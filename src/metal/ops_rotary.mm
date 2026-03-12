// src/metal/ops_rotary.mm
//
// M6.3 — metal::rotary_metal<T>() — Metal rotary position embedding kernel.
//
// Free function (no StorageView dependency) used by:
//   - src/ops/rotary_metal.mm  (Rotary::compute<Device::MPS> wrapper)
//   - tests/metal/rotary_test.mm (standalone correctness test)
//
// Algorithm:
//   Dispatch: 2D grid [total_vecs, depth] — one thread per output element.
//
//   Time index (mirrors CUDA Rotary kernel):
//     is_transposed=false (FA2 layout):  t = vec / head_size
//     is_transposed=true  (std layout):  t = vec % max_time
//
//   Non-interleave (LLaMA-style):
//     middle = ndims / 2
//     y[d]         = x[d]        * cos[t,d]    - x[d+middle] * sin[t,d]   d < middle
//     y[d+middle]  = x[d+middle] * cos[t,d]    + x[d]        * sin[t,d]   d < middle
//     y[d]         = x[d]  for d in [ndims, depth)
//
//   Interleave (GPT-NeoX-style):
//     y[2i]   = x[2i]   * cos[t,2i]   - x[2i+1] * sin[t,2i]   i < ndims/2
//     y[2i+1] = x[2i+1] * cos[t,2i+1] + x[2i]   * sin[t,2i+1] i < ndims/2
//     y[d]    = x[d]  for d in [ndims, depth)
//
// All arithmetic is performed in float32 regardless of storage type T.

#include "metal/primitives_infra.h"
#include "metal/ops_metal.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

// ---------------------------------------------------------------------------
// MSL source — embedded raw string
// ---------------------------------------------------------------------------

// NOTE: MSL is embedded inline here (not in a .metal file) and is therefore
// NOT tracked by tools/gen_msl_strings.py / check_msl_sync.
static constexpr const char* kRotaryMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

// One thread per output element.
// Grid:   [total_vecs, depth, 1]
// t computation mirrors CUDA Rotary kernel:
//   is_transposed=0 (FA2 layout): t = vec / head_size
//   is_transposed=1 (std layout): t = vec % max_time
//
// Arithmetic always in float32.

#define DEFINE_ROTARY(T)                                                       \
kernel void rotary_##T(                                                        \
    device const T*    input        [[buffer(0)]],                             \
    device const T*    sin_buf      [[buffer(1)]],                             \
    device const T*    cos_buf      [[buffer(2)]],                             \
    device       T*    output       [[buffer(3)]],                             \
    constant uint&     ndims        [[buffer(4)]],                             \
    constant uint&     depth        [[buffer(5)]],                             \
    constant uint&     max_time     [[buffer(6)]],                             \
    constant uint&     head_size    [[buffer(7)]],                             \
    constant uint&     interleave   [[buffer(8)]],                             \
    constant uint&     is_transposed[[buffer(9)]],                             \
    uint2 gid [[thread_position_in_grid]])                                     \
{                                                                              \
    uint vec = gid.x;                                                          \
    uint d   = gid.y;                                                          \
    /* time index for sin/cos table */                                         \
    uint t = (is_transposed != 0u) ? (vec % max_time) : (vec / head_size);    \
    float xi = float(input[vec * depth + d]);                                  \
    if (d >= ndims) {                                                          \
        output[vec * depth + d] = (T)xi;                                       \
        return;                                                                \
    }                                                                          \
    float sin_d = float(sin_buf[t * ndims + d]);                               \
    float cos_d = float(cos_buf[t * ndims + d]);                               \
    float result;                                                              \
    if (interleave == 0u) {                                                    \
        uint middle = ndims / 2u;                                              \
        if (d < middle) {                                                      \
            float partner = float(input[vec * depth + d + middle]);            \
            result = xi * cos_d - partner * sin_d;                             \
        } else {                                                               \
            float partner = float(input[vec * depth + d - middle]);            \
            result = xi * cos_d + partner * sin_d;                             \
        }                                                                      \
    } else {                                                                   \
        if (d % 2u == 0u) {                                                    \
            float partner = float(input[vec * depth + d + 1u]);                \
            result = xi * cos_d - partner * sin_d;                             \
        } else {                                                               \
            float partner = float(input[vec * depth + d - 1u]);                \
            result = xi * cos_d + partner * sin_d;                             \
        }                                                                      \
    }                                                                          \
    output[vec * depth + d] = (T)result;                                       \
}

DEFINE_ROTARY(float)
DEFINE_ROTARY(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_ROTARY(bfloat)
#endif
)msl";

// ---------------------------------------------------------------------------
// M12.17 — Decode RoPE kernel using half-sized cos/sin tables.
//
// Grid:   [num_vecs, depth, 1]  — one thread per element.
// Each thread computes RoPE for one (vec, d) using the cos/sin row at
// the given position (passed as a pointer offset, not computed from time).
//
// Half-table format: cos_row/sin_row have half_dim elements.
//   Non-interleave:
//     d < half_dim:        y[d]             = x[d]             * cos[d] - x[d+half_dim] * sin[d]
//     half_dim <= d < 2*half_dim: y[d]      = x[d]             * cos[d-half_dim] + x[d-half_dim] * sin[d-half_dim]
//   Interleave:
//     d even:              y[d]   = x[d]   * cos[d/2] - x[d+1] * sin[d/2]
//     d odd:               y[d]   = x[d]   * cos[d/2] + x[d-1] * sin[d/2]
// Elements d >= ndims are unchanged.
// ---------------------------------------------------------------------------
// M12.18: Fixed WAR race condition from M12.17.
// Original kernel read partner element data[d±half_dim] while another thread
// simultaneously wrote to it.  Fix: load entire vector into threadgroup
// memory, barrier, then compute RoPE from the scratch copy.
// Threadgroup memory: depth * sizeof(float) per threadgroup (~256 bytes).
// Dispatch: (num_vecs, depth, 1) with threadgroup (1, depth, 1).
// Requires depth == threadgroup y-dimension (all elements in one threadgroup).
static constexpr const char* kDecodeRopeMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#define DEFINE_DECODE_ROPE(T)                                                  \
kernel void decode_rope_##T(                                                   \
    device       T*    data      [[buffer(0)]],                                \
    device const T*    cos_row   [[buffer(1)]],                                \
    device const T*    sin_row   [[buffer(2)]],                                \
    constant  uint&    half_dim  [[buffer(3)]],                                \
    constant  uint&    depth     [[buffer(4)]],                                \
    constant  uint&    interleave[[buffer(5)]],                                \
    threadgroup float* scratch   [[threadgroup(0)]],                           \
    uint2 gid [[thread_position_in_grid]])                                     \
{                                                                              \
    uint vec = gid.x;                                                          \
    uint d   = gid.y;                                                          \
    uint ndims = half_dim * 2u;                                                \
    /* Load entire vector into threadgroup memory (all threads in this */      \
    /* threadgroup share the same vec). */                                     \
    if (d < depth)                                                             \
        scratch[d] = float(data[vec * depth + d]);                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    if (d >= ndims) return;                                                    \
    float xi = scratch[d];                                                     \
    float result;                                                              \
    if (interleave == 0u) {                                                    \
        if (d < half_dim) {                                                    \
            float c = float(cos_row[d]);                                       \
            float s = float(sin_row[d]);                                       \
            result = xi * c - scratch[d + half_dim] * s;                       \
        } else {                                                               \
            uint partner_d = d - half_dim;                                     \
            float c = float(cos_row[partner_d]);                               \
            float s = float(sin_row[partner_d]);                               \
            result = xi * c + scratch[partner_d] * s;                          \
        }                                                                      \
    } else {                                                                   \
        uint pair_idx = d / 2u;                                                \
        float c = float(cos_row[pair_idx]);                                    \
        float s = float(sin_row[pair_idx]);                                    \
        if (d % 2u == 0u) {                                                    \
            result = xi * c - scratch[d + 1u] * s;                             \
        } else {                                                               \
            result = xi * c + scratch[d - 1u] * s;                             \
        }                                                                      \
    }                                                                          \
    data[vec * depth + d] = (T)result;                                         \
}

DEFINE_DECODE_ROPE(float)
DEFINE_DECODE_ROPE(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_DECODE_ROPE(bfloat)
#endif
)msl";

// ---------------------------------------------------------------------------
// Library / PSO cache
// ---------------------------------------------------------------------------

namespace {

static id<MTLLibrary> get_rotary_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kRotaryMSL, "rotary");
}

static id<MTLComputePipelineState> get_rotary_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_rotary_library, name);
}

static id<MTLLibrary> get_decode_rope_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kDecodeRopeMSL, "decode_rope");
}

static id<MTLComputePipelineState> get_decode_rope_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_decode_rope_library, name);
}

}  // namespace

// ---------------------------------------------------------------------------
// metal::rotary_metal<T>
// ---------------------------------------------------------------------------

namespace ctranslate2 {
  namespace metal {

    template <typename T>
    void rotary_metal(const T* input, const T* sin_buf, const T* cos_buf,
                      T* output,
                      dim_t total_vecs, dim_t depth, dim_t ndims,
                      dim_t max_time, dim_t head_size,
                      bool interleave, bool is_transposed) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "rotary_%s", MetalTypeName<T>::value);
      id<MTLComputePipelineState> pso = get_rotary_pso(kname);

      const uint32_t u_ndims       = ct2_u32(ndims);
      const uint32_t u_depth       = ct2_u32(depth);
      const uint32_t u_max_time    = ct2_u32(max_time);
      const uint32_t u_head_size   = ct2_u32(head_size);
      const uint32_t u_interleave  = interleave    ? 1u : 0u;
      const uint32_t u_transposed  = is_transposed ? 1u : 0u;

      id<MTLComputeCommandEncoder> enc =
          create_compute_encoder();
      [enc setComputePipelineState:pso];

      NSUInteger off_in = 0, off_sin = 0, off_cos = 0, off_out = 0;
      [enc setBuffer:metal_buffer_for_ptr(input,   &off_in)  offset:off_in  atIndex:0];
      [enc setBuffer:metal_buffer_for_ptr(sin_buf, &off_sin) offset:off_sin atIndex:1];
      [enc setBuffer:metal_buffer_for_ptr(cos_buf, &off_cos) offset:off_cos atIndex:2];
      [enc setBuffer:metal_buffer_for_ptr(output,  &off_out) offset:off_out atIndex:3];
      [enc setBytes:&u_ndims      length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&u_depth      length:sizeof(uint32_t) atIndex:5];
      [enc setBytes:&u_max_time   length:sizeof(uint32_t) atIndex:6];
      [enc setBytes:&u_head_size  length:sizeof(uint32_t) atIndex:7];
      [enc setBytes:&u_interleave length:sizeof(uint32_t) atIndex:8];
      [enc setBytes:&u_transposed length:sizeof(uint32_t) atIndex:9];

      // 2D dispatch: outer = total_vecs, inner = depth; one thread per element.
      const NSUInteger depth_ns = static_cast<NSUInteger>(depth);
      NSUInteger tg_depth = std::min(depth_ns,
          static_cast<NSUInteger>(pso.maxTotalThreadsPerThreadgroup));
      [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_vecs), depth_ns, 1)
          threadsPerThreadgroup:MTLSizeMake(1, tg_depth, 1)];
      [enc endEncoding];
      [enc release];
    }

#define DECLARE_ROTARY_METAL(T)                                               \
    template void rotary_metal<T>(const T*, const T*, const T*, T*,           \
                                  dim_t, dim_t, dim_t, dim_t, dim_t, bool, bool);
    DECLARE_ROTARY_METAL(float)
    DECLARE_ROTARY_METAL(ct2_f16)
    DECLARE_ROTARY_METAL(ct2_bf16)
#undef DECLARE_ROTARY_METAL

    // -----------------------------------------------------------------------
    // M12.17 — GPU decode RoPE with half-table format (encode-only).
    //
    // Applies RoPE in-place to `num_vecs` head vectors of size `depth`.
    // cos_row / sin_row point to the single position row in half-tables
    // (half_dim elements each).
    // -----------------------------------------------------------------------
    template <typename T>
    void decode_rope_metal(T* data,
                           const T* cos_row,
                           const T* sin_row,
                           dim_t num_vecs,
                           dim_t depth,
                           dim_t half_dim,
                           bool interleave) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "decode_rope_%s", MetalTypeName<T>::value);
      id<MTLComputePipelineState> pso = get_decode_rope_pso(kname);

      const uint32_t u_half_dim    = ct2_u32(half_dim);
      const uint32_t u_depth       = ct2_u32(depth);
      const uint32_t u_interleave  = interleave ? 1u : 0u;

      id<MTLComputeCommandEncoder> enc = create_compute_encoder();
      [enc setComputePipelineState:pso];

      NSUInteger off_data = 0, off_cos = 0, off_sin = 0;
      [enc setBuffer:metal_buffer_for_ptr(data,    &off_data) offset:off_data atIndex:0];
      [enc setBuffer:metal_buffer_for_ptr(cos_row, &off_cos)  offset:off_cos  atIndex:1];
      [enc setBuffer:metal_buffer_for_ptr(sin_row, &off_sin)  offset:off_sin  atIndex:2];
      [enc setBytes:&u_half_dim    length:sizeof(uint32_t) atIndex:3];
      [enc setBytes:&u_depth       length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&u_interleave  length:sizeof(uint32_t) atIndex:5];

      // Threadgroup memory for scratch buffer (WAR race fix).
      // depth floats per threadgroup — one full head vector.
      const NSUInteger depth_ns = static_cast<NSUInteger>(depth);
      [enc setThreadgroupMemoryLength:depth_ns * sizeof(float) atIndex:0];

      // All depth elements must be in the same threadgroup for the barrier
      // to synchronize the scratch load.  depth <= 512 in practice.
      NSUInteger tg_depth = std::min(depth_ns,
          static_cast<NSUInteger>(pso.maxTotalThreadsPerThreadgroup));
      [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(num_vecs), depth_ns, 1)
          threadsPerThreadgroup:MTLSizeMake(1, tg_depth, 1)];
      [enc endEncoding];
      [enc release];
    }

#define DECLARE_DECODE_ROPE_METAL(T)                                          \
    template void decode_rope_metal<T>(T*, const T*, const T*,                \
                                       dim_t, dim_t, dim_t, bool);
    DECLARE_DECODE_ROPE_METAL(float)
    DECLARE_DECODE_ROPE_METAL(ct2_f16)
    DECLARE_DECODE_ROPE_METAL(ct2_bf16)
#undef DECLARE_DECODE_ROPE_METAL

  }  // namespace metal
}  // namespace ctranslate2
