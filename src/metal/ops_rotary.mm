// src/metal/ops_rotary.mm
//
// M6.3 — metal::rotary_metal<T>() — Metal rotary position embedding kernel.
//
// Free function (no StorageView dependency) used by:
//   - src/ops/rotary_metal.mm  (Rotary::compute<Device::METAL> wrapper)
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

      id<MTLCommandBuffer> cmd = get_current_command_buffer();
      id<MTLComputeCommandEncoder> enc =
          [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
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
    }

#define DECLARE_ROTARY_METAL(T)                                               \
    template void rotary_metal<T>(const T*, const T*, const T*, T*,           \
                                  dim_t, dim_t, dim_t, dim_t, dim_t, bool, bool);
    DECLARE_ROTARY_METAL(float)
    DECLARE_ROTARY_METAL(ct2_f16)
    DECLARE_ROTARY_METAL(ct2_bf16)
#undef DECLARE_ROTARY_METAL

  }  // namespace metal
}  // namespace ctranslate2
