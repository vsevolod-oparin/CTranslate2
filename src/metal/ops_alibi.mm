// src/metal/ops_alibi.mm
//
// M6.4 — metal::alibi_add_metal<T>() — Metal ALiBi positional bias kernel.
//
// Free function (no StorageView dependency) used by:
//   - src/ops/alibi_add_metal.mm  (AlibiAdd::compute<Device::METAL> wrapper)
//   - tests/metal/alibi_test.mm   (standalone correctness test)
//
// Algorithm:
//   Input: attention scores [batch_size, num_heads, query_length, key_length]
//   ALiBi: slopes table     [1, num_heads, 1, cached_key_length]
//   Output: input + ALiBi (broadcast over batch and query dims)
//
//   For each row (b, h, q) and key position k:
//     output[b,h,q,k] = input[b,h,q,k] + alibi[0,h,0, alibi_offset+k]
//
//   alibi_offset = use_positive_positions ? 0 : cached_key_length - key_length
//   (computed by AlibiAdd::operator() before calling compute<METAL>)
//
//   Dispatch: 2D grid [total_rows, key_length]
//     total_rows = batch_size * num_heads * query_length
//   All arithmetic in float32.

#include "metal/primitives_infra.h"
#include "metal/ops_metal.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

// ---------------------------------------------------------------------------
// MSL source — embedded raw string
// ---------------------------------------------------------------------------

static constexpr const char* kAlibiMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

// One thread per output element (vec, k).
// vec = gid.x  — row index in [0, total_rows), where total_rows = batch*nh*ql
// k   = gid.y  — key position in [0, key_length)
//
// Head index:  h = (vec / query_length) % num_heads
// ALiBi index: alibi[h * cached_kl + alibi_offset + k]
//
// Arithmetic always in float32.

#define DEFINE_ALIBI_ADD(T)                                                    \
kernel void alibi_add_##T(                                                     \
    device const T*    input        [[buffer(0)]],                             \
    device const T*    alibi        [[buffer(1)]],                             \
    device       T*    output       [[buffer(2)]],                             \
    constant uint&     num_heads    [[buffer(3)]],                             \
    constant uint&     query_length [[buffer(4)]],                             \
    constant uint&     key_length   [[buffer(5)]],                             \
    constant uint&     cached_kl    [[buffer(6)]],                             \
    constant uint&     alibi_offset [[buffer(7)]],                             \
    uint2 gid [[thread_position_in_grid]])                                     \
{                                                                              \
    uint vec = gid.x;                                                          \
    uint k   = gid.y;                                                          \
    uint h   = (vec / query_length) % num_heads;                               \
    uint in_idx    = vec * key_length + k;                                     \
    uint alibi_idx = h * cached_kl + alibi_offset + k;                        \
    output[in_idx] = (T)(float(input[in_idx]) + float(alibi[alibi_idx]));     \
}

DEFINE_ALIBI_ADD(float)
DEFINE_ALIBI_ADD(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_ALIBI_ADD(bfloat)
#endif
)msl";

// ---------------------------------------------------------------------------
// Library / PSO cache
// ---------------------------------------------------------------------------

namespace {

static id<MTLLibrary> get_alibi_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kAlibiMSL, "alibi");
}

static id<MTLComputePipelineState> get_alibi_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_alibi_library, name);
}

}  // namespace

// ---------------------------------------------------------------------------
// metal::alibi_add_metal<T>
// ---------------------------------------------------------------------------

namespace ctranslate2 {
  namespace metal {

    template <typename T>
    void alibi_add_metal(const T* input, const T* alibi, T* output,
                         dim_t batch_size, dim_t num_heads,
                         dim_t query_length, dim_t key_length,
                         dim_t cached_key_length, dim_t alibi_offset) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "alibi_add_%s", MetalTypeName<T>::value);
      id<MTLComputePipelineState> pso = get_alibi_pso(kname);

      const dim_t total_rows = batch_size * num_heads * query_length;

      const uint32_t u_num_heads    = ct2_u32(num_heads);
      const uint32_t u_query_length = ct2_u32(query_length);
      const uint32_t u_key_length   = ct2_u32(key_length);
      const uint32_t u_cached_kl    = ct2_u32(cached_key_length);
      const uint32_t u_alibi_offset = ct2_u32(alibi_offset);

      id<MTLCommandBuffer> cmd = get_current_command_buffer();
      id<MTLComputeCommandEncoder> enc =
          [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
      [enc setComputePipelineState:pso];

      NSUInteger off_in = 0, off_alibi = 0, off_out = 0;
      [enc setBuffer:metal_buffer_for_ptr(input,  &off_in)    offset:off_in    atIndex:0];
      [enc setBuffer:metal_buffer_for_ptr(alibi,  &off_alibi) offset:off_alibi atIndex:1];
      [enc setBuffer:metal_buffer_for_ptr(output, &off_out)   offset:off_out   atIndex:2];
      [enc setBytes:&u_num_heads    length:sizeof(uint32_t) atIndex:3];
      [enc setBytes:&u_query_length length:sizeof(uint32_t) atIndex:4];
      [enc setBytes:&u_key_length   length:sizeof(uint32_t) atIndex:5];
      [enc setBytes:&u_cached_kl    length:sizeof(uint32_t) atIndex:6];
      [enc setBytes:&u_alibi_offset length:sizeof(uint32_t) atIndex:7];

      // 2D dispatch: outer = total_rows, inner = key_length; one thread per element.
      const NSUInteger kl_ns   = static_cast<NSUInteger>(key_length);
      const NSUInteger rows_ns = static_cast<NSUInteger>(total_rows);
      NSUInteger tg_kl = std::min(kl_ns,
          static_cast<NSUInteger>(pso.maxTotalThreadsPerThreadgroup));
      [enc dispatchThreads:MTLSizeMake(rows_ns, kl_ns, 1)
          threadsPerThreadgroup:MTLSizeMake(1, tg_kl, 1)];
      [enc endEncoding];
    }

#define DECLARE_ALIBI_METAL(T)                                               \
    template void alibi_add_metal<T>(const T*, const T*, T*,                 \
                                     dim_t, dim_t, dim_t, dim_t, dim_t, dim_t);
    DECLARE_ALIBI_METAL(float)
    DECLARE_ALIBI_METAL(ct2_f16)
    DECLARE_ALIBI_METAL(ct2_bf16)
#undef DECLARE_ALIBI_METAL

  }  // namespace metal
}  // namespace ctranslate2
