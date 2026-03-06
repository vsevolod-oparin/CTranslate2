// src/metal/ops_fused_norm_gemm.mm
//
// Fused LayerNorm/RMSNorm + GEMV dispatch infrastructure.
//
// Implements ctranslate2::metal::fused_layer_norm_gemm_metal<T>() and
// ctranslate2::metal::fused_rms_norm_gemm_metal<T>() declared in
// src/metal/ops_metal.h.
//
// Design: one threadgroup of 256 threads per input row.
// Phase 1: normalize row into threadgroup memory (float32).
// Phase 2: each thread computes ceil(N/256) output columns.
// Threadgroup memory: K * sizeof(float) + 256 * sizeof(float) for reduction.

#include "metal/primitives_infra.h"

namespace {

static constexpr uint32_t kFusedBlock = 256;

// ---------------------------------------------------------------------------
// Fused norm+gemm library
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_fused_norm_gemm_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kFusedNormGemmMSL, "fused_norm_gemm");
}

static id<MTLComputePipelineState> get_fused_norm_gemm_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_fused_norm_gemm_library, name);
}

// ---------------------------------------------------------------------------
// Dispatch: fused LayerNorm + GEMV
//
// Buffer layout (fused_norm_gemm.metal):
//   0: x, 1: gamma, 2: beta, 3: W, 4: y
//   5: K (uint), 6: N (uint), 7: has_beta (uint), 8: eps (float)
//   threadgroup(0): float[K] (normalized row) + float[256] (reduction scratch)
// ---------------------------------------------------------------------------
static void dispatch_fused_ln_gemm(const char* kname,
                                    const void* x, const void* gamma,
                                    const void* beta, const void* W, void* y,
                                    ctranslate2::dim_t outer_size,
                                    ctranslate2::dim_t K, ctranslate2::dim_t N,
                                    float eps) {
  if (outer_size == 0 || K == 0 || N == 0) return;
  uint32_t K_u32     = ct2_u32(K);
  uint32_t N_u32     = ct2_u32(N);
  uint32_t has_beta  = (beta != nullptr) ? 1u : 0u;

  id<MTLComputePipelineState> pso = get_fused_norm_gemm_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];

  NSUInteger off_x = 0, off_g = 0, off_b = 0, off_W = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x)                  offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(gamma, &off_g)              offset:off_g atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(beta ? beta : x, &off_b)    offset:off_b atIndex:2];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(W, &off_W)                  offset:off_W atIndex:3];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y)                  offset:off_y atIndex:4];
  [enc setBytes:&K_u32     length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&N_u32     length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&has_beta  length:sizeof(uint32_t) atIndex:7];
  [enc setBytes:&eps       length:sizeof(float)    atIndex:8];

  // Threadgroup memory: K floats for norm_row + 256 floats for reduction scratch
  NSUInteger tg_mem = (NSUInteger)K * sizeof(float) + kFusedBlock * sizeof(float);
  [enc setThreadgroupMemoryLength:tg_mem atIndex:0];

  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)outer_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kFusedBlock, 1, 1)];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// Dispatch: fused RMSNorm + GEMV
//
// Buffer layout:
//   0: x, 1: gamma, 2: W, 3: y
//   4: K (uint), 5: N (uint), 6: eps (float)
//   threadgroup(0): float[K] + float[256]
// ---------------------------------------------------------------------------
static void dispatch_fused_rms_gemm(const char* kname,
                                     const void* x, const void* gamma,
                                     const void* W, void* y,
                                     ctranslate2::dim_t outer_size,
                                     ctranslate2::dim_t K, ctranslate2::dim_t N,
                                     float eps) {
  if (outer_size == 0 || K == 0 || N == 0) return;
  uint32_t K_u32 = ct2_u32(K);
  uint32_t N_u32 = ct2_u32(N);

  id<MTLComputePipelineState> pso = get_fused_norm_gemm_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];

  NSUInteger off_x = 0, off_g = 0, off_W = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x)      offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(gamma, &off_g)  offset:off_g atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(W, &off_W)      offset:off_W atIndex:2];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y)      offset:off_y atIndex:3];
  [enc setBytes:&K_u32 length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&N_u32 length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&eps   length:sizeof(float)    atIndex:6];

  NSUInteger tg_mem = (NSUInteger)K * sizeof(float) + kFusedBlock * sizeof(float);
  [enc setThreadgroupMemoryLength:tg_mem atIndex:0];

  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)outer_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kFusedBlock, 1, 1)];
  [enc endEncoding];
}

}  // anonymous namespace

namespace ctranslate2 {
  namespace metal {

    template <typename T>
    void fused_layer_norm_gemm_metal(const T* x, const T* gamma, const T* beta,
                                      const T* W, T* y,
                                      dim_t outer_size, dim_t K, dim_t N,
                                      float epsilon) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "fused_layer_norm_gemm_%s",
                    MetalTypeName<T>::value);
      dispatch_fused_ln_gemm(kname, x, gamma, beta, W, y, outer_size, K, N, epsilon);
    }

    template <typename T>
    void fused_rms_norm_gemm_metal(const T* x, const T* gamma,
                                    const T* W, T* y,
                                    dim_t outer_size, dim_t K, dim_t N,
                                    float epsilon) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "fused_rms_norm_gemm_%s",
                    MetalTypeName<T>::value);
      dispatch_fused_rms_gemm(kname, x, gamma, W, y, outer_size, K, N, epsilon);
    }

    // Explicit instantiations
    template void fused_layer_norm_gemm_metal<float>(const float*, const float*, const float*, const float*, float*, dim_t, dim_t, dim_t, float);
    template void fused_layer_norm_gemm_metal<float16_t>(const float16_t*, const float16_t*, const float16_t*, const float16_t*, float16_t*, dim_t, dim_t, dim_t, float);
    template void fused_layer_norm_gemm_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t, dim_t, float);

    template void fused_rms_norm_gemm_metal<float>(const float*, const float*, const float*, float*, dim_t, dim_t, dim_t, float);
    template void fused_rms_norm_gemm_metal<float16_t>(const float16_t*, const float16_t*, const float16_t*, float16_t*, dim_t, dim_t, dim_t, float);
    template void fused_rms_norm_gemm_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t, dim_t, float);

  }  // namespace metal
}  // namespace ctranslate2
