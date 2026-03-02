// src/metal/ops_conv1d.mm
//
// M8.3 — Conv1D Metal dispatch (im2col + GEMM).
//
// Implements metal::conv1d_metal<T>() declared in src/metal/ops_metal.h.
// Called from src/ops/conv1d_metal.mm.
//
// Algorithm:
//   1. im2col (GPU kernel): input[B, C_in, T_in] → im2col[B, T_out, C_in*K]
//      Encoded into the per-thread command buffer (encode-only for f32/f16;
//      the BF16 GEMM path flushes the CB before reading im2col results).
//
//   2. GEMM (per batch b):
//        A = weight    [C_out, CK]     trans_a=false, same for all b
//        B = im2col[b] [T_out, CK]     trans_b=true  → treat as [CK, T_out]
//        C = output[b] [C_out, T_out]
//      Uses primitives<Device::METAL>::gemm<T,T> (MPS for f32/f16;
//      MPSGraph for bf16 — commits synchronously but reads already-flushed data).
//
// The im2col buffer is allocated via the Metal allocator (Shared, registered
// in MetalAllocator::_live) so metal_buffer_for_ptr() can locate it.
//
// Groups == 1 constraint is enforced by the caller (conv1d_metal.mm).

#include "metal/primitives_infra.h"
#include "metal/ops_metal.h"
#include "ctranslate2/allocator.h"

namespace {

// ---------------------------------------------------------------------------
// RAII allocator-registered temporary buffer (same pattern as ops_sdpa.mm).
//
// Buffers allocated via get_allocator<Device::METAL>() are tracked in
// MetalAllocator::_live so metal_buffer_for_ptr() can find them.
// ---------------------------------------------------------------------------

struct MetalTempBuf {
  void* ptr = nullptr;

  MetalTempBuf() = default;

  explicit MetalTempBuf(size_t n_bytes) {
    ptr = ctranslate2::get_allocator<ctranslate2::Device::METAL>().allocate(n_bytes, 0);
  }

  ~MetalTempBuf() {
    if (ptr)
      ctranslate2::get_allocator<ctranslate2::Device::METAL>().free(ptr, 0);
  }

  MetalTempBuf(const MetalTempBuf&) = delete;
  MetalTempBuf& operator=(const MetalTempBuf&) = delete;
};

// ---------------------------------------------------------------------------
// im2col PSO infrastructure
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_conv1d_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kConv1dMSL, "conv1d");
}

static id<MTLComputePipelineState> get_conv1d_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_conv1d_library, name);
}

// ---------------------------------------------------------------------------
// dispatch_im2col<T>
//
// Encodes the im2col kernel into the current command buffer (encode-only).
//
// Buffer layout (conv1d.metal):
//   buffer(0): input  [B, C_in, T_in]
//   buffer(1): output [B, T_out, C_in*K]
//   buffer(2): dims0  uint4 = {B, C_in, T_in, T_out}
//   buffer(3): dims1  uint4 = {K, stride, padding, dilation}
//
// One thread per output element.  total = B * T_out * C_in * K.
// ---------------------------------------------------------------------------

template <typename T>
static void dispatch_im2col(
    const T* input, T* output,
    ctranslate2::dim_t B,     ctranslate2::dim_t C_in,
    ctranslate2::dim_t T_in,  ctranslate2::dim_t T_out,
    ctranslate2::dim_t K,
    ctranslate2::dim_t stride, ctranslate2::dim_t padding,
    ctranslate2::dim_t dilation) {
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "im2col_%s", MetalTypeName<T>::value);

  id<MTLComputePipelineState> pso = get_conv1d_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];

  NSUInteger off_in = 0, off_out = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(input,  &off_in)  offset:off_in  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(output, &off_out) offset:off_out atIndex:1];

  const uint32_t dims0[4] = {
    ct2_u32(B), ct2_u32(C_in), ct2_u32(T_in), ct2_u32(T_out)
  };
  const uint32_t dims1[4] = {
    ct2_u32(K), ct2_u32(stride), ct2_u32(padding), ct2_u32(dilation)
  };
  [enc setBytes:dims0 length:sizeof(dims0) atIndex:2];
  [enc setBytes:dims1 length:sizeof(dims1) atIndex:3];

  const ctranslate2::dim_t total = B * T_out * C_in * K;
  const NSUInteger tpg = std::min((NSUInteger)256,
                                   pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
}

}  // anonymous namespace

namespace ctranslate2 {
  namespace metal {

    // -----------------------------------------------------------------------
    // conv1d_metal<T>
    //
    // Steps:
    //   1. Allocate im2col buffer [B, T_out, CK] via Metal allocator.
    //   2. Encode im2col kernel (encode-only, no commit).
    //   3. Per-batch GEMM: weight[C_out,CK] × im2col[b,T_out,CK]^T → out[b,C_out,T_out].
    //
    // For f32/f16: all three steps encode into the CB; the caller commits.
    // For bf16: first GEMM (via dispatch_bf16_gemm) calls commit_and_wait(),
    //   which flushes the im2col kernel.  Subsequent batches run synchronously.
    // -----------------------------------------------------------------------

    template <typename T>
    void conv1d_metal(const T* input, const T* weight, T* output,
                      dim_t B,     dim_t C_in, dim_t T_in,
                      dim_t C_out, dim_t K,    dim_t T_out,
                      dim_t stride, dim_t padding, dim_t dilation) {
      const dim_t CK = C_in * K;

      // Allocate im2col buffer via registered allocator so GEMM can find it.
      MetalTempBuf im2col_alloc(static_cast<size_t>(B * T_out * CK) * sizeof(T));
      T* p = static_cast<T*>(im2col_alloc.ptr);

      // Step 1: encode im2col.
      dispatch_im2col(input, p, B, C_in, T_in, T_out, K, stride, padding, dilation);

      // Step 2: batched GEMM (stride_a=0: weight is shared across all batches).
      primitives<Device::METAL>::template gemm_batch_strided<T, T>(
          false, true,                   // trans_a=false, trans_b=true
          C_out, T_out, CK,              // m, n, k
          1.0f,
          weight, CK, 0,                 // A = weight [C_out, CK], shared (stride=0)
          p, CK, T_out * CK,             // B = im2col [B, T_out, CK], per-batch
          0.0f,
          output, T_out, C_out * T_out,  // C = output [B, C_out, T_out], per-batch
          B);
    }

    // Explicit instantiations.
    template void conv1d_metal<float>(
        const float*, const float*, float*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t);
    template void conv1d_metal<float16_t>(
        const float16_t*, const float16_t*, float16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t);
    template void conv1d_metal<bfloat16_t>(
        const bfloat16_t*, const bfloat16_t*, bfloat16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, dim_t);

  }  // namespace metal
}  // namespace ctranslate2
