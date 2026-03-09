// src/metal/ops_quantize.mm
//
// M9.1 — INT8 Quantize / Dequantize GPU dispatch for Metal.
//
// Implements the three metal:: free functions declared in ops_metal.h:
//   metal::quantize_int8_metal<T>()           — per-row INT8 quantization
//   metal::dequantize_int8_metal<T>()         — per-element INT8 dequantization
//   metal::dequantize_gemm_output_metal<T>()  — rescale int32 GEMM output
//
// Called from:
//   src/ops/quantize_metal.mm   (Quantize::quantize<METAL,T,int8_t>)
//   src/ops/dequantize_metal.mm (Dequantize::dequantize<METAL,int8_t,T> +
//                                Dequantize::dequantize_gemm_output<METAL,T>)
//   tests/metal/m91_test.mm     (direct test harness)
//
// All three dispatch helpers encode GPU kernels into the current command buffer
// (encode-only; no commit).  Caller calls commit_and_wait() when ready to read
// results from CPU.

#include "metal/primitives_infra.h"
#include "metal/ops_metal.h"

namespace {

// ---------------------------------------------------------------------------
// PSO infrastructure (all three kernel families share one MSL library)
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_quantize_library() {
  static id<MTLLibrary>  lib  = nil;
  static std::once_flag  flag;
  return compile_library_once(flag, lib, kQuantizeMSL, "quantize");
}

static id<MTLComputePipelineState> get_quantize_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_quantize_library, name);
}

// ---------------------------------------------------------------------------
// dispatch_quantize<T>
//
// Encodes quantize_T into the current CB.
//
// Buffer layout (quantize.metal):
//   buffer(0): input   — T   [batch_size, depth]
//   buffer(1): output  — char [batch_size, depth]  (int8_t)
//   buffer(2): scales  — float [batch_size]
//   buffer(3): dims    — uint2 {batch_size, depth}
//   threadgroup(0):    — float[256]
//
// One threadgroup per row, 256 threads per threadgroup.
// ---------------------------------------------------------------------------

constexpr uint32_t kQuantBlock = 256;

template <typename T>
static void dispatch_quantize(
    const T* input, int8_t* output, float* scales,
    uint32_t batch_size, uint32_t depth) {
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "quantize_%s", MetalTypeName<T>::value);

  id<MTLComputePipelineState> pso = get_quantize_pso(kname);
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_in = 0, off_out = 0, off_sc = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(input,  &off_in)  offset:off_in  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(output, &off_out) offset:off_out atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(scales, &off_sc)  offset:off_sc  atIndex:2];

  const uint32_t dims[2] = { batch_size, depth };
  [enc setBytes:dims length:sizeof(dims) atIndex:3];

  [enc setThreadgroupMemoryLength:kQuantBlock * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kQuantBlock, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// ---------------------------------------------------------------------------
// dispatch_dequantize<T>
//
// Encodes dequantize_T into the current CB.
//
// Buffer layout:
//   buffer(0): input   — char [batch_size, depth]  (int8_t)
//   buffer(1): scales  — float [batch_size]
//   buffer(2): output  — T    [batch_size, depth]
//   buffer(3): dims    — uint2 {batch_size, depth}
//
// One thread per output element.
// ---------------------------------------------------------------------------

template <typename T>  // OutT — type of output
static void dispatch_dequantize(
    const int8_t* input, const float* scales, T* output,
    uint32_t batch_size, uint32_t depth) {
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "dequantize_%s", MetalTypeName<T>::value);

  id<MTLComputePipelineState> pso = get_quantize_pso(kname);
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_in = 0, off_sc = 0, off_out = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(input,  &off_in)  offset:off_in  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(scales, &off_sc)  offset:off_sc  atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(output, &off_out) offset:off_out atIndex:2];

  const uint32_t dims[2] = { batch_size, depth };
  [enc setBytes:dims length:sizeof(dims) atIndex:3];

  const NSUInteger total = (NSUInteger)batch_size * depth;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// ---------------------------------------------------------------------------
// dispatch_dequantize_gemm_output<T>
//
// Encodes dequantize_gemm_output_T into the current CB.
//
// Buffer layout:
//   buffer(0): c        — int  [batch, depth]  (int32_t)
//   buffer(1): a_scales — float
//   buffer(2): b_scales — float
//   buffer(3): bias     — T    [depth] (or dummy buffer when has_bias == 0)
//   buffer(4): y        — T    [batch, depth]  (output)
//   buffer(5): dims     — uint4 {batch, depth, has_bias, act_type}
//   buffer(6): trans    — uint2 {transpose_a, transpose_b}
//
// One thread per output element.
// ---------------------------------------------------------------------------

template <typename T>
static void dispatch_dequantize_gemm_output(
    const int32_t* c,
    const float*   a_scales,
    const float*   b_scales,
    const void*    bias_ptr,   // T* if has_bias, else any valid Metal-registered ptr
    T*             y,
    uint32_t batch,     uint32_t depth,
    uint32_t trans_a,   uint32_t trans_b,
    uint32_t has_bias,  uint32_t act_type) {
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "dequantize_gemm_output_%s",
                MetalTypeName<T>::value);

  id<MTLComputePipelineState> pso = get_quantize_pso(kname);
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_c=0, off_as=0, off_bs=0, off_bi=0, off_y=0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(c,        &off_c)  offset:off_c  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a_scales, &off_as) offset:off_as atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b_scales, &off_bs) offset:off_bs atIndex:2];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(bias_ptr, &off_bi) offset:off_bi atIndex:3];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y,        &off_y)  offset:off_y  atIndex:4];

  const uint32_t dims[4] = { batch, depth, has_bias, act_type };
  const uint32_t trs[2]  = { trans_a, trans_b };
  [enc setBytes:dims length:sizeof(dims) atIndex:5];
  [enc setBytes:trs  length:sizeof(trs)  atIndex:6];

  const NSUInteger total = (NSUInteger)batch * depth;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

}  // anonymous namespace

namespace ctranslate2 {
  namespace metal {

    // -----------------------------------------------------------------------
    // quantize_int8_metal<T>
    //
    // Per-row INT8 quantization.
    //   scale[row] = 127 / max(abs(input[row, :]))
    //   output[row, i] = round(float(input[row, i]) * scale[row])  → int8
    // -----------------------------------------------------------------------

    template <typename T>
    void quantize_int8_metal(const T* input, int8_t* output, float* scales,
                             dim_t batch_size, dim_t depth) {
      dispatch_quantize<T>(input, output, scales,
                           ct2_u32(batch_size), ct2_u32(depth));
    }

    // -----------------------------------------------------------------------
    // dequantize_int8_metal<T>
    //
    // Per-element INT8 dequantization.
    //   output[row, i] = T(float(input[row, i]) / scales[row])
    // -----------------------------------------------------------------------

    template <typename T>
    void dequantize_int8_metal(const int8_t* input, const float* scales, T* output,
                               dim_t batch_size, dim_t depth) {
      dispatch_dequantize<T>(input, scales, output,
                             ct2_u32(batch_size), ct2_u32(depth));
    }

    // -----------------------------------------------------------------------
    // dequantize_gemm_output_metal<T>
    //
    // Rescale int32 GEMM output to floating point.
    //   y[i,j] = c[i,j] / (a_scales[ta?j:i] * b_scales[tb?j:i])
    //          + (has_bias ? bias[j] : 0)
    //   y = activation(y)  if activation_type >= 0
    //
    // activation_type: -1 = none; otherwise static_cast<int>(ActivationType).
    // Maps to kernel: 0=none, 1=relu, 2=gelu_tanh, 3=swish, 4=gelu,
    //                 5=gelu_sigmoid, 6=tanh, 7=sigmoid.
    //
    // bias_ptr: T* if has_bias, else any valid Metal-registered pointer
    //   (e.g. c itself cast to T*); the kernel will not read it when has_bias=0.
    // -----------------------------------------------------------------------

    template <typename T>
    void dequantize_gemm_output_metal(
        const int32_t* c,
        const float*   a_scales,
        const float*   b_scales,
        const void*    bias_ptr,
        T*             y,
        dim_t batch,       dim_t depth,
        bool  transpose_a, bool  transpose_b,
        bool  has_bias,    int   activation_type) {
      // Map activation_type (-1=none) to kernel encoding (0=none, 1+).
      uint32_t act = (activation_type < 0)
                         ? 0u
                         : static_cast<uint32_t>(activation_type) + 1u;
      dispatch_dequantize_gemm_output<T>(
          c, a_scales, b_scales, bias_ptr, y,
          ct2_u32(batch), ct2_u32(depth),
          transpose_a ? 1u : 0u, transpose_b ? 1u : 0u,
          has_bias ? 1u : 0u, act);
    }

    // Explicit instantiations.
    template void quantize_int8_metal<float>(
        const float*, int8_t*, float*, dim_t, dim_t);
    template void quantize_int8_metal<float16_t>(
        const float16_t*, int8_t*, float*, dim_t, dim_t);
    template void quantize_int8_metal<bfloat16_t>(
        const bfloat16_t*, int8_t*, float*, dim_t, dim_t);

    template void dequantize_int8_metal<float>(
        const int8_t*, const float*, float*, dim_t, dim_t);
    template void dequantize_int8_metal<float16_t>(
        const int8_t*, const float*, float16_t*, dim_t, dim_t);
    template void dequantize_int8_metal<bfloat16_t>(
        const int8_t*, const float*, bfloat16_t*, dim_t, dim_t);

    template void dequantize_gemm_output_metal<float>(
        const int32_t*, const float*, const float*, const void*, float*,
        dim_t, dim_t, bool, bool, bool, int);
    template void dequantize_gemm_output_metal<float16_t>(
        const int32_t*, const float*, const float*, const void*, float16_t*,
        dim_t, dim_t, bool, bool, bool, int);
    template void dequantize_gemm_output_metal<bfloat16_t>(
        const int32_t*, const float*, const float*, const void*, bfloat16_t*,
        dim_t, dim_t, bool, bool, bool, int);

  }  // namespace metal
}  // namespace ctranslate2
