// src/metal/ops_metal.h
//
// Declarations for Metal dispatch functions used by high-level ops
// (LayerNorm, RMSNorm, SoftMax, Gather, SDPA, Conv1D, Quantize/Dequantize).
// Implemented in src/metal/ops_norm_gather.mm, src/metal/ops_sdpa.mm,
// src/metal/ops_conv1d.mm, src/metal/ops_quantize.mm.
//
// Include only from .mm files compiled with Metal support (CT2_WITH_MPS).
// Part of M5.2 — Metal op specializations.
// Part of M6.1 — Scaled dot-product attention.
// Part of M8.3 — Conv1D via im2col + GEMM.
// Part of M9.1 — INT8 Quantize / Dequantize.

#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace metal {

    // layer_norm_metal: normalize each row of x, scale by gamma, shift by beta.
    //   - gamma and beta may be nullptr (identity scale / zero bias).
    //   - Only the last-axis case (inner_size == 1) is dispatched to GPU;
    //     the caller (normalization_metal.mm) throws for other axes.
    //   - One threadgroup of 256 threads per outer element (row).
    template <typename T>
    void layer_norm_metal(const T* x, const T* gamma, const T* beta,
                          T* y, dim_t outer_size, dim_t axis_size, float epsilon);

    // rms_norm_metal: normalize each row of x by its RMS, scale by gamma.
    //   - One threadgroup of 256 threads per batch item.
    template <typename T>
    void rms_norm_metal(const T* x, const T* gamma, T* y,
                        dim_t batch_size, dim_t depth, float epsilon);

    // softmax_metal: compute softmax (or log-softmax) over the last dimension.
    //   - lengths may be nullptr (no masking); log_mode=true → log-softmax.
    //   - One threadgroup of 256 threads per batch item.
    template <typename T>
    void softmax_metal(const T* x, const int32_t* lengths, T* y,
                       dim_t batch_size, dim_t depth, bool log_mode);

    // gather_metal: for each output slot, copy copy_size elements from src.
    //   dst[slot*copy_size + j] = src[batch_index*batch_stride + indices[slot]*copy_size + j]
    //   - One thread per output element.
    //   - Encode-only (no commit).  Callers that read output on CPU must
    //     call synchronize_stream() or rely on copy_from / .to(CPU) which
    //     sync internally (storage_view.cc:417).
    //   - The two-argument Gather::operator() (in-place clone hazard) adds
    //     its own synchronize_stream() in gather.cc.
    template <typename T>
    void gather_metal(const T* src, T* dst, const int32_t* indices,
                      dim_t copy_size, dim_t batch_stride,
                      dim_t num_indices_per_batch, dim_t total_elements);

    // alibi_add_metal: add ALiBi positional bias to attention scores.
    //   input/output: [batch_size, num_heads, query_length, key_length]
    //   alibi:        [1, num_heads, 1, cached_key_length]
    //   alibi_offset: start column within alibi (= cached_kl - key_length when
    //                 use_positive_positions=false; = 0 otherwise).
    template <typename T>
    void alibi_add_metal(const T* input, const T* alibi, T* output,
                         dim_t batch_size, dim_t num_heads,
                         dim_t query_length, dim_t key_length,
                         dim_t cached_key_length, dim_t alibi_offset);

    // rotary_metal: apply rotary position embeddings to a 2-D view of
    //   [total_vecs, depth].
    //   sin/cos: [max_time, ndims] tables.
    //   is_transposed=false: t = vec / head_size  (FA2 layout)
    //   is_transposed=true:  t = vec % max_time   (std layout)
    template <typename T>
    void rotary_metal(const T* input, const T* sin_buf, const T* cos_buf,
                      T* output,
                      dim_t total_vecs, dim_t depth, dim_t ndims,
                      dim_t max_time, dim_t head_size,
                      bool interleave, bool is_transposed);

    // sdpa_metal: scaled dot-product attention.
    //   q/k/v layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
    //   output layout: same shape as q.
    //   scale: multiplied into Q * K^T before softmax.
    //   is_causal: apply causal mask (scores[col > row] = large_neg).
    //   kv_batch_stride: number of elements between consecutive batches in K/V.
    //     When 0 (default), uses seqlen_k * num_heads_k * head_dim.
    //     Set to total_cache * num_heads_k * head_dim for KV-cache decode
    //     where the cache is pre-allocated larger than seqlen_k_eff.
    //   beam_size: when > 1, Q has batch_size batches but K/V have
    //     batch_size/beam_size batches.  K/V batch index = Q batch / beam_size.
    //     Used by flash cross-attention to avoid tiling K/V for beam search.
    template <typename T>
    void sdpa_metal(const T* q, const T* k, const T* v, T* output,
                    dim_t batch_size, dim_t seqlen_q, dim_t seqlen_k,
                    dim_t num_heads, dim_t num_heads_k, dim_t head_dim,
                    float scale, bool is_causal,
                    dim_t kv_batch_stride = 0,
                    dim_t beam_size = 1);

    // conv1d_metal: 1-D convolution via im2col + GEMM (groups == 1 only).
    //   input:  [B, C_in, T_in]    — NCT layout
    //   weight: [C_out, C_in, K]
    //   output: [B, C_out, T_out]  — pre-allocated by the caller
    //   T_out must equal (T_in + 2*padding - dilation*(K-1) - 1) / stride + 1
    //   dilation must be >= 1.
    template <typename T>
    void conv1d_metal(const T* input, const T* weight, T* output,
                      dim_t B,    dim_t C_in, dim_t T_in,
                      dim_t C_out, dim_t K,   dim_t T_out,
                      dim_t stride, dim_t padding, dim_t dilation);

    // quantize_int8_metal: per-row INT8 quantization.
    //   scale[row] = 127 / max(abs(input[row, :]))
    //   output[row, i] = round(float(input[row, i]) * scale[row])  → int8
    //   One threadgroup per row (256 threads); encode-only.
    template <typename T>
    void quantize_int8_metal(const T* input, int8_t* output, float* scales,
                             dim_t batch_size, dim_t depth);

    // dequantize_int8_metal: per-element INT8 dequantization.
    //   output[row, i] = T(float(input[row, i]) / scales[row])
    //   One thread per element; encode-only.
    template <typename T>
    void dequantize_int8_metal(const int8_t* input, const float* scales, T* output,
                               dim_t batch_size, dim_t depth);

    // dequantize_gemm_output_metal: rescale int32 GEMM output to floating point.
    //   y[i,j] = c[i,j] / (a_scales[ta?j:i] * b_scales[tb?j:i])
    //          + (has_bias ? bias[j] : 0)
    //   then apply optional activation.
    //
    //   activation_type: -1 = none; otherwise static_cast<int>(ActivationType).
    //   bias_ptr: T* if has_bias, else any valid Metal-registered pointer.
    //   One thread per element; encode-only.
    template <typename T>
    void dequantize_gemm_output_metal(
        const int32_t* c,
        const float*   a_scales,
        const float*   b_scales,
        const void*    bias_ptr,
        T*             y,
        dim_t batch,       dim_t depth,
        bool  transpose_a, bool  transpose_b,
        bool  has_bias,    int   activation_type);

    // fused_layer_norm_gemm_metal: fused LayerNorm + GEMV in a single dispatch.
    //   x: [outer_size, K], gamma: [K], beta: [K] (nullable), W: [N, K], y: [outer_size, N].
    //   One threadgroup (256 threads) per row; encode-only.
    template <typename T>
    void fused_layer_norm_gemm_metal(const T* x, const T* gamma, const T* beta,
                                      const T* W, T* y,
                                      dim_t outer_size, dim_t K, dim_t N,
                                      float epsilon);

    // fused_rms_norm_gemm_metal: fused RMSNorm + GEMV in a single dispatch.
    //   x: [outer_size, K], gamma: [K], W: [N, K], y: [outer_size, N].
    //   One threadgroup (256 threads) per row; encode-only.
    template <typename T>
    void fused_rms_norm_gemm_metal(const T* x, const T* gamma,
                                    const T* W, T* y,
                                    dim_t outer_size, dim_t K, dim_t N,
                                    float epsilon);

    // topk_metal: GPU top-k selection.
    //   k=1: argmax kernel. k>1: single-pass fused top-k (local insertion sort + tree merge).
    //   Finds the top-k values and their indices in each row of [batch_size, depth].
    //   Output: values [batch_size, k], indices [batch_size, k].
    //   One threadgroup (256 threads) per batch item; encode-only.
    template <typename T>
    void topk_metal(const T* input, T* values, int32_t* indices,
                    dim_t batch_size, dim_t depth, dim_t k = 1);

  }  // namespace metal
}  // namespace ctranslate2
