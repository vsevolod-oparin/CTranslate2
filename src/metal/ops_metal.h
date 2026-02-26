// src/metal/ops_metal.h
//
// Declarations for Metal dispatch functions used by high-level ops
// (LayerNorm, RMSNorm, SoftMax, Gather, SDPA).  Implemented in
// src/metal/ops_norm_gather.mm and src/metal/ops_sdpa.mm.
//
// Include only from .mm files compiled with Metal support (CT2_WITH_METAL).
// Part of M5.2 — Metal op specializations.
// Part of M6.1 — Scaled dot-product attention.

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
    //   M6.1 scope: offset == 0 only (no KV cache).
    template <typename T>
    void sdpa_metal(const T* q, const T* k, const T* v, T* output,
                    dim_t batch_size, dim_t seqlen_q, dim_t seqlen_k,
                    dim_t num_heads, dim_t num_heads_k, dim_t head_dim,
                    float scale, bool is_causal);

  }  // namespace metal
}  // namespace ctranslate2
