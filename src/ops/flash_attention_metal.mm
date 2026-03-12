// src/ops/flash_attention_metal.mm
//
// M6.1 — FlashAttention::compute<Device::MPS>.
// M6.2 — KV-cache update (offset > 0 path).
// M6.3 — Rotary embeddings (RoPE) for the decode path.
//
// Implements scaled dot-product attention on the Metal backend.
// Calls metal::sdpa_metal<T> from src/metal/ops_sdpa.mm.
//
// M6.1 scope (offset == 0): SDPA over full Q/K/V without cache.
//   RoPE: applied by the layer (RotaryEmbeddings::apply) before this call;
//   rotary_cos/sin are nullptr here.
//
// M6.2 scope (offset > 0):  KV-cache update + SDPA decode step.
//   Algorithm:
//     1. commit_and_wait()  — flush any pending GPU writes so CPU can read
//                             the new keys/values (written by prior linear ops).
//     2. CPU RoPE (M6.3)    — apply rotary position `offset` to Q and new K
//                             using the half-sized cos/sin tables supplied by
//                             the layer (rotary_cos/sin != nullptr).
//     3. CPU memcpy         — write new K/V into cached_keys/values at position
//                             `offset` (unified memory, zero-copy GPU side).
//     4. sdpa_metal         — attend Q over the full valid cache [0, offset+seqlen_new).
//
// KV cache layout: [batch, total_cache_slots, num_heads_k, head_dim].
//   Per-batch slice starts at b * total_cache_slots * num_heads_k * head_dim.
//   Per-position stride (row_elements): num_heads_k * head_dim.
//
// Causal masking for decode: when seqlen_q == 1, causal masking is
//   unnecessary — the KV cache itself provides the temporal boundary.
//   Using is_causal=true with sq=1 would mask all but position 0, so
//   we force is_causal=false when sq==1 (mirrors the CUDA FlashAttention
//   implementation).
//
// RoPE half-table format (rotary_cos / rotary_sin for decode):
//   Shape [total_positions, ndims/2] — only the "positive-frequency" half.
//   rotary_cos->dim(1) = ndims/2,  rotary_sin->dim(1) = ndims/2.
//   ndims = rotary_cos->dim(1) * 2.
//   Row `offset` gives the cos/sin for position `offset`.
//
// Still unsupported in M6.3 (guard throws):
//   ALiBi, sliding window, attention weight output,
//   seqlen_q > 1 with offset > 0 (chunk-prefill into KV cache).

#include "ctranslate2/ops/flash_attention.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "metal/ops_metal.h"
#include "metal/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // -------------------------------------------------------------------------
    // apply_rope_half — CPU RoPE for one token using half-sized cos/sin tables.
    //
    //   x:       token vector [depth] (in-place)
    //   cos_row: cos[offset, 0..half_dim)   (half_dim = ndims/2)
    //   sin_row: sin[offset, 0..half_dim)
    //   ndims:   number of dimensions that rotate  (depth >= ndims)
    //   depth:   full head dimension
    //   interleave: false → non-interleave (LLaMA-style half rotation)
    //               true  → interleave     (GPT-NeoX-style pair rotation)
    //
    // Non-interleave:
    //   middle = ndims/2
    //   y[d]        = x[d]        * cos[d]      - x[d+middle] * sin[d]   d < middle
    //   y[d+middle] = x[d+middle] * cos[d]      + x[d]        * sin[d]   d < middle
    //
    // Interleave:
    //   y[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]   i < ndims/2
    //   y[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]   i < ndims/2
    //
    // Elements d in [ndims, depth) are passed through unchanged.
    // -------------------------------------------------------------------------
    template <typename T>
    static void apply_rope_half(T* x,
                                const T* cos_row,
                                const T* sin_row,
                                dim_t ndims,
                                dim_t depth,
                                bool interleave) {
      const dim_t half = ndims / 2;

      if (!interleave) {
        // Non-interleave: load ndims elements, compute, write back.
        // Write back only 2*half elements: for odd ndims, the last unpaired
        // element (index 2*half = ndims-1) is left unchanged rather than
        // silently zeroed (Bug 1.1 fix — matches rotary_cpu.cc behaviour).
        //
        // Stack buffer for small ndims (≤ 512), heap for larger.
        // 2*half floats are needed (one for each rotated element).
        constexpr dim_t kStackLimit = 512;
        float stack_tmp[kStackLimit];
        std::unique_ptr<float[]> heap_tmp;
        float* tmp = stack_tmp;
        if (2 * half > kStackLimit) {
          heap_tmp.reset(new float[2 * half]);
          tmp = heap_tmp.get();
        }
        for (dim_t d = 0; d < half; ++d) {
          const float xd  = float(x[d]);
          const float xp  = float(x[d + half]);
          const float cd  = float(cos_row[d]);
          const float sd  = float(sin_row[d]);
          tmp[d]        = xd * cd - xp * sd;
          tmp[d + half] = xp * cd + xd * sd;
        }
        for (dim_t d = 0; d < 2 * half; ++d) x[d] = T(tmp[d]);
      } else {
        // Interleave: pairs (2i, 2i+1) → can update in-place without temp.
        for (dim_t i = 0; i < half; ++i) {
          const float xe = float(x[2 * i]);
          const float xo = float(x[2 * i + 1]);
          const float ci = float(cos_row[i]);
          const float si = float(sin_row[i]);
          x[2 * i]     = T(xe * ci - xo * si);
          x[2 * i + 1] = T(xo * ci + xe * si);
        }
      }
      // Elements [ndims, depth) are passed through unchanged.
    }

    template <Device D>
    void FlashAttention::compute(StorageView& queries,
                                  StorageView& keys,
                                  StorageView& values,
                                  StorageView& output,
                                  StorageView* cached_keys,
                                  StorageView* cached_values,
                                  StorageView* attention,
                                  bool return_normalized_attention,
                                  StorageView* rotary_cos,
                                  StorageView* rotary_sin,
                                  const bool rotary_interleave,
                                  StorageView* alibi,
                                  dim_t offset,
                                  dim_t beam_size) const {
      // Guards for features not yet supported.
      if (alibi) {
        // ALiBi is applied as a standalone AlibiAdd::compute<METAL> op (M6.4).
        // The layer code (src/layers/flash_attention.cc) always passes nullptr
        // here — this guard is a defensive check only.
        throw std::invalid_argument(
            "Metal FlashAttention: ALiBi via FlashAttention is not supported; "
            "use AlibiAdd::compute<METAL> (M6.4)");
      }
      if (_sliding_window > 0) {
        throw std::invalid_argument(
            "Metal FlashAttention: sliding window is not supported (M6.3 scope)");
      }
      if (return_normalized_attention && attention) {
        throw std::invalid_argument(
            "Metal FlashAttention: attention weight output is not supported (M6.3 scope)");
      }

      // Input shape: [batch, seqlen_q, num_heads, head_dim]
      const dim_t batch_size  = queries.dim(0);
      const dim_t seqlen_q    = queries.dim(1);
      const dim_t num_heads   = queries.dim(2);
      const dim_t head_dim    = queries.dim(3);
      const dim_t num_heads_k = keys.dim(2);

      output.resize(queries.shape());

      if (offset > 0) {
        // -----------------------------------------------------------------------
        // M6.2/M6.3 — KV-cache decode path
        // -----------------------------------------------------------------------
        if (!cached_keys || !cached_values) {
          throw std::invalid_argument(
              "Metal FlashAttention: offset > 0 requires cached_keys and cached_values");
        }
        if (seqlen_q > 1) {
          throw std::invalid_argument(
              "Metal FlashAttention: chunk-prefill (seqlen_q > 1 with offset > 0) "
              "is not supported in M6.3");
        }

        const dim_t seqlen_new   = keys.dim(1);
        const dim_t total_cache  = cached_keys->dim(1);
        const dim_t row_elements = num_heads_k * head_dim;
        const dim_t seqlen_k_eff = offset + seqlen_new;

        const bool need_rope = (rotary_cos != nullptr && rotary_sin != nullptr);

        TYPE_DISPATCH(queries.dtype(), {
          T*       q_ptr   = queries.data<T>();
          T*       k_new   = keys.data<T>();
          T*       k_cache = cached_keys->data<T>();
          T*       v_cache = cached_values->data<T>();
          const T* v_new   = values.data<T>();

          // Two KV-cache update paths:
          //
          // (A) CPU path (bf16 with need_rope):
          //     commit_and_wait → CPU RoPE on Q/K → CPU memcpy to cache.
          //     BF16 uses synchronous MPSGraph for SDPA anyway, so the
          //     commit cost is unavoidable.
          //
          // (B) GPU blit path (f32/f16 with force_layer_rope, need_rope=false):
          //     No CPU access needed — RoPE already applied by the layer's GPU
          //     RotaryEmbeddings kernel.  Encode-only blit_copy for K/V cache.

          if (need_rope) {
            // --- Path (A): CPU RoPE + CPU memcpy ---
            // GPU RoPE was tested (M12.18) but produces FMA-induced rounding
            // differences vs CPU RoPE that compound over 60+ tokens for f16.
            // CPU RoPE is negligible (~0.5µs) so the only cost is the
            // commit_and_wait to flush prior GPU GEMM output.
            metal::commit_and_wait();

            const dim_t half_dim = rotary_cos->dim(1);
            const dim_t ndims    = half_dim * 2;
            const T* cos_row     = rotary_cos->data<T>() + offset * half_dim;
            const T* sin_row     = rotary_sin->data<T>() + offset * half_dim;

            for (dim_t b = 0; b < batch_size; ++b) {
              for (dim_t h = 0; h < num_heads; ++h) {
                T* qvec = q_ptr + (b * seqlen_q * num_heads + h) * head_dim;
                apply_rope_half(qvec, cos_row, sin_row, ndims, head_dim, rotary_interleave);
              }
            }
            for (dim_t b = 0; b < batch_size; ++b) {
              for (dim_t h = 0; h < num_heads_k; ++h) {
                T* kvec = k_new + (b * seqlen_new * num_heads_k + h) * head_dim;
                apply_rope_half(kvec, cos_row, sin_row, ndims, head_dim, rotary_interleave);
              }
            }

            // CPU memcpy K/V to cache (unified memory, zero-copy GPU side).
            {
              const size_t row_bytes = seqlen_new * row_elements * sizeof(T);
              for (dim_t b = 0; b < batch_size; ++b) {
                T*       kd = k_cache + (b * total_cache  + offset) * row_elements;
                T*       vd = v_cache + (b * total_cache  + offset) * row_elements;
                const T* ks = k_new   +  b * seqlen_new * row_elements;
                const T* vs = v_new   +  b * seqlen_new * row_elements;
                std::memcpy(kd, ks, row_bytes);
                std::memcpy(vd, vs, row_bytes);
              }
            }
          } else {
            // --- Path (B): GPU blit copy only (f32 force_layer_rope) ---
            const size_t row_bytes = seqlen_new * row_elements * sizeof(T);
            for (dim_t b = 0; b < batch_size; ++b) {
              void*       kd = k_cache + (b * total_cache  + offset) * row_elements;
              void*       vd = v_cache + (b * total_cache  + offset) * row_elements;
              const void* ks = k_new   +  b * seqlen_new * row_elements;
              const void* vs = v_new   +  b * seqlen_new * row_elements;
              metal::blit_copy(ks, kd, row_bytes);
              metal::blit_copy(vs, vd, row_bytes);
            }
          }

          const bool eff_causal = _is_causal && (seqlen_q > 1);

          metal::sdpa_metal<T>(
              q_ptr, k_cache, v_cache, output.data<T>(),
              batch_size, seqlen_q, seqlen_k_eff,
              num_heads, num_heads_k, head_dim,
              _queries_scale, eff_causal,
              total_cache * num_heads_k * head_dim,
              beam_size);
        });

      } else {
        // -----------------------------------------------------------------------
        // M6.1 — Prefill / no-cache path (offset == 0)
        // rotary_cos/sin == nullptr here: the layer (RotaryEmbeddings::apply)
        // already applied RoPE to Q and K before calling FlashAttention.
        // -----------------------------------------------------------------------
        const dim_t seqlen_k = keys.dim(1);

        TYPE_DISPATCH(queries.dtype(),
                      metal::sdpa_metal<T>(
                          queries.data<T>(), keys.data<T>(), values.data<T>(),
                          output.data<T>(),
                          batch_size, seqlen_q, seqlen_k,
                          num_heads, num_heads_k, head_dim,
                          _queries_scale, _is_causal,
                          /*kv_batch_stride=*/0, beam_size));
      }
    }

    template void FlashAttention::compute<Device::MPS>(
        StorageView&, StorageView&, StorageView&, StorageView&,
        StorageView*, StorageView*, StorageView*, bool,
        StorageView*, StorageView*, const bool, StorageView*, dim_t, dim_t) const;

  }  // namespace ops
}  // namespace ctranslate2
