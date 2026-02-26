// src/ops/flash_attention_metal.mm
//
// M6.1 — FlashAttention::compute<Device::METAL>.
// M6.2 — KV-cache update (offset > 0 path).
//
// Implements scaled dot-product attention on the Metal backend.
// Calls metal::sdpa_metal<T> from src/metal/ops_sdpa.mm.
//
// M6.1 scope (offset == 0): SDPA over full Q/K/V without cache.
// M6.2 scope (offset > 0):  KV-cache update + SDPA decode step.
//   Algorithm:
//     1. commit_and_wait()  — flush any pending GPU writes so CPU can read
//                             the new keys/values (written by prior linear ops).
//     2. CPU memcpy         — write new K/V into cached_keys/values at position
//                             `offset` (unified memory, zero-copy GPU side).
//     3. sdpa_metal         — attend Q over the full valid cache [0, offset+seqlen_new).
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
// Still unsupported in M6.2 (guard throws):
//   rotary embeddings, ALiBi, sliding window, attention weight output,
//   seqlen_q > 1 with offset > 0 (chunk-prefill into KV cache).

#include "ctranslate2/ops/flash_attention.h"

#include <cstring>
#include <stdexcept>

#include "metal/ops_metal.h"
#include "metal/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

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
                                  const bool /*rotary_interleave*/,
                                  StorageView* alibi,
                                  dim_t offset) const {
      // Guards for features not yet supported.
      if (rotary_cos || rotary_sin) {
        throw std::invalid_argument(
            "Metal FlashAttention: rotary embeddings are not supported (M6.2 scope)");
      }
      if (alibi) {
        throw std::invalid_argument(
            "Metal FlashAttention: ALiBi is not supported (M6.2 scope)");
      }
      if (_sliding_window > 0) {
        throw std::invalid_argument(
            "Metal FlashAttention: sliding window is not supported (M6.2 scope)");
      }
      if (return_normalized_attention && attention) {
        throw std::invalid_argument(
            "Metal FlashAttention: attention weight output is not supported (M6.2 scope)");
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
        // M6.2 — KV-cache decode path
        // -----------------------------------------------------------------------
        if (!cached_keys || !cached_values) {
          throw std::invalid_argument(
              "Metal FlashAttention: offset > 0 requires cached_keys and cached_values");
        }
        if (seqlen_q > 1) {
          throw std::invalid_argument(
              "Metal FlashAttention: chunk-prefill (seqlen_q > 1 with offset > 0) "
              "is not supported in M6.2");
        }

        // Flush pending GPU writes (linear projections wrote keys/values via
        // GPU kernels; CPU must see the results before the memcpy below).
        metal::commit_and_wait();

        const dim_t seqlen_new   = keys.dim(1);
        const dim_t total_cache  = cached_keys->dim(1);
        const dim_t row_elements = static_cast<dim_t>(num_heads_k) * head_dim;
        const dim_t seqlen_k_eff = offset + seqlen_new;

        // Write new K/V into cache at position `offset` (one batch at a time).
        TYPE_DISPATCH(queries.dtype(), {
          T*       k_cache = cached_keys->data<T>();
          T*       v_cache = cached_values->data<T>();
          const T* k_new   = keys.data<T>();
          const T* v_new   = values.data<T>();

          for (dim_t b = 0; b < batch_size; ++b) {
            T*       kd = k_cache + (b * total_cache  + offset) * row_elements;
            T*       vd = v_cache + (b * total_cache  + offset) * row_elements;
            const T* ks = k_new   +  b * seqlen_new * row_elements;
            const T* vs = v_new   +  b * seqlen_new * row_elements;
            std::memcpy(kd, ks, seqlen_new * row_elements * sizeof(T));
            std::memcpy(vd, vs, seqlen_new * row_elements * sizeof(T));
          }

          // Decode with sq==1: causal mask is irrelevant — all cached tokens
          // are "in the past" of the current query.
          const bool eff_causal = _is_causal && (seqlen_q > 1);

          metal::sdpa_metal<T>(
              queries.data<T>(), k_cache, v_cache, output.data<T>(),
              batch_size, seqlen_q, seqlen_k_eff,
              num_heads, num_heads_k, head_dim,
              _queries_scale, eff_causal);
        });

      } else {
        // -----------------------------------------------------------------------
        // M6.1 — Prefill / no-cache path (offset == 0)
        // -----------------------------------------------------------------------
        const dim_t seqlen_k = keys.dim(1);

        TYPE_DISPATCH(queries.dtype(),
                      metal::sdpa_metal<T>(
                          queries.data<T>(), keys.data<T>(), values.data<T>(),
                          output.data<T>(),
                          batch_size, seqlen_q, seqlen_k,
                          num_heads, num_heads_k, head_dim,
                          _queries_scale, _is_causal));
      }
    }

    template void FlashAttention::compute<Device::METAL>(
        StorageView&, StorageView&, StorageView&, StorageView&,
        StorageView*, StorageView*, StorageView*, bool,
        StorageView*, StorageView*, const bool, StorageView*, dim_t) const;

  }  // namespace ops
}  // namespace ctranslate2
