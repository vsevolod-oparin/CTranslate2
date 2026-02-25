// src/ops/flash_attention_metal.mm
//
// M6.1 — FlashAttention::compute<Device::METAL>.
//
// Implements scaled dot-product attention on the Metal backend.
// Calls metal::sdpa_metal<T> from src/metal/primitives_sdpa.mm.
//
// M6.1 scope: no KV cache (offset must be 0), no rotary embeddings,
// no ALiBi, no sliding window, no attention weight output.

#include "ctranslate2/ops/flash_attention.h"

#include <stdexcept>

#include "metal/ops_metal.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D>
    void FlashAttention::compute(StorageView& queries,
                                  StorageView& keys,
                                  StorageView& values,
                                  StorageView& output,
                                  StorageView* /*cached_keys*/,
                                  StorageView* /*cached_values*/,
                                  StorageView* attention,
                                  bool return_normalized_attention,
                                  StorageView* rotary_cos,
                                  StorageView* rotary_sin,
                                  const bool /*rotary_interleave*/,
                                  StorageView* alibi,
                                  dim_t offset) const {
      // M6.1 scope guards.
      if (offset != 0) {
        throw std::invalid_argument(
            "Metal FlashAttention: KV cache (offset > 0) is not supported in M6.1");
      }
      if (rotary_cos || rotary_sin) {
        throw std::invalid_argument(
            "Metal FlashAttention: rotary embeddings are not supported in M6.1");
      }
      if (alibi) {
        throw std::invalid_argument(
            "Metal FlashAttention: ALiBi is not supported in M6.1");
      }
      if (_sliding_window > 0) {
        throw std::invalid_argument(
            "Metal FlashAttention: sliding window is not supported in M6.1");
      }
      if (return_normalized_attention && attention) {
        throw std::invalid_argument(
            "Metal FlashAttention: attention weight output is not supported in M6.1");
      }

      // Input shape: [batch, seqlen_q, num_heads, head_dim]
      const dim_t batch_size  = queries.dim(0);
      const dim_t seqlen_q    = queries.dim(1);
      const dim_t num_heads   = queries.dim(2);
      const dim_t head_dim    = queries.dim(3);
      const dim_t seqlen_k    = keys.dim(1);
      const dim_t num_heads_k = keys.dim(2);

      output.resize(queries.shape());

      TYPE_DISPATCH(queries.dtype(),
                    metal::sdpa_metal<T>(
                        queries.data<T>(), keys.data<T>(), values.data<T>(),
                        output.data<T>(),
                        batch_size, seqlen_q, seqlen_k,
                        num_heads, num_heads_k, head_dim,
                        _queries_scale, _is_causal));
    }

    template void FlashAttention::compute<Device::METAL>(
        StorageView&, StorageView&, StorageView&, StorageView&,
        StorageView*, StorageView*, StorageView*, bool,
        StorageView*, StorageView*, const bool, StorageView*, dim_t) const;

  }  // namespace ops
}  // namespace ctranslate2
