// src/ops/alibi_add_metal.mm
//
// M6.4 — AlibiAdd::compute<Device::MPS>
//
// Thin wrapper: extracts dimensions from StorageView and calls
// metal::alibi_add_metal<T>() (implemented in src/metal/ops_alibi.mm).
//
// See src/metal/ops_alibi.mm for the MSL kernel and algorithm details.

#include "ctranslate2/ops/alibi_add.h"

#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void AlibiAdd::compute(const StorageView& input,
                           const StorageView& alibi,
                           const dim_t alibi_offset,
                           StorageView& output) const {
      // input: [batch_size, num_heads, query_length, key_length]
      // alibi: [1, num_heads, 1, cached_key_length]
      const dim_t batch_size       = input.dim(0);
      const dim_t num_heads        = input.dim(1);
      const dim_t query_length     = input.dim(2);
      const dim_t key_length       = input.dim(3);
      const dim_t cached_key_length = alibi.dim(-1);

      metal::alibi_add_metal<T>(
          input.data<T>(), alibi.data<T>(), output.data<T>(),
          batch_size, num_heads, query_length, key_length,
          cached_key_length, alibi_offset);
    }

#define DECLARE_IMPL(T)                                                       \
    template void                                                             \
    AlibiAdd::compute<Device::MPS, T>(const StorageView&,                   \
                                        const StorageView&,                   \
                                        const dim_t,                          \
                                        StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)

  }  // namespace ops
}  // namespace ctranslate2
