// src/ops/quantize_metal.mm
//
// M9.1 — Quantize::quantize<Device::MPS, T, int8_t>
//
// Thin wrapper that extracts batch_size/depth from StorageView and delegates
// to metal::quantize_int8_metal<T>() (src/metal/ops_quantize.mm).
//
// INT8 path only (quantize.cc enforces that; INT16 stays on CPU).
// _shift_to_uint8 is not supported on Metal (throws invalid_argument).

#include "ctranslate2/ops/quantize.h"
#include "metal/ops_metal.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename InT, typename OutT>
    void Quantize::quantize(const StorageView& input,
                            StorageView& output,
                            StorageView& scale) const {
      if (_shift_to_uint8)
        throw std::invalid_argument("Quantize: shift_to_uint8 is not supported on Metal");

      const dim_t batch_size = scale.size();
      const dim_t depth      = input.dim(-1);

      metal::quantize_int8_metal<InT>(
          input.data<InT>(),
          output.data<OutT>(),
          scale.data<float>(),
          batch_size, depth);
    }

#define DECLARE_IMPL(T)                                                        \
    template void Quantize::quantize<Device::MPS, T, int8_t>(               \
        const StorageView&, StorageView&, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
