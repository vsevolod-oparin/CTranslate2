// src/ops/dequantize_metal.mm
//
// M9.1 — Dequantize::dequantize<Device::MPS, int8_t, T>
//      + Dequantize::dequantize_gemm_output<Device::MPS, T>
//
// Thin wrappers that extract dims from StorageViews and delegate to
// metal::dequantize_int8_metal<T>() and metal::dequantize_gemm_output_metal<T>()
// (both implemented in src/metal/ops_quantize.mm).

#include "ctranslate2/ops/dequantize.h"
#include "metal/ops_metal.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

namespace ctranslate2 {
  namespace ops {

    // -----------------------------------------------------------------------
    // dequantize<METAL, int8_t, T>
    //
    // output[row, i] = T(float(input[row, i]) / scales[row])
    // -----------------------------------------------------------------------

    template <Device D, typename InT, typename OutT>
    void Dequantize::dequantize(const StorageView& input,
                                const StorageView& scale,
                                StorageView& output) const {
      const dim_t depth      = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      metal::dequantize_int8_metal<OutT>(
          input.data<InT>(),
          scale.data<float>(),
          output.data<OutT>(),
          batch_size, depth);
    }

    // -----------------------------------------------------------------------
    // dequantize_gemm_output<METAL, T>
    //
    // y[i,j] = c[i,j] / (a_scales[ta?j:i] * b_scales[tb?j:i])
    //        + (bias ? bias[j] : 0)
    // then apply optional activation.
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Dequantize::dequantize_gemm_output(const StorageView& c,
                                            const StorageView& a_scale,
                                            const StorageView& b_scale,
                                            const bool transpose_a,
                                            const bool transpose_b,
                                            const StorageView* bias,
                                            StorageView& y) const {
      const dim_t batch = a_scale.size();
      const dim_t depth = c.dim(-1);

      // Map ActivationType (may be nullptr) to int: -1 = none.
      int act_type = (_activation_type == nullptr)
                         ? -1
                         : static_cast<int>(*_activation_type);

      // When no bias, pass c's buffer as a dummy (kernel reads bias[j] only
      // when has_bias != 0, so any valid Metal-registered pointer is safe).
      const void* bias_ptr = bias
                                 ? static_cast<const void*>(bias->data<T>())
                                 : static_cast<const void*>(c.data<int32_t>());

      metal::dequantize_gemm_output_metal<T>(
          c.data<int32_t>(),
          a_scale.data<float>(),
          b_scale.data<float>(),
          bias_ptr,
          y.data<T>(),
          batch, depth,
          transpose_a, transpose_b,
          bias != nullptr, act_type);
    }

#define DECLARE_IMPL(T)                                                        \
    template void Dequantize::dequantize<Device::MPS, int8_t, T>(           \
        const StorageView&, const StorageView&, StorageView&) const;          \
    template void Dequantize::dequantize_gemm_output<Device::MPS, T>(       \
        const StorageView&, const StorageView&, const StorageView&,           \
        const bool, const bool, const StorageView*, StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
