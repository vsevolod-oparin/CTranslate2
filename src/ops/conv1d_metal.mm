// src/ops/conv1d_metal.mm
//
// M8.3 — Conv1D for Device::METAL.
//
// Thin wrapper that:
//   1. Validates groups == 1.
//   2. Delegates to metal::conv1d_metal<T>() (src/metal/ops_conv1d.mm) for
//      the im2col + GEMM pass.
//   3. Applies bias + activation via apply_bias_and_activation.
//
// Only groups == 1 is supported (Whisper uses groups == 1 throughout).
// The Metal implementation requires no CPU primitives or CPU dispatch paths.

#include "ctranslate2/ops/conv1d.h"
#include "ctranslate2/ops/gemm.h"
#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv1D::compute(
        const StorageView& input,
        const StorageView& weight,
        const StorageView* bias,
        StorageView& output,
        const StorageView*) const {
      if (_groups != 1)
        throw std::invalid_argument(
            "Metal Conv1D: only groups=1 is supported (groups=" +
            std::to_string(_groups) + ")");

      const dim_t B     = input.dim(0);
      const dim_t C_in  = input.dim(1);
      const dim_t T_in  = input.dim(2);
      const dim_t C_out = weight.dim(0);
      const dim_t K     = weight.dim(2);
      const dim_t T_out = output.dim(2);
      if (_dilation < 1)
        throw std::invalid_argument(
            "Metal Conv1D: dilation must be >= 1 (got " +
            std::to_string(_dilation) + ")");
      const dim_t dil = _dilation;

      metal::conv1d_metal<T>(input.data<T>(), weight.data<T>(), output.data<T>(),
                              B, C_in, T_in, C_out, K, T_out,
                              _stride, _padding, dil);

      apply_bias_and_activation(output, bias, _activation_type, nullptr, -2);
    }

#define DECLARE_IMPL(T)                                           \
    template void                                                 \
    Conv1D::compute<Device::METAL, T>(const StorageView& input,  \
                                      const StorageView& weight, \
                                      const StorageView* bias,   \
                                      StorageView& output,       \
                                      const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
