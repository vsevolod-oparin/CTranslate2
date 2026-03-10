// src/ops/bias_add_metal.mm
//
// M5.2 — Metal implementation of the BiasAdd op.
//
// Reuses the already-implemented Metal broadcast primitives:
//   primitives<Device::MPS>::add_batch_broadcast  (last-axis bias)
//   primitives<Device::MPS>::add_block_broadcast  (other axes)

#include "ctranslate2/ops/bias_add.h"
#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/add.h"
#include "ctranslate2/primitives.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void BiasAdd::compute(const StorageView& value,
                          const StorageView& bias,
                          StorageView& output,
                          const StorageView* residual) const {
      if (_axis == -1 || _axis == value.rank() - 1) {
        primitives<D>::add_batch_broadcast(bias.data<T>(),
                                           value.data<T>(),
                                           output.data<T>(),
                                           bias.size(),
                                           value.size());
      } else {
        const dim_t axis = _axis < 0 ? value.rank() + _axis : _axis;
        dim_t width = 1;
        for (dim_t i = axis + 1; i < value.rank(); ++i)
          width *= value.dim(i);
        primitives<D>::add_block_broadcast(bias.data<T>(),
                                           value.data<T>(),
                                           output.data<T>(),
                                           width,
                                           bias.size(),
                                           value.size());
      }
      if (residual)
        Add()(*residual, output, output);
      if (_activation_type)
        get_activation_op(*_activation_type)(output, output);
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    BiasAdd::compute<Device::MPS, T>(const StorageView& value,\
                                        const StorageView& bias,\
                                        StorageView& output,    \
                                        const StorageView* residual) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
