// src/ops/mean_metal.mm
//
// M7 — Metal implementation of the Mean op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// compute the mean (or sum) in float32 on shared-memory pointers.
// Mean is float-only on both CPU and Metal.

#include "ctranslate2/ops/mean.h"

#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       const bool get_sum,
                       StorageView& output) const {
      CT2_COMMIT_AND_WAIT();

      const T* src = input.data<T>();
      T* dst = output.data<T>();

      for (dim_t i = 0; i < outer_size; ++i) {
        for (dim_t j = 0; j < inner_size; ++j) {
          float sum = 0.f;
          for (dim_t k = 0; k < axis_size; ++k)
            sum += static_cast<float>(src[i * axis_size * inner_size
                                          + k * inner_size + j]);
          dst[i * inner_size + j] = T(get_sum ? sum : sum / static_cast<float>(axis_size));
        }
      }
    }

#define DECLARE_IMPL(T)                                             \
    template void                                                   \
    Mean::compute<Device::MPS, T>(const StorageView& input,       \
                                     const dim_t outer_size,        \
                                     const dim_t axis_size,         \
                                     const dim_t inner_size,        \
                                     const bool get_sum,            \
                                     StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
