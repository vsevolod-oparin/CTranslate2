// src/ops/gather_metal.mm
//
// M5.2 — Metal implementation of the Gather op.
//
// Uses the same unspecialized template-body pattern as *_gpu.cu files.
// Only axis == batch_dims is supported (same constraint as the CPU path).

#include "ctranslate2/ops/gather.h"

#include <stdexcept>

#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Gather::compute(const StorageView& data,
                         const StorageView& input,
                         const dim_t axis,
                         const dim_t batch_dims,
                         StorageView& output) const {
      if (axis != batch_dims)
        throw std::invalid_argument(
            "Metal Gather: only indexing the first non-batch dimension is supported");

      const dim_t copy_size             = data.stride(axis);
      const dim_t batch_stride          = axis > 0 ? data.stride(axis - 1) : data.size();
      const dim_t batch_size            = data.size() / batch_stride;
      const dim_t num_indices           = input.size();
      const dim_t num_indices_per_batch = num_indices / batch_size;
      const dim_t total_elements        = num_indices * copy_size;

      metal::gather_metal<T>(data.data<T>(),
                              output.data<T>(),
                              input.data<int32_t>(),
                              copy_size,
                              batch_stride,
                              num_indices_per_batch,
                              total_elements);
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Gather::compute<Device::METAL, T>(const StorageView& data,  \
                                       const StorageView& input,\
                                       const dim_t axis,        \
                                       const dim_t batch_dims,  \
                                       StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)
    DECLARE_IMPL(int32_t)
    DECLARE_IMPL(int16_t)
    DECLARE_IMPL(int8_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
