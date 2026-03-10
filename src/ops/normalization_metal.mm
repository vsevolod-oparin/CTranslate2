// src/ops/normalization_metal.mm
//
// M5.2 — Metal implementations of LayerNorm, RMSNorm, and SoftMax.
//
// Uses the same unspecialized template-body pattern as *_gpu.cu files:
// the template is defined here for all (D, T), but only explicit
// instantiations for Device::MPS are emitted, so the linker always
// resolves METAL calls to this TU.
//
// Constraints:
//   LayerNorm: only last-axis normalization (inner_size == 1) is supported.
//   RMSNorm:   use_residual = true is not supported on Metal.

#include "ctranslate2/ops/layer_norm.h"
#include "ctranslate2/ops/rms_norm.h"
#include "ctranslate2/ops/softmax.h"

#include <stdexcept>

#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    // -----------------------------------------------------------------------
    // LayerNorm
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t /*axis*/,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      // The Metal kernel lays out each row as a contiguous block of axis_size
      // elements: row_off = tgid * axis_size.  This is valid when all dimensions
      // trailing the normalization axis are size 1 (inner_size == 1), because
      // then consecutive rows are adjacent in memory regardless of whether axis
      // is literally the last dimension.  The equivalent check is inner_size == 1,
      // which is more permissive than "axis == rank-1" while remaining correct.
      if (inner_size != 1)
        throw std::invalid_argument(
            "Metal LayerNorm: only normalization over a memory-contiguous axis "
            "is supported (all dimensions after the normalization axis must be 1, "
            "i.e. inner_size == 1)");
      metal::layer_norm_metal<T>(input.data<T>(),
                                  gamma ? gamma->data<T>() : nullptr,
                                  beta  ? beta->data<T>()  : nullptr,
                                  output.data<T>(),
                                  outer_size, axis_size, _epsilon);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::MPS, T>(const StorageView* beta,       \
                                         const StorageView* gamma,      \
                                         const StorageView& input,      \
                                         const dim_t axis,              \
                                         const dim_t outer_size,        \
                                         const dim_t axis_size,         \
                                         const dim_t inner_size,        \
                                         StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // RMSNorm
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      if (_use_residual)
        throw std::invalid_argument(
            "Metal RMSNorm: use_residual is not supported on Metal");
      const dim_t depth      = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      metal::rms_norm_metal<T>(input.data<T>(), gamma.data<T>(), output.data<T>(),
                                batch_size, depth, _epsilon);
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    RMSNorm::compute<Device::MPS, T>(const StorageView&,      \
                                        const StorageView&,     \
                                        StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // SoftMax (also covers LogSoftMax via the _log flag)
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth      = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      metal::softmax_metal<T>(input.data<T>(),
                               lengths ? lengths->data<int32_t>() : nullptr,
                               output.data<T>(),
                               batch_size, depth, _log);
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::MPS, T>(const StorageView& input,        \
                                        const StorageView* lengths,     \
                                        StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    DECLARE_IMPL(bfloat16_t)

#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
