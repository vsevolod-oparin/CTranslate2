// src/ops/rotary_metal.mm
//
// M6.3 — Rotary::compute<Device::MPS>
//
// Thin wrapper: extracts dimensions from StorageView and calls
// metal::rotary_metal<T>() (implemented in src/metal/ops_rotary.mm).
//
// See src/metal/ops_rotary.mm for the MSL kernel and algorithm details.

#include "ctranslate2/ops/rotary.h"

#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Rotary::compute(const StorageView& input,
                         const StorageView& sin,
                         const StorageView& cos,
                         StorageView& output,
                         bool is_transposed) const {
      // Dimensions follow the CPU and CUDA kernel convention:
      //   max_time  = is_transposed ? dim(-2) : dim(-3)
      //   head_size = is_transposed ? dim(-3) : dim(-2)   (num_heads)
      //   depth     = dim(-1)
      const dim_t max_time   = is_transposed ? input.dim(-2) : input.dim(-3);
      const dim_t head_size  = is_transposed ? input.dim(-3) : input.dim(-2);
      const dim_t depth      = input.dim(-1);
      const dim_t ndims      = _ndims == 0 ? depth : _ndims;
      const dim_t total_vecs = input.size() / depth;

      metal::rotary_metal<T>(
          input.data<T>(), sin.data<T>(), cos.data<T>(), output.data<T>(),
          total_vecs, depth, ndims, max_time, head_size,
          _interleave, is_transposed);
    }

#define DECLARE_IMPL(T)                                                    \
    template void                                                          \
    Rotary::compute<Device::MPS, T>(const StorageView&,                  \
                                      const StorageView&,                  \
                                      const StorageView&,                  \
                                      StorageView&,                        \
                                      bool) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)

  }  // namespace ops
}  // namespace ctranslate2
