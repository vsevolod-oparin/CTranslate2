// src/ops/median_filter_metal.mm
//
// M7 — Metal implementation of the MedianFilter op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// apply the sliding median filter on shared-memory pointers using
// std::nth_element.  MedianFilter is float-only.
//
// GPU opportunity assessment (M15.5):
//   Call site: Whisper alignment only (whisper.cc).  Called once per audio
//   segment, NOT in the decode loop.  Typical shape: [batch, ~tokens, ~3000 frames],
//   width=7.  The CPU takes ~6 ms (M7 benchmark).  Immediately followed by
//   Mean + synchronize_stream + move_to(CPU) for DTW, so the pipeline stall
//   from commit_and_wait() is fully masked by the subsequent CPU sync.
//   A GPU bitonic-sort-based median filter is possible but complex for
//   negligible end-to-end impact.  NOT worth implementing.

#include "ctranslate2/ops/median_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void MedianFilter::compute(const StorageView& input,
                               const dim_t axis_size,
                               StorageView& output) const {
      CT2_COMMIT_AND_WAIT();

      const T* src = input.data<T>();
      T* dst = output.data<T>();

      const dim_t depth = axis_size;
      const dim_t batch_size = input.size() / depth;
      const dim_t rank = _width / 2;

      if (depth <= rank)
        return;

      std::vector<float> window(static_cast<std::size_t>(_width));

      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t offset = i * depth;
        const T* in = src + offset;
        T*       out = dst + offset;

        for (dim_t j = 0; j < depth; ++j) {
          for (dim_t k = -rank; k <= rank; ++k) {
            dim_t read = std::abs(j + k);
            if (read >= depth)
              read = depth - (read - depth) - 2;
            window[static_cast<std::size_t>(k + rank)] =
                static_cast<float>(in[read]);
          }
          std::nth_element(window.begin(),
                           window.begin() + rank,
                           window.end());
          out[j] = T(window[static_cast<std::size_t>(rank)]);
        }
      }
    }

#define DECLARE_IMPL(T)                                                     \
    template void                                                           \
    MedianFilter::compute<Device::MPS, T>(const StorageView& input,      \
                                             const dim_t axis_size,         \
                                             StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
