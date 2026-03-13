// src/ops/topp_mask_metal.mm
//
// M7 — Metal implementation of the TopPMask op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// execute the nucleus sampling mask CPU algorithm on shared-memory
// pointers.  TopPMask is float-only (same constraint as CPU).
//
// max_num_classes<Device::MPS> returns numeric_limits<dim_t>::max()
// (same as CPU — no artificial vocabulary limit on Metal).
//
// GPU opportunity assessment (M15.5, see also M12.15 report):
//   Call site: RandomSampler::sample() when sampling_topp < 1.0 (sampling.cc).
//   NOT invoked during beam search (the primary benchmark).  A GPU version would
//   need a radix sort + parallel prefix sum over vocab_size (~32K–128K), which is
//   significant MSL complexity.  M12.15 explicitly evaluated and rejected this:
//   even in sampling mode, the sort is ~5 ms once per step vs ~10–20 ms GEMM time,
//   and the encode-only dispatch pattern already amortizes kernel launch costs.
//   NOT worth implementing unless nucleus sampling becomes a bottleneck.

#include "ctranslate2/ops/topp_mask.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <vector>

#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void TopPMask::compute(const StorageView& input,
                           const StorageView& probs,
                           StorageView& output) const {
      CT2_COMMIT_AND_WAIT();

      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      const T* x    = input.data<T>();
      const T* prob = probs.data<T>();
      T*       y    = output.data<T>();

      std::vector<dim_t> ids(static_cast<std::size_t>(depth));

      for (dim_t i = 0; i < batch_size; ++i) {
        const T* x_i    = x    + i * depth;
        const T* prob_i = prob + i * depth;
        T*       y_i    = y    + i * depth;

        std::iota(ids.begin(), ids.end(), dim_t(0));
        std::sort(ids.begin(), ids.end(), [&prob_i](const dim_t a, const dim_t b) {
          return static_cast<float>(prob_i[a]) > static_cast<float>(prob_i[b]);
        });

        float total_p = 0.f;
        for (const auto id : ids) {
          y_i[id] = total_p < _p ? x_i[id] : T(_mask_value);
          total_p += static_cast<float>(prob_i[id]);
        }
      }
    }

    template <>
    dim_t TopPMask::max_num_classes<Device::MPS>() {
      return std::numeric_limits<dim_t>::max();
    }

#define DECLARE_IMPL(T)                                                     \
    template void TopPMask::compute<Device::MPS, T>(const StorageView&,   \
                                                       const StorageView&,   \
                                                       StorageView&) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
