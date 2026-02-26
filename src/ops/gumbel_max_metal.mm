// src/ops/gumbel_max_metal.mm
//
// M7 — Metal implementation of GumbelMax::add_gumbel_noise.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// add Gumbel noise using the process-wide CPU random generator on
// shared-memory pointers.  GumbelMax is float-only.  The TopK step is
// dispatched by GumbelMax::operator() via the TopK op (which now has a
// Metal specialization in topk_metal.mm).

#include "ctranslate2/ops/gumbel_max.h"

#include <cmath>
#include <limits>
#include <random>

#include "ctranslate2/random.h"
#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void GumbelMax::add_gumbel_noise(const StorageView& x, StorageView& y) const {
      metal::commit_and_wait();

      auto& generator = get_random_generator();
      std::uniform_real_distribution<float> distribution(
          std::numeric_limits<float>::min(), 1.f);

      const T* src = x.data<T>();
      T* dst = y.data<T>();

      for (dim_t i = 0; i < x.size(); ++i) {
        const float z = -std::log(distribution(generator));
        dst[i] = T(static_cast<float>(src[i]) + z);
      }
    }

#define DECLARE_IMPL(T)                                                   \
    template void                                                         \
    GumbelMax::add_gumbel_noise<Device::METAL, T>(const StorageView& x,  \
                                                   StorageView& y) const;

    DECLARE_IMPL(float)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
