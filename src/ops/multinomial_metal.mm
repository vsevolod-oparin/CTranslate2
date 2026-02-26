// src/ops/multinomial_metal.mm
//
// M7 — Metal implementation of the Multinomial op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// sample using std::discrete_distribution on shared-memory pointers.
// Multinomial is float-only (input probability distribution).

#include "ctranslate2/ops/multinomial.h"

#include <random>

#include "ctranslate2/random.h"
#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      metal::commit_and_wait();

      const dim_t class_size  = input.dim(-1);
      const dim_t batch_size  = input.size() / class_size;

      const T*      inp = input.data<T>();
      int32_t*      out = output.data<int32_t>();

      auto& generator = get_random_generator();

      for (dim_t i = 0; i < batch_size; ++i) {
        const T*     row_in  = inp + i * class_size;
        int32_t*     row_out = out + i * _sample_size;

        // Build a float copy for discrete_distribution (required to accept
        // a range of double-convertible values).
        std::vector<float> weights(row_in, row_in + class_size);
        std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
        for (dim_t j = 0; j < _sample_size; ++j)
          row_out[j] = dist(generator);
      }
    }

#define DECLARE_IMPL(T)                                                   \
    template void                                                         \
    Multinomial::compute<Device::METAL, T>(const StorageView& input,     \
                                            StorageView& output) const;

    DECLARE_IMPL(float)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
