// src/ops/topk_metal.mm
//
// M7 — Metal implementation of the TopK op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// execute the CPU sorting algorithm directly on the shared-memory
// pointers.  Metal buffers use MTLResourceStorageModeShared (unified
// memory) so no device copy is needed.  Result is written directly to
// the output Metal buffers; the next GPU op will see the data without
// an additional flush.
//
// Explicit cast to float32 is used in all comparisons so the code
// compiles correctly for bfloat16_t (which has operator float() but no
// comparison operators) as well as float and float16_t.

#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      CT2_COMMIT_AND_WAIT();

      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      const DataType* x_data = x.data<DataType>();
      DataType* v_data = values.data<DataType>();
      IndexType* i_data = indices.data<IndexType>();

      if (_k == 1) {
        for (dim_t i = 0; i < batch_size; ++i) {
          const DataType* row = x_data + i * depth;
          const DataType* mx = std::max_element(row, row + depth,
              [](const DataType& a, const DataType& b) {
                return static_cast<float>(a) < static_cast<float>(b);
              });
          v_data[i] = *mx;
          i_data[i] = static_cast<IndexType>(std::distance(row, mx));
        }
      } else {
        std::vector<IndexType> ids(static_cast<std::size_t>(depth));
        for (dim_t i = 0; i < batch_size; ++i) {
          const DataType* inp = x_data + i * depth;
          DataType* val = v_data + i * _k;
          IndexType* ind = i_data + i * _k;

          std::iota(ids.begin(), ids.end(), IndexType(0));
          std::partial_sort(ids.begin(), ids.begin() + _k, ids.end(),
              [&inp](const IndexType& i1, const IndexType& i2) {
                return static_cast<float>(inp[i1]) > static_cast<float>(inp[i2]);
              });
          for (dim_t j = 0; j < _k; ++j) {
            ind[j] = ids[static_cast<std::size_t>(j)];
            val[j] = inp[ind[j]];
          }
        }
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    TopK::compute<Device::METAL, T, int32_t>(const StorageView& x,     \
                                              StorageView& values,      \
                                              StorageView& indices) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
