// src/ops/topk_metal.mm
//
// Metal implementation of the TopK op.
//
// k=1: GPU argmax kernel (encode-only, no commit_and_wait).
//   The caller (Sampler::operator()) syncs via copy_from which
//   calls synchronize_stream(Device::METAL) internally.
//
// k>1: CPU partial_sort is still faster than GPU iterative argmax
//   for typical beam search shapes (k=5, depth=51865).  The GPU
//   kernel exists (topk_k_<T>) but the k sequential passes with
//   barriers cannot beat CPU partial_sort (~33µs) given CB overhead.

#include "ctranslate2/ops/topk.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "metal/ops_metal.h"
#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      DataType* v_data = values.data<DataType>();
      IndexType* i_data = indices.data<IndexType>();

      if (_k == 1) {
        // GPU argmax — encode-only, no sync.
        metal::topk_metal<DataType>(x.data<DataType>(), v_data, i_data,
                                     batch_size, depth);
      } else {
        // CPU fallback for k>1: flush GPU, then partial_sort on shared memory.
        // GPU topk_k kernel exists but is slower for typical shapes
        // (k sequential full-vocab scans ~1ms vs CPU partial_sort ~33µs).
        CT2_COMMIT_AND_WAIT();

        const DataType* x_data = x.data<DataType>();
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
