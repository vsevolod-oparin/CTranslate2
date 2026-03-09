// src/ops/topk_metal.mm
//
// Metal implementation of the TopK op.
//
// k=1: GPU argmax kernel (encode-only).
// k>1: GPU single-pass fused top-k kernel (encode-only).
//
// Both paths are encode-only — no commit_and_wait.
// The caller (Sampler::operator()) syncs via copy_from which
// calls synchronize_stream(Device::METAL) internally.

#include "ctranslate2/ops/topk.h"

#include "metal/ops_metal.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename DataType, typename IndexType>
    void TopK::compute(const StorageView& x,
                       StorageView& values,
                       StorageView& indices) const {
      const dim_t depth = x.dim(-1);
      const dim_t batch_size = x.size() / depth;

      metal::topk_metal<DataType>(x.data<DataType>(),
                                   values.data<DataType>(),
                                   indices.data<IndexType>(),
                                   batch_size, depth, _k);
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
