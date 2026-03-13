// src/ops/tile_metal.mm
//
// M7 — Metal implementation of the Tile op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// CPU-side memcpy in a nested outer × num_tiles loop.  Metal buffers
// use MTLResourceStorageModeShared (unified memory) so no device copy
// is required.
//
// GPU opportunity assessment (M15.5):
//   Call sites: KV-cache replication for beam expansion (language_model.cc, axis=0,
//   repeats=batch_size ≈ 2–8) and GQA head replication (attention.cc, axis=2,
//   repeats=num_heads/num_kv_heads ≈ 4–8).  Both are small repeat counts over
//   contiguous memory — std::memcpy on unified memory is effectively a DMA at
//   ~50 GB/s, competitive with a GPU blit.  The commit_and_wait() pipeline stall
//   is the real cost, but Tile is called infrequently (once per generation init
//   for KV-cache, once per layer for GQA).  A GPU blit_copy version could avoid
//   the stall but adds complexity for marginal gain.  Low priority.

#include "ctranslate2/ops/tile.h"

#include <cstring>

#include "metal/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Tile::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t inner_size,
                       StorageView& output) const {
      CT2_COMMIT_AND_WAIT();

      const T* src = input.data<T>();
      T* dst = output.data<T>();

      for (dim_t i = 0; i < outer_size; ++i) {
        for (dim_t t = 0; t < _num_tiles; ++t) {
          std::memcpy(dst, src, inner_size * sizeof(T));
          dst += inner_size;
        }
        src += inner_size;
      }
    }

#define DECLARE_IMPL(T)                                           \
    template void                                                 \
    Tile::compute<Device::MPS, T>(const StorageView& input,    \
                                     const dim_t outer_size,     \
                                     const dim_t inner_size,     \
                                     StorageView& output) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
