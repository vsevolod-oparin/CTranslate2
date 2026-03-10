// src/ops/tile_metal.mm
//
// M7 — Metal implementation of the Tile op.
//
// Strategy: commit_and_wait() to flush pending GPU writes, then
// CPU-side memcpy in a nested outer × num_tiles loop.  Metal buffers
// use MTLResourceStorageModeShared (unified memory) so no device copy
// is required.

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
