// src/ops/concat_split_slide_metal.mm
//
// M7 — Metal implementations of Concat, Split, and Slide.
//
// Strategy: commit_and_wait() to flush any pending GPU writes into shared
// memory, then do sequential CPU-side memcpy.  Metal buffers use
// MTLResourceStorageModeShared (unified memory), so the contents pointer
// is simultaneously valid for both CPU and GPU.  After the memcpy the
// written data is immediately visible to the next GPU op encoded into the
// fresh command buffer.

#include "ctranslate2/ops/concat.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/ops/slide.h"

#include <cstring>

#include "metal/utils.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace ops {

    // -----------------------------------------------------------------------
    // Local helpers — same semantics as the private statics in
    // concat_split_slide_cpu.cc.
    // -----------------------------------------------------------------------

    static dim_t compute_copy_size(const StorageView& x, dim_t axis) {
      dim_t s = 1;
      for (dim_t i = axis; i < x.rank(); ++i)
        s *= x.dim(i);
      return s;
    }

    static dim_t compute_iter_size(const StorageView& x, dim_t axis) {
      dim_t s = 1;
      for (dim_t i = 0; i < axis; ++i)
        s *= x.dim(i);
      return s;
    }

    // -----------------------------------------------------------------------
    // Concat
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      metal::commit_and_wait();

      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      T* output_data = output.data<T>();

      for (const StorageView* inp : inputs) {
        const StorageView& x = *inp;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        const T* x_data = x.data<T>();
        for (dim_t i = 0; i < iter_size; ++i)
          std::memcpy(output_data + i * step_size,
                      x_data     + i * copy_size,
                      copy_size * sizeof(T));
        output_data += copy_size;
      }
    }

#define DECLARE_IMPL(T)                                                         \
    template void                                                               \
    Concat::compute<Device::METAL, T>(const std::vector<const StorageView*>&,  \
                                       StorageView&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // Split
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      metal::commit_and_wait();

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* input_data = input.data<T>();

      for (StorageView* out : outputs) {
        StorageView& x = *out;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        T* x_data = x.data<T>();
        for (dim_t i = 0; i < iter_size; ++i)
          std::memcpy(x_data     + i * copy_size,
                      input_data + i * step_size,
                      copy_size * sizeof(T));
        input_data += copy_size;
      }
    }

#define DECLARE_IMPL(T)                                                       \
    template void                                                             \
    Split::compute<Device::METAL, T>(const StorageView&,                     \
                                      std::vector<StorageView*>&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // Slide
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Slide::compute(const StorageView& input,
                        StorageView& output,
                        const dim_t& index) const {
      metal::commit_and_wait();

      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t step_size = input.dim(axis) * stride_axis;

      const T* input_data = input.data<T>() + index * stride_axis;
      T* x_data = output.data<T>();

      const dim_t copy_size = compute_copy_size(output, axis);
      if (copy_size == 0)
        return;
      const dim_t iter_size = compute_iter_size(output, axis);

      for (dim_t i = 0; i < iter_size; ++i)
        std::memcpy(x_data    + i * copy_size,
                    input_data + i * step_size,
                    copy_size * sizeof(T));
    }

#define DECLARE_IMPL(T)                                                         \
    template void                                                               \
    Slide::compute<Device::METAL, T>(const StorageView&, StorageView&,          \
                                      const dim_t&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
