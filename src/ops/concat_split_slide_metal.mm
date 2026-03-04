// src/ops/concat_split_slide_metal.mm
//
// M7 — Metal implementations of Concat, Split, and Slide.
//
// Strategy: GPU-side blit copy via the deferred command buffer.
// No commit_and_wait() needed — all copies are encoded as GPU blit
// commands that execute in-order with preceding compute kernels.

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
    // Concat — GPU blit copy
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Concat::compute(const std::vector<const StorageView*>& inputs,
                         StorageView& output) const {
      const dim_t axis = _axis < 0 ? output.rank() + _axis : _axis;
      const dim_t step_size = output.dim(axis) * output.stride(axis);
      const T* output_base = output.data<T>();
      dim_t output_offset_elems = 0;

      for (const StorageView* inp : inputs) {
        const StorageView& x = *inp;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        const T* x_data = x.data<T>();
        T* out_data = const_cast<T*>(output_base) + output_offset_elems;
        for (dim_t i = 0; i < iter_size; ++i) {
          metal::blit_copy(x_data   + i * copy_size,
                           out_data + i * step_size,
                           copy_size * sizeof(T));
        }
        output_offset_elems += copy_size;
      }
    }

#define DECLARE_IMPL(T)                                                         \
    template void                                                               \
    Concat::compute<Device::METAL, T>(const std::vector<const StorageView*>&,  \
                                       StorageView&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // Split — GPU blit copy
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Split::compute(const StorageView& input,
                        std::vector<StorageView*>& outputs) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t step_size = input.dim(axis) * input.stride(axis);
      const T* input_base = input.data<T>();
      dim_t input_offset_elems = 0;

      for (StorageView* out : outputs) {
        StorageView& x = *out;
        const dim_t copy_size = compute_copy_size(x, axis);
        if (copy_size == 0)
          continue;
        const dim_t iter_size = compute_iter_size(x, axis);
        T* x_data = x.data<T>();
        const T* in_data = input_base + input_offset_elems;
        for (dim_t i = 0; i < iter_size; ++i) {
          metal::blit_copy(in_data + i * step_size,
                           x_data  + i * copy_size,
                           copy_size * sizeof(T));
        }
        input_offset_elems += copy_size;
      }
    }

#define DECLARE_IMPL(T)                                                       \
    template void                                                             \
    Split::compute<Device::METAL, T>(const StorageView&,                     \
                                      std::vector<StorageView*>&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

    // -----------------------------------------------------------------------
    // Slide — GPU blit copy
    // -----------------------------------------------------------------------

    template <Device D, typename T>
    void Slide::compute(const StorageView& input,
                        StorageView& output,
                        const dim_t& index) const {
      const dim_t axis = _axis < 0 ? input.rank() + _axis : _axis;
      const dim_t stride_axis = input.stride(axis) == 0 ? 1 : input.stride(axis);
      const dim_t step_size = input.dim(axis) * stride_axis;

      const T* input_data = input.data<T>() + index * stride_axis;
      T* x_data = output.data<T>();

      const dim_t copy_size = compute_copy_size(output, axis);
      if (copy_size == 0)
        return;
      const dim_t iter_size = compute_iter_size(output, axis);

      for (dim_t i = 0; i < iter_size; ++i) {
        metal::blit_copy(input_data + i * step_size,
                         x_data     + i * copy_size,
                         copy_size * sizeof(T));
      }
    }

#define DECLARE_IMPL(T)                                                         \
    template void                                                               \
    Slide::compute<Device::METAL, T>(const StorageView&, StorageView&,          \
                                      const dim_t&) const;

    DECLARE_ALL_TYPES(DECLARE_IMPL)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
