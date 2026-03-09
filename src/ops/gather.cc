#include "ctranslate2/ops/gather.h"

#include <algorithm>
#include <vector>

#include "ctranslate2/devices.h"
#include "dispatch.h"

#ifdef CT2_WITH_METAL
#include "metal/ops_metal.h"
#include "metal/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    static inline Shape compute_output_shape(const StorageView& data,
                                             const StorageView& input,
                                             const dim_t axis) {
      Shape output_shape(input.shape());
      for (dim_t i = axis + 1; i < data.rank(); ++i)
        output_shape.push_back(data.dim(i));
      return output_shape;
    }

    static bool support_gather_batch_inplace(const StorageView& data, const StorageView& input) {
      // We can gather in place if the output is not larger than data and indices are in
      // strictly increasing order (i.e. we never need to gather from a previous index).
      const auto* input_begin = input.data<int32_t>();
      const auto* input_end = input_begin + input.size();
      return (input.device() == Device::CPU
              && input.size() <= data.dim(0)
              && std::adjacent_find(input_begin, input_end, std::greater_equal<int32_t>()) == input_end);
    }

    template <typename T>
    void gather_batch_inplace(StorageView& data, const StorageView& input) {
      const auto* indices = input.data<int32_t>();
      auto* dst = data.data<T>();
      const auto* src = dst;
      const auto copy_dim = data.stride(0);
      for (dim_t i = 0; i < input.size(); ++i) {
        const dim_t index = indices[i];
        if (index != i)
          primitives<Device::CPU>::copy(src + index * copy_dim, dst, copy_dim);
        dst += copy_dim;
      }
    }


    Gather::Gather(const dim_t axis, const dim_t batch_dims)
      : _axis(axis)
      , _batch_dims(batch_dims) {
    }

    void Gather::operator()(StorageView& data, const StorageView& input) const {
      if (_axis == 0 && _batch_dims == 0 && support_gather_batch_inplace(data, input)) {
        PROFILE("Gather");
        TYPE_DISPATCH(data.dtype(), (gather_batch_inplace<T>(data, input)));
        data.resize(compute_output_shape(data, input, _axis));
      } else {
        StorageView clone(std::move(data));
        operator()(clone, input, data);
#ifdef CT2_WITH_METAL
        if (data.device() == Device::METAL) {
          // M11.21: Protect clone and indices from premature reuse.
          metal::protect_buffer(clone.buffer());
          if (input.device() == Device::METAL)
            metal::protect_buffer(input.buffer());
        }
#endif
      }
    }

    void Gather::operator()(const StorageView& data,
                            const StorageView& input,
                            StorageView& output) const {
      PROFILE("Gather");

      if (_batch_dims > 0) {
        if (data.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of data should greater than or equal to "
                                      + std::to_string(_batch_dims));
        if (input.rank() < _batch_dims)
          throw std::invalid_argument("Gather: rank of input should greater than or equal to "
                                      + std::to_string(_batch_dims));

        const auto& data_shape = data.shape();
        const auto& input_shape = input.shape();
        if (!std::equal(data_shape.begin(),
                        data_shape.begin() + _batch_dims,
                        input_shape.begin()))
          throw std::invalid_argument("Gather: first " + std::to_string(_batch_dims)
                                      + " dimensions of data and input should match");
      }

      const dim_t axis = _axis < 0 ? data.rank() + _axis : _axis;
      output.resize(compute_output_shape(data, input, axis));
      DEVICE_AND_TYPE_DISPATCH(data.device(), data.dtype(),
                               (compute<D, T>(data, input, axis, _batch_dims, output)));
    }


    void Gather::batch_gather_in_place(std::vector<StorageView*>& data_views,
                                       const StorageView& indices) {
      if (data_views.empty())
        return;

#ifdef CT2_WITH_METAL
      if (data_views[0]->device() == Device::METAL) {
        // M11.1: encode all gathers, then one commit.
        // 1. Clone all data views — clones hold the source data.
        std::vector<StorageView> clones;
        clones.reserve(data_views.size());
        for (auto* view : data_views)
          clones.emplace_back(std::move(*view));

        // 2. For each clone+output pair, resize output and encode gather.
        for (size_t i = 0; i < data_views.size(); ++i) {
          StorageView& src = clones[i];
          StorageView& dst = *data_views[i];
          const dim_t axis = 0;

          // Compute output shape (same logic as compute_output_shape with axis=0).
          Shape output_shape(indices.shape());
          for (dim_t d = axis + 1; d < src.rank(); ++d)
            output_shape.push_back(src.dim(d));
          dst.resize(output_shape);

          // Gather parameters (axis=0, batch_dims=0).
          const dim_t copy_size             = src.stride(axis);
          const dim_t batch_stride          = src.size();
          const dim_t num_indices           = indices.size();
          const dim_t num_indices_per_batch = num_indices;
          const dim_t total_elements        = num_indices * copy_size;

          TYPE_DISPATCH(src.dtype(),
            (metal::gather_metal<T>(
                src.data<T>(),
                dst.data<T>(),
                indices.data<int32_t>(),
                copy_size, batch_stride,
                num_indices_per_batch, total_elements)));
        }

        // 3. M11.21: Protect all GPU-referenced buffers from premature reuse.
        //    Clone buffers: read by GPU gather kernels.
        //    Indices buffer: read by GPU gather kernels (indices may be freed
        //    by the caller before the next commit_and_wait).
        for (auto& clone : clones)
          metal::protect_buffer(clone.buffer());
        if (indices.device() == Device::METAL)
          metal::protect_buffer(indices.buffer());
        return;
      }
#endif

      // Fallback: sequential gathers for non-Metal devices.
      const Gather gather;
      for (auto* view : data_views)
        gather(*view, indices);
    }

  }
}
