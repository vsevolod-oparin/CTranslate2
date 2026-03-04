#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Gather : public BinaryOp {
    public:
      Gather(const dim_t axis = 0, const dim_t batch_dims = 0);
      using BinaryOp::operator();

      void operator()(StorageView& data, const StorageView& input) const;
      void operator()(const StorageView& data,
                      const StorageView& input,
                      StorageView& output) const override;

      // Batch multiple in-place gathers (axis=0) into one GPU submission.
      // Metal: encodes all gathers, then one commit.
      // Non-Metal: falls back to sequential gathers.
      static void batch_gather_in_place(std::vector<StorageView*>& data_views,
                                        const StorageView& indices);

    private:
      template <Device D, typename T>
      void compute(const StorageView& data,
                   const StorageView& input,
                   const dim_t axis,
                   const dim_t batch_dims,
                   StorageView& output) const;

      const dim_t _axis;
      const dim_t _batch_dims;
    };

  }
}
