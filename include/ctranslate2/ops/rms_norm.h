#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class RMSNorm : public Op {
    public:
      // use_residual=true is not supported on Device::METAL and will throw
      // std::invalid_argument at runtime when that backend is used.
      RMSNorm(const float epsilon = 1e-6, const bool use_residual = false);

      void operator()(const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const;

    private:
      template <Device D, typename T>
      void compute(const StorageView& gamma,
                   const StorageView& input,
                   StorageView& output) const;

      const float _epsilon;
      const bool _use_residual;
    };

  }
}
