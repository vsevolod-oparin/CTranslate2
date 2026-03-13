// src/ops/nccl_metal.mm
//
// Metal linker stubs for distributed collective ops (ReduceAll, GatherAll).
//
// Metal is a single-GPU backend — Apple Silicon has one unified GPU, so
// multi-device collective communication (NCCL/MPI) is not applicable.
// These stubs satisfy the linker; at runtime they throw if called.
//
// Unlike AWQ, this is a permanent architectural limitation, not a missing
// implementation.  Multi-GPU Apple systems do not exist in the consumer
// or server space, so there is no path to implementing these.

#include "ctranslate2/ops/nccl_ops.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void ReduceAll::compute(const StorageView&, StorageView&) const {
      throw std::runtime_error(
          "ReduceAll is not supported on Metal (distributed ops require NCCL/MPI)");
    }

    template <Device D, typename T>
    void GatherAll::compute(const StorageView&, StorageView&) const {
      throw std::runtime_error(
          "GatherAll is not supported on Metal (distributed ops require NCCL/MPI)");
    }

#define DECLARE_NCCL(T) \
    template void ReduceAll::compute<Device::MPS, T>( \
        const StorageView&, StorageView&) const; \
    template void GatherAll::compute<Device::MPS, T>( \
        const StorageView&, StorageView&) const;

    DECLARE_NCCL(float)
    DECLARE_NCCL(int8_t)
    DECLARE_NCCL(int16_t)
    DECLARE_NCCL(int32_t)
    DECLARE_NCCL(ctranslate2::float16_t)
    DECLARE_NCCL(ctranslate2::bfloat16_t)
#undef DECLARE_NCCL

  }  // namespace ops
}  // namespace ctranslate2
