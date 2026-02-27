// src/ops/nccl_metal.mm
//
// Metal stubs for distributed NCCL ops (ReduceAll, GatherAll).
//
// Metal is a single-device backend; distributed collective ops are not
// supported.  These stubs satisfy the linker; at runtime they throw if called.

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
    template void ReduceAll::compute<Device::METAL, T>( \
        const StorageView&, StorageView&) const; \
    template void GatherAll::compute<Device::METAL, T>( \
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
