// src/ops/awq_metal.mm
//
// Metal stubs for AWQ (activation-aware weight quantization) ops.
//
// Metal has no native AWQ (INT4-weight quantized) kernel.
// These stubs satisfy the linker; at runtime, the AWQ path is only triggered
// when running AWQ-quantized models, which are not yet supported on Metal.

#include "ctranslate2/ops/awq/dequantize_awq.h"
#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/ops/awq/gemv.h"

#include "metal/utils.h"

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename InT, typename OutT>
    void DequantizeAwq::dequantize(const StorageView&,
                                   const StorageView&,
                                   const StorageView&,
                                   StorageView&) const {
      throw std::runtime_error(
          "DequantizeAwq is not supported on Metal (AWQ not yet implemented)");
    }

    template void DequantizeAwq::dequantize<Device::MPS, int32_t, ctranslate2::float16_t>(
        const StorageView&, const StorageView&, const StorageView&, StorageView&) const;
    template void DequantizeAwq::dequantize<Device::MPS, int32_t, ctranslate2::bfloat16_t>(
        const StorageView&, const StorageView&, const StorageView&, StorageView&) const;
    template void DequantizeAwq::dequantize<Device::MPS, int32_t, float>(
        const StorageView&, const StorageView&, const StorageView&, StorageView&) const;

    // ---------------------------------------------------------------------------

    template <Device D, typename In, typename Out>
    void GemmAwq::compute(const StorageView&, const StorageView&,
                          const StorageView&, const StorageView&,
                          StorageView&) const {
      throw std::runtime_error(
          "GemmAwq is not supported on Metal (AWQ not yet implemented)");
    }

    template void GemmAwq::compute<Device::MPS, ctranslate2::float16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemmAwq::compute<Device::MPS, ctranslate2::bfloat16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemmAwq::compute<Device::MPS, float, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;

    // ---------------------------------------------------------------------------

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv(const StorageView&, const StorageView&,
                                const StorageView&, const StorageView&,
                                StorageView&) const {
      throw std::runtime_error(
          "GemvAwq is not supported on Metal (AWQ not yet implemented)");
    }

    template <Device D, typename In, typename Out>
    void GemvAwq::compute_gemv2(const StorageView&, const StorageView&,
                                 const StorageView&, const StorageView&,
                                 StorageView&) const {
      throw std::runtime_error(
          "GemvAwq (gemv2) is not supported on Metal (AWQ not yet implemented)");
    }

    template void GemvAwq::compute_gemv<Device::MPS, ctranslate2::float16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv2<Device::MPS, ctranslate2::float16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv<Device::MPS, ctranslate2::bfloat16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv2<Device::MPS, ctranslate2::bfloat16_t, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv<Device::MPS, float, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;
    template void GemvAwq::compute_gemv2<Device::MPS, float, int32_t>(
        const StorageView&, const StorageView&, const StorageView&,
        const StorageView&, StorageView&) const;

  }  // namespace ops
}  // namespace ctranslate2
