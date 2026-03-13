// src/ops/awq_metal.mm
//
// Metal linker stubs for AWQ (activation-aware weight quantization) ops.
//
// AWQ packs weights as INT4 (4-bit) inside INT32 containers and dequantizes
// on-the-fly during GEMM.  MPS has no native INT4 matmul, so implementing
// this would require a custom MSL dequant+GEMM kernel.
//
// These stubs satisfy the linker so the multi-backend build compiles.
// At runtime, AWQ is only triggered when loading AWQ-quantized models
// (e.g. ct2-opus-mt-en-de-awq), which are not yet supported on Metal.
// Attempting to load one will throw here.
//
// To implement: write an MSL kernel that unpacks INT4→FP16/FP32 and
// fuses with GEMM, similar to CUDA's gemm_awq_kernel.  Low priority —
// INT8 quantization (already supported) covers most Metal use cases.

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
