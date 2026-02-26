// src/metal/primitives.mm — SPLIT
//
// This file has been decomposed into focused translation units:
//
//   primitives_infra.h          Shared infrastructure (included by all below)
//   primitives_memory.mm        M4.1  at / fill / copy / convert / cross_device
//   primitives_elementwise.mm   M4.2  add / sub / mul / min / max + M4.5 activations + M4.6 broadcasts
//   primitives_reduction.mm     M4.3  sum / max / amax / max_element / logsumexp
//   primitives_gemm.mm          M4.4  gemm / gemm_batch_strided (MPS + BF16 MPSGraph)
//   primitives_transpose.mm     M4.8  transpose_2d / 3d / 4d
//   primitives_beam_search.mm   M4.7  penalize_previous_tokens / prepare_length_mask
//   ops_norm_gather.mm          M5.2  layer_norm / rms_norm / softmax / gather (metal:: wrappers)
//   ops_sdpa.mm                 M6.1  sdpa_metal — scaled dot-product attention (metal:: wrapper)
//
// See CMakeLists.txt (METAL_SOURCES) for the build list.
