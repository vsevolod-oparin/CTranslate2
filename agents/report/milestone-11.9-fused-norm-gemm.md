# M11.9 — Fused LayerNorm + GEMM Kernel

## Summary

Added a fused LayerNorm/RMSNorm + GEMV (matrix-vector) MSL kernel that combines normalization and weight projection into a single GPU dispatch. Targeted at **BF16 inference** where MPSGraph GEMM requires `commit_and_wait()` — fusing eliminates one CB submission (~400 µs) per fused call.

**Key constraint:** Only activates for BF16, `outer_size <= 16`, `K <= 4096`. For FP32/FP16, MPS GEMM is encode-only and vastly outperforms the naive MSL GEMV, so fusion is skipped.

## Motivation

In the BF16 decode path (outer_size=1, K=512 for whisper-base), every Dense layer that follows a LayerNorm dispatches:
1. LayerNorm MSL kernel (encode-only)
2. `commit_and_wait()` — required for MPSGraph BF16 GEMM
3. MPSGraph BF16 GEMM

By fusing steps 1-3 into a single MSL kernel dispatch, we eliminate the `commit_and_wait()` entirely. The fused kernel handles normalization in threadgroup shared memory (float32), then each thread computes ceil(N/256) output columns via dot products.

## Algorithm

### Phase 1: Normalize row into threadgroup memory

- 256 threads cooperatively load a row of K elements
- Tree reduction computes mean (LayerNorm) or sum-of-squares (RMSNorm)
- Each thread normalizes its slice: `norm_row[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]`
- All computation in float32 regardless of storage type

### Phase 2: GEMV in-place

- Weight layout: W[N, K] row-major (trans_b=true convention)
- Each thread handles ceil(N/256) output columns
- For each output column j: dot product of norm_row[0..K-1] with W[j, 0..K-1]
- Output cast back to storage type T

### Threadgroup memory

- K floats for the normalized row + 256 floats for reduction scratch
- Total: `(K + 256) * sizeof(float)` bytes per threadgroup

## Integration

`Dense::fused_norm_and_project()` is a new method on the `Dense` layer class:
- Takes a `LayerNorm` reference and input tensor
- Returns `true` if fusion was applied, `false` to fall back to separate norm + Dense
- Guards: Metal device, BF16 only, no quantization, no activation, no partial weight, outer_size <= 16, K <= 4096

Call sites:
1. **`MultiHeadAttention::operator()`** — fuses pre-norm LayerNorm with QKV projection (`_linear[0]`)
2. **`FeedForwardNetwork::operator()`** — fuses pre-norm LayerNorm with FF1 projection (only when `_ff1_noact` is false)

Both call sites fall through to separate norm + Dense when fusion returns false.

`LayerNorm` gained public accessors: `gamma()`, `beta()`, `epsilon()`, `has_beta()`.

## Changes

| File | Change |
|------|--------|
| `include/ctranslate2/layers/common.h` | Added `Dense::fused_norm_and_project()`, `LayerNorm` public accessors |
| `src/layers/common.cc` | Implemented `fused_norm_and_project()` with BF16/size guards |
| `src/layers/attention.cc` | Fused pre-norm + QKV projection in MultiHeadAttention |
| `src/layers/transformer.cc` | Fused pre-norm + FF1 projection in FeedForwardNetwork |
| `src/metal/kernels/fused_norm_gemm.metal` | MSL kernel: `DEFINE_FUSED_LN_GEMM(T)` and `DEFINE_FUSED_RMS_GEMM(T)` |
| `src/metal/ops_fused_norm_gemm.mm` | Dispatch: `fused_layer_norm_gemm_metal<T>()` and `fused_rms_norm_gemm_metal<T>()` |
| `src/metal/ops_metal.h` | Declarations for fused functions |
| `src/metal/msl_strings.h` | Regenerated with fused_norm_gemm kernel |
| `tools/gen_msl_strings.py` | Added `("fused_norm_gemm", "kFusedNormGemmMSL")` |
| `CMakeLists.txt` | Added `ops_fused_norm_gemm.mm` to METAL_SOURCES, `fused_norm_gemm.metal` to MSL sources |
| `tests/metal/fused_norm_gemm_test.mm` | 334 lines — standalone correctness tests |
| `tests/metal/fused_norm_gemm_bench.mm` | 209 lines — benchmark vs separate norm+GEMM |

## Performance Notes

- BF16 decode (outer_size=1): eliminates ~400 µs `commit_and_wait()` per fused site
- FP32/FP16: fusion disabled — MPS encode-only GEMM is faster than naive MSL GEMV
- Whisper-base has K=512, N=512/2048: well within fusion thresholds
- Net effect on Whisper BF16: modest (few CB eliminations per step), but cumulative across layers

## Test Results

| Test Suite | Count | Result |
|---|---|---|
| Standalone fused tests (`fused_norm_gemm_test.mm`) | All pass | **PASS** |
| Translation | 90/90 | PASS |
| Beam search | 39/39 | PASS |
| Whisper | 13/13 | PASS |
| Faster-whisper | 8/8 | PASS |
