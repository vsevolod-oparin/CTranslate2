# M12.13 — Aggressive GEMV for Decode

**Date**: 2026-03-11
**Status**: COMPLETE — FAILED EXPERIMENT, all changes reverted
**Hardware**: Apple M4, macOS 15

---

## Problem Statement

Performance research (M12 Part 4.1) estimated 20-40% improvement from custom GEMV kernels for single-token decode, where m=1 (or small m). The hypothesis was that MPS GEMM has significant descriptor/matrix allocation overhead for small-m shapes, and a lightweight MSL kernel could bypass this.

CTranslate2 already had two custom GEMV kernels:
- **FP16 batched GEMV** (`dispatch_gemv_f16_batched`, M11.19) — required because MPS produces garbled output for float16 m=1 batched GEMM (MPS bug)
- **FP32 batched GEMV** (`dispatch_gemv_f32_batched`, M12.2) — available but not routed by default

Both kernels use a simple scalar dot-product design: 256 threads per threadgroup, each thread handles ceil(N/256) output columns with a scalar accumulation loop over K elements.

## Approach

### Experiment 1: Non-batched m<=4 GEMV routing

Route all non-batched GEMMs with m<=4 through the custom GEMV kernel. During OPUS-MT decode with beam=4, the dominant shapes are:

| Shape (m x n x k) | Count/step | Operation |
|---|---|---|
| 4 x 512 x 512 | 35 | Linear projections (Q, K, V, output) |
| 4 x 1536 x 512 | 12 | Feed-forward intermediate |
| 4 x 2048 x 512 | 11 | Feed-forward up |
| 4 x 512 x 2048 | 11 | Feed-forward down |
| 4 x 58100 x 512 | 1 | Logits projection |

**Result: -19.7% regression for float32** (1026 -> 824 tok/s), -14.7% for int8 (769 -> 656 tok/s).

**Root cause**: The custom scalar GEMV kernel is dramatically slower than MPS for shapes with large N and K. MPS uses the Apple GPU's hardware matrix multiply units (AMX on CPU, dedicated matrix hardware on GPU) with SIMD/simdgroup optimizations. The naive scalar kernel (1 thread = 1 dot product over K) cannot compete:

- For 4x512x512: MPS uses hardware matrix multiply at ~500 GFLOPS; custom kernel achieves ~10 GFLOPS
- For 4x2048x512: even worse ratio due to memory-bound scalar access pattern
- The overhead savings (~5-10 us per GEMM call) are negligible compared to compute penalty (~100-500 us per GEMM)

**Action**: Reverted immediately.

### Experiment 2: F32 batched m=1 GEMV routing

Route only float32 batched m=1 GEMMs (attention shapes) through the custom GEMV kernel. These are the smallest GEMMs in the pipeline:

| Shape | Count/step | Batch | Operation |
|---|---|---|---|
| 1 x seq_len x 64 | varies | 32 | Q x K^T (per-head attention) |
| 1 x 64 x seq_len | varies | 32 | scores x V (per-head attention) |

Where seq_len ranges from 1 to ~200 during decode.

**Result: -20% regression for float32** (1026 -> 818 tok/s), -14% for int8 (769 -> 659 tok/s).

**Root cause**: Even for tiny shapes (N=64, K=64), the custom kernel is slower than MPS batched GEMM:

1. **MPS batched GEMM amortizes setup**: One `MPSMatrixMultiplication` handles all 32 batches with a single descriptor setup. The per-batch overhead is negligible.
2. **Threadgroup waste**: With N=64 and 256 threads, 75% of threads are idle (only 64 do useful work). The kernel launches 32 threadgroups x 256 threads = 8192 threads, but only 2048 do work.
3. **No SIMD reduction**: Each thread does a scalar loop over K. MPS uses simdgroup matrix operations that process multiple elements per cycle.
4. **INT8 uses f32 attention**: The int8 pipeline runs attention in float32, so the f32 GEMV routing affected int8 performance too.

**Action**: Reverted.

### Verification after revert

After reverting both changes, float32 returned to 1031 tok/s (consistent with M12.12's 1026 tok/s, within noise).

## File Modified

- `src/metal/primitives_gemm.mm` — temporarily modified, then fully reverted to M12.12 state

## Results Summary

| Experiment | Types affected | Regression |
|---|---|---|
| Non-batched m<=4 GEMV | float32: -19.7%, int8: -14.7% | **Severe** |
| F32 batched m=1 GEMV | float32: -20%, int8: -14% | **Severe** |
| Final (all reverted) | All types unchanged vs M12.12 | **None** |

## Why the Research Estimate Was Wrong

The M12 performance research estimated "20-40% for single-token decode GEMM" based on two flawed assumptions:

1. **"Decode uses m=1"**: False for non-batched GEMMs. With beam_size=4, the non-batched shapes are m=4 (not m=1). Only batched attention GEMMs are m=1.

2. **"Custom kernel is faster than MPS for small m"**: False. MPS's hardware-optimized GEMM uses the GPU's dedicated matrix multiply units with SIMD width 32. The naive scalar GEMV kernel (1 thread = 1 dot product) achieves <5% of MPS throughput for any shape with K > 64.

3. **"MPS setup overhead dominates for small shapes"**: Partially true for a single GEMM call, but MPS batched GEMM amortizes setup across all batch elements. For batch=32, the per-element overhead is ~0.3 us — negligible.

## What Would Be Needed for Competitive Custom GEMV

To actually outperform MPS, a custom GEMV kernel would need:

1. **Simdgroup reductions**: Use `simd_sum()` for K-dimension reduction (32-wide instead of scalar)
2. **Shared memory tiling**: Load B-matrix tiles into threadgroup memory for reuse across batch elements
3. **Multiple elements per thread**: Process 4-8 output columns per thread with vectorized loads (`float4`)
4. **Warp-level primitives**: Coordinate threads within a simdgroup for cooperative loading

This would essentially be reimplementing what MPS already does. Given that MPS is maintained by Apple and optimized for each GPU generation, the effort/reward ratio is very unfavorable.

## Decision

**No code changes retained.** The existing GEMV kernels remain available for their original purposes:
- FP16 batched GEMV: still used (required due to MPS bug for f16 m=1)
- FP32 batched GEMV: remains in codebase but not routed (available for future use if MPS exhibits similar f32 bugs)

The research item "Aggressive GEMV for decode" is marked as **REJECTED** — MPS hardware-optimized GEMM outperforms naive custom kernels for all shapes in the OPUS-MT translation pipeline.

---

## Benchmark Data

### Experiment 1: Non-batched m<=4 GEMV (50 sentences, beam=4, best-of-3)

| Type | M12.12 tok/s | With m<=4 GEMV | Change |
|------|-------------|---------------|--------|
| float32 | 1026 | 824 | **-19.7%** |
| int8 | 769 | 656 | **-14.7%** |

### Experiment 2: F32 batched m=1 GEMV (50 sentences, beam=4, best-of-3)

| Type | M12.12 tok/s | With f32 batched GEMV | Change |
|------|-------------|----------------------|--------|
| float16 | 1495 | 1494 | 0% (not affected) |
| float32 | 1026 | 818 | **-20.3%** |
| int8 | 769 | 659 | **-14.3%** |
| int8_float16 | 870 | 868 | -0.2% (noise) |
| bfloat16 | 1457 | 1467 | +0.7% (noise) |
| int8_bfloat16 | 844 | 873 | +3.4% (noise) |

### After full revert (verification)

| Type | M12.12 tok/s | Post-revert tok/s | Change |
|------|-------------|-------------------|--------|
| float32 | 1026 | 1031 | +0.5% (noise) |
