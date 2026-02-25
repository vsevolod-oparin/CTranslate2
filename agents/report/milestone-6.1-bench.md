# M6.1 SDPA Benchmark — Metal vs CPU

**Date:** 2026-02-25
**Hardware:** Apple M4
**File:** `tests/metal/m61_bench.mm`

---

## Setup

- **GPU path**: `metal::sdpa_metal<T>()` + `metal::commit_and_wait()` (includes ~0.4 ms CB overhead)
- **CPU reference**: single-threaded float32 `ref_sdpa()` — no BLAS, no SIMD
- **Accuracy**: Metal output (converted to float32) vs CPU float32 reference
- **Timing**: median of N runs; GPU is encode+commit_and_wait end-to-end
- **Shapes**: 8 shapes covering prefill (sq=sk), decode (sq=1), and GQA (nhk < nh)

> **Note on CB overhead**: the ~0.4 ms fixed overhead is amortised across many ops in a real
> pipeline. Standalone GPU times are conservative — the true crossover is lower.

> **Note on BF16**: the bfloat16 path uses MPSGraph (synchronous per head). This adds
> ~4 ms base cost regardless of shape, making it uncompetitive for small or decode shapes.

---

## Results

### float32 (tolerance: 1e-5)

| Shape                   | max_abs_err | Status | GPU (µs) | CPU (µs) |  Speedup |
|-------------------------|-------------|--------|----------|----------|----------|
| b1 sq64   sk64   nh8 hd64 | 1.79e-07 | PASS | 1017 | 1267 | **1.25x** GPU |
| b1 sq128  sk128  nh8 hd64 | 1.49e-07 | PASS |  917 | 6722 | **7.33x** GPU |
| b1 sq256  sk256  nh8 hd64 | 1.79e-07 | PASS | 1121 | 28165 | **25.13x** GPU |
| b1 sq512  sk512  nh8 hd64 | 1.49e-07 | PASS | 3323 | 122139 | **36.75x** GPU |
| b1 sq1024 sk1024 nh8 hd64 | 1.49e-07 | PASS | 7610 | 523762 | **68.83x** GPU |
| b1 sq1 sk256 nh8 hd64     | 0.00e+00 | PASS |  644 | 112 | 0.17x CPU |
| b1 sq1 sk1024 nh8 hd64    | 0.00e+00 | PASS |  688 | 496 | 0.72x CPU |
| b1 sq256 sk256 nh16/4 hd64 | 1.94e-07 | PASS | 3354 | 56333 | **16.79x** GPU |

### float16 (tolerance: 0.02)

| Shape                   | max_abs_err | Status | GPU (µs) | CPU (µs) |  Speedup |
|-------------------------|-------------|--------|----------|----------|----------|
| b1 sq64   sk64   nh8 hd64 | 5.62e-04 | PASS |  675 | 1297 | **1.92x** GPU |
| b1 sq128  sk128  nh8 hd64 | 5.58e-04 | PASS | 1218 | 6860 | **5.63x** GPU |
| b1 sq256  sk256  nh8 hd64 | 4.76e-04 | PASS |  906 | 28431 | **31.38x** GPU |
| b1 sq512  sk512  nh8 hd64 | 5.68e-04 | PASS | 3280 | 123194 | **37.55x** GPU |
| b1 sq1024 sk1024 nh8 hd64 | 5.38e-04 | PASS | 7376 | 532027 | **72.13x** GPU |
| b1 sq1 sk256 nh8 hd64     | 2.44e-04 | PASS |  649 | 112 | 0.17x CPU |
| b1 sq1 sk1024 nh8 hd64    | 2.43e-04 | PASS |  686 | 508 | 0.74x CPU |
| b1 sq256 sk256 nh16/4 hd64 | 5.69e-04 | PASS | 3053 | 56989 | **18.67x** GPU |

### bfloat16 — synchronous MPSGraph per head (tolerance: 0.10)

| Shape                   | max_abs_err | Status | GPU (µs) | CPU (µs) |  Speedup |
|-------------------------|-------------|--------|----------|----------|----------|
| b1 sq64   sk64   nh8 hd64 | 4.01e-03 | PASS | 4281 | 1298 | 0.30x CPU |
| b1 sq128  sk128  nh8 hd64 | 4.01e-03 | PASS | 4431 | 6894 | **1.56x** GPU |
| b1 sq256  sk256  nh8 hd64 | 4.23e-03 | PASS | 6648 | 28553 | **4.29x** GPU |
| b1 sq512  sk512  nh8 hd64 | 4.61e-03 | PASS | 7009 | 124066 | **17.70x** GPU |
| b1 sq1024 sk1024 nh8 hd64 | 4.59e-03 | PASS | 12159 | 584043 | **48.04x** GPU |
| b1 sq1 sk256 nh8 hd64     | 1.95e-03 | PASS | 4193 | 112 | 0.03x CPU |
| b1 sq1 sk1024 nh8 hd64    | 1.95e-03 | PASS | 4913 | 500 | 0.10x CPU |
| b1 sq256 sk256 nh16/4 hd64 | 4.72e-03 | PASS | 10659 | 57041 | **5.35x** GPU |

**Accuracy summary: 24 passed, 0 failed.**

---

## Analysis

### GPU crossover points (standalone, no CB amortisation)

| dtype    | Prefill crossover | Note |
|----------|-------------------|------|
| float32  | sq ≥ 64 (1.25x)   | Wins immediately, scales to 69x at sq=1024 |
| float16  | sq ≥ 64 (1.92x)   | Slightly faster than float32 at small shapes |
| bfloat16 | sq ≥ 128 (1.56x)  | ~4 ms MPSGraph base cost per `sdpa_metal` call |

### Decode scenario (sq=1)

All dtypes lose to CPU for sq=1. The CB overhead (~0.4 ms for FP32/FP16, ~4 ms for BF16)
completely dominates the tiny compute. In a real inference pipeline where the CB covers the
full decoder layer, the overhead is amortised and the GPU would be competitive.

### Accuracy observations

- **float32**: near-exact vs reference (~1.5–1.9e-07), matches expected float32 rounding.
- **float16**: stable ~5e-04 error across all shapes. The per-element error does not grow
  with sequence length, which confirms numerically stable softmax (causal masking → bounded scores).
- **bfloat16**: ~2–5e-03 error, also stable. MPSGraph is accurate despite synchronous execution.

### GQA (nh=16, nhk=4)

With 4× more query heads than KV heads, the total work is proportionally higher.
GPU speedups are good (16–19x for FP32/FP16) because each MPS GEMM call covers more heads
with the same CB overhead.

---

## Recommendation for M6.2+

- **FP32/FP16 prefill** (sq ≥ 64): GPU is already beneficial standalone; excellent in pipeline.
- **FP32/FP16 decode** (sq=1): GPU cannot win standalone due to CB overhead; acceptable in pipeline.
- **BF16 decode**: the synchronous MPSGraph per head (~500 µs each × 8 heads = ~4 ms) makes
  BF16 decode impractical even in a pipeline. A future M6.x milestone could investigate
  fusing all 8 heads into a single MPSGraph call to reduce overhead.
