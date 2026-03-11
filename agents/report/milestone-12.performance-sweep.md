# M12 Performance Sweep — Translation Pipeline (OPUS-MT En→De) on Apple M4

**Date**: 2026-03-11
**Benchmark**: CTranslate2 `Translator.translate_batch()` via `m12_perf_sweep.py`
**Model**: OPUS-MT En→De (Helsinki-NLP/opus-mt-en-de)
**Dataset**: WMT14 newstest2014 (50 sentences for fast types, 10 for slow types)
**Method**: Beam=4, max_batch=32, best-of-3 runs
**Script**: `tools/benchmark/m12_perf_sweep.py`
**CPU baseline**: float32, 4 threads, 50 sentences → 1969 ms, 1549 tokens, 787 tok/s (M12.0)
**CPU baseline**: float32, 4 threads, 50 sentences → 1875 ms, 1549 tokens, 826 tok/s (M12.1, bucketed allocator)
**CPU baseline**: float32, 4 threads, 50 sentences → 2092 ms, 1549 tokens, 741 tok/s (M12.8, post code review)
**CPU baseline**: float32, 4 threads, 50 sentences → 1904 ms, 1549 tokens, 813 tok/s (M12.9, post M/L fixes)
**Chart**: `agents/report/milestone-12.performance-chart.html`

---

## Final Summary (M12.9 — Post All Fixes, all compute types, 50 sentences)

| Backend | Type | tok/s | ms | vs CPU f32 | Commits | GPU% | Notes |
|---------|------|-------|-----|-----------|---------|------|-------|
| **MPS** | **float16** | **1496** | **1036** | **1.84×** | 90 | 41% | Best overall |
| **MPS** | **bfloat16** | **1305** | **1188** | **1.60×** | 90 | 44% | Auto-promoted to f16 (M12.5) |
| **MPS** | **float32** | **1059** | **1458** | **1.30×** | 96 | 54% | Precision-sensitive |
| CPU | float32 | 813 | 1904 | 1.00× | — | — | Baseline (4 threads, AMX) |
| CPU | int8 (RUY) | 540 | 2882 | 0.66× | — | — | Memory-constrained only (M12.7) |
| **MPS** | int8_float16 | 496 | 3122 | 0.61× | 3446 | 41% | GPU dequant (M12.6) |
| **MPS** | int8_bfloat16 | 483 | 3201 | 0.59× | 3446 | 41% | Auto-promoted to int8_f16 |
| **MPS** | int8 | 467 | 3321 | 0.57× | 3522 | 39% | GPU dequant (M12.6) |

Note: CPU baseline varies between runs (741–813 tok/s). The "vs CPU f32" ratios above use the M12.9 baseline (813 tok/s). MPS absolute throughput is stable across M12.8→M12.9.

### CPU INT8 Thread Scaling (50 sentences, beam=4, RUY)

| Threads | CPU f32 tok/s | CPU int8 tok/s | INT8/FP32 ratio |
|---------|--------------|---------------|-----------------|
| 1 | — | 257 | — |
| 2 | — | 424 | — |
| 4 | 741 | 540 | 0.73× |
| 8 | — | 460 | — |

---

## Pre-M12 Baseline (all compute types)

| Compute Type | Sentences | Best (ms) | tok/s | Commits | GPU% | Status |
|-------------|-----------|-----------|-------|---------|------|--------|
| **float16** | 50 | **1201** | **1286** | 188 | 41% | **OK** — fastest |
| **float32** | 50 | 1673 | 923 | 188 | 54% | **OK** |
| **int8** | 10 | 2512 | 77 | 5743 | 14% | **Slow** — CPU int8↔f32 conversion + 2 syncs/GEMM |
| **int8_float16** | 10 | 2447 | 78 | 5630 | 16% | **Slow** — same as int8 |
| **int8_bfloat16** | 10 | 23786 | 8 | 6955 | 2% | **Very slow** — int8 overhead + MPSGraph sync |
| **bfloat16** | 10 | 22261 | 9 | 2895 | 1% | **Very slow** — MPSGraph synchronous GEMM |
| **int16** | — | — | — | — | — | **Unsupported** — `int16 compute type not supported` |

---

## Float32 Results (50 sentences)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label | vs CPU |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|--------|
| 1 | c55d4e9d | 1673 | 1818, 1673, 1673 | 1544 | 188 | 54% | 923 | M12 baseline (pre-optimization) | 1.18x |
| 2 | 8e325364 | 1480 | 1516, 1516, 1480 | 1544 | 96 | 55% | 1043 | M12.1 prepare_length_mask non-blocking commit + bucketed allocator | 1.29x |
| 3 | cbe1afee | 1462 | 1485, 1462, 1481 | 1544 | 96 | 53% | 1056 | M12.2 cached rowBytes (ObjC overhead <0.5%, negligible) | 1.28x |
| 4 | 1ef14c1a | 1515 | 1537, 1520, 1515 | 1544 | 96 | 54% | 1019 | M12.3 256-entry ptr cache (within noise) | |
| 5 | — | — | — | — | 96 | 54% | ~1000 | M12.5 BF16 auto-promotion (no f32 change) | |
| 6 | — | — | — | — | 96 | 54% | ~1000 | M12.6 INT8 GPU dequant (no f32 change) | |
| 7 | fda694a1 | 1515 | 1528, 1515, 1523 | 1544 | 96 | 54% | 1019 | M12.8 code review fixes | 1.38x |
| 8 | 093ae223 | 1458 | 1493, 1480, 1458 | 1544 | 96 | 54% | 1059 | M12.9 MEDIUM/LOW fixes (no perf change) | 1.30x |

## Float16 Results (50 sentences)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label | vs CPU |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|--------|
| 1 | c55d4e9d | 1201 | 1216, 1224, 1201 | 1544 | 188 | 41% | 1286 | M12 baseline (pre-optimization) | 1.64x |
| 2 | 8e325364 | 1033 | 1084, 1034, 1033 | 1550 | 90 | 42% | 1500 | M12.1 prepare_length_mask non-blocking commit + bucketed allocator | 1.81x |
| 3 | cbe1afee | 1012 | 1078, 1021, 1012 | 1550 | 90 | 42% | 1531 | M12.2 cached rowBytes (negligible delta) | 1.85x |
| 4 | 1ef14c1a | 1050 | 1094, 1056, 1050 | 1550 | 90 | 41% | 1476 | M12.3 256-entry ptr cache (within noise) | |
| 5 | — | — | — | — | 90 | 41% | ~1476 | M12.5 BF16 auto-promotion (no f16 change) | |
| 6 | — | 1049 | — | 1550 | 90 | 41% | 1464 | M12.6 INT8 GPU dequant (no f16 change) | |
| 7 | fda694a1 | 1035 | 1084, 1035, 1036 | 1550 | 90 | 42% | 1497 | M12.8 code review fixes | 1.86x |
| 8 | 093ae223 | 1036 | 1125, 1036, 1043 | 1550 | 90 | 41% | 1496 | M12.9 MEDIUM/LOW fixes (no perf change) | 1.84x |

## INT8 Results (50 sentences, post-M12.6)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 1 | 051a64f5 | 2512 | 2638, 2591, 2512 | 194 | 5743 | 14% | 77 | M12 baseline |
| 2 | 8e325364 | 2336 | 2380, 2348, 2336 | 194 | 5692 | 15% | 83 | M12.1 bucketed allocator |
| 3 | cbe1afee | 2303 | 2346, 2303, 2310 | 194 | 5692 | 15% | 84 | M12.2 cached rowBytes |
| 4 | 1ef14c1a | 2315 | 2361, 2331, 2315 | 194 | 5692 | 15% | 84 | M12.3 256-entry ptr cache |
| 5 | — | — | — | — | — | — | — | M12.5 (no int8 change) |
| 6 | — | — | — | 194 | 3522 | 40% | 453 | **M12.6 GPU dequant (5.4× speedup)** |
| 7 | fda694a1 | 3363 | 3363, 3389, 3375 | 1552 | 3522 | 39% | 462 | M12.8 code review fixes (50 sent) |
| 8 | 093ae223 | 3321 | 3451, 3325, 3321 | 1552 | 3522 | 39% | 467 | M12.9 MEDIUM/LOW fixes (no perf change) |

## INT8+Float16 Results (50 sentences, post-M12.6)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 1 | 051a64f5 | 2447 | 2630, 2447, 2460 | 192 | 5630 | 16% | 78 | M12 baseline |
| 2 | 8e325364 | 2266 | 2286, 2266, 2276 | 193 | 5580 | 15% | 85 | M12.1 bucketed allocator |
| 3 | cbe1afee | 2252 | 2278, 2252, 2343 | 193 | 5580 | 15% | 86 | M12.2 cached rowBytes |
| 4 | 1ef14c1a | 2254 | 2290, 2254, 2271 | 193 | 5580 | 15% | 86 | M12.3 256-entry ptr cache |
| 5 | — | — | — | — | — | — | — | M12.5 (no int8_f16 change) |
| 6 | — | — | — | 193 | 3446 | 42% | 494 | **M12.6 GPU dequant (5.7× speedup)** |
| 7 | fda694a1 | 3186 | 3186, 3242, 3196 | 1547 | 3446 | 42% | 486 | M12.8 code review fixes (50 sent) |
| 8 | 093ae223 | 3122 | 3302, 3139, 3122 | 1547 | 3446 | 41% | 496 | M12.9 MEDIUM/LOW fixes (no perf change) |

## BFloat16 Results (50 sentences, post-M12.5)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 1 | a8a2bf16 | 22261 | 22261, 22410, 22334 | 195 | 2895 | 1% | 9 | M12 baseline |
| 2 | 8e325364 | 21387 | 21401, 21501, 21387 | 195 | 2844 | 1% | 9 | M12.1 bucketed allocator |
| 3 | cbe1afee | 21329 | 21344, 21511, 21329 | 195 | 2844 | 1% | 9 | M12.2 cached rowBytes |
| 4 | 1ef14c1a | 21245 | 21361, 21328, 21245 | 195 | 2844 | 1% | 9 | M12.3 256-entry ptr cache |
| 5 | — | — | — | 1556 | 90 | 41% | 1426 | **M12.5 BF16→FP16 auto-promotion (158× speedup)** |
| 6 | — | — | — | — | 90 | 41% | ~1426 | M12.6 (no bf16 change) |
| 7 | fda694a1 | 1043 | 1134, 1043, 1046 | 1556 | 90 | 41% | 1492 | M12.8 code review fixes (50 sent) |
| 8 | 093ae223 | 1188 | 2818, 1188, 1215 | 1550 | 90 | 44% | 1305 | M12.9 MEDIUM/LOW fixes (run 1 cold JIT) |

## INT8+BFloat16 Results (50 sentences, post-M12.5)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 1 | 051a64f5 | 23786 | 27672, 25010, 23786 | 195 | 6955 | 2% | 8 | M12 baseline |
| 2 | 8e325364 | 22042 | 22528, 22042, 22251 | 195 | 6904 | 2% | 9 | M12.1 bucketed allocator |
| 3 | cbe1afee | 22664 | 22707, 22806, 22664 | 195 | 6904 | 2% | 9 | M12.2 cached rowBytes |
| 4 | 1ef14c1a | 22690 | 22785, 22766, 22690 | 195 | 6904 | 2% | 9 | M12.3 256-entry ptr cache |
| 5 | — | — | — | — | 3446 | 41% | 454 | **M12.5 auto-promoted to int8_f16** |
| 6 | — | — | — | — | 3446 | 41% | ~454 | M12.6 (benefits from int8_f16 GPU dequant) |
| 7 | fda694a1 | 3153 | 3154, 3217, 3153 | 1547 | 3446 | 41% | 491 | M12.8 code review fixes (50 sent) |
| 8 | 093ae223 | 3201 | 3424, 3220, 3201 | 1547 | 3446 | 41% | 483 | M12.9 MEDIUM/LOW fixes (no perf change) |

---

## M12 Optimization Impact Summary

| Milestone | Change | Biggest Impact |
|-----------|--------|----------------|
| **M12.0** | Baseline measurements | — |
| **M12.1** | Bucketed allocator + encode_barrier | f16: 1286→1500 tok/s (+17%), commits 188→90 (−52%) |
| **M12.2** | Cached rowBytesForColumns | <1% (within noise) |
| **M12.3** | 256-entry pointer cache | <1% (within noise) |
| **M12.5** | BF16→FP16 auto-promotion | bf16: 9→1426 tok/s (**158× speedup**) |
| **M12.6** | INT8 GPU dequantize kernels | int8: 84→453 tok/s (**5.4×**), int8_f16: 86→494 (**5.7×**) |
| **M12.7** | CPU INT8 build (RUY) | CPU int8: 540 tok/s (new, slower than CPU f32 on Apple Silicon) |
| **M12.8** | Code review fixes (8 HIGH) | protect_buffer, rounding fix, atomic counters, etc. |
| **M12.9** | Code review fixes (5 MED + 2 LOW) | ct2_u32 consistency, dead code removal — no perf impact |

---

## Baseline Analysis

### Performance Tiers
1. **Tier 1 — Production-ready**: float16 (1286 tok/s), float32 (923 tok/s)
2. **Tier 2 — Functional but slow**: int8 (77 tok/s), int8_float16 (78 tok/s) — ~16x slower than f16
3. **Tier 3 — Unusable**: bfloat16 (9 tok/s), int8_bfloat16 (8 tok/s) — ~150x slower than f16
4. **Tier 4 — Broken**: int16 (not supported on MPS)

### Root Causes by Tier

**Tier 2 (int8/int8_float16)** — 5600-5700 commits for 10 sentences:
- No native Metal INT8 GEMM — CPU converts int8→f32, GPU runs f32 GEMM, CPU converts back
- Each GEMM needs 2 extra `commit_and_wait()` for CPU↔GPU data transfer
- ~30x more commits than f32/f16 at comparable sentence count
- Fix: M12.6 — GPU dequantize kernel (int8→f16 on GPU, skip CPU round-trip)

**Tier 3 (bfloat16/int8_bfloat16)** — 2900-7000 commits for 10 sentences:
- MPSGraph `runWithMTLCommandQueue:` is synchronous (~1ms per GEMM call)
- Every BF16 GEMM forces a full command buffer commit+wait
- int8_bfloat16 compounds both problems (int8 conversion + MPSGraph sync)
- GPU utilization: 1-2% — GPU idle 98-99% of the time
- Fix: M12.5 — async MPSGraph, or auto-promote BF16→FP16 for GEMM

### Key Observations
- **188 commits** per 50-sentence batch for f32/f16 — sync overhead is dtype-independent
- **GPU utilization**: f32=54%, f16=41% — f16 compute is faster so CPU overhead fraction is higher
- **f16 vs f32**: 1.39x speedup (1673 → 1201 ms) from faster MPS GEMM
- **f16 vs CPU**: 1.64x speedup — GPU already ahead of 4-thread CPU
- **f32 vs CPU**: 1.18x — barely ahead; sync overhead nearly offsets GPU compute advantage
- **int8 ≈ int8_float16**: Both ~77-78 tok/s — the int8 conversion overhead dominates, accumulate type doesn't matter

### Commit Breakdown (f32/f16, per decode step)
With ~50 sentences and beam=4, the 188 commits come from:
- `prepare_length_mask`: ~2 commits/step × ~70 steps ≈ 140 commits
- Sampler `synchronize_stream`: ~1 commit/step × ~70 steps ≈ 48 commits

### Optimization Targets (from M12 plan)
1. **M12.1**: Replace `prepare_length_mask` commit_and_wait → encode_barrier (~140 commits → 0)
2. **M12.2**: Batch sampler sync (48 commits → fewer)
3. **M12.3**: Fused decoder layer ops
4. **M12.4**: Async prefill pipeline
5. **M12.5**: BF16 GEMM fix (async MPSGraph or auto-promote to FP16)
6. **M12.6**: INT8 GPU dequantize (eliminate CPU round-trip)

### Expected Impact
Eliminating prepare_length_mask syncs (M12.1) should reduce commits from 188 → ~48 for f32/f16, improving GPU utilization significantly. For f32, this could push the speedup from 1.18x to ~1.5-1.8x vs CPU. For f16, from 1.64x to ~2.0-2.5x.

---

## How to Add a Row

After each optimization commit:
```bash
# Primary types (50 sentences, ~30s each)
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type float32 --label "description"
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type float16 --label "description"

# Slow types (10 sentences to keep runtime reasonable)
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type int8 --num_sentences 10 --label "description"
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type int8_float16 --num_sentences 10 --label "description"
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type bfloat16 --num_sentences 10 --label "description"
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type int8_bfloat16 --num_sentences 10 --label "description"
```
Copy the "Table row" output and append to the appropriate table above.

---

## M12.1 Memory Analysis — Bucketed Allocator (2026-03-11)

### Root Cause of System Crash (pre-M12.1)
The Metal allocator pool keyed on **exact requested size**. During translation, each internal
batch has different sentence lengths → different tensor dimensions → different buffer sizes.
Almost none of these sizes were reused across batches, so the pool accumulated dead MTLBuffers
indefinitely. On Apple Silicon (unified memory), each `MTLResourceStorageModeShared` buffer
consumes real DRAM. Without a cap, the full WMT14 (2737 sentences) grew the pool to **30+ GB**,
exhausting system memory and crashing the machine.

### Fix: Power-of-2 Size-Class Bucketing
`allocate()` now rounds sizes up to the next power-of-2. The pool keys on bucket size instead
of exact size. A 131,000-byte request and a 130,500-byte request both bucket to 131,072 → the
freed buffer gets reused. The pool converges after the first few batches.

### Memory Profile: Full WMT14 (2737 sentences, float16, beam=4)

| Metric | Before (exact-size) | After (bucketed) |
|--------|-------------------|-----------------|
| Pool after 500 sent | **15,436 MB** | **702 MB** (22x reduction) |
| Pool after full WMT14 | **~80 GB** (crash) | **2,774 MB** |
| Same-data repeat growth | +150 MB each | **0 MB** (perfect reuse) |
| Peak RSS (full WMT14) | system crash | **1,108 MB** |
| Translation correctness | baseline | exact match vs CPU |

### Performance Impact of Bucketing
Benchmarks show no measurable performance difference across pool cap sizes (256 MB to uncapped),
confirming the old exact-size pool entries were dead weight with ~0% reuse.

### Chunk-Level Memory Stability (2737 sent, chunk_size=200)

| Chunk | Sentences | Pool (MB) | RSS (MB) | tok/s |
|-------|-----------|-----------|----------|-------|
| 1 | 1-200 | 1,653 | 869 | 1,758 |
| 2 | 201-400 | 2,518 | 957 | 1,348 |
| 3-5 | 401-1000 | 2,542-2,554 | 957 | 1,412-1,490 |
| 6-9 | 1001-1800 | 2,554 | 957 | 989-1,440 |
| 10-14 | 1801-2737 | 2,562-2,774 | 959-1,108 | 1,153-1,456 |

Pool stabilizes by chunk 3 (~2,550 MB) with small growth as new bucket sizes are encountered
in later sentence-length ranges. RSS stays under 1.2 GB throughout.

### Environment Variable (optional safety valve)
`CT2_METAL_POOL_MAX_MB=N` caps the pool at N megabytes. Excess buffers are released to the system.
Default: unlimited (bucketing alone prevents unbounded growth).
