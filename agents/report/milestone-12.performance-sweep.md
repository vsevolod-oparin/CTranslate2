# M12 Performance Sweep — Translation Pipeline (OPUS-MT En→De) on Apple M4

**Date**: 2026-03-11
**Benchmark**: CTranslate2 `Translator.translate_batch()` via `m12_perf_sweep.py`
**Model**: OPUS-MT En→De (Helsinki-NLP/opus-mt-en-de)
**Dataset**: WMT14 newstest2014, first 50 sentences
**Method**: Beam=4, max_batch=32, best-of-3 runs
**Script**: `tools/benchmark/m12_perf_sweep.py`
**CPU baseline**: float32, 4 threads → 1969 ms, 1549 tokens, 787 tok/s

---

## Float32 Results

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label | vs CPU |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|--------|
| 1 | c55d4e9d | 1673 | 1818, 1673, 1673 | 1544 | 188 | 54% | 923 | M12 baseline (pre-optimization) | 1.18x |

## Float16 Results

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label | vs CPU |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|--------|
| 1 | c55d4e9d | 1201 | 1216, 1224, 1201 | 1544 | 188 | 41% | 1286 | M12 baseline (pre-optimization) | 1.64x |

## BFloat16 Results (10 sentences — too slow for 50)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 1 | a8a2bf16 | 22261 | 22261, 22410, 22334 | 195 | 2895 | 1% | 9 | M12 baseline (pre-optimization) |

---

## Baseline Analysis

### Key Observations
- **188 commits** per 50-sentence batch for both f32 and f16 — sync overhead is dtype-independent
- **GPU utilization**: f32=54%, f16=41% — f16 compute is faster so CPU overhead fraction is higher
- **f16 vs f32**: 1.39x speedup (1673 → 1201 ms) from faster MPS GEMM
- **f16 vs CPU**: 1.64x speedup — GPU already ahead of 4-thread CPU
- **f32 vs CPU**: 1.18x — barely ahead; sync overhead nearly offsets GPU compute advantage
- **BF16**: 9 tok/s, **2895 commits** for just 10 sentences — **15x more commits** than f32/f16 (50 sent). MPSGraph `runWithMTLCommandQueue:` is synchronous, forcing a commit per BF16 GEMM. GPU utilization is just 1%.
- **BF16 extrapolated to 50 sentences**: ~110s vs 1.2s for f16 — **~90x slower**

### Commit Breakdown (per decode step)
With ~50 sentences and beam=4, the 188 commits come from:
- `prepare_length_mask`: ~2 commits/step × ~70 steps ≈ 140 commits
- Sampler `synchronize_stream`: ~1 commit/step × ~70 steps ≈ 48 commits

### Optimization Targets (from M12 plan)
1. **M12.1**: Replace `prepare_length_mask` commit_and_wait → encode_barrier (~140 commits → 0)
2. **M12.2**: Batch sampler sync (48 commits → fewer)
3. **M12.3**: Fused decoder layer ops
4. **M12.4**: Async prefill pipeline

### Expected Impact
Eliminating prepare_length_mask syncs (M12.1) should reduce commits from 188 → ~48, improving GPU utilization significantly. For f32, this could push the speedup from 1.18x to ~1.5-1.8x vs CPU. For f16, from 1.64x to ~2.0-2.5x.

---

## How to Add a Row

After each optimization commit:
```bash
# Float32
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type float32 --label "description"
# Float16
conda run -n ct2 python tools/benchmark/m12_perf_sweep.py --compute_type float16 --label "description"
```
Copy the "Table row" output and append to the appropriate table above.
