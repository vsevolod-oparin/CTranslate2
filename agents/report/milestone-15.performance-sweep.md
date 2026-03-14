# M15 Performance Sweep — Post-Cleanup Benchmark (2026-03-14)

**Commit**: `d04ab8b5` (M15.7 Cross-model beam search validation)
**Branch**: `metal-backend`
**Changes since last sweep (M13, `972abc70`)**: M14.1–M14.8 (precision audits, beam search, logits sync fix, regression gate) + M15.1–M15.7 (cleanup milestone)

---

## Key Finding: M14.5 `synchronize_stream` Regression

M14.5 added `synchronize_stream(Device::MPS)` after every decoder call in `decoding.cc` (both beam search and greedy paths). This is **required for correctness** — without it, logits processors (repetition_penalty, no_repeat_ngram) read stale GPU data, causing page faults and wrong results.

**Impact**: commit count ~2× across all compute types, f16/bf16 throughput regressed ~25%.

| Metric | M13 (pre-fix) | M15 (post-fix) | Change |
|--------|---------------|----------------|--------|
| f16 commits | 90 | 190 | +111% |
| f32 commits | 96 | 184 | +92% |
| int8 commits | 97 | 186 | +92% |
| f16 tok/s | 1435 | 1073 | **−25%** |
| bf16 tok/s | 1399 | 1076 | **−23%** |
| f32 tok/s | 1070 | 1042 | −3% (noise) |
| int8 tok/s | 768 | 768 | 0% |
| int8_f16 tok/s | 850 | 884 | +4% (noise) |

**Why f16/bf16 regressed more**: f16 decode is GPU-bound (~48% GPU time), so each added `commit_and_wait()` creates a pipeline stall. f32/int8 already had heavier sync patterns and are less affected.

**M15 cleanup itself has zero hot-path impact** — all performance changes are from M14.5.

---

## M12 Translation Sweep (OPUS-MT En→De, 50 sentences, beam=4, best-of-3)

**CPU baseline**: float32, 4 threads → 1898 ms, 1549 tokens, 816 tok/s

| Type | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | vs CPU |
|------|-----------|-----------|--------|---------|------|-------|--------|
| **float32** | 1482 | 1537, 1482, 1522 | 1544 | 184 | 54% | 1042 | 1.28× |
| **float16** | 1439 | 1488, 1458, 1439 | 1544 | 190 | 48% | 1073 | 1.32× |
| **int8** | 2020 | 2032, 2038, 2020 | 1552 | 186 | 54% | 768 | 0.94× |
| **int8_float16** | 1750 | 1819, 1750, 1757 | 1547 | 182 | 44% | 884 | 1.08× |
| **bfloat16** | 1436 | 1530, 1478, 1436 | 1544 | 190 | 50% | 1076 | 1.32× |
| **int8_bfloat16** | 1786 | 1866, 1786, 1805 | 1547 | 182 | 45% | 866 | 1.06× |

---

## M11 Whisper Sweep (whisper-large-v3-turbo, f16, 30s audio)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Label | Speedup |
|---|--------|-----------|-----------|--------|-------|---------|
| 52 | a4046a63 | 1,812 | 1811, 1818, 1807 | 123 | M12.26 Residency sets | 22.72× |
| 53 | d04ab8b5 | 2,239 | 2266, 2239, 2271 | 127 | M15 Cleanup | 18.39× ⚠ |

⚠ **Methodology change**: commit 53 uses HF `WhisperFeatureExtractor` on real audio (`sample.mp3`, 30s) instead of random mel features. Produces 127 tokens vs ~123 previously. The M15 code changes (dead file deletion, documentation, BF16 multinomial kernel) cannot affect Whisper f16 inference.

---

## Potential M14.5 Regression Mitigation

The `synchronize_stream` in `decoding.cc` is a blunt instrument. Possible optimizations (future work):

1. **Conditional sync**: only sync when logits processors that read logits are active (repetition_penalty != 1.0, no_repeat_ngram_size > 0, etc.). Skip sync for default beam search with no processors.
2. **Encode-only barrier**: use `encode_barrier()` (MTLSharedEvent) instead of `commit_and_wait()` — GPU-side sync without CPU stall. This is the same pattern that recovered 50% of the BUG-2 regression in M11.29.
3. **Batch sync**: sync once per batch step rather than once per decoder call.

These are not M15 scope (cleanup only) but are tracked for future optimization.
