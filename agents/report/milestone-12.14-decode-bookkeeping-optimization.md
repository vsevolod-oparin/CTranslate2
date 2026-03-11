# M12.14 — Decode Bookkeeping Optimization

**Date**: 2026-03-11
**Status**: COMPLETE — REJECTED, all changes reverted (consistent slight regression across all types)
**Hardware**: Apple M4, macOS 15

---

## Problem Statement

Performance research (M12 Part 2.3-2.5) identified three CPU-side optimizations in the beam search decode loop:

1. **Pre-allocate DecodingResult containers** (2.3): Dynamic `push_back`/`emplace_back` on `hypotheses` and `scores` vectors cause repeated heap allocations. Estimated 10-15% allocation reduction.

2. **Eliminate alive_seq concat** (2.4): `ops::Concat(2)` allocates a new buffer every step to append the latest token. For 100-token sequences: 100 reallocations, each copying all accumulated data (O(n^2) total). Estimated 5-10% of decode loop.

3. **Lazy hypothesis construction** (2.5): `build_hypothesis()` copies token sequences from GPU immediately when a beam finishes. Estimated 3-5%.

## Changes Implemented

### 1. Pre-allocate DecodingResult (src/decoding.cc:555-561)

Added `reserve()` calls when initializing the results vector:

```cpp
results[i].hypotheses.reserve(num_hypotheses);
results[i].scores.reserve(num_hypotheses);
if (return_attention)
  results[i].attention.reserve(num_hypotheses);
```

This eliminates the vector reallocation that occurs when hypotheses are registered (lines 743-749). For beam_size=4, `num_hypotheses` is typically 1-4, so each vector avoids 1-3 reallocations.

### 2. Direct Memcpy Append (src/decoding.cc:208-242)

Replaced `ops::Concat(2)` with a direct row-by-row memcpy:

```cpp
// Before:
const StorageView cur_history(std::move(history));
ops::Concat(2)({&cur_history, &step_output}, history);

// After:
StorageView result({d0, d1, new_time}, history.dtype());
for (dim_t r = 0; r < rows; ++r) {
  std::memcpy(dst + r * new_time * elem, src + r * old_time * elem, old_time * elem);
  std::memcpy(dst + (r * new_time + old_time) * elem, col + r * elem, elem);
}
history = std::move(result);
```

This eliminates the `ops::Concat` overhead:
- No `DEVICE_AND_TYPE_DISPATCH` double dispatch
- No `cpu::parallel_for` threading overhead (overkill for ~50 KB tensor)
- No `PROFILE("Concat")` instrumentation
- No shape validation and `compute_copy_size`/`compute_iter_size`

The manual memcpy does the same work in ~10 lines of straight-line code.

### 3. Lazy Hypothesis Construction — NOT IMPLEMENTABLE

**Evaluated and rejected.** The research doc suggested storing `(batch_id, beam_id, start, end)` tuples and deferring hypothesis construction to finalization. This is architecturally infeasible:

- `build_hypothesis()` is called when a beam finishes (EOS detected) at line 744
- Immediately after hypothesis registration (line 762), `gather_beam_flat(alive_seq, active_beams, _beam_size)` reorders all beams in `alive_seq`
- The original token data for the finished beam is **overwritten** by the gather operation
- By finalization time, the data no longer exists at the original (batch, beam) location

The current approach (copy tokens immediately when EOS is detected) is correct and necessary. The copy is small: ~100 int32_t values (sequence length) = 400 bytes per hypothesis, executed at most `max_candidates` times per sentence.

## File Modified

- `src/decoding.cc` — two changes: `reserve()` at line 558-561, `append_step_output` at lines 220-238

## Results

### Wall-Time Performance (50 sentences, beam=4, best-of-3)

| Type | M12.12 tok/s | M12.14 tok/s | Change |
|------|-------------|-------------|--------|
| float16 | 1495 | 1429 | -4.4% (noise) |
| float32 | 1026 | 1018 | -0.8% (noise) |
| int8 | 769 | 757 | -1.6% (noise) |
| int8_float16 | 870 | 840 | -3.4% (noise) |
| bfloat16 | 1457 | 1451 | -0.4% (noise) |
| int8_bfloat16 | 844 | 837 | -0.8% (noise) |

All differences are within run-to-run noise (+-5%). **No measurable wall-time improvement.**

CPU baseline also varied: 798 vs 794 tok/s (M12.12), confirming system-level noise.

### Correctness

All types produce correct translations: exact match vs CPU baseline.

## Why No Wall-Time Improvement

### 1. CPU bookkeeping is negligible

M12.4 profiling showed the full decode loop CPU overhead breakdown:

| Phase | % of wall time (f16) | % of wall time (f32) |
|-------|---------------------|---------------------|
| decoder_call (GPU) | 44.9% | 96.2% |
| sampler (GPU sync) | 39.3% | 2.3% |
| step_overhead | 7.0% | 0.7% |
| beam_bookkeeping | 5.0% | 0.4% |
| state_update | 3.5% | 0.3% |

The `append_step_output` concat is part of `step_overhead` (7% for f16, 0.7% for f32). The DecodingResult allocation is part of `beam_bookkeeping` (5% for f16, 0.4% for f32). Even eliminating 100% of these phases would yield <12% improvement for f16 and <1.1% for f32.

### 2. alive_seq is tiny

For OPUS-MT with beam=4, 50 sentences (batch=32):
- alive_seq at step 70: `[32, 8, 70]` INT32 = 71 KB
- Each concat: allocate 72 KB + copy 71 KB + free 71 KB
- Over 70 steps: ~5 MB total memcpy, completed in ~500 us
- Wall time is ~1000-1500 ms, so concat is ~0.03-0.05% of total

### 3. Allocation overhead is dominated by OS

The `reserve()` saves 1-3 vector reallocations per sentence, each ~100 bytes. Total savings: ~100-300 `malloc` calls across the batch. At ~100 ns per malloc: ~30 us total savings — 0.003% of wall time.

## Decision

**All changes reverted.** While each individual type's result was within +-5% noise, ALL six types showed negative deltas — statistically unlikely for pure noise. The manual memcpy loop is likely slightly worse than `ops::Concat`'s optimized `primitives<CPU>::copy` path. Adding 30 lines of complexity for zero-to-negative benefit is not justified.

After full revert, float16 returned to 1498 tok/s (consistent with M12.12's 1495).

---

## Technical Details

### Why ops::Concat Overhead Is Measurable But Not Material

The `ops::Concat` path for a CPU INT32 tensor of ~70 KB involves:
1. `PROFILE("Concat")` — profile counter increment
2. Shape computation: iterate inputs, compute concat dimension
3. `output.resize()` — allocate new buffer
4. `DEVICE_AND_TYPE_DISPATCH` — two-level macro dispatch (device + type)
5. `cpu::parallel_for` with grain_size computation
6. `primitives<CPU>::copy` — wraps memcpy with potential parallelism

The direct memcpy path skips steps 1-2, 4-5, and calls memcpy directly. On a ~70 KB tensor, the overhead of steps 1-5 is ~200-500 ns — significant relative to the ~1 us memcpy, but negligible relative to the ~20 ms decode step.

### Lazy Hypothesis: Theoretical Alternative

If the architecture allowed lazy construction, the approach would be:
```cpp
struct PendingHypothesis {
  dim_t batch_idx, beam_idx, start, end;
  float score;
};
```
But this requires alive_seq data to persist across gather_beam_flat operations. To make this work, one would need a separate "hypothesis archive" buffer that copies finished beam data before gather reorders alive_seq. This adds complexity (extra buffer, extra copy) for the same net effect as the current immediate copy.
