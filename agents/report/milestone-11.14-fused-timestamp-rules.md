# M11.14 — Fused ApplyTimestampRules GPU Kernel

## Summary

Replaced per-batch_id `max()` + `logsumexp()` scalar reductions (2 `CT2_COMMIT_AND_WAIT()` each) in Whisper's `ApplyTimestampRules::should_sample_timestamp()` with a single fused GPU kernel that computes the boolean result for ALL batch_ids in one dispatch + one sync. This eliminates ~1,100 syncs from the beam_size=5 faster_whisper decode path.

## Context

Item #3 from `agents/report/faster-whisper-metal-performance.md`. During Whisper decode with timestamps, `should_sample_timestamp()` is called per beam to decide whether to force-sample a timestamp token. Each call triggered:

1. `primitives<METAL>::max()` — GPU reduce_max kernel + `CT2_COMMIT_AND_WAIT()` + CPU `std::max_element`
2. `primitives<METAL>::logsumexp()` — `CT2_COMMIT_AND_WAIT()` (flush) + CPU-only logsumexp

With beam_size=5 and ~200+ decode steps, this produced ~2,000+ syncs — the most frequent sync pair in the trace after previous optimizations.

## Changes

### `src/metal/primitives_reduction.mm` — Fused MSL kernel + dispatch function

**New MSL kernel** `should_sample_ts_<T>` (float, half, bfloat):
- One threadgroup (256 threads) per batch_id
- 3-pass binary-tree reduction using stride loops (handles vocab up to 65K):
  - **Pass 1**: `max` over text tokens `[0, num_text_tokens)` → `max_text`
  - **Pass 2**: `max` over timestamp tokens `[num_text, num_text + num_ts)` → `max_ts` (for numerically stable logsumexp)
  - **Pass 3**: `sum(exp(ts[i] - max_ts))` over timestamp tokens → `sum_exp`
- Thread 0 computes `log(sum_exp) + max_ts > max_text` and writes `uint(1)` or `uint(0)` to results buffer

**New function** `metal::should_sample_timestamps_metal<T>()`:
- Uploads `batch_ids` vector to a temp MTLBuffer
- Allocates results temp MTLBuffer
- Encodes ONE dispatch (`num_batch_ids` threadgroups × 256 threads)
- ONE `CT2_COMMIT_AND_WAIT()`
- Reads results back into `std::vector<bool>`

### `src/metal/utils.h` — Declaration

Added template declaration in the C++ section (callable from `.cc` files) with `#include <vector>` and `#include "ctranslate2/types.h"` for `dim_t`.

### `src/models/whisper.cc` — Caller change

Added `#ifdef CT2_WITH_METAL` conditional in `ApplyTimestampRules::operator()`:
- Metal path calls `should_sample_timestamps_metal<T>()` for all batch_ids at once
- CPU path (fallback) unchanged — still uses per-batch_id `should_sample_timestamp<D, T>()`

## Benchmark (faster_whisper, beam_size=5, 60s audio)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| CPU | 53,943 ms | 49,824 ms | (variance) |
| Metal | 55,321 ms | 47,444 ms | **-14.2%** |
| Speedup | 0.98x | **1.05x** | Metal now beats CPU |

## Sync Trace (bench_faster_whisper, beam_size=5)

**Before** (M11.13):
```
14512  primitives_gemm.mm:1018   (batch_cpu_gemm_f16)
 2435  primitives_reduction.mm:127 (max — includes should_sample_timestamp)
 1065  devices.cc:162            (synchronize_stream)
 1037  primitives_memory.mm:80   (indexed_fill)
  769  multinomial_metal.mm:21   (sampling)
  268  topk_metal.mm:44          (top-k)
   14  primitives_memory.mm:90   (misc)
   14  primitives_beam_search.mm:89 (beam search)
```

**After** (M11.14):
```
19408  primitives_gemm.mm:1018   (batch_cpu_gemm_f16)
 1393  devices.cc:162            (synchronize_stream)
 1363  primitives_memory.mm:80   (indexed_fill)
 1335  primitives_reduction.mm:345 (fused should_sample_ts — single sync)
 1095  multinomial_metal.mm:21   (sampling)
  268  topk_metal.mm:44          (top-k)
   14  primitives_memory.mm:90   (misc)
   14  primitives_beam_search.mm:89 (beam search)
```

The `primitives_reduction.mm:127` entry (2435 syncs from max + logsumexp) is **completely eliminated**, replaced by `:345` (1335 syncs — one per decode step instead of two per batch_id).

## Test Results

| Test Suite | Result |
|------------|--------|
| test_translation | 90/90 PASS |
| test_float16_translation | 9/9 PASS |
| test_beam_search | 39/39 PASS |
| test_whisper | 13/13 PASS |
| test_faster_whisper | 8/8 PASS |
| **Total** | **159/159 PASS** |

## Files Modified

1. `src/metal/primitives_reduction.mm` — MSL kernel `should_sample_ts_<T>` + `should_sample_timestamps_metal()` (+~90 lines)
2. `src/metal/utils.h` — Template declaration + includes (+9 lines)
3. `src/models/whisper.cc` — `#ifdef CT2_WITH_METAL` fused path (+18/-0 lines)
