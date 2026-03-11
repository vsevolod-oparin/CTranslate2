# M12.10 — INT8 Per-Dense Sync Elimination (protect_buffer)

**Date**: 2026-03-11
**Status**: COMPLETE
**Hardware**: Apple M4, macOS 15

---

## Problem

Every INT8 Dense layer called `synchronize_stream(device)` at `src/layers/common.cc:411` before local quantization temporaries (`qinput`, `qinput_scale`, `qoutput`) went out of scope. This was necessary because GPU-encoded dequantize kernels read these buffers — if the buffers were freed and reused before the GPU executed, it would read stale data.

This caused **36 syncs per decode step** (6 Dense layers × 6 encoder/decoder layers), resulting in **3,500+ commit_and_wait() calls** per 50-sentence translation. Each sync blocks the CPU until all GPU work completes, serializing the pipeline and destroying throughput.

## Solution

Replaced `synchronize_stream(device)` with `protect_buffer()` calls for the three local StorageViews:

```cpp
// Before (M12.9):
if (device == Device::MPS)
    synchronize_stream(device);

// After (M12.10):
#ifdef CT2_WITH_MPS
if (device == Device::MPS) {
    metal::protect_buffer(qinput.buffer());
    metal::protect_buffer(qinput_scale.buffer());
    metal::protect_buffer(qoutput.buffer());
}
#endif
```

`protect_buffer()` marks the buffer as GPU-referenced. When the StorageView destructor frees the buffer, it goes to a deferred queue instead of the reuse pool. The deferred buffers are only returned to the pool after the next `commit_and_wait()` completes all prior GPU work — guaranteeing the GPU has finished reading them.

This is the same pattern used for Gather (M11.21) and INT8 GEMM input buffers (M12 code review H1/M1).

## File Modified

- `src/layers/common.cc` — 3 lines changed (1 include added, sync → protect_buffer)

## Results

### Correctness
- INT8 translation test: **10/10 pass** (greedy + beam=4, exact match vs CPU-INT8)

### Performance (50 sentences, beam=4, best-of-3)

| Type | M12.9 tok/s | M12.10 tok/s | Speedup | Commits before → after |
|------|-------------|-------------|---------|----------------------|
| **int8** | 467 | **779** | **1.67×** | 3,522 → 97 (**97% reduction**) |
| **int8_float16** | 496 | **889** | **1.79×** | 3,446 → 93 (**97% reduction**) |
| **int8_bfloat16** | 483 | **887** | **1.83×** | 3,446 → 93 (**97% reduction**) |
| float16 | 1,496 | 1,490 | ~1.00× | 90 → 90 (no change) |
| float32 | 1,059 | 1,053 | ~1.00× | 96 → 96 (no change) |
| bfloat16 | 1,305 | 1,490 | ~1.00× | 90 → 90 (no change) |
| CPU f32 | 813 | 830 | — | — |

### Key Observations

1. **INT8_float16 is now the fastest INT8 variant** at 889 tok/s — surpassing CPU float32 baseline (830 tok/s) for the first time.
2. **97% commit reduction**: INT8 commits dropped from 3,500 to ~95, now comparable to float16/float32 (90-96 commits).
3. **No regression** for non-INT8 types — the change only affects the quantized GEMM code path.
4. **GPU utilization improved**: INT8 GPU% went from 39% to 55%, indicating the GPU is now better utilized with fewer idle stalls.
5. **Remaining 97 commits** are from sampler sync (1/step) + RoPE/other mandatory syncs — further reduction requires different optimizations.

### Cumulative INT8 Progress (from pre-M12 baseline)

| Milestone | INT8 tok/s | INT8_f16 tok/s | Key Change |
|-----------|-----------|---------------|------------|
| Pre-M12 | 77 | 78 | CPU int8↔f32 conversion |
| M12.6 | 453 | 494 | GPU dequantize kernels |
| **M12.10** | **779** | **889** | **protect_buffer sync elimination** |
| Total speedup | **10.1×** | **11.4×** | From pre-M12 baseline |

---

## Technical Details

### Why protect_buffer works here

The three local StorageViews (`qinput`, `qinput_scale`, `qoutput`) are allocated from the Metal allocator pool. Their buffers contain data that GPU-encoded kernels (quantize, GEMM, dequantize_gemm_output) need to read. Previously, the `synchronize_stream()` call ensured all GPU work completed before the destructors ran.

With `protect_buffer()`, the destructors still run at scope exit, but instead of returning the MTLBuffer to the reuse pool immediately, the buffer goes to a deferred-free queue. The next `commit_and_wait()` (which happens at the sampler sync point, once per decode step) calls `flush_pending_frees()` to return all deferred buffers to the pool. By that time, the GPU has already completed the kernels that read those buffers.

### Why this doesn't cause memory pressure

Each Dense layer produces 3 temporary buffers. With 36 Dense calls per step (worst case), that's 108 deferred buffers. Each is typically small (batch × depth × element_size ≈ a few KB to a few hundred KB). The total deferred memory is negligible compared to the pool size.
