# M11.16 — BeamSearch GPU Acceleration: `prepare_length_mask` GPU Kernel

**Date:** 2026-03-08
**Status:** Complete

## Task Description

Replace the CPU-based `prepare_length_mask` implementation in `primitives_beam_search.mm` with a GPU MSL kernel, mirroring the CUDA backend. This completes item #6 (BeamSearch GPU acceleration) from M11.5 optimization list.

## Problem

`prepare_length_mask` on Metal was implemented as:
1. `CT2_COMMIT_AND_WAIT()` — flush pending GPU writes to `lengths` input
2. CPU loop — write mask values to shared memory

The CUDA backend uses a GPU kernel (`primitives.cu:321-351`). The Metal backend should match.

## Solution

### MSL Kernel (`beam_search.metal` / `kBeamSearchMSL`)

Added `prepare_length_mask` kernel with 2D grid `[batch_size, num_heads * num_queries]`. Each thread writes one mask element. Logic mirrors the CUDA kernel:

```metal
kernel void prepare_length_mask(
    device const int* lengths, device int* mask,
    constant uint& num_heads, constant uint& num_queries,
    constant uint& mask_future, constant uint& multi_query,
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.x, i = gid.y;
    int length = lengths[b];
    int val;
    if (mask_future) {
        uint idx = multi_query ? (i / num_heads) : (i % num_queries);
        val = min(length, (int)(idx + 1));
    } else {
        val = length;
    }
    mask[b * num_heads * num_queries + i] = val;
}
```

### Host-side dispatch (`primitives_beam_search.mm`)

Replaced the CPU loop with `get_beam_search_pso("prepare_length_mask")` + 2D Metal dispatch. The `CT2_COMMIT_AND_WAIT()` is **retained** at the start of the function (see Key Finding below).

## Key Finding: Accidental Fence

**The original `CT2_COMMIT_AND_WAIT()` cannot be removed.** It served as a synchronization fence not only for `lengths` (its stated purpose) but for ALL pending GPU work. Removing it causes downstream ops to read stale data from prior GPU writes, producing incorrect attention masks during beam search decode.

Investigation steps:
1. Removed `CT2_COMMIT_AND_WAIT()` entirely → beam search tests fail (12/39, translation 72/90)
2. Added `CT2_COMMIT_AND_WAIT()` AFTER GPU encode only → passes (but doesn't help sync count)
3. Added `CT2_COMMIT_AND_WAIT()` BEFORE GPU encode only → passes (same sync count as original)
4. Conclusion: the fence at the START is the required behavior

The failing pattern was characteristic: token repetition at position 3+ in beam search (e.g., "Die Katze saß **Die Katze**" instead of "Die Katze saß **auf der**"), indicating the causal attention mask was corrupted.

## Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| `primitives_beam_search.mm` syncs | 14 | 14 | 0 (fence retained) |
| Mask computation | CPU loop | GPU kernel encode | GPU offloaded |
| Sync count reduction | — | — | None (accidental fence is required) |

The optimization is **architecturally correct** (GPU kernel matches CUDA backend) but provides no sync count reduction because the fence is load-bearing. Future work could investigate and eliminate the implicit data dependency to make this fully encode-only.

## Files Modified

| File | Change |
|------|--------|
| `src/metal/kernels/beam_search.metal` | +35 lines: `prepare_length_mask` MSL kernel |
| `src/metal/msl_strings.h` | Regenerated via `gen_msl_strings.py` |
| `src/metal/primitives_beam_search.mm` | GPU dispatch replaces CPU loop (fence retained) |
| `agents/report/milestone-11.5-perf-optimization.md` | Item #6 marked DONE |

## Test Results

| Test Suite | Result |
|-----------|--------|
| `test_beam_search.py` | **39/39 PASS** |
| `test_translation.py` | **90/90 PASS** |
| `test_whisper.py` | **13/13 PASS** |
| `test_faster_whisper.py` | **8/8 PASS** |

## Future Work

To make this fully encode-only (eliminating 14 syncs), one would need to:
1. Trace ALL callers of `prepare_length_mask` and identify which preceding GPU writes they depend on
2. Ensure those writes are either on the same command buffer or already committed
3. Only then can the fence be removed safely
