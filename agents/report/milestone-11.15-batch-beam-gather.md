# M11.15 — Batch Beam Search Gathers to Reduce Syncs

**Date:** 2026-03-08
**Status:** Complete
**Agent used:** manual (no agent files available in tree)

## Task Description

Batch the 4-5 individual `gather_beam_flat()` calls per decode step (lines 683-688 in `decoding.cc`) into a single GPU dispatch + single `synchronize_stream`, reducing per-step syncs from 4-5 to 1.

## Problem

After M11.14 (fused timestamp rules), the sync trace showed **1393** `synchronize_stream` calls at `devices.cc:162`. The dominant source was beam search state reordering in `decoding.cc` lines 683-688, where 4-5 individual `gather_beam_flat()` calls each trigger a separate sync (via the two-arg `Gather::operator()` clone lifetime hazard at `gather.cc:67`).

Each `gather_beam_flat` does: `merge_batch_beam` (reshape) -> `gather` (GPU encode + sync) -> `split_batch_beam` (reshape).

## Solution

Added `batch_gather_beam_flat()` static helper in `src/decoding.cc` (guarded by `CT2_WITH_METAL`):

1. `merge_batch_beam` all views (reshapes only, no GPU work)
2. `ops::Gather::batch_gather_in_place(views, indices)` — encodes all gathers into one command buffer + single sync
3. `split_batch_beam` all views (reshapes only)

Modified the caller at lines 683-688 to use the batched path on Metal, with fallback to sequential `gather_beam_flat` for non-Metal devices.

## Key Design Decisions

- **Reuses existing `batch_gather_in_place`**: No new GPU primitives needed. The M11.1 infrastructure (clone-all, encode-all, single-sync) handles the batching.
- **Metal-only `#ifdef`**: Non-Metal backends are unaffected; they keep the original sequential path.
- **Device check at call site**: `topk_ids.device() == Device::METAL` gates the batched path, so CPU-only runs have zero overhead.

## Files Modified

| File | Change |
|------|--------|
| `src/decoding.cc` | +20 lines: `batch_gather_beam_flat()` helper + caller change at beam reordering |

## Sync Count Results

Measured with `CT2_METAL_TRACE=1 python bench_faster_whisper.py whisper-large-v3-turbo 5`:

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| `devices.cc:162` syncs | 1393 | 852 | **-541 (-39%)** |

Full trace breakdown (after):
```
  11360  primitives_gemm.mm:1018
    852  devices.cc:162         (gather syncs — was 1393)
    827  primitives_memory.mm:80
    799  primitives_reduction.mm:345
    559  ops/multinomial_metal.mm:21
    268  ops/topk_metal.mm:44
     14  primitives_memory.mm:90
     14  primitives_beam_search.mm:89
```

Estimated wall-clock savings: ~0.22 seconds (541 syncs x ~0.4 ms/sync).

## Test Results

| Test Suite | Result |
|-----------|--------|
| `test_beam_search.py` | **39/39 PASS** |
| `test_translation.py` | **90/90 PASS** |
| `test_float16_translation.py` | **9/9 PASS** |
| `test_whisper.py` | **13/13 PASS** |
| `test_faster_whisper.py` | **8/8 PASS** |

## Next Steps

- The remaining 852 `devices.cc:162` syncs come from other `gather` call sites (decoder `update_state`, non-finished batch filtering, greedy search gathers). Further batching could target these.
- The top sync sources are now `primitives_gemm.mm` (11360) and `primitives_reduction.mm` (799) — these are structural (one per op invocation) and harder to batch.
