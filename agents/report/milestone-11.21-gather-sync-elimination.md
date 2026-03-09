# M11.21 — Gather Sync Elimination

## Summary

Replaced `synchronize_stream(Device::METAL)` with `metal::protect_buffer()` in both Gather paths (single in-place and batch), eliminating 264 `commit_and_wait()` syncs per whisper-large-v3-turbo inference (beam_size=5, 60s audio).

## Problem

The Gather op's in-place path creates a clone of the source data, encodes a GPU gather kernel, then calls `synchronize_stream()` to ensure the clone buffer survives until the GPU finishes reading from it. This sync commits and waits for the entire command buffer, creating a pipeline bubble.

Two call sites:
1. **Single gather** (`Gather::operator()(data, input)`) — clone + gather + sync
2. **Batch gather** (`Gather::batch_gather_in_place()`) — clone all + encode all + sync

## Solution

Replace `synchronize_stream()` with `metal::protect_buffer()` which marks buffers for deferred-free recycling. Protected buffers go to `_pending_free` instead of the pool when freed, and are only returned to the pool after the next `commit_and_wait()` completes all prior GPU work.

### Critical Bug Discovery

Initial implementation only protected the **clone buffers** (data source). This caused 7/39 beam search test failures. The root cause: the **indices buffer** passed to `batch_gather_in_place` is also referenced by the pending GPU gather kernels. The indices are a by-value parameter in `update_state()`, so the copy is destroyed when `update_state` returns. Its buffer goes back to the pool and can be reused by `topk_ids.to(device)` in the next decode step — which has the SAME size (batch_size × beam_size × sizeof(int32)). The GPU then reads topk_ids data instead of beam indices, producing garbled output.

**Fix**: Protect both clone buffers AND the indices buffer:

```cpp
// Protect clone buffers (data source for GPU gathers)
for (auto& clone : clones)
    metal::protect_buffer(clone.buffer());
// Protect indices buffer (also read by GPU gathers)
if (indices.device() == Device::METAL)
    metal::protect_buffer(indices.buffer());
```

## Sync Trace

### whisper-large-v3-turbo, beam_size=5, 60s audio

| Source | Before M11.21 | After M11.21 |
|--------|-------------:|-------------:|
| `devices.cc:162` (synchronize_stream) | 2,536 | **2,272** |
| `primitives_memory.mm:80` (indexed_fill) | 2,508 | 2,508 |
| `topk_metal.mm:44` (CPU sort) | 268 | 268 |
| `primitives_gemm.mm:1124` (cblas) | 16 | 16 |
| **Total** | **5,328** | **5,064** |

Net reduction: **264 fewer syncs** from Gather operations.

Note: The original plan estimated ~2,500 syncs from `devices.cc:162` were from Gather. In reality, the majority of `synchronize_stream` calls come from:
- `common.cc:411`: INT8 quantized Dense layer sync (~2,000)
- `storage_view.cc:417/438`: Metal→CPU copy and type conversion syncs
- `whisper.cc:106`: Model-level sync

These are separate optimization targets for future milestones.

## Files Modified

| File | Change |
|------|--------|
| `src/ops/gather.cc` | Replace `synchronize_stream` with `protect_buffer` for clone + indices in both single and batch gather paths |

## Test Results

All 150 e2e tests pass:

| Suite | Result |
|-------|--------|
| beam_search | 39/39 PASS |
| translation | 90/90 PASS |
| whisper | 13/13 PASS |
| faster_whisper | 8/8 PASS |

## Architecture Notes

### Why protect_buffer works for clone buffers but not alone

The deferred-free mechanism (`protect_buffer` → `_pending_free` → `flush_pending_frees` after `commit_and_wait`) ensures:
1. Protected buffers cannot be recycled from the pool while GPU work is pending
2. Buffers are only returned to the pool after `commit_and_wait()` completes all prior GPU work
3. Metal's in-order command buffer execution guarantees gather kernels complete before subsequent operations on the same buffer

### Why indices must also be protected

`Decoder::update_state()` takes `beam_indices` by value. The by-value copy is used in `batch_gather_in_place` as the indices for GPU gather kernels. When `update_state` returns, the copy is destroyed and its buffer freed to the pool. The next `topk_ids.to(device)` allocation matches this buffer's size (both are `batch_size × beam_size × sizeof(int32)`), causing the GPU to read topk_ids data instead of beam indices.

### Debugging journey

The bug was elusive because:
1. Adding a sync at the start of `TransformerDecoder::decode()` (after update_state returns, before any KV cache use) **still failed** — the indices buffer was already recycled between `update_state` returning and `decode()` being called
2. Adding a sync right after `update_state()` in the decode loop **passed** — the sync committed the gathers before `beam_indices` was destroyed
3. `protect_buffer` always found and marked all clone buffers (0 misses) — the issue was the INDICES buffer, not the clones
