# M11.18 — Encode-Only MPS Padded GEMM for m=1 Decode Attention

## Summary

Eliminated 8,712 `commit_and_wait()` syncs from whisper-large-v3-turbo inference by routing m=1 padded attention GEMMs through the existing `dispatch_mps_gemm_batched_padded` path (encode-only) instead of the CPU cblas fallback (1 sync per call). A targeted deferred-free mechanism in the MetalAllocator prevents encode-only GPU operations from referencing recycled buffers.

## Problem

After M11.17, whisper-large-v3-turbo (beam_size=5, 60s audio, float32) was 0.65x CPU speed. Root cause: **8,728 syncs from CPU cblas GEMM fallback** for m=1 decode-step attention GEMMs (QK^T and scores×V, k=64, n=3–354). These triggered MPS row-padding due to `n % 4 != 0`, and the existing `m*n ≤ 4096` threshold routed them to cblas with a per-call `CT2_COMMIT_AND_WAIT()`.

## Solution

### Part 1: Route m=1 to MPS Padded Path

Changed the threshold in `gemm_batch_strided` from:
```
if (m*n > 4096) → MPS batched padded
else → cblas
```
to:
```
if (m*n > 4096 || m == 1) → MPS batched padded
else → cblas
```

`dispatch_mps_gemm_batched_padded` is entirely encode-only:
1. GPU `row_copy` packs A/B to padded temp buffers
2. `commit_command_buffer()` (non-blocking) separates row_copy from MPS GEMM
3. `MPSMatrixMultiplication` encodes into the command buffer
4. GPU `row_copy` unpacks padded C back to original buffer

Zero `CT2_COMMIT_AND_WAIT()` calls.

### Part 2: Targeted Deferred-Free Allocator

The encode-only path crashes without protection because:
- After `gemm_batch_strided` returns, the caller may free StorageViews for A, B, or C
- The MetalAllocator returns the freed buffer to its pool
- A subsequent `allocate()` of the same size reuses the buffer
- The new owner writes data to the recycled buffer
- When the GPU finally executes (at the next `commit_and_wait()`), it reads/writes stale or corrupted data

**Solution**: `protect_buffer(ptr)` marks a live allocation so that its `free()` defers recycling until `flush_pending_frees()` is called (automatically from `commit_and_wait_impl()`).

Only buffers passed to `dispatch_mps_gemm_batched_padded` are protected — all other allocator operations remain unchanged. This avoids the massive allocation overhead of global deferred-free (which caused 0.23x regression due to creating new MTLBuffers instead of reusing pooled ones).

### Float16 Limitation

The m=1 MPS padded optimization is **float32 only**. MPS batched GEMM produces incorrect results for float16 with m=1 small matrices — verified both with and without post-sync. The output is non-deterministically garbled, indicating an MPS internal issue with float16 batched GEMM for this specific dimension configuration.

Float16 m=1 attention GEMMs continue to use the cblas fallback (1 sync per call). This is acceptable because:
- whisper-large-v3-turbo defaults to float32 on Metal (auto-converted from float16 saved model)
- The float16 path is only hit when explicitly requested with `compute_type="float16"`

### Failed Approaches

1. **Custom GEMV MSL kernel (encode-only)**: Crashed — directly references original buffers which get freed/reused before GPU execution. Same root cause as M11.17's indexed_fill failure.

2. **Custom GEMV with post-sync**: Works correctly but 1 sync per call = same count as cblas. Benchmark showed 0.47–0.94x (GPU dispatch overhead for tiny matrices not worth it).

3. **Global deferred-free**: All `free()` calls deferred until `commit_and_wait()`. Works but causes 0.23x regression — prevents ALL buffer reuse, forcing new MTLBuffer allocation for every request between syncs.

4. **Float16 MPS padded m=1**: Even with post-sync (eliminating buffer lifetime issues), MPS produces garbled output for float16 batched GEMM with m=1 and small n. Reverted to cblas for float16.

## Changes

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Route `m == 1` padded GEMMs to MPS batched padded path; protect A/B/C buffers **only for m=1** (protecting all padded GEMMs caused 0.46x regression from pool starvation) |
| `src/metal/allocator.mm` | `gpu_protected` flag on LiveEntry; `protect_buffer()` and `flush_pending_frees()` methods; deferred free for protected buffers |
| `src/metal/utils.h` | Declare `protect_buffer()` and `flush_pending_frees()` |
| `src/metal/utils.mm` | Call `flush_pending_frees()` from `commit_and_wait_impl()` |

## Results

### Sync Count (whisper-large-v3-turbo, beam_size=5, 60s audio)

| Source | Before (M11.17) | After (M11.18) | Delta |
|--------|-----------------|----------------|-------|
| `primitives_gemm.mm` (CPU cblas) | 8,728 | **16** | **-8,712** |
| `devices.cc` (synchronize_stream) | 1,506 | 1,078 | -428 |
| `primitives_memory.mm` (indexed_fill) | 1,455 | 1,024 | -431 |
| `multinomial_metal.mm` (sampling) | 1,187 | 756 | -431 |
| `topk_metal.mm` (CPU sort) | 268 | 268 | 0 |
| **Total** | **13,144** | **3,142** | **-10,002** |

The additional reductions in devices.cc, primitives_memory.mm, and multinomial are likely from reduced command buffer contention with fewer syncs.

### Performance (whisper-large-v3-turbo, beam_size=5, 60s audio, float32)

| Metric | Before | After |
|--------|--------|-------|
| CPU | ~28s | ~28s |
| Metal | ~42.5s | ~24–27s |
| Speedup | 0.65x | **1.03–1.22x** |

Note: High variance from thermal throttling on Apple M4. Both CPU and Metal times vary ±30% between runs.

### whisper-base (13 tests, beam_size=5, 30s audio)

| Metric | Before | After |
|--------|--------|-------|
| Metal/CPU ratio | ~1.98x | **3.35x** |

### Test Results

| Test | Result |
|------|--------|
| `test_beam_search.py` | 39/39 PASS |
| `test_translation.py` | 90/90 PASS |
| `test_whisper.py` | 13/13 PASS |
| `test_faster_whisper.py` | 8/8 PASS |

## Architecture

### Deferred-Free Mechanism

```
allocate(size) → check pool → return cached or new MTLBuffer
                  ↑
                  │ flush_pending_frees()
                  │ (called from commit_and_wait_impl)
                  │
free(ptr) → is gpu_protected? ─yes→ _pending_free queue
                               ─no→  _pool (immediate reuse)
                                       ↑
protect_buffer(ptr) → set gpu_protected=true on LiveEntry
```

Only buffers explicitly protected by `protect_buffer()` are deferred. This preserves the O(1) pool reuse for the vast majority of allocations.

### MPS Padded GEMM Flow for m=1

For QK^T (m=1, n=seq_len, k=64, transpose_b=true):
- A: no padding (64 cols × 4 = 256 bytes ≥ MPS min)
- B: no padding (64 cols × 4 = 256 bytes ≥ MPS min)
- C: padded when `seq_len % 4 != 0` → tmp_c + GPU row_copy back

For scores×V (m=1, n=64, k=seq_len, transpose_b=false):
- A: no padding
- B: padded when `seq_len % 4 != 0` → tmp_b
- C: no padding (64 cols × 4 = 256 bytes ≥ MPS min)
