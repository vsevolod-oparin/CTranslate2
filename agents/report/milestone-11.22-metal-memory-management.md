# M11.22 — Metal Memory Management

## Problem

Running benchmarks caused system-wide memory pressure on macOS — browser tabs offloading, terminal blanking, cursor hanging. Root cause: **five** issues compounding.

### Issue 1: ARC not enabled — all ObjC objects leaked

The Metal `.mm` files compile **without ARC** (`-fobjc-arc` not set). This means:
- `[[MPSMatrix alloc] init...]` → +1 retained, **never released**
- `[[MPSMatrixMultiplication alloc] init...]` → +1 retained, **never released**
- `[[MPSGraphTensorData alloc] init...]` → +1 retained, **never released**
- `MetalAllocator._pool` containing `id<MTLBuffer>` → `.clear()` destroys the C++ vector but does NOT send `[release]` to the ObjC objects

Every GEMM call leaked 4 MPS objects (3 MPSMatrix + 1 MPSMatrixMultiplication). Whisper runs hundreds of GEMMs per transcription → **~16 GB VSIZE growth per transcription run** that was never reclaimed.

ARC cannot be enabled globally because `utils.mm` uses `thread_local id<MTLCommandBuffer>` which ARC rejects (`non-trivial ownership`). Manual `[release]` is required.

### Issue 2: Autoreleased objects leaked on Python threads

Python threads have no Cocoa autorelease pool. Without a pool, autoreleased objects (`[queue commandBuffer]`, `[cmd computeCommandEncoder...]`, `[cmd blitCommandEncoder]`) are never drained — they accumulate forever.

- Command buffers: `[queue commandBuffer]` returns +0 autoreleased. Stored in `thread_local` without `retain`, the autorelease was the only reference — and it never drained.
- Compute encoders: 30 sites across 13 files create autoreleased encoders that accumulate.
- Blit encoders: 1 site in `blit_copy()`.

### Issue 3: Temporary MTLBuffer leak (`alloc_temp_buffer`)

`alloc_temp_buffer()` in `primitives_infra.h` creates `MTLBuffer` objects via `[device newBufferWithLength:]` (returns +1 retained). The comment said "ARC-managed" but **ARC is not enabled**. Every temp buffer leaked.

Temp buffers are used for:
- GEMM row-padding (MPS requires `rowBytesForColumns:` alignment)
- BF16 GEMM offset workaround (MPSGraphTensorData has no byte-offset parameter)
- INT8 GEMM intermediate buffers
- Reduction partial results

~20+ call sites, some called hundreds of times per transcription → **the dominant remaining leak** after fixing MPS objects and encoders.

### Issue 4: Metal allocator pool never cleared

`unload_model()` only cleared CUDA cache, never Metal.
On Apple Silicon unified memory, pooled MTLBuffers = system RAM.

### Issue 5: Benchmarks loaded two model copies simultaneously

`bench_faster_whisper.py` loaded CPU + Metal models together (~8 GB peak for whisper-large-v3-turbo).

## Changes

### Fix 1: Manual ObjC release (MPS object leak)

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `[matA/matB/matC/gemm_op release]` after every `encodeToCommandBuffer:` (4 sites). Added `[tdA/tdB release]` after MPSGraph execution (1 site). |
| `src/metal/ops_sdpa.mm` | Same pattern: `[release]` after MPS encode (1 site) and MPSGraph (1 site) |
| `src/metal/allocator.mm` | `clear_cache()` now sends `[buf release]` before clearing pool/pending. Added destructor with explicit release for pool, pending, and live buffers. |

### Fix 2: Autorelease pool management (command buffer + encoder leak)

| File | Change |
|------|--------|
| `src/metal/utils.mm` | `get_current_command_buffer()`: wrap `[queue commandBuffer]` in `@autoreleasepool{}` + `retain`. `commit_command_buffer()`: `[release]` before nil. `commit_and_wait_impl()`: retain `buf` independently before `commit_command_buffer()` releases slot; `@autoreleasepool` around wait; `[buf release]` after GPU timing read. `blit_copy()`: wrap encoder in `@autoreleasepool{}` + retain/release. New `create_compute_encoder()` helper. |
| `src/metal/utils.h` | Added `create_compute_encoder()` declaration. |
| 13 primitive/ops files | Replaced 30 `[cmd computeCommandEncoderWithDispatchType:]` sites with `create_compute_encoder()` (which wraps in `@autoreleasepool` + retain). Added `[enc release]` after every `[enc endEncoding]`. |

Files updated for encoder drain:
`ops_sdpa.mm`, `primitives_transpose.mm`, `ops_conv1d.mm`, `ops_fused_norm_gemm.mm` (2),
`ops_topk.mm` (2), `ops_rotary.mm`, `ops_norm_gather.mm` (4), `primitives_elementwise.mm` (5),
`primitives_beam_search.mm` (2), `primitives_gemm.mm` (2), `primitives_reduction.mm` (5),
`ops_quantize.mm` (3), `ops_alibi.mm`.

### Fix 3: Temporary buffer release (`alloc_temp_buffer` leak)

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `[tmp_a/tmp_b/tmp_c release]` in 5 functions: `dispatch_mps_gemm`, `run_bf16_gemm_inner`, `dispatch_int8_gemm`, `dispatch_mps_gemm_batched_padded`, batched INT8 GEMM (14 release sites total). |
| `src/metal/ops_sdpa.mm` | Added `[tmp_a/tmp_b/tmp_c release]` in `sdpa_mps_gemm` (3 sites) and `sdpa_bf16_gemm` (2 sites). |
| `src/metal/primitives_reduction.mm` | Added `[out_buf/vals_buf/idxs_buf release]` in `sum`, `max_element`, `max`, `amax`, `fuse_timestamp_check_and_disable_metal` (6 sites). |

Metal command buffers retain all referenced resources, so releasing our +1 after encoding is safe.

### Fix 4: Memory management API

| File | Change |
|------|--------|
| `python/cpp/module.cc` | Added `clear_device_cache(device)` Python binding |
| `python/cpp/replica_pool.h` | `unload_model()` clears Metal cache (parity with CUDA) |
| `python/ctranslate2/__init__.py` | Export `clear_device_cache` |
| `src/metal/allocator.mm` | `pool_bytes()` / `live_bytes()` introspection |
| `src/metal/utils.h` | Declarations for stats functions |

### Fix 5: Test / Benchmark sequential loading

| File | Change |
|------|--------|
| `bench_faster_whisper.py` | Sequential model loading (CPU → delete → Metal) |
| `test_faster_whisper.py` | Same pattern |
| `test_whisper.py` | Same pattern |

## Memory Impact (whisper-large-v3-turbo, test_faster_whisper.py — 4 transcriptions)

| Metric | Before all fixes | After all fixes |
|--------|-----------------|-----------------|
| VSIZE growth per transcription | ~16 GB (never reclaimed) | **~0 GB (stable)** |
| VSIZE during Metal inference | ~435+ GB (growing) | **~400–402 GB (stable)** |
| Peak RSS during Metal inference | ~4–5 GB | **~740–940 MB** |
| RSS during CPU phase | ~4.4 GB | ~4.4 GB (unchanged) |
| Peak RSS during benchmark | ~8 GB (2× model) | ~4.4 GB (1× model) |
| Metal speed (beam_size=5) | 0.66x CPU | **1.95x CPU** |

VSIZE is now stable across transcriptions — no per-transcription growth. The 1.95x Metal speedup (up from 0.66x) is partly due to eliminating memory pressure that was throttling the GPU.

## Root Cause Explanation

Without ARC, the Cocoa memory management rules apply:
- `alloc+init` returns +1 retained → caller MUST `release`
- `@autoreleasepool` only releases `autorelease`d objects (from convenience constructors like `matrixDescriptorWithRows:`)
- `alloc+init` objects are NOT autoreleased — they need explicit `[obj release]`
- Storing `id<MTLBuffer>` in C++ containers (std::vector, std::unordered_map) and then clearing the container destroys the C++ wrapper but does NOT send `-release` to the ObjC object
- `[device newBufferWithLength:]` returns +1 retained — the `new` prefix means the caller owns it. Without ARC, `alloc_temp_buffer` callers MUST explicitly `[release]`
- Python threads have no autorelease pool — autoreleased objects from `[queue commandBuffer]` and `[cmd computeCommandEncoder...]` accumulate indefinitely unless wrapped in explicit `@autoreleasepool{}` blocks
- `retain` + `@autoreleasepool` drains the autorelease reference; explicit `[release]` later brings retain count to 0
- Metal command buffers retain all referenced GPU resources during encoding, so releasing our +1 after encoding is safe — the GPU will still access the data until the command buffer completes

## Test Results

| Test | Result |
|------|--------|
| `test_beam_search.py` | 39/39 PASS |
| `test_translation.py` | 90/90 PASS |
| `test_whisper.py` | 13/13 PASS |
| `test_faster_whisper.py` | 8/8 PASS |
| **Total** | **150/150 PASS** |
