# M11.22 — Metal Memory Audit Report

## Executive Summary

The CTranslate2 Metal backend had **six categories of memory leaks** causing unbounded virtual memory growth (~16 GB per whisper transcription) and system-wide memory pressure (browser tab eviction, terminal hangs, cursor freezing). All leaks have been identified and fixed. After fixes, VSIZE is stable across transcriptions, RSS during Metal inference dropped from ~5 GB to ~800 MB, and Metal inference speed improved from 0.66× to 1.88× CPU.

---

## Fundamental Constraint

**ARC (Automatic Reference Counting) cannot be enabled** for any `.mm` file in this project.

`utils.mm` uses `thread_local id<MTLCommandBuffer>` and `thread_local id<MTLCommandQueue>`, which ARC rejects with:
```
error: thread-local variable has non-trivial ownership: type is '__strong id<MTLCommandBuffer>'
```

Additionally, `compile_library_once()` takes `id<MTLLibrary>&` output parameters, which ARC disallows for strong pointers.

All ObjC memory management must therefore be **manual** — every `+1 retained` object requires an explicit `[obj release]`.

---

## Leak Categories and Fixes

### Leak 1: MPS Objects — `alloc+init` Without Release

**Scope:** 6 dispatch sites across 2 files (called hundreds of times per transcription)

Without ARC, `[[MPSMatrix alloc] initWithBuffer:...]` returns `+1 retained`. These were never released — every GEMM call leaked 4 MPS objects (3 `MPSMatrix` + 1 `MPSMatrixMultiplication`), and every BF16 GEMM leaked 2 `MPSGraphTensorData`.

**Impact:** ~16 GB VSIZE growth per transcription (the largest single leak).

**Fix:** Added explicit `[release]` after each `encodeToCommandBuffer:` or graph execution.

| File | Sites | Objects Released |
|------|------:|-----------------|
| `primitives_gemm.mm` | 4 MPS GEMM + 1 BF16 | `[matA release]`, `[matB release]`, `[matC release]`, `[gemm_op release]`, `[tdA release]`, `[tdB release]` |
| `ops_sdpa.mm` | 1 MPS GEMM + 1 BF16 | Same pattern |

**Safety:** `encodeToCommandBuffer:` internally retains its inputs. Releasing our `+1` after encoding is safe — the command buffer holds its own reference until GPU completion.

---

### Leak 2: Temporary MTLBuffers — `new` Prefix Without Release

**Scope:** 25 call sites across 3 files (called hundreds of times per transcription)

`alloc_temp_buffer()` calls `[device newBufferWithLength:options:]`. The `new` prefix means the caller receives `+1 retained` — but the function's original comment incorrectly said "ARC-managed". Every temp buffer leaked.

Temp buffers are used for:
- **GEMM row-padding** — MPS requires `rowBytesForColumns:` alignment; data is copied to a wider temp, GEMM is encoded, results are copied back
- **BF16 GEMM offset workaround** — `MPSGraphTensorData` has no byte-offset parameter; sub-buffers are copied to zero-offset temps
- **INT8 GEMM intermediates** — temp buffers for quantized matmul pipeline
- **Reduction partial results** — GPU produces per-threadgroup partials, CPU reads and reduces

**Impact:** The dominant remaining leak after MPS object fix (~5.5 GB VSIZE growth per transcription).

**Fix:** Added `[tmp release]` at each site, after the last use of the buffer.

| File | Sites | Pattern |
|------|------:|---------|
| `primitives_gemm.mm` | 14 | `if (tmp_a) [tmp_a release];` after GEMM encode + row_copy |
| `ops_sdpa.mm` | 5 | Same pattern for SDPA GEMM |
| `primitives_reduction.mm` | 6 | `[out_buf release]` after CPU readback |

**Safety for encode-only buffers:** Metal command buffers retain all resources referenced by encoded commands. Releasing our `+1` after encoding is safe — the GPU can still access the buffer data until the command buffer completes and releases its reference.

**Safety for reduction buffers:** These are read by CPU after `CT2_COMMIT_AND_WAIT()` (GPU has already finished). Release happens after the CPU read.

---

### Leak 3: Command Buffers — Autoreleased on Poolless Threads

**Scope:** `utils.mm` — `get_current_command_buffer()`, called once per commit cycle (~268 per transcription)

`[queue commandBuffer]` returns an autoreleased (`+0`) object. On threads with an autorelease pool, the pool owns the reference and drains it eventually. But **Python threads have no Cocoa autorelease pool** — autoreleased objects are never drained, leaking indefinitely.

The original code stored the autoreleased pointer directly in `thread_local` without `retain`:
```objc
// BEFORE (leaked):
_thread_buffer = [get_metal_command_queue() commandBuffer];
```

**Fix:** Wrap in `@autoreleasepool` + `retain` to take explicit ownership, then `release` in `commit_command_buffer()`:
```objc
// AFTER (correct):
@autoreleasepool {
    _thread_buffer = [[get_metal_command_queue() commandBuffer] retain];
}
// The @autoreleasepool drains the autorelease reference.
// Only our retain keeps the buffer alive.
// Released later in commit_command_buffer() → [_thread_buffer release].
```

**Related fix in `commit_and_wait_impl()`:** The function captures the buffer pointer for `waitUntilCompleted` and GPU timing access. Since `commit_command_buffer()` now releases the thread-local slot, the captured pointer must be independently retained:
```objc
id<MTLCommandBuffer> buf = [_thread_buffer retain];  // +1 independent
commit_command_buffer();                               // releases thread-local
[buf waitUntilCompleted];                              // safe — we hold +1
// ... read buf.GPUEndTime, buf.GPUStartTime ...
[buf release];                                         // balanced
```

---

### Leak 4: Compute & Blit Encoders — Autoreleased on Poolless Threads

**Scope:** 30 compute encoder sites across 13 files + 1 blit encoder in `utils.mm`

`[cmd computeCommandEncoderWithDispatchType:]` and `[cmd blitCommandEncoder]` return autoreleased (`+0`) objects. Same problem as command buffers — on Python threads, they're never drained.

**Fix:** Created `create_compute_encoder()` helper in `utils.mm`:
```objc
id<MTLComputeCommandEncoder> create_compute_encoder() {
    id<MTLCommandBuffer> cmd = get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc;
    @autoreleasepool {
        enc = [[cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
    }
    return enc;  // +1 retained, caller must [enc release] after [enc endEncoding]
}
```

Replaced all 30 sites across 13 files. Added `[enc release]` after every `[enc endEncoding]`. Same pattern applied to `blit_copy()` blit encoder.

**Files updated:** `ops_sdpa.mm`, `ops_conv1d.mm`, `ops_fused_norm_gemm.mm`, `ops_topk.mm`, `ops_rotary.mm`, `ops_norm_gather.mm`, `ops_quantize.mm`, `ops_alibi.mm`, `primitives_transpose.mm`, `primitives_elementwise.mm`, `primitives_beam_search.mm`, `primitives_gemm.mm`, `primitives_reduction.mm`.

---

### Leak 5: MTLCompileOptions and MTLFunction — Static Init Leaks

**Scope:** `primitives_infra.h` — `compile_library_once()` and `make_pso()` (called once per kernel family, ~18 families)

Two `+1 retained` objects were never released:

1. `default_msl_compile_options()` creates `[[MTLCompileOptions alloc] init]` and returns it without release. Called from `compile_library_once()` — the compile options are needed only during compilation, not after.

2. `make_pso()` creates `[lib newFunctionWithName:]` (`+1 retained`, `new` prefix) but never released the function after passing it to `newComputePipelineStateWithFunction:`.

**Fix:**
```objc
// compile_library_once: release options after library compilation
if (!opts) [effective_opts release];

// make_pso: release function after PSO creation
[fn release];
```

**Impact:** Small (18 objects at process startup), but fixes a correctness issue detectable by Metal debug tools.

---

### Leak 6: Allocator Pool — C++ Container Clear Without ObjC Release

**Scope:** `allocator.mm` — `MetalAllocator::clear_cache()` and destructor

`_pool` stores `id<MTLBuffer>` objects inside `std::vector` containers. Calling `_pool.clear()` destroys the C++ `vector` objects but does NOT send `[release]` to the ObjC `id<MTLBuffer>` pointers inside. On Apple Silicon unified memory, pooled `MTLBuffer` objects consume physical RAM directly.

**Fix:**
```objc
void clear_cache() override {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& [sz, bufs] : _pool)
        for (id<MTLBuffer> buf : bufs)
            [buf release];        // <-- was missing
    _pool.clear();
    for (auto& entry : _pending_free)
        [entry.buffer release];   // <-- was missing
    _pending_free.clear();
}
```

Added matching release logic in the destructor for `_pool`, `_pending_free`, and `_live`.

---

## Post-Fix Audit: Complete Object Lifecycle Verification

After applying all fixes, a comprehensive audit of every `.mm` and `.h` file in `src/metal/` verified correct memory management:

### Retained Objects (`+1`) — All Balanced

| Object Type | Create Pattern | Release Pattern | Count | Status |
|------------|----------------|-----------------|------:|--------|
| `MPSMatrix` | `[[MPSMatrix alloc] init...]` | `[matA release]` after encode | 15 | ✅ |
| `MPSMatrixMultiplication` | `[[... alloc] initWithDevice:]` | `[gemm_op release]` after encode | 5 | ✅ |
| `MPSGraphTensorData` | `[[... alloc] initWithMTLBuffer:]` | `[tdA release]` inside `@autoreleasepool` | 4 | ✅ |
| `MTLBuffer` (temp) | `[device newBufferWithLength:]` | `[tmp release]` after encode or CPU read | 25 | ✅ |
| `MTLBuffer` (allocator) | `[device newBufferWithLength:]` | `[buf release]` in destructor/clear_cache | all | ✅ |
| `MTLCommandBuffer` | `[queue commandBuffer]` + retain | `[release]` in commit_command_buffer | per-commit | ✅ |
| `MTLComputeCommandEncoder` | `create_compute_encoder()` (retain) | `[enc release]` after endEncoding | 30 | ✅ |
| `MTLBlitCommandEncoder` | `[cmd blitCommandEncoder]` + retain | `[blit release]` after endEncoding | 1 | ✅ |
| `MTLCompileOptions` | `[[... alloc] init]` | `[effective_opts release]` | 18 | ✅ |
| `MTLFunction` | `[lib newFunctionWithName:]` | `[fn release]` after PSO creation | per-PSO | ✅ |
| `MTLCommandQueue` | `[device newCommandQueue]` | thread_local (intentional) | per-thread | ✅ (by design) |
| `MTLDevice` | `MTLCreateSystemDefaultDevice()` | static singleton (intentional) | 1 | ✅ (by design) |
| `MPSGraph` | `[[MPSGraph alloc] init]` | static cache (intentional) | 4 | ✅ (by design) |
| `MTLLibrary` | `[device newLibraryWithSource:]` | static cache (intentional) | 18 | ✅ (by design) |
| `MTLComputePipelineState` | `[device newComputePipeline...]` | PSOCache (intentional) | ~50 | ✅ (by design) |

### Autoreleased Objects (`+0`) — All Inside `@autoreleasepool`

| Object Type | Create Pattern | Drain Mechanism | Count | Status |
|------------|----------------|-----------------|------:|--------|
| `MPSMatrixDescriptor` | `[... matrixDescriptorWithRows:]` | `@autoreleasepool{}` around GEMM | 19 | ✅ |
| `NSArray` literals | `@[@(...), @(...)]` | `@autoreleasepool{}` or `std::call_once` | ~10 | ✅ |
| `NSDictionary` literals | `@{key: val}` | `@autoreleasepool{}` around MPSGraph run | 2 | ✅ |
| `NSString` | `[NSString stringWithUTF8String:]` | `std::call_once` (static init context) | 2 | ✅ |
| `NSError` | API output parameter | Stack variable, never retained | 2 | ✅ |

### Intentional Non-Released Objects

| Object | Reason | Lifetime |
|--------|--------|----------|
| `MTLDevice` (singleton) | Process-global, created once | Process |
| `MTLCommandQueue` (thread_local) | One per thread, acceptable minor leak on thread exit | Thread |
| `MPSGraph` (static cache) | 4 instances, compiled once for BF16 GEMM transposes | Process |
| `MTLLibrary` (static cache) | 18 MSL shader libraries, compiled once | Process |
| `MTLComputePipelineState` (PSOCache) | ~50 PSOs cached by kernel name, created on first use | Process |

These are all **compile-once/use-many** objects with process lifetime — not leaks.

### Thread Safety — Verified

| Mechanism | Protection | Status |
|-----------|-----------|--------|
| `MetalAllocator::_mutex` | Guards `_live`, `_pool`, `_pending_free` | ✅ |
| `_trace_mutex` | Guards commit trace counters | ✅ |
| `_commit_count`, `_gpu_time_elapsed` | `std::atomic` with relaxed ordering | ✅ |
| `_pso_hits`, `_pso_misses` | `std::atomic` with relaxed ordering | ✅ |
| `PSOCache::mtx` | Guards PSO cache map per kernel family | ✅ |
| `Bf16GemmEntry` static init | `std::mutex` + `initialized[]` flag | ✅ |
| `_thread_queue`, `_thread_buffer` | `thread_local` — no sharing | ✅ |

---

## Measured Results

### Test: whisper-large-v3-turbo, `test_faster_whisper.py` (4 transcriptions, beam_size=5)

| Metric | Before All Fixes | After All Fixes |
|--------|:----------------:|:---------------:|
| VSIZE growth per transcription | ~16 GB (unbounded) | **~0 GB (stable)** |
| VSIZE during Metal inference | ~435+ GB (growing) | **~400–403 GB (stable)** |
| Peak RSS during Metal inference | ~4–5 GB | **~400–800 MB** |
| Peak RSS during CPU phase | ~4.4 GB | ~4.4 GB (unchanged) |
| Peak RSS during full benchmark | ~8 GB (2× model loaded) | **~4.4 GB (1× model)** |
| Metal speed vs CPU (beam_size=5) | 0.66× CPU | **1.88× CPU** |

### Progression of Fixes

| Fix Applied | VSIZE/transcription | Cumulative Reduction |
|------------|:-------------------:|:--------------------:|
| Baseline (no fixes) | ~16 GB | — |
| + MPS object release | ~12 GB | −25% |
| + Command buffer retain/release | ~6 GB | −63% |
| + Compute encoder drain | ~5.5 GB | −66% |
| + **Temp buffer release** | **~0 GB** | **−100%** |

### Test Results — 150/150 PASS

| Test Suite | Tests | Result |
|-----------|------:|--------|
| `test_beam_search.py` | 39 | PASS |
| `test_translation.py` | 90 | PASS |
| `test_whisper.py` | 13 | PASS |
| `test_faster_whisper.py` | 8 | PASS |
| **Total** | **150** | **ALL PASS** |

---

## Files Modified

### Core Memory Fixes

| File | Changes |
|------|---------|
| `src/metal/primitives_infra.h` | Fixed `alloc_temp_buffer` comment ("ARC-managed" → "manual release required"). Released `MTLCompileOptions` after library compilation. Released `MTLFunction` after PSO creation. |
| `src/metal/utils.mm` | Command buffer: `@autoreleasepool{retain}` + release. `commit_and_wait_impl`: independent retain for wait + `@autoreleasepool` around wait. Blit encoder: retain/release. New `create_compute_encoder()` helper. |
| `src/metal/utils.h` | Added `create_compute_encoder()` declaration. |
| `src/metal/allocator.mm` | Destructor releases pool/pending/live buffers. `clear_cache()` releases before clearing. |
| `src/metal/primitives_gemm.mm` | Released MPS objects at 4 GEMM sites + 1 BF16. Released temp buffers at 5 functions (14 sites). Switched to `create_compute_encoder()` (2 sites). |
| `src/metal/ops_sdpa.mm` | Released MPS objects at 1 GEMM + 1 BF16. Released temp buffers (5 sites). Switched to `create_compute_encoder()` (1 site). |
| `src/metal/primitives_reduction.mm` | Released temp buffers at 5 reduction functions (6 sites). Switched to `create_compute_encoder()` (5 sites). |

### Encoder Drain (13 files, 30 sites total)

| File | Sites |
|------|------:|
| `primitives_elementwise.mm` | 5 |
| `primitives_reduction.mm` | 5 |
| `ops_norm_gather.mm` | 4 |
| `ops_quantize.mm` | 3 |
| `primitives_beam_search.mm` | 2 |
| `primitives_gemm.mm` | 2 |
| `ops_fused_norm_gemm.mm` | 2 |
| `ops_topk.mm` | 2 |
| `primitives_transpose.mm` | 1 |
| `ops_conv1d.mm` | 1 |
| `ops_rotary.mm` | 1 |
| `ops_alibi.mm` | 1 |
| `ops_sdpa.mm` | 1 |

### Memory Management API

| File | Change |
|------|--------|
| `python/cpp/module.cc` | Added `clear_device_cache(device)` Python binding |
| `python/cpp/replica_pool.h` | `unload_model()` clears Metal cache (parity with CUDA) |
| `python/ctranslate2/__init__.py` | Export `clear_device_cache` |

### Test / Benchmark

| File | Change |
|------|--------|
| `bench_faster_whisper.py` | Sequential model loading (CPU → delete → Metal) |
| `test_faster_whisper.py` | Same pattern |
| `test_whisper.py` | Same pattern |

---

## Root Cause Reference: ObjC Memory Rules Without ARC

| Pattern | Ownership | Action Required |
|---------|:---------:|-----------------|
| `[[Foo alloc] init...]` | +1 caller owns | Caller MUST `[obj release]` |
| `[obj newXxx]` | +1 caller owns | Caller MUST `[result release]` |
| `[obj copy]` / `[obj mutableCopy]` | +1 caller owns | Caller MUST `[result release]` |
| `[Foo fooWithBar:]` (convenience) | +0 autoreleased | Needs `@autoreleasepool` to drain |
| `@[...]` / `@{...}` / `@(...)` literals | +0 autoreleased | Needs `@autoreleasepool` to drain |
| `std::vector<id<T>>.clear()` | C++ only | Does NOT send `[release]` — must iterate and release first |
| `thread_local id<T>` | No ARC | Thread exit does not send `[release]` |

Python threads have **no Cocoa autorelease pool**. Autoreleased objects on Python threads leak indefinitely. The fix is `@autoreleasepool { obj = [[... method] retain]; }` — the pool drains the autorelease reference, and the explicit `retain` takes ownership for manual release later.

Metal command buffers retain all GPU resources referenced by encoded commands. It is safe to release a buffer after encoding — the command buffer holds its own `+1` reference until GPU completion.
