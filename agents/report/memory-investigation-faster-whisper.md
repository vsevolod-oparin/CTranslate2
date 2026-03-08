# Memory Investigation: test_faster_whisper Virtual Address Space Growth

## Observation

Running `test_faster_whisper.py` shows up to ~48 GB in macOS Activity Monitor / `top`. This looks alarming but is **not a real memory leak** — it's virtual address space (VSZ) inflation, not physical memory (RSS) usage.

## Measurements

### Physical Memory (RSS) — actual DRAM usage

| Stage | RSS |
|-------|-----|
| After imports | 0.27 GB |
| After WhisperModel(cpu) | 3.57 GB |
| After WhisperModel(metal) | 5.46 GB |
| After CPU transcribe | 5.98 GB |
| After Metal transcribe | 1.87 GB (model pages swapped) |
| Peak (maxRSS) | **7.30 GB** |
| After cleanup + gc | 2.06 GB |

**Physical memory stays under 8 GB.** The RSS actually drops after Metal transcription because macOS swaps out idle model pages.

### Virtual Address Space (VSZ) — what `top` shows

| Stage | VSZ | Delta |
|-------|-----|-------|
| After imports | 392 GB | — |
| After WhisperModel(metal) | 398 GB | +6 GB |
| After Metal transcribe #1 | 414 GB | +16 GB |
| After Metal transcribe #2 | 429 GB | +15 GB |
| After Metal transcribe #3 | 440 GB | +11 GB |
| After Metal transcribe #4 | 451 GB | +11 GB |

**VSZ grows ~12-16 GB per Metal transcription** and never shrinks.

Note: Python baseline VSZ is already ~392 GB on macOS — this is normal for modern macOS processes with MALLOC_NANO and framework mappings.

### vmmap Breakdown — What's Growing

| Region | After load | After 1 transcribe | After 3 transcribes |
|--------|-----------|-------------------|---------------------|
| MALLOC_NANO | 512 MB | 1.0 GB | 1.5 GB |
| MALLOC_TINY | 32 MB | 117 MB | 234 MB |
| MALLOC_MEDIUM | 640 MB | 896 MB | 896 MB |
| IOKit (Metal GPU) | 48 KB | 64 KB | 64 KB |

## Root Cause

### `alloc_temp_buffer()` — throwaway MTLBuffer churn

`alloc_temp_buffer()` in `primitives_infra.h` calls `[MTLDevice newBufferWithLength:options:]` to create a fresh MTLBuffer for each temporary need (reduction partials, GEMM padding, etc.). These buffers are ARC-released at scope exit, but **macOS does not return the virtual address pages** to the process.

Per transcription (beam_size=5, 60s audio):
- ~17,600 `commit_and_wait()` calls
- Each sync often involves 1-3 temp buffers from `alloc_temp_buffer()`
- Estimated ~35,000-50,000 MTLBuffers created and destroyed per transcription
- Each MTLBuffer reserves virtual address space (16 KB minimum + page-aligned size)
- Result: ~12-16 GB of virtual pages claimed but never reclaimed

Call sites (28 total across the codebase):
- `primitives_reduction.mm` — 6 sites (reduction partials for sum, max, amax, max_element, fused timestamp)
- `primitives_gemm.mm` — 15 sites (GEMM A/B/C padding for single and batched)
- `ops_sdpa.mm` — 6 sites (attention A/B/C padding)

### The MetalAllocator pool — different issue

The `MetalAllocator` (used for StorageView data) pools freed buffers by size in `_pool`. These are intentionally retained for reuse and account for the stable ~2 GB RSS of model weights. This pool does NOT contribute to VSZ growth because the buffers are reused, not constantly created/destroyed.

### Why `top` shows large numbers

macOS Activity Monitor / `top` shows either VSIZE or Memory depending on column settings:
- **VSIZE** column: shows VSZ (~440+ GB) — this is the virtual address space, mostly uncommitted pages
- **Memory** column: shows RSS (~2-7 GB) — this is the actual physical memory

The 48 GB figure likely corresponds to the VSIZE at a point during the test with both models loaded + multiple transcriptions completed.

## Impact Assessment

| Concern | Impact |
|---------|--------|
| Physical memory usage | **Low** — RSS stays under 8 GB |
| Swap pressure | **None** — macOS efficiently swaps idle pages |
| VM exhaustion | **None** — macOS supports 128 TB virtual address space |
| Performance | **None** — virtual page table entries have negligible cost |
| Appearance | **Confusing** — looks like a memory leak in `top` |

## Potential Fix: Temp Buffer Pool

Add a simple per-size cache for `alloc_temp_buffer()`, mirroring `MetalAllocator`'s pool pattern:

```cpp
static id<MTLBuffer> alloc_temp_buffer(NSUInteger bytes) {
    static std::mutex mutex;
    static std::unordered_map<NSUInteger, std::vector<id<MTLBuffer>>> pool;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = pool.find(bytes);
    if (it != pool.end() && !it->second.empty()) {
        id<MTLBuffer> buf = it->second.back();
        it->second.pop_back();
        return buf;
    }
    // No cached buffer — create new
    id<MTLBuffer> buf = [get_metal_device()
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (buf == nil)
        throw std::runtime_error("Metal: failed to allocate temporary buffer");
    return buf;
}

static void free_temp_buffer(id<MTLBuffer> buf) {
    static std::mutex mutex;
    static std::unordered_map<NSUInteger, std::vector<id<MTLBuffer>>> pool;
    std::lock_guard<std::mutex> lock(mutex);
    pool[buf.length].push_back(buf);
}
```

This would require changing all call sites to explicitly return buffers, or using RAII wrappers. The main challenge: temp buffers are currently ARC-managed with scope-based lifetimes — adding pooling would require tracking ownership.

**Trade-off**: The fix addresses a cosmetic concern (VSZ in `top`), not a real memory problem. Implementing it would add complexity to 28 call sites for no performance or stability benefit.

## Recommendation

**No action needed.** The virtual address space growth is a macOS quirk with Metal buffer allocation, not a memory leak. Physical memory usage is well-bounded. If the cosmetic issue matters (e.g., for user-facing applications), a temp buffer pool could be added as a separate optimization.

## Files Referenced

| File | Role |
|------|------|
| `src/metal/primitives_infra.h:160` | `alloc_temp_buffer()` — source of temp MTLBuffers |
| `src/metal/allocator.mm` | `MetalAllocator` — caching pool for StorageView buffers |
| `src/metal/primitives_reduction.mm` | 6 `alloc_temp_buffer` call sites |
| `src/metal/primitives_gemm.mm` | 15 `alloc_temp_buffer` call sites |
| `src/metal/ops_sdpa.mm` | 6 `alloc_temp_buffer` call sites |
