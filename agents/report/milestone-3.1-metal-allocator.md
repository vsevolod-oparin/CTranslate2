# Milestone 3.1 — Metal Allocator

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Implement `MetalAllocator` — a caching allocator backed by `MTLResourceStorageModeShared`
buffers — and register it as `get_allocator<Device::METAL>()`.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/allocator.mm` | New — `MetalAllocator` + `get_allocator<Device::METAL>()` |
| `CMakeLists.txt` | `src/metal/allocator.mm` added to `METAL_SOURCES` |
| `tests/metal/allocator_test.mm` | New — 14 assertions |

---

## Design

### Why `MTLResourceStorageModeShared`

On Apple Silicon all memory is physically unified — CPU and GPU share the same DRAM.
A Shared-mode `MTLBuffer` provides:

- `[buf contents]` — a stable `void*` that is simultaneously valid for CPU reads/writes
  and GPU access (Metal ops, MPS kernels).
- No `cudaMemcpy` equivalent needed for CPU↔Metal transfers; the pointer *is* the memory.
- `StorageView` stores the `void*` directly; no indirection layer required.

`MTLResourceStorageModePrivate` (GPU-only) would require blit commands for every
CPU access. There is no performance benefit for Apple Silicon inference workloads.

### Pool design

```
_live  : void* → {requested_size, id<MTLBuffer>}   (currently allocated)
_pool  : requested_size → [id<MTLBuffer>, ...]      (available for reuse)
```

Pool key = *requested* size (not `buf.length`, which Metal may round up internally).
This ensures `allocate(N)` always finds buffers freed by `free()` of an N-byte allocation.

ARC handles `MTLBuffer` lifetime automatically:
- Pooled buffers are retained by the `std::vector`.
- `clear_cache()` clears the vectors; ARC releases the `MTLBuffer` objects.

### Thread safety

A single `std::mutex` guards both maps. Allocation is not on the hot path —
`StorageView` buffers are long-lived in CTranslate2's inference loop.

### `get_allocator<Device::METAL>()` registration

Defined at the bottom of `allocator.mm` following the same pattern as CPU and CUDA:

```cpp
template<>
Allocator& get_allocator<Device::METAL>() {
  static metal::MetalAllocator allocator;
  return allocator;
}
```

The existing `get_allocator(Device device)` dispatch (in `src/allocator.cc`) routes
`Device::METAL` to this specialisation via `DEVICE_DISPATCH`.

---

## Implementation (`src/metal/allocator.mm`)

```objc
void* allocate(size_t size, int) override {
  // 1. Lock
  // 2. Check _pool[size] — if non-empty, pop and return cached buffer
  // 3. Otherwise: [device newBufferWithLength:size options:MTLResourceStorageModeShared]
  // 4. Insert {size, buf} into _live[ptr]; return ptr
}

void free(void* ptr, int) override {
  // 1. Lock
  // 2. Look up _live[ptr] to get {size, buf}
  // 3. Erase from _live
  // 4. Push buf into _pool[size]  (ARC retains it)
}

void clear_cache() override {
  // Lock; _pool.clear()  →  ARC releases all pooled MTLBuffers
}
```

---

## Verification

### Build and run

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/allocator_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  src/metal/allocator.mm \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o allocator_test && ./allocator_test
```

### Output

```
=== M3.1: MetalAllocator tests ===

  PASS  allocate(1024 * sizeof(float)) — no error
  PASS  allocate() returns non-null
  PASS  CPU write to allocated buffer — no error
  PASS  CPU read-back [0] == 42.f
  PASS  CPU read-back [1] == -1.f
  PASS  CPU read-back [1023] == 7.f
  PASS  free() — no error
  PASS  pool hit: second alloc same size returns same ptr
  PASS  different sizes: small != large
  PASS  clear_cache() — no error
  PASS  allocate after clear_cache() — no error
  PASS  allocate after clear_cache() non-null
  PASS  free(nullptr) — no error
  PASS  free(unknown ptr) — throws

14 passed, 0 failed
```

### What the tests verify

| # | Assertion |
|---|-----------|
| 1–2 | `allocate()` succeeds and returns non-null |
| 3 | Returned pointer is CPU-writable (unified memory) |
| 4–6 | CPU can read back written values immediately |
| 7 | `free()` does not throw |
| 8 | Pool hit: second same-size alloc returns the same `void*` (same `MTLBuffer`) |
| 9 | Different sizes get distinct allocations (no cross-bucket bleed) |
| 10 | `clear_cache()` does not throw |
| 11–12 | Allocation after `clear_cache()` works |
| 13 | `free(nullptr)` is a safe no-op |
| 14 | `free()` of an unknown pointer throws `std::runtime_error` |

---

## Known Limitations / Next Steps

- The pool grows unboundedly until `clear_cache()` is called. A high-watermark
  or LRU eviction policy could be added later if memory pressure is observed.
- No alignment guarantee beyond what `MTLBuffer` provides (which is at least 16 bytes).
  If a future op requires stronger alignment, pass `MTLResourceOptionCPUCacheModeDefault`
  with a padded size.
- Next: **M3.2** — integrate `MetalAllocator` with `StorageView`
  (`cross_device_primitives` copy between CPU and Metal).
