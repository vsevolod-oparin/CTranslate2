# Milestones 2.2 / 2.3 / 2.4 — Synchronize, ScopedDeviceSetter, Error Handling

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

- **M2.2** — Wire `synchronize_stream` / `synchronize_device` for `Device::METAL`
- **M2.3** — Extend `ScopedDeviceSetter` template for Metal (get/set device index)
- **M2.4** — Error handling strategy: `CT2_METAL_CHECK_BUFFER` / `CT2_METAL_CHECK_OBJ`

All three were **already fully implemented** by the time this milestone was formally addressed:
- M2.2 was completed in M2.1 (`src/devices.cc` wired to `metal::commit_and_wait()`)
- M2.3 was completed in M1.1 (`src/metal/device.h` — `get_device_index` / `set_device_index` template specialisations)
- M2.4 was completed in M2.1 (`src/metal/utils.h` — both error macros defined and in use)

This milestone writes verification tests and formal documentation.

---

## Files Changed

| File | Change |
|------|--------|
| `tests/metal/sync_scoped_test.mm` | New — standalone verification (10 assertions) |

No source file changes were needed — all implementation was already complete.

---

## M2.2 — `synchronize_stream` / `synchronize_device`

### Implementation (in M2.1)

`src/devices.cc`:

```cpp
void synchronize_stream(Device device) {
#ifdef CT2_WITH_METAL
  if (device == Device::METAL)
    metal::commit_and_wait();
#endif
}

void synchronize_device(Device device, int index) {
#ifdef CT2_WITH_METAL
  if (device == Device::METAL) {
    (void)index;
    metal::commit_and_wait();
  }
#endif
}
```

`commit_and_wait()` (`src/metal/utils.mm`):

```objc
void commit_and_wait() {
  if (_thread_buffer == nil) return;   // no-op if nothing encoded
  id<MTLCommandBuffer> buf = _thread_buffer;
  commit_command_buffer();             // commits, resets _thread_buffer to nil
  [buf waitUntilCompleted];            // blocks until GPU done
  CT2_METAL_CHECK_BUFFER(buf);         // throws on GPU error
}
```

### Test assertions (M2.2 section of `sync_scoped_test.mm`)

1. `commit_and_wait()` with no encoded commands — no error, no hang
2. Encode real blit (copy 64 bytes, two shared buffers) → `commit_and_wait()` — no error
3. Blit result correct after `commit_and_wait()` (CPU verifies GPU wrote correct bytes)
4. Fresh command buffer ready after sync (new buffer ≠ committed buffer)

---

## M2.3 — `ScopedDeviceSetter` for Metal

### Implementation (in M1.1)

`src/metal/device.h` defines Metal specialisations of the `get_device_index` /
`set_device_index` function templates:

```cpp
template<>
int get_device_index<Device::METAL>() {
  return 0;          // Metal: always device 0 (one GPU per Apple Silicon system)
}

template<>
void set_device_index<Device::METAL>(int index) {
  if (index != 0)
    throw std::invalid_argument(
        "Invalid Metal device index: " + std::to_string(index));
}
```

The existing `ScopedDeviceSetter` RAII template (in `include/ctranslate2/devices.h`) uses
these specialisations without modification:

```cpp
// Constructor: saves prev index, calls set_device_index if changed
// Destructor:  restores prev index, calls set_device_index if changed
```

Because Metal always has index 0, the constructor never calls `set_device_index` in
practice (prev == new == 0). This is a safe no-op.

### Test assertions (M2.3 section of `sync_scoped_test.mm`)

1. `get_device_index<METAL>() == 0`
2. `set_device_index<METAL>(0)` — no error
3. `set_device_index<METAL>(1)` — throws `std::invalid_argument`
4. ScopedDeviceSetter pattern with index 0: `if (prev != 0) set_device_index(0)` — no error
   (simulates RAII scope; confirms constructor path is safe)

---

## M2.4 — Error Handling Strategy

### Implementation (in M2.1)

Defined in the `#ifdef __OBJC__` section of `src/metal/utils.h`:

```objc
// After [buf waitUntilCompleted] — check for GPU-reported errors:
#define CT2_METAL_CHECK_BUFFER(buf)                                        \
  do {                                                                      \
    if ((buf).status == MTLCommandBufferStatusError) {                     \
      throw std::runtime_error(                                             \
          std::string("Metal command buffer error: ") +                    \
          [(buf).error.localizedDescription UTF8String]);                  \
    }                                                                       \
  } while(0)

// After alloc/init — check that an MPS/Metal object is non-nil:
#define CT2_METAL_CHECK_OBJ(obj, name)                                     \
  do {                                                                      \
    if ((obj) == nil) {                                                     \
      throw std::runtime_error("Metal: failed to create " name);           \
    }                                                                       \
  } while(0)
```

### When to use each macro

| Pattern | Macro | Notes |
|---------|-------|-------|
| After `[buf waitUntilCompleted]` | `CT2_METAL_CHECK_BUFFER(buf)` | GPU async errors only surface here |
| After `[[MPSFoo alloc] init...]` | `CT2_METAL_CHECK_OBJ(obj, "name")` | MPS returns nil on failure, no NSError |
| After `newLibraryWithSource:options:error:` | Inline `if (error) throw...` | Some Metal APIs do take `NSError**` |

### Error recovery (documented design)

- **Command buffer error:** throw `std::runtime_error` with GPU diagnostics from
  `error.localizedDescription`. Let the caller handle (typically: fail the batch).
- **Out of memory (nil buffer):** `CT2_METAL_CHECK_OBJ` throws. Caller may retry after
  `allocator.clear_cache()`.
- **GPU fault / device lost:** macOS recovers the GPU automatically; subsequent command
  buffers will succeed. The current batch fails — throw and let the engine retry.

---

## Verification

### Build and run

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/sync_scoped_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o sync_scoped_test && ./sync_scoped_test
```

### Output

```
=== M2.2 + M2.3 tests ===

--- M2.2: synchronize (commit_and_wait) ---
  PASS  commit_and_wait() empty buffer — no error
  PASS  commit_and_wait() after blit encode — no error
  PASS  blit result correct after commit_and_wait()
  PASS  fresh command buffer ready after sync
  PASS  fresh buffer differs from committed one

--- M2.3: Metal device index ---
  PASS  get_device_index<METAL>() == 0
  PASS  set_device_index<METAL>(0) — no error
  PASS  set_device_index<METAL>(1) — throws
  PASS  prev index == 0
  PASS  ScopedDeviceSetter(METAL, 0) — no error

10 passed, 0 failed
```

---

## Known Limitations / Next Steps

- `commit_and_wait()` is the **only** sync point. Primitives must never call it themselves —
  only `synchronize_stream()` / `synchronize_device()` may commit.
- `set_device_index<METAL>(n)` for n > 0 throws — Metal multi-GPU (rare, Mac Pro only)
  is out of scope. A future multi-GPU pass would need an `MTLDevice` array.
- Next: **M3** — Metal allocator (`MetalAllocator` backed by `MTLResourceStorageModeShared`).
  This unblocks `StorageView` for Metal and, together with M4 primitives stub, unblocks the
  Python extension linker.
