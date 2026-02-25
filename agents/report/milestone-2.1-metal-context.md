# Milestone 2.1 — Metal Context Module

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Establish the Metal execution context: process-wide device singleton, per-thread command
queue, per-thread deferred command buffer, and commit/wait helpers. Wire
`synchronize_stream` and `synchronize_device` in `devices.cc` to use the new context.

---

## Execution Model (confirmed from M0.3)

```
One MTLDevice   per process  — singleton, lazy C++11 static init (thread-safe)
One MTLCommandQueue per thread — thread_local, created on first access
One MTLCommandBuffer per thread — thread_local, lazy; reset to nil after each commit
  → ops encode into the current buffer (never commit themselves)
  → only synchronize_stream() / commit_and_wait() commits
```

This mirrors the CUDA stream model and avoids the 48% GPU-utilisation penalty of
per-op commits (confirmed in M0.3 at ~0.4 ms fixed overhead per submission).

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/utils.h` | New — C++ + ObjC++ interface, error-check macros |
| `src/metal/utils.mm` | New — context implementation |
| `src/devices.cc` | `synchronize_stream` and `synchronize_device` wired to `commit_and_wait()` |
| `CMakeLists.txt` | `src/metal/utils.mm` added to `METAL_SOURCES` |
| `tests/metal/context_test.mm` | New — standalone verification test (10 assertions) |

---

## Implementation Details

### `src/metal/utils.h`

Uses `#ifdef __OBJC__` to split the header into two sections:

- **Always visible (plain C++):** `commit_and_wait()` — callable from `devices.cc`
  without Objective-C knowledge.
- **ObjC++ only:** `get_metal_device()`, `get_metal_command_queue()`,
  `get_current_command_buffer()`, `commit_command_buffer()`, plus the two error macros.

This single-header design keeps ObjC types out of `.cc` consumers while giving `.mm`
op files the full API through one include.

### Error macros

```objc
// After [buf waitUntilCompleted] — check for GPU-reported errors:
CT2_METAL_CHECK_BUFFER(buf)

// After alloc/init — check that an MPS/Metal object is non-nil:
CT2_METAL_CHECK_OBJ(obj, "descriptive name")
```

Most MPS objects return `nil` on failure (no `NSError`). Operations that do take
`NSError**` (e.g. `newLibraryWithSource:`) require an inline `if (error) throw...`
pattern — they cannot use `CT2_METAL_CHECK_OBJ`.

### `src/metal/utils.mm` — key design choices

| Choice | Rationale |
|--------|-----------|
| `static id<MTLDevice> device = MTLCreateSystemDefaultDevice()` | C++11 static-local init is thread-safe; no `dispatch_once` boilerplate needed |
| `thread_local id<MTLCommandQueue>` at anonymous-namespace file scope | Must be accessible from both `get_metal_command_queue()` and `commit_command_buffer()`; function-local statics would be invisible across functions |
| `commit_command_buffer()` captures `buf` before resetting to `nil` | Keeps a strong ARC reference so `waitUntilCompleted` can be called after the slot is cleared |
| No-op when `_thread_buffer == nil` | Avoids errors when `synchronize_stream` is called before any op has been encoded |

### `src/devices.cc`

```cpp
void synchronize_stream(Device device) {
  ...
#ifdef CT2_WITH_METAL
  if (device == Device::METAL)
    metal::commit_and_wait();
#endif
}

void synchronize_device(Device device, int index) {
  ...
#ifdef CT2_WITH_METAL
  if (device == Device::METAL) {
    (void)index;
    metal::commit_and_wait();
  }
#endif
}
```

---

## Verification

### Standalone test

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/context_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o context_test && ./context_test
```

```
=== Metal context tests ===
  PASS  get_metal_device() != nil
  PASS  get_metal_device() is singleton
  PASS  get_metal_command_queue() != nil
  PASS  get_metal_command_queue() is per-thread singleton
  PASS  get_current_command_buffer() != nil
  PASS  get_current_command_buffer() stable before commit
  PASS  new buffer issued after commit_command_buffer()
  PASS  new buffer differs from committed buffer
  PASS  commit_and_wait() with empty buffer — no error
  PASS  fresh buffer ready after commit_and_wait()

10 passed, 0 failed
```

### CMake build

```
[ 97%] Building OBJCXX object .../src/metal/device.mm.o  ✅
[ 98%] Building OBJCXX object .../src/metal/utils.mm.o   ✅
clang++: error: linker command failed  ← expected (primitives<Device::METAL> still M2+)
```

---

## Known Limitations / Next Steps

- `commit_and_wait()` with an empty buffer is a no-op (returns immediately). Once real
  ops encode commands, the wait will block until the GPU finishes.
- `thread_local` ARC strong ObjC references may leak on thread exit for short-lived
  threads. Acceptable for CTranslate2's long-lived inference threads.
- `MTLCommandQueue` label (for GPU profiler visibility) not yet set — add in a later
  milestone when debugging tools are needed.
- Next: M2.2 adds a `primitives<Device::METAL>` stub to unblock the linker and enable
  Python extension builds.
