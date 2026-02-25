# Milestone 3.2 — StorageView CPU↔Metal Integration

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Wire CPU↔Metal memory copies into `StorageView::copy_from`, implement
`cross_device_primitives<CPU,METAL>` and `<METAL,CPU>` as zero-copy memcpy
over unified memory, and provide the complete `primitives<Device::METAL>` stub
that satisfies the linker for the full library build.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/primitives.mm` | New — `cross_device_primitives`, `primitives<METAL>::at/copy`, stubs for all other methods, explicit instantiations |
| `src/storage_view.cc` | `copy_from`: Metal cross-device block added before the CUDA block |
| `CMakeLists.txt` | `src/metal/primitives.mm` added to `METAL_SOURCES` |
| `tests/metal/storage_view_test.mm` | New — 13 assertions |

---

## Design

### Unified memory means "copy" is memcpy

On Apple Silicon, `MTLResourceStorageModeShared` buffers expose a `void*` via
`[buf contents]` that is simultaneously valid for both CPU and GPU access.
There is no separate GPU VRAM to copy into — the Metal pointer **is** the CPU
pointer.

```
CUDA model:                         Metal/Apple Silicon model:
  CPU RAM ──cudaMemcpy──► GPU VRAM    CPU RAM == GPU VRAM (same DRAM)
                                      [buffer contents] is void* for both
```

`cross_device_primitives<CPU,METAL>` and `<METAL,CPU>` are therefore both plain
`std::memcpy`. No DMA engine, no asynchronous transfer, no pinned memory.

### Synchronisation direction

The only asymmetry between the two directions is GPU→CPU synchronisation:

| Direction | Sync needed | When |
|-----------|-------------|------|
| CPU → Metal | No | CPU writes are immediately visible to subsequent GPU commands |
| Metal → CPU | Yes | Must `commit_and_wait()` to flush any pending GPU writes before the CPU reads |

`storage_view.cc::copy_from` handles this:

```cpp
#ifdef CT2_WITH_METAL
  if (device != _device && (device == Device::METAL || _device == Device::METAL)) {
    if (device == Device::METAL) {
      // Metal → CPU: flush GPU writes first.
      synchronize_stream(Device::METAL);
      cross_device_primitives<Device::METAL, Device::CPU>::copy(...);
    } else {
      // CPU → Metal: no sync needed.
      cross_device_primitives<Device::CPU, Device::METAL>::copy(...);
    }
  } else
#endif
#ifdef CT2_WITH_CUDA
  ...
```

The Metal block fires before the CUDA block, so Metal+CUDA builds are
handled correctly (each path only matches its own device pair).

### `primitives<Device::METAL>` stub

The full `primitives<Device::METAL>` template must be instantiated for the
linker to be satisfied whenever `storage_view.cc` (or any other translation
unit) is compiled with `CT2_WITH_METAL`. Two methods are real implementations:

| Method | Implementation |
|--------|---------------|
| `at<T>(x, index)` | `return x[index]` — unified memory, always CPU-readable |
| `copy<T>(x, y, size)` | `std::memcpy(y, x, size * sizeof(T))` — both pointers are shared |

All other methods (`fill`, `add`, `mul`, `gemm`, …) throw
`std::runtime_error("not yet implemented (scheduled for M4)")`. They will be
replaced with real MPS/shader implementations in M4.

### `DECLARE_ALL_TYPES` in `.mm` files

`DECLARE_ALL_TYPES` is defined in `src/type_dispatch.h`. When used in an
`.mm` file, it must be included explicitly (it is not pulled in by
`ctranslate2/primitives.h`). Added: `#include "type_dispatch.h"`.

---

## Verification

### Build and run

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/storage_view_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  src/metal/allocator.mm \
  src/metal/primitives.mm \
  src/allocator.cc \
  src/devices.cc \
  src/cpu/allocator.cc \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o storage_view_test && ./storage_view_test
```

### Output

```
=== M3.2: CPU↔Metal primitives tests ===

--- M3.2a: cross_device_primitives (CPU↔Metal) ---
  PASS  Metal alloc non-null
  PASS  CPU→Metal copy — no error
  PASS  CPU→Metal: values visible via Metal ptr (unified memory)
  PASS  Metal→CPU copy — no error
  PASS  Metal→CPU: values correct
  PASS  int32 round-trip CPU↔Metal

--- M3.2b: primitives<METAL>::at and ::copy ---
  PASS  primitives<METAL>::at(0) == 1.f
  PASS  primitives<METAL>::at(3) == 4.f
  PASS  primitives<METAL>::copy — no error
  PASS  primitives<METAL>::copy: values match

--- M3.2c: synchronize_stream fence before Metal→CPU read ---
  PASS  fence+copy: dst[0] == 9.f
  PASS  fence+copy: dst[3] == 6.f
  PASS  synchronize_stream(METAL) + Metal→CPU — no error

13 passed, 0 failed
```

### StorageView full integration

The `StorageView::to(Device::METAL)` / `StorageView::to(Device::CPU)` paths
(which call `copy_from` internally) require `primitives<Device::CPU>` to be
linked, which in turn requires `src/cpu/primitives.cc` → `BS_thread_pool_light.hpp`
(third-party, available only in the full CMake build). These paths are verified
via CMake build rather than the standalone test above.

---

## Known Limitations / Next Steps

- All `primitives<Device::METAL>` methods except `at` and `copy` throw
  "not yet implemented". These are replaced in M4.
- `gemm`, `gemm_batch_strided`, and `gemm_pack_b` only have explicit
  instantiations for `float`/`float32` pairs; additional type pairs (int8, fp16,
  bf16) are added in M4 when the actual implementations are written.
- The linker now resolves for basic builds. Running end-to-end inference still
  requires M4 (fill, arithmetic) and M5+ (layer ops).
- Next: **M4.1** — real `primitives<Device::METAL>::fill` using a Metal compute
  kernel or `MPSMatrixCopy`, which unblocks `StorageView::zero()` and all ops
  that initialise output tensors.
