# Milestone 4.1 — Memory Primitives (fill, strided_fill, indexed_fill, convert)

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Replace the M3.2 stub implementations of `fill`, `strided_fill`, `indexed_fill`,
and `convert` in `primitives<Device::METAL>` with real, working implementations.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/primitives.mm` | `fill`, `strided_fill`, `indexed_fill`, `convert` stubs replaced with real implementations; file-level comment updated |
| `tests/metal/primitives_test.mm` | New — 15 assertions |

---

## Design: CPU-side operations on unified memory

All Metal buffers use `MTLResourceStorageModeShared`. Their `contents` pointer is
simultaneously valid for CPU reads/writes and GPU access. On Apple Silicon, CPU
writes to shared memory are visible to subsequent GPU command encoding without any
explicit flush — this is guaranteed by the unified memory architecture.

Implementing these primitives as CPU-side operations is therefore both **correct**
and **sufficient** for inference use cases:

| Method | Implementation | Rationale |
|--------|---------------|-----------|
| `fill<T>(x, a, size)` | `std::fill(x, x + size, a)` | CPU write; GPU sees result before next encoding |
| `strided_fill<T>(x, a, inc, size)` | stride loop | Same |
| `indexed_fill<T>(x, a, idx, n)` | index loop | Same |
| `convert<U,V>(x, y, size)` | `std::copy(x, x + size, y)` | Relies on `half_float::half` and `bfloat16_t` implicit conversion operators |

A GPU compute kernel for `fill` would require shader compilation, kernel
dispatch overhead, and command buffer encoding — all unnecessary when the
destination is already CPU-accessible shared memory.

### `convert` type coverage

`std::copy` with implicit conversion handles all six dtype pairs that
`storage_view.cc` instantiates:

| From | To | Mechanism |
|------|----|-----------|
| `float` | `float16_t` | `half_float::half` assignment from `float` |
| `float16_t` | `float` | `half_float::half` implicit cast to `float` |
| `float` | `bfloat16_t` | `bfloat16_t` assignment from `float` |
| `bfloat16_t` | `float` | `bfloat16_t` implicit cast to `float` |
| `float16_t` | `bfloat16_t` | via `float` intermediary (both define `operator float`) |
| `bfloat16_t` | `float16_t` | same |

### Include change

Added `#include "ctranslate2/types.h"` (provides `float16_t`, `bfloat16_t`, `dim_t`)
and `#include <algorithm>` (for `std::fill`, `std::copy`). Removed the now-unneeded
`#include "ctranslate2/primitives.h"` (pulled in transitively via `types.h`).

---

## Verification

### Build and run

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/primitives_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  src/metal/allocator.mm \
  src/metal/primitives.mm \
  src/allocator.cc \
  src/devices.cc \
  src/cpu/allocator.cc \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o primitives_test && ./primitives_test
```

### Output

```
=== M4.1: Memory primitives (fill, strided_fill, indexed_fill, convert) ===

--- M4.1: primitives<METAL>::fill ---
  PASS  fill<float>(3.14f) — no error
  PASS  fill<float>: all elements == 3.14f
  PASS  fill<int32>(0): all elements == 0
  PASS  fill<int32>(42): all elements == 42
  PASS  fill<float16>(1.5f): all elements ≈ 1.5f

--- M4.1: primitives<METAL>::strided_fill ---
  PASS  strided_fill(stride=2): even positions == 7.f
  PASS  strided_fill(stride=2): odd positions untouched (0.f)

--- M4.1: primitives<METAL>::indexed_fill ---
  PASS  indexed_fill: p[1] == 9.f
  PASS  indexed_fill: p[3] == 9.f
  PASS  indexed_fill: p[5] == 9.f
  PASS  indexed_fill: p[0] untouched (0.f)
  PASS  indexed_fill: p[2] untouched (0.f)

--- M4.1: primitives<METAL>::convert ---
  PASS  convert float32→float16→float32 round-trip (tol 0.1)
  PASS  convert float32→bfloat16→float32 round-trip (tol 0.1)
  PASS  convert float16→bfloat16→float16 round-trip (tol 0.1)

15 passed, 0 failed
```

---

## What is now unblocked

With `fill`, `copy`, and `convert` all working:

- `StorageView::zero()` — calls `primitives<D>::fill(data<T>(), T(0), size)` ✅
- `StorageView::fill(value)` — same path ✅
- `StorageView::to(DataType)` — calls `primitives<D>::convert(...)` ✅
- Any op that zero-initialises its output tensor before accumulating results ✅

---

## Architecture Decision: CPU-side is permanent (not a TODO)

`fill`, `strided_fill`, `indexed_fill`, and `convert` are **permanently CPU-side**.
This is the correct design for Apple Silicon unified memory, not a temporary stub.

**Reasoning (M0.3 data):** Metal command buffer commit has ~0.4 ms fixed overhead.
A CPU `fill` on a 4096-element float32 buffer (~16 KB) takes ~0.1 µs at M4's
120 GB/s memory bandwidth. The GPU dispatch overhead alone is **4000× higher**.

Estimated crossover (CPU fill vs GPU kernel including dispatch overhead):

| Buffer | CPU fill | GPU overhead alone | Winner |
|--------|----------|--------------------|--------|
| 4K floats (16 KB) | ~0.1 µs | ~400 µs | CPU |
| 1M floats (4 MB) | ~33 µs | ~400 µs | CPU |
| 100M floats (400 MB) | ~3 ms | ~400 µs | GPU |

Tensors of 100M+ elements do not appear in CTranslate2's inference workloads.

**Contrast with CUDA:** CUDA uses `cudaMemset` because the CPU physically cannot
write to discrete GPU VRAM — there is a PCIe copy to amortize. On Apple Silicon
there is no separate VRAM; the CPU write *is* the device write.

**Conclusion:** These four primitives are finalised at their M4.1 implementations.
No Metal shader will be added for them.

## Next Steps

- Next: **M4.2** — arithmetic primitives (`add`, `mul`, `sub`) via Metal compute
  shaders (element-wise kernels). These *do* need GPU encoding because they are
  called between other GPU ops and must not force a sync.
