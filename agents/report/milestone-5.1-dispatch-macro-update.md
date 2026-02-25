# Milestone 5.1 — Update `DEVICE_AND_FLOAT_DISPATCH` for Metal FP16/BF16

**Date:** 2026-02-25
**Status:** ✅ DONE — 14/14 tests pass

---

## Goal

Wire the `DEVICE_AND_FLOAT_DISPATCH` macro to recognise `Device::METAL` as a valid
GPU device for FP16 and BF16 types, and make the macro usable from `.mm` compilation
units (Objective-C++ files that include Metal headers).

---

## Files Modified

| File | Change |
|------|--------|
| `src/dispatch.h` | Qualify `float16_t`/`bfloat16_t` as `ctranslate2::` in `TYPE_CASE` calls |
| `src/type_dispatch.h` | Same qualification in `TYPE_DISPATCH` and `DECLARE_ALL_TYPES` (same root cause) |

## Files Added

| File | Description |
|------|-------------|
| `tests/metal/dispatch_test.mm` | 14-test runtime guard suite for M5.1 PASS criterion |

---

## Changes in Detail

### `src/dispatch.h` and `src/type_dispatch.h` — `.mm` usability fix

The `DEVICE_AND_FLOAT_DISPATCH` Metal guard was already in place from M1.1 work:

```cpp
// At least one GPU backend (CUDA, Metal, or both).
TYPE_CASE(ctranslate2::float16_t, {
  if (DEVICE != Device::CUDA && DEVICE != Device::METAL)
    throw std::invalid_argument("FP16 " NAME " is only supported on GPU");
  DEVICE_DISPATCH(DEVICE, (STMTS));
})
TYPE_CASE(ctranslate2::bfloat16_t, {
  if (DEVICE != Device::CUDA && DEVICE != Device::METAL)
    throw std::invalid_argument("BF16 " NAME " is only supported on GPU");
  DEVICE_DISPATCH(DEVICE, (STMTS));
})
```

**New in M5.1:** the type tokens were changed from bare `float16_t`/`bfloat16_t` to
`ctranslate2::float16_t`/`ctranslate2::bfloat16_t`.

**Why:** Metal SDK headers (`arm_vector_types.h`, `arm_bf16.h`) inject `::float16_t`
and `::bfloat16_t` at global scope in every `.mm` translation unit. When `dispatch.h`
is `#include`d from a `.mm` file — as it will be for every op's `*_metal.mm`
implementation in M5.2 — the bare names are ambiguous. Fully qualifying them resolves
the ambiguity with zero cost to `.cc` callers (fully-qualified names are always valid).

**Compile-time verification** (all four build configurations):

```
CPU-only   : OK  (dispatch.h #if !GPU block; float16/bfloat16 cases absent)
Metal-only : OK  (new qualified names, Metal ARM headers injected)
CUDA-only  : OK  (was already working; no change to CUDA-only path)
CUDA+Metal : OK  (both backends, new qualified names)
```

---

## Test Results

```
=== M5.1 DEVICE_AND_FLOAT_DISPATCH runtime guard tests ===

--- 1. float32 + Device::METAL ---
  PASS  float32 + Metal: no throw
  PASS  float32 + Metal: D == METAL
  PASS  float32 + Metal: sizeof(T) == 4
  PASS  float32 Metal add: out[i] == 10+i

--- 2. float16 + Device::METAL (M5.1 PASS) ---
  PASS  float16 + Metal: no throw
  PASS  float16 + Metal: D == METAL
  PASS  float16 + Metal: sizeof(T) == 2
  PASS  float16 Metal add: out[i] ≈ 5+i

--- 3. bfloat16 + Device::METAL (M5.1 PASS) ---
  PASS  bfloat16 + Metal: no throw
  PASS  bfloat16 + Metal: D == METAL
  PASS  bfloat16 + Metal: sizeof(T) == 2
  PASS  bfloat16 Metal add: out[i] ≈ 3+i

--- 4. float16 + Device::CPU: expected throw ---
  PASS  float16 + CPU: throws invalid_argument containing "FP16"

--- 5. bfloat16 + Device::CPU: expected throw ---
  PASS  bfloat16 + CPU: throws invalid_argument containing "BF16"

=== Results: 14 passed, 0 failed ===
```

### Test build command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/dispatch_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives.mm src/allocator.cc src/devices.cc \
    src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o dispatch_test && ./dispatch_test
```

---

## Design Notes

### DispatchResult pattern for macro guard tests

The test uses a small aggregate to capture `(D, sizeof(T))` inside
`DEVICE_AND_FLOAT_DISPATCH` without calling any device primitives:

```cpp
struct DispatchResult { Device d; std::size_t sz_T; };
DispatchResult dr;
CHECK_NOTHROW("float16 + Metal: no throw",
  DEVICE_AND_FLOAT_DISPATCH("test", Device::METAL, DataType::FLOAT16,
    (dr = DispatchResult{D, sizeof(T)})));
```

This avoids the CPU-path template-instantiation / link issue: the `DEVICE_DISPATCH`
switch in Metal-only builds generates a dead CPU case with `D = Device::CPU`. If
`STMTS` called `primitives<D>::add(...)`, the compiler would instantiate
`primitives<Device::CPU>::add<float16_t>` — a symbol not present in the test's
link set. Using `sizeof(T)` and a struct assignment is valid for any D and T,
so the dead branch compiles and links fine.

End-to-end Metal correctness (float16 and bfloat16 GPU adds produce correct results)
is confirmed via direct `primitives<Device::METAL>::add(...)` calls, consistent with
the M4.2 test pattern.

---

## Next Step

**M5.2** — Add `Device::METAL` template specializations for ops (LayerNorm, RMSNorm,
Softmax, Gemm/MatMul, Add/Mul/BiasAdd, Transpose, Gather) in `src/ops/*_metal.mm`.
