# Milestone 1.1 — Device::METAL Enum & Dispatch Infrastructure

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Introduce `Device::METAL` into CTranslate2's device enumeration and wire up all dispatch macro infrastructure so that every existing `DEVICE_DISPATCH` / `DEVICE_AND_FLOAT_DISPATCH` call site automatically handles the Metal device without any individual changes.

---

## Files Changed

| File | Change |
|------|--------|
| `include/ctranslate2/devices.h` | Added `METAL` to `enum class Device` |
| `src/metal/device.h` | New header — `metal::get_device_count()` |
| `src/metal/device.mm` | New implementation — calls `MTLCreateSystemDefaultDevice()` |
| `src/devices.cc` | Added Metal branches to all 6 device functions |
| `src/device_dispatch.h` | Rewritten with 2×2 CUDA/Metal preprocessor matrix |
| `src/dispatch.h` | Updated FP16/BF16 guards and `constexpr D` binding for Metal |
| `tests/test_dispatch_macros.cc` | New compile-only verification test |

---

## Implementation Details

### `include/ctranslate2/devices.h`

```cpp
enum class Device {
  CPU,
  CUDA,
  METAL   // added
};
```

### `src/metal/device.h` + `src/metal/device.mm`

Minimal shim exposing `ctranslate2::metal::get_device_count()`.
Implementation uses `MTLCreateSystemDefaultDevice() != nil ? 1 : 0`; always 1 on Apple Silicon.

### `src/devices.cc`

All six device-related functions updated:

| Function | Metal behaviour |
|----------|----------------|
| `str_to_device` | `"metal"` / `"METAL"` → `Device::METAL` (behind `CT2_WITH_METAL`); `"auto"` picks Metal when CUDA absent |
| `device_to_str` | `case Device::METAL: return "metal"` |
| `get_device_count` | delegates to `metal::get_device_count()` |
| `get_device_index<METAL>` | always returns 0 (single unified device) |
| `set_device_index<METAL>` | throws if index != 0 |
| `synchronize_device` | no-op + comment (unified memory; buffer flush in M2) |
| `synchronize_stream` | no-op + comment (deferred commit implemented in M2) |

### `src/device_dispatch.h`

Replaced the original CUDA-only switch with a full 2×2 preprocessor matrix:

| `CT2_WITH_CUDA` | `CT2_WITH_METAL` | CUDA case | METAL case |
|:-:|:-:|:-:|:-:|
| OFF | OFF | UNSUPPORTED | UNSUPPORTED |
| ON  | OFF | DEVICE_CASE | UNSUPPORTED |
| OFF | ON  | UNSUPPORTED | DEVICE_CASE |
| ON  | ON  | DEVICE_CASE | DEVICE_CASE |

Each `DEVICE_CASE` sets `constexpr Device D` so downstream templates specialise correctly.

### `src/dispatch.h`

`DEVICE_AND_FLOAT_DISPATCH` updated in two ways:

1. **CPU-only guard** changed from `#ifndef CT2_WITH_CUDA` → `#if !defined(CT2_WITH_CUDA) && !defined(CT2_WITH_METAL)` so the Metal build uses the GPU branch.

2. **FP16/BF16 runtime guard** changed from `D == Device::CUDA` → `DEVICE != Device::CUDA && DEVICE != Device::METAL` to allow half-precision on both GPU backends.

3. **`constexpr D` binding** — was hardcoded `D = Device::CUDA`; now uses `DEVICE_DISPATCH(DEVICE, (STMTS))` so D is correctly `Device::METAL` when Metal is the runtime device.

---

## Verification

Four-combination syntax-only compilation check (`tests/test_dispatch_macros.cc`):

```bash
while IFS='|' read -r label flags; do
  printf "%-20s " "$label:";
  clang++ -std=c++17 -fsyntax-only -I include -I src $flags tests/test_dispatch_macros.cc \
    2>&1 && echo "OK" || echo "FAIL";
done <<'EOF'
CPU-only            |
Metal-only          |-DCT2_WITH_METAL
CUDA-only           |-DCT2_WITH_CUDA
CUDA+Metal          |-DCT2_WITH_CUDA -DCT2_WITH_METAL
EOF
```

```
CPU-only            : OK
Metal-only          : OK
CUDA-only           : OK
CUDA+Metal          : OK
```

### Macro comma-protection pattern

Template calls with two type parameters cannot be passed directly as a macro argument
because the preprocessor tokenises the comma as an argument separator.
The correct idiom (used in the verification test and throughout the codebase via `SINGLE_ARG`):

```cpp
// Wrong — preprocessor sees 5 args to a 4-arg macro:
DEVICE_AND_FLOAT_DISPATCH("op", d, t, check<D, T>());

// Correct — outer parens protect the comma:
DEVICE_AND_FLOAT_DISPATCH("op", d, t, (check<D, T>()));

// Also correct — SINGLE_ARG variadic absorbs the comma at the inner call level:
DEVICE_DISPATCH(d, SINGLE_ARG(foo<D, T>()));
```

---

## Known Limitations / Next Steps

- `synchronize_stream(Device::METAL)` is a no-op; real deferred command-buffer commit implemented in **Milestone 2**.
- No Metal-specific primitives yet — all `primitives<Device::METAL>` specialisations will throw `unsupported device` until implemented in subsequent milestones.
- `src/metal/device.mm` requires `-framework Metal -framework Foundation` at link time; CMake integration is **Milestone 1.2**.
