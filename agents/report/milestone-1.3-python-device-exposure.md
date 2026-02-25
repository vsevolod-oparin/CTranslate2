# Milestone 1.3 — Python Device Exposure

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Expose `Device::METAL` to Python so that `"metal"` is a valid device string throughout
the Python API and `get_supported_devices()` returns it on Metal-capable builds.

---

## Files Changed

| File | Change |
|------|--------|
| `python/cpp/storage_view.cc` | Added `"metal"` value to the `Device` enum pybind11 binding |
| `python/cpp/module.cc` | Added `get_supported_devices()` helper and `get_metal_device_count()` |
| `python/ctranslate2/__init__.py` | Exported both new symbols from `_ext` |

---

## Implementation Details

### `python/cpp/storage_view.cc` — Device enum

```cpp
py::enum_<Device>(m, "Device")
  .value("cpu",   Device::CPU)
  .value("cuda",  Device::CUDA)
  .value("metal", Device::METAL)   // added
  ;
```

This makes `ctranslate2.Device.metal` valid Python and allows `device="metal"` to be
accepted by all model constructors (`Translator`, `Generator`, `Encoder`, etc.) that
call `str_to_device()` internally.

### `python/cpp/module.cc` — new functions

**`get_supported_devices()`** — returns the devices available at runtime in this build:

```cpp
static std::vector<std::string> get_supported_devices() {
  std::vector<std::string> devices = {"cpu"};
  if (ctranslate2::get_device_count(ctranslate2::Device::CUDA) > 0)
    devices.push_back("cuda");
  if (ctranslate2::get_device_count(ctranslate2::Device::METAL) > 0)
    devices.push_back("metal");
  return devices;
}
```

Uses `get_device_count()` (already handles `#ifdef CT2_WITH_*` internally) so no
preprocessor guards are needed here. On a CPU-only build `get_device_count(METAL)`
returns 0 and `"metal"` is never added.

**`get_metal_device_count()`** — symmetric with the existing `get_cuda_device_count()`:

```cpp
m.def("get_metal_device_count", []() {
  return ctranslate2::get_device_count(ctranslate2::Device::METAL);
}, "Returns the number of visible Metal devices (0 or 1).");
```

### `python/ctranslate2/__init__.py`

```python
from ctranslate2._ext import (
    ...
    get_cuda_device_count,
    get_metal_device_count,    # added
    get_supported_compute_types,
    get_supported_devices,     # added
    ...
)
```

---

## PASS Criteria (to verify once Python extension is built)

```python
import ctranslate2

# On a WITH_METAL=ON build with Apple Silicon:
assert "metal" in ctranslate2.get_supported_devices()
assert ctranslate2.get_metal_device_count() == 1

# Device enum accessible:
assert ctranslate2.Device.metal == ctranslate2.Device.metal

# On a CPU-only build:
assert ctranslate2.get_supported_devices() == ["cpu"]
assert ctranslate2.get_metal_device_count() == 0
```

`ctranslate2.Translator("model", device="metal")` will raise at model-load time (no
Metal primitives yet), but must not raise an `ImportError` or `ValueError` for the
device string itself — `str_to_device("metal")` succeeds since M1.1.

---

## Known Limitations / Next Steps

- The Python extension must be rebuilt (`pip install -e .` or `cmake --build`) for
  these changes to take effect.
- `get_supported_compute_types("metal")` will fall back to `float32` only until
  `mayiuse_bfloat16` / `mayiuse_float16` are implemented for Metal (M4+).
- Python wheel packaging (`setup.py` / `pyproject.toml`) does not yet pass
  `-DWITH_METAL=ON` to CMake — wheel build integration is out of scope for M1.

---

## ⚠️ Do Not Attempt to Build the Python Extension After M1.3

The Python extension links against `libctranslate2.dylib`, which currently **fails to
link** due to missing `primitives<Device::METAL, T>` specialisations. Every op file
that calls `DEVICE_AND_FLOAT_DISPATCH` instantiates Metal template specialisations at
link time, and none are defined yet. You will see dozens of errors like:

```
"void ctranslate2::ops::LayerNorm::compute<(ctranslate2::Device)2, float>(...)"
— undefined symbol
```

**The Python extension becomes buildable in M2**, once a `primitives<Device::METAL>`
stub (throwing `std::runtime_error("not implemented")`) is in place to satisfy the
linker. Do not attempt to build or test the extension before that point.
