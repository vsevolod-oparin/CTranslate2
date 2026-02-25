# Milestone 1.2 — CMake Integration

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Wire the Metal backend into the project's CMake build system behind a `WITH_METAL` option,
following the same pattern as `WITH_CUDA` and `WITH_HIP`.

---

## Files Changed

| File | Change |
|------|--------|
| `CMakeLists.txt` | Added `WITH_METAL` option, `METAL_SOURCES` list, configuration block, and framework linking |

---

## Implementation

### Changes to `CMakeLists.txt`

**1. Option declaration** (alongside `WITH_CUDA` / `WITH_HIP`):
```cmake
option(WITH_METAL "Compile with Apple Metal backend" OFF)
```

**2. Source list** (alongside `CUDA_SOURCES`):
```cmake
set(METAL_SOURCES
  src/metal/device.mm
)
```

**3. Configuration block** (before the `if(WITH_CUDA)` library-creation section):
```cmake
if(WITH_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "WITH_METAL=ON requires macOS (Apple platform)")
  endif()
  if(CMAKE_VERSION VERSION_LESS "3.16")
    message(FATAL_ERROR "WITH_METAL=ON requires CMake 3.16 or later (OBJCXX language support)")
  endif()
  message(STATUS "Compiling with Apple Metal backend")
  # Override the global 10.13 minimum: MPSGraph BF16 and Apple9 GPU family require macOS 14+.
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")
  enable_language(OBJCXX)
  add_definitions(-DCT2_WITH_METAL)
  list(APPEND SOURCES ${METAL_SOURCES})
endif()
```

**4. Framework linking** (after `target_link_libraries(... PRIVATE ${LIBRARIES})`):
```cmake
if(WITH_METAL)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph"
  )
endif()
```

---

## Design Decisions

### `CMAKE_OSX_DEPLOYMENT_TARGET` set to `14.0`, not `13.0`

The plan originally suggested `13.0`. Based on M0.2 findings:
- `MPSGraph` (needed for BF16 GEMM) requires macOS 12.0+
- BF16 hardware support (`MTLGPUFamilyApple9`) is M3+ which ships with macOS 14
- Setting `14.0` accurately represents the minimum OS for which BF16 is actually available

The global default remains `10.13` for non-Metal builds (unchanged).

### `enable_language(OBJCXX)` requires CMake ≥ 3.16

Native OBJCXX language support was added in CMake 3.16. The guard ensures a clean error
rather than a cryptic linker failure on older CMake versions.

### Metal sources appended to `SOURCES` (not a separate list)

Unlike `CUDA_SOURCES` (which require `cuda_add_library`), `.mm` files are compiled by the
same host compiler (AppleClang) and need no special compiler macro. Appending to `SOURCES`
makes them compile correctly in all three library-creation branches (`WITH_CUDA`,
`WITH_HIP`, plain `add_library`).

### Frameworks linked separately via `target_link_libraries`

Keeps the framework list visible and easy to extend as more Metal APIs are used in later
milestones (e.g., `MetalPerformanceShadersGraph` is already included for M0.2 BF16 path).

---

## Verification

### Prerequisites

Required submodules must be initialised before configuring:
```bash
git submodule update --init --recursive
```
Without this, CMake will fail on `third_party/spdlog` (missing `CMakeLists.txt`) and
`third_party/cxxopts` (CLI dependency) before reaching the Metal block.

### Step 1 — `cmake` configure

```bash
cmake -S . -B build \
  -DWITH_METAL=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=NONE \
  -DENABLE_CPU_DISPATCH=OFF
```

```
-- Compiling with Apple Metal backend
-- The OBJCXX compiler identification is AppleClang 17.0.0.17000603
-- Configuring done
-- Generating done
```

The object file does **not** exist yet at this point — configure only generates build files.

### Step 2 — `cmake --build`

```bash
cmake --build build --target ctranslate2 -j$(sysctl -n hw.logicalcpu)
```

```
[ 98%] Building OBJCXX object CMakeFiles/ctranslate2.dir/src/metal/device.mm.o  ✅
[100%] Linking CXX shared library libctranslate2.dylib
clang++: error: linker command failed  ← expected, see below
```

```bash
ls build/CMakeFiles/ctranslate2.dir/src/metal/device.mm.o  # confirms compilation
```

**Expected linker errors:** `primitives<Device::METAL, T>` specialisations are not yet
implemented (M2+). All `undefined symbol` errors at link time reference
`Device::METAL` template instantiations — this is by design.

### `cmake` configure — Metal OFF (regression check)

```bash
cmake -S . -B build -DWITH_METAL=OFF -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE
```

```
-- Configuring done   ← no Metal messages, no OBJCXX, no regression ✅
```

---

## Known Limitations / Next Steps

- Link will fail until `primitives<Device::METAL>` specialisations are provided (M2+).
- `MetalPerformanceShadersGraph.framework` is already in the link list; no extra step
  needed when the BF16 GEMM path is wired up in M4.
- Python wheel packaging (`setup.py` / `pyproject.toml`) not yet updated to pass
  `-DWITH_METAL=ON` — deferred to M1.3.
