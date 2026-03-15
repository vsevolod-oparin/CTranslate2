# M13.4: Documentation

**Date:** 2026-03-15
**Status:** ✅ Complete

## Summary

Added Apple Silicon / Metal backend documentation across four files: hardware support, installation, environment variables, and architecture reference.

## Changes

### 1. `docs/hardware_support.md`
- Renamed "GPU" section to "GPU (NVIDIA)" for clarity
- Added "GPU (Apple Silicon / Metal)" section with:
  - Requirements (M1+, macOS 14+, source build)
  - Compute type support table (f32, f16, bf16, int8, int8_f16, int8_bf16)
  - Known limitations (AWQ, gemm_pack_b, RMSNorm residual, BF16 chip requirement)
  - Environment variable tips referencing `environment_variables.md`

### 2. `docs/installation.md`
- Added `WITH_METAL` row to CMake build options table
- Added Metal dependency note (macOS 14+, Apple Silicon, system frameworks)
- Added Metal+Accelerate multi-backend example
- Added "Building on Apple Silicon" subsection with cmake/make commands
- Documented `OPENMP_RUNTIME=NONE` recommendation for PyTorch coexistence

### 3. `docs/environment_variables.md`
- Added `CT2_MPS_ALLOW_BF16` — BF16 hardware detection override
- Added `CT2_METAL_POOL_MAX_MB` — Metal buffer pool size cap
- Added `CT2_MPS_TRACE` — command buffer tracing for profiling
- Added `CT2_DECODE_PROFILE` — decode loop timing breakdown

### 4. `ARCHITECTURE.md` Section 12
- Fixed env var names: `CT2_METAL_ALLOW_BF16` → `CT2_MPS_ALLOW_BF16`, `CT2_METAL_TRACE` → `CT2_MPS_TRACE`
- Added `CT2_METAL_POOL_MAX_MB` and `CT2_DECODE_PROFILE` to the table
- Note: Sections 6 (dispatch) and 9 (memory/allocator) already contained Metal backend entries from prior work

## Files Modified
- `docs/hardware_support.md`
- `docs/installation.md`
- `docs/environment_variables.md`
- `ARCHITECTURE.md`
- `APPLE_M4_METAL_PLAN.md` (marked M13.4 ✅)
