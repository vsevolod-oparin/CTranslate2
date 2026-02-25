# Milestone 5.2 — Metal Op Specializations

**Date:** 2026-02-25
**Status:** ✅ DONE — 30/30 tests pass

---

## Goal

Add `Device::METAL` template specializations for ops (LayerNorm, RMSNorm, Softmax,
BiasAdd, Gather) in `src/ops/*_metal.mm`. Wire them to GPU kernels implemented in
new `.metal` files and registered via `gen_msl_strings.py`.

---

## Scope

Ops that already work on Metal without new files (handled by inline code or existing
dispatch): Add, Mul, Transpose, Gemm/MatMul.

Ops that needed new `*_metal.mm` files: BiasAdd, LayerNorm, RMSNorm, SoftMax, Gather.

---

## Files Added

| File | Description |
|------|-------------|
| `src/metal/kernels/normalization.metal` | MSL kernels: layer_norm, rms_norm, softmax (float/half/bfloat) |
| `src/metal/kernels/gather.metal` | MSL gather kernel (float/half/bfloat/int/short/char) |
| `src/metal/ops_metal.h` | C++ declarations for the 4 dispatch wrapper templates |
| `src/ops/normalization_metal.mm` | LayerNorm, RMSNorm, SoftMax Metal specializations |
| `src/ops/gather_metal.mm` | Gather Metal specialization (all 6 types) |
| `src/ops/bias_add_metal.mm` | BiasAdd Metal specialization (reuses existing broadcast prims) |
| `tests/metal/normalization_gather_test.mm` | 30-test correctness suite |

## Files Modified

| File | Change |
|------|--------|
| `tools/gen_msl_strings.py` | Added normalization + gather to KERNELS list (8 total) |
| `src/metal/msl_strings.h` | Regenerated (now has kNormalizationMSL + kGatherMSL) |
| `src/metal/primitives.mm` | Added dispatch functions + ctranslate2::metal wrapper templates |
| `CMakeLists.txt` | Added 3 .mm files to METAL_SOURCES; 2 .metal files to _MSL_METAL_SOURCES |

---

## Architecture

### MSL Kernel Design (normalization.metal)

All three normalization kernels use a "one threadgroup per row" design:
- Grid = [num_rows, 1, 1]; threadgroup = [256, 1, 1] (kNormBlock = 256)
- Threadgroup memory: `float[256]` for two-pass reduction
- All accumulation in float32 for correctness with half/bfloat inputs

**LayerNorm**: two-pass stable algorithm (pass 1: mean, pass 2: variance). Supports
optional gamma/beta via `has_gamma`/`has_beta` flags (null → bind input as dummy).

**RMSNorm**: single-pass (accumulate sum-of-squares, then normalize).

**SoftMax**: three-pass (max → sum_exp → normalize). Supports optional lengths masking
and log-softmax mode. Out-of-range slots zeroed when masking is active.

### MSL Kernel Design (gather.metal)

One thread per output element. `gid → slot = gid/copy_size, j = gid%copy_size`.
Supports batched gather via `num_indices_per_batch` argument.

### Dispatch Infrastructure (primitives.mm)

Anonymous-namespace dispatch functions for each kernel family:
- `dispatch_layer_norm()`, `dispatch_rms_norm()`, `dispatch_softmax()`
- `dispatch_gather()`

Exposed via `ctranslate2::metal::layer_norm_metal<T>` etc. (declared in ops_metal.h),
with explicit instantiations for all relevant types.

### Op Files Pattern

The op `*_metal.mm` files use the same **unspecialized template-body** pattern as
`*_gpu.cu` files — NOT partial specialization (which C++ doesn't allow for function
templates). The full template body `template <Device D, typename T> void Foo::compute`
is defined in the file, but only explicit instantiations for `Device::METAL` are
emitted. The linker resolves each concrete `(D, T)` call site to the appropriate TU.

### Constraints

| Op | Metal constraint |
|----|-----------------|
| LayerNorm | Only inner_size == 1 (last-axis normalization); throws for other axes |
| RMSNorm | use_residual = true not supported; throws if set |
| SoftMax | No constraints (full feature parity with CUDA) |
| Gather | Only axis == batch_dims (same as CPU path) |
| BiasAdd | Full feature parity (reuses existing broadcast primitives) |

---

## Test Results

```
=== M5.2 normalization + gather Metal tests ===

--- layer_norm ---
  PASS  layer_norm warmup (no exception)
  PASS  layer_norm f32: output mean ≈ 0 (got 0)
  PASS  layer_norm f32: output variance ≈ 1 (got 0.99999)
  PASS  layer_norm f32 gamma/beta: output mean ≈ 1 (got 1)
  PASS  layer_norm f32 gamma/beta: output variance ≈ 4 (got 4)
  PASS  layer_norm f16: output mean ≈ 0 (got 0)
  PASS  layer_norm f16: output variance ≈ 1 (got 0.99976)

--- rms_norm ---
  PASS  rms_norm warmup (no exception)
  PASS  rms_norm f32: output RMS ≈ 1 (got 1)
  PASS  rms_norm f32: y[0] ≈ x[0]/rms(x) (got 0.36515)
  PASS  rms_norm f16: output RMS ≈ 1 (got 0.99992)

--- softmax ---
  PASS  softmax warmup (no exception)
  PASS  softmax f32: output sums to 1 (got 1)
  PASS  softmax f32: all outputs positive
  PASS  log_softmax f32: all outputs <= 0
  PASS  log_softmax f32: exp(y) sums to 1 (got 1)
  PASS  softmax masked: active slots sum to 1 (got 1)
  PASS  softmax masked: inactive slots are zero

--- gather ---
  PASS  gather warmup (no exception)
  PASS  gather f32 copy_size=1: dst[0]=40 (got 40)
  PASS  gather f32 copy_size=1: dst[1]=20 (got 20)
  PASS  gather f32 copy_size=1: dst[2]=10 (got 10)
  PASS  gather f32 copy_size=4: rows correct
  PASS  gather int32: dst[0]=400
  PASS  gather int32: dst[1]=0
  PASS  gather int32: dst[2]=200
  PASS  gather batched: dst[0]=3  (b0,row3) (got 3)
  PASS  gather batched: dst[1]=1  (b0,row1) (got 1)
  PASS  gather batched: dst[2]=12 (b1,row2) (got 12)
  PASS  gather batched: dst[3]=10 (b1,row0) (got 10)

=== Results: 30 passed, 0 failed ===
```

### Test build command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/normalization_gather_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o normalization_gather_test && ./normalization_gather_test
```

---

## Key Design Notes

1. **Unspecialized template-body pattern**: Cannot partially specialize function
   templates in C++. The CUDA + Metal approach: define the full template body
   `template <Device D, typename T> void Foo::compute(...)` in each device-specific
   TU; only instantiate for the target device. Linker resolves correctly.

2. **Null gamma/beta/lengths buffers**: Metal cannot bind nil buffers. Solution: bind
   the input buffer (x) as a dummy and use `has_gamma`/`has_beta`/`has_lengths` flags
   to let the kernel skip scale/bias/masking application.

3. **threadgroupMemoryLength**: Normalization kernels use `float[256]` threadgroup
   scratch for all element types (accumulation always in float32).

4. **dispatchThreadgroups vs dispatchThreads**: Normalization uses
   `dispatchThreadgroups:threadsPerThreadgroup:` (fixed groups = outer_size). Gather
   uses `dispatchThreads:threadsPerThreadgroup:` (total threads = total_elements).

5. **gen_msl_strings.py + CMakeLists.txt**: Updated to include normalization.metal
   and gather.metal in both the code-generation pipeline and the CI sync check.

---

## Next Step

**M5.3** — Integration test: run the full CTranslate2 translation pipeline on Metal
to exercise all M5.2 op specializations end-to-end with a real model.
