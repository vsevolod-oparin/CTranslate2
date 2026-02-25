# Milestone 5.2 — Metal Op Specializations

**Date:** 2026-02-25
**Status:** ✅ DONE — 22 test files, 601 assertions pass; m52_bench.mm 25/25 accuracy checks pass

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
| `tests/metal/normalization_gather_test.mm` | Correctness suite (extended to 25 tests after code review) |
| `tests/metal/normalization_comparison_test.mm` | CPU-reference vs Metal comparison (6 tests; added in code review 4.5) |
| `tests/metal/bias_add_test.mm` | BiasAdd primitive-level tests (11 assertions; added in code review 4.6) |
| `tests/metal/m52_bench.mm` | Accuracy + performance benchmark vs single-thread CPU (25 checks) |

## Files Modified

| File | Change |
|------|--------|
| `tools/gen_msl_strings.py` | Added normalization + gather to KERNELS list (8 total) |
| `src/metal/msl_strings.h` | Regenerated (now has kNormalizationMSL + kGatherMSL) |
| `src/metal/primitives_norm_gather.mm` | Dispatch functions + ctranslate2::metal wrapper templates (split from primitives.mm) |
| `CMakeLists.txt` | Added 3 .mm files to METAL_SOURCES; 2 .metal files to _MSL_METAL_SOURCES |
| `include/ctranslate2/ops/rms_norm.h` | Added comment: `use_residual=true` unsupported on Device::METAL (code review 2.4) |
| `tests/metal/pso_warmup_test.mm` | Extended to 17 tests (8 libraries; added normalization + gather warmup) |
| `APPLE_M4_METAL_PLAN.md` | M5.2 section updated to ✅ DONE with op table and review summary |

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

The full test suite is run via `tests/metal/run_all.sh`. Final state: **22/22 test files,
601 total assertions, all pass**. Key test files for M5.2:

| Test file | Assertions | Coverage |
|-----------|-----------|---------|
| `normalization_gather_test.mm` | 51+ | LayerNorm/RMSNorm/SoftMax/Gather f32/f16/bf16/int; multi-row; masked |
| `pso_warmup_test.mm` | 17 | All 8 MSL libraries compile; normalization + gather warmup |
| `normalization_comparison_test.mm` | 6 | C++ reference vs Metal; max abs error checks for all 3 norm ops + softmax |
| `bias_add_test.mm` | 11 | add_batch_broadcast (f32/f16), add_block_broadcast, residual path |

### Test build commands

```bash
# Full suite (recommended)
bash tests/metal/run_all.sh

# Individual test (split primitives — use this, not primitives.mm)
clang++ -std=c++17 -O0 \
    -I include -I src -DCT2_WITH_METAL \
    tests/metal/normalization_gather_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/metal/primitives_norm_gather.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o normalization_gather_test && ./normalization_gather_test
```

---

## Accuracy and Performance (m52_bench.mm, Apple M4)

Benchmarks: Metal standalone (encode + commit_and_wait) vs single-threaded C++.
GPU times include the ~0.4 ms command-buffer overhead; in a real pipeline this is
amortised, so the GPU crossover is lower than shown below.

All **25 accuracy checks pass** (max abs err within tolerance for all ops and shapes).

```
=== LayerNorm (float32, gamma+beta) ===
  Shape           max_abs_err  Status   GPU (µs)  CPU (µs)  Speedup
  [1 x 64]          5.6e-09    PASS        261.5       0.2     0.00x  CPU wins
  [4 x 256]         7.2e-07    PASS        260.0       2.5     0.01x  CPU wins
  [32 x 512]        1.2e-06    PASS        322.0      46.0     0.14x  CPU wins
  [256 x 1024]      1.9e-06    PASS        440.1     653.0     1.48x  GPU wins
  [512 x 4096]      3.8e-06    PASS        993.1    3549.6     3.57x  GPU wins

=== RMSNorm (float32, with gamma) ===
  [1 x 64]          2.4e-07    PASS        196.2       0.0     0.00x  CPU wins
  [32 x 512]        1.2e-06    PASS        177.7      12.9     0.07x  CPU wins
  [256 x 1024]      1.7e-06    PASS        241.3     204.5     0.85x  CPU wins
  [512 x 4096]      4.3e-06    PASS        436.5    1524.2     3.49x  GPU wins

=== SoftMax (float32, no masking) ===
  [32 x 512]        8.4e-09    PASS        175.9      68.8     0.39x  CPU wins
  [256 x 1024]      9.8e-09    PASS        206.1    1119.9     5.43x  GPU wins
  [512 x 8192]      3.3e-09    PASS       1817.8   18739.0    10.31x  GPU wins

=== Gather (float32) ===
  16  × 64  (src=512)    0.0e+00  PASS    186.2       0.0     0.00x  CPU wins
  256 × 256 (src=4096)   0.0e+00  PASS    198.6       2.8     0.01x  CPU wins
  4096× 512 (src=32768)  0.0e+00  PASS    978.1     133.1     0.14x  CPU wins

=== BiasAdd batch_broadcast (float32) ===
  total=4096   (bias=64)   0.0e+00  PASS    173.4       1.9     0.01x  CPU wins
  total=65536  (bias=128)  0.0e+00  PASS    177.5      30.7     0.17x  CPU wins
  total=1048576(bias=256)  0.0e+00  PASS    258.6     500.0     1.93x  GPU wins
  total=4194304(bias=512)  0.0e+00  PASS    576.5    2068.5     3.59x  GPU wins

=== BiasAdd block_broadcast (float32) ===
  [4,32,64]   total=8K    0.0e+00  PASS    178.0       7.4     0.04x  CPU wins
  [16,64,256] total=256K  0.0e+00  PASS    287.0     246.5     0.86x  CPU wins
  [64,128,512] total=4M   0.0e+00  PASS    688.2    3924.5     5.70x  GPU wins
```

**Key takeaways:**
- **Normalization** (LayerNorm/RMSNorm) GPU crossover ≈ 200K–2M elements; 3.5–3.6x at 2M.
- **SoftMax** GPU crossover ≈ 200K elements; 10x at 4M — best scaling of any M5.2 op.
- **Gather** GPU loses even at 2M output elements — random-access scatter is
  memory-bandwidth-bound in ways that favour the CPU L3 cache over GPU dispatch overhead.
- **BiasAdd** GPU crossover ≈ 1M elements (batch_broadcast) / ~1M (block_broadcast).
- In real pipeline (CB overhead amortised): all crossovers shift to much smaller sizes.

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

## Code Review

Full code review in `agents/report/milestone-5-review.md`. All items resolved:

| Section | Items | Status |
|---------|-------|--------|
| 1 — Bugs | 1.1 `primitives<D>` fix, 1.2 LayerNorm guard comment, 1.3 log(0) comment | ✅ All fixed |
| 2 — Quality | 2.1 ops_metal.h comment, 2.2 pso_warmup stale count, 2.3 normalization comment, 2.4 rms_norm.h annotation | ✅ All fixed |
| 3 — Performance | norm threadgroup sizing, softmax recompute, gather 1D dispatch | Deferred to M7+ |
| 4 — Tests | pso warmup (4.1), BF16 norm (4.2), gather types (4.3), edge cases (4.4), comparison (4.5), BiasAdd (4.6) | ✅ All added |

---

## Next Step

**M6** — Attention mechanisms on Metal (SDPA via MPS, KV-cache, RoPE, ALiBi).
