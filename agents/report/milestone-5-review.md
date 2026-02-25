# M5 Metal Op Specializations — Code Review

This review covers Milestone 5 of the Apple Metal backend:

- **M5.1** — `DEVICE_AND_FLOAT_DISPATCH` updated to allow Metal FP16/BF16 (complete, 14/14 tests).
- **M5.2** — Metal op specializations: `LayerNorm`, `RMSNorm`, `SoftMax`, `BiasAdd`, `Gather`
  (`Add`, `Mul`, `Transpose`, `Gemm` are served by M4 primitives via header-inline `compute`
  methods and need no separate `_metal.mm` file).

Files reviewed:

| File | Role |
|------|------|
| `src/metal/kernels/normalization.metal` | MSL kernels: layer_norm, rms_norm, softmax |
| `src/metal/kernels/gather.metal` | MSL kernel: gather |
| `src/metal/primitives_norm_gather.mm` | Dispatch wrappers (M5.2) |
| `src/metal/ops_metal.h` | Declarations used by op files |
| `src/ops/normalization_metal.mm` | LayerNorm, RMSNorm, SoftMax specializations |
| `src/ops/gather_metal.mm` | Gather specialization |
| `src/ops/bias_add_metal.mm` | BiasAdd specialization |
| `tests/metal/normalization_gather_test.mm` | M5.2 correctness tests |
| `tests/metal/dispatch_test.mm` | M5.1 dispatch guard tests |
| `tests/metal/pso_warmup_test.mm` | Library compilation warmup |

---

## 1. Bugs and Correctness Risks

### 1.1 `bias_add_metal.mm` hardcodes `Device::METAL` instead of template parameter `D`

**Severity: Low**

`BiasAdd::compute` is declared as `template <Device D, typename T>` but its body always
calls `primitives<Device::METAL>::...` instead of `primitives<D>::...`:

```cpp
// bias_add_metal.mm — current
primitives<Device::METAL>::add_batch_broadcast(bias.data<T>(), ...);
primitives<Device::METAL>::add_block_broadcast(bias.data<T>(), ...);
```

The CPU version in `bias_add_cpu.cc` correctly uses `primitives<D>`. Since the only explicit
instantiations in this file are for `Device::METAL`, calling the Metal version from a non-Metal
context is currently impossible — so there is no runtime bug today. However, if the file is
ever used as a template for another backend, or if `DEVICE_AND_FLOAT_DISPATCH` routes a
different device here by mistake, the wrong primitives would be called silently.

**Fix:** Replace `primitives<Device::METAL>` with `primitives<D>` in both call sites.

---

### 1.2 LayerNorm: non-last-axis case silently falls through for `inner_size == 1` when `axis != last`

**Severity: Low**

The guard in `normalization_metal.mm` is:

```cpp
if (inner_size != 1)
  throw std::invalid_argument(
      "Metal LayerNorm: only last-axis normalization is supported ...");
```

This allows calls where `inner_size == 1` but `axis` is not the last axis — for example,
normalizing axis 0 of a 1-D tensor has `inner_size == 1` and would pass through without error.
The GPU kernel uses `row_off = tgid * N` and `gamma[j]` / `beta[j]` indexed by column, which
is correct only for last-axis normalization. For a degenerate case like a 2-D tensor
`[batch=1, features=N]` normalized on axis 0 (`outer_size=1, axis_size=batch=1, inner_size=N`),
the guard fires (`inner_size=N != 1`), so it is caught.

A more precise guard would check `axis == input.rank() - 1` (consistent with what the CUDA
path optimises for), but for M5 the functional impact is minimal: `inner_size == 1` with a
non-last axis only arises in contrived shapes. Documenting the true invariant would prevent
future confusion.

---

### 1.3 Softmax: `active_N == 0` produces `log(0) = -inf` and `NaN` if `log_mode` reads max_val

**Severity: Low (edge case)**

When `has_lengths == 1` and `lengths[tgid] == 0` (a fully-masked row), the kernel:

1. Pass 1 (max): `mx = -FLT_MAX` for all threads → `max_val = -FLT_MAX` after reduction.
2. Pass 2 (sum-exp): loop over `[0, active_N)` doesn't execute → `sum_e = 0` →
   `total_sum = 0`.
3. Pass 3 (log-softmax): `log(total_sum) = log(0) = -inf`. The output loop over
   `[0, active_N)` doesn't execute, so no values are written.
4. Zero-fill: `for (uint j = active_N + tid; j < N; j += NORM_BLOCK)` fills all N slots with
   `(T)0`.

So the output for a fully-masked row is all zeros — the same as the softmax path — and is
numerically benign. **No bug, but the logic is non-obvious.** A comment explaining why
`log(0) = -inf` is harmless here (because the loop body that would divide/subtract by it
never executes) would help future readers.

---

## 2. Code Quality and Maintainability

### 2.1 `ops_metal.h` comment is stale after the primitives split

**Severity: Low**

The header comment reads:

```cpp
// Declarations for Metal dispatch functions used by high-level ops
// (LayerNorm, RMSNorm, SoftMax, Gather).  Implemented in primitives.mm.
```

After the M4 refactoring split, these functions are now implemented in
`src/metal/primitives_norm_gather.mm`, not `primitives.mm` (which is now an empty stub).

**Fix:** Update the comment to say `Implemented in primitives_norm_gather.mm`.

---

### 2.2 `pso_warmup_test.mm` describes "six kernel groups" but there are now eight

**Severity: Low**

The prose comment at line 3 of `pso_warmup_test.mm` says:

```
// Each of the six kernel groups in primitives.mm has its own MSL source string
```

M5.2 added two more MSL libraries (`normalization` and `gather`), bringing the total to eight.
The test only covers the original six and makes no mention of the two new ones.

This is both a stale comment and an incomplete test — see section 4.1 for the test gap.

---

### 2.3 `normalization_metal.mm` comment on `inner_size` constraint could be more precise

**Severity: Low (documentation)**

The comment and error message say "only last-axis normalization is supported (inner_size must
be 1)". As noted in 1.2, the true invariant is `axis == rank - 1`, not just `inner_size == 1`.
The CUDA implementation (in `layer_norm_gpu.cu`) makes this distinction explicit by checking
`axis == input.rank() - 1` to choose between `LayerNormForwardCUDAKernel` and
`LayerNormAxisForwardCUDAKernel`.

---

### 2.4 RMSNorm `use_residual` limitation is undocumented at the API level

**Severity: Low**

`rms_norm.h` declares `RMSNorm(const float epsilon = 1e-6, const bool use_residual = false)`.
The `use_residual` flag fuses a residual add into the norm (used in Qwen-style models).

The Metal implementation throws:
```cpp
if (_use_residual)
  throw std::invalid_argument("Metal RMSNorm: use_residual is not supported on Metal");
```

This is a runtime exception with no static / compile-time indication to the caller. A comment in
`rms_norm.h` noting the Metal limitation (similar to how some CTranslate2 ops document
backend-specific constraints) would help model authors targeting Metal.

---

## 3. Performance Observations

### 3.1 `NORM_BLOCK = 256` is suboptimal for small axis sizes

**Severity: Low**

All three normalization kernels always launch threadgroups of 256 threads, regardless of
`axis_size` / `depth`. For small hidden dimensions (e.g. N = 64 or N = 128), most threads do
no useful work in the inner loops (the `for (uint j = tid; j < N; ...)` loop is empty), but
still participate in the barrier-reduction passes. For N = 64, 75% of threads are idle throughout
the computation.

Typical transformer hidden dimensions (512, 768, 1024, 2048, 4096) are all ≥ 256, so this is
not an issue for production models. For small models or small intermediate projections, a
dynamic threadgroup size (clamped to the next power of two ≤ N and ≤ 256) would improve
utilisation — but this requires the MSL kernel to treat the threadgroup size as a runtime
parameter rather than a compile-time constant, which complicates the reduction loop.

**Suggested approach (M5+ or M7):** Pass threadgroup size as a `constant uint& TGSIZE` kernel
argument; dispatch `min(kNormBlock, next_pow2_geq(N))` threads per group. This is the same
approach used by many production softmax kernels.

---

### 3.2 Softmax: `exp(x - max_val)` computed twice for regular softmax

**Severity: Low**

In the regular softmax path (`log_mode == false`), the kernel computes:

- Pass 2: `sum_e += exp(x[j] - max_val)` (for the denominator)
- Pass 3: `y[j] = (T)(exp(x[j] - max_val) / total_sum)` (writes each output)

Each output element requires two `exp` evaluations. For log-softmax, the output pass uses
`x[j] - max_val - log_sum` (no `exp`), so it is more efficient.

For standard softmax with N ≈ vocabulary size (e.g. 32k–128k tokens), re-evaluating
`exp` for every output element doubles the transcendental cost. A cache of intermediate `exp`
values would require N floats of storage per row, which is impractical for large N. The
standard mitigation is a two-pass approach that stores `exp` values in registers between passes
— possible when N ≤ threadgroup size, but not for large N.

This is a known trade-off in softmax implementations. For attention scores (N = sequence
length, typically ≤ 4096), the cost is acceptable. For vocabulary softmax (N = 32k–128k),
the double `exp` is the dominant cost. **No immediate action needed;** document as a known
limitation for large-vocabulary softmax.

---

### 3.3 `gather_metal` uses a 1D dispatch for arbitrarily large `total_elements`

**Severity: Low (forward-looking)**

`dispatch_gather` launches `total_elements` threads in a 1D grid:

```objc
[enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_elements), 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
```

For a vocabulary gather with 32,000 indices × embedding_dim 4096 = 131M elements, this is a
131M-wide 1D grid. Metal on M4 supports up to 2^32 threads in x, so this is within bounds.
However, very large 1D grids can have scheduling inefficiencies on some GPU families; a 2D grid
(`[copy_size, num_indices, 1]`) would expose more parallelism structure to the scheduler.

Practically, for current model sizes this is not a measurable issue. Flag for future review if
large embedding lookups become a bottleneck.

---

## 4. Suggested New Tests

### 4.1 Extend `pso_warmup_test` to cover normalization and gather libraries

**Status: Missing**

`pso_warmup_test.mm` tests 6 of the 8 MSL libraries. The normalization and gather libraries
added in M5.2 are not covered. If their MSL source has a syntax error, the warmup test would
not catch it; the failure would only surface when `normalization_gather_test` runs (and it
would appear as an exception rather than a build failure).

**Suggested addition:** Two new entries in `pso_warmup_test.mm`:

```
--- warmup: normalization library ---
  PASS  normalization library compiles (layer_norm_float)
  PASS  normalization library compiles (softmax_half)

--- warmup: gather library ---
  PASS  gather library compiles (gather_float)
  PASS  gather library compiles (gather_int)
```

The test count would increase from 12 to ≥16.

---

### 4.2 Test BF16 for LayerNorm and RMSNorm

**Status: Missing**

`normalization_gather_test.mm` tests LayerNorm and RMSNorm for `float` and `half` but not
`bfloat`. Both types are instantiated (`DECLARE_IMPL(bfloat16_t)`) and the MSL kernels are
compiled under `#if defined(__HAVE_BFLOAT__)`. Without a test, a silent MSL compile failure
(on pre-Apple9 hardware or an SDK that doesn't define `__HAVE_BFLOAT__`) would go undetected.

**Suggested tests:**

```
layer_norm bf16: output mean ≈ 0, output variance ≈ 1 (tol 0.1)
rms_norm   bf16: output RMS ≈ 1 (tol 0.1)
softmax    bf16: outputs sum to 1 (tol 0.01)
```

---

### 4.3 Test all gather element types

**Status: Partially covered**

`normalization_gather_test.mm` tests `gather_metal<float>` and `gather_metal<int32_t>` but
not `float16_t`, `bfloat16_t`, `int16_t`, or `int8_t`. Six types are instantiated; four are
untested.

For the non-float integer types (`int16_t`, `int8_t`), the MSL kernel maps them to `short` and
`char` — both present in Metal's built-in type set. If the type mapping in
`MetalTypeName<T>::value` (in `primitives_infra.h`) were wrong for any of these types, the
kernel lookup would fail at runtime. A smoke test for each type catches this early.

**Suggested tests:** `gather_metal<float16_t>`, `gather_metal<bfloat16_t>`,
`gather_metal<int16_t>`, `gather_metal<int8_t>` — basic copy_size=1 correctness.

---

### 4.4 Test exception paths for LayerNorm `inner_size != 1` and RMSNorm `use_residual`

**Status: Missing**

The two documented limitations have no tests:

1. `LayerNorm: inner_size != 1` → throws `std::invalid_argument`
2. `RMSNorm: use_residual = true` → throws `std::invalid_argument`

Without these tests, a refactoring that removes the guards (e.g. someone who implements the
missing axis support and forgets to remove the guard, or conversely someone who accidentally
removes a guard without implementing the feature) would not be caught.

**Suggested tests:**

```cpp
// inner_size != 1 throws (axis=0 on a 2D tensor)
metal::layer_norm_metal<float>(x, gamma, beta, y, N, 1, /*epsilon*/1e-5f);  // outer=1, axis_size=N
// ^ must call through normalization_metal.mm with a shape where inner_size = product_after_axis > 1

// use_residual = true throws
RMSNorm rms(1e-6, /*use_residual=*/true);
rms(gamma, input, output);  // Device::METAL → must throw
```

---

### 4.5 CPU-vs-Metal comparison test for normalization and softmax

**Status: Missing**

All current M5.2 tests check analytical properties (mean ≈ 0, sum ≈ 1, etc.) but do not
compare Metal output against a CPU reference. The CUDA op test pattern (from the plan's M5.2
spec) calls for:

```cpp
// CPU reference
LayerNorm()(beta_cpu, gamma_cpu, input_cpu, output_cpu);
// Metal
LayerNorm()(beta_metal, gamma_metal, input_metal, output_metal);
// Metal→CPU copy then compare
CHECK: max(|metal - cpu|) < tol
```

Analytical checks miss off-by-one errors in the row-offset computation (`row_off = tgid * N`)
and subtle numerical differences between the two-pass Metal algorithm and the one-pass CUDA
algorithm. A comparison test using randomly-filled inputs would catch both.

**Suggested approach:** A new `normalization_comparison_test.mm` that:
1. Generates random f32 input, gamma, beta on CPU Metal buffer.
2. Runs CPU LayerNorm / RMSNorm / SoftMax (commit_and_wait first to get CPU access).
3. Runs Metal LayerNorm / RMSNorm / SoftMax on the same data.
4. Syncs and compares element-wise: max abs error < 1e-4 for f32.

---

### 4.6 BiasAdd Metal test

**Status: Missing**

`bias_add_metal.mm` has no test. The implementation reuses `add_batch_broadcast` /
`add_block_broadcast` from M4.6, but three untested paths exist:

1. **Last-axis bias** (`_axis == -1` or `_axis == rank-1`): calls `add_batch_broadcast`.
2. **Non-last-axis bias** (`_axis` < rank-1): calls `add_block_broadcast` with `width`
   computation from axis dimensions.
3. **Residual addition** (`residual != nullptr`): calls `Add()(*residual, output, output)`.
4. **Activation fusion** (`_activation_type != nullptr`): calls activation on output.

Path 2 and paths 3+4 in combination are completely untested on Metal. Path 2 especially uses
a `width` calculation that could be wrong for non-trivial axis values.

**Suggested tests:** A new `bias_add_test.mm` with:
```
bias_add last-axis f32: output[i] == value[i] + bias[i % bias_size]
bias_add mid-axis f32: block broadcast correctness
bias_add with residual: output == value + bias + residual
bias_add with relu activation: negative values clamped to 0
```

---

## 5. Summary Table

| # | Category | Severity | Status | Item |
|---|----------|----------|--------|------|
| 1.1 | Bug | Low | ✅ Fixed (2026-02-25) | `bias_add_metal.mm` hardcodes `Device::METAL` instead of template param `D` |
| 1.2 | Bug | Low | ✅ Fixed (2026-02-25) | LayerNorm guard comment/message clarified: `inner_size == 1` is the correct invariant |
| 1.3 | Bug | Low | ✅ Fixed (2026-02-25) | Softmax `active_N == 0` log(0) path explained with inline comment in MSL kernel |
| 2.1 | Quality | Low | Open | `ops_metal.h` comment says "Implemented in primitives.mm" (stale after split) |
| 2.2 | Quality | Low | Open | `pso_warmup_test.mm` says "six kernel groups" — now 8; normalization/gather not tested |
| 2.3 | Quality | Low | Open | LayerNorm constraint description imprecise (`inner_size == 1` vs `axis == rank-1`) |
| 2.4 | Quality | Low | Open | `use_residual` Metal limitation not documented in `rms_norm.h` API |
| 3.1 | Perf | Low | Deferred (M7+) | `NORM_BLOCK = 256` wastes threads for small N; dynamic threadgroup size would improve utilisation |
| 3.2 | Perf | Low | Deferred | Softmax recomputes `exp(x - max)` twice per element; unavoidable for large N but worth documenting |
| 3.3 | Perf | Low | Deferred | `gather_metal` 1D dispatch; 2D grid would expose more scheduler parallelism for large embeddings |
| 4.1 | Test | — | Open | `pso_warmup_test` missing normalization and gather library warmup entries |
| 4.2 | Test | — | Open | No BF16 tests for LayerNorm, RMSNorm, SoftMax |
| 4.3 | Test | — | Open | Gather: only f32 and int32 tested; f16, bf16, int16, int8 untested |
| 4.4 | Test | — | Open | No tests for `inner_size != 1` throw and `use_residual` throw paths |
| 4.5 | Test | — | Open | No CPU-vs-Metal comparison test for norm/softmax (only analytical checks) |
| 4.6 | Test | — | Open | `BiasAdd::compute<Device::METAL>` has no test (mid-axis, residual, activation paths) |

---

## 6. Completeness Assessment (M5.2 Scope vs. Implementation)

The M5.2 plan required Metal specializations for:

| Op | Required | Implemented | Notes |
|----|----------|-------------|-------|
| `LayerNorm` | ✅ | ✅ | Last-axis only; non-last-axis throws |
| `RMSNorm` | ✅ | ✅ | `use_residual=true` throws |
| `SoftMax` / `LogSoftMax` | ✅ | ✅ | Full (masking + log mode) |
| `Gemm` / `MatMul` | ✅ | ✅ | Via M4.4 primitives; no `_metal.mm` file needed |
| `Add` | ✅ | ✅ | Header-inline via `primitives<D>::add` |
| `Mul` | ✅ | ✅ | Header-inline via `primitives<D>::mul` |
| `BiasAdd` | ✅ | ✅ | `bias_add_metal.mm`; untested beyond broadcast smoke test |
| `Transpose` | ✅ | ✅ | Via M4.8 primitives |
| `Gather` | ✅ | ✅ | `gather_metal.mm`; only f32/i32 tested |

All M5.2 ops are implemented. The functional gaps are documented limitations (not regressions):
- LayerNorm restricted to last axis
- RMSNorm `use_residual` not supported

The primary outstanding work is **test coverage** (items 4.1–4.6), with three low-severity
code quality fixes (1.1, 2.1, 2.3).

---

**Review date:** 2026-02-25
**Reviewer:** code-reviewer agent
**Tests at review time:** 20/20 Metal test files pass (558 total assertions)
