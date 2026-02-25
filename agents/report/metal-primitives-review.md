# Metal Primitives Code Review

This review covers the Apple Metal backend primitive layer as implemented across milestones M4.1 through M4.8. The primary source under review is `src/metal/primitives.mm` (~2048 lines), with supporting analysis of `src/metal/utils.mm`, `src/metal/utils.h`, and `src/metal/allocator.mm`. The review does not propose code changes; it catalogues correctness risks, code quality observations, performance notes, and missing test coverage.

---

## 1. Bugs and Correctness Risks

### 1.1 `convert<U,V>` reads stale GPU data ✅ FIXED (2026-02-25)

**Was: Medium — now resolved.**

`primitives<Device::METAL>::convert` was implemented as `std::copy(x, x + size, y)` — a CPU operation executed directly against the unified-memory pointer, with no guard to flush pending GPU writes first.

**Applied fix** (`src/metal/primitives.mm`): Added `metal::commit_and_wait()` at the start of `convert`, consistent with the guard already used by `at()`, `logsumexp()`, and `prepare_length_mask()`:

```cpp
template<>
template <typename U, typename V>
void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
  metal::commit_and_wait();  // flush pending GPU writes before CPU read
  std::copy(x, x + size, y);
}
```

**Test added:** `tests/metal/convert_test.mm` (8/8 pass).
The primary test (`convert flushes GPU writes`) encodes an `add_scalar` GPU kernel on the source buffer, then immediately calls `convert()` without an explicit sync and verifies the result reflects the GPU-written values (not stale pre-GPU values).

Original description:

If `x` was written by a pending (uncommitted) GPU kernel, `std::copy` reads stale data. On Apple Silicon, the GPU and CPU share physical DRAM, but the Metal API gives no guarantee that GPU writes are visible to the CPU until the command buffer containing those writes has been committed and `waitUntilCompleted` has returned.

All other CPU-reads-after-GPU-write patterns in this file guard with `metal::commit_and_wait()`:

- `at<T>()` (line 1354): `metal::commit_and_wait()` with comment "flush any pending GPU writes to x".
- `logsumexp<T>()` (line 1774): `metal::commit_and_wait()` with comment "flush any pending GPU writes to x".
- `prepare_length_mask()` (line 1696): `metal::commit_and_wait()` with comment "flush any pending GPU writes to lengths".

`convert` was the only CPU-read primitive that did not flush.

---

### 1.2 `min` and `max` element-wise overloads are runtime stubs ✅ FIXED (2026-02-25)

**Was: High (runtime throw) — now resolved.**

Four public API overloads unconditionally throw `std::runtime_error` at runtime via `METAL_STUB`:

```
// src/metal/primitives.mm:1611-1631
template<>
template <typename T>
void primitives<Device::METAL>::min(T a, const T* x, T* y, dim_t size) {
  METAL_STUB(min);
}

template<>
template <typename T>
void primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size) {
  METAL_STUB(min);
}

template<>
template <typename T>
void primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size) {
  METAL_STUB(max);
}

template<>
template <typename T>
void primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size) {
  METAL_STUB(max);
}
```

These are the element-wise clamp (scalar-vs-vector) and element-wise min/max (vector-vs-vector) operations. They are distinct from the reduction `max(const T* array, dim_t size)` which is implemented.

The full MSL infrastructure for element-wise binary and scalar dispatch is already in place (`dispatch_binary`, `dispatch_scalar`, `kElementwiseMSL`, `get_elementwise_pso`). The only missing pieces are MSL kernel definitions for the min/max operations and corresponding C++ dispatch wiring — a straightforward extension of the pattern already established by `add`, `sub`, and `mul`.

One specific caution: MSL's `min()` and `max()` overloads for `bfloat` may be absent in some SDK versions, which is precisely why `reduce_max` uses an explicit `>` comparison rather than calling `max()`. Any new min/max element-wise kernels should follow the same convention (ternary with cast through `float`) rather than relying on MSL standard library overloads for `bfloat`.

---

### 1.3 `uint32_t` truncation of `dim_t` in GPU kernel arguments ✅ FIXED (2026-02-25)

**Was: Low (theoretical) — now resolved.**

**Applied fix** (`src/metal/primitives.mm`):
- Added `ct2_u32(dim_t v)` helper: validated narrowing cast that throws `std::runtime_error` with a clear message instead of silently truncating values above 2^32-1. Also rejects negative values.
- Added `(void)ct2_u32(size);` guard at the top of all six dispatch helpers (`dispatch_binary`, `dispatch_scalar`, `dispatch_unary`, `dispatch_broadcast1`, `dispatch_broadcast2`, `dispatch_transpose`) — covers the implicit MSL `uint gid` truncation path.
- Replaced all 16 raw `static_cast<uint32_t>(dim_t_expr)` call-sites in reduction, broadcast, penalize, and transpose dispatches with `ct2_u32(...)`.
- Added `#include <limits>` to support `std::numeric_limits<uint32_t>::max()`.

**Test added:** `tests/metal/truncation_test.mm` (9/9 pass).
Key tests: `dispatch_binary` and `dispatch_scalar` paths throw on `UINT32_MAX + 1`; negative `dim_t` also throws; normal-sized dispatch produces correct results (regression guard).

Original description:

Several primitives cast `dim_t` (which is `int64_t` on 64-bit platforms) to `uint32_t` before passing it as a kernel argument silently. The MSL kernels use `uint gid [[thread_position_in_grid]]` (32-bit); for element counts above 2^32, `gid` wraps to zero and elements beyond that index are silently skipped. The truncation was silent — corruption would only be discovered through a future data bug.

---

## 2. Code Quality and Maintainability

### 2.1 PSO creation boilerplate repeated across six kernel groups ✅ FIXED (2026-02-25)

**Was: Medium — now resolved.**

**Applied fix** (`src/metal/primitives.mm`): Added `make_pso(lib, name)` and `PSOCache` struct before the first library function. Each of the six `get_*_pso` functions is now a 3-line one-liner:

```cpp
static id<MTLComputePipelineState> get_elementwise_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_elementwise_library, name);
}
```

`PSOCache` holds a `std::unordered_map` + `std::mutex` and a templated `get(LibFn, name)` method. `make_pso` performs the function lookup and PSO creation with uniform error handling. ~180 lines of duplicated boilerplate reduced to ~18 lines (6 × 3).

---

### 2.2 Library compilation boilerplate repeated across six kernel groups ✅ FIXED (2026-02-25)

**Was: Medium — now resolved.**

**Applied fix** (`src/metal/primitives.mm`): Added `compile_library_once(flag, lib_out, msl_src, label, opts=nil)` helper. Each of the six `get_*_library` functions is now a 3-line one-liner:

```cpp
static id<MTLLibrary> get_elementwise_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kElementwiseMSL, "elementwise");
}
```

~90 lines of duplicated boilerplate reduced to ~18 lines (6 × 3).

---

### 2.3 MSL source duplicated between `.metal` files and embedded C strings ✅ FIXED (2026-02-25)

**Was: Medium (source of silent divergence) — now resolved.**

At the time of the review, 4 of 6 `.metal`/`k*MSL` pairs had already diverged (documentation comments had drifted). The fix eliminates the duplication by making `primitives.mm` consume the `.metal` files as the single source of truth.

**Applied fix:**

1. **`tools/gen_msl_strings.py`** (new): reads the 6 canonical `.metal` files and emits `src/metal/msl_strings.h` containing the 6 `k*MSL` constexpr string definitions. Two modes:
   - `python3 tools/gen_msl_strings.py` — regenerate after editing a `.metal` file.
   - `python3 tools/gen_msl_strings.py --check` — verify in sync; exits 1 with a clear error if not (for CI).

2. **`src/metal/msl_strings.h`** (new, auto-generated, checked in): replaces the 406 lines of inline string definitions in `primitives.mm`. The "do not edit" header and generation instructions make the file's origin unambiguous.

3. **`src/metal/primitives.mm`**: the 6 inline `k*MSL` string definitions (and their preceding "MSL source for..." comment blocks) are removed. A single `#include "metal/msl_strings.h"` replaces them.

4. **`CMakeLists.txt`**: added `check_msl_sync` custom target inside `if(WITH_METAL)`. Uses `add_custom_command` with the `.metal` files and `msl_strings.h` as dependencies — the check re-runs only when any of those files change, and touches a stamp file on success. The build fails with an actionable error message if `msl_strings.h` is out of sync.

**Verified:**
- `python3 tools/gen_msl_strings.py --check` exits 0 on clean state and exits 1 after a simulated `.metal` edit.
- All prior test suites still pass after the change (33 broadcast, 72 minmax, 8 convert, 9 truncation, 12 pso_warmup, 7 large_transpose, 6 reduce_sum_precision).

---

### 2.4 `MTLLanguageVersion3_1` on the activation library ✅ FIXED (2026-02-25)

**Was: Low — now resolved.**

**Applied fix** (`src/metal/primitives.mm`):
- Removed the `MTLCompileOptions` object and `MTLLanguageVersion3_1` from `get_activation_library()`. Now passes `nil` options, matching all other five groups.
- Updated the `ct2_erf` comment to state: "Metal Shading Language does not provide erf() in metal_stdlib, *even when MTLLanguageVersion3_1 is requested*. We therefore use a polynomial approximation; this also makes the activation library compile-option-free."
- The refactoring in 2.1/2.2 reinforces this: `get_activation_library` is now a one-liner with no options argument, making the absence of special options visually obvious.

---

### 2.5 Inconsistent error message format across kernel groups ✅ FIXED (2026-02-25)

**Was: Low — resolved as side effect of 2.1.**

All six `get_*_pso` functions now go through the single `make_pso` helper which emits:
- `"Metal: kernel not found: <name>"` for missing functions
- `"Metal: PSO creation failed for <name>: <Metal error>"` for creation failures

The former per-group prefixes (`"activation kernel not found"`, `"beam_search kernel not found"`, `"beam_search PSO creation failed for"`, etc.) are gone.

---

### 2.6 CPU-GPU coherency in `dispatch_mps_gemm` ✅ FIXED (2026-02-25)

**Was: Low — now resolved.**

**Applied fix** (`src/metal/primitives.mm`): Added an explanatory comment above the `pad_a`/`pad_b` buffer preparation block:

> CPU writes to freshly-allocated `MTLResourceStorageModeShared` buffers are immediately visible to the GPU on Apple Silicon unified memory. No explicit flush is required between the `memcpy` calls and the subsequent MPS encode because (a) each buffer is newly allocated so no prior GPU encoding holds a reference to it, and (b) the encode happens in the same thread immediately after the copy completes.

---

### 2.7 `buffer_for_ptr` performs a linear scan through all live allocations

**Severity: Low — deferred (out of scope)**

Acknowledged in the existing comment. O(N) scan is acceptable for current workloads (typically a few dozen live allocations). Replace with an interval tree or sorted array when/if KV cache growth makes it measurable.

---

## 3. Performance Observations

### 3.1 `alloc_temp_buffer` bypasses the MetalAllocator pool

**Severity: Low**

`alloc_temp_buffer(bytes)` calls `[device newBufferWithLength:bytes options:MTLResourceStorageModeShared]` directly, bypassing the `MetalAllocator` pool (lines 988-996). The allocator pool exists specifically to avoid repeated OS-level Metal buffer allocation/deallocation.

The functions that call `alloc_temp_buffer` are:

- `sum()` — once per call, for the partial results buffer.
- `max()` — once per call, for the partial results buffer.
- `amax()` — once per call, for the partial results buffer.
- `max_element()` — twice per call (values buffer and indices buffer).
- `dispatch_mps_gemm` — up to three times when `pad_a`, `pad_b`, or `pad_c` is true.
- `run_bf16_gemm_inner` — up to twice, when input buffers have non-zero offsets.

For inference workloads where reductions are called repeatedly (e.g., `amax` once per transformer layer for dynamic quantization scale computation, typically 24-96 times per forward pass for large models), this creates continuous OS-level allocation and deallocation pressure. Metal's internal allocator does maintain its own free list, so the cost is not as severe as a raw `malloc`/`free`, but it is measurable and avoidable.

**Suggested improvement:** Either route temporary allocations through `MetalAllocator` (potentially by adding a `free_after_sync` deferred-release path), or maintain a small set of pre-allocated reduction scratch buffers inside the reduction dispatch functions, keyed by size, reusing them across calls.

---

### 3.2 BF16 batched GEMM is fully sequential and each item incurs a full GPU round-trip

**Severity: Medium**

`gemm_batch_strided` for BF16 (lines 1877-1889):

```cpp
metal::commit_and_wait();  // flush once before batch
for (dim_t i = 0; i < batch_size; ++i) {
  run_bf16_gemm_inner(...);  // synchronous MPSGraph run per item
}
```

`run_bf16_gemm_inner` calls `[entry.graph runWithMTLCommandQueue:queue feeds:... targetTensors:... targetOperations:... ]`, which is documented to be synchronous — it submits a command buffer and blocks until completion. For a batch of 4, this is 4 × ~0.4 ms (per M0.3 latency measurements) = approximately 1.6 ms of serial GPU round-trip overhead for the batch loop itself, regardless of the matrix dimensions.

MPSGraph provides an asynchronous API: `runAsyncWithMTLCommandQueue:feeds:targetTensors:targetOperations:executionDescriptor:` which returns an `MPSGraphExecutionResults` future. Using it would allow the batch items to be submitted to the GPU command queue without waiting for each to complete before submitting the next, reducing the total overhead from `batch_size * per_CB_latency` to approximately `1 * per_CB_latency`.

This is particularly relevant for multi-head attention in decode mode (batch_size = number of heads or number of KV heads), where BF16 batch GEMM is called with many small matrices.

**Suggested improvement:** Investigate `runAsyncWithMTLCommandQueue:` for BF16 batch GEMM in a future milestone (M5+), measuring whether the async submission path reduces wall-clock latency for typical batch sizes.

---

### 3.3 Small-matrix FP32/FP16 GEMM with `pad_c` always commits the command buffer

**Severity: Low**

In `dispatch_mps_gemm`, when `pad_c` is true (lines 1157-1163), the function calls `ctranslate2::metal::commit_and_wait()` to flush the GPU before reading back the padded output. This unconditionally breaks the deferred-commit pipeline and pays the full ~0.4 ms command buffer overhead.

The condition `pad_c` arises when `ldc * sizeof(T)` is less than the MPS hardware minimum row stride. For FP16 with very few columns (e.g., n=1, n=2 — common in decode-phase attention with single output positions), MPS requires a minimum of 16 bytes per row while the natural stride is 2 or 4 bytes.

For decode-phase single-head attention (batch=1, seq=1, n=1), this GEMM call shape may occur multiple times per layer, each paying the 0.4 ms commit tax. For a 32-layer model, this could add 12+ ms per decode step.

**Suggested improvement (M5+):** For the `pad_c` path, encode the row-unpack step as a blit (copy) kernel into the same command buffer rather than committing and doing the unpack on the CPU. This keeps the deferred-commit pipeline intact. Alternatively, accumulate the GEMM result in a persistently allocated padded buffer and unpack lazily only when the result is actually needed by a CPU-read primitive.

---

### 3.4 `sum<T>` reduction threadgroup memory is allocated as `sizeof(T)` per thread

**Severity: Low (potential type-specific issue)**

In `sum()` (line 1434):

```objc
[enc setThreadgroupMemoryLength:kReductionTGS * sizeof(T) atIndex:0];
```

For `T = bfloat16_t`, `sizeof(T) == 2`. The MSL kernel `reduce_sum_bfloat` uses `threadgroup bfloat* shmem` and accumulates in `bfloat` precision. For large reductions (e.g., summing 65536 bfloat values in 256 partial groups of 256), the partial sum within each threadgroup accumulates up to 256 bfloat additions. BF16 has only 7 bits of mantissa, so the partial sum can lose precision for values that span a large dynamic range.

By contrast, `reduce_amax` always accumulates in `float` in its threadgroup memory, regardless of the input type — this is the correct design for precision-sensitive operations.

The `sum` operation is used for accumulation tasks where precision matters (e.g., computing normalisation denominators). Consider whether `reduce_sum` should also accumulate in `float` internally, especially for `bfloat` and `half` types, as `reduce_amax` does.

---

### 3.5 `convert` is CPU-only, missing a GPU path for large type conversions

**Severity: Low (forward-looking)**

`convert<U, V>` performs type conversion (e.g., float32 to float16, bfloat16 to float32) on the CPU via `std::copy`. For large tensors — for example, reading back a 50k-vocabulary logit tensor in float16 — this incurs a sequential CPU conversion loop over potentially millions of elements.

On Apple Silicon with unified memory, a CPU conversion loop is already fast compared to a DMA transfer on discrete GPU architectures. However, for very large conversions (10 MB+), a single-pass GPU kernel can complete in under a millisecond using wide SIMD parallelism, while leaving the CPU free for other work.

The existing `dispatch_unary` / `dispatch_binary` infrastructure could accommodate a type-conversion kernel. The MSL definition is straightforward: `y[gid] = (V)x[gid]` in a kernel parameterized on input and output types. This is a forward-looking optimization for M5+, not an urgent correctness fix.

---

## 4. Suggested New Tests

### 4.1 `convert_gpu_flush_test` ✅ COVERED (2026-02-25)

**Already implemented** in `tests/metal/convert_test.mm::test_convert_flushes_gpu_writes()` as part of Fix 1.1.

The test follows the exact procedure described here:
1. CPU fills `d_src[N]` with `1.0f`.
2. GPU encodes `add(5.0f, d_src, d_src, N)` — result 6.0, not yet committed.
3. `convert<float, float16_t>(d_src, d_dst, N)` is called with no intervening sync.
4. Asserts `d_dst[i] == 6.0f` (GPU-written value), not `1.0f` (stale CPU value).

The values differ slightly from the spec above (initial=1 + addend=5 instead of initial=0 + addend=1) but the mechanism and stale-vs-fresh distinction are identical. The test is named `"convert flushes GPU writes (f32→f16)"` and is part of the 8-test `convert_test` suite (8/8 pass).

---

### 4.2 `pso_warmup_test` ✅ COVERED (2026-02-25)

**Implemented** in `tests/metal/pso_warmup_test.mm` (12/12 pass).

One kernel from each of the six library groups is triggered via the public API and wrapped in a `no_exception()` helper. If any `newLibraryWithSource:` or `newComputePipelineStateWithFunction:` call fails, the test FAILS:

| Group | Trigger call | Types tested |
|-------|-------------|-------------|
| elementwise | `add(vec, vec, out, N)` | f32, f16 |
| activation | `relu(x, y, N)`, `gelu(x, y, N)` | f32 (relu), f32 (gelu/ct2_erf), bf16 |
| broadcast | `add_batch_broadcast(a, b, c, 2, 4)` | f32, f16 |
| beam_search | `penalize_previous_tokens(…)` | f32 |
| transpose | `transpose_2d(a, dims, b)` | f32, f16 |
| reduction | `sum(x, N)`, `max_element(x, N)` | f32 |

Original description:

Verify that all six MSL libraries compile cleanly at process startup.

Call each of the six `get_*_library()` functions (or any kernel lookup that triggers lazy compilation) and assert that no exception is thrown. This catches MSL syntax errors in any of the six embedded source strings that would otherwise surface as runtime crashes during the first transformer layer forward pass.

This test is particularly valuable because MSL source errors are not detected at C++ compile time. They surface as runtime exceptions from `newLibraryWithSource:`, and the error message from Metal is not always easy to correlate with the specific line in the embedded string.

---

### 4.3 `min_max_element_wise_test` ✅ COVERED (2026-02-25)

**Already present** (42 tests) and **extended** (30 new tests) in `tests/metal/minmax_test.mm`. Total: 72/72 pass.

New functions added to satisfy the full 4.3 spec:

**`run_scalar_clamp_subcases<T>`** (6 tests × 3 float types = 18 new tests):
- `all_below`: x[i]=1..16, scalar=20 — `min(20,x)=x`; `max(20,x)=20` (scalar always wins for max)
- `all_above`: x[i]=1..16, scalar=0  — `min(0,x)=0` (scalar always wins); `max(0,x)=x`
- `edge`: x[i]=i−8, scalar=−8 (== min element) — `min(−8,x)=−8` for all i; `max(−8,x)=x` for all i

**`run_vector_sign_subcases<T>`** (4 tests × 3 float types = 12 new tests):
- `all_positive`: a[i]=i+1, b[i]=i+2 (a < b everywhere) — `min(a,b)=a`; `max(a,b)=b`
- `all_negative`: a[i]=−(i+2), b[i]=−(i+1) (a < b < 0 everywhere) — `min(a,b)=a`; `max(a,b)=b`

**Bfloat ternary verification**: the bfloat16 sub-cases implicitly validate that the MSL kernel uses ternary comparison operators rather than `min()`/`max()` overloads. If the overload were used and absent from the SDK, the kernel would fail to compile or produce wrong values for the bfloat-typed inputs.

---

### 4.4 `large_transpose_test` ✅ COVERED (2026-02-25)

**Implemented** in `tests/metal/large_transpose_test.mm` (7/7 pass).

All cases compare GPU output against the CPU reference from `transpose_test.mm`. Shapes exercise `gid` values up to 1,048,575 — confirming the MSL `uint gid` arithmetic (`i0 = gid/b_s0`, `i1 = (gid/b_s1)%bd1`, `i2 = gid%b_s1`) is correct throughout:

| Case | Shape | Elements | Permutation(s) |
|------|-------|----------|----------------|
| 2D large | [1024, 512] | 524,288 | [1,0] |
| 3D attention | [32, 128, 256] | 1,048,576 | [2,0,1], [0,2,1], [1,0,2] |
| 4D MHA | [4, 32, 64, 128] | 1,048,576 | [0,2,1,3], [3,2,1,0] |
| 4D decode | [1, 16, 512, 128] | 1,048,576 | [0,2,1,3] |

Original description:

Test 2D, 3D, and 4D transpose for tensors with per-dimension sizes around 1024 to verify that the index decomposition arithmetic in the MSL kernels (`gid / b_s0`, `gid % bd1`, etc.) remains correct for large element counts.

Current tests likely cover small tensors used in development. A test with, for example, a 3D tensor of shape [32, 128, 256] (1,048,576 elements) and a non-identity permutation exercises the full range of `gid` values that would be used in a production multi-head attention transpose. This also incidentally validates that `static_cast<NSUInteger>(n)` in the dispatch helper handles six-digit element counts correctly.

---

### 4.5 `reduce_sum_precision_test` ✅ COVERED (2026-02-25)

**Implemented** in `tests/metal/reduce_sum_precision_test.mm` (6/6 pass).

PASS/FAIL checks only that the result is finite and no exception is thrown. Relative errors are informational baselines. Observed results on M4 (N=65536):

| Type | fill=1.0 (exact=65536) | fill=1/3 (exact≈21845) |
|------|----------------------|----------------------|
| float32 | 0.00e+00 (exact) | 7.15e-07 |
| float16 | overflow (inf) — expected, max=65504 | 7.57e-03 |
| bfloat16 | 0.00e+00 (exact) | **3.91e-02** |

Key findings:
- For `fill=1.0` the bfloat16 result is exact: all partial sums are powers of 2 (256.0 per group, 65536.0 total), exactly representable in BF16 — the precision problem doesn't manifest here.
- For `fill=1/3` the bfloat16 error is 3.91% versus 1.95e-03 for float32-over-BF16 reference (CPU accumulating in float32 over the same BF16 inputs). This confirms finding 3.4: the GPU accumulation in BF16 adds ~20× more error than the input quantisation noise alone.
- The reference (CPU float32 accumulation over BF16 inputs) gives 1.95e-03 — this is the floor achievable if the kernel switched to float threadgroup memory as `reduce_amax` does.

Original description:

Verify that `sum<bfloat16_t>` produces an acceptably accurate result for large arrays with a known analytic sum.

Suggested case: fill an array of N = 65536 bfloat16 values with 1.0. The exact sum is 65536.0. Measure the relative error of the GPU reduction result. Document the observed error as a baseline. This test does not assert a pass/fail criterion today (since the current implementation accumulates in bfloat), but it establishes a regression baseline and makes explicit the precision trade-off noted in finding 3.4.

---

## 5. Summary Table

| # | Category | Severity | Status | Item |
|---|----------|----------|--------|------|
| 1.1 | Bug | Medium | ✅ Fixed | `convert` missing `commit_and_wait()` flush before CPU read |
| 1.2 | Bug | High | ✅ Fixed | `min`/`max` element-wise were runtime throw stubs |
| 1.3 | Latent | Low | ✅ Fixed | `uint32_t` truncation of `dim_t` kernel args; silent for N > 2^32 |
| 2.1 | Quality | Medium | ✅ Fixed | PSO creation boilerplate repeated six times; extract `PSOCache` helper |
| 2.2 | Quality | Medium | ✅ Fixed | Library compile boilerplate repeated six times; extract `compile_library_once` |
| 2.3 | Quality | Medium | ✅ Fixed | MSL source duplicated in `.metal` files and embedded strings |
| 2.4 | Quality | Low | ✅ Fixed | `MTLLanguageVersion3_1` on activation library has no effect; comment was inaccurate |
| 2.5 | Quality | Low | ✅ Fixed | Error message format inconsistent across six kernel groups (side effect of 2.1) |
| 2.6 | Quality | Low | ✅ Fixed | CPU memcpy to temp buffer before GPU encoding in `dispatch_mps_gemm`; correctness non-obvious |
| 2.7 | Quality | Low | Deferred | `buffer_for_ptr` is O(N) linear scan; acceptable now, revisit if KV cache growth makes it measurable |
| 3.1 | Perf | Low | Deferred (M5+) | `alloc_temp_buffer` bypasses pool; allocation pressure on every reduction call |
| 3.2 | Perf | Medium | Deferred (M5+) | BF16 batch GEMM is fully sequential; each item pays ~0.4 ms CB overhead |
| 3.3 | Perf | Low | Deferred (M5+) | Small-matrix FP32/FP16 GEMM with `pad_c` commits command buffer unconditionally |
| 3.4 | Perf | Low | Baselined | `reduce_sum<bfloat16_t>` accumulates in bfloat precision; 3.91% rel-err documented (see 4.5) |
| 3.5 | Perf | Low | Deferred (M5+) | `convert` is CPU-only; GPU kernel would be faster for large type conversions |
| 4.1 | Test | — | ✅ Covered | `convert_gpu_flush_test` — `tests/metal/convert_test.mm` (8/8) |
| 4.2 | Test | — | ✅ Covered | `pso_warmup_test` — `tests/metal/pso_warmup_test.mm` (12/12) |
| 4.3 | Test | — | ✅ Covered | `min_max_element_wise_test` — `tests/metal/minmax_test.mm` (72/72) |
| 4.4 | Test | — | ✅ Covered | `large_transpose_test` — `tests/metal/large_transpose_test.mm` (7/7) |
| 4.5 | Test | — | ✅ Covered | `reduce_sum_precision_test` — `tests/metal/reduce_sum_precision_test.mm` (6/6) |

**Status as of 2026-02-25:** All correctness bugs (section 1) and all code quality items (section 2) are resolved. Section 3 performance items are deferred to M5+, with finding 3.4 baselined via `reduce_sum_precision_test`. All suggested tests (section 4) are implemented and passing.
