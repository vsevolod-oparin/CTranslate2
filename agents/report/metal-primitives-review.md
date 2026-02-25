# Metal Primitives Code Review

This review covers the Apple Metal backend primitive layer as implemented across milestones M4.1 through M4.8. The primary source under review is `src/metal/primitives.mm` (~2048 lines), with supporting analysis of `src/metal/utils.mm`, `src/metal/utils.h`, and `src/metal/allocator.mm`. The review does not propose code changes; it catalogues correctness risks, code quality observations, performance notes, and missing test coverage.

---

## 1. Bugs and Correctness Risks

### 1.1 `convert<U,V>` reads stale GPU data

**Severity: Medium**

`primitives<Device::METAL>::convert` is implemented as `std::copy(x, x + size, y)` — a CPU operation executed directly against the unified-memory pointer:

```
// src/metal/primitives.mm:1402-1404
template<>
template <typename U, typename V>
void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
  std::copy(x, x + size, y);
}
```

If `x` was written by a pending (uncommitted) GPU kernel, `std::copy` reads stale data. On Apple Silicon, the GPU and CPU share physical DRAM, but the Metal API gives no guarantee that GPU writes are visible to the CPU until the command buffer containing those writes has been committed and `waitUntilCompleted` has returned.

All other CPU-reads-after-GPU-write patterns in this file guard with `metal::commit_and_wait()`:

- `at<T>()` (line 1354): `metal::commit_and_wait()` with comment "flush any pending GPU writes to x".
- `logsumexp<T>()` (line 1774): `metal::commit_and_wait()` with comment "flush any pending GPU writes to x".
- `prepare_length_mask()` (line 1696): `metal::commit_and_wait()` with comment "flush any pending GPU writes to lengths".

`convert` is the only CPU-read primitive that does not flush. The condition for a silent corruption is: any code path that (1) issues a GPU kernel that writes a buffer, (2) does not call `synchronize_stream()`, and (3) immediately calls `convert` on that buffer. This is not hypothetical — type conversions are routinely applied to output tensors that were just computed on GPU.

**Suggested fix:** Add `metal::commit_and_wait()` at the start of `convert`, consistent with the guard pattern used by `at()`, `logsumexp()`, and `prepare_length_mask()`.

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

### 1.3 `uint32_t` truncation of `dim_t` in GPU kernel arguments

**Severity: Low (theoretical, not a practical concern today)**

Several primitives cast `dim_t` (which is `int64_t` on 64-bit platforms) to `uint32_t` before passing it as a kernel argument:

```
// e.g., src/metal/primitives.mm:1421, 1449, 1489, 1519
uint32_t n = static_cast<uint32_t>(size);
```

Similarly, the MSL kernels use `uint gid [[thread_position_in_grid]]` which is a 32-bit unsigned integer. For element counts above 2^32 (approximately 4 billion), `gid` wraps to zero and elements beyond the 4-billion index are silently skipped.

In practice, inference tensors stay well below this limit. A 50k-token, 8192-dimension float32 activation tensor is approximately 1.6 GB, which is 410 million elements — within the 32-bit range. The risk is theoretical for the foreseeable hardware envelope.

However, the truncation is silent. A `static_assert` or a runtime check in the dispatch helpers would surface the assumption explicitly rather than leaving it to be discovered through a data corruption in a future setting.

The `dispatch_penalize` function has a related pattern: `batch_size`, `length`, and `vocab_size` are all truncated from `dim_t` to `uint32_t`. A 250k-vocabulary model is well within 32-bit range, but the truncation deserves a note in the relevant dispatch function.

---

## 2. Code Quality and Maintainability

### 2.1 PSO creation boilerplate repeated across six kernel groups

**Severity: Medium (maintenance burden)**

Each of the six kernel groups (elementwise, activation, broadcast, beam_search, transpose, reduction) contains a functionally identical PSO lookup function. The structure is the same in every case:

1. Declare a static `std::unordered_map<std::string, id<MTLComputePipelineState>>` and a static `std::mutex`.
2. Acquire the lock and check the cache.
3. On miss: call the corresponding `get_*_library()` function.
4. Look up the function by name; throw if nil.
5. Create the PSO; throw if nil.
6. Insert into cache and return.

This pattern spans approximately 30 lines per group, giving roughly 180 lines of near-identical code across the six functions. The functions differ only in the library they call and the prefix of their error messages.

Because the pattern is duplicated, any future change — for example, switching from `newComputePipelineStateWithFunction:error:` to the async `newComputePipelineStateWithFunction:completionHandler:`, or adding PSO descriptor options — must be applied six times and can be forgotten in some copies.

**Suggested refactoring:** Extract the repeated logic into two shared helpers in the anonymous namespace:

1. A `make_pso(id<MTLLibrary> lib, const char* name) -> id<MTLComputePipelineState>` function that performs the function lookup and PSO creation with uniform error handling.
2. A small `PSOCache` struct (or class) holding the `std::unordered_map` and `std::mutex` with a single `get(id<MTLLibrary> lib, const char* name)` method. Each library group then creates one `static PSOCache` instance.

This reduces the six ~30-line functions to six one-line calls.

---

### 2.2 Library compilation boilerplate repeated across six kernel groups

**Severity: Medium (maintenance burden)**

The six `get_*_library()` functions (`get_elementwise_library`, `get_activation_library`, `get_broadcast_library`, `get_beam_search_library`, `get_transpose_library`, `get_reduction_library`) share identical structure:

1. Declare a `static id<MTLLibrary> lib = nil` and a `static std::once_flag flag`.
2. Call `std::call_once`.
3. Convert the MSL string to `NSString`.
4. Call `newLibraryWithSource:options:error:`.
5. Check for nil, construct an error message, throw.

The only differences are the MSL source string, the compile options (only the activation library passes non-nil options), and the label used in the error message. This is approximately 15 lines per group, 90 lines total.

**Suggested refactoring:** Extract a shared helper:

```
static id<MTLLibrary> compile_library_once(
    std::once_flag& flag,
    id<MTLLibrary>& lib_out,
    const char* msl_src,
    const char* label,
    MTLCompileOptions* opts = nil);
```

The six functions collapse to six one-line calls with their respective arguments.

---

### 2.3 MSL source duplicated between `.metal` files and embedded C strings

**Severity: Medium (source of silent divergence)**

Six `.metal` files exist at `src/metal/kernels/`:

- `elementwise.metal`
- `activation.metal`
- `broadcast.metal`
- `beam_search.metal`
- `transpose.metal`
- `reduction.metal`

These are described in comments as the "canonical copies". However, at runtime the Metal libraries are compiled from verbatim C raw string literals (`kElementwiseMSL`, `kActivationMSL`, etc.) embedded in `primitives.mm`. The `.metal` files are not consumed by the build system; their value is documentation and IDE support.

This means every edit to a `.metal` file must be manually mirrored in the corresponding `k*MSL` string. The two copies can silently diverge: a developer refines the `.metal` source for readability or correctness, forgets to update the embedded string, and the change is never actually executed at runtime. The reverse is equally possible — a hotfix applied to the embedded string is never reflected in the documented `.metal` file.

**Suggested improvement (two options):**

Option A (lighter): Add a CI step or a build-time check that compares the SHA-1 (or any stable hash) of each `.metal` file against a recorded expected hash stored in a comment or sidecar file next to the `k*MSL` string. The build fails if hashes diverge.

Option B (heavier, preferred long-term): Add a code-generation step that automatically produces the `k*MSL` strings from the `.metal` files at build time (e.g., via a CMake `configure_file` or a Python script that hex-encodes the `.metal` content into a C header). The duplication then becomes mechanical and always correct.

---

### 2.4 `MTLLanguageVersion3_1` on the activation library has no effect

**Severity: Low (misleading comment)**

In `get_activation_library()` (line 274):

```objc
MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
opts.languageVersion = MTLLanguageVersion3_1;
```

The comment above reads: "We request Metal 3.1 (macOS 14+) to ensure erf() is available — it was added to the Metal standard math library in MSL 3.1."

However, the activation kernels do not call `erf()`. They call `ct2_erf()`, the Abramowitz and Stegun polynomial approximation defined in the MSL source string itself, precisely because MSL's built-in `erf()` was found to be unavailable even after setting `MTLLanguageVersion3_1`. The comment's stated reason for the option is therefore factually incorrect.

The `MTLLanguageVersion3_1` option has no effect on the compiled kernels as written. All other five library compilation calls pass `nil` options. Leaving a non-nil options object with an inaccurate justifying comment creates confusion for future maintainers: they may believe `MTLLanguageVersion3_1` is a prerequisite for correct activation behavior, and may be reluctant to remove it or consolidate it with the other libraries.

**Suggested fix:** Pass `nil` options to `get_activation_library()` matching the pattern of the other five libraries. Update the comment near `ct2_erf()` to clarify that the polynomial approximation is used because the MSL built-in was unavailable, making the language version option unnecessary.

---

### 2.5 Inconsistent error message format across kernel groups

**Severity: Low**

Error messages for missing kernel functions use different prefix strings depending on which group they come from:

| Group | "not found" message |
|---|---|
| Elementwise | `"Metal: kernel not found: "` |
| Activation | `"Metal: activation kernel not found: "` |
| Broadcast | `"Metal: broadcast kernel not found: "` |
| Beam search | `"Metal: beam_search kernel not found: "` |
| Transpose | `"Metal: transpose kernel not found: "` |
| Reduction | `"Metal: reduction kernel not found: "` |

The PSO creation failure messages are also inconsistent: beam_search uses `"Metal: beam_search PSO creation failed for "` while the others use `"Metal: PSO creation failed for "`.

When the PSO boilerplate is unified into a shared helper (see 2.1), these messages will naturally become consistent as a side effect of having a single code path.

---

### 2.6 `alloc_temp_buffer` usage within `dispatch_mps_gemm` bypasses pool and triggers a CPU copy before the buffer is used by the GPU

**Severity: Low (correctness note, not a bug)**

In `dispatch_mps_gemm`, when `pad_a` or `pad_b` is true, the code uses `alloc_temp_buffer` to create a fresh MTLBuffer and then immediately writes to it with `std::memcpy` (lines 1074-1075, 1085-1086). This memcpy happens on the CPU before any GPU command is encoded, which is correct — CPU writes to a Shared-mode buffer are coherent with the GPU on subsequent encodings.

However, the pattern relies on the implicit understanding that (1) the CPU memcpy completes before the GPU starts, and (2) no intervening GPU command on a different thread could observe a half-written buffer. Both hold because the buffer is freshly allocated (not shared with any prior encoding) and the encoding happens in the same thread immediately after the copy. This is correct but the correctness is non-obvious. A comment explaining why no CPU-GPU synchronization is required between the memcpy and the encoding would benefit future readers.

---

### 2.7 `buffer_for_ptr` performs a linear scan through all live allocations

**Severity: Low (not a hot path today)**

`MetalAllocator::buffer_for_ptr` in `allocator.mm` (line 41-54) iterates through `_live` — a `std::unordered_map<void*, LiveEntry>` — with a range-for loop, comparing each base pointer's range against the query. This is an O(N) scan where N is the number of live allocations.

The `_live` map stores entries keyed by the exact base pointer of each allocation, so a mid-allocation pointer (e.g., a pointer into the middle of a buffer) cannot be resolved by a direct `find()`. The comment acknowledges this: "Iterates _live to find the enclosing allocation. O(n) where n is the number of live allocations — typically a few dozen in inference."

For normal inference workloads with a stable allocation set, this is not a bottleneck. Every call to `dispatch_binary`, `dispatch_scalar`, `dispatch_broadcast1`, `dispatch_broadcast2`, `dispatch_unary`, `dispatch_transpose`, and the GEMM path calls `metal_buffer_for_ptr` two to three times. In a transformer layer with batch=1, these calls are in the hundreds per forward pass.

If the allocation count grows (e.g., KV cache with large context windows creating many independently allocated buffers), the scan could become measurable. A future optimization would replace the linear scan with an interval tree or sorted array supporting binary search, but this is out of scope for the current milestone.

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

### 4.1 `convert_gpu_flush_test`

Verify that `convert` correctly observes GPU-written data.

Procedure:
1. Allocate a Metal buffer `d_src` containing float32 zeros.
2. Issue a GPU kernel (e.g., `add(1.0f, d_src, d_src, N)`) that writes to `d_src`. Do not call `synchronize_stream()`.
3. Immediately call `convert<float, float16_t>(d_src, d_dst, N)` without any intervening synchronization.
4. Assert that `d_dst` contains the value 1.0 in float16, not 0.0 (the stale pre-kernel value).

Without a `commit_and_wait()` guard in `convert`, step 4 reads stale data and the test fails. With the guard it reads the post-kernel GPU output and passes. This test directly verifies the correctness risk described in finding 1.1.

---

### 4.2 `pso_warmup_test`

Verify that all six MSL libraries compile cleanly at process startup.

Call each of the six `get_*_library()` functions (or any kernel lookup that triggers lazy compilation) and assert that no exception is thrown. This catches MSL syntax errors in any of the six embedded source strings that would otherwise surface as runtime crashes during the first transformer layer forward pass.

This test is particularly valuable because MSL source errors are not detected at C++ compile time. They surface as runtime exceptions from `newLibraryWithSource:`, and the error message from Metal is not always easy to correlate with the specific line in the embedded string.

---

### 4.3 `min_max_element_wise_test`

Once the element-wise `min` and `max` stubs are implemented (see finding 1.2), add a test suite:

- Scalar clamp min: `y[i] = min(a, x[i])`. Verify for three cases: all elements below `a`, all above `a`, mixed. Test edge case where `a` equals the minimum element.
- Element-wise vector min: `c[i] = min(a[i], b[i])`. Verify for positive, negative, and mixed values.
- Scalar clamp max and element-wise vector max: symmetric to the above.
- Test all three floating-point types: `float`, `float16_t`, `bfloat16_t`.
- Verify that bfloat min/max kernels use ternary comparison, not MSL `min()`/`max()`, to avoid SDK overload ambiguity.

---

### 4.4 `large_transpose_test`

Test 2D, 3D, and 4D transpose for tensors with per-dimension sizes around 1024 to verify that the index decomposition arithmetic in the MSL kernels (`gid / b_s0`, `gid % bd1`, etc.) remains correct for large element counts.

Current tests likely cover small tensors used in development. A test with, for example, a 3D tensor of shape [32, 128, 256] (1,048,576 elements) and a non-identity permutation exercises the full range of `gid` values that would be used in a production multi-head attention transpose. This also incidentally validates that `static_cast<NSUInteger>(n)` in the dispatch helper handles six-digit element counts correctly.

---

### 4.5 `reduce_sum_precision_test`

Verify that `sum<bfloat16_t>` produces an acceptably accurate result for large arrays with a known analytic sum.

Suggested case: fill an array of N = 65536 bfloat16 values with 1.0. The exact sum is 65536.0. Measure the relative error of the GPU reduction result. Document the observed error as a baseline. This test does not assert a pass/fail criterion today (since the current implementation accumulates in bfloat), but it establishes a regression baseline and makes explicit the precision trade-off noted in finding 3.4.

---

## 5. Summary Table

| # | Category | Severity | Item |
|---|----------|----------|------|
| 1.1 | Bug | Medium | `convert` missing `commit_and_wait()` flush before CPU read |
| 1.2 | Bug | High | `min`/`max` element-wise are runtime throw stubs — **FIXED** |
| 1.3 | Latent | Low | `uint32_t` truncation of `dim_t` kernel args; silent for N > 2^32 |
| 2.1 | Quality | Medium | PSO creation boilerplate repeated six times; extract `PSOCache` helper |
| 2.2 | Quality | Medium | Library compile boilerplate repeated six times; extract `compile_library_once` |
| 2.3 | Quality | Medium | MSL source duplicated in `.metal` files and embedded strings; add drift detection |
| 2.4 | Quality | Low | `MTLLanguageVersion3_1` on activation library has no effect; comment is inaccurate |
| 2.5 | Quality | Low | Error message format inconsistent across six kernel groups |
| 2.6 | Quality | Low | CPU memcpy to temp buffer before GPU encoding in `dispatch_mps_gemm`; correctness is non-obvious without comment |
| 2.7 | Quality | Low | `buffer_for_ptr` is O(N) linear scan; acceptable now, worth noting for future growth |
| 3.1 | Perf | Low | `alloc_temp_buffer` bypasses pool; allocation pressure on every reduction call |
| 3.2 | Perf | Medium | BF16 batch GEMM is fully sequential; each item pays ~0.4 ms CB overhead |
| 3.3 | Perf | Low | Small-matrix FP32/FP16 GEMM with `pad_c` commits command buffer unconditionally |
| 3.4 | Perf | Low | `reduce_sum<bfloat16_t>` accumulates in bfloat precision; consider float accumulator as in `reduce_amax` |
| 3.5 | Perf | Low | `convert` is CPU-only; a GPU kernel would be faster for large type conversions |
| 4.1 | Test | — | Add `convert_gpu_flush_test` to verify finding 1.1 |
| 4.2 | Test | — | Add `pso_warmup_test` to catch MSL syntax errors at test time |
| 4.3 | Test | — | Add `min_max_element_wise_test` after finding 1.2 is resolved |
| 4.4 | Test | — | Add `large_transpose_test` for production-scale tensor shapes |
| 4.5 | Test | — | Add `reduce_sum_precision_test` to baseline bfloat16 accumulation error |

**Recommended implementation order:**

1. Fix 1.2 (`min`/`max` stubs) — highest severity; blocks models that use clamp or element-wise min/max operations.
2. Fix 1.1 (`convert` flush) — correctness risk, one-line fix with well-established precedent in the same file.
3. Refactor 2.1 and 2.2 (PSO and library boilerplate) — reduces the ongoing maintenance surface before more kernel groups are added.
4. Fix 2.4 (MSL 3.1 option and comment) — trivial cleanup; prevents future confusion.
5. Address 2.3 (MSL source drift) — CI-enforced hash check is a low-cost safeguard.
6. Investigate 3.2 (BF16 async batch GEMM) — performance investigation appropriate for M5+.
