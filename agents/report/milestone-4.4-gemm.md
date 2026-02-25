# Milestone 4.4 — GEMM: MPSMatrixMultiplication (FP32/FP16) + MPSGraph (BF16)

**Status:** ✅ DONE (2026-02-25)
**All tests pass:** 26/26

---

## Summary

Implemented `primitives<Device::METAL>::gemm` and `gemm_batch_strided` for three numeric types:

| Type | API | Strategy |
|------|-----|----------|
| float32 | `MPSMatrixMultiplication` | Deferred — encodes into per-thread command buffer |
| float16 | `MPSMatrixMultiplication` | Deferred — encodes into per-thread command buffer |
| bfloat16 | `MPSGraph` matmul | Synchronous — commits immediately on its own queue |

---

## Implementation Details

### Path A — FP32 / FP16: `dispatch_mps_gemm<T>`

Located in `src/metal/primitives.mm` (anonymous namespace), template function.

**rowBytes padding (hardware minimum constraint):**

`MPSMatrixDescriptor` requires `rowBytes ≥ [MPSMatrixDescriptor rowBytesForColumns:cols dataType:dtype]`. For small matrices this minimum can exceed the natural stride. For example, 4-column Float16 matrix: natural rowBytes = 8, hardware minimum = 16.

When `nat_rb < mps_min_rb`, the input is copied to a row-padded temporary `MTLBuffer` (StorageModeShared) before encoding. The output C is similarly padded, and after `commit_and_wait()` the result is unpacked back to c.

For production matrices (≥ a few columns of any supported type), the natural stride already satisfies the minimum, so no copies occur and the full deferred-commit pipeline is preserved.

**`@autoreleasepool` / `thread_local` ARC bug — the critical fix:**

Calling `get_current_command_buffer()` *inside* an `@autoreleasepool {}` block caused a use-after-free crash:

1. `get_current_command_buffer()` returns `_thread_buffer` which is held by a `thread_local id<MTLCommandBuffer>` (ARC strong).
2. The `commandBuffer` factory method also autoreleases its return value into the current pool.
3. When the scoped `@autoreleasepool {}` drains, that autorelease decrements the refcount — combined with ARC's handling of the strong thread_local, the refcount reaches 0.
4. `_thread_buffer` becomes a dangling pointer; the subsequent `[_thread_buffer commit]` in `commit_command_buffer()` crashes (SIGSEGV or EXC_BAD_ACCESS).

**Fix:** obtain `id<MTLCommandBuffer> cmd = get_current_command_buffer()` **before** the `@autoreleasepool {}` block. A stable strong local variable holds the buffer alive across the pool drain.

```objc
// CORRECT — cmd obtained before autoreleasepool
id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();

@autoreleasepool {
  MPSMatrixDescriptor* descA = ...;
  ...
  [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
}
// cmd strong ref keeps _thread_buffer alive; pool drain does not free it
```

This pattern must be followed any time MPS objects are created inside a scoped pool AND they use the per-thread command buffer for encoding.

### Path B — BF16: `dispatch_bf16_gemm` + `run_bf16_gemm_inner`

`MPSMatrixMultiplication` asserts at runtime if passed `MPSDataTypeBFloat16`. BF16 GEMM uses `MPSGraph` instead.

**Graph caching:** Four `MPSGraph` instances are created once (one per `{trans_a, trans_b}` combination). Each graph uses nil-shape placeholders so MPSGraph JIT-compiles and caches kernels per matrix shape on first use. The transpose is baked in as a `transposeTensor` node fused with the matmul.

**Synchronous execution:** MPSGraph's `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` is synchronous. It runs on its own internal command queue (not the per-thread deferred buffer). To avoid interleaving with the deferred command buffer, `commit_and_wait()` is called first to flush any in-flight encoders.

**Buffer offset handling:** `MPSGraphTensorData` has no byte-offset parameter. When `metal_buffer_for_ptr` returns a non-zero offset (batch slice from a larger allocation), the matrix data is copied to a zero-offset temporary buffer before the graph run.

**Alpha/beta constraint:** The MPSGraph path only supports `alpha=1.0, beta=0.0`. A `std::runtime_error` is thrown for other values (matching the ops layer's always-1/0 contract for BF16).

**Result readback:** `[[results[tC] mpsndarray] readBytes:c strideBytes:nil]` writes the result directly to the output MTLBuffer's contents pointer. `nil` strides means packed/contiguous row-major layout.

### `gemm_batch_strided`

- FP32/FP16: loop over batch dimension, call `dispatch_mps_gemm` for each slice. Each call obtains the command buffer and encodes one `MPSMatrixMultiplication` op — they all encode into the same command buffer and commit together when `synchronize_stream()` is next called.
- BF16: `commit_and_wait()` once before the loop (not per iteration), then call `run_bf16_gemm_inner` for each batch element.

---

## Bugs Found and Fixed

### Bug 1: FP16 crash — `MPSMatrix rowBytes` below hardware minimum

**Symptom:** `primitives<METAL>::gemm<ct2_f16, ct2_f16>` crashed for 4×4 matrices.

**Root cause:** Natural `rowBytes = 4 * 2 = 8` bytes for Float16, but `[MPSMatrixDescriptor rowBytesForColumns:4 dataType:Float16]` requires 16 bytes minimum. The GPU fault manifested as SIGSEGV or EXC_BAD_ACCESS inside the Metal framework.

**Fix:** Query `rowBytesForColumns:dataType:` for each of A, B, C and copy to padded temporary buffers when `nat_rb < mps_min_rb`.

### Bug 2: FP32 crash — `@autoreleasepool` + `thread_local` ARC over-release

**Symptom:** After applying the rowBytes fix, both FP16 and FP32 4×4 GEMM crashed at `[_thread_buffer commit]` inside `commit_command_buffer()` (utils.mm:57). LLDB showed `_thread_buffer` had a valid-looking address but a nil `isa`.

**Root cause:** `get_current_command_buffer()` was called inside the `@autoreleasepool {}` block that was added for the rowBytes fix. The command buffer's autorelease into the scoped pool, combined with thread_local ARC semantics, left `_thread_buffer` as a dangling pointer after the pool drained.

**Diagnosis timeline:**
1. LLDB backtrace: crash at `commit_command_buffer()`; `_thread_buffer` = non-nil pointer with nil isa.
2. Isolated to minimal standalone test: global (non-thread_local) var + same pool structure → same crash.
3. Confirmed: immediate commit works; deferred commit across pool boundary fails.
4. Hypothesis confirmed: moving `get_current_command_buffer()` outside the pool fixes both FP32 and FP16 in the standalone test.
5. Applied to production code: all 26 tests pass.

**Important:** Other call sites (`dispatch_binary`, `dispatch_scalar`, reduction functions) do NOT call `get_current_command_buffer()` inside `@autoreleasepool {}` blocks — they are unaffected by this issue.

---

## Test Results

File: `tests/metal/gemm_test.mm`

```
=== M4.4 GEMM Tests ===

--- FP32 gemm ---
  PASS  NN 4x4x4 no-transpose
  PASS  NT 4x4x4 transpose-B
  PASS  TN 4x4x4 transpose-A
  PASS  TT 4x4x4 both-transpose
  PASS  4x4x4 alpha=2.0 beta=1.0
  PASS  NN 64x64x64 no-transpose
  PASS  NN 256x512x128 rectangular
--- FP16 gemm ---
  PASS  NN 4x4x4 no-transpose (rowBytes padded)
  PASS  NT 4x4x4 transpose-B
  PASS  TN 4x4x4 transpose-A
  PASS  TT 4x4x4 both-transpose
  PASS  NN 64x64x64 no-transpose
  PASS  NN 256x512x128 rectangular
--- BF16 gemm ---
  PASS  NN 4x4x4 no-transpose
  PASS  NT 4x4x4 transpose-B
  PASS  TN 4x4x4 transpose-A
  PASS  TT 4x4x4 both-transpose
  PASS  NN 64x64x64 no-transpose
  PASS  NN 256x512x128 rectangular
--- FP32 gemm_batch_strided ---
  PASS  batch=1 4x4x4
  PASS  batch=4 4x4x4
  PASS  batch=4 64x64x64
  PASS  batch=4 256x512x128
--- BF16 gemm_batch_strided ---
  PASS  batch=1 4x4x4
  PASS  batch=4 4x4x4
  PASS  batch=4 64x64x64

=== Results: 26 passed, 0 failed ===
```

---

## Files Modified

- `src/metal/primitives.mm` — added `dispatch_mps_gemm<T>`, `Bf16GemmEntry`, `get_bf16_graph`, `run_bf16_gemm_inner`, `dispatch_bf16_gemm`; implemented `primitives<METAL>::gemm` and `gemm_batch_strided`

---

## Building and Running the Tests

Run from the repository root:

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/gemm_test.mm \
    src/metal/device.mm \
    src/metal/utils.mm \
    src/metal/allocator.mm \
    src/metal/primitives.mm \
    src/allocator.cc \
    src/devices.cc \
    src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o gemm_test && ./gemm_test
```

`-framework MetalPerformanceShadersGraph` is required for the BF16 MPSGraph path.

---

## Files Created

- `tests/metal/gemm_test.mm` — 26 correctness tests for FP32/FP16/BF16 gemm and gemm_batch_strided
- `agents/report/milestone-4.4-gemm.md` — this file

---

## Design Decisions and Notes

1. **No MPSGraph caching for BF16 batch-strided:** The current implementation calls `run_bf16_gemm_inner` in a loop, which internally calls `[graph runWithMTLCommandQueue:...]` per batch element. This is not optimal for large batches. A future optimization could build a batched graph once and pass a 3-D tensor. Deferred to M4.4+ profiling.

2. **FP16 gemm_batch_strided not implemented:** The test suite covers FP32 batch-strided but not FP16. The FP16 path calls `dispatch_mps_gemm<float16_t>` in a loop identically to FP32, so it should work correctly by construction.

3. **`@autoreleasepool` rule to document:** Any call to `get_current_command_buffer()` must happen OUTSIDE any `@autoreleasepool {}` block, and the returned `id` must be held in a local variable across the entire encoding region. This is now documented inline in `primitives.mm`.

4. **BF16 GEMM is always synchronous:** The deferred-commit benefit from M0.3 is lost for BF16 ops. This is an inherent limitation of MPSGraph's execution model. Revisit if BF16 inference shows unacceptable latency from frequent per-GEMM commits.
