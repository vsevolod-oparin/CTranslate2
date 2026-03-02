# Milestone 9 Code Review — INT8 Quantization + GEMM

**Date:** 2026-03-02
**Reviewer:** Claude Opus 4.6
**Scope:** M9.1 (INT8 Quantize/Dequantize), M9.2 (INT8 GEMM), M9.3 (gemm_pack_b), M9.4 (compute_u8_compensation)

---

## Executive Summary

M9.1 introduces three GPU kernel families (quantize, dequantize, dequantize_gemm_output) plus their host-side dispatch and StorageView wrappers. M9.2 adds INT8 GEMM via float32 intermediate (CPU conversion + MPS GEMM). M9.3 and M9.4 are trivial stubs (gemm_pack_b returns 0, compute_u8_compensation is a no-op). All 24 tests pass (15+9).

No critical bugs found. The primary concern is performance: batch_strided INT8 GEMM issues 2 `commit_and_wait` calls per batch element, creating a synchronization bottleneck.

| Severity | Count |
|----------|-------|
| Critical | 0 |
| Medium   | 2 |
| Low      | 4 |
| Test gap | 7 |

---

## MEDIUM

### M1. INT8 `gemm_batch_strided` issues 2*B synchronization points

**Location:** `src/metal/primitives_gemm.mm:538-545`

**Issue:** The INT8 path in `gemm_batch_strided` loops over batch elements, calling `dispatch_int8_gemm` for each. Every `dispatch_int8_gemm` call executes two `commit_and_wait` calls (line 406 and 454): one to flush before CPU reads int8 inputs, one to wait for the GPU GEMM result.

```cpp
// gemm_batch_strided INT8 path:
for (dim_t i = 0; i < batch_size; ++i)
    dispatch_int8_gemm(..., a + i * stridea, ..., b + i * strideb, ..., c + i * stridec, ...);
```

For batch_size=B, this is **2*B** GPU synchronization points, each adding ~0.4ms command buffer overhead on M4.

**Impact:** For beam search with B=4, this is 8 synchronization points (~3.2ms overhead) per INT8 GEMM layer, compared to the ideal 2 (one flush before all CPU conversions, one wait after all GPU GEMMs).

**Fix:** Restructure into three phases:
1. One `commit_and_wait` to flush pending GPU work
2. CPU: convert all B batches of int8 A and B to float32 temp buffers
3. Encode all B MPS GEMMs (or use `MPSMatrixMultiplication` batch API)
4. One `commit_and_wait` to wait for all GPU GEMMs
5. CPU: round all B float32 results to int32

This reduces synchronization from 2*B to 2, regardless of batch size. Note: the float32/float16 paths already benefit from encode-only (no sync per batch element), and the BF16 path wisely does `commit_and_wait` once before the batch loop (line 532).

### M2. CPU int8-to-float32 conversion is unvectorized

**Location:** `src/metal/primitives_gemm.mm:430-443`

**Issue:** The int8 → float32 conversion uses a scalar nested loop:
```cpp
for (NSUInteger r = 0; r < rows_a; ++r) {
    for (NSUInteger ci = 0; ci < cols_a; ++ci)
        dst[ci] = static_cast<float>(src[ci]);
}
```

For a typical INT8 GEMM with m=512, k=512, this converts 262K elements with scalar casts.

**Impact:** On Apple M4 with NEON, `vld1q_s8` + `vcvtq_f32_s32` could process 16 int8 values per iteration. Alternatively, `vDSP_vflt8` from Accelerate.framework converts int8→float vectorized. The current scalar loop may be masked by the two `commit_and_wait` calls, but if M1 is fixed (reducing to 2 syncs), this conversion becomes the bottleneck.

**Fix:** Use `vDSP_vflt8` for each row, or write a simple NEON loop. Low urgency until M1 is addressed.

---

## LOW

### L1. Missing `@autoreleasepool` in m92_test.mm

**Location:** `tests/metal/m92_test.mm:523-538`

The `main()` function does not wrap test invocations in `@autoreleasepool`:
```cpp
int main() {
    std::printf("=== M9.2 INT8 GEMM on Metal ===\n\n");
    test_int8_gemm_basic();
    // ... no @autoreleasepool
}
```

Compare with `m91_test.mm:554` which correctly uses `@autoreleasepool { ... }`. MPS objects created during tests may not be released promptly.

### L2. `dispatch_mps_gemm_buf` hardcodes beta=0.0

**Location:** `src/metal/primitives_gemm.mm:377`

```cpp
MPSMatrixMultiplication* gemm_op =
    [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                         ...
                                              beta:0.0];
```

The `beta` parameter from `dispatch_int8_gemm` is validated but not passed through to `dispatch_mps_gemm_buf`. Currently safe because INT8 GEMM always uses beta=0, but the early validation at line 393-394 creates a false sense of generality.

### L3. Build command in m91_test.mm references non-existent file

**Location:** `tests/metal/m91_test.mm:28`

```
src/metal/primitives_norm_gather.mm
```

The actual file is `src/metal/ops_norm_gather.mm` (matching the standard naming convention). The build command in the file header would fail if copy-pasted. The m92_test.mm correctly uses `ops_norm_gather.mm`.

### L4. Test 7 (m91) uses b_scales sized to B (not D) without comment

**Location:** `tests/metal/m91_test.mm:414`

```cpp
std::vector<float> host_as(B), host_bs(B), host_bias(D);
```

With `trans_a=false, trans_b=false`, the kernel indexes both a_scales and b_scales by `i` (the batch/row index). Having b_scales sized to B (=3) is correct for this configuration, but differs from test 6 which uses `host_bs(D)` with the same transpose settings. The inconsistency makes it unclear whether the test is exercising a realistic configuration.

In production, CTranslate2 always uses `trans_b=true` for INT8 GEMM (weights stored transposed), where b_scales is indexed by `j` (column) and has `n` entries. Neither m91 test exercises this common configuration for `dequantize_gemm_output`.

---

## TEST GAPS

### T1. No `trans_a=true` test for INT8 GEMM

**Location:** `tests/metal/m92_test.mm`

Tests cover `trans_a=false` + `trans_b=false` and `trans_a=false` + `trans_b=true`. No test exercises `trans_a=true`. While uncommon in production (activation matrices are not transposed), the `dispatch_int8_gemm` implementation handles it (lines 400-401), and it should be verified.

### T2. No `k > 1040` precision boundary test

**Location:** `tests/metal/m92_test.mm`

The largest k tested is 512 (test 4). The report documents that float32 exact integer accumulation works for k <= 1040 (127*127*k < 2^24). No test exercises k near this boundary (e.g., k=1024 with worst-case +-127 values) or beyond it (k=2048 with large values) to verify that precision loss is graceful rather than catastrophic.

### T3. Only ReLU activation tested in dequantize_gemm_output

**Location:** `tests/metal/m91_test.mm`

Test 8 exercises ReLU (activation_type=0) only. The kernel implements 7 activation functions:
- ReLU (tested)
- GELUTanh (untested) — used by many transformer models
- Swish (untested) — used by some models
- GELU (untested)
- GELUSigmoid (untested)
- Tanh (untested)
- Sigmoid (untested)

The GELU-tanh implementation (line 165-166 of quantize.metal) and the `ct2_erf` polynomial (line 21-27) are non-trivial and should be verified against CPU reference values. At minimum, GELUTanh (used in Whisper's feed-forward layers with INT8) and GELU should be tested.

### T4. No fp16/bf16 tests for dequantize_gemm_output

**Location:** `tests/metal/m91_test.mm`

All three `dequantize_gemm_output` tests (6, 7, 8) use float32 only. The kernel is instantiated for half and bfloat (quantize.metal lines 181-184), and the dispatch has explicit instantiations for all three types (ops_quantize.mm lines 268-276), but neither half nor bfloat are exercised.

Float16 precision may interact with the bias addition and activation computation in the kernel (all done in float32 arithmetic, then cast to T at the end — line 178), but the cast-to-T step should be verified for correctness.

### T5. No test through StorageView wrapper path

**Location:** `src/ops/quantize_metal.mm`, `src/ops/dequantize_metal.mm`

All tests call the `metal::*` free functions directly, bypassing the StorageView wrappers. The wrappers contain logic that should be verified:
- `quantize_metal.mm`: validates `_shift_to_uint8` is not supported (throws)
- `dequantize_metal.mm`: maps `ActivationType*` to int, handles null bias by passing `c` buffer as dummy pointer

These wrappers will only be exercised by M10 end-to-end model tests, but a simple unit test catching the `_shift_to_uint8` exception would ensure the guard doesn't regress.

### T6. No test for `dequantize_gemm_output` with `trans_b=true`

**Location:** `tests/metal/m91_test.mm`

Tests 6-8 all use `trans_a=false, trans_b=false`. The production configuration is `trans_a=false, trans_b=true` (INT8 weights are stored transposed). Test 6 in m92_test.mm exercises trans_b=true through the full pipeline, but the standalone dequantize_gemm_output tests don't isolate this path.

The scale indexing is the critical difference: `b_scales[j]` (column index) with trans_b=true vs `b_scales[i]` (row index) with trans_b=false. A dedicated test with mismatched B/D dimensions would catch any indexing errors.

### T7. No test for beta != 0 rejection in INT8 GEMM

**Location:** `src/metal/primitives_gemm.mm:393-394`

```cpp
if (beta != 0.0f)
    throw std::runtime_error("Metal INT8 GEMM: only beta=0 is supported");
```

This guard is documented but not tested. A simple test that verifies the expected exception would prevent silent regression.

---

## Architecture Notes (Non-Issues)

### Float32 intermediate strategy is correct

The int8→float32→MPS GEMM→float32→int32 approach is the only viable strategy on current Metal (no native INT8 MPS matmul). The float32 accumulation is mathematically exact for k <= 1040, covering typical CTranslate2 inference dimensions. The two `commit_and_wait` calls per single GEMM are unavoidable for the CPU-conversion approach, though batch amortization (see M1) can reduce the overhead.

### `dispatch_mps_gemm_buf` reuse pattern is sound

Creating a new low-level helper that takes `id<MTLBuffer>` directly (bypassing `metal_buffer_for_ptr`) is the correct approach. The temp buffers from `alloc_temp_buffer` are not in the MetalAllocator live map, so `dispatch_mps_gemm<T>` would throw. The helper shares MPS matrix setup logic without code duplication.

### Quantize kernel tree reduction is correct

The 256-thread tree reduction for per-row abs-max follows standard GPU reduction patterns: strided accumulation, threadgroup_barrier, power-of-2 halving. The barrier after scale broadcast (line 74) ensures all threads see the computed scale before the quantization step.

### Activation encoding is correct

The mapping from `ActivationType` enum to kernel encoding (+1 offset, with 0=none) is consistent between host code (`ops_quantize.mm:243-245`) and kernel (`quantize.metal:162-177`). All 7 activation implementations match their mathematical definitions (verified GELU-tanh formula, ct2_erf polynomial, swish=x*sigmoid(x), etc.).

---

## Summary of Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| **P1** | Fix M1: batch INT8 GEMM to use 2 syncs instead of 2*B | Medium |
| **P1** | Add T3: test all 7 activation types in dequantize_gemm_output | Small |
| **P2** | Add T1: trans_a=true INT8 GEMM test | Small |
| **P2** | Add T6: dequantize_gemm_output with trans_b=true | Small |
| **P2** | Add T4: fp16/bf16 dequantize_gemm_output tests | Small |
| **P2** | Fix M2: vectorize int8→float32 CPU conversion | Small |
| **P3** | Fix L1: add @autoreleasepool to m92_test.mm | Trivial |
| **P3** | Fix L3: correct build command in m91_test.mm header | Trivial |
| **P3** | Add T2: k > 1040 boundary test | Small |
| **P3** | Add T5: StorageView wrapper path test | Small |
| **P3** | Add T7: beta != 0 rejection test | Trivial |
