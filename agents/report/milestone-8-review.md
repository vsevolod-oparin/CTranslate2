# Milestone 8 Code Review — Transformer Layers + Conv1D

**Date:** 2026-03-02
**Reviewer:** Claude Opus 4.6
**Scope:** M8.1 (Encoder Layer), M8.2 (Decoder Layer), M8.3 (Conv1D)

---

## Executive Summary

M8.1 and M8.2 are pure integration tests — no new production code was written. M8.3 introduces new production code for Conv1D (im2col kernel + GEMM dispatch). All 29 tests pass (11+11+7).

**Critical finding:** A latent batch-stride bug in `sdpa_metal` (originating in M6.1/M6.2) is exposed by the M8.2 KV-cache decode pattern when `batch_size > 1`. This affects production inference with beam search.

| Severity | Count |
|----------|-------|
| Critical | 1 |
| Medium   | 2 |
| Low      | 5 |
| Test gap | 8 |

---

## CRITICAL

### C1. KV-cache batch stride mismatch in `sdpa_metal` (batch > 1)

**Location:** `src/metal/ops_sdpa.mm:510` / `src/ops/flash_attention_metal.mm:232`

**Bug:** `sdpa_metal` computes the per-batch offset for K/V as:
```cpp
const T* k_row0 = k + (b * seqlen_k * num_heads_k + hk) * head_dim;
```
This assumes K/V are laid out as `[batch, seqlen_k, nhk, hd]` with contiguous seqlen_k rows per batch.

But when called from `flash_attention_metal.mm` with a KV cache, the actual layout is `[batch, total_cache_slots, nhk, hd]` where `total_cache_slots > seqlen_k_eff`. The batch stride in the cache is `total_cache * nhk * hd`, but sdpa_metal uses `seqlen_k * nhk * hd`.

**Impact:** For `batch_size > 1` (beam search, batched inference), `sdpa_metal` reads the wrong memory for batches b >= 1 during KV-cache decode.

**Why not caught:** All M8.2 tests use `B=1`. The M10.1 beam_size>1 fix addressed gather, not this SDPA stride issue.

**Fix:** Add explicit `kv_batch_stride` parameter to `sdpa_metal`, or pass the `total_cache` size so the correct batch offset can be computed. The CUDA path avoids this because cuDNN/flash-attn kernels take explicit stride parameters.

**Workaround:** The current code works correctly when:
- `batch_size == 1` (most decode scenarios), OR
- `total_cache == seqlen_k_eff` (offset == 0 prefill path), OR
- The cache StorageView is exactly sized to `seqlen_k_eff` (never happens in practice — caches are pre-allocated larger)

---

## MEDIUM

### M1. Conv1D per-batch GEMM loop instead of batched GEMM

**Location:** `src/metal/ops_conv1d.mm:156-167`

**Issue:** Conv1D loops over batches and calls `primitives<METAL>::gemm` individually:
```cpp
for (dim_t b = 0; b < B; ++b) {
    primitives<Device::METAL>::template gemm<T, T>(
        false, false, false, true,
        C_out, T_out, CK, 1.0f,
        weight, CK,
        p + b * T_out * CK, CK,
        0.0f,
        output + b * C_out * T_out, T_out);
}
```

The CUDA implementation uses `gemm_batch_strided` which batches all GEMMs into a single GPU dispatch. Metal has `primitives<METAL>::gemm_batch_strided` available (defined in `primitives_gemm.mm:507`).

**Impact:** For batch > 1, each GEMM call adds command buffer encoding overhead. For B=4 with Whisper (C_out=128, K=3, T_out~3000), this is 4 separate GEMM dispatches instead of 1.

**Fix:** Replace the per-batch loop with:
```cpp
primitives<Device::METAL>::gemm_batch_strided<T, T>(
    false, true, C_out, T_out, CK, 1.0f,
    weight, CK, 0,                      // weight is shared (stride_a=0)
    p, CK, T_out * CK,                  // im2col stride per batch
    0.0f,
    output, T_out, C_out * T_out,       // output stride per batch
    B);
```

Note: verify `stride_a=0` (shared weight) is supported by the Metal `gemm_batch_strided` implementation.

### M2. `MetalTempBuf` duplicated in two files

**Location:** `src/metal/ops_conv1d.mm:38-54` and `src/metal/ops_sdpa.mm:40-60`

**Issue:** Identical RAII struct copy-pasted. Any bug fix or improvement must be applied in both places. The SDPA version has an additional `as<T>()` method that the Conv1D version lacks.

**Fix:** Extract to `src/metal/primitives_infra.h` or a new `src/metal/metal_temp_buf.h` header.

---

## LOW

### L1. Silent dilation clamping

**Location:** `src/ops/conv1d_metal.mm:39`

```cpp
const dim_t dil = (_dilation > 0) ? _dilation : 1;
```

The CUDA/CPU implementations pass `_dilation` directly without clamping. Silently converting 0 to 1 can mask upstream bugs. A `_dilation < 1` should throw an `invalid_argument` (matching the `groups != 1` validation on line 28).

### L2. Missing `@autoreleasepool` in m81_test.mm and m82_test.mm

**Location:** `tests/metal/m81_test.mm:563-575`, `tests/metal/m82_test.mm:840-850`

`m83_test.mm` correctly wraps all tests in `@autoreleasepool {}`, but the encoder and decoder tests don't. MPS objects created during tests may not be released promptly, leaking memory during the test run.

### L3. Test harness helper inconsistency

**Location:** All three test files

- m81/m82: `metal_from<T>()` — element-by-element copy via typed assignment, `metal_to_host()` — element-by-element read
- m83: `make_f32_buf()` — `memcpy`, `read_f32()` — `memcpy`

While both are correct, the inconsistency makes maintenance harder. Consider unifying into a shared test utility header (`tests/metal/metal_test_utils.h`).

### L4. No move semantics for `MetalTempBuf`

**Location:** `src/metal/ops_conv1d.mm:52-53`

The copy constructor and assignment are deleted, but move constructor/assignment are not defined. This prevents returning `MetalTempBuf` from functions or storing them in containers. Low priority since current usage is always stack-local.

### L5. Conv1D test tolerance comment imprecision

**Location:** `tests/metal/m83_test.mm:351-352`

```cpp
// float16 accumulates ~C_in*K=24 terms; tolerance 2e-2
```

This comment is accurate for test 6 (C_in=8, K=3 → CK=24), but the tolerance should be evaluated against the actual GEMM accumulation size which is `C_in * K` terms per output element. The tolerance of 2e-2 is generous — float16 with 24 accumulations should be accurate to ~1e-3. Consider tightening to catch potential issues.

---

## TEST GAPS

### T1. No batch > 1 test for KV-cache decode (Critical)

**Location:** `tests/metal/m82_test.mm`

All KV-cache decode tests use `B=1`. This masks the critical batch stride bug (C1).

**Needed:** `B=2` or `B=4` KV-cache decode test with `MAX_CACHE > seqlen_k_eff`.

### T2. No float16/bfloat16 tests for encoder/decoder layers

**Location:** `tests/metal/m81_test.mm`, `tests/metal/m82_test.mm`

Both test files only exercise `float32`. The plan (APPLE_M4_METAL_PLAN.md line 144) explicitly deferred these: "Float16 / BF16 encoder test — Extend m81_test with fp16/bf16 variants". These should be added before M10 end-to-end model tests, since production Whisper models use float16.

### T3. No multi-step decode sequence test

**Location:** `tests/metal/m82_test.mm`

Current test does a single decode step at `offset=4`. No test verifies growing the cache through a sequence of steps: `offset=0` (prefill), then `offset=4, 5, 6, ...`. This would catch:
- Cache position arithmetic errors at different offsets
- Stale data in unused cache slots being read
- Memory alignment issues at non-power-of-2 offsets

### T4. No Conv1D padding=0 test

**Location:** `tests/metal/m83_test.mm`

All tests use `padding >= 1`. Padding=0 is a valid configuration that changes the T_out calculation and eliminates all zero-padded positions in im2col.

### T5. No Conv1D K=1 test (point convolution)

Kernel size 1 is a degenerate case where im2col is essentially a no-op reshape. Worth testing to ensure the indexing doesn't break.

### T6. No Conv1D bias+activation test through the op wrapper

**Location:** `tests/metal/m83_test.mm`

Tests call `metal::conv1d_metal<T>()` directly, bypassing `Conv1D::compute<METAL>` which applies `apply_bias_and_activation()`. Bias + GELU activation is used by Whisper's Conv1D layers. This path is completely untested.

### T7. No stress test at realistic shapes

Test dimensions are tiny (dim=16, seq=4). Production Whisper uses dim=512/1024, seq=1500+. Numerical errors may compound differently at scale (e.g., float16 accumulation over C_in=512*K=3=1536 terms).

### T8. No test for `groups > 1` error path

**Location:** `src/ops/conv1d_metal.mm:28-31`

The `throw` for `groups != 1` is noted in the report as "validated by the throw in conv1d_metal.mm (not separately tested)". A simple test that verifies the expected exception would ensure this guard doesn't silently regress.

---

## Architecture Notes (Non-Issues)

### im2col + GEMM is the correct strategy

The plan originally suggested `MPSCNNConvolution`. The implementation chose im2col + GEMM instead, matching the CUDA `conv1d_gpu.cu` pattern. This is correct because:
1. `MPSCNNConvolution` is designed for 2D spatial convolution with image-centric layout (NHWC)
2. CTranslate2's Conv1D uses NCT layout which maps cleanly to im2col + GEMM
3. The GEMM path reuses the existing MPS/MPSGraph GEMM infrastructure

### Two-layer file separation is consistent

`ops_conv1d.mm` (raw pointers, no StorageView) + `conv1d_metal.mm` (StorageView wrapper) follows the established pattern from SDPA (M6.1), Rotary (M6.3), and ALiBi (M6.4).

### Test methodology is sound

Testing at the raw-pointer level (bypassing StorageView/model-loading) is the right approach for integration tests — it isolates Metal correctness from the dispatch machinery.

---

## Summary of Recommended Actions

| Priority | Action | Effort |
|----------|--------|--------|
| **P0** | Fix C1: add KV batch stride to sdpa_metal | Medium |
| **P0** | Add T1: batch>1 KV-cache decode test | Small |
| **P1** | Fix M1: use gemm_batch_strided in Conv1D | Small |
| **P1** | Add T6: Conv1D bias+activation test | Small |
| **P2** | Fix M2: extract MetalTempBuf to shared header | Small |
| **P2** | Add T2: fp16/bf16 encoder/decoder tests | Medium |
| **P2** | Add T3: multi-step decode sequence test | Medium |
| **P3** | Fix L1: throw on invalid dilation | Trivial |
| **P3** | Fix L2: add @autoreleasepool to m81/m82 tests | Trivial |
| **P3** | Add T4/T5: padding=0 and K=1 Conv1D tests | Small |
