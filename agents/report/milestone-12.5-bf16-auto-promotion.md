# M12.5 — BF16→FP16 Auto-Promotion on MPS

**Date**: 2026-03-11
**Status**: Complete — 181× speedup for bf16, 64× for int8_bf16
**Model**: OPUS-MT En→De (d_model=512, 6 layers)
**Hardware**: Apple M4, macOS 15

---

## 1. Problem

BF16 GEMM on MPS uses `MPSGraph runWithMTLCommandQueue:`, which is inherently synchronous (~11ms per GEMM call). With ~36 GEMMs per decode step, this results in **424ms/step** (99.8% of decode time) — a **65× slowdown** compared to FP16's 6.5ms/step.

Root cause: Apple's MPS framework provides no native BF16 `MPSMatrixMultiplication`. The only BF16 GEMM path is through `MPSGraph`, which compiles, executes, and synchronizes per call. On CUDA, `cublasGemmEx` handles BF16 natively at FP16 speed.

### Why BF16 exists in CTranslate2

BF16 was added for CUDA (Ampere+) where it runs at FP16 speed with the advantage of FP32-range exponents (8-bit vs FP16's 5-bit), reducing overflow risk in attention scores and gradient-heavy operations. On MPS, this hardware advantage doesn't exist — Apple's AMX handles FP16 natively but has no BF16 acceleration path.

---

## 2. Solution

**Auto-promote BF16→FP16 on MPS** at model load time, with an environment variable escape hatch.

### Behavior

| Request | MPS effective type | Notes |
|---------|-------------------|-------|
| `compute_type="bfloat16"` | **float16** | Auto-promoted with warning |
| `compute_type="int8_bfloat16"` | **int8_float16** | Auto-promoted with warning |
| `compute_type="bfloat16"` + `CT2_MPS_NATIVE_BF16=1` | bfloat16 | Native BF16, 65× slower |
| `compute_type="int8_bfloat16"` + `CT2_MPS_NATIVE_BF16=1` | int8_bfloat16 | Native BF16, 65× slower |

### Warning message

```
Requested compute type bfloat16 was automatically promoted to float16 on MPS
for performance (BF16 uses synchronous MPSGraph, ~65x slower).
Set CT2_MPS_NATIVE_BF16=1 to force native BF16.
```

### Precision impact

FP16 has reduced exponent range (5 bits vs BF16's 8 bits) but higher mantissa precision (10 bits vs BF16's 7 bits). For transformer inference:
- **Attention scores**: FP16 is fine — values are bounded by softmax
- **Weight representation**: FP16 has better precision for the typical [-1, 1] weight range
- **Accumulation**: Both use FP32 accumulation in MPS GEMM
- **Token-level output**: BF16 and FP16 produce identical greedy/beam outputs for OPUS-MT (verified in M11.3)

The only scenario where BF16's wider exponent range matters is models with unusually large intermediate values (not typical for inference).

---

## 3. Implementation

### Files modified

| File | Change | Purpose |
|------|--------|---------|
| `src/types.cc` | Added MPS BF16→FP16 promotion in `resolve_compute_type()` | Auto-promote BFLOAT16→FLOAT16 and INT8_BFLOAT16→INT8_FLOAT16 on MPS |
| `src/models/model.cc` | Added explicit warning for compute type promotion | Log when requested != effective type (not just DEFAULT) |

### Code changes

**`src/types.cc`** — `resolve_compute_type()`:

```cpp
case ComputeType::BFLOAT16: {
  if (support_bfloat16) {
#ifdef CT2_WITH_MPS
    if (device == Device::MPS) {
      static const bool native_bf16 = read_bool_from_env("CT2_MPS_NATIVE_BF16");
      if (!native_bf16 && support_float16)
        return ComputeType::FLOAT16;
    }
#endif
    return ComputeType::BFLOAT16;
  }
  // fallback chain: FLOAT16 → FLOAT32
}
```

Same pattern for `ComputeType::INT8_BFLOAT16` → `ComputeType::INT8_FLOAT16`.

The fallback chains for non-MPS (or when BF16 isn't supported at all) were also improved to try FLOAT16 before FLOAT32.

---

## 4. Performance Results

### 50 sentences, beam_size=2, best-of-3

| Type | Before (tok/s) | After (tok/s) | Speedup | Effective type |
|------|---------------|--------------|---------|----------------|
| **bfloat16** | **9** | **1631** | **181×** | float16 |
| **int8_bfloat16** | **9** | **574** | **64×** | int8_float16 |
| float16 (baseline) | 1585 | 1585 | — | float16 |
| int8_float16 (baseline) | 568 | 568 | — | int8_float16 |

The promoted types match their native FP16 counterparts within noise:
- bf16→f16: 1631 vs 1585 tok/s (within run-to-run variance)
- int8_bf16→int8_f16: 574 vs 568 tok/s (within variance)

### Escape hatch verification

```bash
CT2_MPS_NATIVE_BF16=1  # Forces native BF16, no promotion, no warning
```

Verified: `effective_compute_type = bfloat16` when env var is set.

---

## 5. Design Decisions

### Why auto-promote instead of error?

- **User intent preservation**: Users requesting BF16 want "good performance with wide-range floats." FP16 delivers both on MPS.
- **Model compatibility**: Models distributed as BF16 (e.g., from Hugging Face) should work on MPS without manual intervention.
- **Precedent**: CTranslate2 already falls back INT16→INT8_FLOAT32→FLOAT32 when hardware doesn't support the requested type.

### Why env var escape hatch?

- **Debugging**: Users may need to verify BF16 behavior matches a reference implementation.
- **Future hardware**: If Apple adds native BF16 GEMM support, users can opt in before we update the detection logic.
- **Principle**: Auto-promotion is a pragmatic default, not a mandate.

### Why not fix MPSGraph to be async?

Possible but high-effort and limited benefit:
- Would require managing a separate MPSGraph command buffer lifecycle alongside the Metal command buffer
- Even with async dispatch, each BF16 GEMM still takes ~10ms (vs ~0.2ms for FP16 via MPSMatrixMultiplication)
- Net improvement would be ~50-100ms/step (async dispatch) vs ~6.5ms/step (FP16 promotion)
- FP16 promotion is zero-effort and gives the full speedup

---

## 6. Updated Performance Summary (all types, post-M12.5)

| Type | tok/s | vs CPU (826) | Bottleneck | Status |
|------|-------|-------------|------------|--------|
| **float16** | **1585** | **1.92x** | GPU compute + sync balanced | Optimal |
| **float32** | **1019** | **1.23x** | f32 GEMM slower, longer sync waits | Optimal |
| **bfloat16** | **1631** | **1.97x** | Auto-promoted to float16 | **Fixed (M12.5)** |
| **int8** | **84** | **0.10x** | CPU int8↔f32 conversion + syncs | M12.6 target |
| **int8_float16** | **568** | **0.69x** | CPU int8↔f32 conversion + syncs | M12.6 target |
| **int8_bfloat16** | **574** | **0.69x** | Auto-promoted to int8_float16 | **Fixed (M12.5)** |
