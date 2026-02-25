# Milestone 0.2 – Validate BF16 Availability

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Agent:** `cpp-pro`
**Plan ref:** `APPLE_M4_METAL_PLAN.md` §0.2

---

## Task Description

- Check `[device supportsFamily:MTLGPUFamilyApple9]` (M3+) for BF16 hardware support.
- Run BF16 MPS matmul and compare correctness against FP32 reference.
- Document architecture findings.

---

## Results on Apple M4

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| GPU family | Apple9 | ≥ Apple9 | **PASS** |
| BF16 HW support | YES | Required | **PASS** |
| Correctness (max rel diff) | `3.6e-03` | `< 1e-02` | **PASS** |
| BF16 MPSGraph avg latency | 44.69 ms (3.08 TFLOPS) | — | — |
| BF16 vs FP32 Metal (M0.1) | 1.04× | informational | — |
| BF16 vs FP32 CPU | 2.0× | informational | — |

**Overall: PASS**

---

## Key Findings

### 1. BF16 hardware is available (Apple9 GPU family)
M4 reports `MTLGPUFamilyApple9` as the highest supported tier. This confirms hardware BF16 support is present, as the plan required.

### 2. Critical: `MPSMatrixMultiplication` does NOT support BF16
The first implementation attempt triggered a runtime assertion:
```
failed assertion `Input data type must be one of
MPSDataTypeFloat32, MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16.'
```
`MPSMatrixMultiplication` is limited to FP32, FP16, and integer types. **BF16 matrix multiply requires `MPSGraph.matrixMultiplicationWithPrimaryTensor:secondaryTensor:`.**

This directly validates the plan's guidance:
> *"Use MPSGraph only where MPS doesn't have a single-op equivalent."*

GEMM is the primary primitive in CTranslate2. The infrastructure choice is now clear:
- **FP32**: `MPSMatrixMultiplication` (eager, no graph overhead)
- **BF16**: `MPSGraph` matmul (graph compilation amortised over runs)

### 3. BF16 vs FP32 Metal: only ~1.04× faster (same throughput class)
Both BF16 MPSGraph and FP32 MPSMatrixMultiplication run in the ~44–47 ms range at ~3 TFLOPS. M4 does not exhibit the 2× BF16 throughput advantage seen on discrete GPUs (e.g. NVIDIA A100: FP16/BF16 is 2× FP32 via Tensor Cores).

**Implication:** BF16 on M4 is primarily a memory-bandwidth advantage (2 bytes vs 4 bytes per element), not a compute-throughput advantage. For weight-memory-bound inference (e.g. long token generation), BF16 will still be beneficial.

### 4. Correctness methodology: reference must use BF16-quantised inputs
Comparing BF16 Metal output against `cblas_sgemm(fp32A, fp32B)` produced 7% max relative error — misleadingly "failing" due to input quantisation noise (each A/B element is off by ≤ 0.78% after BF16 conversion, accumulated over K=4096 terms).

The correct comparison: run `cblas_sgemm` on the **same BF16-quantised inputs** (round-tripped through `float_to_bf16 → bf16_to_float`). This isolates hardware accumulation precision from input quantisation. After this fix:
- Max rel diff: `3.6e-3` (0.36%) — comfortably within 1% tolerance.
- This implies MPS accumulates internally in FP32 (or better), with only the output quantised to BF16.

---

## Architecture Implications for CTranslate2

| Operation | API to use | Reason |
|-----------|-----------|--------|
| FP32 GEMM | `MPSMatrixMultiplication` | Eager, variable shapes, no compilation |
| BF16 GEMM | `MPSGraph` matmul | Only BF16-capable API for matrix multiply |
| Other MPS ops (FP16 attention, etc.) | `MPSMatrixMultiplication` | Supported |
| RMS Norm, Rotary emb | `MPSGraph` | No MPS single-op equivalent (as planned) |

---

## Files Created / Modified

| File | Action |
|------|--------|
| `tools/metal_poc/bf16_poc.mm` | Created |

---

## Build & Run

```bash
clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
    -DACCELERATE_NEW_LAPACK \
    -o bf16_poc tools/metal_poc/bf16_poc.mm \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -framework Accelerate
./bf16_poc
```

Note: Requires macOS 14.0+ for `MPSDataTypeBFloat16`.

---

## Recommendations / Next Steps

- **Proceed to Milestone 0.3** (command buffer latency test). Both correctness and hardware support are confirmed.
- **Update plan §Milestone 2+**: GEMM primitive dispatch must branch on dtype — FP32 uses `MPSMatrixMultiplication`, BF16 uses `MPSGraph`.
- **BF16 performance note**: throughput parity with FP32 is still useful for inference — 2× smaller memory footprint enables larger batches and longer context windows within the M4's unified memory budget.
