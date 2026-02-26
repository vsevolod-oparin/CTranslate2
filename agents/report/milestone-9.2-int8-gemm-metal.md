# Milestone 9.2 — INT8 GEMM on Metal

**Date:** 2026-02-26
**Status:** ✅ DONE

---

## Summary

M9.2 implements `primitives<Device::METAL>::gemm<int8_t, int32_t>()` and
`gemm_batch_strided<int8_t, int32_t>()`.

Metal MPS has no native INT8 matmul (macOS 14).  The workaround uses
float32 as an intermediate:

```
int8 A, int8 B
    │
    │ CPU: int8 → float32 (exact, ±127 fits in float32 mantissa)
    ↓
float32 A', float32 B' (in alloc_temp_buffer MTLBuffers)
    │
    │ GPU: float32 MPS GEMM (MPSMatrixMultiplication, encode + commit)
    ↓
float32 C'
    │
    │ CPU: float32 → int32 (std::lroundf)
    ↓
int32 C
```

**Test result: 9/9 pass** (`tests/metal/m92_test.mm`)

---

## Architecture

### Why float32 (not float16)?

`max accumulator = 127 * 127 * k`.  For k = 512, max = 8,257,792.
Float16 overflows at 65,504 (k ≥ 5 for worst case); float32 is exact for
integers up to 2^24 = 16,777,216, covering k ≤ 1040.  For typical inference
`k ≤ 1024`, the float32 path gives mathematically exact integer accumulation.

### Why CPU conversion (not a GPU kernel)?

INT8 GEMM always begins with `commit_and_wait()` (it takes over the CPU
path anyway), so the CPU conversion of A and B is "free" pipeline-wise.
No new MSL kernels are needed.

### `alloc_temp_buffer` vs MetalAllocator

Temporary buffers for the float32 A'/B'/C' conversion are allocated via
`alloc_temp_buffer()` (direct `MTLDevice newBufferWithLength:`), which
bypasses the MetalAllocator live-map.  The existing `dispatch_mps_gemm<T>`
calls `metal_buffer_for_ptr()` internally (live-map lookup) and cannot be
used with these temps.

Solution: a new helper `dispatch_mps_gemm_buf()` takes `id<MTLBuffer>`
arguments directly (offset=0), bypassing `metal_buffer_for_ptr`.

### Row padding

`dispatch_int8_gemm` queries `MPSMatrixDescriptor rowBytesForColumns:` and
pre-allocates temp buffers with at least that row stride, so no additional
padding copy is needed before MPS.

### beta

Only `beta = 0` is supported (INT8 GEMM in CTranslate2 is always called
with `beta = 0`).  A non-zero beta throws.

---

## Files Modified

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `dispatch_mps_gemm_buf()`, `dispatch_int8_gemm()`; added int8/int32 branch to `gemm` and `gemm_batch_strided`; added explicit instantiations |
| `METAL_ARCHITECTURE.md` | Updated status, GEMM dispatch table, milestone history |

## Files Created

| File | Description |
|------|-------------|
| `tests/metal/m92_test.mm` | 9 tests: basic GEMM, trans_b, alpha, large k, zero, full pipeline, batch_strided, gemm_pack_b |
| `agents/report/milestone-9.2-int8-gemm-metal.md` | This report |

---

## Results (Apple M4)

| Test | Description | Result | Pass |
|------|-------------|--------|------|
| 1 | Basic INT8 GEMM (m=4,n=8,k=16, no transpose) | all elements exact | ✅ |
| 2 | trans_b=true (m=8,n=4,k=32) | all elements exact | ✅ |
| 3 | alpha=2.5 (m=4,n=4,k=8) | all elements exact | ✅ |
| 4 | Large k=512 (m=8,n=16) | all elements exact; accum < 2^24 | ✅ |
| 5 | Zero matrix | all zeros | ✅ |
| 6 | Full pipeline: float→quant→INT8 GEMM→dequant→float | norm_err=0.43% | ✅ |
| 7 | batch_strided (batch=3,m=4,n=8,k=16) | all elements exact | ✅ |
| 8 | gemm_pack_b<int8_t> returns 0 | result=0 | ✅ |

Tests 1–5, 7: outputs match CPU INT8 reference exactly (integer accumulation,
exact in float32).

Test 6 pipeline error: max(|Y - Y_ref|) / max(|Y_ref|) = **0.43%** (well
within 1%).  The error is due to INT8 quantization of both A and B; the
per-element relative error appears higher (2.2%) for the smallest output
values, but the absolute error is only 0.045 vs a peak output of 10.5.

---

## Build Command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m92_test.mm \
    src/metal/ops_quantize.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm \
    src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm \
    src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm \
    src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m92_test && ./m92_test
```

---

## Key Findings

1. **Float32 is exact for typical k**: Max INT8 accumulator (127*127*512 =
   8.25M) fits well within float32 exact integer range (2^24 = 16.8M).
   Tests 1–5 all produce bit-exact int32 output matching the CPU reference.

2. **`dispatch_mps_gemm_buf` reuse**: The new low-level helper shares the
   same MPS matrix setup and encoding logic as `dispatch_mps_gemm` but
   accepts pre-prepared `id<MTLBuffer>` arguments.  The existing `dispatch_mps_gemm`
   cannot be used for temp buffers since `metal_buffer_for_ptr` requires
   allocations to be registered in the MetalAllocator live-map.

3. **Two commit_and_wait calls**: `dispatch_int8_gemm` calls `commit_and_wait`
   twice — once to flush before CPU reads int8 inputs, once after GPU GEMM.
   This is unavoidable for the dequantize-before-GEMM strategy but is
   acceptable since INT8 ops are already on the CPU-synchronous path.

4. **Full pipeline 0.43% error**: The INT8 quantization of both A and B
   (two independent rounding steps) accumulates error of ~0.4% relative to
   the output peak, consistent with INT8 round-trip error of ~0.4% seen in M9.1.

---

## Deferred Items (M10+)

| Item | Notes |
|------|-------|
| Native INT8 MPS (when available) | Swap `dispatch_int8_gemm` implementation if Apple adds INT8 to MPS |
| Full INT8 model loading test | Requires M10 model loading infrastructure |
| Performance benchmark | INT8 pipeline: 2× commit_and_wait → profile in full model context |
