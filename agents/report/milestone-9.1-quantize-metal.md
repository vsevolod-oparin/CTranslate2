# Milestone 9.1 — INT8 Quantize / Dequantize on Metal

**Date:** 2026-02-26
**Status:** ✅ DONE

---

## Summary

M9.1 implements `Quantize::quantize<Device::METAL, T, int8_t>()`,
`Dequantize::dequantize<Device::METAL, int8_t, T>()`, and
`Dequantize::dequantize_gemm_output<Device::METAL, T>()` using GPU MSL kernels.

Also implements M9.3 (`gemm_pack_b` returns 0 — already done) and M9.4
(`compute_u8_compensation` changed from METAL_STUB to no-op).

**Test result: 15/15 pass** (`tests/metal/m91_test.mm`)

---

## Architecture

Three GPU kernel families in `src/metal/kernels/quantize.metal`:

```
quantize_T        — one threadgroup per row, 256-thread tree reduction
                    finds abs-max, computes scale=127/amax, rounds to int8
dequantize_T      — one thread per element: output[row,i] = T(int8/scale[row])
dequantize_gemm_output_T
                  — one thread per element: rescales int32 GEMM output,
                    adds optional bias, applies optional activation
```

Two-layer design (same pattern as Conv1D in M8.3):

```
src/metal/ops_quantize.mm     metal::quantize_int8_metal<T>()
                               metal::dequantize_int8_metal<T>()
                               metal::dequantize_gemm_output_metal<T>()
                                         │
src/ops/quantize_metal.mm    Quantize::quantize<METAL,T,int8_t>
src/ops/dequantize_metal.mm  Dequantize::dequantize<METAL,int8_t,T>
                             Dequantize::dequantize_gemm_output<METAL,T>
```

### quantize_T kernel

```
// One threadgroup per row; 256 threads.
// threadgroup float sdata[256] for tree reduction.
//
// Step 1: each thread strides over row to find local abs-max
//   thread_max = max(|row[tid]|, |row[tid+256]|, |row[tid+512]|, ...)
// Step 2: tree reduction in sdata[] → row abs-max in sdata[0]
//   scale = 127 / sdata[0]   (or 1 if zero row)
//   thread 0 writes scale[row]
// Step 3: output[row,i] = char(round(float(input[row,i]) * scale))
```

### Null-bias handling in dequantize_gemm_output

When `bias == nullptr`, the `c` buffer is passed as a dummy for buffer slot 3;
`has_bias = 0` prevents the kernel from reading it. Same pattern as
normalization kernels using null gamma/beta.

### Activation encoding

```
activation_type passed to metal:: = static_cast<int>(ActivationType)
                                   (-1 for no activation)
kernel act_type = (activation_type < 0) ? 0 : activation_type + 1

0=none  1=relu  2=gelu_tanh  3=swish  4=gelu  5=gelu_sigmoid  6=tanh  7=sigmoid
```

---

## Files Created

| File | Description |
|------|-------------|
| `src/metal/kernels/quantize.metal` | MSL kernels: quantize/dequantize/dequantize_gemm_output |
| `src/metal/ops_quantize.mm` | `metal::quantize_int8_metal<T>()` + `metal::dequantize_int8_metal<T>()` + `metal::dequantize_gemm_output_metal<T>()` |
| `src/ops/quantize_metal.mm` | `Quantize::quantize<METAL>` thin wrapper |
| `src/ops/dequantize_metal.mm` | `Dequantize::dequantize<METAL>` + `dequantize_gemm_output<METAL>` thin wrappers |
| `tests/metal/m91_test.mm` | 10 tests (15 assertions) |
| `agents/report/milestone-9.1-quantize-metal.md` | This report |

## Files Modified

| File | Change |
|------|--------|
| `tools/gen_msl_strings.py` | Added `("quantize", "kQuantizeMSL")`; updated docstring to "eleven" |
| `src/metal/msl_strings.h` | Regenerated (auto-generated) |
| `src/metal/ops_metal.h` | Added `quantize_int8_metal`, `dequantize_int8_metal`, `dequantize_gemm_output_metal` declarations; updated header comment |
| `src/metal/primitives_memory.mm` | `compute_u8_compensation`: METAL_STUB → no-op (M9.4) |
| `CMakeLists.txt` | Added `ops_quantize.mm`, `quantize_metal.mm`, `dequantize_metal.mm`; added `quantize.metal` to `_MSL_METAL_SOURCES` |
| `METAL_ARCHITECTURE.md` | Updated status + milestone table + directory structure |

---

## Results (Apple M4, float32/float16/bfloat16)

| Test | Description | Result | Pass |
|------|-------------|--------|------|
| 1 | f32 round-trip B=4,D=128 | round-trip_err=0.0000% | ✅ |
| 2 | f32 zero-row guard (scale=1) | max_err=0.0000 | ✅ |
| 3 | f16 round-trip B=2,D=64 | rel=0.378% | ✅ |
| 4 | bf16 round-trip B=2,D=64 | rel=0.628% | ✅ |
| 5 | f32 large B=8,D=512 | max_err=0.394% | ✅ |
| 6 | dequantize_gemm_output (no bias, no act) | max_err=9.54e-07 | ✅ |
| 7 | dequantize_gemm_output (with bias) | max_err=1.91e-06 | ✅ |
| 8 | dequantize_gemm_output (relu) | max_err=0.00, min≥0 | ✅ |
| 9 | gemm_pack_b returns 0 (M9.3) | result=0 | ✅ |
| 10 | compute_u8_compensation no-op (M9.4) | no throw, buf unchanged | ✅ |

All round-trip errors well within 1% criterion.
Float32: ~0% (quantization error is negligible for the test values).
Float16/BF16: < 0.7% (within expected ~1 decimal digit precision).

---

## Build Command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m91_test.mm \
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
    -o m91_test && ./m91_test
```

---

## Key Findings

1. **GPU quantize correctness**: Float32 round-trip error is essentially zero
   for the test data (values on a regular grid). The ULP-level quantization
   error only appears for truly random inputs; even there < 0.4%.

2. **Zero-row guard works**: When an entire row is zero, `scale = 1` is
   assigned (instead of dividing by zero). Output stays all zeros.

3. **BF16 GEMM output rescaling**: Works correctly since the kernel operates
   in float32 internally — `int32_t → float32 → T` cast path is always safe.

4. **ct2_erf polynomial**: Used for GELU activation in
   `dequantize_gemm_output`; MSL `erf()` availability varies by target, the
   polynomial is always safe (max error 1.5e-7).

5. **M9.3/M9.4 already trivial**: `gemm_pack_b` was already returning 0 from
   M4.4. `compute_u8_compensation` changed from METAL_STUB to empty function.

---

## Deferred Items (M9.2+)

| Item | Notes |
|------|-------|
| INT8 GEMM on Metal | M9.2: dequantize INT8 weights to FP16 → FP16 GEMM |
| Full INT8 model loading test | Requires M9.2 + model loading infrastructure |
| dequantize_gemm_output f16/bf16 tests | f32 sufficient for M9.1 criterion |
