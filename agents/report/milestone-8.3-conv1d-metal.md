# Milestone 8.3 — Conv1D on Metal (im2col + GEMM)

**Date:** 2026-02-26
**Status:** ✅ DONE

---

## Summary

M8.3 implements `Conv1D::compute<Device::METAL, T>` using the im2col + GEMM
approach (same algorithm as `conv1d_gpu.cu`), supporting `float32`, `float16`,
and `bfloat16`.

Constraint: `groups == 1` only (Whisper uses groups == 1 throughout).

**Test result: 7/7 tests pass** (`tests/metal/m83_test.mm`)

---

## Architecture

Two-layer design (matching the pattern established for SDPA in M6.1):

```
src/metal/ops_conv1d.mm     metal::conv1d_metal<T>()
                                      │
        ┌─────────────────────────────┴──────────────────────────────┐
        │ 1. dispatch_im2col<T>()                                    │
        │    im2col kernel: input[B, C_in, T_in] → buf[B, T_out, CK]│
        │    kConv1dMSL, encode-only                                 │
        │                                                            │
        │ 2. per-batch GEMM loop                                     │
        │    primitives<METAL>::gemm<T,T>(trans_b=true)              │
        │    weight[C_out, CK] × im2col[b, T_out, CK]^T             │
        │    → output[b, C_out, T_out]                               │
        └────────────────────────────────────────────────────────────┘

src/ops/conv1d_metal.mm     Conv1D::compute<Device::METAL, T>
                                calls metal::conv1d_metal<T>()
                                then apply_bias_and_activation()
```

### im2col Kernel (conv1d.metal)

```
// gid ∈ [0, B * T_out * C_in * K)
//   b  =  gid / (T_out * C_in * K)
//   ti = (gid / (C_in * K)) % T_out
//   c  = (gid % (C_in * K)) / K
//   k  =  gid % K
//
// win = ti*stride - padding + k*dilation
// output[gid] = (win ∈ [0, T_in)) ? input[b, c, win] : 0
```

### GEMM Convention

```
trans_a = false: A = weight [C_out, CK], lda = CK
trans_b = true:  B = im2col [T_out, CK], ldb = CK  (transposed → [CK, T_out])
C = output [C_out, T_out], ldc = T_out
```

For float32/float16: encode-only (GEMM encoded after im2col in same CB).
For bfloat16: first GEMM calls `commit_and_wait()` (flushes im2col + BF16 setup), then runs MPSGraph synchronously for each batch.

### im2col Buffer Allocation

Allocated via `get_allocator<Device::METAL>().allocate()` (RAII `MetalTempBuf`)
so `metal_buffer_for_ptr()` can locate it for the GEMM pass.

---

## Files Created

| File | Description |
|------|-------------|
| `src/metal/kernels/conv1d.metal` | MSL im2col kernel (float/half/bfloat) |
| `src/metal/ops_conv1d.mm` | `metal::conv1d_metal<T>()` dispatch |
| `src/ops/conv1d_metal.mm` | `Conv1D::compute<METAL>` thin wrapper |
| `tests/metal/m83_test.mm` | 7 tests: f32 (5 shapes) + f16 + bf16 |
| `agents/report/milestone-8.3-conv1d-metal.md` | This report |

## Files Modified

| File | Change |
|------|--------|
| `tools/gen_msl_strings.py` | Added `("conv1d", "kConv1dMSL")` entry |
| `src/metal/msl_strings.h` | Regenerated (auto-generated) |
| `src/metal/ops_metal.h` | Added `conv1d_metal<T>()` declaration |
| `CMakeLists.txt` | Added `ops_conv1d.mm` + `conv1d.metal` to MSL sync list |
| `METAL_ARCHITECTURE.md` | Updated status + milestone table |

---

## Results (Apple M4, float32/float16/bfloat16)

| Test | Shape | max_abs_diff | Pass |
|------|-------|-------------|------|
| 1 | f32 B=1, C_in=4, T_in=8, C_out=8, K=3, s=1 | 2.38e-07 | ✅ |
| 2 | f32 stride=2 | 1.49e-07 | ✅ |
| 3 | f32 dilation=2, batch=2 | 2.38e-07 | ✅ |
| 4 | f32 batch=3 | 2.38e-07 | ✅ |
| 5 | f32 Whisper-like C_in=80, C_out=128, K=3 | 0.00e+00 | ✅ |
| 6 | f16 B=1, C_in=8, T_in=16, C_out=8, K=3 | 4.80e-04 | ✅ |
| 7 | bf16 B=1, C_in=8, T_in=16, C_out=8, K=3 | 2.74e-03 | ✅ |

Float32 errors at the ULP level (~1e-7). Float16/BF16 errors well within
the expected ~3 / ~2 decimal digit precision.

---

## Build Command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m83_test.mm \
    src/metal/ops_conv1d.mm \
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
    -o m83_test && ./m83_test
```

---

## Key Findings

1. **im2col + GEMM is exact**: Float32 errors are at the single-ULP level (~1e-7),
   matching M8.1/M8.2 encoder/decoder results.

2. **Whisper-like shape (C_in=80, C_out=128)**: Zero error on the Whisper-like
   test (Test 5 with small T_in=32). The im2col correctly handles larger
   channel sizes without numerical issues.

3. **BF16 GEMM flush pattern works**: The MPSGraph-based BF16 GEMM calls
   `commit_and_wait()` on the first batch, which also flushes the im2col kernel
   encoded before the GEMM loop. Subsequent batch GEMMs read the already-flushed
   im2col buffer via Shared memory. No additional coordination needed.

4. **Architecture follows established pattern**: `metal::conv1d_metal<T>()` in
   `ops_conv1d.mm` mirrors `metal::sdpa_metal<T>()` in `ops_sdpa.mm`.
   Tests call the Metal function directly (no CPU deps), same as m82_test.mm.

5. **groups != 1 throws cleanly**: Validated by the throw in `conv1d_metal.mm`
   (not separately tested since Whisper requires groups==1 only).

---

## Deferred Items (M8.4+)

| Item | Notes |
|------|-------|
| Full WhisperEncoder test | Conv1d×2 + N transformer encoder layers |
| Bias + activation via op layer | Requires CPU deps in test or mock dispatch |
| groups > 1 | Requires GEMM loop over groups × batches |
| Dilation = 0 guard | Currently clamped to 1 in conv1d_metal.mm |
