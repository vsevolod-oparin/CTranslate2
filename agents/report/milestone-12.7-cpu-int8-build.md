# M12.7 — CPU INT8 Build Support (RUY)

**Date**: 2026-03-11
**Status**: Complete — CPU INT8 works, but slower than CPU FP32 on Apple Silicon
**Hardware**: Apple M4 (10-core), macOS 15

---

## 1. Problem

CPU INT8 inference failed with:
```
ValueError: Requested int8_float32 compute type, but the target device or
backend do not support efficient int8_float32 computation.
```

The build had `WITH_RUY=OFF`, `WITH_MKL=OFF`, `WITH_DNNL=OFF`. Without any INT8 GEMM backend, `cpu::has_gemm_backend(ComputeType::INT8)` returns false.

---

## 2. Fix

Enabled RUY (Google's matrix multiplication library, already bundled at `third_party/ruy/`):

```bash
cmake -DWITH_RUY=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
```

The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag is needed because RUY's `cpuinfo` dependency has an outdated CMake minimum version that conflicts with CMake 3.30+.

### Build details

- RUY source: `third_party/ruy/` (already bundled, no download needed)
- RUY uses ARM NEON SIMD for INT8 GEMM on Apple Silicon
- `cpuinfo` (CPU detection library) built as a static dependency
- Adds `CT2_WITH_RUY` preprocessor define
- INT8 GEMM path: `ruy::Mul()` with `ruy::MulParams` for alpha/beta scaling

---

## 3. Performance Results

### 50 sentences, beam=4, max_batch_size=32, best-of-3

| Configuration | tok/s | ms | Notes |
|--------------|-------|-----|-------|
| **MPS float16** | **1473** | 1049 | Best overall |
| **CPU float32** (threads=4) | **807** | 1920 | Accelerate/AMX |
| **CPU int8** (threads=4, RUY) | **580** | 2683 | New (M12.7) |
| CPU int8 (threads=2, RUY) | 429 | 3633 | |
| CPU int8 (threads=1, RUY) | 261 | 5964 | |
| MPS int8_float16 | 488 | 3168 | M12.6 GPU dequant |
| MPS int8 | 455 | 3412 | M12.6 GPU dequant |

### CPU thread scaling

| Threads | CPU float32 (tok/s) | CPU int8 RUY (tok/s) | INT8/FP32 ratio |
|---------|--------------------|--------------------|-----------------|
| 1 | 743 | 261 | 0.35× |
| 2 | 742 | 429 | 0.58× |
| 4 | 807 | 552 | 0.68× |
| 8 | 750 | 495 | 0.66× |

---

## 4. Key Finding: INT8 Is Slower Than FP32 on Apple Silicon

CPU INT8 with RUY is **28-65% slower** than CPU FP32 with Accelerate on Apple M4.

### Why?

1. **Apple AMX advantage**: Accelerate's FP32 GEMM dispatches to the Apple Matrix Extension (AMX), specialized hardware for matrix multiplication that achieves near-peak throughput. RUY's INT8 GEMM uses general-purpose ARM NEON SIMD instructions.

2. **Quantize/dequantize overhead**: Each INT8 GEMM requires:
   - Quantize activation: FP32 → INT8 (per-row scale + round)
   - INT8 GEMM (RUY, NEON)
   - Dequantize output: INT32 → FP32 (rescale + bias + activation)

   This overhead is amortized on x86 servers where INT8 GEMM is 2-4× faster than FP32, but on Apple Silicon the FP32 GEMM is so fast that the overhead exceeds the savings.

3. **INT8 is designed for x86**: CTranslate2's INT8 path primarily targets Intel VNNI (AVX-512) and similar server hardware where INT8 ops are ~4× faster than FP32. On Apple Silicon, the FP32 AMX path is already near-optimal.

### Practical recommendation

On Apple Silicon:
- **Use MPS float16** for best speed (1473 tok/s)
- **Use CPU float32** as fallback (807 tok/s)
- **INT8 is only useful for memory reduction** (2× less weight memory), not speed
- MPS int8_float16 (488 tok/s) and CPU int8 (580 tok/s) are comparable but both slower than CPU f32

---

## 5. Updated Performance Summary (all backends, post M12.7)

| Backend | Type | tok/s | vs CPU f32 | Memory | Use case |
|---------|------|-------|-----------|--------|----------|
| MPS | float16 | 1473 | 1.82× | 1× | **Default choice** |
| MPS | bfloat16 | 1426 | 1.77× | 1× | Auto-promoted to f16 |
| MPS | float32 | 1000 | 1.24× | 2× | Precision-sensitive |
| CPU | float32 | 807 | 1.00× | 2× | Baseline |
| CPU | int8 (RUY) | 580 | 0.72× | 0.5× | **Memory-constrained only** |
| MPS | int8_f16 | 488 | 0.60× | 0.5× | Memory + GPU |
| MPS | int8 | 455 | 0.56× | 0.5× | Memory + GPU |

---

## 6. Files Modified

| File | Change |
|------|--------|
| `build/CMakeCache.txt` | `WITH_RUY=ON` (runtime reconfigure) |

No source code changes were needed — RUY integration was already complete in the codebase. Only the build configuration flag was toggled.

The `CMAKE_POLICY_VERSION_MINIMUM=3.5` workaround is needed for CMake 3.30+ compatibility with RUY's `cpuinfo` dependency.
