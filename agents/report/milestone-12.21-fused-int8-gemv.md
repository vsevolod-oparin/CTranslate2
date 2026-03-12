# M12.21 — Fused INT8 GEMV Kernel

**Date**: 2026-03-12
**Status**: COMPLETE
**Branch**: `metal-backend`

## Summary

Implemented a fused MSL kernel for INT8 GEMV (matrix-vector multiply, m=1) that reads int8 A and B directly, accumulates in int32, and outputs int32 — eliminating the previous 3-kernel pipeline (int8→f32 dequant A, int8→f32 dequant B, MPS f32 GEMM, f32→int32 round) and 3 temporary f32 buffer allocations per GEMM.

## Performance Results (TinyLlama-1.1B, Apple M4, greedy beam=1, max_length=100)

### Before vs After (standard MHA)

| Compute Type | Before (3-kernel) | After (fused GEMV) | Speedup |
|-------------|-------------------|-------------------|---------|
| **int8** | 3.1 tok/s | **34.3 tok/s** | **11.1x** |
| **int8_f16** | 3.6 tok/s | **31.9 tok/s** | **8.9x** |

### Full benchmark (all compute types, after optimization)

| Compute Type | Standard MHA | Flash MHA | Flash speedup |
|-------------|-------------|-----------|---------------|
| **f32** | 18.9 | 18.5 | 0.98x |
| **f16** | 25.3 | 38.9 | 1.54x |
| **bf16** (→f16) | 30.8 | 39.7 | 1.29x |
| **int8** (→int8_f16) | **34.3** | **41.2** | 1.20x |
| **int8_f16** | 31.9 | 39.5 | 1.24x |
| **int8_bf16** (→int8_f16) | 33.8 | 39.5 | 1.17x |

INT8 is now the **fastest standard MHA path** at 34.3 tok/s, beating even f16 (25.3 tok/s). Flash INT8 at 41.2 tok/s is the fastest configuration overall.

## Root Cause Analysis

### Why INT8 was slow (3.1 tok/s before)

The previous INT8 GEMM decode path required **4 GPU kernel dispatches + 3 temp buffer allocs per GEMM**:

```
For each of 36 GEMMs per decode step:
  1. alloc_temp_buffer(rows_a × rb_a)     // f32 temp A
  2. alloc_temp_buffer(rows_b × rb_b)     // f32 temp B (16MB for 2048×2048)
  3. alloc_temp_buffer(m × rb_c)          // f32 temp C
  4. encode int8_to_float32(A)            // GPU kernel #1
  5. encode int8_to_float32(B)            // GPU kernel #2
  6. encode MPS GEMM(f32)                 // GPU kernel #3
  7. encode float32_to_int32(C)           // GPU kernel #4
  8. [release 3 temp buffers]
```

For decode (m=1) with a 2048×2048 weight matrix:
- **Old path memory reads**: A as int8 (2KB) → f32 (8KB), B as int8 (4MB) → f32 (16MB), then MPS GEMM reads 16MB+8KB. **Total: ~36MB** per GEMM.
- **New path memory reads**: A as int8 (2KB), B as int8 (4MB). **Total: ~4MB** per GEMM.
- **Bandwidth reduction**: ~9× per GEMM, ~36 GEMMs/step → saves ~1.2GB/step of memory traffic.

Additionally, `alloc_temp_buffer()` calls `[MTLDevice newBufferWithLength:]` (ObjC alloc), costing ~50-100µs each. 3 allocs × 36 GEMMs = 108 allocs/step → ~5-10ms/step overhead.

## Optimization Implemented

### Fused INT8 GEMV MSL Kernel

**File**: `src/metal/primitives_gemm.mm`

For decode (m=1, trans_b=true, contiguous layout), replaced the 4-kernel pipeline with a single fused MSL kernel:

```metal
kernel void fused_int8_gemv(
    device const char* a,       // [1, k] int8 vector
    device const char* b,       // [n, k] int8 matrix (transposed)
    device int*        c,       // [1, n] int32 output
    constant uint2&    params,  // {n, k}
    constant float&    alpha,
    uint gid [[thread_position_in_grid]])
```

Key design decisions:
1. **char4 vectorized reads**: Processes 4 int8 values per loop iteration for 4× bandwidth efficiency.
2. **int32 accumulation**: More accurate than f32 for large k. int32 is exact for k ≤ 133K (vs f32 exact for k ≤ 1040). Same accumulation method as CPU RUY backend.
3. **One thread per output**: Each thread computes one dot product c[gid] = round(alpha × sum(a[i] × b[gid, i])).
4. **Fallback for non-GEMV**: m>1 (prefill), trans_a, or non-contiguous layouts fall back to the existing 3-kernel MPS GEMM path (which uses AMX hardware acceleration).

### Integration

In `dispatch_int8_gemm()`, added routing logic:
```cpp
if (m == 1 && !transpose_a && transpose_b && lda == k && ldb == k) {
    dispatch_fused_int8_gemv(n, k, alpha, a, lda, b, ldb, c, ldc);
    return;
}
// Fall through to existing 3-kernel path...
```

## Correctness

### Translation tests (seq2seq): 100/100 PASS

| Test Suite | Result |
|-----------|--------|
| Translation (90 tests, 6 compute types × 15 configs) | **90/90 PASS** |
| INT8 translation (10 tests, greedy/beam/batch) | **10/10 PASS** |

### Generator (decoder-only): same behavior as old path

| Test | Fused GEMV | Old 3-kernel | Status |
|------|-----------|-------------|--------|
| Greedy (4 prompts, 50 tokens) | Matches old MPS path exactly | — | **PASS** (identical to old) |
| CPU vs MPS int8 (TinyLlama) | Diverges at token 5-25 | Same divergence points | **Pre-existing** |

The CPU/MPS divergence is pre-existing and caused by different accumulation methods between CPU RUY (int32) and MPS f32 GEMM. The fused GEMV actually uses int32 accumulation (matching RUY), but the divergence persists because the prefill step (m>1) still uses the f32 MPS GEMM path, introducing initial differences that compound during autoregressive decoding.

## Architecture

### Decode INT8 GEMM path (m=1, after optimization):

```
dispatch_int8_gemm(m=1, trans_b=true)
  → dispatch_fused_int8_gemv()
    → 1 MSL kernel: reads int8 A and B, accumulates int32, outputs int32
    → No temp buffers, no ObjC allocs
    → Encode-only (no commit)
```

### Prefill/non-decode INT8 GEMM path (m>1, unchanged):

```
dispatch_int8_gemm(m>1)
  → alloc 3 f32 temp buffers
  → encode int8_to_float32(A)       // GPU kernel
  → encode int8_to_float32(B)       // GPU kernel
  → encode MPS f32 GEMM             // AMX hardware
  → encode float32_to_int32(C)      // GPU kernel
  → release temps (CB retains)
```

## Files Modified

| File | Changes |
|------|---------|
| `src/metal/primitives_gemm.mm` | Added `kFusedInt8GemvMSL` kernel, PSO cache, `dispatch_fused_int8_gemv()`, routing in `dispatch_int8_gemm()` |

## Key Technical Decisions

1. **int32 accumulation over f32**: For int8×int8 with k=2048, max sum = 127×127×2048 = 33M, well within int32 range (2.1B). int32 accumulation is exact, while f32 loses precision after k≈1040 terms. This actually improves accuracy over the old path.

2. **One thread per output (no reduction)**: For GEMV (m=1), each thread independently computes one dot product. No threadgroup memory or reduction needed — simpler and efficient for the n=2048 output width typical in LLMs.

3. **char4 vectorization**: MSL char4 loads 4 bytes per instruction, aligning with Metal's 4-byte minimum load width and improving ALU utilization for the int8 multiply-accumulate.

4. **Contiguous layout guard**: The fused kernel only fires when lda==k and ldb==k (row-major contiguous). Non-contiguous strided layouts fall back to the 3-kernel path where the strided conversion kernels handle padding.

5. **Prefill fallback**: m>1 stays on MPS f32 GEMM (AMX hardware), which is optimal for larger matrix multiplies. The fused kernel is specifically designed for the decode bottleneck (m=1, bandwidth-bound).
