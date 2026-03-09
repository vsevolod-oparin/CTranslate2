# M11.19 — Float16 m=1 Custom GEMV Kernel (Encode-Only)

## Summary

Eliminated ~8,700 `commit_and_wait()` syncs from whisper-large-v3-turbo float16 inference by replacing the CPU cblas fallback for m=1 decode attention GEMMs with a custom MSL GEMV kernel. The kernel runs encode-only (zero syncs) with `protect_buffer` preventing buffer-reuse crashes.

## Problem

After M11.18, float32 m=1 GEMMs route through encode-only MPS padded path (zero syncs). But float16 m=1 still used `batch_cpu_gemm_f16` (1 sync per call) because MPS batched GEMM produces garbled output for float16 m=1 (MPS bug, verified in M11.18). For whisper-large-v3-turbo default inference (float16), this caused ~8,700 syncs — the dominant remaining bottleneck.

## Solution

### Custom MSL GEMV Kernel

A custom `gemv_half` MSL kernel avoids MPS entirely — it performs its own dot-product computation, so it is unaffected by the MPS float16 batched GEMM bug.

**Kernel design:**
- One threadgroup per batch element, 256 threads per threadgroup
- Each thread computes `ceil(N/256)` output columns
- Float32 accumulation (avoids half-precision rounding in inner loop), cast to half on write
- Supports both `transpose_b=true` (QK^T) and `transpose_b=false` (scores×V)
- Handles alpha/beta scaling

**Dispatch:**
- Encode-only: encodes into `get_current_command_buffer()`, no `commit_and_wait()`
- `protect_buffer(a)`, `protect_buffer(b)`, `protect_buffer(c)` after encoding prevents buffer-reuse crashes (same mechanism as M11.18 float32 path)

### Integration

Added m=1 intercept at the **top** of the float16 branch in `gemm_batch_strided`, before the `needs_padding()` check:

```cpp
if (m == 1 && batch_size > 0) {
    dispatch_gemv_f16_batched(transpose_b, n, k, alpha, beta,
                              a, lda, stridea, b, ldb, strideb,
                              c, ldc, stridec, batch_size);
    return;
}
```

This intercepts ALL m=1 float16 batch GEMMs — both padding and non-padding cases.

### Why This Works Now (But Didn't Before)

M11.18's report listed "Custom GEMV MSL kernel (encode-only)" as a failed approach because it crashed due to buffer-lifetime issues. But M11.18 then introduced `protect_buffer` to solve exactly this problem for the float32 MPS padded path. The same mechanism works for the custom GEMV kernel.

## Changes

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Inline `kGemvF16MSL` kernel, `get_gemv_f16_library()`/`get_gemv_f16_pso()`, `dispatch_gemv_f16_batched()`, m=1 intercept in float16 branch |

## Results

### Sync Count (whisper-large-v3-turbo, beam_size=5, 60s audio, float16)

| Source | Before (M11.18) | After (M11.19) | Delta |
|--------|-----------------|----------------|-------|
| `primitives_gemm.mm` (cblas) | ~8,700 | **16** | **~-8,684** |
| `devices.cc` (synchronize_stream) | ~1,078 | 651 | -427 |
| `primitives_memory.mm` (indexed_fill) | ~1,024 | 600 | -424 |
| `multinomial_metal.mm` (sampling) | ~756 | 332 | -424 |
| `topk_metal.mm` (CPU sort) | 268 | 268 | 0 |
| **Total** | **~11,826** | **1,867** | **~-9,959** |

The additional reductions in devices.cc, primitives_memory.mm, and multinomial are consistent with M11.18's observation — fewer syncs reduce command buffer contention.

### Performance (whisper-large-v3-turbo, beam_size=5, 60s audio)

| Metric | Before (float16) | After (float16) |
|--------|-------------------|-----------------|
| GEMM syncs | ~8,700 | 16 |
| Metal speed | ~0.55x CPU | **0.70–1.16x CPU** |

Note: High variance (0.70x–1.16x) from thermal throttling on Apple M4. Both CPU and Metal times vary ±30% between runs. The sync elimination is the definitive metric.

### whisper-base (13 tests, beam_size=5, 30s audio)

| Metric | Before | After |
|--------|--------|-------|
| Metal/CPU ratio | ~1.98x | **3.67x** |

### Test Results

| Test | Result |
|------|--------|
| `test_beam_search.py` | 39/39 PASS |
| `test_translation.py` | 90/90 PASS |
| `test_whisper.py` | 13/13 PASS |
| `test_faster_whisper.py` | 8/8 PASS |

## Architecture

### GEMV Kernel Flow

```
gemm_batch_strided (float16, m=1)
  → dispatch_gemv_f16_batched
    → metal_buffer_for_ptr(A/B/C) → get offsets
    → setBuffer(A,B,C) + setBytes(K,N,strides,ldb,alpha,beta,tb)
    → dispatchThreadgroups([batch_size,1,1], [256,1,1])
    → protect_buffer(A), protect_buffer(B), protect_buffer(C)
    → return (encode-only, no sync)
```

### Kernel Details

```metal
kernel void gemv_half(...)
{
    // One threadgroup per batch element
    // Each thread handles ceil(N/TGS) output columns
    for (uint j = tid; j < N; j += tgs) {
        float acc = 0.0f;  // float32 accumulation
        if (tb) {
            // B transposed: row-major access for each output column
            for (uint i = 0; i < K; ++i)
                acc += (float)a_row[i] * (float)b_row[i];
        } else {
            // B not transposed: column-strided access
            for (uint i = 0; i < K; ++i)
                acc += (float)a_row[i] * (float)b_mat[i * ldb + j];
        }
        c_row[j] = (half)(alpha * acc + beta * old_c);
    }
}
```

## Remaining Sync Sources (float16 path)

| Source | Count | Notes |
|--------|-------|-------|
| `devices.cc` (synchronize_stream) | 651 | Framework sync, type conversions |
| `primitives_memory.mm` (indexed_fill) | 600 | DisableTokens first apply() |
| `multinomial_metal.mm` (CPU sampling) | 332 | std::discrete_distribution on CPU |
| `topk_metal.mm` (CPU sort) | 268 | std::partial_sort for k>1 beam search |
| `primitives_gemm.mm` (cblas) | 16 | Non-m=1 float16 padded GEMMs |
