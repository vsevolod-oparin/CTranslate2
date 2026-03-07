# M11.7 — Batched MPS GEMM for Non-Padded Attention

## Summary

Added `dispatch_mps_gemm_batched<T>()` that encodes all batch elements (e.g. multi-head attention heads) in a single `MPSMatrixMultiplication` call using MPS batch matrix descriptors, instead of looping per-element. Applied to both FP32 and FP16 non-padded paths in `gemm_batch_strided`.

## Context

Multi-head attention computes `QK^T` and `attn*V` per head. With 8 heads (whisper-base) or 20 heads (whisper-large-v3-turbo), the previous code encoded one MPS GEMM per head. While each is encode-only (no sync), the per-element ObjC overhead of creating `MPSMatrix`, `MPSMatrixMultiplication`, and encoding to the command buffer adds up.

MPS provides a batch API via `matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:` that describes the full batch in one descriptor, and `MPSMatrixMultiplication.batchSize` that processes all elements in a single encode call.

## Changes

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `dispatch_mps_gemm_batched<T>()` template function; updated `gemm_batch_strided` FP32/FP16 paths to try batched first, fallback to per-element |

## Implementation

### `dispatch_mps_gemm_batched<T>()`

```cpp
template <typename T>
static bool dispatch_mps_gemm_batched(
    bool transpose_a, bool transpose_b,
    dim_t m, dim_t n, dim_t k, float alpha,
    const T* a, dim_t lda, dim_t stridea,
    const T* b, dim_t ldb, dim_t strideb,
    float beta,
    T* c, dim_t ldc, dim_t stridec,
    dim_t batch_size);
```

- Returns `true` if successfully dispatched, `false` if MPS batch constraints not met
- Uses batch matrix descriptors with `matrixBytes` = stride between batch elements
- Single `MPSMatrixMultiplication` with `batchSize` set encodes all heads in one call
- Entirely encode-only (no `commit_and_wait`)

### MPS Batch Constraints

MPS requires for each matrix:
- `matrixBytes % rowBytes == 0`
- `matrixBytes >= rows * rowBytes`

When these constraints aren't met (e.g. non-contiguous batch elements), falls back to the per-element loop.

### Routing in `gemm_batch_strided`

```
gemm_batch_strided (FP32/FP16, no padding needed):
  ├── Try dispatch_mps_gemm_batched<T>()  →  single batched MPS encode
  └── Fallback: per-element dispatch_mps_gemm<T>() loop
```

## Test Results

All e2e tests pass:
- 90/90 translation
- 39/39 beam search
- 13/13 whisper
- 12/12 GPT-2
- 9/9 float16
- 11/11 long-form
- 13/13 BF16
