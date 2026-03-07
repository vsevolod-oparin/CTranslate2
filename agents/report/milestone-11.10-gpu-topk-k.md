# M11.10 — GPU TopK Kernel for k>1

## Summary

Implemented a GPU MSL kernel for TopK with k>1 using iterative argmax with an excluded-index list. The kernel works correctly (30/30 standalone tests) but **benchmarking showed CPU `std::partial_sort` is faster** for typical beam search shapes (k=5, depth=51865: GPU ~1.5 ms vs CPU ~33 µs).

**Decision:** The GPU kernel exists in the codebase (`topk_k_<T>` in `topk.metal`) but production code retains the CPU fallback for k>1. The k=1 GPU argmax path (from M11.6) remains unchanged.

## Motivation

Beam search with k>1 previously used `commit_and_wait()` + CPU `std::partial_sort`. The hypothesis was that a GPU parallel reduction could avoid the CB submission overhead (~400 µs) and execute faster on the GPU.

## Algorithm

The GPU TopK kernel (`topk_k_<T>`) uses iterative argmax with an excluded-index list:
- `TOPK_MAX_K = 64` (compile-time limit)
- One threadgroup of 256 threads per batch row
- For each iteration `iter` in `[0, k)`:
  1. Each thread scans a strided slice of the row, skipping indices in `excluded[0..iter-1]`
  2. Tree reduction in shared memory finds the global max value and index
  3. Thread 0 writes result to output and records `excluded[iter] = found_index`
  4. `threadgroup_barrier(mem_threadgroup)` before next iteration
- All comparisons in float32 (cast from T)

## Why CPU Wins

For k=5, depth=51865 (Whisper vocab):
- **GPU**: 5 sequential full-vocab scans with barriers ~1.5 ms
- **CPU**: `std::partial_sort` on unified memory ~33 µs (O(n log k), single pass)

The GPU kernel's k sequential barrier-synchronized passes over the full vocabulary cannot compete with CPU's cache-friendly single-pass partial sort. The CB overhead (~400 µs) that the GPU kernel was meant to avoid is already much less than the kernel's execution time.

A more competitive GPU approach would need bitonic sort or radix-based selection, but the complexity is not justified given that beam search already has many CPU-bound operations (hypothesis management, score sorting).

## Changes

| File | Change |
|------|--------|
| `src/metal/kernels/topk.metal` | Added `DEFINE_TOPK_K(T)` macro — iterative argmax with excluded-index list (float/half/bfloat) |
| `src/metal/ops_topk.mm` | Added `dispatch_topk_k()` function; `topk_metal<T>()` updated to accept `k` parameter and route k==1 to argmax, k>1 to topk_k |
| `src/metal/ops_metal.h` | Updated `topk_metal` declaration with `dim_t k = 1` parameter |
| `src/ops/topk_metal.mm` | k=1: GPU argmax (unchanged); k>1: CPU `partial_sort` fallback (retained — GPU kernel slower) |
| `src/metal/msl_strings.h` | Regenerated with topk_k kernel |
| `tests/metal/topk_test.mm` | Extended from 12 to 30 tests: k=1 argmax + k>1 top-k for f32/f16/bf16 at various depths |
| `tests/metal/topk_bench.mm` | **NEW**: GPU TopK vs CPU partial_sort benchmark |

## Benchmark Results (Apple M4)

| Shape | k | GPU TopK | CPU partial_sort | Ratio |
|-------|---|----------|------------------|-------|
| [1, 51865] | 5 | ~1500 µs | ~33 µs | 0.02x |
| [1, 51865] | 1 | ~15 µs (argmax) | ~45 µs | 3.0x |

The k=1 GPU argmax path remains a clear win (encode-only, no CB overhead). The k>1 kernel is retained in the codebase for potential future use with different workloads (smaller vocabularies, larger k) but is not used in production.

## Production Path Summary

```
TopK::compute<Device::METAL>:
  k == 1 → metal::topk_metal<T>(batch, depth)     [GPU argmax, encode-only]
  k > 1  → CT2_COMMIT_AND_WAIT() + CPU partial_sort  [CPU fallback]
```

## Test Results

| Test Suite | Count | Result |
|---|---|---|
| Standalone TopK (`topk_test.mm`) | 30/30 | PASS |
| Translation | 90/90 | PASS |
| Beam search | 39/39 | PASS |
| Whisper | 13/13 | PASS |
| Faster-whisper | 8/8 | PASS |
