# M11.4 — Metal Profiling Integration

## Summary
Added GPU-native timing to CTranslate2's profiling system using Metal command buffer `GPUStartTime`/`GPUEndTime`. The existing `PROFILE` macro and `ScopeProfiler` now report both wall-clock time and GPU execution time for Metal operations.

## Key Design
- **Thread-local GPU time accumulator**: `_gpu_time_elapsed` in `utils.mm` accumulates `(GPUEndTime - GPUStartTime)` after each `commit_and_wait()`.
- **ScopeProfiler integration**: Constructor snapshots `gpu_time_elapsed()` before the scope; destructor computes the delta after `synchronize_stream()` flushes GPU work.
- **Backward compatible**: GPU-ms column only appears in dump output when any scope has non-zero GPU time (i.e., Metal device only). CUDA/CPU output unchanged.

## Files Modified
| File | Change |
|------|--------|
| `src/metal/utils.h` | Added `gpu_time_elapsed()`, `reset_gpu_time()` declarations |
| `src/metal/utils.mm` | Thread-local `_gpu_time_elapsed` accumulator in `commit_and_wait()` |
| `include/ctranslate2/profiler.h` | Added `_gpu_start` field to `ScopeProfiler` |
| `src/profiler.cc` | `gpu_time` in `ScopeProfile`; GPU timing in ctor/dtor; GPU column in dump |
| `tests/metal/e2e/test_profiling.py` | **NEW** — 7 tests: CLI profiling + Python API |
| `agents/report/milestone-11.4-profiling.md` | **NEW** — this report |

## Test Results (7/7 PASS)
- CLI: per-op names (Gemm, LayerNorm, SoftMax) present ✅
- CLI: GPU timing column (`ms(gpu)`) appears ✅
- CLI: 21 ops have non-zero GPU time ✅
- CLI: profiling table parses correctly (26 rows) ✅
- Python: `reset_gpu_time()` works ✅
- Python: GPU time functions callable, inference completes ✅
- Python: translation produces correct output with profiling build ✅

## Sample Output (opus-mt-en-de, beam_size=2, "Hello world")
```
 60.82%  60.82%  60.82% MatMul                                 1010.40ms  22.29ms(gpu)
 14.28%  27.67%  75.10% Gemm                                   237.24ms  53.16ms(gpu)
  8.20%  13.38%  83.31% BiasAdd                                136.27ms  3.35ms(gpu)
  3.97%   3.97%  87.28% LayerNorm                              65.96ms  2.15ms(gpu)
  2.54%   2.54%  96.41% SoftMax                                42.19ms  1.34ms(gpu)
```

## Insight: Wall vs GPU Time
The profiling reveals massive overhead from per-op `commit_and_wait()`:
- MatMul: 1010ms wall → 22ms GPU (98% overhead from per-op CB submission)
- Total GPU: ~84ms vs ~1680ms wall time

This confirms the M11.1 finding: amortizing command buffer submissions is critical for Metal performance. In production (without profiling), CB batching reduces this to 1 commit per decode step.
