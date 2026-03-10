# M11.28 — Indexed Fill Pre-Sync Elimination (Float16)

## Summary

Eliminated the `CT2_COMMIT_AND_WAIT()` pre-sync in `indexed_fill` for float16 inference, cutting per-step syncs from 2 to 1 — a **50% reduction in sync count** for the f16 decode loop. Float32 retains the pre-sync due to an unresolved MPS driver ordering issue.

## Problem

After M11.25-27, the decode loop had 2 syncs per step:

```
Per decode step (before):
  GPU: [layer ops, MPS GEMM] → COMMIT_AND_WAIT (indexed_fill pre-sync)
                                → [indexed_fill GPU kernel] → [more ops]
                                → COMMIT_AND_WAIT (sampler copy_from)
                                → CPU reads tokens
```

The indexed_fill pre-sync accounted for 38% of CPU time in the comprehensive profiler (M11.4 analysis). It was originally required when indexed_fill was a CPU scatter loop (needed GPU data flushed to shared memory before CPU reads). After M11.25 moved indexed_fill to a GPU kernel, the pre-sync was preserved because float32 inference produced garbage without it.

### Re-Investigation

The hypothesis: since the GPU indexed_fill kernel and the prior MPS GEMM are now both encoded into the **same command buffer**, Metal's in-order execution guarantee should make the pre-sync unnecessary. The logits buffer is written by MPS GEMM (GPU) and read by indexed_fill (GPU) — no CPU involvement in the data path. The indices buffer is CPU-written into a StorageModeShared buffer, which is coherent on Apple Silicon unified memory.

### Test Results Without Pre-Sync

| Type | Model | Test | Result |
|------|-------|------|--------|
| float16 | whisper-large-v3-turbo | test_faster_whisper.py | 8/8 PASS |
| float16 | whisper-large-v3-turbo | sync trace | 0 indexed_fill syncs |
| float32 | whisper-base | test_whisper.py | **5/13 FAIL** |
| float32 | whisper-base | batch output | Metal 7 chars vs CPU 110 chars — garbage |
| mixed | opus-mt-en-de | test_beam_search.py | 39/39 PASS |
| mixed | opus-mt-en-de | test_translation.py | 90/90 PASS |

Float32 whisper fails: single inference (Metal 31 chars vs CPU 671 chars), timestamps, and batch modes all produce truncated/garbage output. The translation model (opus-mt-en-de) passes, suggesting the issue is specific to whisper's float32 GEMM → indexed_fill interaction.

### Root Cause (Unresolved)

The float32 failure is reproducible and specific. Possible explanations:

1. **MPS float32 GEMM internal implementation**: May use different encoder patterns than float16, with weaker ordering guarantees against subsequent compute encoders within the same command buffer.
2. **Accumulator precision effects**: Float32 GEMMs may produce results with different memory coherence timing than float16 on the MPS hardware path.
3. **Metal driver bug**: The MPS framework may not properly enforce execution ordering between its internal encoders and subsequent compute encoders for float32 matrices.

Without access to MPS internals or a Metal GPU debugger showing per-encoder execution, the exact cause cannot be determined.

## Solution: Type-Conditional Pre-Sync

```cpp
metal::protect_buffer(indices);
if constexpr (!std::is_same_v<T, ctranslate2::float16_t>) {
  CT2_COMMIT_AND_WAIT();
}
```

- **float16**: No pre-sync. Indexed_fill encodes into the same command buffer as prior work. Zero syncs.
- **float32/int/other**: Pre-sync preserved. Prior GPU work committed and completed before indexed_fill.

The `if constexpr` ensures zero overhead for the f16 path (the branch is eliminated at compile time).

### Pipeline: Before vs After (Float16)

```
Before (2 syncs/step):
  GPU: [layer ops] → SYNC → [indexed_fill] → [more ops] → SYNC → CPU
                     ↑ bubble                               ↑ necessary

After (1 sync/step):
  GPU: [layer ops → indexed_fill → more ops] → SYNC → CPU
       ↑ all encode-only, uninterrupted        ↑ necessary
```

## Sync Trace

### Float16 (whisper-large-v3-turbo, beam_size=5, raw API, 112 tokens)

| Source | Before | After | Change |
|--------|--------|-------|--------|
| `devices.cc:162` (synchronize_stream) | ~112 | 112 | 0 |
| `primitives_memory.mm:95` (indexed_fill pre-sync) | ~112 | **0** | **-112 (100%)** |
| Other (tile, beam_search) | ~2 | 2 | 0 |
| **Total** | **~226** | **114** | **-112 (50%)** |

Per-step syncs reduced from 2 to 1.

### Float32 (whisper-base, beam_size=5, raw API, 98 tokens)

| Source | Before | After | Change |
|--------|--------|-------|--------|
| `devices.cc:162` | 101 | 101 | 0 |
| `primitives_memory.mm:102` (indexed_fill pre-sync) | 100 | 100 | 0 |
| Other | 1 | 1 | 0 |
| **Total** | **202** | **202** | **0** |

Float32 unchanged — pre-sync preserved.

## Files Modified

| File | Change |
|------|--------|
| `src/metal/primitives_memory.mm` | Type-conditional pre-sync: skip for float16, keep for others |

## Test Results

All tests pass with correct output:

| Test | Result | Details |
|------|--------|---------|
| `test_whisper.py` (f32, whisper-base) | 13/13 PASS | WER 0.00%, exact match |
| `test_faster_whisper.py` (f16, whisper-large-v3-turbo) | 8/8 PASS | Output comparable to CPU |
| `test_beam_search.py` | 39/39 PASS | All beam sizes correct |
| `test_translation.py` | 90/90 PASS | All combinations |

### Performance

Raw CTranslate2 API benchmarks (Apple M4, 60s audio, beam_size=5):

| Model | Type | Mean (ms) | CV | Speedup vs CPU |
|-------|------|-----------|----|---------------|
| whisper-large-v3-turbo | f16 | 1700 | 0.5% | — (no CPU f16 baseline) |
| whisper-base | f32 | 753 | 7.5% | 3.05x |

`test_whisper.py` (f32): CPU 6059ms, Metal 2017ms → **3.00x** (was 2.77x before M11.27/28).

`test_faster_whisper.py` (f16): varies 2.37-4.68x due to faster_whisper seek retry non-determinism (see M11.27 variance analysis). Raw API timing is stable.

## Why 50% Sync Reduction Matters

Each `commit_and_wait()` creates a GPU pipeline bubble:

1. **Submit**: CPU commits command buffer to GPU
2. **Wait**: CPU blocks until GPU finishes (~0.4ms minimum CB overhead)
3. **Resume**: CPU wakes up, sets up next operations, encodes new GPU work
4. **Gap**: GPU sits idle during step 3 until new work arrives

With 2 syncs/step, there are 2 bubbles. The indexed_fill bubble was particularly damaging because it fell in the **middle of the forward pass** — between the logits GEMM and the sampling operations. Removing it allows the GPU to run the entire forward pass + indexed_fill + sampling setup as one uninterrupted sequence, with only a single sync at the end when the CPU must read the sampled tokens.

## Remaining Sync Sources (Float16)

| Sync | Count/Step | Source | Avoidable? |
|------|-----------|--------|------------|
| sampler copy_from | 1 | `devices.cc:162` via `sampling.cc` | No — CPU must read sampled tokens |

**1 sync per step is the practical minimum** without architectural changes (GPU-side EOS check, speculative decoding).

## Future Work

1. **Investigate float32 pre-sync requirement**: The MPS float32 GEMM ordering issue remains unresolved. Options:
   - `MTLFence` between MPS encoder and compute encoder (if MPS exposes its encoder boundaries)
   - `[buffer didModifyRange:]` on the indices buffer (lighter than full commit_and_wait)
   - File a Feedback with Apple about MPS float32 encoder ordering guarantees
2. **GPU-side EOS check**: Move end-of-sequence detection to GPU to eliminate the sampler sync entirely. Would require a GPU kernel that checks if all beams have terminated, avoiding the CPU roundtrip.
3. **Speculative decoding**: Submit N decode steps in one command buffer, syncing once for N steps instead of N times. Requires predicting which tokens will be generated.
