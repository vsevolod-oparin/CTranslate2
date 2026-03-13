# M14.5: Logits Processor Sync Fix — MPS Driver Coherency

**Date:** 2026-03-14
**Goal:** Fix GPU page faults with `repetition_penalty` and `no_repeat_ngram_size` on Metal.
**Outcome:** Both bugs fixed. 20/20 tests pass across float32, float16, int8, int8_float16.

---

## Root Cause

**MPS driver coherency issue** between `MPSMatrixMultiplication` (framework ops) and custom
compute encoders within the same command buffer.

The decoder's final output GEMM uses `MPSMatrixMultiplication` (for K > 32 in the f16
promoted path). When logits processors (RepetitionPenalty's Gather + penalize_previous_tokens,
DisableTokens' indexed_fill) encode custom compute kernels that read/write the same logits
buffer in the same command buffer, the MPS driver doesn't guarantee data coherency.

This manifested as:
- `kIOGPUCommandBufferCallbackErrorPageFault` — penalize kernel reading stale/unmapped logits data
- `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored` — cascade from prior CB failure

### Trigger conditions
- 8+ diverse-length sentences in a batch (specific memory layout/allocation pattern)
- `beam_size=4` (32 active hypotheses = batch_size * beam_size)
- f16 compute type (uses MPSMatrixMultiplication via promoted path)
- Also affects f32 (MPSMatrixMultiplication used directly for all sizes)

---

## Fix

**`synchronize_stream(device)` after decoder call, before logits processing** in `decoding.cc`.

This commits the decoder's command buffer and waits for GPU completion, ensuring logits data
is fully coherent before any logits processor reads or writes it.

Applied to both code paths:
1. `beam_search()` — line ~607 (after decoder call, before DisableTokens + logits processors)
2. `greedy_search()` — line ~1004 (after decoder call, before DisableTokens + logits processors)

### Why this position and not inside RepetitionPenalty::apply()

Placing `commit_and_wait()` inside `RepetitionPenalty::apply()` (before Gather) fails because
the GPU error has already been triggered in a prior command buffer. The `prepare_length_mask`
kernel (called during decoder execution) does a `commit_command_buffer()` (non-blocking CB split),
and the decoder's subsequent MPS operations in the NEW CB after that split are the ones
that cause the coherency issue. By the time `RepetitionPenalty::apply()` runs, the faulty
CB has already been committed (during decoder execution), and `commit_and_wait()` sees
"Ignored (for causing prior/excessive GPU errors)".

The fix must come between the decoder call completing and any custom compute encoder
accessing the logits buffer.

---

## Performance Impact

`synchronize_stream(device)` adds one `commit_and_wait()` per decode step. This is ~0.4ms
per call (measured in M12.4 profiling). For a typical 10-step beam search decode, this adds
~4ms total — negligible compared to the ~800ms total decode time.

This sync was effectively already happening in most pipelines (e.g., sampler does
`synchronize_stream` internally). The new sync ensures correctness when logits processors
are used before the sampler.

---

## Files Modified

- `src/decoding.cc` — Added `synchronize_stream(device)` after decoder call (beam_search + greedy_search)
- `src/decoding_utils.cc` — Removed redundant `commit_and_wait()` from RepetitionPenalty::apply()
- `src/metal/utils.h` — Added `non_blocking_commit()` C++ interface (retained for future use)
- `src/metal/utils.mm` — Added `non_blocking_commit()` implementation

## Tests

- `tests/metal/e2e/test_m14_5_logits_processor_sync.py` — 20/20 pass
  - 4 compute types × 5 test cases (rep_penalty, no_repeat_ngram, combined, greedy, 5x stress)
- `tests/metal/e2e/test_beam_search.py` — 39/39 pass (no regression)
