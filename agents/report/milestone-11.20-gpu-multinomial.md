# M11.20 — GPU Multinomial Sampling

## Summary

Replaced CPU-side `std::discrete_distribution` with an encode-only GPU kernel for multinomial sampling (`sample_size == 1`). This eliminates all `commit_and_wait()` syncs from the Multinomial op during Whisper and translation inference.

## Problem

The Metal `Multinomial::compute()` called `CT2_COMMIT_AND_WAIT()` to flush pending GPU writes before reading softmax probabilities on CPU via `std::discrete_distribution`. For whisper-large-v3-turbo with `beam_size=1` (sampling mode), this caused ~756 syncs — one per decode step per batch element.

## Solution

### MSL Kernel

Inline MSL kernel `multinomial_float` / `multinomial_half` in `multinomial_metal.mm`:

**Algorithm** (for `sample_size == 1`):
1. **Phase 1**: Each thread sums its chunk of the probability vector (strided access: thread `tid` handles indices `tid, tid+tgs, tid+2*tgs, ...`).
2. **Phase 2**: Thread 0 computes exclusive prefix sum of per-thread partial sums in shared memory. Also scales the random threshold by total probability sum (handles unnormalized distributions).
3. **Phase 3**: Each thread re-scans its chunk with the cumulative offset. The first thread whose running sum exceeds the scaled threshold records its index.
4. **Phase 4**: Thread 0 finds the minimum winning index across all threads (break ties by smallest index, matching `std::discrete_distribution` semantics).

- One threadgroup per batch element, 256 threads per threadgroup
- Float32 accumulation (even for half inputs) for numerical stability
- Shared memory: 2 × 1024 × sizeof(float) + 1024 × sizeof(int) ≈ 12KB (well within 32KB limit)

### Host Dispatch

- Random values generated on CPU via `std::uniform_real_distribution<float>(0.0f, 1.0f)` using the existing `get_random_generator()` mt19937
- Random buffer passed to GPU as a small `MTLBuffer` (batch_size × 4 bytes)
- Input/output buffers protected via `protect_buffer_by_base()` (O(1) lookup)
- Encode-only: no `commit_and_wait()`

### Fallback

- `sample_size > 1`: Falls back to CPU path (only used by GumbelMax, not Multinomial in practice)
- `bfloat16_t` input: Falls back to CPU path (BF16 MSL kernel not needed — BF16 inference doesn't use sampling)

## Sync Trace

### beam_size=1 (sampling mode, whisper-large-v3-turbo)

| Source | Before M11.20 | After M11.20 |
|--------|-------------:|-------------:|
| `multinomial_metal.mm` | 756 | **0** |

### beam_size=5 (beam search mode)

| Source | Before M11.20 | After M11.20 |
|--------|-------------:|-------------:|
| `multinomial_metal.mm` | 0* | 0 |

*Beam search uses TopK, not Multinomial.

## Files Modified

| File | Change |
|------|--------|
| `src/ops/multinomial_metal.mm` | GPU kernel + dispatch for `sample_size == 1`; CPU fallback for other cases |

## Test Results

All 150 e2e tests pass:

| Suite | Result |
|-------|--------|
| beam_search | 39/39 PASS |
| translation | 90/90 PASS |
| whisper | 13/13 PASS |
| faster_whisper | 8/8 PASS |

### Sampling Correctness

Verified GPU multinomial produces varied, valid results across 10 sampling runs with `sampling_topk=10, sampling_temperature=0.8` — different translations each run, all valid German tokens from opus-mt-en-de vocabulary.

## Performance

Speed measurements are affected by thermal throttling from consecutive heavy workloads in this session. The primary impact is sync elimination:

- **756 fewer syncs** in the sampling path (beam_size=1)
- At ~0.4ms per sync, estimated direct savings: ~302ms per inference
- Additional savings from reduced GPU pipeline bubbles (fewer stall points)

## Remaining Sync Sources (beam_size=5)

| Source | Syncs | Status |
|--------|------:|--------|
| `devices.cc:162` (synchronize_stream) | 2,536 | M11.21 target |
| `primitives_memory.mm:80` (indexed_fill) | 2,508 | M11.23 target |
| `topk_metal.mm:44` (CPU sort) | 268 | M11.22 target |
| `primitives_gemm.mm` (cblas) | 16 | Irreducible |
| **Total** | **5,328** | |

## Architecture Notes

- The kernel uses strided access (thread `tid` reads every `tgs`-th element) rather than contiguous chunks. This simplifies the prefix-sum logic and ensures coalesced memory access across threads within a warp/SIMD group.
- The two-pass approach (sum + scan) avoids needing `class_size` floats in shared memory. Only `tgs` floats are needed for the prefix sum.
- Random number generation remains on CPU (mt19937) for reproducibility and seed control. Only the threshold value is passed to GPU — no GPU RNG state management needed.
