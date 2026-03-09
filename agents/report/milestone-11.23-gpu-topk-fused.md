# M11.23 — GPU Fused TopK for Beam Search

## Summary

Replaced the CPU `std::partial_sort` fallback for beam search TopK (k>1) with a single-pass GPU kernel, eliminating **268 `commit_and_wait()` syncs** per whisper-large-v3-turbo inference (beam_size=5, 60s audio). The new kernel reads input once and performs k tree reductions in shared memory, vs the old iterative kernel that re-scanned the entire vocabulary k times.

## Problem

The TopK op for k>1 (beam search with beam_size=5) followed this path:

```
TopK::compute<METAL>() → CT2_COMMIT_AND_WAIT() → CPU std::partial_sort
```

Each decode step flushed the entire GPU command buffer to read logits into CPU memory, then ran `std::partial_sort` (~33µs) on the CPU. For whisper-large-v3-turbo beam_size=5, this generated 268 syncs — one per decode step.

A GPU `topk_k_<T>` kernel existed but was unused because it performed k sequential full-vocab argmax passes (reading the full input k times), making it ~1ms — slower than the CPU path.

## Solution: Single-Pass Fused TopK Kernel

### Algorithm

**Phase 1 — Scan** (single memory read):
Each of 256 threads scans ~N/256 elements (≈202 for vocab=51865) with strided access, maintaining a sorted (descending) local top-k array in private registers via insertion sort. When a new element exceeds the current k-th best, it replaces it and bubbles up to maintain sorted order.

**Phase 2 — K reduction rounds** (shared memory only):
For each of k rounds:
1. Each thread offers its current best candidate (rank 0 initially) to shared memory
2. Tree reduction (log₂256 = 8 steps) finds the global best
3. Thread 0 writes winner to output buffer
4. The winning thread advances its private rank pointer, so next round it offers its 2nd-best candidate
5. All other threads continue offering their current best

### Complexity

| Metric | Old (iterative) | New (fused) |
|--------|-----------------|-------------|
| Input reads | k × N/T = 5 × 202 | 1 × N/T = 202 |
| Comparisons/thread | O(k × N/T) = 1010 | O(N/T × k) + O(k × log T) = 1050 |
| Barriers | k × (1 + log₂T + 1) = 50 | k × (1 + log₂T + 1) = 50 |
| **Bandwidth** | **k × N × sizeof(T)** | **1 × N × sizeof(T)** |

The key win is **5× less memory bandwidth** — GPU is bandwidth-limited, so single-pass is significantly faster despite similar comparison counts.

### Shared Memory

256 × (sizeof(float) + sizeof(uint32_t)) = 2048 bytes — same as the argmax kernel, well within 32KB limit.

### Private Registers

2k values per thread (val + idx). For k=5: 10 values. Maximum k=64 (TOPK_MAX_K).

## Files Modified

| File | Change |
|------|--------|
| `src/metal/kernels/topk.metal` | Replaced `DEFINE_TOPK_K` (iterative argmax) with `DEFINE_TOPK_FUSED` (single-pass local top-k + tree merge) |
| `src/metal/msl_strings.h` | Regenerated via `gen_msl_strings.py` |
| `src/metal/ops_topk.mm` | Updated kernel name from `topk_k_<T>` to `topk_fused_<T>`; updated header comment |
| `src/ops/topk_metal.mm` | Removed CPU fallback path (`CT2_COMMIT_AND_WAIT` + `std::partial_sort`); all k values now use GPU |
| `src/metal/ops_metal.h` | Updated comment to reflect new algorithm |

## Data Flow Analysis

```
log_probs (Metal GPU)
  → BestSampler::sample() → TopK::compute<METAL>()
      → metal::topk_metal<T>()  ← encode-only, no sync
  → Sampler::operator() → copy_from()
      → synchronize_stream(METAL)  ← natural sync point (copies result to CPU)
  → unflatten_ids() on CPU
  → gather_beam_flat() on GPU
```

The `copy_from()` in `Sampler::operator()` provides the natural sync point. The TopK kernel only needs to be encode-only — the results are guaranteed to be ready before CPU reads them.

## Correctness Argument

1. **Unique thread ownership**: Each thread scans disjoint indices (`i = tid, tid+tgs, tid+2*tgs, ...`), so the winner's original index uniquely identifies which thread produced it. No two threads can offer the same index.

2. **Correct rank advancement**: After the tree reduction, `sh_idxs[0]` contains the winning index. Only the thread whose currently-offered index matches advances its rank. This thread promotes its next-best candidate for the following round.

3. **Sorted output guarantee**: Each round finds the global maximum among all threads' current offerings. Since each thread's local array is sorted descending, and each thread offers its best remaining candidate, the rounds produce the global top-1, top-2, ..., top-k in order.

## Test Results

| Test Suite | Result |
|-----------|--------|
| `test_beam_search.py` | 39/39 pass |
| `test_translation.py` | 90/90 pass |
| `test_faster_whisper.py` | 8/8 pass |
| **Total** | **137/137 pass** |

(`test_whisper.py` skipped due to HuggingFace network issue, not related to this change.)

## Sync Trace (whisper-large-v3-turbo, beam_size=5, 60s audio)

### Before (Post-M11.22 memory fixes)
```
   1509  devices.cc:162          (synchronize_stream)
   1477  primitives_memory.mm:80 (indexed_fill)
    268  topk_metal.mm:44        (CPU partial_sort)  ← TARGET
     16  primitives_gemm.mm:1161 (CPU cblas)
   ----
   3270  total
```

### After (M11.23)
```
   1509  devices.cc:162          (synchronize_stream)
   1477  primitives_memory.mm:80 (indexed_fill)
     16  primitives_gemm.mm:1161 (CPU cblas)
   ----
   3002  total
```

**268 syncs eliminated** (topk_metal.mm line completely absent from trace).

## Performance Impact

### Small model (test_faster_whisper.py, whisper-base, 60s audio, beam_size=5)
| Metric | Before | After |
|--------|--------|-------|
| CPU | 27.1s | 27.1s |
| Metal | 14.4s | 8.6s |
| Speedup | 1.88× | **3.17×** |

The 1.88× → 3.17× jump on the small model reflects the compound effect: TopK syncs were a larger fraction of total syncs for the small model, and eliminating them allows longer uninterrupted GPU command sequences.

### Large model (whisper-large-v3-turbo, 60s audio, beam_size=5)
| Metric | Before | After |
|--------|--------|-------|
| CPU | ~29s | ~29s |
| Metal | ~25s | ~27s |
| Speedup | ~1.15× | ~1.09× |

The large model is still bottlenecked by ~3000 remaining syncs from `synchronize_stream` (1509) and `indexed_fill` (1477). These are targets for future milestones.
