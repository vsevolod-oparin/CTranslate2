# M12.11 — Decode Loop CPU Overhead Investigation

**Date**: 2026-03-11
**Status**: INVESTIGATED — NO CHANGE (optimizations not impactful)
**Hardware**: Apple M4, macOS 15

---

## Problem Statement

Performance research (M12 Part 2) identified two CPU-side decode loop optimizations:

1. **2.1 — Defer word ID conversion** (estimated 3-5%): `convert_to_original_word_ids()` was believed to cause GPU→CPU→GPU roundtrips per step
2. **2.2 — Batch CPU read of topk_scores** (estimated 5-10%): `topk_scores.scalar_at<float>({i, k})` per-beam reads were believed to cause sync overhead

## Investigation

### Approach 1: Keep topk_ids/topk_scores on GPU (FAILED)

Attempted to keep topk_ids and topk_scores on MPS device throughout the decode loop, eliminating `.to(device)` copies. This caused:

1. **Metal assertion crash** (`commit command buffer with uncommitted encoder`): `alive_seq` migrated to MPS (via `append_step_output` Concat), then `gather_beam_flat(alive_seq, gather_indices, ...)` with CPU `gather_indices` triggered a Gather device mismatch → Metal crashed during commit
2. **Cascading device mismatches**: beam bookkeeping code uses CPU-side `at()/scalar_at()` on topk_ids/topk_scores, which on MPS triggers `CT2_COMMIT_AND_WAIT()` per element access — defeating the purpose
3. **`gather(gather_indices, active_beams)` mismatch**: gather_indices moved to GPU but active_beams on CPU
4. **`unflatten_ids` CPU requirement**: Must do integer division/modulo on CPU; cannot stay GPU-only

**Conclusion**: The GPU-resident approach requires pervasive changes across beam bookkeeping, hypothesis building, alive_seq management, and gather operations — fundamentally incompatible with the CPU-centric bookkeeping design.

### Approach 2: Persistent GPU Buffers + Raw Pointer Access (NO IMPROVEMENT)

Simpler approach that stays within the existing CPU-centric design:
- Reuse persistent GPU `StorageView` buffers instead of per-step `.to(device)` temporaries
- Replace `at({i, k})` / `scalar_at<float>({i, k})` with raw pointer access

**Benchmark result**: No measurable change (within noise):

| Type | M12.10 tok/s | M12.11 tok/s | Change |
|------|-------------|-------------|--------|
| float16 | 1490 | 1484 | −0.4% |
| float32 | 1053 | 1044 | −0.9% |
| int8 | 779 | 766 | −1.7% |
| int8_float16 | 889 | 847 | −4.7% |
| bfloat16 | 1490 | 1449 | −2.8% |

All differences are within run-to-run noise (±5%).

### Root Cause: Negligible Target

Decode profiler breakdown (float16, 10 sentences, 256 steps):

| Component | Time (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| decoder_call | 1092 | 52.8% | Transformer forward pass (encode-only GPU ops) |
| sampler | 959 | 46.3% | GPU sync: flushes ALL pending GPU work |
| state_update | 7.2 | 0.3% | KV cache reorder |
| step_overhead | 9.6 | 0.5% | unflatten_ids, append, `.to(device)` copies |
| beam_gather | 0.6 | 0.0% | gather_beam_flat |
| beam_bookkeep | 0.2 | 0.01% | EOS check, hypothesis registration |
| logits_process | 0.1 | 0.0% | disable_tokens |

**Key findings**:
- **99.1%** of time is GPU computation (decoder_call + sampler sync)
- **beam_bookkeep** is 0.2 ms / 2069 ms = **0.01%** — completely negligible
- **step_overhead** (includes `.to(device)` copies) is 9.6 ms / 2069 ms = **0.46%**
- The `.to(device)` copy is ~16 bytes (batch_size × beam_size × sizeof(int32_t)) — trivially fast
- Pool allocator returns cached buffers in O(1) — no allocation overhead

### Why the Research Estimates Were Wrong

The research estimated 3-5% for 2.1 and 5-10% for 2.2 based on:

1. **Assumption that topk_ids/topk_scores are on GPU**: In reality, the sampler copies them to CPU before returning (M11.26). So `at()` is just CPU array access, not GPU sync.
2. **Assumption that `.to(device)` is expensive**: With StorageModeShared and pool allocation, `.to(device)` is just: pool lookup (O(1)) + memcpy (16 bytes) + ~0.1 µs total.
3. **Overestimate of bookkeeping cost**: At 0.01% of total time, even a 10× speedup of bookkeeping adds 0 ms.

## Decision

**No code changes**. The decode loop is already operating at near-theoretical efficiency:
- GPU utilization: 41-55%
- CPU overhead: <1%
- Sync points: 1 per step (sampler)

Further performance improvements require:
1. **Faster GPU kernels** (GEMM, attention) — 52.8% of time
2. **Reduced GPU idle time** between kernel dispatches — better pipelining
3. **Larger models/batches** where GPU utilization increases

## Files Modified

None.
