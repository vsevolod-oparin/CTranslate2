# Roadmap: 2x Metal Speed for whisper-large-v3-turbo

## Current State (Post-M11.23)

- **Target**: whisper-large-v3-turbo, beam_size=5, 60s audio, float32
- **CPU**: ~29s | **Metal**: ~27s | **Speedup**: ~1.09x
- **Remaining syncs**: 3,002

| Source | Syncs | Description |
|--------|------:|-------------|
| `devices.cc` (synchronize_stream) | 1,509 | Gather clone hazard, cross-device copy, framework sync |
| `primitives_memory.mm` (indexed_fill) | 1,477 | `DisableTokens::apply()` CPU scatter |
| `primitives_gemm.mm` (CPU cblas) | 16 | Float16-only remainder |
| **Total** | **3,002** | |

At ~0.4ms per sync, the remaining syncs account for ~1.2s of pure overhead plus significant GPU pipeline bubbles (idle time between syncs).

### Completed Milestones

| Milestone | Syncs Eliminated | Status |
|-----------|-----------------|--------|
| M11.20 — GPU Multinomial Sampling | 756 → 0 | ✅ Done |
| M11.21 — Gather Sync Elimination | 264 → 0 | ✅ Done |
| M11.22 — Metal Memory Management | N/A (leak fix) | ✅ Done |
| M11.23 — GPU Fused TopK | 268 → 0 | ✅ Done |

## Goal

Metal **2.0x CPU** → target Metal time ~14s (from ~27s).

Need to save ~13s. Sync elimination alone (~1.2s direct + pipeline bubble reduction) won't reach 2x — also need compute overlap and reduced framework overhead.

## Phase 1: Remaining Sync Elimination (est. -2–4s)

### M11.24 — Reduce synchronize_stream Calls (1,509 → ~200)

**Problem**: `synchronize_stream()` called from data type conversions and framework-level operations. Many are unnecessary when both source and destination are Metal-allocated.

**Solution**: Audit all `synchronize_stream()` call sites. Categories:
1. **Type conversions** (float32↔float16): If both buffers are Metal, encode a GPU cast kernel instead of sync+CPU conversion
2. **StorageView copies**: If source data is already on GPU, skip sync
3. **Framework-level syncs**: Some are required (e.g., before CPU-side decisions), but many can be deferred

**Priority**: HIGH — largest remaining sync source. Requires careful auditing to avoid breaking correctness.

### M11.25 — Batch indexed_fill Across Decode Steps (1,477 → ~50)

**Problem**: GPU `indexed_fill` kernel is encode-only (M11.17), but `DisableTokens::apply()` is called multiple times per decode step with separate index lists. Each call may trigger a sync from the caller.

**Solution**: Batch all token disabling into a single GPU dispatch per decode step.
- Accumulate all disable indices, then dispatch once
- Requires restructuring `DisableTokens` to defer the actual fill

**Priority**: MEDIUM — the indexed_fill kernel itself is already encode-only. The remaining syncs come from callers that sync before reading logits. Needs investigation of exact call patterns.

## Phase 2: Compute Optimization (est. -2–4s)

### M11.26 — Persistent Command Encoder

**Problem**: ~15 compute encoder create/end cycles per decoder layer × 4 layers × ~200 decode steps = ~12,000 transitions at ~10µs each ≈ ~120ms.

**Solution**: Keep a single compute encoder open across multiple kernel dispatches where possible. Requires careful barrier management between kernels that read each other's output.

**Priority**: LOW — modest savings alone, but improves GPU utilization.

### M11.27 — Float32 Fused LayerNorm+Linear

**Problem**: LayerNorm and the following GEMM are separate dispatches with intermediate buffer materialization.

**Solution**: Fused MSL kernel that computes LayerNorm and feeds result directly to GEMM (or at least avoids writing/reading the intermediate buffer). Currently BF16-only via MPSGraph; extend to float32 with custom MSL.

**Priority**: LOW — saves ~100-200ms from reduced memory traffic.

### M11.28 — MPS Object Caching

**Problem**: `MPSMatrix` and `MPSMatrixMultiplication` objects recreated per GEMM call (~1-5µs each).

**Solution**: Cache MPS objects keyed by (rows, cols, dataType, rowBytes). Invalidate on shape change.

**Priority**: LOW — microseconds per call, thousands of calls, but total is only ~5-20ms.

## Phase 3: Architectural (est. -3–5s)

### M11.29 — Decode-Step Pipeline Fusion

**Problem**: Each decode step has ~15+ separate GPU dispatches (LayerNorm, GEMM×4, SDPA, Add, ...) with encoder transitions and intermediate buffers between them.

**Solution**: Fuse the entire decoder layer into fewer, larger GPU dispatches:
1. Fused Q/K/V projection (3 GEMMs → 1 concatenated GEMM)
2. Fused attention output projection + residual add
3. Fused FFN (Linear+ReLU+Linear+Add)

**Priority**: MEDIUM-HIGH — largest potential savings but highest complexity. Each fusion needs correctness verification.

### M11.30 — Cross-Attention KV Reuse Optimization

**Problem**: Cross-attention K/V are cached but the cache lookup and SDPA dispatch still have per-step overhead.

**Solution**: Since cross-attention K/V are constant across decode steps, pre-compute and cache the `MPSMatrix` objects for K/V. Possibly pre-factorize attention if beam patterns allow.

**Priority**: LOW — cross-attention is already efficient due to caching.

## Priority Summary

| Milestone | Est. Savings | Effort | Priority | Status |
|-----------|-------------|--------|----------|--------|
| M11.20 GPU Multinomial | -0.3s | Low | **HIGH** | ✅ Done |
| M11.21 Gather Sync Elimination | -0.1s | Low | **HIGH** | ✅ Done |
| M11.22 Metal Memory Management | perf fix | Medium | **CRITICAL** | ✅ Done |
| M11.23 GPU Fused TopK | -0.1s + compound | Medium | MEDIUM | ✅ Done |
| M11.24 Reduce sync_stream | -1–2s | Medium | **HIGH** | Planned |
| M11.25 Batch indexed_fill | -0.4s | Medium | MEDIUM | Planned |
| M11.29 Decode Pipeline Fusion | -3–5s | High | **MEDIUM-HIGH** | Planned |
| M11.26 Persistent Encoder | -0.1s | Low | LOW | Planned |
| M11.27 Fused LN+Linear | -0.2s | Medium | LOW | Planned |
| M11.28 MPS Object Cache | -0.02s | Low | LOW | Planned |

## Recommended Execution Order

1. ~~**M11.20** — GPU Multinomial~~ ✅
2. ~~**M11.21** — Gather sync elimination~~ ✅
3. ~~**M11.22** — Metal memory management~~ ✅
4. ~~**M11.23** — GPU Fused TopK~~ ✅
5. **M11.24** — synchronize_stream audit (biggest remaining sync source)
6. **M11.25** — Batch indexed_fill (cleanup remaining scatter syncs)
7. **M11.29** — Decode pipeline fusion (biggest compute win, do after syncs are minimized)

After Phase 1 (M11.24–M11.25), expected: ~1.5–2.0x CPU.
After Phase 3 (M11.29), expected: ~2.0–2.5x CPU.

## Non-Sync Analysis

Most remaining Metal overhead is sync-related. Non-sync optimizations (MPS object caching, encoder transitions, fused ops) would add ~5-10% improvement at most. The path to 2x is primarily:

1. Eliminate remaining ~3,000 syncs → longer uninterrupted GPU command sequences
2. Reduce pipeline bubbles → better GPU utilization from fewer sync points
3. Fuse decoder operations → fewer dispatches, less intermediate memory traffic
