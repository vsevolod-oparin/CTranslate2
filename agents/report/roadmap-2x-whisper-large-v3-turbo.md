# Roadmap: 2x Metal Speed for whisper-large-v3-turbo

## Current State (Post-M16, 2026-03-15)

- **Target**: whisper-large-v3-turbo, beam_size=5, FLEURS 6-language benchmark
- **CPU f32**: RTF 0.488 | **Metal f16**: RTF 0.172 | **Metal Flash f16**: RTF 0.154
- **Speedup**: Metal f16 = **2.83x CPU** | Metal Flash f16 = **3.17x CPU**
- **vs competitors**: 1.71x faster than whisper.cpp F16, 1.26x faster than mlx-whisper f16

**The original 2x target has been achieved and exceeded.**

### Performance Evolution

| State | Metal Time | CPU Time | Speedup |
|-------|-----------|----------|---------|
| Post-M11.23 (original roadmap) | ~27s | ~29s | 1.09x |
| Post-M12 (allocator + sync elim) | — | — | ~1.8x (est.) |
| Post-M12.19 (FlashMHA fusion) | — | — | ~2.8x (f16) |
| Post-M16 (current, FLEURS) | RTF 0.154 | RTF 0.488 | **3.17x** |

### Sync Elimination Progress (cumulative)

| Milestone | What | Syncs Eliminated |
|-----------|------|-----------------|
| M11.20 ✅ | GPU Multinomial Sampling | 756 → 0 |
| M11.21 ✅ | Gather Sync Elimination | 264 → 0 |
| M11.22 ✅ | Metal Memory Management (leak fix) | N/A |
| M11.23 ✅ | GPU Fused TopK | 268 → 0 |
| M11.25 ✅ | GPU Indexed Fill Kernel | CPU scatter → GPU kernel |
| M11.26 ✅ | Padded GEMM → MPS Routing | 548 CPU cblas syncs → 0 |
| M11.27 ✅ | MPS GEMM Object Cache | Alloc overhead eliminated |
| M11.28 ✅ | Indexed Fill F16 Sync Elimination | Pre-sync removed (f16 only) |
| M12.1 ✅ | Bucketed Allocator | Pool 30GB→2.5GB, commits 188→90 |
| M12.10 ✅ | INT8 protect_buffer Sync Elimination | Commits 3500→97 |
| M12.19 ✅ | FlashMHA Commit Optimization | Fused SDPA + GPU blit |
| M12.21 ✅ | Fused INT8 GEMV | int8 3.1→34.3 (11.1x) |
| M13 ✅ | Float16 Custom MSL GEMM (f32 accum) | BLEU 22.48→25.74 |
| M14.5 ✅ | Logits Processor Sync Fix | Correctness fix (added sync) |
| M16 ✅ | Conditional Sync Skip | Skip decoder sync when no logits processors |

**Current syncs per decode step (f16)**: ~1 (sampler copy_from only, unavoidable for beam search)

## Original Roadmap Items — What Was Done

### Completed

| Item | How It Was Done |
|------|----------------|
| **M11.25** — Batch indexed_fill | GPU kernel implemented (M11.25). F16 pre-sync eliminated (M11.28). F32 retains pre-sync due to MPS ordering issue. |
| **M11.28** — MPS Object Caching | GEMM PSO cache and SDPA GEMM cache. Keyed by (transpose, m, n, k, alpha, beta, batch_size). 50-200 entries typical. |
| **M11.24** — Reduce synchronize_stream | Not done as a single audit, but the goal was achieved through targeted work across M11.25–M11.28, M12.1, M12.10, M16. Syncs reduced from ~3,002 to ~1/step (f16). |

### Additional Completed Work (beyond original roadmap)

| Milestone | Impact |
|-----------|--------|
| M12.1 Bucketed Allocator | Pool convergence, eliminated allocation-related syncs |
| M12.19 FlashMHA Fusion | Fused SDPA + GPU blit: f32 5x, f16 1.28x |
| M12.21 Fused INT8 GEMV | int8 decode 11.1x faster |
| M13 Float16 GEMM Fix | Custom MSL with f32 accum, BLEU recovery |
| M14.4 Beam Search Tuning | beam=6 + length_penalty=0.6 closes f16 quality gap |
| M14.5 Logits Sync Fix | Correctness fix for repetition_penalty/no_repeat_ngram |
| M16 Whisper Benchmark | Full competitive benchmark, WER validation |

### Not Yet Done (still viable)

These items were not pursued because the 2x target was met through other work. They remain valid optimization directions for future gains:

| Item | Est. Savings | Why Not Yet |
|------|-------------|-------------|
| **M11.26** — Persistent Command Encoder | 2–5% | Modest savings; not blocking at 3.17x |
| **M11.27** — Fused LayerNorm+Linear | 2–5% | BF16-only via MPSGraph exists; custom f32 MSL not yet needed |
| **M11.29** — Decode Pipeline Fusion | 10–20% | Largest remaining compute win, but high complexity |
| **M11.30** — Cross-Attention KV Reuse | ~1% | Cross-attention already cached efficiently |

## Remaining Opportunities (Beyond 3x)

The 2x target is met. Further optimization could target 4x+ CPU:

| Opportunity | Est. Savings | Effort | Priority |
|-------------|-------------|--------|----------|
| Decode Pipeline Fusion (M11.29) | 10–20% | High | MEDIUM |
| GPU-side EOS Check | 5–10% (eliminate last sync) | High | LOW |
| Speculative Decoding | 30–50% (architectural) | Very High | MEDIUM |
| Persistent Command Encoder | 2–5% | Low | LOW |
| Fused LN+Linear | 2–5% | Medium | LOW |
| Larger SIMD GEMM Tiles (32x32+) | 5–10% | Medium | LOW |

## Attempted and Rejected Optimizations (M12.13–M12.17)

These were tried during M12 and found to be ineffective or counterproductive on Apple M4:

| Attempt | Result | Why |
|---------|--------|-----|
| M12.13 GEMV float4 vectorization | 20% slower | Memory-bound; wider loads didn't help |
| M12.14 Bookkeeping optimization | Regression | Overhead already negligible (0.4% CPU) |
| M12.15 BiasAdd fusion | Noise | Within measurement error |
| M12.16 Pooling optimization | Noise | Within measurement error |
| M12.17 GPU RoPE for decode | Dead code, no impact | sq=1 decode too small for GPU benefit |

## Conclusion

The roadmap's original 2x target (**Metal ~14s, CPU ~29s**) has been exceeded. Metal Flash f16 achieves **3.17x CPU** on the FLEURS benchmark. The primary drivers were:

1. **Sync elimination** (M11.20–M11.28, M12.1, M12.10): reduced commits from ~3,000 to ~1/step
2. **FlashMHA fusion** (M12.19): fused SDPA + GPU blit for dramatic decode speedup
3. **Allocator bucketing** (M12.1): eliminated pool bloat and allocation-related syncs
4. **Precision fixes** (M13, M14): f32-accumulate GEMM + beam search tuning restored quality

Further gains would require architectural changes (speculative decoding, pipeline fusion) with diminishing returns relative to effort.
