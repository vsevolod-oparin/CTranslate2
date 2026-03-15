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
| **M11.26** — Persistent Command Encoder | **< 1%** (revised down from 2–5%) | Evaluated 2026-03-15: only 41 custom kernel sites affected (~80µs/step); MPS GEMMs manage own encoders |
| **M11.27** — Fused LayerNorm+Linear (f32/f16) | 1–2% | BF16-only exists (`fused_norm_and_project`); f32/f16 skipped because MPS GEMM is already encode-only |
| **M11.29** — Decode Pipeline Fusion | **2–5%** (revised down from 10–20%) | See below — most fusion already implemented |
| **M11.30** — Cross-Attention KV Reuse | ~1% | Cross-attention already cached efficiently |

### M11.29 — Decode Pipeline Fusion: Revised Assessment

Code review (2026-03-15) reveals that most fusion from the original M11.29 proposal is **already implemented** through other milestones:

| Fusion Target | Status | Where |
|---------------|--------|-------|
| Q/K/V → single GEMM | **Done** | `flash_attention.cc:43-49` — `_linear[0]` produces `fused_proj`, then Split |
| SDPA decode (Q@K + softmax + S@V) | **Done** | `ops_sdpa.mm:702-766` — `fused_sdpa_decode` MSL kernel, single dispatch per (batch,head) |
| LayerNorm + Linear | **Done (BF16)** | `common.cc:469-544` — `fused_norm_and_project`; returns false for f32/f16 |
| Output projection + residual | **Done** | `flash_attention.cc:141` — `_linear.back(context, output, &queries)` |
| FFN Linear2 + residual | **Done** | `transformer.cc:50` — `_ff2(inner, output, &input)` |

**What remains unfused:**

1. **LayerNorm + Linear for f32/f16** — Currently skipped because f32/f16 MPS GEMM is encode-only (no sync overhead to amortize). The fused kernel only helps BF16 where MPSGraph is synchronous.

2. **FFN Linear1 + activation** — Activation applied as a separate GPU dispatch after the GEMM. A fused GEMM+activation kernel would save one encoder creation (~10µs) and one intermediate buffer write/read.

**Revised estimate**: 2–5% (down from original 10–20%), since the major fusion targets (QKV GEMM, SDPA decode, output+residual) are already done. All remaining f32/f16 ops are encode-only and deferred into a single command buffer, so further fusion primarily saves encoder creation overhead (~10µs each) and intermediate buffer traffic.

## Remaining Opportunities (Beyond 3x)

The 2x target is met. Further optimization could target 4x+ CPU:

| Opportunity | Est. Savings | Effort | Priority | Notes |
|-------------|-------------|--------|----------|-------|
| GPU-side EOS Check | 5–10% (eliminate last sync) | High | LOW | Sampler is 33% of f16 decode time |
| Speculative Decoding (self) | **10–15%** (revised) | Very High | LOW | See detailed assessment below |
| Cross-Attention Fused Decode | 2–5% | Medium | LOW | Extend `fused_sdpa_decode` to cross-attn (batch=1 only) |
| FFN Linear+Activation Fusion | 1–3% | Low | LOW | Save one encoder + one intermediate buffer per layer |
| LayerNorm+Linear for f32/f16 | 1–2% | Medium | LOW | f32/f16 MPS GEMM already encode-only; minimal benefit |

### Speculative Decoding: Detailed Assessment (2026-03-15)

**Original estimate**: 30–50%. **Revised**: **10–15%** for whisper-large-v3-turbo.

The 30–50% estimate assumed a GPU-bound workload with many decoder layers. Whisper-turbo has only 4 decoder layers and the sampler sync (33% of decode time) is unaffected by speculative decoding.

**How it works**: Run a cheap "draft" N steps ahead, then verify all N tokens with the full model in one forward pass. Accepted tokens are free; rejected tokens trigger rollback.

**Best approach for this model**: Self-speculative with early exit — use layers 0–1 as draft (~2x faster/step), verify with all 4 layers. No external draft model needed.

**Why the gain is limited for whisper-turbo:**
- Only 4 decoder layers → draft (2 layers) is only ~2x faster, not 10x
- Sampler is 33% of decode time → unaffected by speculative decoding
- Net: ~15–25% speedup on decoder_call (66% of loop) = **~10–15% total**
- Would push 3.17x CPU → ~3.5–3.6x

**Five architecture blockers in current code:**

| Blocker | Current State | What's Needed |
|---------|--------------|---------------|
| **Multi-token forward** | Decoder takes 1 token/step; full-sequence path exists but not wired into decode loop | Extend full-sequence path to work mid-generation with existing KV cache |
| **KV cache rollback** | `DecoderState` grows monotonically; `update_state()` is destructive gather | Add `checkpoint()` / `restore()` (deep copy ~100KB at seq=50) |
| **Position encoding** | Uses single `step` parameter | Support position range `[step, step+N-1]` in one call |
| **Logits processors** | Read `alive_seq` which is only updated after sampling | Temporarily append draft tokens, remove on rollback |
| **Beam search interaction** | All beams advance uniformly per step | Speculative + beam search is an open research problem |

**Implementation phases:**

| Phase | Scope | Effort | Gain |
|-------|-------|--------|------|
| 1. Self-speculative (early exit) | KV checkpoint/restore, partial-layer forward, verify loop | ~2 weeks | 10–15% |
| 2. Multi-token verify | Wire full-sequence decoder into mid-generation verify | ~1 week | Enables higher N (more tokens per verify) |
| 3. External draft model | Dual model loading, shared vocab, stochastic acceptance | ~2 weeks | Better for larger models (32-layer whisper-large-v3) |

**Verdict**: High complexity, moderate reward for whisper-turbo specifically. Would benefit whisper-large-v3 (32 decoder layers) much more. Not recommended as next step given the effort-to-gain ratio.

### Evaluated and Downgraded

| Opportunity | Original Est. | Revised Est. | Why |
|-------------|--------------|-------------|-----|
| **Speculative Decoding** | 30–50% | **10–15%** | Whisper-turbo has only 4 decoder layers; sampler sync (33%) unaffected; self-speculative draft (2 layers) only ~2x faster |
| **Persistent Command Encoder** | 2–5% | **< 1%** | 41 custom kernel sites × ~1-2µs each = ~80µs/step. MPS GEMMs (the dominant cost) manage their own encoders internally — unaffected. Against 8.6ms/step, this is < 1%. |
| **Larger SIMD GEMM Tiles** | 5–10% | **N/A** | Apple Silicon `simdgroup_matrix` is hardware-fixed at 8×8 — cannot be enlarged. The custom MSL GEMM kernel is not active in the main f16 path anyway (MPS float32 promotion is faster). |

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

Most incremental fusion opportunities (QKV projection, SDPA decode, output+residual) are already implemented. The remaining low-level fusions (FFN activation, LN+Linear for f32/f16) offer diminishing returns since all ops are already encode-only. Speculative decoding — the largest remaining opportunity — was evaluated and revised down to 10–15% for whisper-turbo (4 decoder layers, 33% sampler overhead unaffected) with very high implementation complexity. The decode pipeline is approaching its practical optimization ceiling for this model architecture.
