# M12 Plan v2 — Revised After Performance Analysis

**Date**: 2026-03-11
**Status**: M12.1-12.4 complete. **Superseded by M12.4 findings** — see section 3a below.
**Model**: OPUS-MT En→De (d_model=512, 6 layers, base Transformer)
**Hardware**: Apple M4, macOS 15

> **NOTE**: Section 2 ("Revised Bottleneck Analysis") in this document was written before M12.4
> decode loop profiling and contains **incorrect estimates** (e.g. "beam bookkeeping 300-400ms").
> M12.4 measured beam bookkeeping at **0.1ms** (0.0%). See `milestone-12.4-decode-loop-profiling.md`
> for the correct analysis and the updated plan in `APPLE_M4_METAL_PLAN.md` M12 section.

---

## 1. What We Learned (M12.1-12.3)

### Original plan estimates vs reality

| Item | Original Estimate | Actual Result |
|------|-------------------|---------------|
| M12.1 encode_barrier | GPU util 40%→55-60%, tok/s +20-40% | Commits 188→90 (f16), **tok/s 1286→1500 (+17%)**, GPU util unchanged at 42% |
| M12.2 ObjC reduction | 10-20% wall-time reduction | **<0.5%** — total ObjC overhead is 2.7ms/batch (0.18%) |
| M12.3 buffer lookup | "MEDIUM" impact | **0%** — O(log n) already ~70ns/lookup; 50.7% cache hit saves ~75ms in noise |
| Cumulative target | f16 1,006→1,500-1,800 tok/s | **f16 ~1,500 tok/s** (hit low end, but from M12.1 alone) |
| GPU utilization target | 65% | **42%** (unchanged from baseline) |

### The original analysis was wrong about the bottleneck

The original M12 plan attributed CPU overhead to:
- Commit sync overhead (correct — fixed by M12.1)
- ObjC autorelease/alloc cycles (wrong — 0.18% of total)
- Buffer lookup linear scan (wrong — already O(log n), not O(n))
- "10-20% from ObjC reduction" (wrong — measured at <0.5%)

### Where the time actually goes (100 sentences, f16)

| Component | Time (ms) | % of wall | Status |
|-----------|-----------|-----------|--------|
| **GPU compute** | 742 | 41.5% | Actual useful work |
| **Commit sync overhead** | 54 | 3.0% | Fixed by M12.1 (was ~115ms) |
| **Non-commit CPU overhead** | **991** | **55.5%** | **Dominant bottleneck** |
| Total wall | 1787 | 100% | |

The ~991ms of non-commit CPU overhead is **dtype-independent** (f32: 1088ms vs f16: 1045ms), which means it's not in the GEMM path at all. It's in the **autoregressive decode loop infrastructure**.

### Commit trace (post M12.1)

| Source | f16 commits | f32 commits | Notes |
|--------|-------------|-------------|-------|
| `devices.cc:162` (sampler sync) | 136 | 144 | **Only remaining sync** — required |
| `primitives_memory.mm:114` (indexed_fill) | 0 | 4 | f32-only pre-sync |
| `primitives_beam_search.mm` | 0 | 0 | **Eliminated by M12.1** |

All syncs are now at the theoretical minimum: 1 per decode step (sampler needs token IDs on CPU for next step).

---

## 2. Revised Bottleneck Analysis

### 2.1 The ~1000ms non-commit CPU overhead

This is the new primary bottleneck. Estimated breakdown:

| Source | Est. time (ms) | Evidence | Can optimize? |
|--------|----------------|----------|---------------|
| **Beam search bookkeeping** | 300-400 | CPU sort/gather/state mgmt per step, O(beam×vocab) | Hard — algorithmic |
| **Decoder loop C++ overhead** | 200-300 | Layer iteration, tensor metadata, op dispatch | Medium — fuse ops |
| **Encoder (one-time)** | 100-150 | ~100ms for 32-sentence batch encode | No — amortized |
| **Sampler CPU-side work** | 50-100 | Token extraction, next-step prep | Small |
| **Python→C++ boundary** | 20-50 | Tokenization, batch management | Small |
| **Memory allocator** | 20-50 | Bucketed pool, already optimized | Done |
| **ObjC/buffer lookup** | <5 | Measured: negligible | Done |

### 2.2 Why GPU utilization is stuck at ~42%

GPU utilization = GPU_time / wall_time. Even with zero sync overhead, GPU util would be:
- f16: 742 / (742 + 991) = **42.8%** — the CPU overhead floor
- f32: 1267 / (1267 + 1088) = **53.8%**

The GPU finishes its work and sits idle while the CPU does beam management, tensor bookkeeping, and loop iteration. **This is a pipeline bubble problem, not a sync problem.**

### 2.3 Small model penalty

OPUS-MT (d_model=512) has tiny GEMMs (m=1-32, k=512, n=512). Each GEMM takes ~5-20µs on GPU. With 6 layers × ~6 GEMMs per step = 36 GEMMs × 15µs = ~0.5ms of GPU compute per decode step. But the CPU overhead per step (beam management, tensor ops, dispatch) is ~5-10ms. The GPU:CPU ratio is fundamentally unfavorable.

For comparison, Whisper large-v3-turbo (d_model=1280, 32 layers) achieves 22.1x speedup over CPU because GEMMs are ~10x larger.

---

## 3. Revised Plan

### Completed (keep as-is)

| # | Item | Status | Impact |
|---|------|--------|--------|
| 12.1 | encode_barrier + bucketed allocator | **Done** | +17% f16 tok/s, commits halved |
| 12.2 | ObjC overhead investigation | **Done** | <0.5% — not a bottleneck |
| 12.3 | Pointer cache in buffer lookup | **Done** | 50.7% hit rate, no measurable wall-time gain |

### Completed

| # | Item | Status | Impact |
|---|------|--------|--------|
| 12.1 | encode_barrier + bucketed allocator | **Done** | +17% f16 tok/s, commits halved |
| 12.2 | ObjC overhead investigation | **Done** | <0.5% — not a bottleneck |
| 12.3 | Pointer cache in buffer lookup | **Done** | 50.7% hit rate, no measurable wall-time gain |
| 12.4 | Decode loop profiling | **Done** | CPU overhead is 0.4%, not 58% — see below |

### Dropped (after M12.4 findings)

| # | Item | Why dropped |
|---|------|-------------|
| ~~12.7 Op fusion~~ | ~~Fuse LayerNorm+add, etc.~~ | GPU encode time is <2ms/step. Savings <1%. |
| ~~12.8 Pipeline overlap~~ | ~~Overlap GPU step N with CPU step N-1~~ | CPU work per step is 0.04ms. Nothing to overlap. |

---

## 3a. M12.4 Decode Loop Profiling — Corrected Analysis

**M12.4 overturned the "~1000ms CPU overhead" theory.** The decode loop is already near-optimal.

Actual breakdown (f16, 62 steps):
- decoder_call: 401ms (50%) — transformer forward pass (all GPU compute)
- sampler: 398ms (50%) — TopK + `synchronize_stream()` (mandatory sync)
- ALL other (beam bookkeep + gather + state update + overhead): **3ms (0.4%)**

The "CPU overhead" was an artifact of measuring `wall_time - GPU_time`. The GPU timer only
records committed execution time, not the interleaved encode/wait pattern in autoregressive decode.

**Three bottleneck profiles by compute type:**

| Category | Types | decoder_call % | Root cause |
|----------|-------|---------------|------------|
| GPU-balanced | f16 | 50% | GPU compute + sync, pipeline optimal |
| Sync-heavy | f32 | 43% | f32 GEMM 1.5× slower, longer sync waits |
| Decoder-dominated | int8, int8_f16, bf16, int8_bf16 | 98-99.8% | Internal syncs inside decoder forward pass |

Per-step cost:
- f16: 12.9ms (GPU compute is the clock)
- f32: 18.0ms (GPU compute + 18ms logits_process anomaly)
- int8: 46.5ms (2 syncs/GEMM × 36 GEMMs = 29ms of commit_and_wait alone)
- bf16: 424.5ms (synchronous MPSGraph per GEMM, ~11ms each)

See `milestone-12.4-decode-loop-profiling.md` for full details.

---

## 4. Recommended Next Steps (Priority Order — updated after M12.4)

1. **M12.5** — BF16→FP16 auto-promotion. 424ms→6.5ms/step = **65× improvement**. Low effort.
2. **M12.6** — INT8 GPU dequantize. 45.6ms→~13ms/step = **3.5× improvement**. Medium effort.
3. **M12.8** — Larger model benchmarks. Validates f16/f32 scale with model size.
4. f32 indexed_fill pre-sync fix — 18ms saving for f32 (1.2% of loop time)

---

## 5. Updated Performance Summary

### Current state (post M12.1-12.3, 50 sentences, best-of-3)

| Type | tok/s | vs CPU (826) | Commits | GPU% | Bottleneck |
|------|-------|-------------|---------|------|-----------|
| **float16** | **1476** | **1.79x** | 90 | 41% | CPU decode loop overhead |
| **float32** | **1019** | **1.23x** | 96 | 54% | CPU decode loop overhead |
| **int8** | 84 | 0.10x | 5692 | 15% | CPU int8↔f32 conversion |
| **int8_float16** | 86 | 0.10x | 5580 | 15% | CPU int8↔f32 conversion |
| **bfloat16** | 9 | 0.01x | 2844 | 1% | Synchronous MPSGraph |
| **int8_bfloat16** | 9 | 0.01x | 6904 | 2% | Both INT8 + MPSGraph |

### Optimization runway remaining

| Target | Current | Achievable | Approach |
|--------|---------|------------|----------|
| f16 tok/s | 1476 | ~1800-2000 | M12.7/12.8 (decode loop optimization) |
| f32 tok/s | 1019 | ~1200-1400 | Same as f16 |
| bf16 tok/s | 9 | ~1400 | M12.5 (auto-promote to f16) |
| int8 tok/s | 84 | ~500-1000 | M12.6 (GPU dequantize) |

---

## 6. Key Insight: This Is a Small-Model Problem

For OPUS-MT (d_model=512), the GPU:CPU compute ratio is unfavorable. Each decode step does ~0.5ms of GPU work but ~5-10ms of CPU bookkeeping. No amount of GPU-side optimization can fix this — the CPU is the bottleneck.

For larger models (d_model≥1024, 24+ layers), GPU compute per step scales quadratically while CPU overhead stays constant. The M12.1 sync elimination matters much more for large models.

**Recommendation**: Before investing in complex optimizations (op fusion, pipeline overlap), benchmark a larger model to confirm the GPU backend scales as expected. If it does, the OPUS-MT numbers are acceptable and optimization effort should focus on BF16/INT8 correctness fixes (M12.5/12.6) rather than squeezing more from f16/f32.
