# M14.2–14.3: Precision Experiments — SDPA, Elementwise, Beam Search

**Date:** 2026-03-13
**Goal:** Close the MPS f16 BLEU gap (25.74 vs f32 27.65 = 1.91 gap).
**Outcome:** Gap is NOT from compute precision. Root cause identified as f16 weight quantization + beam search degeneration.

---

## Experiments Conducted

### 14.2: SDPA GEMM f32 Accumulation
**Change:** `ops_sdpa.mm` — route f16 SDPA GEMMs through f32-accumulation path (half→f32→MPS f32 GEMM→f32→half) using per-thread `SdpaF16TempCache`.

**Result:** No BLEU change (25.71 standard, 25.70 flash).

**Why:** Standard MHA (`flash_attention=False`, the default) uses `dot_product_attention()` → `ops::MatMul` → main GEMM dispatch, which already has f32 accumulation from M13. SDPA code only runs with `flash_attention=True`. Flash path K=64 was already precise enough.

### 14.3: Elementwise/Broadcast f32 Promotion for Half
**Change:** `kernels/elementwise.metal` and `kernels/broadcast.metal` — half add/sub/mul now promote operands to f32: `c[gid] = (T)((float)a[gid] op (float)b[gid])`.

**Result:** No BLEU change (25.64 ±0.1). Performance unchanged (1160 tok/s vs 1148 baseline).

**Why:** Consistent with CUDA evidence — CUDA f16 elementwise also runs in native f16 and has zero BLEU loss. The precision of individual elementwise ops is not the bottleneck.

**Kept:** Yes — zero overhead (memory-bound ops), sound practice, may help other models.

### Additional: Beam Score f32 Accumulation
**Change:** `decoding.cc` — `topk_scores` always f32, promote `log_probs` to f32 before beam score accumulation and sampling.

**Result:** Marginal improvement (+0.1 BLEU to 25.83) but **15% throughput loss** (977 vs 1148 tok/s) from converting the [batch×beam × vocab] log_probs tensor to f32 every step.

**Reverted:** Overhead not justified by marginal improvement.

### Additional: Full f32 GEMM (Threshold=0)
**Change:** `primitives_gemm.mm` — set `kF16DirectThreshold = 0` to force ALL f16 GEMMs through the promoted path (f32×f32→f32 instead of f16×f16→f32 SIMD kernel).

**Result:** No BLEU change (25.75). 17% throughput loss.

**Reverted:** No benefit, worse performance.

---

## Root Cause Analysis: Beam Search Degeneration

### Key Discovery: The Gap Only Exists with Beam Search

| Beam Size | f16 BLEU | f32 BLEU | Gap | Notes |
|-----------|----------|----------|-----|-------|
| 1 (greedy) | 19.48 | 18.50 | **f16 better by 0.98** | No precision issue |
| 2 | 18.80 | 26.27 | f32 better by 7.47 | f16 DROPS below greedy! |
| 4 | 25.73 | 27.65 | f32 better by 1.92 | Standard benchmark |
| 4 + no_repeat_3gram | 26.27 | 27.57 | f32 better by 1.30 | Repetition accounts for ~0.6 |

**Critical observation:** With greedy decoding, f16 actually OUTPERFORMS f32. The BLEU gap is entirely a beam search phenomenon.

### Per-Sentence Analysis

Compared all 2737 sentences between f16 and f32 (beam=4):
- **980 divergent** (35.8% of sentences)
- **f32 wins 548**, f16 wins 319, tie 113

**Worst f16 failures** (examples):

| Sentence | f16 BLEU | f32 BLEU | f16 Output | Issue |
|----------|----------|----------|------------|-------|
| #2640 | 0.3 | 84.3 | "Johanna Rah Rah Rah Rah..." | **Repetition loop** — score -0.096 (higher than f32's -0.170!) |
| #1968 | 4.2 | 60.3 | Garbled translation | Score -1.509 vs f32's -0.341 |
| #287 | 2.8 | 53.3 | Garbled with repetition | Score -1.768 vs f32's -0.312 |
| #315 | 2.4 | 52.8 | Garbled translation | Score -1.556 vs f32's -0.350 |

The repetition loops are most damaging — the model assigns HIGHER probability to degenerate output than to correct translations.

### Mechanism

1. **f16 weight quantization** (10-bit mantissa vs f32's 23-bit) creates subtly different probability distributions
2. These differences are individually tiny but **compound across decoder layers** (6 layers × multi-head attention × feed-forward)
3. **Beam search amplifies** the differences by selecting the highest-scoring hypotheses across beams
4. For some sentences, the f16 probability landscape has "degenerate attractors" — repetition patterns that become self-reinforcing and score higher than correct translations
5. **Greedy decoding** doesn't trigger this because it makes local decisions without cross-beam comparison

### Why CUDA Has Zero Gap

CUDA f16 also uses f16 weights and f16 elementwise but has zero BLEU loss. This is likely because:
- cuBLAS tensor core FMA uses different rounding/accumulation order than MPS GEMM
- These tiny numerical differences cause beam search to explore different trajectories
- CUDA's specific numerical behavior happens to not trigger degeneration for this model
- This is a **butterfly effect** — identical mathematical operations with different implementation produce different beam search paths

### Implications

The ~2 BLEU gap is **inherent to the specific numerical behavior** of the MPS backend with this model. It cannot be closed by improving compute precision (already all f32 intermediate). Potential mitigations:
1. **no_repeat_ngram_size**: Reduces gap from 1.92 to 1.30 (0.6 BLEU recovery)
2. **repetition_penalty**: Untestable — GPU page fault bug in current Metal implementation
3. **Model-specific**: Different models may have different sensitivity to f16 quantization

---

## Summary of Changes Kept

| File | Change | Impact |
|------|--------|--------|
| `src/metal/kernels/elementwise.metal` | Half add/sub/mul promote to f32 | Zero overhead |
| `src/metal/kernels/broadcast.metal` | Half broadcast add/mul promote to f32 | Zero overhead |
| `src/metal/msl_strings.h` | Auto-regenerated from .metal files | — |
| `src/metal/ops_sdpa.mm` | SDPA f16 GEMM f32 accumulation | Flash attention only |

## Summary of Changes Reverted

| Change | Why Reverted |
|--------|-------------|
| Beam score f32 accumulation (`decoding.cc`) | 15% throughput loss, +0.1 BLEU only |
| GEMM threshold=0 (`primitives_gemm.mm`) | 17% throughput loss, no BLEU change |

---

## Recommended Next Steps

1. **Investigate beam search degeneration mechanism** — understand WHY f16 probability distributions trigger repetition loops
2. **Test repetition_penalty** — fix the GPU page fault bug first, then test if it closes the gap
3. **Cross-model validation** — check if other models (Whisper, TinyLlama) have the same f16 BLEU gap pattern
4. **Consider f16-specific beam search heuristics** — e.g., automatic no_repeat_ngram_size when compute_type=float16
