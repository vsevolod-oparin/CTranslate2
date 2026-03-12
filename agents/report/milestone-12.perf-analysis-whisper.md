# Performance Analysis — Whisper (Encoder-Decoder) on Metal

**Date**: 2026-03-12
**Hardware**: Apple M4 (10-core GPU, 16 GB unified memory), macOS 15
**Branch**: `metal-backend`
**Models**: whisper-large-v3 (32+32), whisper-large-v3-turbo (32+4), OPUS-MT En→De (6+6)

---

## Executive Summary

| Finding | Impact | Status |
|---------|--------|--------|
| **MPS FP16 is 1.8× faster than CPU** for OPUS-MT, **6.4× for whisper-large-v3** | Production-ready | Achieved |
| **99.1% of decode time is GPU compute** — CPU overhead is negligible | No CPU optimization opportunity | Confirmed |
| **Speculative decoding**: potential 1.5-2× additional speedup for whisper | High potential | Not yet implemented |
| **INT4 quantization**: could halve memory bandwidth (decode bottleneck) | High potential | Not yet implemented |
| **Whisper-MLA**: 87.5% KV cache reduction via latent attention conversion | Research | arxiv 2603.00563 |
| **Patience parameter**: patience=2 with beam=5 generates 2× more hypotheses | Direct user impact | Confirmed |
| **Whisper-large-v3 MPS correctness**: Fixed via iterative prompt + per-step sync | Critical fix | Implemented |

---

## 1. Current Performance Baselines

### 1.1 OPUS-MT Translation (Small Model, d_model=512, 6+6 layers)

50 sentences, beam=4, best-of-3, Apple M4.

| Backend | Type | tok/s | ms | vs CPU f32 | Commits | GPU% |
|---------|------|-------|-----|-----------|---------|------|
| **MPS** | **float16** | **1462** | **1060** | **1.79×** | 90 | 41% |
| **MPS** | **bfloat16** | **1461** | **1061** | **1.79×** | 90 | 41% |
| **MPS** | **float32** | **1032** | **1496** | **1.26×** | 96 | 54% |
| **MPS** | **int8_float16** | **901** | **1717** | **1.10×** | 93 | 45% |
| **MPS** | **int8_bfloat16** | **899** | **1721** | **1.10×** | 93 | 45% |
| CPU | float32 | 817 | 1895 | 1.00× | — | — |
| **MPS** | **int8** | **773** | **2008** | **0.95×** | 97 | 56% |
| CPU | int8 (RUY) | 540 | 2882 | 0.66× | — | — |

**Key observations**:
- FP16 is the optimal path for OPUS-MT (1.79× vs CPU)
- BF16 auto-promotes to FP16 with identical performance (M12.5)
- INT8_FP16 exceeds CPU baseline for the first time (1.10×, M12.10)
- GPU utilization is 41-56% — remaining idle time is CB submission + sync overhead

### 1.2 Whisper-large-v3 (d_model=1280, 32 enc + 32 dec)

~60s Russian audio, beam=5, patience=2, raw ctranslate2 API (confirmed correct).

| Backend | Type | Enc ms | Dec ms | Total ms | Speedup | tok/s |
|---------|------|--------|--------|----------|---------|-------|
| CPU | f32 | 3359 | 23398 | 26757 | 1.00× | 3 |
| **MPS** | **f32** | **1132** | **5518** | **6650** | **4.02×** | **12** |
| **MPS** | **f16** | **983** | **3186** | **4169** | **6.42×** | **19** |
| **MPS** | **bf16** | **1126** | **2965** | **4090** | **6.54×** | **19** |
| MPS | int8 | — | — | — | ERROR | — |
| MPS | int8_f16 | — | — | — | ERROR | — |

**Key observations**:
- **GPU advantage scales with model size**: 1.79× for OPUS-MT → 6.54× for whisper-large-v3
- Encoder (single forward pass, large matrices) gets 3.0-3.4× speedup alone
- Decoder (autoregressive, 32 layers) gets 4.2-7.9× speedup
- INT8 whisper models fail with type mismatch error (pre-existing bug, model not INT8-quantized)
- BF16 auto-promotes to FP16 and achieves best overall speedup (6.54×)

### 1.3 Whisper-large-v3-turbo (32 enc / 4 dec)

| Type | Best ms | Speedup | Correctness |
|------|---------|---------|-------------|
| CPU f32 | 12,803 | baseline | — |
| MPS f32 | 13,044 | 0.98× | Exact match |
| MPS f16 | 8,941 | **1.43×** | Minor token diffs |
| MPS bf16→f16 | 8,992 | **1.42×** | Same as f16 |

Turbo is decode-dominated (4 decoder layers × many autoregressive steps). CPU AMX handles small GEMMs efficiently, limiting GPU advantage.

---

## 2. Optimization History (Translation/Whisper Path)

### 2.1 Successful Optimizations

| Milestone | Optimization | Impact |
|-----------|-------------|--------|
| **M12.1** | Bucketed allocator + encode_barrier | f16: 1286→1500 tok/s (+17%), commits 188→90 (−52%) |
| **M12.5** | BF16→FP16 auto-promotion | bf16: 9→1426 tok/s (**158×**) |
| **M12.6** | INT8 GPU dequantize kernels | int8: 84→453 tok/s (**5.4×**) |
| **M12.10** | INT8 protect_buffer sync elimination | int8: 467→779 tok/s (**1.67×**), commits 97% reduced |
| **M12.12** | Pointer cache 2-way set-associative | Hit rate 20→49% (no wall-time gain) |

**Total improvement from M12 baseline to M12.25**:
- FP16: 1286 → 1462 tok/s (1.14×)
- INT8: 77 → 773 tok/s (**10×**)
- BF16: 9 → 1461 tok/s (**162×**)
- INT8_BF16: 8 → 899 tok/s (**112×**)

### 2.2 Rejected Optimizations

| Milestone | Optimization | Why Rejected |
|-----------|-------------|-------------|
| M12.2 | ObjC cached rowBytes | <1% impact |
| M12.3 | 256-entry pointer cache | <1% impact |
| M12.11 | Defer word ID conversion | No-op in common case |
| M12.11 | Batch CPU read of topk_scores | Already on CPU after sampler sync |
| M12.13 | Aggressive GEMV for decode | Naive GEMV 20% *slower* than MPS AMX |
| M12.14 | Pre-allocate DecodingResult | Slight regression |
| M12.14 | Eliminate alive_seq concat | memcpy worse than `ops::Concat` |
| M12.14 | Lazy hypothesis construction | Architecturally infeasible |
| M12.15 | BiasAdd + Activation + Residual fusion | ±3% noise, <0.5% theoretical |
| M12.16 | Object pooling temporaries | ±3% noise, allocator already O(1) |

**Lesson**: CPU-side decode loop overhead is **0.4%** of total time (M12.4). All CPU micro-optimizations are noise. Remaining gains must come from GPU kernel efficiency or algorithmic improvements.

---

## 3. Whisper Decode Pipeline Analysis

### 3.1 GEMMs Per Decode Step (whisper-large-v3, 32 layers)

| Component | GEMMs per layer | Total (32 layers) |
|-----------|----------------|-------------------|
| Self-attention Q projection | 1 | 32 |
| Self-attention KV projection | 1 | 32 |
| Self-attention output projection | 1 | 32 |
| Cross-attention Q projection | 1 | 32 |
| Cross-attention output projection | 1 | 32 |
| FFN linear_0 (d→4d) | 1 | 32 |
| FFN linear_1 (4d→d) | 1 | 32 |
| **Total per step (steps > 0)** | **7** | **224** |

Cross-attention K/V projections (2 GEMMs/layer) are cached from step 0 — only Q projection needed thereafter.

**Decoder step = ~224 GEMMs + 64 LayerNorms + 32 Softmax + 64 Add/Residual + sampling**

### 3.2 Synchronization Budget (After All M12 Optimizations)

| Sync Point | Count/Step | Cost | Eliminable? |
|-----------|-----------|------|-------------|
| Sampler GPU→CPU | 1 | 0.4 ms | No (needs token IDs on CPU) |
| SDPA small-tensor fallback | ~1 | 0.4 ms | No (CPU faster for sq=1, small sk) |
| **Total** | **~2** | **~0.8 ms** | Theoretical minimum: 1 |

Float paths (f16/f32/bf16) are at near-theoretical sync limits.

### 3.3 Time Distribution (FP16 Decode Step, OPUS-MT)

| Component | % of Time | Notes |
|-----------|-----------|-------|
| decoder_call (GPU GEMM + attention) | 50% | All GEMMs encoded, no per-op commit |
| sampler (TopK + sync + memcpy) | 50% | Single sync flushes all pending GPU work |
| beam bookkeeping (CPU) | 0.01% | ~0.2 ms |
| logits processing (CPU) | 0.3% | Suppress tokens, min length |
| state update (CPU) | 0.5% | Beam reordering |

**The sampler's 50% is NOT CPU overhead** — it's the time waiting for GPU to complete all GEMMs. The decoder_call returns instantly (encode-only), and the sampler sync flushes everything.

### 3.4 Patience Impact on Beam Search

With `beam_size=5`:

| Patience | max_candidates | Effect |
|----------|---------------|--------|
| 1.0 | 5 | Standard — finishes when 5 hypotheses found |
| **2.0** | **10** | **Extended** — needs 10 hypotheses before stopping |

**Impact**: patience=2 can roughly double the number of decode steps for some sequences, as the search continues well past when the top beam finishes.

**Recommendation**: Use `patience=1` for latency-sensitive applications. Quality difference is negligible for most ASR/translation tasks.

### 3.5 Decode Is Bandwidth-Bound

For autoregressive decode (m=1):
- Each GEMM reads the entire weight matrix (N×K bytes) but only computes 1 output row
- For whisper-large-v3 (d_model=1280): arithmetic intensity = 1.0 FLOP/byte (FP16)
- Apple M4 GPU compute-bandwidth ratio: 36 FLOPs/byte
- **Decode GEMM is 36× below compute saturation** — pure bandwidth bottleneck

This is why INT4 quantization (4× less bandwidth) is the highest-impact GPU optimization available.

---

## 4. GPU-Side Optimization Opportunities

### 4.1 INT4 Quantization (Priority: ★★★★)

**The decode bottleneck is memory bandwidth.** Reducing weight precision from FP16 (2 bytes/weight) to INT4 (0.5 bytes/weight) cuts memory bandwidth by 4×.

**llama.cpp Q4_K on Metal**:
- Block size 256, 4-bit weights with per-block scale+min
- ~4.5 bits per weight effective
- On Apple M4, memory bandwidth ~120 GB/s — a 2048×2048 weight matrix:
  - FP16: 8 MB, reads in ~67 µs
  - Q4_K: 2.3 MB, reads in ~19 µs — **3.5× faster**

**Quality evidence from research**:

| Source | Method | Model Size | WER Impact | Speed Impact |
|--------|--------|-----------|------------|-------------|
| arxiv 2503.09905 | INT4 (whisper.cpp) | -45% | Preserved (LibriSpeech clean) | -19% latency |
| arxiv 2511.08093 | HQQ INT4 | -70% | **Degraded in noise** | Significant |
| arxiv 2511.08093 | Dynamic INT8 (Quanto) | -57% | **Improved** over baseline | Moderate |
| Dropbox blog | HQQ INT4 + static cache | -75% | Minimal | **6× speedup** |
| WhisperKit | OD-MBP (mixed-bit) | <1 GB | Within 1% WER | Optimized |

**Key finding**: INT4 works well on clean speech but degrades in noisy conditions. Attention and LayerNorm layers should remain at 8-bit minimum.

**Important whisper.cpp finding**: Quantized models (Q4_0, Q4_1) are reportedly **slower on Metal than FP16** in whisper.cpp. Custom dequant kernels can't compete with MPS hardware GEMM. INT4 on Metal only wins if the dequant+GEMM is properly fused.

**Implementation path**:
1. Add Q4_K block format to CTranslate2's type system
2. Write MSL GEMV kernel for Q4_K×FP16 (similar to existing fused INT8 GEMV)
3. Convert whisper/OPUS-MT weights to Q4_K format
4. Expected: **2-4× decode speedup** from bandwidth reduction

### 4.2 QKV Projection Fusion (Priority: ★★★)

**Current** (confirmed via `src/layers/attention.cc:490-596`): Q and KV use **separate Dense layers** for standard MHA:
```
Q  = input @ W_q   // Dense linear[0]: GEMM 1
KV = input @ W_kv  // Dense linear[1]: GEMM 2 (then split into K, V)
```

**Proposed**: Merge Q and KV weights into single GEMM:
```
QKV = input @ [W_q; W_kv]  // 1 GEMM, wider output
```

**Expected impact**: 10-15% decode speedup from reduced dispatch overhead (saves ~32 dispatches/step for whisper-large-v3).

**Effort**: Medium — requires weight tensor concatenation at model load time. Model format change needed.

### 4.3 Flash Attention for Encoder-Decoder (Priority: ★★)

FlashMultiHeadAttention on MPS produces garbage for encoder-decoder models (465 commits/token). Standard MultiHeadAttention works correctly.

**Root cause**: FlashMHA decode path's KV-cache management is designed for decoder-only (causal) models. Cross-attention has different patterns.

**If fixed**: 1.25-1.5× decode speedup (based on TinyLlama flash results). High effort.

### 4.4 KV-Cache Quantization (Priority: ★★ for LLMs, ★ for Whisper)

For whisper-large-v3: KV cache ~16 MB total at 100 steps. Not a bottleneck. Low priority for whisper/translation (short sequences). High priority for LLM generator with long contexts.

### 4.5 Metal Residency Sets (Priority: ★, Robustness)

`MTLResidencySet` (macOS 15+) pins model weight buffers in physical memory. Zero normal impact, prevents 10-100× latency spikes. Easy to implement (~20 lines).

---

## 5. Algorithmic Optimization Opportunities

### 5.1 Speculative Decoding for Whisper (Priority: ★★★★★)

**The single highest-impact optimization opportunity available.**

Uses a smaller "draft" model to propose multiple tokens, then the main model validates in a single forward pass:

1. Draft model (distil-whisper, 2 decoder layers) generates N candidate tokens
2. Main model (whisper-large-v3, 32 layers) runs one forward pass with all N tokens
3. Verify: accept matching prefix, reject from first divergence
4. **Outputs are mathematically identical** to the main model alone

**Expected speedup**: 1.5-2× (HuggingFace benchmarks, ~70-80% acceptance rate)

**Why especially good on Metal**:
- Whisper transcription is highly predictable — ASR follows strong linguistic patterns
- Draft model runs fast (~6% cost of main model per token)
- Verification converts bandwidth-bound GEMV (m=1) to compute-efficient GEMM (m=N)
- GPU utilization jumps from ~41% to ~70-80% during verification

**CTranslate2 reference**: arxiv paper "Model-free Speculative Decoding for ASR" (2507.21522) was implemented **using CTranslate2** — the infrastructure supports it.

**Implementation path**:
1. Load two models (main + assistant) sharing the same encoder
2. Assistant generates N candidate tokens (N=5-10) with 2-layer decoder
3. Main model validates in single forward pass
4. Accept/reject logic
5. Works best at beam=1; at beam≥4, speculative decoding becomes marginal or negative

### 5.2 Distilled Models (Priority: ★★★★)

**Fastest path to 2-6× speedup without any code changes.**

| Model | Decoder Layers | Params | Expected Speed | WER Impact |
|-------|---------------|--------|----------------|------------|
| whisper-large-v3 | 32 | 1.5B | 1× (baseline) | — |
| whisper-large-v3-turbo | 4 | 809M | **~3-4×** | <1% WER increase |
| distil-whisper-large-v3 | 2 | 756M | **~6×** | <1% WER increase |

### 5.3 Batched Processing (Priority: ★★★)

Process multiple 30-second segments in a single batch. Increases GPU utilization from 41% → 70-90%. Expected 2-4× throughput. Already supported in CTranslate2's API.

### 5.4 Patience & Beam Size Tuning (Priority: ★★★, Zero Code Change)

| Setting | Decode Cost | Quality |
|---------|------------|---------|
| beam=1, patience=1 | 1× | Good |
| beam=2, patience=1 | 2× | Very good |
| beam=5, patience=1 | 5× | Excellent |
| beam=5, patience=2 | **~10×** | Marginal gain over patience=1 |

**Recommendation**: beam=1 or beam=2 with patience=1 for latency-sensitive applications.

---

## 6. Whisper-Specific Correctness Findings

### 6.1 Whisper-large-v3 MPS Fix

**Problem**: 32-decoder-layer model produced zero transcription segments on MPS.

**Root cause**: Processing multi-token prompts at once caused KV-cache corruption in the 32-layer decoder. GPU work from step N wasn't completed before step N+1 read the KV cache.

**Fix** (in `src/layers/whisper.cc`): Iterative prompt processing — one token at a time with `synchronize_stream()` between each step.

**Result**: Raw ctranslate2 API produces **exact token-level match** between CPU and MPS for all configurations.

### 6.2 Vocabulary Mismatch (faster_whisper Integration)

CT2 vocabulary includes `<|yue|>` (Cantonese) at position 50358, absent from the HuggingFace tokenizer. Shifts `<|transcribe|>` and `<|notimestamps|>` IDs by +1. **Not a Metal bug** — affects CPU equally.

### 6.3 INT8 on MPS

`ValueError: expected storage to be of type float16, but is of type int8` — pre-existing bug; whisper models aren't INT8-quantized.

---

## 7. Comparison with Other Whisper Implementations

### 7.1 Whisper-MLA (arxiv 2603.00563, March 2026)

- Converts MHA to Multi-Head Latent Attention (MLA), inspired by DeepSeek-V3
- Projects K/V into a shared low-rank latent space (dimension 96 vs 1280)
- **KV cache reduced by up to 87.5%**
- Apply **only to decoder self-attention** (not cross-attention or encoder)
- Requires only minimal fine-tuning from pretrained Whisper weights

### 7.2 Comparative Benchmarks

| Implementation | Model | Hardware | Speed | Notes |
|---------------|-------|----------|-------|-------|
| **MLX whisper** | large-v3 | M4 Max | ~10× real-time | Turbo variant |
| **whisper.cpp** | large-v3 | M4 | ~5× real-time (est.) | Metal + CoreML encoder |
| **Lightning Whisper MLX** | various | Apple Silicon | Claims 10× whisper.cpp | Heavily optimized |
| **CTranslate2 (Metal)** | large-v3-turbo | M4 | 4.1× real-time | beam=1, no timestamps |
| **CTranslate2 (Metal)** | large-v3 | M4 | **14× real-time** | beam=5, patience=2, 60s audio |

**Gap analysis**: MLX achieves higher performance through lazy evaluation (graph-level fusion) and native Metal kernels. For large models, CTranslate2's MPS GEMM path is competitive because GEMM dominates and MPS uses hardware AMX.

---

## 8. Implementation Roadmap

### Tier 1: High Impact, Achievable

| # | Optimization | Expected Impact | Effort | Risk |
|---|-------------|----------------|--------|------|
| 1 | **Use distil-whisper or turbo** | 3-6× faster | Zero (model swap) | Low |
| 2 | **Patience=1 with beam≤2** | 2-5× faster | Zero (parameter) | Low |
| 3 | **Speculative decoding** | 1.5-2× faster | High | Medium |
| 4 | **INT4 quantization** | 2-4× decode speedup | High | Medium |

### Tier 2: Medium Impact

| # | Optimization | Expected Impact | Effort | Risk |
|---|-------------|----------------|--------|------|
| 5 | Batch processing (multi-segment) | 2-4× throughput | Low | Low |
| 6 | FlashMHA for encoder-decoder | 1.25-1.5× decode | High | High |
| 7 | QKV projection fusion | 10-15% decode | Medium | Low |
| 8 | Metal Residency Sets | Robustness | Easy | None |

### Tier 3: Future / Research

| # | Optimization | Expected Impact | Effort | Risk |
|---|-------------|----------------|--------|------|
| 9 | Whisper-MLA (KV cache 87.5% reduction) | Bandwidth savings | Very High | Retraining |
| 10 | Custom Metal flash attention (tiled, simdgroup) | 1.5-2× attention | Very High | High |
| 11 | Metal 4 tensor API (M5+ hardware) | 2-4× | High | Hardware-dependent |
| 12 | Continuous batching / PagedAttention | Server throughput | Very High | High |

---

## 9. Techniques Definitively Ruled Out

| Technique | Why Not | Evidence |
|-----------|---------|----------|
| Custom Metal GEMM kernels | 7-12% of MPS throughput on M4 | Benchmarked (M12.13) |
| Apple Neural Engine (CoreML) | 0.095 ms XPC overhead per op | Too slow for autoregressive decode |
| CPU-side micro-optimizations | 99.6% of time is GPU | M12.11, M12.14, M12.16 |
| BiasAdd/activation/residual fusion | ±3% noise | M12.15 benchmark |
| Object pooling for temporaries | Metal allocator already O(1) | M12.16 benchmark |
| AMX INT8 (BNNSMatMul) | AMX doesn't natively support INT8 | MIT thesis confirmation |

---

## Appendix: CPU-Side Analysis

### CPU Performance

| Metric | Value |
|--------|-------|
| AMX FP32 throughput | ~3.2 TFLOPS (single cluster) |
| AMX: 2 perf clusters | ~6.4 TFLOPS total |
| Memory bandwidth | ~120 GB/s (shared with GPU) |
| CPU f32 translation (OPUS-MT) | 817 tok/s |
| CPU f32 whisper-large-v3 | 3 tok/s |
| Neural Engine | 38 TOPS (INT8), 19 TFLOPS (FP16), 16 cores, 2.8W |

### CPU INT8 (RUY)

540 tok/s — **slower than CPU f32 (817 tok/s)** on Apple Silicon. AMX accelerates f32 GEMM; RUY uses NEON, not AMX.

### Multi-Threading

| Threads | CPU f32 tok/s | CPU int8 tok/s |
|---------|--------------|---------------|
| 1 | ~250 | 257 |
| 2 | ~500 | 424 |
| 4 | 741 | 540 |
| 8 | ~700 | 460 |

Scales well to 4 threads (2 performance clusters × AMX), then degrades.

---

## Sources

### Papers
- **arxiv 2603.00563** (March 2026): Whisper-MLA — 87.5% KV cache reduction
- **arxiv 2507.21522**: Model-free Speculative Decoding for ASR using CTranslate2
- **arxiv 2503.09905** (March 2025): INT4 quantization for Whisper
- **arxiv 2511.08093** (Nov 2025): Quantizing Whisper — INT4/NF4 degrade noisy robustness
- **arxiv 2412.11272** (Dec 2024): WhisperFlow — real-time streaming

### Frameworks & Tools
- HuggingFace speculative decoding blog, Whisper-Medusa (aiola)
- Distil-Whisper: 6× faster, <1% WER increase
- WhisperKit (Argmax): CoreML-optimized, ANE encoder
- Lightning Whisper MLX: Claims 10× whisper.cpp
- Dropbox blog: Static KV cache + torch.compile + HQQ INT4 = 6× speedup
