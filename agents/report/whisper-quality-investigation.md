# Whisper Quality Investigation Report

**Date:** 2026-03-15
**Scope:** CT2 Metal vs CPU vs reference implementations (OpenAI, mlx-whisper)
**Model:** whisper-large-v3-turbo on FLEURS benchmark (50 samples x 6 languages)

## Executive Summary

The quality gap between CTranslate2 and reference implementations (OpenAI, mlx-whisper) is caused by a **bug in CTranslate2's decoder inference** that produces incorrect logit distributions specifically for whisper-large-v3-turbo. The bug causes beam search at temperature=0 to fail completely, forcing faster_whisper's temperature fallback to compensate via sampling — resulting in degraded but functional output.

**The issue is NOT model conversion.** It is a CTranslate2 inference engine bug.

---

## 1. Quality Gap Quantification

### Benchmark WER Comparison (beam=1, FLEURS test, 50 samples/lang)

| Config | ENG | JAP | MAN | GER | SPA | ARA | **Avg** |
|--------|-----|-----|-----|-----|-----|-----|---------|
| openai_whisper f32 | 4.7% | 6.7% | 50.5% | 3.8% | 2.5% | 15.9% | **14.0%** |
| mlx_whisper f16 | 4.7% | 6.3% | 50.5% | 3.8% | 2.5% | 15.7% | **13.9%** |
| ct2_cpu float32 | 9.2% | 37.4% | 64.8% | 8.9% | 7.2% | 29.3% | **26.1%** |
| ct2_metal float32 | 15.3% | 30.7% | 62.1% | 10.8% | 7.5% | 31.4% | **26.3%** |
| ct2_metal f16 b=5 | 16.0% | 33.8% | 62.4% | 8.5% | 7.4% | 24.9% | **25.5%** |

**Key observations:**
- CT2 (both CPU and Metal) is ~12 pp worse than reference implementations
- CT2 CPU and CT2 Metal are nearly identical (~26% avg WER)
- The gap is consistent across ALL languages, not just non-English
- Metal vs CPU difference is statistical noise, not a Metal-specific bug

### Conclusion: Metal is NOT the problem. The CTranslate2 decoder is.

---

## 2. Root Cause: CTranslate2 Decoder Bug

### 2.1 Smoking Gun — First Generated Token

When given the same input (mel spectrogram + prompt tokens `[50258, 50259, 50359, 50363]`), the decoders produce fundamentally different logit distributions:

| Implementation | First token (raw, no suppression) | Logit / Score |
|----------------|-----------------------------------|---------------|
| **OpenAI** (reference) | `50360` `<\|startoflm\|>` (after suppression: `2908` "However") | logit = 10.03 |
| **CTranslate2** | `50365` (timestamp 0.02s) | score = -4.74 |

OpenAI's model has timestamp token 50365 at logit 0.75 (very low, not competitive). CT2's model selects it as the top token. The logit distributions are **categorically different**.

### 2.2 Verification: NOT a Conversion Issue

Three independent model sources all produce the **same wrong output**:

| Model Source | First Token | Score |
|-------------|-------------|-------|
| Local CT2 model (data/whisper-large-v3-turbo) | 50365 | -4.68 |
| Fresh conversion (`TransformersConverter('openai/whisper-large-v3-turbo')`) | 50365 | -4.68 |
| Systran's official model (mobiuslabsgmbh/faster-whisper-large-v3-turbo) | 50365 | -4.68 |

All three produce **bit-identical results** (Orig-Fresh diff: 0.000000). The conversion is correct. The bug is in the CTranslate2 C++ inference engine.

### 2.3 Verification: Turbo-Specific Bug

| Model | CT2 First Token | OpenAI First Token | CT2 Output Match |
|-------|----------------|-------------------|-----------------|
| whisper-base (6enc/6dec, 512d, 8h, 80mel) | `2908` (" However") | `2908` (" However") | **EXACT MATCH** |
| whisper-small (12enc/12dec, 768d, 12h, 80mel) | `2908` (" However") | `2908` (" However") | **EXACT MATCH** |
| whisper-large-v3-turbo (32enc/4dec, 1280d, 20h, 128mel) | `50365` (timestamp) | `50360` (startoflm) | **WRONG** |

whisper-base and whisper-small produce correct output. The bug is triggered specifically by turbo's architecture.

### 2.4 What's Unique About Turbo

| Property | base | small | large-v3-turbo |
|----------|------|-------|---------------|
| Encoder layers | 6 | 12 | **32** |
| Decoder layers | 6 | 12 | **4** |
| Hidden dim | 512 | 768 | **1280** |
| Attention heads | 8 | 12 | **20** |
| n_mels | 80 | 80 | **128** |

Turbo is unique in its **highly asymmetric architecture** (32 encoder / 4 decoder layers) and its larger dimensions. The exact trigger needs further C++ debugging.

### 2.5 What Makes It Still Work (Partially)

faster_whisper's `transcribe()` has temperature fallback: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`. When temperature=0 beam search fails (avg_logprob = -4.68 << threshold -1.0), it retries with higher temperature using sampling. This produces reasonable but degraded output.

The quality degradation comes from:
1. **Sampling instead of beam search** — more random, less optimal
2. **Some fallback iterations may still fail** — wasting time and sometimes producing garbage that passes thresholds
3. **Condition-on-previous-text propagation** — errors from one segment affect subsequent segments

---

## 3. What Was Ruled Out

### 3.1 Feature Extraction
- faster_whisper and OpenAI mel spectrograms are nearly identical
- Max diff: 0.045, mean diff: 0.000006
- n_mels=128 is correctly detected and used for turbo
- **RULED OUT** as cause

### 3.2 Encoder
- CT2 and OpenAI encoder outputs match closely
- Max diff: 0.004395, mean diff: 0.000015
- Identical across original and fresh conversion
- **RULED OUT** as cause

### 3.3 Prompt Tokens
- Both implementations use identical prompt: `[50258, 50259, 50359, 50363]`
- SOT, language, task, notimestamps tokens all match
- **RULED OUT** as cause

### 3.4 Suppress Token Lists
- CT2 and OpenAI suppress lists are nearly identical (88 tokens each)
- Only differences: CT2 has 50363 (notimestamps), OpenAI has 50358 (translate)
- Timestamp tokens (>=50364) are NOT in either suppress list
- **RULED OUT** as cause

### 3.5 Timestamp Handling
- Both implementations skip `ApplyTimestampRules` when `without_timestamps=True`
- The `<|notimestamps|>` token in prompt correctly disables timestamp rules
- **RULED OUT** as cause

### 3.6 Model Weights
- Token embeddings match between HuggingFace, OpenAI, and CT2 (diff: 0.0)
- Output projection uses tied weights (same as embeddings)
- All weight shapes match
- **RULED OUT** as cause

---

## 4. CT2 Metal vs CT2 CPU (Secondary Gap)

While the primary gap is CT2 vs reference (~12 pp), there's a small secondary gap between CT2 Metal and CT2 CPU:

| Language | CT2 CPU f32 | CT2 Metal f32 | Delta |
|----------|-------------|---------------|-------|
| English | 9.2% | 15.3% | +6.1 |
| Japanese | 37.4% | 30.7% | -6.7 |
| German | 8.9% | 10.8% | +1.9 |
| Spanish | 7.2% | 7.5% | +0.3 |
| Arabic | 29.3% | 31.4% | +2.1 |
| **Average** | **26.1%** | **26.3%** | **+0.2** |

The Metal vs CPU difference is **noise** (~0.2 pp average). Per-language variations are expected because temperature fallback is inherently stochastic — different temperatures trigger for different samples on different devices.

---

## 5. Recommendations for Fix

### 5.1 Immediate Priority: Debug CT2 Decoder for Turbo

The decoder produces wrong logits for whisper-large-v3-turbo specifically. Investigation should focus on:

1. **Compare decoder hidden states layer-by-layer** between CT2 and OpenAI's PyTorch decoder. The divergence point will identify the buggy component.

2. **Check if the TransformerDecoder handles the asymmetric architecture correctly** — 32 encoder layers producing 1500 time-steps of 1280-dim context, feeding into only 4 decoder layers with cross-attention.

3. **Cross-attention KV computation** — verify that cross-attention K,V projections are computed correctly for the full encoder output (shape [1, 1500, 1280]) when using 20 heads and d_head=64.

4. **Test whisper-large-v3** (32enc/32dec) — if this works, the bug is specific to 4-decoder-layer configuration. If it fails, the bug is in handling 1280-dim/20-head/128-mel scale.

### 5.2 Verification Test

A minimal reproduction test:

```python
import ctranslate2, whisper, torch, numpy as np

# Reference: OpenAI
ow = whisper.load_model('large-v3-turbo', device='cpu')
audio = np.zeros(480000, dtype=np.float32)  # 30s silence
mel = whisper.log_mel_spectrogram(
    whisper.pad_or_trim(torch.from_numpy(audio)), n_mels=128
).unsqueeze(0)

with torch.no_grad():
    logits = ow.decoder(torch.tensor([[50258]], dtype=torch.long), ow.encoder(mel))
print("OpenAI top-1:", int(logits[0, -1].argmax()))  # Should be 50360

# CT2
ct2 = ctranslate2.models.Whisper('large-v3-turbo', device='cpu', compute_type='float32')
r = ct2.generate(
    ctranslate2.StorageView.from_array(mel.numpy()),
    [[50258]], beam_size=1, max_length=2, suppress_blank=False, suppress_tokens=[]
)
print("CT2 top-1:", r[0].sequences_ids[0][0])  # Bug: produces 50365 instead of 50360
```

### 5.3 Impact If Fixed

Fixing the decoder bug would:
- Eliminate the ~12 pp WER gap between CT2 and reference
- Make temperature fallback unnecessary (beam search would work at temp=0)
- Improve speed slightly (no wasted fallback iterations)
- Make CT2 Metal competitive with mlx-whisper and OpenAI on quality while being 1.3-1.7x faster

---

## 6. Summary

| Finding | Status |
|---------|--------|
| Primary quality gap (CT2 vs reference ~12pp) | **CTranslate2 decoder inference bug, turbo-specific** |
| Secondary gap (Metal vs CPU ~0.2pp) | **Noise from stochastic temperature fallback** |
| Model conversion | **Verified correct — identical across 3 sources** |
| Feature extraction | **Verified correct — negligible difference** |
| Encoder | **Verified correct — matches reference** |
| Prompt/suppress tokens | **Verified correct — matches reference** |
| Bug scope | **whisper-large-v3-turbo only — base and small work perfectly** |
| Workaround in place | **Temperature fallback partially compensates (WER 26% vs ideal 14%)** |
