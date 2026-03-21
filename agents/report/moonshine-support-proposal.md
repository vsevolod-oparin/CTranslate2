# Moonshine Model Support for CTranslate2 — Technical Proposal

**Date:** 2025-03-21
**Author:** Research analysis for MPS/Metal backend integration
**Status:** Proposal

---

## 1. Executive Summary

Moonshine is a family of encoder-decoder ASR models by Useful Sensors, purpose-built for real-time streaming speech recognition on edge devices. Adding Moonshine support to CTranslate2 with the MPS backend would create the **fastest Moonshine runtime on Apple Silicon** — no other runtime offers Metal-accelerated Moonshine inference with INT8 quantization.

**Key value proposition:**
- CTranslate2 already has ~90% of the required ops (RoPE, sliding window attention, SwiGLU, KV caching)
- Moonshine's architecture is a close relative of Whisper — same encoder-decoder transformer pattern
- The MPS backend is the unique differentiator — ONNX Runtime's CoreML provider is the only alternative on macOS
- Streaming with encoder KV caching would be a novel CTranslate2 capability

---

## 2. Moonshine Architecture (Detailed)

### 2.1 Model Variants

| Variant | Encoder Layers | Decoder Layers | Enc Hidden | Dec Hidden | Heads | Params | WER (avg) |
|---------|---------------|---------------|------------|------------|-------|--------|-----------|
| Tiny | 6 | 6 | 288 | 288 | 6 | 34M | 12.01% |
| Small | 10 | 10 | 620 | 512 | 8 | 123M | 7.84% |
| Medium | 14 | 14 | 768 | 640 | 10 | 245M | 6.65% |

All variants: head_dim=64, vocab=32768 BPE tokens, 16kHz mono input.

### 2.2 Audio Frontend

The audio frontend replaces Whisper's mel spectrogram + sinusoidal positional embedding:

```
Raw 16kHz audio (float32)
    │
    ▼
Conv1d(1, hidden/2, kernel=127, stride=64, padding=63)  ← causal, stride-2 time reduction
    │ + GELU
    ▼
Conv1d(hidden/2, hidden, kernel=7, stride=2, padding=3)  ← stride-2 again, total 4x reduction
    │ + GELU
    ▼
[batch, hidden, time/4] → transpose → [batch, time/4, hidden]
    │
    ▼
CMVN (cepstral mean-variance normalization, running stats)
    │
    ▼
Encoder input: 50Hz features (one frame per 20ms)
```

**Key differences from Whisper frontend:**
- Raw waveform input (not mel spectrogram) — Whisper uses 80/128-channel log-mel
- Learned convolutions (not fixed STFT + mel filterbank)
- **No positional embedding** in the encoder — positions are captured by the sliding window attention pattern
- 50Hz output rate (20ms/frame) vs Whisper's 50Hz (same rate, different computation)

**CTranslate2 impact:** Needs `Conv1d` with stride and padding. CTranslate2 already has `Conv1d` for Whisper's frontend. The kernel sizes and strides are different but the op is the same.

### 2.3 Encoder

```
MoonshineEncoder:
  ├── audio_frontend (Conv1d × 2 + CMVN)
  ├── layers × 14:
  │     ├── self_attention:
  │     │     ├── q_proj, k_proj, v_proj (Linear, no bias)
  │     │     ├── out_proj (Linear, no bias)
  │     │     ├── sliding_window: [window_size, lookahead]
  │     │     │     layers 0-1, 12-13: [16, 4]  (80ms lookahead)
  │     │     │     layers 2-11: [16, 0]  (fully causal)
  │     │     └── NO position embeddings
  │     ├── feed_forward:
  │     │     ├── fc1 (Linear, hidden → 4*hidden)
  │     │     ├── GELU activation
  │     │     └── fc2 (Linear, 4*hidden → hidden)
  │     └── layer_norm × 2 (pre-norm)
  └── final_layer_norm
```

**CTranslate2 mapping:**
- `sliding_window` attention: **Already supported** (`attention_layer.cc:141`, per-layer `sliding_window` attribute)
- No position embeddings: Just omit them (simpler than Whisper)
- GELU activation: **Already supported**
- Pre-norm transformer: **Already supported** (standard TransformerEncoderLayer)

**Novel feature — Encoder KV caching for streaming:**
Because the encoder uses **causal sliding window attention** (no future context in layers 2-11), encoder activations can be cached. When new audio arrives:
1. Only encode the new audio chunk through the frontend
2. Concatenate with cached encoder KV states
3. Apply sliding window attention (only attends to last `window_size` frames)
4. Cache the new KV states, evict frames outside the window

This means re-encoding is O(new_chunk) not O(total_audio). CTranslate2 has KV caching for decoders but **not for encoders**. Adding encoder KV caching is the biggest new feature.

### 2.4 Adapter (Encoder → Decoder bridge)

```
MoonshineAdapter:
  ├── learned_position_embeddings (Embedding, max_pos=4096, dim=dec_hidden)
  ├── linear_projection (Linear, enc_hidden → dec_hidden)  [if enc_hidden != dec_hidden]
  └── layer_norm
```

Maps encoder output (768-dim) to decoder input (640-dim) and adds learned positional embeddings.

**CTranslate2 mapping:** Simple linear + embedding + layer_norm. All ops exist.

### 2.5 Decoder

```
MoonshineDecoder:
  ├── embed_tokens (Embedding, vocab=32768, dim=640)
  ├── layers × 14:
  │     ├── self_attention:
  │     │     ├── q_proj, k_proj, v_proj (Linear, no bias)
  │     │     ├── out_proj (Linear, no bias)
  │     │     ├── RoPE (partial_rotary_factor=0.5, theta=10000)
  │     │     └── causal mask (standard autoregressive)
  │     ├── cross_attention:
  │     │     ├── q_proj, k_proj, v_proj (Linear, no bias)
  │     │     ├── out_proj (Linear, no bias)
  │     │     └── attends to encoder output
  │     ├── feed_forward:
  │     │     ├── gate_proj + up_proj (SwiGLU pattern)
  │     │     ├── SiLU activation on gate
  │     │     └── down_proj
  │     └── layer_norm × 3 (pre-norm, self-attn + cross-attn + ffn)
  ├── final_layer_norm
  └── lm_head (Linear, 640 → 32768, no tie with embed)
```

**CTranslate2 mapping:**
- RoPE with partial rotary: **Already supported** (`rotary_dim` attribute, `partial_rotary_factor=0.5` means 32 of 64 dims get rotation)
- SwiGLU (SiLU-gated): **Already supported** (Llama decoder uses this exact pattern)
- Cross-attention with KV caching: **Already supported** (Whisper decoder does this)
- Causal self-attention with KV caching: **Already supported**

The Moonshine decoder is essentially a **Llama-style decoder + cross-attention** — both patterns already exist in CTranslate2.

---

## 3. Gap Analysis: CTranslate2 vs Moonshine Requirements

### 3.1 Already Supported (no changes needed)

| Component | CTranslate2 Status | Used By |
|-----------|-------------------|---------|
| RoPE (partial rotary) | ✅ `rotary_dim`, MPS kernel | Llama, Mistral |
| Sliding window attention | ✅ `sliding_window` attribute | Mistral |
| SwiGLU feed-forward | ✅ `Activation::SWISH` + gate | Llama |
| GELU activation | ✅ | Whisper, BERT |
| Cross-attention + KV cache | ✅ | Whisper decoder |
| Causal self-attention + KV cache | ✅ | All decoder models |
| Conv1d (encoder frontend) | ✅ | Whisper encoder |
| Layer normalization | ✅ (RMSNorm + LayerNorm) | All models |
| BPE tokenizer | ✅ | All models |
| INT8 quantization | ✅ | All models |
| Float16 inference | ✅ + MPS | All models |

### 3.2 Needs Implementation

| Component | Effort | Details |
|-----------|--------|---------|
| **MoonshineSpec** (Python) | Small | New model spec class, ~200 lines. Defines weight mapping from HuggingFace safetensors to CT2 format. |
| **MoonshineModel** (C++) | Small | New model class (~100 lines). Register in `model_factory.cc`. Very similar to `WhisperModel`. |
| **MoonshineEncoder** (C++) | Medium | New encoder class. Audio frontend (Conv1d with different params than Whisper) + transformer layers with per-layer sliding window sizes. ~200 lines. |
| **MoonshineDecoder** (C++) | Small | Minimal — can likely reuse existing `TransformerDecoder` with RoPE + cross-attention configuration. Or a thin wrapper. ~100 lines. |
| **Adapter layer** (C++) | Small | Linear projection + learned positional embedding + layer norm. ~50 lines. |
| **Model converter** (Python) | Medium | `moonshine.py` converter to map HuggingFace weight names to CT2 spec. ~150 lines. |
| **CMVN normalization** | Small | Running mean/variance normalization for audio. ~30 lines. |
| **Encoder KV caching** (C++, optional) | Large | New capability: cache encoder attention KV states for incremental encoding. This is the big streaming win but can be deferred. ~300-500 lines. |
| **Per-layer sliding window config** | Small | Current CT2 has one `sliding_window` value per model. Moonshine needs per-layer values `[16,4], [16,0], ...]`. ~50 lines to add per-layer config. |

### 3.3 What Does NOT Need Implementation

- **ConvTranspose1d**: Earlier analysis suggested this was needed. Looking at the actual config, Moonshine streaming uses two **Conv1d** layers (stride-2), not ConvTranspose. The non-streaming Moonshine uses ConvTranspose but the streaming variant does not.
- **New attention mechanism**: Sliding window + RoPE are both existing CT2 features
- **New quantization**: Existing INT8/FP16 work as-is
- **New MPS ops**: All required ops already have MPS implementations

---

## 4. Implementation Plan

### Phase 1: Basic Inference (offline mode)
**Estimated effort:** 1-2 weeks

1. **MoonshineSpec** — Python model specification
   - Map HuggingFace `model.safetensors` weight names to CT2 internal names
   - Handle encoder/decoder dimension mismatch (768 → 640 via adapter)
   - Configure per-layer sliding window sizes
   - Configure partial RoPE (factor=0.5)

2. **Model converter** — `python/ctranslate2/converters/moonshine.py`
   - Load from HuggingFace `UsefulSensors/moonshine-streaming-{tiny,small,medium}`
   - Convert safetensors → CT2 format with quantization support

3. **MoonshineModel + MoonshineReplica** — C++ model class
   - Register `"MoonshineSpec"` in model_factory
   - Implement `encode()` with audio frontend (Conv1d layers + CMVN)
   - Implement `generate()` reusing existing decoder infrastructure

4. **Per-layer sliding window** — extend TransformerEncoderLayer
   - Add `sliding_window` as per-layer attribute (currently per-model)
   - Moonshine encoder layers 0-1, 12-13 use window [16, 4]; layers 2-11 use [16, 0]

5. **Testing**
   - Verify against ONNX Runtime reference output
   - WER on LibriSpeech test-clean (target: match HuggingFace reported 2.08% for Medium)
   - Performance benchmark vs ONNX Runtime on Apple Silicon

### Phase 2: Streaming with Encoder KV Caching
**Estimated effort:** 2-3 weeks (can be deferred)

1. **Encoder KV cache infrastructure**
   - Extend `MultiHeadAttention` to optionally cache and return KV states from encoder
   - Add `encode_chunk()` method that accepts previous KV cache + new audio
   - Sliding window eviction: only keep last `window_size` frames in cache

2. **Streaming API**
   - New `MoonshineReplica::encode_streaming(audio_chunk, encoder_state)` method
   - Returns encoder output + updated state for next call
   - Decoder runs normally on the accumulated encoder output

3. **Integration with metalwhisper**
   - Replace the Whisper streaming pipeline with Moonshine streaming
   - Proper incremental encoding (only new audio processed)
   - Expected latency: <200ms per update (vs 100-300ms current Whisper)

---

## 5. Weight Mapping (HuggingFace → CTranslate2)

Based on the HuggingFace Moonshine model structure:

```
HuggingFace name                          → CTranslate2 name
─────────────────────────────────────────────────────────────
encoder.conv1.weight                      → encoder/conv1/weight
encoder.conv1.bias                        → encoder/conv1/bias
encoder.conv2.weight                      → encoder/conv2/weight
encoder.conv2.bias                        → encoder/conv2/bias
encoder.layers.{i}.self_attn.q_proj.weight → encoder/layer_{i}/self_attention/linear_0/weight
encoder.layers.{i}.self_attn.k_proj.weight → encoder/layer_{i}/self_attention/linear_1/weight
encoder.layers.{i}.self_attn.v_proj.weight → encoder/layer_{i}/self_attention/linear_2/weight [*]
encoder.layers.{i}.self_attn.o_proj.weight → encoder/layer_{i}/self_attention/linear_3/weight [*]
encoder.layers.{i}.mlp.fc1.weight         → encoder/layer_{i}/ffn/linear_0/weight
encoder.layers.{i}.mlp.fc2.weight         → encoder/layer_{i}/ffn/linear_1/weight
encoder.layers.{i}.input_layernorm.*      → encoder/layer_{i}/self_attention/layer_norm/*
encoder.layers.{i}.post_attention_layernorm.* → encoder/layer_{i}/ffn/layer_norm/*
encoder.final_layer_norm.*                → encoder/layer_norm/*

adapter.pos_emb.weight                    → adapter/position_embeddings/weight [**]
adapter.proj.weight                       → adapter/projection/weight
adapter.proj.bias                         → adapter/projection/bias
adapter.norm.*                            → adapter/layer_norm/*

decoder.embed_tokens.weight               → decoder/embeddings/weight
decoder.layers.{i}.self_attn.*            → decoder/layer_{i}/self_attention/*
decoder.layers.{i}.encoder_attn.*         → decoder/layer_{i}/attention/*  (cross-attention)
decoder.layers.{i}.mlp.gate_proj.weight   → decoder/layer_{i}/ffn/linear_0/weight  (SwiGLU gate)
decoder.layers.{i}.mlp.up_proj.weight     → decoder/layer_{i}/ffn/linear_0_noact/weight  (SwiGLU up)
decoder.layers.{i}.mlp.down_proj.weight   → decoder/layer_{i}/ffn/linear_1/weight
decoder.lm_head.weight                    → decoder/projection/weight
```

[*] Exact CT2 naming depends on whether attention uses fused QKV or separate projections.
[**] New: adapter is a novel component not in Whisper; needs explicit handling.

---

## 6. Performance Expectations

### Inference Speed (Apple Silicon M4)

| Operation | Whisper Turbo (current) | Moonshine Medium (expected) |
|-----------|------------------------|----------------------------|
| Encode 1s audio | ~60ms (but pads to 30s = wasted) | ~15ms (no padding) |
| Encode 5s audio | ~150ms (pads to 30s) | ~40ms |
| Encode 10s audio | ~280ms (pads to 30s) | ~70ms |
| Encode 30s audio | ~280ms | ~200ms |
| Decode (per token) | ~5ms | ~4ms (smaller decoder) |
| **Streaming update** | ~280ms (re-encode full window) | ~20ms (encode new chunk only, with caching) |

The streaming speedup from encoder caching is the key win: **10-15x faster per update** vs re-encoding the full window.

### Quality (WER on standard benchmarks)

| Model | LibriSpeech clean | LibriSpeech other | Average (8 datasets) |
|-------|-------------------|-------------------|---------------------|
| Whisper Turbo (809M) | 2.5% | 5.0% | 7.75% |
| Moonshine Medium (245M) | 2.08% | 5.0% | 6.65% |
| Moonshine Small (123M) | 2.49% | 6.78% | 7.84% |

Moonshine Medium achieves **better accuracy than Whisper Turbo with 3.3x fewer parameters**.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Per-layer sliding window may break existing CT2 abstractions | Start with uniform window, add per-layer as follow-up |
| Encoder KV caching is a new paradigm for CT2 | Phase 2 — defer if Phase 1 works well enough |
| CMVN normalization may drift during streaming | Use Moonshine's reference implementation |
| Moonshine is English-only (Medium variant) | Multilingual variants are in development (Tiny covers 8 languages) |
| Model converter weight mapping may have edge cases | Validate against ONNX Runtime golden outputs |

---

## 8. References

- [Moonshine Paper](https://arxiv.org/abs/2410.15608) — Jeffries et al., 2024
- [Moonshine GitHub](https://github.com/moonshine-ai/moonshine) — Official repo with iOS, Android, macOS examples
- [HuggingFace Model](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) — Weights + config
- [Moonshine ONNX](https://huggingface.co/Mazino0/moonshine-streaming-medium-onnx) — ONNX export reference
- [Flavors of Moonshine Paper](https://arxiv.org/abs/2509.02523) — Multilingual tiny models, 2025
