# M17.3 — MoonshineSpec Python Model Specification

**Date:** 2026-03-21
**Status:** Complete
**Tests:** 18 checks across 2 test scripts, all passing

---

## What Was Implemented

`python/ctranslate2/specs/moonshine_spec.py` — Python model specification for Moonshine Streaming ASR, defining the weight structure that the C++ model loader expects.

### Classes

| Class | Purpose | Fields |
|-------|---------|--------|
| `MoonshineSpec` | Top-level model spec (extends `LanguageModelSpec`) | `encoder`, `adapter`, `decoder` |
| `MoonshineEncoderSpec` | Encoder with audio frontend + transformer layers | `frontend`, `layer_norm`, `layer[]`, `num_heads`, `pre_norm`, `activation` |
| `MoonshineAudioFrontendSpec` | Audio preprocessing weights | `log_k`, `linear`, `conv1`, `conv2` |
| `MoonshineAdapterSpec` | Encoder→decoder bridge | `position_embeddings`, `projection` |
| `MoonshineConfig` | Model configuration | (inherits `ModelConfig`) |

### Key Design Decisions

**1. Reuse `TransformerDecoderSpec` for the decoder.** Moonshine's decoder is a standard Llama-style transformer with RoPE + SwiGLU + cross-attention — all features already supported by `TransformerDecoderSpec`. No custom decoder spec needed. Configured with:
- `activation=SWISH` (SiLU = Swish)
- `ffn_glu=True` (SwiGLU pattern: `linear_0` + `linear_0_noact`)
- `with_encoder_attention=True` (cross-attention to encoder)
- `rotary_dim=32` (partial_rotary_factor=0.5 × head_dim=64)
- `rotary_interleave=False` (Moonshine uses non-interleaved RoPE)
- `scale_embeddings=False` (no sqrt(d_model) scaling)

**2. Reuse `TransformerEncoderLayerSpec` for encoder layers.** Each encoder layer is a standard pre-norm transformer encoder layer. The per-layer sliding window is passed through to `MultiHeadAttentionSpec` via the existing `sliding_window` parameter (which now accepts `[left, right]` tuples from M17.1).

**3. Per-layer sliding window via constructor loop.** Instead of the existing pattern of creating identical layers in a list comprehension, `MoonshineEncoderSpec` uses a for-loop to pass different `sliding_window` values per layer. Validated: length must equal `num_layers`.

**4. SwiGLU weight split is a converter concern.** Moonshine's HF model uses a fused `fc1` weight `[2*intermediate, hidden]`. The spec declares separate `linear_0` (value) and `linear_0_noact` (gate) following CT2's existing GLU pattern. The converter (M17.4) will split the fused weight during conversion.

**5. Adapter as a separate spec.** The adapter (position embeddings + linear projection) is a novel Moonshine component not found in Whisper. Stored under `spec.adapter` with `EmbeddingsSpec` for position embeddings and `LinearSpec` for the projection.

## Weight Structure

When serialized, the model will have this weight tree:

```
encoder/
  frontend/
    log_k                           # scalar float
    linear/weight, linear/bias      # [hidden, 80], [hidden]
    conv1/weight, conv1/bias        # [hidden*2, hidden, 5], [hidden*2]
    conv2/weight, conv2/bias        # [hidden, hidden*2, 5], [hidden]
  layer_norm/gamma, layer_norm/beta
  layer_{i}/
    self_attention/
      layer_norm/gamma, layer_norm/beta
      linear_0/weight               # fused QKV or Q projection
      linear_1/weight               # K/V or separate KV
      sliding_window                 # int32 (left window size)
      sliding_window_right           # int32 (right window size, 0 = causal)
    ffn/
      layer_norm/gamma, layer_norm/beta
      linear_0/weight, linear_0/bias
      linear_1/weight, linear_1/bias

adapter/
  position_embeddings/weight        # [max_pos, dec_hidden]
  projection/weight                 # [enc_hidden, dec_hidden] (no bias)

decoder/
  embeddings/weight                 # [vocab, dec_hidden]
  layer_norm/gamma, layer_norm/beta
  projection/weight                 # [dec_hidden, vocab]
  layer_{i}/
    self_attention/                  # with RoPE
      ...
    attention/                       # cross-attention to encoder
      ...
    ffn/                            # SwiGLU: linear_0 + linear_0_noact + linear_1
      ...
```

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `python/ctranslate2/specs/moonshine_spec.py` | +180 | Full model specification |

## Test Results

### Primary test (Moonshine Medium configuration):
```
Spec name: MoonshineSpec
Spec revision: 1
Encoder layers: 14
Decoder layers: 14
Encoder num_heads: 10
Encoder has frontend: True
Has adapter: True
Layer  0: sw=16, swr=4   (bidirectional)
Layer  1: sw=16, swr=4   (bidirectional)
Layer  2: sw=16, swr=0   (causal)
  ...
Layer 11: sw=16, swr=0   (causal)
Layer 12: sw=16, swr=4   (bidirectional)
Layer 13: sw=16, swr=4   (bidirectional)
Decoder has cross-attention: True
Decoder has SwiGLU: True
Decoder rotary_dim: 32
Frontend has log_k: True
```

### Edge case tests:
- Moonshine Tiny (6 layers): PASS
- No sliding window: PASS (no `sliding_window` attribute set)
- Wrong `sliding_windows` length: PASS (raises `ValueError`)
- Validate on unset spec: PASS (raises `ValueError` for unset weights)

## Backward Compatibility

- No existing code modified — new file only
- Reuses existing `TransformerDecoderSpec`, `TransformerEncoderLayerSpec`, `MultiHeadAttentionSpec`
- The `sliding_window=[left, right]` tuple feature was added in M17.1

## Next Steps

- **M17.4:** Model converter — maps HuggingFace weight names to this spec structure, handles fused SwiGLU weight split
- **M17.5:** C++ MoonshineModel/MoonshineReplica — loads this spec and runs inference
