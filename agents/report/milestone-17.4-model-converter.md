# M17.4 — Moonshine Model Converter (HuggingFace → CT2)

**Date:** 2026-03-21
**Status:** Complete
**Tests:** Full conversion of moonshine-streaming-tiny (float32 + float16), all validations pass

---

## What Was Implemented

`python/ctranslate2/converters/moonshine.py` — Converter that loads Moonshine Streaming models from HuggingFace and exports to CTranslate2 format.

### Usage
```python
from ctranslate2.converters.moonshine import MoonshineConverter

converter = MoonshineConverter("UsefulSensors/moonshine-streaming-medium")
converter.convert("moonshine-medium-ct2", quantization="float16")
```

## Verified Weight Mapping

Mapping verified against actual `moonshine-streaming-tiny` safetensors (161 tensors):

### Encoder Frontend
```
model.encoder.embedder.comp.log_k      → encoder/frontend/log_k          (scalar)
model.encoder.embedder.linear.weight    → encoder/frontend/linear/weight  [320, 80]
model.encoder.embedder.conv1.weight     → encoder/frontend/conv1/weight   [640, 320, 5]
model.encoder.embedder.conv1.bias       → encoder/frontend/conv1/bias     [640]
model.encoder.embedder.conv2.weight     → encoder/frontend/conv2/weight   [320, 640, 5]
model.encoder.embedder.conv2.bias       → encoder/frontend/conv2/bias     [320]
```

### Encoder Layers
```
model.encoder.layers.{i}.input_layernorm.gamma       → layer_{i}/self_attention/layer_norm/gamma
model.encoder.layers.{i}.self_attn.{q,k,v}_proj.w    → layer_{i}/self_attention/linear_0/weight  (FUSED Q+K+V)
model.encoder.layers.{i}.self_attn.o_proj.weight      → layer_{i}/self_attention/linear_1/weight
model.encoder.layers.{i}.post_attention_layernorm.gamma → layer_{i}/ffn/layer_norm/gamma
model.encoder.layers.{i}.mlp.fc1.weight               → layer_{i}/ffn/linear_0/weight  [4*H, H]
model.encoder.layers.{i}.mlp.fc1.bias                  → layer_{i}/ffn/linear_0/bias
model.encoder.layers.{i}.mlp.fc2.weight               → layer_{i}/ffn/linear_1/weight
model.encoder.layers.{i}.mlp.fc2.bias                  → layer_{i}/ffn/linear_1/bias
model.encoder.final_norm.gamma                         → encoder/layer_norm/gamma
```

### Adapter
```
model.decoder.pos_emb.weight           → adapter/position_embeddings/weight  [4096, dec_H]
model.decoder.proj.weight              → adapter/projection/weight           [enc_H, dec_H]  (only if enc_H ≠ dec_H)
```

### Decoder Layers
```
model.decoder.embed_tokens.weight      → decoder/embeddings/weight
model.decoder.norm.weight              → decoder/layer_norm/gamma  (+zero beta)
model.decoder.layers.{i}.input_layernorm.weight → layer_{i}/self_attention/layer_norm/gamma (+zero beta)
model.decoder.layers.{i}.self_attn.{q,k,v}_proj → layer_{i}/self_attention/linear_0  (FUSED Q+K+V)
model.decoder.layers.{i}.self_attn.o_proj       → layer_{i}/self_attention/linear_1
model.decoder.layers.{i}.post_attention_layernorm → layer_{i}/attention/layer_norm (+zero beta)
model.decoder.layers.{i}.encoder_attn.q_proj    → layer_{i}/attention/linear_0
model.decoder.layers.{i}.encoder_attn.{k,v}_proj → layer_{i}/attention/linear_1  (FUSED K+V)
model.decoder.layers.{i}.encoder_attn.o_proj    → layer_{i}/attention/linear_2
model.decoder.layers.{i}.final_layernorm        → layer_{i}/ffn/layer_norm (+zero beta)
model.decoder.layers.{i}.mlp.fc1.weight[:mid]   → layer_{i}/ffn/linear_0/weight       (SPLIT: value path)
model.decoder.layers.{i}.mlp.fc1.weight[mid:]   → layer_{i}/ffn/linear_0_noact/weight (SPLIT: gate path)
model.decoder.layers.{i}.mlp.fc2                → layer_{i}/ffn/linear_1
proj_out.weight                                  → decoder/projection/weight
```

## Key Findings During Implementation

### 1. Encoder does NOT use SwiGLU
The original proposal assumed the encoder uses SwiGLU like the decoder. **Wrong.** The encoder uses standard GELU FFN (`fc1` shape `[4*H, H]`). Only the decoder uses SwiGLU (`fc1` shape `[8*H, H]` = 2× intermediate for gate+value).

### 2. Encoder norms use `.gamma` (RMS-style), decoder norms use `.weight` (standard)
- Encoder: `input_layernorm.gamma`, `post_attention_layernorm.gamma`, `final_norm.gamma` — no beta
- Decoder: `input_layernorm.weight`, `post_attention_layernorm.weight`, `final_layernorm.weight`, `norm.weight` — needs zero beta for CT2's LayerNorm

### 3. Adapter projection is Identity when enc_hidden == dec_hidden
Moonshine Tiny has enc_hidden=dec_hidden=320, so `decoder.proj` is `nn.Identity()`. The converter detects this and skips setting the projection weight. The spec's `MoonshineAdapterSpec(project=False)` omits the `projection` field entirely.

### 4. SwiGLU fc1 split order
HuggingFace: `hidden_states, gate = fc1(x).chunk(2, dim=-1)`
- First half (rows 0..mid-1) = value path → `linear_0` (with SiLU activation)
- Second half (rows mid..end) = gate path → `linear_0_noact`

### 5. Tiny has partial_rotary_factor=0.8, not 0.5
The plan assumed 0.5 (from Medium config). Tiny uses 0.8 → rotary_dim=51 (0.8×64). The converter reads this from config correctly.

## Spec Changes (M17.3 updates)

- `MoonshineConfig` now extends `LanguageModelConfig` (not `ModelConfig`) — gets `bos_token`, `eos_token`, `unk_token` fields
- `MoonshineEncoderSpec` now uses `rms_norm=True` by default — encoder norms have gamma only
- `MoonshineAdapterSpec` accepts `project: bool` parameter — omits projection when enc_hidden == dec_hidden

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `python/ctranslate2/converters/moonshine.py` | +190 | Full converter implementation |
| `python/ctranslate2/specs/moonshine_spec.py` | ~5 | Fixed: MoonshineConfig → LanguageModelConfig; rms_norm=True for encoder; adapter project flag |

## Test Results

### moonshine-streaming-tiny (float32)
```
Converted to: /tmp/moonshine-tiny-ct2
  config.json: 72 bytes
  model.bin: 134,289,504 bytes
  vocabulary.json: 520,709 bytes
```

### moonshine-streaming-tiny (float16)
```
  config.json: 72 bytes
  model.bin: 67,150,944 bytes (0.50x of float32 — correct)
  vocabulary.json: 520,709 bytes
```

### Validation
- `spec.validate()`: PASS (all 161 weights mapped, vocabulary set)
- `spec.get_vocabulary_size()`: 32768
- Per-layer sliding windows: correctly read from config (`[[16,4],[16,4],[16,0],[16,0],[16,4],[16,4]]`)
- Encoder frontend: `log_k` scalar, `linear` [320,80], `conv1` [640,320,5], `conv2` [320,640,5]
- Decoder SwiGLU split: `linear_0` [1280,320], `linear_0_noact` [1280,320] (from `fc1` [2560,320])

## Not Yet Tested

- moonshine-streaming-medium (1GB download — not cached locally)
- int8 / int8_float16 quantization (requires CT2 model loading, M17.5)
- End-to-end inference with converted model (M17.7)

## Next Steps

- **M17.5:** MoonshineModel C++ class — loads this converted format and runs inference
