# M17.5 — MoonshineModel + MoonshineReplica C++ Model Class

**Date:** 2026-03-21
**Status:** Complete
**Build:** PASS (libctranslate2.mps.dylib, zero errors)
**Model load:** PASS (MoonshineSpec recognized, model instantiated)

---

## What Was Implemented

C++ model infrastructure for Moonshine: `MoonshineModel`, `MoonshineReplica`, and `Moonshine` pool class, following the Whisper pattern.

### Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `MoonshineModel` | Model loading, vocabulary, weight management | `include/ctranslate2/models/moonshine.h` |
| `MoonshineReplica` | Per-thread inference: encode + generate | `src/models/moonshine.cc` |
| `Moonshine` | Thread-safe pool wrapping `MoonshineReplica` | Both files |
| `MoonshineOptions` | Beam search, sampling, length parameters | Header |
| `MoonshineResult` | Output: sequences, IDs, scores | Header |

### Architecture

```
MoonshineReplica::encode(audio)
  │
  ▼  MoonshineAudioFrontend (M17.2)
  │    raw waveform → [batch, time/4, enc_hidden]
  │
  ▼  TransformerEncoder layers (reused, bypassing embedding layer)
  │    iterate layers manually: layer(input, nullptr, output)
  │    apply output norm
  │    → [batch, time/4, enc_hidden]
  │
  ▼  Adapter
  │    + position embeddings (broadcast add)
  │    + linear projection (if enc_hidden ≠ dec_hidden)
  │    → [batch, time/4, dec_hidden]
  │
  output

MoonshineReplica::generate(audio, prompts, options)
  │
  ▼  maybe_encode(audio)  — encode if not already encoded
  ▼  state["memory"] = encoder_output
  ▼  decode() — standard CT2 beam search / greedy
  ▼  Convert DecodingResult → MoonshineResult
  │
  output: sequences + scores
```

### Key Design Decisions

**1. Bypass TransformerEncoder's embedding path.** `TransformerEncoder::operator()` expects `vector<StorageView>` token IDs as input and runs them through embeddings + position encoder. Moonshine's encoder input comes from the audio frontend, not embeddings. Solution: added `get_layers()` and `get_output_norm()` public accessors to `TransformerEncoder`, and iterate layers directly in `MoonshineReplica::encode()`.

**2. Adapter as raw weight access, not a Layer subclass.** The adapter is two operations (position embedding add + optional linear projection). Using `get_variable_if_exists` to access weights directly is simpler than creating a new Layer subclass for 10 lines of logic.

**3. Standard decode() API for generation.** Moonshine's decoder is a standard `TransformerDecoder` — the existing `decode()` function handles beam search, greedy, sampling, etc. No custom decoding logic needed (unlike Whisper which has timestamps, language detection, alignment).

**4. MPS sync between encoder and decoder.** Following the Whisper pattern (M12.8 fix): `synchronize_stream(device)` after encoding on MPS to prevent command buffer overflow with deep models.

**5. Separate from Whisper.** `MoonshineReplica` does NOT extend `WhisperReplica`. Moonshine has no timestamps, no language tokens, no 30s chunking, no alignment — keeping them separate avoids complexity.

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `include/ctranslate2/models/moonshine.h` | +120 | Model, Replica, Pool class declarations + Options/Result structs |
| `src/models/moonshine.cc` | +290 | Full implementation: encode, adapter, generate, pool methods |
| `src/models/model_factory.cc` | +2 | Register `MoonshineModel` for `"MoonshineSpec"` |
| `include/ctranslate2/layers/transformer.h` | +9 | Added `get_layers()` and `get_output_norm()` public accessors |
| `CMakeLists.txt` | +1 | Added `src/models/moonshine.cc` to SOURCES |

## Test Results

### Build
```
[100%] Built target ctranslate2
```
Zero errors, zero warnings from moonshine.cc.

### Model Registration
```python
>>> ctranslate2.Translator('/tmp/moonshine-tiny-ct2', device='cpu')
RuntimeError: This model cannot be used as a sequence-to-sequence model
```
Model is recognized (no "Unknown model" error) — correctly rejects use as a Translator since Moonshine needs its own Python class.

### What's NOT Tested Yet
- End-to-end inference (requires Python bindings — M17.8)
- Encoder output correctness vs HuggingFace reference (M17.7)
- Multi-device (CPU vs MPS) output comparison
- Adapter projection path (needs Medium model — Tiny has enc_hidden == dec_hidden)

## Implementation Notes

### Adapter Position Embeddings
The adapter adds learned position embeddings to the encoder output before optional projection:
```cpp
// Slice pos_emb to [time, hidden], reshape to [1, time, hidden]
// Broadcast add to encoder_output [batch, time, hidden]
primitives<D>::add_batch_broadcast(pos_emb, result, pos_emb_size, result_size)
```

### Encoder Layer Iteration
Since TransformerEncoder bundles embedding + position encoding + layers, and Moonshine doesn't use embeddings, we bypass the `operator()` and iterate layers directly:
```cpp
const auto& layers = _encoder->get_layers();
for (const auto& layer : layers) {
    (*layer)(input, nullptr, encoder_output);
    input = std::move(encoder_output);
}
_encoder->get_output_norm()(input, encoder_output);
```

### Vocabulary
Moonshine uses simple BPE tokens: BOS=`<s>` (id=1), EOS=`</s>` (id=2), UNK=`<unk>` (id=0). No special Whisper-style tokens (timestamps, languages, etc).

## Next Steps

- **M17.6:** Tokenizer integration — verify CT2 can load Moonshine's `tokenizer.json`
- **M17.7:** End-to-end accuracy validation — compare against HuggingFace reference
- **M17.8:** Python API — `ctranslate2.models.Moonshine` Python bindings
