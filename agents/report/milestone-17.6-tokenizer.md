# M17.6 — Tokenizer Integration

**Date:** 2026-03-21
**Status:** Complete
**Tests:** All checks pass — vocabulary round-trip, special tokens, config alignment

---

## What Was Verified

Moonshine's tokenizer integrates with CT2's existing vocabulary system with **zero code changes**. The converter (M17.4) already saves the vocabulary correctly.

### Tokenizer Architecture

Moonshine uses a **HuggingFace `PreTrainedTokenizerFast`** with a standard BPE model:

| Property | Value |
|----------|-------|
| Type | BPE (tokenizers library) |
| Base vocab | 32,000 tokens |
| Added tokens | 768 special tokens (`<<ST_0>>` through `<<ST_767>>`) |
| Total vocab | 32,768 |
| Format | `tokenizer.json` (HuggingFace fast tokenizer) |
| BOS | `<s>` (ID 1) |
| EOS | `</s>` (ID 2) |
| UNK | `<unk>` (ID 0) |
| PAD | `<unk>` (ID 0, same as UNK) |

### How CT2 Handles Tokenization

CT2 does **not** perform tokenization internally. The pattern (same as Whisper via faster-whisper):

1. **Conversion time**: The converter extracts the vocabulary as a flat list of token strings, saved as `vocabulary.json` in the model directory
2. **C++ load time**: `MoonshineModel::initialize()` loads `vocabulary.json` into a `Vocabulary` object (ID→token and token→ID mappings)
3. **Inference time**: The caller (Python API or application) uses the HuggingFace tokenizer to encode input text into IDs and decode output IDs back to text
4. **CT2 internal**: The `Vocabulary` object maps IDs to tokens for the generate() output

### Integration Points Verified

**1. Vocabulary file format**
- Saved as `vocabulary.json`: flat JSON array of 32768 strings
- IDs are implicit (array index = token ID)
- CT2's `_save_vocabulary` / `load_vocabulary` handles this format natively

**2. Token ID alignment**
```
HF tokenizer ID ↔ CT2 vocabulary index
    0 = <unk>     ✓
    1 = <s>        ✓ (BOS)
    2 = </s>       ✓ (EOS)
  3-255 = byte tokens (<0x00> through <0xFF>)
  256-31999 = BPE tokens (▁Hello, ▁world, etc.)
  32000-32767 = special tokens (<<ST_0>> through <<ST_767>>)
```

**3. Encode/decode round-trip**
```
"Hello world"        → [15043, 3186]        → ['▁Hello', '▁world']        → "Hello world"     ✓
"The quick brown fox" → [450, 4996, 17354, 1701, 29916] → ['▁The', '▁quick', '▁brown', '▁fo', 'x'] ✓
"Testing one two three" → [4321, 292, 697, 1023, 2211]  → ['▁Test', 'ing', '▁one', '▁two', '▁three'] ✓
```

**4. Special token alignment with C++ model**
- `MoonshineModel::initialize()` sets `unk_token="<unk>"`, `bos_token="<s>"`, `eos_token="</s>"`
- `MoonshineReplica` reads `_bos_id = vocabulary.bos_id()` (=1), `_eos_id = vocabulary.eos_id()` (=2)
- Config `config.json` stores matching values

**5. C++ model loads vocabulary successfully**
- `ext.Translator('/tmp/moonshine-tiny-ct2')` → "cannot be used as seq2seq" (model + vocab loaded, type check rejects)
- `ext.Whisper('/tmp/moonshine-tiny-ct2')` → "not a Whisper model" (model + vocab loaded, type check rejects)
- No "vocabulary" or "cannot load" errors — vocabulary loads correctly

## Files Changed

None. Zero code changes required — existing infrastructure handles everything.

## Key Finding: No `tokenizer.json` Needed in CT2 Model Directory

Unlike HuggingFace models which ship `tokenizer.json` for BPE tokenization, CT2 only needs `vocabulary.json` (the flat token list). The actual BPE tokenization is done externally by the caller using the HuggingFace tokenizer. This matches the Whisper/faster-whisper pattern.

For a Moonshine Python API wrapper (M17.8), tokenization would be handled like:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("UsefulSensors/moonshine-streaming-tiny")

# Encode: tokenizer handles BPE
prompt_ids = tokenizer.encode("<s>", add_special_tokens=False)

# Decode: tokenizer handles detokenization
text = tokenizer.decode(output_ids, skip_special_tokens=True)
```

## Next Steps

- **M17.7:** End-to-end accuracy validation
- **M17.8:** Python API with tokenizer integration
