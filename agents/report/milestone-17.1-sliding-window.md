# M17.1 — Per-Layer Bidirectional Sliding Window Attention

**Date:** 2026-03-21
**Status:** Complete
**Build:** PASS (libctranslate2.mps.dylib)
**Tests:** 84/84 assertions, 7 test cases, 0 failures

---

## Problem

CTranslate2's sliding window attention had two limitations blocking Moonshine support:

1. **Per-model, not per-layer.** Stored once on `TransformerEncoderSpec` / `TransformerDecoderSpec`, applied uniformly to all layers. Moonshine needs per-layer config: layers 0-1 and 12-13 use `[16, 4]` (with lookahead), layers 2-11 use `[16, 0]` (causal).

2. **Past-only.** Implemented as KV cache trimming (`ops::Slide` on cached keys/values after they exceed `_sliding_window` length). No support for attending to future frames. Moonshine's `[16, 4]` means "attend to 16 past + 3 future frames" — a bidirectional window.

Additionally, the existing implementation only applied to **decoder** KV caches. Moonshine's sliding window is on the **encoder** (no KV cache in offline mode).

## Design

### Architecture audit (pre-implementation)

| Backend | FlashAttention sliding window | Attention mask tensor |
|---------|-------------------------------|----------------------|
| CPU | N/A (uses `dot_product_attention`) | Works via existing mask path |
| CUDA | Supports `window_size_left/right` natively | Works via existing mask path |
| MPS | **Throws exception** if `sliding_window > 0` | Works via existing mask path |

**Decision:** Implement as an additive attention mask tensor (not KV cache manipulation, not FlashAttention parameters). This is the only approach that works on all three backends.

### Mask generation

`make_sliding_window_mask(seq_q, seq_k, left_window, right_window, dtype, device)` produces a `[1, seq_q, seq_k]` float tensor:
- `0.0` for positions within the window (valid attention)
- `-1e9` for positions outside the window (effectively masked after softmax)

Window semantics (matching HuggingFace's `modeling_moonshine_streaming.py`):
- `left_window`: query can attend to keys where `dist = q - k` satisfies `0 <= dist < left_window`
- `right_window`: query can attend to keys where `dist = q - k` satisfies `dist < 0 && -dist < right_window`
- `[16, 4]` → self + 15 past + 3 future = 19 positions per query
- `[16, 0]` → self + 15 past = 16 positions per query (causal)
- `[1, 0]` → self only (diagonal mask)

The mask is created on CPU then moved to the target device. Shape `[1, seq_q, seq_k]` broadcasts over the `batch * num_heads` dimension via `add_batch_broadcast`.

### Application point

The mask is added to attention scores (`Q @ K^T`) **before** softmax, inside `dot_product_attention()`. This is the standard approach for attention masking — `-inf` values become 0 after softmax.

```
scores = Q @ K^T * scale
scores += relative_position_bias  (if any)
scores += alibi                   (if any)
scores += sliding_window_mask     (if any)  ← NEW
attn = softmax(scores, lengths)
output = attn @ V
```

### Activation conditions

The mask is only generated when ALL of these are true:
- `_sliding_window > 0` (layer has a sliding window configured)
- `_self_attention` (not cross-attention)
- `!_is_decoder` (encoder layer — decoder uses KV cache trimming instead)
- `!cached_keys` (no KV cache — confirms it's an encoder forward pass)

This ensures:
- Existing decoder sliding window (Mistral, Gemma3) is unaffected — still uses KV cache trimming
- Cross-attention is unaffected
- Encoder layers without sliding window are unaffected

## Files Changed

| File | Lines | Change |
|------|-------|--------|
| `python/ctranslate2/specs/attention_spec.py` | +5 | Support `sliding_window=[left, right]` tuple; stores `sliding_window_right` int32 attribute when bidirectional |
| `include/ctranslate2/layers/attention_layer.h` | +1 | Added `const dim_t _sliding_window_right` member |
| `src/layers/attention_layer.cc` | +1 | Read `sliding_window_right` from model scope (default 0) |
| `src/layers/attention.cc` | +40 | `make_sliding_window_mask()` function, mask generation in `MultiHeadAttention::operator()`, `sliding_window_mask` parameter in `dot_product_attention`, mask application before softmax |
| `tests/metal/sliding_window_test.mm` | +195 | 7 test cases, 84 assertions |

## Test Results

```
=== Sliding Window Mask Tests (M17.1) ===

  PASS: causal_window_4_0           — 8 assertions
  PASS: bidirectional_window_16_4   — 8 assertions
  PASS: small_window_2_1            — 12 assertions
  PASS: symmetric_window_3_3        — 6 assertions
  PASS: mask_shape                  — 5 assertions
  PASS: full_window_covers_all      — 25 assertions
  PASS: self_only_window_1_0        — 20 assertions

=== Results: 84 passed, 0 failed ===
```

**Build command:**
```bash
clang++ -std=c++17 -O0 \
    -I include -I src -DCT2_WITH_METAL \
    tests/metal/sliding_window_test.mm \
    -L build -lctranslate2.mps \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
    -Wl,-rpath,build \
    -o sliding_window_test && ./sliding_window_test
```

## Backward Compatibility

- `sliding_window_right` defaults to 0 in both Python spec and C++ constructor
- When `sliding_window_right == 0`, no mask is generated — existing behavior preserved
- Existing models (Mistral, Gemma3) set `sliding_window` as a single int → stored as `sliding_window` only, no `sliding_window_right` attribute → C++ reads default 0 → no mask generated → decoder KV cache trimming works as before
- Python spec accepts both `sliding_window=1024` (backward compatible) and `sliding_window=[16, 4]` (new tuple syntax)

## Device Agnosticism

Verified by design:
- `make_sliding_window_mask` generates mask on CPU, moves to target device via `StorageView::to(device)`
- Mask application uses `primitives<D>::add_batch_broadcast` via `DEVICE_AND_TYPE_DISPATCH` — dispatches to CPU/CUDA/MPS automatically
- Zero `#ifdef CT2_WITH_*` guards in the sliding window code
- No device-specific code paths

## Performance Notes

- Mask is `[1, seq_q, seq_k]` float32 — for Moonshine's 50Hz encoder on 30s audio: `[1, 1500, 1500]` = 9 MB. Acceptable.
- Mask is regenerated on every `MultiHeadAttention::operator()` call. For encoder (called once per input), this is fine.
- `-1e9` used instead of `-inf` for fp16 safety (avoids NaN in edge cases).

## Not Yet Tested

- Integration with actual Moonshine model (requires M17.3+ model class)
- Cross-device output comparison (CPU vs MPS) — requires model
- Per-layer varying windows in a single encoder (requires Moonshine converter M17.4)

## Next Steps

- **M17.2:** Audio frontend (CausalConv1d, CMVN, asinh compression)
- **M17.3:** MoonshineSpec Python model specification
- **M17.4:** Model converter (HuggingFace → CT2)
