# M17.7 — End-to-End Validation

**Date:** 2026-03-21
**Status:** Complete — encoder output matches HuggingFace reference (max diff 3.3e-3)
**Tests:** 10/10 assertions pass

---

## What Was Tested

### Model Loading ✅
- CT2 model loads via `Model::load("/tmp/moonshine-tiny-ct2")`
- Correctly identified as `MoonshineModel` via `dynamic_cast`
- `MoonshineReplica` created successfully
- Frontend, encoder layers, decoder, adapter weights all loaded

### Frontend Output ✅ (exact match)
- CT2 frontend output matches HuggingFace reference to **5+ significant digits**
- First 5 values: CT2 `[-0.212987, -0.018252, 0.099077, -0.022723, 0.026664]` vs HF `[-0.21298657, -0.01825169, 0.09907554, -0.02272246, 0.02666389]`
- Shape: `[1, 50, 320]` for 1s audio — correct (50Hz output)
- CMVN, asinh compression, linear projection, SiLU, CausalConv1d — all correct

### Output Shape ✅
- 1s audio (16000 samples) → `[1, 50, 320]` ✓
- 5s audio (80000 samples) → `[1, 250, 320]` ✓
- Matches expected 50Hz output rate (250 frames / 5s = 50Hz)

### Encoder + Adapter Numerical Match ✅
- Max abs diff: **3.3e-3** (after 6 transformer layers + adapter — acceptable float32 accumulation)
- Mean abs diff: **7.9e-4**
- Layer 0 output: CT2 `mean=-0.017055` vs HF `mean=-0.017060` — near exact

## Bugs Found and Fixed During Testing

### Bug 1: Missing SiLU activation between Conv1d layers
The HF forward pass applies `silu()` after `conv1` and before `conv2`. The original CT2 implementation had no activation between convolutions. Fixed in `moonshine.cc`.

### Bug 2 (ROOT CAUSE): Wrong LayerNorm type + missing unit_offset
**The encoder's `MoonshineStreamingLayerNorm` is NOT RMSNorm.** It wraps standard `nn.LayerNorm(elementwise_affine=False)` and multiplies by `(gamma + unit_offset)` where `unit_offset=1.0`.

The spec initially set `rms_norm=True` based on the `.gamma`-only naming convention in the HF weights. But the actual computation is standard LayerNorm (mean subtraction + variance normalization), not RMS normalization.

**Fix applied:**
1. Changed encoder spec to `rms_norm=False`
2. In converter: save `gamma + 1.0` (apply unit_offset during conversion) and add zero `beta`

Before fix: max diff 6.5 (broken). After fix: max diff 3.3e-3 (correct).

## What Remains

- [ ] Full E2E text generation test (needs Python bindings — M17.8)
- [ ] WER on LibriSpeech test-clean
- [ ] INT8/float16 quantization validation
- [ ] Performance benchmarks (tokens/sec on CPU and MPS)

## Files Changed

| File | Description |
|------|-------------|
| `tests/metal/moonshine_e2e_test.mm` | E2E validation test (model load, encoder comparison, shape check) |
| `src/layers/moonshine.cc` | Fixed: added SiLU between conv1 and conv2 |
| `python/ctranslate2/specs/moonshine_spec.py` | Fixed: `rms_norm=False` for encoder |
| `python/ctranslate2/converters/moonshine.py` | Fixed: apply unit_offset to encoder norm gamma, add zero beta |

## Test Output

```
=== Moonshine E2E Validation Tests (M17.7) ===

  PASS: model_load
  Audio: 16000 samples
  CT2 encoder output: [1, 50, 320]
  HF reference: [1, 50, 320]
  Max abs diff: 3.255337e-03
  Mean abs diff: 7.859057e-04
  PASS: encoder_output
  5s audio → output shape: [1, 250, 320]
  PASS: encoder_shape_5s

=== Results: 10 passed, 0 failed ===
```

## Next Steps

- **M17.8:** Python bindings — enables full E2E text generation + WER validation
