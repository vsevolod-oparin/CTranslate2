# M17.2 — Moonshine Audio Frontend

**Date:** 2026-03-21
**Status:** Complete
**Build:** PASS (libctranslate2.mps.dylib, zero warnings from moonshine.cc)
**Tests:** 40/40 assertions, 13 test cases, 0 failures

---

## What Was Implemented

`MoonshineAudioFrontend` — a new CT2 Layer class that converts raw 16kHz waveform to encoder input features. Follows the HuggingFace `MoonshineStreamingPreprocessor` architecture.

### Pipeline

```
raw audio [batch, samples]
  │
  ▼  Frame into [batch, num_frames, 80]  (5ms frames at 16kHz)
  │
  ▼  Per-frame CMVN: (x - mean) / sqrt(var + eps)
  │
  ▼  Asinh compression: asinh(x * exp(log_k))  [log_k is learned]
  │
  ▼  Linear(80 → hidden_size) via Dense layer
  │
  ▼  SiLU activation
  │
  ▼  Transpose to [batch, hidden, time]
  │
  ▼  CausalConv1d(hidden → hidden*2, k=5, s=2)  [left-pad by 4]
  │
  ▼  CausalConv1d(hidden*2 → hidden, k=5, s=2)  [left-pad by 4]
  │
  ▼  Transpose to [batch, time/4, hidden]
  │
  output [batch, num_frames/4, hidden_size]  (50Hz features)
```

### Key Design Decisions

**1. CMVN and asinh on CPU.** These operate on small data (80 floats per frame) and run once per input. GPU dispatch overhead would dominate. The data lives in shared memory (`MTLResourceStorageModeShared`), so CPU access is zero-copy on Apple Silicon.

**2. CausalConv1d via manual left-padding + Conv1D(padding=0).** CT2's Conv1D uses symmetric padding (`2 * padding`). Causal convolution requires left-only padding. Implementation: create a zero tensor `[batch, channels, kernel_size-1]`, concat with input on dim 2, then Conv1D with `padding=0`. Uses existing `ops::Concat` — device-agnostic.

**3. SiLU applied separately from Dense.** CT2's `Dense` layer wrapper doesn't expose activation as a constructor parameter (unlike `Conv1D`). Applied via `ops::get_activation_op(ActivationType::Swish)` after the Dense forward pass. SiLU = Swish = `x * sigmoid(x)`.

**4. Framing by reshape (no copy).** Raw waveform `[batch, samples]` is reshaped to `[batch, num_frames, frame_size]` — this is pointer arithmetic, no data movement. Trailing samples that don't fill a full frame are truncated (matching HF behavior).

**5. Asinh decomposition not needed.** The plan suggested decomposing `asinh(x) = log(x + sqrt(x² + 1))`. Instead, used `std::asinhf()` directly since the operation runs on CPU. Simpler and exact.

### Output Shape

For 5 seconds of 16kHz audio:
- Input: `[1, 80000]`
- After framing: `[1, 1000, 80]`
- After CMVN + asinh + Linear + SiLU: `[1, 1000, hidden_size]`
- After CausalConv1d #1 (s=2): `[1, hidden*2, 500]`
- After CausalConv1d #2 (s=2): `[1, hidden, 250]`
- After transpose: `[1, 250, hidden_size]` → 50Hz output (250 frames / 5s)

## Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `include/ctranslate2/layers/moonshine.h` | +55 | `MoonshineAudioFrontend` class declaration |
| `src/layers/moonshine.cc` | +155 | Full implementation: CMVN, asinh, framing, Conv1d pipeline |
| `CMakeLists.txt` | +1 | Added `src/layers/moonshine.cc` to SOURCES |
| `tests/metal/moonshine_frontend_test.mm` | +240 | 13 test cases, 40 assertions |

## Model Weights Required

The frontend loads these weights from the model scope:

| Weight | Shape | Description |
|--------|-------|-------------|
| `{scope}/log_k` | scalar | Asinh compression parameter (learned, initialized to 0.75) |
| `{scope}/linear/weight` | `[hidden_size, 80]` | Frame-to-feature projection |
| `{scope}/linear/bias` | `[hidden_size]` | Frame-to-feature bias |
| `{scope}/conv1/weight` | `[hidden*2, hidden, 5]` | CausalConv1d #1 kernel |
| `{scope}/conv1/bias` | `[hidden*2]` | CausalConv1d #1 bias |
| `{scope}/conv2/weight` | `[hidden, hidden*2, 5]` | CausalConv1d #2 kernel |
| `{scope}/conv2/bias` | `[hidden]` | CausalConv1d #2 bias |

These will be populated by the Python model converter (M17.4).

## Test Results

```
=== Moonshine Audio Frontend Tests (M17.2) ===

  PASS: cmvn_zero_mean         — verifies mean ≈ 0 after normalization
  PASS: cmvn_unit_variance     — verifies RMS ≈ 1 after normalization
  PASS: cmvn_multi_frame       — independent normalization per frame
  PASS: cmvn_constant_frame    — constant input → output ≈ 0 (eps prevents NaN)
  PASS: cmvn_realistic         — 80-sample frame with DC offset + sine wave
  PASS: asinh_identity          — asinh(1) with log_k=0
  PASS: asinh_with_scale        — asinh(x * exp(0.75))
  PASS: asinh_zero              — asinh(0) = 0
  PASS: asinh_negative          — asinh is odd: asinh(-x) = -asinh(x)
  PASS: causal_pad              — left-pad [1,2,3] → [0,0,0,0,1,2,3] on dim 2
  PASS: framing                 — 240 samples → 3 frames of 80, boundary check
  PASS: framing_truncation      — 250 samples → 3 frames (10 truncated)
  PASS: output_shape            — conv1d stride math: 1000 → 500 → 250 (50Hz)

=== Results: 40 passed, 0 failed ===
```

**Build command:**
```bash
clang++ -std=c++17 -O0 \
    -I include -I src -DCT2_WITH_METAL \
    tests/metal/moonshine_frontend_test.mm \
    -L build -lctranslate2.mps \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
    -Wl,-rpath,build \
    -o moonshine_frontend_test && ./moonshine_frontend_test
```

## Device Agnosticism

- CMVN and asinh: CPU-side on shared memory (zero-copy on Apple Silicon; explicit sync on discrete GPU)
- Dense, SiLU, Conv1D: all dispatch via `DEVICE_AND_FLOAT_DISPATCH` — work on CPU, CUDA, MPS
- Causal padding: uses `ops::Concat` — device-agnostic
- Transpose: uses `ops::Transpose` — device-agnostic
- Zero `#ifdef CT2_WITH_*` guards in the implementation

## Not Yet Tested

- Full end-to-end pipeline (requires model weights — M17.4)
- fp16/bf16 paths (CMVN/asinh run in fp32; Dense/Conv dispatch by dtype)
- Comparison against HuggingFace reference output (M17.7)

## Next Steps

- **M17.3:** MoonshineSpec Python model specification
- **M17.4:** Model converter (HuggingFace → CT2) — will populate the weight names above
