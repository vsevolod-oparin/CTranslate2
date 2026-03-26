# Release Notes — metal-dev-0.3

**Base:** CTranslate2 4.7.1 upstream
**Supported platforms:** macOS 14+ (Apple Silicon, arm64)
**Python:** 3.9 – 3.14

---

## What's New

### Moonshine ASR Support

Full end-to-end support for [Moonshine](https://github.com/usefulsensors/moonshine) — UsefulSensors' compact, high-accuracy speech recognition model optimized for real-time on-device inference.

- **Audio frontend** (`MoonshineAudioFrontend`): per-frame CMVN normalization, learned asinh compression (`log_k`), causal Conv1d padding
- **Bidirectional sliding window attention** (`[left, right]` window pairs, used in Moonshine's boundary encoder layers)
- **Model converter**: `ctranslate2.converters.MoonshineConverter` — converts HuggingFace Moonshine checkpoints to CT2 format
- **Python API**: `ctranslate2.Moonshine` class with `.generate()` interface compatible with faster-whisper patterns
- **End-to-end validated**: greedy and beam search accuracy verified against HuggingFace reference on real audio

Convert and run a Moonshine model:

```python
import ctranslate2, tokenizers

ct2_model = ctranslate2.converters.MoonshineConverter("usefulsensors/moonshine-base", device="mps")
ct2_model.convert("moonshine-base-ct2", quantization="float16")

model = ctranslate2.Moonshine("moonshine-base-ct2", device="mps", compute_type="float16")
tokenizer = tokenizers.Tokenizer.from_pretrained("usefulsensors/moonshine-base")

results = model.generate(audio)  # audio: list[float], 16 kHz
print(tokenizer.decode(results[0].sequences_ids[0]))
```

### Bug Fixes

**MPS stale data corruption in F16 GEMM** (`src/metal/primitives_gemm.mm`, `src/metal/ops_sdpa.mm`):
The `row_copy` MSL kernel used 4-byte (`uint`) pointer casts to copy matrix rows, but MPS row strides
are only guaranteed 2-byte aligned (e.g. 205 float16 columns = 410 bytes). Misaligned `uint*` casts
on Metal GPUs cause silent write failures, leaving stale NaN values from pooled buffers in the
destination. Fix: switched to 2-byte (`ushort`) copies, always safe for float16 data. Additionally,
cached temp buffers for F16 GEMM accumulation are now zeroed on reuse to prevent stale data from a
larger prior allocation corrupting a smaller one. This manifested as incorrect results when switching
from long to short sequences (e.g. beam-search 3000 frames → greedy 200 frames).

**Allocator `bucket_size` integer overflow** (`src/metal/allocator.mm`):
For `requested > 2^63`, the next-power-of-2 bit manipulation fills all 64 bits and `v + 1` wraps to
zero, causing `newBufferWithLength:0` to be called. Added an explicit guard that throws before the
overflow occurs.

**CMVN division by zero** (`src/layers/moonshine.cc`):
`apply_cmvn` with a zero-length feature dimension computed `0.0 / 0.0 = NaN` as an intermediate.
Added an early `continue` guard.

### Flash Attention Enabled on MPS

`use_flash_attention=True` is now recognized on `Device::MPS` — previously the device check only
covered CUDA/HIP, so the option was silently ignored. Flash attention is now correctly enabled for
MPS and is the recommended mode for long-context inference.

### Library Renamed

The shared library is now `libctranslate2.mps` (was `libctranslate2`) and the Python package is
`ctranslate2-mps`. This avoids conflicts with the upstream CPU-only `ctranslate2` package when both
are installed in the same environment.

### iOS Cross-Compilation

CMake now supports building for iOS targets (`CMAKE_SYSTEM_NAME=iOS`), with the deployment target
correctly set for both macOS and iOS builds.

### Adversarial Test Suite

New test files covering allocator edge cases and frontend numerical corner cases:

- `tests/metal/adversarial_allocator_test.mm` (24 tests): interior pointer lookup, double free,
  SIZE_MAX overflow, protect/flush lifecycle, concurrent alloc/free, pointer cache invalidation
- `tests/metal/adversarial_frontend_test.mm` (25 tests): zero-length sequences, zero windows,
  cross-attention shapes, NaN/Inf/FLT_MAX inputs to CMVN, asinh IEEE 754 edge values

---

## Assets

| File | Description |
|------|-------------|
| `ctranslate2_mps-4.7.1-cp39-cp39-macosx_14_0_arm64.whl` | Python 3.9 wheel |
| `ctranslate2_mps-4.7.1-cp310-cp310-macosx_14_0_arm64.whl` | Python 3.10 wheel |
| `ctranslate2_mps-4.7.1-cp311-cp311-macosx_14_0_arm64.whl` | Python 3.11 wheel |
| `ctranslate2_mps-4.7.1-cp312-cp312-macosx_14_0_arm64.whl` | Python 3.12 wheel |
| `ctranslate2_mps-4.7.1-cp313-cp313-macosx_14_0_arm64.whl` | Python 3.13 wheel |
| `ctranslate2_mps-4.7.1-cp314-cp314-macosx_14_0_arm64.whl` | Python 3.14 wheel |
| `libctranslate2.mps.4.7.1.dylib` | Shared library (macOS arm64) |
| `libctranslate2.mps.a` | Static library (macOS arm64) |
