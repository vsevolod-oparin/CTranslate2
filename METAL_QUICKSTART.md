# CTranslate2 Metal Backend — Quick Start Guide

GPU-accelerated Whisper transcription on Apple Silicon (M1–M4) using the Metal backend.

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/)
- Xcode Command Line Tools: `xcode-select --install`
- ~5 GB disk space (model + dependencies)

## Directory Structure

We'll set up everything under one parent folder:

```
whisper-metal/
├── CTranslate2/          # source code (cloned from git)
├── models/               # converted Whisper models
└── workspace/            # your scripts and audio files
    ├── audio.mp3
    └── transcribe.py
```

## Step 1 — Create the project folder

```bash
mkdir -p whisper-metal && cd whisper-metal
```

## Step 2 — Create a conda environment

```bash
conda create -n whisper-metal python=3.12 -y
conda activate whisper-metal
```

> You must run `conda activate whisper-metal` in every new terminal session.

## Step 3 — Install Python dependencies

```bash
pip install torch transformers sentencepiece faster-whisper
```

## Step 4 — Clone and build CTranslate2

```bash
git clone -b metal-backend https://github.com/vsevolod-oparin/CTranslate2.git
cd CTranslate2
git submodule update --init --recursive

# Generate Metal shader headers
python3 tools/gen_msl_strings.py

# Configure
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_METAL=ON -DWITH_ACCELERATE=ON -DWITH_MKL=OFF -DWITH_DNNL=OFF -DOPENMP_RUNTIME=NONE -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)")
cd ..

# Build and install the C++ library
cmake --build build -j$(sysctl -n hw.logicalcpu)
cmake --install build

# Install Python bindings
cd python
pip install .
cd ../..
```

Verify the installation:

```bash
python -c "import ctranslate2; print('CTranslate2', ctranslate2.__version__)"
```

## Step 5 — Convert a Whisper model

```bash
mkdir -p models

ct2-transformers-converter \
    --model openai/whisper-large-v3-turbo \
    --output_dir models/whisper-large-v3-turbo \
    --quantization float16 \
    --copy_files tokenizer.json preprocessor_config.json
```

This downloads the model from Hugging Face (~3 GB), converts it, and saves to `models/`.

> **Important:** `--copy_files tokenizer.json preprocessor_config.json` is required.
> Without `tokenizer.json`, word error rate degrades severely.
> Without `preprocessor_config.json`, the feature extractor uses wrong mel bin count.

Other model choices:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `openai/whisper-large-v3-turbo` | ~1.6 GB | Fastest | Great |
| `openai/whisper-large-v3` | ~3.1 GB | Slower | Best |
| `openai/whisper-base` | ~150 MB | Very fast | Good for testing |

## Step 6 — Transcribe audio

Create the workspace and add your audio file:

```bash
mkdir -p workspace
cp /path/to/your/audio.mp3 workspace/audio.mp3
```

Create `workspace/transcribe.py`:

```python
import argparse
import os
import time
from faster_whisper import WhisperModel

parser = argparse.ArgumentParser(description="Transcribe audio with CTranslate2 Whisper")
parser.add_argument("--device", default="mps", choices=["mps", "cpu"],
                    help="Device to use: mps (Metal GPU) or cpu (default: mps)")
parser.add_argument("--audio", default=os.path.join(os.path.dirname(__file__), "audio.mp3"),
                    help="Path to audio file")
args = parser.parse_args()

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "whisper-large-v3-turbo")
compute_type = "float16" if args.device == "mps" else "float32"

print(f"Loading model ({args.device}, {compute_type})...")
model = WhisperModel(MODEL_DIR, device=args.device, compute_type=compute_type)

# Warmup run (excludes JIT/pipeline compilation from timing)
print("Warmup run...")
list(model.transcribe(args.audio, beam_size=5, language=None)[0])

# Timed run
print("Timed run...")
t0 = time.monotonic()
segments, info = model.transcribe(
    args.audio,
    beam_size=5,
    language=None,
    temperature=0.0,                   # no fallback retries (default tries 6 temperatures)
    condition_on_previous_text=False,   # skip cross-segment dependency
    vad_filter=True,                    # skip silent regions
)
segments = list(segments)
elapsed = time.monotonic() - t0

print(f"\nDevice: {args.device}  Compute: {compute_type}")
print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")
print(f"Inference time: {elapsed:.2f}s\n")

for seg in segments:
    print(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s]  {seg.text.strip()}")

del model
import gc; gc.collect()
if args.device == "mps":
    import ctranslate2; ctranslate2.clear_device_cache("mps")
```

Run it:

```bash
cd whisper-metal

# Metal GPU (default)
python workspace/transcribe.py

# CPU for comparison
python workspace/transcribe.py --device cpu
```

### Example results (Apple M4, ~60s Russian audio, whisper-large-v3-turbo)

| Device | Compute | Inference time |
|--------|---------|----------------|
| Metal GPU (`mps`) | float16 | **7.4s** |
| CPU | float32 | 27.8s |

**Metal GPU is 3.7x faster** than CPU on this workload.

## Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| `device` | `"mps"` | Metal GPU acceleration |
| `compute_type` | `"float16"` | Best speed/quality balance |
| `beam_size` | `5` | Good default |
| `language` | `None` | Auto-detect from audio |

For maximum translation quality with float16, use `beam_size=6` and
pass `length_penalty=0.6` to `model.transcribe()`.

## Troubleshooting

**`NameError: name 'torch' is not defined`** during conversion
→ Make sure `pip install torch transformers` was run inside the `whisper-metal` conda env,
and that `ct2-transformers-converter` resolves to the correct env:
```bash
which ct2-transformers-converter
# Should show: .../envs/whisper-metal/bin/ct2-transformers-converter
```

**`Invalid input features shape: expected (1, 128, 3000) but got (1, 80, 3000)`**
→ The model directory is missing `preprocessor_config.json`. Re-run the converter with
`--copy_files tokenizer.json preprocessor_config.json`, or copy it manually:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('openai/whisper-large-v3-turbo', 'preprocessor_config.json', local_dir='models/whisper-large-v3-turbo')"
```

**Build error: `msl_strings.h is out of sync`**
→ Run `python3 tools/gen_msl_strings.py` from the CTranslate2 directory, then rebuild.

**`OPENMP_RUNTIME` / double-init segfault**
→ Make sure cmake was configured with `-DOPENMP_RUNTIME=NONE`.
