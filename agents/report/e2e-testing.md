# End-to-End Testing: Metal Backend

How to build CTranslate2 with the Metal backend from source, set up a conda environment, prepare test data, and run the Python e2e tests.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Anaconda or Miniconda installed
- Git checkout of the `metal-backend` branch

## 1. Create the Conda Environment

```bash
conda create -n ct2 python=3.14 -y
conda activate ct2/
```

Install Python dependencies:

```bash
pip install pybind11 numpy pyyaml transformers sentencepiece librosa
```

## 2. Build the C++ Library

From the repo root:

```bash
cmake -S . -B build \
    -DWITH_METAL=ON \
    -DWITH_ACCELERATE=ON \
    -DWITH_MKL=OFF \
    -DOPENMP_RUNTIME=NONE \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(sysctl -n hw.logicalcpu)
```

This produces `build/libctranslate2.dylib`.

## 3. Install the Library into the Conda Environment

Copy the built library into the conda env so it is self-contained (no `sudo`, no `/usr/local`):

```bash
CT2_ENV_LIB="$(python -c 'import sys; print(sys.prefix)')/lib"

cp build/libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/"
ln -sf libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/libctranslate2.4.dylib"
ln -sf libctranslate2.4.dylib      "$CT2_ENV_LIB/libctranslate2.dylib"
```

After a C++ rebuild (step 2), only the `cp` line needs repeating — symlinks stay valid.

## 4. Install the Python Bindings

From the repo root:

```bash
cd python
CTRANSLATE2_ROOT=../build pip install -e . --no-build-isolation
cd ..
```

`CTRANSLATE2_ROOT` points to the build directory for headers during compilation.
The runtime library is loaded from the conda env lib path set up in step 3.

Verify the installation:

```bash
python -c "import ctranslate2; print(ctranslate2.get_supported_devices())"
# Should include 'metal' in the output
```

## 4. Prepare Test Data

The e2e tests need two models and one audio file. Create a data directory (outside the repo is recommended):

```bash
mkdir -p /path/to/data
```

### Translation model (opus-mt-en-de)

```bash
# Convert the HuggingFace model to CT2 format
ct2-opus-mt-converter --model_name Helsinki-NLP/opus-mt-en-de \
    --output_dir /path/to/data/opus-mt-en-de
```

The output directory should contain: `model.bin`, `shared_vocabulary.json`, `config.json`.

### Whisper model (whisper-base)

```bash
ct2-transformers-converter --model openai/whisper-base \
    --output_dir /path/to/data/whisper-base
```

### Audio sample

Place any MP3 audio file at `/path/to/data/sample.mp3`. A short (10–60s) speech clip works well.

### Final data directory layout

```
/path/to/data/
├── opus-mt-en-de/
│   ├── config.json
│   ├── model.bin
│   └── shared_vocabulary.json
├── whisper-base/
│   └── ...
└── sample.mp3
```

## 5. Rebuild Shortcut

After a C++ change (no Python binding changes), only steps 2–3 are needed:

```bash
cmake --build build -j$(sysctl -n hw.logicalcpu)
cp build/libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/"
```

If the Python binding code changes, also re-run step 4.

## 6. Run the E2E Tests

Set the `CT2_TEST_DATA` environment variable pointing to your data directory, then run each test:

```bash
export CT2_TEST_DATA=/path/to/data

python tests/metal/e2e/test_translation.py    # 90 tests: CPU vs Metal, 5 sentences × 3 beams × 6 max_len
python tests/metal/e2e/test_beam_search.py    # 28 tests: beam sweep + batch consistency
python tests/metal/e2e/test_whisper.py        #  3 tests: Whisper ASR validation
```

Each script exits 0 on all-pass, 1 on any failure.

### Run all at once

```bash
export CT2_TEST_DATA=/path/to/data
python tests/metal/e2e/test_translation.py \
  && python tests/metal/e2e/test_beam_search.py \
  && python tests/metal/e2e/test_whisper.py \
  && echo "All e2e tests passed"
```

## Test Descriptions

| Test | Source | What it covers |
|------|--------|----------------|
| `test_translation.py` | `full_e2e_test.py`, `beam_toks_test.py`, `test1.py` | CPU vs Metal output equality across sentences, beam sizes (1/2/4), and decoding lengths |
| `test_beam_search.py` | `beam_step_test.py`, `batch_compare_test.py` | Step-by-step beam regression (max_len 1..11), batch=2 vs single consistency, beam1 vs beam2 hyp[0] |
| `test_whisper.py` | `whisper_test.py` | Whisper ASR on Metal: non-empty output, reasonable length, no error tokens |

## Shared Config (`conftest.py`)

All tests import from `tests/metal/e2e/conftest.py`, which provides:

- `DATA_DIR` — resolved from `CT2_TEST_DATA` env var (defaults to `../../../../data` relative to the script)
- `model_path(name)` / `audio_path(name)` — absolute path helpers
- `load_marian_tokenizer()` — loads `Helsinki-NLP/opus-mt-en-de` tokenizer
- `tokenize()` / `decode()` — MarianTokenizer encode/decode wrappers

## Troubleshooting

**`AttributeError: module 'ctranslate2' has no attribute 'Translator'`**
The C extension (`_ext.so`) isn't built. Re-run `pip install -e .` from the `python/` directory with `CTRANSLATE2_ROOT` set.

**`Library not loaded: libctranslate2.4.dylib`**
The shared library isn't in the conda env lib path. Re-run step 3 (copy + symlinks into `$CT2_ENV_LIB`).

**`FileNotFoundError` on model path**
Set `CT2_TEST_DATA` to the directory containing `opus-mt-en-de/`, `whisper-base/`, and `sample.mp3`.

**Whisper test fails with `librosa` import error**
Install librosa: `pip install librosa`.
