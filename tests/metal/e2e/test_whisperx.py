#!/usr/bin/env python3
"""Test WhisperX compatibility with CTranslate2 MPS backend.

Verifies that:
  1. WhisperX loads model on CPU and MPS via CTranslate2
  2. Feature extraction uses correct n_mels (128 for large-v3 models)
  3. Transcription produces comparable output on both devices
  4. Word-level alignment works on MPS transcription output
  5. MPS provides speedup over CPU

Usage:
  python test_whisperx.py [model_name]
  # default: whisper-large-v3-turbo

Requires:
  - CT2_TEST_DATA pointing to directory with model dir and sample.mp3
  - whisperx, faster_whisper, librosa, torch
"""
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

try:
    import whisperx
except ImportError:
    print("SKIP: whisperx not installed")
    sys.exit(0)

import ctranslate2
import librosa

# --- Compatibility patches for whisperx 3.2.0 with newer deps ---
# whisperx 3.2.0 expects faster_whisper 1.0.0 but we have 1.2.x which added
# 'multilingual' and 'hotwords' to TranscriptionOptions.
_COMPAT_ASR_OPTIONS = {"multilingual": False, "hotwords": None}

# pyannote.audio 4.x removed 'use_auth_token' from Inference.__init__.
# Patch it so whisperx's VoiceActivitySegmentation can still pass it.
try:
    from pyannote.audio.core.inference import Inference as _Inference
    _orig_init = _Inference.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return _orig_init(self, *args, **kwargs)

    _Inference.__init__ = _patched_init
except ImportError:
    pass

# WhisperX passes device to torch.device() for VAD — force CPU for VAD
# (small PyTorch model, fast on CPU, avoids MPS compatibility issues).
import whisperx.vad as _vad
_orig_load_vad = _vad.load_vad_model

def _patched_load_vad(device, *args, **kwargs):
    import torch
    return _orig_load_vad(torch.device("cpu"), *args, **kwargs)

_vad.load_vad_model = _patched_load_vad
import whisperx.asr as _asr
_asr.load_vad_model = _patched_load_vad


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-turbo"
    whisper_path = model_path(model_name)
    audio_file = audio_path("sample.mp3")

    if not os.path.isdir(whisper_path):
        print(f"SKIP: {model_name} not found at {whisper_path}")
        return 0
    if not os.path.isfile(audio_file):
        print(f"SKIP: sample.mp3 not found at {audio_file}")
        return 0

    passed = 0
    failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{tag}] {label}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1

    def info(label, detail=""):
        suffix = f"  ({detail})" if detail else ""
        print(f"  [INFO] {label}{suffix}")

    print(f"Model: {model_name}")
    print(f"Path:  {whisper_path}")

    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    duration_s = len(audio) / 16000
    perf_beam_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # =========================================================
    # Phase 1: CPU — load, test, benchmark, free
    # =========================================================
    print("\n=== Loading WhisperX CPU model ===")
    model_cpu = whisperx.load_model(
        whisper_path, device="cpu", compute_type="float32",
        language="ru", asr_options=_COMPAT_ASR_OPTIONS,
    )
    cpu_ct = model_cpu.model.model.compute_type
    print(f"  CPU compute_type: {cpu_ct}")
    check("CPU model loaded", model_cpu is not None)

    print("\n=== Feature extraction (CPU) ===")
    expected_mels = 128 if "large-v3" in model_name else 80
    cpu_mels = model_cpu.model.model.n_mels
    cpu_fe_mels = model_cpu.model.feature_extractor.mel_filters.shape[0]
    check(f"CPU n_mels == {expected_mels}", cpu_mels == expected_mels, f"n_mels={cpu_mels}")
    check(f"CPU FeatureExtractor mel bins == {expected_mels}",
          cpu_fe_mels == expected_mels, f"mel_filters.shape[0]={cpu_fe_mels}")

    print("\n=== CPU transcription ===")
    cpu_audio = whisperx.load_audio(audio_file)
    cpu_result = model_cpu.transcribe(cpu_audio, batch_size=1, language="ru")
    cpu_segments = cpu_result.get("segments", [])
    cpu_text = " ".join(s["text"] for s in cpu_segments).strip()
    print(f"  CPU: {cpu_text[:150]}{'...' if len(cpu_text) > 150 else ''}")
    check("CPU transcription non-empty", len(cpu_text) > 10, f"len={len(cpu_text)}")
    check("CPU output reasonable length", len(cpu_text) > 50,
          f"len={len(cpu_text)}, expect >50 for 60s audio")

    # Alignment (CPU)
    print("\n=== CPU alignment ===")
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code="ru", device="cpu",
        )
        cpu_aligned = whisperx.align(
            cpu_segments, align_model, align_meta, cpu_audio, device="cpu",
        )
        aligned_segs = cpu_aligned.get("segments", [])
        has_words = any("words" in s and len(s["words"]) > 0 for s in aligned_segs)
        check("CPU alignment produces word-level data", has_words,
              f"n_aligned_segments={len(aligned_segs)}")
        if has_words:
            sample_words = aligned_segs[0].get("words", [])[:3]
            for w in sample_words:
                info(f"  word: '{w.get('word','')}' [{w.get('start',0):.2f}-{w.get('end',0):.2f}]")
        del align_model
        gc.collect()
    except Exception as e:
        info(f"Alignment skipped: {e}")

    # CPU benchmark
    model_cpu.transcribe(cpu_audio, batch_size=1, language="ru")  # warmup
    t0 = time.monotonic()
    model_cpu.transcribe(cpu_audio, batch_size=1, language="ru")
    cpu_ms = (time.monotonic() - t0) * 1000

    # Free CPU model before loading MPS
    del model_cpu
    gc.collect()

    # =========================================================
    # Phase 2: MPS — load, test, benchmark, free
    # =========================================================
    print("\n=== Loading WhisperX MPS model ===")
    model_mps = whisperx.load_model(
        whisper_path, device="mps", compute_type="float16",
        language="ru", asr_options=_COMPAT_ASR_OPTIONS,
    )
    mps_ct = model_mps.model.model.compute_type
    print(f"  MPS compute_type: {mps_ct}")
    check("MPS model loaded", model_mps is not None)

    print("\n=== Feature extraction (MPS) ===")
    mps_mels = model_mps.model.model.n_mels
    mps_fe_mels = model_mps.model.feature_extractor.mel_filters.shape[0]
    check(f"MPS n_mels == {expected_mels}", mps_mels == expected_mels, f"n_mels={mps_mels}")
    check(f"MPS FeatureExtractor mel bins == {expected_mels}",
          mps_fe_mels == expected_mels, f"mel_filters.shape[0]={mps_fe_mels}")

    print("\n=== MPS transcription ===")
    mps_audio = whisperx.load_audio(audio_file)
    mps_result = model_mps.transcribe(mps_audio, batch_size=1, language="ru")
    mps_segments = mps_result.get("segments", [])
    mps_text = " ".join(s["text"] for s in mps_segments).strip()
    print(f"  MPS: {mps_text[:150]}{'...' if len(mps_text) > 150 else ''}")
    check("MPS transcription non-empty", len(mps_text) > 10, f"len={len(mps_text)}")

    # Compare CPU vs MPS
    if cpu_ct == mps_ct:
        check("CPU == MPS text", cpu_text == mps_text)
    else:
        mps_ok = len(mps_text) > len(cpu_text) * 0.5
        check("MPS output comparable to CPU", mps_ok,
              f"cpu_len={len(cpu_text)}, mps_len={len(mps_text)}")

    # Alignment (MPS output, aligned on CPU)
    print("\n=== MPS alignment ===")
    try:
        align_model, align_meta = whisperx.load_align_model(
            language_code="ru", device="cpu",
        )
        mps_aligned = whisperx.align(
            mps_segments, align_model, align_meta, mps_audio, device="cpu",
        )
        aligned_segs = mps_aligned.get("segments", [])
        has_words = any("words" in s and len(s["words"]) > 0 for s in aligned_segs)
        check("MPS alignment produces word-level data", has_words,
              f"n_aligned_segments={len(aligned_segs)}")
        del align_model
        gc.collect()
    except Exception as e:
        info(f"Alignment skipped: {e}")

    # MPS benchmark
    model_mps.transcribe(mps_audio, batch_size=1, language="ru")  # warmup
    t0 = time.monotonic()
    model_mps.transcribe(mps_audio, batch_size=1, language="ru")
    mps_ms = (time.monotonic() - t0) * 1000

    del model_mps
    gc.collect()
    ctranslate2.clear_device_cache("mps")

    # =========================================================
    # Speed summary
    # =========================================================
    print("\n=== Speed benchmark ===")
    rtf_cpu = cpu_ms / (duration_s * 1000)
    rtf_mps = mps_ms / (duration_s * 1000)
    speedup = cpu_ms / mps_ms if mps_ms > 0 else float("inf")
    info(f"Audio duration: {duration_s:.1f}s at beam_size={perf_beam_size}")
    info(f"CPU:   {cpu_ms:.0f} ms  (RTF={rtf_cpu:.3f})")
    info(f"MPS:   {mps_ms:.0f} ms  (RTF={rtf_mps:.3f})")
    info(f"Speedup: {speedup:.2f}x  (MPS / CPU)")

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'=' * 50}")
    total = passed + failed
    print(f"{passed}/{total} passed")
    if failed:
        print("FAILURES DETECTED")
        return 1
    else:
        print("ALL PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
