#!/usr/bin/env python3
"""Test faster_whisper compatibility with CTranslate2 Metal backend.

Verifies that:
  1. Feature extraction uses correct n_mels (128 for large-v3 models)
  2. CPU and Metal produce comparable transcription through faster_whisper
  3. without_timestamps mode produces correct output

Usage:
  python test_faster_whisper.py [model_name]
  # default: whisper-large-v3-turbo

Requires:
  - CT2_TEST_DATA pointing to directory with model dir and sample.mp3
  - faster_whisper, librosa
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

import ctranslate2
import librosa
from faster_whisper import WhisperModel


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
    print(f"Path: {whisper_path}")

    # --- 1. Load models ---
    print("\n=== Loading models ===")
    model_cpu = WhisperModel(whisper_path, device="cpu")
    model_metal = WhisperModel(whisper_path, device="metal")

    cpu_ct = model_cpu.model.compute_type
    metal_ct = model_metal.model.compute_type
    print(f"  CPU compute_type: {cpu_ct}")
    print(f"  Metal compute_type: {metal_ct}")

    # --- 2. Check n_mels ---
    print("\n=== Feature extraction ===")
    expected_mels = 128 if "large-v3" in model_name else 80
    cpu_mels = model_cpu.model.n_mels
    metal_mels = model_metal.model.n_mels
    cpu_fe_mels = model_cpu.feature_extractor.mel_filters.shape[0]
    metal_fe_mels = model_metal.feature_extractor.mel_filters.shape[0]

    check(f"CPU n_mels == {expected_mels}", cpu_mels == expected_mels, f"n_mels={cpu_mels}")
    check(f"Metal n_mels == {expected_mels}", metal_mels == expected_mels, f"n_mels={metal_mels}")
    check(f"CPU FeatureExtractor mel bins == {expected_mels}",
          cpu_fe_mels == expected_mels, f"mel_filters.shape[0]={cpu_fe_mels}")
    check(f"Metal FeatureExtractor mel bins == {expected_mels}",
          metal_fe_mels == expected_mels, f"mel_filters.shape[0]={metal_fe_mels}")

    # --- 3. Transcribe without timestamps (most reliable mode) ---
    print("\n=== Transcription (without_timestamps=True) ===")
    cpu_segments, cpu_info = model_cpu.transcribe(
        audio_file, language="ru", beam_size=5, without_timestamps=True,
    )
    cpu_segments = list(cpu_segments)
    cpu_text = " ".join(s.text for s in cpu_segments).strip()
    print(f"  CPU: {cpu_text[:150]}{'...' if len(cpu_text)>150 else ''}")

    metal_segments, metal_info = model_metal.transcribe(
        audio_file, language="ru", beam_size=5, without_timestamps=True,
    )
    metal_segments = list(metal_segments)
    metal_text = " ".join(s.text for s in metal_segments).strip()
    print(f"  Metal: {metal_text[:150]}{'...' if len(metal_text)>150 else ''}")

    check("CPU transcription non-empty", len(cpu_text) > 10, f"len={len(cpu_text)}")
    check("Metal transcription non-empty", len(metal_text) > 10, f"len={len(metal_text)}")
    check("CPU output reasonable length", len(cpu_text) > 50,
          f"len={len(cpu_text)}, expect >50 for 60s audio")

    # Compare CPU vs Metal
    if cpu_ct == metal_ct:
        check("CPU == Metal text", cpu_text == metal_text)
    else:
        # Different compute types: just check Metal produced substantial output
        metal_ok = len(metal_text) > len(cpu_text) * 0.5
        check("Metal output comparable to CPU", metal_ok,
              f"cpu_len={len(cpu_text)}, metal_len={len(metal_text)}")

    # --- 4. Timestamps mode (informational) ---
    print("\n=== Timestamps mode (informational) ===")
    ts_segments, _ = model_cpu.transcribe(
        audio_file, language="ru", beam_size=5,
    )
    ts_segments = list(ts_segments)
    ts_text = " ".join(s.text for s in ts_segments).strip()
    info(f"Timestamps mode: {len(ts_segments)} segments, {len(ts_text)} chars")
    for seg in ts_segments[:5]:
        info(f"  [{seg.start:.1f}-{seg.end:.1f}] {seg.text[:80]}")

    # --- 5. Speed benchmark ---
    print("\n=== Speed benchmark ===")
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    duration_s = len(audio) / 16000

    # Warmup both backends
    info("Warming up...")
    for m in (model_cpu, model_metal):
        segs, _ = m.transcribe(audio_file, language="ru", beam_size=5, without_timestamps=True)
        list(segs)  # consume generator

    # Benchmark CPU
    t0 = time.monotonic()
    cpu_segs, _ = model_cpu.transcribe(
        audio_file, language="ru", beam_size=5, without_timestamps=True,
    )
    list(cpu_segs)
    cpu_ms = (time.monotonic() - t0) * 1000

    # Benchmark Metal
    t0 = time.monotonic()
    metal_segs, _ = model_metal.transcribe(
        audio_file, language="ru", beam_size=5, without_timestamps=True,
    )
    list(metal_segs)
    metal_ms = (time.monotonic() - t0) * 1000

    rtf_cpu = cpu_ms / (duration_s * 1000)
    rtf_metal = metal_ms / (duration_s * 1000)
    speedup = cpu_ms / metal_ms if metal_ms > 0 else float("inf")

    info(f"Audio duration: {duration_s:.1f}s")
    info(f"CPU:   {cpu_ms:.0f} ms  (RTF={rtf_cpu:.3f})")
    info(f"Metal: {metal_ms:.0f} ms  (RTF={rtf_metal:.3f})")
    info(f"Speedup: {speedup:.2f}x  (Metal / CPU)")

    # --- Summary ---
    print(f"\n{'='*50}")
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
