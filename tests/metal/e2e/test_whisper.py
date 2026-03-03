#!/usr/bin/env python3
"""M10.4 — Whisper ASR end-to-end test on Metal.

Verifies that the Metal backend produces correct transcription for whisper-base:
  - Metal and CPU transcriptions compared via Word Error Rate (WER)
  - WER difference must be < 1% (i.e., Metal WER within 1 point of CPU WER)
  - Exact transcript match checked (informational)
  - Timing benchmarks (informational, speed optimization deferred to M11)

Requires:
  - CT2_TEST_DATA pointing to a directory containing whisper-base/ and sample.mp3
  - librosa, transformers, jiwer
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

import numpy as np
import ctranslate2
import librosa
from transformers import WhisperProcessor
from jiwer import wer

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30  # Whisper's native window
WER_TOLERANCE = 0.01  # WER difference must be < 1%

# Whisper control tokens: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
PREFIX_TOKENS = [50258, 50263, 50359, 50363]


def transcribe(model, processor, audio_array, label=""):
    """Transcribe audio using a CTranslate2 Whisper model. Returns (text, elapsed_ms)."""
    chunk_size = CHUNK_SECONDS * SAMPLE_RATE
    full_text = ""

    t0 = time.monotonic()

    for i in range(0, len(audio_array), chunk_size):
        chunk = audio_array[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

        inputs = processor(chunk, return_tensors="np", sampling_rate=SAMPLE_RATE)
        features = ctranslate2.StorageView.from_array(inputs.input_features)
        results = model.generate(features, [PREFIX_TOKENS])

        output_tokens = results[0].sequences_ids[0][len(PREFIX_TOKENS):]
        text = processor.decode(output_tokens, skip_special_tokens=True)
        full_text += text + " "

    elapsed_ms = (time.monotonic() - t0) * 1000
    return full_text.strip(), elapsed_ms


def main():
    whisper_path = model_path("whisper-base")
    audio_file = audio_path("sample.mp3")

    if not os.path.isdir(whisper_path):
        print(f"SKIP: whisper-base model not found at {whisper_path}")
        return 0
    if not os.path.isfile(audio_file):
        print(f"SKIP: sample.mp3 not found at {audio_file}")
        return 0

    print("Loading processor and audio...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
    duration_s = len(audio) / SAMPLE_RATE
    print(f"  Audio: {audio_file} ({duration_s:.1f}s, {len(audio)} samples)")

    print("Loading models...")
    model_cpu = ctranslate2.models.Whisper(whisper_path, device="cpu")
    model_metal = ctranslate2.models.Whisper(whisper_path, device="metal")

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

    # --- Transcription ---
    print("\n=== Transcription ===")
    cpu_text, cpu_ms = transcribe(model_cpu, processor, audio, "CPU")
    metal_text, metal_ms = transcribe(model_metal, processor, audio, "Metal")

    print(f"  CPU  : \"{cpu_text}\"")
    print(f"  Metal: \"{metal_text}\"")

    # --- Correctness checks ---
    print("\n=== Correctness ===")

    check("CPU transcription non-empty", len(cpu_text) > 0)
    check("Metal transcription non-empty", len(metal_text) > 0)
    check("No error tokens in CPU output", "<|" not in cpu_text)
    check("No error tokens in Metal output", "<|" not in metal_text)

    # WER comparison
    if cpu_text and metal_text:
        # WER of Metal vs CPU (treating CPU as reference)
        wer_metal_vs_cpu = wer(cpu_text, metal_text)
        print(f"\n  Metal-vs-CPU WER: {wer_metal_vs_cpu:.4f} ({wer_metal_vs_cpu*100:.2f}%)")

        check(
            f"Metal-vs-CPU WER < {WER_TOLERANCE*100:.0f}%",
            wer_metal_vs_cpu < WER_TOLERANCE,
            f"WER={wer_metal_vs_cpu*100:.2f}%",
        )

        exact_match = cpu_text == metal_text
        check(
            "Exact transcript match (CPU == Metal)",
            exact_match,
            "identical" if exact_match else "differ",
        )

    # --- Reasonable output checks ---
    print("\n=== Output sanity ===")
    check("CPU output > 10 chars", len(cpu_text) > 10, f"len={len(cpu_text)}")
    check("Metal output > 10 chars", len(metal_text) > 10, f"len={len(metal_text)}")

    # --- Speed benchmarks (informational) ---
    print("\n=== Speed benchmark (informational) ===")
    speedup = cpu_ms / metal_ms if metal_ms > 0 else float("inf")
    rtf_cpu = cpu_ms / (duration_s * 1000)
    rtf_metal = metal_ms / (duration_s * 1000)
    info(f"CPU:   {cpu_ms:.0f} ms  (RTF={rtf_cpu:.3f})")
    info(f"Metal: {metal_ms:.0f} ms  (RTF={rtf_metal:.3f})")
    info(f"Ratio: {speedup:.2f}x (target: >=1.5x after M11 optimization)")

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
