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
# Note: <|en|> (50263) is intentional — this is a CPU-vs-Metal correctness test,
# not a transcription quality test. The audio may be non-English but both backends
# receive the same prefix tokens, so the comparison is valid regardless of language.
PREFIX_TOKENS = [50258, 50263, 50359, 50363]

WARMUP_SECONDS = 5  # short warmup to exclude pipeline compilation overhead


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

    # --- Warmup (exclude pipeline compilation from timing) ---
    warmup_samples = WARMUP_SECONDS * SAMPLE_RATE
    warmup_chunk = audio[:warmup_samples]
    if len(warmup_chunk) < SAMPLE_RATE * CHUNK_SECONDS:
        warmup_chunk = np.pad(warmup_chunk, (0, SAMPLE_RATE * CHUNK_SECONDS - len(warmup_chunk)),
                              mode="constant")
    warmup_features = ctranslate2.StorageView.from_array(
        processor(warmup_chunk, return_tensors="np", sampling_rate=SAMPLE_RATE).input_features)
    model_cpu.generate(warmup_features, [PREFIX_TOKENS])
    model_metal.generate(warmup_features, [PREFIX_TOKENS])
    print("  Warmup complete (cold-start overhead excluded from timing)")

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

    # --- T6: Whisper with timestamps ---
    print("\n=== Timestamps mode ===")
    # <|startoftranscript|> <|en|> <|transcribe|> (no <|notimestamps|>)
    ts_prefix = [50258, 50263, 50359]
    # Use first 30s chunk only
    ts_chunk = audio[:CHUNK_SECONDS * SAMPLE_RATE]
    if len(ts_chunk) < CHUNK_SECONDS * SAMPLE_RATE:
        ts_chunk = np.pad(ts_chunk, (0, CHUNK_SECONDS * SAMPLE_RATE - len(ts_chunk)),
                          mode="constant")
    ts_inputs = processor(ts_chunk, return_tensors="np", sampling_rate=SAMPLE_RATE)
    ts_features = ctranslate2.StorageView.from_array(ts_inputs.input_features)
    cpu_ts = model_cpu.generate(ts_features, [ts_prefix])
    metal_ts = model_metal.generate(ts_features, [ts_prefix])
    cpu_ts_ids = cpu_ts[0].sequences_ids[0]
    metal_ts_ids = metal_ts[0].sequences_ids[0]
    ts_match = cpu_ts_ids == metal_ts_ids
    check("Timestamps mode: CPU == Metal tokens", ts_match,
          f"cpu_len={len(cpu_ts_ids)}, metal_len={len(metal_ts_ids)}")
    if not ts_match:
        print(f"      CPU  : {cpu_ts_ids[:20]}...")
        print(f"      Metal: {metal_ts_ids[:20]}...")
    # Verify timestamp tokens are present (token IDs >= 50364 are timestamps)
    has_ts = any(t >= 50364 for t in cpu_ts_ids)
    check("Timestamps mode: output contains timestamp tokens", has_ts)

    # --- T7: Batched Whisper transcription ---
    print("\n=== Batched transcription (batch_size=2) ===")
    # Create two different-length chunks
    chunk1 = audio[:CHUNK_SECONDS * SAMPLE_RATE]
    half_len = min(CHUNK_SECONDS * SAMPLE_RATE // 2, len(audio))
    chunk2 = audio[:half_len]
    # Pad both to 30s
    if len(chunk1) < CHUNK_SECONDS * SAMPLE_RATE:
        chunk1 = np.pad(chunk1, (0, CHUNK_SECONDS * SAMPLE_RATE - len(chunk1)), mode="constant")
    chunk2 = np.pad(chunk2, (0, CHUNK_SECONDS * SAMPLE_RATE - len(chunk2)), mode="constant")

    feat1 = processor(chunk1, return_tensors="np", sampling_rate=SAMPLE_RATE).input_features
    feat2 = processor(chunk2, return_tensors="np", sampling_rate=SAMPLE_RATE).input_features
    batch_features = ctranslate2.StorageView.from_array(np.concatenate([feat1, feat2], axis=0))

    cpu_batch = model_cpu.generate(batch_features, [PREFIX_TOKENS, PREFIX_TOKENS])
    metal_batch = model_metal.generate(batch_features, [PREFIX_TOKENS, PREFIX_TOKENS])

    for idx in range(2):
        cpu_ids = cpu_batch[idx].sequences_ids[0]
        metal_ids = metal_batch[idx].sequences_ids[0]
        batch_match = cpu_ids == metal_ids
        check(f"Batch item {idx}: CPU == Metal tokens", batch_match,
              f"cpu_len={len(cpu_ids)}, metal_len={len(metal_ids)}")
        if not batch_match:
            cpu_txt = processor.decode(cpu_ids[len(PREFIX_TOKENS):], skip_special_tokens=True)
            metal_txt = processor.decode(metal_ids[len(PREFIX_TOKENS):], skip_special_tokens=True)
            print(f"      CPU  : {cpu_txt[:80]}")
            print(f"      Metal: {metal_txt[:80]}")

    # Verify batch item 0 matches single-item result
    single_cpu = model_cpu.generate(
        ctranslate2.StorageView.from_array(feat1), [PREFIX_TOKENS])
    single_ids = single_cpu[0].sequences_ids[0]
    batch0_ids = cpu_batch[0].sequences_ids[0]
    check("Batch[0] == single consistency (CPU)", single_ids == batch0_ids)

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
