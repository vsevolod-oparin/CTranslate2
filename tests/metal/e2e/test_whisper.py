#!/usr/bin/env python3
"""Whisper ASR end-to-end test on Metal."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import ctranslate2
import librosa
from transformers import WhisperProcessor
from conftest import model_path, audio_path

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = ctranslate2.models.Whisper(model_path("whisper-base"), device="metal")

audio, _ = librosa.load(audio_path("sample.mp3"), sr=16000, mono=True)

# Whisper control tokens: <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
prefix_tokens = [50258, 50263, 50359, 50363]

sample_rate = 16000
chunk_size = 60 * sample_rate  # 60s chunks
full_transcription = ""

t_begin = time.monotonic()

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i + chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

    inputs = processor(chunk, return_tensors="np", sampling_rate=sample_rate)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    results = model.generate(features, [prefix_tokens])

    output_tokens = results[0].sequences_ids[0][len(prefix_tokens):]
    text = processor.decode(output_tokens, skip_special_tokens=True)
    full_transcription += text + " "

elapsed_ms = (time.monotonic() - t_begin) * 1000
transcription = full_transcription.strip()

print(f"Transcription: {transcription}")
print(f"Total time: {elapsed_ms:.0f} ms")

# Basic validation
passed = 0
failed = 0


def check(label, ok):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    if ok:
        passed += 1
    else:
        failed += 1


check("Non-empty transcription", len(transcription) > 0)
check("Reasonable length (>10 chars)", len(transcription) > 10)
check("No error tokens in output", "<|" not in transcription)

print()
total = passed + failed
print(f"{passed}/{total} passed")
if failed:
    print("FAILURES DETECTED")
    sys.exit(1)
else:
    print("ALL PASS")
