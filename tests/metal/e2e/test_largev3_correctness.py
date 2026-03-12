#!/usr/bin/env python3
"""Quick correctness test for whisper-large-v3 on MPS vs CPU.

Verifies the iterative prompt fix (M12.8) produces correct transcriptions.
"""

import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

MODEL_DIR = model_path("whisper-large-v3")
AUDIO_FILE = audio_path("sample.mp3")

if not os.path.isdir(MODEL_DIR):
    print(f"SKIP: whisper-large-v3 not found at {MODEL_DIR}")
    sys.exit(1)

from faster_whisper import WhisperModel
import ctranslate2

def run_test(device, compute_type, patience=2, beam_size=5):
    model = WhisperModel(MODEL_DIR, device=device, compute_type=compute_type)
    t0 = time.monotonic()
    segments, info = model.transcribe(
        AUDIO_FILE, language="ru", beam_size=beam_size,
        without_timestamps=True, patience=patience,
    )
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.monotonic() - t0
    del model
    gc.collect()
    return text, elapsed

print("=== whisper-large-v3 Correctness Test ===\n")

# CPU baseline
print("Running CPU f32 (beam=5, patience=2)...")
cpu_text, cpu_time = run_test("cpu", "float32")
print(f"  Time: {cpu_time:.2f}s")
print(f"  Text: {cpu_text[:150]}...")
print(f"  Len:  {len(cpu_text)}")

# MPS float16
for ct in ["float16", "float32"]:
    print(f"\nRunning MPS {ct} (beam=5, patience=2)...")
    try:
        mps_text, mps_time = run_test("mps", ct)
        print(f"  Time: {mps_time:.2f}s")
        print(f"  Text: {mps_text[:150]}...")
        print(f"  Len:  {len(mps_text)}")

        match = cpu_text.strip() == mps_text.strip()
        print(f"  MATCH: {'YES' if match else 'NO'}")
        if not match and len(mps_text) > 0:
            # Check overlap
            cpu_words = set(cpu_text.lower().split())
            mps_words = set(mps_text.lower().split())
            if cpu_words and mps_words:
                overlap = len(cpu_words & mps_words) / max(len(cpu_words), len(mps_words))
                print(f"  Word overlap: {overlap:.0%}")

        ctranslate2.clear_device_cache("mps")
        gc.collect()
    except Exception as e:
        import traceback
        traceback.print_exc()

# Also test beam=1 (greedy)
print(f"\nRunning MPS float16 beam=1 (greedy)...")
try:
    mps_g_text, mps_g_time = run_test("mps", "float16", patience=1, beam_size=1)
    cpu_g_text, cpu_g_time = run_test("cpu", "float32", patience=1, beam_size=1)
    print(f"  CPU Time: {cpu_g_time:.2f}s  Text: {cpu_g_text[:80]}...")
    print(f"  MPS Time: {mps_g_time:.2f}s  Text: {mps_g_text[:80]}...")
    match = cpu_g_text.strip() == mps_g_text.strip()
    print(f"  MATCH: {'YES' if match else 'NO'}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
