#!/usr/bin/env python3
"""M12.8 — Larger model benchmark: whisper-large-v3-turbo (d_model=1280).

Comprehensive benchmark across compute types with best-of-3 methodology.
Tests GPU scaling beyond OPUS-MT (d_model=512).

Usage:
    conda run -n ct2 python bench_whisper_m12_8.py [--beam 5] [--runs 3]
"""
import os, sys, time, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path, detect_language_fw

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=int, default=5)
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()

MODEL_NAME = "whisper-large-v3-turbo"
BEAM_SIZE = args.beam
NUM_RUNS = args.runs

# Compute types to benchmark
COMPUTE_TYPES = ["float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16"]

whisper_path = model_path(MODEL_NAME)
audio_file = audio_path("sample.mp3")

if not os.path.isdir(whisper_path):
    print(f"SKIP: {MODEL_NAME} not found at {whisper_path}")
    sys.exit(1)

import librosa
audio, _ = librosa.load(audio_file, sr=16000, mono=True)
dur = len(audio) / 16000
print(f"Model: {MODEL_NAME}  beam_size={BEAM_SIZE}  audio={dur:.1f}s  runs={NUM_RUNS}")
print(f"Model path: {whisper_path}")
print("=" * 80)

from faster_whisper import WhisperModel
import ctranslate2


_detected_language = None

def bench_one(device, compute_type, beam_size, num_runs):
    """Load model, warmup, run num_runs timed transcriptions. Return best_ms and text."""
    global _detected_language
    model = WhisperModel(whisper_path, device=device, compute_type=compute_type)

    # Auto-detect language on first call
    if _detected_language is None:
        _detected_language = detect_language_fw(model, audio_file)
        print(f"  Detected language: {_detected_language}")

    # Warmup
    segments, _ = model.transcribe(
        audio_file, language=_detected_language, beam_size=beam_size, without_timestamps=True
    )
    text = " ".join(seg.text.strip() for seg in segments)

    # Timed runs
    times = []
    for i in range(num_runs):
        t0 = time.monotonic()
        segs, _ = model.transcribe(
            audio_file, language=_detected_language, beam_size=beam_size, without_timestamps=True
        )
        list(segs)  # consume generator
        elapsed = (time.monotonic() - t0) * 1000
        times.append(elapsed)

    del model
    gc.collect()
    if device == "mps":
        ctranslate2.clear_device_cache("mps")

    best_ms = min(times)
    return best_ms, times, text


# --- CPU baseline (f32 only) ---
print(f"\n{'CPU float32':=^80}")
cpu_ms, cpu_times, cpu_text = bench_one("cpu", "float32", BEAM_SIZE, NUM_RUNS)
print(f"  Times: {['%.0f' % t for t in cpu_times]}")
print(f"  Best:  {cpu_ms:.0f} ms")
print(f"  Text:  {cpu_text[:120]}...")

results = {}
results["cpu_f32"] = {"best_ms": cpu_ms, "times": cpu_times, "text": cpu_text}

# --- MPS benchmarks ---
for ct in COMPUTE_TYPES:
    label = f"MPS {ct}"
    print(f"\n{label:=^80}")
    try:
        ms, times, text = bench_one("mps", ct, BEAM_SIZE, NUM_RUNS)
        speedup = cpu_ms / ms
        print(f"  Times: {['%.0f' % t for t in times]}")
        print(f"  Best:  {ms:.0f} ms  ({speedup:.2f}x vs CPU)")
        print(f"  Text:  {text[:120]}...")

        # Quick correctness check: compare first 50 chars to CPU
        match = "MATCH" if text[:50] == cpu_text[:50] else "DIFF"
        print(f"  vs CPU: {match}")

        results[f"mps_{ct}"] = {"best_ms": ms, "times": times, "text": text, "speedup": speedup}
    except Exception as e:
        print(f"  ERROR: {e}")
        results[f"mps_{ct}"] = {"error": str(e)}

# --- Summary ---
print(f"\n{'SUMMARY':=^80}")
print(f"Model: {MODEL_NAME} (d_model=1280, 4 decoder layers)")
print(f"Audio: {dur:.1f}s, beam_size={BEAM_SIZE}, best-of-{NUM_RUNS}")
print()
print(f"{'Type':<20} {'Best ms':>10} {'Speedup':>10} {'Status':>10}")
print("-" * 55)
print(f"{'CPU f32':<20} {cpu_ms:>10.0f} {'baseline':>10} {'---':>10}")

for ct in COMPUTE_TYPES:
    key = f"mps_{ct}"
    r = results.get(key, {})
    if "error" in r:
        print(f"{'MPS ' + ct:<20} {'ERROR':>10} {'---':>10} {'FAIL':>10}")
    elif key in results:
        ms = r["best_ms"]
        sp = r["speedup"]
        status = "PASS" if sp > 1.0 else "SLOW"
        print(f"{'MPS ' + ct:<20} {ms:>10.0f} {sp:>9.2f}x {status:>10}")

# Pass criterion from plan: f16 > 3x CPU speedup
f16_result = results.get("mps_float16", {})
if "speedup" in f16_result:
    criterion = f16_result["speedup"] >= 3.0
    print(f"\nM12.8 PASS criterion (f16 > 3x CPU): {f16_result['speedup']:.2f}x — {'PASS' if criterion else 'FAIL'}")
else:
    print("\nM12.8 PASS criterion: f16 not available")
