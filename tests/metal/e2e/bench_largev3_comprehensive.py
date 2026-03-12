#!/usr/bin/env python3
"""Comprehensive whisper-large-v3 performance benchmark on MPS.

Tests all compute types with raw ctranslate2 API (confirmed correct).
Measures encoder time, decoder time, and total throughput.
"""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path, detect_language_ct2, get_whisper_prefix_strings

MODEL_DIR = model_path("whisper-large-v3")
AUDIO_FILE = audio_path("sample.mp3")
NUM_RUNS = 3

import ctranslate2
import numpy as np
import librosa
from faster_whisper.feature_extractor import FeatureExtractor

# Load audio and extract features
audio, _ = librosa.load(AUDIO_FILE, sr=16000, mono=True)
dur = len(audio) / 16000
fe = FeatureExtractor(feature_size=128)
features = fe(audio)[:, :3000]
features = np.expand_dims(features, 0).astype(np.float32)
print(f"Audio: {dur:.1f}s, features: {features.shape}")

# Auto-detect language from audio
_detect_model = ctranslate2.models.Whisper(MODEL_DIR, device="cpu")
from transformers import WhisperProcessor as _WP
_detect_proc = _WP.from_pretrained("openai/whisper-large-v3")
_language = detect_language_ct2(_detect_model, _detect_proc, audio)
del _detect_model, _detect_proc
print(f"Detected language: {_language}")

PROMPT = get_whisper_prefix_strings(_language)
COMPUTE_TYPES = ["float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16"]


def bench_one(device, compute_type, beam_size=5, patience=2, runs=NUM_RUNS):
    try:
        model = ctranslate2.models.Whisper(MODEL_DIR, device=device, compute_type=compute_type)
    except Exception as e:
        return {"error": str(e)}

    # Warmup
    feat_sv = ctranslate2.StorageView.from_array(features)
    results = model.generate(feat_sv, [PROMPT], beam_size=beam_size, patience=patience,
                              return_no_speech_prob=True)
    result = results[0]
    tokens = result.sequences[0]
    text = "".join(t for t in tokens if not t.startswith("<|"))
    n_tokens = len(tokens)

    # Encoder-only timing
    enc_times = []
    for _ in range(runs):
        feat_sv = ctranslate2.StorageView.from_array(features)
        t0 = time.monotonic()
        enc_out = model.encode(feat_sv)
        enc_times.append((time.monotonic() - t0) * 1000)

    # Full generate timing
    gen_times = []
    for _ in range(runs):
        feat_sv = ctranslate2.StorageView.from_array(features)
        t0 = time.monotonic()
        r = model.generate(feat_sv, [PROMPT], beam_size=beam_size, patience=patience)
        _ = r[0].sequences
        gen_times.append((time.monotonic() - t0) * 1000)

    del model
    gc.collect()
    if device == "mps":
        ctranslate2.clear_device_cache("mps")

    return {
        "enc_ms": min(enc_times),
        "gen_ms": min(gen_times),
        "dec_ms": min(gen_times) - min(enc_times),
        "tokens": n_tokens,
        "text": text[:80],
        "no_speech_prob": result.no_speech_prob,
        "tok_per_s": n_tokens / (min(gen_times) / 1000) if min(gen_times) > 0 else 0,
    }


# --- CPU baseline ---
print(f"\n{'CPU float32':=^70}")
cpu = bench_one("cpu", "float32")
if "error" in cpu:
    print(f"  ERROR: {cpu['error']}")
    sys.exit(1)
print(f"  Encoder: {cpu['enc_ms']:.0f} ms")
print(f"  Total:   {cpu['gen_ms']:.0f} ms  (decoder: {cpu['dec_ms']:.0f} ms)")
print(f"  Tokens:  {cpu['tokens']}  ({cpu['tok_per_s']:.0f} tok/s)")
print(f"  Text:    {cpu['text']}")

# --- MPS benchmarks ---
results = {"cpu_f32": cpu}

print(f"\n{'MPS BENCHMARKS':=^70}")
for ct in COMPUTE_TYPES:
    label = f"MPS {ct}"
    print(f"\n{label:-^70}")
    r = bench_one("mps", ct)
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        results[f"mps_{ct}"] = r
        continue

    speedup = cpu["gen_ms"] / r["gen_ms"]
    enc_speedup = cpu["enc_ms"] / r["enc_ms"]
    dec_speedup = cpu["dec_ms"] / r["dec_ms"]
    match = "MATCH" if r["text"][:40] == cpu["text"][:40] else "DIFF"

    print(f"  Encoder: {r['enc_ms']:.0f} ms ({enc_speedup:.2f}x vs CPU)")
    print(f"  Decoder: {r['dec_ms']:.0f} ms ({dec_speedup:.2f}x vs CPU)")
    print(f"  Total:   {r['gen_ms']:.0f} ms ({speedup:.2f}x vs CPU)")
    print(f"  Tokens:  {r['tokens']}  ({r['tok_per_s']:.0f} tok/s)")
    print(f"  Text:    {match} {r['text'][:60]}")

    r["speedup"] = speedup
    r["enc_speedup"] = enc_speedup
    r["dec_speedup"] = dec_speedup
    results[f"mps_{ct}"] = r

# --- Summary table ---
print(f"\n{'SUMMARY':=^70}")
print(f"{'Type':<20} {'Enc ms':>8} {'Dec ms':>8} {'Total':>8} {'Speedup':>8} {'tok/s':>8}")
print("-" * 60)
print(f"{'CPU f32':<20} {cpu['enc_ms']:>8.0f} {cpu['dec_ms']:>8.0f} {cpu['gen_ms']:>8.0f} {'1.00x':>8} {cpu['tok_per_s']:>8.0f}")
for ct in COMPUTE_TYPES:
    key = f"mps_{ct}"
    r = results.get(key, {})
    if "error" in r:
        print(f"{'MPS '+ct:<20} {'ERR':>8} {'':>8} {'':>8} {'':>8}")
    else:
        print(f"{'MPS '+ct:<20} {r['enc_ms']:>8.0f} {r['dec_ms']:>8.0f} {r['gen_ms']:>8.0f} {r.get('speedup',0):>7.2f}x {r['tok_per_s']:>8.0f}")
