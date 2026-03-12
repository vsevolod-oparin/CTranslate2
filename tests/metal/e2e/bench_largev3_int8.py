#!/usr/bin/env python3
"""Benchmark int8 variants for whisper-large-v3."""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path

MODEL_DIR = model_path("whisper-large-v3")
AUDIO_FILE = audio_path("sample.mp3")
NUM_RUNS = 3

import ctranslate2, numpy as np, librosa
from faster_whisper.feature_extractor import FeatureExtractor

audio, _ = librosa.load(AUDIO_FILE, sr=16000, mono=True)
fe = FeatureExtractor(feature_size=128)
features = fe(audio)[:, :3000]
features = np.expand_dims(features, 0).astype(np.float32)

PROMPT = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]

for ct in ["int8", "int8_float16", "int8_bfloat16"]:
    for device in ["cpu", "mps"]:
        label = f"{device} {ct}"
        print(f"\n{label:-^60}")
        try:
            model = ctranslate2.models.Whisper(MODEL_DIR, device=device, compute_type=ct)

            # Warmup + correctness
            feat_sv = ctranslate2.StorageView.from_array(features)
            r = model.generate(feat_sv, [PROMPT], beam_size=5, patience=2,
                                return_no_speech_prob=True)[0]
            tokens = r.sequences[0]
            text = "".join(t for t in tokens if not t.startswith("<|"))
            print(f"  Tokens: {len(tokens)}, nsp={r.no_speech_prob:.4f}")
            print(f"  Text: {text[:80]}")

            # Timed runs
            times = []
            enc_times = []
            for _ in range(NUM_RUNS):
                feat_sv = ctranslate2.StorageView.from_array(features)
                t0 = time.monotonic()
                enc = model.encode(feat_sv)
                enc_times.append((time.monotonic() - t0) * 1000)

                feat_sv = ctranslate2.StorageView.from_array(features)
                t0 = time.monotonic()
                model.generate(feat_sv, [PROMPT], beam_size=5, patience=2)[0]
                times.append((time.monotonic() - t0) * 1000)

            best = min(times)
            enc_best = min(enc_times)
            print(f"  Encoder: {enc_best:.0f} ms, Total: {best:.0f} ms, "
                  f"Decoder: {best-enc_best:.0f} ms, tok/s: {len(tokens)/(best/1000):.0f}")

            del model
            gc.collect()
            if device == "mps":
                ctranslate2.clear_device_cache("mps")
        except Exception as e:
            print(f"  ERROR: {e}")
