#!/usr/bin/env python3
"""Quick faster_whisper speed benchmark — skip correctness, minimal overhead."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, audio_path
from faster_whisper import WhisperModel
import librosa

model_name = sys.argv[1] if len(sys.argv) > 1 else "whisper-large-v3-turbo"
beam_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
whisper_path = model_path(model_name)
audio_file = audio_path("sample.mp3")

if not os.path.isdir(whisper_path):
    print(f"SKIP: {model_name} not found at {whisper_path}"); sys.exit(0)

audio, _ = librosa.load(audio_file, sr=16000, mono=True)
dur = len(audio) / 16000
print(f"Model: {model_name}  beam_size={beam_size}  audio={dur:.0f}s")

# Benchmark CPU first, then free before loading Metal to avoid 2x memory.
model_cpu = WhisperModel(whisper_path, device="cpu", compute_type="float32")
list(model_cpu.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
t0 = time.monotonic()
list(model_cpu.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
cpu_ms = (time.monotonic() - t0) * 1000
del model_cpu
import gc; gc.collect()

model_metal = WhisperModel(whisper_path, device="mps", compute_type="float32")
list(model_metal.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
t0 = time.monotonic()
list(model_metal.transcribe(audio_file, language="ru", beam_size=beam_size, without_timestamps=True)[0])
metal_ms = (time.monotonic() - t0) * 1000
del model_metal
gc.collect()
import ctranslate2; ctranslate2.clear_device_cache("mps")

print(f"CPU: {cpu_ms:.0f}ms  Metal: {metal_ms:.0f}ms  Speedup: {cpu_ms/metal_ms:.2f}x")
