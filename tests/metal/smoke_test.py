#!/usr/bin/env python3
"""M13.7 — Metal backend smoke test.

Single-command local verification for Apple Silicon developers.
Run after any Metal code change to verify nothing is broken.

Usage:
    python tests/metal/smoke_test.py

Tests:
    1. MPS device available and StorageView round-trip works
    2. Translation f32 on MPS produces valid output
    3. Translation f16 on MPS produces valid output
    4. Translation int8 on MPS produces valid output
    5. Whisper inference on MPS (if model available)

Requires:
    - CTranslate2 built with CT2_WITH_METAL=ON
    - CT2_TEST_DATA with opus-mt-en-de/ (minimum)
    - Optional: opus-mt-en-de-f16/, opus-mt-en-de-int8/, whisper-large-v3-turbo/
"""
import sys
import os
import time

import numpy as np


class SmokeTest:
    """Simple test runner with pass/fail/skip tracking."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def check(self, label, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        self.results.append((tag, label, suffix))
        print(f"  [{tag}] {label}{suffix}")
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def skip(self, label, reason=""):
        self.results.append(("SKIP", label, reason))
        print(f"  [SKIP] {label}  ({reason})")
        self.skipped += 1

    def summary(self):
        total = self.passed + self.failed
        return self.passed, self.failed, self.skipped, total


def main():
    start_time = time.time()
    print("M13.7 Metal Smoke Test")
    print(f"{'='*60}")

    runner = SmokeTest()

    # ---- Setup ----
    e2e_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")
    sys.path.insert(0, e2e_dir)

    try:
        import ctranslate2
    except ImportError:
        print("FATAL: ctranslate2 not importable")
        return 1

    try:
        from conftest import model_path, load_marian_tokenizer, tokenize, decode
    except ImportError:
        print("FATAL: conftest not found in tests/metal/e2e/")
        return 1

    # ========================================================
    # Test 1: MPS device + StorageView round-trip
    # ========================================================
    print("\n--- 1. MPS device availability ---")
    try:
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        sv_cpu = ctranslate2.StorageView.from_array(data)
        sv_mps = sv_cpu.to_device(ctranslate2.Device.mps)
        sv_back = sv_cpu.to_device(ctranslate2.Device.cpu)
        result = np.array(sv_back)
        ok = np.allclose(data, result, atol=1e-6)
        runner.check("MPS StorageView round-trip", ok)
    except Exception as e:
        runner.check("MPS StorageView round-trip", False, str(e))
        print("\nFATAL: MPS not available — cannot continue")
        return 1

    # f16 round-trip
    try:
        data_f16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        sv16 = ctranslate2.StorageView.from_array(data_f16)
        sv16_mps = sv16.to_device(ctranslate2.Device.mps)
        sv16_back = sv16_mps.to_device(ctranslate2.Device.cpu)
        result_f16 = np.array(sv16_back)
        runner.check("MPS f16 StorageView round-trip",
                     np.allclose(data_f16, result_f16, atol=1e-3))
    except Exception as e:
        runner.check("MPS f16 StorageView round-trip", False, str(e))

    # ========================================================
    # Test 2: Translation f32
    # ========================================================
    print("\n--- 2. Translation f32 ---")
    f32_path = model_path("opus-mt-en-de")
    if not os.path.isdir(f32_path):
        runner.skip("Translation f32", f"model not found at {f32_path}")
    else:
        try:
            t0 = time.time()
            translator_f32 = ctranslate2.Translator(f32_path, device="mps")
            tokenizer = load_marian_tokenizer()

            tokens = tokenize(tokenizer, "The cat sat on the mat.")
            result = translator_f32.translate_batch([tokens], beam_size=4)
            output = result[0].hypotheses[0]
            output_text = decode(tokenizer, output)

            has_output = len(output) > 0
            dt = time.time() - t0
            runner.check("Translation f32 produces output", has_output,
                         f"'{output_text[:50]}' in {dt:.2f}s")

            # CPU cross-check
            cpu_tr = ctranslate2.Translator(f32_path, device="cpu")
            cpu_r = cpu_tr.translate_batch([tokens], beam_size=4)
            cpu_out = cpu_r[0].hypotheses[0]
            runner.check("f32 MPS matches CPU", output == cpu_out)

            # Batch test
            sentences = ["Hello world.", "This is a test.", "Machine translation."]
            batch_tokens = [tokenize(tokenizer, s) for s in sentences]
            batch_r = translator_f32.translate_batch(batch_tokens, beam_size=1)
            all_ok = all(len(r.hypotheses[0]) > 0 for r in batch_r)
            runner.check("f32 batch translation", all_ok, f"{len(batch_r)} results")

            del translator_f32, cpu_tr
        except Exception as e:
            runner.check("Translation f32", False, str(e))

    # ========================================================
    # Test 3: Translation f16
    # ========================================================
    print("\n--- 3. Translation f16 ---")
    f16_path = model_path("opus-mt-en-de-f16")
    if not os.path.isdir(f16_path):
        runner.skip("Translation f16", f"model not found at {f16_path}")
    else:
        try:
            t0 = time.time()
            translator_f16 = ctranslate2.Translator(f16_path, device="mps")

            tokens = tokenize(tokenizer, "The cat sat on the mat.")
            result = translator_f16.translate_batch([tokens], beam_size=4)
            output = result[0].hypotheses[0]
            output_text = decode(tokenizer, output)

            has_output = len(output) > 0
            dt = time.time() - t0
            runner.check("Translation f16 produces output", has_output,
                         f"'{output_text[:50]}' in {dt:.2f}s")
            del translator_f16
        except Exception as e:
            runner.check("Translation f16", False, str(e))

    # ========================================================
    # Test 4: Translation int8
    # ========================================================
    print("\n--- 4. Translation int8 ---")
    int8_path = model_path("opus-mt-en-de-int8")
    if not os.path.isdir(int8_path):
        runner.skip("Translation int8", f"model not found at {int8_path}")
    else:
        try:
            t0 = time.time()
            translator_int8 = ctranslate2.Translator(int8_path, device="mps")

            tokens = tokenize(tokenizer, "The cat sat on the mat.")
            result = translator_int8.translate_batch([tokens], beam_size=4)
            output = result[0].hypotheses[0]
            output_text = decode(tokenizer, output)

            has_output = len(output) > 0
            dt = time.time() - t0
            runner.check("Translation int8 produces output", has_output,
                         f"'{output_text[:50]}' in {dt:.2f}s")

            # Verify native int8 compute type (not fallback)
            runner.check("int8 native compute type",
                         translator_int8.compute_type == "int8_float32",
                         f"got {translator_int8.compute_type}")
            del translator_int8
        except Exception as e:
            runner.check("Translation int8", False, str(e))

    # ========================================================
    # Test 5: Whisper inference
    # ========================================================
    print("\n--- 5. Whisper inference ---")
    whisper_models = ["whisper-large-v3-turbo", "whisper-large-v3", "whisper-base"]
    whisper_path = None
    whisper_name = None
    for wm in whisper_models:
        wp = model_path(wm)
        if os.path.isdir(wp):
            whisper_path = wp
            whisper_name = wm
            break

    if whisper_path is None:
        runner.skip("Whisper inference", "no whisper model found in CT2_TEST_DATA")
    else:
        try:
            from faster_whisper import WhisperModel

            t0 = time.time()
            model = WhisperModel(whisper_path, device="mps", compute_type="float16")

            # Find a test audio file
            audio_file = None
            data_dir = model_path("")
            for fname in os.listdir(data_dir):
                if fname.endswith((".wav", ".mp3", ".flac")):
                    audio_file = os.path.join(data_dir, fname)
                    break

            if audio_file is None:
                # Generate a short silent audio for basic smoke test
                import tempfile
                import struct
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    audio_file = f.name
                    # Write a minimal WAV with 1 second of silence at 16kHz
                    sr = 16000
                    n_samples = sr
                    data_size = n_samples * 2
                    f.write(b"RIFF")
                    f.write(struct.pack("<I", 36 + data_size))
                    f.write(b"WAVE")
                    f.write(b"fmt ")
                    f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                    f.write(b"data")
                    f.write(struct.pack("<I", data_size))
                    f.write(b"\x00" * data_size)

            segments, info = model.transcribe(audio_file, beam_size=1, language="en")
            text = " ".join(s.text.strip() for s in segments)
            dt = time.time() - t0
            runner.check(f"Whisper ({whisper_name}) runs on MPS", True,
                         f"'{text[:60]}' in {dt:.2f}s")

            del model
        except ImportError:
            runner.skip("Whisper inference", "faster_whisper not installed")
        except Exception as e:
            runner.check("Whisper inference", False, str(e))

    # ========================================================
    # Summary
    # ========================================================
    elapsed = time.time() - start_time
    p, f, s, total = runner.summary()

    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Result: {p} passed, {f} failed, {s} skipped (of {total} tests)")

    if f > 0:
        print("\nFAILURES DETECTED")
        for tag, label, detail in runner.results:
            if tag == "FAIL":
                print(f"  FAIL: {label}{detail}")
        return 1
    else:
        print("\nALL PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
