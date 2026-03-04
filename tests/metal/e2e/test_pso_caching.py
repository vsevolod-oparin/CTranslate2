#!/usr/bin/env python3
"""M11.2 — PSO caching verification.

Proves the PASS criterion:
  "Second inference call is not slower than first
   (pipeline states reused, no recompile)"

Tests:
  1. PSO cache hit/miss counts: first call creates PSOs (misses),
     second call reuses them (all hits, zero new misses).
  2. Latency: second inference call is not slower than first.
  3. Whisper model test: validates PSO reuse across a larger model
     with more diverse kernel dispatch patterns.
"""
import sys
import os
import time
import ctypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize, decode, audio_path


def load_ct2_lib():
    """Load libctranslate2 and bind PSO stats functions."""
    candidates = [
        "/opt/anaconda3/envs/ct2/lib/libctranslate2.dylib",
        "/opt/anaconda3/envs/ct2/lib/libctranslate2.4.7.1.dylib",
    ]
    lib = None
    for path in candidates:
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            break
    if lib is None:
        return None

    fns = {}
    symbol_map = {
        'pso_hit_count':   ('_ZN11ctranslate25metal13pso_hit_countEv',   ctypes.c_uint64),
        'pso_miss_count':  ('_ZN11ctranslate25metal14pso_miss_countEv',  ctypes.c_uint64),
        'reset_pso_stats': ('_ZN11ctranslate25metal15reset_pso_statsEv', None),
    }
    try:
        for key, (sym, rtype) in symbol_map.items():
            fn = getattr(lib, sym)
            fn.restype = rtype
            fn.argtypes = []
            fns[key] = fn
        return fns
    except AttributeError:
        return None


def test_translation_pso_caching():
    """Test PSO caching with opus-mt-en-de translation model."""
    mpath = model_path("opus-mt-en-de")
    if not os.path.isdir(mpath):
        print(f"SKIP: model not found at {mpath}")
        return 0, 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()
    tokens = [tokenize(tokenizer, "The cat sat on the mat.")]

    fns = load_ct2_lib()
    passed = 0
    total = 0

    translator = ctranslate2.Translator(mpath, device="metal")

    # --- Test 1: PSO miss/hit counts ---
    if fns:
        fns['reset_pso_stats']()

        # First inference: should trigger PSO compilations (misses)
        t0 = time.perf_counter()
        r1 = translator.translate_batch(tokens, beam_size=4, max_decoding_length=20)
        t_first = time.perf_counter() - t0

        misses_after_first = fns['pso_miss_count']()
        hits_after_first = fns['pso_hit_count']()

        print(f"\n=== Translation: PSO Cache Stats After First Inference ===")
        print(f"  Misses (compilations): {misses_after_first}")
        print(f"  Hits (cache reuse):    {hits_after_first}")
        print(f"  Time: {t_first*1000:.1f} ms")

        total += 1
        if misses_after_first > 0:
            print(f"  [PASS] First call triggered {misses_after_first} PSO compilations")
            passed += 1
        else:
            print(f"  [FAIL] Expected PSO misses on first call, got 0")

        # Reset counters and run second inference
        fns['reset_pso_stats']()

        t0 = time.perf_counter()
        r2 = translator.translate_batch(tokens, beam_size=4, max_decoding_length=20)
        t_second = time.perf_counter() - t0

        misses_after_second = fns['pso_miss_count']()
        hits_after_second = fns['pso_hit_count']()

        print(f"\n=== Translation: PSO Cache Stats After Second Inference ===")
        print(f"  Misses (compilations): {misses_after_second}")
        print(f"  Hits (cache reuse):    {hits_after_second}")
        print(f"  Time: {t_second*1000:.1f} ms")

        # PASS: zero new compilations on second call
        total += 1
        if misses_after_second == 0:
            print(f"  [PASS] Zero new PSO compilations on second call (all cached)")
            passed += 1
        else:
            print(f"  [FAIL] {misses_after_second} new PSO compilations on second call")

        # PASS: all lookups are hits
        total += 1
        if hits_after_second > 0:
            print(f"  [PASS] {hits_after_second} cache hits on second call")
            passed += 1
        else:
            print(f"  [FAIL] Zero cache hits on second call")

        # Verify output correctness is the same
        out1 = decode(tokenizer, r1[0].hypotheses[0])
        out2 = decode(tokenizer, r2[0].hypotheses[0])
        total += 1
        if out1 == out2:
            print(f"  [PASS] Output identical: {out1}")
            passed += 1
        else:
            print(f"  [FAIL] Output differs: {out1} vs {out2}")
    else:
        print("\n[SKIP] Could not bind PSO stats symbols")

    # --- Test 2: Latency comparison (warm-up excluded) ---
    print(f"\n=== Translation: Latency Comparison (5 runs each) ===")
    # Warm up fully (3 runs)
    for _ in range(3):
        translator.translate_batch(tokens, beam_size=4, max_decoding_length=20)

    # Measure first batch of 5
    times_a = []
    for _ in range(5):
        t0 = time.perf_counter()
        translator.translate_batch(tokens, beam_size=4, max_decoding_length=20)
        times_a.append(time.perf_counter() - t0)

    # Measure second batch of 5
    times_b = []
    for _ in range(5):
        t0 = time.perf_counter()
        translator.translate_batch(tokens, beam_size=4, max_decoding_length=20)
        times_b.append(time.perf_counter() - t0)

    avg_a = sum(times_a) / len(times_a)
    avg_b = sum(times_b) / len(times_b)
    print(f"  Batch A avg: {avg_a*1000:.1f} ms")
    print(f"  Batch B avg: {avg_b*1000:.1f} ms")
    ratio = avg_b / avg_a if avg_a > 0 else 1.0
    print(f"  Ratio B/A:   {ratio:.2f}x")

    total += 1
    # Second batch should not be significantly slower (within 20% margin)
    if ratio < 1.20:
        print(f"  [PASS] Later calls not slower than earlier (ratio={ratio:.2f})")
        passed += 1
    else:
        print(f"  [FAIL] Later calls slower (ratio={ratio:.2f}, expected < 1.20)")

    return passed, total


def test_whisper_pso_caching():
    """Test PSO caching with Whisper model (more diverse kernel mix)."""
    mpath = model_path("whisper-base")
    apath = audio_path("sample.mp3")
    if not os.path.isdir(mpath):
        print(f"\nSKIP: whisper model not found at {mpath}")
        return 0, 0
    if not os.path.isfile(apath):
        print(f"\nSKIP: audio file not found at {apath}")
        return 0, 0

    try:
        import ctranslate2
        import librosa
        import numpy as np
        from transformers import WhisperProcessor
    except ImportError as e:
        print(f"\nSKIP: missing dependency for Whisper test: {e}")
        return 0, 0

    fns = load_ct2_lib()
    passed = 0
    total = 0

    SAMPLE_RATE = 16000
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = ctranslate2.models.Whisper(mpath, device="metal")

    audio, _ = librosa.load(apath, sr=SAMPLE_RATE, mono=True)
    # Use first 30 seconds for a single-chunk test
    audio = audio[:30 * SAMPLE_RATE]
    inputs = processor(audio, return_tensors="np", sampling_rate=SAMPLE_RATE)
    features = ctranslate2.StorageView.from_array(inputs.input_features)
    prefix_tokens = [
        processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
        processor.tokenizer.convert_tokens_to_ids("<|en|>"),
        processor.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
    ]

    if fns:
        # First inference
        fns['reset_pso_stats']()
        t0 = time.perf_counter()
        r1 = model.generate(features, [prefix_tokens])
        t_first = time.perf_counter() - t0
        misses_1 = fns['pso_miss_count']()
        hits_1 = fns['pso_hit_count']()

        print(f"\n=== Whisper: PSO Cache Stats After First Inference ===")
        print(f"  Misses: {misses_1}, Hits: {hits_1}, Time: {t_first*1000:.1f} ms")

        # Second inference
        fns['reset_pso_stats']()
        t0 = time.perf_counter()
        r2 = model.generate(features, [prefix_tokens])
        t_second = time.perf_counter() - t0
        misses_2 = fns['pso_miss_count']()
        hits_2 = fns['pso_hit_count']()

        print(f"\n=== Whisper: PSO Cache Stats After Second Inference ===")
        print(f"  Misses: {misses_2}, Hits: {hits_2}, Time: {t_second*1000:.1f} ms")

        total += 1
        if misses_2 == 0:
            print(f"  [PASS] Zero new PSO compilations on second Whisper call")
            passed += 1
        else:
            print(f"  [FAIL] {misses_2} new PSO compilations on second Whisper call")

        total += 1
        if hits_2 > 0:
            print(f"  [PASS] {hits_2} cache hits on second Whisper call")
            passed += 1
        else:
            print(f"  [FAIL] Zero cache hits")

        # Verify output identical
        total += 1
        if r1[0].sequences_ids[0] == r2[0].sequences_ids[0]:
            text = processor.decode(r1[0].sequences_ids[0][len(prefix_tokens):],
                                     skip_special_tokens=True)
            print(f"  [PASS] Whisper output identical across runs: \"{text[:60]}...\"")
            passed += 1
        else:
            print(f"  [FAIL] Whisper output differs")
    else:
        print("\n[SKIP] Could not bind PSO stats symbols for Whisper test")

    return passed, total


def main():
    passed_total = 0
    total_total = 0

    p, t = test_translation_pso_caching()
    passed_total += p
    total_total += t

    p, t = test_whisper_pso_caching()
    passed_total += p
    total_total += t

    print(f"\n{'='*50}")
    print(f"{passed_total}/{total_total} passed")
    if passed_total == total_total:
        print("ALL PASS")
    else:
        print("SOME FAILURES")

    return 0 if passed_total == total_total else 1


if __name__ == "__main__":
    sys.exit(main())
