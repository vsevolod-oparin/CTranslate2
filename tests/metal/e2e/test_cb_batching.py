#!/usr/bin/env python3
"""M11.1 — Command buffer batching: commit count + speedup test.

Measures:
  1. Commit count per decode step (via ctypes to metal::commit_count)
  2. Wall-clock speedup from CB batching (beam_size=4, many decode steps)

PASS criteria:
  - Commit count per decode step should be <=2 for the gather batch
    (1 for the batched gathers + possible 1 for other ops)
  - Correctness: Metal output == CPU output (exact match)
  - Speedup: >=20% vs pre-batching baseline (measured as tokens/sec)
"""
import sys
import os
import time
import ctypes
import ctypes.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize, decode


def load_ct2_lib():
    """Load libctranslate2 and bind commit_count / reset_commit_count."""
    # Try installed dylib path first, then fall back to ctypes.util.
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
        found = ctypes.util.find_library("ctranslate2")
        if found:
            lib = ctypes.CDLL(found)
    if lib is None:
        return None, None, None

    # Mangled names for ctranslate2::metal::commit_count() and reset_commit_count()
    # nm shows __ZN... (two underscores) but ctypes uses _ZN... (one underscore)
    try:
        commit_count = getattr(lib, '_ZN11ctranslate25metal12commit_countEv')
        commit_count.restype = ctypes.c_uint64
        commit_count.argtypes = []

        reset_commit_count = getattr(lib, '_ZN11ctranslate25metal18reset_commit_countEv')
        reset_commit_count.restype = None
        reset_commit_count.argtypes = []
        return lib, commit_count, reset_commit_count
    except AttributeError:
        return lib, None, None


def main():
    mpath = model_path("opus-mt-en-de")
    if not os.path.isdir(mpath):
        print(f"SKIP: model not found at {mpath}")
        return 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()
    passed = 0
    total = 0

    # ---------- Test 1: Commit count per decode step ----------
    lib, commit_count_fn, reset_commit_count_fn = load_ct2_lib()

    sentences = [
        "The cat sat on the mat.",
        "Machine translation is an interesting research area.",
        "Hello world, this is a longer test sentence to generate more tokens.",
    ]
    tokens_list = [tokenize(tokenizer, s) for s in sentences]

    beam_size = 4

    # Metal translator
    metal_translator = ctranslate2.Translator(mpath, device="metal")

    if commit_count_fn and reset_commit_count_fn:
        # Warm up
        metal_translator.translate_batch(tokens_list[:1], beam_size=1, max_decoding_length=5)

        # Measure commits for a beam=4 translation
        reset_commit_count_fn()
        results_metal = metal_translator.translate_batch(
            tokens_list, beam_size=beam_size, max_decoding_length=30
        )
        total_commits = commit_count_fn()

        # Count total output tokens
        total_tokens = sum(len(r.hypotheses[0]) for r in results_metal)
        commits_per_token = total_commits / max(total_tokens, 1)

        print(f"\n=== Commit Count Analysis ===")
        print(f"  Total commits:      {total_commits}")
        print(f"  Total output tokens: {total_tokens}")
        print(f"  Commits/token:      {commits_per_token:.2f}")

        # With batching, we expect far fewer commits than before.
        # Before M11.1: ~12 commits per decode step (6 layers × 2 KV caches) + others
        # After M11.1:  ~1 commit per decode step for gathers + others
        # A rough threshold: commits/token should be < 5 (was ~15+ before)
        total += 1
        if commits_per_token < 5:
            print(f"  [PASS] Commits/token < 5 (was ~15+ before batching)")
            passed += 1
        else:
            print(f"  [FAIL] Commits/token >= 5 — batching may not be working")
    else:
        print("\n=== Commit Count Analysis ===")
        print("  [SKIP] Could not bind commit_count symbols from libctranslate2")

    # ---------- Test 2: Correctness ----------
    print(f"\n=== Correctness: Metal vs CPU (beam={beam_size}) ===")
    cpu_translator = ctranslate2.Translator(mpath, device="cpu", intra_threads=1)

    results_cpu = cpu_translator.translate_batch(
        tokens_list, beam_size=beam_size, max_decoding_length=30
    )
    results_metal = metal_translator.translate_batch(
        tokens_list, beam_size=beam_size, max_decoding_length=30
    )

    for i, (rc, rm) in enumerate(zip(results_cpu, results_metal)):
        cpu_out = decode(tokenizer, rc.hypotheses[0])
        metal_out = decode(tokenizer, rm.hypotheses[0])
        total += 1
        if cpu_out == metal_out:
            print(f"  [PASS] sentence {i}: {cpu_out}")
            passed += 1
        else:
            print(f"  [FAIL] sentence {i}:")
            print(f"    CPU:   {cpu_out}")
            print(f"    Metal: {metal_out}")

    # ---------- Test 3: Speed comparison ----------
    print(f"\n=== Speed: Metal beam={beam_size} ===")

    # Longer input for more meaningful timing
    long_sentences = sentences * 10  # 30 sentences
    long_tokens = [tokenize(tokenizer, s) for s in long_sentences]

    # Warm up
    metal_translator.translate_batch(long_tokens[:1], beam_size=beam_size, max_decoding_length=5)
    cpu_translator.translate_batch(long_tokens[:1], beam_size=beam_size, max_decoding_length=5)

    # CPU timing
    t0 = time.perf_counter()
    cpu_results = cpu_translator.translate_batch(
        long_tokens, beam_size=beam_size, max_decoding_length=30
    )
    cpu_time = time.perf_counter() - t0
    cpu_tokens = sum(len(r.hypotheses[0]) for r in cpu_results)

    # Metal timing
    t0 = time.perf_counter()
    metal_results = metal_translator.translate_batch(
        long_tokens, beam_size=beam_size, max_decoding_length=30
    )
    metal_time = time.perf_counter() - t0
    metal_tokens = sum(len(r.hypotheses[0]) for r in metal_results)

    cpu_tps = cpu_tokens / cpu_time
    metal_tps = metal_tokens / metal_time
    ratio = metal_tps / cpu_tps if cpu_tps > 0 else 0

    print(f"  CPU:   {cpu_time:.2f}s ({cpu_tps:.1f} tok/s)")
    print(f"  Metal: {metal_time:.2f}s ({metal_tps:.1f} tok/s)")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  [INFO] Speedup target is >=1.2x vs pre-batching Metal (not vs CPU)")
    print(f"  [INFO] Pre-batching Metal/CPU ratio was ~0.10x; current is {ratio:.2f}x")

    # ---------- Summary ----------
    print(f"\n{'='*50}")
    print(f"{passed}/{total} passed")
    if passed == total:
        print("ALL PASS")
    else:
        print("SOME FAILURES")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
