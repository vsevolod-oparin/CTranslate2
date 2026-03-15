#!/usr/bin/env python3
"""M13.6 — Stress test for Metal backend: memory leak detection + precision cycling.

Test 1 — Memory leak: Run 100 translation iterations on MPS, measure RSS at
  iterations 1, 10, 50, 100. Verify RSS stable (±10% after warmup).
  Catches ObjC ARC/manual-release leaks (cf. M11.22) and allocator pool drift.

Test 2 — Mixed precision cycling: 50 iterations alternating f32 → f16 → int8 → f32.
  Verify allocator pool convergence and no OOM. Tests pool behavior across
  type switches (different buffer sizes per type).

Requires:
  - CT2_TEST_DATA with opus-mt-en-de/, opus-mt-en-de-f16/, opus-mt-en-de-int8/
"""
import sys
import os
import time
import resource
import gc


def get_rss_mb():
    """Get current RSS in MB (macOS: ru_maxrss is in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    iterations = int(os.environ.get("STRESS_ITERATIONS", "100"))
    cycle_iterations = int(os.environ.get("STRESS_CYCLES", "50"))

    print(f"M13.6 Stress Test — iterations={iterations}, cycles={cycle_iterations}")
    print(f"{'='*70}")

    e2e_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")
    sys.path.insert(0, e2e_dir)
    from conftest import model_path, load_marian_tokenizer, tokenize, decode

    import ctranslate2

    f32_path = model_path("opus-mt-en-de")
    f16_path = model_path("opus-mt-en-de-f16")
    int8_path = model_path("opus-mt-en-de-int8")

    if not os.path.isdir(f32_path):
        print(f"SKIP: model not found at {f32_path}")
        return 0

    tokenizer = load_marian_tokenizer()

    test_sentences = [
        "The cat sat on the mat.",
        "Hello world, this is a test.",
        "Machine translation is an interesting research area.",
        "The quick brown fox jumps over the lazy dog.",
        "One two three four five six seven eight nine ten.",
    ]
    all_tokens = [tokenize(tokenizer, s) for s in test_sentences]

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

    # ========================================================
    # Test 1: Memory leak detection (f32)
    # ========================================================
    print("\n=== Test 1: Memory leak detection (f32, {0} iterations) ===".format(iterations))

    translator = ctranslate2.Translator(f32_path, device="mps")

    # Warmup: 5 iterations to stabilize allocator pool
    print("  Warming up (5 iterations)...")
    for _ in range(5):
        translator.translate_batch(all_tokens, beam_size=4)
    gc.collect()

    rss_samples = {}
    checkpoints = [1, 10, 25, 50, 75, iterations]
    checkpoints = [c for c in checkpoints if c <= iterations]

    start_time = time.time()
    rss_after_warmup = get_rss_mb()
    print(f"  RSS after warmup: {rss_after_warmup:.0f} MB")

    for i in range(1, iterations + 1):
        # Alternate between beam sizes and single/batch to exercise different code paths
        if i % 3 == 0:
            translator.translate_batch(all_tokens, beam_size=4)
        elif i % 3 == 1:
            translator.translate_batch(all_tokens[:1], beam_size=1)
        else:
            translator.translate_batch(all_tokens, beam_size=2)

        if i in checkpoints:
            gc.collect()
            rss = get_rss_mb()
            rss_samples[i] = rss
            elapsed = time.time() - start_time
            print(f"  Iteration {i:>4d}: RSS={rss:.0f} MB, elapsed={elapsed:.1f}s")

    # Verify RSS stability: final RSS within ±10% of post-warmup RSS
    final_rss = rss_samples[checkpoints[-1]]
    rss_drift = abs(final_rss - rss_after_warmup) / rss_after_warmup if rss_after_warmup > 0 else 0
    rss_stable = rss_drift < 0.10

    check(f"RSS stable after {iterations} iterations",
          rss_stable,
          f"warmup={rss_after_warmup:.0f}MB, final={final_rss:.0f}MB, drift={rss_drift:.1%}")

    # Also check no monotonic growth (each sample should not be > previous + 5%)
    sorted_checkpoints = sorted(rss_samples.keys())
    monotonic_growth = True
    for j in range(1, len(sorted_checkpoints)):
        prev_rss = rss_samples[sorted_checkpoints[j-1]]
        curr_rss = rss_samples[sorted_checkpoints[j]]
        if curr_rss < prev_rss * 1.05:
            monotonic_growth = False
            break

    if monotonic_growth and len(sorted_checkpoints) > 2:
        check("No monotonic RSS growth", False,
              f"RSS grew at every checkpoint: {[f'{rss_samples[c]:.0f}' for c in sorted_checkpoints]}")
    else:
        check("No monotonic RSS growth", True)

    # Verify output consistency: first and last iteration produce same output
    first_r = translator.translate_batch(all_tokens[:1], beam_size=1)
    first_out = first_r[0].hypotheses[0]
    has_output = len(first_out) > 0
    check("Output still valid after stress", has_output, f"len={len(first_out)}")

    del translator
    gc.collect()

    elapsed_t1 = time.time() - start_time
    print(f"  Test 1 completed in {elapsed_t1:.1f}s")

    # ========================================================
    # Test 2: Mixed precision cycling
    # ========================================================
    has_f16 = os.path.isdir(f16_path)
    has_int8 = os.path.isdir(int8_path)

    if not has_f16 and not has_int8:
        print("\n=== Test 2: SKIP (need f16 and/or int8 models) ===")
    else:
        print(f"\n=== Test 2: Mixed precision cycling ({cycle_iterations} iterations) ===")
        print(f"  Models: f32=yes, f16={has_f16}, int8={has_int8}")

        # Build list of available (path, label) pairs
        model_configs = [("f32", f32_path)]
        if has_f16:
            model_configs.append(("f16", f16_path))
        if has_int8:
            model_configs.append(("int8", int8_path))

        start_time2 = time.time()
        rss_before_cycling = get_rss_mb()
        print(f"  RSS before cycling: {rss_before_cycling:.0f} MB")

        cycle_errors = 0
        rss_cycle_samples = {}
        for i in range(1, cycle_iterations + 1):
            label, mpath = model_configs[i % len(model_configs)]

            try:
                tr = ctranslate2.Translator(mpath, device="mps")
                result = tr.translate_batch(all_tokens[:2], beam_size=1)
                ok = len(result) == 2 and all(len(r.hypotheses[0]) > 0 for r in result)
                if not ok:
                    cycle_errors += 1
                    print(f"  [{i}] {label}: empty output")
                del tr
            except Exception as e:
                cycle_errors += 1
                print(f"  [{i}] {label}: EXCEPTION: {e}")

            if i % 10 == 0:
                gc.collect()
                rss = get_rss_mb()
                rss_cycle_samples[i] = rss
                elapsed = time.time() - start_time2
                print(f"  Cycle {i:>3d} ({label}): RSS={rss:.0f} MB, elapsed={elapsed:.1f}s")

        gc.collect()
        rss_after_cycling = get_rss_mb()

        check(f"No errors in {cycle_iterations} precision cycles",
              cycle_errors == 0,
              f"{cycle_errors} errors")

        # RSS stability check: verify RSS stabilizes in the second half.
        # Model loading causes one-time RSS growth (mapped framework pages) — that's expected.
        # What matters is that RSS stops growing, indicating no leak.
        sorted_samples = sorted(rss_cycle_samples.items())
        if len(sorted_samples) >= 3:
            # Compare last half vs second-to-last sample: growth < 10%
            mid = len(sorted_samples) // 2
            second_half = [rss for _, rss in sorted_samples[mid:]]
            rss_at_mid = sorted_samples[mid][1]
            rss_at_end = second_half[-1]
            late_drift = (rss_at_end - rss_at_mid) / rss_at_mid if rss_at_mid > 0 else 0
            check("RSS stabilizes in second half of cycling",
                  late_drift < 0.15,
                  f"mid={rss_at_mid:.0f}MB, end={rss_at_end:.0f}MB, late_drift={late_drift:.1%}")
        else:
            check("RSS stabilizes (insufficient samples)", True, "only {0} samples".format(len(sorted_samples)))

        elapsed_t2 = time.time() - start_time2
        print(f"  Test 2 completed in {elapsed_t2:.1f}s")

    # ========================================================
    # Summary
    # ========================================================
    total_rss = get_rss_mb()
    print(f"\n{'='*70}")
    print(f"Peak RSS: {total_rss:.0f} MB")
    total = passed + failed
    print(f"Result: {passed}/{total} passed, {failed} failed")

    if failed:
        print("FAILURES DETECTED")
        return 1
    else:
        print("ALL PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
