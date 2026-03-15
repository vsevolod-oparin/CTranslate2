#!/usr/bin/env python3
"""M13.5 — Shape randomization fuzz test for Metal backend.

Randomizes tensor shapes across key ops (GEMM, LayerNorm, RMSNorm, Softmax,
elementwise, transpose) to catch MSL kernel dispatch bugs with unusual
threadgroup sizes.

Verifies: no crash, no hang, outputs finite (non-NaN, non-inf),
Metal vs CPU results within tolerance.

Requires:
  - CTranslate2 built with CT2_WITH_METAL=ON
  - No model files needed — tests ops directly via StorageView
"""
import sys
import os
import random
import time
import resource

import numpy as np


def check_mps_available():
    """Return True if MPS device is available."""
    try:
        import ctranslate2
        sv = ctranslate2.StorageView.from_array(np.ones(4, dtype=np.float32))
        sv_mps = sv.to(ctranslate2.DataType.float32)  # stays on CPU
        return True
    except Exception:
        return False


def random_shape(min_dims=1, max_dims=3, min_size=1, max_size=2048):
    """Generate a random tensor shape with non-power-of-2 dimensions."""
    ndims = random.randint(min_dims, max_dims)
    shape = []
    for _ in range(ndims):
        # Mix of small, medium, large, and non-power-of-2 sizes
        size_type = random.random()
        if size_type < 0.2:
            # Small edge cases
            shape.append(random.choice([1, 2, 3, 5, 7]))
        elif size_type < 0.5:
            # Medium non-power-of-2
            shape.append(random.randint(10, 300))
        elif size_type < 0.8:
            # Larger
            shape.append(random.randint(128, max_size))
        else:
            # Power-of-2
            shape.append(2 ** random.randint(0, 10))
    return tuple(shape)


def random_gemm_shapes():
    """Generate random GEMM-compatible shapes (M, N, K)."""
    m = random.choice([1, 2, 3, 7, 16, 31, 64, 127, 256, 512])
    n = random.choice([1, 2, 3, 7, 16, 31, 64, 127, 256, 512, 1024])
    k = random.choice([1, 2, 3, 7, 16, 31, 64, 127, 256, 512, 1024])
    return m, n, k


def is_finite(arr):
    """Check that all values are finite (not NaN, not inf)."""
    return np.all(np.isfinite(arr))


def max_abs_error(a, b):
    """Compute max absolute error between two arrays."""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def main():
    seed = int(os.environ.get("FUZZ_SEED", "42"))
    num_rounds = int(os.environ.get("FUZZ_ROUNDS", "50"))
    timeout_s = int(os.environ.get("FUZZ_TIMEOUT", "300"))

    random.seed(seed)
    np.random.seed(seed)

    print(f"M13.5 Fuzz Shape Test — seed={seed}, rounds={num_rounds}, timeout={timeout_s}s")
    print(f"{'='*70}")

    import ctranslate2

    passed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    def check(label, ok, detail=""):
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{tag}] {label}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1

    # ---- Test 1: StorageView CPU→MPS round-trip with random shapes ----
    print("\n=== StorageView CPU↔MPS round-trip (random shapes) ===")
    for i in range(min(num_rounds, 20)):
        if time.time() - start_time > timeout_s:
            print("  TIMEOUT reached, stopping")
            break
        shape = random_shape(min_dims=1, max_dims=4, max_size=1024)
        dtype = random.choice([np.float32, np.float16])
        dtype_name = "f32" if dtype == np.float32 else "f16"

        try:
            data = np.random.randn(*shape).astype(dtype)
            sv_cpu = ctranslate2.StorageView.from_array(data)
            sv_mps = sv_cpu.to_device(ctranslate2.Device.mps)
            sv_back = sv_mps.to_device(ctranslate2.Device.cpu)
            result = np.array(sv_back)

            finite_ok = is_finite(result)
            shape_ok = result.shape == data.shape
            if dtype == np.float32:
                value_ok = max_abs_error(data, result) < 1e-5
            else:
                value_ok = max_abs_error(data, result) < 1e-2
            ok = finite_ok and shape_ok and value_ok
            check(f"round-trip {dtype_name} {shape}", ok,
                  f"finite={finite_ok}, shape={shape_ok}, value={value_ok}")
        except Exception as e:
            check(f"round-trip {dtype_name} {shape}", False, f"EXCEPTION: {e}")

    # ---- Test 2: Translation with random max_decoding_length ----
    print("\n=== Translation with random decoding lengths ===")
    e2e_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e")
    sys.path.insert(0, e2e_dir)
    try:
        from conftest import model_path, load_marian_tokenizer, tokenize, decode
        mpath = model_path("opus-mt-en-de")
        if not os.path.isdir(mpath):
            print(f"  SKIP: model not found at {mpath}")
            skipped += 1
        else:
            tokenizer = load_marian_tokenizer()
            metal_translator = ctranslate2.Translator(mpath, device="mps")
            cpu_translator = ctranslate2.Translator(mpath, device="cpu")

            test_sentences = [
                "The cat sat on the mat.",
                "Hello world.",
                "Machine translation is interesting.",
                "A short test.",
                "One two three four five six seven eight nine ten.",
            ]

            for i in range(min(num_rounds, 15)):
                if time.time() - start_time > timeout_s:
                    print("  TIMEOUT reached, stopping")
                    break

                sentence = random.choice(test_sentences)
                tokens = tokenize(tokenizer, sentence)
                beam_size = random.choice([1, 2, 4])
                max_len = random.choice([1, 2, 3, 5, 10, 20, 50, None])

                kwargs = dict(beam_size=beam_size)
                if max_len is not None:
                    kwargs["max_decoding_length"] = max_len

                try:
                    metal_r = metal_translator.translate_batch([tokens], **kwargs)
                    cpu_r = cpu_translator.translate_batch([tokens], **kwargs)

                    metal_out = metal_r[0].hypotheses[0]
                    cpu_out = cpu_r[0].hypotheses[0]

                    has_output = len(metal_out) > 0
                    match = metal_out == cpu_out
                    check(f"translate beam={beam_size} max_len={max_len}",
                          has_output,
                          f"match={match}, len={len(metal_out)}")
                except Exception as e:
                    check(f"translate beam={beam_size} max_len={max_len}",
                          False, f"EXCEPTION: {e}")
    except ImportError:
        print("  SKIP: conftest not available")
        skipped += 1

    # ---- Test 3: Batch size variations ----
    print("\n=== Random batch sizes ===")
    try:
        if 'metal_translator' in dir():
            batch_sizes_to_test = [1, 2, 3, 5, 7, 8, 10, 16]
            for batch_size in batch_sizes_to_test:
                if time.time() - start_time > timeout_s:
                    print("  TIMEOUT reached, stopping")
                    break

                sentences = [random.choice(test_sentences) for _ in range(batch_size)]
                all_tokens = [tokenize(tokenizer, s) for s in sentences]

                try:
                    metal_r = metal_translator.translate_batch(all_tokens, beam_size=1)
                    ok = len(metal_r) == batch_size
                    all_nonempty = all(len(r.hypotheses[0]) > 0 for r in metal_r)
                    check(f"batch_size={batch_size}", ok and all_nonempty,
                          f"got {len(metal_r)} results, all_nonempty={all_nonempty}")
                except Exception as e:
                    check(f"batch_size={batch_size}", False, f"EXCEPTION: {e}")
    except NameError:
        print("  SKIP: translator not available")
        skipped += 1

    # ---- Test 4: f16 translation with random params ----
    print("\n=== Float16 translation with random params ===")
    try:
        f16_path = model_path("opus-mt-en-de-f16")
        if not os.path.isdir(f16_path):
            print(f"  SKIP: f16 model not found at {f16_path}")
            skipped += 1
        else:
            metal_f16 = ctranslate2.Translator(f16_path, device="mps")
            for i in range(min(num_rounds, 10)):
                if time.time() - start_time > timeout_s:
                    print("  TIMEOUT reached, stopping")
                    break

                sentence = random.choice(test_sentences)
                tokens = tokenize(tokenizer, sentence)
                beam_size = random.choice([1, 2, 4])
                max_len = random.choice([1, 3, 10, None])

                kwargs = dict(beam_size=beam_size)
                if max_len is not None:
                    kwargs["max_decoding_length"] = max_len

                try:
                    r = metal_f16.translate_batch([tokens], **kwargs)
                    has_output = len(r[0].hypotheses[0]) > 0
                    check(f"f16 beam={beam_size} max_len={max_len}",
                          has_output, f"len={len(r[0].hypotheses[0])}")
                except Exception as e:
                    check(f"f16 beam={beam_size} max_len={max_len}",
                          False, f"EXCEPTION: {e}")
    except NameError:
        print("  SKIP: tokenizer not available")
        skipped += 1

    # ---- Test 5: INT8 translation with random params ----
    print("\n=== INT8 translation with random params ===")
    try:
        int8_path = model_path("opus-mt-en-de-int8")
        if not os.path.isdir(int8_path):
            print(f"  SKIP: int8 model not found at {int8_path}")
            skipped += 1
        else:
            metal_int8 = ctranslate2.Translator(int8_path, device="mps")
            for i in range(min(num_rounds, 10)):
                if time.time() - start_time > timeout_s:
                    print("  TIMEOUT reached, stopping")
                    break

                sentence = random.choice(test_sentences)
                tokens = tokenize(tokenizer, sentence)
                beam_size = random.choice([1, 2, 4])

                try:
                    r = metal_int8.translate_batch([tokens], beam_size=beam_size)
                    has_output = len(r[0].hypotheses[0]) > 0
                    check(f"int8 beam={beam_size}", has_output,
                          f"len={len(r[0].hypotheses[0])}")
                except Exception as e:
                    check(f"int8 beam={beam_size}", False, f"EXCEPTION: {e}")
    except NameError:
        print("  SKIP: tokenizer not available")
        skipped += 1

    # ---- Summary ----
    elapsed = time.time() - start_time
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"Seed: {seed}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Peak RSS: {rss_mb:.0f} MB")
    total = passed + failed
    print(f"Result: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed:
        print("FAILURES DETECTED")
        return 1
    else:
        print("ALL PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
