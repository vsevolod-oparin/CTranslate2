#!/usr/bin/env python3
"""M10 Review T4 — INT8 model translation test on Metal.

Tests that INT8-quantized models loaded on Metal produce correct output.
Currently the Metal backend does NOT advertise INT8 support (mayiuse_int8
returns false), so the runtime auto-converts INT8 weights to float32.
This test verifies:
  1. INT8 model loads on Metal without error (auto-fallback to float32)
  2. The fallback produces correct translations (matches CPU float32)
  3. Greedy, beam search, and batch modes all work

Note: M9.2 implemented INT8 GEMM kernels (quantize/dequantize + MPS GEMM),
but the end-to-end INT8 pipeline has a bug (produces garbage output — empty
for greedy, infinite repetition for beam search). Enabling mayiuse_int8 for
Metal is blocked until this pipeline bug is fixed. See milestone-10-review.md
for details.

Requires:
  - CT2_TEST_DATA with opus-mt-en-de/ (float32) and opus-mt-en-de-int8/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize, decode


def main():
    f32_path = model_path("opus-mt-en-de")
    int8_path = model_path("opus-mt-en-de-int8")

    if not os.path.isdir(f32_path):
        print(f"SKIP: float32 model not found at {f32_path}")
        return 0
    if not os.path.isdir(int8_path):
        print(f"SKIP: INT8 model not found at {int8_path}")
        return 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()

    print("Loading models...")
    cpu_f32 = ctranslate2.Translator(f32_path, device="cpu")
    # INT8 model on Metal: runtime auto-converts to float32 (no native INT8 yet)
    metal_int8 = ctranslate2.Translator(int8_path, device="metal")

    test_sentences = [
        "The cat sat on the mat.",
        "Hello world, this is a test.",
        "Machine translation is an interesting research area.",
        "The quick brown fox jumps over the lazy dog.",
    ]

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

    # --- Verify auto-fallback compute type ---
    print("\n=== Compute type verification ===")
    # INT8 models are auto-converted to float32 on Metal (mayiuse_int8 returns false)
    actual_ct = metal_int8.compute_type
    check("INT8 model loads on Metal", True)
    check("Auto-fallback compute type is float32",
          actual_ct == "float32",
          f"got {actual_ct}")

    # --- Greedy decoding ---
    print("\n=== Greedy: INT8-fallback Metal vs float32 CPU ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = cpu_f32.translate_batch([tokens], beam_size=1)
        metal_r = metal_int8.translate_batch([tokens], beam_size=1)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Greedy sentence {i}", match)
        if not match:
            print(f"      CPU  : {decode(tokenizer, cpu_out)}")
            print(f"      Metal: {decode(tokenizer, metal_out)}")

    # --- Beam search ---
    print("\n=== Beam=4: INT8-fallback Metal vs float32 CPU ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = cpu_f32.translate_batch([tokens], beam_size=4)
        metal_r = metal_int8.translate_batch([tokens], beam_size=4)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Beam4 sentence {i}", match)
        if not match:
            print(f"      CPU  : {decode(tokenizer, cpu_out)}")
            print(f"      Metal: {decode(tokenizer, metal_out)}")

    # --- Batch decoding ---
    print("\n=== Batch decoding consistency ===")
    all_tokens = [tokenize(tokenizer, s) for s in test_sentences]
    metal_batch = metal_int8.translate_batch(all_tokens, beam_size=1)
    metal_single = [
        metal_int8.translate_batch([t], beam_size=1)[0].hypotheses[0]
        for t in all_tokens
    ]
    batch_ok = all(
        metal_batch[i].hypotheses[0] == metal_single[i]
        for i in range(len(all_tokens))
    )
    check("Batch == single consistency", batch_ok)

    # --- Summary ---
    print(f"\n{'='*50}")
    total = passed + failed
    print(f"{passed}/{total} passed")
    if failed:
        print("FAILURES DETECTED")
        return 1
    else:
        print("ALL PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
