#!/usr/bin/env python3
"""Float16 model end-to-end translation test on Metal.

Tests that float16-quantized models produce correct output on Metal.
The Metal backend does not currently advertise float16 support via
mayiuse_float16(), so the runtime auto-converts float16 weights to float32.

Verifies:
  1. Float16 model loads on Metal without error
  2. Output matches CPU float32 reference (greedy, beam, batch)

Requires:
  - CT2_TEST_DATA with opus-mt-en-de/ (float32) and opus-mt-en-de-f16/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize, decode


def main():
    f32_path = model_path("opus-mt-en-de")
    f16_path = model_path("opus-mt-en-de-f16")

    if not os.path.isdir(f32_path):
        print(f"SKIP: float32 model not found at {f32_path}")
        return 0
    if not os.path.isdir(f16_path):
        print(f"SKIP: float16 model not found at {f16_path}")
        return 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()

    print("Loading models...")
    cpu_f32 = ctranslate2.Translator(f32_path, device="cpu")
    metal_f16 = ctranslate2.Translator(f16_path, device="mps")

    actual_ct = metal_f16.compute_type
    print(f"  Metal float16 model compute_type: {actual_ct}")

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

    # --- Greedy ---
    print("\n=== Greedy: Metal-f16 vs CPU-f32 ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = cpu_f32.translate_batch([tokens], beam_size=1)
        metal_r = metal_f16.translate_batch([tokens], beam_size=1)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Greedy sentence {i}", match)
        if not match:
            print(f"      CPU-f32  : {decode(tokenizer, cpu_out)}")
            print(f"      Metal-f16: {decode(tokenizer, metal_out)}")

    # --- Beam search ---
    print("\n=== Beam=4: Metal-f16 vs CPU-f32 ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = cpu_f32.translate_batch([tokens], beam_size=4)
        metal_r = metal_f16.translate_batch([tokens], beam_size=4)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Beam4 sentence {i}", match)
        if not match:
            print(f"      CPU-f32  : {decode(tokenizer, cpu_out)}")
            print(f"      Metal-f16: {decode(tokenizer, metal_out)}")

    # --- Batch consistency ---
    print("\n=== Batch decoding consistency (Metal-f16) ===")
    all_tokens = [tokenize(tokenizer, s) for s in test_sentences]
    metal_batch = metal_f16.translate_batch(all_tokens, beam_size=1)
    metal_single = [
        metal_f16.translate_batch([t], beam_size=1)[0].hypotheses[0]
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
