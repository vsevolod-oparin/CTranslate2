#!/usr/bin/env python3
"""INT8 native end-to-end translation test on Metal.

Tests that INT8-quantized models run natively on Metal (not fallback to float32).
The Metal backend now advertises INT8 support (mayiuse_int8 returns true), so
the runtime uses the native INT8 GEMM pipeline (quantize -> int8 GEMM -> dequantize).

Verifies:
  1. INT8 model loads on Metal with compute_type "int8_float32" (native, not fallback)
  2. Metal-INT8 output matches CPU-INT8 output exactly (same quantization path)
  3. Greedy, beam search, and batch modes all produce correct translations
  4. Informational comparison vs CPU-float32 (may differ due to quantization loss)

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
    cpu_int8 = ctranslate2.Translator(int8_path, device="cpu")
    metal_int8 = ctranslate2.Translator(int8_path, device="mps")

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

    # --- Verify native INT8 compute type ---
    print("\n=== Compute type verification ===")
    actual_ct = metal_int8.compute_type
    check("Metal INT8 compute type is int8_float32 (native)",
          actual_ct == "int8_float32",
          f"got {actual_ct}")

    cpu_int8_ct = cpu_int8.compute_type
    cpu_has_int8 = cpu_int8_ct == "int8_float32"
    if cpu_has_int8:
        print(f"  [INFO] CPU supports native INT8 ({cpu_int8_ct})")
    else:
        print(f"  [INFO] CPU lacks INT8 backend, fell back to {cpu_int8_ct} — using CPU-f32 as reference")

    # Use CPU-INT8 as reference if available, otherwise CPU-float32
    # (When CPU lacks INT8 backend, cpu_int8 auto-falls back to float32 anyway)
    ref = cpu_int8 if cpu_has_int8 else cpu_f32
    ref_label = "CPU-INT8" if cpu_has_int8 else "CPU-f32"

    # --- Greedy: Metal-INT8 vs reference ---
    print(f"\n=== Greedy: Metal-INT8 vs {ref_label} (exact match expected) ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = ref.translate_batch([tokens], beam_size=1)
        metal_r = metal_int8.translate_batch([tokens], beam_size=1)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Greedy sentence {i}", match)
        if not match:
            print(f"      {ref_label:12s}: {decode(tokenizer, cpu_out)}")
            print(f"      Metal-INT8: {decode(tokenizer, metal_out)}")

    # --- Beam search: Metal-INT8 vs reference ---
    print(f"\n=== Beam=4: Metal-INT8 vs {ref_label} (exact match expected) ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = ref.translate_batch([tokens], beam_size=4)
        metal_r = metal_int8.translate_batch([tokens], beam_size=4)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        check(f"Beam4 sentence {i}", match)
        if not match:
            print(f"      {ref_label:12s}: {decode(tokenizer, cpu_out)}")
            print(f"      Metal-INT8: {decode(tokenizer, metal_out)}")

    # --- Batch decoding consistency ---
    print("\n=== Batch decoding consistency (Metal-INT8) ===")
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

    # --- Informational: Metal-INT8 vs CPU-float32 ---
    print("\n=== Informational: Metal-INT8 vs CPU-float32 (may differ) ===")
    for i, sentence in enumerate(test_sentences):
        tokens = tokenize(tokenizer, sentence)
        cpu_r = cpu_f32.translate_batch([tokens], beam_size=1)
        metal_r = metal_int8.translate_batch([tokens], beam_size=1)
        cpu_out = cpu_r[0].hypotheses[0]
        metal_out = metal_r[0].hypotheses[0]
        match = cpu_out == metal_out
        label = "match" if match else "differ (expected due to quantization)"
        print(f"  [INFO] Sentence {i}: {label}")
        if not match:
            print(f"      CPU-f32   : {decode(tokenizer, cpu_out)}")
            print(f"      Metal-INT8: {decode(tokenizer, metal_out)}")

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
