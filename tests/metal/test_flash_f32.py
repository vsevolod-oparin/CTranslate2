#!/usr/bin/env python3
"""Test FlashMHA f32 correctness on MPS.

Compares standard vs flash path for float32 compute type.
Run with CT2_ATTN_NORM_DEBUG=1 to get per-layer norm dumps.

Usage:
  # Build first, then:
  CT2_ATTN_NORM_DEBUG=1 conda run -n ct2 python tests/metal/test_flash_f32.py 2>flash_debug.log

  # Then diff layer norms:
  grep '\\[FLASH' flash_debug.log | head -44 > flash_norms.txt
  grep '\\[STD'   flash_debug.log | head -44 > std_norms.txt
  diff flash_norms.txt std_norms.txt
"""
import ctranslate2
import os
import sys

MODEL_PATH = "/Users/smileijp/projects/branch/data/tinyllama-ct2"
PROMPT_TOKENS = [["<s>", "Hello", ",", " world"]]
MAX_LENGTH = 20

def run_path(flash_attention, compute_type="float32"):
    label = "FLASH" if flash_attention else "STANDARD"
    print(f"\n--- {label} (compute_type={compute_type}) ---")
    gen = ctranslate2.Generator(
        MODEL_PATH,
        device="mps",
        compute_type=compute_type,
        flash_attention=flash_attention,
    )
    results = gen.generate_batch(
        PROMPT_TOKENS,
        max_length=MAX_LENGTH,
        beam_size=1,
        sampling_topk=1,
    )
    tokens = results[0].sequences[0]
    print(f"  Tokens ({len(tokens)}): {' '.join(tokens[:30])}")
    return tokens

def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    for ct in ["float32"]:
        print(f"\n{'='*60}")
        print(f"  Compute type: {ct}")
        print(f"{'='*60}")

        std_tokens = run_path(flash_attention=False, compute_type=ct)
        flash_tokens = run_path(flash_attention=True, compute_type=ct)

        match = std_tokens == flash_tokens
        print(f"\n  Match: {match}")
        if not match:
            for i in range(max(len(std_tokens), len(flash_tokens))):
                st = std_tokens[i] if i < len(std_tokens) else "<end>"
                ft = flash_tokens[i] if i < len(flash_tokens) else "<end>"
                if st != ft:
                    print(f"  First divergence at generated token {i}:")
                    print(f"    standard = '{st}'")
                    print(f"    flash    = '{ft}'")
                    break

    # Also test with prompt length 1
    print(f"\n{'='*60}")
    print(f"  Prompt length 1 (float32)")
    print(f"{'='*60}")
    for flash in [False, True]:
        label = "FLASH" if flash else "STD"
        gen = ctranslate2.Generator(
            MODEL_PATH, device="mps", compute_type="float32",
            flash_attention=flash)
        r = gen.generate_batch(
            [["<s>"]],
            max_length=10, beam_size=1, sampling_topk=1)
        toks = r[0].sequences[0]
        print(f"  [{label}] {' '.join(toks)}")

if __name__ == "__main__":
    main()
