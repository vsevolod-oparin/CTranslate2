#!/usr/bin/env python3
"""Beam search regression tests: step-by-step sweep and batch consistency.

Consolidates: beam_step_test.py, batch_compare_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import model_path, load_marian_tokenizer, tokenize, decode


def main():
    mpath = model_path("opus-mt-en-de")
    if not os.path.isdir(mpath):
        print(f"SKIP: model not found at {mpath}")
        return 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()
    cpu = ctranslate2.Translator(mpath, device="cpu")
    metal = ctranslate2.Translator(mpath, device="metal")

    sentence = "The cat sat on the mat."
    tokens = tokenize(tokenizer, sentence)

    passed = 0
    failed = 0

    def check(label, ok):
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}")
        if ok:
            passed += 1
        else:
            failed += 1

    # --- Section 1: Step-by-step beam regression (max_len sweep) ---
    print(f"Input: {tokens}")
    print()

    for beam_size in [2, 4, 8]:
        print(f"=== beam_size={beam_size}, max_len sweep 1..11 ===")
        for max_len in range(1, 12):
            cpu_r = cpu.translate_batch(
                [tokens], beam_size=beam_size,
                max_decoding_length=max_len, num_hypotheses=beam_size,
            )
            metal_r = metal.translate_batch(
                [tokens], beam_size=beam_size,
                max_decoding_length=max_len, num_hypotheses=beam_size,
            )
            cpu_hyps = cpu_r[0].hypotheses
            metal_hyps = metal_r[0].hypotheses
            ok = cpu_hyps == metal_hyps
            check(f"beam={beam_size} max_len={max_len:2d}", ok)
            if not ok:
                for i, (ch, mh) in enumerate(zip(cpu_hyps, metal_hyps)):
                    hm = "OK" if ch == mh else "DIFF"
                    print(f"      hyp[{i}]: {hm}  cpu={ch}  metal={mh}")
        print()

    # --- Section 2: Batch consistency (from batch_compare_test.py) ---
    print("=== Batch consistency ===")

    # beam_size=2, all hypotheses
    cpu_beam2 = cpu.translate_batch([tokens], beam_size=2, num_hypotheses=2)
    metal_beam2 = metal.translate_batch([tokens], beam_size=2, num_hypotheses=2)
    for i, (ch, mh) in enumerate(zip(cpu_beam2[0].hypotheses, metal_beam2[0].hypotheses)):
        check(f"beam2 hyp[{i}] CPU==Metal", ch == mh)

    # beam1 vs beam2 hypothesis[0]
    metal_beam1 = metal.translate_batch([tokens], beam_size=1)
    h0_b2 = metal_beam2[0].hypotheses[0]
    h0_b1 = metal_beam1[0].hypotheses[0]
    check("Metal beam1 == beam2 hyp[0]", h0_b1 == h0_b2)

    # CPU beam1 vs Metal beam1
    cpu_beam1 = cpu.translate_batch([tokens], beam_size=1)
    check("CPU beam1 == Metal beam1", cpu_beam1[0].hypotheses[0] == h0_b1)

    # Batch=2 identical sentences
    metal_2x = metal.translate_batch([tokens, tokens], beam_size=1)
    check("Batch-of-2 result[0] == result[1]", metal_2x[0].hypotheses[0] == metal_2x[1].hypotheses[0])
    check("Batch-of-2 result[0] == single beam1", metal_2x[0].hypotheses[0] == h0_b1)

    print()
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
