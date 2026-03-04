#!/usr/bin/env python3
"""CPU vs Metal translation comparison across multiple sentences and beam sizes.

Consolidates: full_e2e_test.py, beam_toks_test.py, test1.py
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
    cpu_translator = ctranslate2.Translator(mpath, device="cpu")
    metal_translator = ctranslate2.Translator(mpath, device="metal")

    test_sentences = [
        "The cat sat on the mat.",
        "Hello world, this is a test.",
        "Machine translation is an interesting research area.",
        "a b c",
        "a b c d e f g",
    ]

    beam_sizes = [1, 2, 4]
    max_lens = [1, 2, 3, 5, 10, None]  # None = no limit (full decode)

    passed = 0
    failed = 0

    print(f"{'beam':<5} {'max_len':<8} {'input':<42} {'match':<5} {'CPU output':<35} {'Metal output'}")
    print("-" * 140)

    for sentence in test_sentences:
        tokens = tokenize(tokenizer, sentence)
        for beam_size in beam_sizes:
            for max_len in max_lens:
                kwargs = dict(beam_size=beam_size)
                if max_len is not None:
                    kwargs["max_decoding_length"] = max_len

                cpu_r = cpu_translator.translate_batch([tokens], **kwargs)
                metal_r = metal_translator.translate_batch([tokens], **kwargs)

                cpu_out = decode(tokenizer, cpu_r[0].hypotheses[0])
                metal_out = decode(tokenizer, metal_r[0].hypotheses[0])

                ok = cpu_out == metal_out
                if ok:
                    passed += 1
                else:
                    failed += 1

                ml_str = str(max_len) if max_len else "full"
                tag = "PASS" if ok else "FAIL"
                print(f"{beam_size:<5} {ml_str:<8} {sentence[:40]:<42} {tag:<5} {cpu_out[:33]:<35} {metal_out[:33]}")

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
