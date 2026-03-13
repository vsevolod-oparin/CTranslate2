#!/usr/bin/env python3
"""M14.5 regression test: logits processor GPU page fault fix.

Tests repetition_penalty and no_repeat_ngram_size with diverse-length
batches that previously triggered MPS driver coherency page faults.
The bug required 8 diverse-length sentences with beam_size=4 to trigger.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import model_path, load_marian_tokenizer, tokenize


def main():
    mpath = model_path("opus-mt-en-de")
    if not os.path.isdir(mpath):
        print(f"SKIP: model not found at {mpath}")
        return 0

    import ctranslate2

    tokenizer = load_marian_tokenizer()

    # The exact 8 diverse-length sentences that triggered the page fault.
    sentences = [
        "The cat sat on the mat.",
        "I love programming in Python and building machine learning models.",
        "Hello world!",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Good morning.",
        "Artificial intelligence is transforming the way we interact with technology in our daily lives.",
        "She sells seashells by the seashore.",
        "To be or not to be, that is the question, whether it is nobler in the mind.",
    ]
    tokens_batch = [tokenize(tokenizer, s) for s in sentences]

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

    # Test all compute types that use different GEMM paths
    for ctype in ["float32", "float16", "int8", "int8_float16"]:
        try:
            metal = ctranslate2.Translator(mpath, device="mps", compute_type=ctype)
        except Exception as e:
            print(f"  [SKIP] {ctype}: {e}")
            continue

        print(f"=== compute_type={ctype} ===")

        # Test 1: repetition_penalty with diverse batch (the crash case)
        try:
            results = metal.translate_batch(
                tokens_batch,
                beam_size=4,
                repetition_penalty=1.2,
            )
            check(f"{ctype} repetition_penalty=1.2 beam=4 batch=8", True)
        except Exception as e:
            check(f"{ctype} repetition_penalty=1.2 beam=4 batch=8: {e}", False)

        # Test 2: no_repeat_ngram_size (second reported bug)
        try:
            results = metal.translate_batch(
                tokens_batch,
                beam_size=4,
                no_repeat_ngram_size=2,
            )
            check(f"{ctype} no_repeat_ngram=2 beam=4 batch=8", True)
        except Exception as e:
            check(f"{ctype} no_repeat_ngram=2 beam=4 batch=8: {e}", False)

        # Test 3: both combined
        try:
            results = metal.translate_batch(
                tokens_batch,
                beam_size=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )
            check(f"{ctype} rep_pen+no_repeat beam=4 batch=8", True)
        except Exception as e:
            check(f"{ctype} rep_pen+no_repeat beam=4 batch=8: {e}", False)

        # Test 4: greedy with repetition_penalty
        try:
            results = metal.translate_batch(
                tokens_batch,
                beam_size=1,
                repetition_penalty=1.2,
            )
            check(f"{ctype} repetition_penalty=1.2 greedy batch=8", True)
        except Exception as e:
            check(f"{ctype} repetition_penalty=1.2 greedy batch=8: {e}", False)

        # Test 5: stress - multiple iterations
        try:
            for i in range(5):
                metal.translate_batch(
                    tokens_batch,
                    beam_size=4,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                )
            check(f"{ctype} 5x stress iterations", True)
        except Exception as e:
            check(f"{ctype} 5x stress iterations: {e}", False)

        del metal
        print()

    total = passed + failed
    print(f"{passed}/{total} passed")
    if failed:
        print("FAILURES DETECTED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
