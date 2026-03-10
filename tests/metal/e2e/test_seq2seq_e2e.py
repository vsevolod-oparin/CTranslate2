#!/usr/bin/env python3
"""M10.2 — Seq2seq (Transformer) end-to-end: BLEU and speed comparison.

Uses WMT14 en-de (first 100 sentences) via sacrebleu.
Hard pass criteria (correctness):
  - BLEU score within 0.5 of CPU result (greedy and beam_size=4)
  - 100% exact token match between CPU and Metal
Informational benchmarks (speed):
  - Metal vs CPU latency for batch_size=1 (greedy and beam=4)
  - Speed optimization is deferred to M11 (command buffer batching)
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ctranslate2
import sacrebleu
from conftest import model_path, load_marian_tokenizer, tokenize, decode

NUM_SENTENCES = 100
BEAM_SIZES = [1, 4]  # greedy and beam=4
BLEU_TOLERANCE = 0.5
WARMUP_SENTENCES = 5  # warm up before timing


def load_wmt14_subset(n):
    """Load first n source sentences and references from WMT14 en-de."""
    src_file = sacrebleu.get_source_file("wmt14", "en-de")
    ref_files = sacrebleu.get_reference_files("wmt14", "en-de")
    with open(src_file) as f:
        sources = [line.strip() for line in f][:n]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f][:n]
    return sources, references


def translate_all_timed(translator, tokenizer, sentences, beam_size, warmup=0):
    """Translate with optional warmup. Returns (outputs, elapsed_seconds)."""
    for sentence in sentences[:warmup]:
        tokens = tokenize(tokenizer, sentence)
        translator.translate_batch([tokens], beam_size=beam_size)

    t0 = time.monotonic()
    outputs = []
    for sentence in sentences:
        tokens = tokenize(tokenizer, sentence)
        result = translator.translate_batch([tokens], beam_size=beam_size)
        text = decode(tokenizer, result[0].hypotheses[0])
        outputs.append(text)
    elapsed = time.monotonic() - t0
    return outputs, elapsed


def compute_bleu(hypotheses, references):
    """Compute corpus BLEU score."""
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def main():
    tokenizer = load_marian_tokenizer()
    mpath = model_path("opus-mt-en-de")

    print(f"Loading WMT14 en-de ({NUM_SENTENCES} sentences)...")
    sources, references = load_wmt14_subset(NUM_SENTENCES)
    print(f"  Loaded {len(sources)} source / {len(references)} reference sentences")

    cpu_translator = ctranslate2.Translator(mpath, device="cpu")
    metal_translator = ctranslate2.Translator(mpath, device="mps")

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

    def info(label, detail=""):
        suffix = f"  ({detail})" if detail else ""
        print(f"  [INFO] {label}{suffix}")

    # --- Combined correctness + timing (single pass per beam size) ---
    for beam_size in BEAM_SIZES:
        mode = "greedy" if beam_size == 1 else f"beam={beam_size}"
        print(f"\n=== BLEU + speed: {mode} ===")

        cpu_outputs, cpu_time = translate_all_timed(
            cpu_translator, tokenizer, sources, beam_size,
            warmup=WARMUP_SENTENCES,
        )
        metal_outputs, metal_time = translate_all_timed(
            metal_translator, tokenizer, sources, beam_size,
            warmup=WARMUP_SENTENCES,
        )

        # Correctness: BLEU
        cpu_bleu = compute_bleu(cpu_outputs, references)
        metal_bleu = compute_bleu(metal_outputs, references)
        bleu_diff = abs(cpu_bleu - metal_bleu)

        print(f"  CPU BLEU:   {cpu_bleu:.2f}")
        print(f"  Metal BLEU: {metal_bleu:.2f}")
        print(f"  Difference: {bleu_diff:.2f}")

        check(
            f"BLEU diff <= {BLEU_TOLERANCE} ({mode})",
            bleu_diff <= BLEU_TOLERANCE,
            f"diff={bleu_diff:.2f}",
        )

        # Correctness: exact match
        exact_matches = sum(1 for c, m in zip(cpu_outputs, metal_outputs) if c == m)
        print(f"  Exact match: {exact_matches}/{len(sources)}")
        check(
            f"Exact match rate ({mode})",
            exact_matches == len(sources),
            f"{exact_matches}/{len(sources)}",
        )

        # Speed (informational)
        speedup = cpu_time / metal_time if metal_time > 0 else float("inf")
        info(f"CPU:   {cpu_time:.2f}s  ({len(sources)/cpu_time:.1f} sent/s)")
        info(f"Metal: {metal_time:.2f}s  ({len(sources)/metal_time:.1f} sent/s)")
        info(f"Ratio: {speedup:.2f}x (target: >=1.5x after M11 optimization)")

    # --- Summary ---
    print(f"\n{'='*50}")
    total = passed + failed
    print(f"{passed}/{total} passed")
    if failed:
        print("FAILURES DETECTED")
        sys.exit(1)
    else:
        print("ALL PASS")


if __name__ == "__main__":
    main()
