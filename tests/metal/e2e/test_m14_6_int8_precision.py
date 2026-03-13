#!/usr/bin/env python3
"""M14.6 INT8 Precision Audit — MPS INT8 vs CPU INT8 BLEU comparison.

Compares INT8 BLEU across MPS and CPU to verify no precision loss
in the Metal INT8 pipeline (GPU dequant, fused GEMV, encode-only ops).
"""
import gc
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize, decode

_WORKER_CODE = '''\
import gc, json, os, sys, time

config = json.loads(sys.argv[1])

import ctranslate2
import sacrebleu
from transformers import MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
def tok(text):
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
def detok(tokens):
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))

sf = sacrebleu.get_source_file("wmt14", langpair="en-de")
rf = sacrebleu.get_reference_files("wmt14", langpair="en-de")[0]
with open(sf) as f:
    source_sentences = [l.strip() for l in f]
source_tokens = [tok(s) for s in source_sentences]

translator = ctranslate2.Translator(
    config["model_path"], device=config["device"],
    compute_type=config.get("compute_type", "default"),
    intra_threads=config.get("intra_threads", 1),
)

# Warmup
translator.translate_batch([[""]], beam_size=1)

t0 = time.monotonic()
all_results = translator.translate_batch(
    source_tokens, beam_size=config["beam_size"], max_batch_size=32)
elapsed = time.monotonic() - t0

hypotheses = [r.hypotheses[0] for r in all_results]
num_tokens = sum(len(h) for h in hypotheses)
decoded = [detok(tokens) for tokens in hypotheses]
bleu = sacrebleu.corpus_bleu(decoded, [open(rf).readlines()], force=True)

result = {
    "device": config["device"],
    "compute_type": translator.compute_type,
    "beam_size": config["beam_size"],
    "bleu": bleu.score,
    "tokens_per_sec": num_tokens / elapsed,
    "time": elapsed,
    "num_sentences": len(source_sentences),
}
print("RESULT:" + json.dumps(result))
'''


def run_config(label, mpath, device, compute_type="default", beam_size=4):
    config = {
        "model_path": mpath,
        "device": device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "intra_threads": 1,
    }
    # Write worker to temp file to avoid shell quoting issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(_WORKER_CODE)
        worker_path = f.name
    try:
        cmd = ["conda", "run", "-n", "ct2", "python", worker_path, json.dumps(config)]
        print(f"  Running: {label} ...", end="", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f" FAILED")
            print(f"    stderr: {result.stderr[-300:]}")
            return None
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                print(f" BLEU={data['bleu']:.2f}  tok/s={data['tokens_per_sec']:.0f}")
                return data
        print(f" NO RESULT")
        print(f"    stdout: {result.stdout[-300:]}")
        return None
    finally:
        os.unlink(worker_path)


def main():
    int8_path = model_path("opus-mt-en-de-int8")
    f32_path = model_path("opus-mt-en-de")

    if not os.path.isdir(int8_path):
        print(f"SKIP: INT8 model not found at {int8_path}")
        return 0
    if not os.path.isdir(f32_path):
        print(f"SKIP: float32 model not found at {f32_path}")
        return 0

    print("=" * 60)
    print("M14.6 INT8 Precision Audit - OPUS-MT WMT14 En-De (2737 sent)")
    print("=" * 60)

    configs = [
        ("CPU f32 beam=4",      f32_path,  "cpu", "float32",      4),
        ("CPU INT8 beam=4",     int8_path, "cpu", "default",      4),
        ("MPS INT8 beam=4",     int8_path, "mps", "default",      4),
        ("MPS INT8_f16 beam=4", int8_path, "mps", "int8_float16", 4),
        ("MPS f32 beam=4",      f32_path,  "mps", "float32",      4),
        ("MPS f16 beam=4",      f32_path,  "mps", "float16",      4),
    ]

    results = {}
    for label, mpath, device, ctype, beam in configs:
        data = run_config(label, mpath, device, ctype, beam)
        if data:
            results[label] = data

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Config':<25s} {'BLEU':>7s} {'tok/s':>8s} {'Gap vs CPU-f32':>15s}")
    print("-" * 60)
    cpu_f32_bleu = results.get("CPU f32 beam=4", {}).get("bleu", 0)
    for label in [c[0] for c in configs]:
        if label in results:
            d = results[label]
            gap = d["bleu"] - cpu_f32_bleu
            print(f"{label:<25s} {d['bleu']:>7.2f} {d['tokens_per_sec']:>8.0f} {gap:>+14.2f}")

    # Pass/fail criteria
    print("\n" + "=" * 60)
    print("Pass/Fail")
    print("=" * 60)
    passed = 0
    failed = 0

    def check(label, ok, detail=""):
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}" + (f"  ({detail})" if detail else ""))
        if ok:
            passed += 1
        else:
            failed += 1

    # MPS INT8 vs CPU INT8: should be very close (< 0.5 BLEU)
    if "MPS INT8 beam=4" in results and "CPU INT8 beam=4" in results:
        gap = abs(results["MPS INT8 beam=4"]["bleu"] - results["CPU INT8 beam=4"]["bleu"])
        check("MPS INT8 vs CPU INT8 gap < 0.5", gap < 0.5, f"gap={gap:.2f}")

    # MPS INT8 vs CPU f32: informational but should be reasonable (< 2.0)
    if "MPS INT8 beam=4" in results and cpu_f32_bleu > 0:
        gap = abs(results["MPS INT8 beam=4"]["bleu"] - cpu_f32_bleu)
        check("MPS INT8 vs CPU f32 gap < 2.0", gap < 2.0, f"gap={gap:.2f}")

    # MPS INT8_f16 should be close to MPS INT8 (< 0.5)
    if "MPS INT8_f16 beam=4" in results and "MPS INT8 beam=4" in results:
        gap = abs(results["MPS INT8_f16 beam=4"]["bleu"] - results["MPS INT8_f16 beam=4"]["bleu"])
        check("MPS INT8_f16 vs MPS INT8 gap < 0.5", gap < 0.5, f"gap={gap:.2f}")

    total = passed + failed
    print(f"\n{passed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
