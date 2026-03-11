#!/usr/bin/env python3
"""MPS (Apple Metal) benchmark for CTranslate2.

Translates En->De newstest2014 (WMT14) with the OPUS-MT model on CPU and MPS,
matching the methodology of the existing README benchmarks:
  - Reports target tokens per second (higher is better)
  - Excludes model loading time (warmup with empty input)
  - Aggregates over multiple runs (best of N)
  - Reports BLEU score via sacrebleu

Each configuration runs in a SEPARATE SUBPROCESS to ensure Metal GPU resources
are fully released between runs (avoids VSIZE/memory accumulation).

Usage:
  python benchmark_mps.py [--num_samples N] [--num_cpus N] [--beam_size N]

Requires:
  - CT2_TEST_DATA env var pointing to directory with opus-mt-en-de* model dirs
  - sacrebleu, transformers, ctranslate2
"""
import argparse
import json
import os
import subprocess
import sys
import textwrap

import ctranslate2
import sacrebleu


# Worker script template — runs in a subprocess to isolate Metal memory
_WORKER_SCRIPT = textwrap.dedent(r'''
import gc
import json
import os
import resource
import sys
import time

import ctranslate2
import sacrebleu
from transformers import MarianTokenizer

def get_max_rss_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024

config = json.loads(sys.argv[1])

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

source_file = sacrebleu.get_source_file(config["test_set"], langpair=config["langpair"])
reference_file = sacrebleu.get_reference_files(config["test_set"], langpair=config["langpair"])[0]
with open(source_file) as f:
    source_sentences = [line.strip() for line in f]

source_tokens = [
    tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
    for s in source_sentences
]

device = config["device"]
translator = ctranslate2.Translator(
    config["model_path"],
    device=device,
    compute_type=config["compute_type"],
    intra_threads=config["intra_threads"],
)

# Warmup
translator.translate_batch([[""]], beam_size=1)

best_time = None
num_target_tokens = 0
bleu_score = 0.0

for sample in range(config["num_samples"]):
    gc.collect()
    if device == "mps":
        ctranslate2.clear_device_cache("mps")

    t0 = time.monotonic()
    results = translator.translate_batch(
        source_tokens,
        beam_size=config["beam_size"],
        max_batch_size=32,
    )
    elapsed = time.monotonic() - t0

    if best_time is None or elapsed < best_time:
        best_time = elapsed
        hypotheses = [r.hypotheses[0] for r in results]
        num_target_tokens = sum(len(h) for h in hypotheses)
        decoded = [
            tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
            for tokens in hypotheses
        ]
        bleu = sacrebleu.corpus_bleu(
            decoded,
            [open(reference_file).readlines()],
            force=True,
        )
        bleu_score = bleu.score

max_rss = get_max_rss_mb()

del translator
gc.collect()
if device == "mps":
    ctranslate2.clear_device_cache("mps")

result = {
    "time": best_time,
    "tokens_per_sec": num_target_tokens / best_time,
    "num_tokens": num_target_tokens,
    "max_rss_mb": max_rss,
    "bleu": bleu_score,
}
print("RESULT:" + json.dumps(result))
''')


def run_benchmark_subprocess(config):
    """Run a single benchmark config in a subprocess, return result dict or None."""
    config_json = json.dumps(config)
    result = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, config_json],
        capture_output=True, text=True, timeout=1200,
    )

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        stderr = result.stderr.strip()
        if stderr:
            # Print last 5 lines of stderr
            for line in stderr.split("\n")[-5:]:
                print(f"    {line}")
        return None

    # Parse result from stdout
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            return json.loads(line[7:])

    print(f"  FAILED: no result in output")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="MPS benchmark for CTranslate2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of runs per configuration (report best)")
    parser.add_argument("--num_cpus", type=int, default=4,
                        help="Number of CPU threads for CPU baseline")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam size for translation")
    parser.add_argument("--test_set", type=str, default="wmt14",
                        help="sacrebleu test set name")
    parser.add_argument("--langpair", type=str, default="en-de",
                        help="Language pair")
    args = parser.parse_args()

    # Locate model directory
    data_dir = os.environ.get(
        "CT2_TEST_DATA",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")),
    )

    models = {
        "float32": os.path.join(data_dir, "opus-mt-en-de"),
        "float16": os.path.join(data_dir, "opus-mt-en-de-f16"),
        # int8 excluded: no native MPS INT8 support, uses CPU fallback (~4x slower)
        # bfloat16 excluded: MPSGraph overhead makes it too slow for large test sets
    }

    # Validate models
    available = {k: v for k, v in models.items() if os.path.isdir(v)}
    if not available:
        print(f"ERROR: No models found in {data_dir}")
        print(f"Expected: opus-mt-en-de, opus-mt-en-de-f16, opus-mt-en-de-int8, opus-mt-en-de-bf16")
        return 1

    print(f"Models found: {', '.join(available.keys())} in {data_dir}")

    # Load test set info
    print(f"Test set: {args.test_set} ({args.langpair})")
    source_file = sacrebleu.get_source_file(args.test_set, langpair=args.langpair)
    with open(source_file) as f:
        num_sentences = sum(1 for _ in f)
    print(f"Source sentences: {num_sentences}")
    print(f"Beam size: {args.beam_size}")
    print(f"Samples per config: {args.num_samples}")
    print()

    # Determine which devices to benchmark
    devices_to_test = ["cpu"]
    if ctranslate2.get_mps_device_count() > 0:
        devices_to_test.append("mps")
    else:
        print("WARNING: No MPS device available, CPU-only benchmark")

    # Define configurations
    # CPU: float32 only (int8 requires special model, float16/bfloat16 not supported)
    # MPS: all available types
    configs = []
    for compute_type, model_dir in available.items():
        for device in devices_to_test:
            if device == "cpu" and compute_type != "float32":
                continue
            configs.append((device, compute_type, model_dir))

    # Run benchmarks (each in a subprocess for memory isolation)
    results = []
    for device, compute_type, model_dir in configs:
        label = f"CTranslate2 ({device}, {compute_type})"
        print(f"Benchmarking: {label}...", flush=True)

        config = {
            "model_path": model_dir,
            "device": device,
            "compute_type": compute_type,
            "beam_size": args.beam_size,
            "num_samples": args.num_samples,
            "intra_threads": args.num_cpus if device == "cpu" else 1,
            "test_set": args.test_set,
            "langpair": args.langpair,
        }

        r = run_benchmark_subprocess(config)
        results.append((label, device, compute_type, r))
        if r:
            print(f"  {r['tokens_per_sec']:.1f} tok/s, "
                  f"{r['time']:.2f}s, "
                  f"RSS={r['max_rss_mb']:.0f}MB, "
                  f"BLEU={r['bleu']:.2f}", flush=True)

    # Print summary table
    print()
    print("=" * 80)
    print(f"OPUS-MT En->De benchmark on {args.test_set} "
          f"(beam={args.beam_size}, best of {args.num_samples})")
    print("=" * 80)
    print()

    # CPU table
    cpu_results = [(l, r) for l, d, ct, r in results if d == "cpu" and r]
    if cpu_results:
        print("#### CPU")
        print()
        print(f"| | Tokens per second | Max. memory | BLEU |")
        print(f"| --- | --- | --- | --- |")
        print(f"| **OPUS-MT model** | | | |")
        for label, r in cpu_results:
            ct = label.split(", ")[1].rstrip(")")
            name = f"CTranslate2 - {ct}" if ct != "float32" else "CTranslate2"
            print(f"| {name} | {r['tokens_per_sec']:.1f} | "
                  f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")
        print()
        print(f"Executed with {args.num_cpus} threads on Apple M4.")
        print()

    # MPS table
    mps_results = [(l, r) for l, d, ct, r in results if d == "mps" and r]
    if mps_results:
        print("#### MPS (Apple Metal)")
        print()
        print(f"| | Tokens per second | Max. memory | BLEU |")
        print(f"| --- | --- | --- | --- |")
        print(f"| **OPUS-MT model** | | | |")
        for label, r in mps_results:
            ct = label.split(", ")[1].rstrip(")")
            name = f"CTranslate2 - {ct}" if ct != "float32" else "CTranslate2"
            print(f"| {name} | {r['tokens_per_sec']:.1f} | "
                  f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")
        print()
        print(f"Executed on Apple M4 GPU (Metal Performance Shaders).")
        print()

    # Speedup comparison
    if cpu_results and mps_results:
        print("#### CPU vs MPS Speedup")
        print()
        cpu_f32 = next((r for l, d, ct, r in results if d == "cpu" and ct == "float32" and r), None)
        if cpu_f32:
            print(f"| Configuration | CPU tok/s | MPS tok/s | Speedup |")
            print(f"| --- | --- | --- | --- |")
            for label, r in mps_results:
                ct = label.split(", ")[1].rstrip(")")
                cpu_match = next((cr for cl, cd, cct, cr in results if cd == "cpu" and cct == ct and cr), None)
                cpu_ref = cpu_match if cpu_match else cpu_f32
                speedup = r["tokens_per_sec"] / cpu_ref["tokens_per_sec"]
                print(f"| {ct} | {cpu_ref['tokens_per_sec']:.1f} | "
                      f"{r['tokens_per_sec']:.1f} | {speedup:.2f}x |")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
