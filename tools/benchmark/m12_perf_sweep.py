#!/usr/bin/env python3
"""M12 Translation Performance Sweep — fast benchmark for tracking optimizations.

Translates a small subset of WMT14 En→De with OPUS-MT to measure per-commit
performance changes. Designed to run in <30s total.

Usage:
  python m12_perf_sweep.py [--compute_type float32|float16] [--label "commit message"]

Output: single line suitable for pasting into the sweep table.
"""
import argparse
import ctypes
import gc
import os
import sys
import time

import ctranslate2
import sacrebleu
from transformers import MarianTokenizer


NUM_SENTENCES = 50       # Small subset for speed
NUM_RUNS = 3             # Best-of-3
BEAM_SIZE = 4
MAX_BATCH_SIZE = 32


def load_ct2_lib():
    """Load libctranslate2 for profiling counters."""
    prefix = sys.prefix
    for name in ["libctranslate2.dylib", "libctranslate2.4.dylib",
                 "libctranslate2.4.7.1.dylib"]:
        path = os.path.join(prefix, "lib", name)
        if os.path.isfile(path):
            return ctypes.CDLL(path)
    return None


def get_profiling_fns(lib):
    """Set up commit_count and gpu_time functions."""
    fns = {}
    if lib is None:
        return fns
    try:
        fn = lib._ZN11ctranslate25metal12commit_countEv
        fn.restype = ctypes.c_uint64; fn.argtypes = []
        fns["commit_count"] = fn

        fn = lib._ZN11ctranslate25metal18reset_commit_countEv
        fn.restype = None; fn.argtypes = []
        fns["reset_commit_count"] = fn

        fn = lib._ZN11ctranslate25metal16gpu_time_elapsedEv
        fn.restype = ctypes.c_double; fn.argtypes = []
        fns["gpu_time_elapsed"] = fn

        fn = lib._ZN11ctranslate25metal14reset_gpu_timeEv
        fn.restype = None; fn.argtypes = []
        fns["reset_gpu_time"] = fn
    except AttributeError:
        pass
    return fns


def run_benchmark(translator, source_tokens, fns):
    """Run one timed translation, return (wall_ms, tokens, commits, gpu_ms)."""
    if "reset_commit_count" in fns:
        fns["reset_commit_count"]()
    if "reset_gpu_time" in fns:
        fns["reset_gpu_time"]()

    gc.collect()

    t0 = time.monotonic()
    results = translator.translate_batch(
        source_tokens, beam_size=BEAM_SIZE, max_batch_size=MAX_BATCH_SIZE,
    )
    wall = (time.monotonic() - t0) * 1000

    tokens = sum(len(r.hypotheses[0]) for r in results)
    commits = fns["commit_count"]() if "commit_count" in fns else 0
    gpu_ms = (fns["gpu_time_elapsed"]() * 1000) if "gpu_time_elapsed" in fns else 0

    return wall, tokens, commits, gpu_ms


def main():
    parser = argparse.ArgumentParser(
        description="M12 fast translation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compute_type", type=str, default="float16",
                        choices=["float32", "float16"])
    parser.add_argument("--label", type=str, default="",
                        help="Label for this run (e.g. commit message)")
    parser.add_argument("--cpu_baseline", action="store_true",
                        help="Also run CPU baseline")
    parser.add_argument("--num_sentences", type=int, default=NUM_SENTENCES)
    args = parser.parse_args()

    data_dir = os.environ.get(
        "CT2_TEST_DATA",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")),
    )
    model_map = {
        "float32": os.path.join(data_dir, "opus-mt-en-de"),
        "float16": os.path.join(data_dir, "opus-mt-en-de-f16"),
    }
    model_path = model_map[args.compute_type]
    if not os.path.isdir(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1

    # Load test data
    source_file = sacrebleu.get_source_file("wmt14", langpair="en-de")
    with open(source_file) as f:
        source_sentences = [line.strip() for line in f][:args.num_sentences]

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    source_tokens = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
        for s in source_sentences
    ]

    lib = load_ct2_lib()
    fns = get_profiling_fns(lib)

    # Get git commit hash
    import subprocess
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    # ---- CPU baseline ----
    cpu_ms = None
    if args.cpu_baseline:
        translator_cpu = ctranslate2.Translator(
            model_map["float32"], device="cpu", compute_type="float32",
            intra_threads=4,
        )
        translator_cpu.translate_batch([[""]], beam_size=1)  # warmup
        cpu_runs = []
        for _ in range(NUM_RUNS):
            w, t, _, _ = run_benchmark(translator_cpu, source_tokens, {})
            cpu_runs.append(w)
        cpu_ms = min(cpu_runs)
        cpu_tokens = t
        del translator_cpu; gc.collect()
        print(f"CPU baseline: {cpu_ms:.0f} ms, {cpu_tokens} tokens, "
              f"{cpu_tokens/(cpu_ms/1000):.0f} tok/s")

    # ---- MPS benchmark ----
    translator = ctranslate2.Translator(
        model_path, device="mps", compute_type=args.compute_type,
        intra_threads=1,
    )
    translator.translate_batch([[""]], beam_size=1)  # warmup + PSO compile

    runs = []
    for i in range(NUM_RUNS):
        gc.collect()
        ctranslate2.clear_device_cache("mps")
        w, tokens, commits, gpu_ms = run_benchmark(translator, source_tokens, fns)
        runs.append((w, tokens, commits, gpu_ms))

    # Pick best wall time
    best = min(runs, key=lambda r: r[0])
    wall_ms, tokens, commits, gpu_ms = best
    tok_s = tokens / (wall_ms / 1000)
    gpu_pct = (gpu_ms / wall_ms * 100) if wall_ms > 0 else 0
    all_walls = [r[0] for r in runs]

    speedup_str = ""
    if cpu_ms:
        speedup = cpu_ms / wall_ms
        speedup_str = f"{speedup:.2f}x"

    label = args.label or commit

    # Print detailed output
    print(f"\nMPS {args.compute_type}: {wall_ms:.0f} ms (best of {NUM_RUNS})")
    print(f"  Runs: {', '.join(f'{w:.0f}' for w in all_walls)} ms")
    print(f"  Tokens: {tokens}, tok/s: {tok_s:.0f}")
    print(f"  Commits: {commits}, GPU: {gpu_ms:.0f} ms ({gpu_pct:.0f}%)")
    if speedup_str:
        print(f"  vs CPU: {speedup_str}")

    # Print table row
    runs_str = ", ".join(f"{w:.0f}" for w in all_walls)
    print(f"\n--- Table row ---")
    print(f"| {commit} | {wall_ms:.0f} | {runs_str} | {tokens} | "
          f"{commits} | {gpu_pct:.0f}% | {tok_s:.0f} | {label} | {speedup_str} |")

    del translator; gc.collect()
    ctranslate2.clear_device_cache("mps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
