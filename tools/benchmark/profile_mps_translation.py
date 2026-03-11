#!/usr/bin/env python3
"""Profile MPS translation to identify performance bottlenecks.

Measures per-step commit counts, GPU time, wall time, and identifies
where time is spent in the translation pipeline.

Usage:
  python profile_mps_translation.py [--compute_type float32|float16]
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


# ---------------------------------------------------------------------------
# Load C library for Metal profiling counters
# ---------------------------------------------------------------------------
def load_ct2_lib():
    """Load the CTranslate2 shared library for direct symbol access."""
    lib_dir = os.path.dirname(ctranslate2.__file__)
    # Try common library paths
    for name in ["libctranslate2.dylib", "libctranslate2.4.dylib",
                 "libctranslate2.4.7.1.dylib"]:
        path = os.path.join(lib_dir, "..", "..", "..", name)
        path = os.path.normpath(path)
        if os.path.isfile(path):
            return ctypes.CDLL(path)
    # Try conda env lib
    prefix = sys.prefix
    for name in ["libctranslate2.dylib", "libctranslate2.4.dylib",
                 "libctranslate2.4.7.1.dylib"]:
        path = os.path.join(prefix, "lib", name)
        if os.path.isfile(path):
            return ctypes.CDLL(path)
    raise FileNotFoundError("Cannot find libctranslate2 shared library")


def setup_profiling_fns(lib):
    """Set up ctypes function signatures for Metal profiling."""
    fns = {}
    try:
        # Commit count
        fn = lib._ZN11ctranslate25metal12commit_countEv
        fn.restype = ctypes.c_uint64
        fn.argtypes = []
        fns["commit_count"] = fn

        fn = lib._ZN11ctranslate25metal18reset_commit_countEv
        fn.restype = None
        fn.argtypes = []
        fns["reset_commit_count"] = fn

        # GPU time
        fn = lib._ZN11ctranslate25metal16gpu_time_elapsedEv
        fn.restype = ctypes.c_double
        fn.argtypes = []
        fns["gpu_time_elapsed"] = fn

        fn = lib._ZN11ctranslate25metal14reset_gpu_timeEv
        fn.restype = None
        fn.argtypes = []
        fns["reset_gpu_time"] = fn

        # Commit trace
        fn = lib._ZN11ctranslate25metal19enable_commit_traceEb
        fn.restype = None
        fn.argtypes = [ctypes.c_bool]
        fns["enable_commit_trace"] = fn

        fn = lib._ZN11ctranslate25metal17dump_commit_traceEv
        fn.restype = None
        fn.argtypes = []
        fns["dump_commit_trace"] = fn

        fn = lib._ZN11ctranslate25metal18reset_commit_traceEv
        fn.restype = None
        fn.argtypes = []
        fns["reset_commit_trace"] = fn

        # PSO stats
        fn = lib._ZN11ctranslate25metal14pso_hit_countEv
        fn.restype = ctypes.c_uint64
        fn.argtypes = []
        fns["pso_hit_count"] = fn

        fn = lib._ZN11ctranslate25metal15pso_miss_countEv
        fn.restype = ctypes.c_uint64
        fn.argtypes = []
        fns["pso_miss_count"] = fn

        fn = lib._ZN11ctranslate25metal15reset_pso_statsEv
        fn.restype = None
        fn.argtypes = []
        fns["reset_pso_stats"] = fn

        # Allocator stats
        fn = lib._ZN11ctranslate25metal10pool_bytesEv
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
        fns["pool_bytes"] = fn

        fn = lib._ZN11ctranslate25metal10live_bytesEv
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
        fns["live_bytes"] = fn

    except AttributeError as e:
        print(f"WARNING: Some profiling symbols not found: {e}")

    return fns


def main():
    parser = argparse.ArgumentParser(
        description="Profile MPS translation bottlenecks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compute_type", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Compute type to profile")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_sentences", type=int, default=100,
                        help="Max sentences to translate (0 = all)")
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--profile_cpu", action="store_true",
                        help="Also profile CPU for comparison")
    args = parser.parse_args()

    # Locate model
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

    # Load test set
    source_file = sacrebleu.get_source_file("wmt14", langpair="en-de")
    with open(source_file) as f:
        source_sentences = [line.strip() for line in f]
    if args.max_sentences > 0:
        source_sentences = source_sentences[:args.max_sentences]

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    source_tokens = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
        for s in source_sentences
    ]

    print(f"Model: {model_path}")
    print(f"Compute type: {args.compute_type}")
    print(f"Sentences: {len(source_sentences)}")
    print(f"Beam size: {args.beam_size}")
    print(f"Max batch size: {args.max_batch_size}")
    print()

    # Load profiling functions
    lib = load_ct2_lib()
    fns = setup_profiling_fns(lib)

    # =========================================================================
    # Phase 1: CPU baseline (optional)
    # =========================================================================
    if args.profile_cpu:
        print("=" * 70)
        print("PHASE 1: CPU BASELINE")
        print("=" * 70)
        translator_cpu = ctranslate2.Translator(
            model_map["float32"], device="cpu", compute_type="float32",
            intra_threads=4,
        )
        translator_cpu.translate_batch([[""]], beam_size=1)  # warmup

        t0 = time.monotonic()
        results_cpu = translator_cpu.translate_batch(
            source_tokens, beam_size=args.beam_size,
            max_batch_size=args.max_batch_size,
        )
        cpu_time = time.monotonic() - t0

        cpu_tokens = sum(len(r.hypotheses[0]) for r in results_cpu)
        print(f"  Wall time:  {cpu_time*1000:.0f} ms")
        print(f"  Tokens:     {cpu_tokens}")
        print(f"  Tok/s:      {cpu_tokens/cpu_time:.1f}")
        print()

        del translator_cpu
        gc.collect()

    # =========================================================================
    # Phase 2: MPS profiled run
    # =========================================================================
    print("=" * 70)
    print(f"PHASE 2: MPS {args.compute_type.upper()} PROFILED RUN")
    print("=" * 70)

    translator = ctranslate2.Translator(
        model_path, device="mps", compute_type=args.compute_type,
        intra_threads=1,
    )

    # Warmup (also compiles PSOs)
    translator.translate_batch([[""]], beam_size=1)

    # Reset all counters
    if "reset_commit_count" in fns:
        fns["reset_commit_count"]()
    if "reset_gpu_time" in fns:
        fns["reset_gpu_time"]()
    if "reset_pso_stats" in fns:
        fns["reset_pso_stats"]()
    if "enable_commit_trace" in fns:
        fns["enable_commit_trace"](True)
    if "reset_commit_trace" in fns:
        fns["reset_commit_trace"]()

    gc.collect()
    ctranslate2.clear_device_cache("mps")

    # Profile full translation
    t0 = time.monotonic()
    results = translator.translate_batch(
        source_tokens, beam_size=args.beam_size,
        max_batch_size=args.max_batch_size,
    )
    wall_time = time.monotonic() - t0

    # Read counters
    commits = fns["commit_count"]() if "commit_count" in fns else 0
    gpu_time = fns["gpu_time_elapsed"]() if "gpu_time_elapsed" in fns else 0.0
    pso_hits = fns["pso_hit_count"]() if "pso_hit_count" in fns else 0
    pso_misses = fns["pso_miss_count"]() if "pso_miss_count" in fns else 0
    live = fns["live_bytes"]() if "live_bytes" in fns else 0
    pool = fns["pool_bytes"]() if "pool_bytes" in fns else 0

    num_tokens = sum(len(r.hypotheses[0]) for r in results)
    overhead_time = wall_time - gpu_time
    commit_overhead = commits * 0.0004  # ~0.4ms per commit

    print(f"\n--- Results ---")
    print(f"  Wall time:         {wall_time*1000:.0f} ms")
    print(f"  GPU time:          {gpu_time*1000:.0f} ms")
    print(f"  CPU overhead:      {overhead_time*1000:.0f} ms ({overhead_time/wall_time*100:.1f}%)")
    print(f"  Tokens generated:  {num_tokens}")
    print(f"  Tok/s:             {num_tokens/wall_time:.1f}")
    print()
    print(f"--- Sync Analysis ---")
    print(f"  Total commits:     {commits}")
    print(f"  Commits/token:     {commits/max(num_tokens,1):.2f}")
    print(f"  Est. commit overhead: {commit_overhead*1000:.0f} ms ({commit_overhead/wall_time*100:.1f}%)")
    print(f"  GPU utilization:   {gpu_time/wall_time*100:.1f}%")
    print()
    print(f"--- PSO Cache ---")
    print(f"  Hits:   {pso_hits}")
    print(f"  Misses: {pso_misses}")
    print(f"  Hit rate: {pso_hits/(pso_hits+pso_misses)*100:.1f}%" if pso_hits+pso_misses > 0 else "  N/A")
    print()
    print(f"--- Memory ---")
    print(f"  Live: {live/1024/1024:.1f} MB")
    print(f"  Pool: {pool/1024/1024:.1f} MB")
    print()

    # =========================================================================
    # Phase 3: Commit trace dump (which callers are syncing)
    # =========================================================================
    print("=" * 70)
    print("COMMIT TRACE (sorted by count)")
    print("=" * 70)
    if "dump_commit_trace" in fns:
        fns["dump_commit_trace"]()
    print()

    # =========================================================================
    # Phase 4: Per-batch analysis
    # =========================================================================
    print("=" * 70)
    print("PER-BATCH ANALYSIS")
    print("=" * 70)

    # Reset and run batch by batch to see per-batch commit counts
    batch_stats = []
    idx = 0
    while idx < len(source_tokens):
        batch = source_tokens[idx:idx + args.max_batch_size]
        idx += args.max_batch_size

        if "reset_commit_count" in fns:
            fns["reset_commit_count"]()
        if "reset_gpu_time" in fns:
            fns["reset_gpu_time"]()

        t0 = time.monotonic()
        batch_results = translator.translate_batch(
            batch, beam_size=args.beam_size,
            max_batch_size=args.max_batch_size,
        )
        batch_wall = time.monotonic() - t0

        batch_commits = fns["commit_count"]() if "commit_count" in fns else 0
        batch_gpu = fns["gpu_time_elapsed"]() if "gpu_time_elapsed" in fns else 0.0
        batch_tokens = sum(len(r.hypotheses[0]) for r in batch_results)
        # Average output length per sentence
        avg_out_len = batch_tokens / max(len(batch), 1)
        # Max decode steps is roughly avg_out_len
        decode_steps = avg_out_len

        batch_stats.append({
            "batch_size": len(batch),
            "tokens": batch_tokens,
            "wall_ms": batch_wall * 1000,
            "gpu_ms": batch_gpu * 1000,
            "commits": batch_commits,
            "avg_out_len": avg_out_len,
            "commits_per_step": batch_commits / max(decode_steps, 1),
        })

    print(f"\n{'Batch':>5} | {'Size':>4} | {'Tokens':>6} | {'Wall ms':>8} | "
          f"{'GPU ms':>7} | {'Commits':>7} | {'C/step':>6} | {'GPU%':>5} | {'tok/s':>7}")
    print("-" * 85)
    for i, s in enumerate(batch_stats):
        gpu_pct = s["gpu_ms"] / s["wall_ms"] * 100 if s["wall_ms"] > 0 else 0
        toks = s["tokens"] / (s["wall_ms"] / 1000) if s["wall_ms"] > 0 else 0
        print(f"{i:>5} | {s['batch_size']:>4} | {s['tokens']:>6} | "
              f"{s['wall_ms']:>8.0f} | {s['gpu_ms']:>7.0f} | "
              f"{s['commits']:>7} | {s['commits_per_step']:>6.1f} | "
              f"{gpu_pct:>4.0f}% | {toks:>7.0f}")

    # Summary
    total_wall = sum(s["wall_ms"] for s in batch_stats)
    total_gpu = sum(s["gpu_ms"] for s in batch_stats)
    total_commits = sum(s["commits"] for s in batch_stats)
    total_tokens = sum(s["tokens"] for s in batch_stats)
    avg_commits_per_step = sum(s["commits_per_step"] for s in batch_stats) / len(batch_stats)

    print(f"\n--- Per-batch Summary ---")
    print(f"  Total wall time:     {total_wall:.0f} ms")
    print(f"  Total GPU time:      {total_gpu:.0f} ms")
    print(f"  Total commits:       {total_commits}")
    print(f"  Avg commits/step:    {avg_commits_per_step:.1f}")
    print(f"  Avg GPU utilization: {total_gpu/total_wall*100:.1f}%")
    print(f"  Effective tok/s:     {total_tokens/(total_wall/1000):.1f}")
    print()

    # =========================================================================
    # Phase 5: Bottleneck diagnosis
    # =========================================================================
    print("=" * 70)
    print("BOTTLENECK DIAGNOSIS")
    print("=" * 70)

    gpu_util = gpu_time / wall_time * 100
    commit_pct = commit_overhead / wall_time * 100

    if gpu_util < 50:
        print(f"  [!!] LOW GPU UTILIZATION: {gpu_util:.0f}% — GPU mostly idle")
        print(f"       The GPU spends {100-gpu_util:.0f}% of time waiting.")
        print(f"       Root cause: {commits} commit_and_wait() calls")
        print(f"       Each commit blocks CPU→GPU for ~0.4ms")
        print(f"       Estimated commit overhead: {commit_overhead*1000:.0f}ms / {wall_time*1000:.0f}ms = {commit_pct:.0f}%")
        print()

    if commits / max(num_tokens, 1) > 1.5:
        print(f"  [!!] HIGH SYNC RATE: {commits/num_tokens:.1f} commits per token")
        print(f"       Ideal: ~1 commit per decode step (batch of tokens)")
        print(f"       Each extra commit adds ~0.4ms latency")
        print()

    remaining_overhead = overhead_time - commit_overhead
    if remaining_overhead > wall_time * 0.1:
        print(f"  [!]  CPU OVERHEAD BEYOND COMMITS: {remaining_overhead*1000:.0f} ms")
        print(f"       This includes: tokenization, beam management, memory alloc,")
        print(f"       Python→C++ boundary, encoder creation, obj-c overhead")
        print()

    if pso_misses > 0 and pso_misses > pso_hits * 0.01:
        print(f"  [!]  PSO CACHE MISSES: {pso_misses} ({pso_misses/(pso_hits+pso_misses)*100:.1f}%)")
        print(f"       Each miss triggers an MSL kernel compilation (~1-5ms)")
        print()

    del translator
    gc.collect()
    ctranslate2.clear_device_cache("mps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
