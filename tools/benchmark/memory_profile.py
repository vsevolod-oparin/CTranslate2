#!/usr/bin/env python3
"""Memory profiling for MPS translation inference.

Samples RSS, Metal live_bytes, and Metal pool_bytes at regular intervals
while translating in configurable chunks. Detects memory leaks and aborts
if RSS exceeds a safety limit.

Usage:
    python memory_profile.py [--compute_type float16] [--max_sentences 200]
    python memory_profile.py --clear_between_chunks --rss_limit_mb 4000
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MemorySample:
    """Single point-in-time memory measurement."""

    timestamp_s: float
    rss_mb: float
    live_mb: float
    pool_mb: float
    label: str = ""


@dataclass
class MemoryTimeline:
    """Accumulated memory samples over the profiling run."""

    samples: list[MemorySample] = field(default_factory=list)
    t0: float = 0.0

    def add(self, rss_mb: float, live_mb: float, pool_mb: float,
            label: str = "") -> None:
        self.samples.append(MemorySample(
            timestamp_s=time.monotonic() - self.t0,
            rss_mb=rss_mb,
            live_mb=live_mb,
            pool_mb=pool_mb,
            label=label,
        ))

    @property
    def peak_rss(self) -> float:
        return max((s.rss_mb for s in self.samples), default=0.0)

    @property
    def peak_live(self) -> float:
        return max((s.live_mb for s in self.samples), default=0.0)

    @property
    def peak_pool(self) -> float:
        return max((s.pool_mb for s in self.samples), default=0.0)

    @property
    def final_live(self) -> float:
        return self.samples[-1].live_mb if self.samples else 0.0

    @property
    def final_pool(self) -> float:
        return self.samples[-1].pool_mb if self.samples else 0.0

    def pool_is_monotonic(self) -> bool:
        """Return True if pool_mb never decreases -- a leak indicator."""
        pool_values = [s.pool_mb for s in self.samples if s.pool_mb > 0]
        if len(pool_values) < 3:
            return False
        for i in range(1, len(pool_values)):
            if pool_values[i] < pool_values[i - 1] - 0.01:
                return False
        return True


# ---------------------------------------------------------------------------
# C library helpers (matches profile_mps_translation.py conventions)
# ---------------------------------------------------------------------------
def load_ct2_lib() -> ctypes.CDLL:
    """Load the CTranslate2 shared library for direct symbol access."""
    import ctranslate2 as _ct2

    lib_dir = os.path.dirname(_ct2.__file__)
    candidates: list[str] = []

    # Relative to the Python package
    for name in ("libctranslate2.dylib", "libctranslate2.4.dylib",
                 "libctranslate2.4.7.1.dylib"):
        candidates.append(os.path.normpath(
            os.path.join(lib_dir, "..", "..", "..", name)))

    # Conda env lib directory
    prefix = sys.prefix
    for name in ("libctranslate2.dylib", "libctranslate2.4.dylib",
                 "libctranslate2.4.7.1.dylib"):
        candidates.append(os.path.join(prefix, "lib", name))

    for path in candidates:
        if os.path.isfile(path):
            return ctypes.CDLL(path)

    raise FileNotFoundError(
        "Cannot find libctranslate2 shared library. "
        "Searched:\n  " + "\n  ".join(candidates)
    )


def setup_memory_fns(
    lib: ctypes.CDLL,
) -> dict[str, Callable[[], int]]:
    """Bind pool_bytes and live_bytes from the Metal allocator."""
    fns: dict[str, Callable[[], int]] = {}
    try:
        fn = lib._ZN11ctranslate25metal10pool_bytesEv
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
        fns["pool_bytes"] = fn
    except AttributeError:
        print("WARNING: pool_bytes symbol not found")

    try:
        fn = lib._ZN11ctranslate25metal10live_bytesEv
        fn.restype = ctypes.c_size_t
        fn.argtypes = []
        fns["live_bytes"] = fn
    except AttributeError:
        print("WARNING: live_bytes symbol not found")

    return fns


# ---------------------------------------------------------------------------
# RSS measurement
# ---------------------------------------------------------------------------
def get_rss_mb() -> float:
    """Return current process RSS in megabytes (macOS / Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def get_rss_mb_precise() -> float:
    """Return current RSS via /proc or task_info. Falls back to ru_maxrss."""
    if sys.platform == "darwin":
        # Use mach API through ctypes for actual (not peak) RSS
        try:
            libc = ctypes.CDLL("libSystem.B.dylib")
            # mach_task_self() -> mach_port_t
            task = libc.mach_task_self()

            class TaskBasicInfo(ctypes.Structure):
                _fields_ = [
                    ("suspend_count", ctypes.c_int32),
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("user_time_seconds", ctypes.c_int32),
                    ("user_time_microseconds", ctypes.c_int32),
                    ("system_time_seconds", ctypes.c_int32),
                    ("system_time_microseconds", ctypes.c_int32),
                    ("policy", ctypes.c_int32),
                ]

            MACH_TASK_BASIC_INFO = 20
            info = TaskBasicInfo()
            count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
            kr = libc.task_info(
                task, MACH_TASK_BASIC_INFO,
                ctypes.byref(info), ctypes.byref(count),
            )
            if kr == 0:
                return info.resident_size / (1024 * 1024)
        except Exception:
            pass
    elif os.path.isfile("/proc/self/statm"):
        try:
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except Exception:
            pass
    return get_rss_mb()


# ---------------------------------------------------------------------------
# Background sampler thread
# ---------------------------------------------------------------------------
class MemorySampler:
    """Background thread that samples memory at a fixed interval."""

    def __init__(
        self,
        timeline: MemoryTimeline,
        metal_fns: dict[str, Callable[[], int]],
        interval_s: float = 0.1,
        rss_limit_mb: float = 8000.0,
    ) -> None:
        self._timeline = timeline
        self._metal_fns = metal_fns
        self._interval_s = interval_s
        self._rss_limit_mb = rss_limit_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mem-sampler",
        )

    def start(self) -> None:
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _sample(self) -> tuple[float, float, float]:
        rss = get_rss_mb_precise()
        live_b = self._metal_fns["live_bytes"]() if "live_bytes" in self._metal_fns else 0
        pool_b = self._metal_fns["pool_bytes"]() if "pool_bytes" in self._metal_fns else 0
        live_mb = live_b / (1024 * 1024)
        pool_mb = pool_b / (1024 * 1024)
        return rss, live_mb, pool_mb

    def sample_now(self, label: str = "") -> tuple[float, float, float]:
        """Take one explicit sample (called from the main thread)."""
        rss, live_mb, pool_mb = self._sample()
        self._timeline.add(rss, live_mb, pool_mb, label=label)
        return rss, live_mb, pool_mb

    def _run(self) -> None:
        while not self._stop.is_set():
            rss, live_mb, pool_mb = self._sample()
            self._timeline.add(rss, live_mb, pool_mb)

            # Safety: abort the whole process if RSS exceeds limit
            if rss > self._rss_limit_mb:
                msg = (
                    f"\n*** RSS LIMIT EXCEEDED: {rss:.0f} MB > "
                    f"{self._rss_limit_mb:.0f} MB -- aborting ***\n"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                os._exit(1)

            self._stop.wait(self._interval_s)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_plot(timeline: MemoryTimeline, path: str) -> bool:
    """Save a memory timeline chart to PNG. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ts = [s.timestamp_s for s in timeline.samples]
    rss = [s.rss_mb for s in timeline.samples]
    live = [s.live_mb for s in timeline.samples]
    pool = [s.pool_mb for s in timeline.samples]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, rss, label="RSS", linewidth=1.5, color="#2196F3")
    ax.plot(ts, live, label="Metal live", linewidth=1.5, color="#4CAF50")
    ax.plot(ts, pool, label="Metal pool", linewidth=1.5, color="#FF9800")

    # Mark labeled samples (chunk boundaries)
    labeled = [(s.timestamp_s, s.rss_mb, s.label)
               for s in timeline.samples if s.label]
    for t, r, lbl in labeled:
        ax.axvline(x=t, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("MPS Translation Memory Profile")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def print_ascii_table(timeline: MemoryTimeline) -> None:
    """Print a down-sampled ASCII table of the timeline."""
    samples = timeline.samples
    if not samples:
        print("  (no samples)")
        return

    # Down-sample to at most 40 rows
    step = max(1, len(samples) // 40)
    print(f"  {'Time(s)':>8}  {'RSS(MB)':>9}  {'Live(MB)':>9}  {'Pool(MB)':>9}  Label")
    print(f"  {'--------':>8}  {'---------':>9}  {'---------':>9}  {'---------':>9}  -----")
    for i in range(0, len(samples), step):
        s = samples[i]
        lbl = f"  {s.label}" if s.label else ""
        print(f"  {s.timestamp_s:>8.2f}  {s.rss_mb:>9.1f}  {s.live_mb:>9.1f}  "
              f"{s.pool_mb:>9.1f}{lbl}")
    # Always show the last sample
    s = samples[-1]
    lbl = f"  {s.label}" if s.label else ""
    print(f"  {s.timestamp_s:>8.2f}  {s.rss_mb:>9.1f}  {s.live_mb:>9.1f}  "
          f"{s.pool_mb:>9.1f}{lbl}")


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
def diagnose(timeline: MemoryTimeline) -> None:
    """Print a clear diagnosis of where memory is accumulating."""
    print("=" * 70)
    print("MEMORY DIAGNOSIS")
    print("=" * 70)

    print(f"\n  Peak RSS:        {timeline.peak_rss:>9.1f} MB")
    print(f"  Peak Metal live: {timeline.peak_live:>9.1f} MB")
    print(f"  Peak Metal pool: {timeline.peak_pool:>9.1f} MB")
    print(f"  Final Metal live:{timeline.final_live:>9.1f} MB")
    print(f"  Final Metal pool:{timeline.final_pool:>9.1f} MB")
    print()

    # Check for pool growth (leak indicator)
    monotonic = timeline.pool_is_monotonic()
    if monotonic:
        print("  [!!] POOL IS MONOTONICALLY GROWING -- likely Metal buffer leak")
        print("       The pool never shrinks, meaning buffers are allocated but")
        print("       never returned. Check for missing [release] calls on MPS")
        print("       objects or MTLBuffer allocations without deallocation.")
    else:
        print("  [OK] Pool is NOT monotonically growing (no obvious leak pattern)")
    print()

    # Check if live stays high after translation
    if timeline.final_live > timeline.peak_live * 0.8 and timeline.peak_live > 10:
        print("  [!!] LIVE MEMORY STAYS HIGH after translation completes")
        print(f"       Final live ({timeline.final_live:.1f} MB) is "
              f"{timeline.final_live / timeline.peak_live * 100:.0f}% of peak "
              f"({timeline.peak_live:.1f} MB)")
        print("       Model weights are expected to persist; check for extra")
        print("       retained buffers (MPS temporaries, cached command buffers).")
    print()

    # Check pool vs live divergence
    if timeline.peak_pool > timeline.peak_live * 2 and timeline.peak_pool > 50:
        ratio = timeline.peak_pool / max(timeline.peak_live, 0.01)
        print(f"  [!]  POOL >> LIVE: pool/live ratio = {ratio:.1f}x")
        print("       The allocator is caching significantly more memory than")
        print("       what is actively used. Consider calling clear_device_cache()")
        print("       between batches or tuning the allocator pool limits.")
    print()

    # RSS vs pool check
    rss_overhead = timeline.peak_rss - timeline.peak_pool
    if rss_overhead > 500:
        print(f"  [!]  RSS OVERHEAD: {rss_overhead:.0f} MB beyond Metal pool")
        print("       This includes: Python heap, model weights in CPU memory,")
        print("       tokenizer data, framework overhead, fragmentation.")
    print()

    # Growth rate analysis on labeled samples (chunk boundaries)
    labeled = [s for s in timeline.samples if s.label.startswith("chunk_")]
    if len(labeled) >= 3:
        pool_deltas = [
            labeled[i].pool_mb - labeled[i - 1].pool_mb
            for i in range(1, len(labeled))
        ]
        avg_delta = sum(pool_deltas) / len(pool_deltas)
        if avg_delta > 1.0:
            print(f"  [!!] POOL GROWS ~{avg_delta:.1f} MB per chunk on average")
            print("       Extrapolating: translating 1000 chunks would add "
                  f"~{avg_delta * 1000:.0f} MB")
        elif avg_delta > 0.1:
            print(f"  [!]  POOL GROWS ~{avg_delta:.2f} MB per chunk (slow leak)")
        else:
            print(f"  [OK] Pool growth per chunk: {avg_delta:+.2f} MB (stable)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Memory profiling for MPS translation inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compute_type", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16",
                                 "int8", "int8_float16", "int8_bfloat16"],
                        help="Compute type for the translator")
    parser.add_argument("--max_sentences", type=int, default=200,
                        help="Total number of sentences to translate")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Sentences per translation chunk")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam search width")
    parser.add_argument("--rss_limit_mb", type=float, default=8000.0,
                        help="Abort if RSS exceeds this limit (MB)")
    parser.add_argument("--clear_between_chunks", action="store_true",
                        help="Call clear_device_cache + gc.collect between chunks")
    parser.add_argument("--sample_interval_ms", type=int, default=100,
                        help="Background memory sampling interval (ms)")
    parser.add_argument("--output_png", type=str, default="",
                        help="Path for the output plot PNG (default: auto-named)")
    args = parser.parse_args()

    import ctranslate2
    import sacrebleu
    from transformers import MarianTokenizer

    # ----- Locate model -----
    data_dir = os.environ.get(
        "CT2_TEST_DATA",
        os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "data")),
    )
    model_map = {
        "float32": os.path.join(data_dir, "opus-mt-en-de"),
        "float16": os.path.join(data_dir, "opus-mt-en-de-f16"),
        "bfloat16": os.path.join(data_dir, "opus-mt-en-de"),
        "int8": os.path.join(data_dir, "opus-mt-en-de"),
        "int8_float16": os.path.join(data_dir, "opus-mt-en-de"),
        "int8_bfloat16": os.path.join(data_dir, "opus-mt-en-de"),
    }
    model_path = model_map[args.compute_type]
    if not os.path.isdir(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1

    # ----- Load test data -----
    source_file = sacrebleu.get_source_file("wmt14", langpair="en-de")
    with open(source_file) as f:
        source_sentences = [line.strip() for line in f][:args.max_sentences]

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    source_tokens = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(s))
        for s in source_sentences
    ]

    # ----- Load ctypes symbols -----
    lib = load_ct2_lib()
    metal_fns = setup_memory_fns(lib)

    print("=" * 70)
    print("MPS TRANSLATION MEMORY PROFILE")
    print("=" * 70)
    print(f"  Model:               {model_path}")
    print(f"  Compute type:        {args.compute_type}")
    print(f"  Total sentences:     {len(source_tokens)}")
    print(f"  Chunk size:          {args.chunk_size}")
    print(f"  Beam size:           {args.beam_size}")
    print(f"  RSS limit:           {args.rss_limit_mb:.0f} MB")
    print(f"  Clear between chunks:{args.clear_between_chunks}")
    print(f"  Sample interval:     {args.sample_interval_ms} ms")
    print()

    # ----- Set up timeline and sampler -----
    timeline = MemoryTimeline()
    timeline.t0 = time.monotonic()

    sampler = MemorySampler(
        timeline=timeline,
        metal_fns=metal_fns,
        interval_s=args.sample_interval_ms / 1000.0,
        rss_limit_mb=args.rss_limit_mb,
    )

    # Take a baseline sample before loading the model
    rss, live, pool = sampler.sample_now(label="before_model_load")
    print(f"  Before model load:   RSS={rss:.1f} MB  live={live:.1f} MB  pool={pool:.1f} MB")

    # ----- Load model -----
    translator = ctranslate2.Translator(
        model_path, device="mps", compute_type=args.compute_type,
        intra_threads=1,
    )

    rss, live, pool = sampler.sample_now(label="after_model_load")
    print(f"  After model load:    RSS={rss:.1f} MB  live={live:.1f} MB  pool={pool:.1f} MB")

    # Warmup (compiles PSOs)
    translator.translate_batch([[""]], beam_size=1)

    rss, live, pool = sampler.sample_now(label="after_warmup")
    print(f"  After warmup:        RSS={rss:.1f} MB  live={live:.1f} MB  pool={pool:.1f} MB")
    print()

    # ----- Start background sampling -----
    sampler.start()

    # ----- Translate in chunks -----
    print(f"{'Chunk':>5}  {'Sent':>5}  {'Tokens':>6}  {'Time(ms)':>9}  "
          f"{'RSS(MB)':>9}  {'Live(MB)':>9}  {'Pool(MB)':>9}  {'Tok/s':>7}")
    print("-" * 78)

    total_tokens = 0
    total_time_ms = 0.0
    chunk_idx = 0

    for start in range(0, len(source_tokens), args.chunk_size):
        chunk = source_tokens[start:start + args.chunk_size]
        chunk_idx += 1

        t0 = time.monotonic()
        results = translator.translate_batch(
            chunk, beam_size=args.beam_size,
            max_batch_size=args.chunk_size,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        tokens = sum(len(r.hypotheses[0]) for r in results)
        total_tokens += tokens
        total_time_ms += elapsed_ms

        rss, live, pool = sampler.sample_now(label=f"chunk_{chunk_idx}")
        tok_s = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        sent_range = f"{start + 1}-{start + len(chunk)}"
        print(f"{chunk_idx:>5}  {sent_range:>5}  {tokens:>6}  {elapsed_ms:>9.0f}  "
              f"{rss:>9.1f}  {live:>9.1f}  {pool:>9.1f}  {tok_s:>7.0f}")

        if args.clear_between_chunks:
            gc.collect()
            ctranslate2.clear_device_cache("mps")
            rss_after, live_after, pool_after = sampler.sample_now(
                label=f"chunk_{chunk_idx}_cleared")
            print(f"       (cleared)              "
                  f"{rss_after:>9.1f}  {live_after:>9.1f}  {pool_after:>9.1f}")

    # ----- Stop sampler -----
    sampler.stop()

    # Final sample after cleanup
    gc.collect()
    rss, live, pool = sampler.sample_now(label="after_translation")
    print()
    print(f"  After all chunks:    RSS={rss:.1f} MB  live={live:.1f} MB  pool={pool:.1f} MB")

    # Unload model
    del translator
    gc.collect()
    ctranslate2.clear_device_cache("mps")

    rss, live, pool = sampler.sample_now(label="after_cleanup")
    print(f"  After cleanup:       RSS={rss:.1f} MB  live={live:.1f} MB  pool={pool:.1f} MB")

    # ----- Summary -----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total sentences:  {len(source_tokens)}")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Total time:       {total_time_ms:.0f} ms")
    if total_time_ms > 0:
        print(f"  Overall tok/s:    {total_tokens / (total_time_ms / 1000):.0f}")
    print()

    # ----- Diagnosis -----
    diagnose(timeline)

    # ----- Plot or ASCII table -----
    output_png = args.output_png
    if not output_png:
        output_png = os.path.join(
            os.path.dirname(__file__),
            f"memory_profile_{args.compute_type}.png",
        )

    if save_plot(timeline, output_png):
        print(f"  Plot saved to: {output_png}")
    else:
        print("  matplotlib not available -- printing ASCII table instead")
        print()
        print_ascii_table(timeline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
