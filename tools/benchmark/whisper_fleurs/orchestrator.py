#!/usr/bin/env python3
"""Orchestrator: runs the full benchmark matrix and produces a unified report.

Usage:
    # Full matrix (all frameworks, all quants, beam 1+5)
    python orchestrator.py --all

    # Quick mode (10 samples, greedy only, turbo only, 3 frameworks)
    python orchestrator.py --quick

    # Single framework
    python orchestrator.py --frameworks ct2_metal,whisper_cpp

    # Resume (skip already-computed results in results/)
    python orchestrator.py --all --resume

    # Report only (from existing results)
    python orchestrator.py --report-only

    # Custom
    python orchestrator.py --frameworks ct2_metal --quants float16 --beams 1 --samples 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))
from common import (
    DEFAULT_LANGUAGES,
    LANGUAGE_NAMES,
    BenchmarkResult,
    download_fleurs,
)


# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    framework: str
    runner: str  # script filename
    model: str
    quant: str
    beam: int
    extra_args: list[str] | None = None

    @property
    def result_key(self) -> str:
        return f"{self.framework}_{self.model}_{self.quant}_b{self.beam}"

    @property
    def result_path(self) -> Path:
        return RESULTS_DIR / f"{self.framework}_{self.model}_{self.quant}_b{self.beam}.json"


# Full matrix definition
FULL_MATRIX: list[RunConfig] = [
    # CT2 Metal (standard MHA)
    RunConfig("ct2_metal", "runner_ct2_metal.py", "whisper-large-v3-turbo", "float16", 1),
    RunConfig("ct2_metal", "runner_ct2_metal.py", "whisper-large-v3-turbo", "float16", 5),
    RunConfig("ct2_metal", "runner_ct2_metal.py", "whisper-large-v3-turbo", "float32", 1),
    RunConfig("ct2_metal", "runner_ct2_metal.py", "whisper-large-v3-turbo", "float32", 5),
    # CT2 Metal FlashMHA
    RunConfig("ct2_metal_flash", "runner_ct2_metal_flash.py", "whisper-large-v3-turbo", "float16", 1),
    RunConfig("ct2_metal_flash", "runner_ct2_metal_flash.py", "whisper-large-v3-turbo", "float16", 5),
    RunConfig("ct2_metal_flash", "runner_ct2_metal_flash.py", "whisper-large-v3-turbo", "float32", 1),
    # CT2 CPU
    RunConfig("ct2_cpu", "runner_ct2_cpu.py", "whisper-large-v3-turbo", "float32", 1),
    RunConfig("ct2_cpu", "runner_ct2_cpu.py", "whisper-large-v3-turbo", "int8", 1),
    # whisper.cpp
    RunConfig("whisper_cpp", "runner_whisper_cpp.py", "whisper-large-v3-turbo", "F16", 1),
    RunConfig("whisper_cpp", "runner_whisper_cpp.py", "whisper-large-v3-turbo", "Q5_0", 1),
    # mlx-whisper
    RunConfig("mlx_whisper", "runner_mlx_whisper.py", "whisper-large-v3-turbo", "f16", 1),
    # OpenAI whisper (reference)
    RunConfig("openai_whisper", "runner_openai_whisper.py", "whisper-large-v3-turbo", "f32", 1),
]

QUICK_MATRIX: list[RunConfig] = [
    RunConfig("ct2_metal", "runner_ct2_metal.py", "whisper-large-v3-turbo", "float16", 1),
    RunConfig("ct2_metal_flash", "runner_ct2_metal_flash.py", "whisper-large-v3-turbo", "float16", 1),
    RunConfig("whisper_cpp", "runner_whisper_cpp.py", "whisper-large-v3-turbo", "F16", 1),
    RunConfig("mlx_whisper", "runner_mlx_whisper.py", "whisper-large-v3-turbo", "f16", 1),
]


def filter_matrix(
    matrix: list[RunConfig],
    frameworks: list[str] | None = None,
    quants: list[str] | None = None,
    beams: list[int] | None = None,
) -> list[RunConfig]:
    out = matrix
    if frameworks:
        out = [c for c in out if c.framework in frameworks]
    if quants:
        out = [c for c in out if c.quant in quants]
    if beams:
        out = [c for c in out if c.beam in beams]
    return out


# ---------------------------------------------------------------------------
# Runner execution
# ---------------------------------------------------------------------------

def run_config(
    cfg: RunConfig,
    languages: str,
    samples: int,
    resume: bool = False,
) -> tuple[bool, float]:
    """Run a single benchmark config. Returns (success, elapsed_seconds)."""
    if resume and cfg.result_path.exists():
        print(f"  SKIP (cached): {cfg.result_key}")
        return True, 0.0

    runner_path = SCRIPT_DIR / cfg.runner
    cmd = [
        sys.executable, str(runner_path),
        "--model", cfg.model,
        "--quant", cfg.quant,
        "--beam", str(cfg.beam),
        "--languages", languages,
        "--samples", str(samples),
    ]
    if cfg.extra_args:
        cmd.extend(cfg.extra_args)

    print(f"\n{'='*70}")
    print(f"  Running: {cfg.result_key}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR.parent.parent.parent),  # CTranslate2 root
            timeout=3600,  # 1 hour max per config
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            print(f"  FAILED: {cfg.result_key} (exit code {proc.returncode})")
            return False, elapsed
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"  TIMEOUT: {cfg.result_key} after {elapsed:.0f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  ERROR: {cfg.result_key}: {e}")
        return False, elapsed


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def load_all_results() -> list[BenchmarkResult]:
    """Load all JSON result files from results directory."""
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = BenchmarkResult.from_json(p)
            if r.languages:  # skip empty results
                results.append(r)
        except Exception as e:
            print(f"  Warning: skipping {p.name}: {e}")
    return results


def framework_label(r: BenchmarkResult) -> str:
    """Human-readable label for a benchmark result."""
    flash = " (Flash)" if r.flash_attention else ""
    beam = f" b={r.beam_size}" if r.beam_size > 1 else ""
    return f"{r.framework}{flash} {r.quant}{beam}"


def generate_report(results: list[BenchmarkResult], output_path: Path | None = None) -> str:
    """Generate a Markdown benchmark report from results."""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Milestone 16: Multi-Language Whisper Benchmark Report")
    lines.append(f"\n**Generated:** {now}")
    lines.append(f"**Model:** whisper-large-v3-turbo")
    lines.append(f"**Dataset:** FLEURS test split (50 samples/language)")
    lines.append(f"**Platform:** Apple M4, macOS")
    lines.append("")

    # Sort by avg RTF (speed)
    results_sorted = sorted(results, key=lambda r: r.avg_rtf())

    # --------------- Main comparison table ---------------
    langs = DEFAULT_LANGUAGES
    lang_headers = [LANGUAGE_NAMES.get(l, l)[:3].upper() for l in langs]

    lines.append("## Speed Comparison (RTF — lower is better)")
    lines.append("")
    header = f"| {'Configuration':<40} | {'Avg RTF':>8} |"
    for lh in lang_headers:
        header += f" {lh:>5} |"
    header += f" {'RSS MB':>7} |"
    lines.append(header)
    sep = f"|{'-'*42}|{'-'*10}|"
    for _ in lang_headers:
        sep += f"{'-'*7}|"
    sep += f"{'-'*9}|"
    lines.append(sep)

    for r in results_sorted:
        label = framework_label(r)
        row = f"| {label:<40} | {r.avg_rtf():>8.3f} |"
        for l in langs:
            lr = r.languages.get(l)
            if lr:
                row += f" {lr.rtf:>5.3f} |"
            else:
                row += f" {'—':>5} |"
        row += f" {r.peak_rss_mb:>7.0f} |"
        lines.append(row)

    lines.append("")

    # --------------- WER table ---------------
    lines.append("## Quality Comparison (WER/CER % — lower is better)")
    lines.append("*Japanese and Chinese use CER (character error rate)*")
    lines.append("")
    header = f"| {'Configuration':<40} | {'Avg':>6} |"
    for lh in lang_headers:
        header += f" {lh:>6} |"
    lines.append(header)
    sep = f"|{'-'*42}|{'-'*8}|"
    for _ in lang_headers:
        sep += f"{'-'*8}|"
    lines.append(sep)

    for r in results_sorted:
        label = framework_label(r)
        row = f"| {label:<40} | {r.avg_wer()*100:>5.1f}% |"
        for l in langs:
            lr = r.languages.get(l)
            if lr:
                row += f" {lr.wer*100:>5.1f}% |"
            else:
                row += f" {'—':>6} |"
        lines.append(row)

    lines.append("")

    # --------------- Speedup analysis ---------------
    lines.append("## Speedup Analysis")
    lines.append("")

    # Find reference configs
    ct2_metal_f16_b1 = next((r for r in results if r.framework == "ct2_metal" and r.quant == "float16" and r.beam_size == 1), None)
    ct2_flash_f16_b1 = next((r for r in results if r.framework == "ct2_metal_flash" and r.quant == "float16" and r.beam_size == 1), None)
    ct2_cpu_f32_b1 = next((r for r in results if r.framework == "ct2_cpu" and r.quant == "float32" and r.beam_size == 1), None)
    whisper_cpp_f16 = next((r for r in results if r.framework == "whisper_cpp" and r.quant == "F16"), None)
    mlx_f16 = next((r for r in results if r.framework == "mlx_whisper"), None)

    if ct2_metal_f16_b1:
        ref_rtf = ct2_metal_f16_b1.avg_rtf()
        lines.append(f"**CT2 Metal f16 (baseline RTF: {ref_rtf:.3f})**")
        lines.append("")

        comparisons = [
            ("vs CT2 CPU f32", ct2_cpu_f32_b1),
            ("vs whisper.cpp F16", whisper_cpp_f16),
            ("vs mlx-whisper f16", mlx_f16),
            ("vs CT2 Metal Flash f16", ct2_flash_f16_b1),
        ]
        for label, other in comparisons:
            if other:
                speedup = other.avg_rtf() / ref_rtf
                lines.append(f"- {label}: **{speedup:.2f}x** faster (RTF {other.avg_rtf():.3f} → {ref_rtf:.3f})")
        lines.append("")

    if ct2_flash_f16_b1 and ct2_metal_f16_b1:
        flash_rtf = ct2_flash_f16_b1.avg_rtf()
        std_rtf = ct2_metal_f16_b1.avg_rtf()
        lines.append(f"**FlashMHA vs Standard MHA (f16 beam=1):**")
        speedup = std_rtf / flash_rtf
        lines.append(f"- RTF: {std_rtf:.3f} → {flash_rtf:.3f} ({speedup:.2f}x)")
        lines.append("")

    # --------------- Key findings ---------------
    lines.append("## Key Findings")
    lines.append("")

    if results_sorted:
        fastest = results_sorted[0]
        lines.append(f"1. **Fastest config:** {framework_label(fastest)} (RTF {fastest.avg_rtf():.3f})")

    # Best WER
    best_wer = min(results, key=lambda r: r.avg_wer())
    lines.append(f"2. **Best quality:** {framework_label(best_wer)} (avg WER {best_wer.avg_wer()*100:.1f}%)")

    if ct2_metal_f16_b1 and ct2_cpu_f32_b1:
        speedup = ct2_cpu_f32_b1.avg_rtf() / ct2_metal_f16_b1.avg_rtf()
        lines.append(f"3. **Metal GPU speedup over CPU:** {speedup:.1f}x (f16 Metal vs f32 CPU)")

    # Note about whisper.cpp non-English WER
    if whisper_cpp_f16:
        non_en_wer = [lr.wer for l, lr in whisper_cpp_f16.languages.items() if not l.startswith("en")]
        if non_en_wer and max(non_en_wer) > 0.5:
            lines.append(f"4. **whisper.cpp caveat:** Very high WER for non-English languages "
                         f"(likely pywhispercpp language setting issue, not a model problem)")

    lines.append("")

    # --------------- Notes ---------------
    lines.append("## Notes")
    lines.append("")
    lines.append("- RTF = wall_time / audio_duration (lower = faster, <1.0 means faster than real-time)")
    lines.append("- WER = Word Error Rate; CER = Character Error Rate (used for ja/zh)")
    lines.append("- CT2 Metal INT8 skipped: MPS backend doesn't support INT8 encoder input")
    lines.append("- mlx-whisper: beam search not yet implemented, greedy only")
    lines.append("- OpenAI whisper: CPU only (MPS broken), extremely slow — partial results only")
    lines.append("- All configs use whisper-large-v3-turbo model")
    lines.append("- FLEURS test split, 50 samples per language, 6 languages")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"\nReport saved to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Whisper FLEURS benchmark orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Run full matrix")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 frameworks, 10 samples)")
    parser.add_argument(
        "--frameworks", default=None,
        help="Comma-separated frameworks: ct2_metal,ct2_metal_flash,ct2_cpu,whisper_cpp,mlx_whisper,openai_whisper",
    )
    parser.add_argument("--quants", default=None, help="Comma-separated quant types")
    parser.add_argument("--beams", default=None, help="Comma-separated beam sizes")
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated languages (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument("--samples", type=int, default=50, help="Samples per language (default: 50)")
    parser.add_argument("--resume", action="store_true", help="Skip configs with existing results")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    parser.add_argument(
        "--report-path", default=None,
        help="Report output path (default: agents/report/milestone-16-whisper-fleurs-benchmark.md)",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path) if args.report_path else (
        SCRIPT_DIR.parent.parent.parent / "agents" / "report" / "milestone-16-whisper-fleurs-benchmark.md"
    )

    if args.report_only:
        results = load_all_results()
        if not results:
            print("No results found in results/")
            return 1
        report = generate_report(results, report_path)
        print(report)
        return 0

    # Build matrix
    if args.quick:
        matrix = QUICK_MATRIX
        args.samples = 10
    elif args.all:
        matrix = FULL_MATRIX
    elif args.frameworks:
        matrix = FULL_MATRIX
    else:
        parser.print_help()
        return 1

    # Apply filters
    frameworks = args.frameworks.split(",") if args.frameworks else None
    quants = args.quants.split(",") if args.quants else None
    beams = [int(b) for b in args.beams.split(",")] if args.beams else None
    matrix = filter_matrix(matrix, frameworks, quants, beams)

    if not matrix:
        print("No configs match the specified filters.")
        return 1

    print(f"=== Whisper FLEURS Benchmark Orchestrator ===")
    print(f"Configs: {len(matrix)}")
    print(f"Languages: {args.languages}")
    print(f"Samples: {args.samples}")
    print(f"Resume: {args.resume}")
    print()
    for cfg in matrix:
        status = "CACHED" if args.resume and cfg.result_path.exists() else "PENDING"
        print(f"  [{status}] {cfg.result_key}")

    # Ensure FLEURS data is cached
    print("\n--- Checking FLEURS data cache ---")
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    download_fleurs(languages, num_samples=args.samples)

    # Run each config
    total_t0 = time.monotonic()
    successes = 0
    failures = 0

    for i, cfg in enumerate(matrix, 1):
        print(f"\n[{i}/{len(matrix)}] {cfg.result_key}")
        ok, elapsed = run_config(cfg, args.languages, args.samples, resume=args.resume)
        if ok:
            successes += 1
        else:
            failures += 1

    total_elapsed = time.monotonic() - total_t0

    print(f"\n{'='*70}")
    print(f"  Completed: {successes} success, {failures} failed, {total_elapsed:.0f}s total")
    print(f"{'='*70}")

    # Generate report
    results = load_all_results()
    if results:
        report = generate_report(results, report_path)
        print(report)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
