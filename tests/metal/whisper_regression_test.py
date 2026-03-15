#!/usr/bin/env python3
"""Whisper regression test — quick RTF + WER check using FLEURS benchmark infra.

Runs CT2 Metal (standard + flash) on a small FLEURS subset and compares
against stored baselines. Catches performance regressions and accuracy
degradation from kernel changes, sync bugs, or precision issues.

Usage:
    conda run -n ct2 python tests/metal/whisper_regression_test.py
    conda run -n ct2 python tests/metal/whisper_regression_test.py --lang en_us --samples 10
    conda run -n ct2 python tests/metal/whisper_regression_test.py --all-langs

Requires:
    - CTranslate2 built with CT2_WITH_METAL=ON and installed in ct2 env
    - FLEURS cache (auto-downloads if missing)
    - faster_whisper, jiwer, whisper_normalizer (or whisper)
    - CT2 model at CT2_TEST_DATA/whisper-large-v3-turbo (or ../../../data/)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR.parent.parent / "tools" / "benchmark" / "whisper_fleurs"
RESULTS_DIR = BENCH_DIR / "results"

sys.path.insert(0, str(BENCH_DIR))

from common import (
    WHISPER_LANG_CODE,
    BenchmarkResult,
    download_fleurs,
    load_cached_data,
    compute_wer,
)

# ---------------------------------------------------------------------------
# Regression thresholds — relative to stored baselines
# ---------------------------------------------------------------------------
RTF_REGRESSION_FACTOR = 1.5   # FAIL if RTF > baseline * 1.5
RTF_WARN_FACTOR = 1.2         # WARN if RTF > baseline * 1.2
WER_ABSOLUTE_CEILING = 0.30   # FAIL if WER > 30% (catches degenerate output)
WER_REGRESSION_DELTA = 0.05   # FAIL if WER > baseline + 5 percentage points


def load_baseline(framework: str, quant: str, beam: int) -> BenchmarkResult | None:
    """Load a stored baseline result JSON."""
    pattern = f"{framework}_whisper-large-v3-turbo_{quant}_b{beam}.json"
    path = RESULTS_DIR / pattern
    if path.exists():
        return BenchmarkResult.from_json(path)
    return None


def resolve_model_path(model_name: str) -> str:
    """Find CT2 model directory (same logic as FLEURS runners)."""
    if os.environ.get("CT2_TEST_DATA"):
        base = os.environ["CT2_TEST_DATA"]
    else:
        # CTranslate2 repo root is SCRIPT_DIR/../../, data is sibling: ../../data
        base = str(SCRIPT_DIR.parent.parent.parent / "data")
    path = os.path.join(base, model_name)
    if os.path.isdir(path):
        return path
    # Fallback: e2e conftest convention (../../../data from tests/metal/)
    alt = str(SCRIPT_DIR / ".." / ".." / ".." / "data" / model_name)
    alt = os.path.normpath(alt)
    if os.path.isdir(alt):
        return alt
    return path


def run_transcription(
    model_path: str,
    quant: str,
    flash: bool,
    lang: str,
    samples: int,
    warmup: int = 2,
) -> tuple[float, float, int]:
    """Run whisper transcription and return (rtf, wer, num_samples).

    Uses faster_whisper API, same as the FLEURS benchmark runners.
    """
    from faster_whisper import WhisperModel
    import ctranslate2

    audios, refs, durs = load_cached_data(lang)
    n = min(len(audios), samples)
    audios, refs, durs = audios[:n], refs[:n], durs[:n]
    total_audio_s = sum(durs)
    whisper_lang = WHISPER_LANG_CODE.get(lang, lang.split("_")[0])

    model = WhisperModel(
        model_path, device="mps", compute_type=quant,
        flash_attention=flash,
    )

    # Warmup
    for i in range(min(warmup, n)):
        segs, _ = model.transcribe(
            audios[i], language=whisper_lang, beam_size=1,
            without_timestamps=True,
        )
        list(segs)

    # Timed inference
    hypotheses: list[str] = []
    total_wall = 0.0
    for i in range(n):
        t0 = time.monotonic()
        segs, _ = model.transcribe(
            audios[i], language=whisper_lang, beam_size=1,
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segs)
        elapsed = time.monotonic() - t0
        total_wall += elapsed
        hypotheses.append(text)

    rtf = total_wall / total_audio_s if total_audio_s > 0 else 0.0
    wer_val = compute_wer(hypotheses, refs, lang)

    del model
    gc.collect()
    ctranslate2.clear_device_cache("mps")

    return rtf, wer_val, n


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper regression test")
    parser.add_argument(
        "--lang", default="en_us",
        help="Single language to test (default: en_us)",
    )
    parser.add_argument(
        "--all-langs", action="store_true",
        help="Test all 6 FLEURS languages (slower)",
    )
    parser.add_argument(
        "--samples", type=int, default=0,
        help="Samples per language (default: match baseline, or 10 if no baseline)",
    )
    parser.add_argument(
        "--model", default="whisper-large-v3-turbo",
        help="CT2 model directory name",
    )
    parser.add_argument(
        "--skip-flash", action="store_true",
        help="Skip FlashMHA test (faster)",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not os.path.isdir(model_path):
        print(f"SKIP: model not found at {model_path}")
        return 0

    langs = (
        ["en_us", "ja_jp", "zh_cn", "de_de", "es_419", "ar_eg"]
        if args.all_langs else [args.lang]
    )

    # Configs to test: (label, framework_key, quant, flash)
    configs = [
        ("CT2 Metal f16", "ct2_metal", "float16", False),
    ]
    if not args.skip_flash:
        configs.append(
            ("CT2 Metal Flash f16", "ct2_metal_flash", "float16", True),
        )

    # Determine sample count: match baseline if --samples not explicitly set
    if args.samples <= 0:
        for _, fw_key, quant, _ in configs:
            bl = load_baseline(fw_key, quant, beam=1)
            if bl:
                # Use num_samples from the first language in baseline
                first_lang = next(iter(bl.languages.values()), None)
                if first_lang:
                    args.samples = first_lang.num_samples
                    break
        if args.samples <= 0:
            args.samples = 10  # fallback

    # Ensure FLEURS data is cached
    download_fleurs(langs, num_samples=args.samples)

    passed = 0
    failed = 0
    warned = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        tag = "PASS" if ok else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{tag}] {label}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1

    def warn(label: str, detail: str = "") -> None:
        nonlocal warned
        warned += 1
        print(f"  [WARN] {label}  ({detail})")

    print("=" * 60)
    print("Whisper Regression Test")
    print(f"  Model: {args.model}")
    print(f"  Languages: {', '.join(langs)}")
    print(f"  Samples: {args.samples} (matched to baseline)")
    print("=" * 60)

    for label, fw_key, quant, flash in configs:
        baseline = load_baseline(fw_key, quant, beam=1)

        print(f"\n--- {label} (flash={flash}) ---")

        for lang in langs:
            # Match exact sample count from baseline for this language
            effective_samples = args.samples
            if baseline and lang in baseline.languages:
                effective_samples = baseline.languages[lang].num_samples

            print(f"\n  [{lang}]")
            rtf, wer_val, n = run_transcription(
                model_path, quant, flash, lang, effective_samples,
            )
            print(f"    RTF={rtf:.3f}  WER={wer_val:.1%}  (n={n})")

            # Check against absolute ceiling
            check(
                f"{lang} WER < {WER_ABSOLUTE_CEILING:.0%}",
                wer_val < WER_ABSOLUTE_CEILING,
                f"WER={wer_val:.1%}",
            )

            # Check against baseline if available
            if baseline and lang in baseline.languages:
                bl = baseline.languages[lang]
                bl_rtf = bl.rtf
                bl_wer = bl.wer

                # RTF regression
                rtf_ratio = rtf / bl_rtf if bl_rtf > 0 else 1.0
                if rtf_ratio > RTF_REGRESSION_FACTOR:
                    check(
                        f"{lang} RTF regression",
                        False,
                        f"RTF {rtf:.3f} vs baseline {bl_rtf:.3f} ({rtf_ratio:.1f}x slower)",
                    )
                elif rtf_ratio > RTF_WARN_FACTOR:
                    check(f"{lang} RTF within ceiling", True, f"RTF={rtf:.3f}")
                    warn(
                        f"{lang} RTF above typical",
                        f"{rtf:.3f} vs baseline {bl_rtf:.3f} ({rtf_ratio:.1f}x)",
                    )
                else:
                    check(
                        f"{lang} RTF OK",
                        True,
                        f"{rtf:.3f} vs baseline {bl_rtf:.3f} ({rtf_ratio:.1f}x)",
                    )

                # WER regression
                wer_delta = wer_val - bl_wer
                if wer_delta > WER_REGRESSION_DELTA:
                    check(
                        f"{lang} WER regression",
                        False,
                        f"WER {wer_val:.1%} vs baseline {bl_wer:.1%} (+{wer_delta:.1%})",
                    )
                else:
                    check(
                        f"{lang} WER OK",
                        True,
                        f"{wer_val:.1%} vs baseline {bl_wer:.1%} (delta {wer_delta:+.1%})",
                    )
            else:
                warn(f"{lang} no baseline", "cannot compare, first run?")

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} passed", end="")
    if warned:
        print(f", {warned} warnings", end="")
    print()

    if failed:
        print("  REGRESSION DETECTED")
        return 1
    print("  ALL PASS — no regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
