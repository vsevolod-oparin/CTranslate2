#!/usr/bin/env python3
"""Runner: CTranslate2 Metal backend (FlashMHA) via faster_whisper.

Thin wrapper around runner_ct2_metal.py with flash_attention=True.

Usage:
    python runner_ct2_metal_flash.py --model whisper-large-v3-turbo --quant float16 --beam 1
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="CTranslate2 Metal FlashMHA whisper benchmark")

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from common import (
        WHISPER_LANG_CODE,
        BenchmarkResult,
        LanguageResult,
        add_common_args,
        auto_output_path,
        compute_wer,
        get_peak_rss_mb,
        load_cached_data,
        parse_languages,
    )

    add_common_args(parser)
    parser.add_argument(
        "--model-dir", default=None,
        help="Directory containing CT2 whisper models",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup samples (default: 3)",
    )
    args = parser.parse_args()

    import os
    if args.model_dir:
        model_base = args.model_dir
    elif os.environ.get("CT2_TEST_DATA"):
        model_base = os.environ["CT2_TEST_DATA"]
    else:
        model_base = str(__import__("pathlib").Path(__file__).parent.parent.parent.parent.parent / "data")

    model_path = os.path.join(model_base, args.model)
    if not os.path.isdir(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1

    languages = parse_languages(args)

    print(f"=== CTranslate2 Metal (FlashMHA) ===")
    print(f"Model: {args.model} ({model_path})")
    print(f"Quant: {args.quant}, Beam: {args.beam}, Warmup: {args.warmup}")
    print(f"Languages: {', '.join(languages)} ({args.samples} samples each)")

    from faster_whisper import WhisperModel
    import ctranslate2

    print(f"\nLoading model on MPS with compute_type={args.quant}, flash_attention=True...")
    t_load = time.monotonic()
    model = WhisperModel(model_path, device="mps", compute_type=args.quant, flash_attention=True)
    print(f"  Loaded in {time.monotonic() - t_load:.1f}s")
    print(f"  Actual compute_type: {model.model.compute_type}")

    result = BenchmarkResult(
        framework="ct2_metal_flash",
        model=args.model,
        quant=args.quant,
        beam_size=args.beam,
        flash_attention=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    for lang in languages:
        audios, refs, durs = load_cached_data(lang)
        n = min(len(audios), args.samples)
        audios, refs, durs = audios[:n], refs[:n], durs[:n]
        total_audio_s = sum(durs)
        whisper_lang = WHISPER_LANG_CODE.get(lang, lang.split("_")[0])

        print(f"\n  [{lang}] {n} samples, {total_audio_s:.0f}s audio, whisper_lang={whisper_lang}")

        n_warmup = min(args.warmup, n)
        try:
            for i in range(n_warmup):
                segs, _ = model.transcribe(
                    audios[i], language=whisper_lang, beam_size=args.beam,
                    without_timestamps=True,
                )
                list(segs)
        except (ValueError, RuntimeError) as e:
            print(f"    ERROR during warmup: {e}")
            print(f"    Skipping {lang} for this config")
            continue

        hypotheses: list[str] = []
        total_wall = 0.0
        for i in range(n):
            t0 = time.monotonic()
            segs, _ = model.transcribe(
                audios[i], language=whisper_lang, beam_size=args.beam,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segs)
            elapsed = time.monotonic() - t0
            total_wall += elapsed
            hypotheses.append(text)

        wer_val = compute_wer(hypotheses, refs, lang)
        rtf = total_wall / total_audio_s if total_audio_s > 0 else 0.0

        result.languages[lang] = LanguageResult(
            wer=round(wer_val, 4),
            rtf=round(rtf, 4),
            wall_s=round(total_wall, 2),
            audio_s=round(total_audio_s, 1),
            num_samples=n,
        )
        print(f"    WER: {wer_val:.2%}  RTF: {rtf:.3f}  Wall: {total_wall:.1f}s")
        if hypotheses:
            print(f"    Sample: {hypotheses[0][:100]}...")

    del model
    gc.collect()
    ctranslate2.clear_device_cache("mps")

    result.peak_rss_mb = round(get_peak_rss_mb(), 1)

    print(f"\n{'=' * 60}")
    print(f"{'Lang':<10} {'WER':>8} {'RTF':>8} {'Wall(s)':>8} {'Audio(s)':>8}")
    print("-" * 60)
    for lang, lr in result.languages.items():
        print(f"{lang:<10} {lr.wer:>7.2%} {lr.rtf:>8.3f} {lr.wall_s:>8.1f} {lr.audio_s:>8.1f}")
    print("-" * 60)
    print(f"{'Average':<10} {result.avg_wer():>7.2%} {result.avg_rtf():>8.3f}")
    print(f"Peak RSS: {result.peak_rss_mb:.0f} MB")

    output = args.output or str(auto_output_path("ct2_metal_flash", args.model, args.quant, args.beam))
    result.save(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
