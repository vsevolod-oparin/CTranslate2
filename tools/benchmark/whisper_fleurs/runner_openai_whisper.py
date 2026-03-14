#!/usr/bin/env python3
"""Runner: OpenAI whisper (CPU only, WER accuracy reference).

Usage:
    python runner_openai_whisper.py --model whisper-large-v3-turbo --beam 1

Note: OpenAI whisper MPS backend is broken. Always uses CPU.
This runner is primarily for WER accuracy reference, not speed.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Map our model names to OpenAI whisper model names
OPENAI_MODEL_MAP = {
    "whisper-large-v3-turbo": "large-v3-turbo",
    "whisper-large-v3": "large-v3",
    "whisper-base": "base",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAI whisper benchmark (CPU, WER reference)")

    sys.path.insert(0, str(Path(__file__).parent))
    from common import (
        WHISPER_LANG_CODE,
        BenchmarkResult,
        LanguageResult,
        add_common_args,
        auto_output_path,
        compute_wer,
        export_wav,
        get_peak_rss_mb,
        load_cached_data,
        parse_languages,
    )

    add_common_args(parser)
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup samples (default: 1)",
    )
    parser.set_defaults(quant="f32")
    args = parser.parse_args()

    # Check whisper availability
    try:
        import whisper
    except ImportError:
        print("SKIP: openai-whisper not installed. Install with: pip install openai-whisper")
        return 0

    languages = parse_languages(args)

    openai_model = OPENAI_MODEL_MAP.get(args.model, args.model.replace("whisper-", ""))

    print(f"=== OpenAI Whisper (CPU, accuracy reference) ===")
    print(f"Model: {openai_model} (from {args.model})")
    print(f"Beam: {args.beam}")
    print(f"Languages: {', '.join(languages)} ({args.samples} samples each)")

    # Load model (CPU only — MPS is broken)
    print(f"\nLoading model '{openai_model}' on CPU...")
    t_load = time.monotonic()
    model = whisper.load_model(openai_model, device="cpu")
    print(f"  Loaded in {time.monotonic() - t_load:.1f}s")

    result = BenchmarkResult(
        framework="openai_whisper",
        model=args.model,
        quant="f32",
        beam_size=args.beam,
        flash_attention=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    for lang in languages:
        audios, refs, durs = load_cached_data(lang)
        n = min(len(audios), args.samples)
        refs, durs = refs[:n], durs[:n]
        total_audio_s = sum(durs)
        whisper_lang = WHISPER_LANG_CODE.get(lang, lang.split("_")[0])

        # OpenAI whisper.transcribe accepts file paths or numpy arrays
        wav_paths = export_wav(lang)[:n]

        print(f"\n  [{lang}] {n} samples, {total_audio_s:.0f}s audio, whisper_lang={whisper_lang}")

        # Warmup
        n_warmup = min(args.warmup, n)
        try:
            for i in range(n_warmup):
                whisper.transcribe(
                    model, str(wav_paths[i]),
                    language=whisper_lang,
                    beam_size=args.beam,
                    without_timestamps=True,
                    fp16=False,
                )
        except Exception as e:
            print(f"    ERROR during warmup: {e}")
            print(f"    Skipping {lang}")
            continue

        # Timed inference
        hypotheses: list[str] = []
        total_wall = 0.0
        for i in range(n):
            t0 = time.monotonic()
            out = whisper.transcribe(
                model, str(wav_paths[i]),
                language=whisper_lang,
                beam_size=args.beam,
                without_timestamps=True,
                fp16=False,
            )
            text = out.get("text", "").strip()
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

    result.peak_rss_mb = round(get_peak_rss_mb(), 1)

    print(f"\n{'=' * 60}")
    print(f"{'Lang':<10} {'WER':>8} {'RTF':>8} {'Wall(s)':>8} {'Audio(s)':>8}")
    print("-" * 60)
    for lang, lr in result.languages.items():
        print(f"{lang:<10} {lr.wer:>7.2%} {lr.rtf:>8.3f} {lr.wall_s:>8.1f} {lr.audio_s:>8.1f}")
    print("-" * 60)
    print(f"{'Average':<10} {result.avg_wer():>7.2%} {result.avg_rtf():>8.3f}")
    print(f"Peak RSS: {result.peak_rss_mb:.0f} MB")

    output = args.output or str(auto_output_path("openai_whisper", args.model, "f32", args.beam))
    result.save(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
