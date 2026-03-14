#!/usr/bin/env python3
"""Runner: mlx-whisper (Apple MLX framework, native Metal).

Usage:
    python runner_mlx_whisper.py --model whisper-large-v3-turbo --quant f16 --beam 1

Note: mlx-whisper uses HuggingFace model repos from mlx-community.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Map our model names to mlx-community HuggingFace repos
MLX_MODEL_MAP = {
    "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "whisper-large-v3": "mlx-community/whisper-large-v3",
    "whisper-base": "mlx-community/whisper-base",
}

# Quantized variants (if available on HuggingFace)
MLX_QUANT_MAP = {
    "whisper-large-v3-turbo": {
        "f16": "mlx-community/whisper-large-v3-turbo",
        "int4": "mlx-community/whisper-large-v3-turbo-4bit",
        "int8": "mlx-community/whisper-large-v3-turbo-8bit",
    },
    "whisper-large-v3": {
        "f16": "mlx-community/whisper-large-v3",
        "int4": "mlx-community/whisper-large-v3-mlx-4bit",
        "int8": "mlx-community/whisper-large-v3-mlx-8bit",
    },
    "whisper-base": {
        "f16": "mlx-community/whisper-base",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="mlx-whisper benchmark")

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
        "--warmup", type=int, default=3,
        help="Number of warmup samples (default: 3)",
    )
    parser.set_defaults(quant="f16")
    args = parser.parse_args()

    # Check mlx_whisper availability
    try:
        import mlx_whisper
    except ImportError:
        print("SKIP: mlx-whisper not installed. Install with: pip install mlx-whisper")
        return 0

    languages = parse_languages(args)

    # Resolve HuggingFace repo
    quant_map = MLX_QUANT_MAP.get(args.model, {})
    hf_repo = quant_map.get(args.quant)
    if hf_repo is None:
        available = list(quant_map.keys()) if quant_map else ["(none)"]
        print(f"ERROR: No mlx model for {args.model} quant={args.quant}. Available: {available}")
        return 1

    print(f"=== mlx-whisper (Apple MLX, Metal) ===")
    print(f"Model: {args.model}, Quant: {args.quant}, Beam: {args.beam}")
    print(f"HF repo: {hf_repo}")
    print(f"Languages: {', '.join(languages)} ({args.samples} samples each)")

    result = BenchmarkResult(
        framework="mlx_whisper",
        model=args.model,
        quant=args.quant,
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

        # mlx_whisper.transcribe accepts file paths
        wav_paths = export_wav(lang)[:n]

        print(f"\n  [{lang}] {n} samples, {total_audio_s:.0f}s audio, whisper_lang={whisper_lang}")

        # Build decode_options (language, beam_size, etc.)
        decode_opts: dict = {"language": whisper_lang, "without_timestamps": True}
        # mlx-whisper doesn't support beam search yet — force greedy
        effective_beam = args.beam
        if args.beam > 1:
            if lang == languages[0]:
                print(f"    NOTE: mlx-whisper does not support beam search, using greedy")
            effective_beam = 1

        # Warmup
        n_warmup = min(args.warmup, n)
        try:
            for i in range(n_warmup):
                mlx_whisper.transcribe(
                    str(wav_paths[i]),
                    path_or_hf_repo=hf_repo,
                    **decode_opts,
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
            out = mlx_whisper.transcribe(
                str(wav_paths[i]),
                path_or_hf_repo=hf_repo,
                **decode_opts,
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

    output = args.output or str(auto_output_path("mlx_whisper", args.model, args.quant, args.beam))
    result.save(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
