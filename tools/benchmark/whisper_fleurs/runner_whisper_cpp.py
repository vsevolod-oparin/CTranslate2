#!/usr/bin/env python3
"""Runner: whisper.cpp via pywhispercpp (Metal GPU).

Usage:
    python runner_whisper_cpp.py --model whisper-large-v3-turbo --quant F16 --beam 1
    python runner_whisper_cpp.py --model whisper-large-v3-turbo --quant Q5_0 --beam 5

Quant options: F16, Q8_0, Q5_0, Q4_0
Models are auto-downloaded from HuggingFace ggerganov/whisper.cpp.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Mapping from our model names to GGML model file names on HuggingFace
GGML_MODEL_MAP = {
    "whisper-large-v3-turbo": "ggml-large-v3-turbo",
    "whisper-large-v3": "ggml-large-v3",
    "whisper-base": "ggml-base",
}

# Quant suffix mapping
QUANT_SUFFIX = {
    "F16": "",        # base model is F16
    "Q8_0": "-q8_0",
    "Q5_0": "-q5_0",
    "Q4_0": "-q4_0",
}


def download_ggml_model(model_name: str, quant: str, cache_dir: Path) -> Path:
    """Download GGML model from HuggingFace if not cached. Returns path to .bin file."""
    from huggingface_hub import hf_hub_download

    base = GGML_MODEL_MAP.get(model_name)
    if base is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(GGML_MODEL_MAP.keys())}")

    suffix = QUANT_SUFFIX.get(quant)
    if suffix is None:
        raise ValueError(f"Unknown quant: {quant}. Available: {list(QUANT_SUFFIX.keys())}")

    filename = f"{base}{suffix}.bin"
    models_dir = cache_dir / "ggml_models"
    local_path = models_dir / filename

    if local_path.exists():
        print(f"  GGML model cached: {local_path}")
        return local_path

    print(f"  Downloading {filename} from ggerganov/whisper.cpp...")
    downloaded = hf_hub_download(
        repo_id="ggerganov/whisper.cpp",
        filename=filename,
        local_dir=str(models_dir),
    )
    return Path(downloaded)


def main() -> int:
    parser = argparse.ArgumentParser(description="whisper.cpp benchmark via pywhispercpp")

    sys.path.insert(0, str(Path(__file__).parent))
    from common import (
        CACHE_DIR,
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
    # Override default quant for whisper.cpp
    parser.set_defaults(quant="F16")
    args = parser.parse_args()

    # Check pywhispercpp availability
    try:
        from pywhispercpp.model import Model as WhisperCppModel
    except ImportError:
        print("SKIP: pywhispercpp not installed. Install with: pip install pywhispercpp")
        return 0

    languages = parse_languages(args)

    print(f"=== whisper.cpp (pywhispercpp, Metal) ===")
    print(f"Model: {args.model}, Quant: {args.quant}, Beam: {args.beam}")
    print(f"Languages: {', '.join(languages)} ({args.samples} samples each)")

    # Download/locate GGML model
    model_path = download_ggml_model(args.model, args.quant, CACHE_DIR)
    print(f"  Model path: {model_path} ({model_path.stat().st_size / 1e9:.2f} GB)")

    # Load model — params are set at construction time
    sampling_strategy = 0 if args.beam <= 1 else 1  # 0=GREEDY, 1=BEAM_SEARCH
    model_kwargs = {
        "n_threads": os.cpu_count() or 4,
    }
    if args.beam > 1:
        model_kwargs["beam_search"] = {"beam_size": args.beam, "patience": -1.0}

    print(f"\nLoading whisper.cpp model (strategy={'beam' if sampling_strategy else 'greedy'})...")
    t_load = time.monotonic()
    model = WhisperCppModel(
        str(model_path),
        params_sampling_strategy=sampling_strategy,
        redirect_whispercpp_logs_to=None,  # suppress verbose logs
        **model_kwargs,
    )
    print(f"  Loaded in {time.monotonic() - t_load:.1f}s")

    result = BenchmarkResult(
        framework="whisper_cpp",
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

        # Export WAV files (whisper.cpp needs file paths)
        wav_paths = export_wav(lang)[:n]

        print(f"\n  [{lang}] {n} samples, {total_audio_s:.0f}s audio, whisper_lang={whisper_lang}")

        # Set language for this batch
        model.params["language"] = whisper_lang

        # Warmup
        n_warmup = min(args.warmup, n)
        try:
            for i in range(n_warmup):
                model.transcribe(str(wav_paths[i]))
        except Exception as e:
            print(f"    ERROR during warmup: {e}")
            print(f"    Skipping {lang}")
            continue

        # Timed inference
        hypotheses: list[str] = []
        total_wall = 0.0
        for i in range(n):
            t0 = time.monotonic()
            segments = model.transcribe(str(wav_paths[i]))
            text = " ".join(seg.text.strip() for seg in segments)
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

    output = args.output or str(auto_output_path("whisper_cpp", args.model, args.quant, args.beam))
    result.save(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
