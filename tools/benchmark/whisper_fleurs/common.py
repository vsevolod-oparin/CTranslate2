#!/usr/bin/env python3
"""Shared infrastructure for FLEURS whisper benchmarks.

Handles dataset download/caching, WER computation, result schema,
WAV export, and RSS measurement.

Usage (standalone — download and cache dataset):
    python common.py --download --languages en_us,ja_jp --samples 50
    python common.py --download  # all 6 default languages, 50 samples
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LANGUAGES = ["en_us", "ja_jp", "zh_cn", "de_de", "es_419", "ar_eg"]

LANGUAGE_NAMES = {
    "en_us": "English",
    "ja_jp": "Japanese",
    "zh_cn": "Mandarin",
    "de_de": "German",
    "es_419": "Spanish",
    "ar_eg": "Arabic",
}

# FLEURS dataset uses different language codes than our short names
FLEURS_LANG_CODE = {
    "zh_cn": "cmn_hans_cn",
    "es_419": "es_419",
}

# Whisper language codes for each benchmark language (used by runners)
WHISPER_LANG_CODE = {
    "en_us": "en",
    "ja_jp": "ja",
    "zh_cn": "zh",
    "de_de": "de",
    "es_419": "es",
    "ar_eg": "ar",
}

SAMPLE_RATE = 16000

DEFAULT_SAMPLES = 50

CACHE_DIR = Path(__file__).parent / "cache"

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class LanguageResult:
    wer: float
    rtf: float
    wall_s: float
    audio_s: float
    num_samples: int


@dataclass
class BenchmarkResult:
    framework: str
    model: str
    quant: str
    beam_size: int
    flash_attention: bool = False
    languages: dict[str, LanguageResult] = field(default_factory=dict)
    peak_rss_mb: float = 0.0
    timestamp: str = ""

    def avg_wer(self) -> float:
        vals = [lr.wer for lr in self.languages.values()]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_rtf(self) -> float:
        total_wall = sum(lr.wall_s for lr in self.languages.values())
        total_audio = sum(lr.audio_s for lr in self.languages.values())
        return total_wall / total_audio if total_audio > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Results saved to {path}")

    @classmethod
    def from_json(cls, path: str | Path) -> "BenchmarkResult":
        with open(path) as f:
            d = json.load(f)
        langs = {}
        for k, v in d.pop("languages", {}).items():
            langs[k] = LanguageResult(**v)
        return cls(**d, languages=langs)


# ---------------------------------------------------------------------------
# FLEURS dataset loading and caching
# ---------------------------------------------------------------------------


def _cache_path(lang: str) -> Path:
    return CACHE_DIR / f"fleurs_{lang}.npz"


def _refs_path(lang: str) -> Path:
    return CACHE_DIR / f"fleurs_{lang}_refs.json"


def is_cached(lang: str) -> bool:
    return _cache_path(lang).exists() and _refs_path(lang).exists()


def download_fleurs(
    languages: list[str] | None = None,
    num_samples: int = DEFAULT_SAMPLES,
    force: bool = False,
) -> dict[str, int]:
    """Download FLEURS test split via streaming and cache as .npz + .json.

    Returns dict of {lang: num_samples_cached}.
    """
    from datasets import load_dataset

    if languages is None:
        languages = DEFAULT_LANGUAGES

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for lang in languages:
        if not force and is_cached(lang):
            # Load existing to get count
            refs = json.loads(_refs_path(lang).read_text())
            counts[lang] = len(refs)
            print(f"  {LANGUAGE_NAMES.get(lang, lang):>10}: {counts[lang]} samples (cached)")
            continue

        print(f"  {LANGUAGE_NAMES.get(lang, lang):>10}: downloading {num_samples} samples...", end="", flush=True)
        t0 = time.monotonic()

        fleurs_code = FLEURS_LANG_CODE.get(lang, lang)
        ds = load_dataset("google/fleurs", fleurs_code, split="test", streaming=True, trust_remote_code=True)

        audios: list[np.ndarray] = []
        refs: list[str] = []
        durations: list[float] = []

        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            # Resample if needed
            if sr != SAMPLE_RATE:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            audios.append(audio_array)
            refs.append(sample["transcription"])
            durations.append(len(audio_array) / SAMPLE_RATE)

        # Save audio arrays as .npz (variable-length arrays stored as object array)
        np.savez(
            _cache_path(lang),
            **{f"audio_{i}": a for i, a in enumerate(audios)},
            durations=np.array(durations, dtype=np.float32),
        )

        # Save references as JSON
        _refs_path(lang).write_text(json.dumps(refs, ensure_ascii=False, indent=2))

        elapsed = time.monotonic() - t0
        total_dur = sum(durations)
        counts[lang] = len(audios)
        print(f" {counts[lang]} samples, {total_dur:.1f}s audio, {elapsed:.1f}s download")

    return counts


def load_cached_data(lang: str) -> tuple[list[np.ndarray], list[str], list[float]]:
    """Load cached FLEURS data for a language.

    Returns (audios, references, durations).
    """
    if not is_cached(lang):
        raise FileNotFoundError(
            f"No cached data for '{lang}'. Run: python common.py --download --languages {lang}"
        )

    data = np.load(_cache_path(lang), allow_pickle=False)
    durations = data["durations"].tolist()
    n = len(durations)
    audios = [data[f"audio_{i}"] for i in range(n)]

    refs = json.loads(_refs_path(lang).read_text())
    return audios, refs, durations


# ---------------------------------------------------------------------------
# WAV export (for frameworks that need file paths)
# ---------------------------------------------------------------------------

WAV_CACHE_DIR = CACHE_DIR / "wav"


def export_wav(lang: str) -> list[Path]:
    """Export cached audio to WAV files. Returns list of WAV file paths."""
    import soundfile as sf

    audios, _, _ = load_cached_data(lang)
    wav_dir = WAV_CACHE_DIR / lang
    wav_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i, audio in enumerate(audios):
        wav_path = wav_dir / f"{i:04d}.wav"
        if not wav_path.exists():
            sf.write(str(wav_path), audio, SAMPLE_RATE)
        paths.append(wav_path)
    return paths


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------

# Lazy-loaded normalizer
_normalizer = None


def _get_normalizer():
    """Get Whisper's text normalizer (handles CJK, punctuation, etc.)."""
    global _normalizer
    if _normalizer is None:
        try:
            from whisper_normalizer.basic import BasicTextNormalizer
            _normalizer = BasicTextNormalizer()
        except ImportError:
            try:
                from whisper.normalizers import BasicTextNormalizer
                _normalizer = BasicTextNormalizer()
            except ImportError:
                # Fallback: simple lowercase + strip
                print("  WARNING: No whisper normalizer found. Using simple lowercase normalization.")
                _normalizer = lambda x: x.lower().strip()  # noqa: E731
    return _normalizer


def compute_wer(hypotheses: list[str], references: list[str], language: str = "en") -> float:
    """Compute Word Error Rate with Whisper text normalization.

    For CJK languages (ja, zh), uses Character Error Rate (CER) since
    word boundaries are not well-defined.
    """
    from jiwer import wer as jiwer_wer, cer as jiwer_cer

    normalizer = _get_normalizer()

    norm_hyps = [normalizer(h) for h in hypotheses]
    norm_refs = [normalizer(r) for r in references]

    # Filter out empty references after normalization
    pairs = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
    if not pairs:
        return 0.0
    norm_hyps, norm_refs = zip(*pairs)

    # Use CER for CJK languages
    lang_prefix = language.split("_")[0]
    if lang_prefix in ("ja", "zh"):
        return jiwer_cer(list(norm_refs), list(norm_hyps))
    return jiwer_wer(list(norm_refs), list(norm_hyps))


# ---------------------------------------------------------------------------
# RSS measurement
# ---------------------------------------------------------------------------


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard benchmark CLI arguments."""
    parser.add_argument(
        "--model", default="whisper-large-v3-turbo",
        help="Model name (default: whisper-large-v3-turbo)",
    )
    parser.add_argument(
        "--quant", default="float16",
        help="Quantization / compute type (default: float16)",
    )
    parser.add_argument(
        "--beam", type=int, default=1,
        help="Beam size (default: 1 = greedy)",
    )
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated language codes (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Number of samples per language (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--data-dir", default=str(CACHE_DIR),
        help=f"FLEURS cache directory (default: {CACHE_DIR})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (default: auto-generated in results/)",
    )


def parse_languages(args: argparse.Namespace) -> list[str]:
    """Parse comma-separated language codes from args."""
    return [l.strip() for l in args.languages.split(",") if l.strip()]


def auto_output_path(framework: str, model: str, quant: str, beam: int, flash: bool = False) -> Path:
    """Generate output JSON path from config."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    flash_suffix = "_flash" if flash else ""
    return results_dir / f"{framework}_{model}_{quant}_b{beam}{flash_suffix}.json"


# ---------------------------------------------------------------------------
# Standalone: download dataset
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="FLEURS dataset manager for whisper benchmarks")
    parser.add_argument("--download", action="store_true", help="Download and cache FLEURS data")
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated language codes (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_SAMPLES,
        help=f"Samples per language (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--export-wav", action="store_true", help="Export cached audio to WAV files")
    parser.add_argument("--info", action="store_true", help="Show cache info")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    if args.download:
        print(f"Downloading FLEURS test split ({args.samples} samples per language)...")
        counts = download_fleurs(languages, num_samples=args.samples, force=args.force)
        total_samples = sum(counts.values())
        print(f"\nTotal: {total_samples} samples across {len(counts)} languages")
        print(f"Cache: {CACHE_DIR}")
        return 0

    if args.export_wav:
        print("Exporting cached audio to WAV...")
        for lang in languages:
            if not is_cached(lang):
                print(f"  {lang}: not cached, skipping")
                continue
            paths = export_wav(lang)
            print(f"  {lang}: {len(paths)} WAV files → {WAV_CACHE_DIR / lang}")
        return 0

    if args.info:
        print(f"Cache directory: {CACHE_DIR}")
        for lang in languages:
            if is_cached(lang):
                audios, refs, durs = load_cached_data(lang)
                total_dur = sum(durs)
                print(f"  {LANGUAGE_NAMES.get(lang, lang):>10}: {len(audios)} samples, {total_dur:.1f}s audio")
            else:
                print(f"  {LANGUAGE_NAMES.get(lang, lang):>10}: not cached")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
