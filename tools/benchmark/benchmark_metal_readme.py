#!/usr/bin/env python3
"""Metal (MPS) benchmark for CTranslate2 README — both models, all types.

Translates En->De newstest2014 (WMT14, 2737 sentences) with:
  1. OpenNMT-py WMT14 model (base Transformer, sentencepiece tokenization)
  2. OPUS-MT model (Helsinki-NLP/opus-mt-en-de, MarianTokenizer)

Each config runs in a SUBPROCESS to isolate Metal GPU memory.

Usage:
  conda run -n ct2 python tools/benchmark/benchmark_metal_readme.py

Expects models in ../data/ relative to repo root:
  opennmt-py-wmt14, opennmt-py-wmt14-f16, opennmt-py-wmt14-int8
  opus-mt-en-de, opus-mt-en-de-f16, opus-mt-en-de-int8
"""
import json
import os
import subprocess
import sys
import textwrap
import time

# ---------------------------------------------------------------------------
# Worker script — runs in subprocess for memory isolation
# ---------------------------------------------------------------------------
_WORKER = textwrap.dedent(r'''
import gc, json, os, resource, sys, time

def get_max_rss_mb():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else u.ru_maxrss / 1024

config = json.loads(sys.argv[1])
model_kind = config["model_kind"]  # "opennmt" or "opus"

import ctranslate2
import sacrebleu

# Load tokenizer
if model_kind == "opus":
    from transformers import MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    def tokenize(text):
        return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    def detokenize(tokens):
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
else:
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(config["sp_model"])
    def tokenize(text):
        return sp.encode(text, out_type=str)
    def detokenize(tokens):
        return sp.decode(tokens)

# Load test set
sf = sacrebleu.get_source_file("wmt14", langpair="en-de")
rf = sacrebleu.get_reference_files("wmt14", langpair="en-de")[0]
with open(sf) as f:
    source_sentences = [l.strip() for l in f]
max_sent = config.get("max_sentences")
if max_sent:
    source_sentences = source_sentences[:max_sent]
source_tokens = [tokenize(s) for s in source_sentences]

device = config["device"]
translator = ctranslate2.Translator(
    config["model_path"],
    device=device,
    compute_type=config["compute_type"],
    intra_threads=config.get("intra_threads", 1),
    flash_attention=config.get("flash_attention", False),
)

# Warmup
translator.translate_batch([[""]], beam_size=1)

best_time = None
num_target_tokens = 0
bleu_score = 0.0
batch_size = 32

for sample in range(config["num_samples"]):
    gc.collect()
    if device == "mps":
        ctranslate2.clear_device_cache("mps")

    t0 = time.monotonic()
    # Pass all sentences at once; translate_batch sorts by length internally,
    # which minimizes padding within each sub-batch.  Chunking in sequential
    # order causes severe f16 quality degradation (padding-sensitive).
    all_results = translator.translate_batch(
        source_tokens, beam_size=config["beam_size"], max_batch_size=batch_size)
    elapsed = time.monotonic() - t0

    if best_time is None or elapsed < best_time:
        best_time = elapsed
        hypotheses = [r.hypotheses[0] for r in all_results]
        num_target_tokens = sum(len(h) for h in hypotheses)
        decoded = [detokenize(tokens) for tokens in hypotheses]
        bleu = sacrebleu.corpus_bleu(
            decoded, [open(rf).readlines()], force=True)
        bleu_score = bleu.score

max_rss = get_max_rss_mb()
del translator
gc.collect()
if device == "mps":
    ctranslate2.clear_device_cache("mps")

result = {
    "time": best_time,
    "tokens_per_sec": num_target_tokens / best_time,
    "num_tokens": num_target_tokens,
    "max_rss_mb": max_rss,
    "bleu": bleu_score,
}
print("RESULT:" + json.dumps(result))
''')

# ---------------------------------------------------------------------------
# Transformers MPS benchmark worker
# ---------------------------------------------------------------------------
_TRANSFORMERS_WORKER = textwrap.dedent(r'''
import gc, json, os, resource, sys, time
import torch

def get_max_rss_mb():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else u.ru_maxrss / 1024

config = json.loads(sys.argv[1])

from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

sf = sacrebleu.get_source_file("wmt14", langpair="en-de")
rf = sacrebleu.get_reference_files("wmt14", langpair="en-de")[0]
with open(sf) as f:
    source_sentences = [l.strip() for l in f]
max_sent = config.get("max_sentences")
if max_sent:
    source_sentences = source_sentences[:max_sent]

device = config["device"]
dtype = {"float32": torch.float32, "float16": torch.float16}.get(
    config.get("dtype", "float32"), torch.float32)

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de",
                                       torch_dtype=dtype).to(device)
model.eval()

# Warmup
with torch.no_grad():
    inp = tokenizer(["Hello"], return_tensors="pt", padding=True).to(device)
    model.generate(**inp, num_beams=1, max_new_tokens=5)

best_time = None
num_target_tokens = 0
bleu_score = 0.0
chunk_size = 32
beam_size = config["beam_size"]

for sample in range(config["num_samples"]):
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    all_decoded = []
    all_token_count = 0
    t0 = time.monotonic()

    with torch.no_grad():
        for ci in range(0, len(source_sentences), chunk_size):
            chunk = source_sentences[ci:ci + chunk_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                               truncation=True).to(device)
            outputs = model.generate(**inputs, num_beams=beam_size,
                                     max_new_tokens=512)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_decoded.extend(decoded)
            # Count tokens (space-separated approximation matching CT2 method)
            for d in decoded:
                all_token_count += len(tokenizer.encode(d))
            if device == "mps":
                gc.collect()
                torch.mps.empty_cache()

    elapsed = time.monotonic() - t0

    if best_time is None or elapsed < best_time:
        best_time = elapsed
        num_target_tokens = all_token_count
        bleu = sacrebleu.corpus_bleu(
            all_decoded, [open(rf).readlines()], force=True)
        bleu_score = bleu.score

max_rss = get_max_rss_mb()
del model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()

result = {
    "time": best_time,
    "tokens_per_sec": num_target_tokens / best_time,
    "num_tokens": num_target_tokens,
    "max_rss_mb": max_rss,
    "bleu": bleu_score,
}
print("RESULT:" + json.dumps(result))
''')


def run_worker(script, config, label, timeout=1800):
    """Run a benchmark worker in a subprocess."""
    print(f"  {label}...", end="", flush=True)
    config_json = json.dumps(config)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, config_json],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f" TIMEOUT ({timeout}s)")
        return None

    if result.returncode != 0:
        print(f" FAILED (exit {result.returncode})")
        for line in result.stderr.strip().split("\n")[-3:]:
            print(f"    {line}")
        return None

    for line in result.stdout.split("\n"):
        if line.startswith("RESULT:"):
            r = json.loads(line[7:])
            print(f" {r['tokens_per_sec']:.1f} tok/s, "
                  f"RSS={r['max_rss_mb']:.0f}MB, BLEU={r['bleu']:.2f}")
            return r

    print(f" FAILED: no result")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Metal README benchmark")
    parser.add_argument("--precheck", action="store_true",
                        help="Quick sanity check: 50 sentences, 1 sample")
    args = parser.parse_args()

    data_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    sp_model = os.path.join(data_dir, "sentencepiece.model")

    if args.precheck:
        NUM_SAMPLES = 1
        BEAM_SIZE = 4
        CPU_THREADS = 4
        MAX_SENTENCES = 50
    else:
        NUM_SAMPLES = 2
        BEAM_SIZE = 4
        CPU_THREADS = 4
        MAX_SENTENCES = None  # all sentences

    # Common config extras
    extra = {}
    if MAX_SENTENCES:
        extra["max_sentences"] = MAX_SENTENCES

    # Define all benchmark configurations
    benchmarks = []

    # === OpenNMT-py WMT14 model ===
    opennmt_models = {
        "float32": os.path.join(data_dir, "opennmt-py-wmt14"),
        # f16/bf16: use f32 model with runtime cast (same as OPUS-MT pattern)
        "float16": os.path.join(data_dir, "opennmt-py-wmt14"),
        "bfloat16": os.path.join(data_dir, "opennmt-py-wmt14"),
        "int8":    os.path.join(data_dir, "opennmt-py-wmt14-int8"),
        "int8_float16": os.path.join(data_dir, "opennmt-py-wmt14-int8"),
    }
    for ct, path in opennmt_models.items():
        if os.path.isdir(path):
            # CPU baseline (float32 only)
            if ct == "float32":
                benchmarks.append(("OpenNMT-py WMT14", "cpu", ct, {
                    "model_path": path, "device": "cpu", "compute_type": ct,
                    "model_kind": "opennmt", "sp_model": sp_model,
                    "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES,
                    "intra_threads": CPU_THREADS, **extra,
                }))
            # MPS
            benchmarks.append(("OpenNMT-py WMT14", "mps", ct, {
                "model_path": path, "device": "mps", "compute_type": ct,
                "model_kind": "opennmt", "sp_model": sp_model,
                "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES, **extra,
            }))

    # === OPUS-MT model ===
    opus_models = {
        "float32": os.path.join(data_dir, "opus-mt-en-de"),
        # f16/bf16: use f32 model with runtime cast (pre-converted f16 model has weight issues)
        "float16": os.path.join(data_dir, "opus-mt-en-de"),
        "bfloat16": os.path.join(data_dir, "opus-mt-en-de"),
        "int8":    os.path.join(data_dir, "opus-mt-en-de-int8"),
        "int8_float16": os.path.join(data_dir, "opus-mt-en-de-int8"),
    }
    for ct, path in opus_models.items():
        if os.path.isdir(path):
            if ct == "float32":
                benchmarks.append(("OPUS-MT", "cpu", ct, {
                    "model_path": path, "device": "cpu", "compute_type": ct,
                    "model_kind": "opus",
                    "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES,
                    "intra_threads": CPU_THREADS, **extra,
                }))
            benchmarks.append(("OPUS-MT", "mps", ct, {
                "model_path": path, "device": "mps", "compute_type": ct,
                "model_kind": "opus",
                "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES, **extra,
            }))

    # === Flash attention variants (MPS only) ===
    # OpenNMT-py: f32+flash only (f16+flash has known degenerate output issue)
    opennmt_f32 = opennmt_models.get("float32")
    if opennmt_f32 and os.path.isdir(opennmt_f32):
        benchmarks.append(("OpenNMT-py WMT14 flash", "mps", "float32", {
            "model_path": opennmt_f32, "device": "mps", "compute_type": "float32",
            "model_kind": "opennmt", "sp_model": sp_model,
            "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES,
            "flash_attention": True, **extra,
        }))

    # OPUS-MT: both f32+flash and f16+flash
    opus_base = opus_models.get("float32")
    opus_f16 = opus_models.get("float16")
    if opus_base and os.path.isdir(opus_base):
        benchmarks.append(("OPUS-MT flash", "mps", "float32", {
            "model_path": opus_base, "device": "mps", "compute_type": "float32",
            "model_kind": "opus",
            "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES,
            "flash_attention": True, **extra,
        }))
    if opus_f16 and os.path.isdir(opus_f16):
        benchmarks.append(("OPUS-MT flash", "mps", "float16", {
            "model_path": opus_f16, "device": "mps", "compute_type": "float16",
            "model_kind": "opus",
            "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES,
            "flash_attention": True, **extra,
        }))

    # === Transformers (PyTorch MPS) — OPUS-MT model ===
    transformers_configs = []
    for dtype_name, device in [("float32", "cpu"), ("float32", "mps"),
                                ("float16", "mps")]:
        transformers_configs.append(("Transformers OPUS-MT", device, dtype_name, {
            "device": device, "dtype": dtype_name,
            "beam_size": BEAM_SIZE, "num_samples": NUM_SAMPLES, **extra,
        }))

    mode = "PRECHECK (50 sentences)" if args.precheck else "FULL"
    n_sent = MAX_SENTENCES if MAX_SENTENCES else 2737
    print(f"=== Metal README Benchmark [{mode}] ===")
    print(f"Test set: wmt14 en-de ({n_sent} sentences)")
    print(f"Beam size: {BEAM_SIZE}, Samples: {NUM_SAMPLES}")
    print(f"CPU threads: {CPU_THREADS}")
    print()

    # Run CTranslate2 benchmarks
    results = {}
    print("--- CTranslate2 benchmarks ---")
    for model_name, device, ct, config in benchmarks:
        label = f"{model_name} / {device} / {ct}"
        r = run_worker(_WORKER, config, label)
        results[(model_name, device, ct, "ct2")] = r

    # Run Transformers benchmarks
    print()
    print("--- Transformers (PyTorch) benchmarks ---")
    for model_name, device, dtype_name, config in transformers_configs:
        label = f"{model_name} / {device} / {dtype_name}"
        r = run_worker(_TRANSFORMERS_WORKER, config, label, timeout=3600)
        results[(model_name, device, dtype_name, "transformers")] = r

    # Print summary tables
    print()
    print("=" * 90)
    print("RESULTS — MPS (Apple Metal) section for README")
    print("=" * 90)
    print()

    # Format as README markdown
    print("#### MPS (Apple Metal)")
    print()
    print("| | Tokens per second | Max. memory | BLEU |")
    print("| --- | --- | --- | --- |")

    # OpenNMT-py WMT14
    print("| **OpenNMT-py WMT14 model** | | | |")
    r = results.get(("OpenNMT-py WMT14", "cpu", "float32", "ct2"))
    if r:
        print(f"| CTranslate2 - CPU float32 ({CPU_THREADS} threads) | "
              f"{r['tokens_per_sec']:.1f} | {r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")
    for ct in ["float32", "float16", "bfloat16", "int8", "int8_float16"]:
        r = results.get(("OpenNMT-py WMT14", "mps", ct, "ct2"))
        if r:
            label = f"CTranslate2 - MPS {ct}"
            print(f"| {label} | {r['tokens_per_sec']:.1f} | "
                  f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")
    r = results.get(("OpenNMT-py WMT14 flash", "mps", "float32", "ct2"))
    if r:
        print(f"| CTranslate2 - MPS float32 (flash) | {r['tokens_per_sec']:.1f} | "
              f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")

    # OPUS-MT
    print("| **OPUS-MT model** | | | |")

    # Transformers CPU
    r = results.get(("Transformers OPUS-MT", "cpu", "float32", "transformers"))
    if r:
        print(f"| Transformers (PyTorch) - CPU float32 ({CPU_THREADS} threads) | "
              f"{r['tokens_per_sec']:.1f} | {r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")

    # Transformers MPS
    for dtype_name in ["float32", "float16"]:
        r = results.get(("Transformers OPUS-MT", "mps", dtype_name, "transformers"))
        if r:
            print(f"| Transformers (PyTorch) - MPS {dtype_name} | "
                  f"{r['tokens_per_sec']:.1f} | {r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")

    # CT2 CPU
    r = results.get(("OPUS-MT", "cpu", "float32", "ct2"))
    if r:
        print(f"| CTranslate2 - CPU float32 ({CPU_THREADS} threads) | "
              f"{r['tokens_per_sec']:.1f} | {r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")

    # CT2 MPS
    for ct in ["float32", "float16", "bfloat16", "int8", "int8_float16"]:
        r = results.get(("OPUS-MT", "mps", ct, "ct2"))
        if r:
            label = f"CTranslate2 - MPS {ct}"
            print(f"| {label} | {r['tokens_per_sec']:.1f} | "
                  f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")
    for ct in ["float32", "float16"]:
        r = results.get(("OPUS-MT flash", "mps", ct, "ct2"))
        if r:
            print(f"| CTranslate2 - MPS {ct} (flash) | {r['tokens_per_sec']:.1f} | "
                  f"{r['max_rss_mb']:.0f}MB | {r['bleu']:.2f} |")

    print()
    print(f"Executed on Apple M4 with Metal Performance Shaders. "
          f"Beam size {BEAM_SIZE}, best of {NUM_SAMPLES} runs. "
          f"CPU baselines use {CPU_THREADS} threads.")

    # Dump raw results as JSON for report
    raw = {}
    for k, v in results.items():
        if v:
            raw[f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"] = v
    print()
    print("RAW_JSON:" + json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
