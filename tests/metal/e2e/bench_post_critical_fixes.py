#!/usr/bin/env python3
"""Post-critical-fixes benchmark (C1/C2/C3).

Measures:
  1. Translation throughput (OPUS-MT En→De, beam=4, 50 sentences) — all compute types
  2. Generator throughput (TinyLlama, greedy, max_length=100) — all compute types, std + flash MHA

Methodology: best-of-3 runs, matching M12 performance sweep.
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import model_path, load_marian_tokenizer, tokenize

import ctranslate2


# ── WMT14 sentences (subset) ──
WMT14_SENTENCES = [
    "Parliament Does Not Support Amendment Recognised as Offensive",
    "Today the amendment is no longer528 in force",
    "The name of the amendment is not specified",
    "A Czech delegation iscurrently visiting the US",
    "Moody's said it would downgrade the country's credit rating",
    "The amendment was not approved by the committee",
    "The president of the commission said that the amendment was not necessary",
    "The prime minister said that the country would not accept the amendment",
    "The court ruled that the amendment was unconstitutional",
    "The government said that the amendment would not be implemented",
    "The opposition party said that the amendment was not in the interest of the people",
    "The foreign minister said that the country would not sign the treaty",
    "The defense minister said that the country would increase its military spending",
    "The finance minister said that the budget would be balanced",
    "The health minister said that the country would invest more in healthcare",
    "The education minister said that the country would reform the education system",
    "The transport minister said that the country would build new roads",
    "The energy minister said that the country would invest in renewable energy",
    "The environment minister said that the country would reduce its carbon emissions",
    "The justice minister said that the country would reform the judicial system",
    "The interior minister said that the country would increase security measures",
    "The agriculture minister said that the country would support farmers",
    "The labor minister said that the country would create more jobs",
    "The culture minister said that the country would invest in arts and culture",
    "The technology minister said that the country would promote innovation",
    "The trade minister said that the country would sign new trade agreements",
    "The economy grew by three percent last year",
    "Unemployment fell to its lowest level in a decade",
    "Inflation remained stable at two percent",
    "Exports increased by five percent compared to the previous year",
    "The central bank kept interest rates unchanged",
    "Consumer confidence rose to a record high",
    "Industrial production increased by four percent",
    "Retail sales grew by two percent",
    "Housing prices continued to rise in major cities",
    "The stock market reached a new all-time high",
    "Foreign investment increased by ten percent",
    "Tourism revenue grew by eight percent",
    "The manufacturing sector expanded for the third consecutive month",
    "Government spending increased by six percent",
    "Tax revenue exceeded expectations by two billion euros",
    "The poverty rate declined by one percentage point",
    "Wage growth outpaced inflation for the first time in five years",
    "The trade deficit narrowed significantly",
    "Research and development spending increased",
    "Productivity improved across all sectors",
    "The construction sector saw strong growth",
    "Small businesses reported improved conditions",
    "Digital transformation accelerated across industries",
    "The services sector remained the largest contributor to GDP",
]


def bench_translation():
    """Translation benchmark: OPUS-MT En→De, beam=4, 50 sentences."""
    mpath = model_path("opus-mt-en-de")
    if not os.path.isdir(mpath):
        print(f"SKIP: model not found at {mpath}")
        return

    tokenizer = load_marian_tokenizer()
    tokens = [tokenize(tokenizer, s) for s in WMT14_SENTENCES]

    compute_types = ["float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16"]

    # CPU baseline
    print("=== Translation Benchmark (OPUS-MT, beam=4, 50 sent) ===\n")
    print(f"{'Type':<18} {'Device':<8} {'tok/s':>8} {'ms':>8} {'vs CPU f32':>10}")
    print("-" * 56)

    cpu_tps = None

    for ct in ["float32"]:
        translator = ctranslate2.Translator(mpath, device="cpu", compute_type=ct,
                                            inter_threads=1, intra_threads=4)
        best_ms = float('inf')
        best_toks = 0
        for _ in range(3):
            t0 = time.monotonic()
            results = translator.translate_batch(tokens, beam_size=4, max_batch_size=32)
            elapsed = (time.monotonic() - t0) * 1000
            total_toks = sum(len(r.hypotheses[0]) for r in results)
            if elapsed < best_ms:
                best_ms = elapsed
                best_toks = total_toks
        tps = best_toks / (best_ms / 1000)
        cpu_tps = tps
        print(f"{'float32':<18} {'CPU':<8} {tps:>8.0f} {best_ms:>8.0f} {'1.00x':>10}")
        del translator

    # Metal
    for ct in compute_types:
        try:
            translator = ctranslate2.Translator(mpath, device="mps", compute_type=ct)
        except Exception as e:
            print(f"{ct:<18} {'MPS':<8} {'ERROR':>8} — {e}")
            continue

        # Warmup
        translator.translate_batch(tokens[:2], beam_size=4)

        best_ms = float('inf')
        best_toks = 0
        for _ in range(3):
            t0 = time.monotonic()
            results = translator.translate_batch(tokens, beam_size=4, max_batch_size=32)
            elapsed = (time.monotonic() - t0) * 1000
            total_toks = sum(len(r.hypotheses[0]) for r in results)
            if elapsed < best_ms:
                best_ms = elapsed
                best_toks = total_toks
        tps = best_toks / (best_ms / 1000)
        ratio = f"{tps/cpu_tps:.2f}x" if cpu_tps else "—"
        print(f"{ct:<18} {'MPS':<8} {tps:>8.0f} {best_ms:>8.0f} {ratio:>10}")
        del translator


def bench_generator():
    """Generator benchmark: TinyLlama, greedy, max_length=100."""
    gpt2_path = model_path("gpt2-ct2")
    tinyllama_path = model_path("tinyllama-ct2")

    # Try TinyLlama first, fall back to GPT-2
    gen_model = None
    model_name = None
    for name, path in [("TinyLlama", tinyllama_path), ("GPT-2", gpt2_path)]:
        if os.path.isdir(path):
            gen_model = path
            model_name = name
            break

    if not gen_model:
        print(f"\nSKIP: No generator model found")
        return

    prompts = [["The", "Ġquick", "Ġbrown", "Ġfox"]]
    max_len = 100

    compute_types = ["float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16"]

    print(f"\n\n=== Generator Benchmark ({model_name}, greedy, max_length={max_len}) ===\n")
    print(f"{'Type':<18} {'MHA':<8} {'tok/s':>8}")
    print("-" * 36)

    for ct in compute_types:
        for use_flash in [False, True]:
            mha_label = "flash" if use_flash else "std"
            try:
                gen = ctranslate2.Generator(gen_model, device="mps", compute_type=ct,
                                            flash_attention=use_flash)
            except Exception as e:
                print(f"{ct:<18} {mha_label:<8} {'ERROR':>8} — {e}")
                continue

            # Warmup
            gen.generate_batch(prompts, max_length=10, sampling_topk=1)

            best_ms = float('inf')
            best_toks = 0
            for _ in range(3):
                t0 = time.monotonic()
                results = gen.generate_batch(prompts, max_length=max_len, sampling_topk=1)
                elapsed = (time.monotonic() - t0) * 1000
                total_toks = len(results[0].sequences[0])
                if elapsed < best_ms:
                    best_ms = elapsed
                    best_toks = total_toks
            tps = best_toks / (best_ms / 1000)
            print(f"{ct:<18} {mha_label:<8} {tps:>8.1f}")
            del gen


if __name__ == "__main__":
    bench_translation()
    bench_generator()
    print("\nDone.")
