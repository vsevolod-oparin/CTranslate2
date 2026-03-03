# Milestone 10.2 — Seq2seq (Transformer) End-to-End

**Date:** 2026-03-03
**Status:** ✅ DONE (correctness); speed deferred to M11

---

## Summary

M10.2 validates that the Metal backend produces identical translation output to CPU across a 100-sentence WMT14 en-de test set, using both greedy decoding and beam search (beam_size=4).

**Correctness: perfect.** BLEU difference = 0.00 for both greedy and beam=4. All 100 sentences produce byte-identical output on CPU and Metal.

**Speed: not yet competitive.** The per-op commit model (each Metal op submits its own command buffer) adds ~0.4ms overhead per submission. For a small model like opus-mt-en-de with many short decoder steps, this overhead completely dominates. Speed optimization is explicitly deferred to M11 (command buffer batching).

---

## Test Results (Apple M4, conda env `ct2`, Python 3.14)

### Correctness (hard pass/fail)

| Test | Result |
|------|--------|
| BLEU diff <= 0.5 (greedy) | **PASS** — diff=0.00 |
| Exact match rate (greedy) | **PASS** — 100/100 |
| BLEU diff <= 0.5 (beam=4) | **PASS** — diff=0.00 |
| Exact match rate (beam=4) | **PASS** — 100/100 |

### BLEU Scores

| Mode | CPU BLEU | Metal BLEU | Difference |
|------|----------|------------|------------|
| Greedy (beam=1) | 26.01 | 26.01 | 0.00 |
| Beam search (beam=4) | 26.73 | 26.73 | 0.00 |

### Speed Benchmarks (informational)

| Mode | CPU | Metal | Ratio |
|------|-----|-------|-------|
| Greedy, batch=1 | 10.43s (9.6 sent/s) | 134.88s (0.7 sent/s) | 0.08x |
| Beam=4, batch=1 | 38.17s (2.6 sent/s) | 352.70s (0.3 sent/s) | 0.11x |

---

## Speed Analysis

The Metal backend is ~10-13x slower than CPU for sentence-by-sentence translation on opus-mt-en-de. This is consistent with the benchmark data from earlier milestones:

- **Per-op commit overhead:** ~0.4ms fixed cost per `commit_and_wait()`. A single decoder step involves ~15-20 ops (LayerNorm, GEMM×4, SDPA, Add, etc.), each triggering a command buffer submission.
- **Small matrices:** opus-mt-en-de has d_model=512, d_ff=2048. At these sizes, the Metal GPU advantage over CPU is minimal (crossover for most ops is at ~200K-1M elements).
- **Autoregressive decode:** Each output token requires a full forward pass with sq=1 (single-token decode). The GPU never gets large enough work to amortize the overhead.

**Expected fix (M11):** Command buffer batching — collect all ops for a decoder step into a single command buffer, reducing submissions from ~15-20 per step to 1. Based on M0.3 findings, this should recover ~48% of GPU throughput wasted on per-op submission.

For larger models (deeper, wider), the GPU compute advantage will dominate the fixed overhead and Metal should be competitive or faster.

---

## Dataset

- **Source:** WMT14 English-German via sacrebleu (`sacrebleu.get_source_file("wmt14", "en-de")`)
- **Size:** First 100 sentences (of 2737 total)
- **References:** Official WMT14 reference translations
- **Model:** opus-mt-en-de (Helsinki-NLP), converted to CT2 format

---

## Files

| File | Description |
|------|-------------|
| `tests/metal/e2e/test_seq2seq_e2e.py` | M10.2 validation script (4 hard tests + 2 info benchmarks) |
| `APPLE_M4_METAL_PLAN.md` | Updated M10.2 status |

No C++ or build changes required — this milestone is purely validation.

---

## Pass Criteria Assessment

| Criterion | Plan Target | Result | Status |
|-----------|-------------|--------|--------|
| BLEU within 0.5 of CPU (greedy) | <= 0.5 | 0.00 | ✅ |
| BLEU within 0.5 of CPU (beam=4) | <= 0.5 | 0.00 | ✅ |
| Speed >= 1.5x CPU (batch=1) | >= 1.5x | 0.08x | Deferred to M11 |

The correctness criteria are fully met. The speed criterion requires M11's command buffer batching optimization and will be revisited after that milestone.

---

## Running the Test

```bash
export CT2_TEST_DATA=/path/to/data   # must contain opus-mt-en-de/
pip install sacrebleu                 # if not already installed
python tests/metal/e2e/test_seq2seq_e2e.py
```

Runtime: ~10 minutes (dominated by Metal translation of 100 sentences × 2 beam sizes × 2 timed runs).
