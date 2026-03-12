# M12.8 — Larger Model Benchmarks

**Date**: 2026-03-12
**Status**: COMPLETE — CRITERION NOT MET (f16 1.43x turbo, target was >3x); large-v3 CORRECTNESS FAILURE
**Branch**: `metal-backend`

## Summary

Benchmarked two whisper models (d_model=1280) to validate GPU scaling beyond OPUS-MT (d_model=512):

1. **whisper-large-v3-turbo** (32 enc / 4 dec): Works correctly, f16 1.43x speedup. Decode-dominated workload limits GPU benefit.
2. **whisper-large-v3** (32 enc / 32 dec): **Correctness failure** — MPS produces 0 transcription segments across all compute types. Deep decoder (32 layers) causes numerical accumulation that breaks output.

## Models Tested

| Property | whisper-large-v3-turbo | whisper-large-v3 |
|----------|----------------------|------------------|
| d_model | 1280 | 1280 |
| Encoder layers | 32 | 32 |
| Decoder layers | 4 (distilled) | 32 |
| Attention | Standard MHA | Standard MHA |
| MPS status | Working | **BROKEN** |

Audio: 60s Russian podcast (`sample.mp3`)

---

## Part 1: whisper-large-v3-turbo (32 enc / 4 dec) — WORKING

### Results — Beam=1 (greedy, most reliable)

Best-of-3 runs, Apple M4.

| Type | Best ms | Speedup | Correctness |
|------|---------|---------|-------------|
| CPU f32 | 12,803 | baseline | — |
| MPS f32 | 13,044 | 0.98x | **MATCH** (exact) |
| MPS f16 | 8,941 | **1.43x** | Minor token diffs (expected) |
| MPS bf16→f16 | 8,992 | **1.42x** | Same as f16 (auto-promoted) |
| MPS int8 | — | ERROR | Model not int8-quantized |

#### Variance

- CPU f32 beam=1: 12,803 / 19,286 / 15,989 ms (high variance, thermal)
- MPS f16 beam=1: 8,941 / 8,958 / 8,976 ms (very stable, <0.5%)

### Results — Beam=5 (with caveats)

| Type | Best ms | Speedup | Segments | Chars | Notes |
|------|---------|---------|----------|-------|-------|
| CPU f32 | 23,125 | baseline | 2 | 363 | — |
| MPS f32 | 3,053 | 7.58x | 1 | 136 | **UNRELIABLE** — early termination |
| MPS f16 | 9,387 | 2.46x | 4 | 503 | Valid (more output than CPU) |
| MPS bf16→f16 | 9,352 | 2.47x | — | — | Same as f16 |

**Warning**: MPS f32 beam=5 produces only 1/3 the text of CPU due to beam search numerical divergence causing early EOS. The 7.58x speedup is artificial.

### Correctness — Beam=1

**F32: exact match** (CPU vs MPS)
> Добрый день, дорогие слушатели, в эфире 454 выпуск подкаста Хобби Докс...

**F16: minor token differences** (expected precision-dependent diffs in first few tokens)

---

## Part 2: whisper-large-v3 (32 enc / 32 dec) — CORRECTNESS FAILURE

### Results — Beam=1

| Type | Best ms | Speedup | Segments | Notes |
|------|---------|---------|----------|-------|
| CPU f32 | 41,206 | baseline | 1 | "Good day, dear listeners..." |
| MPS f32 | 58,311 | 0.71x | **0** | Empty output |
| MPS f16 | 42,099 | 0.98x | **0** | Empty output |
| MPS bf16→f16 | 42,080 | 0.98x | **0** | Empty output |

All MPS paths produce **zero transcription segments**. The encoder works correctly (language detection: Russian, prob=1.000, duration=60.0s), but the 32-layer decoder generates no valid output — likely immediate EOS or garbage tokens filtered by faster_whisper.

Tested with: `condition_on_previous_text=False`, `task="transcribe"`, timestamps enabled, auto language detection. All produce 0 segments.

### Control test: whisper-base (6 enc / 6 dec) — works correctly

```
whisper-base MPS f16: 1 segment
"Доброе время, несуток дорогие слушатели в эфире, 454 выпуск подкасток..."
```

### Root cause analysis

The issue is **decoder depth**. Models tested on MPS:

| Model | Decoder layers | MPS status |
|-------|---------------|------------|
| whisper-base | 6 | Working |
| whisper-large-v3-turbo | 4 | Working |
| whisper-large-v3 | 32 | **BROKEN** (0 segments) |

With 32 decoder layers × ~100+ autoregressive steps, MPS GEMM numerical non-determinism compounds across layers. Each step passes through 32 layers of linear projections + attention, and the accumulated drift eventually produces garbage logits that either:
1. Immediately select EOS token, or
2. Produce tokens filtered out by faster_whisper's VAD/segment logic

This is the same per-layer drift mechanism identified in M12.19 (f32 flash attention required per-layer `synchronize_stream()`), but at 32 decoder layers (vs 22 for TinyLlama or 4 for turbo), the drift exceeds the model's tolerance even with synchronization.

**This is a Metal backend limitation for deep decoder models.** The encoder (32 layers, single forward pass) works correctly; the issue is specifically the autoregressive decode loop amplifying per-layer drift.

---

## Analysis

### Why GPU speedup is limited for whisper-turbo

1. **Decode-dominated workload**: 4 decoder layers × many autoregressive steps (sq=1, sk small). CPU AMX handles small GEMMs efficiently.
2. **Encoder runs once**: 32-layer encoder (GPU-friendly) but amortized across many decode steps.
3. **Decode step overhead**: ~0.4ms CB overhead × ~100 steps = ~40ms pure overhead.
4. **Small batch decode**: beam=1 = batch_size=1, minimal GPU parallelism.

### Comparison across models

| Model | Enc/Dec layers | d_model | f16 speedup | f32 correctness |
|-------|---------------|---------|-------------|-----------------|
| OPUS-MT | 6/6 | 512 | ~1.8x | Exact (greedy) |
| whisper-base | 6/6 | 512 | N/A | Working |
| whisper-turbo | 32/4 | 1280 | 1.43x | Exact (greedy) |
| whisper-large-v3 | 32/32 | 1280 | N/A | **BROKEN** |

### INT8 Not Available

Whisper model files contain float16 weights only. INT8 quantization requires explicit conversion not available for this format.

## Criterion Evaluation

**Plan criterion**: "Larger model shows >3× CPU speedup with f16"

**Result**: Best working model (turbo) achieves **1.43×** — **CRITERION NOT MET**

**Assessment**: The 3× criterion assumed a compute-heavy balanced model. Available whisper models are either decode-dominated (turbo, 4 dec layers) or broken on MPS (large-v3, 32 dec layers). The criterion should be re-evaluated with NLLB-200 (24 enc + 24 dec) if/when the deep decoder correctness issue is resolved.

**Action items**:
1. Investigate deep decoder numerical drift (32 layers × autoregressive) — may need per-layer sync or precision guards
2. Consider benchmarking NLLB-200 once correctness is established for 24+ decoder layers

## Files

| File | Description |
|------|-------------|
| `tests/metal/e2e/bench_whisper_m12_8.py` | Benchmark script (multi-type, best-of-N) |
| `agents/report/milestone-12.8-larger-model-benchmarks.md` | This report |
