# Milestone 10.4 — Whisper ASR End-to-End on Metal

**Date:** 2026-03-03
**Status:** Complete (8/8 tests pass)

## Summary

Whisper-base (encoder-decoder ASR model with Conv1D frontend) runs end-to-end on the Metal
backend with **exact transcript match** against CPU. The Metal transcription is character-for-character
identical to CPU output, yielding 0.00% WER — well within the 1% tolerance.

No new Metal ops or bug fixes were required — all components (Conv1D, Transformer encoder/decoder,
SDPA, KV-cache, beam search) were already implemented and verified in M1–M10.3.

## Test Setup

| Parameter | Value |
|-----------|-------|
| Model | `whisper-base` (74M params, 6 encoder + 6 decoder layers) |
| Audio | `sample.mp3` — 60s Russian podcast (Hobby Talks #454), mono, 44.1kHz → resampled to 16kHz |
| Chunk size | 30s (Whisper's native window) |
| Prefix tokens | `<\|startoftranscript\|> <\|en\|> <\|transcribe\|> <\|notimestamps\|>` |
| WER library | `jiwer 4.0.0` |
| Platform | Apple M4, macOS |

## Test Results

```
$ CT2_TEST_DATA=.../data python3 tests/metal/e2e/test_whisper.py

Loading processor and audio...
  Audio: .../data/sample.mp3 (60.0s, 960070 samples)
Loading models...

=== Transcription ===
  CPU  : ", несуток дорогие слушатели в эфире, 454 выпуск подкаста, ..."
  Metal: ", несуток дорогие слушатели в эфире, 454 выпуск подкаста, ..."

=== Correctness ===
  [PASS] CPU transcription non-empty
  [PASS] Metal transcription non-empty
  [PASS] No error tokens in CPU output
  [PASS] No error tokens in Metal output

  Metal-vs-CPU WER: 0.0000 (0.00%)
  [PASS] Metal-vs-CPU WER < 1%  (WER=0.00%)
  [PASS] Exact transcript match (CPU == Metal)  (identical)

=== Output sanity ===
  [PASS] CPU output > 10 chars  (len=671)
  [PASS] Metal output > 10 chars  (len=671)

=== Speed benchmark (informational) ===
  [INFO] CPU:   6209 ms  (RTF=0.103)
  [INFO] Metal: 27087 ms  (RTF=0.451)
  [INFO] Ratio: 0.23x (target: >=1.5x after M11 optimization)

==================================================
8/8 passed
ALL PASS
```

## Correctness Analysis

| Metric | Value | Tolerance | Status |
|--------|-------|-----------|--------|
| Metal-vs-CPU WER | 0.00% | < 1% | PASS |
| Exact transcript match | Yes (identical) | — | PASS |
| Transcript length | 671 chars | > 10 | PASS |
| Error tokens | None | None | PASS |

The exact match demonstrates that the entire Whisper pipeline on Metal — from mel-spectrogram
feature extraction through Conv1D encoder frontend, Transformer encoder, cross-attention decoder
with KV-cache, and beam search — produces numerically identical results to CPU.

## Speed Analysis

| Device | Time (ms) | RTF | Notes |
|--------|-----------|-----|-------|
| CPU | 6,209 | 0.103 | Single-thread, optimized |
| Metal | 27,087 | 0.451 | Per-op commit overhead |
| Ratio | 0.23x | — | Metal slower due to CB overhead |

The Metal backend is currently ~4.3x slower than CPU for Whisper. This is expected and consistent
with the per-op command buffer commit overhead observed in all previous benchmarks. Each Metal op
incurs ~0.4ms CB overhead, and Whisper involves hundreds of ops per chunk (Conv1D + 6 encoder layers
+ 6 decoder layers × many tokens). Command buffer batching in M11 will amortize this overhead
across all ops in a layer, which should bring Metal throughput above CPU.

## Whisper Architecture on Metal

The Whisper model exercises these Metal ops per inference:

**Encoder (per chunk):**
1. **Conv1D** × 2 — im2col + GEMM (M8.3)
2. **GELU** activation
3. **PositionEmbedding** (sinusoidal)
4. **Transformer Encoder** × 6 layers:
   - LayerNorm → 4× GEMM (Q/K/V/Out) → SDPA → Residual → LayerNorm → 2× GEMM + ReLU → Residual

**Decoder (per token):**
1. **Embedding** lookup
2. **PositionEmbedding** (learned)
3. **Transformer Decoder** × 6 layers:
   - LayerNorm → Self-attn (GEMM×4 + SDPA + KV-cache) → Residual
   - LayerNorm → Cross-attn (GEMM×4 + SDPA) → Residual
   - LayerNorm → FFN (GEMM×2 + ReLU) → Residual
4. **Final LayerNorm** → Linear projection → beam search/greedy

## Files Modified

| File | Change |
|------|--------|
| `tests/metal/e2e/test_whisper.py` | Rewritten: Metal-vs-CPU comparison with WER, timing, 8 checks |
| `APPLE_M4_METAL_PLAN.md` | Updated M10.4 status to ✅ |

## Dependencies

- `jiwer>=3.0` — Word Error Rate computation
- `librosa` — audio loading and resampling
- `transformers` — WhisperProcessor for tokenization

## Milestone 10 Complete

With M10.4, all Milestone 10 subtasks are now complete:

| Subtask | Model Type | Status |
|---------|------------|--------|
| M10.1 | Beam search fix (gather bug) | ✅ |
| M10.2 | Seq2seq (opus-mt-en-de) | ✅ BLEU exact match |
| M10.3 | Language model (GPT-2) | ✅ Token-exact match |
| M10.4 | Whisper ASR (whisper-base) | ✅ Transcript exact match |

The Metal backend now supports all three major CTranslate2 model families end-to-end.
Next milestone: M11 (Performance Optimization — command buffer batching).
