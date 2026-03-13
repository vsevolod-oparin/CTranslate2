# M14.7: README Benchmark Update

**Date:** 2026-03-14
**Goal:** Update README MPS table with current benchmark numbers and f16 precision notes.
**Outcome:** Done. All numbers refreshed with best-of-2 runs on Apple M4.

---

## Updated MPS Benchmark Table

| Config | tok/s | RSS (MB) | BLEU | Gap vs f32 |
|--------|-------|----------|------|------------|
| **OpenNMT-py WMT14** | | | | |
| CPU f32 (4 threads) | 1023.5 | 1804 | 26.57 | — |
| MPS f32 | 1643.4 | 1193 | 26.57 | 0.00 |
| MPS f16 | 1611.2 | 1075 | 26.20 | -0.37 |
| MPS int8 | 1079.7 | 660 | 26.55 | -0.02 |
| MPS int8_f16 | 1023.7 | 1107 | 26.23 | -0.34 |
| **OPUS-MT** | | | | |
| CPU f32 (4 threads) | 838.9 | 1650 | 27.65 | — |
| MPS f32 | 1080.7 | 1075 | 27.65 | 0.00 |
| MPS f16 | 992.4 | 939 | 25.76 | -1.89 |
| MPS int8 | 733.1 | 653 | 27.57 | -0.08 |
| MPS int8_f16 | 770.1 | 593 | 25.60 | -2.05 |

### F16 BLEU Gap: Model-Dependent

| Model | f16 BLEU Gap | Notes |
|-------|-------------|-------|
| OpenNMT-py WMT14 | **-0.37** | Excellent — within tolerance at beam=4 |
| OPUS-MT | **-1.89** | Beam search degeneration (M14.4); use beam=6 for parity |

The f16 BLEU gap is model-dependent, not a code precision issue (M14.1-14.4 confirmed all
compute paths use f32 intermediate). OpenNMT-py's smaller vocabulary and different weight
distribution make it more robust to f16 beam search pruning artifacts.

## Changes from Previous README

1. **Number updates**: All MPS numbers refreshed (previous numbers were from earlier milestone)
   - OpenNMT MPS f32: 1027 -> 1643 tok/s (60% faster, M12 optimizations)
   - OPUS MPS f32: 727 -> 1081 tok/s (49% faster)
   - OPUS MPS int8: 481 -> 733 tok/s (52% faster, M12.6+M12.10)

2. **Added OpenNMT f16**: 1611.2 tok/s, BLEU 26.20 (gap -0.37) — previously excluded due
   to overflow bug that was fixed by M13/M14 precision work (f32-accumulation GEMM)

3. **Added int8_float16 for both models**: OpenNMT 1023.7 tok/s, OPUS 770.1 tok/s

4. **Removed Transformers comparison**: Versions were outdated (5.2.0/2.10.0), not maintainable

5. **Removed flash attention rows**: `fused_sdpa_decode` kernel not compiled in current build
   - Flash attention is functional for prefill/standard MHA (ops_sdpa.mm)
   - Fused decode kernel MSL source is missing — needs implementation

6. **Updated summary text**: f16 quality is model-dependent — OpenNMT gap 0.37 (good),
   OPUS-MT gap ~2 (use beam=6+length_penalty=0.6 for parity)

7. **Fixed benchmark script**: Re-enabled OpenNMT f16 in `benchmark_metal_readme.py`

## Flash Attention Status

Flash attention rows were removed because the `fused_sdpa_decode_float/half` kernel referenced
in `ops_sdpa.mm:718` has no corresponding MSL source. The dispatch function exists but the
kernel was never implemented. This needs to be added as a future task or the flash attention
code path should fall back to non-fused decode for encoder-decoder models.

## Test

Benchmark script: `/tmp/bench_readme_update.py` (temp file, results above)
