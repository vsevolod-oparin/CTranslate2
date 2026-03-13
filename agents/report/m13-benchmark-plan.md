# M13 Float16 Benchmark Plan

## What Was Done (M13 Fix Summary)

### Problem
Float16 `translate_batch` on MPS produced garbage output (only periods). Root cause: two compounding issues:
1. **MPS float16 accumulation**: `MPSMatrixMultiplication` accumulates in float16, causing precision loss for K >= 512
2. **Metal encoder tracking limit**: >~128 encode-only compute encoders in a single command buffer causes unreliable resource tracking

### Fix Applied
1. **Custom MSL GEMM kernel** (`gemm_f16_acc32`): half inputs, float32 accumulation, half output — replaces MPS float16 GEMM entirely
2. **Periodic CT2_COMMIT_AND_WAIT**: every 64 GEMMs to keep encoder count per command buffer within Metal's reliable tracking limit
3. **Buffer protection**: `protect_buffer_by_base()` on A, B, C to prevent allocator recycling between syncs
4. **SDPA float16 path**: Added CT2_COMMIT_AND_WAIT between MPS f32 GEMM and f32→half conversion in `ops_sdpa.mm`

### Verification Results
- **Float16 E2E test (OPUS-MT)**: 9/9 pass — f16 output matches CPU f32 exactly (greedy, beam=4, batch)
- **Main translation test (OPUS-MT)**: 88/90 pass — same baseline as f32 (2 pre-existing edge cases)
- **Whisper E2E test**: 18/18 pass — 0.0% WER, exact transcript match with CPU f32

### Files Modified
- `src/metal/primitives_gemm.mm` — Custom f16 GEMM kernel + periodic flush + buffer protection
- `src/metal/ops_sdpa.mm` — CT2_COMMIT_AND_WAIT for SDPA f16 promotion path

## Precheck Results (50 sentences, beam=4, Apple M4)

### OpenNMT-py WMT14

| Config | tok/s | Time | Tokens | Quality |
|--------|-------|------|--------|---------|
| f32 | 1530.5 | 1.1s | 1703 | OK |
| f16 | 166.4 | 10.2s | 1699 | OK |
| f32+flash | 354.7 | 4.6s | 1648 | OK |
| **f16+flash** | **258.6** | **25.7s** | **6639** | **BROKEN** — 24/50 sentences degenerate (>100 tok) |
| int8 | 1065.3 | 1.6s | 1697 | OK |

### OPUS-MT

| Config | tok/s | Time | Tokens | Quality |
|--------|-------|------|--------|---------|
| f32 | 975.8 | 1.6s | 1544 | OK |
| f16 | 141.4 | 10.9s | 1544 | OK |
| f32+flash | 318.0 | 3.3s | 1040 | OK |
| f16+flash | 153.7 | 6.7s | 1035 | OK |

### Key Findings

1. **f16 is ~9x slower than f32** in batch mode due to periodic CT2_COMMIT_AND_WAIT (every 64 GEMMs). Single-sentence was only ~3x slower because batch processing triggers more GEMMs.

2. **OpenNMT-py f16+flash is broken**: Single-sentence decoding produces degenerate output (256 tokens of periods) for ~50% of sentences. Batch mode produces reasonable output but with different quality. This is a separate SDPA float16 path issue, not related to the main GEMM fix. **Excluded from full benchmark.**

3. **OPUS-MT f16+flash works correctly**: All outputs reasonable, token counts match across modes.

4. **f32+flash is slower than standard f32** for these small models: Flash attention has overhead that doesn't pay off for small sequence lengths.

5. **Full WMT14 time estimates** (2737 sentences, ×2 samples):
   - f32: ~2 min, f16: ~18 min, f32+flash: ~5 min
   - OPUS-MT f16+flash: ~12 min
   - Total: ~1 hour including all configs + Transformers baselines

## Benchmark Plan

### Configurations to Run

**CTranslate2 (standard attention):**
| Model | Device | Compute Type |
|-------|--------|-------------|
| OpenNMT-py WMT14 | CPU | float32 (4 threads) |
| OpenNMT-py WMT14 | MPS | float32 |
| OpenNMT-py WMT14 | MPS | float16 |
| OpenNMT-py WMT14 | MPS | int8 |
| OPUS-MT | CPU | float32 (4 threads) |
| OPUS-MT | MPS | float32 |
| OPUS-MT | MPS | float16 |
| OPUS-MT | MPS | int8 |

**CTranslate2 (flash attention):**
| Model | Device | Compute Type | Status |
|-------|--------|-------------|--------|
| OpenNMT-py WMT14 | MPS | float32 + flash | OK |
| ~~OpenNMT-py WMT14~~ | ~~MPS~~ | ~~float16 + flash~~ | **EXCLUDED** — degenerate single-sentence output |
| OPUS-MT | MPS | float32 + flash | OK |
| OPUS-MT | MPS | float16 + flash | OK |

**Transformers (PyTorch) — OPUS-MT only:**
| Device | Dtype |
|--------|-------|
| CPU | float32 |
| MPS | float32 |
| MPS | float16 |

### FasterTransformer
**NOT applicable for MPS.** FasterTransformer v5.3 is NVIDIA CUDA-only (requires `nvcr.io/nvidia/pytorch` Docker image, CUDA compilation). It cannot run on Apple Silicon / Metal. The existing `benchmark_all.py` includes it for GPU comparison only.

### Methodology
- Test set: WMT14 newstest2014 En→De (2737 sentences)
- Beam size: 4
- Samples: 2 (best of 2 runs)
- CPU threads: 4
- Each config runs in subprocess for memory isolation
- Metrics: tokens/sec, max RSS (MB), BLEU score
- Script: `tools/benchmark/benchmark_metal_readme.py` (updated with flash_attention support)
