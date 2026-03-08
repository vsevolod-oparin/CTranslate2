# Metal Performance Analysis: whisper-large-v3-turbo

## Context

After completing M11.17 (fused timestamp check + disable kernel, eliminating ~1346 syncs), a benchmark of whisper-large-v3-turbo with beam_size=5 on 60s audio revealed Metal is **0.65x CPU speed** — significantly slower despite all prior optimizations (M11.5–M11.17).

This report documents the root cause investigation.

## Benchmark Configuration

- **Model**: whisper-large-v3-turbo (d_model=1280, 20 heads, 4 decoder layers, 32 encoder layers, vocab=51866)
- **Audio**: 60s Russian podcast (`sample.mp3`)
- **Beam size**: 5
- **Compute type**: float32 (both CPU and Metal, for fair comparison)
- **Hardware**: Apple M4

## Results

```
CPU: 27699ms  Metal: 42516ms  Speedup: 0.65x
```

Metal is **53% slower** than CPU.

## Sync Trace (CT2_METAL_TRACE=1)

| Source | Syncs | % of Total | Description |
|--------|------:|---:|-------------|
| `primitives_gemm.mm:1020` | **8728** | **66.4%** | CPU cblas GEMM fallback for tiny padded matrices |
| `devices.cc:162` | 1506 | 11.5% | `synchronize_stream()` from data conversions |
| `primitives_memory.mm:80` | 1455 | 11.1% | `indexed_fill` CPU scatter (DisableTokens::apply) |
| `multinomial_metal.mm:21` | 1187 | 9.0% | CPU `std::discrete_distribution` sampling |
| `topk_metal.mm:44` | 268 | 2.0% | CPU `std::partial_sort` for k>1 beam search |
| **Total** | **13,144** | **100%** | |

At ~0.4ms per `commit_and_wait()`, the syncs alone account for **~5.3 seconds** of pure overhead.

## GEMM Fallback Deep Dive

### Why CPU Fallback?

The `gemm_batch_strided` function (M11.5) falls back to CPU cblas when:
1. MPS requires row padding (`cols * sizeof(T) % 16 != 0`)
2. Output matrix is small (`rows_c * cols_c <= 4096`)

For decode-step attention GEMMs, **both conditions are always met**:
- `m=1` → output has 1 row, always ≤ 4096 elements
- Sequence lengths often have `n % 4 != 0` (float32: `n*4 % 16 != 0` → padding)

### GEMM Dimension Trace

Added temporary tracing to `batch_cpu_gemm_f32` to capture every fallback. Key findings:

**All 8728 calls are m=1 decode-step attention GEMMs with k=64 (d_k = 1280/20 heads).**

Two patterns appear in pairs:

| Pattern | Shape | Description |
|---------|-------|-------------|
| QK^T | `m=1, n=seq_len, k=64, tb=1` | Query × Key^T → attention scores |
| scores×V | `m=1, n=64, k=seq_len, tb=0` | Scores × Values → attention output |

### Batch Size Distribution

| Batch | Calls | % | Interpretation |
|------:|------:|---:|----------------|
| 20 | 6600 | 75.6% | 20 heads × 1 (cross-attention via non-flash path, or partial beams) |
| 100 | 1744 | 20.0% | 20 heads × 5 beams (full self-attention batch) |
| 80 | 112 | 1.3% | 20 heads × 4 beams (partially finished beams) |
| 60 | 156 | 1.8% | 20 heads × 3 beams |
| 40 | 116 | 1.3% | 20 heads × 2 beams |

The batch=20 dominance (75.6%) suggests many calls come from attention with a single effective beam dimension — likely cross-attention or attention on short segments.

### Sequence Length Distribution

The variable dimension (`n` in QK^T, `k` in scores×V) ranges from **3 to 354**, representing the KV cache position count at each decode step. Nearly all values have `val % 4 != 0`, triggering MPS padding.

## Why Prior Optimizations Didn't Catch This

whisper-large-v3-turbo differs from whisper-base (used for M11.5 benchmarking):

| Property | whisper-base | whisper-large-v3-turbo |
|----------|-------------|----------------------|
| d_model | 512 | 1280 |
| num_heads | 8 | 20 |
| d_k | 64 | 64 |
| Decoder layers | 6 | 4 |
| Encoder layers | 6 | 32 |
| Vocab | 51865 | 51866 |
| Typical decode steps | ~50 | ~200+ |

With 20 heads (vs 8) and longer outputs, the number of m=1 attention GEMMs is much higher.

## Other Sync Sources

### devices.cc:162 — synchronize_stream (1506 syncs)

Called by data type conversions and framework-level stream synchronization. These are architectural and cannot be easily eliminated without restructuring the data flow.

### primitives_memory.mm:80 — indexed_fill (1455 syncs)

`DisableTokens::apply()` uses CPU scatter fill with a fence sync. M11.17 reduced this from ~1205 to ~1035 for the M11.16 baseline. The remaining syncs come from the first `apply()` in `ApplyTimestampRules` which accumulates suppress/ordering tokens. Cannot be eliminated without restructuring `DisableTokens` to be fully deferred.

### multinomial_metal.mm:21 — CPU Sampling (1187 syncs)

Multinomial sampling uses `std::discrete_distribution` which requires CPU access to GPU-computed probabilities. Each decode step with beam_size=5 does one sampling call. A GPU sampling kernel would eliminate these.

### topk_metal.mm:44 — CPU TopK (268 syncs)

For k>1 (beam search), TopK uses CPU `std::partial_sort`. A GPU TopK kernel for small k (k=5) would eliminate these, but the impact is relatively minor.

## Solution: M11.18 — Encode-Only MPS Padded GEMM + Deferred-Free Allocator

**Implemented.** See `agents/report/milestone-11.18-mps-padded-gemm.md`.

Routed m=1 padded GEMMs through the existing `dispatch_mps_gemm_batched_padded` path (encode-only, zero syncs) with a targeted deferred-free mechanism in the MetalAllocator to prevent buffer reuse before GPU execution completes.

### Actual Impact

| Source | Before | After M11.18 | Delta |
|--------|--------|-------------|-------|
| `primitives_gemm.mm` (CPU GEMM) | 8,728 | **16** | **-8,712** |
| Other sources | 4,416 | 3,126 | -1,290 |
| **Total** | **13,144** | **3,142** | **-10,002** |

Performance: Metal 0.65x → **1.03–1.22x** CPU (whisper-large-v3-turbo, beam_size=5, 60s audio).

## Comparison with whisper-base

For reference, the M11.5 whisper-base benchmark showed Metal at **1.32x CPU** (later improved to ~1.98x by M11.8, and ~3.35x by M11.18). whisper-base has fewer heads and shorter outputs, producing far fewer m=1 GEMM calls. The CPU GEMM fallback was adequate for that model but does not scale to larger models.
