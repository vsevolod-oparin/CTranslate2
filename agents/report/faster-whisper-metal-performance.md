# faster_whisper + Metal Performance Report

**Date:** 2026-03-07
**Model:** whisper-large-v3-turbo (d_model=1280, n_heads=20, 4 decoder layers)
**Audio:** 60s Russian podcast (`sample.mp3`)
**Platform:** Apple M4

---

## Benchmark Results

| Pipeline | CPU (ms) | Metal (ms) | RTF | Speedup |
|----------|----------|------------|-----|---------|
| Direct CT2 API (beam=1, no timestamps) | 19,873 | 14,571 | 0.243 | **1.36x** |
| faster_whisper (beam=5, timestamps) | 37,607 | 37,425 | 0.624 | **1.00x** |

Metal shows a 1.36x speedup through the direct CT2 API but **zero speedup** through
faster_whisper.  The pipeline overhead and synchronization points completely erase
the GPU compute advantage.

---

## Root Cause Analysis

### Why faster_whisper is 2x slower than direct API (both backends)

faster_whisper defaults differ from the direct test:

| Setting | Direct CT2 test | faster_whisper |
|---------|----------------|----------------|
| `beam_size` | 1 (greedy) | **5** |
| Timestamps | `<\|notimestamps\|>` | Active (timestamp rules every step) |
| `return_no_speech_prob` | No | Yes |
| Temperature fallback | None | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] |
| `condition_on_previous_text` | No | Yes (prompt accumulation) |

Beam search with size 5 alone roughly doubles the computation per decode step
(5x hypotheses, TopK over vocab, state gathering).

### Why Metal gains vanish with beam_size=5

The decode path is **synchronization-bound**, not compute-bound.  Each decode step
triggers multiple `commit_and_wait()` calls that force the CPU to wait for the GPU
and vice versa.  The GPU compute between syncs is too small to amortize the ~0.4 ms
fixed command buffer overhead.

#### Synchronization points per decode step

| Source | Syncs | File | Line | Cause |
|--------|-------|------|------|-------|
| **TopK (k=10)** | 1 | `src/ops/topk_metal.mm` | 40 | k>1 → `commit_and_wait()` + CPU `partial_sort` over 51,865 vocab |
| **ApplyTimestampRules: max()** | 1 | `src/metal/primitives_reduction.mm` | 127 | `primitives<METAL>::max()` over text tokens |
| **ApplyTimestampRules: logsumexp()** | 1 | `src/metal/primitives_reduction.mm` | 169 | `primitives<METAL>::logsumexp()` over timestamp tokens |
| **Beam state Gather** | 1 | `src/ops/gather.cc` | 144 | `synchronize_stream(METAL)` to reorder beam states |
| **Logits GEMM pad_c** | 1 | `src/metal/primitives_gemm.mm` | 249 | n=51,865 misaligned → `commit_and_wait()` + CPU unpack |
| **Small GEMM CPU fallback** | 1-2 | `src/metal/primitives_gemm.mm` | 102 | Q/K/V projections (m=1, n=1280, 1280≤4096) → cblas |
| **KV-cache update** | 1 | `src/ops/flash_attention_metal.mm` | — | `commit_and_wait()` + CPU memcpy for cache |
| **Total** | **~7-9** | | | |

With ~200-400 decode steps × ~8 syncs × ~0.4 ms ≈ **0.6–1.3 seconds** of pure
synchronization overhead.  But worse than the raw sync cost: each sync serializes
the CPU↔GPU pipeline, preventing overlap of GPU compute with CPU work.

#### Tiny GEMM CPU fallback

The `kCpuGemmThresh = 4096` threshold routes Q/K/V projections (m=1, n=1280,
`rows_c * cols_c = 1280`) through CPU cblas.  Before this, it calls
`commit_and_wait()` to flush pending GPU work.  This means:

1. GPU was computing previous layers → sync
2. CPU runs cblas GEMM → GPU sits idle
3. Next layer encodes GPU work → CPU sits idle

This ping-pong between CPU and GPU per layer is the primary throughput killer.

---

## Recommendations for Performance Improvement

### High Impact

#### 1. GPU TopK kernel for k>1
**Current:** `commit_and_wait()` + CPU `std::partial_sort` over 51,865 elements.
**Fix:** MSL kernel using bitonic sort or radix-select for top-k.  For k=10 and
vocab=51,865 this is a clear GPU win.  Eliminates 1 sync per step.
**Expected gain:** ~10-15% end-to-end for beam=5.

#### 2. Eliminate logits GEMM pad_c sync
**Current:** n=51,865 needs MPS row alignment → padded temp buffer → `commit_and_wait()`
+ CPU row-by-row unpack.
**Fix:** Use GPU blit copy (MTLBlitCommandEncoder) to unpack padded rows, or
pre-allocate output with MPS-aligned row bytes and adjust downstream consumers
to use strided access.
**Expected gain:** Eliminates 1 sync per step.  Combined with #1, removes 2 of
the ~8 syncs.

#### 3. Fuse ApplyTimestampRules into GPU
**Current:** `max()` and `logsumexp()` each trigger `commit_and_wait()` (2 syncs).
**Fix:** Write a single MSL kernel that computes both `max(text_logprobs)` and
`logsumexp(timestamp_logprobs)` and writes the boolean result to a shared-memory
flag.  The CPU reads the flag after the next natural sync point.
**Expected gain:** Eliminates 2 syncs per step (the most frequent pair).

#### 4. Raise kCpuGemmThresh or use MPS for all decode-path GEMMs
**Current:** `kCpuGemmThresh = 4096` routes m=1, n=1280 projections to cblas with
a preceding sync.
**Fix:** Lower the threshold (e.g., to 256) or remove the CPU fallback for the
decode path entirely.  For m=1, n=1280, MPS GEMM encode-only is ~30 µs vs cblas
~15 µs, but the 0.4 ms sync cost to switch to CPU dwarfs the GEMM difference.
Keeping everything on GPU avoids the pipeline bubble.
**Expected gain:** Eliminates 1-2 syncs per layer per step.  With 4 decoder layers
this is 4-8 fewer syncs per step — potentially the **single biggest win**.

### Medium Impact

#### 5. Batch reduction ops
When multiple scalar reductions are needed in the same step (max, logsumexp,
no_speech_prob gather), batch them into a single kernel dispatch + single sync
instead of individual syncs.

#### 6. Async KV-cache update
**Current:** `commit_and_wait()` + CPU memcpy for KV-cache.
**Fix:** Use GPU blit encoder to copy new K/V into cache (Metal shared memory
means no actual data transfer — just pointer arithmetic).  This was explored in
M6.2 but deferred.

### Lower Impact (but easy)

#### 7. faster_whisper configuration tuning
Users can improve speed without code changes:
```python
model.transcribe(
    audio,
    beam_size=1,              # greedy — 2-5x faster, similar quality
    without_timestamps=True,  # avoids ApplyTimestampRules syncs
    condition_on_previous_text=False,
)
```
This alone should bring Metal closer to the 1.36x speedup seen in the direct API.

---

## Sync Budget Summary

| Optimization | Syncs removed per step | Difficulty |
|-------------|----------------------|------------|
| GPU TopK (k>1) | 1 | Medium (MSL kernel) |
| GPU blit for pad_c | 1 | Low (blit encoder) |
| Fused timestamp rules | 2 | Medium (MSL kernel) |
| Remove tiny GEMM CPU fallback | 4-8 (across layers) | Low (threshold change) |
| Async KV-cache | 1 | Low (blit encoder) |
| **Total** | **9-13 of ~30-40** | |

Reducing from ~8 syncs/step to ~3 syncs/step should yield a **1.5-2x speedup** on
Metal for the beam=5 faster_whisper pipeline, bringing it to a meaningful advantage
over CPU.

---

## Files Referenced

| File | Relevance |
|------|-----------|
| `src/ops/topk_metal.mm:40` | TopK k>1 CPU fallback |
| `src/metal/primitives_gemm.mm:101-103` | kCpuGemmThresh tiny GEMM fallback |
| `src/metal/primitives_gemm.mm:248-254` | pad_c output unpack sync |
| `src/metal/primitives_reduction.mm:127,169` | max() and logsumexp() syncs |
| `src/ops/gather.cc:144` | Beam state gather sync |
| `src/models/whisper.cc:836-849` | ApplyTimestampRules::should_sample_timestamp |
| `src/ops/flash_attention_metal.mm` | KV-cache update sync |
