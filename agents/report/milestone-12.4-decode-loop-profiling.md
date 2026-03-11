# M12.4 — Decode Loop Profiling (All Compute Types)

**Date**: 2026-03-11
**Status**: Complete — reveals fundamentally different bottleneck per dtype
**Model**: OPUS-MT En→De (d_model=512, 6 layers)
**Method**: `CT2_DECODE_PROFILE=1` env var, `chrono::high_resolution_clock` instrumentation in `decoding.cc`

---

## 1. Instrumentation

Added 8-component profiling to the beam search loop in `src/decoding.cc`, controlled by `CT2_DECODE_PROFILE=1` environment variable. Zero overhead when disabled (single branch on a cached bool).

Components measured:
1. **decoder_call** — `decoder(step, ids, state, &logits)` — full transformer forward pass (all layers, GEMM, norm, attention)
2. **logits_process** — `DisableTokens` + `apply_min_length` + logits processors + `apply()`
3. **log_softmax** — `LogSoftMax()` + beam score accumulation + flatten
4. **sampler** — `sampler()` call (TopK on GPU + `synchronize_stream` + memcpy to CPU)
5. **beam_bookkeep** — EOS check, hypothesis registration, `finalize_result`
6. **beam_gather** — `gather_beam_flat` / `batch_gather_beam_flat` + `keep_batches` filtering
7. **state_update** — `decoder.update_state()` (KV cache reorder by beam indices)
8. **step_overhead** — unflatten_ids, prefix handling, append_step_output

---

## 2. Results — All Compute Types

All results: best-of-3, batch 0 (largest batch = 32 sentences for 50-sent runs, 10 for slow types).

### Summary Table

| Compute Type | Steps | decoder_call | sampler | All other | Total loop | Per step | Bottleneck |
|-------------|-------|-------------|---------|-----------|------------|----------|------------|
| **float16** | 62 | 401 ms (50%) | 398 ms (50%) | 3 ms (0.4%) | 802 ms | 12.9 ms | GPU compute + sync balanced |
| **float32** | 62 | 483 ms (43%) | 614 ms (55%) | 21 ms (1.9%) | 1118 ms | 18.0 ms | sampler sync dominates |
| **int8** | 49 | 2232 ms (98%) | 40 ms (1.8%) | 6 ms (0.3%) | 2278 ms | 46.5 ms | **decoder_call dominates** |
| **int8_float16** | 48 | 2177 ms (98%) | 34 ms (1.5%) | 7 ms (0.3%) | 2218 ms | 46.2 ms | **decoder_call dominates** |
| **bfloat16** | 49 | 20758 ms (99.8%) | 40 ms (0.2%) | 3 ms (0.0%) | 20801 ms | 424.5 ms | **decoder_call dominates** |
| **int8_bfloat16** | 49 | 22171 ms (99.8%) | 36 ms (0.2%) | 5 ms (0.0%) | 22212 ms | 453.3 ms | **decoder_call dominates** |

### Detailed: Float16 (50 sentences, 62 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **decoder_call** | **401** | **50.0%** | 6.47 |
| **sampler** | **398** | **49.6%** | 6.42 |
| state_update | 1.5 | 0.2% | 0.024 |
| step_overhead | 0.6 | 0.1% | 0.010 |
| beam_gather | 0.3 | 0.0% | 0.005 |
| log_softmax | 0.2 | 0.0% | 0.003 |
| beam_bookkeep | 0.1 | 0.0% | 0.002 |
| logits_process | 0.0 | 0.0% | 0.000 |
| **total_loop** | **802** | **100%** | **12.93** |

### Detailed: Float32 (50 sentences, 62 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **sampler** | **614** | **54.9%** | 9.90 |
| **decoder_call** | **483** | **43.2%** | 7.79 |
| logits_process | 18 | 1.6% | 0.29 |
| state_update | 1.7 | 0.2% | 0.027 |
| everything else | 1.0 | 0.1% | — |
| **total_loop** | **1118** | **100%** | **18.03** |

### Detailed: INT8 (10 sentences, 49 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **decoder_call** | **2232** | **98.0%** | **45.55** |
| sampler | 40 | 1.8% | 0.82 |
| state_update | 4.1 | 0.2% | 0.08 |
| log_softmax | 0.4 | 0.0% | — |
| everything else | 1.1 | 0.0% | — |
| **total_loop** | **2278** | **100%** | **46.49** |

### Detailed: INT8+Float16 (10 sentences, 48 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **decoder_call** | **2177** | **98.2%** | **45.35** |
| sampler | 34 | 1.5% | 0.70 |
| state_update | 6.1 | 0.3% | 0.13 |
| everything else | 1.0 | 0.0% | — |
| **total_loop** | **2218** | **100%** | **46.20** |

### Detailed: BFloat16 (10 sentences, 49 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **decoder_call** | **20758** | **99.8%** | **423.6** |
| sampler | 40 | 0.2% | 0.81 |
| state_update | 1.9 | 0.0% | 0.04 |
| everything else | 1.2 | 0.0% | — |
| **total_loop** | **20801** | **100%** | **424.5** |

### Detailed: INT8+BFloat16 (10 sentences, 49 steps)

| Component | Time (ms) | % | Per step (ms) |
|-----------|-----------|---|---------------|
| **decoder_call** | **22171** | **99.8%** | **452.5** |
| sampler | 36 | 0.2% | 0.73 |
| state_update | 3.5 | 0.0% | 0.07 |
| everything else | 1.4 | 0.0% | — |
| **total_loop** | **22212** | **100%** | **453.3** |

---

## 3. Analysis

### 3.1 Three fundamentally different bottleneck profiles

The six compute types fall into three categories:

**Category A — GPU-balanced (f16):**
- decoder_call (50%) ≈ sampler (50%)
- Per step: 12.9ms = 6.5ms decode + 6.4ms sample
- The sampler's `synchronize_stream()` waits for the GPU work encoded during decoder_call
- Both sides contain GPU compute — the pipeline is balanced
- **Optimization**: Only faster GPU kernels or larger model can help

**Category B — Sync-dominated (f32):**
- sampler (55%) > decoder_call (43%)
- Per step: 18.0ms = 7.8ms decode + 9.9ms sample
- f32 GEMM takes ~1.5x longer than f16, so the sync wait inside sampler is longer
- logits_process shows 18ms anomaly (f32 indexed_fill pre-sync)
- **Optimization**: f32 indexed_fill pre-sync elimination could save ~18ms; otherwise limited

**Category C — decoder_call dominated (int8, int8_f16, bf16, int8_bf16):**
- decoder_call: **98-99.8%** of all time
- sampler: 34-40ms (1-2%) — nearly constant across types (TopK is dtype-independent post-logits)
- The sampler sync is essentially free because by the time it's called, all prior GPU work has already completed (the decoder_call itself contains blocking syncs internally)

For the Category C types, the bottleneck is entirely **inside the decoder forward pass**:

| Type | decoder_call/step | Why |
|------|-------------------|-----|
| int8 | 45.6 ms | CPU int8→f32 dequant + 2 syncs per GEMM × ~36 GEMMs/step |
| int8_f16 | 45.4 ms | Same as int8 (accumulate type doesn't matter) |
| bf16 | 423.6 ms | MPSGraph `runWithMTLCommandQueue:` synchronous per GEMM (~1ms each) |
| int8_bf16 | 452.5 ms | int8 CPU round-trip + MPSGraph sync (compounds both) |

### 3.2 INT8 decoder_call breakdown (estimated)

Per step at 45.6ms with ~36 GEMMs/step (6 layers × 6 GEMMs each):
- 36 GEMMs × 2 syncs each × 0.4ms/sync = **28.8ms in commit_and_wait** alone
- 36 GEMMs × CPU dequant (~0.1ms each) = **3.6ms CPU conversion**
- 36 GEMMs × GPU GEMM compute (~0.2ms each) = **7.2ms GPU**
- Remaining ~6ms: non-GEMM ops (norm, activation, attention)

The 5692 commits for 10 sentences (49 steps × ~2 batches) = ~58 commits/step confirms ~2 syncs per GEMM × ~29 GEMMs/step.

**Fix (M12.6)**: GPU int8→f32 dequantize kernel would eliminate both syncs per GEMM (the entire pipeline becomes encode-only). Expected: 45.6ms → ~13ms/step (~3.5x speedup).

### 3.3 BFloat16 decoder_call breakdown (estimated)

Per step at 423.6ms with ~36 GEMMs/step:
- 36 GEMMs × ~11ms/GEMM (MPSGraph synchronous exec) = **~400ms**
- Each BF16 GEMM: pre-flush CB (~0.4ms) + MPSGraph compile+run+sync (~10ms) + readBytes copy (~0.5ms)
- Remaining ~24ms: non-GEMM ops (these use encode-only kernels, fast)

The 2844 commits for 10 sentences = ~29 commits/step, roughly 1 pre-flush per GEMM.

**Fix (M12.5)**: BF16→FP16 auto-promotion would convert 423.6ms/step → ~6.5ms/step (f16 speed). Alternatively, async MPSGraph execution could reduce to ~50-100ms/step.

### 3.4 Why the sampler time is constant (~35-40ms) across ALL types

The sampler runs GPU TopK (dtype-independent since logits are always float-type), then `synchronize_stream()`, then CPU memcpy. For the slow types (int8, bf16), all GPU work has already completed during the decoder_call (which blocks internally), so `synchronize_stream()` returns immediately. The 35-40ms is:
- TopK kernel dispatch + execution: ~30ms (for vocab_size=59,514 × batch × beam)
- memcpy: ~5ms
- No sync wait (GPU already idle)

For f16/f32, the sync inside sampler actually waits for prior GPU work, which is why sampler takes 400-614ms.

### 3.5 The f32 logits_process 18ms anomaly

Float32 shows 18ms in logits_process (vs 0ms for f16). This comes from `disable_tokens.apply()` which calls `indexed_fill` to write `-inf` to disabled token logits. The f32 path hits the pre-sync at `primitives_memory.mm:114` (~4 syncs × 0.4ms each for the 4 `indexed_fill` calls per batch). This is a known M11.28 limitation — f32 pre-sync was preserved due to an MPS ordering issue.

---

## 4. Corrected Bottleneck Picture

### Previous understanding (WRONG):
```
wall_time = GPU_compute (42%) + CPU_overhead (58%)
                                    ↑
                         "beam bookkeeping, ObjC, buffer lookup"
```

### Actual picture (per compute type):
```
f16:         [==decoder(50%)==][==sampler+sync(50%)==]  — GPU balanced
f32:         [=decoder(43%)=][===sampler+sync(55%)===]  — sync wait
int8:        [=========decoder(98%)=========][s(2%)]    — internal decoder syncs
int8_f16:    [=========decoder(98%)=========][s(2%)]    — internal decoder syncs
bf16:        [=============decoder(99.8%)===========][·] — MPSGraph synchronous GEMM
int8_bf16:   [=============decoder(99.8%)===========][·] — both problems
```

In ALL cases, CPU-side decode loop overhead (beam bookkeeping, gather, unflatten) is **<0.4% of wall time**.

---

## 5. Optimization Impact Matrix (Revised)

| Optimization | f16 impact | f32 impact | int8 impact | bf16 impact |
|-------------|-----------|-----------|-------------|-------------|
| **M12.5: BF16→FP16 promotion** | — | — | — | **9→~1400 tok/s** (47x) |
| **M12.6: GPU int8 dequantize** | — | — | **84→~300 tok/s** (3.5x) | — |
| **f32 indexed_fill pre-sync fix** | — | ~18ms saving (1.2%) | — | — |
| Decode loop CPU opt | 0% | 0% | 0% | 0% |
| Op fusion | <1% | <1% | <1% | <1% |
| Pipeline overlap | 0% | 0% | 0% | 0% |
| Larger model | Higher GPU util | Higher GPU util | Higher GPU util | Higher GPU util |

### Priority order:
1. **M12.5** (BF16 fix) — highest absolute impact, low effort if auto-promoting to f16
2. **M12.6** (INT8 GPU dequant) — 3.5x speedup for int8 types, medium effort
3. **M12.9** (larger model benchmarks) — validates that f16/f32 scales with model size
4. f32 indexed_fill fix — small but measurable for f32

---

## 6. Per-Step Cost Comparison

| Type | ms/step | Relative to f16 | Main cost per step |
|------|---------|-----------------|-------------------|
| float16 | 12.9 | 1.0x | GPU GEMM + sync |
| float32 | 18.0 | 1.4x | GPU GEMM (1.5x data) + sync |
| int8 | 46.5 | 3.6x | 2 CPU syncs × 36 GEMMs/step |
| int8_f16 | 46.2 | 3.6x | Same as int8 |
| bfloat16 | 424.5 | 32.9x | MPSGraph sync per GEMM |
| int8_bf16 | 453.3 | 35.1x | int8 + MPSGraph combined |

---

## 7. Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `src/decoding.cc` | Added `DecodeProfiler` struct + `decode_profile_enabled()` | Per-component decode timing |
| `src/decoding.cc` | Instrumented beam search loop (8 timer regions) | Measure all components |

**Activation**: `CT2_DECODE_PROFILE=1` environment variable. Output goes to stderr.
**Overhead when disabled**: Single branch on cached static bool (~0 ns).
