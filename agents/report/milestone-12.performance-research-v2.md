# M12 Performance Research — Comprehensive Optimization Analysis

**Date**: 2026-03-11
**Hardware**: Apple M4 (10-core GPU), macOS 15, 16 GB unified memory
**Model**: OPUS-MT En→De (Helsinki-NLP/opus-mt-en-de), d_model=512, 6 layers, 8 heads
**Benchmark**: 50 sentences, beam=4, best-of-3 runs
**Methodology**: 7 parallel research agents (GPU pipeline, CPU overhead, INT8 audit, sampler/beam, elementwise ops, profiling, web research)

---

## Executive Summary

CTranslate2's Metal backend has been optimized through M12.11 to near-theoretical limits for **all float paths** (1 sync/step, 41-55% GPU utilization). INT8 sync elimination (M12.10) brought INT8_float16 above CPU baseline for the first time. Decode loop CPU overhead investigation (M12.11) confirmed **99.1% of time is GPU compute** — CPU-side optimizations (2.1-2.6) are not impactful.

**Remaining optimization opportunities**:
1. **Faster GPU kernels** (GEMM, attention) — 52.8% of decode time
2. **Reduced GPU idle time** between kernel dispatches — better pipelining
3. **Larger models/batches** where GPU utilization naturally increases
4. **Pointer cache**: Only 20-24% hit rate (256-entry direct-mapped, heavy collisions)
5. **Small-tensor dispatch**: Elementwise ops on ≤512K elements pay more dispatch overhead than compute

### Pre-Research Performance Baseline (M12.9)

| Backend | Type | tok/s | ms | Commits | GPU% | vs CPU f32 |
|---------|------|-------|-----|---------|------|-----------|
| MPS | float16 | 1496 | 1036 | 90 | 41% | 1.84× |
| MPS | float32 | 1059 | 1458 | 96 | 54% | 1.30× |
| MPS | int8 | 467 | 3321 | 3522 | 39% | 0.57× |
| MPS | int8_float16 | 496 | 3122 | 3446 | 41% | 0.61× |
| MPS | bfloat16 | 1305 | 1188 | 90 | 44% | 1.60× |
| MPS | int8_bfloat16 | 483 | 3201 | 3446 | 41% | 0.59× |
| CPU | float32 | 813 | 1904 | — | — | 1.00× |
| CPU | int8 (RUY) | 540 | 2882 | — | — | 0.66× |

### Decode Profiler Breakdown (CT2_DECODE_PROFILE=1, 10 sentences)

| Component | float16 | float32 | int8 |
|-----------|---------|---------|------|
| decoder_call | 44.9% | 41.6% | **96.2%** |
| sampler | 54.2% | 55.6% | 3.5% |
| state_update | 0.6% | 0.5% | 0.1% |
| logits_process | 0.3% | 2.2% | 0.2% |

**Key insight**: For f16/f32, "sampler" time is actually "flush ALL pending GPU work + TopK + memcpy" — the single sync per step. For INT8, decoder_call dominates because each Dense layer syncs.

---

## Part 1: INT8 Optimization (CRITICAL — 3522 → ~90 commits)

### 1.1 INT8 Per-Dense Sync Elimination — ✅ COMPLETED (M12.10)

**Root cause**: `src/layers/common.cc:411` — `synchronize_stream(Device::MPS)` fires after every quantized Dense layer because local temporaries (`qinput`, `qinput_scale`, `qoutput`) go out of scope and their MTLBuffers would be recycled before GPU reads them.

**Fix applied**: Replaced `synchronize_stream()` with `protect_buffer()` for the three temporary buffers. See `agents/report/milestone-12.10-int8-protect-buffer.md`.

**Actual results** (vs estimated 3-4×):

| Type | M12.9 tok/s | M12.10 tok/s | Speedup | Commits |
|------|-------------|-------------|---------|---------|
| int8 | 467 | **779** | **1.67×** | 3522 → 97 (97% reduction) |
| int8_float16 | 496 | **889** | **1.79×** | 3446 → 93 (97% reduction) |
| int8_bfloat16 | 483 | **887** | **1.83×** | 3446 → 93 (97% reduction) |

**Note**: The estimated 3-4× improvement was too optimistic. The actual 1.67-1.83× reflects that INT8 compute (GPU dequantize + GEMM) is inherently slower than FP16 GEMM, so removing sync overhead doesn't close the full gap. INT8_float16 at 889 tok/s now surpasses CPU float32 (830 tok/s) for the first time.

### 1.2 MPSGraph Fused Dequantize+MatMul (Priority: ★★★)

**What**: MPSGraph can fuse `dequantize + matmul` into a single kernel that dequantizes INT8 weights on the fly without storing an intermediate FP32 buffer.

**Current INT8 path**: GPU int8→f32 → FP32 MPS GEMM → GPU f32→i32 (3 kernels, encode-only)
**Proposed**: MPSGraph fused dequant+matmul (1 kernel)

**Impact**: Eliminates intermediate FP32 buffer materialization (~2× bandwidth savings)
**Risk**: MPSGraph is synchronous (same issue as BF16), would need `waitUntilScheduled` pattern
**Effort**: High (MPSGraph integration for INT8)
**Verdict**: Defer until 1.1 is done and measured. If INT8 reaches ~1200+ tok/s with protect_buffer alone, this optimization has diminishing returns for the OPUS-MT model.

### 1.3 Custom Metal INT8 GEMM Kernel (Priority: ★★)

**What**: Write a dedicated MSL kernel that reads INT8 weights, dequantizes in-register, and accumulates in FP32 (llama.cpp style).

**Problem**: Benchmarks show custom Metal GEMM kernels achieve only 7-12% of MPS throughput on M4. Even with bandwidth savings from fused dequant, the compute loss negates it.

**Verdict**: Not recommended. MPS GEMM is irreplaceable on Apple Silicon.

---

## Part 2: CPU-Side Decode Loop Optimization

### 2.1 Defer Word ID Conversion — ❌ INVESTIGATED, NOT IMPACTFUL (M12.11)

**Location**: `src/decoding.cc:599`
**Original estimate**: 3-5% (GPU→CPU→GPU roundtrip per step)
**Actual finding**: `convert_to_original_word_ids()` is a **no-op** when `output_layer_is_updated()` returns false (the common case). Even when active, it operates on CPU data (topk_ids is already on CPU after sampler sync). Cannot be deferred because the decoder needs original word IDs for embedding lookup at the next step.
**Profiler data**: beam_bookkeep + step_overhead = 0.5% of total time. No measurable target.

### 2.2 Batch CPU Read of topk_scores — ❌ INVESTIGATED, NOT IMPACTFUL (M12.11)

**Location**: `src/decoding.cc:743`
**Original estimate**: 5-10% (GPU memory per-beam reads causing sync overhead)
**Actual finding**: The sampler copies topk_ids/topk_scores to CPU before returning (M11.26 optimization). So `at()` / `scalar_at()` are **plain CPU array access**, not GPU sync. With `StorageModeShared` and pool allocation, `.to(device)` copies ~16 bytes in ~0.1 µs. The entire beam bookkeeping section is **0.01%** of total time (0.2 ms / 2069 ms).
**Why estimates were wrong**: Assumed topk data lived on GPU; in reality, sampler already copies to CPU.

See `agents/report/milestone-12.11-decode-loop-cpu-overhead.md` for full investigation including decode profiler breakdown.

### 2.3 Pre-allocate DecodingResult Containers — ✅ REJECTED (M12.14)

**Location**: `src/decoding.cc:555`
**Tested**: Added `reserve(num_hypotheses)` for hypotheses, scores, attention vectors.
**Result**: **REJECTED.** Consistent slight regression across all 6 types (all negative deltas). Allocation overhead is ~30 us total (<0.003% of wall time) — too small to measure. All code reverted.
**Report**: `agents/report/milestone-12.14-decode-bookkeeping-optimization.md`

### 2.4 Eliminate alive_seq Concat Per Step — ✅ REJECTED (M12.14)

**Location**: `src/decoding.cc:208-220`
**Tested**: Replaced `ops::Concat(2)` with direct row-by-row memcpy. Also evaluated full pre-allocation to `max_step` width — rejected because `gather_beam_flat` creates new tensors (would lose pre-allocation benefit, wider tensors make gather MORE expensive).
**Result**: **REJECTED.** Manual memcpy slightly worse than `ops::Concat`'s optimized `primitives<CPU>::copy`. alive_seq concat is ~500 us total (<0.05% of wall time). All code reverted.
**Report**: `agents/report/milestone-12.14-decode-bookkeeping-optimization.md`

### 2.5 Lazy Hypothesis Construction — ✅ REJECTED (M12.14)

**Location**: `src/decoding.cc:742-749`
**Issue**: Research proposed deferring hypothesis construction to finalization.
**Result**: **Architecturally infeasible.** `gather_beam_flat` immediately after hypothesis registration overwrites beam data in alive_seq. Token data for finished beams is destroyed before finalization. Current immediate copy is correct and necessary.
**Report**: `agents/report/milestone-12.14-decode-bookkeeping-optimization.md`

### 2.6 Object Pooling for Per-Step Temporaries (Priority: ★★★)

**Location**: `src/decoding.cc:713, 677, 826`
**Issue**: `non_finished_index`, `gather_indices`, `keep_batches` allocated fresh each step.
**Fix**: Allocate before the loop, clear and reuse.
**Impact**: 3-5%
**Effort**: Low-Medium

---

## Part 3: Synchronization Audit (All Types)

### Complete Sync Point Inventory

| Sync Point | File:Line | Fires | Condition | Cost | Eliminable? |
|-----------|-----------|-------|-----------|------|-------------|
| Sampler GPU→CPU | sampling.cc:29 | 1/step | Always | 0.4 ms | No (needs token IDs) |
| INT8 Dense layer | common.cc:411 | 36/step | Quantized only | 14.4 ms | **YES** (protect_buffer) |
| SDPA RoPE | flash_attention_metal.mm:184 | 1/step | offset>0 | 0.4 ms | Yes (GPU RoPE kernel) |
| TopPMask sort | topp_mask_metal.mm:28 | 0-1/step | topp<1.0 | 1-5 ms | Yes (GPU nucleus sampling) |
| BF16 MPSGraph | ops_sdpa.mm:279 | 0-1/step | BF16 only | 0.4 ms | No (framework) |
| SDPA small-tensor | ops_sdpa.mm:589 | ~1/step | sq×sk≤32 | 0.4 ms | No (CPU faster) |
| SDPA padding | ops_sdpa.mm:212 | rare | pad_c | 0.4 ms | No (MPS requirement) |
| prepare_length_mask | beam_search.mm:102 | 1/step | Always | ~0 ms | N/A (non-blocking) |

### Sync Budget Per Step (After Proposed Optimizations)

| Type | M12.9 | M12.10 (actual) | After RoPE GPU | Theoretical Min |
|------|-------|-----------------|----------------|-----------------|
| float16 | 2-3 | 2-3 | 1-2 | 1 |
| float32 | 2-3 | 2-3 | 1-2 | 1 |
| int8 | 38 | **2-3** ✅ | 1-2 | 1 |
| int8_float16 | 38 | **2-3** ✅ | 1-2 | 1 |
| bfloat16 | 2-3 | 2-3 | 1-2 | 1 |

---

## Part 4: GPU Kernel Optimization

### 4.1 BiasAdd + Activation + Residual Fusion (Priority: ★★★)

**Current**: 3 separate kernel dispatches per transformer sub-layer
**Proposed**: Single fused kernel: `out[i] = activation(value[i] + bias[i % bias_size]) + residual[i]`
**Saves**: 24 dispatches → 12 per step (180 µs encoder overhead)
**Impact**: ~2% for OPUS-MT (d_model=512), ~5-15% for d_model≥2048
**Effort**: Medium (new MSL kernel + dispatch function)

### 4.2 Aggressive GEMV Dispatch During Decode — ✅ REJECTED (M12.13)

**Location**: `src/metal/primitives_gemm.mm:1003-1156` — GEMV kernels exist (`kGemvF16MSL`, `kGemvF32MSL`)
**Issue**: During decode, one dimension is typically 1 or beam_size. Custom GEMV was hypothesized to be 20-40% faster than MPS for these narrow shapes.
**Result**: **REJECTED.** Naive scalar GEMV is 20% *slower* than MPS for all shapes. MPS uses hardware matrix units; custom scalar dot-product kernel cannot compete. Both non-batched m≤4 routing and f32 batched m=1 routing caused severe regressions. All code reverted.
**Report**: `agents/report/milestone-12.13-aggressive-gemv-decode.md`

### 4.3 GPU Nucleus Sampling (Priority: ★★★)

**Current**: CPU `std::sort` on GPU probabilities (1-5 ms per step when topp<1.0)
**Proposed**: MSL kernel for top-p marking (single pass + threadgroup tree reduction)
**Impact**: 10-70% speedup when nucleus sampling is enabled
**Effort**: Medium (MSL kernel)
**Note**: Only matters when `topp < 1.0` (not default for beam search)

### 4.4 GPU Rotary Embeddings (Priority: ★★)

**Current**: CPU RoPE on GPU K/V requires `commit_and_wait()` — 0.4 ms per step
**Proposed**: MSL kernel for batch RoPE
**Impact**: ~0.4 ms/step savings
**Effort**: High (interleaved-pair layout is complex on GPU)

---

## Part 5: Memory & Allocator Optimization

### 5.1 Pointer Cache Improvement — ✅ COMPLETED (M12.12)

**Before (M12.3)**: 256-entry direct-mapped cache, 20-30% hit rate
**After (M12.12)**: 512-set × 2-way set-associative + Fibonacci hash, **~49% hit rate**

**Changes applied**:
1. Multiplicative (Fibonacci) hash: `(v >> 4) * golden_ratio >> (64 - bits)` — eliminates clustering from Metal's page-aligned allocations
2. 2-way set-associative: prevents temporary tensor lookups from evicting stable model weight entries
3. 1024 total entries (512 sets × 2 ways)

**Result**: Hit rate improved 2.5× (20-30% → 49%), **no wall-time gain** (within ±5% noise). O(log n) fallback on ~50-100 live entries is ~20 ns — too fast for cache optimization to matter. See `agents/report/milestone-12.12-pointer-cache-improvement.md`.

**Effort**: Low (allocator.mm only)

### 5.2 Metal Residency Sets (Priority: ★★★)

**What**: `MTLResidencySet` (macOS 15+) pins model weight buffers in physical memory, preventing OS eviction under memory pressure.
**Impact**: 0% normal; prevents 10-100× latency spikes under memory pressure
**Effort**: Easy (10-20 lines)
**Location**: Add to `MetalAllocator` init + model weight allocation path

### 5.3 Stride-Aware Allocation for MPS Padding (Priority: ★★)

**Issue**: MPS GEMM requires minimum `rowBytes` alignment. When natural stride is below minimum, CTranslate2 copies to padded temp buffers + adds a conditional `commit_and_wait()`.
**Fix**: Allocate tensors with MPS-preferred row alignment at allocator level.
**Impact**: Eliminates ~0.4 ms per padded GEMM (rare, small-n shapes)
**Effort**: Medium

---

## Part 6: Techniques Evaluated and Rejected

### Already Implemented (No Further Action)

| Technique | Status | Reference |
|-----------|--------|-----------|
| Deferred command buffer encoding | ✅ M11.1, M12.1 | Equivalent to llama.cpp whole-graph encoding |
| PSO caching | ✅ M11.2 | 100% hit rate (37K-74K hits, 0-5 misses) |
| MPS GEMM object cache | ✅ M11.27 | Zero alloc/release per GEMM |
| Buffer pool with bucketed allocation | ✅ M11.22, M12.1 | Power-of-2 bucketing, deferred recycling |
| GPU TopK (single-pass) | ✅ M11.23 | Eliminated 268 syncs |
| GPU Multinomial | ✅ M11.20 | Eliminated 756 syncs |
| BF16→FP16 auto-promotion | ✅ M12.5 | 158× speedup |
| INT8 GPU dequant kernels | ✅ M12.6 | 5.4× speedup |
| Encode barrier (non-blocking CB split) | ✅ M12.1 | Commits 188→90 |

### Not Recommended

| Technique | Why Not |
|-----------|---------|
| Custom Metal GEMM kernels | 7-12% of MPS throughput on M4 (benchmarked) |
| Apple Neural Engine (CoreML) | 0.095 ms XPC overhead per op; INT8 is FP16 internally on ANE |
| Indirect Command Buffers | Incompatible with MPS ops; dynamic shapes prevent reuse |
| MTLSharedEvent async | No CPU work to overlap in decode loop (bookkeeping is 0.02%) |
| Full computation graph (MLX-style) | Would require complete architecture rewrite |
| AMX INT8 (BNNSMatMul) | AMX doesn't natively support INT8 compute (<5% improvement) |
| Metal 4 Tensor API | M5+ hardware only (future) |

---

## Part 7: Implementation Roadmap

### Phase 1: Critical

| # | Task | Impact | Status | Result |
|---|------|--------|--------|--------|
| 1.1 | INT8 protect_buffer sync elimination | INT8: 1.67-1.83× | ✅ M12.10 | int8 467→779, int8_f16 496→889, commits 97% reduced |
| 2.1 | Defer word ID conversion | ~~All: 3-5%~~ | ❌ M12.11 | No-op in common case; CPU data, not GPU roundtrip |
| 2.2 | Batch CPU read of topk_scores | ~~All: 5-10%~~ | ❌ M12.11 | Already CPU after sampler sync; bookkeeping = 0.01% |
| 5.1 | Pointer cache improvement | ~~All: 2-5%~~ | ✅ M12.12 | Hit rate 20→49%; no wall-time gain (O(log n) fast enough) |

### Phase 2: High Impact (Expected: 10-20% additional)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 2.3 | ~~Pre-allocate DecodingResult~~ | ✅ REJECTED (M12.14) | — | Slight regression, reverted |
| 2.4 | ~~Eliminate alive_seq concat~~ | ✅ REJECTED (M12.14) | — | Slight regression, reverted |
| 4.2 | ~~Aggressive GEMV for decode~~ | ✅ REJECTED (M12.13) | — | Naive GEMV 20% slower than MPS |
| 2.5 | ~~Lazy hypothesis construction~~ | ✅ REJECTED (M12.14) | — | Architecturally infeasible |

### Phase 3: Medium Impact (Expected: 5-10% additional)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 4.1 | BiasAdd+Act+Residual fusion | All: 2-5% | Medium | New MSL kernel |
| 4.3 | GPU nucleus sampling | Sampling: 10-70% | Medium | New MSL kernel |
| 5.2 | Metal Residency Sets | Robustness | Easy | `allocator.mm` |
| 2.6 | Object pooling temporaries | All: 3-5% | Low-Med | `decoding.cc` |

### Phase 4: Future / Large Models

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 4.4 | GPU Rotary Embeddings | 0.4 ms/step | High | New MSL kernel |
| 1.2 | MPSGraph fused dequant+matmul | INT8 2× | High | New MPSGraph path |
| — | Tiled Flash Attention | Large models | Very High | New MSL kernel |
| — | Metal 4 migration | M5+: 2-4× | High | Architecture |

---

## Part 8: Actual Performance After Phase 1 (M12.10 + M12.11)

| Backend | Type | Pre-M12 tok/s | M12.10 tok/s | Total Speedup | vs CPU f32 |
|---------|------|--------------|-------------|---------------|-----------|
| MPS | float16 | 1286 | **1490** | 1.16× | **1.80×** |
| MPS | float32 | 923 | **1053** | 1.14× | **1.27×** |
| MPS | **int8** | **77** | **779** | **10.1×** | **0.94×** |
| MPS | **int8_float16** | **78** | **889** | **11.4×** | **1.07×** |
| MPS | bfloat16 | 9 | **1490** | 166× | **1.80×** |
| MPS | **int8_bfloat16** | **8** | **887** | **111×** | **1.07×** |
| CPU | float32 | 813 | 830 | — | 1.00× |

**Key outcomes**:
- INT8_float16 surpasses CPU f32 for the first time (1.07×)
- Float16/bfloat16 are at near-theoretical limits (1 sync/step, 41% GPU utilization)
- Decode loop CPU overhead is **0.5%** — further CPU-side optimizations (2.1-2.6) are not impactful
- Remaining gains require faster GPU kernels (GEMM, attention) or larger models/batches

---

## Appendix: Key Source References

| File | Relevance |
|------|-----------|
| `src/layers/common.cc:408-411` | INT8 per-Dense sync (CRITICAL FIX) |
| `src/decoding.cc:599` | Word ID conversion (GPU→CPU→GPU) |
| `src/decoding.cc:742-749` | Hypothesis building / scores read |
| `src/decoding.cc:208-220` | alive_seq concat (O(n²)) |
| `src/sampling.cc:29` | Sampler sync (mandatory, already optimized) |
| `src/metal/primitives_gemm.mm:1003-1156` | GEMV kernels |
| `src/metal/allocator.mm` | Pointer cache (256-entry, 20% hit rate) |
| `src/ops/flash_attention_metal.mm:184` | RoPE sync |
| `src/ops/topp_mask_metal.mm:28` | TopP sort sync |
| `src/ops/bias_add_metal.mm:17-44` | BiasAdd + Act + Residual (fusion candidate) |
| `src/layers/transformer.cc:21-284` | Layer sequence (fusion analysis) |
| `src/metal/ops_sdpa.mm` | SDPA / attention |

## Appendix: Web Research Sources

- llama.cpp: NORM+MUL+ADD kernel fusion (PR #16220), flash attention tiling, quantized GEMM kernels
- MLX: lazy evaluation, `mx.compile()` JIT fusion, memory management
- Apple: Metal 4 tensors (M5+), residency sets (macOS 15+), simdgroup_matrix intrinsics
- Benchmarks: Custom Metal GEMM = 7-12% of MPS throughput (metal_performance_testing)
- AMX: No native INT8 compute support confirmed (MIT thesis, meekolab research)
- Draw Things: Metal FlashAttention v2 with simdgroup_async_copy (20% improvement)
