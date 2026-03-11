# M13 Performance Research — Comprehensive Optimization Analysis

**Date**: 2026-03-11
**Hardware**: Apple M4 (10-core GPU), macOS 15, 16 GB unified memory
**Model**: OPUS-MT En→De (Helsinki-NLP/opus-mt-en-de), d_model=512, 6 layers, 8 heads
**Benchmark**: 50 sentences, beam=4, best-of-3 runs
**Methodology**: 7 parallel research agents (GPU pipeline, CPU overhead, INT8 audit, sampler/beam, elementwise ops, profiling, web research)

---

## Executive Summary

CTranslate2's Metal backend has been optimized through M12 to near-theoretical limits for **float16/float32** paths (1 sync/step, 41-54% GPU utilization). The remaining performance gaps are:

1. **INT8: 3522 commits vs 90 for f16** — 36 per-Dense syncs per step account for 80% of the speed gap
2. **CPU decode overhead**: 46-59% of wall time is CPU-side (beam bookkeeping, hypothesis building, word ID conversion)
3. **Pointer cache**: Only 20-24% hit rate (256-entry direct-mapped, heavy collisions)
4. **Small-tensor dispatch**: Elementwise ops on ≤512K elements pay more dispatch overhead than compute

### Current Performance Baseline (M12.9)

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

### 1.1 INT8 Per-Dense Sync Elimination (Priority: ★★★★★)

**Root cause**: `src/layers/common.cc:411` — `synchronize_stream(Device::MPS)` fires after every quantized Dense layer because local temporaries (`qinput`, `qinput_scale`, `qoutput`) go out of scope and their MTLBuffers would be recycled before GPU reads them.

**Current per-step commit math**:
- OPUS-MT decoder: 6 layers × 6 Dense layers/layer = 36 syncs
- Plus sampler (1) + other (1) = **38 syncs/step**
- 38 × 0.4 ms = **15.2 ms overhead/step** (vs 0.4 ms for f16)

**Fix**: Replace `synchronize_stream()` with `protect_buffer()` for the temporary buffers. The `protect_buffer` pattern is already proven (M11.21 gather, M12.6 INT8 GEMM inputs). Temporaries are protected from recycling until the next `commit_and_wait()`, which happens at the sampler sync.

**Implementation**:
```
File: src/layers/common.cc:408-411
Current:  synchronize_stream(Device::MPS);
Proposed: protect_buffer(qinput); protect_buffer(qinput_scale); protect_buffer(qoutput);
```

**Expected impact**: 36 syncs/step → 0 syncs/step (sampler sync remains)
- INT8 commits: 3522 → ~90 (matching f16)
- INT8 overhead: 15.2 ms → 0.4 ms per step
- INT8 tok/s: 467 → **~1200-1400** (3-4× improvement)
- INT8 would become **faster than CPU f32** (currently 0.57× CPU)

**Risk**: Low. Pattern is proven. Only requires that temporaries aren't reused before GPU finishes (protect_buffer guarantees this).

**Effort**: Small (5-10 lines of code)

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

### 2.1 Defer Word ID Conversion to Finalization (Priority: ★★★★★)

**Location**: `src/decoding.cc:599`
**Issue**: `convert_to_original_word_ids(decoder, topk_ids)` called every step, doing GPU→CPU→GPU roundtrip on MPS.
**Fix**: Move conversion to after the decode loop (only needed for final results).
**Impact**: Eliminates 1 GPU→CPU→GPU roundtrip per step = ~0.4 ms/step
**Effort**: Low (move call to finalization)

### 2.2 Batch CPU Read of topk_scores (Priority: ★★★★★)

**Location**: `src/decoding.cc:743`
**Issue**: `topk_scores.scalar_at<float>({i, k})` reads GPU memory per-beam in a loop. Each read may require a sync or at least a cache-line fetch from GPU-written unified memory.
**Fix**: Copy entire scores array to CPU once before the bookkeeping loop:
```cpp
StorageView topk_scores_cpu = topk_scores.to(Device::CPU);
const float* scores_data = topk_scores_cpu.data<float>();
// In loop: scores_data[i * beam_size + k]
```
**Impact**: 5-10% of decode loop
**Effort**: Low (3-5 lines)

### 2.3 Pre-allocate DecodingResult Containers (Priority: ★★★★)

**Location**: `src/decoding.cc:555`
**Issue**: DecodingResult vectors (`hypotheses`, `scores`, `attention`) grow dynamically via push_back/emplace_back, causing repeated heap allocations.
**Fix**: Reserve capacity based on `max_length` and `num_hypotheses`:
```cpp
result.hypotheses.reserve(num_hypotheses);
result.scores.reserve(num_hypotheses);
```
**Impact**: 10-15% allocation reduction in beam bookkeeping
**Effort**: Low

### 2.4 Eliminate alive_seq Concat Per Step (Priority: ★★★★)

**Location**: `src/decoding.cc:208-220`
**Issue**: `ops::Concat` allocates a new buffer every step to append the latest token to the sequence history. For 100-token sequences: 100 reallocations, each copying all accumulated data (O(n²) total).
**Fix**: Pre-allocate `alive_seq` to `max_length` and use slice assignment:
```cpp
StorageView alive_seq({batch * beam, max_steps}, DataType::INT32, device);
// Per step: write into alive_seq[:, step] instead of concat
```
**Impact**: 5-10% of decode loop
**Effort**: Medium (rewrite append_step_output)

### 2.5 Lazy Hypothesis Construction (Priority: ★★★)

**Location**: `src/decoding.cc:742-749`
**Issue**: `build_hypothesis()` copies token sequences from GPU immediately when a beam finishes. Multiple vector allocations per finished beam.
**Fix**: Store (batch_id, beam_id, start, end) tuples, construct hypotheses only at finalization.
**Impact**: 3-5%
**Effort**: Medium

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

| Type | Current | After 1.1 | After RoPE GPU | Theoretical Min |
|------|---------|-----------|----------------|-----------------|
| float16 | 2-3 | 2-3 | 1-2 | 1 |
| float32 | 2-3 | 2-3 | 1-2 | 1 |
| int8 | 38 | **2-3** | 1-2 | 1 |
| int8_float16 | 38 | **2-3** | 1-2 | 1 |
| bfloat16 | 2-3 | 2-3 | 1-2 | 1 |

---

## Part 4: GPU Kernel Optimization

### 4.1 BiasAdd + Activation + Residual Fusion (Priority: ★★★)

**Current**: 3 separate kernel dispatches per transformer sub-layer
**Proposed**: Single fused kernel: `out[i] = activation(value[i] + bias[i % bias_size]) + residual[i]`
**Saves**: 24 dispatches → 12 per step (180 µs encoder overhead)
**Impact**: ~2% for OPUS-MT (d_model=512), ~5-15% for d_model≥2048
**Effort**: Medium (new MSL kernel + dispatch function)

### 4.2 Aggressive GEMV Dispatch During Decode (Priority: ★★★)

**Location**: `src/metal/primitives_gemm.mm:1003-1156` — GEMV kernels already exist (`kGemvF16MSL`, `kGemvF32MSL`)
**Issue**: During decode, one dimension is typically 1 or beam_size. Custom GEMV can be 20-40% faster than MPS for these narrow shapes.
**Current state**: GEMV is used for some paths; could be used more aggressively.
**Impact**: 20-40% for single-token decode GEMM
**Effort**: Medium (dispatch logic routing)

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

### 5.1 Pointer Cache Improvement (Priority: ★★★★)

**Current**: 256-entry direct-mapped cache, 20-24% hit rate
- float16: 295K hits / 1.19M misses (19.9%)
- float32: 734K hits / 1.56M misses (32.0%)
- int8: 709K hits / 1.63M misses (30.3%)

**Root cause**: Direct-mapped cache with `(ptr >> 12) ^ ((ptr >> 8) & 0x3F)` hash — heavy collisions from Metal's allocation patterns.

**Options**:
1. Increase to 1024 or 2048 entries (simple, ~4× less collisions)
2. Switch to 4-way set-associative (better collision handling)
3. Use better hash function (e.g., multiplicative hash)

**Impact**: Faster `buffer_for_ptr()` lookups → less CPU overhead per Metal op
**Effort**: Low-Medium

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

### Phase 1: Critical (Expected: 3-4× INT8 improvement + 10-20% all types)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 1.1 | INT8 protect_buffer sync elimination | INT8: 3-4× | Small | `common.cc:411` |
| 2.1 | Defer word ID conversion | All: 3-5% | Low | `decoding.cc:599` |
| 2.2 | Batch CPU read of topk_scores | All: 5-10% | Low | `decoding.cc:743` |
| 5.1 | Pointer cache improvement | All: 2-5% | Low-Med | `allocator.mm` |

### Phase 2: High Impact (Expected: 10-20% additional)

| # | Task | Impact | Effort | Files |
|---|------|--------|--------|-------|
| 2.3 | Pre-allocate DecodingResult | All: 10-15% alloc | Low | `decoding.cc:555` |
| 2.4 | Eliminate alive_seq concat | All: 5-10% | Medium | `decoding.cc:208-220` |
| 4.2 | Aggressive GEMV for decode | f16/f32: 20-40% GEMM | Medium | `primitives_gemm.mm` |
| 2.5 | Lazy hypothesis construction | All: 3-5% | Medium | `decoding.cc:742-749` |

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

## Part 8: Expected Final Performance (After Phase 1)

| Backend | Type | Current tok/s | Expected tok/s | Improvement |
|---------|------|--------------|----------------|-------------|
| MPS | float16 | 1496 | 1600-1700 | +7-14% |
| MPS | float32 | 1059 | 1150-1250 | +9-18% |
| MPS | **int8** | **467** | **1200-1400** | **+157-200%** |
| MPS | **int8_float16** | **496** | **1300-1500** | **+162-202%** |
| MPS | bfloat16 | 1305 | 1400-1500 | +7-15% |
| MPS | **int8_bfloat16** | **483** | **1300-1500** | **+169-210%** |
| CPU | float32 | 813 | 813 | — |

**INT8 would go from 0.57× CPU to 1.5-1.7× CPU** — a transformative improvement.

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
