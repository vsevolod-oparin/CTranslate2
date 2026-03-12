# M12.14 — FlashMHA Commit Count Optimization

**Date**: 2026-03-12
**Status**: COMPLETE
**Branch**: `metal-backend`

## Summary

Optimized FlashMultiHeadAttention decode performance on MPS by replacing 1408 per-head MPS GEMM ObjC calls per step with 22 fused MSL kernel dispatches, and replacing 22 CPU-roundtrip KV-cache commits with encode-only GPU blit copies. Result: flash f32 **3.5 → 18.6 tok/s (5.3x speedup)**, now 1.8x faster than standard f32.

## Performance Results (TinyLlama, Apple M4, greedy beam=1, max_length=100)

| Path | Before | After | Speedup |
|------|--------|-------|---------|
| **flash f32** | **3.5 tok/s** | **18.6 tok/s** | **5.3x** |
| std f32 | 10.5 | 10.4 | — |
| flash f16 | 28.6 | 29.7 | 1.04x |
| std f16 | 31.3 | 29.6 | — |

Flash f32 was the primary bottleneck: 3x slower than standard before, now 1.8x faster.

## Root Cause Analysis

### Why flash f32 was slow (3.5 tok/s)

The flash f32 decode path had **44 commit_and_waits per step** from two sources:

1. **CPU KV-cache memcpy** (22 commits/step): Each layer called `commit_and_wait()` to flush GPU GEMM output before CPU `memcpy()` could copy new K/V into the cache. Required because CPU cannot read GPU-written data without a sync barrier.

2. **Per-layer correctness sync** (22 commits/step): Each layer called `synchronize_stream()` after SDPA to prevent non-deterministic numerical drift. Without this barrier, 1408 small MPS GEMMs (64 per layer × 22 layers) accumulated in one command buffer, causing MPS to schedule them non-deterministically.

On top of the 44 commits, each commit_and_wait costs ~0.4ms (fixed CB submission overhead), adding ~18ms/step. But the **dominant cost** was actually the **1408 MPS GEMM ObjC encoding calls** per step: each requires MPSMatrix alloc (×3), MPSMatrixMultiplication alloc, encodeToCommandBuffer, and release. At ~150µs per GEMM, that's ~210ms/step.

## Optimizations Implemented

### Optimization 1: GPU Blit Copy for KV Cache

**File**: `src/ops/flash_attention_metal.mm`

For f32 MPS (where `force_layer_rope` means no CPU RoPE is needed), replaced:
```
commit_and_wait() → CPU memcpy(K/V to cache)
```
With:
```
metal::blit_copy(K → cache)  // encode-only, no commit
metal::blit_copy(V → cache)  // encode-only, no commit
```

The blit encoder copies between MTLBuffers within the same command buffer — no CPU roundtrip, no commit needed. Both source (linear GEMM output) and destination (cached K/V) are in MetalAllocator-managed shared memory.

**Impact**: Eliminated 22 commit_and_waits per step (one per transformer layer).

**Scope**: f32 MPS only. f16/bf16 still use the CPU path because they need `commit_and_wait()` for CPU RoPE anyway (the `need_rope` flag controls which path is taken).

### Optimization 2: Fused MSL SDPA Decode Kernel

**Files**: `src/metal/msl_strings.h`, `src/metal/ops_sdpa.mm`

For decode (seqlen_q=1), replaced the per-head loop:
```
for each head (32):
  MPS GEMM: scores = Q_h @ K_hk^T     // ObjC alloc + encode
  GPU kernel: softmax(scores)
  MPS GEMM: out_h = scores @ V_hk      // ObjC alloc + encode
// Total: 64 MPS GEMMs per layer, 1408 per step
```
With a single fused MSL compute kernel:
```
dispatch_fused_sdpa_decode<float>(...)  // 1 kernel, all heads
// Total: 1 dispatch per layer, 22 per step
```

The kernel `fused_sdpa_decode_float` processes one (batch, head) pair per threadgroup:
1. **Scores**: Each thread computes `scale * dot(Q_h, K_hk[j])` for assigned sk positions
2. **Softmax**: Parallel max/sum reduction in threadgroup memory (256-wide tree reduction)
3. **Output**: Each thread computes `sum_j prob[j] * V_hk[j, d]` for assigned head_dim positions

Threadgroup memory holds softmax scores as `float[seqlen_k]`, limiting max sk to 8192 (32KB / 4 bytes). Falls back to per-head MPS GEMM for larger sequences.

Dispatch: `MTLSizeMake(batch_size, num_heads, 1)` with threadgroup size 256.

**Impact**: Eliminated 1408 MPS GEMM ObjC calls per step. The single kernel dispatch has negligible encoding overhead compared to 64 MPS GEMM setups per layer.

**Scope**: f32 only. f16 keeps the existing CPU SDPA path (identical numerics to standard, zero beam-search regression). See "F16 Optimization Opportunities" below.

## Correctness

### Greedy (beam=1): 100% exact match

| Type | Prompt "Hello, world" | Prompt "The quick brown fox" | Prompt "<s>" | Prompt "Once upon a time" |
|------|----------------------|------------------------------|--------------|--------------------------|
| f32 | OK | OK | OK | OK |
| f16 | OK | OK | OK | OK |

### Beam search: pre-existing numerical sensitivity

| Type | beam=2 pass rate | Notes |
|------|-----------------|-------|
| f32 | 6/8 (75%) | Same as before optimization; fused kernel uses float accumulation matching MPS GEMM |
| f16 | 4/8 (50%) | Pre-existing; flash and standard have fundamentally different layouts/paths |

The per-layer `synchronize_stream` is still required for f32 correctness. Without it, linear projection MPS GEMMs (Q/K/V/output projections — 4 per layer, 88 total) accumulate across 22 layers and cause non-deterministic drift. The fused SDPA kernel eliminated SDPA-internal drift, but the linear projection GEMMs remain the source.

## F16 Optimization Opportunities

Flash f16 is currently **29.7 tok/s** vs standard f16 **29.6 tok/s** — near parity. The f16 decode path uses:

1. `commit_and_wait()` — flush GPU GEMM output for CPU RoPE access (22/step)
2. CPU RoPE — apply rotary embeddings on CPU (~0.5µs, negligible)
3. CPU memcpy — write K/V into cache
4. `commit_and_wait()` + CPU SDPA — attention computation (22/step)

Total: ~44 commits/step, same structure as old f32 path. However f16 is already fast because:
- CPU SDPA for sq=1 is very cheap (~8µs per layer vs ~210ms for 64 MPS GEMMs)
- The commits are the bottleneck, not the computation

### Potential f16 optimizations (not implemented):

1. **GPU RoPE to eliminate CPU-path commit** (~22 commits saved): The M12.17 GPU RoPE kernel was reverted due to an in-place WAR race condition (partner elements d±half_dim read while simultaneously overwritten). A fixed version using a two-pass approach or a separate output buffer would allow the entire decode path to stay on GPU:
   - GPU blit copy K/V to cache (encode-only)
   - GPU RoPE on Q and cached K (encode-only, with separate output buffer)
   - Fused SDPA kernel (encode-only)
   - Total: 0 commits from attention (vs 22 currently)

2. **Fused SDPA kernel for f16**: The kernel already has a `fused_sdpa_decode_half` variant compiled in MSL. Currently disabled because it changes f16 accumulation order (GPU kernel uses float32 accumulation, CPU SDPA also uses float32, but different reduction order) causing beam search regression. Could be enabled with an opt-in flag for users who only need greedy decoding.

3. **GPU RoPE + fused SDPA combined**: If both optimizations are applied, f16 flash would have zero attention-related commits per step (vs 44 today). Estimated speedup: **29.7 → 40+ tok/s** based on the f32 optimization trajectory (commit elimination was the dominant factor).

### Why f16 optimization was deferred:

- f16 is already at parity with standard (29.7 vs 29.6 tok/s)
- The GPU RoPE race condition fix requires careful design (separate output buffer or two-pass kernel)
- The fused SDPA kernel changes beam search behavior for f16
- f32 was the urgent bottleneck (3.5 tok/s, unusable)

## Files Modified

| File | Changes |
|------|---------|
| `src/ops/flash_attention_metal.mm` | GPU blit copy path (B) for f32 MPS; CPU path (A) preserved for f16 |
| `src/metal/msl_strings.h` | `fused_sdpa_decode_float/half` MSL kernel + `FusedSdpaDecodeParams` struct |
| `src/metal/ops_sdpa.mm` | `dispatch_fused_sdpa_decode()` function + integration into `sdpa_metal()` entry point |
| `src/layers/flash_attention.cc` | Updated per-layer sync comment documenting remaining necessity |

## Architecture

### Decode path flow (f32 MPS, after optimization):

```
Linear projections (GPU GEMM, encode-only)
  → GPU RoPE via force_layer_rope (encode-only)
  → Split heads (reshape, no GPU work)
  → GPU blit copy K/V to cache (encode-only)     ← NEW: replaces commit + CPU memcpy
  → Fused SDPA kernel (encode-only)               ← NEW: replaces 64 MPS GEMMs
  → Combine heads (reshape)
  → Output linear (GPU GEMM, encode-only)
  → synchronize_stream()                          ← 1 commit per layer (correctness)
```

### Decode path flow (f16, unchanged):

```
Linear projections (GPU GEMM, encode-only)
  → GPU RoPE at offset=0 only; offset>0 deferred to CPU
  → Split heads (reshape)
  → commit_and_wait()                             ← needed for CPU RoPE
  → CPU RoPE on Q and K
  → CPU memcpy K/V to cache
  → commit_and_wait() + CPU SDPA                  ← needed for CPU read of Q/K/V
  → Combine heads (reshape)
  → Output linear (GPU GEMM, encode-only)
```

## Key Technical Decisions

1. **f32-only fused kernel**: Applying the fused kernel to f16 would change beam search behavior (different accumulation order from CPU SDPA). Since f16 is already fast, the risk/reward didn't justify it.

2. **Per-layer sync preserved**: Attempted removing it with the fused kernel (hypothesis: per-head GEMM drift was the cause). Disproved — the drift comes from linear projection GEMMs accumulating across layers. The fused kernel eliminated SDPA-internal drift but not projection-level drift.

3. **Threadgroup memory for scores**: Dynamic allocation via `setThreadgroupMemoryLength:` instead of fixed array. Supports any sk up to 8192 (32KB/4 bytes for float). Falls back to per-head MPS GEMM for larger sequences.

4. **beam_size in kernel params**: The fused kernel handles beam search (K/V batch broadcasting via `kv_b = b / beam_size`) natively, matching the per-head loop's behavior.
