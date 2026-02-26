# Milestone 6.2 — KV-Cache Update Report
**Date:** 2026-02-26
**Branch:** `metal-backend`

---

## Summary

M6.2 adds KV-cache decode support to `FlashAttention::compute<Device::METAL>`.
Prior to this milestone, `offset > 0` threw an exception. After M6.2 the Metal
backend can run multi-step autoregressive decode with a growing KV cache.

---

## Files Changed

| File | Change |
|------|--------|
| `src/ops/flash_attention_metal.mm` | Added `offset > 0` decode path; removed guard exception |
| `tests/metal/kv_cache_test.mm` | New — 6 tests for multi-step decode correctness |
| `APPLE_M4_METAL_PLAN.md` | M6.2 section updated to ✅ DONE |

---

## Design

### KV-Cache Layout

```
cached_keys/values shape: [batch, total_cache_slots, num_heads_k, head_dim]
Row stride (row_elements): num_heads_k * head_dim
Position offset within batch b: b * total_cache_slots * row_elements + offset * row_elements
```

The layer (`FlashMultiHeadAttention`) pre-allocates cache buffers with `_offset_free_space = 512`
extra slots. The op writes into these slots at position `offset` each decode step.

### Algorithm (offset > 0)

```
1.  commit_and_wait()
    ─ Flush the current command buffer and wait.
    ─ Reason: linear projection kernels wrote keys/values into Metal buffers via GPU
      compute shaders. Unified memory makes them CPU-readable only after the CB completes.

2.  CPU memcpy (per batch item)
    for b in [0, batch):
        k_cache[b, offset:offset+seqlen_new, :, :] = keys[b, :, :, :]
        v_cache[b, offset:offset+seqlen_new, :, :] = values[b, :, :, :]
    ─ O(seqlen_new × num_heads_k × head_dim) bytes copied per batch.
    ─ For typical decode (seqlen_new=1, hd=64, nhk=8): 512 bytes — negligible.
    ─ Unified memory: written bytes are immediately GPU-visible; no explicit flush needed.

3.  sdpa_metal(Q, cached_K, cached_V, output,
               seqlen_k = offset + seqlen_new,
               is_causal = false)
    ─ seqlen_k_eff = offset + seqlen_new  (only the valid range, not total_cache_slots).
    ─ is_causal=false: with sq==1, all cache positions [0..offset+seqlen_new-1] are in the
      past of the current query. The causal mask (col > row) would incorrectly mask
      positions 1..seqlen_k-1 when row=0. Setting is_causal=false is correct and mirrors
      the CUDA FlashAttention implementation.
```

### Why CPU memcpy, Not a GPU Kernel

The same reasoning as `prepare_length_mask` (M4.7):
- `seqlen_new == 1` for decode → copy is O(head_dim × num_heads_k × batch) bytes (~512 B typical).
- A GPU kernel launch costs ~0.4 ms in CB overhead; the copy itself takes < 1 µs on CPU.
- CPU wins by a wide margin for this size; GPU kernel only pays off at seqlen_new > ~100K elements.
- A GPU encode-only copy could avoid the `commit_and_wait()` cost, but would require a new kernel
  and is an M6.x performance optimisation, not needed for correctness.

### Causal Masking Decision

| Scenario | seqlen_q | is_causal passed | Reasoning |
|----------|----------|------------------|-----------|
| Prefill (offset=0) | > 1 | `_is_causal` | Standard causal prefix mask |
| Decode (offset>0) | 1 | `false` | All K positions ≤ current pos; mask would break decode |
| Chunk decode (offset>0, sq>1) | > 1 | `throws` | Needs offset-aware causal mask; deferred |

---

## Test Results

File: `tests/metal/kv_cache_test.mm` — **6 passed, 0 failed**

| Test | Config | Steps | Max Err | Threshold | Result |
|------|--------|-------|---------|-----------|--------|
| float32 decode | batch=1, nh=4, nh_k=4, hd=32, prefill=4 | 10 | 2.98e-07 | 1e-4 | ✅ |
| float32 GQA decode | batch=1, nh=4, nh_k=2, hd=16, prefill=4 | 5 | 8.94e-08 | 1e-4 | ✅ |
| float32 batch=2 | batch=2, nh=2, nh_k=2, hd=16, prefill=3 | 5 | 1.19e-07 | 1e-4 | ✅ |
| float16 decode | batch=1, nh=4, nh_k=4, hd=32, prefill=4 | 5 | 4.23e-04 | 0.05 | ✅ |
| bfloat16 decode | batch=1, nh=2, nh_k=2, hd=16, prefill=3 | 3 | 3.49e-03 | 0.1 | ✅ |
| Cache contents | 8 sequential writes, verify slot + no overwrite | — | — | exact | ✅ |

M6.1 SDPA regression: **8/8 tests pass** (offset=0 path unchanged).

### Error Profile

Float32 errors (~1–3e-7) are at floating-point rounding noise level — near-perfect match.
Float16 (~4e-4) and BF16 (~3e-3) errors are consistent with quantisation noise in the
softmax numerics, stable across decode steps.

---

## Benchmark (Metal vs CPU, Apple M4)

File: `tests/metal/m62_bench.mm` — **33/33 accuracy checks pass**

One decode step: `sq=1`, `offset = sk-1` (new token appended, SDPA over `[0, sk)`).
GPU timing = `commit_and_wait` + `memcpy` new K/V + `sdpa_metal` + `commit_and_wait`.
CPU timing = same `memcpy` + single-threaded float32 reference SDPA.

### float32

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 sq1 sk64 nh8 hd64 | 1237 | 19 | 0.02x | CPU |
| b1 sq1 sk128 nh8 hd64 | 630 | 48 | 0.08x | CPU |
| b1 sq1 sk256 nh8 hd64 | 641 | 102 | 0.16x | CPU |
| b1 sq1 sk512 nh8 hd64 | 665 | 223 | 0.34x | CPU |
| b1 sq1 sk1024 nh8 hd64 | 684 | 481 | 0.70x | CPU |
| **b1 sq1 sk2048 nh8 hd64** | **837** | **999** | **1.19x** | **GPU** |
| b1 sq1 sk256 nh16/4 hd64 | 1053 | 208 | 0.20x | CPU |
| b1 sq1 sk512 nh16/4 hd64 | 1070 | 457 | 0.43x | CPU |
| b1 sq1 sk1024 nh16/4 hd64 | 1102 | 980 | 0.89x | CPU |
| b1 sq1 sk128 nh4 hd32 | 427 | 12 | 0.03x | CPU |
| b1 sq1 sk512 nh4 hd32 | 436 | 58 | 0.13x | CPU |

### float16

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 sq1 sk64 nh8 hd64 | 627 | 19 | 0.03x | CPU |
| b1 sq1 sk128 nh8 hd64 | 629 | 51 | 0.08x | CPU |
| b1 sq1 sk256 nh8 hd64 | 630 | 107 | 0.17x | CPU |
| b1 sq1 sk512 nh8 hd64 | 652 | 225 | 0.34x | CPU |
| b1 sq1 sk1024 nh8 hd64 | 665 | 486 | 0.73x | CPU |
| **b1 sq1 sk2048 nh8 hd64** | **687** | **1015** | **1.48x** | **GPU** |
| b1 sq1 sk256 nh16/4 hd64 | 1044 | 212 | 0.20x | CPU |
| b1 sq1 sk512 nh16/4 hd64 | 1057 | 464 | 0.44x | CPU |
| b1 sq1 sk1024 nh16/4 hd64 | 1096 | 989 | 0.90x | CPU |
| b1 sq1 sk128 nh4 hd32 | 427 | 12 | 0.03x | CPU |
| b1 sq1 sk512 nh4 hd32 | 426 | 58 | 0.14x | CPU |

### bfloat16 (synchronous MPSGraph per head)

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 sq1 sk64 nh8 hd64 | 4225 | 19 | 0.00x | CPU |
| b1 sq1 sk128 nh8 hd64 | 4162 | 48 | 0.01x | CPU |
| b1 sq1 sk256 nh8 hd64 | 4216 | 102 | 0.02x | CPU |
| b1 sq1 sk512 nh8 hd64 | 4568 | 223 | 0.05x | CPU |
| b1 sq1 sk1024 nh8 hd64 | 4754 | 481 | 0.10x | CPU |
| b1 sq1 sk2048 nh8 hd64 | 4864 | 1000 | 0.21x | CPU |
| b1 sq1 sk256 nh16/4 hd64 | 8447 | 203 | 0.02x | CPU |
| b1 sq1 sk512 nh16/4 hd64 | 8561 | 447 | 0.05x | CPU |
| b1 sq1 sk1024 nh16/4 hd64 | 9243 | 963 | 0.10x | CPU |
| b1 sq1 sk128 nh4 hd32 | 2298 | 12 | 0.01x | CPU |
| b1 sq1 sk512 nh4 hd32 | 2327 | 59 | 0.03x | CPU |

### Performance Analysis

**GPU floor (float32/float16):** ~630–640 µs CB submission overhead dominates for all
realistic decode sizes. The SDPA kernel itself is fast; GPU starts winning at sk≥2048
for standard MHA (1.19–1.48x). GQA (nh16/nhk4) has higher overhead because the Metal
SDPA internally loops over 16 query heads vs 4 K/V heads, approaching crossover near sk≈2048.

**BF16 decode via MPSGraph:** per-head MPSGraph launches cost ~2.3–4.2 ms per step
(scales with nh). CPU wins at all tested shapes. A dedicated MSL kernel would eliminate
this overhead; deferred to M6.x performance work.

**Real-pipeline context:** The ~0.4 ms CB overhead is amortised across all ops in a
layer (linear projections, norm, decode) when the CB is committed at `synchronize_stream()`.
In that context float32/float16 GPU decode would be competitive at much lower sk than 2048.

### Benchmark Build Command

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m62_bench.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m62_bench && ./m62_bench
```

---

## Limitations (deferred to future milestones)

| Feature | Status |
|---------|--------|
| Chunk-prefill into cache (sq > 1, offset > 0) | Throws — needs offset-aware causal mask |
| Rotary embeddings (M6.3) | Throws |
| ALiBi (M6.4) | Throws |
| Sliding window attention | Throws |
| Attention weight output | Throws |

---

## Build Command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/kv_cache_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o kv_cache_test && ./kv_cache_test
```
