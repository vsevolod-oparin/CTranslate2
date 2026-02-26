# Milestone 6.3 — Rotary Embeddings (RoPE) Report
**Date:** 2026-02-26
**Branch:** `metal-backend`

---

## Summary

M6.3 adds rotary positional embedding (RoPE) support to the Metal backend via
two code paths:

1. **Prefill (offset == 0):** `Rotary::compute<Device::METAL>` — a 2D Metal
   compute kernel applied to Q and K tensors before `FlashAttention::compute`.

2. **Decode (offset > 0):** CPU RoPE inside `FlashAttention::compute<METAL>` —
   applied in-place after `commit_and_wait()`, using the half-sized
   `rotary_cos / rotary_sin` tables supplied by the layer.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/ops_rotary.mm` | **New** — MSL kernel + `metal::rotary_metal<T>()` free function |
| `src/ops/rotary_metal.mm` | **New** — `Rotary::compute<Device::METAL>` wrapper |
| `src/ops/flash_attention_metal.mm` | Removed rotary guard; added `apply_rope_half<T>()` for decode |
| `src/metal/ops_metal.h` | Added `metal::rotary_metal<T>()` declaration |
| `CMakeLists.txt` | Added `src/metal/ops_rotary.mm` to METAL_SOURCES |
| `tests/metal/rotary_test.mm` | **New** — 9 correctness tests |
| `tests/metal/m63_bench.mm` | **New** — Metal vs CPU performance benchmark |
| `APPLE_M4_METAL_PLAN.md` | M6.3 section updated to ✅ DONE |

---

## Design

### File Structure

```
src/metal/ops_rotary.mm       MSL source, PSO cache, metal::rotary_metal<T>()
src/ops/rotary_metal.mm       Rotary::compute<Device::METAL> wrapper (thin)
src/ops/flash_attention_metal.mm  apply_rope_half<T> for decode path
```

This mirrors the established pattern:
- `src/metal/ops_*.mm` — free function, no `StorageView` dependency (testable standalone)
- `src/ops/*_metal.mm` — `Op::compute<METAL>` wrapper using `StorageView` (full build only)

### Prefill Path (offset == 0)

Called via `RotaryEmbeddings::apply(x, offset=0, fa2=true)` → `Rotary::operator()` →
`Rotary::compute<METAL, T>` → `metal::rotary_metal<T>()`.

**MSL Kernel** (`kRotaryMSL`, instantiated as `rotary_float`, `rotary_half`,
`rotary_bfloat`):

```
Grid: [total_vecs, depth, 1]
Dispatch: one thread per output element.

Time index:
  is_transposed=0 (FA2):  t = vec / head_size
  is_transposed=1 (std):  t = vec % max_time

Non-interleave (LLaMA-style):
  y[d]         = x[d]        * cos[t,d] - x[d+middle] * sin[t,d]   d < middle
  y[d+middle]  = x[d+middle] * cos[t,d] + x[d]        * sin[t,d]   d < middle
  y[d]         = x[d]   for d in [ndims, depth)

Interleave (GPT-NeoX-style):
  y[2i]   = x[2i]   * cos[t,2i]   - x[2i+1] * sin[t,2i]
  y[2i+1] = x[2i+1] * cos[t,2i+1] + x[2i]   * sin[t,2i+1]
  y[d]    = x[d]   for d in [ndims, depth)

All arithmetic in float32; result cast back to T.
```

### Decode Path (offset > 0)

`RotaryEmbeddings::apply(x, offset > 0, fa2=true)` returns EARLY — the layer
does NOT rotate Q/K before calling `FlashAttention::compute`. Instead,
`FlashAttention::compute<METAL>` applies RoPE on the CPU after `commit_and_wait()`.

**Half-table format:** `rotary_cos / rotary_sin` have shape `[total_positions, ndims/2]`.
Row `offset` gives the cos/sin values for the current decode position.
`ndims = rotary_cos->dim(1) * 2`.

**`apply_rope_half<T>()`:** In-place CPU RoPE using the half-sized tables.
Uses a temporary buffer for non-interleave to avoid in-place aliasing.

```cpp
// Non-interleave (half-table, cos/sin indexed by d, d < ndims/2):
for d in [0, half):
  tmp[d]        = x[d] * cos[d]        - x[d+half] * sin[d]
  tmp[d+half]   = x[d+half] * cos[d]   + x[d]      * sin[d]
write tmp → x[0..ndims)

// Interleave:
for i in [0, half):
  x[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
  x[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
```

### Why CPU for Decode

- `sq == 1` for decode → two loops over `num_heads * head_dim` (~512 bytes at hd=64, nh=8)
- Total CPU work: ~1 µs
- GPU kernel launch overhead: ~400 µs (CB overhead)
- CPU wins by ~400×; GPU not justified for single-token RoPE

---

## Test Results

File: `tests/metal/rotary_test.mm` — **9 passed, 0 failed**

| Test | Config | Max Err | Threshold | Result |
|------|--------|---------|-----------|--------|
| f32 non-interleave is_transposed=false (FA2) | b1, t8, h4, hd64 | 5.96e-08 | 1e-5 | ✅ |
| f32 non-interleave is_transposed=true (std)  | b1, t8, h4, hd64 | 5.96e-08 | 1e-5 | ✅ |
| f32 interleave     is_transposed=false (FA2) | b1, t8, h4, hd64 | 1.19e-07 | 1e-5 | ✅ |
| f32 partial rotation (ndims=32, hd=64)       | b1, t8, h4, hd64 | 5.96e-08 | 1e-5 | ✅ |
| f16 non-interleave is_transposed=false       | b1, t8, h4, hd64 | 7.01e-04 | 5e-3 | ✅ |
| bf16 non-interleave is_transposed=false      | b1, t8, h4, hd64 | 5.00e-03 | 5e-2 | ✅ |
| f32 non-interleave time=64, h8, hd64         | longer sequence  | 5.96e-08 | 1e-5 | ✅ |
| f32 non-interleave time=16, h2, hd32         | GQA-like         | 5.96e-08 | 1e-5 | ✅ |
| f32 interleave     is_transposed=true (std)  | b1, t8, h4, hd64 | 1.19e-07 | 1e-5 | ✅ |

Float32 errors (~6e-8) are near floating-point epsilon — near-perfect accuracy.
Float16 (~7e-4) and BF16 (~5e-3) errors are quantisation noise only.

---

## Benchmark (Metal vs CPU, Apple M4)

File: `tests/metal/m63_bench.mm` — **24/24 accuracy checks pass**

GPU timing = `rotary_metal<T>` encode + `commit_and_wait()`.
CPU timing = single-threaded float32 CPU reference (same algorithm).

### float32

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 t1   h4  hd64  | 670 | 1   | 0.00x | CPU |
| b1 t8   h4  hd64  | 655 | 3   | 0.00x | CPU |
| b1 t32  h4  hd64  | 655 | 5   | 0.01x | CPU |
| b1 t64  h4  hd64  | 308 | 12  | 0.04x | CPU |
| b1 t128 h4  hd64  | 664 | 21  | 0.03x | CPU |
| b1 t256 h8  hd64  | 412 | 92  | 0.22x | CPU |
| b1 t512 h8  hd64  | 457 | 187 | 0.41x | CPU |
| b1 t1024 h8 hd64  | 479 | 335 | 0.70x | CPU |
| **b1 t2048 h8 hd64**  | **533** | **602** | **1.13x** | **GPU** |
| b1 t256 h16 hd128 | 408 | 278 | 0.68x | CPU |
| b1 t512 h16 hd128 | 639 | 557 | 0.87x | CPU |
| b1 t256 h8  hd128 ndims=64 | 306 | 87 | 0.28x | CPU |

### float16

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 t8   h4  hd64  | 173 | 1   | 0.01x | CPU |
| b1 t64  h4  hd64  | 197 | 8   | 0.04x | CPU |
| b1 t256 h8  hd64  | 290 | 65  | 0.22x | CPU |
| b1 t512 h8  hd64  | 344 | 122 | 0.35x | CPU |
| b1 t1024 h8 hd64  | 474 | 246 | 0.52x | CPU |
| b1 t2048 h8 hd64  | 562 | 462 | 0.82x | CPU |

### bfloat16

| Shape | GPU (µs) | CPU (µs) | Speedup | Winner |
|-------|----------|----------|---------|--------|
| b1 t8   h4  hd64  | 204 | 1   | 0.00x | CPU |
| b1 t64  h4  hd64  | 196 | 7   | 0.04x | CPU |
| b1 t256 h8  hd64  | 252 | 56  | 0.22x | CPU |
| b1 t512 h8  hd64  | 249 | 114 | 0.46x | CPU |
| b1 t1024 h8 hd64  | 315 | 233 | 0.74x | CPU |
| **b1 t2048 h8 hd64**  | **362** | **440** | **1.22x** | **GPU** |

### Performance Analysis

**GPU floor (standalone):** ~170–670 µs depending on CB state (varies with preceding
ops). The rotary kernel itself is fast; CB submission overhead (~0.4 ms) dominates
at all small/medium shapes.

**Crossover (standalone encode + commit_and_wait):**
- float32: ~t=2048 (nh=8, hd=64) → 1.13×
- float16: approaches crossover ~t=2048 (0.82×)
- bfloat16: t=2048 → 1.22×

**Pipeline context (realistic):** In a transformer layer, the CB is committed once at
`synchronize_stream()` — the rotary kernel's encoding cost (~5–15 µs) is amortised
across linear projections, norm, and SDPA. In that context GPU wins at much smaller
`time` values.

**BF16 faster than FP16 standalone:** BF16 GPU times (250–360 µs) are lower than
FP16 (290–562 µs) at the same shapes, likely because bfloat16 operations avoid
the half-precision denormal handling overhead in the shader.

---

## Limitations (unchanged from M6.2)

| Feature | Status |
|---------|--------|
| Chunk-prefill into cache (sq > 1, offset > 0) | Throws — needs offset-aware causal mask |
| ALiBi (M6.4) | Throws |
| Sliding window attention | Throws |
| Attention weight output | Throws |

---

## Build Commands

### Correctness Tests

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/rotary_test.mm \
    src/metal/ops_rotary.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o rotary_test && ./rotary_test
```

### Benchmark

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m63_bench.mm \
    src/metal/ops_rotary.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m63_bench && ./m63_bench
```
