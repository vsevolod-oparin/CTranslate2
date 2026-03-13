# M14.1: Metal Float16 Precision Audit

**Date:** 2026-03-13
**Goal:** Catalog every f16 compute path in the Metal backend, identify precision loss sources, and prioritize fixes to close the 1.91 BLEU gap (f16 25.74 vs f32 27.65).

**Reference:** CUDA f16 has zero BLEU loss vs f32 (27.90 vs 27.92 OPUS-MT, 26.77 vs 26.77 OpenNMT-py).

---

## Methodology

Six parallel investigation agents audited the entire Metal codebase:
1. MSL elementwise/broadcast kernels
2. MSL normalization/activation/quantize kernels
3. SDPA and attention pipeline
4. GEMM and transpose dispatch paths
5. All remaining ops (reductions, beam search, rotary, gather, sampling, etc.)
6. C++ layer dispatch code and StorageView type propagation

---

## Complete Precision Map

### SAFE: Operations with f32 Intermediate Precision

| Category | Operations | Precision Behavior | Files |
|----------|-----------|-------------------|-------|
| **GEMM (main)** | All f16×f16→f16 | m≤32: SIMD kernel f32 accum; m>32: half→f32→MPS f32→f32→half | `primitives_gemm.mm` |
| **GEMM (batched GEMV)** | m=1 decode | Per-thread f32 accumulation | `primitives_gemm.mm` |
| **Activations** | exp, log, relu, gelu, gelu_tanh, sigmoid, swish, tanh | `float v = (float)x[gid]; y[gid] = (T)(expr)` — all math in f32 | `activation.metal` |
| **LayerNorm** | Mean, variance, normalize | Threadgroup float32 accumulation, f32 scale+bias | `normalization.metal:33-79` |
| **RMSNorm** | Sum-of-squares, rsqrt, scale | Threadgroup float32 accumulation | `normalization.metal:101-129` |
| **Softmax** | max, exp-sum, normalize | Three-pass, all f32. Threadgroup float32 | `normalization.metal:153-220` |
| **Rotary (RoPE)** | sin/cos rotation | `float xi = float(input[...]); result = xi*cos - partner*sin` all f32 | `ops_rotary.mm` (kRotaryMSL) |
| **Decode RoPE** | In-place rotation | Threadgroup float32 scratch buffer | `ops_rotary.mm` (kDecodeRopeMSL) |
| **ALiBi** | Position bias add | `(T)(float(input) + float(alibi))` — f32 arithmetic | `ops_alibi.mm` |
| **Quantize** | f16→int8 | `abs(float(x))` for scale, `round(float(x)*scale)` | `quantize.metal:36-68` |
| **Dequantize** | int8→f16 | `T(float(input) / scales[row])` — f32 division | `quantize.metal:90-102` |
| **Dequant GEMM output** | int32→f16 with act | All rescaling, bias, activation in f32 | `quantize.metal:133-169` |
| **Fused norm+GEMV** | LayerNorm+GEMV, RMSNorm+GEMV | Norm in f32 threadgroup, GEMV accum in f32 | `fused_norm_gemm.metal` |
| **SDPA decode (fused)** | sq=1, all heads | Float32 threadgroup memory for scores | `ops_sdpa.mm:603-641` |
| **TopK** | argmax, top-k selection | `float v = (float)row[i]; if (v > best_val)` — f32 comparison | `topk.metal` |
| **Multinomial** | GPU sampling | `float local_sum += float(row[i])` — f32 accumulation | `multinomial_metal.mm` |
| **Mean** | Axis mean | `float sum += static_cast<float>(src[...])` — CPU f32 | `mean_metal.mm` |
| **MedianFilter** | Window median | `std::vector<float> window` — f32 working buffer | `median_filter_metal.mm` |
| **GumbelMax** | Noise sampling | `float z = -std::log(...)` — f32 arithmetic | `gumbel_max_metal.mm` |
| **reduce_amax** | Absolute max | Threadgroup float32 | `reduction.metal:105-123` |
| **reduce_max_element** | Max with index | Float32 comparison | `reduction.metal:143-171` |
| **logsumexp** | Log-sum-exp | CPU, all float: `sum += std::exp((float)x[i] - maxval)` | `primitives_reduction.mm:273-283` |
| **Transpose** | 2D/3D/4D | Bit-exact copy, no arithmetic | `primitives_transpose.mm` |
| **Concat/Split/Slide** | Memory rearrange | GPU blit copy, no arithmetic | `concat_split_slide_metal.mm` |
| **Tile** | Repeat tensor | CPU memcpy, no arithmetic | `tile_metal.mm` |
| **Gather** | Index-based copy | No arithmetic | `ops_norm_gather.mm` |
| **Copy/Fill/Convert** | Memory ops | memcpy or CPU loops | `primitives_memory.mm` |
| **TopPMask** | Top-p masking | CPU std::sort with float cast | `topp_mask_metal.mm` |
| **C++ layer dispatch** | All op wrappers | Generic template `T` flows unchanged | All `ops/*_metal.mm` |
| **StorageView propagation** | Intermediate tensors | `input.dtype()` preserved | `transformer.cc`, `decoder.cc` |

### ISSUE: Operations Computing Directly in Native f16

| Operation | Risk | Calls/Forward | File:Line | Kernel Code | Notes |
|-----------|------|--------------|-----------|-------------|-------|
| **Elementwise Add** | HIGH | ~180-240 | `elementwise.metal:32` | `c[gid] = a[gid] + b[gid]` | Residual connections, scaling |
| **Elementwise Sub** | HIGH | ~10-20 | `elementwise.metal:32` | `c[gid] = a[gid] - b[gid]` | Rare but same pattern |
| **Elementwise Mul** | HIGH | ~120-180 | `elementwise.metal:32` | `c[gid] = a[gid] * b[gid]` | Attention scaling, masking |
| **Scalar Add** | HIGH | ~60-90 | `elementwise.metal:42` | `y[gid] = a + x[gid]` | Bias-like additions |
| **Scalar Mul** | HIGH | ~48-72 | `elementwise.metal:42` | `y[gid] = a * x[gid]` | Post-GEMM scaling (alpha) |
| **Broadcast Add (batch)** | HIGH | ~60-90 | `broadcast.metal:54` | `c[gid] = a[gid%a_size] + b[gid]` | Bias addition |
| **Broadcast Add (depth)** | HIGH | ~18-30 | `broadcast.metal:65` | `c[gid] = a[gid/depth] + b[gid]` | Per-head bias |
| **Broadcast Add (block)** | HIGH | ~30-48 | `broadcast.metal:78` | `c[gid] = a[(gid/block)%a_size] + b[gid]` | Block-level bias |
| **Broadcast Mul** | MEDIUM | ~12-18 | `broadcast.metal:54` | `c[gid] = a[gid%a_size] * b[gid]` | Scaling operations |
| **Min/Max** | LOW | ~12-18 | `elementwise.metal:78` | Ternary selection | No arithmetic, just comparison |
| **SDPA GEMM (QK^T)** | MEDIUM | 48 (8h×6L) | `ops_sdpa.mm:297-299` | `MPSMatrixMultiplication` f16 | K=64, MPS f16 accumulation |
| **SDPA GEMM (AV)** | MEDIUM | 48 (8h×6L) | `ops_sdpa.mm:486-490` | `MPSMatrixMultiplication` f16 | K=seq_len, MPS f16 accumulation |
| **reduce_sum** | LOW | ~5-10 | `reduction.metal:36` | `shmem[tid] += shmem[tid+s]` | Threadgroup f16 accum |
| **reduce_max** | SAFE | ~5-10 | `reduction.metal:81` | Ternary comparison | No arithmetic loss |

---

## Analysis: Why CUDA Has Zero Gap But MPS Has 1.91

### What CUDA Does Differently

1. **GEMM**: cuBLAS uses `CUBLAS_COMPUTE_32F` for f16 — f32 accumulation. **MPS main GEMM: FIXED (M13).**
2. **SDPA GEMM**: cuBLAS also uses f32 accumulation for attention GEMMs. **MPS SDPA: still native f16.**
3. **Elementwise**: CUDA f16 elementwise ops also run in native f16 (half2 SIMD). Yet CUDA has zero BLEU loss.
4. **Conclusion**: The elementwise f16 precision is NOT the primary cause. If CUDA can achieve zero gap with f16 elementwise, so can we.

### Root Cause Narrowing

The remaining 1.91 BLEU gap is most likely dominated by:

1. **SDPA GEMM in native f16** — This is the single biggest remaining difference vs CUDA. CUDA uses f32 accumulation for ALL GEMMs including attention. MPS SDPA deliberately uses native f16 MPS GEMM.
   - 96 attention GEMMs per forward pass (QK^T + AV × 8 heads × 6 layers)
   - The AV product has K=seq_len which can be large (up to 512+ for long sentences)
   - Attention scores feed softmax (which is f32) but the AV product's precision directly affects output quality

2. **Elementwise operations** — Unlikely to be the main cause given CUDA evidence, but could contribute marginally through different error accumulation patterns on different hardware.

3. **reduce_sum in f16** — Minor contributor, rarely on critical path.

---

## Prioritized Fix Plan

### Priority 1: SDPA GEMM f32 Accumulation (Expected: ~1.0-1.5 BLEU recovery)

**What:** Route SDPA f16 GEMMs through f32-accumulation path instead of native MPS f16.

**Options:**
- **A. Use promoted path** (half→f32→MPS f32→f32→half): Most reliable, matches main GEMM. Cost: 2 extra conversion kernels per GEMM × 96 GEMMs/step. Since SDPA matrices are small (m=seq, n=head_dim=64, k=64 or k=seq), the conversion overhead may dominate.
- **B. Use SIMD kernel** (gemm_f16_acc32): Already exists for m≤32. SDPA prefill has m=seq_len which can be large, but we could extend the SIMD kernel threshold. Performance untested for larger m.
- **C. Selective promotion**: Only promote the AV product (where K=seq_len can be large and precision matters more for output) while keeping QK^T native (K=64, errors masked by softmax).

**Recommendation:** Start with Option A (promoted path for both QK^T and AV). Measure BLEU and speed. If speed regression is acceptable (<15%), keep it. If not, try Option C.

### Priority 2: Measure After SDPA Fix (Expected: confirms or disproves elementwise theory)

**What:** After fixing SDPA, re-measure BLEU. If gap closes to <0.5, elementwise is not the issue. If gap remains >0.5, proceed to Priority 3.

### Priority 3: Elementwise f32 Promotion (Only if needed after Priority 2)

**What:** In `elementwise.metal` and `broadcast.metal`, promote operands to f32 for computation:
```metal
// Current:
c[gid] = a[gid] + b[gid];          // native f16

// Proposed:
c[gid] = (T)(float(a[gid]) + float(b[gid]));  // f32 arithmetic
```

**Performance impact:** Elementwise ops are memory-bandwidth-bound (read 2 values, write 1). The extra ALU for f32 add/mul is essentially free on modern GPU ALUs. The only cost is the f16→f32→f16 conversion, which Metal handles in hardware (1 cycle each). Expected regression: <2%.

**Scope:** Only Add and Mul variants (Sub is rare, Min/Max are lossless):
- `add_<T>`, `sub_<T>`, `mul_<T>`, `add_scalar_<T>`, `mul_scalar_<T>`
- `add_batch_broadcast_<T>`, `add_depth_broadcast_<T>`, `add_block_broadcast_<T>`
- `mul_batch_broadcast_<T>`

### Priority 4: reduce_sum f32 Accumulation (Low priority)

**What:** Change `reduce_sum` threadgroup memory from `T` to `float`:
```metal
// Current:
threadgroup T shmem[256];
shmem[tid] += shmem[tid + s];  // f16 accumulation

// Proposed:
threadgroup float shmem[256];
shmem[tid] = (float)input[gid];  // promote on load
// ... reduce in f32 ...
output[0] = (T)shmem[0];  // demote on store
```

**Performance impact:** Doubles shared memory usage (2 bytes→4 bytes per element), but 256×4=1KB is trivial. No measurable throughput impact expected.

---

## Summary Table

| # | Fix | Expected BLEU Recovery | Perf Cost | Confidence |
|---|-----|----------------------|-----------|------------|
| 1 | SDPA GEMM f32 accum | 1.0–1.5 | 5–15% SDPA | HIGH — direct CUDA analogy |
| 2 | Measure gap | — | — | Gate for further work |
| 3 | Elementwise f32 promo | 0.1–0.5 | <2% | LOW — CUDA doesn't need it |
| 4 | reduce_sum f32 accum | <0.1 | ~0% | LOW — rarely on critical path |

**Total expected recovery:** 1.1–2.0 BLEU, bringing f16 to within 0.0–0.8 of f32.

---

## Files Audited (Complete List)

### MSL Kernels
- `src/metal/kernels/elementwise.metal` — binary ops, scalar ops
- `src/metal/kernels/broadcast.metal` — batch/depth/block broadcast
- `src/metal/kernels/activation.metal` — all activations (f32 promoted)
- `src/metal/kernels/normalization.metal` — LayerNorm, RMSNorm, softmax (f32)
- `src/metal/kernels/reduction.metal` — sum, max, amax, max_element
- `src/metal/kernels/quantize.metal` — quantize, dequantize, dequant_gemm_output (f32)
- `src/metal/kernels/sdpa.metal` — causal mask
- `src/metal/kernels/metal_math.metalh` — ct2_safe_tanh, ct2_erf
- `src/metal/kernels/fused_norm_gemm.metal` — fused norm+GEMV (f32)
- `src/metal/kernels/topk.metal` — argmax, topk (f32 comparison)

### C++ Metal Primitives
- `src/metal/primitives_gemm.mm` — all GEMM dispatch paths
- `src/metal/primitives_elementwise.mm` — elementwise dispatch + activation
- `src/metal/primitives_reduction.mm` — reduction dispatch
- `src/metal/primitives_memory.mm` — copy, fill, convert
- `src/metal/primitives_transpose.mm` — transpose dispatch
- `src/metal/primitives_beam_search.mm` — beam search ops

### Metal Ops (Low-Level)
- `src/metal/ops_sdpa.mm` — SDPA implementation
- `src/metal/ops_norm_gather.mm` — norm and gather
- `src/metal/ops_rotary.mm` — rotary embeddings
- `src/metal/ops_alibi.mm` — ALiBi

### Op Dispatch (C++ Wrappers)
- `src/ops/bias_add_metal.mm`
- `src/ops/normalization_metal.mm`
- `src/ops/gather_metal.mm`
- `src/ops/flash_attention_metal.mm`
- `src/ops/rotary_metal.mm`
- `src/ops/alibi_add_metal.mm`
- `src/ops/quantize_metal.mm`
- `src/ops/dequantize_metal.mm`
- `src/ops/concat_split_slide_metal.mm`
- `src/ops/tile_metal.mm`
- `src/ops/topk_metal.mm`
- `src/ops/topp_mask_metal.mm`
- `src/ops/mean_metal.mm`
- `src/ops/median_filter_metal.mm`
- `src/ops/gumbel_max_metal.mm`
- `src/ops/multinomial_metal.mm`

### Layer Code
- `src/layers/transformer.cc`
- `src/layers/decoder.cc`
- `src/layers/flash_attention.cc`
- `src/layers/common.cc`
- `include/ctranslate2/padder.h`
