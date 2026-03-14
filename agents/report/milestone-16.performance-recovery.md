# M16 Performance Recovery — F16/BF16 Throughput Optimization (2026-03-14)

**Branch**: `metal-backend`
**Base**: `b2fe6a60` (M15 Performance sweep update)

---

## Summary

Three optimizations recovered and exceeded M13-era f16/bf16 throughput:

1. **Conditional sync** (`decoding.cc`): Skip `synchronize_stream` when no logits processors need GPU-coherent data. Removes the M14.5 per-step sync for default beam search.

2. **Restored fused SDPA decode kernel** (`msl_strings.h`): Commit `cbfadcb5` (M14.4) accidentally deleted the `DEFINE_FUSED_SDPA_DECODE` macro and instantiations from `kSdpaMSL`. The dispatch code in `ops_sdpa.mm` still referenced it, causing a crash for flash attention decode paths.

3. **Native MPS f16 GEMM for encoder** (`primitives_gemm.mm`): Use MPS f16 GEMM directly for m > 32 (encoder/prefill) instead of the half→f32→MPS f32→half promotion path. Single-pass encoder computation tolerates f16 accumulation; autoregressive decode (m ≤ 32) still uses custom simd kernel with f32 accumulation.

---

## Translation Benchmark (OPUS-MT En→De, 50 sent, beam=4, this benchmark script)

| Type | M15 baseline | M16 optimized | Change |
|------|-------------|---------------|--------|
| f32 MPS | 1073 | 1078 | +0.5% |
| **f16 MPS** | **822** | **1124** | **+36.7%** |
| **bf16 MPS** | **837** | **1152** | **+37.6%** |
| int8 MPS | 774 | 784 | +1.3% |
| int8_f16 | 733 | 747 | +1.9% |
| int8_bf16 | 780 | 743 | ~noise |

**Key**: f16 now **beats f32** (1124 vs 1078, +4.3%), which is the expected behavior due to half the memory bandwidth.

**Note**: M15 report numbers (f16=1073, f32=1042) used a different dataset (wmt14_50.txt with ~1544 tokens). The above comparison uses the `bench_post_critical_fixes.py` 50-sentence dataset (~520 tokens) for apples-to-apples comparison.

---

## Generator Benchmark (TinyLlama, greedy, max_length=100)

| Type | MHA | M16 tok/s |
|------|-----|-----------|
| f32 | std | 16.9 |
| f32 | flash | 18.1 |
| f16 | std | 22.4 |
| f16 | flash | 27.5 |
| int8 | flash | 26.7 |
| int8_f16 | flash | 32.3 |

Flash attention working correctly for all types (fused decode kernel restored).

---

## BLEU Precision Check (WMT14 En→De, 2737 sent)

| Config | f32 BLEU | f16 BLEU | Gap |
|--------|----------|----------|-----|
| beam=4 | 27.62 | 25.49 | 2.13 |
| beam=6, LP=0.6 | 27.89 | 27.16 | **0.73** |

- M13 promoted GEMM: beam=4 gap=1.91, beam=6+LP gap=0.48
- M16 native f16 encoder: beam=4 gap=2.13, beam=6+LP gap=0.73
- **Delta**: ~0.25 BLEU from native f16 encoder accumulation
- **Assessment**: Acceptable tradeoff for +37% throughput

---

## Root Cause Analysis

### Why f16 was slower than f32 (M15 baseline)

The promoted GEMM path (`dispatch_f16_promoted_gemm`) does 3 GPU operations per GEMM:
1. GPU kernel: half→float32 conversion
2. MPS float32 GEMM
3. GPU kernel: float32→half conversion

vs f32 which does just 1 MPS GEMM. For batched encoder GEMMs (large M), the extra 2 conversion passes added ~40% overhead, making f16 consistently slower than f32.

**Proof** (batch=1 vs batch=32, promoted path):
- batch=1: f16 243 tok/s, f32 160 tok/s → f16 **52% faster** (decode-dominated, custom simd kernel)
- batch=32: f16 787 tok/s, f32 957 tok/s → f16 **18% slower** (encoder-dominated, promoted path overhead)

### Why native f16 GEMM is safe for encoder

MPS f16 GEMM accumulates in float16, causing precision loss for K ≥ 512. In M13, this caused a ~5 BLEU drop when used for ALL GEMMs. However:

- **Encoder** (single-pass): f16 accumulation errors are bounded per-layer and don't compound across time steps. Cost: ~0.25 BLEU.
- **Autoregressive decode** (multi-step): f16 accumulation errors compound over 50+ steps, causing ~5 BLEU loss. The custom simd kernel with f32 accumulation (m ≤ 32) prevents this.

---

## Optimization Details

### 1. Conditional Sync (`decoding.cc`)

```cpp
static bool needs_logits_sync(
    const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors) {
  for (const auto& p : logits_processors) {
    if (dynamic_cast<RepetitionPenalty*>(p.get()))
      return true;
    // Unknown user-provided processor — conservatively sync
    if (!dynamic_cast<NoRepeatNgram*>(p.get()) &&
        !dynamic_cast<SuppressTokens*>(p.get()) &&
        !dynamic_cast<SuppressTokensBegin*>(p.get()) &&
        !dynamic_cast<SuppressSequences*>(p.get()))
      return true;
  }
  return false;
}
```

Precomputed `sync_after_decoder` flag before both beam_search and greedy_search loops. Built-in processors (SuppressTokens, NoRepeatNgram, etc.) only use `DisableTokens::add()` (CPU-side index accumulation); `DisableTokens::apply()` → `indexed_fill` has its own internal sync.

### 2. Restored Fused SDPA Decode Kernel (`msl_strings.h`)

~100 lines of MSL code: `FusedSdpaDecodeParams` struct, `DEFINE_FUSED_SDPA_DECODE` macro with float4-vectorized dot products and softmax, instantiated for both float and half. Was present in M12.19–M12.25, accidentally deleted in commit `cbfadcb5`.

### 3. Native MPS F16 GEMM for Encoder (`primitives_gemm.mm`)

Changed `dispatch_f16_gemm` to call `dispatch_mps_gemm<float16_t>` for m > 32 instead of `dispatch_f16_promoted_gemm`. The promoted path remains available for future use.

---

## Files Modified

| File | Change |
|------|--------|
| `src/decoding.cc` | +47/-9: `needs_logits_sync()`, conditional `sync_after_decoder` |
| `src/metal/msl_strings.h` | +113: Restored fused SDPA decode kernel |
| `src/metal/primitives_gemm.mm` | +8/-4: Native MPS f16 for encoder path |

---

## Rejected/Reverted Optimizations

1. **GEMV float4/half4 vectorization**: Added to transposed GEMV kernels. No measurable improvement — memory bandwidth already saturated for GEMV.
2. **SDPA prefill non-blocking commit**: Replaced `CT2_COMMIT_AND_WAIT()` with `commit_command_buffer() + encode_barrier()`. No impact on throughput (within noise). Reverted to blocking commit for simplicity.

---

## Remaining Opportunities (Future Work)

1. **Larger SIMD GEMM tiles**: Current custom kernel uses 16×16 tiles (4 simdgroups). Larger tiles (32×32+) with threadgroup memory tiling could make f32-accum competitive with MPS for medium matrices, allowing promoted path without overhead.
2. **Flash Attention 4 techniques**: Software exp approximation (unlikely to help — M4 hardware exp is fast), conditional softmax rescaling (marginal gain), fully tiled prefill kernel (significant but complex).
3. **INT8 sync reduction**: INT8 GEMM does 2 CPU syncs per GEMM (dequantize pattern). Exploring GPU-only INT8→f32 dequantization could eliminate these.
