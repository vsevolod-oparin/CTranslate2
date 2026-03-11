# M12.15 — BiasAdd+Act+Residual Fusion & GPU Nucleus Sampling: REJECTED

**Date**: 2026-03-11
**Status**: REJECTED — no measurable performance gain; all code reverted

---

## 1. BiasAdd+Activation+Residual Fusion

### Motivation
In the Transformer FFN, `BiasAdd` is called with combinations of:
- **_ff1**: bias + activation (ReLU/GELU/Swish/etc.), no residual
- **_ff2**: bias + residual, no activation
- **Attention output**: bias + residual, no activation

The unfused path dispatches 2-3 separate GPU kernels per call:
1. `add_batch_broadcast` (bias)
2. `get_activation_op()` (activation, if present)
3. `Add()` (residual, if present)

Each Metal dispatch costs ~3-5 µs on Apple M4. Fusing into a single kernel eliminates 1-2 dispatches per BiasAdd call.

### Implementation

**MSL kernel** (`msl_strings.h`): `fused_bias_act_res_<T>` — single kernel with:
- 4 buffers: bias, value, output, residual
- 3 uint constants: bias_size, act_type (0=none, 1=ReLU…7=Sigmoid), has_residual
- `apply_act()` switch function with all 8 activation types including `ct2_safe_tanh()`
- Defined for float, half, bfloat types

**Dispatch function** (`primitives_elementwise.mm`): `metal::dispatch_fused_bias_act_res<T>()`
- Lazy-compiled library via `compile_library_once`
- PSO cached via `PSOCache::get()`
- Encode-only pattern (no commit)

**Routing** (`bias_add_metal.mm`): Routes to fused path when:
- Last-axis bias (most common in Transformers)
- AND (activation OR residual present)
- Falls back to unfused path for bias-only or non-last-axis

### Benchmark Results (50 sentences, beam=4, best-of-3)

| Type | M12.12 baseline (tok/s) | M12.15 fused (tok/s) | Delta |
|------|------------------------|---------------------|-------|
| float16 | 1495 | 1452 | -2.9% |
| float32 | 1026 | 1038 | +1.2% |
| int8 | 769 | 748 | -2.7% |
| int8_float16 | 870 | 861 | -1.0% |
| bfloat16 | 1457 | 1428 | -2.0% |
| int8_bfloat16 | 844 | 867 | +2.7% |

All results within ±3% noise — **no measurable improvement**.

### Why No Improvement

Each fused dispatch saves 1-2 kernel launches × ~3-5 µs = ~3-10 µs per BiasAdd call.
Per decode step: ~8 BiasAdd calls × ~5 µs savings = ~40 µs.
Per 50-sentence benchmark: ~90 steps × 40 µs = **~3.6 ms total savings**.
Total runtime: 1000-2000 ms → savings are **0.2-0.4%**, well below measurement noise.

The bottleneck is GEMM compute time, not dispatch overhead. The encode-only dispatch pattern already amortizes kernel launch costs within the command buffer.

### Decision
**REJECTED** — complexity (new MSL kernel + dispatch function + routing logic) not justified for <0.5% theoretical savings. All code reverted.

---

## 2. GPU Nucleus Sampling (TopP)

### Analysis

Nucleus sampling (`TopPMask`) is only active when:
1. `RandomSampler` is used (requires `sampling_topk > 1` or `temperature > 0`)
2. AND `sampling_topp < 1.0`

The beam search benchmark uses `BestSampler` (equivalent to TopK k=1), so `TopPMask` is **never invoked** during the standard benchmark.

### Current Implementation (`src/ops/topp_mask_metal.mm`)
- CPU-based: `commit_and_wait()` → `std::sort` (descending by probability) → cumulative sum → mask
- The sort is O(V log V) where V = vocab_size (32768 for OPUS-MT)
- Only fires with explicit `sampling_topp < 1.0` parameter

### Why Not Implemented
1. **Not exercised by benchmark**: Beam search uses `BestSampler`, not `RandomSampler`
2. **Marginal impact**: Even in sampling mode, TopP is one step in a long pipeline; the sort is ~5ms once per step, negligible vs ~10-20ms GEMM time per step
3. **Complexity**: GPU radix sort + parallel prefix sum for cumulative probabilities requires significant MSL code
4. **No user demand**: The primary use case (translation with beam search) doesn't use nucleus sampling

### Decision
**NOT IMPLEMENTED** — not applicable to the primary benchmark, and marginal theoretical impact even in sampling mode.

---

## Summary

| Optimization | Status | Impact |
|-------------|--------|--------|
| BiasAdd+Act+Residual fusion | **REJECTED** | ±3% noise, <0.5% theoretical savings |
| GPU nucleus sampling | **NOT IMPLEMENTED** | Not exercised by beam search benchmark |

**Performance unchanged from M12.12.**
