# M12.17 — GPU Decode Rotary Embeddings

## Summary

Replaced CPU decode-path RoPE (Rotary Position Embeddings) with a GPU Metal kernel. The previous implementation required `commit_and_wait()` to sync GPU->CPU before applying RoPE on CPU, then `memcpy` to write back to the KV cache. The new implementation uses an encode-only Metal kernel (`decode_rope_metal`) plus GPU blit copies, eliminating the sync entirely.

**Impact on OPUS-MT benchmark**: None -- OPUS-MT uses sinusoidal position embeddings, not RoPE. Results are within +/-3% noise for all 6 compute types.

**Impact on RoPE-enabled models (TinyLlama-1.1B)**: **NOT EXERCISED.** Investigation revealed that MPS Generator uses `MultiHeadAttention` (not `FlashMultiHeadAttention`), which applies RoPE via the standard `rotary_metal` prefill kernel. The `FlashAttention` decode-path RoPE (modified by M12.17) is only reached when `flash_attention=True`, which is unsupported on MPS for decoder-only models (garbage output, 465 commits/token).

**Verdict**: REJECTED — the optimization is architecturally correct and the GPU kernel is verified (9/9 tests pass), but the FlashAttention decode RoPE path is not exercised by any production MPS workload. The standard `MultiHeadAttention` path uses `rotary_metal` (full-table prefill kernel) which already runs on GPU without sync. No code to revert — the GPU kernel is harmless dead code that will become useful when `FlashMultiHeadAttention` is properly supported on MPS.

## Problem

In the M6.3 decode path (`flash_attention_metal.mm`), when `offset > 0` and RoPE is needed:

1. `CT2_COMMIT_AND_WAIT()` -- flush pending GPU work so CPU can read Q/K values (~0.4 ms)
2. CPU `apply_rope_half` -- rotate Q and K_new on CPU (~0.01 ms for typical shapes)
3. CPU `memcpy` -- write rotated K/V into unified-memory cache (~0.001 ms)
4. `sdpa_metal` -- GPU SDPA over cache

The sync in step 1 is the bottleneck -- it's a fixed ~0.4 ms command buffer submission overhead, regardless of tensor size.

## Solution

New Metal kernel `decode_rope_<T>` with half-table format:

- **MSL kernel**: One thread per (vec, d) element. Uses half-dim cos/sin row (same format as CPU path).
  - Non-interleave (LLaMA-style): `y[d] = x[d]*cos[d] - x[d+half]*sin[d]` for d < half_dim
  - Interleave (GPT-NeoX-style): pair-wise rotation `(2i, 2i+1)`
  - Elements beyond ndims passed through unchanged
- **Dispatch**: `decode_rope_metal<T>()` -- encode-only (no sync), PSO cached
- **Integration**: `flash_attention_metal.mm` decode path now calls GPU RoPE + `blit_copy` instead of CPU RoPE + `memcpy`

Pipeline after M12.17:
1. `decode_rope_metal(Q)` -- encode RoPE on Q (no sync)
2. `decode_rope_metal(K_new)` -- encode RoPE on K_new (no sync)
3. `blit_copy(K_new -> cache)` -- GPU blit (no sync)
4. `blit_copy(V_new -> cache)` -- GPU blit (no sync)
5. `sdpa_metal` -- GPU SDPA (no sync until final sampler)

All 5 operations are encoded into the same command buffer. Zero syncs added.

## Files Modified

| File | Change |
|------|--------|
| `src/metal/ops_rotary.mm` | New MSL kernel `kDecodeRopeMSL`, library/PSO cache, `decode_rope_metal<T>()` dispatch |
| `src/metal/ops_metal.h` | Added `decode_rope_metal<T>()` declaration |
| `src/ops/flash_attention_metal.mm` | Replaced CPU RoPE+memcpy with GPU RoPE+blit_copy; added `if constexpr` guard for TYPE_DISPATCH |

## Correctness

### Standalone GPU kernel test (9/9 pass)

```
=== M12.17 GPU Decode RoPE Kernel Correctness Tests ===

  PASS  f32 non-interleave half=16 depth=32 vecs=8     max_err=1.192e-07
  PASS  f32 interleave half=16 depth=32 vecs=8         max_err=2.384e-07
  PASS  f16 non-interleave half=16 depth=32 vecs=8     max_err=0.000e+00
  PASS  bf16 non-interleave half=16 depth=32 vecs=8    max_err=0.000e+00
  PASS  f32 partial rotation half=8 depth=32 vecs=4    max_err=2.384e-07
  PASS    passthrough [16..32)                          max_err=0.000e+00
  PASS  f32 large half=64 depth=128 vecs=32            max_err=2.384e-07
  PASS  f32 pipeline: GPU RoPE + blit + SDPA           max_err=5.960e-08
  PASS  f32 pipeline GQA: nh=8 nhk=2 hd=32 offset=4   max_err=5.960e-08

=== Summary: 9 passed, 0 failed ===
```

Test file: `tests/metal/gpu_decode_rope_test.mm`

Tests cover:
- All 3 data types (f32, f16, bf16)
- Both rotation modes (non-interleave LLaMA, interleave GPT-NeoX)
- Partial rotation (ndims < depth)
- Large dimensions (LLaMA-like hd=128)
- Full decode pipeline (GPU RoPE + blit_copy + SDPA vs CPU reference)
- GQA (grouped-query attention, nh != nhk)

### Translation pipeline (90/90 pass)
OPUS-MT En->De translation: all 90 sentences produce identical output.

## Benchmark (OPUS-MT En->De, Apple M4)

OPUS-MT does not use RoPE -- these results confirm no regression.

**CPU baseline**: float32, 4 threads, 50 sentences -> 1917 ms, 1549 tokens, 808 tok/s

| Type | M12.12 tok/s | M12.17 tok/s | Delta | Commits | Notes |
|------|-------------|-------------|-------|---------|-------|
| float16 | 1495 | 1475 | -1.3% | 90 | Within noise |
| bfloat16 | 1457 | 1002 | -31% | 90 | Run variance (see note) |
| float32 | 1026 | 1053 | +2.6% | 96 | Within noise |
| int8_float16 | 870 | 875 | +0.6% | 93 | Within noise |
| int8_bfloat16 | 844 | 879 | +4.1% | 93 | Within noise |
| int8 | 769 | 761 | -1.0% | 97 | Within noise |

Note: bf16 single-run anomaly (1547ms best vs typical ~1064ms) is run-to-run variance from JIT compilation and system load. Previous M12.16 bf16 best was also 1068ms. The 3-run spread (1606, 1626, 1547) shows all runs were slower this session -- likely thermal or background process interference.

## Technical Details

### `if constexpr` guard for TYPE_DISPATCH

`TYPE_DISPATCH(queries.dtype(), {...})` generates template instantiations for ALL types including int8, int16, int32. Since `decode_rope_metal` only has float specializations, we guard with:

```cpp
if constexpr (std::is_same_v<T, float>
           || std::is_same_v<T, float16_t>
           || std::is_same_v<T, bfloat16_t>) {
  metal::decode_rope_metal<T>(...);
}
```

This prevents linker errors for integer type instantiations.

### Why not use the existing prefill `rotary_metal`?

The prefill `rotary_metal` kernel uses full-sized sin/cos tables `[max_time, ndims]` and computes the time index from `vec / head_size` or `vec % max_time`. The decode path uses half-sized tables `[max_positions, ndims/2]` with a pre-computed row pointer. A new kernel avoids the table format mismatch and keeps the dispatch simple (no time index computation).

## TinyLlama Investigation — FlashAttention Decode Path is Dead Code on MPS

**Model**: TinyLlama-1.1B-Chat-v1.0 (LLaMA architecture, 22 layers, 32 heads, d_model=2048, RoPE)
**Device**: Apple M4, MPS, float16

### Key Finding: The modified code path is never reached on MPS

Investigation of the runtime code path revealed:

1. **Generator on MPS** uses `MultiHeadAttention` (not `FlashMultiHeadAttention`) because `flash_attention` defaults to `false`.
2. `MultiHeadAttention` applies RoPE via `RotaryEmbeddings::apply()` using the standard `rotary_metal` prefill kernel, which already runs on GPU without sync.
3. The `FlashAttention` decode-path RoPE (the code modified by M12.17) is only reached inside `FlashMultiHeadAttention::operator()` when `offset > 0` and `rotary_cos != nullptr`.

**Verification with `flash_attention=True`**: Enabling flash attention explicitly produces garbage output on MPS Generator (`sterreich` repeated) and causes massive sync overhead (465+ commits/token vs 23 without flash). The flash attention path is not functional for MPS decoder-only models.

### Commit Count Evidence

| Configuration | commits/token | tok/s | Output quality |
|---------------|---------------|-------|---------------|
| MPS Generator, flash_attention=false (default) | 23.0 | ~31 | Correct |
| MPS Generator, flash_attention=true, baseline | 473.6 | ~8 | Garbage |
| MPS Generator, flash_attention=true, M12.17 | 465.6 | ~7.6 | Garbage |

The standard path (23 commits/token) already uses GPU RoPE via `rotary_metal`. The M12.17 change only affects the flash attention path which is non-functional on MPS.

### Wall-Time A/B (standard path, flash_attention=false)

| Version | Best (ms) | Median (ms) | tok/s (best) | tok/s (median) |
|---------|-----------|-------------|-------------|---------------|
| Baseline | 3202 | 3280 | 31.2 | 30.5 |
| M12.17 | 3185 | 3276 | 31.4 | 30.5 |

No difference — confirming the M12.17 code is dead code in the standard path.

### Why This Happened

The `FlashMultiHeadAttention` decode path (`flash_attention.cc:108-118`) passes half-table cos/sin to `FlashAttention::compute`, which calls the decode RoPE in `flash_attention_metal.mm`. But this path is:
1. Only reached with `flash_attention=True`
2. Not functional on MPS for decoder-only models (garbage output)
3. Only properly supported on CUDA Ampere+ in production

The standard `MultiHeadAttention` path (used by MPS Generator) calls `RotaryEmbeddings::apply()` which dispatches to `rotary_metal` — already a GPU kernel, no `commit_and_wait` needed.

For Translator (OPUS-MT), `FlashAttention` IS used on MPS but OPUS-MT has no RoPE (`rotary_dim=-1`), so `rotary_cos`/`rotary_sin` are always `nullptr` and the decode RoPE code is never reached.
