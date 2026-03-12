# FlashMHA Code Review — Metal Backend

**Date**: 2026-03-12
**Scope**: Full code review of FlashMultiHeadAttention on MPS
**Branch**: `metal-backend`

## Files Reviewed

| File | Lines | Role |
|---|---|---|
| `src/layers/flash_attention.cc` | 197 | Layer orchestration, RoPE routing, f32 sync barrier |
| `src/ops/flash_attention_metal.mm` | 293 | MPS compute: CPU RoPE, blit copy, SDPA dispatch |
| `src/metal/ops_sdpa.mm` | 763 | SDPA kernels: per-head MPS, BF16 MPSGraph, fused decode, CPU fast-path |
| `src/metal/kernels/sdpa.metal` | 56 | Causal mask MSL kernels |
| `src/metal/msl_strings.h:1070-1162` | 92 | Fused SDPA decode MSL kernel |
| `include/ctranslate2/ops/flash_attention.h` | 47 | Op interface |
| `include/ctranslate2/layers/flash_attention.h` | 53 | Layer interface |
| `tests/metal/sdpa_test.mm` | 407 | SDPA unit tests (12 cases) |
| `tests/metal/flash_mha_decode_test.mm` | 853 | Decode path tests (6 cases) |
| `src/layers/attention_layer.cc` | (force_layer_rope) | RotaryEmbeddings::apply routing |

---

## CRITICAL — Potential Bugs / Data Corruption

### C1. Stack buffer overflow in `apply_rope_half` (latent)

`flash_attention_metal.mm:96`:
```cpp
float tmp[512];  // generous bound: head_dim never exceeds 512
```

This buffer holds `2 * half` elements where `half = ndims / 2`. For `ndims == head_dim`, this means `head_dim` entries. So `tmp[512]` supports head_dim up to 512. Current models (TinyLlama hd=64, GPT-2 hd=64, Llama-2 hd=128) are safe, but models with head_dim > 512 (some research models) would corrupt the stack silently. Should be a `std::vector` or at minimum a bounds check.

**Severity**: Low today (no production model hits it), high if new models are added without review.

### C2. Fused SDPA decode `tg_reduce[256]` hardcoded

`msl_strings.h:1120`:
```metal
threadgroup float tg_reduce[256];
```

This must match `kTgSize = 256` in `ops_sdpa.mm:547`. The coupling is implicit — changing one without the other causes out-of-bounds threadgroup memory writes and GPU hangs. Should use a constant or static_assert.

### C3. Comment says f16 max sk = 16384, actual limit is 8192

`msl_strings.h:1070`:
```
// Max sk: 32768/sizeof(T) (8192 for float, 16384 for half).
```

The kernel uses `threadgroup float* tg_scores` (always float32 regardless of T), so the limit is always 32768/4 = 8192. The comment is misleading and could cause someone to raise `kFusedSdpaMaxSk` for f16, resulting in a GPU hang from exceeding the 32KB threadgroup memory limit.

---

## HIGH — Missing Protections / Defensive Gaps

### H1. No `protect_buffer` for fused SDPA decode inputs

`ops_sdpa.mm:527-531` — `dispatch_fused_sdpa_decode` looks up Q/K/V/output via `metal_buffer_for_ptr()` and encodes the kernel, but does NOT call `protect_buffer_by_base()` on Q or K/V. Compare with the INT8 GEMM path (`primitives_gemm.mm:758-759`) which protects both A and B.

In practice this is safe because Q is alive through the layer function scope, and K/V are the persistent cache. But if the allocator ever reclaims these buffers before the CB commits (e.g., during a StorageView resize between encode and commit), the GPU would read garbage. This gap violates the project's established pattern.

### H2. `MetalTempBuf` RAII frees before GPU execution

In `sdpa_head_mps`, `MetalTempBuf scores_buf` is freed (returned to pool) when the function returns. But the GPU hasn't executed the encode-only GEMM/softmax kernels yet. If the next head iteration's `MetalTempBuf` allocation recycles the same pool buffer, the new GEMM write would overwrite the old scores.

This is **safe in practice** because Metal serial dispatch guarantees in-order execution within a command buffer — the GPU processes head 0's write-then-read before head 1's write. But it relies on an undocumented invariant (the pool returns the same buffer for same-size sequential alloc/free cycles). A `protect_buffer` call would make this explicit.

### H3. Causal mask offset hardcoded to 0

`ops_sdpa.mm:63`:
```cpp
uint32_t offset = 0u;
```

The MSL kernel supports `col > row + offset`, but the dispatch always passes 0. For the current usage (prefill-only causal mask, decode forces `is_causal=false`), this is correct. But if chunk-prefill into KV cache is ever implemented (the code already guards against it at `flash_attention_metal.mm:170`), the causal mask would be wrong. The offset should be a parameter, not hardcoded.

---

## MEDIUM — Performance Optimizations

### P1. SDPA MPSMatrixMultiplication not cached

`ops_sdpa.mm:194-207` — Each `sdpa_mps_gemm` call creates and destroys a fresh `MPSMatrixMultiplication`. `primitives_gemm.mm` caches these via `get_cached_mps_gemm()` (M11.27), saving ~15us per GEMM.

For prefill: 2 GEMMs/head x 32 heads x 22 layers = 1408 allocs. At ~15us each = **~21ms per prefill wasted on ObjC alloc**. For a typical 200ms prefill, this is ~10% overhead. The fused decode kernel avoids this (no MPS GEMMs), but the prefill path still pays.

**Fix**: Use `get_cached_mps_gemm()` instead of `[[MPSMatrixMultiplication alloc] initWithDevice:...]`.

### P2. SDPA `rowBytesForColumns` not cached

`ops_sdpa.mm:122-126` — Calls `[MPSMatrixDescriptor rowBytesForColumns:dataType:]` every time. `primitives_gemm.mm` caches these (M12.2). Minor but free to fix.

### P3. Fused SDPA decode kernel: inner loop not vectorized

`msl_strings.h:1113`:
```metal
for (uint d = 0; d < p.head_dim; ++d)
    dot += float(q_h[d]) * float(k_row[d]);
```

This is a scalar dot product. For head_dim=64 (typical), a SIMD-group reduction or `float4` vectorized accumulation could significantly reduce ALU cycles. The output loop (line 1153-1157) has the same issue — iterating `j` over seqlen_k inside a `d` loop means poor cache locality for large sk.

### P4. `beam_size` always 1 in FlashMHA

`flash_attention.cc:52`: `dim_t beam_size = 1;` — never updated. The fused SDPA kernel has `kv_b = b / beam_size` beam broadcasting logic, but it's dead code since beam_size is always 1. The cache is per-beam in FlashMHA (unlike standard MHA), so broadcasting isn't needed. This unused parameter adds complexity.

---

## LOW — Code Quality

### Q1. Dead Path C comments

`flash_attention_metal.mm:200-204`:
```cpp
// (C) GPU RoPE + blit path (f16 with need_rope):
//     M12.18: GPU decode_rope_metal (WAR race fixed with threadgroup
//     scratch) applies RoPE on Q/K, then blit_copy writes K/V to cache.
//     All encode-only — zero commits from attention.
```

Path C was attempted but reverted. The comment describes abandoned code but the implementation doesn't exist. This is confusing for future readers.

### Q2. `_offset_free_space{512}` magic number

`flash_attention.h:50`:
```cpp
static constexpr dim_t _offset_free_space{512};
```

This controls KV cache growth chunk size. 512 is undocumented — what's the rationale? Too small wastes time on frequent reallocation; too large wastes memory. Should at minimum have a comment.

### Q3. `fl_attn_ops` variable name

`flash_attention.cc:125`: `ops::FlashAttention fl_attn_ops(...)` — constructed and called immediately. The variable name is unclear. Could just be:
```cpp
ops::FlashAttention(_queries_scale, _sliding_window)(queries_proj, keys_proj, ...);
```

### Q4. Explicit instantiations for int types that throw

`ops_sdpa.mm:752-760` — `sdpa_metal<int8_t>`, `sdpa_metal<int16_t>`, `sdpa_metal<int32_t>` are instantiated just to satisfy `TYPE_DISPATCH` link requirements, but they throw at runtime. A compile-time static_assert or `if constexpr` filter would be cleaner.

---

## Missing Tests

### T1. No fused SDPA decode test with large seqlen_k

The sdpa_test.mm decode tests use sk=16. The fused kernel has a complex threadgroup reduction (max + exp + sum, tree reduction in 256 threads) that's not stressed at sk=16. Test with sk=1024+ and sk near the 8192 limit.

### T2. No beam search test with fused SDPA kernel

The fused kernel has `kv_b = b / beam_size` logic but beam_size is always 1 in all tests. If this code is ever activated with beam_size > 1, it needs coverage.

### T3. No f16 fused SDPA decode test (flash_mha_decode_test.mm)

`flash_mha_decode_test.mm` only tests f32. Given the documented f16 FMA divergence issues, f16 decode tests would catch regressions from future changes to force_layer_rope.

### T4. No test for fused SDPA fallback boundary

When `seqlen_k == 8193`, the code should fall back from fused kernel to per-head MPS GEMM. No test verifies this boundary produces correct results.

### T5. No test for chunk-prefill guard

`flash_attention_metal.mm:170` throws for `seqlen_q > 1 with offset > 0`. No test verifies this guard.

### T6. No sliding window / ALiBi guard tests

The `throw` guards at lines 144 and 148 are untested. While simple, they protect against silent corruption.

---

## Summary

| Category | Count | Top Priority |
|---|---|---|
| Critical (potential data corruption) | 3 | C1 (stack overflow), C3 (misleading comment leading to future GPU hang) |
| High (defensive gaps) | 3 | H1 (missing protect_buffer), P1 (21ms prefill overhead) |
| Medium (performance) | 4 | P1, P3 (SDPA kernel vectorization) |
| Low (code quality) | 4 | Q1 (dead comments) |
| Missing tests | 6 | T1 (large sk), T3 (f16 decode) |

## Recommended Priority Order

1. **P1** — Cache MPSMatrixMultiplication in SDPA (free ~10% prefill speedup)
2. **C1** — Replace `float tmp[512]` with bounds-checked allocation
3. **H1** — Add `protect_buffer_by_base()` in fused SDPA dispatch
4. **C3** — Fix the misleading threadgroup memory comment
5. **T1/T3** — Add large-sk and f16 fused decode tests
