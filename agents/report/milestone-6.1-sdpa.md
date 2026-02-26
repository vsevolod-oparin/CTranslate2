# Milestone 6.1 — Scaled Dot-Product Attention for Device::METAL

**Status: ✅ DONE** (2026-02-25)

## Summary

Implemented `metal::sdpa_metal<T>` (the Metal SDPA primitive) and
`FlashAttention::compute<Device::METAL>` (the high-level op wrapper).

---

## Algorithm

For each (batch b, query head h):

```
hk = h % num_heads_k              (grouped query attention)
S  = scale * Q[b,h] @ K[b,hk]^T  [seqlen_q × seqlen_k]
if is_causal: S[col > row] = large_neg
A  = softmax(S, dim=-1)
O[b,h] = A @ V[b,hk]              [seqlen_q × head_dim]
```

Layout: `[batch, seqlen, num_heads, head_dim]` (interleaved heads).
Row stride for Q: `q_lda = num_heads * head_dim`
Row stride for K/V: `kv_lda = num_heads_k * head_dim`

---

## Implementation Details

### FP32 / FP16 path (`sdpa_head_mps<T>`)
- `MPSMatrixMultiplication` (encode-only), handles non-contiguous strides via MPS rowBytes.
- All four ops (Q@K^T, causal_mask, softmax, scores@V) encode into the same deferred CB.
- **Edge-case fix**: when `seqlen_k * sizeof(T) < MPS_min_rowBytes`, `sdpa_mps_gemm` packs the scores matrix CPU-side (`pad_a=true`). A conditional `commit_and_wait()` is inserted before step 4 to flush causal_mask + softmax on the GPU first. This path only triggers for very small seqlen_k (< 8 for float32, < 16 for float16) — atypical in real transformer inference.

### BF16 path (`sdpa_head_bf16`)
- `MPSGraph` matmul (synchronous), packs Q/K/V slices to contiguous allocator-registered buffers.
- Scale is folded into the Q pack loop (multiply each element by `scale`).
- `sdpa_bf16_gemm` calls `commit_and_wait()` before each GEMM — correctly flushes causal_mask+softmax before the second GEMM.

### Causal mask
- MSL kernel `causal_mask_{float,half,bfloat}` in `src/metal/kernels/sdpa.metal`.
- Dispatch: 1 thread per element, `seqlen_q * seqlen_k` total threads.
- Sets `scores[gid] = large_neg` when `col > row + 0` (offset=0 for M6.1).
- `large_neg`: float → `-1e9f`, half → `half(-65504.0f)`, bfloat → `bfloat(-1e9f)`.

### Scores buffer
- Always allocated via `get_allocator<Device::METAL>()` (RAII `MetalTempBuf` struct).
- Registered in `MetalAllocator::_live` so `metal_buffer_for_ptr()` can locate it for the causal_mask kernel and softmax kernel.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/kernels/sdpa.metal` | New — causal_mask_float/half/bfloat kernels |
| `tools/gen_msl_strings.py` | Added `("sdpa", "kSdpaMSL")` to KERNELS list (9 total) |
| `src/metal/msl_strings.h` | Regenerated — kSdpaMSL constant added |
| `src/metal/ops_metal.h` | Added `metal::sdpa_metal<T>` declaration |
| `src/metal/ops_sdpa.mm` | New — full SDPA implementation |
| `src/ops/flash_attention_metal.mm` | New — `FlashAttention::compute<Device::METAL>` wrapper |
| `CMakeLists.txt` | Added ops_sdpa.mm, flash_attention_metal.mm to METAL_SOURCES; added sdpa.metal to _MSL_METAL_SOURCES |
| `tests/metal/sdpa_test.mm` | New — 8 correctness tests |

---

## M6.1 Scope Limitations

`FlashAttention::compute<Device::METAL>` throws `std::invalid_argument` for:
- `offset > 0` — no KV cache
- `rotary_cos / rotary_sin` — no rotary embeddings
- `alibi` — no ALiBi bias
- `sliding_window > 0` — no sliding window
- `return_normalized_attention && attention` — no attention weight output

---

## Test Results

**8/8 tests pass** (`tests/metal/sdpa_test.mm`):

| Test | Max abs err |
|------|------------|
| float32 non-causal, sq=4 sk=4 h=1 hd=8 | 5.96e-08 |
| float32 causal, sq=4 sk=4 h=1 hd=8 | 5.96e-08 |
| float32 causal, batch=2 sq=3 sk=3 h=2 hd=4 | 1.19e-07 |
| float32 non-causal GQA, nh=4 nhk=2 hd=4 | 5.96e-08 |
| float16 non-causal, sq=4 sk=4 h=1 hd=8 | 2.44e-04 |
| float16 causal, sq=4 sk=4 h=1 hd=8 | 2.44e-04 |
| bfloat16 non-causal, sq=4 sk=4 h=1 hd=8 | 1.95e-03 |
| bfloat16 causal, sq=4 sk=4 h=1 hd=8 | 1.56e-03 |

---

## Key Bug Fixed During Implementation

**Root cause**: In `sdpa_head_mps`, when `seqlen_k * sizeof(T) < MPS_min_rowBytes`
(small matrices), `sdpa_mps_gemm` packs the scores input matrix CPU-side (`pad_a`
path) before the pending causal_mask/softmax GPU encoders have executed. This
causes GEMM 2 to use stale pre-softmax values.

**Fix**: Added a one-time `@autoreleasepool` query of `rowBytesForColumns:seqlen_k`.
If the natural stride is below the MPS minimum, `commit_and_wait()` is called
to flush the GPU before the CPU pack. For typical inference (seqlen_k ≥ 8),
this check is a no-op.
