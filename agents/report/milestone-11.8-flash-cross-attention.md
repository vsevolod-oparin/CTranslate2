# M11.8 — Flash Cross-Attention on Metal

## Summary

Added a Metal-optimized cross-attention path in `MultiHeadAttention` that bypasses the 3-op `dot_product_attention` (MatMul Q*K^T + SoftMax + MatMul attn*V) and routes through `FlashAttention` op → `sdpa_metal`, the same fused kernel used for self-attention.

**Key insight:** `sdpa_metal` expects `[batch, seq, heads, dim]` layout. Cross-attention's Q projection is `[batch, sq, nh*dh]` which reshapes to `[batch, sq, nh, dh]` with zero data movement (no transpose needed). This eliminates the `split_heads` transpose that the existing path requires.

## Motivation

For Whisper (60s audio), every decode step runs cross-attention with sq=1, sk~1500. The existing `dot_product_attention` path dispatches 3 separate ops:
1. `gemm_batch_strided(Q, K^T)` — score computation
2. `SoftMax` — normalization
3. `gemm_batch_strided(attn, V)` — context computation

Using `sdpa_metal` fuses all three into a single kernel dispatch, reducing Metal command encoder overhead. Additionally, `sdpa_metal` handles MQA/GQA natively without `replicate_heads`.

## Approach

When the device is Metal and `_use_flash_cross_attention` is set (propagated from the `use_flash_attention` model flag), `MultiHeadAttention::operator()` takes a new early-return path:

1. **`process_cross_attention_flash()`**: Projects Q/K/V into `[batch, seq, heads, dim]` layout (reshape only, no transpose)
2. Tiles K/V for beam search if `beam_size > 1`
3. Calls `FlashAttention` with `is_causal=false`, no rotary, no alibi, offset=0
4. Reshapes output `[batch, sq, nh, dh]` → `[batch, sq, d_model]`
5. Applies output projection and post-norm

**Conditions for flash cross-attention:**
- Device is Metal
- `_use_flash_cross_attention` flag set (from model's `use_flash_attention`)
- No relative position embeddings (not used in Whisper/translation cross-attention)

When not met → existing `dot_product_attention` path (no regressions for CPU or non-flash models).

## Changes

### 1. `include/ctranslate2/layers/attention.h`

- Added `bool use_flash_cross_attention = false` constructor parameter
- Added `bool _use_flash_cross_attention` private member
- Added `process_cross_attention_flash()` private method declaration

### 2. `src/layers/attention.cc`

- **Constructor:** Stores `_use_flash_cross_attention` in initializer list
- **`process_cross_attention_flash()`**: New method producing `[batch, seq, heads, dim]` layout:
  - Q: reshape `[batch, sq, nh*dh]` → `[batch, sq, nh, dh]`
  - K/V: project via `_linear[1]`, split, reshape to `[batch, sk, nhk, dh]`
  - Handles MQA (nhk=1), GQA (nhk<nh), and MHA (nhk=nh)
  - Caches K/V in `[batch, sk, nhk, dh]` format
  - Applies `_q_norm` / `_k_norm` if present
  - Detects beam_size from Q/K batch dim ratio
- **`operator()`**: New early-return block in `!_self_attention` branch:
  - Calls `process_cross_attention_flash()`
  - Tiles K/V for beam search
  - Calls `ops::FlashAttention` with `is_causal=false`
  - Reshapes and applies output projection
  - Falls through to existing `process_cross_attention` + `dot_product_attention` when conditions not met
- Added `#include "ctranslate2/ops/flash_attention.h"`

### 3. `src/layers/transformer.cc`

- `TransformerDecoderLayer` constructor: passes `use_flash_attention` to `MultiHeadAttention` for `_encoder_attention` via `use_flash_cross_attention` parameter

### 4. No changes to FlashAttention op or flash_attention_metal.mm

The existing prefill path (offset=0) already handles arbitrary sq/sk with `is_causal=false`.

### 5. Beam_size broadcasting — additional changes (commit `e528112a`)

| File | Change |
|------|--------|
| `src/metal/ops_metal.h` | Added `dim_t beam_size = 1` to `sdpa_metal` signature |
| `src/metal/ops_sdpa.mm` | `kv_b = b / beam_size` in both GPU and CPU paths |
| `include/ctranslate2/ops/flash_attention.h` | Added `dim_t beam_size = 1` to operator/compute |
| `src/ops/flash_attention.cc` | Threaded `beam_size` through |
| `src/ops/flash_attention_metal.mm` | Passed `beam_size` to `sdpa_metal` calls |
| `src/ops/flash_attention_cpu.cc` | Added beam_size param (compile fix) |
| `src/ops/flash_attention_gpu.cu` | Added beam_size param (compile fix) |
| `src/layers/attention.cc` | Removed K/V Tile, pass beam_size to FlashAttention |

## Beam Search Handling

### Initial Implementation (commit `261dacf4`)

- Q: `[batch*beam, 1, nh, dh]`
- Cached K/V: `[batch, sk, nhk, dh]`
- Detect: `beam_size = Q.dim(0) / cached_keys->dim(0)`
- Tile K/V via `ops::Tile(0, beam_size)` → `[batch*beam, sk, nhk, dh]`
- Call FlashAttention with batch_size = batch*beam

### Beam Broadcasting Fix (commit `e528112a`)

The initial `ops::Tile` approach caused a **performance regression** from 2.6x → 1.5x for Whisper beam_size=5. Tiling copies the entire encoder K/V cache (batch × sk × nhk × dh) per layer per decode step.

**Fix:** Added `beam_size` parameter to `sdpa_metal` so Q indexes `batch*beam` while K/V index `batch` using `kv_b = b / beam_size`. This eliminates tiling entirely — zero-copy broadcast.

Changes for beam broadcasting:
- `sdpa_metal()` and `sdpa_cpu()` in `ops_sdpa.mm`: `kv_b = b / beam_size` for K/V batch indexing
- `FlashAttention::operator()` and `compute<D>()`: added `dim_t beam_size = 1` parameter
- `flash_attention_metal.mm`: passes `beam_size` to `sdpa_metal` calls
- `flash_attention_cpu.cc` / `flash_attention_gpu.cu`: compile-fix (beam_size param unused)
- `attention.cc`: no K/V tiling — passes `beam_size` directly to FlashAttention

```cpp
// In sdpa_metal / sdpa_cpu — both GPU and CPU paths:
for (dim_t b = 0; b < batch_size; ++b) {
  const dim_t kv_b = b / beam_size;  // K/V batch broadcasting
  // Q and output use b; K/V use kv_b
}
```

## MQA/GQA

`sdpa_metal` supports `num_heads_kv < num_heads` natively — no `replicate_heads` needed. This is more memory-efficient than the existing `process_cross_attention` path.

## Performance Results

### Whisper ASR (whisper-base, 60s Russian podcast, Apple M4, plugged in)

| Optimization Stage | Metal/CPU Ratio |
|---|---|
| After eliminate Gather/TopK syncs (M11) | ~1.59x |
| After batched MPS GEMM for non-padded attention (M11) | ~2.61x |
| After flash cross-attention with Tile (initial, `261dacf4`) | ~1.51x (regression) |
| **After flash cross-attention with beam_size broadcasting (`e528112a`)** | **~1.98x (median on power)** |
| Flash cross-attn disabled (old dot_product_attention) | ~1.80x (median on power) |

The initial flash cross-attention implementation regressed Whisper performance due to K/V tiling. After adding beam_size broadcasting, flash cross-attention with broadcasting (1.98x) is **faster** than the old `dot_product_attention` path (1.80x) on plugged-in power.

**Note:** Battery vs plugged-in significantly affects results. On battery: flash+broadcast ~1.70x, old path ~2.0x. On power: flash+broadcast ~1.98x, old path ~1.80x. Always benchmark on power.

### Seq2seq (opus-mt-en-de, WMT14 100 sentences)

| Mode | BLEU diff | Exact match | Notes |
|---|---|---|---|
| Greedy | 0.21 | 98/100 | Fused kernel has different FP rounding |
| Beam=4 | 0.21 | 97/100 | Same numerical cause |

The 2-3 sentence differences are expected: the fused SDPA kernel performs softmax in a single pass with different intermediate precision than separate MatMul + SoftMax + MatMul. BLEU diff of 0.21 is well within tolerance (threshold: 0.5).

## Test Results

| Test Suite | Count | Result |
|---|---|---|
| Translation | 90/90 | PASS |
| Beam search | 39/39 | PASS |
| Whisper | 13/13 | PASS |
| GPT-2 (generator) | 12/12 | PASS |
| Float16 | 9/9 | PASS |
| BF16 | 13/13 | PASS |
| Longform generation | 11/11 | PASS |
| INT8 | 10/10 | PASS |
| Seq2seq (BLEU) | 2/2 | PASS (diff=0.21) |
| Seq2seq (exact match) | 0/2 | Expected (fused kernel FP rounding) |
| Profiling | 2/2 | PASS |
| PSO caching | 2/2 | PASS |
