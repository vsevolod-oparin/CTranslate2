# Milestone 8.2 — Transformer Decoder Layer (End-to-End Integration Test)

**Date:** 2026-02-26
**Status:** ✅ DONE

---

## Summary

M8.2 validates that the Metal backend can run a complete Pre-LayerNorm Transformer Decoder
Layer end-to-end with float32 accuracy matching the CPU reference to within floating-point
rounding error (~1e-7).

Two elements not exercised by M8.1 (encoder layer) are validated here:

| New element | Description | Implementation |
|-------------|-------------|----------------|
| Cross-attention | Q from decoder (sq), KV from encoder (sk), sq ≠ sk | `metal::sdpa_metal` (M6.1) |
| KV-cache decode | offset > 0: commit_and_wait + CPU memcpy + sdpa | M6.2 pattern in `flash_attention_metal.mm` |

No new Metal kernels or op specializations were required. All ops were already implemented
in M1–M7.

**Test result: 11/11 tests pass** (`tests/metal/m82_test.mm`)

---

## Architecture Under Test

```
x_dec --+--[LN1]--{Wqs,Wks,Wvs}--[causal self-SDPA]--[Wos]--+--> h1
        +----------------------------------------------------^

h1  ----+--[LN2]--{Wqc}--\
        |                 +--> [cross-SDPA]--[Woc]--+--> h2
        |  enc_ctx--{Wkc,Wvc}--/                    ^
        +-------------------------------------------/

h2  ----+--[LN3]--[W1]--[ReLU]--[W2]--+--> output
        +------------------------------^
```

**Prefill parameters:** batch=1, dec_seq=4 (DEC_T), enc_seq=6 (ENC_T), dim=16,
heads=2, head_dim=8, ffn_dim=32, max_cache=8, scale=1/√8, self_attn=causal,
cross_attn=non-causal.

**Decode parameters:** same weights, sq=1 (one new token), offset=4 (prior 4 tokens in cache).

---

## KV-Cache Decode Pattern (M6.2)

The decode step mirrors the implementation in `flash_attention_metal.mm` lines 174–236:

```
1. Compute Q_new, K_new, V_new via GPU GEMM on the new token.
2. commit_and_wait()  ← flush GPU so CPU can read K_new/V_new from Shared memory.
3. memcpy K_new/V_new into k_cache/v_cache at position offset=DEC_T.
4. sdpa_metal(q, k_cache, v_cache, out, B=1, sq=1, sk=DEC_T+1=5, is_causal=false)
```

Key invariant: `is_causal=false` for decode when sq=1, because all cached tokens are
already in the temporal past of the new query — no upper-triangular masking is needed.

---

## Results (Apple M4, float32)

| Test | Stage | max_abs_diff | Pass |
|------|-------|-------------|------|
| 1 | Cross-attn SDPA [1,4,2,8]×[1,6,2,8] | 5.96e-08 | ✅ |
| 2 | KV-cache decode [sq=1, sk=5, offset=4] | 8.94e-08 | ✅ |
| 3a | Full decoder prefill output [4×16] | 9.54e-07 | ✅ (< 2e-4) |
| 3b | Causal self-attn intermediate | 7.15e-07 | ✅ |
| 3c | Cross-attn intermediate | 2.38e-07 | ✅ |
| 3d | h1 (self-attn residual) | 7.75e-07 | ✅ |
| 3e | h2 (cross-attn residual) | 9.54e-07 | ✅ |
| 4a | Full decoder decode output [1×16] | 9.54e-07 | ✅ (< 2e-4) |
| 4b | Self-attn decode (sq=1, sk=5) | 3.58e-07 | ✅ |
| 4c | Cross-attn decode (sq=1, sk=6) | 1.49e-07 | ✅ |
| 4d | h2 decode (cross-attn residual) | 2.38e-07 | ✅ |

**All errors are at the float32 ULP level (~1e-7).** Consistent with M8.1 encoder results.

---

## Files Created

### Test file

`tests/metal/m82_test.mm` — 11 tests covering:
- Unit tests: cross-attention SDPA (sq≠sk), KV-cache decode step (Tests 1–2)
- Full decoder layer prefill with per-stage intermediate checks (Test 3a–3e)
- Full decoder layer decode step with intermediate checks (Test 4a–4d)

### Build command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m82_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm \
    src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm \
    src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm \
    src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm \
    src/metal/ops_sdpa.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m82_test && ./m82_test
```

---

## Key Findings

1. **Cross-attention (sq ≠ sk) is exact**: `sdpa_metal` handles arbitrary combinations of
   seqlen_q and seqlen_k without modification. The MPS matrix multiplication used internally
   is shape-agnostic.

2. **KV-cache decode is correct**: The commit_and_wait + CPU memcpy + sdpa pattern from
   M6.2 produces numerically identical results to a CPU reference that builds the full K/V
   array explicitly. Shared memory eliminates any coherency issues.

3. **Full decoder pipeline accuracy**: The 10-stage pipeline (3 LayerNorm, 3 GEMM groups,
   1 causal + 1 non-causal SDPA, 1 ReLU, 2 residual adds) accumulates at most ~1e-6 error
   over float32 ULP — no rounding accumulation is visible.

4. **Decode vs prefill parity**: The decode step (sq=1) and prefill (sq=4) produce the same
   ULP-level errors. The commit_and_wait + memcpy path does not introduce any additional
   numerical error compared to running the attention directly.

---

## Deferred Items (M8.3+)

| Item | Notes |
|------|-------|
| M8.3: Conv1d / WhisperEncoder | `MPSCNNConvolution` wrapping; Conv1d×2 + encoder layers |
| Multi-step decode sequence | Test offset=0,1,2,...,N to verify cache grows correctly |
| Float16 / BF16 decoder test | Extend with fp16/bf16 variants |
| Batch > 1 decode | Batch decode requires cache stride = MAX_CACHE×NK×HD per batch |
