# Milestone 10.3 — Language Model (GPT-style) End-to-End on Metal

**Date:** 2026-03-03
**Status:** Complete (12/12 tests pass)

## Summary

GPT-2 (decoder-only, pre-norm architecture with `gelu_tanh` activation) runs end-to-end
on the Metal backend with token-for-token CPU match across greedy decoding, beam search,
and batch inference.

A Metal GPU bug was discovered and fixed: the MSL `tanh()` function produces NaN for
arguments with absolute value greater than ~44, because it internally computes
`(exp(2x)-1)/(exp(2x)+1)` and `exp(2x)` overflows float32 to infinity, yielding `inf/inf = NaN`.

## Bug Investigation

### Symptom
GPT-2 generation on Metal produced all-NaN logits after the first decoder layer.
The first NaN appeared at index 1612 of the 3072-element FFN hidden state (ff1 output).

### Root Cause Analysis

1. **Model weights are clean** — parsed all 158 variables from `model.bin`; no NaN in any weight.
2. **MPS GEMM output is clean** — verified by reading output immediately after `commit_and_wait()`.
3. **BiasAdd output is clean** — the broadcast-add kernel correctly adds bias to GEMM output.
4. **NaN is produced by the `gelu_tanh` MSL kernel** — confirmed by reading the kernel input
   (all finite, including value 11.628224 at index 1612) and observing NaN only in the output.
5. **Standalone reproduction** — a minimal 50-line Metal test dispatching only `gelu_tanh_float`
   with input 11.628224 at index 1612 reproduces the NaN, confirming this is a Metal shader bug,
   not a CT2 pipeline issue.

### Mathematical Analysis

The `gelu_tanh` formula is:
```
y = 0.5 * v * (1 + tanh(0.7978 * (v + 0.044715 * v^3)))
```

For `v = 11.628224`:
- Inner term: `0.044715 * 11.628^3 ≈ 70.27`
- Tanh argument: `0.7978 * (11.628 + 70.27) ≈ 65.3`
- Metal's `tanh(65.3)`: computes `exp(130.6)` which overflows float32 (max ~3.4e38, exp saturates at ~88.7)
- Result: `(inf - 1) / (inf + 1) = inf/inf = NaN`

The CPU `tanhf(65.3)` correctly returns 1.0 because the C library uses a different implementation
that handles large arguments.

## Fix

Added `ct2_safe_tanh()` — a clamped wrapper around Metal's `tanh()`:

```metal
static float ct2_safe_tanh(float x) {
    return tanh(clamp(x, -10.f, 10.f));
}
```

Since `tanh(10) = 0.99999999587...` (within 4e-9 of 1.0), this clamp introduces zero practical
error while eliminating the NaN for all possible float32 inputs.

Applied to:
- `src/metal/kernels/activation.metal` — `tanh` and `gelu_tanh` activation ops
- `src/metal/kernels/quantize.metal` — inline `gelu_tanh` and `tanh` in `dequantize_gemm_output`
- `src/metal/msl_strings.h` — regenerated via `gen_msl_strings.py`

## Files Modified

| File | Change |
|------|--------|
| `src/metal/kernels/activation.metal` | Added `ct2_safe_tanh()`, use in `tanh` and `gelu_tanh` ops |
| `src/metal/kernels/quantize.metal` | Added `ct2_safe_tanh()`, use in dequantize_gemm_output |
| `src/metal/msl_strings.h` | Regenerated (auto-generated from .metal files) |
| `tests/metal/e2e/test_generator.py` | New: GPT-2 e2e test (12 test cases) |
| `APPLE_M4_METAL_PLAN.md` | Updated M10.3 status |

## Debug Code Removed

All temporary debug instrumentation from the investigation was cleaned up:

| File | Removed |
|------|---------|
| `src/ops/gemm.cc` | `extern "C" metal_commit_and_wait_debug` declaration + barrier call |
| `src/metal/utils.mm` | `metal_commit_and_wait_debug()` function |
| `src/metal/primitives_gemm.mm` | `gemm_call_count` counter + debug comment |
| `src/metal/primitives_elementwise.mm` | PRE-UNARY NaN dump (40+ lines) + broadcast debug comment |
| `src/ops/bias_add_metal.mm` | `synchronize_stream` barrier + `#include devices.h` |
| `src/layers/transformer.cc` | FFN NaN trace, decoder layer NaN trace, decoder-level NaN trace (~80 lines) |

## Test Results

```
$ CT2_TEST_DATA=.../data python3 tests/metal/e2e/test_generator.py
[PASS] Greedy prompt 0 (len=21)
[PASS] Greedy prompt 1 (len=21)
[PASS] Greedy prompt 2 (len=21)
[PASS] Greedy prompt 3 (len=21)
[PASS] Beam2 prompt 0 (len=16)
[PASS] Beam2 prompt 1 (len=16)
[PASS] Beam4 prompt 0 (len=16)
[PASS] Beam4 prompt 1 (len=16)
[PASS] Batch prompt 0 (len=11)
[PASS] Batch prompt 1 (len=11)
[PASS] Batch prompt 2 (len=11)
[PASS] Batch prompt 3 (len=11)

12/12 passed
ALL PASS
```

Existing translation tests remain unaffected (90/90 pass).

## GPT-2 Architecture on Metal

The decoder-only model exercises these Metal ops per layer:
1. **LayerNorm** (pre-norm)
2. **Self-attention**: 4x GEMM (Q/K/V/Out) + SDPA with KV-cache
3. **Residual add**
4. **LayerNorm** (pre-norm)
5. **FFN**: GEMM + BiasAdd + **GELU_TANH** + GEMM + Residual
6. **PositionEmbedding** (absolute, learned)

All ops were previously implemented (M1–M9). No new Metal kernels were required — only
the `tanh()` NaN fix was needed.

## Architecture Notes

- GPT-2 uses `gelu_tanh` (tanh-approximated GELU), not plain `gelu` (erf-based)
- `multi_query_attention: false` in the CT2 config
- Pre-norm architecture (LayerNorm before attention/FFN, not after)
- 12 layers, 768 hidden, 3072 FFN intermediate, 12 heads
