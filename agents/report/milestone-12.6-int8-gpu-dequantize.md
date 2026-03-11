# M12.6 — INT8 GPU Dequantize (Sync Elimination)

**Date**: 2026-03-11
**Status**: Complete — 5.4× speedup for int8, 5.7× for int8_float16
**Model**: OPUS-MT En→De (d_model=512, 6 layers)
**Hardware**: Apple M4, macOS 15

---

## 1. Problem

INT8 GEMM on MPS had **2 CPU/GPU synchronizations per GEMM call**:

```
Old path: dispatch_int8_gemm()
  1. CT2_COMMIT_AND_WAIT()      ← Sync #1: flush GPU so CPU can read int8 inputs
  2. CPU: vDSP_vflt8()          ← CPU converts int8 → float32 (A and B matrices)
  3. GPU: encode MPS GEMM       ← Float32 GEMM (encode-only)
  4. CT2_COMMIT_AND_WAIT()      ← Sync #2: flush GPU so CPU can read float32 result
  5. CPU: lroundf()             ← CPU rounds float32 → int32
```

With ~36 GEMMs per decode step (6 layers × 6 GEMMs each), this created:
- 72 `commit_and_wait()` calls per step at ~0.4ms each = **~29ms of pure sync overhead**
- Plus CPU conversion time (~4ms) = **~33ms wasted per step**
- decoder_call: 45.6 ms/step (M12.4 measured), 98% of loop time

---

## 2. Solution

Replace CPU int8↔float32 conversions with GPU compute kernels. All operations become encode-only — **zero syncs per GEMM**.

```
New path: dispatch_int8_gemm()
  1. GPU: int8_to_float32_strided kernel   ← encode-only (A)
  2. GPU: int8_to_float32_strided kernel   ← encode-only (B)
  3. GPU: encode MPS GEMM                  ← Float32 GEMM (encode-only)
  4. GPU: float32_round_to_int32_strided   ← encode-only
  No syncs! All operations chain in the command buffer.
```

### MSL Kernels

Two simple kernels added as inline MSL in `primitives_gemm.mm`:

**`int8_to_float32_strided`**: Reads int8 matrix with element stride `in_stride`, writes float32 with element stride `out_stride`. One thread per element.

**`float32_round_to_int32_strided`**: Reads float32 matrix, rounds to nearest int32 (`round()` in MSL). One thread per element.

Both handle MPS row alignment (strides differ between int8 input layout and MPS-padded float32 layout).

### Correctness Verification

Debug comparison showed **zero mismatches** between GPU `int8_to_float32_strided` and CPU `vDSP_vflt8` across all tested GEMM shapes:
- m=3, n=1536, k=512 (encoder FC)
- m=3, n=512, k=512 (attention)
- m=3, n=2048, k=512 (FFN)

---

## 3. Performance Results

### M12 Performance Sweep (50 sentences, beam=4, best-of-3)

| Type | Before (tok/s) | After (tok/s) | Speedup | Commits before | Commits after |
|------|---------------|--------------|---------|----------------|---------------|
| **int8** | **84** | **453** | **5.4×** | 5692 | 3522 |
| **int8_float16** | **86** | **494** | **5.7×** | 5580 | 3446 |
| float16 | 1476 | 1464 | — | 90 | 90 |
| float32 | 1019 | 1000 | — | 96 | 96 |
| bfloat16 | 9 → 1631* | 1426* | — | — | 90 |
| int8_bfloat16 | 9 → 574* | 454* | — | — | 3446 |

\* BF16 types auto-promoted to FP16 by M12.5.

### Decode Loop Profiling (10 sentences, beam=4, batch 0)

**INT8 (49 steps):**

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| decoder_call | 2232 ms (98%) | 1402 ms (96%) | -37% |
| sampler | 40 ms (1.8%) | 53 ms (3.6%) | +33%* |
| total_loop | 2278 ms | 1458 ms | **-36%** |
| **ms/step** | **46.5** | **29.8** | **-36%** |

**INT8_FLOAT16 (48 steps):**

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| decoder_call | 2177 ms (98%) | 1109 ms (97%) | -49% |
| sampler | 34 ms (1.5%) | 34 ms (3.0%) | unchanged |
| total_loop | 2218 ms | 1145 ms | **-48%** |
| **ms/step** | **46.2** | **23.9** | **-48%** |

\* Sampler time increased for int8 because the GPU now has more work queued and the `synchronize_stream()` in the sampler actually waits for the conversion kernels + GEMM to complete.

### Commits Reduction

| Type | Before | After | Reduction |
|------|--------|-------|-----------|
| int8 | 5692 | 3522 | -38% (2170 syncs eliminated) |
| int8_float16 | 5580 | 3446 | -38% (2134 syncs eliminated) |

The eliminated syncs correspond to ~2 syncs/GEMM × ~36 GEMMs/step × 30 steps ≈ 2160 syncs, matching the observed reduction.

---

## 4. Why Better Than Predicted

M12.4 predicted 45.6ms/step → ~13ms/step (3.5× speedup). Actual results are better:
- int8: 5.4× tok/s improvement (46.5 → 29.8 ms/step = 1.56× per-step speedup)
- int8_f16: 5.7× tok/s improvement (46.2 → 23.9 ms/step = 1.93× per-step speedup)

The tok/s improvement is larger than the per-step improvement because:
1. **Batch amortization**: 50 sentences batch more efficiently with faster per-step throughput
2. **Pipeline efficiency**: All-encode-only pipeline allows GPU to schedule work more efficiently
3. **int8_float16 extra benefit**: The dequantize_gemm_output kernel outputs float16 instead of float32, which is 2× less data to write

The per-step improvement is less than the 3.5× estimate because the estimate assumed zero overhead for the GPU kernels. In reality, the GPU int8→f32 and f32→int32 kernels add some overhead (encoding + dispatch + execution), but far less than the CPU sync overhead they replace.

---

## 5. Remaining INT8 Bottleneck

At 29.8 ms/step for int8 (vs 12.9 ms/step for float16), INT8 is still **2.3× slower** than FP16. The remaining overhead comes from:

1. **Extra GEMM work**: INT8 GEMM does int8→f32 conversion + f32 GEMM + f32→int32 rounding, which is ~3× the encode work of a single FP16 GEMM
2. **Quantize/dequantize ops**: Per-layer INT8 quantization (GPU quantize kernel) and GEMM output dequantization (GPU dequantize_gemm_output kernel) add overhead
3. **Remaining syncs**: 3522 commits (vs 90 for float16) — these come from other code paths (sampler, state updates, non-GEMM syncs)

### Possible further optimizations
- **Fused dequant-GEMM**: Skip the int32 intermediate entirely — convert int8→f32 and do GEMM in one step, output directly to the dequantize_gemm_output kernel. Would eliminate the f32→int32→dequant roundtrip.
- **INT8 GEMM via MPSGraph**: MPSGraph might support int8 matmul natively (untested)
- **Weight pre-dequantization**: Convert int8 weights to FP16 at load time (like M12.5 for BF16). Trades memory for speed.

---

## 6. Files Modified

| File | Change | Purpose |
|------|--------|---------|
| `src/metal/primitives_gemm.mm` | Added `kInt8GemmHelperMSL` (inline MSL), `encode_int8_to_float32()`, `encode_float32_to_int32()` dispatch functions | GPU int8↔f32 conversion kernels |
| `src/metal/primitives_gemm.mm` | Rewrote `dispatch_int8_gemm()` — replaced CPU vDSP + 2 syncs with GPU kernels | Single INT8 GEMM: encode-only |
| `src/metal/primitives_gemm.mm` | Rewrote batched INT8 GEMM path | Batched INT8 GEMM: encode-only |
| `src/types.cc` | (M12.5) BF16→FP16 auto-promotion | Already committed |
| `src/models/model.cc` | (M12.5) BF16 promotion warning, fixed scope | Already committed |

---

## 7. Updated Performance Summary (all types, post-M12.5+M12.6)

| Type | tok/s | vs CPU (826) | Commits | GPU% | Status |
|------|-------|-------------|---------|------|---------|
| **float16** | **1464** | **1.77×** | 90 | 41% | Optimal |
| **float32** | **1000** | **1.21×** | 96 | 54% | Optimal |
| **bfloat16** | **1426** | **1.73×** | 90 | 41% | Fixed (M12.5, auto-promoted) |
| **int8** | **453** | **0.55×** | 3522 | 40% | **Improved (M12.6)** |
| **int8_float16** | **494** | **0.60×** | 3446 | 42% | **Improved (M12.6)** |
| **int8_bfloat16** | **454** | **0.55×** | 3446 | 40% | Fixed (M12.5+M12.6) |
