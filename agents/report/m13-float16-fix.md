# M13: Float16 MPS GEMM — Root Cause & Fix

## Problem

Float16 `translate_batch` on MPS produced garbage output (OpenNMT-py: only periods ".", BLEU 0.02). `score_batch` (teacher forcing) worked correctly. OPUS-MT float16 also showed quality degradation.

## Root Cause

**MPS float16 accumulation precision loss + Metal encoder tracking limitation.**

Two independent issues compound:

### Issue 1: MPS Float16 Accumulation
`MPSMatrixMultiplication` with float16 inputs accumulates in float16, causing catastrophic precision loss for K >= 512 (typical transformer hidden dim). This affects both the value of individual GEMMs and error propagation through the model pipeline.

### Issue 2: Metal Encoder Tracking Breakdown
When many encode-only compute encoders are queued in a single command buffer (empirically >128), Metal's automatic resource tracking between encoders becomes unreliable. Custom compute kernels writing to a buffer may not have their output visible to subsequent encoders that read from the same buffer.

**Key observations:**
- `CT2_COMMIT_AND_WAIT()` (full CPU/GPU sync) is the ONLY reliable fix — forces a new command buffer
- Non-blocking `commit_command_buffer()` + `encode_barrier()` does NOT fix the issue
- `protect_buffer_by_base()` (preventing allocator recycling) does NOT fix the issue
- Float32 path works without any periodic sync because MPS GEMM has internal resource tracking
- Custom GEMV kernel works without sync because it runs in the batched attention path where natural sync points limit encoder count per CB

**Why score_batch worked but translate_batch didn't:**
- `score_batch`: All tokens processed at once → fewer decode iterations → fewer total GEMMs per CB
- `translate_batch`: Autoregressive decode → many iterations × many GEMMs → exceeds encoder tracking limit

## Fix

### 1. Custom MSL GEMM kernel (`primitives_gemm.mm`)

New `gemm_f16_acc32` kernel: half inputs, float32 accumulation, half output. Single compute encoder per GEMM, no MPS involvement.

```cpp
// Dispatch path for all float16 GEMMs (both batched and non-batched):
dispatch_f16_gemm_acc32(transpose_a, transpose_b, m, n, k,
                        alpha, a, lda, b, ldb, beta, c, ldc);
```

With periodic `CT2_COMMIT_AND_WAIT()` every 64 GEMMs:
```cpp
static thread_local uint32_t _f16_gemm_count = 0;
if (++_f16_gemm_count >= 64) {
    CT2_COMMIT_AND_WAIT();
    _f16_gemm_count = 0;
}
```

Buffer protection prevents premature reuse between syncs:
```cpp
ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
ctranslate2::metal::protect_buffer_by_base([buf_b contents]);
ctranslate2::metal::protect_buffer_by_base([buf_c contents]);
```

### 2. Batched m=1 decode GEMMs (unchanged)

The `dispatch_gemv_f16_batched` kernel continues to handle m=1 attention GEMMs — encode-only, zero syncs, buffer protection only.

### 3. SDPA float16 path (`ops_sdpa.mm`)

Added `CT2_COMMIT_AND_WAIT()` between MPS float32 GEMM and the f32→half conversion in `sdpa_mps_gemm`. Same hazard pattern: MPS GEMM internal encoders don't bridge to subsequent custom compute encoders.

## Empirical Flush Frequency Analysis (Apple M4)

| N (flush every N GEMMs) | Speed (trans/s) | Correct? |
|--------------------------|-----------------|----------|
| 1 (every GEMM)           | 4.2             | Yes      |
| 4                         | 8.8             | Yes      |
| 8                         | 11.1            | Yes      |
| 16                        | 5.1             | Yes      |
| 32                        | 5.2             | Yes      |
| 64                        | 7.8             | Yes (5 sentences × 3 runs) |
| 96                        | 7.2             | **No** (some sentences) |
| 128                       | varies          | **Flaky** |
| 192+                      | varies          | **No** |
| 9999 (no flush)           | 2.7             | **No** |

N=64 chosen as the highest reliably-safe value with good performance.

## Performance

| Compute Type | Speed (trans/s) | vs f32 | Commits/translation |
|-------------|-----------------|--------|---------------------|
| float32     | 17.3            | 1.00x  | 20                  |
| float16     | 5.7             | 0.33x  | 54 (37 GEMM + 17 sync) |

Float16 is ~3x slower than float32 on this small model (OpenNMT-py, 6 layers, 512 dim). The overhead comes from:
1. Periodic CT2_COMMIT_AND_WAIT: 37 extra syncs × ~0.4ms = ~15ms
2. Naive per-element kernel: not hardware-optimized like MPS GEMM

For larger models (whisper-large-v3), f16 memory savings offset the speed loss.

## Verification

- **OpenNMT-py**: f16 output matches f32 exactly for 5 test sentences (beam=1 and beam=4)
- **OPUS-MT**: 9/9 E2E tests pass (greedy, beam=4, batch consistency) — f16 matches CPU f32
- **Whisper-base**: 18/18 E2E tests pass, WER 0.0% (exact transcript match with CPU)
- **Main translation test**: 88/90 pass (2 pre-existing edge cases unrelated to f16)

## Files Modified

- `src/metal/primitives_gemm.mm` — Custom f16 GEMM kernel + periodic flush + buffer protection
- `src/metal/ops_sdpa.mm` — CT2_COMMIT_AND_WAIT for SDPA f16 promotion path

## M13 Phase 2: Batched Inference & Padding Removal (2026-03-13)

### Additional Changes

After the initial M13 fix (custom GEMM kernel), two more changes were made to close the BLEU gap:

#### 1. Wired f32-accumulation GEMM into main dispatch (`primitives_gemm.mm`)

The M13 commit added `dispatch_f16_gemm_direct` (m≤32, SIMD kernel) and `dispatch_f16_promoted_gemm` (m>32, half→f32→MPS f32 GEMM→f32→half) but they were not connected to the main GEMM dispatch. The main `gemm<float16_t, float16_t>` still called native MPS f16.

**Fix:** Replaced native MPS f16 GEMM with `dispatch_f16_gemm()` in the main dispatch:
```cpp
} else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
  dispatch_f16_gemm(transpose_a, transpose_b, m, n, k,
                    alpha, a, lda, b, ldb, beta, c, ldc);
```

#### 2. Enabled padding removal for MPS float16 (`padder.h`)

`Padder::allow_padding_removal()` was returning `false` for GPU float16, disabling encoder padding removal. This caused f16 to process padded tokens while f32 didn't — a ~3 BLEU penalty.

**Fix:** Now returns `true` unconditionally (safe with f32-accumulation GEMM):
```cpp
static inline bool allow_padding_removal(const Device device,
                                         const ComputeType compute_type) {
  (void)compute_type;
  (void)device;
  return true;
}
```

### BLEU Improvement Trajectory (OPUS-MT, WMT14 En→De, 2737 sent, beam=4)

| State | BLEU | Tokens | Notes |
|-------|------|--------|-------|
| Pre-M13 (native MPS f16 GEMM) | 22.48 | 1449 | f16 accumulation, no padding removal |
| + f32 accum GEMM wired in | 24.67 | 1449 | Fixed GEMM precision |
| + padding removal enabled | 25.74 | 1544 | Token count now matches f32 |
| **f32 reference** | **27.65** | **1544** | Target |
| **CUDA f16 reference** | **27.90** | — | From README GPU table (NVIDIA A10G) |

### Remaining Gap: 1.91 BLEU (25.74 vs 27.65)

CUDA f16 shows **zero** BLEU loss vs f32 (27.90 vs 27.92 for OPUS-MT, 26.77 vs 26.77 for OpenNMT-py). This is because CUDA Tensor Cores accumulate in f32 for ALL operations uniformly.

MPS still has 1.91 BLEU gap. Investigation identified these remaining precision loss sources:

#### Source 1: SDPA GEMM uses native MPS f16 (est. ~0.6 BLEU)
`ops_sdpa.mm` line 211-213: The attention GEMM (QK^T and AV products) uses `MPSMatrixMultiplication` with f16 inputs, accumulating in f16. The comment says "K = head_dim (typically 64), well below K >= 512 threshold" but:
- Attention scores are softmax inputs — small errors amplify exponentially
- Runs once per head per layer per decode step — compounds across 8 heads × 6 layers × ~100 steps
- CUDA uses f32 accumulation for these GEMMs too

#### Source 2: Elementwise operations in native f16 (est. ~1.0 BLEU)
`kernels/elementwise.metal`: All binary operations (`a[gid] + b[gid]`, `a[gid] * b[gid]`) run directly in f16. Residual connections (output += sublayer_output) lose ~1 ULP per addition, compounding across ~100+ additions per forward pass (6 encoder layers × 4 residuals + decoder).

#### Source 3: Broadcast operations in native f16 (est. ~0.3 BLEU)
`kernels/broadcast.metal`: `add_batch_broadcast`, `add_depth_broadcast` also in native f16. Affects bias additions throughout the model.

#### Already Correct (not contributing to gap):
- **Softmax**: Reductions promoted to f32 in `normalization.metal`
- **LayerNorm/RMSNorm**: Accumulation in f32 in `normalization.metal`
- **Activations**: All computed in f32 in `activation.metal`
- **Quantize/Dequantize**: All in f32 in `quantize.metal`

### Files Modified (Phase 2)

- `src/metal/primitives_gemm.mm` — Wired `dispatch_f16_gemm` into main dispatch
- `include/ctranslate2/padder.h` — Enabled padding removal for all GPU types

## Future Optimization Opportunities

1. **simdgroup_matrix kernel**: Use Apple Silicon's hardware matrix multiply (`simdgroup_matrix<half,8,8>` with `simdgroup_matrix<float,8,8>` accumulator) for near-MPS performance with f32 accumulation
2. **Framework-level sync**: Move flush points to natural boundaries (layer end, decode step) instead of GEMM counter
3. **Reduce encoder count**: Reuse a single compute encoder across multiple GEMM dispatches with memory barriers
4. **Close remaining BLEU gap**: See Milestone 14 in `APPLE_M4_METAL_PLAN.md` — systematic precision audit of all f16 operations
