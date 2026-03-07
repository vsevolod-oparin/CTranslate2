# M11.11 — Batched Padded GEMM & faster_whisper Investigation

## Summary

Added `dispatch_mps_gemm_batched_padded<T>()` for batched MPS GEMM when matrices require MPS 16-byte row alignment padding. Previously, padded batched GEMMs fell back to per-element `dispatch_mps_gemm<T>()` loops, causing **N syncs per batch** (N = batch_size, typically 20–100 heads). The new function handles padding via temp buffers in a single MPS batched encode + GPU row_copy unpack.

Also fixed a critical regression where per-element dispatch in `gemm_batch_strided` caused **17,700 `commit_and_wait()` calls** per transcription on whisper-large-v3-turbo (float16), slowing Metal to 0.58× CPU. After optimization: ~1.06× on faster_whisper (beam=5), ~1.45× on native API (greedy).

## Context

whisper-large-v3-turbo: d_model=1280, n_heads=20, head_dim=64, 4 decoder layers, encoder output=1500 frames, float16.

For float16, `cols=1500`: `1500 × 2 = 3000 bytes`, but `MPSMatrixDescriptor rowBytesForColumns:` returns 3008 (next 16-byte aligned). `needs_padding()` returns true, triggering CPU fallback for **all** batched attention GEMMs:

| GEMM | Dimensions | Padding | Batch size |
|------|-----------|---------|------------|
| Cross-attn score QK^T (decode) | m=1, n=1500, k=64 | pad_c only | 20–100 |
| Cross-attn context AV (decode) | m=1, n=64, k=1500 | pad_a only | 20–100 |
| Self-attn score QK^T (decode) | m=1, n=~130, k=64 | pad_c | 20–100 |
| Self-attn context AV (decode) | m=1, n=64, k=~130 | pad_a | 20–100 |
| Encoder score QK^T (prefill) | m=1500, n=1500, k=64 | pad_c only | 20 |
| Logits projection | m varies, n=51865, k=1280 | pad_c | 1 |

With beam_size=5, ~110 decode steps × 4 layers × 2 attention types × 2 GEMMs × batch_size ≈ 17,600 individual CPU GEMM calls, each preceded by a `commit_and_wait()`.

## Changes

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `dispatch_mps_gemm_batched_padded<T>()`; added MSL `row_copy` kernel; added `batch_cpu_gemm_f32`/`batch_cpu_gemm_f16` helpers; restructured `gemm_batch_strided` routing |
| `tests/metal/e2e/bench_faster_whisper.py` | Added beam_size CLI parameter for targeted benchmarking |

## Implementation

### Three-tier routing in `gemm_batch_strided`

```
gemm_batch_strided (FP32/FP16, needs_padding=true):
  ├── m*n > 4096  → dispatch_mps_gemm_batched_padded<T>()
  │     MPS batched GEMM with padded temp buffers.
  │     CPU memcpy for A/B packing, GPU row_copy for C unpack.
  │     Syncs: 1 flush (if pad_a/pad_b) + 0 (encode-only MPS + GPU unpack)
  │
  └── m*n ≤ 4096  → batch_cpu_gemm_f32() / batch_cpu_gemm_f16()
        Single CT2_COMMIT_AND_WAIT() + loop of cblas_sgemm.
        Syncs: 1 per batch (was N per batch with per-element dispatch)

gemm_batch_strided (FP32/FP16, needs_padding=false):
  ├── Try dispatch_mps_gemm_batched<T>()  → single batched MPS encode
  └── Fallback: per-element dispatch_mps_gemm<T>() loop
```

### `dispatch_mps_gemm_batched_padded<T>()`

Handles batched GEMM when any of A, B, or C need row-padding for MPS alignment:

1. **Flush pending GPU work** (if pad_a, pad_b, or pad_c with beta≠0) — single `CT2_COMMIT_AND_WAIT()`
2. **CPU memcpy** for A/B packing: copy rows from natural stride to MPS-aligned temp buffer
3. **memset** for C temp (when beta=0, avoids reading stale data from padding bytes)
4. **MPS batched GEMM**: single `MPSMatrixMultiplication` with `batchSize` (encode-only)
5. **GPU row_copy** for C unpack: MSL kernel copies from padded temp back to natural stride (encode-only)

Worst case: 1 sync (flush) + N encode-only operations. Best case (pad_c only, beta=0): 0 syncs.

### MSL `row_copy` kernel

```metal
kernel void row_copy(
    device const char* src, device char* dst,
    constant uint& copy_bytes, constant uint& src_rb, constant uint& dst_rb,
    constant uint& rows_per_mat, constant uint& src_mb, constant uint& dst_mb,
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
```

- One threadgroup per row across all batch elements
- 4-byte aligned copies via `uint*` cast, remainder bytes handled by thread 0
- Dispatched with `dispatchThreadgroups:` (NOT `dispatchThreads:`)
- **Bug fix**: Originally used `thread_position_in_grid` which ranges 0..total_rows×256−1 with `dispatchThreadgroups:`, causing GPU page faults. Fixed to `threadgroup_position_in_grid` (0..total_rows−1).

### `batch_cpu_gemm_f32` / `batch_cpu_gemm_f16`

Lambda helpers for tiny padded batched GEMMs (m×n ≤ 4096):

- **Single** `CT2_COMMIT_AND_WAIT()` for the entire batch
- Loop of `cblas_sgemm` calls (no per-element sync)
- For float16: widen to float32 → cblas → narrow back (stack-allocated up to 4096 elements)
- Reduces syncs from N (per-element dispatch) to 1 per batch

### `kCpuGemmThresh = 4096` correctness boundary

MPS produces incorrect results for very small float16 matrices (observed with opus-mt, ~10×10). The threshold ensures:
- **m×n > 4096**: MPS batched padded (safe for MPS, GPU-accelerated)
- **m×n ≤ 4096**: CPU cblas (guaranteed correct for all sizes)

Attempted routing pad_c-only + beta=0 cases through MPS even below threshold (as it would be encode-only/zero syncs), but:
- MPS encoding overhead for m=1 GEMMs exceeds the sync cost savings
- Performance degraded from 1.06× to 0.71× — reverted

## faster_whisper Investigation

### Why faster_whisper is slower than native API

| Benchmark | API | Decoding | Metal/CPU ratio |
|-----------|-----|----------|----------------|
| test_whisper (large-v3-turbo) | Native `model.generate()` | Greedy | **1.45×** |
| test_faster_whisper (large-v3-turbo) | `WhisperModel.transcribe()` | Beam=5 | **1.13×** |
| bench_faster_whisper (large-v3-turbo) | `WhisperModel.transcribe()` | Beam=5 | **1.06×** |
| bench_faster_whisper (large-v3-turbo) | `WhisperModel.transcribe()` | Beam=1 | **0.41×** |

Key differences:

1. **Beam size**: Native test uses greedy (no beam_size parameter in `model.generate()`). faster_whisper defaults to beam_size=5. Beam search multiplies decode steps and generates larger batches.

2. **Separate encode/decode**: faster_whisper calls `model.encode()` then `model.generate()` separately. Native API does both in a single `model.generate(features, [PREFIX_TOKENS])` call.

3. **Beam=1 paradox**: beam_size=1 via faster_whisper is *slower* than beam_size=5 (0.41× vs 1.06×). This is because beam_size=1 generates much longer output sequences (more decode steps), while beam_size=5 terminates earlier via beam search pruning.

4. **Tiny padded GEMMs dominate decode**: Each decode step requires cross-attention with encoder output (1500 frames). With float16, these GEMMs need padding → CPU cblas → `commit_and_wait()`. Even with batched single-sync optimization, ~8,600 syncs per transcription remain.

### Sync trace (whisper-large-v3-turbo, faster_whisper beam=5, post-optimization)

| Caller | Commits | Source |
|--------|---------|--------|
| `primitives_gemm.mm` (batch CPU GEMM) | 8,644 | Tiny padded attention GEMMs in `gemm_batch_strided` |
| `primitives_reduction.mm` | 1,173 | Softmax/reduction ops |
| `devices.cc` (synchronize_stream) | 626 | synchronize_stream calls |
| `primitives_gemm.mm` (MPS pad_c unpack) | 620 | Logits projection (alignment padding) |
| `multinomial_metal.mm` | 477 | Multinomial sampling |
| `topk_metal.mm` | 134 | TopK |
| `primitives_gemm.mm` (batched padded flush) | 112 | Large padded GEMM flush |

### GEMM batch size distribution (single transcription)

| Dimensions | Batch size | Count | Source |
|-----------|-----------|-------|--------|
| m=1, n=1500, k=64 | 40 | 908 | Cross-attn score (beam=5 × 2 layers?) |
| m=1, n=64, k=1500 | 40 | 908 | Cross-attn context |
| m=1, n=1500, k=64 | 20 | 872 | Cross-attn score (other layers) |
| m=1, n=64, k=1500 | 20 | 872 | Cross-attn context |
| m=5, n=64, k=1500 | 20 | 536 | Batched cross-attn context |
| m=1, n=1500, k=64 | 100 | 68 | Full beam expansion |
| m=1, n=~130, k=64 | 100 | 24 | Self-attn score (cache length) |

## Performance Results

### whisper-large-v3-turbo (float16, 60s Russian audio, Apple M4)

| Configuration | CPU (ms) | Metal (ms) | Ratio |
|--------------|----------|------------|-------|
| Native greedy (test_whisper) | ~26,000 | ~18,000 | **1.45×** |
| faster_whisper beam=5 | ~35,000 | ~34,000 | **1.06×** |
| faster_whisper beam=1 | ~37,000 | ~90,000 | 0.41× |

### whisper-base (float32, 60s audio)

| Configuration | CPU (ms) | Metal (ms) | Ratio |
|--------------|----------|------------|-------|
| Native (test_whisper) | ~3,400 | ~1,900 | **1.81×** |
| faster_whisper beam=5 | ~5,400 | ~2,850 | **1.91×** |

whisper-base sees better Metal speedup because encoder output is 1500 frames but head_dim=64 and n_heads=8 with float32 — most decode GEMMs don't need padding (float32 cols=64: 256 bytes, 16-byte aligned).

## Test Results

| Test Suite | Result |
|-----------|--------|
| Translation (90 tests) | 90/90 PASS |
| Float16 translation (9 tests) | 9/9 PASS |
| Beam search (39 tests) | 39/39 PASS |
| Whisper base (13 tests) | 13/13 PASS |
| Whisper large-v3-turbo (12 tests) | 12/12 PASS |
| faster_whisper (8 tests) | 8/8 PASS |

## Failed Approaches

### 1. GPU row_copy for A/B packing

Attempted using the MSL row_copy kernel for A and B padding (not just C unpack). Result: incorrect output for all models. Root cause undetermined — possibly a race condition between the row_copy compute kernel writing to temp buffers and MPS reading them, or an issue with the buffer offsets. Reverted to CPU memcpy for A/B packing.

### 2. Encode-only MPS for small pad_c-only matrices

When only pad_c needs padding and beta=0, `dispatch_mps_gemm_batched_padded` is theoretically encode-only (zero syncs). Attempted routing m×n ≥ 256 through this path to eliminate syncs for cross-attention score GEMMs (m=1, n=1500).

Result: Performance degraded from 1.06× to 0.71×. The MPS encoding overhead (ObjC descriptor creation, temp buffer allocation, encode calls) for thousands of tiny m=1 matrices outweighs the sync cost savings. CPU cblas for m=1 is extremely fast (~1 row), making the sync the cheaper path.

### 3. Per-element dispatch for small padded batches (original code)

Before this work, `gemm_batch_strided` with padding called per-element `dispatch_mps_gemm<T>()` in a loop. Each element independently checks padding → CT2_COMMIT_AND_WAIT() → cblas. With batch_size=20–100, this caused 20–100 syncs per batch instead of 1. This was the primary cause of the 0.58× regression on faster_whisper.

## Further Optimization Opportunities

1. **Reduction/Softmax syncs (1,173 per transcription)**: These ops sync per-call. Batching or persistent kernels could help.

2. **Multinomial syncs (477)**: Used in beam search sampling. Could be replaced with GPU-side sampling.

3. **Custom MSL GEMV kernel for m=1 padded**: A Metal compute shader for m=1 matrix-vector multiply could bypass MPS entirely, avoiding both MPS overhead and cblas widening. Would be encode-only with no padding needed (custom kernel controls its own row stride).

4. **Fused cross-attention decode**: Fuse QK^T + softmax + AV into a single kernel for sq=1 cross-attention decode, eliminating 3 separate dispatches and their associated syncs.
