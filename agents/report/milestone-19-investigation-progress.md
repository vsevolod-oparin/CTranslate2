# M19 Investigation Progress — Root Cause Analysis In Progress

**Date:** 2026-03-22
**Status:** Investigation ongoing — root cause narrowed but not identified

## Key Findings

### Simplified reproduction
The bug is much simpler than initially described:
- **Any** `beam_size >= 2` on **any** frame count, followed by
- **Any** encode with **odd** mel frame count → **all-NaN encoder output**

Even `beam5(205) → encode(205)` (same frame count!) triggers it. The prior frame count is irrelevant.

### It's not an Apple MPS bug
Four standalone reproduction attempts (f32, f16, pooled, deferred CB, cached GEMM, mixed custom compute+MPS) all produce correct results. The bug is in CTranslate2's code, not Apple's MPS framework.

### NaN location
- The encoder output is **entirely NaN** (131,840/131,840 values for enc_len=103)
- Decoder is innocent — it receives corrupted encoder output
- Only whisper-large-v3-turbo affected (not whisper-base)

### Odd vs even mel frames
| mel_frames | enc_len | After beam | Notes |
|-----------|---------|------------|-------|
| 200 | 100 | OK | |
| 201 | 101 | OK | |
| 204 | 102 | OK | |
| 205 | 103 | **NaN** | Same enc_len as 206! |
| 206 | 103 | OK | |
| 207 | 104 | **NaN** | Same enc_len as 208! |
| 208 | 104 | OK | |

mel=205 and mel=206 both produce enc_len=103, but only 205 (odd) produces NaN. The difference is the **Conv1D stride-2 input length** (205 odd vs 206 even), not the output length.

### What beam search does that triggers it
- `beam_size=1` (greedy) → subsequent encode is fine
- `beam_size>=2` → subsequent odd-frame encode produces NaN
- Beam search expands batch dimension (batch=1 → batch=beam_size)
- This allocates larger intermediate tensors that land in specific pool buckets
- When freed to pool, these buffers are reused by the subsequent encoder

### Pool bucket analysis
For conv1 output [1, 1280, T_out] at float16:
- mel=205 → size = 524,800 bytes → **bucket 1,048,576 (1 MB)**
- mel=204 → size = 522,240 bytes → **bucket 524,288 (512 KB)**

Different pool buckets! The 1 MB bucket likely has a stale buffer from beam search's batch-expanded tensors. The 512 KB bucket does not.

### What the fix (disable pooling) does
With pooling disabled, every allocation creates a fresh zeroed MTLBuffer. No stale data from beam search is ever encountered. This masks the real bug.

## Hypotheses for actual root cause

1. **Conv1D `dispatch_mps_gemm_batched_padded` row_copy bug**: The output row_copy might write incorrect data for specific dimensions where `nat_rb_c` is not 16-byte aligned. With odd T_out, the row stride is `T_out * 2` which may have specific alignment properties that interact with the row_copy kernel.

2. **Stale data in pool buffer read by MPS GEMM**: The MPS GEMM in the padding path reads/writes a fresh temp buffer (alloc_temp_buffer), but the row_copy writes back to the pool buffer. If the pool buffer has stale NaN/Inf values from beam search in the padding bytes between rows, and these leak through...

3. **Pool buffer used as both GEMM input and output**: If the same pool buffer is returned for both the Conv1D input and output (same bucket size), the GEMM might be reading and writing the same buffer concurrently.

## Next steps

1. Add NaN checks after conv1, conv2, transpose, position_embedding, each encoder layer in whisper.cc
2. Specifically check if the 1 MB pool buffer returned for conv1 output contains NaN/Inf from beam search
3. Check if the row_copy kernel produces correct output for the specific dimensions (m=1280, nat_rb=410, mps_rb=416)
4. Check if the same buffer is returned for both conv1 input and conv1 output (aliasing)

## Current code state
- Pooling is RE-ENABLED (bug triggers)
- F16TempCache/SdpaF16TempCache zeroing is applied
- Option 1 (per-model clear_cache) is stashed: `git stash list`
- Test files: `tests/metal/m19_repro.py`, `tests/metal/m19_temp_cache_test.mm`, `tests/metal/m19_perf.py`
