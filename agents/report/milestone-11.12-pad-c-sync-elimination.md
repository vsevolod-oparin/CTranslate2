# M11.12 — Eliminate logits GEMM pad_c sync + Sync Reduction Roadmap

## Summary

Replaced the CPU memcpy pad_c unpack in `dispatch_mps_gemm<T>()` with the encode-only GPU `dispatch_row_copy()` kernel, eliminating one `commit_and_wait()` per decode step for padded output matrices (e.g., logits projection n=51865).

Fixed a correctness bug in `indexed_fill` (token suppression) that was exposed by the deferred GPU execution.

## Changes

### 1. `src/metal/primitives_gemm.mm` — pad_c GPU unpack

**Before** (lines 247-254):
```cpp
if (pad_c) {
    CT2_COMMIT_AND_WAIT();  // sync!
    const auto* src = static_cast<const uint8_t*>([tmp_c contents]);
    auto* dst = reinterpret_cast<uint8_t*>(c);
    for (NSUInteger r = 0; r < rows_c; ++r)
        std::memcpy(dst + r * nat_rb_c, src + r * mps_rb_c, nat_rb_c);
}
```

**After**:
```cpp
if (pad_c) {
    NSUInteger dst_off_c = 0;
    id<MTLBuffer> dst_buf = ctranslate2::metal_buffer_for_ptr(c, &dst_off_c);
    dispatch_row_copy(tmp_c, 0,
                      mps_rb_c, rows_c * mps_rb_c,   // src: padded layout
                      dst_buf, dst_off_c,
                      nat_rb_c, rows_c * nat_rb_c,    // dst: natural layout
                      nat_rb_c, rows_c, 1);            // batch_size=1
}
```

Added forward declaration for `dispatch_row_copy` before `dispatch_mps_gemm` since the definition appears later in the file.

### 2. `src/metal/primitives_memory.mm` — `indexed_fill` sync

Added `CT2_COMMIT_AND_WAIT()` before the CPU loop in `indexed_fill`. This is necessary because the deferred `dispatch_row_copy` may have pending GPU writes to the same buffer that `indexed_fill` modifies via CPU. Without the sync, the GPU row_copy would overwrite the CPU's token suppression writes.

```cpp
template<>
template <typename T>
void primitives<Device::METAL>::indexed_fill(T* x, T a,
                                              const int32_t* indices,
                                              dim_t num_indices) {
    CT2_COMMIT_AND_WAIT();  // flush pending GPU row_copy
    for (dim_t i = 0; i < num_indices; ++i)
        x[indices[i]] = a;
}
```

## Debugging Notes

The `indexed_fill` bug was subtle:
- `dispatch_row_copy` is encode-only (no sync), so the GPU row_copy from padded→natural C is deferred
- `DisableTokens::apply()` calls `indexed_fill` on the same logits buffer to suppress tokens (e.g., end-of-text penalties)
- Without sync, the CPU writes from `indexed_fill` were overwritten when the deferred GPU row_copy eventually executed
- Symptom: Metal whisper produced empty transcription (suppressed tokens not actually suppressed)

## kCpuGemmThresh=0 Experiment

Tested forcing all padded GEMMs through MPS (no CPU cblas fallback):
- **Float32**: All tests pass — MPS f32 GEMM matches CPU cblas f32
- **Float16**: Tests fail — CPU cblas widens f16→f32 before computing; MPS uses native f16 with f32 accumulation. The precision difference compounds over decode steps. This is a fundamental limitation, not a bug.
- **Conclusion**: kCpuGemmThresh=4096 is correct; tiny f16 GEMMs must use CPU for precision parity.

## Test Results

All 159 tests pass:
- `test_translation.py` — 90/90
- `test_float16_translation.py` — 9/9
- `test_beam_search.py` — 39/39
- `test_whisper.py` — 13/13
- `test_faster_whisper.py` — 8/8

---

## Sync Reduction Roadmap

### Current Sync Trace (whisper-large-v3-turbo, beam_size=5)

```
 Syncs  Source                          Description
------  ------------------------------  -------------------------------------------
 14136  primitives_gemm.mm:1016         Batched CPU cblas for tiny padded f16 GEMMs
  2297  primitives_reduction.mm:127     amax() partial GPU reduce + CPU max
  1040  devices.cc:162                  synchronize_stream calls
  1012  primitives_memory.mm:80         indexed_fill CPU loop
   744  ops/multinomial_metal.mm:21     CPU multinomial sampling
   268  ops/topk_metal.mm:44            TopK k>1 CPU partial_sort
   224  primitives_gemm.mm:796          Batched padded GEMM pad_a/pad_b sync
    14  primitives_memory.mm:90         convert
    14  primitives_beam_search.mm:89    beam search gather
```

### Priority 1: GPU `indexed_fill` kernel (1012 syncs → 0)

**Effort**: Low (trivial MSL scatter-write kernel)
**Impact**: Eliminates 1012 syncs per transcription

Replace the CPU loop with an MSL kernel:
```metal
kernel void indexed_fill(device T* x [[buffer(0)]],
                         const device int* indices [[buffer(1)]],
                         constant T& value [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    x[indices[gid]] = value;
}
```

Encode-only — no `commit_and_wait()` needed. The kernel runs in-order after any pending GPU work on the same command buffer.

### Priority 2: GPU full `amax()` reduction (2297 syncs → 0)

**Effort**: Medium (second reduction pass MSL kernel)
**Impact**: Eliminates 2297 syncs per transcription

Currently `amax()` does:
1. GPU partial reduction: each threadgroup produces one max → `partial_buf` (encode-only)
2. `commit_and_wait()` — sync!
3. CPU `std::max_element` over partial results

Fix: Add a second GPU reduction pass over the partial results to produce a single scalar. Both passes encode-only, no sync needed. The consumer of `amax()` (quantization) will sync when it reads the result.

### Priority 3: Batched padded GEMM pad_a/pad_b sync (224 syncs → 0)

**Effort**: Medium
**Impact**: Eliminates 224 syncs per transcription

`dispatch_mps_gemm_batched_padded<T>()` already uses `dispatch_row_copy` for C unpack but still does `commit_and_wait()` for A and B padding. Could use GPU row_copy for A/B pack as well.

### Priority 4: GPU TopK k>1 (268 syncs → 0)

**Effort**: High (GPU top-k algorithm: radix select or bitonic partial sort)
**Impact**: Eliminates 268 syncs per transcription

CPU `partial_sort` over vocab=51865 takes ~33µs. A GPU implementation would need to beat this despite CB overhead. Possible approaches:
- Radix-based k-th element selection
- Warp-level bitonic sort of top-k candidates
- Two-pass: GPU histogram to find threshold, then GPU filter

### Priority 5: GPU multinomial sampling (744 syncs → 0)

**Effort**: High (GPU prefix-sum + binary search)
**Impact**: Eliminates 744 syncs per transcription

Requires:
1. GPU parallel prefix sum over probability distribution
2. GPU binary search with random threshold
3. Result in GPU buffer for downstream consumption

### Priority 6: Batched CPU cblas f16 GEMMs (14136 syncs)

**Effort**: Very High
**Impact**: Numerically largest but practically smallest — these are tiny GEMMs (< 4096 elements) where CPU cblas is genuinely faster than MPS dispatch. The syncs are near-zero cost since there's typically no pending GPU work.

Cannot simply route through MPS due to f16 precision mismatch (see kCpuGemmThresh experiment above). Would require a custom GPU kernel that widens f16→f32, computes in f32, and narrows back — significant engineering for minimal real-world speedup.

### Estimated Impact

| Change | Syncs eliminated | Effort |
|--------|-----------------|--------|
| GPU indexed_fill | 1012 | Low |
| GPU full amax | 2297 | Medium |
| Batched GEMM A/B pack | 224 | Medium |
| GPU TopK k>1 | 268 | High |
| GPU multinomial | 744 | High |
| **Total feasible** | **4545** | |

Priority 1+2 alone would eliminate ~3300 syncs with low-to-medium effort.
