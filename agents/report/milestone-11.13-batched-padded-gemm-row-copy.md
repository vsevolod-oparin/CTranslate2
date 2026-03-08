# M11.13 — GPU row_copy for batched padded GEMM A/B/C packing

## Summary

Replaced CPU `memcpy` A/B/C packing in `dispatch_mps_gemm_batched_padded()` with encode-only GPU `dispatch_row_copy()`, eliminating all `commit_and_wait()` syncs from the large padded batched GEMM path. Uses a non-blocking `commit_command_buffer()` between row_copy and MPS GEMM to work around a Metal constraint where compute encoders before MPS encoders on the same command buffer cause crashes.

## Changes

### `src/metal/primitives_gemm.mm` — `dispatch_mps_gemm_batched_padded()`

**Before** (lines 793-845):
```cpp
// CPU memcpy for A/B packing (needs flush to read GPU data).
if (pad_a || pad_b || (pad_c && beta != 0.0f))
    CT2_COMMIT_AND_WAIT();   // <-- blocking sync!

if (pad_a) {
    tmp_a = alloc_temp_buffer(mb_a * batch_size);
    auto* dst = static_cast<uint8_t*>([tmp_a contents]);
    auto* src = reinterpret_cast<const uint8_t*>(a);
    for (dim_t i = 0; i < batch_size; ++i)
        for (NSUInteger r = 0; r < rows_a; ++r)
            std::memcpy(dst + i*mb_a + r*mps_rb_a,
                        src + i*stridea*elem + r*nat_rb_a, nat_rb_a);
    // ... same pattern for B and C
}
```

**After**:
```cpp
// GPU row_copy for A/B/C packing — encode-only, zero CPU waits.
if (pad_a) {
    tmp_a = alloc_temp_buffer(mb_a * batch_size);
    NSUInteger src_off_a = 0;
    id<MTLBuffer> src_buf_a = metal_buffer_for_ptr(a, &src_off_a);
    dispatch_row_copy(src_buf_a, src_off_a,
                      nat_rb_a, stridea * elem,   // src: natural layout
                      tmp_a, 0,
                      mps_rb_a, mb_a,             // dst: padded layout
                      nat_rb_a, rows_a, batch_size);
    // ... same pattern for B and C
}

// Non-blocking commit: flush row_copy encoders to a separate CB so MPS
// GEMM gets a fresh CB. The serial queue ensures row_copy finishes first.
if (pad_a || pad_b || (pad_c && beta != 0.0f))
    commit_command_buffer();
```

## Key Discovery: Compute-before-MPS Crash

GPU compute encoder (row_copy) followed by MPS GEMM encoding on the **same** command buffer causes silent crashes (exit 138/139). This was verified across multiple approaches:

- Same CB: **crash** (all variants)
- MTLFence between them: **crash**
- memoryBarrierWithScope: **crash**
- Non-blocking commit between them: **works** (separate CBs, serial queue ordering)

The C unpack row_copy (encode-only AFTER MPS GEMM) works fine on the same CB — the issue is specifically compute-before-MPS ordering.

## What Was NOT Changed (by design)

1. **`dispatch_mps_gemm` (single GEMM)** — Small padded matrices (<= 4096 elements) still use CPU cblas with `CT2_COMMIT_AND_WAIT()`. Extending GPU row_copy here triggers the same compute-before-MPS issue and produces corrupt data.

2. **Float16 small padded batched GEMMs** — `batch_cpu_gemm_f16` still uses CPU cblas with float32 accumulation. MPS float16 GEMM has insufficient precision for translation quality (causes wrong translations like "Braunfuchs" -> "Braunf").

3. **Float32 small padded batched GEMMs** — `batch_cpu_gemm_f32` still uses CPU cblas. These go through `dispatch_mps_gemm` per-element, which has its own CPU cblas fallback.

## Failed Approaches

| Approach | Result |
|----------|--------|
| GPU row_copy + MPS on same CB | Crash (exit 138/139) |
| Per-element dispatch for small padded | More syncs (each dispatch_mps_gemm hits own cblas fallback) |
| Remove float16 cblas threshold | 4/9 float16 tests fail (precision loss) |
| GPU row_copy in dispatch_mps_gemm (single) | Crash / corrupt data |

## Sync Trace (bench_faster_whisper, beam_size=5)

The `dispatch_mps_gemm_batched_padded` sync that was at line ~796 is **completely eliminated** from the trace.

Remaining syncs:
```
14512  primitives_gemm.mm:1018   (batch_cpu_gemm_f16 — float16 precision)
 2435  primitives_reduction.mm:127 (amax — returns scalar, unavoidable)
 1065  devices.cc:162            (synchronize_stream)
 1037  primitives_memory.mm:80   (indexed_fill)
  769  multinomial_metal.mm:21   (sampling)
  268  topk_metal.mm:44          (top-k)
   14  primitives_memory.mm:90   (misc)
   14  primitives_beam_search.mm:89 (beam search)
```

## Benchmark

Whisper-large-v3-turbo, 60s audio, beam_size=5:
- **Metal 1.13x CPU** (this session)
- Was 0.22x before M11 optimizations began

## Test Results

| Test Suite | Result |
|------------|--------|
| test_translation | 90/90 PASS |
| test_float16_translation | 9/9 PASS |
| test_beam_search | 39/39 PASS |
| test_whisper | 13/13 PASS |
| test_faster_whisper | 8/8 PASS |
| **Total** | **159/159 PASS** |

## Files Modified

1. `src/metal/primitives_gemm.mm` — GPU row_copy for A/B/C packing in `dispatch_mps_gemm_batched_padded()` (+31/-29 lines)
