# M12 Code Review Report

**Date**: 2026-03-11
**Scope**: M12.1–M12.7 (all changes since M11)
**Hardware**: Apple M4, macOS 15
**Reviewer**: Claude (automated)
**Status**: All HIGH issues FIXED (2026-03-11)

---

## Summary

Reviewed 6 files across 7 milestones. Found **8 HIGH**, **5 MEDIUM**, and **3 LOW** severity issues.

| Severity | Count | Categories |
|----------|-------|------------|
| HIGH | 8 | Data corruption (2), Numerical correctness (1), Maintenance (2), Misleading output (1), Profiling bug (1), Thread safety (1), Documentation (1) |
| MEDIUM | 5 | Data corruption (1), Overflow (1), Test coverage (2), Portability (1) |
| LOW | 3 | Dead code (1), Benchmark coverage (1), Build docs (1) |

---

## Files Reviewed

| File | Milestones | Lines Changed |
|------|-----------|---------------|
| `src/metal/primitives_gemm.mm` | M12.6 | ~200 (GPU INT8 kernels) |
| `src/metal/primitives_beam_search.mm` | M12.1 | ~20 (encode_barrier) |
| `src/types.cc` | M12.5 | ~40 (BF16→FP16 promotion) |
| `src/models/model.cc` | M12.5 | ~15 (BF16 warning) |
| `src/decoding.cc` | M12.4 | ~30 (decode profiler) |
| `src/metal/allocator.mm` | M12.3 | ~80 (pointer cache) |

---

## HIGH Severity

### H1. Missing `protect_buffer` for INT8 encode-only buffers

**File**: `src/metal/primitives_gemm.mm` (`dispatch_int8_gemm`)
**Risk**: Use-after-free / data corruption
**Details**: The GPU int8→f32 conversion kernels read from input buffers in encode-only mode. If the caller frees those buffers before the command buffer executes, the GPU reads stale/reused memory. This is the exact same pattern as the M10.1 gather bug.
**Fix**: Add `protect_buffer()` calls for input A and B buffers before encoding the int8→f32 kernels.

---

### H2. MSL `round()` vs C++ `lroundf()` rounding difference

**File**: `src/metal/primitives_gemm.mm` (`float32_round_to_int32_strided` kernel)
**Risk**: Silent numerical divergence between GPU and CPU paths
**Details**: MSL `round()` uses banker's rounding (round-half-to-even: 0.5→0, 1.5→2). C++ `lroundf()` uses ties-away-from-zero (0.5→1, 1.5→2). For INT8 GEMM outputs that land exactly on .5, the GPU path produces different int32 results than the CPU path. The M12.6 debug comparison showed zero mismatches because tested values didn't hit exact .5 boundaries, but this is a latent correctness issue.
**Fix**: Replace MSL `round()` with `floor(x + 0.5f)` to match C++ `lroundf()` behavior.

---

### H3. Duplicated `static const bool native_bf16` variable

**File**: `src/types.cc` (lines ~208 and ~290)
**Risk**: Future divergence / maintenance bug
**Details**: The same `static const bool native_bf16 = read_bool_from_env("CT2_MPS_NATIVE_BF16")` appears at two independent sites (BFLOAT16 case and INT8_BFLOAT16 case). If one is updated and the other isn't, behavior silently diverges.
**Fix**: Extract to a single helper:
```cpp
static bool mps_native_bf16() {
  static const bool v = read_bool_from_env("CT2_MPS_NATIVE_BF16");
  return v;
}
```

---

### H4. BF16 warning fires on non-MPS devices

**File**: `src/models/model.cc` (lines ~893-903)
**Risk**: Misleading user-facing warning
**Details**: The warning says "on MPS" but the condition only checks compute type pairs (BF16→FP16), not the device. If CUDA falls back from BF16→FP16 (e.g., on older GPUs), users see "automatically promoted ... on MPS" which is incorrect.
**Fix**: Add `device == Device::MPS` guard to the condition. Note: `device` variable may need to be captured or passed to this scope.

---

### H5. Off-by-one step count in decode profiler

**File**: `src/decoding.cc`
**Risk**: Incorrect profiling metrics
**Details**: When early exit triggers (`break`), `++prof.steps` at the loop bottom is skipped. The final step's work is counted in timings but not in the step count, inflating per-step averages.
**Fix**: Move `++prof.steps` to the top of the loop body or increment before each `break` statement.

---

### H6. Stale pointer cache entries after free+reallocate

**File**: `src/metal/allocator.mm` (interior pointer cache)
**Risk**: Data corruption (returns wrong MTLBuffer for a pointer)
**Details**: When a buffer is freed, only `_live` map entries keyed by `base_ptr` are removed. If an interior pointer was cached with a different hash bucket, the stale entry survives. If a smaller buffer is later allocated at the same base address, the stale interior entry now points to the old (now-freed) MTLBuffer with incorrect offset/size.
**Fix**: Invalidate all cache entries on `free()` by scanning, or add a generation counter that invalidates stale entries on lookup.

---

### H7. Thread-unsafe counter reads in allocator

**File**: `src/metal/allocator.mm`
**Risk**: Data race (undefined behavior under ThreadSanitizer)
**Details**: Cache hit/miss counters are read without holding the mutex. On ARM64 this is practically safe (atomic loads for aligned integers), but it's technically UB per C++ memory model.
**Fix**: Use `std::atomic<int64_t>` for the counters, or read them under the lock.

---

### H8. Comment/code mismatch in cache size

**File**: `src/metal/allocator.mm`
**Risk**: Confusion / wrong assumptions in future maintenance
**Details**: Comment says "64 entries" but the code defines `CACHE_SIZE = 256`.
**Fix**: Update the comment to say "256 entries".

---

## MEDIUM Severity

### M1. No `protect_buffer` for batched INT8 GEMM

**File**: `src/metal/primitives_gemm.mm` (~lines 1465-1545)
**Risk**: Same as H1 but for batched path
**Details**: The batched INT8 GEMM path also encodes int8→f32 kernels in encode-only mode without protecting input buffers. Same fix as H1.

---

### M2. `ct2_u32()` not used consistently in INT8 helper dispatch

**File**: `src/metal/primitives_gemm.mm` (`encode_int8_to_float32`, `encode_float32_to_int32`)
**Risk**: Silent overflow for dimensions > 2^31
**Details**: Some parameters are cast with `(uint32_t)` instead of `ct2_u32()`, which checks for overflow and negative values. While current model dimensions are well within range, this violates the project convention established in post-M4.8.

---

### M3. No unit test for INT8 GPU kernel rounding edge cases

**Risk**: Rounding difference (H2) untested
**Details**: The M12.6 debug comparison verified zero mismatches for the tested shapes, but didn't specifically test values at exact .5 boundaries where banker's rounding diverges from ties-away-from-zero.

---

### M4. `high_resolution_clock` used instead of `steady_clock`

**File**: `src/decoding.cc`
**Risk**: Timer can go backwards on clock adjustments
**Details**: `steady_clock` is guaranteed monotonic; `high_resolution_clock` may alias to `system_clock` on some platforms (not macOS, but portability concern for upstream).

---

### M5. INT8 dequantize_gemm_output not tested with activation functions post-M12.6

**Risk**: Regression in edge cases
**Details**: The `ct2_safe_tanh` fix (M10.3) in `dequantize_gemm_output` wasn't re-verified after the M12.6 pipeline changes. If the GPU kernel chain changes the input distribution to the dequantize step, edge-case NaN behavior could resurface.

---

## LOW Severity

### L1. Dead code: old CPU int8 conversion path

**File**: `src/metal/primitives_gemm.mm`
**Details**: The old `vDSP_vflt8` / `lroundf` CPU path is fully replaced by GPU kernels. Any commented-out code or debug scaffolding should be cleaned up.

---

### L2. No benchmark for batched INT8 GEMM

**Details**: M12.6 benchmarks only cover single GEMM dispatch. The batched path (used for multi-head attention) wasn't separately profiled for the sync elimination improvement.

---

### L3. RUY build workaround not documented in CMakeLists.txt

**Details**: The `CMAKE_POLICY_VERSION_MINIMUM=3.5` workaround for RUY's cpuinfo dependency is only documented in the M12.7 milestone report, not in the build file itself. A comment in CMakeLists.txt would help future developers.

---

## Approved (No Issues)

### `src/metal/primitives_beam_search.mm` (M12.1)
- `encode_barrier()` implementation is clean and correct
- Properly uses `MTLBarrierScopeBuffers` with serial dispatch
- No sync elimination concerns — barriers are encode-only by design

---

## Recommendations (Priority Order)

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | H1+M1: Add `protect_buffer` for INT8 buffers | Small | Prevents data corruption |
| 2 | H2: Fix rounding to match CPU (`floor(x+0.5f)`) | Trivial | Numerical correctness |
| 3 | H4: Add device guard to BF16 warning | Trivial | User-facing fix |
| 4 | H5: Fix step counter off-by-one | Trivial | Profiling accuracy |
| 5 | H3: Consolidate `native_bf16` helper | Small | Maintenance |
| 6 | H6: Fix stale cache entries | Medium | Correctness under reallocation |
| 7 | H7: Atomic counters | Trivial | Thread safety compliance |
| 8 | H8: Fix comment | Trivial | Documentation accuracy |
| 9 | M2: Use `ct2_u32()` consistently | Trivial | Convention compliance |
| 10 | M4: Switch to `steady_clock` | Trivial | Portability |
| 11 | M3+M5: Add rounding + activation tests | Medium | Test coverage |
