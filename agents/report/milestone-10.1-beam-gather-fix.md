# Milestone 10.1 — Fix Metal beam_size>1: synchronous gather (use-after-clone hazard)

**Date:** 2026-02-27
**Status:** ✅ DONE

---

## Summary

M10.1 diagnoses and fixes a correctness bug that caused Metal `beam_size>1`
translation to produce completely wrong output (infinite loops of repeated
tokens), while `beam_size=1` and batched single-beam translation worked
correctly.

**Root cause:** The Metal gather kernel was encode-only (deferred).  The
in-place `Gather::operator()(data, input)` pattern creates a temporary
`StorageView clone(std::move(data))`, encodes the Metal gather kernel
(which holds a reference to clone's backing `MTLBuffer`), then immediately
frees the clone when the function returns.  The freed `MTLBuffer` is returned
to `MetalAllocator._pool` and may be reused by the very next allocation.
When the GPU later executes the encoded gather (at the next `commit_and_wait`),
it reads the **new data** written by the reusing allocation — garbage values
that propagate through subsequent decoder steps.

**Fix:** One line: `ctranslate2::metal::commit_and_wait()` at the end of
`dispatch_gather`.  This makes gather synchronous: the GPU reads from the
clone's buffer while it is still valid, before the clone is freed.

**Tests:**
- `tests/metal/e2e/test_beam_search.py`: 28/28 PASS (beam sweep + batch consistency)
- `tests/metal/e2e/test_translation.py`: 90/90 PASS (CPU vs Metal, 5 sentences × 3 beams × 6 max_len)

---

## Root Cause Analysis

### Why beam_size=1 was unaffected

After the first decoder step, the surviving beam's `gather_indices` are
`[0, 1, 2, …, beam_size-1]` — strictly increasing.
`support_gather_batch_inplace` checks:

```cpp
return (input.device() == Device::CPU
        && input.size() <= data.dim(0)
        && std::adjacent_find(input_begin, input_end,
                              std::greater_equal<int32_t>()) == input_end);
```

For `beam_size=1`, indices `[0]` — trivially strictly increasing.
For batch-of-2 `beam_size=1`, indices `[0, 1]` — also strictly increasing.

In both cases the function returns `true`, and the gather is done as a
**CPU in-place scatter** (no Metal kernel, no use-after-clone hazard).

### Why beam_size=2 fails

After step=0 with `beam_size=2`, both surviving beams come from parent beam 0
(the best hypothesis is shared at the first step).  The resulting
`gather_indices = [0, 0]`.

`adjacent_find` finds `0 >= 0` → NOT strictly increasing.
`support_gather_batch_inplace` returns `false`.

The code falls into:

```cpp
StorageView clone(std::move(data));   // clone = temporary source
operator()(clone, input, data);       // Metal kernel: reads clone, writes data
// clone destroyed here               // MTLBuffer returned to _pool!
```

The Metal gather kernel is **encoded but deferred**.  When the clone's
destructor runs, `MetalAllocator::free()` puts its `MTLBuffer` into `_pool`.
The very next allocation may pull it from the pool and overwrite it with new
data.  When the GPU executes the gather (triggered by the next
`commit_and_wait` from GEMM or softmax), it reads the overwritten buffer.

### Diagnostic

The bug was confirmed with `CT2_BEAM_DEBUG=1` (per-step prints added to
`decoding.cc`):

```
step=0: CPU topk_ids = [55, 119, 388, 26]
step=0: Metal topk_ids = [55, 119, 388, 26]   ← identical, correct

step=1: CPU   beam0 max_logit = 12.42
step=1: Metal beam0 max_logit =  9.424         ← diverges here
```

Divergence at step=1 is exactly where `update_state` calls `Gather` with
`gather_indices=[0,0]` (after step=0).  The gather encodes on Metal, then the
clone is freed, its buffer is reused for K/V projection output (GEMM), and the
GPU gather reads that fresh GEMM output instead of the original KV cache.

---

## Fix

**File:** `src/metal/ops_norm_gather.mm`
**Change:** Add `commit_and_wait()` at the end of `dispatch_gather`.

```cpp
// M10.1 fix: commit_and_wait() at the end makes gather synchronous.
//
// The in-place Gather::operator()(data, input) pattern:
//   StorageView clone(std::move(data));     // clone = temporary source
//   operator()(clone, input, data);         // encode gather: reads clone, writes data
//   // clone is freed here                  // src buffer returned to allocator pool
//
// If the gather were encode-only (deferred), the GPU would run after the
// clone's MTLBuffer has been returned to the pool and potentially reused for
// another allocation, causing the GPU to read wrong (stale / overwritten) data.
// Making gather synchronous eliminates this use-after-clone hazard.
static void dispatch_gather(...) {
  ...
  [enc endEncoding];
  // Flush immediately so the GPU reads src before the caller can free it.
  ctranslate2::metal::commit_and_wait();
}
```

---

## General Pattern

> **Any encode-only Metal op that reads from a TEMPORARY buffer (not a
> persistent StorageView) must call `commit_and_wait()` before the caller
> can free the buffer.**

Ops that read from long-lived `StorageView` allocations (model weights,
persistent caches) are safe to leave encode-only.  Ops that read from
temporaries created within the call stack are hazardous unless made
synchronous.

The gather op is the only op in the current Metal backend that is called
with a temporary source.  All other ops (GEMM, softmax, layernorm, etc.)
read from allocations that outlive the dispatch call.

---

## Files Modified

| File | Change |
|------|--------|
| `src/metal/ops_norm_gather.mm` | Added `commit_and_wait()` at end of `dispatch_gather`; added M10.1 comment block |
| `src/metal/primitives_gemm.mm` | Added conditional `commit_and_wait()` before MPS-row-padding memcpy (see GEMM padding hazard below) |

## GEMM Padding Hazard (discovered during build verification)

When building the Python package and running end-to-end tests, a **second
correctness bug** was found in `dispatch_mps_gemm`:

**Problem:** When matrix A or B has a row-byte stride smaller than MPS
requires (`nat_rb < mps_rb`), the code copies the source data into a padded
temp buffer using a CPU `memcpy`.  Those source buffers may have been written
by a preceding GPU kernel (layernorm, transpose, etc.) that is still encoded
but not yet committed.  On unified memory, **GPU writes are not visible to the
CPU until the command buffer has been committed and completed**.  Without a
sync point the CPU `memcpy` reads stale pre-write data, producing wrong GEMM
output.

The `pad_c` output path already had `commit_and_wait()` after the GEMM encode
(to unpack the temp result back to `c`).  The input side had no equivalent.

**Fix:** Added a conditional flush before the padding section:

```cpp
// Padding hazard: CPU memcpy from A/B reads GPU-written data.
// Flush before reading if any input or beta-accumulate padding is needed.
if (pad_a || pad_b || (pad_c && beta != 0.0f))
    ctranslate2::metal::commit_and_wait();
```

This is only triggered when MPS row-padding is actually required (small `n`
values such as `n=2` in the beam-search output projection), so the common
no-padding path pays no extra synchronisation cost.

## Additional Linker Fixes (same milestone)

Building the Python wheel exposed missing explicit instantiations:

| File | Change |
|------|--------|
| `src/metal/ops_sdpa.mm` | Added integer-type instantiations required by `TYPE_DISPATCH` in `flash_attention_metal.mm`; improved `if constexpr` chain |
| `src/ops/gumbel_max_metal.mm` | Added `DECLARE_IMPL` for `float16_t`, `bfloat16_t` |
| `src/ops/mean_metal.mm` | Added `DECLARE_IMPL` for `float16_t`, `bfloat16_t` |
| `src/ops/median_filter_metal.mm` | Added `DECLARE_IMPL` for `float16_t`, `bfloat16_t` |
| `src/ops/multinomial_metal.mm` | Added `DECLARE_IMPL` for `float16_t`, `bfloat16_t` |
| `src/ops/topp_mask_metal.mm` | Added `DECLARE_IMPL` for `float16_t`, `bfloat16_t` |
| `src/ops/awq_metal.mm` | New stub: throws at runtime (AWQ not yet supported on Metal) |
| `src/ops/nccl_metal.mm` | New stub: throws at runtime (distributed ops require NCCL/MPI) |
| `CMakeLists.txt` | Added `awq_metal.mm` and `nccl_metal.mm` to `METAL_SOURCES` |

---

## Building and Testing from Scratch

See `agents/report/e2e-testing.md` for full build and test instructions.

Quick summary from the repo root:

```bash
# Build C++ library
cmake --build build -j$(sysctl -n hw.logicalcpu)

# Copy into conda env (no sudo, no /usr/local)
CT2_ENV_LIB="$(python -c 'import sys; print(sys.prefix)')/lib"
cp build/libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/"

# Run e2e tests
export CT2_TEST_DATA=/path/to/data
python tests/metal/e2e/test_translation.py
python tests/metal/e2e/test_beam_search.py
python tests/metal/e2e/test_whisper.py
```

---

## Test Results (conda env `ct2`, Python 3.14, Apple M4)

```
tests/metal/e2e/test_translation.py:   90/90 passed — ALL PASS
tests/metal/e2e/test_beam_search.py:   28/28 passed — ALL PASS
tests/metal/e2e/test_whisper.py:        3/3  passed — ALL PASS
```

---

## Key Findings

1. **Deferred GPU ops are hazardous with temporary sources.**  The Metal
   backend's encode-only model is safe when all source buffers outlive the
   command buffer.  The in-place `Gather` op violated this invariant because
   the source (`clone`) is freed before the command buffer is committed.

2. **The allocator pool is the mechanism of corruption.**  ARC retains the
   `MTLBuffer` in the pool, so the buffer is not deallocated — but its
   *contents* are overwritten when a new `StorageView` is allocated from the
   pool.  The encoded kernel still holds a Metal reference to the buffer
   (preventing deallocation) but cannot prevent the CPU from writing new data
   through the same buffer handle.

3. **Unified memory does NOT mean CPU–GPU cache coherence is free.**  Even on
   Apple Silicon, CPU reads of GPU-written Shared buffers are only guaranteed
   to be up-to-date after the command buffer has committed and completed.
   This applies both to the gather bug (GPU reads stale pool data) and the
   GEMM padding bug (CPU reads stale input before commit).

4. **commit_and_wait overhead is acceptable for gather.**  Gather is already
   benchmarked as "never faster than CPU" for standalone calls (random-access
   scatter is memory-bandwidth-bound and the CB overhead dominates).  Making
   it synchronous does not change the performance profile; in a full pipeline
   the cost is absorbed into the next existing commit boundary.

5. **GEMM padding flush is conditional, not universal.**  By checking
   `pad_a || pad_b || (pad_c && beta != 0)` before flushing, the common
   no-padding path (large matrix dimensions) pays zero extra synchronisation
   cost.

6. **CT2_BEAM_DEBUG env var** was useful for bisecting the failure to a single
   decode step.  The debug prints were removed before the final commit.

---

## Commits

```
7e904c85  M10.1 — Fix Metal beam_size>1: synchronous gather (use-after-clone hazard)
5a76b818  M10.1 — Add missing Metal explicit instantiations and linker stubs
```
*(The GEMM padding fix in primitives_gemm.mm is a working change not yet committed.)*
