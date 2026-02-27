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
- `ptest/batch_compare_test.py`: all OK (beam_size=2, both hypotheses match CPU)
- `ptest/beam_step_test.py`: ALL PASS (beam=2 and beam=4, max_decoding_length=1..11)
- `ptest/beam_toks_test.py`: ALL PASS (beam=1,2,4; max_len=1,2,3,5,10)

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

Assumes: repo checked out, `ptest/opus-mt-en-de/` model already present,
Xcode command-line tools installed.  Run all commands from the repo root.

### 1 — Create the conda environment

```bash
conda create -n ct2 python=3.14 -y
conda run -n ct2 pip install \
    "pybind11==2.11.1" setuptools wheel \
    numpy "pyyaml>=5.3,<7" \
    transformers sacremoses sentencepiece
```

### 2 — Configure CMake

```bash
cmake -S . -B build \
    -DWITH_METAL=ON \
    -DWITH_ACCELERATE=ON \
    -DWITH_MKL=OFF
```

`WITH_ACCELERATE=ON` picks up Apple's Accelerate framework for CPU BLAS.
`WITH_MKL=OFF` is required on Apple Silicon (no MKL).

### 3 — Build the C++ library

```bash
cmake --build build --target ctranslate2 -j$(sysctl -n hw.logicalcpu)
# Produces: build/libctranslate2.4.dylib (and versioned + unversioned symlinks)
```

### 4 — Install the library into the conda env

The Python extension links against `@rpath/libctranslate2.4.dylib`.
Its rpath list is (in order):
1. `/opt/anaconda3/envs/ct2/lib`  ← checked first
2. `/usr/local/lib`

Copy the built library into the conda env so the install is self-contained
and does not pollute the system library path:

```bash
CT2_ENV_LIB=$(conda run -n ct2 python -c "import sys; print(sys.prefix)")/lib

cp build/libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/"
ln -sf libctranslate2.4.7.1.dylib "$CT2_ENV_LIB/libctranslate2.4.dylib"
ln -sf libctranslate2.4.dylib      "$CT2_ENV_LIB/libctranslate2.dylib"
```

After a library rebuild (step 3), only the `cp` line needs to be repeated —
the symlinks stay valid as long as the version number does not change.

### 5 — Build and install the Python extension (editable)

```bash
cd python
CTRANSLATE2_ROOT=../build \
CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.logicalcpu) \
    conda run -n ct2 pip install -e . --no-build-isolation
cd ..
```

`--no-build-isolation` lets pip use the pybind11 already in the env instead
of downloading a separate build-time copy.  The `-e` (editable) install
builds `_ext.cpython-314-darwin.so` in-place inside `python/ctranslate2/`
so re-running `pip install` after a library rebuild is all that is needed.

### 6 — Run the end-to-end tests

All scripts must be run from `ptest/` (model path `opus-mt-en-de` is relative):

```bash
cd ptest
conda run -n ct2 python full_e2e_test.py      # 5 sentences × beam 1/2/4 — quickest overall check
conda run -n ct2 python batch_compare_test.py  # beam_size=2 hypotheses vs CPU
conda run -n ct2 python beam_step_test.py      # beam=2,4 × max_len=1..11
conda run -n ct2 python beam_toks_test.py      # beam=1,2,4 × max_len=1,2,3,5,10
```

After a library-only change (no Python binding changes), only steps 3–4 are
needed before re-running tests — skip steps 1, 2, and 5.

---

## Test Results (conda env `ct2`, Python 3.14, Apple M4)

### ptest/batch_compare_test.py

```
Input: ['▁The', '▁cat', '▁sat', '▁on', '▁the', '▁mat', '.', '</s>']

=== beam_size=2 results ===
  hyp[0]: OK  cpu=['▁Die', '▁Katze', '▁saß', '▁auf', '▁der', '▁Matt', 'e', '.']
              metal=['▁Die', '▁Katze', '▁saß', '▁auf', '▁der', '▁Matt', 'e', '.']
  hyp[1]: OK  cpu=['▁Die', '▁Katze', '▁saß', '▁auf', '▁dem', '▁Bett', '.']
              metal=['▁Die', '▁Katze', '▁saß', '▁auf', '▁dem', '▁Bett', '.']

=== Metal beam_size=1 vs beam_size=2 hypothesis[0] ===
  beam1_hyp[0] vs beam2_hyp[0]: OK

  cpu_beam1 vs metal_beam1: OK

=== Two separate sentences vs batched ===
  Batch 2 identical sentences: OK
  Batch-of-2 result[0] vs single beam1: OK
```

### ptest/beam_step_test.py

```
=== beam_size=2 ===
  max_len= 1: PASS  …  max_len=11: PASS   (all 11 lengths)

=== beam_size=4 ===
  max_len= 1: PASS  …  max_len=11: PASS   (all 11 lengths)
```

### ptest/full_e2e_test.py

```
beam | input                                    | CPU output                        | Metal output                      | match
1    | The cat sat on the mat.                  | Die Katze saß auf der Matte.      | Die Katze saß auf der Matte.      | PASS
2    | The cat sat on the mat.                  | Die Katze saß auf der Matte.      | Die Katze saß auf der Matte.      | PASS
4    | The cat sat on the mat.                  | Die Katze saß auf der Matte.      | Die Katze saß auf der Matte.      | PASS
1    | Hello world, this is a test.             | Hallo Welt, das ist ein Test.     | Hallo Welt, das ist ein Test.     | PASS
2    | Hello world, this is a test.             | Hallo Welt, das ist ein Test.     | Hallo Welt, das ist ein Test.     | PASS
4    | Hello world, this is a test.             | Hallo Welt, das ist ein Test.     | Hallo Welt, das ist ein Test.     | PASS
1    | Machine translation is an interesting…   | Maschinelle Übersetzung ist…      | Maschinelle Übersetzung ist…      | PASS
2    | Machine translation is an interesting…   | Maschinelle Übersetzung ist…      | Maschinelle Übersetzung ist…      | PASS
4    | Machine translation is an interesting…   | Maschinelle Übersetzung ist…      | Maschinelle Übersetzung ist…      | PASS
1    | a b c                                    | a b c                             | a b c                             | PASS
2    | a b c                                    | a b c                             | a b c                             | PASS
4    | a b c                                    | a b c                             | a b c                             | PASS
1    | a b c d e f g                            | a b c d e f g                     | a b c d e f g                     | PASS
2    | a b c d e f g                            | a b c d e f g                     | a b c d e f g                     | PASS
4    | a b c d e f g                            | a b c d e f g                     | a b c d e f g                     | PASS

ALL PASS
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
