# M11 Code Review — Recent Changes (Commits 41–48)

**Date**: 2026-03-10
**Scope**: All source code changes from `07c2fcf9` (M11.25) through `f12d2300` (OPT-1/OPT-3)
**Files reviewed**: `allocator.mm`, `primitives_gemm.mm`, `primitives_memory.mm`, `sampling.cc`,
`decoding_utils.cc`, `whisper.cc`, `utils.mm`, `utils.h`, `indexed_fill.metal`, `msl_strings.h`

---

## Summary

| Severity | Count | Description |
|----------|------:|-------------|
| **BUG (P0)** | 1 | Missing `protect_buffer` for `previous_ids` in `RepetitionPenalty::apply()` |
| **BUG (P1)** | 1 | F16 `indexed_fill` pre-sync skip has unclear root cause — fragile |
| **WEAKNESS (P2)** | 3 | MPS cache unbounded growth; OPT-1 regression for Metal-resident features; `clear_cache()` ordering |
| **MISSING TESTS** | 5 | `buffer_for_ptr` O(log N), `protect_buffer`, GEMM cache, `indexed_fill` f16 encode-only, `sampling.cc` batched memcpy |

---

## P0 — Bugs (Must Fix)

### BUG-1: Missing `protect_buffer` for `previous_ids` in `RepetitionPenalty::apply()`

**File**: `src/decoding_utils.cc:59-81`
**Severity**: P0 — potential data corruption in beam search with `repetition_penalty > 1.0`

The M11.29 fix correctly protects `previous_scores` from premature reuse, but **misses `previous_ids`**:

```cpp
StorageView previous_ids = sequences.to(device);         // ← local, freed on return
StorageView previous_scores(device, dtype);
ops::Gather(...)(logits, previous_ids, previous_scores);  // encode-only: reads previous_ids

#ifdef CT2_WITH_METAL
if (device == Device::METAL)
    metal::protect_buffer(previous_scores.buffer());       // ✓ protected
    // previous_ids NOT protected                          // ✗ BUG
#endif

primitives<D>::penalize_previous_tokens(
    logits, previous_scores, previous_ids, ...);           // encode-only: reads BOTH
```

When `apply()` returns, `previous_ids`'s destructor frees its Metal buffer back to the pool.
If any subsequent allocation (e.g., TopK intermediates in sampling) reuses that buffer before
the command buffer is committed, the GPU kernel reads garbage indices → wrong penalties →
incorrect beam scores → silent output corruption.

This is the exact same class of bug as M10.1 (Gather use-after-clone). The GPU kernel
`penalize_previous_tokens` reads `previous_ids` as `const int32_t*` — it's a pure read-after-free
race on the Metal buffer.

**Why it may not have manifested yet**: `repetition_penalty` defaults to 1.0 (disabled) in
Whisper's `DecodingOptions`. The `RepetitionPenalty` processor is only added when
`penalty > 1.0`. Standard Whisper benchmarks use default options, so this code path is untested.

**Fix**: Add `metal::protect_buffer(previous_ids.buffer())` alongside the existing protection:

```cpp
if (device == Device::METAL) {
    metal::protect_buffer(previous_scores.buffer());
    metal::protect_buffer(previous_ids.buffer());
}
```

---

## P1 — Fragile Code (Should Fix)

### BUG-2: F16 `indexed_fill` pre-sync skip — unclear root cause

**File**: `src/metal/primitives_memory.mm:96-106`

```cpp
metal::protect_buffer(indices);
if constexpr (!std::is_same_v<T, ctranslate2::float16_t>) {
    CT2_COMMIT_AND_WAIT();   // f32/bf16: sync required
}                             // f16: NO sync
```

The comment admits: *"float32: REQUIRED — without the pre-sync, whisper-base f32 produces garbage
output. Root cause unclear (possible MPS driver ordering issue with f32 GEMMs)."*

**Concern**: The f16 skip is justified by *"Metal guarantees in-order execution [within a command
buffer]"*. But `dispatch_mps_gemm_batched_padded()` calls `commit_command_buffer()` (non-blocking)
at line 1069–1070 to flush row-copy encoders before the MPS GEMM. This creates a NEW command
buffer for the MPS GEMM. The `indexed_fill` kernel encodes into THIS or a LATER command buffer.

Metal command queues are **concurrent by default** — command buffers from the same queue can
execute concurrently on the GPU. The only ordering guarantee is that CBs are *scheduled* (started)
in submission order, but they may *overlap* in execution. If the MPS GEMM writes to a buffer and
`indexed_fill` reads from the same buffer in a different CB, there is no GPU-side dependency
tracking to ensure the write completes before the read.

In practice, the f16 path typically doesn't hit the padded path (m=1 f16 goes through custom GEMV
which doesn't split CBs), so `indexed_fill` ends up in the same CB as the GEMM. But this is
fragile — a future change that introduces a `commit_command_buffer()` between the GEMM and
`indexed_fill` would break correctness silently.

**Recommendation**: Either:
1. **Find and document the real root cause** of the f32 failure, or
2. **Remove the `if constexpr` and always sync** (costs ~0.4ms × number of `indexed_fill` calls per
   inference, likely <5ms total), or
3. **Add a Metal event/fence** between the write and read instead of a full `commit_and_wait()`.

---

## P2 — Weaknesses (Track)

### WEAK-1: MPS GEMM cache grows unbounded

**File**: `src/metal/primitives_gemm.mm:112` (static cache, never evicted)

```cpp
static std::unordered_map<MpsGemmKey, MPSMatrixMultiplication*, MpsGemmKeyHash> cache;
```

Each unique `(transpose, m, n, k, alpha, beta, batch_size)` tuple creates a permanent cache entry.
For Whisper, the set of GEMM shapes is bounded by the model architecture (~50-100 unique shapes).
But for a general-purpose server processing variable-length inputs with different batch sizes,
the cache grows monotonically. Each `MPSMatrixMultiplication` object is ~1-2KB.

**Impact**: Low for Whisper. Could matter for a long-running server with diverse workloads.

**Recommendation**: Add LRU eviction or a `clear_gemm_cache()` function callable from
`clear_device_cache('metal')`.

### WEAK-2: OPT-1 `StorageView(features)` regression for Metal-resident features

**File**: `src/models/whisper.cc:667-671`

`sync_copy()` called `synchronize_stream(device)` before copying — safe for any source device.
`StorageView(features)` uses the copy constructor which calls `copy_from()`. For same-device
Metal→Metal copies, this does `primitives<METAL>::copy()` which is `std::memcpy` — **no GPU sync**.

If the caller passes Metal-resident features with pending GPU writes (e.g., features produced by
a custom preprocessing pipeline on GPU), the copy reads stale data.

**Impact**: Zero for current use (Python/faster_whisper always passes CPU features). The C++ API
could theoretically hit this, but no existing code does.

**Recommendation**: Document the assumption (CPU features only) in a comment, or add a
`synchronize_stream(features.device())` guard for the Metal case.

### WEAK-3: `clear_cache()` calls `commit_and_wait()` outside the mutex

**File**: `src/metal/allocator.mm:170-185`

```cpp
void clear_cache() override {
    metal::commit_and_wait();           // calls flush_pending_frees() → acquires _mutex
    std::lock_guard<std::mutex> lock(_mutex);   // re-acquires _mutex
    // ... release pool buffers
}
```

Between `flush_pending_frees()` returning (inside `commit_and_wait()`) and `clear_cache()`
acquiring the mutex, another thread could allocate buffers from the pool. Those buffers would
be missed by the release loop.

**Impact**: None in current code — `clear_cache()` is documented as "not thread-safe" in
`ReplicaPool::clear_cache()`. But the code structure is misleading.

**Recommendation**: Add a comment noting the thread-safety assumption, or restructure to
acquire the mutex once.

---

## Missing Tests

### TEST-1: `buffer_for_ptr` O(log N) correctness

**Current coverage**: Zero — `allocator_test.mm` tests allocate/free/pool but not `buffer_for_ptr`.

**What to test**:
- Allocate multiple buffers, look up a pointer within each (base, middle, last byte)
- Look up a sub-pointer (offset into a buffer)
- Verify correct MTLBuffer and offset_out returned
- Verify throws for pointer outside any allocation
- Verify after free (pointer no longer in `_live`)

### TEST-2: `protect_buffer` + `flush_pending_frees` lifecycle

**Current coverage**: Zero — no tests for the deferred-free mechanism.

**What to test**:
- Allocate → protect → free → verify NOT in pool (in pending)
- `flush_pending_frees()` → verify NOW in pool
- Protect with sub-pointer (not base) via O(log N) path
- `protect_buffer_by_base()` variant

### TEST-3: `indexed_fill` f16 encode-only correctness

**Current coverage**: `primitives_test.mm` tests float32 `indexed_fill` only.

**What to test**:
- f16 `indexed_fill` followed by a read (after `commit_and_wait()`)
- f16 `indexed_fill` interleaved with GEMM (encode-only chain correctness)
- bf16 `indexed_fill` (verify pre-sync fires)
- Zero-length `num_indices` (early return path)

### TEST-4: MPS GEMM cache correctness

**Current coverage**: Zero — no direct test for `get_cached_mps_gemm`.

**What to test**:
- Same key returns same pointer (cache hit)
- Different key returns different pointer (cache miss)
- Batched vs non-batched keys differentiated
- Object is usable after cache hit (encode a GEMM, commit, verify output)

### TEST-5: `sampling.cc` Metal batched memcpy path

**Current coverage**: Covered indirectly by e2e Whisper tests, but no isolated unit test.

**What to test**:
- GPU TopK → `synchronize_stream()` → `memcpy` → verify sampled_ids and sampled_scores
- Verify `resize_as()` produces correct shape on CPU output
- Verify TYPE_DISPATCH for float16 scores

---

---

## Recommended Fix Priority

1. **BUG-1** (P0): Add `protect_buffer(previous_ids.buffer())` — 1 line fix, prevents data
   corruption when `repetition_penalty > 1.0` on Metal.
2. **TEST-1/2**: `buffer_for_ptr` + `protect_buffer` tests — validates the O(log N) refactor.
3. **TEST-3**: `indexed_fill` f16 test — validates the pre-sync skip.
4. **BUG-2** (P1): Investigate f32 `indexed_fill` root cause or remove the optimization.
5. **TEST-4/5**: GEMM cache + sampling tests — confirms cache correctness and sampling path.
