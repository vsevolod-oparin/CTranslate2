# Audit: M11.19–M11.28 — Comprehensive Review

## Scope

All Metal backend changes from M11.19 (GEMV kernel) through M11.28 (indexed_fill pre-sync elimination). Covers bugs, dead code, thread safety, missing tests, and performance opportunities.

---

## CRITICAL BUGS

### BUG-1: `penalize_previous_tokens` Use-After-Free (Same Pattern as M10.1)

**Files:** `src/decoding_utils.cc:55-66`, `src/metal/primitives_beam_search.mm:68-80`

In `RepetitionPenalty::apply()`:

```cpp
StorageView previous_ids = sequences.to(device);          // line 55
StorageView previous_scores(device, dtype);                // line 56
ops::Gather(/*axis=*/-1, /*batch_dims=*/1)(logits, previous_ids, previous_scores);  // line 57
// ^ Gather encodes GPU kernel, protects clones + previous_ids (indices).
//   previous_scores (output) is NOT protected.

primitives<D>::penalize_previous_tokens(                   // line 59-66
    logits.data<T>(), previous_scores.data<T>(),           // reads previous_scores
    previous_ids.data<int32_t>(), ...);                    // reads previous_ids
// ^ penalize_previous_tokens is encode-only — no commit_and_wait()
```

When `apply()` returns:
- `previous_ids` → protected by Gather (M11.21) → goes to `_pending_free` → **SAFE**
- `previous_scores` → **NOT protected** → returns to pool immediately → **UNSAFE**

The GPU `penalize_previous_tokens` kernel reads `previous_scores` but its buffer can be reused by the very next allocation before the GPU executes. This is the **exact same class of bug** as M10.1 (Gather use-after-clone).

**Why it may not have manifested yet:** The next operation after RepetitionPenalty in the decode loop may not allocate a buffer of the same size, or a `commit_and_wait()` may intervene. But this is luck, not correctness.

**Fix:** Add `metal::protect_buffer(previous_scores.buffer())` after the Gather call, or add `metal::protect_buffer(previous_scores.data<T>())` inside `penalize_previous_tokens`.

**Severity:** CRITICAL — Silent data corruption in beam search decode with `repetition_penalty != 1`.

### BUG-2: `indexed_fill` Dispatches 0 Threads When `num_indices == 0`

**File:** `src/metal/primitives_memory.mm`, lines 87-125

When `num_indices == 0`, the function calls `MTLSizeMake(0, 1, 1)` for grid size and `std::min<NSUInteger>(0, pso.maxTotalThreadsPerThreadgroup)` = 0 for threadgroup size. Metal requires `threadsPerThreadgroup` to have all dimensions >= 1. This is undefined behavior on some GPU families and can cause a command buffer error.

**Fix:** Add `if (num_indices <= 0) return;` at the top of the function.

### BUG-3: `dispatch_mps_gemm_batched_padded` Leaks Temp Buffers on Fallback Path

**File:** `src/metal/primitives_gemm.mm`, lines ~1073-1082

When padding is needed and `tmp_a`/`tmp_b`/`tmp_c` are allocated (lines ~1012-1057), but the MPS batch constraint check fails (line ~1073), the function `return`s at line ~1082 without reaching the `[tmp_a release]` / `[tmp_b release]` / `[tmp_c release]` at lines ~1139-1141. This is a memory leak (ARC is off — these +1 retained MTLBuffers are never released).

**Fix:** Add release calls before the early return:
```cpp
if (final_mb_a % final_rb_a != 0 || ...) {
    if (tmp_a) [tmp_a release];
    if (tmp_b) [tmp_b release];
    if (tmp_c) [tmp_c release];
    for (...) dispatch_mps_gemm<T>(...);
    return;
}
```

### BUG-4: `clear_cache()` Releases Pending-Free Buffers Without commit_and_wait

**File:** `src/metal/allocator.mm`, lines ~165-174

`clear_cache()` releases buffers in `_pending_free` without first committing pending GPU work. These buffers are in `_pending_free` because they are referenced by uncommitted GPU commands. Releasing them while GPU work is in-flight causes GPU-side use-after-free.

**Fix:** Call `commit_and_wait()` at the start of `clear_cache()`, or document that it must only be called with no GPU work in flight.

---

## HIGH-SEVERITY ISSUES

### CODE-1: Dead Code — `batch_cpu_gemm_f32` and `batch_cpu_gemm_f16` Lambdas

**File:** `src/metal/primitives_gemm.mm`, lines ~1219-1279

After M11.26 removed the `m*n > 4096` threshold, these two lambdas (60 lines total) are never called. All padded GEMMs now route through `dispatch_mps_gemm_batched_padded`. The lambdas allocate 48KB of stack arrays each (via `float sa[4096]` etc.) that are never used.

**Fix:** Delete both lambdas.

### CODE-2: GEMV Kernel Missing SIMD Optimizations

**File:** `src/metal/primitives_gemm.mm`, lines 860-898 (kGemvF16MSL)

The f16 GEMV kernel accumulates the dot product serially per thread:
```metal
for (uint i = 0; i < K; ++i)
    acc += float(a[i]) * float(B[col * K + i]);
```

For typical decode shapes (K=64-512), this leaves significant performance on the table:
- **Simdgroup reductions**: `simd_sum()` could parallelize K-accumulation across 32 threads
- **Vectorized loads**: `half4` loads would improve memory throughput 4x
- **Threadgroup shared memory**: `a_row` is broadcast to all threads — loading once to shared memory eliminates redundant global reads

This kernel is on the hot path for every decode step in f16 inference.

### THREAD-1: MPS GEMM Cache Returns Shared Objects for Concurrent Encode

**File:** `src/metal/primitives_gemm.mm`, lines 92-129 (cache), lines 338/553/735/1121 (encode sites)

The `get_cached_mps_gemm()` function returns the **same** `MPSMatrixMultiplication*` to all callers. Apple documents that `MPSKernel` subclasses are NOT thread-safe for concurrent `encodeToCommandBuffer:`. If `inter_threads > 1` is ever used with Metal, two threads encoding the same shape concurrently would race on the kernel's internal state.

**Current risk:** Low — CTranslate2 runs single-threaded per translator. But the cache is `static` (process-wide) with no thread-local isolation.

**Fix options:**
1. Make cache `thread_local` instead of `static` (simplest)
2. Use `[gemm_op copy]` (NSCopying) to return per-thread clones
3. Assert/document single-threaded Metal requirement

---

## MEDIUM-SEVERITY ISSUES

### SYNC-1: `flush_pending_frees()` Is Global but Command Buffers Are Thread-Local

**Files:** `src/metal/utils.mm:126`, `src/metal/allocator.mm:157-163`

`commit_and_wait()` commits the **thread-local** command buffer, then calls `flush_pending_frees()` which releases **all** pending buffers globally. If Thread A protects a buffer and encodes work, then Thread B calls `commit_and_wait()`, Thread B's flush releases Thread A's protected buffers — even though Thread A's GPU work hasn't been committed yet.

**Risk:** Only manifests with multi-threaded GPU encoding, which is not currently used.

### SAFETY-1: indexed_fill Kernel Has No Bounds Checking

**File:** `src/metal/kernels/indexed_fill.metal`, line 17

```metal
x[indices[gid]] = fill_val;  // No check that indices[gid] < x_size
```

A malformed index causes an out-of-bounds GPU scatter write, which can silently corrupt adjacent Metal buffer memory. In current code paths, indices are validated in `DisableTokens::add()`, so this is defensive rather than urgent.

**Fix:** Pass `x_size` as a `constant uint&` parameter; guard with `if (indices[gid] < x_size)`.

### BF16-1: Batched BF16 GEMM Alpha Scaling Ignores `stridec`

**File:** `src/metal/primitives_gemm.mm`, lines ~1356-1360

```cpp
if (alpha != 1.0f) {
    const dim_t total = batch_size * m * n;
    primitives<Device::METAL>::mul(static_cast<bfloat16_t>(alpha), c, c, total);
}
```

This scales a contiguous `batch_size * m * n` block, but if `stridec > m * n` (padding between batch elements), it scales incorrect memory regions. Currently BF16 enforces `ldc == n`, but `stridec` is not validated.

**Fix:** Either assert `stridec == m * ldc` or apply alpha per batch element.

### PERF-1: Float32 Indexed_Fill Pre-Sync Still Unresolved

**File:** `src/metal/primitives_memory.mm`, lines 101-103

The M11.28 optimization only applies to float16. Float32 retains the per-step pre-sync, meaning f32 decode still has 2 syncs/step instead of 1. The root cause (MPS f32 GEMM ordering) remains unresolved.

**Options not yet tried:**
1. `MTLFence` between MPS encoder and compute encoder
2. `[buffer didModifyRange:]` on the indices buffer (lighter than full commit_and_wait)
3. A second command buffer for indexed_fill (avoids sharing CB with MPS)

---

## DEAD CODE & CLEANUP

| Item | File | Lines | Description |
|------|------|-------|-------------|
| `batch_cpu_gemm_f32` lambda | `primitives_gemm.mm` | ~1219-1231 | Unreachable after M11.26 threshold removal |
| `batch_cpu_gemm_f16` lambda | `primitives_gemm.mm` | ~1235-1279 | Unreachable after M11.26 threshold removal |
| `needs_padding` f16 hardcode | `primitives_gemm.mm` | ~1207 | Assumes elem_sz==2 is f16, would misidentify bf16 (not currently reachable) |

---

## TEST COVERAGE GAPS

### Features With NO Standalone Tests

| Feature | Milestone | Current Coverage |
|---------|-----------|-----------------|
| GEMV f16 kernel (`dispatch_gemv_f16_batched`) | M11.19 | E2E only (via inference) |
| `protect_buffer` / deferred free | M11.21 | None — allocator_test doesn't test gpu_protected |
| MPS GEMM cache (`get_cached_mps_gemm`) | M11.27 | None — test_pso_caching tests PSO, not GEMM objects |
| Sampler single-sync batching | M11.26 | Aggregate commit count only (test_cb_batching) |
| `penalize_previous_tokens` GPU kernel | M11.23 | E2E only |

### Features With THIN Coverage

| Feature | Gap |
|---------|-----|
| `indexed_fill` (M11.25) | Only f32 tested (primitives_test.mm). Missing: f16, bf16, int32, 0 indices, large arrays |
| TopK fused kernel (M11.23) | Missing: k > kTopKMaxK throw test, depth=1, k=depth, batch=0 |

### Recommended New Tests (Priority Order)

1. **`protect_buffer_test.mm`** — Alloc, protect, free, re-alloc same size → verify no reuse before commit
2. **Extend `primitives_test.mm`** — indexed_fill for f16, int32, 0-index edge case
3. **`gemv_f16_test.mm`** — Direct GEMV via `gemm_batch_strided<f16>` with m=1, various (n,k,batch)
4. **Extend `gemm_test.mm`** — Repeated same-shape GEMM to exercise cache hit path
5. **`penalize_previous_tokens` test** — With repetition_penalty != 1 and beam_size > 1
6. **TopK edge cases** — k > 64 throws, depth=1, batch=0

---

## PERFORMANCE OPPORTUNITIES

### OPP-1: GEMV Simdgroup Optimization (HIGH impact)

The m=1 f16 GEMV kernel runs on every decode step. Adding simd_sum() for K-dimension reduction and half4 vector loads could yield 2-4x on the kernel itself. Since GEMV is ~30% of decode compute, this could translate to 10-20% end-to-end speedup for f16.

### OPP-2: Float32 Pre-Sync Elimination (MEDIUM impact)

Resolving the MPS f32 ordering issue would halve syncs/step for f32 decode (same as M11.28 achieved for f16). Estimated 10-15% f32 speedup from pipeline bubble elimination.

### OPP-3: GPU-Side EOS Check (MEDIUM impact, HIGH effort)

The last remaining sync per decode step is the CPU reading sampled tokens to check for end-of-sequence. Moving EOS detection to a GPU kernel would allow multi-step GPU submission without CPU roundtrip.

### OPP-4: MPSMatrix Pooling (LOW impact)

3 `MPSMatrix` objects are still allocated/released per GEMM dispatch. Pooling by (buffer, offset, descriptor) could save ~10-20μs/dispatch for 76K+ dispatches per inference.

---

## REPORT-LEVEL ISSUES

### Unverified Performance Claims

| Claim | Report | Issue |
|-------|--------|-------|
| GEMV 0.70-1.16x speedup | M11.19 | Attributed to "thermal throttling"; no reproducible measurement |
| Caching saves ~10-20μs/object | M11.27 | Theoretical estimate, not directly measured |
| TopK large model 1.09x | M11.23 | Actually a slight slowdown — report frames it as improvement |

### Contradictions

| Item | Reports | Resolution |
|------|---------|------------|
| faster_whisper variance cause | M11.27 vs M11.28 | M11.28 has the correct root cause (seek retries from MPS non-determinism) |
| Gather sync savings | M11.21 vs plan | Plan estimated ~2500 syncs; actual was 264 — plan was misdirected |

---

## SUMMARY

| Severity | Count | Action Required |
|----------|-------|-----------------|
| CRITICAL | 4 | BUG-1: penalize use-after-free; BUG-2: 0-index dispatch; BUG-3: temp buffer leak; BUG-4: clear_cache unsafe |
| HIGH | 4 | Dead code removal, GEMV optimization, cache thread safety, BF16 stridec alpha |
| MEDIUM | 4 | Global flush_pending_frees, bounds checking, f32 pre-sync, BF16 stridec |
| Test Gaps | 5+ | Standalone tests for GEMV, protect_buffer, indexed_fill types |
| Performance | 4 | GEMV simdgroup, f32 pre-sync, GPU EOS, MPSMatrix pool |

### Priority Fix Order

1. **BUG-1** (penalize_previous_tokens use-after-free) — Silent data corruption with `repetition_penalty != 1`. Same pattern as M10.1.
2. **BUG-3** (temp buffer leak on fallback) — Memory leak every time MPS batch constraints fail. One-line fix.
3. **BUG-2** (0-index dispatch) — Defensive one-liner. May not trigger in practice but is UB.
4. **BUG-4** (clear_cache without sync) — Only affects explicit cache clearing with pending GPU work.
5. **CODE-1** (dead code) — 60 lines of unreachable lambdas. Clean delete.

**Verdict:** BUG-1 must be fixed before merge. BUG-3 is a real leak. BUG-2 and BUG-4 are defensive fixes.
