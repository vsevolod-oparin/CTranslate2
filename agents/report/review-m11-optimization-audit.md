# M11 Optimization Audit — Cross-Milestone Review

## Scope

Review of all 18 milestone reports (M11.1–M11.18) for potential bugs, missed optimizations, inconsistencies, and performance anti-patterns in the CTranslate2 Metal backend.

## Potential Bugs

### Critical

#### 1. `batch_cpu_gemm_f16` Stride Bug

**File:** `src/metal/primitives_gemm.mm` (widen loop in `batch_cpu_gemm_f16`)

The float16→float32 widen loop uses a linear index `ai[j]` where `j` ranges over `elems_a = rows_a * cols_a`, treating the source as contiguous. But `cblas_sgemm` is called with the original `lda`/`ldb` (which may differ from `cols_a`/`cols_b` for non-contiguous inputs).

```cpp
// Current (buggy for non-contiguous input):
for (NSUInteger j = 0; j < elems_a; ++j)
    fa[j] = static_cast<float>(ai[j]);

// Correct (strided copy):
for (NSUInteger r = 0; r < rows_a; ++r)
    for (NSUInteger c = 0; c < cols_a; ++c)
        fa[r * cols_a + c] = static_cast<float>(ai[r * lda + c]);
```

Works today because all attention GEMMs have `lda == cols_a` (contiguous projections), but any non-contiguous input would silently produce wrong results.

#### 2. Beam Size Detection in Flash Cross-Attention

**File:** `src/layers/attention.cc:486`

```cpp
if (queries_proj.dim(1) == 1 && cached_keys)
    beam_size = queries_proj.dim(0) / cached_keys->dim(0);
```

This integer division assumes `batch_size == 1`. For `batch_size > 1` with beam search, it happens to produce correct results by coincidence (`10 / 2 = 5` for batch=2, beam=5), but the derivation is fragile. Should propagate an explicit `beam_size` parameter from the caller instead of inferring from dim ratios.

#### 3. TopK GPU Kernel Silent Truncation at k > 64

**File:** `src/metal/kernels/topk.metal`

`TOPK_MAX_K = 64` is hardcoded. The `excluded[]` array in threadgroup memory is sized to 64. If the GPU k>1 path is ever activated for k > 64, results are silently truncated with no error.

**Fix:** Add runtime guard in `src/metal/ops_topk.mm`:
```cpp
if (k > TOPK_MAX_K)
    throw std::runtime_error("Metal TopK GPU kernel: k > 64 not supported");
```

### Medium

#### 4. `protect_buffer` O(n) Scan Called Thousands of Times

**File:** `src/metal/allocator.mm`

`protect_buffer()` scans the entire `_live` map (O(n)) to find the enclosing allocation. It is called 3 times per m=1 GEMM (for A, B, C). For whisper-large-v3-turbo decode with beam_size=5: ~20 heads × 4 decoder layers × ~200 decode steps = ~16,000 GEMM calls × 3 protect_buffer calls = ~48,000 O(n) scans.

**Fix:** Cache the MTLBuffer handle at the `gemm_batch_strided` call site (via `metal_buffer_for_ptr` already done inside `dispatch_mps_gemm_batched_padded`) and pass it to a new `protect_buffer_by_ptr(void* base_ptr)` that does O(1) lookup in `_live`.

#### 5. Fused Norm-GEMM Threadgroup Memory Mismatch

**File:** `src/metal/kernels/fused_norm_gemm.metal`

Host allocates `(K + 256) * sizeof(float)` for threadgroup memory via `setThreadgroupMemoryLength:`. But the `red[FUSED_BLOCK]` array (where `FUSED_BLOCK=256`) is declared as a separate `threadgroup float red[256]` inside the kernel body — an implicit threadgroup allocation NOT part of the host-declared size. The `threadgroup(0)` binding only covers `norm_row`.

Metal silently allocates additional per-kernel threadgroup memory for in-kernel arrays on current Apple Silicon, but this is not guaranteed by the Metal specification. A stricter future GPU family or OS could reject the kernel or produce incorrect results.

**Fix:** Move `red` into the host-allocated buffer at offset `K * sizeof(float)`, or declare it as part of `threadgroup(0)` with appropriate indexing.

#### 6. `_env_checked` Data Race in `commit_and_wait_impl`

**File:** `src/metal/utils.mm:86`

```cpp
static bool _env_checked = false;
if (!_env_checked) {
    _env_checked = true;
    if (std::getenv("CT2_METAL_TRACE")) { ... }
}
```

Not thread-safe if two threads call `commit_and_wait_impl` concurrently before the first completes the `if` block. Low risk (CTranslate2 uses one translator per thread), but technically UB.

**Fix:** `std::call_once(_env_flag, [&] { ... });`

#### 7. `gpu_time_elapsed` Is Thread-Local, Invisible from Other Threads

**File:** `src/metal/utils.mm:30,147`

`commit_count()` is a global atomic (visible from any thread), but `gpu_time_elapsed()` is thread-local (returns 0 when called from Python thread or monitoring). Design inconsistency — should either be global atomic or explicitly documented.

## Missed Easy Optimizations

### High Impact

#### Float16 m=1 Custom GEMV + `protect_buffer`

**Estimated savings:** ~8,700 syncs in the default (float16) inference path

The M11.18 report abandoned the custom GEMV kernel because of buffer-lifetime crashes — but `protect_buffer` now solves exactly that problem. The deferred-free mechanism was built to protect encode-only GPU kernels from buffer reuse. A custom float16 GEMV kernel with `protect_buffer` on A/B/C should work, since this is exactly what M11.18 does for float32 MPS padded path.

Note: MPS batched GEMM itself produces garbled output for float16 m=1 (verified in M11.18 with post-sync). But a custom MSL kernel performs its own computation — it is not affected by the MPS bug.

**Implementation:**
1. Write MSL GEMV kernel for float16 (simple dot product per output element)
2. Dispatch as encode-only from `gemm_batch_strided` for float16 m=1
3. Call `protect_buffer(a)`, `protect_buffer(b)`, `protect_buffer(c)` after dispatch

### Medium Impact

#### `primitives<METAL>::max_element` Still Syncs

**File:** `src/metal/primitives_reduction.mm:253-291`

Uses two-phase GPU reduction + `CT2_COMMIT_AND_WAIT()`. Could reuse M11.6's encode-only argmax kernel (`dispatch_argmax` in the same file's anonymous namespace), eliminating syncs from beam search scoring.

#### BF16 Batched GEMM: Sequential Synchronous MPSGraph

**File:** `src/metal/primitives_gemm.mm:1109-1118`

`batch_size=8` (multi-head attention) runs 8 sequential synchronous MPSGraph calls. A single MPSGraph with batch dimensions could eliminate redundant graph compilation overhead.

#### `logsumexp` CPU with Forced Sync

**File:** `src/metal/primitives_reduction.mm:356-367`

Called from beam search log-probability scoring. Could be fused with surrounding operations (same pattern as M11.17's timestamp fusion).

### Low Impact

#### `prepare_length_mask` Fence (14 syncs)

**File:** `src/metal/primitives_beam_search.mm`

The `CT2_COMMIT_AND_WAIT()` fence exists because `lengths` may have pending GPU writes. Audit callers to determine if this is always true — if `lengths` is always CPU-resident at the call site, the fence can be dropped.

#### Dead Code: `should_sample_timestamps_metal`

**File:** `src/metal/primitives_reduction.mm:452-460`

Superseded by M11.17's fused kernel `fuse_timestamp_check_and_disable_metal`. Still compiled but never called from `whisper.cc`. Should be removed to avoid confusion.

## Documentation / Consistency Issues

| Issue | Location |
|-------|----------|
| File comment describes old CPU implementation; M11.16 replaced with GPU kernel | `src/metal/primitives_beam_search.mm:10-17` |
| `kCpuGemmThresh = 4096` lacks comment that it is a **correctness** boundary for float16 (not a perf tuning knob) | `src/metal/primitives_gemm.mm` |
| M11.12 roadmap Priority 3 (A/B GPU packing) is stale — resolved in M11.13 | `agents/report/milestone-11.12-pad-c-sync-elimination.md` |
| M11.2 report implies MPS objects "might be cached"; they are not (resolved by M11.7 batching) | `agents/report/milestone-11.2-pso-caching.md` |
| CB batching test 3/4 PASS since M11.5 — threshold not updated for CPU GEMM syncs | `tests/metal/e2e/test_cb_batching.py` |
| Unexplained ~3,150 sync count change between M11.14 and M11.15 traces — likely measurement variance but not documented | Reports |
| `atexit` + `_trace_mutex` destruction order is technically UB (trivially destructible in practice) | `src/metal/utils.mm:91` |

## Performance Anti-Patterns Still Present

| Sync Source | Count (M11.18, float32) | Notes |
|-------------|------------------------|-------|
| `primitives_gemm.mm` (batch_cpu_gemm_f16) | 16 (f32) / ~8,700 (f16) | Float16 m=1 MPS bug; custom GEMV is viable |
| `devices.cc:162` (synchronize_stream) | 1,078 | Gathers, type conversions, framework sync |
| `primitives_memory.mm:80` (indexed_fill) | 1,024 | DisableTokens first apply() — structural |
| `multinomial_metal.mm:21` (CPU sampling) | 756 | std::discrete_distribution on CPU |
| `topk_metal.mm:44` (CPU sort) | 268 | std::partial_sort for k>1 beam search |

## Priority Action Table

| Priority | Item | Type | Effort | Impact |
|----------|------|------|--------|--------|
| **P0** | Float16 m=1 GEMV + protect_buffer | Missed optimization | Medium | ~8,700 syncs (default path) |
| **P0** | `batch_cpu_gemm_f16` stride bug | Bug fix | Low | Correctness for non-contiguous inputs |
| **P1** | Beam size detection in flash cross-attn | Bug fix | Low | Correctness for batch>1 |
| **P1** | TopK k>64 runtime guard | Bug fix | Low | Prevent silent truncation |
| **P1** | `protect_buffer` O(1) optimization | Performance | Low | Reduce ~48K O(n) scans |
| **P2** | Fused norm-GEMM threadgroup memory | Latent bug | Low | Future-proof Metal compliance |
| **P2** | `_env_checked` thread safety | Bug fix | Low | Eliminate data race |
| **P2** | Remove dead `should_sample_timestamps_metal` | Cleanup | Low | Reduce confusion |
| **P2** | Update stale comments/docs (5 items above) | Documentation | Low | Maintainability |
| **P3** | `max_element` encode-only via argmax | Optimization | Low | Minor sync reduction |
| **P3** | BF16 batched MPSGraph | Optimization | Medium | Latency improvement |
| **P3** | `logsumexp` fusion | Optimization | Medium | Minor sync reduction |
