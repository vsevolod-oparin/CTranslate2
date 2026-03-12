# Metal Test Suite Investigation & Fix Report

**Date**: 2026-03-12
**Branch**: `metal-backend`
**Scope**: All tests in `tests/metal/` (22 in `run_all.sh` + 18 standalone `.mm` tests)

## Executive Summary

All 22 tests in `run_all.sh` failed to build from a clean state (`--clean`). Investigation revealed 6 distinct root causes — all in test infrastructure or test reference code, not in production Metal backend code. All fixes applied; **601/601 assertions now pass across 22 tests**, plus 3 previously-failing standalone GQA tests (22 additional assertions).

## Before / After

| Scope | Before | After |
|-------|--------|-------|
| `run_all.sh` (22 tests, clean build) | 22/22 BUILD-FAIL | **22/22 PASS (601 assertions)** |
| Standalone GQA tests (3 files) | 3 runtime failures | **All pass (22 assertions)** |

## Root Causes & Fixes

### RC1: Stale source file reference — `primitives_norm_gather.mm` (19/22 tests)

**File**: `tests/metal/run_all.sh:65`
**Symptom**: `clang++: error: no such file or directory: .../primitives_norm_gather.mm`
**Cause**: During M5.2, normalization/gather code was moved from `src/metal/primitives_norm_gather.mm` to `src/metal/ops_norm_gather.mm` (following the new `ops_*.mm` naming convention). The `run_all.sh` `PRIM_SRCS` variable was never updated.
**Fix**: Replaced `primitives_norm_gather.mm` with `ops_norm_gather.mm` and added `ops_sdpa.mm` (needed by `primitives_gemm.mm` via `clear_sdpa_gemm_cache()`). Introduced `OPS_SRCS` variable for clarity.

### RC2: Dependency chain outgrew link groups (3/22 tests)

**Files**: `tests/metal/run_all.sh` — `CTX_SRCS`, `ALLOC_SRCS` definitions
**Symptom**: Undefined symbols: `metal_buffer_for_ptr`, `flush_pending_frees`, `clear_gemm_cache`
**Cause**: During M11-M12 milestones, cross-module dependencies grew:
```
utils.mm → allocator.mm (blit_copy, flush_pending_frees)
         → allocator.mm → primitives_gemm.mm (clear_gemm_cache)
                        → ops_sdpa.mm (clear_sdpa_gemm_cache)
```
The "minimal" link groups (`CTX_SRCS` = device+utils, `ALLOC_SRCS` = device+utils+allocator) no longer satisfy the linker.
**Fix**: All 3 tests (`context_test`, `sync_scoped_test`, `allocator_test`) now use `FULL_SRCS` + `FW_GRAPH`. Also added `-framework Accelerate` to `FW_GRAPH` (required by `primitives_gemm.mm` for `cblas_sgemm`).

### RC3: Missing `commit_and_wait()` after GPU `prepare_length_mask` (4 assertion failures)

**File**: `tests/metal/beam_search_test.mm`
**Symptom**: All 4 `prepare_length_mask` tests fail — mask values are zero/stale.
**Cause**: M11.16 moved `prepare_length_mask` from CPU to a GPU kernel. The test code still assumed CPU execution and read results immediately without flushing. Stale comment: *"No GPU work needed — CPU fills mask directly."*
**Fix**: Added `metal::commit_and_wait()` after each `prepare_length_mask()` call.

### RC4: Wrong GQA head mapping in CPU reference SDPA (3 standalone test failures)

**Files**: `kv_cache_test.mm:95`, `decode_rope_test.mm:209`, `gpu_decode_rope_test.mm:233`
**Symptom**: GQA tests (nh != nhk) fail with error ~0.8-1.6 (tolerance 1e-4). Non-GQA tests pass perfectly.
**Cause**: CPU reference used `hk = h % nhk` to map Q heads to KV heads. The correct GQA mapping is `hk = h / (nh / nhk)`. For `nh=4, nhk=2`:

| Q head (h) | `h % nhk` (wrong) | `h / (nh/nhk)` (correct) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | **1** | **0** |
| 2 | **0** | **1** |
| 3 | 1 | 1 |

The GPU implementation (`ops_sdpa.mm:804`, `msl_strings.h:1102`) correctly uses `h / heads_per_kv`. The bug was in the test reference, not the production code.
**Fix**: Changed `h % nhk` to `h / (nh / nhk)` in all 3 test files.

### RC5: ObjC pointer identity after release (1 assertion failure)

**Files**: `sync_scoped_test.mm:111`, `context_test.mm:75`
**Symptom**: `CHECK("fresh buffer differs from committed one", cb2 != cb)` intermittently fails.
**Cause**: After `commit_and_wait()` calls `[_thread_buffer release]`, the Metal runtime may recycle the same ObjC object address for the next `[queue commandBuffer]`. Pointer identity comparison is unreliable after release.
**Fix**: Replaced pointer comparison with status check: `[cb2 status] == MTLCommandBufferStatusNotEnqueued`.

### RC6: ARM `float16_t` name collision (1 build failure)

**File**: `tests/metal/primitives_test.mm`
**Symptom**: `error: reference to 'float16_t' is ambiguous`
**Cause**: `using namespace ctranslate2` brings `ctranslate2::float16_t` into scope, conflicting with ARM's built-in `float16_t` from `arm_vector_types.h` (included transitively via `Metal.h`). This is the known M5.1 pattern.
**Fix**: Qualified all occurrences as `ctranslate2::float16_t` and `ctranslate2::bfloat16_t`.

## Files Modified

| File | Change |
|------|--------|
| `tests/metal/run_all.sh` | Fixed source groups, framework flags, test link commands |
| `tests/metal/context_test.mm` | Replaced pointer identity check with status check |
| `tests/metal/sync_scoped_test.mm` | Replaced pointer identity check with status check |
| `tests/metal/primitives_test.mm` | Qualified `float16_t`/`bfloat16_t` with `ctranslate2::` |
| `tests/metal/beam_search_test.mm` | Added `commit_and_wait()` after `prepare_length_mask` |
| `tests/metal/kv_cache_test.mm` | Fixed GQA head mapping: `h % nhk` → `h / (nh / nhk)` |
| `tests/metal/decode_rope_test.mm` | Fixed GQA head mapping: `h % nhk` → `h / (nh / nhk)` |
| `tests/metal/gpu_decode_rope_test.mm` | Fixed GQA head mapping: `h % nhk` → `h / (nh / nhk)` |

## Key Finding: Production Code Is Correct

All 6 root causes were in **test infrastructure or test reference implementations**, not in the Metal backend production code. The GPU kernels, SDPA implementation, GQA head mapping, and all Metal primitives are functioning correctly.

## Standalone Tests Not in `run_all.sh`

18 additional `.mm` test files exist outside `run_all.sh`. After fixing the same `ops_sdpa.mm` link dependency (RC2), all 18 build and 15/18 pass. The 3 that failed were the GQA reference bug (RC4), now fixed. Full standalone results:

| Test | Pass | Fail | Notes |
|------|------|------|-------|
| m11_review_test | 43 | 0 | |
| m12_review_test | 7 | 0 | |
| decode_rope_test | 7 | 0 | GQA fix applied |
| gpu_decode_rope_test | 9 | 0 | GQA fix applied |
| topk_test | 30 | 0 | |
| kv_cache_test | 6 | 0 | GQA fix applied |
| m7_test | 28 | 0 | |
| m81_test | 13 | 0 | |
| m82_test | 21 | 0 | |
| m83_test | 18 | 0 | |
| m91_test | 12 | 0 | |
| m92_test | 10 | 0 | |
| rotary_test | 11 | 0 | |
| bugfix_test | 14 | 0 | |
| sdpa_test | 22 | 0 | |
| fused_norm_gemm_test | 21 | 0 | |
| alibi_test | 11 | 0 | |
| flash_mha_decode_test | 16 | 0 | |

## Recommendations

1. **Add standalone tests to `run_all.sh`**: The 18 standalone test files (especially `kv_cache_test`, `sdpa_test`, `flash_mha_decode_test`) should be added to the registry for automated coverage.
2. **Update stale build commands in test headers**: Most `.mm` files have build instructions in their header comments that are missing `ops_sdpa.mm` and `-framework Accelerate`.
3. **Consider a shared test build config**: The growing dependency chain means every test now needs `FULL_SRCS`. A shared Makefile or CMake target would prevent future drift.
