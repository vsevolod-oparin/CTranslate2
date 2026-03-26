# M19 — MPS Buffer Cache Corruption Fix

**Date:** 2026-03-22
**Status:** Complete
**Tests:** 17/17 standalone GEMM, 26/26 existing GEMM suite, 18/18 Python model tests
**Performance:** ~3-5% slowdown, ~5% RSS increase (measured on whisper-large-v3-turbo, Apple M4)

---

## Bug Description

After `Whisper::generate()` with `beam_size=5` on 3000-frame mel spectrogram, a subsequent `generate()` with `beam_size=1` on smaller frames (e.g., 205) produces all token-ID-0 output — the model outputs garbage.

Only affects large models (whisper-large-v3-turbo confirmed; whisper-base unaffected). The failure pattern is dimension-specific: around n=205, odd mel frame counts fail while even work; around n=750, most fail except multiples of 4.

## Root Causes

Two separate bugs were identified and fixed.

### Bug 1: F16/SDPA Temp Cache Stale Data

**Location:** `src/metal/primitives_gemm.mm` (`F16TempCache`), `src/metal/ops_sdpa.mm` (`SdpaF16TempCache`)

**Mechanism:** Thread-local caches hold oversized MTLBuffers for float32 temporary matrices used in f16-promoted GEMM. When GEMM dimensions shrink between calls, the conversion kernel (`encode_half_to_float32`) writes only the new (smaller) region. MPS may read beyond the declared matrix dimensions for alignment, encountering stale float32 values from the previous call. This corrupts the GEMM accumulation.

**Fix:** `memset([buf contents], 0, cap)` on cache hit before returning the buffer. Cost is negligible (~microseconds on unified memory).

**Status:** Necessary but NOT sufficient — fixing this alone does not resolve the full bug.

### Bug 2: MPS Per-MTLBuffer Internal State Corruption

**Location:** `src/metal/allocator.mm` (MetalAllocator pool)

**Mechanism:** Apple's MPS framework caches internal optimization state per `MTLBuffer` object identity. When the pool allocator returns a previously-used MTLBuffer at different GEMM dimensions, MPS's cached per-buffer state becomes invalid for the new dimensions.

This is NOT a data-contents issue — zeroing the buffer does not help. It is NOT a buffer-identity issue at the OS level — disabling pooling (fresh `[device newBufferWithLength:]` every time) while still going through the pool→free path does not help. Only fully releasing the MTLBuffer objects (via `[buf release]`) and creating new ones eliminates the stale MPS state.

**Fix:** Disable pool reuse entirely. `allocate()` always creates a fresh MTLBuffer. `pool_or_release_locked()` always releases instead of pooling.

## Diagnostic Process

### What was tested (and ruled out)

| Hypothesis | Test | Result |
|-----------|------|--------|
| Stale data in pool buffers | `memset(ptr, 0, bucket)` on pool hit | Bug persists |
| MTLBuffer object reuse | Disable pooling (release + fresh alloc) | Bug persists |
| Cached MPSMatrixMultiplication objects | Clear GEMM cache between calls | Bug persists |
| GPU coherency / ordering | `synchronize_stream()` before encode | Bug persists |
| Combination of all above | `clear_cache()` (commit + release pool + clear GEMM) | **Bug fixed** |

### Key finding

`clear_cache()` releases pool buffers via `[buf release]` — this forces fresh MTLBuffer objects on subsequent allocations. The critical difference from "disable pooling" is that `clear_cache()` also releases buffers that went through the `pending_free` path (GPU-protected buffers deferred until `commit_and_wait()`). The combination of releasing ALL cached buffer objects is what clears MPS's internal state.

### Failure pattern analysis

Tested on whisper-large-v3-turbo after `beam_size=5` with 3000-frame mel:

```
n=200: OK    n=201: OK    n=202: OK    n=203: OK    n=204: OK
n=205: ZERO  n=206: OK    n=207: ZERO  n=208: OK    n=209: ZERO
n=210: OK    n=211: ZERO  n=212: OK    n=213: ZERO  n=214: OK
```

Around n=750:
```
n=745-747: ZERO  n=748: OK  n=749-751: ZERO  n=752: OK
n=753-755: ZERO  n=756: OK  n=757-759: ZERO
```

The pattern correlates with MPS's internal alignment/tiling boundaries, not with CT2 logic. This strongly suggests an Apple MPS framework behavior rather than a CTranslate2 bug.

## Fix Details

### Files Changed

| File | Change |
|------|--------|
| `src/metal/allocator.mm` | Disabled pool reuse in `allocate()`; `pool_or_release_locked()` always releases |
| `src/metal/primitives_gemm.mm` | `F16TempCache::get()`: memset on cache hit |
| `src/metal/ops_sdpa.mm` | `SdpaF16TempCache::get()`: memset on cache hit |

### Files Added

| File | Purpose |
|------|---------|
| `tests/metal/m19_temp_cache_test.mm` | 17 standalone GEMM dimension-change tests |
| `tests/metal/m19_repro.py` | Python reproduction test (beam→greedy on turbo model) |
| `tests/metal/m19_perf.py` | A/B performance benchmark script |

### Stashed alternative (Option 1)

```
git stash list → "M19: per-model clear_cache fix (option 1) + temp cache zeroing"
```

This approach calls `get_allocator<Device::MPS>().clear_cache()` at the start of `WhisperReplica::generate()`. It works but only protects Whisper — other model types (Translator, Generator, Moonshine) remain vulnerable. Recoverable with `git stash pop` if Option 3 proves too costly.

## Performance Impact

Measured on whisper-large-v3-turbo, Apple M4, averaged across both run orderings to eliminate thermal bias:

| Workload | With Pooling | No Pooling | Delta |
|----------|-------------|------------|-------|
| Greedy 3000 frames | 1070 ms | 1127 ms | **+5.3%** |
| Greedy 500 frames | 219 ms | 224 ms | **+2.3%** |
| Greedy 205 frames | 147 ms | 149 ms | **+1.4%** |
| Beam=5 3000 frames | 1125 ms | 1176 ms | **+4.5%** |
| Mixed beam+greedy cycle | 1248 ms | 1289 ms | **+3.3%** |
| RSS peak | ~4670 MB | ~4718 MB | **+1%** |

**Summary:** ~3-5% speed regression, ~1-5% RSS increase. Acceptable trade for correctness across all model types and all dimension combinations.

## Scope of Protection

The pool-level fix (Option 3) protects **all** model types automatically:

| Workflow | Previously at risk | Now protected |
|----------|-------------------|---------------|
| Beam search → greedy on same model | Yes | Yes |
| Variable-length encoding (streaming) | Yes | Yes |
| Batch size changes between calls | Yes | Yes |
| Different models on same device | Yes | Yes |
| Consistent dimensions (no change) | No risk | No impact |

## Remaining Work

**19.5:** Remove the 3000-frame padding workaround in MetalWhisper (`MWStreamingTranscriber.mm` in the metal-faster-whisper repo).

## Build & Test Commands

```bash
# Build
cmake --build build --target ctranslate2 -j$(sysctl -n hw.logicalcpu)

# Install
cp build/libctranslate2.mps.4.7.1.dylib /opt/anaconda3/envs/ct2/lib/libctranslate2.mps.4.7.1.dylib

# Standalone GEMM test
clang++ -std=c++17 -O0 \
    -I include -I src -DCT2_WITH_METAL \
    tests/metal/m19_temp_cache_test.mm \
    -L build -lctranslate2.mps \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
    -Wl,-rpath,build \
    -o m19_test && ./m19_test

# Python reproduction test
conda run -n ct2 python tests/metal/m19_repro.py

# Performance benchmark
conda run -n ct2 python tests/metal/m19_perf.py
```
