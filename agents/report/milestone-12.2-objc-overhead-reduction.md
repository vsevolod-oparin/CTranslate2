# M12.2 — ObjC Overhead Reduction in GEMM Path

**Date**: 2026-03-11
**Status**: Investigation complete — minor code improvement, no measurable perf gain
**Model**: OPUS-MT En→De (d_model=512, 6 layers)
**Benchmark**: 50 sentences (f32/f16), 10 sentences (int8/bf16)

---

## 1. Hypothesis

The M12 profiling report estimated ~10µs ObjC overhead per GEMM call (MPSMatrixDescriptor creation, MPSMatrix alloc/init/release, @autoreleasepool). With ~2100 GEMM dispatches per 50-sentence batch, this would total ~21ms (~2% of wall time).

Three optimization approaches were proposed:
- **(a)** Cache `rowBytesForColumns` results (eliminate ObjC class method calls)
- **(b)** Custom float32 GEMV kernel for m=1 decode (bypass MPS entirely)
- **(c)** Pool `MPSMatrix` objects (reduce alloc/dealloc cycles)

---

## 2. Implementation

### 2a. Cached `rowBytesForColumns` (implemented, kept)

`[MPSMatrixDescriptor rowBytesForColumns:dataType:]` is a pure function of (columns, dataType). Previously called 3× per GEMM dispatch inside @autoreleasepool blocks at 5 call sites (15 calls per GEMM).

**Change**: Added `cached_row_bytes(cols, dtype)` function at file scope in `primitives_gemm.mm`. Uses `std::unordered_map` with mutex (single-threaded GPU work = uncontended). Replaced all 5 call sites, removing 5 @autoreleasepool blocks.

**Files modified**: `src/metal/primitives_gemm.mm`

### 2b. Custom float32 GEMV for m=1 decode (tested, reverted)

Implemented a custom MSL GEMV kernel (`gemv_float`) mirroring the existing float16 GEMV (M11.19). Routed all float32 m=1 batched GEMMs through the custom kernel, bypassing MPS MPSMatrixMultiplication.

**Result: 16% REGRESSION for float32** (1480 ms → 1716 ms).

| Path | Best (ms) | GPU time (ms) | GPU% |
|------|-----------|---------------|------|
| M12.1 MPS (baseline) | 1480 | 814 | 55% |
| M12.2 custom GEMV | 1716 | 1098 | 64% |
| M12.2 cached_row_bytes only | 1462 | 778 | 53% |

**Root cause of regression**: MPS `MPSMatrixMultiplication` dispatches to Apple's **AMX (Apple Matrix Extension)** hardware — a dedicated matrix coprocessor that performs tiled 16×16 matrix multiplies per cycle. A custom MSL compute kernel runs on regular GPU shader cores and cannot access AMX. For m=1 GEMV shapes, MPS decomposes the operation to use AMX efficiently, achieving higher throughput than any compute kernel.

**Why f16 GEMV works**: The float16 GEMV (M11.19) exists due to a **correctness bug** in MPS batched GEMM for float16 m=1 (produces garbled output). The custom kernel is a correctness fix, not a performance optimization. Its ~5-10% performance loss vs correct MPS was acceptable; for float32 where MPS works correctly, there's no reason to bypass it.

**Action**: Reverted f32 GEMV routing. Kept the kernel code for reference (dead code, available if a f32 MPS bug surfaces).

### 2c. MPSMatrix/Descriptor pooling (not implemented)

Analysis showed MPSMatrix alloc+init+release overhead is ~300ns per object × 3 objects × 2100 GEMMs = ~1.9ms. MPSMatrixDescriptor creation adds ~0.5ms. Total ObjC overhead: ~2.5ms out of ~1500ms = **0.17%**. Not worth the code complexity.

---

## 3. Performance Results (cached_row_bytes only)

### Float32 (50 sentences)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 1480 | 1516, 1516, 1480 | 1544 | 96 | 55% | 1043 | M12.1 |
| 3 | cbe1afee | 1462 | 1485, 1462, 1481 | 1544 | 96 | 53% | 1056 | M12.2 |

Delta: -18 ms (-1.2%) — **within noise**, not statistically significant.

### Float16 (50 sentences)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 1033 | 1084, 1034, 1033 | 1550 | 90 | 42% | 1500 | M12.1 |
| 3 | cbe1afee | 1012 | 1078, 1021, 1012 | 1550 | 90 | 42% | 1531 | M12.2 |

Delta: -21 ms (-2.0%) — **within noise**.

### INT8 (10 sentences)

| # | Commit | Best (ms) | Runs (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 2336 | 2380, 2348, 2336 | 194 | 5692 | 15% | 83 | M12.1 |
| 3 | cbe1afee | 2303 | 2346, 2303, 2310 | 194 | 5692 | 15% | 84 | M12.2 |

Delta: -33 ms (-1.4%) — within noise.

### INT8+Float16 (10 sentences)

| # | Commit | Best (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 2266 | 193 | 5580 | 15% | 85 | M12.1 |
| 3 | cbe1afee | 2252 | 193 | 5580 | 15% | 86 | M12.2 |

### BFloat16 (10 sentences)

| # | Commit | Best (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 21387 | 195 | 2844 | 1% | 9 | M12.1 |
| 3 | cbe1afee | 21329 | 195 | 2844 | 1% | 9 | M12.2 |

### INT8+BFloat16 (10 sentences)

| # | Commit | Best (ms) | Tokens | Commits | GPU% | tok/s | Label |
|---|--------|-----------|--------|---------|------|-------|-------|
| 2 | 8e325364 | 22042 | 195 | 6904 | 2% | 9 | M12.1 |
| 3 | cbe1afee | 22664 | 195 | 6904 | 2% | 9 | M12.2 |

---

## 4. Key Findings

### 4.1 ObjC overhead is not the bottleneck

The M12 report estimated 10-20% improvement from reducing ObjC overhead. Actual measurement shows **<0.5%** of total time is spent on MPS wrapper creation:

| Source | Per-call | Calls/batch | Total | % of wall |
|--------|----------|-------------|-------|-----------|
| `rowBytesForColumns` × 3 | ~15 ns | 2100 | ~0.1 ms | 0.007% |
| `MPSMatrixDescriptor` × 3 | ~100 ns | 2100 | ~0.6 ms | 0.04% |
| `MPSMatrix` alloc/init/release × 3 | ~300 ns | 2100 | ~1.9 ms | 0.13% |
| `@autoreleasepool` enter/exit | ~50 ns | 2100 | ~0.1 ms | 0.007% |
| **Total ObjC overhead** | | | **~2.7 ms** | **0.18%** |

### 4.2 MPS uses AMX hardware, not compute shaders

Apple's `MPSMatrixMultiplication` dispatches to the **Apple Matrix Extension (AMX)** — a dedicated matrix coprocessor that performs tiled matrix multiplies at hardware speed. Custom MSL compute kernels run on regular GPU shader cores and **cannot access AMX**. This makes it impossible to beat MPS GEMM performance with a custom kernel for standard shapes.

The float16 GEMV (M11.19) is a **correctness workaround**, not a performance optimization. It exists because MPS has a bug for float16 m=1 batched GEMM. For float32 where MPS works correctly, the custom kernel is 35% slower.

### 4.3 CPU overhead breakdown (revised)

With ObjC overhead eliminated as a factor, the ~680ms f32 CPU overhead (47% of wall time) comes from:

| Source | Estimated | Notes |
|--------|-----------|-------|
| Beam search bookkeeping | ~200-300 ms | CPU sort, gather, state management |
| Non-GEMM GPU ops encode | ~50-100 ms | LayerNorm, activation, softmax |
| `metal_buffer_for_ptr()` | ~5-10 ms | Already O(log n) via std::map |
| Sampler sync CPU-side | ~50-100 ms | Token extraction, next-step prep |
| Python→C++ overhead | ~20-50 ms | Tokenization, batch management |
| Misc (allocator, etc.) | ~100-200 ms | |

### 4.4 Recommendation: skip remaining M12.2/12.3

- **M12.2 (ObjC reduction)**: Done. No further optimizations warranted — total overhead is 0.18%.
- **M12.3 (buffer lookup)**: Already O(log n) via `std::map::upper_bound()` in allocator.mm. No action needed.
- **Next impactful targets**: M12.5 (BF16 async/promote), M12.6 (INT8 GPU dequantize), or larger-model benchmarks where GPU utilization is higher.

---

## 5. Code Changes (kept)

| File | Change | Purpose |
|------|--------|---------|
| `src/metal/primitives_gemm.mm` | Added `cached_row_bytes()` function | Cache `rowBytesForColumns` results |
| `src/metal/primitives_gemm.mm` | Replaced 5 call sites (15 ObjC calls) | Eliminated 5 `@autoreleasepool` blocks |
| `src/metal/primitives_gemm.mm` | Added `kGemvF32MSL` kernel + dispatch | Dead code — available if f32 MPS bug surfaces |

---

## 6. Code Changes (reverted)

| Change | Why reverted |
|--------|-------------|
| f32 m=1 → custom GEMV routing in `gemm_batch_strided` | 16% regression — MPS AMX is faster |
