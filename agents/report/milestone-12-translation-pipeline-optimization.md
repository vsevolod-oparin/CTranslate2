# Milestone 12: Translation Pipeline Optimization — Profiling Report

**Date:** 2026-03-11
**Status:** Profiling complete, implementation pending
**Model:** OPUS-MT En→De (base Transformer, 6 encoder + 6 decoder layers, d_model=512)
**Test set:** WMT14 newstest2014, 2,737 sentences, beam_size=4
**Hardware:** Apple M4, macOS 15

---

## 1. Benchmark Results (Pre-Optimization)

### Full test set (2,737 sentences, best-of-3)

| Config | tok/s | Time (s) | RSS (MB) | BLEU |
|--------|-------|----------|----------|------|
| CPU float32 (4 threads) | 933.9 | 78.3 | 1,649 | 27.65 |
| MPS float32 | 268.7 | 272.3 | 1,024 | 27.65 |
| MPS float16 | 1,006.3 | 73.0 | 673 | 26.81 |
| MPS int8 | ~131 | ~559 | 828 | 27.57 |
| MPS bfloat16 | timeout (>20 min) | — | — | — |

**Key observation:** MPS float16 barely beats 4-thread CPU (1.08×). MPS float32 is 3.5× *slower* than CPU. For a GPU backend, we'd expect 2-5× speedup minimum.

### Profiled run (100 sentences, detailed instrumentation)

| Metric | MPS float32 | MPS float16 |
|--------|-------------|-------------|
| Wall time | 3,752 ms | 1,991 ms |
| GPU time | 1,487 ms (39.6%) | 834 ms (41.9%) |
| CPU overhead | 2,266 ms (60.4%) | 1,156 ms (58.1%) |
| Total commits | 288 | 282 |
| Commits/token | 0.11 | 0.10 |
| Tok/s | 720 | 1,361 |

**GPU utilization is only ~40%.** The GPU spends 60% of its time idle, waiting for CPU overhead.

---

## 2. Sync Trace Analysis

Commit trace (`CT2_MPS_TRACE=1`) reveals two dominant sync sources:

| Source | Location | Commits | % of Total |
|--------|----------|---------|------------|
| `prepare_length_mask` | `primitives_beam_search.mm:91` | ~140 | 49% |
| `synchronize_stream` (sampler) | `devices.cc:162` via `sampling.cc:29` | ~144 | 50% |
| `indexed_fill` pre-sync | `primitives_memory.mm:114` | 4 | 1% |

### Per-batch breakdown (float16, 100 sentences)

| Batch | Size | Tokens | Wall ms | GPU ms | Commits | C/step | GPU% |
|-------|------|--------|---------|--------|---------|--------|------|
| 0 | 32 | 948 | 792 | 316 | 133 | 4.5 | 40% |
| 1 | 32 | 940 | 755 | 304 | 129 | 4.4 | 40% |
| 2 | 32 | 652 | 594 | 241 | 115 | 5.6 | 41% |
| 3 | 4 | 146 | 410 | 162 | 121 | 3.3 | 40% |

**~4.5 commits per decode step** — well above the theoretical minimum of 1.

---

## 3. Root Cause Analysis

### 3.1 `prepare_length_mask` — unnecessary CPU sync (~140 commits)

**Location:** `src/metal/primitives_beam_search.mm:91`

```cpp
CT2_COMMIT_AND_WAIT();  // Flushes GPU before mask kernel reads lengths
```

**Why it exists:** The `lengths` buffer may have been written by a prior GPU op. The commit ensures GPU writes are visible before the mask kernel reads them.

**Why it's unnecessary in the decode loop:**
- `lengths` (self-attention) comes from decoder state and is stable after step 0
- `memory_lengths` (cross-attention) is set during encoding and never changes
- Even if lengths *were* GPU-written, the prior sampler sync already flushed all pending GPU work
- The mask kernel only reads `lengths` on GPU and writes `mask` on GPU — no CPU access needed

**Called twice per decode step:** once for self-attention mask (line 674 in transformer.cc) and once for cross-attention mask (line 711).

**Fix:** Replace with `encode_barrier()` — a GPU-side-only ordering barrier that ensures prior GPU writes are visible to subsequent GPU kernels without blocking the CPU. Infrastructure already exists at `utils.mm:209-224`.

### 3.2 Sampler sync — required but is the only truly necessary sync

**Location:** `src/sampling.cc:29`

```cpp
synchronize_stream(Device::MPS);  // Flush GPU TopK results to CPU
```

**Why it's required:** CPU needs token IDs for next decode step input. This is the fundamental GPU→CPU sync point in autoregressive decoding — it cannot be eliminated.

After fixing 3.1, this becomes the **only** sync per decode step (~1 commit/step = theoretical minimum).

### 3.3 CPU overhead beyond commits — 2,150ms (57% of wall time)

Estimated breakdown:
- **ObjC autoreleasepool + MPSMatrix alloc/release:** ~30 GEMMs/step × ~70 steps × ~10µs/GEMM = ~21ms (ObjC dispatch overhead)
- **`metal_buffer_for_ptr()` linear scan:** ~10,000+ calls × ~1-5µs = ~10-50ms
- **MPSMatrixDescriptor creation:** 3 per GEMM × 2,100 GEMMs = 6,300 descriptor allocs
- **Beam management (CPU):** sorting, gathering, bookkeeping = ~hundreds ms
- **Tokenization/detokenization:** ~50-100ms (one-time, amortized)
- **Python→C++ boundary:** minimal per batch call

The dominant CPU cost is likely the **beam management and decoder state bookkeeping**, which is pure CPU work that cannot be offloaded. However, reducing ObjC overhead in the GEMM hot path can recover ~10-20%.

---

## 4. Improvement Proposals

### 12.1 — `encode_barrier()` in `prepare_length_mask` (HIGH IMPACT, LOW EFFORT)

**Change:** Single line replacement in `primitives_beam_search.mm:91`:
```cpp
// Before:
CT2_COMMIT_AND_WAIT();
// After:
metal::encode_barrier();
```

**Expected impact:**
- Commits: 288 → ~148 (49% reduction)
- Commit overhead: ~115ms → ~59ms
- GPU utilization: ~40% → ~55-60%
- Float16 tok/s: 1,006 → estimated ~1,200-1,400

**Risk:** Low. `encode_barrier()` provides the same GPU ordering guarantee as `commit_and_wait()` but without blocking the CPU. The mask kernel reads `lengths` (GPU) and writes `mask` (GPU) — no CPU-visible side effects. The next natural sync (sampler) will flush everything before CPU reads any results.

**Validation:** Run profiler before/after; verify commit count drops by ~50%, output tokens identical.

### 12.2 — ObjC overhead reduction in GEMM path (MEDIUM IMPACT, MEDIUM EFFORT)

**Problem:** Each `dispatch_mps_gemm` creates 3 `MPSMatrix` + 3 `MPSMatrixDescriptor` objects per call. With ~2,100 GEMM calls per 100-sentence batch, that's ~12,600 ObjC alloc/release cycles.

**Options (ordered by effort):**

a) **Cache `MPSMatrixDescriptor`** by (rows, cols, rowBytes, dataType) — thread-local cache with LRU eviction. Similar pattern to existing GEMM cache (M11.27).

b) **Custom f32 GEMV kernel for m=1 decode** — the f16 path already has a custom GEMV kernel (M11.19) that bypasses MPS entirely. Adding f32 GEMV would eliminate all MPS matrix wrapper overhead for the decode hot path (m=1 GEMMs, which are ~80% of all GEMMs in autoregressive decoding).

c) **Pool `MPSMatrix` objects** — reuse when buffer/descriptor match. More complex but addresses all GEMM sizes.

**Expected impact:** 10-20% wall-time reduction from reduced ObjC overhead.

### 12.3 — Buffer lookup optimization (MEDIUM IMPACT, MEDIUM EFFORT)

**Problem:** `metal_buffer_for_ptr()` in `allocator.mm` linearly scans `_live` map to find which MTLBuffer contains a given pointer. Called 3× per GEMM + 2× per compute kernel.

**Fix:** Replace `_live` with a `std::map<uintptr_t, ...>` keyed by buffer start address. Use `upper_bound()` for O(log n) lookup. Or maintain a small thread-local cache of recent lookups.

**Expected impact:** Moderate — depends on number of live allocations. For OPUS-MT (~100 tensors), the linear scan is O(100) × 10,000 calls = 1M comparisons. With sorted map: O(7) × 10,000 = 70K comparisons.

### 12.4 — MPS benchmark and README update (after optimizations)

Re-run `tools/benchmark/benchmark_mps.py` after implementing 12.1-12.3. Update README.md with final MPS performance numbers.

---

## 5. Expected Cumulative Impact

| Optimization | Commits | GPU Util | Est. float16 tok/s |
|-------------|---------|----------|---------------------|
| Baseline | 288 | 40% | 1,006 |
| 12.1 (encode_barrier) | ~148 | ~55% | ~1,200-1,400 |
| 12.1 + 12.2 (ObjC reduction) | ~148 | ~60% | ~1,400-1,600 |
| 12.1 + 12.2 + 12.3 (buffer lookup) | ~148 | ~65% | ~1,500-1,800 |

**Target:** MPS float16 ≥ 2× CPU float32 = 1,868 tok/s

---

## 6. Files Involved

| File | Role |
|------|------|
| `src/metal/primitives_beam_search.mm:91` | `prepare_length_mask` commit (12.1) |
| `src/metal/utils.mm:209-224` | `encode_barrier()` implementation |
| `src/metal/primitives_gemm.mm:170-377` | GEMM path with MPSMatrix alloc (12.2) |
| `src/metal/allocator.mm` | `metal_buffer_for_ptr()` lookup (12.3) |
| `src/sampling.cc:29` | Sampler sync (required, not removable) |
| `src/layers/transformer.cc:674,711` | Decoder calls to `prepare_length_mask` |
| `tools/benchmark/benchmark_mps.py` | MPS benchmark script |
| `tools/benchmark/profile_mps_translation.py` | Profiler used for this analysis |

---

## 7. Benchmark Infrastructure Created

### `tools/benchmark/benchmark_mps.py`
- Translates WMT14 En→De (2,737 sentences) with OPUS-MT model
- Tests CPU float32 and MPS float32/float16
- **Subprocess isolation**: each config runs in a separate process to prevent Metal VSIZE accumulation
- Reports tok/s, max RSS, BLEU in README-compatible Markdown tables
- Supports `--num_samples`, `--num_cpus`, `--beam_size` arguments

### `tools/benchmark/profile_mps_translation.py`
- Detailed profiling with Metal counters (commit count, GPU time, PSO stats)
- Per-batch analysis with commits/step breakdown
- Automatic bottleneck diagnosis
- Commit trace dump via `CT2_MPS_TRACE`
- Supports `--compute_type`, `--max_sentences`, `--profile_cpu` flags

---

## 8. Key Insight: Small Model vs Large Model

The OPUS-MT model (d_model=512, 6 layers) is **disadvantaged on GPU** because:
- GEMM sizes are small (m=1-32, n=512, k=512) — below GPU efficiency threshold
- Command buffer overhead (~0.4ms) is a larger fraction of small GEMM compute time
- CPU BLAS (Accelerate) is highly optimized for small matrices on Apple Silicon

For larger models (Whisper large-v3-turbo: d_model=1280, 32 layers), the same MPS backend achieves **22.1× speedup** because GEMM sizes are above the GPU crossover point.

The M12 optimizations will help both model sizes, but the biggest relative improvement will be for small models where sync overhead is a larger fraction of total time.

---

## 9. BF16 / INT8 / CPU-INT8 Issues & Proposals

### 9.1 BF16 — MPSGraph Synchronous Execution (>20 min timeout)

**Benchmark result:** Timed out after 20 minutes for 2,737 sentences (beam=4).

**Root cause:** BF16 GEMM uses `MPSGraph.runWithMTLCommandQueue:` which is **inherently synchronous** — it commits its own internal command buffer and blocks until GPU completion. This is fundamentally different from the FP32/FP16 path which uses `MPSMatrixMultiplication` (encode-only, deferred).

**Per-GEMM overhead chain:**
1. `CT2_COMMIT_AND_WAIT()` — flush prior deferred CB (~0.4ms)
2. `MPSGraphTensorData` alloc (2×) — ObjC alloc per input matrix
3. `runWithMTLCommandQueue:` — graph compilation + synchronous GPU execution
4. `readBytes:` — copy result from MPSNDArray to output buffer
5. `[tdA release]; [tdB release]` — ObjC dealloc

For batched GEMM (e.g., attention with batch_size=4): 1 pre-flush + B synchronous graph runs = **B+1 blocking points per batched GEMM call**.

Each Transformer decoder step has ~6 GEMMs (QKV, attn output, FFN×2). With 6 decoder layers × 6 GEMMs × beam_size=4 = ~144 synchronous graph executions per decode step. At ~1ms per graph execution = ~144ms per decode step (vs ~5ms for FP16).

**Improvement proposals (12.5):**

a) **MPSGraph async execution** — Use `runAsyncWithMTLCommandQueue:` instead of `runWithMTLCommandQueue:`. This encodes the graph's work into a command buffer without blocking. Requires:
   - Tracking the completion event (MTLSharedEvent or command buffer completion handler)
   - Deferring `readBytes:` until the next natural sync point
   - Risk: MPSGraph may not support truly deferred execution with our buffer management

b) **Batched graph execution** — Instead of B separate `runWithMTLCommandQueue:` calls for a batched GEMM, construct a single graph with batch dimension. MPSGraph supports `matrixMultiplicationWithPrimaryTensor:` on 3D tensors. This would reduce B graph runs → 1 graph run.
   - **Expected impact:** 4× fewer graph executions for beam_size=4

c) **FP16 fallback for BF16 models** — Since MPS float16 is fast (1,006 tok/s), automatically promote BF16 weights to FP16 at load time on MPS devices where native BF16 GEMM is too slow. This is a lossy conversion but maintains >99% accuracy for translation.
   - **Expected impact:** Immediate — BF16 models run at FP16 speed

d) **Custom BF16 GEMV kernel for m=1** — For the decode hot path (m=1), bypass MPSGraph entirely with a custom MSL GEMV kernel that operates on BF16 directly. Similar pattern to the existing FP16 GEMV kernel (M11.19). Apple Silicon M3+ has hardware BF16 support in the GPU ALU.
   - **Expected impact:** Eliminates all graph overhead for decode steps

### 9.2 INT8 — Very Slow (~131 tok/s, ~559s for 2737 sentences)

**Benchmark result:** MPS INT8 = 131 tok/s (7× slower than CPU float32).

**Root cause:** No native INT8 GEMM on Metal. The current path:
1. `CT2_COMMIT_AND_WAIT()` — flush pending GPU work for CPU read (~0.4ms)
2. CPU: int8→float32 conversion (via vDSP_vflt8, vectorized)
3. GPU: float32 MPS GEMM (encode-only)
4. `CT2_COMMIT_AND_WAIT()` — wait for GPU GEMM result (~0.4ms)
5. CPU: float32→int32 rounding (scalar loop)

For single GEMM: 2 syncs. For batched GEMM (already optimized): 2 syncs total.

The real issue is the **CPU conversion overhead** (steps 2 and 5) combined with **GEMM running on float32** (4× the memory bandwidth of int8).

**Improvement proposals (12.6):**

a) **GPU int8→float32 kernel** — Replace CPU `int8_to_float32` with a Metal compute shader. This eliminates the first `CT2_COMMIT_AND_WAIT()` entirely — the conversion runs on GPU as an encode-only kernel, feeding directly into the MPS float32 GEMM.
   ```metal
   kernel void int8_to_float32(device const char* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
       output[gid] = float(input[gid]);
   }
   ```
   - **Expected impact:** Eliminates 1 of 2 syncs per GEMM, removes CPU conversion time

b) **GPU float32→int32 rounding kernel** — Similarly, replace the CPU scalar rounding loop with a GPU kernel:
   ```metal
   kernel void float32_to_int32(device const float* input [[buffer(0)]],
                                 device int* output [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
       output[gid] = int(rint(input[gid]));
   }
   ```
   Combined with (a), this makes the entire INT8 GEMM pipeline fully GPU-resident with **0 syncs** (all encode-only).

c) **INT8 dequantize-on-load** — At model load time, dequantize INT8 weights to FP16 and run the model in FP16 mode. This trades 2× memory for FP16-level performance. For OPUS-MT (small model), the memory increase is negligible.
   - **Expected impact:** Immediate — INT8 models run at FP16 speed (~1,006 tok/s)

### 9.3 CPU INT8 — Build Configuration Issue

**Benchmark result:** `ValueError: Requested int8 compute type, but the target device or backend do not support efficient int8 computation.`

**Root cause:** The ct2 conda environment was built **without** MKL, DNNL, or RUY backends. The check at `types.cc:166`:
```cpp
case Device::CPU:
    return cpu::has_gemm_backend(ComputeType::INT8);  // → false
```

CPU INT8 requires one of:
- `CT2_WITH_MKL` — Intel Math Kernel Library (x86 only)
- `CT2_WITH_DNNL` — Intel Deep Neural Network Library (x86 + ARM via ACL)
- `CT2_WITH_RUY` — Google's RUY library (ARM, used by TensorFlow Lite)

**Fix (12.7):**

a) **Build with RUY** — Best option for Apple Silicon. CMake: `cmake -DCT2_WITH_RUY=ON ..`
   RUY provides optimized INT8×INT8→INT32 GEMM for ARM NEON.

b) **Build with DNNL + ACL** — Alternative: DNNL can use Arm Compute Library for INT8 on ARM.

c) **Benchmark script fix** — The benchmark should detect INT8 availability before attempting it:
   ```python
   # Check if INT8 is supported on this device
   try:
       translator = ctranslate2.Translator(model_path, device=device,
                                           compute_type="int8")
   except ValueError:
       print(f"  SKIPPED: int8 not supported on {device}")
   ```

### 9.4 Summary: Compute Type Support Matrix (Current → Target)

| Compute Type | MPS Status | MPS tok/s | Root Cause | Fix Priority | Fix Approach |
|---|---|---|---|---|---|
| float32 | ✅ Works | 269 | CB overhead > compute | HIGH | 12.1 encode_barrier |
| float16 | ✅ Works | 1,006 | ~1× CPU, should be 2-3× | HIGH | 12.1-12.3 |
| bfloat16 | ⚠️ Timeout | <50 est. | MPSGraph synchronous | MEDIUM | 12.5a/b/c/d |
| int8 | ⚠️ Very slow | 131 | CPU conversion + 2 syncs | MEDIUM | 12.6a/b/c |
| CPU int8 | ❌ Error | N/A | Build lacks RUY/MKL/DNNL | LOW | 12.7a build fix |

### 9.5 Implementation Priority (revised)

| # | Proposal | Effort | Impact | Risk |
|---|----------|--------|--------|------|
| 12.1 | `encode_barrier()` in prepare_length_mask | Low | HIGH | Low |
| 12.2 | ObjC overhead reduction in GEMM | Medium | MEDIUM-HIGH | Low |
| 12.3 | Buffer lookup optimization | Medium | MEDIUM | Low |
| 12.4 | Benchmark + README update | Low | — | None |
| 12.5c | BF16→FP16 auto-promotion on MPS | Low | HIGH for BF16 | Low (lossy) |
| 12.5d | Custom BF16 GEMV kernel (m=1) | Medium | HIGH for BF16 | Medium |
| 12.5b | Batched MPSGraph execution | High | HIGH for BF16 | High |
| 12.6c | INT8 dequantize-to-FP16 on load | Low | HIGH for INT8 | Low (2× mem) |
| 12.6a | GPU int8→float32 kernel | Medium | MEDIUM for INT8 | Low |
| 12.7a | Build with RUY for CPU INT8 | Low | Enables CPU INT8 | None |
