# Comprehensive Metal Performance Analysis

## Executive Summary

Previous profiling focused exclusively on `commit_and_wait()` sync counts, leading to a tunnel-vision that missed the memory leak bottleneck (M11.22, 2x improvement) and potentially other major issues. This report uses a multi-dimensional profiling approach to identify ALL bottleneck categories.

**Critical finding: Sync overhead is only ~1-3% of wall time. The real bottleneck is that 50-54% of wall time is non-GPU "CPU/other" time, and GPU utilization averages only 37%.**

---

## Profiling Methodology

### Previous (M11.5–M11.21): Sync-Centric
- Counted `commit_and_wait()` calls
- Assumed ~0.4ms fixed overhead per sync
- Multiplied: `syncs × 0.4ms = total overhead`
- **Blind spots**: Everything else

### New: Multi-Dimensional
| Dimension | Tool | What it captures |
|-----------|------|------------------|
| Wall-clock timing | `time.monotonic()` | End-to-end latency |
| GPU execution time | Metal `GPUStartTime/GPUEndTime` | Actual GPU compute time |
| Sync count/trace | `CT2_METAL_TRACE` | Where syncs happen |
| GPU utilization | IOKit `PerformanceStatistics` | % of time GPU is active |
| Memory (process) | `ps -o rss,vsz` | RSS and VSIZE |
| Memory (GPU) | IOKit `In use system memory` / `Alloc system memory` | Metal allocations |
| CPU profiling | Python `cProfile` | Python + C++ boundary hotspots |
| Stability | Multiple runs | Variance, thermal throttling |

### What's NOT Yet Available (blind spots remaining)
- **Per-kernel GPU timing**: Metal only exposes command-buffer-level timing. Individual kernel (LayerNorm, GEMM, SDPA) times are only measurable in standalone benchmarks, not during e2e inference.
- **GPU bandwidth/occupancy counters**: Apple M4 only exposes `timestamp` counter set programmatically. No ALU utilization, cache hit rates, or memory bandwidth counters via API. These are only visible in Xcode Instruments.
- **C++ function-level profiling**: `cProfile` only sees Python functions. C++ hotspots inside `ctranslate2._ext.encode` and `generate_with_fallback` are opaque.
- **xctrace integration**: `xcrun xctrace record --template "Metal System Trace" --launch python ...` could provide GPU timeline, but requires Xcode and produces large traces that need GUI analysis.

---

## Results: whisper-large-v3-turbo, beam_size=5, 60s audio

### Timing Breakdown

| Metric | Value | % of wall |
|--------|------:|----------:|
| **CPU wall time** | 36,341 ms | — |
| **Metal wall time** | 57,832 ms | 100% |
| GPU execution time | 26,079 ms | 45.1% |
| Sync overhead (est.) | 684 ms | 1.2% |
| **CPU/other time** | 31,069 ms | **53.7%** |
| **Speedup** | **0.63x** | (Metal slower!) |

### GPU Utilization (sampled at 20 Hz over inference)

| Metric | Value |
|--------|------:|
| Average | 36.8% |
| Min | 0% |
| Max | 100% |
| Samples | 688 |

GPU is idle or underutilized ~63% of the time.

### Memory

| Metric | Value |
|--------|------:|
| RSS during inference | 109 – 836 MB |
| GPU in-use memory peak | 4,954 MB |
| GPU alloc system memory | 22,877 MB |
| Model size (est.) | ~3 GB |

**GPU Alloc system memory (23 GB) is 7x the model size.** This suggests either fragmentation, over-allocation, or Metal framework internal overhead.

### Stability (4 consecutive runs)

| Run | Wall (ms) | GPU (ms) | Commits |
|-----|----------:|--------:|--------:|
| 1 | 57,832 | 26,079 | 1,710 |
| 2 | 73,992 | 28,681 | 1,330 |
| 3 | 61,854 | 25,943 | 830 |
| 4 | 78,554 | 29,131 | 1,606 |

**30% variance** between min (57.8s) and max (78.5s). GPU time is more stable (25.9–29.1s, ~12% spread) than wall time, suggesting the variance comes from the CPU/other component.

**Commit count varies 830–1,710** — non-deterministic behavior in the inference path. This is unexpected and warrants investigation.

### CPU Profile (cProfile, top hotspots)

| Self time (s) | Calls | Function |
|--------------:|------:|----------|
| 26.1 | 19 | `_thread.lock.acquire` |
| 10.9 | 2 | `ctranslate2._ext.encode` |
| 7.7 | 2 | `generate_with_fallback` |

The 26s in `_thread.lock.acquire` is the Python main thread waiting for C++ worker threads. The actual C++ execution time is inside `encode` (10.9s) and `generate_with_fallback` (7.7s) = 18.6s in the C++ extension.

But Metal wall time is 49s (cProfile run). So 49 - 18.6 = 30.4s is unaccounted Python/threading overhead + lock contention.

---

## Bottleneck Categories

### Category 1: CPU/Other Time (53.7% of wall time) — CRITICAL

The single largest bottleneck. This 31s of non-GPU time includes:

1. **Metal API overhead**: MPS object creation (`[[MPSMatrix alloc] init...]`), encoder create/end cycles (~15 per layer × 4 layers × ~200 steps), buffer binding (`setBuffer:offset:atIndex:`). Each MPS object creation involves ObjC message dispatch, heap allocation, and MTL validation.

2. **Allocator overhead**: `metal_buffer_for_ptr()` scans the `_live` map (a `std::unordered_map`) on every GPU kernel dispatch to find the backing `MTLBuffer` for a raw pointer. With hundreds of live allocations, this linear scan could be significant.

3. **ObjC/ARC-equivalent overhead**: Every `@autoreleasepool{retain}` pattern involves ObjC runtime calls. Each `[enc release]`, `[fn release]` is an ObjC message + dealloc.

4. **Python↔C++ boundary**: The GIL release/acquire cycle on every ctranslate2 function call. The faster_whisper Python layer calls into C++ many times per segment.

5. **Data copies and conversions**: `copy_from()` for GPU→CPU transfers, `to()` for device transfers, `synchronize_stream()` implicit copies.

### Category 2: GPU Underutilization (37% avg) — HIGH

The GPU is idle 63% of the time. Possible causes:

1. **Pipeline bubbles**: Between successive `commit_and_wait()` calls, the GPU sits idle while the CPU processes results and sets up the next batch.

2. **Small kernel launches**: Decode step kernels operate on small tensors (batch=1–5, seq=1 for decode). The GPU may starve with so little work.

3. **Encoder transition overhead**: Creating/ending compute encoders introduces GPU-side gaps.

4. **Serial dispatch**: All compute encoders use `MTLDispatchTypeSerial`. This prevents the GPU from overlapping independent kernel execution.

### Category 3: Memory Pressure (23 GB GPU alloc) — MEDIUM

`Alloc system memory` (23 GB) far exceeds the model size (3 GB). This could cause:

1. **System memory pressure**: macOS memory compressor and swapper activate, causing page faults
2. **TLB pressure**: Many large mappings stress the TLB, causing address translation slowdowns
3. **Metal driver overhead**: Managing thousands of MTLBuffer objects has CPU cost

### Category 4: Run-to-Run Variance (30%) — MEDIUM

30% variance between consecutive runs suggests:

1. **Thermal throttling**: Apple M4 reduces GPU/CPU clock under sustained load
2. **Memory pressure effects**: Variable paging/compression overhead
3. **Non-deterministic code paths**: Commit count varies 830–1710, suggesting different code paths are taken

### Category 5: Sync Overhead (1.2%) — LOW (previously overestimated)

At 684ms estimated, sync overhead is a small fraction of the 57.8s wall time. Previous analysis overestimated its importance by not measuring the denominator correctly.

---

## Recommended Investigations

### Priority 1: C++ Function-Level Profiling
**Tool**: `xcrun xctrace record --template "Time Profiler" --launch python -- profile_comprehensive.py`
**Goal**: Find which C++ functions consume the 31s of non-GPU time. Suspect: `metal_buffer_for_ptr()`, MPS object creation, allocator mutex contention.

### Priority 2: Metal System Trace
**Tool**: `xcrun xctrace record --template "Metal System Trace" --launch python -- bench_faster_whisper.py`
**Goal**: Visualize GPU timeline — see encoder gaps, idle bubbles, command buffer boundaries. Identify if the GPU starves between command submissions.

### Priority 3: Allocator Profiling
**Tool**: Add timing to `metal_buffer_for_ptr()`, `allocate()`, `free()` in allocator.mm
**Goal**: Measure per-call overhead of buffer lookup. If `_live` map has 1000+ entries, the `std::unordered_map` scan in `metal_buffer_for_ptr()` (which checks ptr ∈ [base, base+size)) could be O(N) per call.

### Priority 4: MPS Object Creation Overhead
**Tool**: Add timing around MPS matrix/GEMM creation in `primitives_gemm.mm`
**Goal**: Measure `[[MPSMatrix alloc] initWithBuffer:...]` and `[[MPSMatrixMultiplication alloc] initWithDevice:...]` overhead per call × thousands of calls.

### Priority 5: Thermal Throttling Detection
**Tool**: `sudo powermetrics --samplers gpu_power,cpu_power -n 10 -i 1000` during inference
**Goal**: Check if GPU/CPU clock frequencies drop during sustained inference, explaining the 30% variance.

### Priority 6: Memory Fragmentation Analysis
**Tool**: Add `pool_bytes()` / `live_bytes()` / live allocation count to the profiler
**Goal**: Track allocator state over time. If pool grows continuously or live allocation count is high, fragmentation may be causing excessive buffer creation.

---

## Key Lesson

**Sync counting is a proxy metric, not a direct bottleneck measurement.** Eliminating syncs is necessary but not sufficient. The real bottleneck is the holistic pipeline: CPU setup time, GPU utilization, memory management, and framework overhead all contribute. A 1% sync overhead doesn't mean the pipeline is 99% efficient — it means the GPU is just idle for different reasons.

The memory leak fix (M11.22) improved performance 2x not because it reduced syncs, but because it eliminated **system-wide memory pressure** that was throttling both CPU and GPU. Similar hidden bottlenecks may exist in allocator overhead, MPS object churn, or thermal throttling.

---

## C++ Function-Level Profiling (macOS `sample` command)

**Tool**: `sample <pid> 120 -file output.txt` — captures native C++ call stacks at 1ms intervals.

### Metal-Only Profile: Worker Thread (100,275 samples over 120s, 3 inference runs)

| Call Path | Samples | % | Description |
|-----------|--------:|--:|-------------|
| `indexed_fill` → `commit_and_wait` → wait | 38,120 | **38.0%** | DisableTokens::apply() forces GPU sync |
| `TransformerDecoder::decode` → layer ops | 3,004 | 3.0% | Decoder layer GPU dispatch (encode-only) |
| `MatMul::compute<Device::METAL>` → MPS GEMM | 360 | 0.36% | Attention Q*K^T dispatch |
| `Gemm::compute<Device::METAL>` → MPS GEMM | ~166 | 0.17% | Dense layer GEMM dispatch |
| `MPSKernelDAG::getDAGAndHash` (SHA256) | ~149 | 0.15% | MPS framework internal hashing |
| `fuse_timestamp_check_and_disable_metal` | 34 | 0.03% | Fused timestamp kernel + AGXBuffer alloc |
| `softmax_metal` dispatch | 35 | 0.03% | GPU softmax encode (fast) |
| `buffer_for_ptr` lookups | 3 | <0.01% | MTLBuffer lookup (fast) |
| `ApplyTimestampRules::apply` (CPU vector ops) | ~220 | 0.22% | std::vector<int>::insert |
| Encoder wait on future | ~28,000 | 28% | Main thread waiting for encode result |
| **Other (decode loop, copy_from, etc.)** | ~30,000 | 30% | Framework overhead, idle |

### Key Findings

1. **GPU GEMM IS correctly dispatched to Metal** — `MatMul::compute<(ctranslate2::Device)2, float>` and `Gemm::compute<(ctranslate2::Device)2, float>` confirm Device::METAL (device index 2). Earlier profiler results showing Device::CPU were from the CPU baseline run captured in the same profiling session.

2. **`indexed_fill` → `commit_and_wait` is the #1 CPU bottleneck (38%)** — `DisableTokens::apply()` calls `indexed_fill` which triggers `commit_and_wait_impl`. The CPU thread then blocks in `_MTLCommandBuffer waitUntilCompleted` → `__psynch_cvwait`. This sync flushes the entire GPU pipeline, creating a bubble.

3. **MPS framework overhead is real but small** — `MPSKernelDAG::getDAGAndHash` (SHA256 of kernel DAG strings) is called every GEMM dispatch. Each call involves string concatenation + SHA256 + dictionary lookup. At ~149 samples (0.15%), it's not dominant but does add up over thousands of GEMM calls.

4. **GPU dispatch (encode-only) is fast** — Metal GEMM dispatch takes only ~0.36% of CPU time. The GPU does the heavy lifting. The bottleneck is not dispatch overhead but pipeline stalls from syncs.

5. **`buffer_for_ptr` is NOT a bottleneck** — Only 3 samples. The `_live` map lookup is fast with the current allocation count.

### Corrected Bottleneck Hierarchy

| Rank | Bottleneck | Impact | Fix |
|------|-----------|--------|-----|
| 1 | `indexed_fill` syncs (DisableTokens) | 38% of CPU time | M11.25: GPU indexed_fill (eliminate sync) |
| 2 | `synchronize_stream` syncs (devices.cc) | ~15% est. | M11.24: audit & reduce |
| 3 | MPS object creation per GEMM | ~2-5% | M11.28: cache MPS objects |
| 4 | GPU pipeline bubbles from syncs | ~10-20% GPU idle | Compound effect of 1+2 |
| 5 | Thermal throttling (30% variance) | 0-30% penalty | Can't fix in software |

---

## Profiling Script

`tests/metal/e2e/profile_comprehensive.py` — measures all dimensions in a single run.

Compiled GPU stats helper: `/tmp/test_gpu_perf` (reads IOKit `PerformanceStatistics`).

Available Instruments templates: `xcrun xctrace list templates` includes "Metal System Trace", "Time Profiler", "Allocations".

### C++ Profiling Command
```bash
# Run Metal-only benchmark, then sample the worker thread
python bench_metal_only2.py whisper-large-v3-turbo 5 &
PYID=$!
sample $PYID 120 -file /tmp/metal_only_sample.txt
```
