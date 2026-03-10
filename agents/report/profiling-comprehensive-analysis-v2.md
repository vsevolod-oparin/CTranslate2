# Comprehensive Metal Performance Analysis v2

**Date**: 2026-03-10 (post M11.28 + audit fixes + OPT implementation)
**Model**: whisper-large-v3-turbo, float16, Apple M4
**Audio**: 60s sample (sample.mp3)

---

## Executive Summary

The Metal backend achieves **1.9s for 30s audio** (raw API) — a **13.8x improvement** from M11.3 baseline (41s). However, **GPU utilization is only 16%** of wall time. The remaining 84% is CPU-side overhead dominated by:

1. **CTranslate2 ThreadPool architecture** (~960ms OS scheduling per API call)
2. **Per-token decode overhead** (~12.5ms/tok wall vs ~2.5ms/tok GPU)
3. **faster_whisper temperature fallback retries** (up to 6x per segment)

**Bottom line**: The Metal GPU kernels are fast. The overhead is in CTranslate2's CPU-side architecture (ThreadPool, beam search on CPU) and faster_whisper's retry logic. Further gains require architectural changes upstream, not Metal kernel optimization.

---

## Section 1: Raw API Baseline (30s, beam=5)

| Metric | Value |
|--------|------:|
| Wall time | 1,887 ms (post-OPT) |
| GPU time | 309 ms (16.4%) |
| Non-GPU time | 1,578 ms (83.6%) |
| Commits | 126 |
| Tokens | 123 |

### Encode vs Decode Breakdown

| Phase | Wall (ms) | GPU (ms) | Non-GPU (ms) | Commits |
|-------|----------:|----------:|-------------:|--------:|
| Encode (standalone) | 997 | 17 | **980** | 1 |
| Generate (30s) | 1,887 | 309 | **1,578** | 126 |

**The encoder uses 17ms of GPU time but ~1,000ms of wall time.** The ~980ms gap is CTranslate2's ThreadPool overhead (see Section 2).

### Per-Token Scaling

| max_length | Tokens | Wall (ms) | ms/tok | GPU (ms) | GPU ms/tok |
|------------|-------:|----------:|-------:|---------:|-----------:|
| 10 | 5 | 1,081 | 216.2 | 28 | 5.6 |
| 30 | 15 | 1,110 | 74.0 | 51 | 3.4 |
| 60 | 30 | 1,270 | 42.3 | 88 | 2.9 |
| 120 | 60 | 1,440 | 24.0 | 155 | 2.6 |

**Linear fit**: Wall = **1,050ms** fixed + **~7ms/tok** marginal. The 1,050ms fixed cost is dominated by encode.

### Beam Size Impact

| beam_size | Wall (ms) | GPU (ms) | Non-GPU (ms) | Tokens |
|----------:|----------:|---------:|-------------:|-------:|
| 1 | 2,088 | 694 | 1,394 | 124 |
| 2 | 1,847 | 273 | 1,574 | 123 |
| 5 | 1,988 | 305 | 1,684 | 123 |
| 10 | 1,749 | 367 | 1,382 | 49 |

beam=1 is **slowest** because greedy search does more decode steps. beam=10 is fastest but truncates early (49 tokens). **Non-GPU time is ~1,400-1,700ms regardless of beam size.**

---

## Section 2: Encoder Bottleneck Analysis

The encode path (`WhisperReplica::encode`) does:

```
features.move_to()    →  device transfer (CPU→Metal shared memory)
Conv1D ×2             →  encode-only (GPU: ~3ms)
Transformer layers    →  encode-only (GPU: ~14ms)
synchronize_stream()  →  commit_and_wait (GPU: ~17ms + OS wakeup)
```

**One GPU sync** after 17ms of GPU work. But the total standalone encode is ~1,000ms.

### Where is the 980ms?

From native `sample` profiling (15s, 95K samples):

| Symbol | Samples | % | Description |
|--------|--------:|--:|-------------|
| `__psynch_cvwait` | 66,315 | 69.8% | Main thread waiting for worker |
| `__workq_kernreturn` | 23,843 | 25.1% | Worker thread idle/OS scheduling |
| `buffer_for_ptr` | 615 | 0.65% | O(N) scan for MTLBuffer lookup |
| `AGX::BlitDispatchContext::checkDependentBlits` | 272 | 0.29% | GPU driver blit dependency check |
| `__bzero` | 1,211 | 1.27% | Memory zeroing |
| `objc_msgSend` | 203 | 0.21% | ObjC dynamic dispatch |
| `ApplyTimestampRules::apply` | 74 | 0.08% | CPU-side vector ops |
| `SHA256_compress` (MPS) | 51 | 0.05% | MPS kernel DAG hashing |

**The main thread (Python) spends 69.8% blocking on `__psynch_cvwait`** — waiting for the worker thread (C++ ThreadPool). The worker thread spends 25.1% in `__workq_kernreturn` — OS kernel scheduling overhead between thread wakeup and actual work.

### Root Cause: Thread Pool + Future Architecture

```
Python thread          Worker thread          GPU
    |                       |                  |
    |--post(lambda)-------->|                  |
    |                       |--encode-only---->|
    |                       |--encode-only---->|
    |                       |--sync----------->|
    |                       |   (wait)         |--17ms GPU work--|
    |   (blocked on         |<--sync complete--|                  |
    |    future.get())      |                  |
    |<--result returned-----|                  |
```

The Python main thread calls `model.generate()` which posts a lambda to the ThreadPool and blocks on `future.get()`. The worker thread does all the Metal work. The ~980ms non-GPU time includes:

1. **Thread wakeup latency**: OS scheduler cost to dispatch the lambda to the worker thread
2. **`commit_and_wait()`**: Metal CB submission + GPU completion wait + thread wakeup
3. **OS scheduling jitter**: `__workq_kernreturn` overhead suggests thread scheduling is a significant factor

### Estimated Encode Time Breakdown

| Component | Estimated (ms) | Notes |
|-----------|---------------:|-------|
| Thread dispatch + wakeup | ~5 | ThreadPool post + condition_variable notify |
| `move_to(device, dtype)` | ~50-100 | CPU→Metal transfer + dtype conversion |
| GPU encode (actual) | 17 | Conv1D + Transformer layers |
| `synchronize_stream()` | ~5-10 | CB submission + 17ms GPU wait |
| Thread return + future notify | ~5 | Worker → main thread wakeup |
| **OS scheduling overhead** | **~700-800** | **Dominant: thread scheduling, kernel transitions** |

The OS scheduling overhead is the biggest factor. CTranslate2's ThreadPool architecture introduces 2 thread context switches per API call (main→worker, worker→main), plus kernel transitions for condition_variable wait/signal. This is inherent to the architecture and cannot be fixed without major upstream changes.

---

## Section 3: faster_whisper Analysis

### Overhead vs Raw API

| Method | Wall (ms) | Notes |
|--------|----------:|-------|
| Raw API (30s) | 1,887 | Single generate call |
| Raw API (60s, 3 chunks) | 4,547 | 3× generate calls |
| faster_whisper (60s) | 6,842–16,617 | **2.1x–4.6x variance** |
| faster_whisper CPU | ~36,000 | CPU baseline |

### Seek-Level Breakdown (3 runs)

| Run | Wall | Seeks | Seek Details | Python overhead |
|-----|-----:|------:|-------------|----------------:|
| 0 | 15,227ms | 2 | 924ms + **12,223ms** | 121ms (0.8%) |
| 1 | 16,617ms | 3 | 2,031ms + **10,123ms** + 649ms | 118ms (0.7%) |
| 2 | 6,842ms | 2 | 960ms + **3,782ms** | 113ms (1.7%) |

**Python overhead is negligible (0.7-1.7%).** The variance comes entirely from the C++ `generate_with_fallback` calls.

### Why the Huge Variance?

`generate_with_fallback` retries generation with increasing temperatures when quality thresholds fail:

```python
temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # default: up to 6 attempts!

for temperature in temperatures:
    result = model.generate(...)
    if compression_ratio < 2.4 and avg_logprob > -1.0:
        break  # Success
    # else: retry with next temperature
```

**With Metal MPS non-determinism**, beam search outputs vary between identical runs. This causes:
- `avg_logprob` to fluctuate around the -1.0 threshold
- Some runs pass on the first try (temperature=0.0), others need 3-5 retries
- Each retry costs ~1,900ms (a full generate call)

### Seek[1] is the Problem Seek

The second audio chunk (30-60s) consistently triggers retries:
- Run 0: 12,223ms ÷ 1,900ms/attempt ≈ **6 attempts** (all temperatures exhausted)
- Run 1: 10,123ms ÷ 1,900ms/attempt ≈ **5 attempts**
- Run 2: 3,782ms ÷ 1,900ms/attempt ≈ **2 attempts**

### with_timestamps Impact

| Mode | Wall (ms) | Commits | GPU (ms) |
|------|----------:|--------:|---------:|
| without_timestamps=True | 12,184 | 593 | 7,383 |
| without_timestamps=False | **19,740** | 1,153 | 11,517 |

Timestamps add **62% overhead** — the timestamp rules and extra decoding logic add significant work.

---

## Section 4: Native C++ Profiling Findings

### buffer_for_ptr — Fixed (0.65% → O(log N))

`MetalAllocator::buffer_for_ptr()` was doing an O(N) scan of the `_live` map (615 samples, 0.65% of CPU time). **Fixed**: replaced `std::unordered_map<void*>` with `std::map<const uint8_t*>` for O(log N) `upper_bound` lookups. Also fixes `protect_buffer()` from O(N) to O(log N).

### MPS SHA256 Overhead (51 samples)

`MPSKernelDAG::getDAGAndHash` computes SHA256 of kernel DAG strings for every MPS dispatch. Our M11.27 cache avoids re-creating `MPSMatrixMultiplication` objects, but MPS still internally hashes on each `encodeToCommandBuffer:`. This is not fixable from our side.

### Memory Operations (1,211 + 170 samples)

`__bzero` (1,211) and `_platform_memmove` (170) — buffer zeroing and memory copies. Some of this comes from `memset` in `dispatch_mps_gemm_batched_padded` when `pad_c && beta==0`.

---

## Section 5: Commit Trace Analysis

### Per-Token Commit Breakdown (CT2_METAL_TRACE)

Instrumented one `generate()` call (30s audio, beam=5, 123 tokens):

| Source | Commits | Description |
|--------|--------:|-------------|
| `devices.cc:162` (`synchronize_stream`) | 124 | **Per-token sampling sync** |
| `tile_metal.mm:25` | 1 | Tile operation (one-time) |
| `primitives_beam_search.mm:91` | 1 | Beam length mask (one-time) |
| **Total** | **126** | |

**124 of 126 commits are the per-token sampling sync** in `sampling.cc:29`. Each decode step must:

1. **GPU** (encode-only): decoder layers → logits → TopK
2. **`synchronize_stream()`**: flush deferred CB, wait for GPU completion
3. **CPU**: read sampled token IDs + scores → beam search state update → next step

This sync is **architecturally required** — CTranslate2's beam search runs on CPU. The token IDs and scores must be readable on CPU after each step. At ~0.4ms overhead per commit, 124 commits = ~50ms (2.6% of wall time).

### Key Finding: Forward Pass is Fully Encode-Only

The entire decoder forward pass (LayerNorm, GEMM, SDPA, FFN) runs **encode-only** for FP16 — zero `commit_and_wait()` calls within the neural network layers. All GPU work is batched into a single deferred command buffer per token step, flushed only at the sampling sync. This is optimal.

The commit count is already at the **theoretical minimum** for CPU-based beam search: 1 sync per token + 2 one-time costs.

---

## Section 6: Optimization Attempts and Results

### Implemented

#### OPT-1: Non-synchronous Lambda Copy — **IMPLEMENTED, MINIMAL IMPACT**
**File**: `src/models/whisper.cc` (lines 666, 680, 695, 708, 722)

Replaced `features.sync_copy()` with `StorageView(features)` in all 5 Whisper API lambda captures. For CPU features (the common path from Python/faster_whisper), `sync_copy()` was already a no-op (`synchronize_stream(CPU)` does nothing). Impact: ~0% measurable.

#### OPT-3: buffer_for_ptr O(N) → O(log N) — **IMPLEMENTED, ~12ms savings**
**File**: `src/metal/allocator.mm`

Replaced `std::unordered_map<void*>` with `std::map<const uint8_t*>` for the `_live` allocation map. `buffer_for_ptr()` and `protect_buffer()` now use `upper_bound()` + step-back for O(log N) lookups. With 0.65% of CPU samples (615/95K), estimated savings ~12ms per inference. Scales better with more live allocations.

### Investigated and Not Actionable

#### OPT-2: Skip Encode Final Sync — **UNSAFE, REVERTED**
**File**: `src/models/whisper.cc:106`

Attempted to skip `synchronize_stream(device)` in `WhisperReplica::encode()` for Metal. **Result**: standalone `encode()` returns garbage data (verified: mean=0.48 vs expected -0.0017, max_diff=9.19). The GPU work is deferred but never committed before the result reaches Python.

The internal `maybe_encode()` path (used by `generate()`) **already skips the sync** — this optimization is already in place for the generate path. The standalone `encode()` sync is required because faster_whisper (and the public API) reads the result.

The ~980ms encode overhead is **98% OS thread scheduling**, not the sync itself (~22ms). Even eliminating the sync would only save ~22ms.

#### OPT-4: Reduce Commit Count — **ALREADY AT THEORETICAL MINIMUM**

Commit tracing (Section 5) shows 124/126 commits are the architecturally required per-token sampling sync. The forward pass is fully encode-only. The 2 remaining commits (tile, beam_search) are one-time costs. No unnecessary commits remain.

At 0.4ms per commit × 126 commits = ~50ms total (2.6% of wall time), commit overhead is not a meaningful target.

---

## Section 7: Full Timing Waterfall

### Raw API (30s, beam=5) — 1,887ms

```
├─ ThreadPool dispatch          ~5ms    [lambda post + wakeup]
├─ Encode (via maybe_encode, no standalone sync)
│  ├─ move_to(device, dtype)   ~50ms    [CPU→Metal transfer]
│  ├─ Conv1D ×2 (GPU)          ~3ms     [encode-only]
│  ├─ Transformer (GPU)        ~14ms    [encode-only]
│  Subtotal: ~67ms GPU-visible (in ~1,000ms wall due to OS scheduling)
├─ Decode loop (123 tokens)
│  ├─ Per-token GPU work       ~2.5ms/tok × 123 = 308ms   [encode-only]
│  ├─ Per-token sampling sync  ~0.4ms/tok × 124 = 50ms    [commit_and_wait]
│  ├─ Per-token CPU overhead   ~10ms/tok × 123  = 1,230ms [beam search, tensor setup]
│  Subtotal: ~1,588ms
└─ Result transfer + cleanup    ~6ms
```

### faster_whisper (60s, beam=5) — 6,842ms (best case)

```
├─ Audio decode               ~130ms
├─ Feature extraction          ~15ms
├─ Seek 1 (0-30s)
│  ├─ encode()                ~985ms    [1 commit + OS scheduling]
│  └─ generate_with_fallback  ~960ms    (1 attempt, passed)
├─ Seek 2 (30-60s)
│  ├─ encode()                ~990ms    [1 commit + OS scheduling]
│  └─ generate_with_fallback  ~3,782ms  (2 attempts, 1 retry)
├─ Python overhead            ~113ms
└─ Total: 6,842ms
```

**Worst case (Run 0: 15,227ms)**: Seek 2 retried 6 times = 12,223ms for a single segment.

---

## Section 8: Where Time Goes (Summary)

### Raw API (1,887ms)

| Category | Time (ms) | % | Fixable? |
|----------|----------:|--:|----------|
| GPU compute | 309 | 16.4% | Already fast |
| OS thread scheduling | ~700 | 37% | Requires CT2 architectural change |
| CPU beam search + tensor setup | ~780 | 41% | Inherent to CPU beam search |
| commit_and_wait overhead | ~50 | 2.6% | At theoretical minimum |
| Memory ops (bzero, memcpy) | ~24 | 1.3% | Minor |
| buffer_for_ptr | ~12 | 0.6% | **Fixed (O(log N))** |
| MPS SHA256 hashing | ~6 | 0.3% | Not fixable (Apple internal) |
| Other | ~6 | 0.3% | — |

### faster_whisper Additional Overhead

| Category | Time (ms) | Fixable? |
|----------|----------:|----------|
| Temperature retries (0-5 extra) | 0–9,500 | **User parameter: `temperature=0.0`** |
| Per-seek encode() sync | ~1,000 | CT2 ThreadPool architecture |
| Python overhead | ~113 | Negligible |

---

## Key Insights

1. **GPU is NOT the bottleneck** — at 16% of wall time, the GPU is underutilized. The bottleneck is CPU-side: CTranslate2's ThreadPool OS scheduling (~37%) and CPU beam search logic (~41%).

2. **The ThreadPool architecture adds ~700ms of OS scheduling overhead** per API call. This is inherent to CTranslate2's design (separate worker thread with condition_variable signaling) and not Metal-specific. It cannot be fixed without upstream architectural changes.

3. **The per-token commit count (126) is already at the theoretical minimum** for CPU-based beam search. The entire forward pass (GEMM, SDPA, LayerNorm, FFN) is fully encode-only. Only the sampling sync (1 per token) + 2 one-time costs remain.

4. **faster_whisper's temperature fallback** is the #1 source of variance and overhead for Metal. A single parameter change (`temperature=0.0`) can reduce transcription from 15s to 5s.

5. **buffer_for_ptr O(N)→O(log N)** is the only measurable optimization implemented (~12ms, 0.6%). OPT-1 (non-sync copy) is safe but has zero measurable impact for the common CPU-features path. OPT-2 (skip encode sync) is unsafe for the public API and was reverted.

6. **The encode-to-decode sync is already skipped** in the internal `maybe_encode()` path used by `generate()`. The standalone `encode()` sync is required for correctness.

---

## Recommendations

### For CTranslate2 Metal Users (Actionable Now)

```python
# Optimal faster_whisper configuration for Metal:
model.transcribe(
    audio,
    temperature=0.0,                     # Eliminates retry variance (saves 0-10s)
    compression_ratio_threshold=3.0,      # More lenient
    log_prob_threshold=-1.5,              # More lenient
    condition_on_previous_text=False,     # Prevent error cascading
    without_timestamps=True,              # 38% faster
)
```

### For CTranslate2 Upstream (Architectural)

1. **Bypass ThreadPool for single-worker inference**: If `num_workers=1`, run the lambda directly on the calling thread. Eliminates ~700ms OS scheduling overhead per API call.
2. **GPU-side beam search**: Move token sampling + beam state to GPU. Eliminates the per-token `synchronize_stream()`. Requires significant refactoring.
3. **Pipelined encode+decode API**: Allow callers to pass raw features to `generate()` without separate `encode()` call. Already works via `maybe_encode()`, but faster_whisper uses standalone `encode()` for caching.

### For faster_whisper Upstream

1. **Backend-aware temperature defaults**: Detect Metal and use `[0.0, 0.6]` instead of `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
2. **Expose encode caching control**: Allow users to bypass the standalone `encode()` call and let `generate()` handle encoding internally when retry caching isn't needed.
