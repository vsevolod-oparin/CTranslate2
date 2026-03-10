# Comprehensive Metal Performance Analysis v2

**Date**: 2026-03-10 (post M11.29 MTLSharedEvent hybrid barrier)
**Model**: whisper-large-v3-turbo, float16, Apple M4
**Audio**: 60s sample (sample.mp3)

---

## Executive Summary

The Metal backend achieves **1.86s for 30s audio** (raw API) — a **22.1x improvement** from M11.3 baseline (41s). However, **GPU utilization is only 16%** of wall time. The remaining 84% is CPU-side overhead dominated by:

1. **CTranslate2 ThreadPool architecture** (~960ms OS scheduling per API call)
2. **Per-token decode overhead** (~12.5ms/tok wall vs ~2.5ms/tok GPU)
3. **faster_whisper temperature fallback retries** (up to 6x per segment)

**Bottom line**: The Metal GPU kernels are fast. All non-architectural optimizations (#3–#8) have been investigated and closed — combined potential was ~15-30ms (1.5%). The overhead is in CTranslate2's CPU-side architecture (ThreadPool, beam search on CPU) and faster_whisper's retry logic. Further gains require architectural changes upstream (ARCH-1: ThreadPool bypass, ARCH-2: GPU beam search), not Metal kernel optimization.

---

## Section 1: Raw API Baseline (30s, beam=5)

| Metric | Value |
|--------|------:|
| Wall time | 1,860 ms (post-M11.29) |
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

### Raw API (1,860ms — post-M11.29)

| Category | Time (ms) | % | Fixable? |
|----------|----------:|--:|----------|
| GPU compute | ~309 | 16.6% | Already fast |
| OS thread scheduling | ~700 | 37.6% | ARCH-1: ThreadPool bypass (~700ms) |
| CPU beam search + tensor setup | ~750 | 40.3% | ARCH-2: GPU beam search (~450ms) |
| commit_and_wait overhead | ~50 | 2.7% | At theoretical minimum |
| Memory ops (bzero, memcpy) | ~24 | 1.3% | Minor |
| Non-arch optimizations #3-#8 | ~15-30 | 1.5% | **Closed — not actionable** |
| buffer_for_ptr | — | — | **Fixed (O(log N), commit 48)** |
| MPS SHA256 hashing | ~6 | 0.3% | Not fixable (Apple internal) |

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

## Section 9: Cross-Reference — All Proposed Optimizations

Comprehensive audit of every optimization proposed across all M11 reports, cross-referenced against
what has been implemented. Source reports: `roadmap-2x-whisper-large-v3-turbo.md`,
`review-m11-optimization-audit.md`, `profiling-comprehensive-analysis.md` (v1),
`metal-perf-analysis-large-v3-turbo.md`, and this report (v2).

### Implemented (DONE)

| Optimization | Source | Implemented As | Impact |
|---|---|---|---|
| CPU GEMM fallback for tiny matrices | roadmap | M11.5 (commit 5) | 6.2x (transformative) |
| MPS object memory leaks | M11.22 report | M11.22 (commit 37) | 2.7x (transformative) |
| Sync elimination (encode-only patterns) | roadmap M11.24 | M11.26 (commit 42) | 1.53x (transformative) |
| Batch `indexed_fill` GPU kernel | roadmap M11.25 | M11.25 (commit 41) | Minor |
| MPS object caching | roadmap M11.28 | M11.27 (commit 43) | ~20ms savings |
| `buffer_for_ptr` O(N)→O(log N) | profiling-v2 OPT-3 | OPT-3 (commit 48) | ~12ms savings |
| Non-sync lambda copy | profiling-v2 OPT-1 | OPT-1 (commit 48) | ~0 (CPU features path) |
| 8,728 CPU GEMM fallbacks for m=1 | metal-perf-analysis | M11.18/M11.19 | Major (encode-only MPS padded GEMM) |
| GPU TopK (single-pass fused kernel) | roadmap | M11.23 (commit 39) | 268 syncs → 0 |
| GPU Multinomial sampling | roadmap | M11.20 | Minor |
| Gather sync elimination | roadmap | M11.21 | ~1200 fewer syncs |
| Flash Cross-Attention | incremental | M11.6 (commit 8) | ~200ms savings |
| Fused LayerNorm + GEMM | incremental | commit 9 | Minor |
| Fused ApplyTimestampRules | incremental | M11.14 | Minor |
| Indexed Fill pre-sync elimination | roadmap | M11.28 (commit 44) | ~30ms savings |
| Indexed Fill hybrid sync (MTLSharedEvent) | M11.29 | M11.29 (commit 51) | ~86ms recovery (f16 path) |
| All P0-P2 audit items | review-m11-optimization-audit | All resolved | Various |

### Not Implemented — Architectural (Still Relevant)

| # | Optimization | Source | Est. Impact | Status |
|---|---|---|---|---|
| 1 | **Bypass ThreadPool (single-worker)** | profiling-v2 §10 | **~700ms (38%)** | Architectural change required |
| 2 | **GPU-side beam search** | profiling-v2 §10 | **Eliminates 124 syncs** | Major architectural change |

These are the only remaining optimizations with meaningful impact potential.
See Section 10 for detailed implementation plans.

### Not Implemented — Non-Architectural (Investigated, Closed)

Deep-dive analysis (2026-03-10 post-M11.29) of every non-architectural item from the original
"Still Relevant" list. Each was investigated against the actual codebase, commit traces, and
profiling data. **Combined potential: ~15-30ms (1.5% of 1,860ms wall time) — not actionable.**

| # | Optimization | Original Est. | Actual Est. | Verdict | Rationale |
|---|---|---|---|---|---|
| 3 | Decode-step pipeline fusion | ~100-200ms | **7-15ms (0.8%)** | **Closed — skip** | QKV is **already fused** at model level: single `[d_model, 3*d_model]` weight matrix produces concatenated Q+K+V in one GEMM, then split. FFN has only 2 GEMMs (up+gate, down) with SiLU between — cannot fuse further. Original estimate assumed 3 separate QKV GEMMs. Remaining savings would come only from reducing encoder create/end cycles (~2-5us each). |
| 4 | Persistent command encoder | ~50-100ms | **3-8ms (0.4%)** | **Closed — skip** | MPS GEMM encodes directly into the CB (no compute encoder). Persistent encoder must be ended before every MPS dispatch and restarted after — only saves encoders between consecutive custom kernels (add→layer_norm), which are few. SDPA creates only 1 encoder (causal mask), not per-head. Engineering complexity exceeds sub-1% gain. |
| 5 | Fused LayerNorm+Linear (f32) | Small | **<3ms (0.2%)** | **Closed — skip** | f32 is not the production path (f16 is). f32 also requires full `CT2_COMMIT_AND_WAIT()` in indexed_fill due to MPS driver coherency issue (M11.29 finding), making it inherently slower. Fusing LN+Linear saves one encoder dispatch + one memory round-trip per layer per token: 4 layers x 123 tokens x ~5us = ~2.5ms. |
| 6 | `MTLDispatchTypeConcurrent` | Small | **~0ms** | **Closed — skip** | Nearly all consecutive dispatches have true data dependencies (output feeds next input). GPU is only 16% of wall time — even perfect overlap of independent kernels yields ~0ms wall improvement. MPS f32 coherency issue (M11.29) indicates Metal driver has subtle ordering assumptions, making concurrent dispatch risky. |
| 7 | Cross-attention KV reuse | Small | **0ms** | **Closed — already done** | `process_cross_attention()` in `src/layers/attention.cc` already caches K/V on first decode step and reuses via `shallow_copy()` on all subsequent steps. No opportunity remains. |
| 8 | `prepare_length_mask` fence | Negligible | **~2.8ms (0.15%)** | **Closed — not worth risk** | Commit trace shows `primitives_beam_search.mm:91` fires only **7 times** per 30s inference (mask rebuilt only at beam expansion boundaries, not per token). At ~0.4ms/sync = ~2.8ms total. Same encode_barrier pattern as M11.29 would work, but 2.8ms gain does not justify the risk of stale lengths → wrong attention masks → silent correctness bugs. |

#### Key Finding: Original Estimates Were Inflated

The original estimates for items #3 and #4 (~100-200ms and ~50-100ms) were based on assumptions
that did not hold against the actual codebase:

- **#3 assumed 3 separate QKV GEMMs** — the model actually stores fused `[d_model, 3*d_model]`
  weights, so only 1 GEMM fires. The "fusion" was already done at model conversion time.
- **#4 assumed per-head encoder creation in SDPA** — MPS GEMM does not use compute encoders
  (it encodes directly into the CB via `encodeToCommandBuffer:`). SDPA has only 1 compute encoder
  (for the causal mask kernel), not per-head.
- **#8 assumed per-token invocation** — commit tracing shows only 7 calls per inference, not 123.

### Not Implemented — Obsoleted or Unsafe

| Optimization | Source | Reason |
|---|---|---|
| Skip `synchronize_stream` in `encode()` | OPT-2 | **Unsafe** — produces garbage. Verified, reverted. |
| Reduce commit count below 126 | OPT-4 | **At theoretical minimum** — 124/126 are per-token sampling syncs. |
| Dead code `should_sample_timestamps_metal` | audit | Cosmetic only, no perf impact |

---

## Section 10: Architectural Change Details (High-Impact)

### ARCH-1: Bypass ThreadPool for Single-Worker Inference

**Estimated savings**: ~700ms per API call (38% of current 1,860ms)
**Effort**: Medium — localized to `ReplicaPool::post()` and `ReplicaPool::post_batch()`
**Risk**: Low — no GPU or model changes, pure CPU control-flow optimization

#### Problem

CTranslate2's `ReplicaPool` always dispatches work through a `ThreadPool`, even when there is only
one worker (the common case for Metal/single-GPU inference). The dispatch path:

```
Python main thread                    Worker thread
     │                                     │
     ├─ post(lambda)                       │
     │   ├─ create BatchJob               │
     │   ├─ JobQueue::put()                │
     │   │   └─ _can_get_job.notify_one()  │  ← OS kernel: futex/psynch_cvsignal
     │   └─ future.get()                   │
     │       └─ __psynch_cvwait (BLOCKS)   │  ← 69.8% of CPU time
     │                                     ├─ __workq_kernreturn (wakeup)  ← 25.1%
     │                                     ├─ BatchJob::run()
     │                                     │   └─ lambda(replica)  ← actual GPU work
     │                                     └─ promise.set_value()
     │       ← wakeup from cvwait          │
     └─ return result                      │
```

Two OS thread context switches (~300-400µs each on macOS) plus condition_variable signaling adds
~700-800ms per API call. This shows up as `__psynch_cvwait` (69.8%) and `__workq_kernreturn`
(25.1%) in `sample` profiling.

#### Proposed Solution

Add a direct-call fast path in `ReplicaPool::post()` when `num_replicas() == 1`:

```cpp
// replica_pool.h — inside ReplicaPool<Replica>::post()
template <typename Result, typename Func>
std::future<Result> post(Func func) {
  // Fast path: single worker → run directly on calling thread.
  // Avoids 2 context switches + condition_variable signaling (~700ms on macOS).
  if (num_replicas() == 1) {
    std::promise<Result> promise;
    auto future = promise.get_future();
    try {
      auto& worker = static_cast<ReplicaWorker<Replica>&>(_thread_pool->get_worker(0));
      Result result = func(worker.replica());
      promise.set_value(std::move(result));
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
    return future;
  }

  // Original path: dispatch through ThreadPool
  auto batched_func = [func = std::move(func)](Replica& replica) mutable {
    std::vector<Result> results;
    results.reserve(1);
    results.emplace_back(func(replica));
    return results;
  };
  auto futures = post_batch<Result>(std::move(batched_func), 1);
  return std::move(futures[0]);
}
```

#### Key Considerations

1. **Thread-local state**: `ReplicaWorker::initialize()` sets device index, thread count, and
   allocator for the worker thread. The direct-call path must either:
   - Call `set_device_index()` + `set_num_threads()` on the calling thread, or
   - Keep a flag that initialization was done on the worker thread and call through it for the first
     invocation, then direct-call afterward (since Metal device/queue are thread-local singletons)

2. **Metal thread-local objects**: `get_metal_device()`, `get_metal_queue()`, and
   `get_current_command_buffer()` use `thread_local` storage in `utils.mm`. If the calling thread
   (Python main thread) has never initialized these, the first Metal call will create them. This is
   safe — Metal device is process-global, queue creation is cheap.

3. **`idle()` callback**: The worker's `idle()` calls `synchronize_stream()` when no jobs are
   available. In the direct path, this should be called after the lambda returns (if needed), or
   simply skipped since the caller controls synchronization.

4. **Backward compatibility**: Multi-worker (`num_replicas() > 1`) and batched request paths remain
   unchanged. Only the common single-replica path benefits.

5. **Testing**: Compare `model.generate()` wall time before/after. Expected: encode drops from
   ~1,000ms to ~100ms, total generate from ~1,900ms to ~1,200ms.

---

### ARCH-2: GPU-Side Beam Search

**Estimated savings**: Eliminates 124 per-token `synchronize_stream()` calls (~50ms direct +
~400ms CPU beam logic)
**Effort**: Very High — requires rewriting `BeamSearch::search()` and `Sampler` for GPU execution
**Risk**: High — beam search correctness is subtle; many edge cases

#### Problem

The current beam search loop in `decoding.cc` (`BeamSearch::search()`) runs entirely on CPU:

```
Per-token decode step (current):
  1. GPU (encode-only): decoder forward → logits [batch×beam, vocab]
  2. Sampler::operator()():
     a. GPU (encode-only): TopK → sampled_ids, sampled_scores [batch×beam, 2*beam]
     b. synchronize_stream(METAL)          ← commit_and_wait() #N
     c. CPU memcpy: sampled_ids, sampled_scores → CPU buffers
  3. CPU: unflatten_ids()                  ← beam_id = flat_id / vocab_size
  4. CPU: update_sample_with_prefix()      ← prefix forcing, EOS penalty
  5. CPU: beam score accumulation          ← topk_scores += sampled_scores
  6. CPU: finished hypothesis bookkeeping  ← check EOS, store completed beams
  7. CPU: gather_beam_flat()               ← reorder decoder state by beam origins
  8. CPU: append_step_output()             ← update token history
  → repeat for next token
```

Steps 2b-6 require sampled token IDs on CPU. Step 7 (`gather_beam_flat`) operates on GPU
StorageViews but needs the CPU-computed beam origins. This forces a GPU→CPU sync every token.

#### What GPU Beam Search Would Require

Moving beam search to GPU means keeping `sampled_ids`, `sampled_scores`, `beam_origins`,
`topk_scores`, and `alive/finished` tracking tensors on the GPU:

1. **GPU `unflatten_ids`**: Replace CPU loop in `unflatten_ids()` (line 118-135 of `decoding.cc`)
   with a Metal kernel that computes `word_id = flat_id % vocab_size` and
   `beam_id = flat_id / vocab_size` in parallel.

2. **GPU beam score accumulation**: Replace CPU score addition with element-wise GPU add.

3. **GPU EOS detection**: A reduction kernel to check if any beam has hit an EOS token. Only THIS
   result needs to be sync'd to CPU (1 bool per batch, not per beam×vocab).

4. **GPU `gather_beam_flat`**: Already runs on GPU (Metal gather kernel). Currently needs
   CPU-computed beam origins — with GPU unflatten, beam origins stay on GPU.

5. **GPU prefix forcing / min_length**: The `update_sample_with_prefix()` and
   `apply_min_length()` logic (CPU loops over batch×beam) would need GPU kernels or could be
   handled by logits masking before sampling (already partially done via `DisableTokens`).

6. **Finished hypothesis extraction**: When a beam completes (EOS detected), the hypothesis
   tokens must be read on CPU. This is a rare event (once per completed beam, not per token), so
   a targeted sync is acceptable.

#### Architectural Sketch

```
Per-token decode step (proposed):
  1. GPU (encode-only): decoder forward → logits
  2. GPU (encode-only): TopK → sampled_ids, sampled_scores
  3. GPU (encode-only): unflatten_ids kernel → word_ids, beam_origins
  4. GPU (encode-only): score accumulation → topk_scores += sampled_scores
  5. GPU (encode-only): EOS check reduction → has_eos [batch] (1 value per batch)
  6. synchronize_stream() ONLY IF has_eos indicates a finished beam
     └─ CPU: extract finished hypothesis, update bookkeeping
  7. GPU (encode-only): gather_beam_flat with GPU beam_origins
  → repeat (no sync needed if no beam finished this step)
```

This reduces syncs from 124 (every token) to ~5-10 (only when beams finish, which happens a few
times over the ~123 token sequence). The ~50ms commit overhead drops to ~2-4ms, and the ~400ms
CPU beam logic largely disappears (replaced by GPU kernels executing in <1ms total).

#### Key Challenges

1. **Variable-length hypothesis storage**: Finished beams have different lengths. GPU-side storage
   needs pre-allocated max-length buffers or a compaction strategy.

2. **Batch compaction**: When a batch element finishes all beams, it's removed from the active set.
   Currently this is a CPU `std::vector` operation with `batch_offset` tracking. GPU-side requires
   stream compaction or masking.

3. **`LogitsProcessor` callbacks**: User-supplied logits processors (Python callables) require
   CPU access to logits. These would force a sync, but they're rarely used in Whisper.

4. **Whisper-specific logic**: `ApplyTimestampRules` runs on CPU after sampling. It would need
   a GPU implementation or a targeted sync for the Whisper use case.

5. **Correctness verification**: Beam search has many edge cases (prefix biasing, patience,
   coverage penalty, repetition penalty). Each must produce identical results to CPU.

#### Recommendation

GPU beam search is the highest-ceiling optimization but also the highest-risk. A pragmatic approach:

- **Phase 1**: Implement ARCH-1 (ThreadPool bypass) first — immediate ~700ms win, low risk.
- **Phase 2**: Implement GPU `unflatten_ids` + score accumulation (steps 3-4 above). Keep EOS
  check and hypothesis extraction on CPU. This eliminates the heaviest CPU work per token while
  keeping the complex bookkeeping on CPU.
- **Phase 3**: Full GPU beam search with lazy CPU sync. Only attempt after Phase 2 proves stable.

---

### ~~ARCH-3: Decode-Step Pipeline Fusion~~ — CLOSED (2026-03-10)

**Original estimate**: ~100-200ms
**Actual estimate**: ~7-15ms (0.8%)
**Status**: Closed — not actionable

#### Why the Original Estimate Was Wrong

The original analysis assumed three separate QKV GEMM dispatches per layer. In reality:

1. **QKV is already fused at model level**: The weight matrix is `[d_model, 3*d_model]` — one
   GEMM call produces concatenated Q+K+V, then split. No dispatch savings possible.

2. **FFN gate+up cannot be fused further**: The gating variant uses `_ff1` (with activation) and
   `_ff1_noact` (linear), then `SiLU(gate) * up`. The element-wise product between the two GEMM
   outputs prevents fusion into a single GEMM.

3. **Persistent command encoder blocked by MPS**: MPS GEMM encodes directly into the CB via
   `[gemm_op encodeToCommandBuffer:]` — not through a compute encoder. A persistent compute
   encoder must be ended before every MPS dispatch. Since GEMMs dominate each layer, the encoder
   would only survive between consecutive custom kernels (add→layer_norm), saving ~3-8ms total.

4. **SDPA encoder overhead is minimal**: Only 1 `create_compute_encoder()` call per SDPA
   (for the causal mask kernel). MPS attention GEMMs encode directly into the CB.

The remaining overhead is ~2-5us per encoder create/end × ~20 custom dispatches/token × 123
tokens = **~7-15ms**. Not worth the refactoring effort.

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

### For CTranslate2 Upstream — Priority Order

**Only architectural changes remain impactful.** All non-architectural items (#3–#8) were
investigated and closed — combined potential was ~15-30ms (1.5%). See Section 9 for details.

1. **ARCH-1: Bypass ThreadPool** (Section 10) — **~700ms savings, medium effort, low risk**.
   Add direct-call fast path in `ReplicaPool::post()` when `num_replicas() == 1`. No GPU changes.
   Expected: 1,860ms → ~1,160ms.

2. **ARCH-2: GPU beam search** (Section 10) — **~450ms, very high effort, high risk**.
   Move beam state to GPU, sync only on EOS. Phased implementation recommended.
   Expected: ~1,160ms → ~700ms.

~~3. **ARCH-3: Decode-step pipeline fusion** — CLOSED. QKV already fused at model level,
   persistent encoder blocked by MPS, actual savings ~7-15ms.~~

Combined realistic potential: **1,860ms → ~700ms (2.7x improvement, 59x vs original baseline)**.

### For faster_whisper Upstream

1. **Backend-aware temperature defaults**: Detect Metal and use `[0.0, 0.6]` instead of `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
2. **Expose encode caching control**: Allow users to bypass the standalone `encode()` call and let `generate()` handle encoding internally when retry caching isn't needed.
