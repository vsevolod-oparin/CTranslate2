# M12.16 — Object Pooling Temporaries: REJECTED

**Date**: 2026-03-11
**Status**: REJECTED — no measurable performance gain; all code reverted

---

## Motivation

The decode loop creates and destroys several `StorageView` temporaries every step:
- **Sampler GPU buffers** (`sampling.cc:20-21`): `sampled_ids_device`, `sampled_scores_device` — GPU-allocated via Metal bucketed allocator
- **`gather_indices`** (`decoding.cc:677`): CPU INT32, size = batch × beam × candidates
- **`active_beams`** (`decoding.cc:717`): CPU INT32, size = batch × beam
- **`non_finished_index`** (`decoding.cc:713`): `std::vector<int32_t>`, up to batch_size

Each alloc/free cycles through the Metal bucketed allocator (GPU buffers) or system malloc (CPU buffers).

---

## Implementation

### 1. Sampler GPU Buffer Pooling
Added `mutable StorageView _ids_device_buf, _scores_device_buf` to the `Sampler` base class. These persist across `operator()` calls, avoiding per-step GPU alloc/free. The buffers are lazily initialized on first use and reused via `resize()` (which is a no-op when the size matches).

### 2. Decode Loop Temporary Hoisting
Hoisted three temporaries before the main decode loop:
- `gather_indices` — CPU INT32 buffer, reused by refactored `unflatten_ids()` (changed from return-by-value to output parameter)
- `gather_indices_device` — GPU INT32 buffer for the `.to(device)` copy needed by `decoder.update_state()`
- `active_beams` — CPU INT32 buffer, resized per step
- `non_finished_index` — `std::vector<int32_t>`, reserved once, `.clear()` per step

### Bug Found During Implementation
The original code does `gather_indices = gather_indices.to(device)` which changes the StorageView from CPU to MPS. With hoisted pooling, this would corrupt the CPU buffer for the next iteration. Fixed by using a separate `gather_indices_device` buffer for the GPU copy.

---

## Correctness
- 90/90 translation tests: **PASS**
- 10/10 INT8 tests: **PASS**
- 39/39 beam search tests: **PASS**

---

## Benchmark Results (50 sentences, beam=4, best-of-3)

| Type | M12.12 baseline (tok/s) | M12.16 pooled (tok/s) | Delta |
|------|------------------------|----------------------|-------|
| float16 | 1495 | 1476 | -1.3% |
| float32 | 1026 | 1039 | +1.3% |
| int8 | 769 | 754 | -2.0% |
| int8_float16 | 870 | 845 | -2.9% |
| bfloat16 | 1457 | 1452 | -0.3% |
| int8_bfloat16 | 844 | 841 | -0.4% |

All results within ±3% noise — **no measurable improvement**.

---

## Why No Improvement

### 1. CPU overhead is 0.4% of total decode time (M12.4 finding)
The decode loop spends 99.1% of time waiting for GPU compute. Allocator overhead is a fraction of the 0.4% CPU time.

### 2. Metal bucketed allocator already provides O(1) pool operations
The allocator uses power-of-2 size-class bucketing with ~49% pointer cache hit rate (M12.12). A `free()` pushes to pool; an `allocate()` pops from pool — both O(1). Eliminating these calls saves single-digit microseconds per step.

### 3. Estimated savings vs measured noise
Per step: ~5 alloc/free pairs × ~1 µs each = ~5 µs saved.
Per benchmark: ~90 steps × 5 µs = **~0.45 ms total savings**.
Total runtime: 1000-2000 ms → savings are **0.02-0.05%**, far below the ±3% measurement noise.

### 4. std::vector and CPU StorageView allocations are essentially free
On modern allocators (tcmalloc/jemalloc/system), small CPU allocations for temporaries under 1KB have negligible overhead (< 100ns each).

---

## Decision

**REJECTED** — the Metal bucketed allocator already handles buffer reuse efficiently. Object pooling adds code complexity (mutable members, output parameters, separate GPU copies) for <0.05% theoretical savings. All code reverted.

---

## Broader Implication

This result, combined with M12.14 (decode bookkeeping) and M12.15 (BiasAdd fusion), confirms that **CPU-side optimizations in the decode loop are exhausted**. The M12.4 profiling finding that 99.1% of time is GPU compute leaves no room for CPU-side improvements at this scale. Further performance gains require:
1. Faster GPU kernels (custom attention, MPS algorithm improvements)
2. Larger models where GPU utilization naturally increases
3. System-level changes (async prefill, continuous batching)
