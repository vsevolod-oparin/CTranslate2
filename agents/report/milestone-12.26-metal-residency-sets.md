# M12.26 — Metal Residency Sets

**Date**: 2026-03-12
**Commit**: a4046a63
**Status**: Implemented, benchmarked, no regression

---

## Summary

Metal Residency Sets (macOS 15+) pin model weight MTLBuffers in physical memory so the OS cannot evict them under memory pressure. This is a proactive defense against performance degradation when running large models alongside other memory-intensive applications.

---

## Motivation

On Apple Silicon, `MTLResourceStorageModeShared` buffers live in unified memory (DRAM). The OS may evict GPU-referenced pages under memory pressure, causing page faults during inference that stall the GPU pipeline. For large models (whisper-large-v3 at ~3 GB, or larger LLMs), eviction risk increases significantly when other applications compete for memory.

The `MTLResidencySet` API (macOS 15+) provides a mechanism to **request** that the OS keep specific allocations resident in physical memory. This is a hint — the OS may still evict under extreme pressure — but it significantly raises the threshold before eviction occurs.

---

## Implementation

### API Surface

```cpp
// src/metal/utils.h
void request_residency();  // Pin all live buffers
void end_residency();      // Release the pin
```

### Core Logic (`src/metal/allocator.mm`)

**`request_residency()`**:
1. Acquires allocator mutex
2. Releases any prior residency set (`end_residency_locked()`)
3. If no live allocations, returns early
4. Creates `MTLResidencySetDescriptor` with label `"ct2_model_weights"`
5. Calls `[device newResidencySetWithDescriptor:desc error:]`
6. Iterates all entries in `_live` map, calling `[set addAllocation:entry.buffer]`
7. Commits and requests residency: `[set commit]` then `[set requestResidency]`

**`end_residency()`**:
1. Acquires allocator mutex
2. Calls `[set endResidency]` then `[set release]`, sets to nil

**Availability guard**: All code wrapped in `@available(macOS 15.0, *)` — complete no-op on older macOS.

**Error handling**: If `newResidencySetWithDescriptor` returns nil, silently returns. Residency is a performance hint, not a correctness requirement.

### Lifecycle Integration

| Event | Action |
|-------|--------|
| `Model::load()` completes `set_device(MPS)` | `metal::request_residency()` called automatically |
| `clear_device_cache("mps")` / `unload_model()` | `end_residency()` called before releasing pool |
| `~MetalAllocator()` destructor | `end_residency_locked()` before buffer cleanup |

The call in `model.cc` (line ~769) is placed immediately after `model->set_device(device, device_index)`, which ensures all weight buffers have been transferred to Metal before pinning.

```cpp
// src/models/model.cc — after set_device()
#ifdef CT2_WITH_METAL
  if (device == Device::MPS)
    metal::request_residency();
#endif
```

### Thread Safety

Both public methods acquire `_mutex` before accessing the `_live` map or `_residency_set`. The private `end_residency_locked()` helper assumes the caller already holds the lock (used from destructor and `clear_cache()`).

### Member State

```objc
id<MTLResidencySet> _residency_set = nil;  // One set per allocator lifetime
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/metal/allocator.mm` | `request_residency()`, `end_residency()`, `end_residency_locked()`, member variable, destructor/clear_cache integration |
| `src/metal/utils.h` | Declarations for `request_residency()` and `end_residency()` |
| `src/models/model.cc` | Auto-trigger after `set_device(MPS)` in `Model::load()` |

---

## Benchmark Results

### OPUS-MT En→De (50 sentences, beam=4) — Small Model (~100 MB)

| Type | M12.25 tok/s | M12.26 tok/s | Change |
|------|-------------|-------------|--------|
| float16 | 1462 | 1478 | within noise |
| float32 | 1032 | 1044 | within noise |
| bfloat16 | 1461 | 1505 | within noise |
| int8 | 773 | 806 | within noise |
| int8_float16 | 901 | 863 | within noise |
| int8_bfloat16 | 899 | 897 | within noise |

**Conclusion**: No measurable impact on small models. Expected — OPUS-MT fits comfortably in memory with no eviction risk.

### Whisper Large-v3-Turbo (30s audio, beam=5) — Medium Model (~800 MB)

| Metric | M11.29 (commit 51) | M12.26 (commit 52) | Change |
|--------|--------------------|--------------------|--------|
| Mean (ms) | 1,860 | 1,812 | −2.6% (within noise) |
| Best (ms) | 1,855 | 1,807 | −2.6% |
| Speedup vs baseline | 22.13× | 22.72× | slightly faster |
| Variance (CV) | 0.3% | 0.3% | same |

**Conclusion**: No regression. Turbo model also fits comfortably. The slight improvement is within run-to-run noise.

### Expected Impact (Not Yet Measured)

Residency sets should provide measurable benefit when:
- **Large models** (whisper-large-v3 at ~3 GB, LLMs at 4–8+ GB) fill a significant fraction of system memory
- **Concurrent workloads** compete for memory (e.g., running inference alongside a web browser or other GPU apps)
- **Long-running inference** where the OS has time to build memory pressure and start evicting pages
- **Batch processing** large datasets where RSS grows from intermediate tensor allocations

Without residency pinning, the OS may evict weight pages, causing ~10–100 µs page faults per evicted page on next GPU access. With pinning, weights stay resident and GPU reads hit physical memory directly.

---

## Design Decisions

### Why pin at model load, not lazily?

Model weights are allocated once and read every inference step. They are the most critical buffers to keep resident. Intermediate tensors are short-lived and managed by the bucketed allocator's reuse pool — pinning them would be wasteful.

### Why one residency set for all buffers?

The `MTLResidencySet` API supports adding many allocations to a single set. One set per model load is simpler than per-buffer tracking and matches the semantic intent: "keep this model in memory."

### Why release on `clear_cache()`?

When the user calls `clear_device_cache("mps")` or `unload_model()`, they're explicitly requesting memory release. Residency must end before buffers return to the pool, otherwise the OS would continue pinning freed buffers.

### Why not pin pool buffers?

Pool buffers are cached but unused. Pinning them would waste physical memory pages on buffers that may never be reused. Only `_live` (actively used) buffers are pinned.

---

## Compatibility

| macOS Version | Behavior |
|---------------|----------|
| 15.0+ (Sequoia) | Full residency set support |
| 14.x and earlier | Complete no-op (`@available` guard) |
| Non-Apple platforms | Not compiled (`#ifdef CT2_WITH_METAL`) |

---

## Test Coverage

No dedicated residency tests. Exercised implicitly through:
- All existing Metal e2e tests (model load triggers `request_residency()`)
- OPUS-MT and whisper benchmark runs confirm no regression
- Manual verification: model loads without errors on macOS 15

A targeted stress test (loading a model under artificial memory pressure) would be valuable future work to validate the eviction-prevention benefit.
