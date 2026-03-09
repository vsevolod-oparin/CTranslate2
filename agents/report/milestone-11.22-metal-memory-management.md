# M11.22 — Metal Memory Management

## Problem

Running benchmarks caused system-wide memory pressure on macOS — browser tabs offloading, terminal blanking, cursor hanging. Root cause: two issues compounding.

### Issue 1: Metal allocator pool never cleared

The Metal allocator (`MetalAllocator` in `allocator.mm`) caches all freed `MTLBuffer` objects in `_pool` for reuse. The `clear_cache()` method exists but was **never called for Metal** — the `unload_model()` path in `replica_pool.h` only cleared CUDA cache:

```cpp
// Before: CUDA-only
if (_device == Device::CUDA)
    _pool->clear_cache();
```

On Apple Silicon, `MTLResourceStorageModeShared` means GPU buffers ARE system RAM. Unlike discrete GPUs with separate VRAM, pooled Metal buffers directly consume system memory and are never released.

### Issue 2: Benchmark loaded two model copies simultaneously

`bench_faster_whisper.py` and `test_faster_whisper.py` loaded both CPU and Metal models at the same time:

```python
model_cpu = WhisperModel(whisper_path, device="cpu")      # ~3 GB
model_metal = WhisperModel(whisper_path, device="metal")   # ~3 GB
# Peak: ~8 GB for whisper-large-v3-turbo alone
```

On a 16 GB machine with ~8 GB available, this alone triggers memory pressure.

## Changes

### C++ / Python API

| File | Change |
|------|--------|
| `python/cpp/module.cc` | Added `clear_device_cache(device)` Python binding via `get_allocator(d).clear_cache()` |
| `python/cpp/replica_pool.h` | `unload_model()` now clears Metal cache (parity with CUDA) |
| `python/ctranslate2/__init__.py` | Export `clear_device_cache` from `_ext` |
| `src/metal/allocator.mm` | Added `pool_bytes()` and `live_bytes()` introspection methods |
| `src/metal/utils.h` | Declarations for `pool_bytes()` / `live_bytes()` |

### Test / Benchmark files

| File | Change |
|------|--------|
| `bench_faster_whisper.py` | Sequential model loading: CPU first → delete → Metal → delete → clear_cache |
| `test_faster_whisper.py` | Same pattern: run all CPU tests → delete → run all Metal tests → delete → clear_cache |
| `test_whisper.py` | Same pattern: collect all CPU results → delete → collect all Metal results → compare |

## Memory Impact (whisper-large-v3-turbo)

| Metric | Before | After |
|--------|--------|-------|
| Peak RSS during benchmark | ~8 GB (2× model) | ~4 GB (1× model) |
| Allocator pool after unload | Never freed | Freed via `clear_cache()` |
| `clear_device_cache('metal')` API | Did not exist | Available |

Verified with `ps -o rss=` (current RSS, not peak):
- Model load: 260 → 4447 MB (+4187 MB)
- After `unload_model(to_cpu=True)` + `load_model()` back: RSS drops to 1469 MB
- macOS DOES reclaim pages when MTLBuffers are released via ARC

## Test Results

| Test | Result |
|------|--------|
| `test_whisper.py` | 13/13 PASS, 3.64x speedup (whisper-base) |
| `test_faster_whisper.py` | 8/8 PASS |
| `test_beam_search.py` | 39/39 PASS |
| `test_translation.py` | 90/90 PASS |
| **Total** | **150/150 PASS** |

## Not Changed

Small-model tests (`test_generator.py`, `test_longform_generation.py`, `test_translation.py`, `test_beam_search.py`) still load both CPU and Metal models simultaneously. These use opus-mt-en-de (~100 MB) or GPT-2 (~500 MB) — not worth the refactoring complexity for minimal memory savings.
