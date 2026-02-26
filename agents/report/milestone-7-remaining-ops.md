# Milestone 7 — Remaining Ops (Complete Coverage)

**Date:** 2026-02-26
**Status:** ✅ DONE

---

## Summary

M7 adds Metal implementations for all CPU-only ops needed for full model support.
Eight new `*_metal.mm` files provide `Device::METAL` specializations for:
`Concat`, `Split`, `Slide`, `Tile`, `TopK`, `TopPMask`, `GumbelMax`, `Multinomial`,
`Mean`, and `MedianFilter`.

**Test result: 20/20 tests pass** (`tests/metal/m7_test.mm`)

---

## Implementation Strategy

### Design decision: commit_and_wait + CPU algorithm on shared memory

All M7 ops use the "deferred-commit + shared-memory CPU" pattern:

```
1. metal::commit_and_wait()   — flush any pending GPU writes visible to CPU
2. CPU algorithm               — operate on Metal-backed shared-memory pointers
3. (implicit)                 — output is instantly visible to the next GPU op
```

This is correct for Apple Silicon: `MTLResourceStorageModeShared` buffers are
simultaneously accessible by both CPU and GPU. No device copy is needed.

**Why not GPU kernels?** These ops are either memory-bound (concat/split/tile),
or require complex sorting/random-number generation (topk/topp_mask/multinomial).
The CPU path via shared memory is correct and immediately functional. GPU-kernel
optimizations are deferred to a future milestone.

**Latency cost:** Each op call that reads GPU-produced data adds one command-buffer
submission (~0.4 ms on M4). In a pipelined model, consecutive GPU ops still batch
efficiently; the flush only occurs at the CPU boundary.

### Ops grouped by category

| Category | Ops | Strategy |
|----------|-----|----------|
| Memory copy | Concat, Split, Slide, Tile | commit_and_wait + std::memcpy loop |
| Sorting/sampling | TopK, TopPMask | commit_and_wait + std::partial_sort |
| Stochastic sampling | GumbelMax, Multinomial | commit_and_wait + CPU RNG |
| Scalar reduction | Mean | commit_and_wait + float32 sum loop |
| Window filter | MedianFilter | commit_and_wait + std::nth_element |

---

## Files Created

### Metal op specializations (`src/ops/`)

| File | Ops | Types |
|------|-----|-------|
| `concat_split_slide_metal.mm` | `Concat::compute`, `Split::compute`, `Slide::compute` | All 6 (DECLARE_ALL_TYPES) |
| `tile_metal.mm` | `Tile::compute` | All 6 (DECLARE_ALL_TYPES) |
| `topk_metal.mm` | `TopK::compute<D,T,int32_t>` | float, float16_t, bfloat16_t |
| `topp_mask_metal.mm` | `TopPMask::compute` + `max_num_classes<METAL>` | float |
| `gumbel_max_metal.mm` | `GumbelMax::add_gumbel_noise` | float |
| `multinomial_metal.mm` | `Multinomial::compute` | float |
| `mean_metal.mm` | `Mean::compute` | float |
| `median_filter_metal.mm` | `MedianFilter::compute` | float |

### Test file (`tests/metal/`)

`m7_test.mm` — 20 tests covering all M7 ops plus memory coherency validation.

### CMakeLists.txt

METAL_SOURCES extended with the 8 new `.mm` files (lines 281–288).

---

## Key Design Details

### Concat / Split / Slide

Same algorithm as `concat_split_slide_cpu.cc` but:
- No `cpu::parallel_for` (unnecessary; algorithm is sequential std::memcpy)
- Helper functions redefined locally (`concat_copy_size`, `concat_iter_size`)
- `commit_and_wait()` before any pointer access

`Concat::compute`: iterates inputs; for each input, memcpy `copy_size` elements
into each `iter_size` stride of the output.

`Split::compute`: same algorithm inverted — reads from input strides into outputs.

`Slide::compute`: single contiguous region copy at `index * stride_axis` offset.

### Tile

Same loop as `tile_cpu.cc` (already generic — no `cpu::parallel_for`):
`outer_size × num_tiles` memcpy calls of `inner_size` elements each.

### TopK

- **k=1**: `std::max_element` with explicit float32 comparator (works for bf16 which lacks `operator<`)
- **k>1**: `std::iota` + `std::partial_sort` on a local `std::vector<IndexType>`, reused across batch rows.

Explicit float32 cast in all comparisons ensures correctness for `bfloat16_t`
(which has `operator float()` but no comparison operators).

### TopPMask

Sorts class indices by probability (descending), accumulates cumulative probability,
masks all classes once cumulative prob ≥ p. `max_num_classes<Device::METAL>` returns
`std::numeric_limits<dim_t>::max()` (same as CPU — no artificial limit).

### GumbelMax

Same as CPU: `z = -log(U(epsilon, 1))` where U is `std::uniform_real_distribution`
using the process-wide `get_random_generator()`. Noise applied as float32 then cast
back to T.

### Multinomial

Builds `std::vector<float>` weights from the input row (needed since
`std::discrete_distribution` requires double-convertible iterators), then samples
`_sample_size` integers per batch row. Output stored directly in the Metal-backed
`int32_t` output buffer.

### Mean

Three nested loops: outer × inner × axis_size. Accumulates in float32, divides
by axis_size unless `get_sum=true`. Float-only (matches CPU limitation).

### MedianFilter

Sliding window of width `_width` with reflect-at-boundary padding.
Uses `std::nth_element` for O(width) median without full sort.
Float-only. Uses a `std::vector<float>` window (re-allocated once, not per row,
in the actual implementation — hot path optimization is deferred).

---

## Test Coverage (20/20 pass)

| Test | Description |
|------|-------------|
| 1 | Memory coherency: GPU add → commit_and_wait → CPU read |
| 2 | Concat axis=0 float32: [6]+[4]=[10] elements |
| 3 | Concat axis=1 float32: [2×2]+[2×3]=[2×5] (per-row interleave) |
| 4 | Split [4×4] axis=0: first half [0..7] correct |
| 5 | Split [4×4] axis=0: second half [8..15] correct |
| 6 | Slide axis=0 index=1: extracts correct row from [3×4] |
| 7 | Tile 1×5 × 3 = 3×5 along axis=0 (float32) |
| 8 | Tile int32 [3×2] × 2 = [3×4] along axis=1 |
| 9 | TopK k=1 float32: argmax correct for 3 rows |
| 10 | TopK k=3 float32: correct top-3 values and indices |
| 11 | TopK k=1 float16: argmax at correct index |
| 12 | TopK k=1 bfloat16: argmax at correct index |
| 13 | TopPMask p=0.75: top-3 kept, remaining masked |
| 14 | Mean last axis [2×4] → [2]: correct means 2.5, 6.5 |
| 15 | Mean axis=0 [4×2] → [2] (inner_size=2): col means 4.0, 5.0 |
| 16 | MedianFilter width=3: correct reflect-padded median values |
| 17 | GumbelMax noise: z = -log(U) > 0 always |
| 18 | GumbelMax: mean noise in expected range [0.3, 1.5] |
| 19 | Multinomial: all samples in valid class range |
| 20 | Multinomial: all classes appear in 100 uniform samples |

---

## Build Command (test)

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m7_test.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm \
    src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm \
    src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm \
    src/metal/primitives_beam_search.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    src/random.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m7_test && ./m7_test
```

---

## Performance Notes

All M7 ops are CPU-side (via `commit_and_wait()` + shared-memory access):
- **Concat/Split/Slide/Tile**: Bandwidth-limited memcpy. At typical transformer sizes
  (multi-head split of [batch, seq, dim]) the CPU memcpy cost is <0.1 ms, comparable
  to a GPU blit at these sizes. GPU blit optimization deferred to M8+.
- **TopK**: CPU sort with O(N log k) complexity. For k=1 (greedy decoding), reduces
  to linear scan. For beam search (k≤5), negligible at typical vocab sizes (≤65536).
- **GumbelMax/Multinomial**: CPU-bound by random number generation. Acceptable for
  stochastic sampling (not in the critical decode path).
- **Mean/MedianFilter**: CPU-bound. Both are infrequent ops in transformer models.

---

## Benchmark (`tests/metal/m7_bench.mm`, observed on Apple M4)

Both paths use Metal-allocated I/O buffers. "Metal" = commit_and_wait() + algorithm.
"CPU" = same algorithm without sync. `commit_and_wait()` with no pending GPU work: **~0 µs**.
In a real pipeline (GPU busy before the op), commit adds ~0.4 ms.

| Op | Shape | Metal µs | CPU µs | Ratio |
|----|-------|----------|--------|-------|
| **commit baseline** | empty (no GPU work) | 0 | — | — |
| **Concat axis=0** | [256×512]+[256×512] | 31 | 24 | 0.77x |
| **Concat axis=0** | [1024×512]+[1024×512] | 79 | 77 | 0.97x |
| **Concat axis=0** | [1024×2048]+[1024×2048] | 344 | 314 | 0.91x |
| **Concat axis=1** | [1024×512]+[1024×1024] | 95 | 94 | 0.99x |
| **Split axis=0** | [512×512]→2×[256×512] | 14 | 14 | 1.00x |
| **Split axis=0** | [2048×512]→2×[1024×512] | 63 | 60 | 0.95x |
| **Tile** | [256×1024]×4→[1024×1024] | 57 | 57 | 1.00x |
| **Tile** | [1024×2048]×4→[4096×2048] | 489 | 436 | 0.89x |
| **TopK k=1** | batch=1, vocab=32768 | 15 | 15 | 1.00x |
| **TopK k=1** | batch=1, vocab=100352 | 46 | 45 | 1.00x |
| **TopK k=5** | batch=1, vocab=32768 | 14 | 14 | 1.00x |
| **TopK k=10** | batch=1, vocab=65536 | 27 | 27 | 1.02x |
| **TopPMask** | p=0.90, vocab=32768 | 1267 | 1251 | 0.99x |
| **TopPMask** | p=0.95, vocab=100352 | 5033 | 4974 | 0.99x |
| **Mean** | [256×1024]→[256] | 106 | 106 | 1.00x |
| **Mean** | [256×512×128]→[256×128] | 9589 | 9631 | 1.00x |
| **MedianFilter** | width=3, [80×1500] | 820 | 856 | 1.04x |
| **MedianFilter** | width=5, [80×3000] | 6052 | 6056 | 1.00x |
| **GumbelMax** | vocab=32768 | 153 | 152 | 1.00x |
| **GumbelMax** | vocab=100352 | 466 | 466 | 1.00x |
| **Multinomial** | vocab=32768, samples=1 | 46 | 46 | 1.00x |

**Key finding**: Metal ≈ CPU (1.00x) for all ops because:
1. `commit_and_wait()` with no pending GPU work is essentially free (~0 µs).
2. The algorithm runs on the CPU in both paths (same Metal-shared memory).
3. In a real pipeline the commit cost (~0.4 ms) would add latency only at the GPU→CPU boundary.

**Most expensive ops**: TopPMask (std::sort O(N log N)), Mean mid-axis (strided access ~9.6 ms for 33M elements), MedianFilter (std::nth_element per position).

### Benchmark build command

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/m7_bench.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o m7_bench && ./m7_bench
```

---

## Deferred Items

| Item | Notes |
|------|-------|
| `Conv1d` | Whisper encoder; deferred to M8 (requires `MPSCNNConvolution`) |
| `Quantize` / `Dequantize` | Deferred to M9 (INT8 milestone) |
| GPU kernels for concat/tile | Blit encoder optimization — deferred to M8+ |
| Parallel TopK for large vocab | `std::partial_sort` is sequential — defer to M8+ |
