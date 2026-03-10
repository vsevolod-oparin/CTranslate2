# M11.27 — MPSMatrixMultiplication Cache

## Summary

Cached `MPSMatrixMultiplication` objects across GEMM calls, eliminating per-call alloc+init overhead (kernel selection, ~10-20us per object). All 4 allocation sites now use a shared cache keyed by `(transpose_a, transpose_b, m, n, k, alpha, beta, batch_size)`. Objects are retained for the process lifetime.

## Problem

Each GEMM dispatch created 7 ObjC objects:
- 3 `MPSMatrixDescriptor` (class factory, autoreleased — lightweight)
- 3 `MPSMatrix` (wraps buffer+descriptor — must be per-call since buffers change)
- 1 `MPSMatrixMultiplication` (expensive: performs kernel selection internally)

With 76K+ GEMM dispatches per whisper inference (measured via PSO hit counter), the `MPSMatrixMultiplication` alloc+init cost adds up. The object only depends on shape parameters `(transpose, m, n, k, alpha, beta)`, not on buffer pointers — it can be reused across encodes.

### Four Allocation Sites

| Site | Function | Line | Use Case |
|------|----------|------|----------|
| 1 | `dispatch_mps_gemm` | ~249 | Single non-batched GEMM (f32/f16) |
| 2 | `dispatch_mps_gemm_buf` | ~471 | INT8 helper (takes raw MTLBuffers) |
| 3 | `dispatch_mps_gemm_batched` | ~657 | Batched GEMM (no padding) |
| 4 | `dispatch_mps_gemm_batched_padded` | ~1050 | Batched GEMM with GPU row_copy padding |

All 4 followed the same pattern:
```objc
MPSMatrixMultiplication* gemm_op =
    [[MPSMatrixMultiplication alloc] initWithDevice:dev
        transposeLeft:... transposeRight:...
        resultRows:m resultColumns:n interiorColumns:k
        alpha:alpha beta:beta];
// For batched: gemm_op.batchSize = batch_size;
[gemm_op encodeToCommandBuffer:cmd leftMatrix:... rightMatrix:... resultMatrix:...];
[gemm_op release];
```

### Prior Art: BF16 Graph Cache

BF16 GEMM already cached `MPSGraph` instances via `struct Bf16GemmEntry` keyed by `(trans_a x trans_b)` — 4 static entries. This established the pattern of caching MPS objects for reuse.

## Solution

### Cache Design

```cpp
struct MpsGemmKey {
  bool transpose_a, transpose_b;
  NSUInteger m, n, k;
  uint64_t alpha_bits, beta_bits;  // bit-exact double comparison
  NSUInteger batch_size;           // 0 for non-batched
};

static MPSMatrixMultiplication* get_cached_mps_gemm(
    bool transpose_a, bool transpose_b,
    NSUInteger m, NSUInteger n, NSUInteger k,
    double alpha, double beta,
    NSUInteger batch_size = 0);
```

- **Key**: `(transpose_a, transpose_b, m, n, k, alpha_bits, beta_bits, batch_size)` — 8 fields
- **Hash**: FNV-1a inspired mixing for good distribution
- **Storage**: `std::unordered_map<MpsGemmKey, MPSMatrixMultiplication*, MpsGemmKeyHash>`
- **Thread safety**: `std::mutex` (same pattern as `PSOCache`)
- **Lifetime**: Objects retained by cache, never released (process-lifetime singleton)
- **batch_size in key**: Avoids mutation race if multiple threads encode with different batch sizes

### Why batch_size Is in the Key

`MPSMatrixMultiplication.batchSize` is a mutable property. If two threads share the cache and set different batchSizes between lookup and encode, the wrong batchSize could be used. Including it in the key gives each (shape, batchSize) combination its own cached object.

### What's NOT Cached

- `MPSMatrixDescriptor`: Class factory method, autoreleased, lightweight (~1us)
- `MPSMatrix`: Wraps a specific `MTLBuffer` + offset — must be created per-call

### Changes Per Site

All 4 sites changed from:
```objc
MPSMatrixMultiplication* gemm_op = [[MPSMatrixMultiplication alloc] initWithDevice:...];
[gemm_op encodeToCommandBuffer:...];
[gemm_op release];
```
To:
```objc
MPSMatrixMultiplication* gemm_op =
    get_cached_mps_gemm(transpose_a, transpose_b, m, n, k, alpha, beta, batch_size);
[gemm_op encodeToCommandBuffer:...];
// No release — owned by cache.
```

## Files Modified

| File | Change |
|------|--------|
| `src/metal/primitives_gemm.mm` | Added `MpsGemmKey`, `MpsGemmKeyHash`, `get_cached_mps_gemm()` cache; updated 4 allocation sites to use cache |

## Test Results

All tests pass with correct output:

| Test | Result | Details |
|------|--------|---------|
| `test_whisper.py` (f32, whisper-base) | 13/13 PASS | WER 0.00%, exact match |
| `test_faster_whisper.py` (f16, whisper-large-v3-turbo) | 8/8 PASS | Output comparable to CPU |
| `test_beam_search.py` | 39/39 PASS | All beam sizes correct |
| `test_translation.py` | 90/90 PASS | All combinations |

### Performance

Speed benchmarks (Apple M4, 60s Russian audio, beam_size=5):

| Test | Run | CPU (ms) | Metal (ms) | Speedup |
|------|-----|----------|------------|---------|
| test_faster_whisper | 1 | 36086 | 12990 | 2.78x |
| test_faster_whisper | 2 | 35405 | 7409 | 4.78x |
| test_faster_whisper | 3 | - | 14927 | 1.66x |
| test_whisper (f32) | 1 | 6279 | 2268 | 2.77x |

#### Benchmark Variance Analysis

The `test_faster_whisper` speedup varies 1.66x–4.78x across runs. The variance is **not** in Metal GPU execution — it's in `faster_whisper`'s seek loop issuing different numbers of `model.generate()` calls per run.

**Raw CTranslate2 `generate()` is rock-solid:**
- whisper-base (f32): 618ms ± 7ms, CV=1.2%, max/min=1.04x over 10 iterations
- whisper-large-v3-turbo (f16): per-call timing proportional to tokens (e.g., 123 tokens ≈ 1000ms ± 50ms consistently)

**`faster_whisper.transcribe()` varies because of seek retries:**

| Run | generate() calls | Total tokens | Final text len | Total gen time |
|-----|-----------------|-------------|----------------|---------------|
| 0 | 7 | 391 | 350 | 4312ms |
| 1 | 7 | **861** | 350 | 9682ms |
| 2 | 7 | 490 | 350 | 7115ms |
| 3 | 7 | 476 | 350 | 7594ms |
| 4 | 7 | **728** | 350 | 8986ms |

Run 1 generates 2.2x more tokens than run 0 for identical final output. The extra tokens come from `generate()` calls hitting `max_length=224` (the cap), producing output that `faster_whisper` discards and retries via its seek logic.

**Root cause**: GPU floating-point non-determinism (thread scheduling order in reductions like softmax/layer_norm) produces slightly different logit scores across runs → different beam search paths → some `generate()` calls produce usable segments, others hit `max_length` and trigger seek retries. More retries = more wall time, despite identical final transcription.

## Why This Works

`MPSMatrixMultiplication` internally performs kernel selection during `initWithDevice:` — choosing the optimal GPU kernel for the given matrix dimensions and transpose configuration. This selection is deterministic for the same parameters, so caching avoids redundant work.

The `encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:` method only reads the buffer pointers from the `MPSMatrix` arguments and the pre-selected kernel from the `MPSMatrixMultiplication` object. The object itself is stateless with respect to the encode — it can be reused safely.

### Cache Size Estimate

In whisper inference, the number of unique GEMM shapes is bounded by:
- Model architecture: ~30 unique (m, n, k) shapes per layer type
- Beam search: beam_size affects m dimension
- Transpose combinations: at most 4 per shape

Typical cache: 50-200 entries, negligible memory (~16KB of ObjC objects).

## Interaction with Prior Milestones

| Milestone | Interaction |
|-----------|-------------|
| M11.22 (Memory management) | Cache objects are never released — no leak since they're process-lifetime singletons. ARC-off is preserved. |
| M11.26 (cblas elimination) | All padded GEMMs now go through MPS (zero cblas syncs). The cache benefits these additional MPS dispatches. |
| M11.25 (GPU indexed_fill) | No interaction — different code path. |

## Future Work

1. **MPSMatrix pooling**: The 3 `MPSMatrix` objects per dispatch are still allocated/released each call. A pool keyed by (buffer, offset, descriptor) could reduce this further, but the objects are lightweight.
2. **Cache statistics**: Add a hit/miss counter (gated by `CT2_METAL_TRACE`) to measure cache effectiveness in production workloads.
3. **LRU eviction**: For long-running servers with varying input shapes, an LRU policy could bound cache growth. Not needed for fixed-model inference.
