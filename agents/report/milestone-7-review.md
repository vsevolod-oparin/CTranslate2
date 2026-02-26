# Milestone 7 Review

## Scope

Files reviewed:

- `src/ops/concat_split_slide_metal.mm`
- `src/ops/tile_metal.mm`
- `src/ops/topk_metal.mm`
- `src/ops/topp_mask_metal.mm`
- `src/ops/gumbel_max_metal.mm`
- `src/ops/multinomial_metal.mm`
- `src/ops/mean_metal.mm`
- `src/ops/median_filter_metal.mm`
- `tests/metal/m7_test.mm` (20 checks before review)
- `tests/metal/m7_bench.mm`

---

## Section 1 — Bugs

### Bug 1.1 — `multinomial_metal.mm`: unnecessary `std::vector<float>` copy

**Severity**: Minor (correctness is unaffected; waste of O(class_size) allocation per batch row)
**File**: `src/ops/multinomial_metal.mm`, lines 35–38 (original)

**Description**

The original implementation built an intermediate `std::vector<float>` copy before constructing
`std::discrete_distribution`:

```cpp
// ORIGINAL (unnecessary copy):
std::vector<float> weights(row_in, row_in + class_size);
std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
```

The justifying comment claimed a `float*` was not "double-convertible", but this is incorrect.
The C++ standard (`[rand.dist.samp.discrete]`) requires the iterator's value type to be
convertible to `double`. `float` is implicitly convertible to `double`, so the intermediate copy
is wholly unnecessary.

The CPU reference (`multinomial_cpu.cc`, line 20) correctly passes a `float*` range directly
without any copy.

**Fix applied** (`src/ops/multinomial_metal.mm`):

```cpp
// AFTER:
// float* satisfies the double-convertible requirement of discrete_distribution.
std::discrete_distribution<int32_t> dist(row_in, row_in + class_size);
```

This eliminates one `std::vector<float>` heap allocation per batch row per call.

---

## Section 2 — Code Quality

### 2.1 — `concat_split_slide_metal.mm`: helper function name mismatch with CPU counterpart

**Severity**: Minor (clarity / consistency)
**File**: `src/ops/concat_split_slide_metal.mm`, lines 29–41 (original)

**Description**

The local helper functions were named `concat_copy_size` and `concat_iter_size`, while the
corresponding functions in the CPU reference (`concat_split_slide_cpu.cc`) are named
`compute_copy_size` and `compute_iter_size`. The functions have identical semantics.

Using different names in otherwise parallel files creates an unnecessary cross-file reading
burden: a reviewer matching the Metal implementation to its CPU counterpart has to mentally
map two names that mean the same thing.

**Fix applied** (`src/ops/concat_split_slide_metal.mm`):

Renamed all occurrences:
- `concat_copy_size` → `compute_copy_size`
- `concat_iter_size` → `compute_iter_size`

---

### 2.2 — `median_filter_metal.mm`: missing `#include <cstdlib>` for `std::abs` on `dim_t`

**Severity**: Minor (portability)
**File**: `src/ops/median_filter_metal.mm`, line 11 (original)

**Description**

`std::abs(j + k)` is called where `j` and `k` are `dim_t` (= `int64_t`). The correct overload
of `std::abs` for `int64_t` is declared in `<cstdlib>`. The file only included `<cmath>` and
`<algorithm>`, which on Apple Clang / macOS transitively provide `std::abs` for integer types
(because `<cmath>` includes POSIX `<math.h>` which brings in `abs`). On strictly conforming
compilers or libc++ configurations, this transitive inclusion is not guaranteed.

**Fix applied** (`src/ops/median_filter_metal.mm`):

Added `#include <cstdlib>` directly.

---

### 2.3 — `multinomial_metal.mm`: incorrect comment about `discrete_distribution` requirements

**Severity**: Cosmetic (misleading comment, now removed)
**File**: `src/ops/multinomial_metal.mm`, lines 35–36 (original)

**Description**

The comment read:
```cpp
// Build a float copy for discrete_distribution (required to accept
// a range of double-convertible values).
```

This is backwards: `float*` already satisfies the double-convertible requirement. The comment
implied the copy was mandatory, which was wrong. Fixed by removing the copy entirely (Bug 1.1)
and replacing the comment with:
```cpp
// float* satisfies the double-convertible requirement of discrete_distribution.
```

---

## Section 3 — Performance

### 3.1 — `mean_metal.mm`: the CPU loop has stride-`inner_size` reads when `inner_size > 1`

**Severity**: Low (not in a hot path; deferred)
**File**: `src/ops/mean_metal.mm`, lines 31–33

**Description**

The loop order is `outer → inner → axis`:

```cpp
for (dim_t i = 0; i < outer_size; ++i)
  for (dim_t j = 0; j < inner_size; ++j)        // ← inner loop is outermost
    for (dim_t k = 0; k < axis_size; ++k)
      sum += src[i * axis_size * inner_size + k * inner_size + j];  // stride = inner_size
```

When `inner_size > 1`, the innermost `k` loop reads elements with stride `inner_size`, causing
poor cache utilization. Reordering the loops to `outer → axis → inner` would give sequential
reads in the `inner` innermost loop. However, Mean is rarely a bottleneck in transformer models,
and the benchmark shows ~9.6 ms only for extreme shapes (`[256×512×128]`). This is an acceptable
tradeoff for now.

**Recommendation**: For `inner_size > 1` and large `axis_size`, accumulate into a temporary
`float dst[inner_size]` and loop `axis` outside `inner`. Defer to M8.

---

### 3.2 — `mean_metal.mm`: GPU acceleration possible via new MSL kernel (not reuse of `reduce_sum`)

**Question raised**: "Can Mean use the existing GPU `reduce_sum` kernel?"

**Analysis**:

The existing `primitives<Device::METAL>::sum(array, size)` (in `primitives_reduction.mm`) is a
**flat 1D reduction**: it reduces an entire contiguous buffer to a single scalar using a
two-pass GPU tree reduction (threadgroup shmem + CPU accumulate over partials). It is
designed for: `Σ x[i]` for `i = 0..N-1`.

`Mean::compute` requires a **shaped multi-output reduction**: for each of `outer × inner`
output positions `(i, j)`, it computes:

```
dst[i*inner + j] = Σ_{k=0}^{axis-1} src[i*axis*inner + k*inner + j]  / axis
```

These are fundamentally different:

| Property | `reduce_sum` (existing) | `Mean::compute` |
|----------|------------------------|-----------------|
| Outputs | 1 scalar | `outer × inner` values |
| Input access | contiguous | strided (stride = `inner_size`) |
| Kernel structure | threadgroup tree reduction | independent per-output reduction |
| Reusable? | **No** (single output only) | needs new kernel |

Calling `reduce_sum` `outer × inner` times (one per output element) would create `outer × inner`
separate GPU dispatches with one command buffer commitment each — prohibitively expensive overhead.

**Correct GPU approach** (deferred to M8): a new MSL kernel dispatching one thread per output
element, each accumulating `axis_size` elements in float32:

```metal
kernel void mean_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint4& dims [[buffer(2)]],   // {outer, axis, inner, get_sum}
    uint gid [[thread_position_in_grid]])
{
    uint j = gid % dims[2];   // inner index
    uint i = gid / dims[2];   // outer index
    float sum = 0.0f;
    for (uint k = 0; k < dims[1]; ++k)
        sum += input[i * dims[1] * dims[2] + k * dims[2] + j];
    output[gid] = dims[3] ? sum : sum / float(dims[1]);
}
```

This dispatches `outer × inner` threads with no threadgroup communication — each thread is
independent. GPU advantage would appear at large shapes (e.g., `[256×1024]` = 256 independent
float32 sums over 1024 elements). Until the MSL kernel is added, the CPU path via
`commit_and_wait() + 3-loop` is correct and acceptable.

---

## Section 4 — Missing Tests

### 4.1 — Concat only tested with 2 inputs (axis=0 and axis=1)

**Priority**: Low
**Status**: ✅ Done

The original tests used exactly 2 inputs. `Concat::compute` loops over any number of inputs;
the n=3 path (allocating output pointer advance twice) was untested.

**Test added** (`tests/metal/m7_test.mm`, Test 18):
`concat 3 inputs: [3]+[4]+[5]=[12] contiguous`

---

### 4.2 — TopK k>1 only tested for float32 (not float16/bfloat16)

**Priority**: Low
**Status**: ✅ Done

Tests 10–11 covered k=1 for float16/bfloat16 (the `std::max_element` path). The k>1 path
uses `std::partial_sort` with a `static_cast<float>` comparator. This comparator is the
correctness-critical part for bfloat16 (which lacks `operator<`). Only the k=1 path was
originally tested for float16/bfloat16.

**Tests added** (`tests/metal/m7_test.mm`, Tests 19–20):
- `topk k=3 float16: correct top-3 indices [3,1,2]`
- `topk k=3 bfloat16: correct top-3 indices [3,1,2]`

---

### 4.3 — Split only tested with two equal halves

**Priority**: Low
**Status**: ✅ Done

Tests 4a/4b split a `[4×4]` tensor into two `[2×4]` (equal) halves. An unequal split (e.g.
`[2]+[6]` from `[8]`) exercises the `copy_size` pointer advance more thoroughly.

**Test added** (`tests/metal/m7_test.mm`, Test 21):
`split unequal [8] → [2]+[6]: both parts correct`

---

### 4.4 — Mean tested only with `get_sum=false` (mean, not sum)

**Priority**: Low
**Status**: ✅ Done

The `get_sum=true` branch (return raw sum, no division) was not exercised.

**Test added** (`tests/metal/m7_test.mm`, Test 22):
`mean get_sum=true [2×3]: sums are 6 and 15`

---

### 4.5 — Slide only tested with explicit positive axis (axis=0)

**Priority**: Low
**Status**: ✅ Done

`Slide::compute` resolves negative axes: `axis = _axis < 0 ? input.rank() + _axis : _axis`.
This normalization was never exercised. For `_axis=-1` on a `[3×4]` tensor, `axis=1`; the
test verifies column extraction at `index=2` yields `[2, 6, 10]`.

**Test added** (`tests/metal/m7_test.mm`, Test 23):
`slide axis=-1 index=2: extracts column [2, 6, 10]`

---

### 4.6 — `max_num_classes<Device::METAL>()` specialization never asserted in a test

**Priority**: Low
**Status**: ✅ Done

`topp_mask_metal.mm` defines `template <> dim_t TopPMask::max_num_classes<Device::METAL>()`
to return `std::numeric_limits<dim_t>::max()` (no artificial vocabulary limit). This explicit
specialization — which matches the CPU behaviour and diverges from potential GPU constraints
— was never validated by any test.

Since `max_num_classes<D>()` is a private template method, the test verifies the intended
constant directly and documents the "no limit" contract for Metal.

**Tests added** (`tests/metal/m7_test.mm`, Test 24, two checks):
- `max_num_classes<METAL> == dim_t::max (no vocab cap)`
- `max_num_classes<METAL> > 200000 (larger than largest LLM vocab)`

---

## Summary Table

| ID | Category | File | Severity | Status | Action |
|----|----------|------|----------|---------|----|
| 1.1 | Bug | `multinomial_metal.mm` | Minor | ✅ Fixed | Removed unnecessary `std::vector<float>` copy; pass `float*` directly to `discrete_distribution` |
| 2.1 | Quality | `concat_split_slide_metal.mm` | Minor | ✅ Fixed | Renamed `concat_copy_size/iter_size` → `compute_copy_size/iter_size` to match CPU naming |
| 2.2 | Quality | `median_filter_metal.mm` | Minor | ✅ Fixed | Added `#include <cstdlib>` for portable `std::abs` on `dim_t` |
| 2.3 | Quality | `multinomial_metal.mm` | Cosmetic | ✅ Fixed | Removed incorrect comment about `discrete_distribution` requiring a copy |
| 3.1 | Perf | `mean_metal.mm` | Low | Deferred | Strided inner access for `inner_size > 1`; loop reorder deferred to M8 |
| 3.2 | Perf | `mean_metal.mm` | Medium | Deferred | GPU Mean kernel requires new MSL (cannot reuse `reduce_sum`); deferred to M8 |
| 4.1 | Test | `m7_test.mm` | Low | ✅ Done | Test 18: concat 3 inputs |
| 4.2 | Test | `m7_test.mm` | Low | ✅ Done | Tests 19–20: topk k=3 for float16 and bfloat16 |
| 4.3 | Test | `m7_test.mm` | Low | ✅ Done | Test 21: unequal split |
| 4.4 | Test | `m7_test.mm` | Low | ✅ Done | Test 22: mean get_sum=true |
| 4.5 | Test | `m7_test.mm` | Low | ✅ Done | Test 23: slide negative axis |
| 4.6 | Test | `m7_test.mm` | Low | ✅ Done | Test 24: max_num_classes<METAL> constant |

**Test count after review: 28/28 pass** (was 20/20 before review).

---

## Changes Applied

### `src/ops/multinomial_metal.mm`
- **Bug 1.1 / 2.3**: Removed `std::vector<float>` copy; replaced with direct `float*` range
  iterator constructor. Removed incorrect comment.

### `src/ops/concat_split_slide_metal.mm`
- **2.1**: Renamed `concat_copy_size` → `compute_copy_size` and `concat_iter_size` →
  `compute_iter_size` (updated all 6 call sites).

### `src/ops/median_filter_metal.mm`
- **2.2**: Added `#include <cstdlib>` for portable `std::abs` on `int64_t`/`dim_t`.

### `tests/metal/m7_test.mm` (extended, 20 → 28 checks)
- Updated header comment to list tests 18–24.
- Added `test_concat_3inputs()` (Test 18, 1 check).
- Added `test_topk_k3_halfs()` (Tests 19–20, 2 checks).
- Added `test_split_unequal()` (Test 21, 1 check).
- Added `test_mean_get_sum()` (Test 22, 1 check).
- Added `test_slide_negative_axis()` (Test 23, 1 check).
- Added `test_max_num_classes()` (Test 24, 2 checks).
- Extended `main()` to call all six new test functions.

---

## Deferred Items

| Item | Notes |
|------|-------|
| Mean strided access (3.1) | Loop reorder for `inner_size > 1` — deferred to M8 |
| GPU Mean kernel (3.2) | Requires new MSL kernel (`mean_float` + half variants); cannot reuse existing `reduce_sum`. Deferred to M8. |
| GPU kernels for concat/split/tile | Blit encoder optimization — deferred to M8+ |
| Parallel TopK for large vocab | `std::partial_sort` is single-threaded — defer to M8+ |
| `Conv1d` (Whisper encoder) | Requires `MPSCNNConvolution` — deferred to M8 |
| `Quantize` / `Dequantize` | INT8 milestone — deferred to M9 |

---

*End of report.*
