# Milestone 6 Review

## Scope

Files reviewed:

- `src/metal/ops_sdpa.mm` (M6.1 SDPA kernel)
- `src/ops/flash_attention_metal.mm` (M6.1/M6.2/M6.3)
- `src/metal/ops_rotary.mm` (M6.3 prefill RoPE)
- `src/ops/rotary_metal.mm` (M6.3 wrapper)
- `src/metal/ops_alibi.mm` (M6.4 ALiBi kernel)
- `src/ops/alibi_add_metal.mm` (M6.4 wrapper)
- `tests/metal/kv_cache_test.mm`, `rotary_test.mm`, `alibi_test.mm`
- `tests/metal/m61_bench.mm`, `m63_bench.mm`, `m64_bench.mm`

---

## Section 1 — Bugs

### Bug 1.1 — `apply_rope_half`: odd `ndims` silently zeroes the unpaired element

**Severity**: Low (odd ndims does not occur in any standard model; always power-of-2 head dimensions)
**File**: `src/ops/flash_attention_metal.mm`, line 89–101

**Description**

`apply_rope_half` is the CPU decode-path RoPE function. The non-interleave loop writes pairs
`(d, d+half)` for `d in [0, half)` where `half = ndims / 2` (integer division). `tmp` is
zero-initialized by `std::vector<float>(ndims)`. For odd `ndims`, the element at index
`2 * half = ndims - 1` is never assigned and stays 0. The write-back loop then sets
`x[ndims-1] = T(0.0f)`, destroying the input value at that position.

Example: `ndims = 5`, `half = 2`.
Loop writes: `tmp[0], tmp[1], tmp[2], tmp[3]`. `tmp[4]` is never touched.
Result: `x[4]` is zeroed instead of being rotated.

The upstream CPU reference (`src/ops/rotary_cpu.cc`) handles all `ndims` elements correctly:

```cpp
// rotary_cpu.cc (correct):
y[i] = x[i] * c[i] + (i < middle ? -x[i + middle] : x[i - middle]) * s[i];
// d=4, middle=2: y[4] = x[4]*c[4] + x[2]*s[4]  <- correct
```

The MSL kernel in `ops_rotary.mm` is also correct (uses the same `d - middle` formula for
`d >= middle`). So the decode CPU path diverges from both the prefill GPU path and the
canonical CPU reference for odd `ndims`.

**Fix** — change the write-back loop to stop at `2 * half`:

```cpp
// before:
for (dim_t d = 0; d < ndims; ++d) x[d] = T(tmp[d]);

// after:
for (dim_t d = 0; d < 2 * half; ++d) x[d] = T(tmp[d]);
// x[2*half .. ndims-1] are unchanged (they are the "unpaired" elements for odd ndims)
```

Or alternatively, add a validation assert/throw in `FlashAttention::compute<METAL>` (or in
`apply_rope_half`) that `ndims % 2 == 0`, which is always true in practice and documents the
invariant explicitly.

---

### Bug 1.2 — Stale scope label in the ALiBi guard

**Severity**: Cosmetic
**File**: `src/ops/flash_attention_metal.mm`, line 133

**Description**

```cpp
if (alibi) {
    throw std::invalid_argument(
        "Metal FlashAttention: ALiBi is not supported (M6.3 scope)");
}
```

M6.4 implemented `AlibiAdd::compute<Device::METAL>`. The guard is still functionally correct —
`FlashAttention` is never called with `alibi != nullptr` in the current layer code — but the
comment says "M6.3 scope" and "not supported", which is misleading.

**Fix**: Update the comment to explain why the guard exists:

```cpp
if (alibi) {
    // ALiBi bias is applied separately by AlibiAdd::compute<METAL> (M6.4).
    // FlashAttention never receives a non-null alibi pointer in the current layer
    // code (src/layers/flash_attention.cc always passes nullptr).
    throw std::invalid_argument(
        "Metal FlashAttention: ALiBi via FlashAttention is not needed (M6.4 scope)");
}
```

---

## Section 2 — Code Quality

### 2.1 — `apply_rope_half`: `tmp` vector allocated even in the interleave path

**File**: `src/ops/flash_attention_metal.mm`, line 89

```cpp
std::vector<float> tmp(static_cast<size_t>(ndims));   // <- always allocated

if (!interleave) {
    // ... uses tmp
} else {
    // ... never uses tmp
}
```

`tmp` is allocated on the heap every call, but the interleave path never uses it. For the decode
path called once per head per token, this is `batch_size * num_heads` heap allocations per step.

**Fix**: Move the declaration inside the `if (!interleave)` branch:

```cpp
if (!interleave) {
    std::vector<float> tmp(static_cast<size_t>(ndims));
    for (dim_t d = 0; d < half; ++d) { ... }
    for (dim_t d = 0; d < 2 * half; ++d) x[d] = T(tmp[d]);
} else {
    for (dim_t i = 0; i < half; ++i) { ... }
}
```

**Additional**: For a decode loop `batch * num_heads` calls deep, a stack-allocated array
avoids heap overhead entirely. `ndims` is always <= `head_dim` <= 256 in typical models, so
`float tmp[512]` on the stack is safe and avoids the allocation cost entirely.

---

### 2.2 — `sdpa_mps_gemm`: `lda`/`ldb`/`ldc` cast to `NSUInteger` without `ct2_u32` guard

**File**: `src/metal/ops_sdpa.mm`, lines 147–149

```cpp
const NSUInteger nat_rb_a = (NSUInteger)lda * elem;
const NSUInteger nat_rb_b = (NSUInteger)ldb * elem;
const NSUInteger nat_rb_c = (NSUInteger)ldc * elem;
```

The `ct2_u32` overflow check is used consistently for GPU kernel arguments elsewhere
(all `setBytes:` paths), but these raw casts are unchecked. In practice `lda = num_heads * head_dim`
is at most ~4096, so overflow is impossible, but the inconsistency is a style gap.

**Fix**: Apply `ct2_u32` or add an assert for values that reach here:

```cpp
const NSUInteger nat_rb_a = static_cast<NSUInteger>(ct2_u32(lda)) * elem;
```

---

### 2.3 — `ops_rotary.mm` / `ops_alibi.mm`: MSL embedded inline, outside `gen_msl_strings.py` tracking

**Files**: `src/metal/ops_rotary.mm`, `src/metal/ops_alibi.mm`

Other Metal kernels (`normalization.metal`, `gather.metal`, etc.) live in `.metal` source files
tracked by `tools/gen_msl_strings.py` → `src/metal/msl_strings.h`. The `check_msl_sync` CMake
target fails the build if a `.metal` file is edited without regenerating the header. `kRotaryMSL`
and `kAlibiMSL` are embedded directly in their `.mm` files and outside this sync mechanism.

This is an acceptable design choice (standalone ops, not part of `primitives.mm`), but means:

- A typo in the inline MSL produces a runtime error, not a compile-time check.
- There is no static `.metal` file that Metal GPU frame-capture tools (Xcode, Metal Debugger) can easily associate with.

**Recommendation**: Document this divergence with a comment in `ops_rotary.mm` and `ops_alibi.mm` at the MSL string, e.g.:

```cpp
// NOTE: This MSL is embedded inline (not in a .metal file) and is therefore
// not tracked by tools/gen_msl_strings.py / check_msl_sync.
```

---

### 2.4 — `SdpaMPSDtype` specialisations: inconsistent spacing

**File**: `src/metal/ops_sdpa.mm`, lines 121–124

```cpp
template <typename T> struct SdpaMPSDtype;
template<> struct SdpaMPSDtype<float>           // <- missing space before <>
{ static const MPSDataType v = MPSDataTypeFloat32; };
```

Compare with the rest of the codebase which uses `template <>` (with space). Minor.

---

### 2.5 — Redundant `static_cast<dim_t>` on a `dim_t` value

**File**: `src/ops/flash_attention_metal.mm`, line 173

```cpp
const dim_t row_elements = static_cast<dim_t>(num_heads_k) * head_dim;
```

`num_heads_k` is already `dim_t`, so `static_cast<dim_t>(num_heads_k)` is a no-op. Remove.

---

## Section 3 — Performance

### 3.1 — SDPA outer head loop: O(B×H) encoder invocations per forward pass

**File**: `src/metal/ops_sdpa.mm`, `sdpa_metal` function

The `for (b) for (h)` loop calls `sdpa_head_mps` (or `sdpa_head_bf16`) once per batch×head.
For B=1, H=32: 32 head-level SDPA calls, each encoding 2 MPS GEMMs + causal_mask + softmax.
Metal encoder creation/end overhead is small, but repeated encoder creation accumulates.

The root cause is the interleaved Q/K/V layout (`[batch, seqlen, num_heads, head_dim]`): the
row strides for Q and K differ, so a single batched-strided MPS GEMM cannot cover all heads
without extracting each head. The alternative would be a custom tiled SDPA kernel (deferred
to future milestones).

**No action required for M6**: This is a known architectural limitation. Flag for future work.

---

### 3.2 — `sdpa_head_bf16`: Q-scale applied element-by-element in scalar loop

**File**: `src/metal/ops_sdpa.mm`, lines 430–436

```cpp
for (ctranslate2::dim_t r = 0; r < seqlen_q; ++r) {
    const BF* src = q_row0 + r * q_lda;
    BF* dst = q_cont + r * head_dim;
    for (ctranslate2::dim_t j = 0; j < head_dim; ++j) {
        dst[j] = BF(float(src[j]) * scale);   // <- scalar loop
    }
}
```

This is `seqlen_q * head_dim` BF16→float32→BF16 conversions per head, per batch item.
At typical prefill shapes (sq=512, hd=64) this is 32K conversions per head × H=32 → ~1M ops.

**Acceptable for M6**: BF16 SDPA already calls `commit_and_wait()` synchronously; scalar pack
overhead is dwarfed by MPSGraph latency. Flag for optimization if a vectorized Q-scale kernel
is added later.

---

### 3.3 — `apply_rope_half`: heap allocation per call in the decode loop

**File**: `src/ops/flash_attention_metal.mm`, line 89

As noted in 2.1, the non-interleave path allocates `std::vector<float>(ndims)` per head per
decode step. At `batch=1, num_heads=32, ndims=64`: 32 alloc+free pairs per step at ~100–200 ns
each → ~3–6 µs wasted per decode step. Trivially fixed with a stack array (see 2.1).

---

## Section 4 — Missing Tests

### 4.1 — `apply_rope_half` (decode RoPE) is completely untested

**Priority**: High

The decode RoPE path in `flash_attention_metal.mm` (`rotary_cos != nullptr && offset > 0`) has
no test coverage. `kv_cache_test.mm` passes `nullptr` for `rotary_cos/sin` and does not exercise
this code path at all. `rotary_test.mm` tests the GPU prefill kernel (`ops_rotary.mm`) only.

**Proposed test** (`tests/metal/decode_rope_test.mm`):

1. Build a reference using `rotary_cpu.cc`-style formula (same as `apply_rope_half`) applied to Q and K.
2. Build cos/sin tables in "half" format `[positions, ndims/2]`.
3. Call `metal::sdpa_metal<T>` with the pre-rotated Q/K/cache, and also call it with
   unrotated Q/K but simulated `apply_rope_half` applied manually.
4. Or: expose `apply_rope_half` as a non-static helper and test it directly.

A minimal test matrix:

| Case | dtype | interleave | ndims | depth | nh | batch | offset |
|------|-------|-----------|-------|-------|----|-------|--------|
| non-interleave | f32 | false | 32 | 64 | 4 | 1 | 4 |
| interleave | f32 | true | 32 | 64 | 4 | 1 | 2 |
| non-interleave | f16 | false | 16 | 32 | 2 | 1 | 1 |

---

### 4.2 — No SDPA correctness test for `is_causal=false` (cross-attention)

**Priority**: Medium

All accuracy checks in `m61_bench.mm` use `is_causal=true`. Cross-attention (encoder-decoder
attention, `is_causal=false`) is a core use case (e.g., Whisper, NLLB). A regression in the
`dispatch_causal_mask` path (e.g., wrong offset) could corrupt cross-attention output silently.

**Proposed addition** to `m61_bench.mm` or a separate `sdpa_test.mm`:

```
Shape [1, sq=8, sk=32, nh=4, nhk=4, hd=64]  is_causal=false   (encoder-decoder)
```

Verify that no positions are masked (all scores should be unmasked).

---

### 4.3 — No dedicated SDPA prefill correctness test file

**Priority**: Medium

The M6.1 milestone has only `m61_bench.mm` (benchmark with embedded accuracy checks) and
`kv_cache_test.mm` (KV-cache decode). There is no `sdpa_test.mm` equivalent to `alibi_test.mm`
or `rotary_test.mm` that documents correctness cases by name, tolerance, and expected error.

**Proposed `tests/metal/sdpa_test.mm`** covering:

| Case | dtype | sq | sk | nh | nhk | hd | is_causal |
|------|-------|----|----|----|-----|----|-----------|
| causal prefill | f32 | 8 | 8 | 4 | 4 | 64 | true |
| cross-attention | f32 | 4 | 16 | 4 | 4 | 64 | false |
| GQA | f32 | 8 | 8 | 8 | 2 | 32 | true |
| causal prefill | f16 | 8 | 8 | 4 | 4 | 64 | true |
| causal prefill | bf16 | 8 | 8 | 4 | 4 | 64 | true |
| decode (sq=1) | f32 | 1 | 16 | 4 | 4 | 64 | true |
| decode (sq=1) | f16 | 1 | 16 | 4 | 4 | 64 | true |
| decode (sq=1) | bf16 | 1 | 16 | 4 | 4 | 64 | true |

---

### 4.4 — Interleave RoPE with partial rotation not tested

**Priority**: Low

`rotary_test.mm` tests partial rotation (`ndims=32, depth=64`) only for the non-interleave path.
The interleave path with `ndims < depth` (partial rotation) is not exercised.

Additionally, interleave + odd `ndims` (even if unlikely in practice) should have a test to
document the current behaviour and catch future regressions.

**Proposed additions** to `rotary_test.mm`:

```
f32 interleave partial (ndims=32, depth=64, is_transposed=false)
f32 interleave partial (ndims=32, depth=64, is_transposed=true)
```

---

### 4.5 — ALiBi test uses only small `key_length` (≤ 32)

**Priority**: Low

`alibi_test.mm` uses at most `key_length=32`. A test with `kl=512` or `kl=1024` would stress
the 2D dispatch and verify the `tg_kl` clamping logic works correctly for larger sequences.

**Proposed addition** to `alibi_test.mm`:

```
f32 [1, 8, 16, 512]   offset=0   (large kl, stresses 2D dispatch threadgroup clamping)
f32 [1, 8, 1, 1024]   offset=0   (decode with long context)
```

---

## Summary Table

| ID | Category | File | Severity | Status | Action |
|----|----------|------|----------|---------|----|
| 1.1 | Bug | `flash_attention_metal.mm` | Low (odd ndims only) | ✅ Fixed | Changed write-back loop to `2 * half`; switched to stack `float[512]` |
| 1.2 | Bug | `flash_attention_metal.mm` | Cosmetic | ✅ Fixed | Updated guard comment to explain M6.4 scope |
| 2.1 | Quality | `flash_attention_metal.mm` | Minor | ✅ Fixed | Moved `tmp` inside `if (!interleave)` branch (fixes 2.1 + 3.3 together) |
| 2.2 | Quality | `ops_sdpa.mm` | Minor | ✅ Fixed | `(NSUInteger)lda/ldb/ldc` replaced with `static_cast<NSUInteger>(ct2_u32(...))` |
| 2.3 | Quality | `ops_rotary.mm`, `ops_alibi.mm` | Info | ✅ Fixed | Added `// NOTE: MSL is embedded inline...` comment to both files |
| 2.4 | Quality | `ops_sdpa.mm` | Cosmetic | ✅ Fixed | `template<>` → `template <>` spacing |
| 2.5 | Quality | `flash_attention_metal.mm` | Cosmetic | ✅ Fixed | Removed redundant `static_cast<dim_t>(num_heads_k)` |
| 3.1 | Perf | `ops_sdpa.mm` | Known limit | Deferred | O(B×H) encoder loop; requires future custom tiled SDPA kernel |
| 3.2 | Perf | `ops_sdpa.mm` | Minor | Deferred | BF16 scalar Q-scale; dwarfed by MPSGraph latency |
| 3.3 | Perf | `flash_attention_metal.mm` | Minor | ✅ Fixed | Fixed alongside 2.1 (stack array eliminates heap alloc) |
| 4.1 | Test | `decode_rope_test.mm` | **High** | ✅ Done | Created `tests/metal/decode_rope_test.mm` — 7/7 pass |
| 4.2 | Test | `sdpa_test.mm` | Medium | ✅ Done | Added cross-attention `sq=4, sk=16, is_causal=false` test — passes |
| 4.3 | Test | `sdpa_test.mm` | Medium | ✅ Done | `tests/metal/sdpa_test.mm` already existed; extended to 12 tests — 12/12 pass |
| 4.4 | Test | `rotary_test.mm` | Low | ✅ Done | Added interleave + partial rotation tests — 11/11 pass |
| 4.5 | Test | `alibi_test.mm` | Low | ✅ Done | Added large kl=512 and kl=1024 tests — 11/11 pass |

---

## Changes Applied

All changes were applied before M7. Summary by file:

### `src/ops/flash_attention_metal.mm`
- **Bug 1.1**: `apply_rope_half` write-back loop changed from `ndims` to `2 * half`.
  `std::vector<float> tmp(ndims)` replaced with `float tmp[512]` stack array.
- **Bug 1.2**: Guard comment updated to explain M6.4 ALiBi separation.
- **2.1 / 3.3**: `tmp` declaration moved inside `if (!interleave)` branch; now stack array.
- **2.5**: Removed `static_cast<dim_t>(num_heads_k)` (no-op cast on a `dim_t`).

### `src/metal/ops_sdpa.mm`
- **2.2**: `nat_rb_a/b/c` row-byte calculations: `(NSUInteger)lda * elem` → `static_cast<NSUInteger>(ct2_u32(lda)) * elem` (and ldb, ldc). Consistent with `ct2_u32` usage everywhere else in Metal ops.
- **2.4**: `template<>` → `template <>` on both `SdpaMPSDtype` specializations.

### `src/metal/ops_rotary.mm`, `src/metal/ops_alibi.mm`
- **2.3**: Added `// NOTE: MSL is embedded inline here...` comment before each inline MSL string.

### `tests/metal/rotary_test.mm` (extended, 9 → 11 tests)
- **4.4**: Added `test_interleave_partial_fa2` and `test_interleave_partial_std`.
  All 11/11 pass.

### `tests/metal/alibi_test.mm` (extended, 9 → 11 tests)
- **4.5**: Added `test_f32_large_kl_decode` (kl=512) and `test_f32_large_kl_prefill` (kl=1024).
  All 11/11 pass.

### `tests/metal/sdpa_test.mm` (extended, 8 → 12 tests)
- **4.2**: Added `test_cross_attention_and_decode()` with:
  - float32 cross-attention `sq=4, sk=16, is_causal=false` (true encoder-decoder path)
  - float32/float16/bfloat16 decode `sq=1, sk=16, is_causal=false`
- **4.3**: File already existed; header comment updated to list new tests.
  All 12/12 pass.

### `tests/metal/decode_rope_test.mm` (new file, 7 tests)
- **4.1**: Created `tests/metal/decode_rope_test.mm`.
  Replicates `apply_rope_half` as local `apply_rope_half_ref<T>`, then exercises
  the full decode pipeline: half-table construction → rotate Q + K_new → update cache
  → sdpa_metal. Tests: f32 non-interleave, f32 interleave, f16, bf16, partial rotation,
  GQA decode, multi-step (5 steps). All 7/7 pass.

---

**Recommended immediate actions (before M7):**

1. ~~Fix Bug 1.1 (`apply_rope_half` odd ndims)~~ ✅
2. ~~Fix Bug 1.2 (stale comment)~~ ✅
3. ~~Apply fixes 2.1 / 3.3 together (combine `tmp` scope fix with stack array)~~ ✅
4. ~~Add test coverage for `apply_rope_half` (4.1) — highest risk gap~~ ✅

Remaining deferred items: 3.1 (custom SDPA kernel), 3.2 (vectorized BF16 Q-scale). Both are performance-only and do not affect correctness.

---

*End of report.*
