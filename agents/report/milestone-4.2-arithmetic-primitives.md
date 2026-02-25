# Milestone 4.2 — Arithmetic Primitives (add, sub, mul)

**Date:** 2026-02-25
**Branch:** `metal-backend`
**Status:** ✅ COMPLETE

---

## Objective

Replace `add`, `sub`, and `mul` stub implementations in `primitives<Device::METAL>`
with real Metal GPU compute kernel implementations using MSL shaders compiled
at runtime via `newLibraryWithSource:options:error:`.

---

## Files Changed

| File | Change |
|------|--------|
| `src/metal/kernels/elementwise.metal` | New — canonical MSL source for element-wise kernels |
| `src/metal/allocator.mm` | Added `buffer_for_ptr(ptr, offset_out)` method + `metal_buffer_for_ptr` free function |
| `src/metal/utils.h` | Declared `metal_buffer_for_ptr` in ObjC++ section |
| `src/metal/primitives.mm` | MSL source embedded as raw string; PSO cache; dispatch helpers; `add`/`sub`/`mul` stubs replaced |
| `tests/metal/arithmetic_test.mm` | New — 21 assertions |

---

## Design

### Why GPU kernels (not CPU-side like M4.1)?

`fill`/`copy`/`convert` are permanently CPU-side because they initialise tensors
before GPU work begins. `add`/`sub`/`mul` are different: they are called **between**
GPU operations and must not introduce a CPU→GPU sync on every call.

Encoding pattern:
```
encode op1 (GPU)
encode add  (GPU) ← this is primitives::add
encode op2 (GPU)
synchronize_stream() → commit_and_wait() → single submit
```

If `add` did a CPU `for` loop, it would produce stale data (op1's output not
yet committed) and force a sync to read it back. A GPU kernel encodes correctly
into the shared command buffer with no intermediate sync.

### MSL kernel structure

Two kernel families:

**Binary vector×vector** — `add_float`, `sub_float`, `mul_float`, etc.:
```metal
kernel void add_float(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device       float* c [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{ c[gid] = a[gid] + b[gid]; }
```

**Scalar×vector** — `add_scalar_float`, `mul_scalar_float`, etc.:
```metal
kernel void add_scalar_float(device const float* x [[buffer(0)]],
                              constant     float& a [[buffer(1)]],
                              device       float* y [[buffer(2)]],
                              uint gid [[thread_position_in_grid]])
{ y[gid] = a + x[gid]; }
```

The scalar `a` is bound via `setBytes:length:atIndex:` (inlined into the
argument table — no MTLBuffer allocation needed for the scalar).

### Types supported

All 6 CTranslate2 arithmetic types:

| C++ type | MSL type |
|----------|----------|
| `float` | `float` |
| `float16_t` | `half` |
| `bfloat16_t` | `bfloat` (compiled conditionally on `__HAVE_BFLOAT__`, i.e. Apple9+) |
| `int8_t` | `char` |
| `int16_t` | `short` |
| `int32_t` | `int` |

### PSO (pipeline state object) caching

The MSL source is compiled once per process into an `id<MTLLibrary>` using
`std::call_once`. Individual kernel `id<MTLComputePipelineState>` objects are
created on first use and cached in a `std::unordered_map<std::string, PSO>`
protected by a mutex. Subsequent calls hit the cache and pay only an
unordered_map lookup.

Compilation happens on the first call to `add`/`sub`/`mul` for each type.
The overhead is amortised across inference (warm-up cost, not per-token cost).

### `metal_buffer_for_ptr` — handling arbitrary offsets

Metal compute encoders require `id<MTLBuffer>` + byte offset; they cannot
accept raw `void*`. `StorageView` operations may be called with a pointer
that is `base + N` bytes into an allocation (e.g., a row of a matrix).

`MetalAllocator::buffer_for_ptr(ptr, offset_out)` scans the `_live` map to
find the enclosing allocation and computes the byte offset:

```cpp
// O(n) where n == number of live allocations (typically a few dozen)
const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
for (auto& [base, entry] : _live) {
  const uint8_t* base_ptr = static_cast<const uint8_t*>(base);
  if (byte_ptr >= base_ptr && byte_ptr < base_ptr + entry.requested_size) {
    *offset_out = static_cast<NSUInteger>(byte_ptr - base_ptr);
    return entry.buffer;
  }
}
```

This correctly handles both allocation-start pointers (offset = 0) and
mid-allocation pointers (offset > 0).

The free function `metal_buffer_for_ptr(ptr, offset_out)` is declared in
`utils.h` (ObjC++ section) and delegates to `MetalAllocator::buffer_for_ptr`.

### Dispatch pattern

```objc
static void dispatch_binary(const char* kernel_name,
                             const void* a, const void* b, void* c,
                             dim_t size) {
  if (size == 0) return;
  id<MTLComputePipelineState> pso = get_elementwise_pso(kernel_name);
  id<MTLCommandBuffer> cmd = get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_a, off_b, off_c;
  [enc setBuffer:metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBuffer:metal_buffer_for_ptr(c, &off_c) offset:off_c atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, size);
  [enc dispatchThreads:MTLSizeMake(size, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}
```

The encoder is committed only when `synchronize_stream(METAL)` → `commit_and_wait()`
is eventually called — never on a per-primitive basis.

### `float16_t` naming conflict

`arm_vector_types.h` (pulled in by `<Metal/Metal.h>`) defines `::float16_t`
as `__fp16` in the global namespace. This conflicts with
`ctranslate2::float16_t` (= `half_float::half`) when `using namespace
ctranslate2` is active in a `.mm` file.

Resolution: In `tests/metal/arithmetic_test.mm` we define distinct aliases
**before** the using-directive:
```cpp
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;
```
The `ct2_f16`/`ct2_bf16` aliases are used throughout the test body.

In `primitives.mm`, `MetalTypeName<float16_t>::value` is defined inside the
anonymous namespace, after the `using namespace ctranslate2` is scoped out
(`namespace ctranslate2` is used instead of the `using` directive in .mm files).

---

## Verification

### Build and run

```bash
clang++ -std=c++17 -O0 \
  -I include -I src \
  -DCT2_WITH_METAL \
  tests/metal/arithmetic_test.mm \
  src/metal/device.mm \
  src/metal/utils.mm \
  src/metal/allocator.mm \
  src/metal/primitives.mm \
  src/allocator.cc \
  src/devices.cc \
  src/cpu/allocator.cc \
  -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -o arithmetic_test && ./arithmetic_test
```

### Output

```
=== M4.2: Arithmetic primitives (add, sub, mul) ===

--- M4.2: primitives<METAL>::add(scalar, vec, out) ---
  PASS  add_scalar float — no error
  PASS  add_scalar float: out[i] == 10 + i
  PASS  add_scalar float16: out[i] ≈ 5 + i
  PASS  add_scalar int32: out[i] == 100 + i

--- M4.2: primitives<METAL>::add(vec, vec, out) ---
  PASS  add_vec float — no error
  PASS  add_vec float: out[i] == i + 1
  PASS  add_vec float16: out[i] ≈ i + 2

--- M4.2: primitives<METAL>::sub(vec, vec, out) ---
  PASS  sub_vec float — no error
  PASS  sub_vec float: out[i] == i
  PASS  sub_vec float: a - a == 0

--- M4.2: primitives<METAL>::mul(scalar, vec, out) ---
  PASS  mul_scalar float — no error
  PASS  mul_scalar float: out[i] == 3*(i+1)
  PASS  mul_scalar float16: out[i] ≈ 2*(i+1)
  PASS  mul_scalar int32: out[i] == 4*(i+1)

--- M4.2: primitives<METAL>::mul(vec, vec, out) ---
  PASS  mul_vec float — no error
  PASS  mul_vec float: out[i] == 2*(i+1)
  PASS  mul_vec float16: out[i] ≈ 3*(i+1)

--- M4.2: zero-size edge cases ---
  PASS  add_scalar size=0 — no error
  PASS  add_scalar size=0: p[0] unchanged
  PASS  mul_vec size=0 — no error
  PASS  mul_vec size=0: p[0] unchanged

21 passed, 0 failed
```

---

## What is now unblocked

- `StorageView::operator+=(value)` and similar — calls `primitives<D>::add(scalar, …)` ✅
- Attention score masking — `add` with broadcast scalar (negative infinity) ✅
- Layer normalisation accumulation — `mul` for scaling ✅
- Residual connections — `add(vec, vec, out)` ✅

Broadcast variants (`add_batch_broadcast`, `add_depth_broadcast`,
`add_block_broadcast`, `mul_batch_broadcast`) still throw "not yet
implemented" — scheduled for M4.3 / M5 along with reductions.

---

## What remains stubbed (arithmetic scope)

| Method | Status |
|--------|--------|
| `add_batch_broadcast` | stub |
| `add_depth_broadcast` | stub |
| `add_block_broadcast` | stub |
| `mul_batch_broadcast` | stub |
| `min(scalar, vec, out)` | stub |
| `min(vec, vec, out)` | stub |
| `max(scalar, vec, out)` | stub |
| `max(vec, vec, out)` | stub |

---

## Next Steps

- **M4.3** — Reduction primitives: `sum`, `max`, `amax`, `max_element`
  via Metal compute shaders (parallel reduction pattern).
- **M4.4** — GEMM: `MPSMatrixMultiplication` for FP32/FP16; `MPSGraph`
  for BF16 (confirmed necessary by M0.2).
