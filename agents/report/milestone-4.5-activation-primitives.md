# Milestone 4.5 — Activation / Transcendental Primitives

**Status:** ✅ DONE (2026-02-25)
**Tests:** 138/138 pass

---

## Summary

Implemented all 11 unary activation and transcendental GPU primitives for Metal:

| Op | Formula | Notes |
|----|---------|-------|
| `exp` | `exp(x)` | MSL built-in |
| `log` | `log(x)` | MSL built-in |
| `cos` | `cos(x)` | MSL built-in |
| `sin` | `sin(x)` | MSL built-in |
| `tanh` | `tanh(x)` | MSL built-in |
| `relu` | `max(x, 0)` | MSL `fmax` |
| `sigmoid` | `1 / (1 + exp(-x))` | in float32 |
| `swish` | `x / (1 + exp(-x))` = `x * sigmoid(x)` | in float32 |
| `gelu` | `0.5 * x * (1 + erf(x / √2))` | `ct2_erf` polynomial |
| `gelu_tanh` | `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))` | MSL tanh |
| `gelu_sigmoid` | `x / (1 + exp(-1.702 * x))` = `x * sigmoid(1.702x)` | in float32 |

Also implemented `logsumexp` (CPU-side).

All three floating-point types are supported: `float` (float32), `half` (float16), `bfloat` (bfloat16).

---

## Implementation

### MSL Kernels

File: `src/metal/kernels/activation.metal` (canonical); also embedded as `kActivationMSL` in `primitives.mm`.

All kernels use a single pattern:

```metal
kernel void <name>_<type>(
    device const T* x [[buffer(0)]],
    device       T* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{ float v = (float)x[gid]; y[gid] = (T)(expr); }
```

**Design choice — always compute in float32:**
All intermediate arithmetic uses `float v = (float)x[gid]`. This ensures:
- `erf`, `sigmoid`, etc. work correctly even when T = `half` or `bfloat`
- Consistent with the CPU reference implementations
- MSL's `exp`, `tanh`, etc. for float are well-tested and hardware-accelerated

### `erf` not available in MSL

Metal Shading Language does not include `erf()` in its standard math library (confirmed on macOS 26.2 / Xcode 26.2 / Metal 3.x). We implement `ct2_erf()` inline using the Abramowitz & Stegun polynomial approximation (formula 7.1.28):

```metal
static float ct2_erf(float x) {
  const float p  = 0.3275911f;
  const float a1 =  0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 =  1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 =  1.061405429f;
  float sign = (x >= 0.f) ? 1.f : -1.f;
  float ax = fabs(x);
  float t  = 1.f / (1.f + p * ax);
  float poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  return sign * (1.f - poly * exp(-ax * ax));
}
```

Max absolute error: 1.5 × 10⁻⁷ — well within float32 precision. No MSL version dependency.

Setting `MTLLanguageVersion3_1` in `MTLCompileOptions` did not resolve the `erf` availability issue; `ct2_erf` is the correct long-term fix.

### Infrastructure in `primitives.mm`

```cpp
// Library compilation (once, lazy):
static id<MTLLibrary> get_activation_library();

// PSO cache (keyed by kernel name string):
static id<MTLComputePipelineState> get_activation_pso(const char* name);

// Dispatch helper (mirrors dispatch_binary, 2 buffers):
static void dispatch_unary(const char* kernel_name, const void* x, void* y, dim_t size);
```

The `METAL_UNARY_OP` macro wires each C++ method to its kernel:

```cpp
#define METAL_UNARY_OP(cpp_name, kernel_prefix)                         \
  template<>                                                            \
  template <typename T>                                                 \
  void primitives<Device::METAL>::cpp_name(const T* x, T* y, dim_t size) { \
    char kname[kKernelNameBufSize];                                     \
    std::snprintf(kname, sizeof(kname), kernel_prefix "_%s",           \
                  MetalTypeName<T>::value);                             \
    dispatch_unary(kname, x, y, size);                                  \
  }

METAL_UNARY_OP(exp,          "exp")
METAL_UNARY_OP(log,          "log")
// ... etc.
```

All 11 kernels encode into the per-thread command buffer (deferred pattern). No `commit_and_wait` needed — they are as lightweight as the M4.2 arithmetic kernels.

### `logsumexp`

Implemented CPU-side (after `commit_and_wait`) using the numerically stable log-sum-exp formula:

```cpp
float maxval = max(x[0..size-1]);
float sum = Σ exp(x[i] - maxval);
return log(sum) + maxval;
```

This avoids a full two-pass GPU reduction + exp kernel. Adequate for the sizes logsumexp is called with in practice (softmax vocabulary dimension ~32K–128K handled by future fused softmax in M5+).

---

## Building and Running the Tests

Run from the repository root:

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/activation_test.mm \
    src/metal/device.mm \
    src/metal/utils.mm \
    src/metal/allocator.mm \
    src/metal/primitives.mm \
    src/allocator.cc \
    src/devices.cc \
    src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o activation_test && ./activation_test
```

---

## Test Results

```
=== M4.5: Activation/Transcendental Primitives ===

--- float32 ---   46 PASS
--- float16 ---   46 PASS
--- bfloat16 ---  46 PASS

138 passed, 0 failed
```

Each type tests: exp, log, cos, sin, tanh, relu, sigmoid, swish, gelu, gelu_tanh, gelu_sigmoid (3-4 values each), logsumexp (3 cases), and zero-size no-crash (5 ops).

---

## Benchmark: Accuracy & Performance vs CPU

`tests/metal/activation_bench.mm` — combined accuracy validation + performance benchmark.

### Building and Running

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/activation_bench.mm \
    src/metal/device.mm \
    src/metal/utils.mm \
    src/metal/allocator.mm \
    src/metal/primitives.mm \
    src/allocator.cc \
    src/devices.cc \
    src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o activation_bench && ./activation_bench
```

### Accuracy Results (N = 10,000 random float32 inputs)

| Op | max\_abs\_diff | rms\_diff | Status |
|----|---------------|-----------|--------|
| exp | 1.144e-05 | 1.336e-06 | PASS |
| log | 4.768e-07 | 1.000e-07 | PASS |
| cos | 1.192e-07 | 3.419e-08 | PASS |
| sin | 1.192e-07 | 3.569e-08 | PASS |
| tanh | 1.192e-07 | 4.053e-08 | PASS |
| relu | 0.000e+00 | 0.000e+00 | PASS |
| sigmoid | 1.192e-07 | 2.063e-08 | PASS |
| swish | 7.153e-07 | 6.457e-08 | PASS |
| gelu | 4.768e-07 | 1.311e-07 | PASS |
| gelu\_tanh | 2.384e-07 | 6.816e-08 | PASS |
| gelu\_sigmoid | 4.768e-07 | 6.011e-08 | PASS |

**11/11 PASS.** `exp` tolerance was widened to 2e-5 (MSL `exp` rounding vs `std::exp` can differ by ~1.2e-5 at the extremes of [-4, 4]).

### Performance Results (median latency, μs)

GPU time includes `commit_and_wait`. CPU is a sequential scalar loop with `-O2`.

**GPU crossover point** (where GPU becomes faster than CPU):

| Op | Crossover N |
|----|------------|
| exp | ~1M |
| log | ~64K |
| cos | ~64K |
| sin | ~64K |
| tanh | ~64K |
| relu | ~1M |
| sigmoid | ~1M |
| swish | ~1M |
| gelu | ~64K |
| gelu\_tanh | ~64K |
| gelu\_sigmoid | ~1M |

**GPU speedup at 16M elements:**

| Op | GPU (μs) | CPU (μs) | Ratio |
|----|----------|----------|-------|
| exp | 2,550 | 25,159 | **9.9×** |
| log | 3,014 | 77,808 | **25.8×** |
| cos | 2,850 | 109,241 | **38.3×** |
| sin | 2,213 | 101,347 | **45.8×** |
| tanh | 2,857 | 54,054 | **18.9×** |
| relu | 3,183 | 16,579 | **5.2×** |
| sigmoid | 1,710 | 33,401 | **19.5×** |
| swish | 2,415 | 35,858 | **14.9×** |
| gelu | 2,491 | 146,377 | **58.8×** |
| gelu\_tanh | 2,998 | 82,547 | **27.5×** |
| gelu\_sigmoid | 2,489 | 35,715 | **14.4×** |

`gelu` achieves the highest ratio (58.8×) because `std::erf` is very slow on the CPU path.

**Notes:**
- For N < ~64K, GPU fixed overhead (~150–350 μs for encode+commit) dominates.
- In real transformer inference, activation layers operate on tensors of size `batch × seq_len × hidden_dim`, typically millions of elements — well into the GPU-wins regime.

---

## Files Created / Modified

- `src/metal/kernels/activation.metal` — canonical MSL source
- `src/metal/primitives.mm` — added `kActivationMSL`, `get_activation_library`, `get_activation_pso`, `dispatch_unary`, `METAL_UNARY_OP` macro; replaced 12 stubs (11 ops + logsumexp)
- `tests/metal/activation_test.mm` — 138-test correctness suite
- `tests/metal/activation_bench.mm` — accuracy + performance benchmark vs CPU
- `agents/report/milestone-4.5-activation-primitives.md` — this file

---

## Notes

- `MTLLanguageVersion3_1` was tried but did not make `erf` available. The `ct2_erf` inline polynomial is the definitive solution.
- The activation library is compiled separately from the elementwise library (M4.2) — same lazy+once pattern, separate PSO cache.
- Integer types (int8, int16, int32) are not instantiated for activation ops — consistent with the CPU implementation which only provides these for `float`.
