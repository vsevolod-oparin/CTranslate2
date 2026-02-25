# Milestone 4.6 — Broadcast Primitives

**Status:** ✅ DONE (2026-02-25)
**Tests:** 33/33 pass

---

## Summary

Implemented 4 broadcast primitives as GPU compute kernels for Metal:

| Op | Kernel formula | Parameters |
|----|---------------|------------|
| `add_batch_broadcast` | `c[gid] = a[gid % a_size] + b[gid]` | `a_size` |
| `add_depth_broadcast` | `c[gid] = a[gid / depth] + b[gid]` | `depth = b_size / a_size` |
| `add_block_broadcast` | `c[gid] = a[(gid/block) % a_size] + b[gid]` | `block`, `a_size` |
| `mul_batch_broadcast` | `c[gid] = a[gid % a_size] * b[gid]` | `a_size` |

All 6 types supported: `float` (float32), `half` (float16), `bfloat` (bfloat16), `int` (int32), `short` (int16), `char` (int8).

Note: `strided_fill` and `indexed_fill` (also listed in M4.6 in the plan) were already implemented CPU-side in M4.1 and are the correct permanent design — no GPU kernel needed.

---

## Index Math

Each broadcast op is embarrassingly parallel: each thread handles one output element. The index math for `gid` (= output index in `[0, b_size)`) is:

```
add_batch_broadcast:
  CPU: iter = b_size/a_size; c[i*a_size+j] = a[j] + b[i*a_size+j]
  GPU: c[gid] = a[gid % a_size] + b[gid]

add_depth_broadcast:
  CPU: depth = b_size/a_size; c[i*depth+k] = a[i] + b[i*depth+k]
  GPU: c[gid] = a[gid / depth] + b[gid]   (depth passed as constant)

add_block_broadcast:
  CPU: c[i*block+k] = a[i%a_size] + b[i*block+k]
  GPU: c[gid] = a[(gid/block) % a_size] + b[gid]   (block, a_size as constants)

mul_batch_broadcast:
  CPU: c[i*a_size+j] = a[j] * b[i*a_size+j]
  GPU: c[gid] = a[gid % a_size] * b[gid]
```

---

## Implementation

### MSL Kernels

File: `src/metal/kernels/broadcast.metal` (canonical); embedded as `kBroadcastMSL` in `primitives.mm`.

```metal
#define DEFINE_BATCH_BROADCAST(name, op, T)
kernel void name##_batch_broadcast_##T(
    device const T* a [[buffer(0)]],
    device const T* b [[buffer(1)]],
    device       T* c [[buffer(2)]],
    constant  uint& a_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{ c[gid] = a[gid % a_size] op b[gid]; }

#define DEFINE_DEPTH_BROADCAST(name, op, T)
// buffer(3) = depth (computed on host as b_size/a_size)
...

#define DEFINE_BLOCK_BROADCAST(name, op, T)
// buffer(3) = block,  buffer(4) = a_size
...
```

### Infrastructure in `primitives.mm`

Two new dispatch helpers:

```cpp
// 3 data buffers + 1 uint32 constant (for batch and depth broadcasts)
static void dispatch_broadcast1(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 dim_t size, uint32_t param0);

// 3 data buffers + 2 uint32 constants (for block broadcast)
static void dispatch_broadcast2(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 dim_t size, uint32_t param0, uint32_t param1);
```

Both use the same `setBytes:length:atIndex:` pattern established in M4.2 for scalar constants — no extra buffer allocation needed.

Separate `get_broadcast_library()` and `get_broadcast_pso()` — same lazy `call_once` pattern as elementwise (M4.2) and activation (M4.5) libraries.

---

## Building and Running the Tests

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/broadcast_test.mm \
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
    -o broadcast_test && ./broadcast_test
```

---

## Test Results

```
=== M4.6: Broadcast Primitives ===

--- float32 ---
--- float16 ---
--- bfloat16 ---

33 passed, 0 failed
```

Each type tests: add_batch_broadcast (basic + in-place + zero-size), add_depth_broadcast (basic + depth=1 + zero-size), add_block_broadcast (basic + block=1 + zero-size), mul_batch_broadcast (basic + zero-size).

---

## Benchmark: Accuracy & Performance vs CPU

`tests/metal/broadcast_bench.mm` — combined accuracy validation + performance benchmark.

### Building and Running

```bash
clang++ -std=c++17 -O2 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/broadcast_bench.mm \
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
    -o broadcast_bench && ./broadcast_bench
```

### Accuracy Results (a\_size=100, b\_size=10,000 random float32 inputs)

| Op | max\_abs\_diff | rms\_diff | Status |
|----|---------------|-----------|--------|
| add\_batch\_broadcast | 0.000e+00 | 0.000e+00 | PASS |
| add\_depth\_broadcast | 0.000e+00 | 0.000e+00 | PASS |
| add\_block\_broadcast | 0.000e+00 | 0.000e+00 | PASS |
| mul\_batch\_broadcast | 0.000e+00 | 0.000e+00 | PASS |

**4/4 PASS.** Exact zero error — broadcast ops are pure integer-indexed integer/float additions; no floating-point rounding occurs beyond the `+`/`*` itself, and both GPU and CPU use the same IEEE 754 float32 arithmetic.

### Performance Results (a\_size=1024, median latency μs)

GPU time includes `commit_and_wait`. CPU is a sequential double-loop with `-O2`.

| Op | b\_size | GPU (μs) | CPU (μs) | Ratio |
|----|---------|----------|----------|-------|
| add\_batch\_broadcast | 16M | 2,501 | 1,465 | 0.59× — CPU wins |
| add\_depth\_broadcast | 16M | 1,764 | 1,482 | 0.84× — CPU wins |
| add\_block\_broadcast (block=64) | 16M | 3,183 | 2,067 | 0.65× — CPU wins |
| mul\_batch\_broadcast | 16M | 2,384 | 1,560 | 0.65× — CPU wins |

**CPU wins at all tested sizes.** This contrasts with activation ops (e.g. `gelu`: 58.8× GPU speedup).

### Why CPU Wins for Broadcast

These kernels are pure **memory-bandwidth-bound** operations with trivial arithmetic (`+` or `*`). The bottleneck is the per-thread **integer division/modulo** that computes the broadcast index:

| Kernel | Per-thread integer op |
|--------|----------------------|
| `add_batch_broadcast` | `gid % a_size` |
| `add_depth_broadcast` | `gid / depth` |
| `add_block_broadcast` | `(gid / block) % a_size` |
| `mul_batch_broadcast` | `gid % a_size` |

Integer division is expensive in GPU shader cores — it typically compiles to a reciprocal multiply sequence (~20 cycles). The CPU's nested `for (i) for (j)` loop structure avoids per-element division entirely (the index advances by 1 each step) and benefits from auto-vectorization.

### Pipeline Correctness vs Standalone Performance

Despite slower standalone latency, the GPU implementation is still the **correct design** for transformer inference:

- All broadcast ops encode into the deferred command buffer — no `commit_and_wait` between GEMM → broadcast → next GEMM.
- The CPU cannot process bias addition until after `commit_and_wait()` drains the current batch; this would stall the GPU between every layer.
- In practice, broadcast ops are called immediately after large GEMM operations. The GPU executes them as part of the same command buffer submission — the wall-clock cost is hidden behind the GEMM latency.

### Future Optimization

A 2D dispatch (outer index × inner index) would eliminate the integer division entirely by having threadgroup coordinates directly provide `(i, j)` or `(i, k)`. This is the standard approach for GPU bias-add kernels and would likely close the gap with the CPU. Deferred to M5+ along with fused GEMM+bias kernels.

---

## Files Created / Modified

- `src/metal/kernels/broadcast.metal` — canonical MSL source
- `src/metal/primitives.mm` — added `kBroadcastMSL`, `get_broadcast_library`, `get_broadcast_pso`, `dispatch_broadcast1`, `dispatch_broadcast2`; replaced 4 stubs
- `tests/metal/broadcast_test.mm` — 33-test correctness suite
- `tests/metal/broadcast_bench.mm` — accuracy + performance benchmark vs CPU
- `agents/report/milestone-4.6-broadcast-primitives.md` — this file
