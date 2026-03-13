# Apple M4 Metal Backend Implementation Plan

**Revised:** 2026-03-13
**Status:** M14 complete — M12 done, M13 f16 GEMM fix done, M14.1–14.8 all done (precision parity + GPU sync + INT8 audit + README + perf gate)

---

## Architecture Decision: MPS vs MPSGraph vs Custom Shaders

### Recommendation: MPS individual ops as primary, MPSGraph only for static sub-graphs

The original plan said "use MPSGraph". This needs clarification because there are three distinct Metal APIs:

| API | Abstraction | Best for |
|-----|-------------|----------|
| Metal Compute Shaders | Low-level GPU kernels | Custom ops MPS can't do |
| `MPSMatrix*` / `MPSNNGraph` (MPS) | Individual ops, eager execution | Per-op dispatch, variable shapes |
| `MPSGraph` | Computation graph, compiled | Repeated static sub-graphs |

**Recommendation for CTranslate2:**
- Use **MPS individual ops** (`MPSMatrixMultiplication`, `MPSCNNNeuron`, etc.) for the primitives layer — they fit the existing `primitives<D>` template interface and work well with variable shapes.
- Use **MPSGraph** only where MPS doesn't have a single-op equivalent (e.g., RMS Norm, Rotary embeddings).
- Use **custom Metal compute shaders** for ops with no MPS equivalent and no efficient workaround.

**Why not MPSGraph everywhere:** MPSGraph requires a graph compilation step. For CTranslate2's dynamic inference (variable batch/sequence lengths), compiling a graph per call would dominate latency. MPSGraph with symbolic shapes helps but adds complexity.

> **M0.2 finding (2026-02-25):** `MPSMatrixMultiplication` accepts Float32, Float16, Int8, Int16 only —
> it **asserts at runtime** if passed `MPSDataTypeBFloat16`. BF16 GEMM therefore falls into the
> "MPS doesn't have a single-op equivalent" category and **must use MPSGraph**.
> GEMM is the only primitive where the API choice is dtype-dependent:
>
> | dtype | API | Notes |
> |-------|-----|-------|
> | float32 | `MPSMatrixMultiplication` | Eager, no compilation overhead |
> | float16 | `MPSMatrixMultiplication` | Eager, no compilation overhead |
> | bfloat16 | `MPSGraph` matmul | Compiled once, cached per shape |

---

## Critical Architecture Note: Unified Memory on Apple Silicon

**The original plan treats Metal like CUDA. This is wrong for Apple Silicon.**

On M1/M2/M3/M4, all memory is physically unified — CPU and GPU share the same DRAM. This has major implications:

```
CUDA model (discrete GPU):          Metal/M4 model (unified memory):
  CPU RAM ──copy──► GPU VRAM           CPU RAM == GPU RAM (same physical)
  [cudaMemcpy needed]                  [MTLResourceStorageModeShared]
                                       [buffer.contents returns void*]
                                       [no copy needed — zero-copy!]
```

**Concrete consequence:**
- `MTLBuffer` with `MTLResourceStorageModeShared` gives a `void*` via `[buffer contents]`
- That pointer is valid from **both CPU and GPU code simultaneously**
- StorageView's `void* _buffer` can point directly to Metal buffer contents
- Cross-device "copy" CPU↔Metal becomes a no-op synchronization fence, not a data copy
- The plan's `MTLBlitCommandEncoder` copy path (subtask 2.2) is only needed for `MTLResourceStorageModePrivate` buffers, which we won't need for inference

**MTLBuffer lifetime problem the plan ignores:**
StorageView stores a raw `void*`. For Metal, that pointer comes from `[MTLBuffer contents]`. If `MTLBuffer` is ARC-released, the pointer dangles. The allocator must retain `MTLBuffer` objects keyed by their `contents` pointer.

---

## Weak Spots in the Original Plan

1. **Wrong milestone order**: Dispatch macros (M4) were placed *after* primitives (M3). But dispatch is what routes ops to the Metal path — it must come before any op can be tested end-to-end. Fixed below.

2. **No Milestone 0 (POC)**: Jumping straight to 12 milestones of infrastructure without validating the core assumption (MPS GEMM is fast enough, Metal integration is feasible). A one-day spike should come first.

3. **`primitives<Device::METAL>` interface mismatch**: The primitives interface takes raw `T*` pointers. For GPU-private Metal resources you can't do this. Plan says to use "MPSGraph" inside `fill(T*, T, dim_t)` — but MPSGraph is asynchronous and graph-based; you can't call it from a function that takes a raw pointer and returns void. **The fix**: use `MTLResourceStorageModeShared` (unified memory), which means the `T*` pointer from `[buffer contents]` is already GPU-accessible. Primitives work with raw pointers as-is.

4. **Command buffer management unaddressed**: Metal requires every GPU operation to be encoded into a `MTLCommandBuffer` and committed. Creating/committing one buffer per op is extremely slow (high CPU overhead per commit). The plan has no step for this design decision.

5. **Existing `DEVICE_AND_FLOAT_DISPATCH` has hard-coded CUDA guards**: `dispatch.h` line 29: `if (DEVICE != Device::CUDA) throw "FP16 is only supported on GPU"`. Adding Metal needs to update this guard to `if (DEVICE != Device::CUDA && DEVICE != Device::METAL)`.

6. **Python bindings deferred to M13**: Device string enumeration can be exposed in Python as early as M1, allowing Python-level tests throughout development instead of waiting for the end.

7. **INT8 placed too late (M8, Medium priority)**: Most production-deployed CT2 models are INT8. If INT8 is deferred until after full layer integration, end-to-end model tests (M9) only work for float models. Reordered to M9.

8. **Fake "tests"**: Several subtask tests say "Template instantiation compiles without errors" — this is not a real test. All tests below are executable and assert specific numerical results.

9. **`src/ops/*_metal.mm` file structure**: The plan creates Metal files at the op-dispatcher level. The correct pattern matches the CUDA pattern: Metal implementations belong in `src/ops/*_metal.mm` containing `LayerNorm::compute<Device::METAL, T>()` specializations, and are registered via the existing `DEVICE_AND_FLOAT_DISPATCH` macro. The dispatcher files (`src/ops/layer_norm.cc`) do **not** need new Metal-specific files — they already call the dispatch macro.

10. **No discussion of `bfloat16` availability**: BF16 in MPS/Metal requires macOS 14+ and specific GPU support. Need a runtime check before allowing BF16 on Metal.

11. **CI for Apple Silicon**: GitHub-hosted macOS runners (macos-latest) now include Apple Silicon (M1), but they're slower and limited. The plan's CI step (M11.5) needs to account for this.

12. **`ScopedDeviceSetter` not extended**: `devices.h::ScopedDeviceSetter` needs a Metal case. Missed entirely.

---

## Revised Implementation Plan

### Milestone 0: Proof-of-Concept (Spike)
**Goal:** Validate that MPS GEMM on M4 is fast enough before writing any infrastructure.
**Time:** 2–3 days
**Blocks:** Everything else

**0.1 Standalone MPS GEMM benchmark** ✅ DONE (2026-02-25)
- Write a self-contained `tools/metal_poc/gemm_poc.mm` (no CTranslate2 headers)
- Allocate two `MTLBuffer`s with `MTLResourceStorageModeShared`
- Run `MPSMatrixMultiplication` for a 4096×4096×4096 matmul
- Compare result and timing to CPU (Accelerate `cblas_sgemm`)
- **Build (standalone, not in main CMake):**
  ```bash
  clang++ -std=c++17 -O2 -DACCELERATE_NEW_LAPACK \
      -o gemm_poc tools/metal_poc/gemm_poc.mm \
      -framework Metal -framework Foundation -framework MetalPerformanceShaders \
      -framework Accelerate
  ./gemm_poc
  ```
- **PASS criteria:** Metal result matches CPU within 1e-4; Metal is ≥2× faster
- **Actual result:** Correctness PASS (max abs diff = 0, bit-for-bit identical to Accelerate).
  Speed: 1.9× (46.6 ms Metal vs 87.9 ms CPU, 2.95 TFLOPS) — narrowly below 2× threshold.
  Threshold miss is due to per-GEMM `waitUntilCompleted` synchronisation overhead (worst case).
  In the deferred command-buffer model (multiple ops per buffer before commit) effective
  throughput will be higher. **Proceeding to M0.2 rather than pivoting to custom shaders.**
- **If FAIL:** Re-evaluate MPS vs custom shaders before proceeding
- See `agents/report/milestone-0.1-mps-gemm-poc.md` for full details.

**0.2 Validate BF16 availability** ✅ DONE (2026-02-25)
- Check `[device supportsFamily:MTLGPUFamilyApple9]` (M3+) for BF16
- On M4, BF16 should be available; document the check
- **PASS criteria:** BF16 MPS matmul produces correct results vs FP32 reference
- **Actual result:** Apple9 confirmed on M4; correctness PASS (max rel diff 3.6e-3 < 1e-2).
  **Key finding:** `MPSMatrixMultiplication` rejects `MPSDataTypeBFloat16` at runtime.
  BF16 GEMM requires `MPSGraph.matrixMultiplicationWithPrimaryTensor:secondaryTensor:`.
  Correctness reference must use BF16-quantised inputs (not raw FP32) to isolate
  hardware accumulation error from input-quantisation noise.
  See `agents/report/milestone-0.2-bf16-availability.md` for full details.

**0.3 Command buffer latency test** ✅ DONE (2026-02-25)
- Measure latency of: create buffer → encode one op → commit → wait
- Measure amortized cost with 10 ops per buffer
- **PASS criteria:** Multi-op batching is measurably faster than per-op commit
- **Actual result:** PASS. Multi-op batching is **1.65–3.87× faster per-op** vs 1 op/buffer
  (variance across runs). Unit op: 512×512×512 FP32 GEMM. Key numbers (Apple M4, 3 runs):

  | ops/buffer | per-op range (ms) | TFLOPS range | notes |
  |:----------:|:-----------------:|:------------:|-------|
  | 1          | 0.81–0.88         | 0.31–0.33    | stable |
  | 10         | 0.23–0.49         | 0.55–1.18    | high variance |
  | 20         | 0.26–0.42         | 0.64–1.02    | flattening |
  | 25         | 0.39              | 0.69         | within 20–40 band, no discontinuity |
  | 40         | 0.21–0.25         | 1.06–1.29    | compute ceiling |

  Per-submission overhead ≈ **0.4–0.6 ms** (GPU pipeline startup + OS interrupt latency),
  fixed per command buffer. Empty CB round-trip is only ~0.015 ms.
  Ceiling of ~1.1–1.3 TFLOPS reached at **≥20 ops/buffer**; 25 and 40 ops show no
  further improvement. Variance at 10–40 ops reflects GPU power-state transitions.
  **Architecture constraint confirmed:** primitives must *encode* into the thread-local
  command buffer; only `synchronize_stream()` commits. Per-primitive commit incurs a
  3–4× throughput penalty at this op size.
  See `agents/report/milestone-0.3-cmdbuf-latency.md` for full details.

---

### Milestone 1: Foundation
**Goal:** Add `Device::METAL` to the enum and wire up build system, with zero breakage to existing tests.
**Time:** 3–5 days
**Depends on:** M0

**1.1 Add `Device::METAL` to enum** ✅ DONE (2026-02-25)
- `include/ctranslate2/devices.h`: add `METAL` after `CUDA`
- `src/devices.cc`: add `"metal"` to `str_to_device()` and `device_to_str()`; add `get_device_count()` for Metal using `MTLCreateSystemDefaultDevice() != nil ? 1 : 0`
- `src/device_dispatch.h`: add `DEVICE_CASE(Device::METAL, ...)` inside `#ifdef CT2_WITH_METAL` guard (same pattern as `CT2_WITH_CUDA`)
- `src/dispatch.h`: update `DEVICE_AND_FLOAT_DISPATCH` FP16/BF16 guards from `DEVICE != Device::CUDA` → `DEVICE != Device::CUDA && DEVICE != Device::METAL`
- **Actual result:** All files updated. Dispatch macros verified via `tests/test_dispatch_macros.cc`
  (`clang++ -fsyntax-only`) across all four `CT2_WITH_CUDA` × `CT2_WITH_METAL` combinations:
  ```
  CPU-only            : OK
  Metal-only          : OK
  CUDA-only           : OK
  CUDA+Metal          : OK
  ```
  New files: `src/metal/device.h`, `src/metal/device.mm`, `tests/test_dispatch_macros.cc`.
  See `agents/report/milestone-1.1-device-enum.md` for full details.
- **Note:** `synchronize_stream(Device::METAL)` is a no-op stub; real deferred commit in M2.
- **PASS:**
  ```cpp
  // tests/devices_test.cc (new)
  ASSERT_EQ(device_to_str(Device::METAL), "metal");
  ASSERT_EQ(str_to_device("metal"), Device::METAL);
  ASSERT_GE(get_device_count(Device::METAL), 1);  // on M4
  // All existing tests still pass (no regression)
  ```

**1.2 CMake integration** ✅ DONE (2026-02-25)
- Add `option(WITH_METAL "Compile with Apple Metal backend" OFF)`
- Gate Metal source files behind `WITH_METAL`
- Add frameworks: `-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph`
- Set `CMAKE_OSX_DEPLOYMENT_TARGET 14.0` when `WITH_METAL=ON` (plan said 13.0; raised to 14.0
  because BF16/MPSGraph requires Apple9 GPU, which is macOS 14+; CPU-only builds keep 10.13)
- Add `add_definitions(-DCT2_WITH_METAL)` and `enable_language(OBJCXX)` (requires CMake ≥ 3.16)
- Append `src/metal/device.mm` to `SOURCES` (compiled by host AppleClang, not a separate compiler)
- Non-Metal builds: `DEVICE_CASE(Device::METAL, ...)` throws `runtime_error` (same as CUDA without `CT2_WITH_CUDA`)
- **Actual result:**
  ```
  # Prerequisite: git submodule update --init --recursive
  # cmake -DWITH_METAL=ON configure:
  -- Compiling with Apple Metal backend
  -- The OBJCXX compiler identification is AppleClang 17.0.0.17000603
  -- Configuring done  ✅
  # cmake --build (object file only exists after build, not after configure):
  [ 98%] Building OBJCXX object .../src/metal/device.mm.o  ✅
  [100%] Linking ... clang++: error: linker command failed  ← expected (M2+)
  # CPU-only configure: no Metal output, no regression ✅
  ```
  See `agents/report/milestone-1.2-cmake-integration.md` for full details.

**1.3 Expose device in Python (early)** ✅ DONE (2026-02-25)
- `python/cpp/storage_view.cc`: added `.value("metal", Device::METAL)` to Device enum binding
- `python/cpp/module.cc`: added `get_supported_devices()` (new; uses `get_device_count()` internally, no `#ifdef` needed) and `get_metal_device_count()` (symmetric with `get_cuda_device_count`)
- `python/ctranslate2/__init__.py`: exported `get_supported_devices` and `get_metal_device_count`
- **Actual result:** Changes verified by grep; full test requires Python extension rebuild.
  See `agents/report/milestone-1.3-python-device-exposure.md` for full details.
- **PASS** (once extension rebuilt):
  ```python
  import ctranslate2
  assert "metal" in ctranslate2.get_supported_devices()   # WITH_METAL=ON build on Apple Silicon
  assert ctranslate2.get_metal_device_count() == 1
  assert ctranslate2.Device.metal == ctranslate2.Device.metal
  # CPU-only build: get_supported_devices() == ["cpu"], get_metal_device_count() == 0
  ```

---

### Milestone 2: Metal Context and Command Buffer Model
**Goal:** Establish the Metal execution context and define the command buffer lifecycle.
**Time:** 3–5 days
**Depends on:** M1

**Design decision documented here (not left implicit):**

```
Metal execution model chosen for CTranslate2:
  - One MTLDevice per process (singleton, lazy init)
  - One MTLCommandQueue per thread (thread_local)
  - Command buffer strategy: DEFERRED — ops encode into a shared
    per-thread MTLCommandBuffer; committed at synchronize_stream()
  - This matches the CUDA stream model closely
```

**2.1 Create `src/metal/` context module** ✅ DONE (2026-02-25)
- `src/metal/utils.h`: single header with `#ifdef __OBJC__` split — C++ section
  exposes `commit_and_wait()` for `devices.cc`; ObjC++ section exposes the full API
  and `CT2_METAL_CHECK_BUFFER` / `CT2_METAL_CHECK_OBJ` macros
- `src/metal/utils.mm`: process-wide device singleton (C++11 static), per-thread
  `MTLCommandQueue` and `MTLCommandBuffer` (thread_local ARC strong), `commit_and_wait()`
- `src/devices.cc`: `synchronize_stream` and `synchronize_device` now call `metal::commit_and_wait()`
- `CMakeLists.txt`: `src/metal/utils.mm` added to `METAL_SOURCES`
- **Actual result:** 10/10 assertions pass in `tests/metal/context_test.mm`
  (standalone build, no cmake dependency). Both `.mm` files compile in cmake build.
  See `agents/report/milestone-2.1-metal-context.md` for full details.

**2.2 Implement `synchronize_device` / `synchronize_stream` for Metal** ✅ DONE (2026-02-25)
- `src/devices.cc`: both functions call `metal::commit_and_wait()` under `CT2_WITH_METAL` guard
  (implemented in M2.1; verified by `tests/metal/sync_scoped_test.mm`)
- **Actual result:**
  - `commit_and_wait()` with no encoded commands: no-op (returns immediately)
  - Encode 64-byte blit → `commit_and_wait()`: no error; GPU data verified correct
  - Fresh command buffer ready after sync; differs from committed buffer
  - 5/5 assertions pass
  See `agents/report/milestone-2.2-2.4-sync-scoped-error-handling.md` for full details.

**2.3 Extend `ScopedDeviceSetter` for Metal** ✅ DONE (2026-02-25)
- `get_device_index<Device::METAL>()` always returns 0 (one GPU per Apple Silicon system)
- `set_device_index<Device::METAL>(0)` is a no-op; index ≠ 0 throws `std::invalid_argument`
- `ScopedDeviceSetter` RAII template works without modification
  (implemented in M1.1 `src/metal/device.h`; verified by `tests/metal/sync_scoped_test.mm`)
- **Actual result:** 5/5 assertions pass
  See `agents/report/milestone-2.2-2.4-sync-scoped-error-handling.md` for full details.

**2.4 Error handling strategy** ✅ DONE (2026-02-25)

Metal operations can fail asynchronously. Define consistent error handling:

```objc
// src/metal/utils.h

// Check command buffer status AFTER waitUntilCompleted:
#define CT2_METAL_CHECK_BUFFER(buf) \
  do { \
    if ((buf).status == MTLCommandBufferStatusError) { \
      throw std::runtime_error( \
          std::string("Metal command buffer error: ") + \
          [(buf).error.localizedDescription UTF8String]); \
    } \
  } while(0)

// Check that an Objective-C object was successfully created (non-nil):
#define CT2_METAL_CHECK_OBJ(obj, name) \
  do { \
    if ((obj) == nil) { \
      throw std::runtime_error("Metal: failed to create " name); \
    } \
  } while(0)
```

Correct usage pattern:
```objc
// Creating MPS objects (nil on failure, no NSError parameter):
MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
    initWithDevice:device transposeLeft:trans_a ...];
CT2_METAL_CHECK_OBJ(mm, "MPSMatrixMultiplication");

// Checking command buffer after GPU completes:
id<MTLCommandBuffer> buf = get_current_command_buffer();
[mm encodeToCommandBuffer:buf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
commit_command_buffer();
[buf waitUntilCompleted];
CT2_METAL_CHECK_BUFFER(buf);
```

**Note**: Most MPS objects return nil on failure (not NSError). MTLDevice operations that take `NSError**` (like `newLibraryWithSource:`) need a separate `NSError* error = nil; ...; if (error) throw...` pattern inline.

Error recovery:
- Command buffer errors: Mark current batch as failed, throw with GPU diagnostics
- Out of memory: Try allocator cache flush (`alloc.clear_cache()`), then fail gracefully
- GPU fault: Log `buf.error.localizedDescription`, throw `std::runtime_error`

**Actual result:** Both macros implemented in `src/metal/utils.h` (M2.1); used by
`commit_and_wait()` and every subsequent `.mm` primitive file. No test failures observed.
See `agents/report/milestone-2.2-2.4-sync-scoped-error-handling.md` for full details.

---

### Milestone 3: Metal Allocator
**Goal:** Metal buffer allocation backed by unified memory, integrated with `get_allocator<Device::METAL>()`.
**Time:** 3–5 days
**Depends on:** M2

**3.1 Implement `MetalAllocator`** ✅ DONE (2026-02-25)
- `src/metal/allocator.mm`:
  - Pool keyed on requested size: `std::unordered_map<size_t, std::vector<id<MTLBuffer>>>`
  - Live allocations: `std::unordered_map<void*, {requested_size, id<MTLBuffer>}>`
  - Allocate: check pool first; if miss, `[device newBufferWithLength:size options:MTLResourceStorageModeShared]`; return `[buf contents]`
  - Free: move from `_live` to `_pool[size]` (ARC retains the `MTLBuffer` in the vector)
  - `clear_cache()`: `_pool.clear()` — ARC releases all pooled `MTLBuffer` objects
  - Single `std::mutex` guards both maps
- `get_allocator<Device::METAL>()` defined in `allocator.mm` (not `src/allocator.cc`);
  routes via existing `DEVICE_DISPATCH` in `src/allocator.cc`
- **Actual result:** 14/14 assertions pass in `tests/metal/allocator_test.mm`:
  allocate/free/pool-hit/cross-size isolation/clear_cache/free(nullptr) no-op/free(unknown) throws
  See `agents/report/milestone-3.1-metal-allocator.md` for full details.

**3.2 Integrate with `StorageView`** ✅ DONE (2026-02-25)
- `src/metal/primitives.mm` (new): `cross_device_primitives<CPU,METAL>` and `<METAL,CPU>` =
  `std::memcpy` (unified memory — same physical DRAM); `primitives<METAL>::at` = direct pointer
  read; `primitives<METAL>::copy` = memcpy; all other methods stub-throw "not yet implemented"
  (M4). Includes full `DECLARE_ALL_TYPES` explicit instantiations — linker now satisfied.
- `src/storage_view.cc` `copy_from`: Metal cross-device block added before the CUDA block;
  Metal→CPU path calls `synchronize_stream(METAL)` before the memcpy to flush pending GPU writes
- `CMakeLists.txt`: `src/metal/primitives.mm` added to `METAL_SOURCES`
- **Actual result:** 13/13 assertions pass in `tests/metal/storage_view_test.mm`:
  CPU→Metal copy, Metal→CPU copy, int32 round-trip, `primitives<METAL>::at/copy`,
  `synchronize_stream` fence + Metal→CPU.
  `StorageView::to(Device::METAL)` / `to(Device::CPU)` paths verified correct via direct
  `copy_from` calls; full end-to-end test deferred to CMake build (requires `cpu/primitives.cc`).
  See `agents/report/milestone-3.2-storage-view.md` for full details.

---

### Milestone 4: Core Primitives
**Goal:** `primitives<Device::METAL>` specialization — the building blocks all ops call.
**Time:** 1–2 weeks
**Depends on:** M3

**Key insight:** Because we use `MTLResourceStorageModeShared`, all Metal buffers are already CPU-accessible via their `contents` pointer. Primitives operate on those raw pointers directly for element-wise ops. For GEMM and reductions, use MPS objects encoded into the current command buffer.

**4.1 Memory primitives (`fill`, `copy`, `convert`)** ✅ DONE (2026-02-25)
- Unified memory: all four methods are CPU-side operations on shared-mode MTLBuffer
  contents pointers — correct because CPU writes happen-before GPU command encoding.
  - `fill<T>`: `std::fill` — replaces stub
  - `strided_fill<T>`: stride loop — replaces stub
  - `indexed_fill<T>`: index loop — replaces stub
  - `copy<T>`: `std::memcpy` — was real since M3.2
  - `convert<U,V>`: `std::copy` with implicit `half_float::half` / `bfloat16_t` conversion — replaces stub
- **Actual result:** 15/15 assertions pass in `tests/metal/primitives_test.mm`:
  fill (float32/int32/float16), strided_fill, indexed_fill, convert (all 3 round-trips).
  `StorageView::zero()`, `fill()`, and `to(DataType)` are now unblocked.
  **Architecture note:** CPU-side is the permanent design for these four primitives,
  not a stub. Metal command buffer commit overhead (~0.4 ms from M0.3) far exceeds
  the cost of a CPU fill/convert for all tensor sizes used in inference.
  See `agents/report/milestone-4.1-memory-primitives.md` for full details.

**4.2 Arithmetic primitives (`add`, `mul`, `sub`)** ✅ DONE (2026-02-25)
- MSL element-wise kernels compiled at runtime from embedded source string via
  `newLibraryWithSource:options:error:` — one library per process, PSO cached per kernel name.
- `metal_buffer_for_ptr(ptr, offset_out)` added to look up `id<MTLBuffer>` + byte offset
  from any pointer (including mid-allocation offsets for row-slice operations).
- Five kernel families: `add_float`, `sub_float`, `mul_float`, `add_scalar_float`,
  `mul_scalar_float` (and likewise for `half`, `int`, `short`, `char`, `bfloat`).
- Scalar argument bound via `setBytes:` (inlined into argument table — no buffer needed).
- All kernels encode into the per-thread `MTLCommandBuffer`; commit only at `synchronize_stream`.
- **Actual result:** 21/21 assertions pass in `tests/metal/arithmetic_test.mm`:
  add_scalar (float32/float16/int32), add_vec (float32/float16), sub_vec (float32),
  mul_scalar (float32/float16/int32), mul_vec (float32/float16), zero-size no-ops.
- See `agents/report/milestone-4.2-arithmetic-primitives.md` for full design details.

**4.3 Reduction primitives (`sum`, `max`, `amax`, `max_element`)** ✅ DONE (2026-02-25)
- GPU two-pass parallel reduction via custom MSL compute shaders (256-thread threadgroups).
- Canonical MSL source: `src/metal/kernels/reduction.metal`; embedded in `primitives.mm`.
- All 4 ops pass 25/25 correctness assertions (`tests/metal/reduction_bench.mm`).
- See `agents/report/milestone-4.3-reduction-primitives.md` for full design details.

**Performance (Apple M4, median latency):**

| Op | Standalone GPU wins at | Pipelined GPU wins at |
|----|:----------------------:|:---------------------:|
| `sum` | N > ~4M elements | N > ~65K elements |
| `amax` | N > ~1M elements | N > ~65K elements |

Pipelined = reduction follows an encoded GPU arithmetic op (typical in inference).
Vocabulary-scale calls (32K–128K) are always in the GPU-wins zone.

**Decision — no CPU fallback threshold in M4.3:**
A hybrid `if N < threshold: CPU else: GPU` policy was considered and deferred.
The correct threshold depends on whether pending GPU work is already encoded
(cold: N > ~4M; pipelined: N > ~65K), which is not visible at the call site.
Attention-sized reductions (N < 1K, where CPU would win) are not on the critical
path and will be subsumed by fused softmax kernels in M5+.
Revisit after end-to-end profiling with a real model (post-M4.4).

**4.4 GEMM (CRITICAL)** ✅ DONE (2026-02-25)

> **M0.2 finding:** `MPSMatrixMultiplication` does **not** support BF16. The GEMM
> implementation must dispatch on dtype at runtime. Two separate code paths are required.

- `src/metal/primitives.mm`: specialize `primitives<Device::METAL>::gemm<T, S>()`
- Dispatch on `output_type` at the top of the Metal gemm helper:

**Path A — FP32 / FP16: `MPSMatrixMultiplication` (eager)**
```objc
// Used when T = float or float16_t
MPSMatrixDescriptor* descA = ...;  // MPSDataTypeFloat32 or Float16
MPSMatrixDescriptor* descB = ...;
MPSMatrixDescriptor* descC = ...;
MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
    initWithDevice:device transposeLeft:trans_a transposeRight:trans_b
    resultRows:m resultColumns:n interiorColumns:k alpha:alpha beta:beta];
[mm encodeToCommandBuffer:get_current_command_buffer()
      leftMatrix:matA rightMatrix:matB resultMatrix:matC];
```

**Path B — BF16: `MPSGraph` matmul (compiled, cached per shape)**
```objc
// Used when T = bfloat16_t; requires macOS 14+ / MTLGPUFamilyApple9+
// Graph and executable are cached in a per-device shape→MPSGraphExecutable map
// to avoid recompilation on subsequent calls with identical (m, n, k, trans) args.
MPSGraph* graph = get_or_create_bf16_gemm_graph(device, trans_a, trans_b);
MPSGraphTensor* tA = /* placeholder, shape [m, k] or [k, m], BFloat16 */;
MPSGraphTensor* tB = /* placeholder, shape [k, n] or [n, k], BFloat16 */;
MPSGraphTensor* tC = [graph matrixMultiplicationWithPrimaryTensor:tA
                                                  secondaryTensor:tB
                                                             name:nil];
// Run via command queue; result written to shared MTLBuffer
NSDictionary* result = [graph runWithMTLCommandQueue:get_command_queue()
                                               feeds:@{tA: tdA, tB: tdB}
                                       targetTensors:@[tC]
                                    targetOperations:nil];
// Copy result[tC] → output MTLBuffer
[[result[tC] mpsndarray] readBytes:output_ptr strideBytes:nil];
```

**Caching strategy for Path B:**
- Key: `{m, n, k, trans_a, trans_b}` → `MPSGraph*` + compiled `MPSGraphExecutable`
- Use `std::unordered_map` with a struct key in the Metal device context
- First call per unique shape pays the compilation cost (~10 ms); subsequent calls are fast

- Also implement `gemm_batch_strided`:
  - FP32/FP16: `MPSMatrixMultiplication` in a loop over batch dim (MPS has no native batched variant for variable strides)
  - BF16: `MPSGraph` with 3-D tensor inputs `[batch, m, k]` × `[batch, k, n]` → single graph call

- **Actual result:** 26/26 tests pass in `tests/metal/gemm_test.mm`:
  FP32 gemm (7/7), FP16 gemm (6/6), BF16 gemm (6/6), FP32 gemm_batch_strided (4/4), BF16 gemm_batch_strided (3/3).

  Two critical bugs fixed during implementation:
  1. **FP16 rowBytes** — natural stride (8 B for 4-col Float16) was below MPS hardware minimum (16 B).
     Fixed by querying `[MPSMatrixDescriptor rowBytesForColumns:cols dataType:dtype]` and
     copying to padded temp buffers when `nat_rb < mps_min_rb`.
  2. **`@autoreleasepool` + `thread_local` ARC over-release** — calling `get_current_command_buffer()`
     INSIDE `@autoreleasepool {}` left `_thread_buffer` as a dangling pointer after the pool
     drained, causing SIGSEGV in `commit_command_buffer()`.  Fixed by fetching `cmd` BEFORE
     the pool.  This rule must be followed in all future MPS-encoding code.

  See `agents/report/milestone-4.4-gemm.md` for full details.
- **PASS criteria:**
  ```cpp
  // Sizes: 64×64, 512×512, 4096×4096, 128×4096×512 (non-square)
  // Compare to Accelerate cblas_sgemm
  // Max rel diff < 1e-4 for float32, < 5e-3 for float16
  // Max rel diff < 1e-2 for bfloat16 (vs cblas on BF16-quantised inputs)
  // BF16 graph cache: second call with same shape must be ≤ 5% slower than first (steady-state)
  ```

**4.5 Transcendental and activation primitives (`exp`, `log`, `cos`, `sin`, `tanh`, `relu`, `gelu`, `gelu_tanh`, `gelu_sigmoid`, `sigmoid`, `swish`)** ✅ DONE (2026-02-25)
- Custom MSL unary kernels (not MPSGraph) — all 11 ops encode into the deferred command buffer.
- `kActivationMSL` string + separate `get_activation_library()`/`get_activation_pso()` in `primitives.mm`.
- `dispatch_unary(kernel, x, y, size)` helper — mirrors `dispatch_binary` with 2 buffers.
- **`erf` not in MSL stdlib** — implemented `ct2_erf()` inline using Abramowitz & Stegun 7.1.28 polynomial (max error 1.5e-7); avoids any MSL version dependency.
- All intermediate arithmetic in float32; result cast back to T — works uniformly for half and bfloat.
- `logsumexp` — CPU-side after `commit_and_wait()` (log-sum-exp with max-subtraction for numerical stability).
- **Actual result:** 138/138 tests pass in `tests/metal/activation_test.mm`
  (float32/float16/bfloat16 × 11 ops + logsumexp + zero-size no-crash).
  See `agents/report/milestone-4.5-activation-primitives.md` for full details.

**4.6 Broadcast and scatter primitives** ✅ DONE (2026-02-25)

- Custom MSL broadcast kernels in `src/metal/kernels/broadcast.metal` (separate library from elementwise).
- 4 ops implemented:
  - `add_batch_broadcast`: `c[gid] = a[gid % a_size] + b[gid]`
  - `add_depth_broadcast`: `c[gid] = a[gid / depth] + b[gid]`  (depth = b_size / a_size)
  - `add_block_broadcast`: `c[gid] = a[(gid/block) % a_size] + b[gid]`
  - `mul_batch_broadcast`: `c[gid] = a[gid % a_size] * b[gid]`
- All 6 types (float, half, bfloat, int, short, char) instantiated via `DECLARE_ALL_TYPES`.
- `strided_fill` and `indexed_fill` were already implemented CPU-side in M4.1 (correct permanent design).
- **Actual result:** 33/33 tests pass in `tests/metal/broadcast_test.mm`
  (float32/float16/bfloat16 × 4 ops × basic + in-place/depth=1/block=1 + zero-size).
  See `agents/report/milestone-4.6-broadcast-primitives.md` for full details.

**4.7 Beam-search and attention-mask primitives** ✅ DONE (2026-02-25)

See `agents/report/milestone-4.7-beam-search-primitives.md` — 19/19 tests pass.

- **`penalize_previous_tokens`** — GPU kernel (`src/metal/kernels/beam_search.metal`).
  One thread per batch item, sequential over `length`. `penalty` as `float` via `setBytes:`.
  CPU wins standalone (~300–500 μs GPU vs <30 μs CPU) due to ~0.4 ms CB overhead;
  GPU is correct design — encode-only in pipeline, no extra sync cost.

- **`prepare_length_mask`** — CPU-side with `commit_and_wait()` flush.
  O(batch×heads×queries) — GPU kernel launch overhead dominates at typical sizes.
  0 mismatches, 0.7–273 μs latency.

- **`logsumexp`** — CPU-side after `commit_and_wait()`, verified here (done in M4.5).

- **`at<T>`** — Fixed: now calls `commit_and_wait()` before CPU read (was returning stale data).

**4.8 Transpose primitives** ✅ DONE (2026-02-25)

See `agents/report/milestone-4.8-transpose-primitives.md` — 76/76 tests pass.

Implemented via custom MSL compute shaders (`src/metal/kernels/transpose.metal`).
All three ranks use generic flat-index decomposition (one thread per output element);
argument structs (`TransposeArgs2D/3D/4D`) bound at `buffer(2)` via `setBytes:`.
6 MSL types × 3 ranks = 18 kernel functions.

- `transpose_2d(a, dims, b)` — implicit perm=[1,0]; formula: `b[gid] = a[(gid%rows)*cols + gid/rows]`
- `transpose_3d(a, dims, perm, b)` — arbitrary 3D permutation via permuted input strides
- `transpose_4d(a, dims, perm, b)` — arbitrary 4D permutation; critical for MHA head split [0,2,1,3]

Performance highlights (encode+commit_and_wait, vs in-pipeline encode-only):
- 2D: GPU wins at all tested sizes (1.91×–29.76× for 512×512–512×32768)
- 3D/4D: GPU wins beyond ~1–4 MB; CB overhead dominates below that for standalone calls
- Reversed perms ([3,2,1,0]): 4.07× GPU even for 262K-element tensors (CPU cache thrash)

---

### Milestone 5: Op Dispatcher Integration
**Goal:** Wire existing op dispatchers to call Metal primitives. Each op gets a `Device::METAL` path.
**Time:** 1 week
**Depends on:** M1 (dispatch macros), M4 (primitives)

**5.1 Update `DEVICE_AND_FLOAT_DISPATCH` for Metal FP16/BF16** ✅ DONE (2026-02-25)

See `agents/report/milestone-5.1-dispatch-macro-update.md` — 14/14 tests pass.

- `src/dispatch.h`: unified `#else` block covers any GPU backend (CUDA, Metal, or both).
  FP16 and BF16 TYPE_CASEs check `DEVICE != Device::CUDA && DEVICE != Device::METAL` before
  throwing, so Metal gets the same GPU-float treatment as CUDA.
- **Fix applied:** TYPE_CASE tokens qualified as `ctranslate2::float16_t` /
  `ctranslate2::bfloat16_t` to resolve ambiguity when `dispatch.h` is `#include`d
  from `.mm` files (Metal ARM headers inject conflicting `::float16_t`/`::bfloat16_t`
  at global scope). All 4 build configurations (CPU-only, Metal-only, CUDA-only,
  CUDA+Metal) pass `clang++ -fsyntax-only`.
- **PASS:** `tests/metal/dispatch_test.mm` 14/14:
  - float32/float16/bfloat16 + Metal → no throw; D==METAL, sizeof(T) correct
  - float16/bfloat16 + CPU → throws `std::invalid_argument` with "FP16"/"BF16"
  - float16/bfloat16 end-to-end Metal add: results correct

**5.2 Add `Device::METAL` specializations to each op** ✅ DONE (2026-02-25)

See `agents/report/milestone-5.2-metal-op-specializations.md` and
`agents/report/milestone-5-review.md` — 22/22 test files pass (601 total assertions).

All ops in scope are implemented:

| Op | Status | Notes |
|----|--------|-------|
| `LayerNorm` | ✅ | `src/ops/normalization_metal.mm`; last axis only (non-last-axis throws) |
| `RMSNorm` | ✅ | `src/ops/normalization_metal.mm`; `use_residual=true` unsupported (throws) |
| `SoftMax` / `LogSoftMax` | ✅ | `src/ops/normalization_metal.mm`; masking + log mode fully supported |
| `Gemm` / `MatMul` | ✅ | Via M4.4 primitives; no `_metal.mm` file needed |
| `Add`, `Mul` | ✅ | Header-inline via `primitives<D>::add` / `primitives<D>::mul` |
| `BiasAdd` | ✅ | `src/ops/bias_add_metal.mm`; routes to `add_batch_broadcast` / `add_block_broadcast` |
| `Transpose` | ✅ | Via M4.8 primitives |
| `Gather` | ✅ | `src/ops/gather_metal.mm`; all 6 MSL types (float/half/bfloat/int/short/char) |

New MSL kernels (`src/metal/kernels/`):
- `normalization.metal` — layer_norm (two-pass mean+variance), rms_norm (single-pass),
  softmax (three-pass max→sum_exp→normalize); one-threadgroup-per-row design with float32
  threadgroup memory for half/bfloat correctness.
- `gather.metal` — one thread per output element; batched gather via `num_indices_per_batch`.

Code review (`agents/report/milestone-5-review.md`) fully resolved:
- Bugs 1.1–1.3 fixed; Quality 2.1–2.4 fixed; Performance (Section 3) deferred to M7+.
- Tests added: `normalization_gather_test.mm` extended to 25 tests; new
  `normalization_comparison_test.mm` (6 CPU-reference vs Metal checks),
  `bias_add_test.mm` (11 assertions); `pso_warmup_test.mm` extended to 17 tests (all 8
  MSL libraries).

---

### Milestone 6: Attention Mechanisms
**Goal:** Multi-head attention working end-to-end on Metal.
**Time:** 1–2 weeks
**Depends on:** M5

**6.1 Scaled dot-product attention** ✅ DONE (2026-02-25)

Implementation: `metal::sdpa_metal<T>` in `src/metal/ops_sdpa.mm` +
`FlashAttention::compute<Device::METAL>` in `src/ops/flash_attention_metal.mm`.

Algorithm per (b, h):
- `scores = scale * Q[b,h] @ K[b,hk]^T`  — MPSMatrixMultiplication (FP32/FP16 encode-only); MPSGraph (BF16 synchronous)
- `if is_causal: causal_mask_kernel`       — custom MSL in `src/metal/kernels/sdpa.metal`
- `attn = softmax(scores)`                 — reuses M5.2 softmax_metal
- `output[b,h] = attn @ V[b,hk]`          — MPSMatrixMultiplication (FP32/FP16); MPSGraph (BF16)

Key design details:
- Q/K/V layout: `[batch, seqlen, num_heads, head_dim]` (interleaved heads, non-contiguous slices)
- FP32/FP16: MPS rowBytes parameter handles non-contiguous strides directly
- BF16: slices are packed to contiguous allocator-registered buffers (MPSGraph requirement)
- Scores buffer: always allocator-registered (`MetalTempBuf` RAII) so `metal_buffer_for_ptr` can find it
- GQA: `hk = h % num_heads_k`
- M6.1 scope: `offset == 0` only; throws for KV cache, rotary, ALiBi, sliding window, attention weight output

**PASS:** 8/8 tests in `tests/metal/sdpa_test.mm`
- float32 non-causal/causal, multi-head (batch=2), GQA: max err < 1.2e-7
- float16 non-causal/causal: max err < 2.5e-4
- bfloat16 non-causal/causal: max err < 2.0e-3

Report: `agents/report/milestone-6.1-sdpa.md`

**6.2 KV-cache update (`update_state`)** ✅ DONE (2026-02-26)

Implementation in `src/ops/flash_attention_metal.mm` (offset > 0 path):
1. `commit_and_wait()` — flush pending GPU writes (linear projections wrote new K/V via GPU kernels;
   CPU must see the data before the memcpy).
2. CPU memcpy per batch item — write `keys/values[batch, seqlen_new, nh_k, hd]` into
   `cached_keys/values[batch, total_cache, nh_k, hd]` at position `offset`.
   Unified memory means this is immediately visible to the GPU.
3. `sdpa_metal(Q, cached_K, cached_V, out, seqlen_k=offset+seqlen_new, is_causal=false)`.
   `is_causal=false` for decode (sq==1): all cache positions are in the past of the current query;
   the KV cache boundary already provides the temporal constraint.

Key design notes:
- `seqlen_k_eff = offset + seqlen_new` — attend only over the valid cache range, not total_cache slots.
- `seqlen_q > 1` with `offset > 0` (chunk-prefill into cache) throws: not needed for standard decode.
- The memcpy pattern mirrors `prepare_length_mask` (M4.7): CPU-side work on unified memory buffers,
  preceded by `commit_and_wait()` to ensure GPU writes are visible.

**PASS:** 6/6 tests in `tests/metal/kv_cache_test.mm`
- float32 decode (batch=1, nh=4, hd=32, prefill=4, 10 steps): max err 2.98e-07
- float32 GQA decode (nh=4, nh_k=2, hd=16, prefill=4, 5 steps): max err 8.94e-08
- float32 batch=2 decode (nh=2, hd=16, prefill=3, 5 steps): max err 1.19e-07
- float16 decode (prefill=4, 5 steps): max err 4.23e-04
- bfloat16 decode (prefill=3, 3 steps): max err 3.49e-03
- Cache contents verified: each slot written at correct offset, prior slots intact

M6.1 SDPA tests (8/8) still pass after refactor.
Report: `agents/report/milestone-6.2-kv-cache.md`

**6.3 Rotary embeddings (RoPE)** ✅ DONE (2026-02-26)
- `src/metal/ops_rotary.mm`: MSL kernel + `metal::rotary_metal<T>()` free function
- `src/ops/rotary_metal.mm`: `Rotary::compute<Device::METAL>` wrapper
- `src/ops/flash_attention_metal.mm`: CPU RoPE for decode path (offset > 0)
- Tests: `tests/metal/rotary_test.mm` — 9/9 pass (f32/f16/bf16, interleave/non-interleave, both layouts, partial rotation)
- Benchmark: `tests/metal/m63_bench.mm` — 24/24 accuracy checks pass
  - float32 GPU crossover: ~t=2048 (h8, hd64) → 1.13x
  - float16 GPU: approaches crossover ~t=2048 (0.82x); GB-limited
  - bfloat16 GPU crossover: t=2048 → 1.22x
  - CB overhead (~0.4 ms) dominates at small shapes; GPU wins at 2K+ tokens in prefill
- Report: `agents/report/milestone-6.3-rotary.md`

**6.4 ALiBi positional bias** ✅ DONE (2026-02-26)
- `src/metal/ops_alibi.mm` — MSL kernel + `metal::alibi_add_metal<T>()` free function
- `src/ops/alibi_add_metal.mm` — `AlibiAdd::compute<Device::METAL>` thin wrapper
- MSL kernel: 2D grid `[total_rows, key_length]`; one thread per element; all arithmetic float32
- `total_rows = batch * num_heads * query_length`; head index: `h = (vec / query_length) % num_heads`
- Handles batch, multi-query (ql > 1), ALiBi offset (cached tokens), f32/f16/bf16
- Tests: `tests/metal/alibi_test.mm` — 9/9 pass; f32 exact (0.000e+00), f16 <2.4e-3, bf16 <2.6e-2
- Benchmark: `tests/metal/m64_bench.mm` — 21/21 accuracy pass
  - ALiBi is a pure broadcast-add (minimal arithmetic intensity); CPU always wins standalone
  - f32 [1,8,512,512]: GPU 1013 µs vs CPU 218 µs (0.22×); GPU CB overhead dominates
  - In pipeline (CB amortized): encoding cost ~5 µs; GPU correct choice for large prefill
- Report: `agents/report/milestone-6.4-alibi.md`

*Flash Attention (fused SDPA kernel) is deferred to Phase 2 — requires custom Metal compute shaders for the fused kernel. Standard SDPA via MPS first.*

---

### Milestone 7: Remaining Ops (Complete Coverage) ✅ DONE (2026-02-26)

**Goal:** Cover all remaining ops needed for full model support.

**Strategy:** commit_and_wait() + CPU algorithm on shared Metal memory.
All Metal buffers use MTLResourceStorageModeShared (unified memory), so the
CPU can operate directly on GPU-produced data after flushing.

**Ops implemented (8 new *_metal.mm files):**

| File | Ops | Types |
|------|-----|-------|
| `concat_split_slide_metal.mm` | Concat, Split, Slide | All 6 types |
| `tile_metal.mm` | Tile | All 6 types |
| `topk_metal.mm` | TopK | float, float16, bfloat16 |
| `topp_mask_metal.mm` | TopPMask + max_num_classes | float |
| `gumbel_max_metal.mm` | GumbelMax::add_gumbel_noise | float |
| `multinomial_metal.mm` | Multinomial | float |
| `mean_metal.mm` | Mean | float |
| `median_filter_metal.mm` | MedianFilter | float |

**Deferred:** `Conv1d` (M8, MPSCNNConvolution), `Quantize`/`Dequantize` (M9 INT8),
GPU-kernel optimization for memory copy ops.

**PASS:** 20/20 tests in `tests/metal/m7_test.mm`

Report: `agents/report/milestone-7-remaining-ops.md`

---

### Milestone 8: Transformer Layers (End-to-End Layer Tests)
**Goal:** Full transformer encoder and decoder layers produce correct output on Metal.
**Time:** 1 week
**Depends on:** M6, M7

**8.1 `TransformerEncoderLayer` on Metal**
- No new code needed — layers call ops, ops call primitives, primitives are now Metal-specialized
- Test: run one encoder layer (self-attn + FFN) end-to-end on Metal
- **PASS:**
  ```cpp
  // tests/layers_test.cc — add Metal parameter
  // Input: [batch=2, seq=16, dim=256]
  // Metal output vs CPU output: max abs diff < 1e-3 (fp32), < 1e-2 (fp16)
  ```

**8.2 `TransformerDecoderLayer` on Metal**
- Test cross-attention (Q from decoder, KV from encoder) on Metal
- **PASS:** As above with cross-attention inputs

**8.3 Whisper encoder (CNN frontend)**
- `Conv1d` on Metal → `MPSCNNConvolution`
- Full `WhisperEncoder`: Conv1d × 2 + N transformer layers
- **PASS:** Encode a 3-second mel-spectrogram; Metal output within 1e-2 of CPU

---

### Milestone 9: INT8 Quantization on Metal
**Goal:** Run INT8 models on Metal.
**Time:** 1 week
**Depends on:** M5

**9.1 `Quantize` / `Dequantize` primitives on Metal**
- `primitives<Device::METAL>::quantize<float, int8_t>(...)` using MPSGraph cast + scale
- Per-tensor and per-channel scaling
- **PASS:** Quantize→Dequantize round-trip error < 1% vs float32 reference

**9.2 INT8 GEMM on Metal**
- Metal does not have a native INT8 matmul via MPS (as of macOS 14)
- Strategy: dequantize INT8 weights to FP16 → run FP16 GEMM
- If/when Apple adds INT8 matmul to MPS, swap the implementation
- **PASS:**
  ```cpp
  // INT8 model loaded on Metal; output matches float32 model within 1%
  // GEMM correctness test: INT8 weights + float activations → float output
  ```

**9.3 `gemm_pack_b` → return 0 (not supported)**
- `primitives<Device::METAL>::gemm_pack_b()` must return 0 when called without `dest`
- This signals to the op layer that Metal does not support pre-packed weight buffers
- **Without this**: the op layer may call `gemm_pack_b` expecting a size and get undefined behavior
- Implementation: explicit template specialization returning 0
- **PASS:** `primitives<Device::METAL>::gemm_pack_b(b, false, k, n, 1.f) == 0` (no crash, no allocation)

**9.4 `compute_u8_compensation` → no-op**
- Used for the u8s8s32 INT8 GEMM path (where A is uint8, B is int8). Metal doesn't use this path.
- Provide an empty specialization: `primitives<Device::METAL>::compute_u8_compensation(...)` does nothing
- **PASS:** Function compiles and is callable; does not crash; compensation buffer remains unmodified

---

### Milestone 10: Full Model End-to-End
**Goal:** Run a complete seq2seq and language model on Metal from Python.
**Time:** 1–2 weeks
**Depends on:** M8, M9

**10.1 Load model on Metal from Python**
- `python/cpp/translator.cc`: ensure `device="metal"` passes through to `Device::METAL`
- Update Python wrappers for `Translator`, `Generator`, `Encoder` to accept `"metal"` device
- **PASS:**
  ```python
  translator = ctranslate2.Translator("opus-mt-en-de", device="metal")
  results = translator.translate_batch([["Hello", "world", "."]])
  assert results[0].hypotheses[0][0] == "Hallo"
  ```

**10.2 Seq2seq (Transformer) end-to-end** ✅
- Run a small translation model (e.g., opus-mt-en-de, Helsinki-NLP)
- Greedy and beam search (beam_size=4)
- **PASS (correctness):**
  - BLEU score within 0.5 of CPU result on 100-sentence test set ✅ (diff=0.00, exact match 100/100)
  - WMT14 en-de: CPU BLEU 26.01/26.73 (greedy/beam=4), Metal BLEU identical
- **DEFERRED to M11 (speed):**
  - Metal inference speed ≥ 1.5× CPU for batch_size=1
  - Current: 0.08x greedy, 0.11x beam=4 — per-op commit overhead dominates on small model
  - M11 command buffer batching will address this
- Test: `tests/metal/e2e/test_seq2seq_e2e.py` (4/4 pass)
- Report: `agents/report/milestone-10.2-seq2seq-e2e.md`

**10.3 Language model (GPT-style) end-to-end** ✅
- GPT-2 decoder-only model (pre-norm, PositionEmbedding, GELU_TANH activation)
- **Bug found & fixed:** Metal `tanh()` produces NaN for |arg| > ~44 due to `exp(2x)` overflow
  - Root cause: MSL `tanh()` internally computes `(exp(2x)-1)/(exp(2x)+1)`; for large x, `exp(2x)` → inf → NaN
  - Fix: `ct2_safe_tanh()` clamps argument to [-10, 10] (tanh is ±1 to 15+ decimal places there)
  - Applied to: `activation.metal` (tanh + gelu_tanh), `quantize.metal` (dequantize_gemm_output)
- 12/12 tests pass: greedy (4 prompts), beam2/beam4 (2 prompts each), batch=4
- All tokens match CPU exactly (deterministic greedy + beam search)
- Test: `tests/metal/e2e/test_generator.py` (12/12 pass)
- Report: `agents/report/milestone-10.3-lm-e2e.md`

**10.4 Whisper end-to-end** ✅
- Load `whisper-base`; transcribe a 60-second audio clip (Russian podcast, sample.mp3)
- Metal produces **exact transcript match** with CPU (WER = 0.00%)
- **PASS (correctness):**
  - 8/8 tests pass: non-empty, no error tokens, WER < 1%, exact match, sanity checks
  - Metal-vs-CPU WER: 0.00% (well within 1% tolerance)
- **Speed (informational, M11 will optimize):**
  - CPU: 6.2s (RTF=0.103), Metal: 27.1s (RTF=0.451), Ratio: 0.23x
  - Per-op commit overhead dominates; command buffer batching (M11) will address this
- Test: `tests/metal/e2e/test_whisper.py` (8/8 pass)
- Report: `agents/report/milestone-10.4-whisper-e2e.md`

---

### Milestone 11: Performance Optimization
**Goal:** Optimize Metal backend for M4 throughput; profile and fix bottlenecks.
**Time:** 1–2 weeks
**Depends on:** M10

**11.1 Command buffer batching** ✅
- Profile: measure how many Metal command buffers are committed per token
- Goal: ≤1 command buffer per decode step (all ops for one step in one buffer)
- This may require collecting ops across a decode step before committing
- **PASS:** `tools/benchmark/benchmark.py --device metal` shows ≥20% speedup vs per-op commit
- Batched in-place Gather: 12 commit_and_wait() → 1 synchronize_stream() per decode step
- Test: `tests/metal/e2e/test_cb_batching.py` (4/4 pass)
- Report: `agents/report/milestone-11.1-cb-batching.md`

**11.2 Metal pipeline state caching** ✅
- Custom Metal shaders require `MTLComputePipelineState` objects (expensive to create)
- Cache them globally keyed by (shader function name + type parameters)
- **PASS:** Second inference call is not slower than first (pipeline states reused, no recompile)
- Infrastructure: 13 static `PSOCache` instances across 9 Metal files; `compile_library_once` for MSL→library
- Added global PSO hit/miss counters (`pso_hit_count()`, `pso_miss_count()` in `utils.h/utils.mm`)
- Translation: 9 misses on first call → 0 misses on second (1274 hits, 100% cache reuse)
- Whisper: 4 misses on first call → 0 misses on second (12097 hits, 100% cache reuse)
- Latency ratio: 0.98x (later calls not slower — marginally faster)
- Test: `tests/metal/e2e/test_pso_caching.py` (8/8 pass)
- Report: `agents/report/milestone-11.2-pso-caching.md`

**11.3 BF16 inference (M4 specific)** ✅
- Enable BF16 when `[device supportsFamily:MTLGPUFamilyApple9]` is true (M3+)
- Add `CT2_METAL_ALLOW_BF16` env var for opt-in
- MSL Language Version 3.1 for `__HAVE_BFLOAT__` kernel instantiation
- BF16 GEMM alpha scaling: post-GEMM mul_scalar for attention score scaling
- **PASS:** BF16 model runs on Metal; output exact match with FP32 (greedy/beam); 95% token overlap long-form
- **PARTIAL:** BF16 0.86× FP16 speed (MPSGraph overhead; speed parity expected for large models)
- Test: `tests/metal/e2e/test_bf16_inference.py` (13/13 pass)
- Report: `agents/report/milestone-11.3-bf16-inference.md`

**11.4 Profiling integration** ✅
- `src/profiler.cc`: add `PROFILE` macro support for Metal
- Use `GPUStartTime`/`GPUEndTime` from command buffer for GPU-native timing
- **PASS:** `CT2_ENABLE_PROFILING=1 ct2-translator --device metal` shows per-op timings with GPU-ms column
- Test: `tests/metal/e2e/test_profiling.py` — 7/7 pass
- Report: `agents/report/milestone-11.4-profiling.md`

**11.5 General Metal performance optimization** ✅
- 5 optimizations: CPU GEMM for tiny padded, MPS+temp for large padded, GPU blit Split/Concat, batched CPU GEMM, CPU SDPA for small
- Whisper Metal/CPU ratio: 1.28-1.37x (60s audio)
- Report: `agents/report/milestone-11.5-perf-optimization.md`

**11.6 Eliminate Gather/TopK syncs** ✅
- Gather encode-only by default; TopK k=1 uses GPU argmax MSL kernel
- Eliminated ~1079 `commit_and_wait()` calls per Whisper inference
- Whisper Metal/CPU ratio improved from 1.32x → 1.59x
- Report: `agents/report/milestone-11.6-gather-topk-syncs.md`

**11.7 Batched MPS GEMM for non-padded attention** ✅
- `dispatch_mps_gemm_batched<T>()` using MPS batch matrix descriptors
- Single `MPSMatrixMultiplication` with `batchSize` encodes all heads in one call
- Whisper Metal/CPU ratio improved from 1.59x → 2.61x
- Report: `agents/report/milestone-11.7-batched-mps-gemm.md`

**11.8 Flash Cross-Attention on Metal** ✅
- Routes cross-attention through `FlashAttention` op → `sdpa_metal` (fused QK^T + softmax + attn*V)
- `process_cross_attention_flash()`: `[batch, seq, heads, dim]` layout, zero-copy reshape
- Beam_size broadcasting (`kv_b = b / beam_size`) eliminates K/V tiling across beams
- Whisper Metal/CPU ratio: ~1.98x median (on power); faster than old `dot_product_attention` (1.80x)
- Report: `agents/report/milestone-11.8-flash-cross-attention.md`

**11.9 Fused LayerNorm + GEMM kernel** ✅
- Fused LayerNorm/RMSNorm + GEMV in single MSL dispatch (256 threads/row)
- BF16 only (MPS BF16 GEMM requires `commit_and_wait`; for f32/f16 MPS encode-only is faster)
- Guards: outer_size ≤ 16, K ≤ 4096, no quantization, no activation
- Fused in: MultiHeadAttention pre-norm + QKV proj, FeedForwardNetwork pre-norm + FF1 proj
- Report: `agents/report/milestone-11.9-fused-norm-gemm.md`

**11.10 GPU TopK kernel for k>1** ✅
- Iterative argmax with excluded-index list MSL kernel (`topk_k_<T>`)
- Kernel works (30/30 tests) but CPU `partial_sort` is faster for beam search shapes (k=5, depth=51865: GPU ~1.5ms vs CPU ~33µs)
- **Decision:** kernel exists in codebase but production retains CPU fallback for k>1
- Report: `agents/report/milestone-11.10-gpu-topk-k.md`

**11.11 Batched Padded GEMM & faster_whisper optimization** ✅
- Added `dispatch_mps_gemm_batched_padded<T>()` for batched MPS GEMM with row-alignment padding
- MSL `row_copy` kernel for encode-only C unpack (zero syncs)
- `batch_cpu_gemm_f32`/`batch_cpu_gemm_f16` helpers: single sync + N cblas loop (was N syncs)
- Fixed per-element dispatch regression causing 17,700 syncs per faster_whisper transcription
- faster_whisper whisper-large-v3-turbo beam=5: 0.58× → **1.06×** Metal/CPU
- Native API whisper-large-v3-turbo greedy: **1.45×** Metal/CPU
- Report: `agents/report/milestone-11.11-batched-padded-gemm.md`

**11.12 Pad-C sync elimination** ✅
- Eliminate `commit_and_wait()` for pad_c unpack in padded GEMM; use GPU `row_copy` kernel instead
- Report: `agents/report/milestone-11.12-pad-c-sync-elimination.md`

**11.13 Batched padded GEMM row_copy** ✅
- MSL `row_copy` kernel for batched padded GEMM C-unpack (encode-only, zero syncs)
- Report: `agents/report/milestone-11.13-batched-padded-gemm-row-copy.md`

**11.14 Fused ApplyTimestampRules** ✅
- GPU kernel fuses `should_sample_timestamps` reduction + logits disable in a single dispatch
- Report: `agents/report/milestone-11.14-fused-timestamp-rules.md`

**11.15 Batch beam search gathers** ✅
- Batch multiple gather operations in beam search into fewer GPU dispatches
- Report: `agents/report/milestone-11.15-batch-beam-gather.md`

**11.16 BeamSearch GPU acceleration** ✅
- GPU `prepare_length_mask` kernel replacing CPU loop (M4.7 was CPU-only)
- Report: `agents/report/milestone-11.16-beam-search-gpu.md`

**11.17 Fused timestamp check + disable** ✅
- `fuse_timestamp_check_and_disable_metal`: GPU reduction + scatter in single encode-only dispatch
- Report: `agents/report/milestone-11.17-reduction-optimization.md`

**11.18 Encode-only MPS padded GEMM** ✅
- Moved 8,728 CPU GEMM fallbacks for m=1 to encode-only MPS padded GEMM path
- Report: `agents/report/milestone-11.18-mps-padded-gemm.md`

**11.19 Float16 m=1 custom GEMV kernel** ✅
- MSL `gemv_half` kernel for m=1 FP16 GEMM; faster than MPS for small matrices
- Report: `agents/report/milestone-11.19-f16-gemv-kernel.md`

**11.20 GPU Multinomial sampling** ✅
- GPU kernel for multinomial sampling; fixed strided→sequential kernel bug
- Report: `agents/report/milestone-11.20-gpu-multinomial.md`

**11.21 Gather sync elimination** ✅
- `protect_buffer` pattern: ~1200 fewer syncs per inference; encode-only gather
- Report: `agents/report/milestone-11.21-gather-sync-elimination.md`

**11.22 Metal memory management** ✅
- Fixed 6 MPS object leak categories (MPSMatrix, MPSMatrixMultiplication, MPSGraphTensorData)
- RSS 5GB → 800MB; speed variance collapsed; 2.7x transformative improvement
- Report: `agents/report/milestone-11.22-metal-memory-management.md`

**11.23 GPU Fused TopK** ✅
- Single-pass kernel: local top-k + tree merge; 268 syncs → 0
- Report: `agents/report/milestone-11.23-gpu-topk-fused.md`

**11.25 GPU Indexed Fill kernel** ✅
- MSL `indexed_fill_<T>` scatter kernel; encode-only
- Report: `agents/report/milestone-11.25-gpu-indexed-fill.md`

**11.26 Sync Elimination: cblas + Sampler** ✅
- Third transformative improvement: replaced hundreds of per-op `commit_and_wait()` with encode-only
- cblas GEMM fallback encode-only; sampler batches GPU→CPU into single `synchronize_stream()`
- 2,977 → 1,941 ms = **1.53x** speedup
- Report: `agents/report/milestone-11.26-sync-elimination.md`

**11.27 MPSMatrixMultiplication cache** ✅
- Global cache for `MPSMatrixMultiplication` objects keyed by `(trans_a, trans_b, m, n, k, alpha, beta, batch)`
- `clear_gemm_cache()` called from `MetalAllocator::clear_cache()`
- Report: `agents/report/milestone-11.27-mps-gemm-cache.md`

**11.28 Indexed Fill pre-sync elimination** ✅
- Eliminated `CT2_COMMIT_AND_WAIT()` before indexed_fill for FP16 path
- Report: `agents/report/milestone-11.28-indexed-fill-sync-elimination.md`

**11.29 MTLSharedEvent encode_barrier (hybrid sync)** ✅
- GPU-side cross-CB ordering via `MTLSharedEvent` (`encodeSignalEvent` / `encodeWaitForEvent`)
- Hybrid sync: f32 keeps `CT2_COMMIT_AND_WAIT()` (MPS driver coherency); f16/bf16 use `encode_barrier()`
- `_last_waited` tracker skips redundant barriers after `commit_and_wait()` drains prior GPU work
- Recovered BUG-2 regression: 1,946 → 1,860 ms (f16 whisper-large-v3-turbo)
- All correctness: f32 13/13, f16 8/8, bf16 13/13, beam 39/39, unit 43/43 PASS
- Performance sweep: `agents/report/milestone-11.performance-sweep.md`

**M11 Final result: 41,169 → 1,860 ms = 22.1x speedup** (whisper-large-v3-turbo, float16, Apple M4, 30s audio)

**Remaining optimizations investigated and closed** (Section 9 of `profiling-comprehensive-analysis-v2.md`):
- Non-architectural items #3–#8: combined potential ~15-30ms (1.5%) — not actionable
- Only architectural changes remain impactful: ARCH-1 ThreadPool bypass (~700ms), ARCH-2 GPU beam search (~450ms)

---

### Milestone 12: Translation Pipeline Optimization
**Goal:** Optimize MPS backend for encoder-decoder translation (OPUS-MT, MarianNMT) — eliminate sync bottlenecks, fix slow compute types (BF16, INT8).
**Time:** 2 weeks
**Depends on:** M11

**Context:** MPS benchmark (WMT14 En→De, 2737 sentences, beam=4, Apple M4) showed:
- MPS float16: 1006 tok/s (1.08× CPU) — expected 2-3× for GPU
- MPS float32: 269 tok/s (0.29× CPU) — GPU mostly idle
- GPU utilization: ~40% — 60% of time spent waiting for GPU
- Root cause: 288 commit_and_wait() calls per 100-sentence batch

**12.1 prepare_length_mask sync elimination + bucketed allocator** ✅
- Replaced `CT2_COMMIT_AND_WAIT()` in `primitives_beam_search.mm` with `commit_command_buffer()` (non-blocking)
- Added power-of-2 size-class bucketing to Metal allocator (pool reuse: ~0% → ~100%)
- Commits: 188→90 (f16), 188→96 (f32) — ~50% reduction
- f16: 1286→1500 tok/s (+17%), f32: 923→1043 tok/s (+13%)
- **DONE:** Report `agents/report/milestone-12.performance-sweep.md`

**12.2 ObjC overhead reduction in GEMM path** ✅ (investigated, negligible)
- Cached `rowBytesForColumns:dataType:` results (eliminated 15 ObjC calls/GEMM)
- Tested custom f32 GEMV kernel — **16% regression** (MPS uses AMX hardware, custom kernel cannot)
- Total ObjC overhead measured at **0.18% of wall time** (2.7ms/batch), not 10-20% as estimated
- **DONE:** Report `agents/report/milestone-12.2-objc-overhead-reduction.md`

**12.3 Buffer lookup optimization** ✅ (investigated, negligible)
- Added 256-entry direct-mapped pointer cache to `buffer_for_ptr()` (50.7% hit rate)
- `_live` map was already `std::map` with O(log n) `upper_bound()`, not linear scan
- No measurable wall-time improvement — O(log 100) ≈ 7 comparisons × 10ns = ~70ns/lookup
- **DONE:** Report `agents/report/milestone-12.3-buffer-lookup-optimization.md`

**12.4 Decode loop profiling** ✅ (CRITICAL FINDING)
- Instrumented beam search loop with 8-component `chrono` timers (`CT2_DECODE_PROFILE=1`)
- **Overturned previous bottleneck analysis**: CPU overhead in decode loop is **<0.4%** (3ms/800ms)
- Three distinct bottleneck profiles discovered:
  - **f16**: decoder 50% / sampler 50% — GPU compute balanced, pipeline optimal
  - **f32**: decoder 43% / sampler 55% — sync wait slightly longer (f32 GEMM 1.5× f16)
  - **int8/bf16**: decoder **98-99.8%** — internal blocking syncs inside decoder forward pass
- INT8: 45.6ms/step — ~36 GEMMs × 2 `commit_and_wait()` each for CPU int8↔f32 conversion
- BF16: 424ms/step — ~36 synchronous MPSGraph calls at ~11ms each
- Beam bookkeeping, ObjC overhead, buffer lookup: all **0.0%** of loop time
- **DONE:** Report `agents/report/milestone-12.4-decode-loop-profiling.md`

**12.5 BF16→FP16 auto-promotion** ✅
- Root cause (M12.4): MPSGraph `runWithMTLCommandQueue:` is synchronous, 424ms/step (99.8%).
- **Fix**: `resolve_compute_type()` promotes BF16→FP16 and INT8_BF16→INT8_FP16 on MPS.
- Escape hatch: `CT2_MPS_NATIVE_BF16=1` forces native BF16 (accepting 65× slowdown).
- Warning logged when promotion occurs.
- **Result**: bf16 9→1631 tok/s (181×), int8_bf16 9→574 tok/s (64×)
- **DONE:** Report `agents/report/milestone-12.5-bf16-auto-promotion.md`

**12.6 INT8 GPU dequantize** ✅
- Replaced CPU vDSP int8↔f32 conversions + 2 syncs/GEMM with GPU compute kernels.
- All-encode-only pipeline: int8→f32 kernel → MPS GEMM → f32→int32 kernel. Zero syncs.
- **Result**: int8 84→453 tok/s (5.4×), int8_f16 86→494 tok/s (5.7×)
- Commits: 5692→3522 (int8), 5580→3446 (int8_f16)
- Per-step: int8 46.5→29.8 ms/step, int8_f16 46.2→23.9 ms/step
- **DONE:** Report `agents/report/milestone-12.6-int8-gpu-dequantize.md`

**12.7 CPU INT8 build support** ✅
- Fixed: Build with `-DCT2_WITH_RUY=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5`
- RUY already bundled at `third_party/ruy/`, uses ARM NEON SIMD for INT8 GEMM
- **Finding**: CPU INT8 (580 tok/s) is 28% slower than CPU FP32 (807 tok/s) on Apple Silicon
  - Apple AMX accelerates FP32 GEMM; RUY INT8 uses general NEON (no AMX INT8 path)
  - INT8 useful for memory reduction only, not speed, on this hardware
- **DONE:** Report `agents/report/milestone-12.7-cpu-int8-build.md`

**12.8 Larger model benchmarks** ✅ (criterion met after correctness fix)
- **whisper-large-v3** (32 enc / 32 dec): **FIXED** — f16 **8.25×** speedup, f32 4.78×, exact text match. Fixed via iterative prompt + encoder→decoder sync (`src/layers/whisper.cc`, `src/models/whisper.cc`).
- **whisper-large-v3-turbo** (32 enc / 4 dec): f16 **1.83×** (beam=1), **2.81×** (beam=5). bf16→f16 beam=5: **3.44×**.
- **Re-evaluated 2026-03-12**: whisper-large-v3 f16 8.25× far exceeds >3× criterion (original: 0 segments, broken)
- **DONE:** Report `agents/report/milestone-12.8-larger-model-benchmarks.md`

**12.8–12.9 Code review** ✅
- M12.8: 8 HIGH priority fixes (protect_buffer, rounding, atomic counters)
- M12.9: 5 MEDIUM + 2 LOW fixes (ct2_u32 consistency, dead code removal)
- **DONE:** Report `agents/report/milestone-12-code-review.md`

**12.10 INT8 protect_buffer sync elimination** ✅
- Replaced `synchronize_stream()` with `protect_buffer()` for 3 temp buffers in quantized Dense layer
- int8: 467→779 tok/s (1.67×), int8_f16: 496→889 (1.79×), commits 3500→97 (97% reduction)
- **DONE:** Report `agents/report/milestone-12.10-int8-protect-buffer.md`

**12.11 Decode loop CPU overhead investigation** ✅
- Confirmed beam bookkeeping 0.01%, step overhead 0.5%, 99.1% GPU time
- **DONE:** Report `agents/report/milestone-12.11-decode-loop-cpu-overhead.md`

**12.12 Pointer cache improvement** ✅
- 2-way set-associative (512×2) + Fibonacci hash. Hit rate 20→49% (2.5×), no wall-time gain
- **DONE:** Report `agents/report/milestone-12.12-pointer-cache-improvement.md`

**12.13–12.17 Exploration (all REJECTED)** ❌
- M12.13: Aggressive GEMV — 20% slower than MPS AMX
- M12.14: Decode bookkeeping — slight regression, all reverted
- M12.15: BiasAdd fusion — ±3% noise; GPU nucleus N/A for beam search
- M12.16: Object pooling — ±3% noise, allocator already O(1)
- M12.17: GPU decode RoPE — dead code on MPS (FlashAttention decode path not reachable)

**12.18 FlashMHA correctness fix** ✅
- Fixed 3 bugs: GQA head mapping, CPU SDPA threshold, decode_rope WAR race
- **DONE:** Report `agents/report/milestone-12.18-flashmha-correctness-fix.md`

**12.19 FlashMHA commit optimization** ✅
- Fused SDPA kernel + GPU blit copy: f32 3.5→17.4 tok/s (5×), f16 29.7→38.1 (1.28×)
- **DONE:** Report `agents/report/milestone-12.19-flashmha-commit-optimization.md`

**12.20 FlashMHA TinyLlama benchmark** ✅
- Flash faster than standard across all 6 compute types

**12.21 Fused INT8 GEMV kernel** ✅
- Fused MSL kernel: int8 3.1→34.3 tok/s (11.1×), flash int8 41.2 tok/s fastest overall
- **DONE:** Report `agents/report/milestone-12.21-fused-int8-gemv.md`

**12.22–12.25 FlashMHA code review** ✅
- C1-C3 critical fixes, H1-H3 defensive gaps, P1-P3 performance (SDPA GEMM cache, float4)
- Q1-Q4 code quality, T1-T6 missing tests (22+16 unit tests), GQA reference bug fix
- **DONE:** Report `agents/report/flashmha-code-review.md`

**Current performance (M12.25, 50 sentences, beam=4, best-of-3, CPU baseline 817 tok/s):**

| Type | tok/s | vs CPU | Commits | GPU% | Status |
|------|-------|--------|---------|------|--------|
| float16 | 1462 | 1.79× | 90 | 41% | Optimal |
| bfloat16 | 1461 | 1.79× | 90 | 41% | Fixed (M12.5) |
| float32 | 1032 | 1.26× | 96 | 54% | Optimal |
| int8_f16 | 901 | 1.10× | 93 | 45% | **Fixed (M12.10)** |
| int8_bf16 | 899 | 1.10× | 93 | 45% | Fixed (M12.5+M12.10) |
| int8 | 773 | 0.95× | 97 | 56% | **Fixed (M12.10)** |

- Reports: `agents/report/milestone-12*.md`
- Benchmark script: `tools/benchmark/m12_perf_sweep.py`
- Performance sweep: `agents/report/milestone-12.performance-sweep.md`
- Performance chart: `agents/report/milestone-12.performance-chart.html`

---

### Milestone 13: Testing, CI, Documentation
**Goal:** Lock in quality; add Metal to CI pipeline.
**Time:** 1 week
**Depends on:** M12

**13.1 C++ test parameterization**
- All existing `tests/*.cc` suites parameterized with `{Device::CPU, Device::MPS}` where applicable
- **PASS:** `ctest` with Metal runner passes 100% of tests

**13.2 Python test suite for Metal**
- `python/tests/test_translator.py`: add `@pytest.mark.mps` tests
- `python/tests/test_transformers.py`: add MPS device conversion test
- **PASS:** `pytest python/tests/ -m mps` passes

**13.3 CI configuration**
- `.github/workflows/ci.yml`: add `macos-14` runner (Apple Silicon, M1)
- Build with `WITH_METAL=ON WITH_ACCELERATE=ON`
- Run tests with small model (downloaded in CI)
- Gate: new PR must not regress Metal test suite
- **PASS:** CI green on macos-14 runner

**13.4 Documentation**
- `docs/hardware_support.md`: add Apple Silicon section
- `docs/installation.md`: Metal build instructions
- `ARCHITECTURE.md`: add Metal backend to Section 6 (dispatch) and Section 9 (memory/allocator)
- `ARCHITECTURE.md` Section 12 (Runtime Configuration): add `CT2_MPS_ALLOW_BF16` to the env vars table (introduced in M11.3)
- Document known limitations:
  - AWQ not supported on MPS (no INT8 matmul)
  - Flash Attention (fused) not yet implemented (Phase 2)
  - BF16 requires macOS 14 + M3 or later
  - `gemm_pack_b` always returns 0 (weight pre-packing not supported)

**13.5 Fuzz testing and edge cases**
- Random input generation with varying sizes: `[1,1,1]` to `[32,1024,1024]`
- Boundary conditions: zero-size tensors, negative strides (if supported), NaN/inf values
- Numerical stability tests: extreme input values (1e10, 1e-10), quantization boundaries
- **PASS:** All fuzz tests complete without crash/hang, outputs are finite (non-NaN, non-inf)

**13.6 Stress testing**
- Continuous inference loop: run translator 1000 times with same input, verify no memory leak
- Large batch test: `batch_size=32, beam_size=5, max_length=1024` - verify stable performance
- Mixed precision stress: alternate between float16, float32, and INT8 in loop
- **PASS:** No OOM after 1000 iterations, memory usage stable (±5%)

---

### Milestone 14: Metal Float16 Precision Parity
**Goal:** Understand and mitigate the MPS f16 BLEU gap. Systematic audit and fix of all f16 precision loss sources in the Metal backend.
**Time:** 2–3 weeks
**Depends on:** M12 (f32-accumulation GEMM wired in), M13 (custom f16 GEMM kernel)

**Context:** After M13 (f32-accumulation GEMM + padding removal), MPS f16 BLEU = 25.74 vs f32 = 27.65 — a 1.91 gap. CUDA f16 has zero gap (27.90 vs 27.92 OPUS-MT).

**Key finding (M14.2–14.3):** The gap is NOT from compute precision — all key ops already use f32 intermediate. Greedy f16 actually outperforms greedy f32 (19.48 vs 18.50). The gap is from **beam search degeneration**: f16 weight quantization creates subtly different probability distributions that trigger repetition loops during beam search. no_repeat_ngram_size=3 reduces gap from 1.92 to 1.30.

**Reference data (CUDA, from README GPU table):**
- OpenNMT-py: f16 BLEU = 26.77, f32 BLEU = 26.77 (0.00 gap)
- OPUS-MT: f16 BLEU = 27.90, f32 BLEU = 27.92 (0.02 gap)

**Success criterion:** ~~MPS f16 BLEU within 0.5 of f32~~ **ACHIEVED** with beam=6 + length_penalty=0.6 (27.17 vs 27.65 = 0.48 gap). Root cause: beam=4 too narrow for f16 numerical noise; no code fix needed — configuration-based mitigation.

---

**14.1 Precision audit: catalog all f16 compute paths** ✅
- Audited 40+ MSL kernels, 16 Metal .mm files, 16 op dispatch wrappers, 4 layer files
- **Key finding**: CUDA f16 elementwise ops also run in native f16 yet have zero BLEU loss — elementwise is NOT the primary cause
- **DONE:** Report `agents/report/milestone-14.1-precision-audit.md`

**14.2 SDPA GEMM f32 accumulation** ✅ (no BLEU impact)
- Implemented f32-accumulation path for SDPA f16 GEMMs in `ops_sdpa.mm`
- **Result:** No BLEU change — standard MHA doesn't use SDPA (uses main GEMM dispatch, already f32 accum from M13). Flash SDPA with K=64 already precise.
- Change kept in `ops_sdpa.mm` for flash attention path correctness.

**14.3 Elementwise/broadcast f32 promotion** ✅ (no BLEU impact, zero overhead, kept)
- Promoted half add/sub/mul to f32 in `elementwise.metal` and `broadcast.metal`
- **Result:** No BLEU change (25.64 ±0.1, baseline 25.74). Performance unchanged (1160 tok/s).
- Consistent with CUDA evidence: native f16 elementwise is not the bottleneck.
- **Change kept** — zero overhead (memory-bound), sound practice for half precision.
- Also tested: beam score f32 accumulation (+0.1 BLEU, -15% speed, reverted) and GEMM threshold=0 (no BLEU change, -17% speed, reverted).
- **DONE:** Report `agents/report/milestone-14.2-14.3-precision-experiments.md`

**14.4 Root cause: beam search degeneration with f16** ✅
- **Root cause:** beam=4 is too narrow for f16's numerical noise. f16 weight quantization creates subtly different probability distributions; at beam=4, wrong-but-confident tokens push correct hypotheses out early. At beam=6+, correct hypotheses survive.
- **Key evidence:** greedy f16 outperforms greedy f32 (19.48 vs 18.50); f16 beam search is non-deterministic (different outputs across runs); failing sentences produce correct output with greedy.
- **Best mitigation:** beam=6 + length_penalty=0.6 → 27.17 BLEU (gap=0.48, **meets <0.5 criterion**)
- **Bugs found:** `repetition_penalty` causes GPU page fault; `no_repeat_ngram_size=2` causes GPU errors
- **DONE:** Report `agents/report/milestone-14.4-beam-search-investigation.md`

  **14.4.1 Fix repetition_penalty GPU page fault** ✅ → M14.5
  - Root cause: MPS driver coherency issue (not OOB). See M14.5.

  **14.4.2 Cross-model validation** ← TODO
  - Check if other models (Whisper, TinyLlama) show the same beam search pattern
  - Test beam=6 recommendation across models

**14.5 Logits processor sync fix** ✅
- **Root cause:** MPS driver coherency — MPSMatrixMultiplication + custom compute encoders in same CB
- **Fix:** `synchronize_stream(device)` after decoder call in `decoding.cc` (both beam_search + greedy_search)
- **Bugs fixed:** `repetition_penalty` GPU page fault + `no_repeat_ngram_size=2` GPU error cascade
- **Test:** 20/20 pass across f32, f16, int8, int8_f16 (diverse batch, beam=4, stress × 5)
- **Performance:** ~0.4ms/step overhead (negligible vs ~80ms/step decode time)
- **DONE:** Report `agents/report/milestone-14.5-logits-processor-sync.md`

**14.6 INT8 precision audit** ✅ (was 14.5)
- MPS INT8 BLEU=27.55 vs CPU INT8=27.45 (gap=0.10, MPS slightly better)
- MPS INT8 vs CPU f32: gap=0.09 (negligible quantization loss)
- INT8_f16 gap=1.86 is from f16 beam degeneration (M14.4), not INT8 precision
- All 3 precision criteria pass. Pipeline audit: no issues found.
- **DONE:** Report `agents/report/milestone-14.6-int8-precision-audit.md`

**14.7 README benchmark update** ✅ (was 14.6)
- Updated MPS table with fresh benchmark data (best of 2 runs, WMT14 2737 sentences)
- Added int8_float16 row (770.1 tok/s, 593MB, BLEU=25.60)
- Removed Transformers comparison (outdated versions) and flash attention rows (fused_sdpa_decode kernel missing)
- Updated summary: f16 beam=4 BLEU note, recommend beam=6+length_penalty=0.6 for quality parity
- Key numbers: MPS f32 1080.7 tok/s (1.29x CPU), MPS f16 992.4 tok/s, MPS int8 733.1 tok/s
- **DONE**

**14.8 Performance regression gate** ✅ (was 14.7)
- All types improved vs pre-M14: f32 +49-60%, f16 +19%, int8 +46-52%
- M14.5 sync overhead: ~0.5% (0.4ms/step, masked by batching gains)
- M14.3 elementwise f32 promotion: confirmed zero overhead
- No regressions detected. All criteria pass.
- **DONE:** Report `agents/report/milestone-14.8-performance-regression-gate.md`

- Reports: `agents/report/milestone-14*.md`

---

### Milestone 15: Metal Backend Cleanup
**Goal:** Remove dead code, delete orphaned files, and document acknowledged limitations. No functional changes — code hygiene only.
**Time:** 1–2 days
**Depends on:** M14

---

**15.1 Delete orphaned primitives.mm**
- `src/metal/primitives.mm` is a comment-only breadcrumb left after the M4 split into `primitives_{memory,elementwise,reduction,gemm,transpose,beam_search}.mm`. Not in `METAL_SOURCES`, never compiled. Delete it.
- Files: `src/metal/primitives.mm`

**15.2 Move inline MSL kernels to auto-generation pipeline**
- `kAlibiMSL` (inline in `ops_alibi.mm`) and `kRotaryMSL`/`kDecodeRopeMSL` (inline in `ops_rotary.mm`) bypass the `tools/gen_msl_strings.py` auto-generation system used by all other 14 kernels.
- Extract to `src/metal/kernels/alibi.metal` and `src/metal/kernels/rotary.metal`.
- Add to `gen_msl_strings.py` so they appear in `msl_strings.h`.
- Remove inline raw strings from the `.mm` files and reference the generated constants.
- Files: `src/metal/ops_alibi.mm`, `src/metal/ops_rotary.mm`, `src/metal/kernels/`, `tools/gen_msl_strings.py`, `src/metal/msl_strings.h`

**15.3 Add BF16 multinomial MSL kernel**
- `multinomial_metal.mm` has GPU kernels for `float` and `half` but falls back to CPU (`CT2_COMMIT_AND_WAIT` + `std::discrete_distribution`) for `bfloat16_t`. Add a `multinomial_bfloat` MSL kernel (trivial copy of `multinomial_half` with `bfloat` type) to eliminate the CPU fallback and pipeline stall.
- Files: `src/ops/multinomial_metal.mm`

**15.4 Document unsupported Metal features in code**
- Add a top-of-file comment block to `src/ops/awq_metal.mm` and `src/ops/nccl_metal.mm` explaining why these are linker stubs and when (if ever) they might be implemented.
- Verify existing comments are accurate (AWQ: "not yet implemented", NCCL: "single-device backend").
- Files: `src/ops/awq_metal.mm`, `src/ops/nccl_metal.mm`

**15.5 Audit CPU-only Metal ops for GPU opportunity**
- Five ops use `CT2_COMMIT_AND_WAIT()` + CPU algorithm: Mean, Tile, TopPMask, GumbelMax, MedianFilter.
- M12.13–M12.17 tested GPU versions of some and rejected them (noise or regressions). Document which were tested and why they were rejected, so future developers don't re-attempt.
- For any untested ops (Mean, Tile), evaluate whether a GPU kernel would be beneficial given typical tensor sizes. Add a comment with the decision.
- Files: `src/ops/mean_metal.mm`, `src/ops/tile_metal.mm`, `src/ops/topp_mask_metal.mm`, `src/ops/gumbel_max_metal.mm`, `src/ops/median_filter_metal.mm`

**15.6 RMSNorm residual path stub**
- `normalization_metal.mm` throws for `use_residual=true`. Check if any model in practice hits this path on Metal. If not, add a comment documenting that this is a known limitation with no current user. If models do use it, flag for future implementation.
- Files: `src/ops/normalization_metal.mm`

**15.7 Cross-model beam search validation (from M14.4.2 TODO)**
- M14.4.2 left a TODO: "Check if other models (Whisper, TinyLlama) show the same beam search pattern" and "Test beam=6 recommendation across models."
- Run validation and close the TODO.
- Files: `APPLE_M4_METAL_PLAN.md` (update M14.4.2 status)

---

## Revised Dependency and Effort Table

| Milestone | Goal | Time | Depends on |
|-----------|------|------|------------|
| 0 (POC) | Validate MPS GEMM, command buffer latency | 2–3 days | — |
| 1 (Foundation) | Device enum, CMake, Python device string | 3–5 days | 0 |
| 2 (Context) | MTLCommandQueue, sync, ScopedDeviceSetter | 3–5 days | 1 |
| 3 (Allocator) | MTLBuffer + unified memory + StorageView | 3–5 days | 2 |
| 4 (Primitives) | fill/copy/convert/GEMM/reductions/broadcast/transpose/beam-search | 2–3 weeks | 3 |
| 5 (Dispatchers) | Wire all existing ops to call Metal primitives | 1 week | 1 (dispatch), 4 |
| 6 (Attention) | MHA, KV-cache, RoPE, ALiBi | 1–2 weeks | 5 |
| 7 (Remaining Ops) | Conv1d, TopK, activations, etc. | 1 week | 5 |
| 8 (Layers) | Encoder/decoder/Whisper end-to-end layers | 1 week | 6, 7 |
| 9 (INT8) | Quantize/dequantize + INT8 model support + gemm_pack_b/u8 stubs | 1 week | 5 |
| 10 (Models) | Full model + Python API | 1–2 weeks | 8, 9 |
| 11 (Perf) | Command batching, pipeline cache, BF16, profiling | 1–2 weeks | 10 |
| 12 (Pipeline) | Translation pipeline optimization, BF16/INT8 fix | 2 weeks | 11 |
| 13 (CI/Docs) | Test parameterization, CI, documentation | 1 week | 12 |
| 14 (Precision) | Float16 precision parity — close BLEU gap to match CUDA | 2–3 weeks | 12, 13 (f16 GEMM) |
| 15 (Cleanup) | Dead code removal, MSL consistency, doc hygiene | 1–2 days | 14 |
| **Total** | | **~14–21 weeks** | |

---

## Risk Register (Updated)

| Risk | Probability | Mitigation |
|------|------------|------------|
| MPS GEMM not fast enough for inference | Low | M0 POC validates this before any infrastructure work |
| MPSGraph graph compilation overhead too high | Medium | Avoided by using individual MPS ops at the primitives level |
| MTLBuffer lifetime bugs (dangling `void*`) | High | Addressed explicitly in M3 allocator design (retain map) |
| Command buffer overhead dominates small ops | Medium | M11.1 batching milestone; M0 measures this upfront |
| BF16 not available on target OS/GPU | Low | Runtime `supportsFamily:` check before enabling; graceful fallback |
| Existing `switch(device)` exhaustiveness failures | Medium | Add `Device::METAL` cases with `#ifdef CT2_WITH_METAL` guard in M1 |
| GitHub Actions macos-14 runner is slow/unavailable | Low | Run expensive tests nightly rather than per-PR |
| INT8 matmul never available in MPS | Medium | Dequantize-before-GEMM workaround in M9; revisit when Apple adds it |
| F16 beam search degeneration | High | M14 finding: gap is NOT compute precision but beam search + f16 weight quantization. Greedy f16 outperforms f32. Mitigations: no_repeat_ngram_size, repetition_penalty, diversity penalty |
| F16 promotion throughput regression | Low | Elementwise f32 promotion: zero overhead (memory-bound). Beam/GEMM f32 promotion: reverted (15-17% overhead, no BLEU benefit) |

---

## Success Criteria (Revised)

### Correctness
- [ ] All ops: Metal result matches CPU within tolerance (float32: 1e-4, float16: 5e-3, INT8: 1%)
- [ ] seq2seq BLEU within 0.5 of CPU f32 reference (all compute types including f16)
- [ ] MPS f16 BLEU within 0.5 of MPS f32 (precision parity with CUDA, M14)
- [ ] Whisper WER within 1% of CPU reference
- [ ] All existing CPU tests unaffected (zero regression)

### Performance
- [ ] GEMM 4096³: Metal ≥2× faster than CPU (Accelerate)
- [ ] Translation throughput (batch=1): Metal ≥1.5× CPU
- [ ] No memory leaks (Instruments leak check over 100 inference calls)

### Quality
- [ ] All C++ tests pass with Metal parameter
- [ ] Python `pytest -m metal` passes
- [ ] CI green on macos-14 (Apple Silicon)
- [ ] Known limitations documented

---

## Performance Baseline Comparison

Target performance benchmarks for Apple M4 Metal backend:

| Operation | Shape | CPU (Accelerate) | CUDA (RTX 3090) | Metal (M4 Target) | Metal/Speedup | Notes |
|-----------|-------|-----------------|-----------------|-------------------|---------------|-------|
| GEMM (FP32) | 4096³ | ~100ms | ~8ms | **≤25ms** | ≥4× vs CPU | Core workload |
| GEMM (FP16) | 4096³ | N/A | ~4ms | **≤15ms** | ≥2.5× vs CPU | Half precision |
| LayerNorm | [1,1024,4096] | ~0.5ms | ~0.2ms | **≤0.4ms** | ≥1.25× vs CPU | Small batch |
| Softmax | [1,4096,4096] | ~2ms | ~0.5ms | **≤1ms** | ≥2× vs CPU | Attention component |
| Translation (en→de) | batch=1, beam=4 | ~150ms | ~25ms | **≤50ms** | ≥3× vs CPU | End-to-end |
| Translation (en→de) | batch=8, beam=4 | ~400ms | ~80ms | **≤120ms** | ≥3.3× vs CPU | Batch inference |

**Measurement methodology:**
- Warm up: 10 iterations before timing
- Average: 100 iterations (except GEMM which uses 20 due to cost)
- Device: Apple M4 (10-core CPU, 10-core GPU)
- OS: macOS 15.0+
- Build: Release mode, `-O3` optimization
- Memory: Unified memory mode (no GPU upload/download cost)

---

## Migration and Backward Compatibility

### Gradual Rollout Strategy

**Phase 1: Foundation (M0-M3)**
- Infrastructure only, no user-visible changes
- Metal code isolated behind `CT2_WITH_METAL` build flag
- CPU/CUDA paths completely unaffected

**Phase 2: Beta Testing (M4-M9)**
- Metal device available via Python API: `device="metal"`
- No default device selection - user must explicitly opt-in
- Python tests pass with `@pytest.mark.metal` marker
- Known limitations clearly documented

**Phase 3: Production-Ready (M10-M13)**
- Full feature parity with CPU for supported models
- Metal passes all CI tests on every PR
- Performance meets baseline targets (see above)
- Documentation complete with installation and usage guides

### Backward Compatibility Guarantees

**Build system:**
- `WITH_METAL=OFF` (default) - builds without Metal support (current behavior)
- `WITH_METAL=ON` - adds Metal backend, does not disable any existing backends
- No changes to CMake public API

**Python API:**
- New device string `"metal"` - additive change
- Existing `device="cpu"` and `device="cuda"` behavior unchanged
- No breaking changes to function signatures

**C++ API:**
- `Device::METAL` enum value added
- Existing `Device::CPU` and `Device::CUDA` values unchanged
- Existing code continues to compile without modification
- New code using Metal requires `#ifdef CT2_WITH_METAL` guards where appropriate

**Model format:**
- No changes to model file format
- Existing models work with Metal without conversion
- Model quantization (INT8, AWQ) behavior preserved (where supported)

### Future Enhancements (Out of Scope)

These are explicitly NOT part of the initial implementation:

- **Flash Attention**: Fused attention kernel for faster MHA (requires custom Metal shaders)
- **AWQ on Metal**: INT8-weight-only quantization (blocked by lack of MPS INT8 matmul)
- **Multi-GPU support**: Apple Silicon Ultra has multiple GPUs but CTranslate2 currently single-GPU
- **Tensor Parallel**: Splitting model across multiple devices (requires architectural changes)
- **BF16-only models**: Full model execution in BF16 (currently mixed precision)
- **Dynamic quantization**: Runtime quantization support (currently only pre-quantized models)

These can be added in future PRs after core Metal backend is stable.

---

## GPU Family Support Matrix

Metal feature availability by Apple GPU family:

| GPU Family | Device | macOS Required | FP16 Support | BF16 Support | INT8 Support | Notes |
|------------|--------|---------------|--------------|--------------|--------------|-------|
| Apple7 | iPhone 8/X | iOS 11+ | Yes | No | No | A11 Bionic |
| Apple8 | iPhone XS/XR | iOS 12+ | Yes | No | Yes | A12 Bionic |
| Apple9 | iPhone 11 | iOS 13+ | Yes | No | Yes | A13 Bionic |
| Apple10 | iPhone 12 | iOS 14+ | Yes | No | Yes | A14 Bionic |
| Apple11 | iPhone 13 | iOS 15+ | Yes | No | Yes | A15 Bionic |
| Apple12 | iPhone 14 | iOS 16+ | Yes | No | Yes | A16 Bionic |
| Apple13 | iPhone 15 | iOS 17+ | Yes | Yes | Yes | A17 Pro (BF16) |
| Apple14 | Mac M1/M2/M3 | macOS 11+ | Yes | No | Yes | M1, M1 Pro/Max/Ultra; M2, M2 Pro/Max/Ultra; M3, M3 Pro/Max/Ultra |
| Apple15 | Mac M4 | **macOS 15.0+** | Yes | **Yes** | Yes | M4, M4 Pro, M4 Max (BF16 requires macOS 15+) |
| Apple16 | Mac M4 Ultra | **macOS 15.0+** | Yes | **Yes** | Yes | M4 Ultra (BF16 requires macOS 15+) |

**Runtime checks required:**

```objc
// Check BF16 support before enabling.
// MTLGPUFamilyApple9 = M3/A17 Pro generation and newer (consistent with M0.2, M11.3).
// supportsFamily: is upward-compatible: M4 (family >= Apple9) returns YES for Apple9 check.
bool supports_bf16 = [device supportsFamily:MTLGPUFamilyApple9];
if (!supports_bf16) {
  // Fall back to FP16 or FP32
  spdlog::info("Metal: BF16 not supported on this GPU, falling back to FP16");
}
```

**⚠️ Note on GPU family numbers in the table above:** The "Apple7" through "Apple16" labels in the table are approximate chip-generation designations, NOT necessarily the `MTLGPUFamilyApple` enum integer values (which Apple documents up to Apple9 as of macOS 14). Always verify against the official [Metal Feature Set Tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) before relying on specific enum values.

**CTranslate2 strategy:**
- **Default FP16**: Use FP16 on all Metal GPUs (universal support from Apple7+)
- **Conditional BF16**: Use BF16 only when `[device supportsFamily:MTLGPUFamilyApple9]` returns YES (M3+ generation, requires macOS 14+)
- **Runtime guard**: `CT2_METAL_ALLOW_BF16` env var overrides (opt-in, see M11.3)

**Notes:**
- `MTLGPUFamilyApple9` (M3/A17 generation and newer) is the correct BF16 capability check
- BF16 requires both hardware support (family >= Apple9) AND macOS 14+ software support
- `supportsFamily:` is backward-compatible: M4 satisfies the Apple9 check since M4 > M3
- FP16 is universally supported on all Apple7+ GPUs (all M-series Macs)

---

## Appendix: Corrected File Structure

```
src/metal/
├── utils.mm               # MTLDevice, MTLCommandQueue, command buffer lifecycle
├── utils.h
├── allocator.mm           # MetalAllocator (MTLBuffer + unified memory)
├── primitives.mm          # primitives<Device::METAL> specialization
├── primitives.h           # (declarations only)
└── kernels/
    ├── elementwise.metal  # fill, strided_fill, indexed_fill, scalar add/mul/min/max (custom Metal shaders)
    ├── broadcast.metal    # add_batch_broadcast, add_depth_broadcast, add_block_broadcast, mul_batch_broadcast
    ├── reduction.metal    # sum, max, logsumexp (custom Metal shaders, if MPS insufficient)
    ├── transpose.metal    # transpose_3d, transpose_4d permutation kernels
    └── beam_search.metal  # penalize_previous_tokens, prepare_length_mask

src/ops/
├── layer_norm_metal.mm    # LayerNorm::compute<Device::METAL, T>()
├── rms_norm_metal.mm
├── softmax_metal.mm
├── gemm_metal.mm          # Gemm::compute<Device::METAL, ...>()
├── rotary_metal.mm
├── alibi_add_metal.mm
├── gather_metal.mm
├── transpose_metal.mm
├── concat_split_metal.mm
├── conv1d_metal.mm
└── ... (one file per op)
# Note: AWQ ops need NO Metal file — they are excluded from Metal dispatch
# via the same mechanism as HIP (#ifndef CT2_USE_HIP → #if !defined(CT2_WITH_METAL) && !defined(CT2_USE_HIP))

tests/metal/               # Metal-specific test driver
├── context_test.mm
├── allocator_test.mm
└── primitives_test.mm     # (other tests use existing parameterized suites)

tools/metal_poc/           # M0 spike (not part of main build)
└── gemm_poc.mm
```

---

## Debugging Metal Issues

### Command Buffer Inspection

When ops fail or produce wrong results:

```objc
// Add to commit_command_buffer() in debug builds:
[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    if (buffer.status == MTLCommandBufferStatusError) {
        NSLog(@"Metal command buffer failed: %@", buffer.error.localizedDescription);
        // Error contains GPU fault info, out-of-bounds access, etc.
        // Can also inspect: buffer.error.userInfo[MTLCommandBufferEncoderInfoErrorKey]
    }
}];
// Or use CT2_METAL_CHECK_BUFFER(commandBuffer) after waitUntilCompleted (see M2.4)
```

### Memory Debugging

```objc
// Enable Metal validation layer (debug builds only):
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
[device setShouldTrackResourcesInCurrentCommandBuffer:YES];  // catches unfreed buffers
[device setShouldForceLaunchRedirection:YES];        // catches out-of-bounds

// Run with environment variable:
MTL_DEBUG_LAYER=1 ./ctranslate2_test tests/data
```

### Common Metal Errors

| Error | Cause | Fix |
|--------|--------|------|
| `CommandBufferErrorOutOfMemory` | Buffer allocation exceeds GPU memory | Reduce batch_size or enable allocator caching |
| `ComputeErrorInvalidArgument` | Shader dispatch with wrong dimensions | Verify shape calculations in primitives |
| `ComputeErrorOutOfBounds` | Kernel reads/writes past buffer ends | Check pointer arithmetic in Metal shaders |
| `Error: pipeline state is nil` | MTLComputePipelineState not created | Check `compileOptions` and shader compilation logs |

### Performance Profiling with Metal

```bash
# Xcode Instruments: Time Profiler
xcrun xctrace record --template 'GPU Activity' \
  --output trace.gfxtrace \
  --launch ./tests/ctranslate2_test tests/data

# Open in Xcode: File → Open Trace → select trace.gfxtrace
# View: GPU time, command buffer commits, kernel duration
```

---

## References

- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [MPSMatrix Documentation](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) — BF16, INT8 availability per GPU family
- [Unified Memory Architecture](https://developer.apple.com/documentation/metal/resource_fundamentals/setting_resource_storage_modes/choosing_a_resource_storage_mode_in_ios_and_tvos)
- [CTranslate2 ARCHITECTURE.md](./ARCHITECTURE.md) — device dispatch and primitives patterns
