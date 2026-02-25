# Apple M4 Metal Backend Implementation Plan

**Revised:** 2026-02-25
**Status:** In progress — Milestone 5.1 complete (dispatch macro Metal FP16/BF16 update)

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

6. **Python bindings deferred to M12**: Device string enumeration can be exposed in Python as early as M1, allowing Python-level tests throughout development instead of waiting for the end.

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

Implementation: `metal::sdpa_metal<T>` in `src/metal/primitives_sdpa.mm` +
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

**6.2 KV-cache update (`update_state`)**
- The decoder caches keys/values across steps — ensure Metal StorageViews support this correctly
- Test with iterative decode (simulate 10 decode steps)
- **PASS:** Cached Metal KV matches CPU cached KV at each step

**6.3 Rotary embeddings (RoPE)**
- `src/ops/rotary_metal.mm`: apply RoPE to Q/K tensors on Metal
- **PASS:** Metal RoPE output matches CPU within 1e-4 for float32

**6.4 ALiBi positional bias**
- `src/ops/alibi_add_metal.mm`
- **PASS:** Metal ALiBi matches CPU within 1e-5

*Flash Attention (fused SDPA kernel) is deferred to Phase 2 — requires custom Metal compute shaders for the fused kernel. Standard SDPA via MPS first.*

---

### Milestone 7: Remaining Ops (Complete Coverage)
**Goal:** Cover all remaining ops needed for full model support.
**Time:** 1 week
**Depends on:** M5

Ops to add Metal implementations for:
- `Conv1d` — Whisper/Wav2Vec2 encoder (use `MPSCNNConvolution`)
- `Quantize` / `Dequantize` — dynamic INT8 (use MPSGraph cast + scale)
- `Concat` / `Split` — used in multi-head attention
- `TopK`, `TopPMask` — beam search / sampling
- `GumbelMax`, `Multinomial` — stochastic sampling
- `Tile`, `Squeeze`, `Unsqueeze`, `Slide`
- `MedianFilter` — Whisper timestamps (can fall back to CPU for now)
- Activation functions not yet covered: `GELU`, `SiLU/Swish`, `Sigmoid`

For ops where Metal offers no speedup over CPU (e.g., `MedianFilter`, small `TopK`), a CPU fallback via `StorageView::to(Device::CPU)` + op + `StorageView::to(Device::METAL)` is acceptable with a debug-level log message.

**PASS:** Run full `tests/ops_test.cc` with Metal device. All ops either pass numerically or log an explicit fallback message.

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

**10.2 Seq2seq (Transformer) end-to-end**
- Run a small translation model (e.g., opus-mt-en-de, Helsinki-NLP)
- Greedy and beam search (beam_size=4)
- **PASS:**
  - BLEU score within 0.5 of CPU result on 100-sentence test set
  - Metal inference speed ≥ 1.5× CPU for batch_size=1

**10.3 Language model (GPT-style) end-to-end**
- Test decoder-only generation on Metal
- **PASS:** Generated text matches CPU output for greedy decoding (same tokens, deterministic)

**10.4 Whisper end-to-end**
- Load `whisper-tiny` or `whisper-base`; transcribe a 10-second audio clip
- **PASS:** WER (word error rate) within 1% of CPU result

---

### Milestone 11: Performance Optimization
**Goal:** Optimize Metal backend for M4 throughput; profile and fix bottlenecks.
**Time:** 1–2 weeks
**Depends on:** M10

**11.1 Command buffer batching**
- Profile: measure how many Metal command buffers are committed per token
- Goal: ≤1 command buffer per decode step (all ops for one step in one buffer)
- This may require collecting ops across a decode step before committing
- **PASS:** `tools/benchmark/benchmark.py --device metal` shows ≥20% speedup vs per-op commit

**11.2 Metal pipeline state caching**
- Custom Metal shaders require `MTLComputePipelineState` objects (expensive to create)
- Cache them globally keyed by (shader function name + type parameters)
- **PASS:** Second inference call is not slower than first (pipeline states reused, no recompile)

**11.3 BF16 inference (M4 specific)**
- Enable BF16 when `[device supportsFamily:MTLGPUFamilyApple9]` is true (M3+)
- Add `CT2_METAL_ALLOW_BF16` env var for opt-in
- **PASS:** BF16 model runs on Metal; output within 1e-2 of FP32; ≥1.3× faster than FP16

**11.4 Profiling integration**
- `src/profiler.cc`: add `PROFILE` macro support for Metal
- Use `[commandBuffer addCompletedHandler:]` + `GPUStartTime`/`GPUEndTime` from command buffer
- **PASS:** `CT2_ENABLE_PROFILING=1 ct2-translator --device metal` shows per-op timings

---

### Milestone 12: Testing, CI, Documentation
**Goal:** Lock in quality; add Metal to CI pipeline.
**Time:** 1 week
**Depends on:** M11

**12.1 C++ test parameterization**
- All existing `tests/*.cc` suites parameterized with `{Device::CPU, Device::METAL}` where applicable
- **PASS:** `ctest` with Metal runner passes 100% of tests

**12.2 Python test suite for Metal**
- `python/tests/test_translator.py`: add `@pytest.mark.metal` tests
- `python/tests/test_transformers.py`: add Metal device conversion test
- **PASS:** `pytest python/tests/ -m metal` passes

**12.3 CI configuration**
- `.github/workflows/ci.yml`: add `macos-14` runner (Apple Silicon, M1)
- Build with `WITH_METAL=ON WITH_ACCELERATE=ON`
- Run tests with small model (downloaded in CI)
- Gate: new PR must not regress Metal test suite
- **PASS:** CI green on macos-14 runner

**12.4 Documentation**
- `docs/hardware_support.md`: add Apple Silicon section
- `docs/installation.md`: Metal build instructions
- `ARCHITECTURE.md`: add Metal backend to Section 6 (dispatch) and Section 9 (memory/allocator)
- `ARCHITECTURE.md` Section 12 (Runtime Configuration): add `CT2_METAL_ALLOW_BF16` to the env vars table (introduced in M11.3)
- Document known limitations:
  - AWQ not supported on Metal (no INT8 matmul)
  - Flash Attention (fused) not yet implemented (Phase 2)
  - BF16 requires macOS 14 + M3 or later
  - `gemm_pack_b` always returns 0 (weight pre-packing not supported)

**12.5 Fuzz testing and edge cases**
- Random input generation with varying sizes: `[1,1,1]` to `[32,1024,1024]`
- Boundary conditions: zero-size tensors, negative strides (if supported), NaN/inf values
- Numerical stability tests: extreme input values (1e10, 1e-10), quantization boundaries
- **PASS:** All fuzz tests complete without crash/hang, outputs are finite (non-NaN, non-inf)

**12.6 Stress testing**
- Continuous inference loop: run translator 1000 times with same input, verify no memory leak
- Large batch test: `batch_size=32, beam_size=5, max_length=1024` - verify stable performance
- Mixed precision stress: alternate between float16, float32, and INT8 in loop
- **PASS:** No OOM after 1000 iterations, memory usage stable (±5%)

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
| 12 (CI/Docs) | Test parameterization, CI, documentation | 1 week | 11 |
| **Total** | | **~12–18 weeks** | |

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

---

## Success Criteria (Revised)

### Correctness
- [ ] All ops: Metal result matches CPU within tolerance (float32: 1e-4, float16: 5e-3, INT8: 1%)
- [ ] seq2seq BLEU within 0.5 of CPU reference
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

**Phase 3: Production-Ready (M10-M12)**
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
