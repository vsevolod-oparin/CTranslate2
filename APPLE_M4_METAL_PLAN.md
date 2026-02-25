# Apple M4 Metal Backend Implementation Plan

**Revised:** 2026-02-25
**Status:** In progress — Milestone 1 complete

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

**2.1 Create `src/metal/` context module**
- `src/metal/utils.h` / `src/metal/utils.mm`:
  ```objc
  id<MTLDevice>       get_metal_device();        // singleton
  id<MTLCommandQueue> get_metal_command_queue();  // thread_local
  id<MTLCommandBuffer> get_current_command_buffer(); // encode into this
  void commit_command_buffer();                  // = synchronize_stream for Metal
  ```
- Add `CT2_METAL_CHECK_BUFFER(buf)` and `CT2_METAL_CHECK_OBJ(obj, name)` macros (see M2.4 for definitions and correct usage pattern)
- **PASS:**
  ```cpp
  // tests/metal/context_test.mm (new)
  auto* dev = get_metal_device();
  ASSERT_NE(dev, nil);
  auto* queue = get_metal_command_queue();
  ASSERT_NE(queue, nil);
  auto* buf = get_current_command_buffer();
  ASSERT_NE(buf, nil);
  commit_command_buffer();
  // New command buffer created after commit
  ASSERT_NE(get_current_command_buffer(), buf);
  ```

**2.2 Implement `synchronize_device` / `synchronize_stream` for Metal**
- `src/devices.cc`: Metal case calls `commit_command_buffer()` then `[commandBuffer waitUntilCompleted]`
- **PASS:**
  ```cpp
  // Encode a no-op blit, synchronize, verify no crash/hang
  synchronize_device(Device::METAL, 0);  // completes within 100ms
  ```

**2.3 Extend `ScopedDeviceSetter` for Metal**
- Metal has one GPU, so `set_device_index(Device::METAL, 0)` is a no-op
- `get_device_index(Device::METAL)` always returns 0
- **PASS:** `ScopedDeviceSetter setter(Device::METAL, 0);` — compiles and runs without error

**2.4 Error handling strategy**

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

---

### Milestone 3: Metal Allocator
**Goal:** Metal buffer allocation backed by unified memory, integrated with `get_allocator<Device::METAL>()`.
**Time:** 3–5 days
**Depends on:** M2

**3.1 Implement `MetalAllocator`**
- `src/metal/allocator.mm`:
  - Allocate: `[device newBufferWithLength:size options:MTLResourceStorageModeShared]`
  - Return `[buffer contents]` — the CPU-accessible `void*` to the shared memory
  - Retain `MTLBuffer` objects in a `std::unordered_map<void*, id<MTLBuffer>>` (protected by mutex)
  - Free: remove from map (ARC releases the MTLBuffer, memory reclaimed)
  - Implement buffer pool: reuse freed buffers of matching size (caching allocator)
- `src/allocator.cc`: `get_allocator<Device::METAL>()` returns `MetalAllocator` singleton
- **PASS:**
  ```cpp
  // tests/metal/allocator_test.mm
  auto& alloc = get_allocator(Device::METAL);
  float* ptr = static_cast<float*>(alloc.allocate(1024 * sizeof(float)));
  ASSERT_NE(ptr, nullptr);
  ptr[0] = 42.f;                         // CPU write to shared memory
  ASSERT_EQ(ptr[0], 42.f);              // CPU read back
  alloc.free(ptr);
  // Pool test: second alloc of same size returns cached buffer
  float* ptr2 = static_cast<float*>(alloc.allocate(1024 * sizeof(float)));
  // (ptr2 should equal ptr if pool works)
  alloc.free(ptr2);
  ```

**3.2 Integrate with `StorageView`**
- `src/storage_view.cc`: add `Device::METAL` case to `cross_device_primitives<Device::METAL, Device::CPU>::copy()` and vice versa
- On Apple Silicon with shared memory, `cross_device_primitives<Device::METAL, Device::CPU>::copy()` is `memcpy` (same physical memory, just a CPU fence)
- Add `synchronize_stream(Device::METAL)` before CPU reads from Metal-written memory
- **PASS:**
  ```cpp
  // tests/metal/storage_view_test.mm
  StorageView cpu_src({4}, DataType::FLOAT32, Device::CPU);
  cpu_src.data<float>()[0] = 1.f; cpu_src.data<float>()[1] = 2.f;
  // Copy to Metal
  StorageView metal_sv = cpu_src.to(Device::METAL);
  ASSERT_EQ(metal_sv.device(), Device::METAL);
  // Copy back to CPU
  StorageView cpu_dst = metal_sv.to(Device::CPU);
  ASSERT_NEAR(cpu_dst.data<float>()[0], 1.f, 1e-6);
  ASSERT_NEAR(cpu_dst.data<float>()[1], 2.f, 1e-6);
  ```

---

### Milestone 4: Core Primitives
**Goal:** `primitives<Device::METAL>` specialization — the building blocks all ops call.
**Time:** 1–2 weeks
**Depends on:** M3

**Key insight:** Because we use `MTLResourceStorageModeShared`, all Metal buffers are already CPU-accessible via their `contents` pointer. Primitives operate on those raw pointers directly for element-wise ops. For GEMM and reductions, use MPS objects encoded into the current command buffer.

**4.1 Memory primitives (`fill`, `copy`, `convert`)**
- `src/metal/primitives.mm`:
  - `fill<T>(T* x, T a, dim_t size)` — encode `MPSMatrixCopy` fill, or use a tiny Metal compute kernel
  - `copy<T>(T* x, T* y, dim_t size)` — `memcpy` (shared memory) + CPU fence, or `MTLBlitCommandEncoder`
  - `convert<float, float16_t>(...)` — MPSGraph cast node, or SIMD CPU conversion (unified memory)
- **PASS:**
  ```cpp
  // tests/metal/primitives_test.mm
  // fill
  std::vector<float> buf(64);
  float* metal_ptr = metal_alloc<float>(64);
  primitives<Device::METAL>::fill(metal_ptr, 3.14f, 64);
  synchronize_stream(Device::METAL);
  for (int i=0; i<64; i++) ASSERT_NEAR(metal_ptr[i], 3.14f, 1e-6);
  // copy
  primitives<Device::METAL>::copy(metal_ptr, dst_ptr, 64);
  synchronize_stream(Device::METAL);
  ASSERT_NEAR(dst_ptr[0], 3.14f, 1e-6);
  ```

**4.2 Arithmetic primitives (`add`, `mul`, `sub`)**
- Use Metal compute shaders (simple element-wise kernels in `src/metal/kernels/elementwise.metal`)
- These are the first custom Metal shaders, but trivial to write and test
- Alternative: MPSGraph for each — but overhead is too high for small ops
- **PASS:**
  ```cpp
  // Reference: CPU result
  // Metal: apply op, sync, compare element-wise (max abs diff < 1e-5 for float32)
  ```

**4.3 Reduction primitives (`sum`, `max`, `amax`, `max_element`)**
- Use `MPSMatrixSum` or a custom reduction kernel
- **PASS:** Compare to CPU `std::accumulate` / `std::max_element` within tolerance

**4.4 GEMM (CRITICAL)**

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

- **PASS:**
  ```cpp
  // Sizes: 64×64, 512×512, 4096×4096, 128×4096×512 (non-square)
  // Compare to Accelerate cblas_sgemm
  // Max rel diff < 1e-4 for float32, < 5e-3 for float16
  // Max rel diff < 1e-2 for bfloat16 (vs cblas on BF16-quantised inputs)
  // Benchmark: 4096³ FP32 GEMM > 2× faster than CPU
  // BF16 graph cache: second call with same shape must be ≤ 5% slower than first (steady-state)
  ```

**4.5 Transcendental and activation primitives (`exp`, `log`, `cos`, `sin`, `tanh`, `relu`, `gelu`, `gelu_tanh`, `gelu_sigmoid`, `sigmoid`, `swish`)**
- Use MPSGraph unary ops for `exp`, `log`, `cos`, `sin`, `tanh`, `sigmoid`
- `relu` → `MPSCNNNeuronReLU` or MPSGraph threshold
- `gelu` (erf form), `gelu_tanh` (tanh approximation), `gelu_sigmoid` → MPSGraph composite or custom Metal shader
- `swish(x) = x * sigmoid(x)` → compose with Metal shader or MPSGraph
- **PASS:** Compare all variants to CPU reference within 1e-4 for float32, 5e-3 for float16

**4.6 Broadcast and scatter primitives**

These are heavily used in transformer layers (bias addition, positional encoding) and are non-trivial — they have different striding/index math than simple element-wise ops.

- `add_batch_broadcast(a, b, c, a_size, b_size)` — broadcasts `a` over batch dim of `b`
- `add_depth_broadcast(a, b, c, a_size, b_size)` — broadcasts `a` over depth dim
- `add_block_broadcast(a, b, c, block, a_size, b_size)` — broadcasts over block-strided layout
- `mul_batch_broadcast(a, b, c, a_size, b_size)` — multiplicative batch broadcast
- `strided_fill(x, a, inc_x, size)` — fill with stride (e.g., diagonal init)
- `indexed_fill(x, a, indices, num_indices)` — fill selected indices (used by Gather-family ops)

All implement as custom Metal compute shaders in `src/metal/kernels/elementwise.metal` (the indexing math is simple but must be correct).

- **PASS:**
  ```cpp
  // tests/metal/primitives_test.mm — broadcast tests
  // add_batch_broadcast: a=[1,2,3], b=[a,b,c,d,e,f] (size 6, a_size=3) → correct broadcast
  // indexed_fill: fill positions [0,2,4] of output with value 7 → verify only those positions changed
  ```

**4.7 Beam-search and attention-mask primitives**

These are required for generation and variable-length batch handling. Missing them means beam search and padded batches silently produce wrong results.

- **`penalize_previous_tokens(scores, previous_scores, previous_ids, penalty, batch, len, vocab_size)`**
  - Applies repetition penalty to logits based on previously generated token IDs
  - Custom Metal compute shader: `src/metal/kernels/beam_search.metal`
  - **PASS:** Scores for repeated tokens are reduced by `penalty`; scores for new tokens are unchanged; verify numerically against CPU reference for batch=2, len=8, vocab=100

- **`prepare_length_mask(lengths, batch, num_heads, num_queries, mask_future, multi_query, mask)`**
  - Writes the `int32_t` mask used by attention to handle padding and causal masking
  - Custom Metal compute shader or CPU fallback (mask creation is cheap; can be done on CPU then uploaded to Metal)
  - **PASS:** Mask matches CPU output for padded batch (lengths=[3,5], max_len=5) and for causal (mask_future=true)

- **`logsumexp(x, size)` → `float`**
  - Returns `log(sum(exp(x_i)))` numerically stably; used in beam search log-probability normalization
  - Note: return type is always `float` even when `T = float16_t` — don't accidentally return `T`
  - Implement as a reduction kernel (find max, subtract, exp, sum, log, add max back)
  - **PASS:** `logsumexp([1.0, 2.0, 3.0]) ≈ 3.408` within 1e-5 of CPU result; test with float32 and float16 inputs

- **`at(const T* x, dim_t index)` → `T`**
  - Reads a single element from a Metal buffer at the given index
  - **Critical subtlety**: Metal GPU writes are not visible to CPU until `synchronize_stream()` is called. `at()` must call `synchronize_stream(Device::METAL)` before reading, or the result will be stale.
  - **PASS:** Write a value via Metal kernel, call `at()`, verify correct value returned

**4.8 Transpose primitives**

The `Transpose` op calls `primitives<D>::transpose_2d/3d/4d` internally. These must be specialized separately from the `Transpose` op dispatcher.

- `transpose_2d(a, dims, b)` — 2D matrix transpose
- `transpose_3d(a, dims, perm, b)` — arbitrary 3D permutation
- `transpose_4d(a, dims, perm, b)` — arbitrary 4D permutation (used for multi-head attention head splitting)

Implementation options:
- Use `MPSMatrixTranspose` for 2D; custom shader for 3D/4D permutations
- Or: all via a generic permutation compute shader parameterized by rank+perm

- **PASS:** Round-trip test (permute then inverse-permute = identity) for all four shapes; compare to CPU `np.transpose` equivalents

---

### Milestone 5: Op Dispatcher Integration
**Goal:** Wire existing op dispatchers to call Metal primitives. Each op gets a `Device::METAL` path.
**Time:** 1 week
**Depends on:** M1 (dispatch macros), M4 (primitives)

**5.1 Update `DEVICE_AND_FLOAT_DISPATCH` for Metal FP16/BF16**
- `src/dispatch.h`: extend the `CT2_WITH_CUDA` block or add a parallel `CT2_WITH_METAL` block
  ```cpp
  #ifdef CT2_WITH_METAL
  TYPE_CASE(float16_t, {
    if (DEVICE != Device::CUDA && DEVICE != Device::METAL)
      throw std::invalid_argument("FP16 " NAME " is only supported on GPU");
    ...
  })
  #endif
  ```
- **PASS:** `DEVICE_AND_FLOAT_DISPATCH` with `Device::METAL` + `float16_t` dispatches without throwing

**5.2 Add `Device::METAL` specializations to each op**
- For each op, add `template<> void LayerNorm::compute<Device::METAL, float>(...)` in new `src/ops/layer_norm_metal.mm`
- Ops to cover in this milestone (by importance):
  1. `LayerNorm` — used in every transformer layer
  2. `RMSNorm` — used in LLaMA/Qwen/modern LLMs
  3. `Softmax` — used in attention
  4. `Gemm` / `MatMul` — via `primitives<Device::METAL>::gemm()`
  5. `Add`, `Mul`, `BiasAdd` — residual connections
  6. `Transpose` — head splitting in attention
  7. `Gather` — embedding lookup
- File naming convention: `src/ops/<op_name>_metal.mm`
- Each file adds specialization to existing op class, not a new class
- **PASS for each op:** Run `tests/ops_test.cc` extended with `Device::METAL` parameter:
  ```cpp
  // Parameterized test: CPU ref vs Metal
  INSTANTIATE_TEST_SUITE_P(Metal, OpTest, ::testing::Values(Device::METAL));
  // Each op test: max(|metal_result - cpu_result|) < tolerance
  ```

---

### Milestone 6: Attention Mechanisms
**Goal:** Multi-head attention working end-to-end on Metal.
**Time:** 1–2 weeks
**Depends on:** M5

**6.1 Scaled dot-product attention**
- `src/ops/flash_attention_metal.mm`: implement SDPA using MPS matmul + softmax
  - `Q * K^T` → `MPSMatrixMultiplication`
  - `/ sqrt(d_k)` → scalar multiply primitive
  - masking → element-wise add (large negative for masked positions)
  - `softmax` → already implemented in M5
  - `* V` → `MPSMatrixMultiplication`
- **PASS:**
  ```cpp
  // tests/attention_test.cc — add Metal case
  // Input: Q/K/V 4×8×64 (batch×heads×dim)
  // Metal output vs CPU output: max abs diff < 1e-3
  ```

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
