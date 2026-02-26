# Metal Backend API Reference

**Branch:** `metal-backend` | **Last updated:** 2026-02-26 | **Status:** M8.1 complete

---

## 1. Command-Buffer / Synchronization API

**Header:** `src/metal/utils.h` (C++ section, no `__OBJC__` guard)

```cpp
namespace ctranslate2::metal {

// Commit the current thread's command buffer and block until the GPU finishes.
// Implements synchronize_stream(Device::METAL) and synchronize_device(Device::METAL).
// No-op if no commands have been encoded since the last commit.
void commit_and_wait();

}
```

**ObjC++ section (`#ifdef __OBJC__` only — include from `.mm` files):**

```cpp
namespace ctranslate2::metal {

// Process-wide MTLDevice singleton (lazy, thread-safe).
id<MTLDevice>        get_metal_device();

// Per-thread MTLCommandQueue (created on first call per thread).
id<MTLCommandQueue>  get_metal_command_queue();

// Per-thread active MTLCommandBuffer.
// Ops must encode into this buffer — NEVER commit it themselves.
// MUST be called OUTSIDE any @autoreleasepool{} block (use-after-free otherwise).
id<MTLCommandBuffer> get_current_command_buffer();

// Commits the current thread's CB and resets the thread-local slot to nil.
// Only commit_and_wait() should call this.
void commit_command_buffer();

}

// Returns the MTLBuffer containing ptr and sets *offset_out to the byte offset
// of ptr within that buffer.  Throws if ptr was not allocated by MetalAllocator.
id<MTLBuffer> metal_buffer_for_ptr(const void* ptr, NSUInteger* offset_out);
```

**Error-check macros (ObjC++ only):**

```cpp
CT2_METAL_CHECK_BUFFER(buf)   // throws if buf.status == MTLCommandBufferStatusError
CT2_METAL_CHECK_OBJ(obj, name) // throws if obj == nil
```

---

## 2. `primitives<Device::METAL>` API

**Header:** `include/ctranslate2/primitives.h`
**Implementation:** `src/metal/primitives_*.mm`

All methods are `static`. `dim_t` = `int64_t`. Pointer args are raw typed pointers
from `MTLResourceStorageModeShared` buffers.

### 2.1 Memory

```cpp
// Read element at index.  Calls commit_and_wait() before CPU read.
template <typename T>
static T at(const T* x, dim_t index);

// Fill size elements with value a (GPU kernel).
template <typename T>
static void fill(T* x, T a, dim_t size);

// Copy size elements x → y (GPU kernel; GPU-to-GPU only via MTLBuffer blit).
template <typename T>
static void copy(const T* x, T* y, dim_t size);

// Type-converting copy U[] → V[].  Calls commit_and_wait() before std::copy.
// Supported pairs: any combination of float/float16_t/bfloat16_t/int8_t/int16_t/int32_t.
template <typename U, typename V>
static void convert(const U* x, V* y, dim_t size);
```

**`cross_device_primitives<D1, D2>::copy`** — copies between Metal and CPU (or Metal and Metal)
using `memcpy` on the Shared-memory pointer after `commit_and_wait()`.

### 2.2 Reduction

```cpp
// Sum over flat array (GPU two-pass tree reduction + CPU accumulate).
// All arithmetic done in float32; result cast back to T.
template <typename T>
static T sum(const T* array, dim_t size);

// Max over flat array (GPU + CPU, same pattern as sum).
template <typename T>
static T max(const T* array, dim_t size);

// Absolute-max (amax) over flat array (GPU + CPU).
template <typename T>
static T amax(const T* array, dim_t size);

// Index of maximum element (GPU + CPU).
template <typename T>
static dim_t max_element(const T* array, dim_t size);

// log(sum(exp(x))) with max-subtraction for numerical stability.
// Calls commit_and_wait() then runs CPU algorithm.
template <typename T>
static float logsumexp(const T* x, dim_t size);
```

### 2.3 Element-wise (binary)

All binary ops accept:
- `(T a, const T* x, T* y, dim_t size)` — scalar op vector, result in y
- `(const T* a, const T* b, T* c, dim_t size)` — vector op vector, result in c
- In-place variants: overloads that write into the first vector argument

```cpp
template <typename T> static void add(T a, const T* x, T* y, dim_t size);
template <typename T> static void add(const T* a, const T* b, T* c, dim_t size);

template <typename T> static void sub(const T* a, const T* b, T* c, dim_t size);

template <typename T> static void mul(T a, const T* x, T* y, dim_t size);
template <typename T> static void mul(const T* a, const T* b, T* c, dim_t size);

template <typename T> static void max(T a, const T* x, T* y, dim_t size);
template <typename T> static void max(const T* a, const T* b, T* c, dim_t size);

template <typename T> static void min(T a, const T* x, T* y, dim_t size);
template <typename T> static void min(const T* a, const T* b, T* c, dim_t size);
```

### 2.4 Broadcast

```cpp
// a is a shorter prefix; b has a_size * N elements.
// c[i*a_size + j] = a[j] OP b[i*a_size + j]
template <typename T>
static void add_batch_broadcast(const T* a, const T* b, T* c,
                                 dim_t a_size, dim_t b_size);
template <typename T>
static void mul_batch_broadcast(const T* a, const T* b, T* c,
                                 dim_t a_size, dim_t b_size);

// a is a deeper prefix; b has depth-broadcast layout.
template <typename T>
static void add_depth_broadcast(const T* a, const T* b, T* c,
                                 dim_t a_size, dim_t b_size);

// Block broadcast: c[b*a_size + j] = a[j] + b[b*a_size + j]  for b in [0, block)
template <typename T>
static void add_block_broadcast(const T* a, const T* b, T* c,
                                 dim_t block, dim_t a_size, dim_t b_size);
template <typename T>
static void mul_block_broadcast(const T* a, const T* b, T* c,
                                 dim_t block, dim_t a_size, dim_t b_size);
```

### 2.5 Activations (element-wise, unary)

```cpp
template <typename T> static void relu(const T* x, T* y, dim_t size);
template <typename T> static void gelu(const T* x, T* y, dim_t size);
template <typename T> static void gelu_tanh(const T* x, T* y, dim_t size);
template <typename T> static void gelu_sigmoid(const T* x, T* y, dim_t size);
template <typename T> static void sigmoid(const T* x, T* y, dim_t size);
template <typename T> static void swish(const T* x, T* y, dim_t size);
template <typename T> static void exp(const T* x, T* y, dim_t size);
template <typename T> static void log(const T* x, T* y, dim_t size);
template <typename T> static void cos(const T* x, T* y, dim_t size);
template <typename T> static void sin(const T* x, T* y, dim_t size);
template <typename T> static void tanh(const T* x, T* y, dim_t size);
```

Note: MSL has no `erf()`. `gelu` uses the Abramowitz & Stegun 7.1.28 polynomial
(`ct2_erf`, max error 1.5e-7). All kernel arithmetic is in float32; result cast back to T.

### 2.6 GEMM

```cpp
// General matrix multiply: C = alpha * op(A) * op(B) + beta * C
// a_is_packed / b_is_packed: always false for Metal (no packing support).
// transpose_a/b: whether to transpose A or B.
// m, n, k: output rows, output cols, inner dim.
// lda: leading dim of A (= k if not transposed, = m if transposed).
// ldb: leading dim of B (= n if not transposed, = k if transposed).
// ldc: leading dim of C (= n).
// In/Out type pairs:
//   float/float, float16_t/float16_t, bfloat16_t/bfloat16_t
// BF16 commits immediately (MPSGraph limitation).
template <typename In, typename Out>
static void gemm(bool a_is_packed, bool b_is_packed,
                 bool transpose_a, bool transpose_b,
                 dim_t m, dim_t n, dim_t k,
                 float alpha,
                 const In* a, dim_t lda,
                 const In* b, dim_t ldb,
                 float beta,
                 Out* c, dim_t ldc,
                 const Out* a_shift_compensation = nullptr);

// Batch-strided GEMM (calls gemm() batch_size times with pointer arithmetic).
template <typename In, typename Out>
static void gemm_batch_strided(bool transpose_a, bool transpose_b,
                               dim_t m, dim_t n, dim_t k,
                               float alpha,
                               const In* a, dim_t lda, dim_t stridea,
                               const In* b, dim_t ldb, dim_t strideb,
                               float beta,
                               Out* c, dim_t ldc, dim_t stridec,
                               dim_t batch_size);
```

**GEMM conventions (critical):**

| Argument | Meaning | Typical value |
|----------|---------|---------------|
| `lda` | columns in A storage (before transpose) | `k` when `transpose_a=false` |
| `ldb` | columns in B storage (before transpose) | `n` when `transpose_b=false`, `k` when `transpose_b=true` |
| `ldc` | columns in C | `n` (always) |

`MPSMatrixMultiplication` requires `rowBytes ≥ rowBytesForColumns:` (MPS minimum alignment).
`primitives_gemm.mm` auto-pads into a temporary buffer when needed.

### 2.7 Transpose

```cpp
// dims: shape of a (2 elements).  Output b has transposed layout.
template <typename T>
static void transpose_2d(const T* a, const dim_t* dims, T* b);

// dims: shape of a (3 elements).  perm: output permutation (3 elements).
template <typename T>
static void transpose_3d(const T* a, const dim_t* dims, const dim_t* perm, T* b);

// dims: shape of a (4 elements).  perm: output permutation (4 elements).
template <typename T>
static void transpose_4d(const T* a, const dim_t* dims, const dim_t* perm, T* b);
```

GPU kernels for all three ranks; 6 types each (float/half/bfloat/int/short/char).
GPU wins at ≥ ~1–4 MB tensors; CB overhead dominates below that.

### 2.8 Beam Search

```cpp
// GPU kernel: for each batch item, divide scores of previously seen tokens by penalty.
// One thread per batch item, sequential over length (safe for duplicate IDs).
template <typename T>
static void penalize_previous_tokens(T* scores,
                                     const T* previous_scores,
                                     const int32_t* previous_ids,
                                     T penalty,
                                     dim_t batch_size,
                                     dim_t length,
                                     dim_t vocabulary_size);

// CPU (commit_and_wait + loop): build attention mask from sequence lengths.
static void prepare_length_mask(const int32_t* lengths,
                                dim_t batch_size,
                                dim_t num_heads,
                                dim_t num_queries,
                                bool mask_future,
                                bool multi_query,
                                int32_t* mask);
```

---

## 3. `metal::` Free-Function API (complex ops)

**Header:** `src/metal/ops_metal.h` — include only from `.mm` files with `CT2_WITH_METAL`.
**Implementations:** `src/metal/ops_norm_gather.mm`, `ops_sdpa.mm`, `ops_rotary.mm`, `ops_alibi.mm`.

```cpp
namespace ctranslate2::metal {

// LayerNorm: normalize each row, scale by gamma, shift by beta.
// gamma/beta may be nullptr (identity / zero).
// CONSTRAINT: inner_size == 1 only (last-axis normalization).
//             Caller (normalization_metal.mm) throws for inner_size > 1.
// One threadgroup of 256 threads per outer element (row).
template <typename T>
void layer_norm_metal(const T* x, const T* gamma, const T* beta,
                      T* y, dim_t outer_size, dim_t axis_size, float epsilon);

// RMSNorm: normalize each row by its RMS, scale by gamma.
// One threadgroup of 256 threads per batch item.
// CONSTRAINT: use_residual is not supported on METAL; caller throws.
template <typename T>
void rms_norm_metal(const T* x, const T* gamma, T* y,
                    dim_t batch_size, dim_t depth, float epsilon);

// SoftMax / log-SoftMax over last dimension.
// lengths may be nullptr (no masking). log_mode=true → log-softmax.
// One threadgroup of 256 threads per batch item.
template <typename T>
void softmax_metal(const T* x, const int32_t* lengths, T* y,
                   dim_t batch_size, dim_t depth, bool log_mode);

// Gather: for each output slot, copy copy_size elements from src.
//   dst[slot*copy_size + j] = src[batch_idx*batch_stride + indices[slot]*copy_size + j]
// One thread per output element.
template <typename T>
void gather_metal(const T* src, T* dst, const int32_t* indices,
                  dim_t copy_size, dim_t batch_stride,
                  dim_t num_indices_per_batch, dim_t total_elements);

// Scaled dot-product attention.
// q/k/v layout: [batch_size, seqlen, num_heads, head_dim]  (interleaved heads)
// output:       same shape as q
// scale:        multiplied into Q*K^T before softmax (typically 1/sqrt(head_dim))
// is_causal:    apply upper-triangular mask (scores[col > row] = large_neg)
// num_heads_k:  KV heads (< num_heads for GQA; == num_heads for MHA)
// f32/f16: MPSMatrixMultiplication (encode-only)
// bf16:    MPSGraph matmul (commits immediately)
template <typename T>
void sdpa_metal(const T* q, const T* k, const T* v, T* output,
                dim_t batch_size, dim_t seqlen_q, dim_t seqlen_k,
                dim_t num_heads, dim_t num_heads_k, dim_t head_dim,
                float scale, bool is_causal);

// Rotary position embeddings applied to [total_vecs, depth] view.
// sin/cos tables: [max_time, ndims]
// is_transposed=false: t = vec_idx / head_size   (Flash-Attention layout)
// is_transposed=true:  t = vec_idx % max_time    (standard layout)
template <typename T>
void rotary_metal(const T* input, const T* sin_buf, const T* cos_buf,
                  T* output,
                  dim_t total_vecs, dim_t depth, dim_t ndims,
                  dim_t max_time, dim_t head_size,
                  bool interleave, bool is_transposed);

// ALiBi positional bias: add alibi slopes to attention scores.
// input/output: [batch_size, num_heads, query_length, key_length]
// alibi:        [1, num_heads, 1, cached_key_length]
// alibi_offset: start column within alibi table.
template <typename T>
void alibi_add_metal(const T* input, const T* alibi, T* output,
                     dim_t batch_size, dim_t num_heads,
                     dim_t query_length, dim_t key_length,
                     dim_t cached_key_length, dim_t alibi_offset);

}  // namespace ctranslate2::metal
```

---

## 4. Internal Infrastructure (`primitives_infra.h`)

Internal header for `primitives_*.mm` and `ops_*.mm` only. Do not include from op layer.

```cpp
// Checked narrowing: dim_t (int64_t) → uint32_t.
// Throws std::runtime_error if v < 0 or v > UINT32_MAX.
// Use wherever a GPU kernel arg is declared `uint` in MSL.
static inline uint32_t ct2_u32(dim_t v);

// Allocate a temporary MTLBuffer (Shared mode, ARC-managed).
static inline id<MTLBuffer> alloc_temp_buffer(NSUInteger bytes);

// Lazy MSL library compilation (thread-safe, compiled once per TU).
// flag and lib_out are static locals of the calling library-getter function.
static inline id<MTLLibrary> compile_library_once(
    std::once_flag& flag,
    id<MTLLibrary>& lib_out,
    const char* msl_src,
    const char* label,
    MTLCompileOptions* opts = nil);

// PSO creation from library + function name.
static inline id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib,
                                                    const char* name);

// Thread-safe PSO cache (one per kernel group, static).
struct PSOCache {
  std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  std::mutex mtx;
  template <typename LibFn>
  id<MTLComputePipelineState> get(LibFn lib_fn, const char* name);
};

// Kernel name buffer size (fits longest name, e.g. "mul_scalar_bfloat").
static constexpr size_t kKernelNameBufSize = 64;

// Throw for unimplemented primitives<METAL> methods.
#define METAL_STUB(name)  // throws std::runtime_error
```

**`MetalTypeName<T>::value`** — maps C++ type to MSL type string for kernel name formatting:

| C++ type | MSL string |
|----------|-----------|
| `float` | `"float"` |
| `float16_t` | `"half"` |
| `bfloat16_t` | `"bfloat"` |
| `int32_t` | `"int"` |
| `int16_t` | `"short"` |
| `int8_t` | `"char"` |

---

## 5. Allocator

**`MetalAllocator`** (`src/metal/allocator.mm`): caching allocator backed by `MTLResourceStorageModeShared`.

- Registered as `Device::METAL` allocator via `register_allocator` in `device.mm`.
- Returned pointer = `[buf contents]` — simultaneously valid for CPU and GPU on Apple Silicon.
- Pool keyed on requested size (not rounded Metal size) for exact-bucket reuse.
- `metal_buffer_for_ptr(ptr, &offset)` — scans `_live` map to find enclosing buffer + byte offset.
  Used by compute encoders which take `(id<MTLBuffer>, offset)` rather than raw `void*`.
- Thread safety: single mutex guards both `_live` and `_pool` maps.

---

## 6. Standard Build Command

```bash
clang++ -std=c++17 -O0 \
    -I include -I src \
    -DCT2_WITH_METAL \
    tests/metal/<test_file>.mm \
    src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
    src/metal/primitives_memory.mm \
    src/metal/primitives_elementwise.mm \
    src/metal/primitives_reduction.mm \
    src/metal/primitives_gemm.mm \
    src/metal/primitives_transpose.mm \
    src/metal/primitives_beam_search.mm \
    src/metal/ops_norm_gather.mm \
    src/allocator.cc src/devices.cc src/cpu/allocator.cc \
    -framework Metal -framework Foundation \
    -framework MetalPerformanceShaders \
    -framework MetalPerformanceShadersGraph \
    -o <test_name> && ./<test_name>
```

**Variations:**

| Need | Change |
|------|--------|
| Benchmarks | `-O2` instead of `-O0` |
| SDPA tests | Add `src/metal/ops_sdpa.mm` |
| RoPE tests | Add `src/metal/ops_rotary.mm` |
| ALiBi tests | Add `src/metal/ops_alibi.mm` |
| M8.1 encoder | Add `ops_sdpa.mm`; drop unused `primitives_beam_search.mm` optional |

---

## 7. Test Harness Boilerplate

```objc
// Standard preamble for Metal test .mm files
#include <cstdio>
#include <cstring>
#include <cmath>

// Float16/BF16 type aliases — MUST be declared before 'using namespace ctranslate2'
// to avoid ambiguity with ARM vector type headers injecting ::float16_t / ::bfloat16_t.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// Allocate n elements of type T in a Metal Shared buffer.
template <typename T>
T* metal_alloc(dim_t n) {
    Allocator& alloc = get_allocator<Device::METAL>();
    return static_cast<T*>(alloc.allocate(n * sizeof(T)));
}

// Free a Metal-allocated pointer.
void metal_free(void* ptr) {
    Allocator& alloc = get_allocator<Device::METAL>();
    alloc.free(ptr);
}

// Assertion helper (prints PASS/FAIL, increments counters).
static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) \
  do { if (cond) { ++g_pass; printf("  PASS: %s\n", msg); } \
       else { ++g_fail; printf("  FAIL: %s\n", msg); } } while(0)
```

---

## 8. Key Constraints Summary

| API | Constraint |
|-----|-----------|
| `get_current_command_buffer()` | MUST be called outside `@autoreleasepool {}` |
| `layer_norm_metal` | `inner_size == 1` only (last-axis); throws otherwise |
| `rms_norm_metal` | `use_residual=true` not supported on METAL |
| `commit_and_wait()` | Must be called before any CPU read from Metal buffers |
| BF16 GEMM (MPSGraph) | Commits the command buffer immediately |
| `ct2_u32()` | Throws if value > UINT32_MAX or negative; use for all MSL `uint` args |
| `gemm` packed args | `a_is_packed=false`, `b_is_packed=false` always for METAL |
| MSL `erf()` | Not available in MSL — uses A&S 7.1.28 polynomial in `activation.metal` |
| Kernel thread type | MUST declare `typedef ctranslate2::float16_t ct2_f16` before `using namespace ctranslate2` in Metal `.mm` files |
