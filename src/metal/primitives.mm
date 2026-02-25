// primitives<Device::METAL> — M3.2 + M4.1 + M4.2 + M4.3 implementation.
//
// All Metal buffers use MTLResourceStorageModeShared (unified memory).
// Their contents pointer is simultaneously valid for CPU and GPU access.
//
// Memory primitives (fill, copy, convert) — M4.1 — are CPU-side operations.
// CPU writes are visible to the GPU before the next command encoding on Apple
// Silicon (unified memory coherency).
//
// Arithmetic primitives (add, sub, mul) — M4.2 — are GPU compute kernels.
// They encode into the thread-local command buffer; results are only
// committed to the GPU when synchronize_stream(METAL) is called.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ctranslate2/types.h"

#include "ctranslate2/primitives.h"
#include "metal/utils.h"
#include "type_dispatch.h"

// ---------------------------------------------------------------------------
// M4.2 — element-wise compute kernel infrastructure
// ---------------------------------------------------------------------------

namespace {

// MSL source for element-wise kernels (add, sub, mul in vector and scalar
// broadcast forms).  This is the exact content of
// src/metal/kernels/elementwise.metal embedded as a C++ string.
static constexpr const char* kElementwiseMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#define DEFINE_BINARY(name, op, T)                                      \
  kernel void name##_##T(                                               \
      device const T* a [[buffer(0)]],                                  \
      device const T* b [[buffer(1)]],                                  \
      device       T* c [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { c[gid] = a[gid] op b[gid]; }

#define DEFINE_SCALAR(name, op, T)                                      \
  kernel void name##_scalar_##T(                                        \
      device const T* x [[buffer(0)]],                                  \
      constant     T& a [[buffer(1)]],                                  \
      device       T* y [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { y[gid] = a op x[gid]; }

#define DEFINE_ALL(T)          \
  DEFINE_BINARY(add, +, T)    \
  DEFINE_BINARY(sub, -, T)    \
  DEFINE_BINARY(mul, *, T)    \
  DEFINE_SCALAR(add, +, T)    \
  DEFINE_SCALAR(mul, *, T)

DEFINE_ALL(float)
DEFINE_ALL(half)
DEFINE_ALL(int)
DEFINE_ALL(short)
DEFINE_ALL(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_ALL(bfloat)
#endif
)msl";

// Lazy-compile the element-wise MSL library.  Thread-safe; compiled once.
static id<MTLLibrary> get_elementwise_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kElementwiseMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:nil
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile elementwise library";
      if (err) {
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      }
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO (pipeline state object) cache.  Keyed by kernel function name.
static id<MTLComputePipelineState> get_elementwise_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_elementwise_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: kernel not found: ") + name);
  }

  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [ctranslate2::metal::get_metal_device()
          newComputePipelineStateWithFunction:fn
                                       error:&err];
  if (pso == nil) {
    std::string msg = std::string("Metal: PSO creation failed for ") + name;
    if (err)
      msg += std::string(": ") + [err.localizedDescription UTF8String];
    throw std::runtime_error(msg);
  }
  cache[name] = pso;
  return pso;
}

// Dispatch a binary vector-op-vector kernel:  c[i] = a[i] op b[i].
static void dispatch_binary(const char* kernel_name,
                             const void* a, const void* b, void* c,
                             ctranslate2::dim_t size) {
  if (size == 0) {
    return;
  }
  id<MTLComputePipelineState> pso = get_elementwise_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(c, &off_c) offset:off_c atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// Dispatch a scalar-op-vector kernel:  y[i] = a op x[i].
// scalar_val points to sizeof(T) bytes of the scalar value; passed via
// setBytes (inlined into the argument table, no buffer allocation needed).
static void dispatch_scalar(const char* kernel_name,
                             const void* scalar_val, size_t scalar_bytes,
                             const void* x, void* y,
                             ctranslate2::dim_t size) {
  if (size == 0) {
    return;
  }
  id<MTLComputePipelineState> pso = get_elementwise_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_x = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x) offset:off_x atIndex:0];
  [enc setBytes:scalar_val length:scalar_bytes atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y) offset:off_y atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// Map from C++ type to its Metal (MSL) type name string.
template <typename T> struct MetalTypeName;
template<> struct MetalTypeName<float>                   { static constexpr const char* value = "float"; };
template<> struct MetalTypeName<ctranslate2::float16_t>  { static constexpr const char* value = "half";  };
template<> struct MetalTypeName<ctranslate2::bfloat16_t> { static constexpr const char* value = "bfloat"; };
template<> struct MetalTypeName<int8_t>                  { static constexpr const char* value = "char";  };
template<> struct MetalTypeName<int16_t>                 { static constexpr const char* value = "short"; };
template<> struct MetalTypeName<int32_t>                 { static constexpr const char* value = "int";   };

// Generous upper bound for any formatted elementwise kernel name
// (e.g. "mul_scalar_bfloat").  Kept large so future op names with longer
// prefixes or type suffixes don't silently truncate via snprintf.
static constexpr size_t kKernelNameBufSize = 64;


// ---------------------------------------------------------------------------
// M4.3 — Parallel reduction infrastructure
// ---------------------------------------------------------------------------

// MSL source for two-pass reduction kernels (sum, max, amax, max_element).
// Canonical copy lives in src/metal/kernels/reduction.metal.
static constexpr const char* kReductionMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

// ---- SUM ----
#define DEFINE_REDUCE_SUM(T, ZERO)                                       \
kernel void reduce_sum_##T(                                              \
    device const T*      inp   [[buffer(0)]],                            \
    device       T*      out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup T*       shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? inp[gid] : ZERO;                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s) { shmem[tid] += shmem[tid + s]; }                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}
DEFINE_REDUCE_SUM(float, 0.f)
DEFINE_REDUCE_SUM(half,  (half)0)
DEFINE_REDUCE_SUM(int,   0)
DEFINE_REDUCE_SUM(short, (short)0)
DEFINE_REDUCE_SUM(char,  (char)0)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_SUM(bfloat, (bfloat)0)
#endif

// ---- MAX ----
// Use explicit comparison instead of max() to avoid MSL overload ambiguity
// for bfloat (no dedicated bfloat max() overload on all SDK versions).
#define DEFINE_REDUCE_MAX(T, NEG_INF)                                    \
kernel void reduce_max_##T(                                              \
    device const T*      inp   [[buffer(0)]],                            \
    device       T*      out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup T*       shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? inp[gid] : NEG_INF;                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s && shmem[tid + s] > shmem[tid]) {                   \
            shmem[tid] = shmem[tid + s];                                 \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}
DEFINE_REDUCE_MAX(float, -FLT_MAX)
DEFINE_REDUCE_MAX(half,  (half)(-FLT_MAX))
DEFINE_REDUCE_MAX(int,   (int)0x80000000)
DEFINE_REDUCE_MAX(short, (short)0x8000)
DEFINE_REDUCE_MAX(char,  (char)0x80)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_MAX(bfloat, (bfloat)(-FLT_MAX))
#endif

// ---- AMAX (output always float*) ----
#define DEFINE_REDUCE_AMAX(T)                                            \
kernel void reduce_amax_##T(                                             \
    device const T*      inp   [[buffer(0)]],                            \
    device       float*  out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup float*   shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? fabs((float)inp[gid]) : 0.f;               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s) { shmem[tid] = max(shmem[tid], shmem[tid + s]); }  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}
DEFINE_REDUCE_AMAX(float)
DEFINE_REDUCE_AMAX(half)
DEFINE_REDUCE_AMAX(int)
DEFINE_REDUCE_AMAX(short)
DEFINE_REDUCE_AMAX(char)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_AMAX(bfloat)
#endif

// ---- MAX_ELEMENT (values in float, indices in uint32_t) ----
#define DEFINE_REDUCE_MAX_ELEMENT(T)                                           \
kernel void reduce_max_element_##T(                                            \
    device const T*        inp      [[buffer(0)]],                             \
    device       float*    out_vals [[buffer(1)]],                             \
    device    uint32_t*    out_idxs [[buffer(2)]],                             \
    constant  uint32_t&    n        [[buffer(3)]],                             \
    threadgroup float*     sh_vals  [[threadgroup(0)]],                        \
    threadgroup uint32_t*  sh_idxs  [[threadgroup(1)]],                        \
    uint gid  [[thread_position_in_grid]],                                     \
    uint tid  [[thread_index_in_threadgroup]],                                 \
    uint tgid [[threadgroup_position_in_grid]],                                \
    uint tgs  [[threads_per_threadgroup]])                                     \
{                                                                              \
    bool in_range = (gid < n);                                                 \
    sh_vals[tid]  = in_range ? (float)inp[gid] : -FLT_MAX;                    \
    sh_idxs[tid]  = in_range ? gid             : 0xFFFFFFFFu;                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                                 \
        if (tid < s && sh_vals[tid + s] > sh_vals[tid]) {                      \
            sh_vals[tid] = sh_vals[tid + s];                                   \
            sh_idxs[tid] = sh_idxs[tid + s];                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    if (tid == 0) {                                                            \
        out_vals[tgid] = sh_vals[0];                                           \
        out_idxs[tgid] = sh_idxs[0];                                          \
    }                                                                          \
}
DEFINE_REDUCE_MAX_ELEMENT(float)
DEFINE_REDUCE_MAX_ELEMENT(half)
DEFINE_REDUCE_MAX_ELEMENT(int)
DEFINE_REDUCE_MAX_ELEMENT(short)
DEFINE_REDUCE_MAX_ELEMENT(char)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_MAX_ELEMENT(bfloat)
#endif
)msl";

// Lazy-compile the reduction MSL library.  Thread-safe; compiled once.
static id<MTLLibrary> get_reduction_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kReductionMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:nil
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile reduction library";
      if (err) {
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      }
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO cache for reduction kernels.
static id<MTLComputePipelineState> get_reduction_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_reduction_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: reduction kernel not found: ") + name);
  }

  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [ctranslate2::metal::get_metal_device()
          newComputePipelineStateWithFunction:fn
                                       error:&err];
  if (pso == nil) {
    std::string msg = std::string("Metal: reduction PSO creation failed for ") + name;
    if (err) {
      msg += std::string(": ") + [err.localizedDescription UTF8String];
    }
    throw std::runtime_error(msg);
  }
  cache[name] = pso;
  return pso;
}

// Fixed threadgroup size for all reduction kernels.
// 256 threads/group gives 8 SIMD waves on Apple Silicon (SIMD width = 32),
// keeping the GPU fully occupied while keeping threadgroup memory small.
static constexpr uint32_t kReductionTGS = 256;

// Allocate a temporary shared-mode MTLBuffer (not through MetalAllocator).
// Managed by ARC — released when the local id<MTLBuffer> goes out of scope.
static id<MTLBuffer> alloc_temp_buffer(NSUInteger bytes) {
  id<MTLBuffer> buf = [ctranslate2::metal::get_metal_device()
      newBufferWithLength:bytes
                 options:MTLResourceStorageModeShared];
  if (buf == nil) {
    throw std::runtime_error("Metal: failed to allocate temporary reduction buffer");
  }
  return buf;
}

}  // anonymous namespace

namespace ctranslate2 {

  // -------------------------------------------------------------------------
  // cross_device_primitives  (CPU ↔ Metal)
  // -------------------------------------------------------------------------

  // CPU → Metal: the destination is a Shared-mode MTLBuffer contents pointer;
  // CPU writes are immediately visible to the GPU at the next command encoding.
  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::METAL>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // Metal → CPU: the source is a Shared-mode MTLBuffer contents pointer.
  // The GPU must have committed and completed any writes to this buffer before
  // this memcpy is called (ensured by synchronize_stream(METAL) in copy_from).
  template<>
  template <typename T>
  void cross_device_primitives<Device::METAL, Device::CPU>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }


  // -------------------------------------------------------------------------
  // primitives<Device::METAL>
  // -------------------------------------------------------------------------

  // at — unified memory: direct CPU read is always valid.
  template<>
  template <typename T>
  T primitives<Device::METAL>::at(const T* x, dim_t index) {
    return x[index];
  }

  // copy — both src and dst are Shared-mode MTLBuffer contents pointers;
  // memcpy is correct.  GPU sync is the caller's responsibility.
  template<>
  template <typename T>
  void primitives<Device::METAL>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

// Macro to generate a stub body for methods not yet implemented in Metal.
#define METAL_STUB(name) \
  throw std::runtime_error("primitives<METAL>::" #name ": not yet implemented (scheduled for M4)")

  // M4.1 — Memory primitives.
  //
  // Unified memory: the Metal buffer's contents pointer is CPU-writable.
  // CPU writes are visible to any subsequent GPU command encoding (Apple
  // Silicon memory coherency guarantee).  No GPU kernel is needed.

  template<>
  template <typename T>
  void primitives<Device::METAL>::fill(T* x, T a, dim_t size) {
    std::fill(x, x + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0; i < size; ++i, x += inc_x) {
      *x = a;
    }
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    for (dim_t i = 0; i < num_indices; ++i) {
      x[indices[i]] = a;
    }
  }

  // convert — std::copy relies on implicit narrowing/widening conversions
  // defined by half_float::half and bfloat16_t assignment operators.
  template<>
  template <typename U, typename V>
  void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
    std::copy(x, x + size, y);
  }

  // M4.3 — Reduction primitives (GPU two-pass parallel reduction).
  //
  // Pass 1 (GPU): each threadgroup of kReductionTGS threads reduces its tile
  //               of input to one partial result in a shared-mode MTLBuffer.
  // Pass 2 (CPU): the host reduces the ceil(N/kReductionTGS) partial results.
  //
  // commit_and_wait() is called before reading partial results; this also
  // flushes any pending GPU writes to the input array.

  template<>
  template <typename T>
  T primitives<Device::METAL>::sum(const T* array, dim_t size) {
    if (size == 0) { return T(0); }
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_sum_%s", MetalTypeName<T>::value);
    uint32_t n = static_cast<uint32_t>(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> out_buf = alloc_temp_buffer(num_groups * sizeof(T));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf offset:inp_off        atIndex:0];
    [enc setBuffer:out_buf offset:0              atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t)     atIndex:2];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(T) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const T* partials = static_cast<const T*>([out_buf contents]);
    return std::accumulate(partials, partials + num_groups, T(0));
  }

  template<>
  template <typename T>
  dim_t primitives<Device::METAL>::max_element(const T* array, dim_t size) {
    if (size == 0) { return 0; }
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_max_element_%s", MetalTypeName<T>::value);
    uint32_t n = static_cast<uint32_t>(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf  = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> vals_buf = alloc_temp_buffer(num_groups * sizeof(float));
    id<MTLBuffer> idxs_buf = alloc_temp_buffer(num_groups * sizeof(uint32_t));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf  offset:inp_off        atIndex:0];
    [enc setBuffer:vals_buf offset:0              atIndex:1];
    [enc setBuffer:idxs_buf offset:0              atIndex:2];
    [enc setBytes:&n length:sizeof(uint32_t)      atIndex:3];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(float)    atIndex:0];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(uint32_t) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const float*    pv = static_cast<const float*>([vals_buf contents]);
    const uint32_t* pi = static_cast<const uint32_t*>([idxs_buf contents]);
    float    best_val = pv[0];
    uint32_t best_idx = pi[0];
    for (uint32_t g = 1; g < num_groups; ++g) {
      if (pv[g] > best_val) {
        best_val = pv[g];
        best_idx = pi[g];
      }
    }
    return static_cast<dim_t>(best_idx);
  }

  template<>
  template <typename T>
  T primitives<Device::METAL>::max(const T* array, dim_t size) {
    if (size == 0) { return T(0); }
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_max_%s", MetalTypeName<T>::value);
    uint32_t n = static_cast<uint32_t>(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> out_buf = alloc_temp_buffer(num_groups * sizeof(T));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf offset:inp_off        atIndex:0];
    [enc setBuffer:out_buf offset:0              atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t)     atIndex:2];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(T) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const T* partials = static_cast<const T*>([out_buf contents]);
    return *std::max_element(partials, partials + num_groups);
  }

  // amax: max of absolute values, returned as type T.
  // The GPU kernel accumulates in float (handles all numeric types uniformly);
  // the output partial buffer is always float*.  CPU converts back to T.
  template<>
  template <typename T>
  T primitives<Device::METAL>::amax(const T* array, dim_t size) {
    if (size == 0) { return T(0); }
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_amax_%s", MetalTypeName<T>::value);
    uint32_t n = static_cast<uint32_t>(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> out_buf = alloc_temp_buffer(num_groups * sizeof(float));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf offset:inp_off          atIndex:0];
    [enc setBuffer:out_buf offset:0                atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t)       atIndex:2];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const float* partials = static_cast<const float*>([out_buf contents]);
    float result = *std::max_element(partials, partials + num_groups);
    return T(result);
  }

  // M4.2 — add(scalar, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::add(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  // M4.2 — add(vector, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::add(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    METAL_STUB(add_batch_broadcast);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_depth_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    METAL_STUB(add_depth_broadcast);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_block_broadcast(
      const T* a, const T* b, T* c, dim_t block, dim_t a_size, dim_t b_size) {
    METAL_STUB(add_block_broadcast);
  }

  // M4.2 — sub(vector, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::sub(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "sub_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::min(T a, const T* x, T* y, dim_t size) {
    METAL_STUB(min);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size) {
    METAL_STUB(min);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size) {
    METAL_STUB(max);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size) {
    METAL_STUB(max);
  }

  // M4.2 — mul(scalar, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  // M4.2 — mul(vector, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    METAL_STUB(mul_batch_broadcast);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::penalize_previous_tokens(
      T*, const T*, const int32_t*, T, dim_t, dim_t, dim_t) {
    METAL_STUB(penalize_previous_tokens);
  }

  template<>
  void primitives<Device::METAL>::prepare_length_mask(
      const int32_t*, dim_t, dim_t, dim_t, bool, bool, int32_t*) {
    METAL_STUB(prepare_length_mask);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    METAL_STUB(transpose_2d);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_3d(
      const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    METAL_STUB(transpose_3d);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_4d(
      const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    METAL_STUB(transpose_4d);
  }

  template<>
  template <typename T>
  float primitives<Device::METAL>::logsumexp(const T* x, dim_t size) {
    METAL_STUB(logsumexp);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::exp(const T* x, T* y, dim_t size) {
    METAL_STUB(exp);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::log(const T* x, T* y, dim_t size) {
    METAL_STUB(log);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::cos(const T* x, T* y, dim_t size) {
    METAL_STUB(cos);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::sin(const T* x, T* y, dim_t size) {
    METAL_STUB(sin);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::tanh(const T* x, T* y, dim_t size) {
    METAL_STUB(tanh);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::relu(const T* x, T* y, dim_t size) {
    METAL_STUB(relu);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::gelu(const T* x, T* y, dim_t size) {
    METAL_STUB(gelu);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::gelu_tanh(const T* x, T* y, dim_t size) {
    METAL_STUB(gelu_tanh);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    METAL_STUB(gelu_sigmoid);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::sigmoid(const T* x, T* y, dim_t size) {
    METAL_STUB(sigmoid);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::swish(const T* x, T* y, dim_t size) {
    METAL_STUB(swish);
  }

  template<>
  void primitives<Device::METAL>::compute_u8_compensation(
      const int8_t*, bool, dim_t, dim_t, float, int32_t*) {
    METAL_STUB(compute_u8_compensation);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::METAL>::gemm_pack_b(
      const T*, bool, dim_t, dim_t, float, T*) {
    return 0;  // Packing not supported.
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::METAL>::gemm(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const In*, dim_t, const In*, dim_t,
      float, Out*, dim_t, const Out*) {
    METAL_STUB(gemm);
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::METAL>::gemm_batch_strided(
      bool, bool, dim_t, dim_t, dim_t,
      float, const In*, dim_t, dim_t, const In*, dim_t, dim_t,
      float, Out*, dim_t, dim_t, dim_t) {
    METAL_STUB(gemm_batch_strided);
  }

#undef METAL_STUB


  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                        \
  template T                                                                   \
  primitives<Device::METAL>::at(const T* x, dim_t index);                     \
  template void                                                                \
  primitives<Device::METAL>::fill(T* x, T a, dim_t size);                     \
  template void                                                                \
  primitives<Device::METAL>::strided_fill(T* x, T a, dim_t inc_x, dim_t size);\
  template void                                                                \
  primitives<Device::METAL>::indexed_fill(T*, T, const int32_t*, dim_t);      \
  template void                                                                \
  primitives<Device::METAL>::copy<T>(const T* x, T* y, dim_t size);           \
  template T                                                                   \
  primitives<Device::METAL>::sum(const T* array, dim_t size);                 \
  template dim_t                                                               \
  primitives<Device::METAL>::max_element(const T* array, dim_t size);         \
  template T                                                                   \
  primitives<Device::METAL>::max(const T* array, dim_t size);                 \
  template T                                                                   \
  primitives<Device::METAL>::amax(const T* array, dim_t size);                \
  template void                                                                \
  primitives<Device::METAL>::add(T a, const T* x, T* y, dim_t size);          \
  template void                                                                \
  primitives<Device::METAL>::add(const T* a, const T* b, T* c, dim_t size);   \
  template void                                                                \
  primitives<Device::METAL>::add_batch_broadcast(const T* a, const T* b,      \
                                                  T* c, dim_t a_size,         \
                                                  dim_t b_size);              \
  template void                                                                \
  primitives<Device::METAL>::add_depth_broadcast(const T* a, const T* b,      \
                                                  T* c, dim_t a_size,         \
                                                  dim_t b_size);              \
  template void                                                                \
  primitives<Device::METAL>::add_block_broadcast(const T* a, const T* b,      \
                                                  T* c, dim_t block,          \
                                                  dim_t a_size, dim_t b_size);\
  template void                                                                \
  primitives<Device::METAL>::sub(const T* a, const T* b, T* c, dim_t size);   \
  template void                                                                \
  primitives<Device::METAL>::min(T a, const T* x, T* y, dim_t size);          \
  template void                                                                \
  primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size);   \
  template void                                                                \
  primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size);          \
  template void                                                                \
  primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size);   \
  template void                                                                \
  primitives<Device::METAL>::mul(T a, const T* x, T* y, dim_t size);          \
  template void                                                                \
  primitives<Device::METAL>::mul(const T* a, const T* b, T* c, dim_t size);   \
  template void                                                                \
  primitives<Device::METAL>::mul_batch_broadcast(const T* a, const T* b,      \
                                                  T* c, dim_t a_size,         \
                                                  dim_t b_size);              \
  template void                                                                \
  primitives<Device::METAL>::penalize_previous_tokens(T*,                     \
                                                       const T*,              \
                                                       const int32_t*,        \
                                                       T,                     \
                                                       dim_t,                 \
                                                       dim_t,                 \
                                                       dim_t);                \
  template void                                                                \
  primitives<Device::METAL>::transpose_2d(const T* a,                         \
                                           const dim_t* dims,                 \
                                           T* b);                             \
  template void                                                                \
  primitives<Device::METAL>::transpose_3d(const T* a,                         \
                                           const dim_t* dims,                 \
                                           const dim_t* perm,                 \
                                           T* b);                             \
  template void                                                                \
  primitives<Device::METAL>::transpose_4d(const T* a,                         \
                                           const dim_t* dims,                 \
                                           const dim_t* perm,                 \
                                           T* b);                             \
  template void                                                                \
  cross_device_primitives<Device::CPU, Device::METAL>::copy<T>(const T*, T*, dim_t); \
  template void                                                                \
  cross_device_primitives<Device::METAL, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

  // convert specialisations (not covered by DECLARE_ALL_TYPES).
  template void primitives<Device::METAL>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::METAL>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::METAL>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::METAL>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::METAL>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::METAL>::convert(const bfloat16_t*, float16_t*, dim_t);

#define DECLARE_FLOAT_IMPL(T)                                                  \
  template void primitives<Device::METAL>::relu(const T*, T*, dim_t);         \
  template void primitives<Device::METAL>::gelu(const T*, T*, dim_t);         \
  template void primitives<Device::METAL>::gelu_tanh(const T*, T*, dim_t);    \
  template void primitives<Device::METAL>::gelu_sigmoid(const T*, T*, dim_t); \
  template void primitives<Device::METAL>::sigmoid(const T*, T*, dim_t);      \
  template void primitives<Device::METAL>::swish(const T*, T*, dim_t);        \
  template float primitives<Device::METAL>::logsumexp(const T*, dim_t);       \
  template void primitives<Device::METAL>::sin(const T*, T*, dim_t);          \
  template void primitives<Device::METAL>::cos(const T*, T*, dim_t);          \
  template void primitives<Device::METAL>::tanh(const T*, T*, dim_t);         \
  template void primitives<Device::METAL>::exp(const T*, T*, dim_t);          \
  template void primitives<Device::METAL>::log(const T*, T*, dim_t);

  DECLARE_FLOAT_IMPL(float)
  DECLARE_FLOAT_IMPL(float16_t)
  DECLARE_FLOAT_IMPL(bfloat16_t)

  // gemm and gemm_pack_b are instantiated per (In, Out) pair in M4.
  // Provide the float32 pair here so the linker is satisfied for basic builds.
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const float*, bool, dim_t, dim_t, float, float*);
  template void primitives<Device::METAL>::gemm<float, float>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, const float*, dim_t,
      float, float*, dim_t, const float*);
  template void primitives<Device::METAL>::gemm_batch_strided<float, float>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, dim_t, const float*, dim_t, dim_t,
      float, float*, dim_t, dim_t, dim_t);

}  // namespace ctranslate2
