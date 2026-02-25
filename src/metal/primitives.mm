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
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

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

// ---------------------------------------------------------------------------
// M4.4 — GEMM infrastructure
// ---------------------------------------------------------------------------

// Map CTranslate2 scalar types to MPSDataType.
// Only FP32 and FP16 are used for Path A (MPSMatrixMultiplication).
template <typename T> struct MPS_Dtype;
template<> struct MPS_Dtype<float>                      { static const MPSDataType value = MPSDataTypeFloat32; };
template<> struct MPS_Dtype<ctranslate2::float16_t>     { static const MPSDataType value = MPSDataTypeFloat16; };

// Path A — FP32 and FP16 GEMM via MPSMatrixMultiplication.
//
// Encodes a single matrix multiplication into the deferred per-thread command
// buffer.  No commit is issued here; the caller flushes via synchronize_stream.
//
// Physical layout of A in memory:
//   !transpose_a  →  rows = m, cols = k, rowBytes = lda * sizeof(T)
//    transpose_a  →  rows = k, cols = m, rowBytes = lda * sizeof(T)
// (Same logic applies to B with its own transpose flag.)
//
// MPSMatrix requires rowBytes >= [MPSMatrixDescriptor rowBytesForColumns:cols dataType:dtype].
// For small matrices (e.g. 4 cols of Float16 → natural 8 bytes vs required 16) the natural
// stride may be below the hardware minimum.  When that happens we copy to a row-padded
// temporary buffer.  For the output matrix we commit+wait and unpack back to c after GEMM.
// For large production matrices the natural stride already satisfies the requirement, so
// no copies occur and the full deferred-commit pipeline is preserved.
template <typename T>
static void dispatch_mps_gemm(bool transpose_a, bool transpose_b,
                               ctranslate2::dim_t m,
                               ctranslate2::dim_t n,
                               ctranslate2::dim_t k,
                               float alpha,
                               const T* a, ctranslate2::dim_t lda,
                               const T* b, ctranslate2::dim_t ldb,
                               float beta,
                               T* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;

  constexpr NSUInteger elem = sizeof(T);
  const MPSDataType dtype = MPS_Dtype<T>::value;

  // Physical dimensions of each matrix as stored in memory.
  const NSUInteger rows_a = transpose_a ? (NSUInteger)k : (NSUInteger)m;
  const NSUInteger cols_a = transpose_a ? (NSUInteger)m : (NSUInteger)k;
  const NSUInteger rows_b = transpose_b ? (NSUInteger)n : (NSUInteger)k;
  const NSUInteger cols_b = transpose_b ? (NSUInteger)k : (NSUInteger)n;
  const NSUInteger rows_c = (NSUInteger)m;
  const NSUInteger cols_c = (NSUInteger)n;

  // Natural rowBytes from the caller's stride.
  const NSUInteger nat_rb_a = (NSUInteger)lda * elem;
  const NSUInteger nat_rb_b = (NSUInteger)ldb * elem;
  const NSUInteger nat_rb_c = (NSUInteger)ldc * elem;

  // MPS hardware-required minimum rowBytes (queried inside an autorelease pool
  // since the class method may create transient ObjC objects internally).
  NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
  @autoreleasepool {
    mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a dataType:dtype];
    mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b dataType:dtype];
    mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:cols_c dataType:dtype];
  }

  const bool pad_a = (nat_rb_a < mps_rb_a);
  const bool pad_b = (nat_rb_b < mps_rb_b);
  const bool pad_c = (nat_rb_c < mps_rb_c);

  // Prepare buffers — copy to row-padded temps when natural stride is too small.
  id<MTLBuffer> buf_a = nil, buf_b = nil, buf_c = nil;
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> tmp_a = nil, tmp_b = nil, tmp_c = nil;

  if (pad_a) {
    tmp_a = alloc_temp_buffer(rows_a * mps_rb_a);
    auto* dst = static_cast<uint8_t*>([tmp_a contents]);
    auto* src = reinterpret_cast<const uint8_t*>(a);
    for (NSUInteger r = 0; r < rows_a; ++r)
      std::memcpy(dst + r * mps_rb_a, src + r * nat_rb_a, nat_rb_a);
    buf_a = tmp_a;
  } else {
    buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  }

  if (pad_b) {
    tmp_b = alloc_temp_buffer(rows_b * mps_rb_b);
    auto* dst = static_cast<uint8_t*>([tmp_b contents]);
    auto* src = reinterpret_cast<const uint8_t*>(b);
    for (NSUInteger r = 0; r < rows_b; ++r)
      std::memcpy(dst + r * mps_rb_b, src + r * nat_rb_b, nat_rb_b);
    buf_b = tmp_b;
  } else {
    buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  }

  if (pad_c) {
    tmp_c = alloc_temp_buffer(rows_c * mps_rb_c);
    auto* dst = static_cast<uint8_t*>([tmp_c contents]);
    if (beta != 0.0f) {
      // Pack existing c into tmp_c so MPS accumulates correctly.
      auto* src = reinterpret_cast<const uint8_t*>(c);
      for (NSUInteger r = 0; r < rows_c; ++r)
        std::memcpy(dst + r * mps_rb_c, src + r * nat_rb_c, nat_rb_c);
    } else {
      std::memset(dst, 0, rows_c * mps_rb_c);
    }
    buf_c = tmp_c;
  } else {
    buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  }

  const NSUInteger rb_a = pad_a ? mps_rb_a : nat_rb_a;
  const NSUInteger rb_b = pad_b ? mps_rb_b : nat_rb_b;
  const NSUInteger rb_c = pad_c ? mps_rb_c : nat_rb_c;

  // Obtain the command buffer BEFORE entering @autoreleasepool.
  // If get_current_command_buffer() is called inside the pool, the returned
  // id<MTLCommandBuffer> gets autoreleased into the scoped pool.  When the pool
  // drains, the autorelease release interacts with the thread_local strong
  // reference in an unexpected way, leaving _thread_buffer dangling.
  // Fetching cmd outside the pool ensures a stable strong local keeps the
  // buffer alive across the pool drain.
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();

  @autoreleasepool {
    MPSMatrixDescriptor* descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                              columns:cols_a
                                             rowBytes:rb_a
                                             dataType:dtype];
    MPSMatrixDescriptor* descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                              columns:cols_b
                                             rowBytes:rb_b
                                             dataType:dtype];
    MPSMatrixDescriptor* descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_c
                                              columns:cols_c
                                             rowBytes:rb_c
                                             dataType:dtype];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:off_a descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:off_b descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:off_c descriptor:descC];

    id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
    MPSMatrixMultiplication* gemm_op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:(BOOL)transpose_a
                                          transposeRight:(BOOL)transpose_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:(double)beta];

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
  }

  // If C was routed to a padded temp buffer, flush the GPU and unpack back to c.
  if (pad_c) {
    ctranslate2::metal::commit_and_wait();
    const auto* src = static_cast<const uint8_t*>([tmp_c contents]);
    auto* dst = reinterpret_cast<uint8_t*>(c);
    for (NSUInteger r = 0; r < rows_c; ++r)
      std::memcpy(dst + r * nat_rb_c, src + r * mps_rb_c, nat_rb_c);
  }
}

// Path B — BF16 GEMM via MPSGraph.
//
// MPSMatrixMultiplication does NOT support BF16 (asserts at runtime).
// MPSGraph::matrixMultiplicationWithPrimaryTensor:secondaryTensor: does.
//
// Four graph instances are cached (one per trans_a × trans_b combination).
// Each graph uses nil-shape placeholders so MPSGraph JIT-compiles and caches
// a kernel per matrix shape on first use.  The transpose is baked into the
// graph as a fused transposeTensor node.
//
// Thread safety note: MPSGraph is not thread-safe.  CTranslate2 runs
// inference single-threaded per translator, so concurrent use is not expected.

struct Bf16GemmEntry {
  MPSGraph*       graph;
  MPSGraphTensor* ph_a;    // placeholder for A (physical layout, before transpose)
  MPSGraphTensor* ph_b;    // placeholder for B (physical layout, before transpose)
  MPSGraphTensor* result;  // output of the matmul node
};

static Bf16GemmEntry& get_bf16_graph(bool trans_a, bool trans_b) {
  static Bf16GemmEntry entries[4];
  static bool initialized[4] = {};
  static std::mutex mtx;

  const int idx = (trans_a ? 2 : 0) | (trans_b ? 1 : 0);
  std::lock_guard<std::mutex> lk(mtx);
  if (!initialized[idx]) {
    MPSGraph* g = [[MPSGraph alloc] init];
    // nil shape = dynamic: shape is supplied at runtime via MPSGraphTensorData.
    MPSGraphTensor* pA = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"A"];
    MPSGraphTensor* pB = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"B"];
    // Bake the transpose into the graph so it fuses with the matmul.
    MPSGraphTensor* opA = trans_a
        ? [g transposeTensor:pA dimension:0 withDimension:1 name:@"AT"] : pA;
    MPSGraphTensor* opB = trans_b
        ? [g transposeTensor:pB dimension:0 withDimension:1 name:@"BT"] : pB;
    MPSGraphTensor* tC = [g matrixMultiplicationWithPrimaryTensor:opA
                                                  secondaryTensor:opB
                                                             name:@"C"];
    entries[idx] = { g, pA, pB, tC };
    initialized[idx] = true;
  }
  return entries[idx];
}

// Run one BF16 GEMM synchronously using MPSGraph.
// Assumes pending GPU work has been flushed (call commit_and_wait() first).
//
// Only contiguous matrices are supported (lda == physical_cols of A, etc.)
// because MPSGraphTensorData has no stride/offset parameter.  The ops layer
// always passes contiguous matrices, so this is not a practical limitation.
//
// Non-zero buffer offsets (e.g. for batch iterations) are handled by copying
// the matrix data to a zero-offset temporary buffer before the graph run.
static void run_bf16_gemm_inner(bool trans_a, bool trans_b,
                                ctranslate2::dim_t m,
                                ctranslate2::dim_t n,
                                ctranslate2::dim_t k,
                                const ctranslate2::bfloat16_t* a, ctranslate2::dim_t lda,
                                const ctranslate2::bfloat16_t* b, ctranslate2::dim_t ldb,
                                ctranslate2::bfloat16_t* c, ctranslate2::dim_t ldc) {
  // Enforce contiguity.
  const ctranslate2::dim_t exp_lda = trans_a ? m : k;
  const ctranslate2::dim_t exp_ldb = trans_b ? k : n;
  if (lda != exp_lda || ldb != exp_ldb || ldc != n) {
    throw std::runtime_error(
        "Metal BF16 GEMM: only contiguous matrices are supported "
        "(lda/ldb/ldc must equal the physical column count)");
  }

  const Bf16GemmEntry& entry = get_bf16_graph(trans_a, trans_b);
  id<MTLCommandQueue> queue = ctranslate2::metal::get_metal_command_queue();

  // Physical shape of A and B as stored in memory (before any transpose
  // that the graph applies internally).
  NSArray<NSNumber*>* shape_a =
      trans_a ? @[@((int)k), @((int)m)] : @[@((int)m), @((int)k)];
  NSArray<NSNumber*>* shape_b =
      trans_b ? @[@((int)n), @((int)k)] : @[@((int)k), @((int)n)];

  const size_t bytes_a = (size_t)m * k * sizeof(ctranslate2::bfloat16_t);
  const size_t bytes_b = (size_t)k * n * sizeof(ctranslate2::bfloat16_t);

  // MPSGraphTensorData has no byte-offset parameter.  Copy to a zero-offset
  // temporary buffer when the input has a non-zero offset within its MTLBuffer
  // (occurs for non-first batch elements).
  NSUInteger off_a = 0, off_b = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

  id<MTLBuffer> tmp_a = nil, tmp_b = nil;
  if (off_a != 0) {
    tmp_a = alloc_temp_buffer(bytes_a);
    std::memcpy([tmp_a contents], a, bytes_a);
    buf_a = tmp_a;
  }
  if (off_b != 0) {
    tmp_b = alloc_temp_buffer(bytes_b);
    std::memcpy([tmp_b contents], b, bytes_b);
    buf_b = tmp_b;
  }

  @autoreleasepool {
    MPSGraphTensorData* tdA = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buf_a shape:shape_a dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData* tdB = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buf_b shape:shape_b dataType:MPSDataTypeBFloat16];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [entry.graph runWithMTLCommandQueue:queue
                                      feeds:@{entry.ph_a: tdA, entry.ph_b: tdB}
                              targetTensors:@[entry.result]
                           targetOperations:nil];

    if (!results || !results[entry.result]) {
      throw std::runtime_error("Metal BF16 GEMM: graph execution returned nil");
    }
    // c is a Shared MTLBuffer contents pointer — valid as CPU destination.
    // nil strides = packed/contiguous layout matching our row-major convention.
    [[results[entry.result] mpsndarray] readBytes:c strideBytes:nil];
  }
}

// Flush pending GPU work and run a single BF16 GEMM.
// Throws if alpha != 1.0 or beta != 0.0 (not supported by the MPSGraph path).
static void dispatch_bf16_gemm(bool trans_a, bool trans_b,
                                ctranslate2::dim_t m,
                                ctranslate2::dim_t n,
                                ctranslate2::dim_t k,
                                float alpha, float beta,
                                const ctranslate2::bfloat16_t* a, ctranslate2::dim_t lda,
                                const ctranslate2::bfloat16_t* b, ctranslate2::dim_t ldb,
                                ctranslate2::bfloat16_t* c, ctranslate2::dim_t ldc) {
  if (alpha != 1.0f || beta != 0.0f) {
    throw std::runtime_error(
        "Metal BF16 GEMM: only alpha=1.0 and beta=0.0 are supported");
  }
  if (m == 0 || n == 0 || k == 0) return;
  // MPSGraph runs synchronously on its own command queue, so we must flush
  // any in-flight encoders from the deferred command buffer first.
  ctranslate2::metal::commit_and_wait();
  run_bf16_gemm_inner(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc);
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
      bool a_is_packed, bool b_is_packed,
      bool transpose_a, bool transpose_b,
      dim_t m, dim_t n, dim_t k,
      float alpha,
      const In* a, dim_t lda,
      const In* b, dim_t ldb,
      float beta,
      Out* c, dim_t ldc,
      const Out* a_shift_compensation) {
    (void)a_is_packed; (void)b_is_packed; (void)a_shift_compensation;
    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      dispatch_mps_gemm<float>(transpose_a, transpose_b, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
      dispatch_mps_gemm<float16_t>(transpose_a, transpose_b, m, n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<In, bfloat16_t> && std::is_same_v<Out, bfloat16_t>) {
      dispatch_bf16_gemm(transpose_a, transpose_b, m, n, k,
                         alpha, beta, a, lda, b, ldb, c, ldc);
    } else {
      METAL_STUB(gemm);
    }
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::METAL>::gemm_batch_strided(
      bool transpose_a, bool transpose_b,
      dim_t m, dim_t n, dim_t k,
      float alpha,
      const In* a, dim_t lda, dim_t stridea,
      const In* b, dim_t ldb, dim_t strideb,
      float beta,
      Out* c, dim_t ldc, dim_t stridec,
      dim_t batch_size) {
    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      for (dim_t i = 0; i < batch_size; ++i) {
        dispatch_mps_gemm<float>(transpose_a, transpose_b, m, n, k,
                                 alpha, a + i * stridea, lda,
                                 b + i * strideb, ldb,
                                 beta, c + i * stridec, ldc);
      }
    } else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
      for (dim_t i = 0; i < batch_size; ++i) {
        dispatch_mps_gemm<float16_t>(transpose_a, transpose_b, m, n, k,
                                     alpha, a + i * stridea, lda,
                                     b + i * strideb, ldb,
                                     beta, c + i * stridec, ldc);
      }
    } else if constexpr (std::is_same_v<In, bfloat16_t> && std::is_same_v<Out, bfloat16_t>) {
      if (alpha != 1.0f || beta != 0.0f) {
        throw std::runtime_error(
            "Metal BF16 GEMM: only alpha=1.0 and beta=0.0 are supported");
      }
      // Flush all pending GPU work once before the batch loop.
      metal::commit_and_wait();
      for (dim_t i = 0; i < batch_size; ++i) {
        run_bf16_gemm_inner(transpose_a, transpose_b, m, n, k,
                            a + i * stridea, lda,
                            b + i * strideb, ldb,
                            c + i * stridec, ldc);
      }
    } else {
      METAL_STUB(gemm_batch_strided);
    }
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

  // GEMM explicit instantiations — one set per (In, Out) type pair.
  // FP32 and FP16 use MPSMatrixMultiplication (Path A).
  // BF16 uses MPSGraph (Path B).
  // INT8→INT32 falls through to the METAL_STUB (throws at runtime).
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const float*, bool, dim_t, dim_t, float, float*);
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const float16_t*, bool, dim_t, dim_t, float, float16_t*);
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const bfloat16_t*, bool, dim_t, dim_t, float, bfloat16_t*);

  template void primitives<Device::METAL>::gemm<float, float>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, const float*, dim_t,
      float, float*, dim_t, const float*);
  template void primitives<Device::METAL>::gemm<float16_t, float16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, const float16_t*, dim_t,
      float, float16_t*, dim_t, const float16_t*);
  template void primitives<Device::METAL>::gemm<bfloat16_t, bfloat16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, const bfloat16_t*, dim_t,
      float, bfloat16_t*, dim_t, const bfloat16_t*);

  template void primitives<Device::METAL>::gemm_batch_strided<float, float>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, dim_t, const float*, dim_t, dim_t,
      float, float*, dim_t, dim_t, dim_t);
  template void primitives<Device::METAL>::gemm_batch_strided<float16_t, float16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, dim_t, const float16_t*, dim_t, dim_t,
      float, float16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::METAL>::gemm_batch_strided<bfloat16_t, bfloat16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, dim_t, const bfloat16_t*, dim_t, dim_t,
      float, bfloat16_t*, dim_t, dim_t, dim_t);

}  // namespace ctranslate2
