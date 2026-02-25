// primitives<Device::METAL> — M3.2 + M4.1 + M4.2 implementation.
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
#include <mutex>
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

  template<>
  template <typename T>
  T primitives<Device::METAL>::sum(const T* array, dim_t size) {
    METAL_STUB(sum);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::METAL>::max_element(const T* array, dim_t size) {
    METAL_STUB(max_element);
  }

  template<>
  template <typename T>
  T primitives<Device::METAL>::max(const T* array, dim_t size) {
    METAL_STUB(max);
  }

  template<>
  template <typename T>
  T primitives<Device::METAL>::amax(const T* array, dim_t size) {
    METAL_STUB(amax);
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
