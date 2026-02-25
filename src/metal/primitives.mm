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

// Min / Max using ternary comparison (avoids MSL min()/max() overload
// ambiguity for bfloat; also correct for integer types without float cast).
#define DEFINE_MINMAX_BINARY(name, sel, T)                              \
  kernel void name##_##T(                                               \
      device const T* a [[buffer(0)]],                                  \
      device const T* b [[buffer(1)]],                                  \
      device       T* c [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { T va = a[gid], vb = b[gid]; c[gid] = (va sel vb) ? va : vb; }

#define DEFINE_MINMAX_SCALAR(name, sel, T)                              \
  kernel void name##_scalar_##T(                                        \
      device const T* x [[buffer(0)]],                                  \
      constant     T& a [[buffer(1)]],                                  \
      device       T* y [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { T vx = x[gid]; y[gid] = (vx sel a) ? vx : a; }

#define DEFINE_MINMAX(T)              \
  DEFINE_MINMAX_BINARY(min, <, T)    \
  DEFINE_MINMAX_BINARY(max, >, T)    \
  DEFINE_MINMAX_SCALAR(min, <, T)    \
  DEFINE_MINMAX_SCALAR(max, >, T)

DEFINE_MINMAX(float)
DEFINE_MINMAX(half)
DEFINE_MINMAX(int)
DEFINE_MINMAX(short)
DEFINE_MINMAX(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_MINMAX(bfloat)
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
// M4.5 — Activation / transcendental kernel infrastructure
// ---------------------------------------------------------------------------

// MSL source for all unary activation and transcendental kernels.
// This is the exact content of src/metal/kernels/activation.metal embedded
// as a C++ string and compiled at runtime via newLibraryWithSource:.
static constexpr const char* kActivationMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

// Metal Shading Language does not provide erf() in its standard library.
// We implement it via the Abramowitz & Stegun polynomial approximation
// (formula 7.1.28, max absolute error 1.5e-7):
//
//   t = 1 / (1 + 0.3275911 * |x|)
//   erf(x) ≈ sign(x) * (1 - poly(t) * exp(-x*x))
//   poly(t) = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
//
// Always operates in float32 regardless of kernel input type.
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

#define DEFINE_UNARY(name, T, expr)                     \
kernel void name##_##T(                                 \
    device const T* x [[buffer(0)]],                    \
    device       T* y [[buffer(1)]],                    \
    uint gid [[thread_position_in_grid]])                \
{ float v = (float)x[gid]; y[gid] = (T)(expr); }

#define DEFINE_ACTIVATION_OPS(T)                                                         \
  DEFINE_UNARY(exp,          T, exp(v))                                                  \
  DEFINE_UNARY(log,          T, log(v))                                                  \
  DEFINE_UNARY(cos,          T, cos(v))                                                  \
  DEFINE_UNARY(sin,          T, sin(v))                                                  \
  DEFINE_UNARY(tanh,         T, tanh(v))                                                 \
  DEFINE_UNARY(relu,         T, fmax(v, 0.f))                                            \
  DEFINE_UNARY(sigmoid,      T, 1.f / (1.f + exp(-v)))                                  \
  DEFINE_UNARY(swish,        T, v / (1.f + exp(-v)))                                    \
  DEFINE_UNARY(gelu,         T, 0.5f * v * (1.f + ct2_erf(v * 0.7071067811865475f)))   \
  DEFINE_UNARY(gelu_tanh,    T, 0.5f * v * (1.f + tanh(0.7978845608028654f *            \
                                (v + 0.044715f * v * v * v))))                           \
  DEFINE_UNARY(gelu_sigmoid, T, v / (1.f + exp(-1.702f * v)))

DEFINE_ACTIVATION_OPS(float)
DEFINE_ACTIVATION_OPS(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_ACTIVATION_OPS(bfloat)
#endif
)msl";

// Lazy-compile the activation MSL library.  Thread-safe; compiled once.
// We request Metal 3.1 (macOS 14+) to ensure erf() is available — it was
// added to the Metal standard math library in MSL 3.1.  Our deployment
// target is already macOS 14 (set in CMakeLists.txt for BF16 support).
static id<MTLLibrary> get_activation_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSString* src = [NSString stringWithUTF8String:kActivationMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:opts
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile activation library";
      if (err)
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO cache for activation kernels.
static id<MTLComputePipelineState> get_activation_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_activation_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: activation kernel not found: ") + name);
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

// Dispatch a unary element-wise activation kernel: y[i] = f(x[i]).
// x is buffer(0), y is buffer(1).
static void dispatch_unary(const char* kernel_name,
                            const void* x, void* y,
                            ctranslate2::dim_t size) {
  if (size == 0) return;
  id<MTLComputePipelineState> pso = get_activation_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_x = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x) offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y) offset:off_y atIndex:1];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}


// ---------------------------------------------------------------------------
// M4.6 — Broadcast kernel infrastructure
// ---------------------------------------------------------------------------

// MSL source for broadcast kernels.
// Canonical copy lives in src/metal/kernels/broadcast.metal.
static constexpr const char* kBroadcastMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#define DEFINE_BATCH_BROADCAST(name, op, T)                              \
kernel void name##_batch_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& a_size [[buffer(3)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[gid % a_size] op b[gid]; }

#define DEFINE_DEPTH_BROADCAST(name, op, T)                              \
kernel void name##_depth_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& depth [[buffer(3)]],                                 \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[gid / depth] op b[gid]; }

#define DEFINE_BLOCK_BROADCAST(name, op, T)                              \
kernel void name##_block_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& block  [[buffer(3)]],                                \
    constant  uint& a_size [[buffer(4)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[(gid / block) % a_size] op b[gid]; }

#define DEFINE_BROADCAST_OPS(T)       \
  DEFINE_BATCH_BROADCAST(add, +, T)   \
  DEFINE_DEPTH_BROADCAST(add, +, T)   \
  DEFINE_BLOCK_BROADCAST(add, +, T)   \
  DEFINE_BATCH_BROADCAST(mul, *, T)

DEFINE_BROADCAST_OPS(float)
DEFINE_BROADCAST_OPS(half)
DEFINE_BROADCAST_OPS(int)
DEFINE_BROADCAST_OPS(short)
DEFINE_BROADCAST_OPS(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_BROADCAST_OPS(bfloat)
#endif
)msl";

// Lazy-compile the broadcast MSL library.  Thread-safe; compiled once.
static id<MTLLibrary> get_broadcast_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kBroadcastMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:nil
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile broadcast library";
      if (err)
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO cache for broadcast kernels.
static id<MTLComputePipelineState> get_broadcast_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_broadcast_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: broadcast kernel not found: ") + name);
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

// Dispatch a broadcast kernel: 3 data buffers + 1 uint32 constant at buffer(3).
// Used by add_batch_broadcast (param0 = a_size) and
//         add_depth_broadcast (param0 = depth = b_size/a_size).
static void dispatch_broadcast1(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 ctranslate2::dim_t size,
                                 uint32_t param0) {
  if (size == 0) return;
  id<MTLComputePipelineState> pso = get_broadcast_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(c, &off_c) offset:off_c atIndex:2];
  [enc setBytes:&param0 length:sizeof(uint32_t) atIndex:3];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// Dispatch a broadcast kernel: 3 data buffers + 2 uint32 constants.
// Used by add_block_broadcast (param0 = block, param1 = a_size).
static void dispatch_broadcast2(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 ctranslate2::dim_t size,
                                 uint32_t param0, uint32_t param1) {
  if (size == 0) return;
  id<MTLComputePipelineState> pso = get_broadcast_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(c, &off_c) offset:off_c atIndex:2];
  [enc setBytes:&param0 length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&param1 length:sizeof(uint32_t) atIndex:4];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// M4.7 — Beam-search kernel infrastructure
// ---------------------------------------------------------------------------

// MSL source for beam-search kernels.
// Canonical copy lives in src/metal/kernels/beam_search.metal.
static constexpr const char* kBeamSearchMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#define DEFINE_PENALIZE(T)                                                      \
kernel void penalize_previous_tokens_##T(                                       \
    device       T*       scores          [[buffer(0)]],                        \
    device const T*       previous_scores [[buffer(1)]],                        \
    device const int*     previous_ids    [[buffer(2)]],                        \
    constant     float&   penalty         [[buffer(3)]],                        \
    constant     uint&    length          [[buffer(4)]],                        \
    constant     uint&    vocab_size      [[buffer(5)]],                        \
    uint batch_idx [[thread_position_in_grid]])                                 \
{                                                                               \
  for (uint j = 0; j < length; ++j) {                                          \
    uint read_idx  = batch_idx * length + j;                                    \
    uint write_idx = batch_idx * vocab_size + (uint)previous_ids[read_idx];    \
    float score = (float)previous_scores[read_idx];                             \
    float penalized = (score < 0.f) ? score * penalty : score / penalty;       \
    scores[write_idx] = (T)penalized;                                           \
  }                                                                             \
}

DEFINE_PENALIZE(float)
DEFINE_PENALIZE(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_PENALIZE(bfloat)
#endif
)msl";

// Lazy-compile the beam-search MSL library.  Thread-safe; compiled once.
static id<MTLLibrary> get_beam_search_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kBeamSearchMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:nil
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile beam_search library";
      if (err)
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO cache for beam-search kernels.
static id<MTLComputePipelineState> get_beam_search_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_beam_search_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: beam_search kernel not found: ") + name);
  }

  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [ctranslate2::metal::get_metal_device()
          newComputePipelineStateWithFunction:fn
                                       error:&err];
  if (pso == nil) {
    std::string msg = std::string("Metal: beam_search PSO creation failed for ") + name;
    if (err)
      msg += std::string(": ") + [err.localizedDescription UTF8String];
    throw std::runtime_error(msg);
  }
  cache[name] = pso;
  return pso;
}

// Dispatch penalize_previous_tokens: one thread per batch item.
// Each thread iterates over `length` previous IDs sequentially.
// Sequential within a thread ensures correct semantics when the same
// token ID appears multiple times (last write wins, matching CPU).
static void dispatch_penalize(const char* kernel_name,
                               void* scores,
                               const void* previous_scores,
                               const void* previous_ids,
                               float penalty,
                               uint32_t batch_size,
                               uint32_t length,
                               uint32_t vocab_size) {
  if (batch_size == 0 || length == 0) return;
  id<MTLComputePipelineState> pso = get_beam_search_pso(kernel_name);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_s = 0, off_ps = 0, off_pi = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(scores, &off_s)
          offset:off_s atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(previous_scores, &off_ps)
          offset:off_ps atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(previous_ids, &off_pi)
          offset:off_pi atIndex:2];
  [enc setBytes:&penalty    length:sizeof(float)    atIndex:3];
  [enc setBytes:&length     length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&vocab_size length:sizeof(uint32_t) atIndex:5];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(batch_size));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// M4.8 — Transpose kernel infrastructure
// ---------------------------------------------------------------------------

// Argument structs (C++ side).  Layout must exactly match the MSL structs.
// All fields are uint32_t, naturally aligned — no padding issues.

struct TransposeArgs2D {
  uint32_t rows, cols;
};

struct TransposeArgs3D {
  uint32_t a_ps0, a_ps1, a_ps2;  // permuted input strides
  uint32_t b_s0, b_s1;           // output strides (b_s2 = 1, implicit)
  uint32_t bd1;                   // output dim 1 (for % decomposition)
};

struct TransposeArgs4D {
  uint32_t a_ps0, a_ps1, a_ps2, a_ps3;
  uint32_t b_s0, b_s1, b_s2;
  uint32_t bd1, bd2;
};

// MSL source for transpose kernels.
// Canonical copy lives in src/metal/kernels/transpose.metal.
static constexpr const char* kTransposeMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

struct TransposeArgs2D { uint rows, cols; };
struct TransposeArgs3D { uint a_ps0, a_ps1, a_ps2; uint b_s0, b_s1; uint bd1; };
struct TransposeArgs4D { uint a_ps0, a_ps1, a_ps2, a_ps3; uint b_s0, b_s1, b_s2; uint bd1, bd2; };

#define DEFINE_TRANSPOSE(T)                                                     \
kernel void transpose_2d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs2D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{ b[gid] = a[(gid % args.rows) * args.cols + (gid / args.rows)]; }             \
                                                                                \
kernel void transpose_3d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs3D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{                                                                               \
  uint i0 =  gid / args.b_s0;                                                  \
  uint i1 = (gid / args.b_s1) % args.bd1;                                      \
  uint i2 =  gid % args.b_s1;                                                  \
  b[gid] = a[i0 * args.a_ps0 + i1 * args.a_ps1 + i2 * args.a_ps2];            \
}                                                                               \
                                                                                \
kernel void transpose_4d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs4D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{                                                                               \
  uint i0 =  gid / args.b_s0;                                                  \
  uint i1 = (gid / args.b_s1) % args.bd1;                                      \
  uint i2 = (gid / args.b_s2) % args.bd2;                                      \
  uint i3 =  gid % args.b_s2;                                                  \
  b[gid] = a[i0 * args.a_ps0 + i1 * args.a_ps1 +                               \
             i2 * args.a_ps2 + i3 * args.a_ps3];                               \
}

DEFINE_TRANSPOSE(float)
DEFINE_TRANSPOSE(half)
DEFINE_TRANSPOSE(int)
DEFINE_TRANSPOSE(short)
DEFINE_TRANSPOSE(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_TRANSPOSE(bfloat)
#endif
)msl";

// Lazy-compile the transpose MSL library.  Thread-safe; compiled once.
static id<MTLLibrary> get_transpose_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  std::call_once(flag, [] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kTransposeMSL];
    lib = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src
                     options:nil
                       error:&err];
    if (lib == nil) {
      std::string msg = "Metal: failed to compile transpose library";
      if (err)
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      throw std::runtime_error(msg);
    }
  });
  return lib;
}

// PSO cache for transpose kernels.
static id<MTLComputePipelineState> get_transpose_pso(const char* name) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(name);
  if (it != cache.end()) {
    return it->second;
  }

  id<MTLLibrary> lib = get_transpose_library();
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil) {
    throw std::runtime_error(std::string("Metal: transpose kernel not found: ") + name);
  }

  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [ctranslate2::metal::get_metal_device()
          newComputePipelineStateWithFunction:fn
                                       error:&err];
  if (pso == nil) {
    std::string msg = std::string("Metal: transpose PSO creation failed for ") + name;
    if (err)
      msg += std::string(": ") + [err.localizedDescription UTF8String];
    throw std::runtime_error(msg);
  }
  cache[name] = pso;
  return pso;
}

// Shared dispatch: 2 data buffers + 1 args struct passed via setBytes:.
// n = total output elements.
static void dispatch_transpose(const char* kname,
                                const void* a, void* b, ctranslate2::dim_t n,
                                const void* args, size_t args_size) {
  if (n == 0) return;
  id<MTLComputePipelineState> pso = get_transpose_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_a = 0, off_b = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBytes:args length:args_size atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(n));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

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

  // at — unified memory: GPU writes are visible to CPU only after the pending
  // command buffer is committed and completes.  Flush before reading so the
  // caller always sees up-to-date data, even when called immediately after a
  // GPU kernel that wrote x (e.g. max_element → scalar readback in beam search).
  template<>
  template <typename T>
  T primitives<Device::METAL>::at(const T* x, dim_t index) {
    metal::commit_and_wait();  // flush any pending GPU writes to x
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
  // commit_and_wait() flushes any pending GPU writes to x before the CPU
  // reads it, matching the same guard used by at(), logsumexp(), and
  // prepare_length_mask().
  template<>
  template <typename U, typename V>
  void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
    metal::commit_and_wait();  // flush pending GPU writes before CPU read
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

  // M4.6 — add_batch_broadcast — GPU kernel.
  // CPU ref: for i in [0, b_size/a_size): c[i*a_size+j] = a[j] + b[i*a_size+j]
  // Kernel:  c[gid] = a[gid % a_size] + b[gid]
  template<>
  template <typename T>
  void primitives<Device::METAL>::add_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_batch_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast1(kname, a, b, c, b_size, static_cast<uint32_t>(a_size));
  }

  // M4.6 — add_depth_broadcast — GPU kernel.
  // CPU ref: depth = b_size/a_size; for i in [0,a_size): c[i*depth+k] = a[i] + b[i*depth+k]
  // Kernel:  c[gid] = a[gid / depth] + b[gid]
  template<>
  template <typename T>
  void primitives<Device::METAL>::add_depth_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_depth_broadcast_%s", MetalTypeName<T>::value);
    uint32_t depth = static_cast<uint32_t>(b_size / a_size);
    dispatch_broadcast1(kname, a, b, c, b_size, depth);
  }

  // M4.6 — add_block_broadcast — GPU kernel.
  // CPU ref: for i in [0,b_size/block): c[i*block+k] = a[i%a_size] + b[i*block+k]
  // Kernel:  c[gid] = a[(gid/block) % a_size] + b[gid]
  template<>
  template <typename T>
  void primitives<Device::METAL>::add_block_broadcast(
      const T* a, const T* b, T* c, dim_t block, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_block_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast2(kname, a, b, c, b_size,
                        static_cast<uint32_t>(block),
                        static_cast<uint32_t>(a_size));
  }

  // M4.2 — sub(vector, vector, out) — GPU kernel
  template<>
  template <typename T>
  void primitives<Device::METAL>::sub(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "sub_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  // M4.2 (extended) — min(scalar, vector, out) — GPU kernel: y[i] = min(a, x[i])
  template<>
  template <typename T>
  void primitives<Device::METAL>::min(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "min_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  // M4.2 (extended) — min(vector, vector, out) — GPU kernel: c[i] = min(a[i], b[i])
  template<>
  template <typename T>
  void primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "min_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  // M4.2 (extended) — max(scalar, vector, out) — GPU kernel: y[i] = max(a, x[i])
  template<>
  template <typename T>
  void primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "max_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  // M4.2 (extended) — max(vector, vector, out) — GPU kernel: c[i] = max(a[i], b[i])
  template<>
  template <typename T>
  void primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "max_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
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

  // M4.6 — mul_batch_broadcast — GPU kernel.
  // CPU ref: for i in [0, b_size/a_size): c[i*a_size+j] = a[j] * b[i*a_size+j]
  // Kernel:  c[gid] = a[gid % a_size] * b[gid]
  template<>
  template <typename T>
  void primitives<Device::METAL>::mul_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_batch_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast1(kname, a, b, c, b_size, static_cast<uint32_t>(a_size));
  }

  // M4.7 — penalize_previous_tokens — GPU kernel (one thread per batch item).
  template<>
  template <typename T>
  void primitives<Device::METAL>::penalize_previous_tokens(
      T* scores, const T* previous_scores, const int32_t* previous_ids,
      T penalty, dim_t batch_size, dim_t length, dim_t vocabulary_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "penalize_previous_tokens_%s",
                  MetalTypeName<T>::value);
    dispatch_penalize(kname,
                      scores, previous_scores, previous_ids,
                      static_cast<float>(penalty),
                      static_cast<uint32_t>(batch_size),
                      static_cast<uint32_t>(length),
                      static_cast<uint32_t>(vocabulary_size));
  }

  // M4.7 — prepare_length_mask — CPU-side with GPU flush.
  //
  // Mask creation is O(batch × heads × queries) — typically small (e.g.
  // 8 × 8 × 512 = 32 K ints).  The GPU overhead of scheduling a kernel
  // dominates for these sizes, so CPU is the correct implementation.
  //
  // `lengths` may have been written by a prior GPU operation (e.g. a gather
  // over a padded-batch lengths tensor).  commit_and_wait() ensures those
  // writes are committed and visible to the CPU before the loop reads them.
  //
  // CPU writes to a Shared-mode MTLBuffer are immediately coherent with the
  // GPU on Apple Silicon — no extra flush is needed before the next kernel.
  template<>
  void primitives<Device::METAL>::prepare_length_mask(
      const int32_t* lengths, dim_t batch_size, dim_t num_heads,
      dim_t num_queries, bool mask_future, bool multi_query, int32_t* mask) {
    metal::commit_and_wait();  // flush any pending GPU writes to lengths
    for (dim_t b = 0; b < batch_size; ++b) {
      const auto length = lengths[b];
      auto* batch_mask = mask + b * num_heads * num_queries;
      for (dim_t i = 0; i < num_heads * num_queries; ++i) {
        batch_mask[i] = (mask_future
                         ? std::min(length,
                                    int32_t((multi_query ? i / num_heads
                                                         : i % num_queries) + 1))
                         : length);
      }
    }
  }

  // M4.8 — transpose_2d — GPU kernel (implicit perm = [1,0]).
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_2d_%s", MetalTypeName<T>::value);
    TransposeArgs2D args{static_cast<uint32_t>(dims[0]), static_cast<uint32_t>(dims[1])};
    dispatch_transpose(kname, a, b, dims[0] * dims[1], &args, sizeof(args));
  }

  // M4.8 — transpose_3d — GPU kernel (arbitrary 3D permutation).
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_3d(
      const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_3d_%s", MetalTypeName<T>::value);
    const uint32_t a_stride[3] = {
      static_cast<uint32_t>(dims[1] * dims[2]),
      static_cast<uint32_t>(dims[2]),
      1u
    };
    const uint32_t bd1 = static_cast<uint32_t>(dims[perm[1]]);
    const uint32_t bd2 = static_cast<uint32_t>(dims[perm[2]]);
    TransposeArgs3D args{
      a_stride[perm[0]], a_stride[perm[1]], a_stride[perm[2]],
      bd1 * bd2, bd2,  // b_s0, b_s1
      bd1
    };
    dispatch_transpose(kname, a, b,
                       dims[0] * dims[1] * dims[2], &args, sizeof(args));
  }

  // M4.8 — transpose_4d — GPU kernel (arbitrary 4D permutation).
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_4d(
      const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_4d_%s", MetalTypeName<T>::value);
    const uint32_t a_stride[4] = {
      static_cast<uint32_t>(dims[1] * dims[2] * dims[3]),
      static_cast<uint32_t>(dims[2] * dims[3]),
      static_cast<uint32_t>(dims[3]),
      1u
    };
    const uint32_t bd1 = static_cast<uint32_t>(dims[perm[1]]);
    const uint32_t bd2 = static_cast<uint32_t>(dims[perm[2]]);
    const uint32_t bd3 = static_cast<uint32_t>(dims[perm[3]]);
    TransposeArgs4D args{
      a_stride[perm[0]], a_stride[perm[1]], a_stride[perm[2]], a_stride[perm[3]],
      bd1 * bd2 * bd3, bd2 * bd3, bd3,  // b_s0, b_s1, b_s2
      bd1, bd2
    };
    dispatch_transpose(kname, a, b,
                       dims[0] * dims[1] * dims[2] * dims[3], &args, sizeof(args));
  }

  // M4.5 — logsumexp: CPU-side after flushing pending GPU work.
  // log(Σ exp(x[i])) computed stably as log(Σ exp(x[i] - max)) + max.
  template<>
  template <typename T>
  float primitives<Device::METAL>::logsumexp(const T* x, dim_t size) {
    if (size == 0) return 0.f;
    metal::commit_and_wait();  // flush any pending GPU writes to x
    float maxval = (float)x[0];
    for (dim_t i = 1; i < size; ++i)
      maxval = std::max(maxval, (float)x[i]);
    float sum = 0.f;
    for (dim_t i = 0; i < size; ++i)
      sum += std::exp((float)x[i] - maxval);
    return std::log(sum) + maxval;
  }

  // M4.5 — GPU unary activation / transcendental kernels.
  // Each dispatches into the per-thread command buffer (encode-only;
  // committed at synchronize_stream).

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
  METAL_UNARY_OP(cos,          "cos")
  METAL_UNARY_OP(sin,          "sin")
  METAL_UNARY_OP(tanh,         "tanh")
  METAL_UNARY_OP(relu,         "relu")
  METAL_UNARY_OP(sigmoid,      "sigmoid")
  METAL_UNARY_OP(swish,        "swish")
  METAL_UNARY_OP(gelu,         "gelu")
  METAL_UNARY_OP(gelu_tanh,    "gelu_tanh")
  METAL_UNARY_OP(gelu_sigmoid, "gelu_sigmoid")

#undef METAL_UNARY_OP

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
