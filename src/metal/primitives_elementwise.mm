// src/metal/primitives_elementwise.mm
//
// M4.2 — Element-wise arithmetic primitives (add, sub, mul, min, max).
// M4.5 — Unary activation / transcendental primitives (exp, log, relu, gelu, …).
// M4.6 — Broadcast primitives (add_batch_broadcast, add_block_broadcast, …).
//
// All kernels encode into the per-thread command buffer (encode-only);
// results are visible to the GPU after synchronize_stream(METAL).

#include "metal/primitives_infra.h"

namespace {

// ---------------------------------------------------------------------------
// Elementwise kernel infrastructure (binary and scalar-op-vector)
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_elementwise_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kElementwiseMSL, "elementwise");
}

static id<MTLComputePipelineState> get_elementwise_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_elementwise_library, name);
}

// c[i] = a[i] op b[i]
static void dispatch_binary(const char* kernel_name,
                             const void* a, const void* b, void* c,
                             ctranslate2::dim_t size) {
  if (size == 0) return;
  (void)ct2_u32(size);
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

// y[i] = scalar op x[i]  (scalar passed via setBytes:, no buffer allocation)
static void dispatch_scalar(const char* kernel_name,
                             const void* scalar_val, size_t scalar_bytes,
                             const void* x, void* y,
                             ctranslate2::dim_t size) {
  if (size == 0) return;
  (void)ct2_u32(size);
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

// ---------------------------------------------------------------------------
// Activation kernel infrastructure (unary: y[i] = f(x[i]))
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_activation_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kActivationMSL, "activation");
}

static id<MTLComputePipelineState> get_activation_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_activation_library, name);
}

static void dispatch_unary(const char* kernel_name,
                            const void* x, void* y,
                            ctranslate2::dim_t size) {
  if (size == 0) return;
  (void)ct2_u32(size);
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
// Broadcast kernel infrastructure
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_broadcast_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kBroadcastMSL, "broadcast");
}

static id<MTLComputePipelineState> get_broadcast_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_broadcast_library, name);
}

// 3 data buffers + 1 uint32 constant (param0).
// Used by add_batch_broadcast (param0 = a_size) and
//          add_depth_broadcast (param0 = depth = b_size/a_size).
static void dispatch_broadcast1(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 ctranslate2::dim_t size,
                                 uint32_t param0) {
  if (size == 0) return;
  (void)ct2_u32(size);
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

// 3 data buffers + 2 uint32 constants.
// Used by add_block_broadcast (param0 = block, param1 = a_size).
static void dispatch_broadcast2(const char* kernel_name,
                                 const void* a, const void* b, void* c,
                                 ctranslate2::dim_t size,
                                 uint32_t param0, uint32_t param1) {
  if (size == 0) return;
  (void)ct2_u32(size);
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

}  // anonymous namespace

namespace ctranslate2 {

  // -------------------------------------------------------------------------
  // M4.2 — Arithmetic: add, sub, min, max, mul
  // -------------------------------------------------------------------------

  template<>
  template <typename T>
  void primitives<Device::METAL>::add(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

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
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "min_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "min_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "max_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "max_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(T a, const T* x, T* y, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_scalar_%s", MetalTypeName<T>::value);
    dispatch_scalar(kname, &a, sizeof(T), x, y, size);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(const T* a, const T* b, T* c, dim_t size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_%s", MetalTypeName<T>::value);
    dispatch_binary(kname, a, b, c, size);
  }

  // -------------------------------------------------------------------------
  // M4.6 — Broadcast arithmetic
  //
  // add_batch_broadcast:  c[gid] = a[gid % a_size] + b[gid]
  // add_depth_broadcast:  c[gid] = a[gid / depth]  + b[gid]
  // add_block_broadcast:  c[gid] = a[(gid/block) % a_size] + b[gid]
  // mul_batch_broadcast:  c[gid] = a[gid % a_size] * b[gid]
  // -------------------------------------------------------------------------

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_batch_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast1(kname, a, b, c, b_size, ct2_u32(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_depth_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_depth_broadcast_%s", MetalTypeName<T>::value);
    uint32_t depth = ct2_u32(b_size / a_size);
    dispatch_broadcast1(kname, a, b, c, b_size, depth);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add_block_broadcast(
      const T* a, const T* b, T* c, dim_t block, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "add_block_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast2(kname, a, b, c, b_size,
                        ct2_u32(block),
                        ct2_u32(a_size));
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul_batch_broadcast(
      const T* a, const T* b, T* c, dim_t a_size, dim_t b_size) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "mul_batch_broadcast_%s", MetalTypeName<T>::value);
    dispatch_broadcast1(kname, a, b, c, b_size, ct2_u32(a_size));
  }

  // -------------------------------------------------------------------------
  // M4.5 — Unary activation / transcendental kernels
  // -------------------------------------------------------------------------

#define METAL_UNARY_OP(cpp_name, kernel_prefix)                           \
  template<>                                                              \
  template <typename T>                                                   \
  void primitives<Device::METAL>::cpp_name(const T* x, T* y, dim_t size) { \
    char kname[kKernelNameBufSize];                                       \
    std::snprintf(kname, sizeof(kname), kernel_prefix "_%s",             \
                  MetalTypeName<T>::value);                               \
    dispatch_unary(kname, x, y, size);                                    \
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

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                          \
  template void                                                                  \
  primitives<Device::METAL>::add(T a, const T* x, T* y, dim_t size);            \
  template void                                                                  \
  primitives<Device::METAL>::add(const T* a, const T* b, T* c, dim_t size);     \
  template void                                                                  \
  primitives<Device::METAL>::add_batch_broadcast(const T* a, const T* b,         \
                                                  T* c, dim_t a_size,            \
                                                  dim_t b_size);                 \
  template void                                                                  \
  primitives<Device::METAL>::add_depth_broadcast(const T* a, const T* b,         \
                                                  T* c, dim_t a_size,            \
                                                  dim_t b_size);                 \
  template void                                                                  \
  primitives<Device::METAL>::add_block_broadcast(const T* a, const T* b,         \
                                                  T* c, dim_t block,             \
                                                  dim_t a_size, dim_t b_size);   \
  template void                                                                  \
  primitives<Device::METAL>::sub(const T* a, const T* b, T* c, dim_t size);     \
  template void                                                                  \
  primitives<Device::METAL>::min(T a, const T* x, T* y, dim_t size);            \
  template void                                                                  \
  primitives<Device::METAL>::min(const T* a, const T* b, T* c, dim_t size);     \
  template void                                                                  \
  primitives<Device::METAL>::max(T a, const T* x, T* y, dim_t size);            \
  template void                                                                  \
  primitives<Device::METAL>::max(const T* a, const T* b, T* c, dim_t size);     \
  template void                                                                  \
  primitives<Device::METAL>::mul(T a, const T* x, T* y, dim_t size);            \
  template void                                                                  \
  primitives<Device::METAL>::mul(const T* a, const T* b, T* c, dim_t size);     \
  template void                                                                  \
  primitives<Device::METAL>::mul_batch_broadcast(const T* a, const T* b,         \
                                                  T* c, dim_t a_size,            \
                                                  dim_t b_size);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

#define DECLARE_FLOAT_IMPL(T)                                                    \
  template void primitives<Device::METAL>::exp(const T*, T*, dim_t);            \
  template void primitives<Device::METAL>::log(const T*, T*, dim_t);            \
  template void primitives<Device::METAL>::cos(const T*, T*, dim_t);            \
  template void primitives<Device::METAL>::sin(const T*, T*, dim_t);            \
  template void primitives<Device::METAL>::tanh(const T*, T*, dim_t);           \
  template void primitives<Device::METAL>::relu(const T*, T*, dim_t);           \
  template void primitives<Device::METAL>::sigmoid(const T*, T*, dim_t);        \
  template void primitives<Device::METAL>::swish(const T*, T*, dim_t);          \
  template void primitives<Device::METAL>::gelu(const T*, T*, dim_t);           \
  template void primitives<Device::METAL>::gelu_tanh(const T*, T*, dim_t);      \
  template void primitives<Device::METAL>::gelu_sigmoid(const T*, T*, dim_t);

  DECLARE_FLOAT_IMPL(float)
  DECLARE_FLOAT_IMPL(float16_t)
  DECLARE_FLOAT_IMPL(bfloat16_t)

#undef DECLARE_FLOAT_IMPL

}  // namespace ctranslate2
