// src/metal/primitives_reduction.mm
//
// M4.3 — Parallel reduction primitives: sum, max, amax, max_element.
// M4.5 — logsumexp (CPU-side scalar after GPU reduction flush).
//
// Algorithm: two-pass.
//   Pass 1 (GPU): each threadgroup of kReductionTGS threads reduces its tile
//                 of the input to one partial result in a Shared MTLBuffer.
//   Pass 2 (CPU): after commit_and_wait(), the host reduces the partial results.

#include "metal/primitives_infra.h"

namespace {

static constexpr uint32_t kReductionTGS = 256;

static id<MTLLibrary> get_reduction_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kReductionMSL, "reduction");
}

static id<MTLComputePipelineState> get_reduction_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_reduction_library, name);
}

}  // anonymous namespace

namespace ctranslate2 {

  // sum: Σ x[i]  (result cast back to T)
  template<>
  template <typename T>
  T primitives<Device::METAL>::sum(const T* array, dim_t size) {
    if (size == 0) return T(0);
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_sum_%s", MetalTypeName<T>::value);
    uint32_t n = ct2_u32(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> out_buf = alloc_temp_buffer(num_groups * sizeof(T));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf offset:inp_off       atIndex:0];
    [enc setBuffer:out_buf offset:0             atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t)    atIndex:2];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(T) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const T* partials = static_cast<const T*>([out_buf contents]);
    return std::accumulate(partials, partials + num_groups, T(0));
  }

  // max_element: index of the maximum element (argmax).
  // GPU reduces each tile to (best_val, best_idx) in float/uint32_t pair buffers.
  template<>
  template <typename T>
  dim_t primitives<Device::METAL>::max_element(const T* array, dim_t size) {
    if (size == 0) return 0;
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_max_element_%s", MetalTypeName<T>::value);
    uint32_t n = ct2_u32(size);
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
    [enc setBuffer:inp_buf  offset:inp_off       atIndex:0];
    [enc setBuffer:vals_buf offset:0             atIndex:1];
    [enc setBuffer:idxs_buf offset:0             atIndex:2];
    [enc setBytes:&n length:sizeof(uint32_t)     atIndex:3];
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

  // max: maximum value in the array.
  template<>
  template <typename T>
  T primitives<Device::METAL>::max(const T* array, dim_t size) {
    if (size == 0) return T(0);
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_max_%s", MetalTypeName<T>::value);
    uint32_t n = ct2_u32(size);
    uint32_t num_groups = (n + kReductionTGS - 1) / kReductionTGS;
    NSUInteger inp_off = 0;
    id<MTLBuffer> inp_buf = metal_buffer_for_ptr(array, &inp_off);
    id<MTLBuffer> out_buf = alloc_temp_buffer(num_groups * sizeof(T));
    id<MTLComputePipelineState> pso = get_reduction_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:inp_buf offset:inp_off       atIndex:0];
    [enc setBuffer:out_buf offset:0             atIndex:1];
    [enc setBytes:&n length:sizeof(uint32_t)    atIndex:2];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(T) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    metal::commit_and_wait();
    const T* partials = static_cast<const T*>([out_buf contents]);
    return *std::max_element(partials, partials + num_groups);
  }

  // amax: max of absolute values.
  // GPU kernel accumulates in float (all types); CPU converts back to T.
  template<>
  template <typename T>
  T primitives<Device::METAL>::amax(const T* array, dim_t size) {
    if (size == 0) return T(0);
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "reduce_amax_%s", MetalTypeName<T>::value);
    uint32_t n = ct2_u32(size);
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

  // logsumexp: log(Σ exp(x[i])) — CPU-side after flushing pending GPU work.
  // Computed stably as log(Σ exp(x[i] - max)) + max.
  template<>
  template <typename T>
  float primitives<Device::METAL>::logsumexp(const T* x, dim_t size) {
    if (size == 0) return 0.f;
    metal::commit_and_wait();
    float maxval = (float)x[0];
    for (dim_t i = 1; i < size; ++i)
      maxval = std::max(maxval, (float)x[i]);
    float sum = 0.f;
    for (dim_t i = 0; i < size; ++i)
      sum += std::exp((float)x[i] - maxval);
    return std::log(sum) + maxval;
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                    \
  template T                                                               \
  primitives<Device::METAL>::sum(const T* array, dim_t size);             \
  template dim_t                                                           \
  primitives<Device::METAL>::max_element(const T* array, dim_t size);     \
  template T                                                               \
  primitives<Device::METAL>::max(const T* array, dim_t size);             \
  template T                                                               \
  primitives<Device::METAL>::amax(const T* array, dim_t size);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

  template float primitives<Device::METAL>::logsumexp(const float*,      dim_t);
  template float primitives<Device::METAL>::logsumexp(const float16_t*,  dim_t);
  template float primitives<Device::METAL>::logsumexp(const bfloat16_t*, dim_t);

}  // namespace ctranslate2
