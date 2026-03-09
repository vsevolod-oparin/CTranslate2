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

// ---------------------------------------------------------------------------
// MSL kernel for fused timestamp check + disable (M11.17).
// 3-pass reduction: max over text tokens, max over timestamp tokens,
// sum(exp(ts - max_ts)). If logsumexp(ts) > max(text), threads cooperatively
// fill logits[batch_id][0..num_text) with -inf.  No CPU readback — encode-only.
// ---------------------------------------------------------------------------
static constexpr const char* kFuseTimestampDisableMSL = R"msl(
#include <metal_stdlib>
using namespace metal;

#define DEFINE_FUSE_TS_DISABLE(T)                                            \
kernel void fuse_ts_disable_##T(                                             \
    device const T*     log_probs   [[buffer(0)]],                           \
    device       T*     logits      [[buffer(1)]],                           \
    device const uint*  batch_ids   [[buffer(2)]],                           \
    constant     uint&  vocab_size  [[buffer(3)]],                           \
    constant     uint&  num_text    [[buffer(4)]],                           \
    constant     uint&  num_ts      [[buffer(5)]],                           \
    threadgroup  float* shmem       [[threadgroup(0)]],                      \
    uint tg_idx [[threadgroup_position_in_grid]],                            \
    uint tid    [[thread_index_in_threadgroup]],                             \
    uint tgs    [[threads_per_threadgroup]])                                  \
{                                                                            \
    uint bid = batch_ids[tg_idx];                                            \
    device const T* row = log_probs + bid * vocab_size;                      \
                                                                             \
    /* Pass 1: max over text tokens [0, num_text) */                         \
    float val = -FLT_MAX;                                                    \
    for (uint i = tid; i < num_text; i += tgs)                               \
        val = max(val, (float)row[i]);                                       \
    shmem[tid] = val;                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                               \
        if (tid < s && shmem[tid + s] > shmem[tid])                          \
            shmem[tid] = shmem[tid + s];                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                        \
    float max_text_val = shmem[0];                                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
                                                                             \
    /* Pass 2: max over timestamp tokens for stable logsumexp */             \
    val = -FLT_MAX;                                                          \
    for (uint i = tid; i < num_ts; i += tgs)                                 \
        val = max(val, (float)row[num_text + i]);                            \
    shmem[tid] = val;                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                               \
        if (tid < s && shmem[tid + s] > shmem[tid])                          \
            shmem[tid] = shmem[tid + s];                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                        \
    float max_ts_val = shmem[0];                                             \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
                                                                             \
    /* Pass 3: sum(exp(ts[i] - max_ts)) */                                   \
    float sum_exp = 0.f;                                                     \
    for (uint i = tid; i < num_ts; i += tgs)                                 \
        sum_exp += exp((float)row[num_text + i] - max_ts_val);               \
    shmem[tid] = sum_exp;                                                    \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                               \
        if (tid < s) shmem[tid] += shmem[tid + s];                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                        \
                                                                             \
    /* Thread 0 decides; broadcast via shmem[0] */                           \
    if (tid == 0) {                                                          \
        float logsumexp_ts = log(shmem[0]) + max_ts_val;                     \
        shmem[0] = (logsumexp_ts > max_text_val) ? 1.f : 0.f;               \
    }                                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
                                                                             \
    /* If should_sample: write -inf to logits[bid][0..num_text) */           \
    if (shmem[0] > 0.5f) {                                                   \
        device T* logits_row = logits + bid * vocab_size;                    \
        T neg_inf = T(-HUGE_VALF);                                           \
        for (uint i = tid; i < num_text; i += tgs)                           \
            logits_row[i] = neg_inf;                                         \
    }                                                                        \
}

DEFINE_FUSE_TS_DISABLE(float)
DEFINE_FUSE_TS_DISABLE(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_FUSE_TS_DISABLE(bfloat)
#endif
)msl";

static id<MTLLibrary> get_fuse_ts_disable_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kFuseTimestampDisableMSL, "fuse_ts_disable");
}

static id<MTLComputePipelineState> get_fuse_ts_disable_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_fuse_ts_disable_library, name);
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
    CT2_COMMIT_AND_WAIT();
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
    CT2_COMMIT_AND_WAIT();
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
    CT2_COMMIT_AND_WAIT();
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
    CT2_COMMIT_AND_WAIT();
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
    CT2_COMMIT_AND_WAIT();
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

  // -------------------------------------------------------------------------
  // Fused timestamp check + disable — M11.17.
  // Performs timestamp probability check and writes -inf directly to logits
  // instead of returning booleans.  Encode-only: no sync.
  // -------------------------------------------------------------------------
  namespace metal {

  template <typename T>
  void fuse_timestamp_check_and_disable_metal(
      const T* log_probs,
      T* logits,
      dim_t vocab_size,
      dim_t num_text_tokens,
      dim_t num_ts_tokens,
      const std::vector<dim_t>& batch_ids) {
    const size_t num = batch_ids.size();
    if (num == 0) return;

    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "fuse_ts_disable_%s", MetalTypeName<T>::value);

    // Upload batch_ids as uint array.
    id<MTLBuffer> ids_buf = alloc_temp_buffer(num * sizeof(uint32_t));
    auto* ids_ptr = static_cast<uint32_t*>([ids_buf contents]);
    for (size_t i = 0; i < num; ++i)
      ids_ptr[i] = static_cast<uint32_t>(batch_ids[i]);

    NSUInteger lp_off = 0, lg_off = 0;
    id<MTLBuffer> lp_buf = metal_buffer_for_ptr(log_probs, &lp_off);
    id<MTLBuffer> lg_buf = metal_buffer_for_ptr(logits, &lg_off);

    uint32_t vocab = ct2_u32(vocab_size);
    uint32_t ntxt  = ct2_u32(num_text_tokens);
    uint32_t nts   = ct2_u32(num_ts_tokens);

    id<MTLComputePipelineState> pso = get_fuse_ts_disable_pso(kname);
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];
    [enc setBuffer:lp_buf  offset:lp_off                  atIndex:0];
    [enc setBuffer:lg_buf  offset:lg_off                  atIndex:1];
    [enc setBuffer:ids_buf offset:0                       atIndex:2];
    [enc setBytes:&vocab   length:sizeof(uint32_t)        atIndex:3];
    [enc setBytes:&ntxt    length:sizeof(uint32_t)        atIndex:4];
    [enc setBytes:&nts     length:sizeof(uint32_t)        atIndex:5];
    [enc setThreadgroupMemoryLength:kReductionTGS * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(num, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReductionTGS, 1, 1)];
    [enc endEncoding];
    // Encode-only — no CT2_COMMIT_AND_WAIT().
  }

  template void fuse_timestamp_check_and_disable_metal<float>(
      const float*, float*, dim_t, dim_t, dim_t,
      const std::vector<dim_t>&);
  template void fuse_timestamp_check_and_disable_metal<float16_t>(
      const float16_t*, float16_t*, dim_t, dim_t, dim_t,
      const std::vector<dim_t>&);
  template void fuse_timestamp_check_and_disable_metal<bfloat16_t>(
      const bfloat16_t*, bfloat16_t*, dim_t, dim_t, dim_t,
      const std::vector<dim_t>&);

  }  // namespace metal

}  // namespace ctranslate2
