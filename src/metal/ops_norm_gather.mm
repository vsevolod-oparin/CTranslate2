// src/metal/ops_norm_gather.mm
//
// M5.2 — Normalization, softmax, and gather dispatch infrastructure.
//
// Implements the ctranslate2::metal:: op-dispatch wrappers declared in
// src/metal/ops_metal.h.  These are called from the Metal op specialization
// files (normalization_metal.mm, gather_metal.mm).
//
// Normalization kernels (layer_norm, rms_norm, softmax):
//   "One threadgroup per row" design.
//   Grid = [outer_size/batch_size, 1, 1]; threads = [kNormBlock, 1, 1].
//   Threadgroup memory: float[kNormBlock] for two-pass reduction (always
//   float32, regardless of element type, for numerical stability).
//
//   Null gamma / beta / lengths: Metal cannot bind nil buffers.  Solution:
//   bind the input buffer (x) as a dummy and use has_gamma / has_beta /
//   has_lengths flags so the kernel skips scale / bias / masking.
//
// Gather kernel:
//   One thread per output element.
//   Grid dispatched via dispatchThreads: (total_elements threads).

#include "metal/primitives_infra.h"

namespace {

static constexpr uint32_t kNormBlock = 256;

// ---------------------------------------------------------------------------
// Normalization library (layer_norm, rms_norm, softmax)
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_normalization_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kNormalizationMSL, "normalization");
}

static id<MTLComputePipelineState> get_normalization_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_normalization_library, name);
}

// Buffer layout (normalization.metal):
//   0: x, 1: gamma, 2: beta, 3: y
//   4: has_gamma (uint), 5: has_beta (uint), 6: N (uint), 7: eps (float)
//   threadgroup(0): float[NORM_BLOCK]
static void dispatch_layer_norm(const char* kname,
                                 const void* x, const void* gamma,
                                 const void* beta, void* y,
                                 ctranslate2::dim_t outer_size,
                                 ctranslate2::dim_t axis_size,
                                 float eps) {
  if (outer_size == 0 || axis_size == 0) return;
  uint32_t has_gamma = (gamma != nullptr) ? 1u : 0u;
  uint32_t has_beta  = (beta  != nullptr) ? 1u : 0u;
  uint32_t N         = ct2_u32(axis_size);
  id<MTLComputePipelineState> pso = get_normalization_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_x = 0, off_g = 0, off_b = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x)                 offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(gamma ? gamma : x, &off_g) offset:off_g atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(beta  ? beta  : x, &off_b) offset:off_b atIndex:2];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y)                 offset:off_y atIndex:3];
  [enc setBytes:&has_gamma length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&has_beta  length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&N         length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&eps       length:sizeof(float)    atIndex:7];
  [enc setThreadgroupMemoryLength:kNormBlock * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)outer_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kNormBlock, 1, 1)];
  [enc endEncoding];
}

// Buffer layout:
//   0: x, 1: gamma, 2: y, 3: N (uint), 4: eps (float)
//   threadgroup(0): float[NORM_BLOCK]
static void dispatch_rms_norm(const char* kname,
                               const void* x, const void* gamma, void* y,
                               ctranslate2::dim_t batch_size,
                               ctranslate2::dim_t depth,
                               float eps) {
  if (batch_size == 0 || depth == 0) return;
  uint32_t N = ct2_u32(depth);
  id<MTLComputePipelineState> pso = get_normalization_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_x = 0, off_g = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x,     &off_x) offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(gamma, &off_g) offset:off_g atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y,     &off_y) offset:off_y atIndex:2];
  [enc setBytes:&N   length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&eps length:sizeof(float)    atIndex:4];
  [enc setThreadgroupMemoryLength:kNormBlock * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kNormBlock, 1, 1)];
  [enc endEncoding];
}

// Buffer layout:
//   0: x, 1: y, 2: lengths, 3: has_lengths (uint), 4: N (uint), 5: log_mode (uint)
//   threadgroup(0): float[NORM_BLOCK]
static void dispatch_softmax(const char* kname,
                              const void* x, const void* lengths, void* y,
                              ctranslate2::dim_t batch_size,
                              ctranslate2::dim_t depth,
                              bool log_mode) {
  if (batch_size == 0 || depth == 0) return;
  uint32_t has_lengths = (lengths != nullptr) ? 1u : 0u;
  uint32_t N           = ct2_u32(depth);
  uint32_t log_m       = log_mode ? 1u : 0u;
  id<MTLComputePipelineState> pso = get_normalization_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_x = 0, off_l = 0, off_y = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(x, &off_x)                     offset:off_x atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(y, &off_y)                     offset:off_y atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(lengths ? lengths : x, &off_l) offset:off_l atIndex:2];
  [enc setBytes:&has_lengths length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&N           length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&log_m       length:sizeof(uint32_t) atIndex:5];
  [enc setThreadgroupMemoryLength:kNormBlock * sizeof(float) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kNormBlock, 1, 1)];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// Gather library
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_gather_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kGatherMSL, "gather");
}

static id<MTLComputePipelineState> get_gather_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_gather_library, name);
}

// Buffer layout:
//   0: src, 1: dst, 2: indices
//   3: copy_size (uint), 4: batch_stride (uint), 5: num_indices_per_batch (uint)
//   grid: total_elements threads (one per output element)
//
// M10.1: commit_and_wait() makes gather synchronous.  Two hazards require this:
//
// (1) In-place Gather::operator()(data, input):
//     StorageView clone(std::move(data));  operator()(clone, input, data);
//     If gather were encode-only, the clone's MTLBuffer would be freed and
//     recycled before the GPU reads it.
//
// (2) Out-of-place gather feeding CPU-side operations:
//     Several call sites (KV-cache update, attention alignment gather) read the
//     gather output from CPU immediately after the call.  Without a sync, the
//     CPU reads stale (pre-gather) data from the unified-memory buffer.
//
// Making gather unconditionally synchronous is the simplest correct solution.
// M11 (command buffer batching) will amortize this by batching all layer ops.
static void dispatch_gather(const char* kname,
                             const void* src, void* dst, const void* indices,
                             ctranslate2::dim_t copy_size,
                             ctranslate2::dim_t batch_stride,
                             ctranslate2::dim_t num_indices_per_batch,
                             ctranslate2::dim_t total_elements,
                             bool sync = true) {
  if (total_elements == 0) return;
  (void)ct2_u32(total_elements);
  uint32_t copy_sz  = ct2_u32(copy_size);
  uint32_t b_stride = ct2_u32(batch_stride);
  uint32_t nipb     = ct2_u32(num_indices_per_batch);
  id<MTLComputePipelineState> pso = get_gather_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_src = 0, off_dst = 0, off_idx = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(src,     &off_src) offset:off_src atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(dst,     &off_dst) offset:off_dst atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(indices, &off_idx) offset:off_idx atIndex:2];
  [enc setBytes:&copy_sz  length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&b_stride length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&nipb     length:sizeof(uint32_t) atIndex:5];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(total_elements));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total_elements), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  // Flush immediately — see comment above for the two hazards this prevents.
  // sync=false is used by batch_gather_in_place (M11.1) which manages its own
  // synchronization after encoding all gathers.
  if (sync)
    CT2_COMMIT_AND_WAIT();
}

}  // anonymous namespace

namespace ctranslate2 {
  namespace metal {

    // -----------------------------------------------------------------------
    // Op-dispatch wrappers (declared in src/metal/ops_metal.h)
    // -----------------------------------------------------------------------

    template <typename T>
    void layer_norm_metal(const T* x, const T* gamma, const T* beta,
                          T* y, dim_t outer_size, dim_t axis_size, float epsilon) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "layer_norm_%s", MetalTypeName<T>::value);
      dispatch_layer_norm(kname, x, gamma, beta, y, outer_size, axis_size, epsilon);
    }

    template <typename T>
    void rms_norm_metal(const T* x, const T* gamma, T* y,
                        dim_t batch_size, dim_t depth, float epsilon) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "rms_norm_%s", MetalTypeName<T>::value);
      dispatch_rms_norm(kname, x, gamma, y, batch_size, depth, epsilon);
    }

    template <typename T>
    void softmax_metal(const T* x, const int32_t* lengths, T* y,
                       dim_t batch_size, dim_t depth, bool log_mode) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "softmax_%s", MetalTypeName<T>::value);
      dispatch_softmax(kname, x, lengths, y, batch_size, depth, log_mode);
    }

    template <typename T>
    void gather_metal(const T* src, T* dst, const int32_t* indices,
                      dim_t copy_size, dim_t batch_stride,
                      dim_t num_indices_per_batch, dim_t total_elements) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "gather_%s", MetalTypeName<T>::value);
      dispatch_gather(kname, src, dst, indices,
                      copy_size, batch_stride, num_indices_per_batch, total_elements);
    }

    template <typename T>
    void gather_metal_encode_only(const T* src, T* dst, const int32_t* indices,
                                   dim_t copy_size, dim_t batch_stride,
                                   dim_t num_indices_per_batch, dim_t total_elements) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "gather_%s", MetalTypeName<T>::value);
      dispatch_gather(kname, src, dst, indices,
                      copy_size, batch_stride, num_indices_per_batch, total_elements,
                      /*sync=*/false);
    }

    // -----------------------------------------------------------------------
    // Explicit instantiations
    // -----------------------------------------------------------------------

    // Normalization: float, float16_t, bfloat16_t
    template void layer_norm_metal<float>(const float*, const float*, const float*, float*, dim_t, dim_t, float);
    template void layer_norm_metal<float16_t>(const float16_t*, const float16_t*, const float16_t*, float16_t*, dim_t, dim_t, float);
    template void layer_norm_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t, float);

    template void rms_norm_metal<float>(const float*, const float*, float*, dim_t, dim_t, float);
    template void rms_norm_metal<float16_t>(const float16_t*, const float16_t*, float16_t*, dim_t, dim_t, float);
    template void rms_norm_metal<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, dim_t, dim_t, float);

    template void softmax_metal<float>(const float*, const int32_t*, float*, dim_t, dim_t, bool);
    template void softmax_metal<float16_t>(const float16_t*, const int32_t*, float16_t*, dim_t, dim_t, bool);
    template void softmax_metal<bfloat16_t>(const bfloat16_t*, const int32_t*, bfloat16_t*, dim_t, dim_t, bool);

    // Gather: all 6 element types
    template void gather_metal<float>(const float*, float*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal<float16_t>(const float16_t*, float16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal<bfloat16_t>(const bfloat16_t*, bfloat16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal<int32_t>(const int32_t*, int32_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal<int16_t>(const int16_t*, int16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal<int8_t>(const int8_t*, int8_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);

    // Gather encode-only (M11.1 batch gather): float types only (KV-cache is always float/f16/bf16)
    template void gather_metal_encode_only<float>(const float*, float*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal_encode_only<float16_t>(const float16_t*, float16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal_encode_only<bfloat16_t>(const bfloat16_t*, bfloat16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal_encode_only<int32_t>(const int32_t*, int32_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal_encode_only<int16_t>(const int16_t*, int16_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);
    template void gather_metal_encode_only<int8_t>(const int8_t*, int8_t*, const int32_t*, dim_t, dim_t, dim_t, dim_t);

  }  // namespace metal
}  // namespace ctranslate2
