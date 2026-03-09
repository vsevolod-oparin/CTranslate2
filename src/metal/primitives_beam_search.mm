// src/metal/primitives_beam_search.mm
//
// M4.7 — Beam-search primitives for Device::METAL.
//
// penalize_previous_tokens: GPU kernel — one thread per batch item.
//   Each thread iterates sequentially over `length` previous IDs.
//   Sequential within a thread guarantees correct semantics when the same
//   token ID appears multiple times (last write wins, matching CPU behaviour).
//
// prepare_length_mask: GPU kernel (M11.16) with a prior GPU flush.
//   `lengths` may have been written by a prior GPU op (e.g. a gather over a
//   padded-batch tensor).  commit_and_wait() ensures those writes are visible
//   before the mask kernel reads them.

#include "metal/primitives_infra.h"

namespace {

static id<MTLLibrary> get_beam_search_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kBeamSearchMSL, "beam_search");
}

static id<MTLComputePipelineState> get_beam_search_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_beam_search_library, name);
}

// One thread per batch item; iterates `length` previous IDs to apply penalty.
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
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(scores,          &off_s)
          offset:off_s  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(previous_scores, &off_ps)
          offset:off_ps atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(previous_ids,    &off_pi)
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

}  // anonymous namespace

namespace ctranslate2 {

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
                      ct2_u32(batch_size),
                      ct2_u32(length),
                      ct2_u32(vocabulary_size));
  }

  template<>
  void primitives<Device::METAL>::prepare_length_mask(
      const int32_t* lengths, dim_t batch_size, dim_t num_heads,
      dim_t num_queries, bool mask_future, bool multi_query, int32_t* mask) {
    if (batch_size == 0) return;

    // Flush pending GPU work: downstream CPU code (or ops that preceded this
    // call) may depend on data written by earlier GPU encodes.  The original
    // CPU implementation had CT2_COMMIT_AND_WAIT() here for the same reason.
    CT2_COMMIT_AND_WAIT();

    id<MTLComputePipelineState> pso = get_beam_search_pso("prepare_length_mask");
    id<MTLCommandBuffer> cmd = metal::get_current_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    [enc setComputePipelineState:pso];

    NSUInteger off_l = 0, off_m = 0;
    [enc setBuffer:metal_buffer_for_ptr(lengths, &off_l) offset:off_l atIndex:0];
    [enc setBuffer:metal_buffer_for_ptr(mask, &off_m)    offset:off_m atIndex:1];
    uint32_t nh = ct2_u32(num_heads);
    uint32_t nq = ct2_u32(num_queries);
    uint32_t mf = mask_future ? 1u : 0u;
    uint32_t mq = multi_query ? 1u : 0u;
    [enc setBytes:&nh length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&nq length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&mf length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&mq length:sizeof(uint32_t) atIndex:5];

    NSUInteger total_items = static_cast<NSUInteger>(num_heads * num_queries);
    NSUInteger tg_y = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, total_items);
    [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(batch_size), total_items, 1)
        threadsPerThreadgroup:MTLSizeMake(1, tg_y, 1)];
    [enc endEncoding];
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                          \
  template void                                                                  \
  primitives<Device::METAL>::penalize_previous_tokens(T*,                        \
                                                       const T*,                 \
                                                       const int32_t*,           \
                                                       T,                        \
                                                       dim_t,                    \
                                                       dim_t,                    \
                                                       dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

}  // namespace ctranslate2
