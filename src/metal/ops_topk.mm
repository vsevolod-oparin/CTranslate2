// src/metal/ops_topk.mm
//
// M11.6 — GPU TopK (argmax for k=1) dispatch.
//
// Implements metal::topk_metal<T>() for k=1 only.
// Uses the argmax_<T> MSL kernel from topk.metal.
// Encode-only: no commit_and_wait().  The caller (Sampler::operator())
// syncs via copy_from which calls synchronize_stream(Device::METAL).

#include "metal/primitives_infra.h"

namespace {

static constexpr uint32_t kTopKBlock = 256;

static id<MTLLibrary> get_topk_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kTopKMSL, "topk");
}

static id<MTLComputePipelineState> get_topk_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_topk_library, name);
}

static void dispatch_argmax(const char* kname,
                             const void* input, void* values, void* indices,
                             ctranslate2::dim_t batch_size,
                             ctranslate2::dim_t depth) {
  if (batch_size == 0 || depth == 0) return;
  uint32_t d = ct2_u32(depth);
  id<MTLComputePipelineState> pso = get_topk_pso(kname);
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off_in = 0, off_val = 0, off_idx = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(input,   &off_in)  offset:off_in  atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(values,  &off_val) offset:off_val atIndex:1];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(indices, &off_idx) offset:off_idx atIndex:2];
  [enc setBytes:&d length:sizeof(uint32_t) atIndex:3];
  [enc setThreadgroupMemoryLength:kTopKBlock * sizeof(float)    atIndex:0];
  [enc setThreadgroupMemoryLength:kTopKBlock * sizeof(uint32_t) atIndex:1];
  [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(batch_size), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kTopKBlock, 1, 1)];
  [enc endEncoding];
}

}  // anonymous namespace

namespace ctranslate2 {
  namespace metal {

    template <typename T>
    void topk_metal(const T* input, T* values, int32_t* indices,
                    dim_t batch_size, dim_t depth) {
      char kname[kKernelNameBufSize];
      std::snprintf(kname, sizeof(kname), "argmax_%s", MetalTypeName<T>::value);
      dispatch_argmax(kname, input, values, indices, batch_size, depth);
    }

    // Explicit instantiations
    template void topk_metal<float>(const float*, float*, int32_t*, dim_t, dim_t);
    template void topk_metal<float16_t>(const float16_t*, float16_t*, int32_t*, dim_t, dim_t);
    template void topk_metal<bfloat16_t>(const bfloat16_t*, bfloat16_t*, int32_t*, dim_t, dim_t);

  }  // namespace metal
}  // namespace ctranslate2
