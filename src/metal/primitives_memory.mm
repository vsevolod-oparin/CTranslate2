// src/metal/primitives_memory.mm
//
// M4.1 — Memory primitives for Device::MPS.
//
// All operations are CPU-side: Metal buffers use MTLResourceStorageModeShared
// (unified memory), so the contents pointer is simultaneously valid for CPU
// and GPU access.  No GPU kernel is required for simple copies or fills.
//
// convert() calls commit_and_wait() first to flush any pending GPU writes to
// the source buffer before the CPU reads it (same pattern as at()).

#include "metal/primitives_infra.h"

namespace ctranslate2 {

  // -------------------------------------------------------------------------
  // cross_device_primitives (CPU ↔ Metal)
  // -------------------------------------------------------------------------

  // CPU → Metal: destination is a Shared MTLBuffer contents pointer;
  // CPU writes are immediately visible to the GPU at the next command encoding.
  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::MPS>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // Metal → CPU: source is a Shared MTLBuffer contents pointer.
  // The caller must ensure the GPU has committed and completed any prior writes
  // (guaranteed by synchronize_stream(METAL) inside copy_from).
  template<>
  template <typename T>
  void cross_device_primitives<Device::MPS, Device::CPU>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // -------------------------------------------------------------------------
  // primitives<Device::MPS> — memory primitives
  // -------------------------------------------------------------------------

  // at: flush pending GPU writes then read one element from CPU.
  template<>
  template <typename T>
  T primitives<Device::MPS>::at(const T* x, dim_t index) {
    CT2_COMMIT_AND_WAIT();
    return x[index];
  }

  // copy: both pointers are Shared MTLBuffer contents pointers; memcpy is correct.
  template<>
  template <typename T>
  void primitives<Device::MPS>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // fill: CPU write to Shared buffer — immediately coherent with GPU.
  template<>
  template <typename T>
  void primitives<Device::MPS>::fill(T* x, T a, dim_t size) {
    std::fill(x, x + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0; i < size; ++i, x += inc_x)
      *x = a;
  }

  // M11.25: indexed_fill GPU kernel infrastructure.
  namespace {
    static id<MTLLibrary> get_indexed_fill_library() {
      static id<MTLLibrary> lib = nil;
      static std::once_flag flag;
      return compile_library_once(flag, lib, kIndexedFillMSL, "indexed_fill");
    }
    static id<MTLComputePipelineState> get_indexed_fill_pso(const char* name) {
      static PSOCache cache;
      return cache.get(get_indexed_fill_library, name);
    }
  }

  template<>
  template <typename T>
  void primitives<Device::MPS>::indexed_fill(T* x, T a,
                                                const int32_t* indices,
                                                dim_t num_indices) {
    if (num_indices <= 0)
      return;

    // M11.29: GPU scatter kernel.  protect_buffer defers index buffer
    // recycling until the next commit_and_wait().
    //
    // Pre-sync strategy depends on dtype:
    //
    //   float32: Full CT2_COMMIT_AND_WAIT() required.  Verified: whisper-base
    //     f32 produces garbage with encode_barrier-only or non-blocking
    //     commit + encode_barrier.  Root cause unclear (possible MPS driver
    //     coherency issue with f32 MPSMatrixMultiplication + subsequent
    //     compute encoder in the same queue).  Cost: ~1.5ms per call.
    //
    //   float16 / bfloat16: GPU-side encode_barrier() is sufficient.
    //     Unlike the original M11.28 approach (which simply skipped sync for
    //     f16), encode_barrier() properly handles CB splits if they occur —
    //     making this robust against future code changes that might introduce
    //     commit_command_buffer() calls in the f16 GEMM path.
    //     BF16 GEMMs use MPSGraph which commits internally, so indexed_fill
    //     always gets a fresh CB; the barrier is a safety net.
    //     Cost: ~0 (GPU-side only, no CPU block).
    metal::protect_buffer(indices);
    if constexpr (std::is_same_v<T, float>) {
      CT2_COMMIT_AND_WAIT();
    } else {
      metal::encode_barrier();
    }

    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "indexed_fill_%s",
                  MetalTypeName<T>::value);
    id<MTLComputePipelineState> pso = get_indexed_fill_pso(kname);

    NSUInteger x_off = 0, idx_off = 0;
    id<MTLBuffer> x_buf   = metal_buffer_for_ptr(x, &x_off);
    id<MTLBuffer> idx_buf = metal_buffer_for_ptr(indices, &idx_off);

    id<MTLComputeCommandEncoder> enc = metal::create_compute_encoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:x_buf   offset:x_off   atIndex:0];
    [enc setBytes:&a       length:sizeof(T) atIndex:1];
    [enc setBuffer:idx_buf offset:idx_off atIndex:2];
    [enc dispatchThreads:MTLSizeMake(ct2_u32(num_indices), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(
           std::min<NSUInteger>(ct2_u32(num_indices),
                                pso.maxTotalThreadsPerThreadgroup), 1, 1)];
    [enc endEncoding];
    [enc release];
  }

  // convert: flush pending GPU writes before the CPU reads x, then copy with
  // implicit type conversion via the half_float::half / bfloat16_t operators.
  template<>
  template <typename U, typename V>
  void primitives<Device::MPS>::convert(const U* x, V* y, dim_t size) {
    CT2_COMMIT_AND_WAIT();
    std::copy(x, x + size, y);
  }

  // compute_u8_compensation: no-op on Metal.
  // Metal does not use the u8s8s32 GEMM path that requires this pre-computation
  // (INT8 weights are dequantized to FP16 before GEMM — see M9.2).
  template<>
  void primitives<Device::MPS>::compute_u8_compensation(
      const int8_t*, bool, dim_t, dim_t, float, int32_t*) {
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                          \
  template T                                                                     \
  primitives<Device::MPS>::at(const T* x, dim_t index);                       \
  template void                                                                  \
  primitives<Device::MPS>::fill(T* x, T a, dim_t size);                       \
  template void                                                                  \
  primitives<Device::MPS>::strided_fill(T* x, T a, dim_t inc_x, dim_t size);  \
  template void                                                                  \
  primitives<Device::MPS>::indexed_fill(T*, T, const int32_t*, dim_t);         \
  template void                                                                  \
  primitives<Device::MPS>::copy<T>(const T* x, T* y, dim_t size);             \
  template void                                                                  \
  cross_device_primitives<Device::CPU, Device::MPS>::copy<T>(                  \
      const T*, T*, dim_t);                                                      \
  template void                                                                  \
  cross_device_primitives<Device::MPS, Device::CPU>::copy<T>(                  \
      const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

  // convert specializations (cross-type pairs, not covered by DECLARE_ALL_TYPES).
  template void primitives<Device::MPS>::convert(const float*,      float16_t*,  dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*,  float*,      dim_t);
  template void primitives<Device::MPS>::convert(const float*,      bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float*,      dim_t);
  template void primitives<Device::MPS>::convert(const float16_t*,  bfloat16_t*, dim_t);
  template void primitives<Device::MPS>::convert(const bfloat16_t*, float16_t*,  dim_t);

}  // namespace ctranslate2
