// src/metal/primitives_memory.mm
//
// M4.1 — Memory primitives for Device::METAL.
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
  void cross_device_primitives<Device::CPU, Device::METAL>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // Metal → CPU: source is a Shared MTLBuffer contents pointer.
  // The caller must ensure the GPU has committed and completed any prior writes
  // (guaranteed by synchronize_stream(METAL) inside copy_from).
  template<>
  template <typename T>
  void cross_device_primitives<Device::METAL, Device::CPU>::copy(
      const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // -------------------------------------------------------------------------
  // primitives<Device::METAL> — memory primitives
  // -------------------------------------------------------------------------

  // at: flush pending GPU writes then read one element from CPU.
  template<>
  template <typename T>
  T primitives<Device::METAL>::at(const T* x, dim_t index) {
    CT2_COMMIT_AND_WAIT();
    return x[index];
  }

  // copy: both pointers are Shared MTLBuffer contents pointers; memcpy is correct.
  template<>
  template <typename T>
  void primitives<Device::METAL>::copy(const T* x, T* y, dim_t size) {
    std::memcpy(y, x, size * sizeof(T));
  }

  // fill: CPU write to Shared buffer — immediately coherent with GPU.
  template<>
  template <typename T>
  void primitives<Device::METAL>::fill(T* x, T a, dim_t size) {
    std::fill(x, x + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0; i < size; ++i, x += inc_x)
      *x = a;
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::indexed_fill(T* x, T a,
                                                const int32_t* indices,
                                                dim_t num_indices) {
    for (dim_t i = 0; i < num_indices; ++i)
      x[indices[i]] = a;
  }

  // convert: flush pending GPU writes before the CPU reads x, then copy with
  // implicit type conversion via the half_float::half / bfloat16_t operators.
  template<>
  template <typename U, typename V>
  void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
    CT2_COMMIT_AND_WAIT();
    std::copy(x, x + size, y);
  }

  // compute_u8_compensation: no-op on Metal.
  // Metal does not use the u8s8s32 GEMM path that requires this pre-computation
  // (INT8 weights are dequantized to FP16 before GEMM — see M9.2).
  template<>
  void primitives<Device::METAL>::compute_u8_compensation(
      const int8_t*, bool, dim_t, dim_t, float, int32_t*) {
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                          \
  template T                                                                     \
  primitives<Device::METAL>::at(const T* x, dim_t index);                       \
  template void                                                                  \
  primitives<Device::METAL>::fill(T* x, T a, dim_t size);                       \
  template void                                                                  \
  primitives<Device::METAL>::strided_fill(T* x, T a, dim_t inc_x, dim_t size);  \
  template void                                                                  \
  primitives<Device::METAL>::indexed_fill(T*, T, const int32_t*, dim_t);         \
  template void                                                                  \
  primitives<Device::METAL>::copy<T>(const T* x, T* y, dim_t size);             \
  template void                                                                  \
  cross_device_primitives<Device::CPU, Device::METAL>::copy<T>(                  \
      const T*, T*, dim_t);                                                      \
  template void                                                                  \
  cross_device_primitives<Device::METAL, Device::CPU>::copy<T>(                  \
      const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

  // convert specializations (cross-type pairs, not covered by DECLARE_ALL_TYPES).
  template void primitives<Device::METAL>::convert(const float*,      float16_t*,  dim_t);
  template void primitives<Device::METAL>::convert(const float16_t*,  float*,      dim_t);
  template void primitives<Device::METAL>::convert(const float*,      bfloat16_t*, dim_t);
  template void primitives<Device::METAL>::convert(const bfloat16_t*, float*,      dim_t);
  template void primitives<Device::METAL>::convert(const float16_t*,  bfloat16_t*, dim_t);
  template void primitives<Device::METAL>::convert(const bfloat16_t*, float16_t*,  dim_t);

}  // namespace ctranslate2
