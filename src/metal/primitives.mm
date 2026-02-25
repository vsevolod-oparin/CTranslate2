// primitives<Device::METAL> — M3.2 implementation.
//
// cross_device_primitives (CPU↔Metal):
//   On Apple Silicon all memory is unified.  MTLResourceStorageModeShared
//   buffers expose a void* via [buf contents] that is valid for both CPU and
//   GPU access.  "Copying" between CPU and Metal is therefore a plain memcpy.
//   The caller is responsible for synchronising the GPU (commit_and_wait) before
//   reading Metal-written data on the CPU.
//
// primitives<Device::METAL>:
//   at() and copy() are real implementations; all other methods throw
//   std::runtime_error("not yet implemented") — they will be replaced in M4.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "ctranslate2/primitives.h"
#include "metal/utils.h"
#include "type_dispatch.h"

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

  template<>
  template <typename T>
  void primitives<Device::METAL>::fill(T* x, T a, dim_t size) {
    METAL_STUB(fill);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    METAL_STUB(strided_fill);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    METAL_STUB(indexed_fill);
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::METAL>::convert(const U* x, V* y, dim_t size) {
    METAL_STUB(convert);
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

  template<>
  template <typename T>
  void primitives<Device::METAL>::add(T a, const T* x, T* y, dim_t size) {
    METAL_STUB(add);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::add(const T* a, const T* b, T* c, dim_t size) {
    METAL_STUB(add);
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

  template<>
  template <typename T>
  void primitives<Device::METAL>::sub(const T* a, const T* b, T* c, dim_t size) {
    METAL_STUB(sub);
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

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(T a, const T* x, T* y, dim_t size) {
    METAL_STUB(mul);
  }

  template<>
  template <typename T>
  void primitives<Device::METAL>::mul(const T* a, const T* b, T* c, dim_t size) {
    METAL_STUB(mul);
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
