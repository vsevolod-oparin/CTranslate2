#pragma once

#include "device_dispatch.h"
#include "type_dispatch.h"

#define DEVICE_AND_TYPE_DISPATCH(DEVICE, TYPE, STMTS)   \
  DEVICE_DISPATCH(DEVICE, TYPE_DISPATCH(TYPE, (STMTS)))


#define NON_FLOAT_CASE(NAME)                                            \
  default:                                                              \
    throw std::invalid_argument(NAME " only supports float types");     \


#if !defined(CT2_WITH_CUDA) && !defined(CT2_WITH_METAL)

// CPU-only build: only float32 is supported.
#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#else

// At least one GPU backend (CUDA, Metal, or both).
// FP16 and BF16 are permitted when the runtime device is any GPU.
// DEVICE_DISPATCH sets `constexpr Device D` correctly for each backend,
// so Metal and CUDA both resolve their own primitives<D> specialisation.
#  define DEVICE_AND_FLOAT_DISPATCH(NAME, DEVICE, TYPE, STMTS)          \
  switch (TYPE) {                                                       \
    TYPE_CASE(float, DEVICE_DISPATCH(DEVICE, (STMTS)))                  \
    TYPE_CASE(ctranslate2::float16_t, {                                  \
      if (DEVICE != Device::CUDA && DEVICE != Device::METAL)            \
        throw std::invalid_argument("FP16 " NAME " is only supported on GPU"); \
      DEVICE_DISPATCH(DEVICE, (STMTS));                                 \
    })                                                                  \
    TYPE_CASE(ctranslate2::bfloat16_t, {                                \
      if (DEVICE != Device::CUDA && DEVICE != Device::METAL)            \
        throw std::invalid_argument("BF16 " NAME " is only supported on GPU"); \
      DEVICE_DISPATCH(DEVICE, (STMTS));                                 \
    })                                                                  \
    NON_FLOAT_CASE(NAME)                                                \
  }

#endif
