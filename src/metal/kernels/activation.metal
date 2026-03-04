// CTranslate2 Metal activation and transcendental kernels — M4.5.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string (kActivationMSL) in src/metal/primitives.mm and compiled at
// runtime via [MTLDevice newLibraryWithSource:options:error:].
//
// Kernel interface
// ----------------
//   y[gid] = f(x[gid])
//   buffer(0) = input  x   (const T*)
//   buffer(1) = output y   (T*)
//
// Naming convention
// -----------------
//   <op>_<type>    e.g. exp_float, relu_half, gelu_bfloat
//
// All intermediate arithmetic is done in float32 so that erf, exp, tanh,
// etc. work uniformly for half and bfloat inputs (which lack some math
// overloads in older MSL versions).  The result is cast back to T before
// storing.
//
// Types supported
// ---------------
//   float   (float32)
//   half    (float16 / float16_t)
//   bfloat  (bfloat16 / bfloat16_t) — only if __HAVE_BFLOAT__ (Apple9+, macOS 14+)

#include <metal_stdlib>
using namespace metal;

#include "metal_math.metalh"

// y[gid] = (T)(expr)  where expr is a float32 computation in variable v = (float)x[gid].
#define DEFINE_UNARY(name, T, expr)                     \
kernel void name##_##T(                                 \
    device const T* x [[buffer(0)]],                    \
    device       T* y [[buffer(1)]],                    \
    uint gid [[thread_position_in_grid]])                \
{ float v = (float)x[gid]; y[gid] = (T)(expr); }

// All eleven ops for a single MSL type T.
#define DEFINE_ACTIVATION_OPS(T)                                                         \
  DEFINE_UNARY(exp,          T, exp(v))                                                  \
  DEFINE_UNARY(log,          T, log(v))                                                  \
  DEFINE_UNARY(cos,          T, cos(v))                                                  \
  DEFINE_UNARY(sin,          T, sin(v))                                                  \
  DEFINE_UNARY(tanh,         T, ct2_safe_tanh(v))                                        \
  DEFINE_UNARY(relu,         T, fmax(v, 0.f))                                            \
  DEFINE_UNARY(sigmoid,      T, 1.f / (1.f + exp(-v)))                                  \
  DEFINE_UNARY(swish,        T, v / (1.f + exp(-v)))                                    \
  DEFINE_UNARY(gelu,         T, 0.5f * v * (1.f + ct2_erf(v * 0.7071067811865475f)))   \
  DEFINE_UNARY(gelu_tanh,    T, 0.5f * v * (1.f + ct2_safe_tanh(0.7978845608028654f *   \
                                (v + 0.044715f * v * v * v))))                           \
  DEFINE_UNARY(gelu_sigmoid, T, v / (1.f + exp(-1.702f * v)))

DEFINE_ACTIVATION_OPS(float)
DEFINE_ACTIVATION_OPS(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_ACTIVATION_OPS(bfloat)
#endif
