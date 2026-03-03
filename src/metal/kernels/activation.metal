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

// Metal Shading Language does not provide erf() in its standard library.
// We implement it via the Abramowitz & Stegun polynomial approximation
// (formula 7.1.28, max absolute error 1.5e-7):
//
//   t = 1 / (1 + 0.3275911 * |x|)
//   erf(x) ≈ sign(x) * (1 - poly(t) * exp(-x*x))
//   poly(t) = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
//
// Always operates in float32 regardless of kernel input type.
static float ct2_erf(float x) {
  const float p  = 0.3275911f;
  const float a1 =  0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 =  1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 =  1.061405429f;
  float sign = (x >= 0.f) ? 1.f : -1.f;
  float ax = fabs(x);
  float t  = 1.f / (1.f + p * ax);
  float poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  return sign * (1.f - poly * exp(-ax * ax));
}

// Metal's tanh() computes (exp(2x)-1)/(exp(2x)+1) which produces NaN when
// |x| > ~44 because exp(2x) overflows float32 to inf, giving inf/inf = NaN.
// Clamp to [-10, 10] where tanh is already ±1 to 15+ decimal places.
static float ct2_safe_tanh(float x) {
  return tanh(clamp(x, -10.f, 10.f));
}

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
