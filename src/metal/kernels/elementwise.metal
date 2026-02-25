// CTranslate2 Metal element-wise kernels — M4.2.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string in src/metal/primitives.mm and compiled at runtime via
// [MTLDevice newLibraryWithSource:options:error:].
//
// Naming convention
// -----------------
//   add_float, sub_float, mul_float     — vector op vector  (3 buffer args)
//   add_scalar_float, mul_scalar_float  — scalar op vector  (setBytes + 2 bufs)
//
// Types supported
// ---------------
//   float  (float32)
//   half   (float16 / float16_t)
//   int    (int32  / int32_t)
//   short  (int16  / int16_t)
//   char   (int8   / int8_t)
//   bfloat (bfloat16 / bfloat16_t) — only if __HAVE_BFLOAT__ (Apple9+, macOS 14+)

#include <metal_stdlib>
using namespace metal;

// --- Binary vector op vector ------------------------------------------------
//   c[i] = a[i] op b[i]
#define DEFINE_BINARY(name, op, T)                                      \
  kernel void name##_##T(                                               \
      device const T* a [[buffer(0)]],                                  \
      device const T* b [[buffer(1)]],                                  \
      device       T* c [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { c[gid] = a[gid] op b[gid]; }

// --- Scalar op vector -------------------------------------------------------
//   y[i] = a op x[i]   (a is bound via setBytes, not a buffer)
#define DEFINE_SCALAR(name, op, T)                                      \
  kernel void name##_scalar_##T(                                        \
      device const T* x [[buffer(0)]],                                  \
      constant     T& a [[buffer(1)]],                                  \
      device       T* y [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { y[gid] = a op x[gid]; }

#define DEFINE_ALL(T)          \
  DEFINE_BINARY(add, +, T)    \
  DEFINE_BINARY(sub, -, T)    \
  DEFINE_BINARY(mul, *, T)    \
  DEFINE_SCALAR(add, +, T)    \
  DEFINE_SCALAR(mul, *, T)

DEFINE_ALL(float)
DEFINE_ALL(half)
DEFINE_ALL(int)
DEFINE_ALL(short)
DEFINE_ALL(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_ALL(bfloat)
#endif

// --- Min / Max using ternary comparison ------------------------------------
//
//   min_<T>(a, b, c):        c[i] = (a[i] < b[i]) ? a[i] : b[i]
//   max_<T>(a, b, c):        c[i] = (a[i] > b[i]) ? a[i] : b[i]
//   min_scalar_<T>(x, a, y): y[i] = (x[i] < a)    ? x[i] : a
//   max_scalar_<T>(x, a, y): y[i] = (x[i] > a)    ? x[i] : a
//
// Ternary comparison is used instead of metal::min()/max() to avoid MSL
// overload-resolution ambiguity for bfloat (no dedicated bfloat overload
// on all SDK versions).  Comparison operators (<, >) are always available.

#define DEFINE_MINMAX_BINARY(name, sel, T)                              \
  kernel void name##_##T(                                               \
      device const T* a [[buffer(0)]],                                  \
      device const T* b [[buffer(1)]],                                  \
      device       T* c [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { T va = a[gid], vb = b[gid]; c[gid] = (va sel vb) ? va : vb; }

#define DEFINE_MINMAX_SCALAR(name, sel, T)                              \
  kernel void name##_scalar_##T(                                        \
      device const T* x [[buffer(0)]],                                  \
      constant     T& a [[buffer(1)]],                                  \
      device       T* y [[buffer(2)]],                                  \
      uint gid [[thread_position_in_grid]])                              \
  { T vx = x[gid]; y[gid] = (vx sel a) ? vx : a; }

#define DEFINE_MINMAX(T)              \
  DEFINE_MINMAX_BINARY(min, <, T)    \
  DEFINE_MINMAX_BINARY(max, >, T)    \
  DEFINE_MINMAX_SCALAR(min, <, T)    \
  DEFINE_MINMAX_SCALAR(max, >, T)

DEFINE_MINMAX(float)
DEFINE_MINMAX(half)
DEFINE_MINMAX(int)
DEFINE_MINMAX(short)
DEFINE_MINMAX(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_MINMAX(bfloat)
#endif
