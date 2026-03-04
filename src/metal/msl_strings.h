// AUTO-GENERATED — DO NOT EDIT.
//
// Source:    src/metal/kernels/*.metal  (canonical MSL source files)
// Generator: tools/gen_msl_strings.py
//
// To regenerate after editing a .metal file:
//     python3 tools/gen_msl_strings.py
//
// To verify that the header is in sync (run in CI):
//     python3 tools/gen_msl_strings.py --check

// Source: src/metal/kernels/elementwise.metal
static constexpr const char* kElementwiseMSL = R"msl(
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
)msl";

// Source: src/metal/kernels/activation.metal
static constexpr const char* kActivationMSL = R"msl(
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

// src/metal/kernels/metal_math.metalh
//
// Shared transcendental math functions for CTranslate2 Metal kernels.
// Included by activation.metal and quantize.metal.

#ifndef CT2_METAL_MATH_H
#define CT2_METAL_MATH_H

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

#endif  // CT2_METAL_MATH_H


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
)msl";

// Source: src/metal/kernels/broadcast.metal
static constexpr const char* kBroadcastMSL = R"msl(
// CTranslate2 Metal broadcast kernels — M4.6.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string (kBroadcastMSL) in src/metal/primitives.mm and compiled at
// runtime via [MTLDevice newLibraryWithSource:options:error:].
//
// Kernel naming convention
// ------------------------
//   add_batch_broadcast_<T>   — c[gid] = a[gid % a_size] + b[gid]
//   add_depth_broadcast_<T>   — c[gid] = a[gid / depth]  + b[gid]
//   add_block_broadcast_<T>   — c[gid] = a[(gid/block) % a_size] + b[gid]
//   mul_batch_broadcast_<T>   — c[gid] = a[gid % a_size] * b[gid]
//
// Each kernel is dispatched with size = b_size (total output elements).
// The index parameter(s) are passed via constant buffers (setBytes).
//
// Types supported
// ---------------
//   float  (float32)
//   half   (float16 / float16_t)
//   int    (int32  / int32_t)
//   short  (int16  / int16_t)
//   char   (int8   / int8_t)
//   bfloat (bfloat16 / bfloat16_t) — only if __HAVE_BFLOAT__ (Apple9+, macOS 14+)
//
// Index math
// ----------
// All three broadcast patterns derive from the CPU reference in primitives.cc:
//
//   add_batch_broadcast:  iter = b_size/a_size
//     for i in [0,iter): c[i*a_size+j] = a[j] + b[i*a_size+j]
//     → per-thread (gid = i*a_size+j): c[gid] = a[gid % a_size] + b[gid]
//
//   add_depth_broadcast:  depth = b_size/a_size
//     for i in [0,a_size): c[i*depth+k] = a[i] + b[i*depth+k]
//     → per-thread (gid = i*depth+k): c[gid] = a[gid / depth] + b[gid]
//
//   add_block_broadcast:
//     for i in [0,b_size/block): c[i*block+k] = a[i%a_size] + b[i*block+k]
//     → per-thread (gid = i*block+k): c[gid] = a[(gid/block) % a_size] + b[gid]

#include <metal_stdlib>
using namespace metal;

// --- Batch broadcast ---------------------------------------------------------
//   buffer(3) = uint a_size
#define DEFINE_BATCH_BROADCAST(name, op, T)                              \
kernel void name##_batch_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& a_size [[buffer(3)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[gid % a_size] op b[gid]; }

// --- Depth broadcast ---------------------------------------------------------
//   buffer(3) = uint depth   (depth = b_size / a_size, computed by the host)
#define DEFINE_DEPTH_BROADCAST(name, op, T)                              \
kernel void name##_depth_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& depth [[buffer(3)]],                                 \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[gid / depth] op b[gid]; }

// --- Block broadcast ---------------------------------------------------------
//   buffer(3) = uint block
//   buffer(4) = uint a_size
#define DEFINE_BLOCK_BROADCAST(name, op, T)                              \
kernel void name##_block_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& block  [[buffer(3)]],                                \
    constant  uint& a_size [[buffer(4)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = a[(gid / block) % a_size] op b[gid]; }

#define DEFINE_BROADCAST_OPS(T)       \
  DEFINE_BATCH_BROADCAST(add, +, T)   \
  DEFINE_DEPTH_BROADCAST(add, +, T)   \
  DEFINE_BLOCK_BROADCAST(add, +, T)   \
  DEFINE_BATCH_BROADCAST(mul, *, T)

DEFINE_BROADCAST_OPS(float)
DEFINE_BROADCAST_OPS(half)
DEFINE_BROADCAST_OPS(int)
DEFINE_BROADCAST_OPS(short)
DEFINE_BROADCAST_OPS(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_BROADCAST_OPS(bfloat)
#endif
)msl";

// Source: src/metal/kernels/beam_search.metal
static constexpr const char* kBeamSearchMSL = R"msl(
// CTranslate2 Metal beam-search primitives — M4.7.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string (kBeamSearchMSL) in src/metal/primitives.mm and compiled at
// runtime via [MTLDevice newLibraryWithSource:options:error:].
//
// Kernel: penalize_previous_tokens
// ---------------------------------
//   One thread per batch item; iterates sequentially over `length` previous
//   token IDs and applies a repetition penalty to the scores buffer in-place.
//
//   For each position j in [0, length):
//     read_idx  = batch_idx * length + j
//     write_idx = batch_idx * vocab_size + previous_ids[read_idx]
//     score     = previous_scores[read_idx]
//     scores[write_idx] = (score < 0) ? score * penalty : score / penalty
//
//   Sequential iteration within each thread ensures that duplicate token IDs
//   across positions produce a deterministic result (last write wins),
//   matching CPU semantics.
//
// Buffer layout
// -------------
//   buffer(0): T*         scores          — in-place output (logits to penalise)
//   buffer(1): const T*   previous_scores — prior step log-probabilities
//   buffer(2): const int* previous_ids    — token IDs generated so far
//   buffer(3): float      penalty         — repetition penalty scalar (> 1 = stronger)
//   buffer(4): uint       length          — number of previous tokens per batch item
//   buffer(5): uint       vocab_size      — vocabulary size
//
// Dispatch: one thread per batch item (grid = batch_size × 1 × 1).
//
// Types supported: float, half, bfloat (Apple9+ / macOS 14+).

#include <metal_stdlib>
using namespace metal;

#define DEFINE_PENALIZE(T)                                                      \
kernel void penalize_previous_tokens_##T(                                       \
    device       T*       scores          [[buffer(0)]],                        \
    device const T*       previous_scores [[buffer(1)]],                        \
    device const int*     previous_ids    [[buffer(2)]],                        \
    constant     float&   penalty         [[buffer(3)]],                        \
    constant     uint&    length          [[buffer(4)]],                        \
    constant     uint&    vocab_size      [[buffer(5)]],                        \
    uint batch_idx [[thread_position_in_grid]])                                 \
{                                                                               \
  for (uint j = 0; j < length; ++j) {                                          \
    uint read_idx  = batch_idx * length + j;                                    \
    uint write_idx = batch_idx * vocab_size + (uint)previous_ids[read_idx];    \
    float score = (float)previous_scores[read_idx];                             \
    float penalized = (score < 0.f) ? score * penalty : score / penalty;       \
    scores[write_idx] = (T)penalized;                                           \
  }                                                                             \
}

DEFINE_PENALIZE(float)
DEFINE_PENALIZE(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_PENALIZE(bfloat)
#endif
)msl";

// Source: src/metal/kernels/transpose.metal
static constexpr const char* kTransposeMSL = R"msl(
// CTranslate2 Metal transpose primitives — M4.8.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string (kTransposeMSL) in src/metal/primitives.mm and compiled at
// runtime via [MTLDevice newLibraryWithSource:options:error:].
//
// Three kernels per element type:
//   transpose_2d_<T> — matrix transpose (implicit perm = [1, 0])
//   transpose_3d_<T> — arbitrary 3D permutation
//   transpose_4d_<T> — arbitrary 4D permutation
//
// Algorithm: one thread per output element (flat index gid).
//   Decompose gid into multi-index (i0, i1, ...) using pre-computed output
//   strides, then compute the input flat index using permuted input strides.
//
// Argument structs are passed via setBytes: at buffer(2).  Their layout must
// match the C++ structs in primitives.mm exactly.
//
// Buffer layout
// -------------
//   buffer(0): const T*  a     — input
//   buffer(1): T*        b     — output
//   buffer(2): struct    args  — dimension/stride constants (see below)
//
// Dispatch: one thread per output element (grid = N × 1 × 1).
//
// Types: float, half, bfloat (Apple9+ / macOS 14+), int, short, char.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Argument structs (layout must match C++ side in primitives.mm)
// ---------------------------------------------------------------------------

// 2D: input shape [rows, cols]; output shape [cols, rows].
struct TransposeArgs2D {
  uint rows;   // input dims[0]
  uint cols;   // input dims[1]
};

// 3D: output shape [bd0, bd1, bd2] derived from perm-reordered input dims.
// b_s0 = bd1 * bd2,  b_s1 = bd2.
struct TransposeArgs3D {
  uint a_ps0, a_ps1, a_ps2;  // permuted input strides: a_stride[perm[k]]
  uint b_s0;                  // output stride 0 (= bd1 * bd2)
  uint b_s1;                  // output stride 1 (= bd2)
  uint bd1;                   // output dim 1 (for % in index decomposition)
};

// 4D: output shape [bd0, bd1, bd2, bd3] from perm-reordered input dims.
// b_s0 = bd1*bd2*bd3,  b_s1 = bd2*bd3,  b_s2 = bd3.
struct TransposeArgs4D {
  uint a_ps0, a_ps1, a_ps2, a_ps3;  // permuted input strides
  uint b_s0, b_s1, b_s2;            // output strides 0-2 (b_s3 = 1)
  uint bd1, bd2;                     // output dims 1,2 (for % in decomposition)
};

// ---------------------------------------------------------------------------
// Kernel macro — instantiated for each element type T
// ---------------------------------------------------------------------------

#define DEFINE_TRANSPOSE(T)                                                     \
                                                                                \
kernel void transpose_2d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs2D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{                                                                               \
  /* Output flat index gid → input: row = gid % rows, col = gid / rows */      \
  b[gid] = a[(gid % args.rows) * args.cols + (gid / args.rows)];               \
}                                                                               \
                                                                                \
kernel void transpose_3d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs3D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{                                                                               \
  uint i0 =  gid / args.b_s0;                                                  \
  uint i1 = (gid / args.b_s1) % args.bd1;                                      \
  uint i2 =  gid % args.b_s1;                                                  \
  b[gid] = a[i0 * args.a_ps0 + i1 * args.a_ps1 + i2 * args.a_ps2];            \
}                                                                               \
                                                                                \
kernel void transpose_4d_##T(                                                   \
    device const T*           a    [[buffer(0)]],                               \
    device       T*           b    [[buffer(1)]],                               \
    constant TransposeArgs4D& args [[buffer(2)]],                               \
    uint gid [[thread_position_in_grid]])                                        \
{                                                                               \
  uint i0 =  gid / args.b_s0;                                                  \
  uint i1 = (gid / args.b_s1) % args.bd1;                                      \
  uint i2 = (gid / args.b_s2) % args.bd2;                                      \
  uint i3 =  gid % args.b_s2;                                                  \
  b[gid] = a[i0 * args.a_ps0 + i1 * args.a_ps1 +                               \
             i2 * args.a_ps2 + i3 * args.a_ps3];                               \
}

DEFINE_TRANSPOSE(float)
DEFINE_TRANSPOSE(half)
DEFINE_TRANSPOSE(int)
DEFINE_TRANSPOSE(short)
DEFINE_TRANSPOSE(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_TRANSPOSE(bfloat)
#endif
)msl";

// Source: src/metal/kernels/reduction.metal
static constexpr const char* kReductionMSL = R"msl(
// Metal reduction kernels for M4.3.
//
// Two-pass design:
//   Pass 1 (GPU):  each threadgroup of 256 threads reduces its tile of input
//                  to one partial result written to out[tgid].
//   Pass 2 (CPU):  the host reduces the ceil(N/256) partial results to a scalar.
//
// The threadgroup size (256) is fixed.  The host sets:
//   setThreadgroupMemoryLength:(256 * sizeof(elem_T)) atIndex:0
// For max_element, a second threadgroup buffer (uint32_t indices) is also set:
//   setThreadgroupMemoryLength:(256 * sizeof(uint32_t)) atIndex:1

#include <metal_stdlib>
using namespace metal;


// ============================================================
// SUM
//   partial[tgid] = sum of input[tgid*256 .. (tgid+1)*256 - 1]
//   Out-of-bounds threads add the identity: 0.
// ============================================================
#define DEFINE_REDUCE_SUM(T, ZERO)                                       \
kernel void reduce_sum_##T(                                              \
    device const T*      inp   [[buffer(0)]],                            \
    device       T*      out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup T*       shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? inp[gid] : ZERO;                           \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s) { shmem[tid] += shmem[tid + s]; }                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}

DEFINE_REDUCE_SUM(float, 0.f)
DEFINE_REDUCE_SUM(half,  (half)0)
DEFINE_REDUCE_SUM(int,   0)
DEFINE_REDUCE_SUM(short, (short)0)
DEFINE_REDUCE_SUM(char,  (char)0)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_SUM(bfloat, (bfloat)0)
#endif


// ============================================================
// MAX (scalar maximum)
//   partial[tgid] = max of input[tgid*256 .. (tgid+1)*256 - 1]
//   Out-of-bounds threads contribute NEG_INF so they never win.
//
// Identity values:
//   float/bfloat: -FLT_MAX            — exact for bfloat, -inf for half
//   half:         (half)(-FLT_MAX)    — half overflows to -inf; fine as identity
//   int:          (int)0x80000000     — INT_MIN  (-2147483648)
//   short:        (short)0x8000       — SHRT_MIN (-32768)
//   char:         (char)0x80          — SCHAR_MIN (-128)
//
// Uses explicit > comparison rather than max() to avoid MSL overload
// ambiguity for bfloat (no dedicated bfloat max() on all SDK versions).
// ============================================================
#define DEFINE_REDUCE_MAX(T, NEG_INF)                                    \
kernel void reduce_max_##T(                                              \
    device const T*      inp   [[buffer(0)]],                            \
    device       T*      out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup T*       shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? inp[gid] : NEG_INF;                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s && shmem[tid + s] > shmem[tid]) {                   \
            shmem[tid] = shmem[tid + s];                                 \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}

DEFINE_REDUCE_MAX(float, -FLT_MAX)
DEFINE_REDUCE_MAX(half,  (half)(-FLT_MAX))
DEFINE_REDUCE_MAX(int,   (int)0x80000000)
DEFINE_REDUCE_MAX(short, (short)0x8000)
DEFINE_REDUCE_MAX(char,  (char)0x80)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_MAX(bfloat, (bfloat)(-FLT_MAX))
#endif


// ============================================================
// AMAX (max of absolute values)
//   Input is read as its native type; abs and compare are done in float.
//   The output buffer (out) is always float* — the host converts to T.
//   Out-of-bounds threads contribute 0.f (identity for max-of-abs).
// ============================================================
#define DEFINE_REDUCE_AMAX(T)                                            \
kernel void reduce_amax_##T(                                             \
    device const T*      inp   [[buffer(0)]],                            \
    device       float*  out   [[buffer(1)]],                            \
    constant  uint32_t&  n     [[buffer(2)]],                            \
    threadgroup float*   shmem [[threadgroup(0)]],                       \
    uint gid  [[thread_position_in_grid]],                               \
    uint tid  [[thread_index_in_threadgroup]],                           \
    uint tgid [[threadgroup_position_in_grid]],                          \
    uint tgs  [[threads_per_threadgroup]])                               \
{                                                                        \
    shmem[tid] = (gid < n) ? fabs((float)inp[gid]) : 0.f;               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
        if (tid < s) { shmem[tid] = max(shmem[tid], shmem[tid + s]); }  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
    }                                                                    \
    if (tid == 0) { out[tgid] = shmem[0]; }                             \
}

DEFINE_REDUCE_AMAX(float)
DEFINE_REDUCE_AMAX(half)
DEFINE_REDUCE_AMAX(int)
DEFINE_REDUCE_AMAX(short)
DEFINE_REDUCE_AMAX(char)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_AMAX(bfloat)
#endif


// ============================================================
// MAX_ELEMENT (index of the maximum value)
//   Comparison is done in float so all input types are handled uniformly.
//   Outputs two partial arrays (one per threadgroup):
//     out_vals[tgid]  — float value of the threadgroup-local maximum
//     out_idxs[tgid]  — uint32_t global index of that maximum
//   Out-of-bounds threads use (-FLT_MAX, 0xFFFFFFFF) so they never win.
// ============================================================
#define DEFINE_REDUCE_MAX_ELEMENT(T)                                           \
kernel void reduce_max_element_##T(                                            \
    device const T*        inp      [[buffer(0)]],                             \
    device       float*    out_vals [[buffer(1)]],                             \
    device    uint32_t*    out_idxs [[buffer(2)]],                             \
    constant  uint32_t&    n        [[buffer(3)]],                             \
    threadgroup float*     sh_vals  [[threadgroup(0)]],                        \
    threadgroup uint32_t*  sh_idxs  [[threadgroup(1)]],                        \
    uint gid  [[thread_position_in_grid]],                                     \
    uint tid  [[thread_index_in_threadgroup]],                                 \
    uint tgid [[threadgroup_position_in_grid]],                                \
    uint tgs  [[threads_per_threadgroup]])                                     \
{                                                                              \
    bool in_range  = (gid < n);                                                \
    sh_vals[tid]   = in_range ? (float)inp[gid] : -FLT_MAX;                   \
    sh_idxs[tid]   = in_range ? gid             : 0xFFFFFFFFu;                \
    threadgroup_barrier(mem_flags::mem_threadgroup);                           \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                                 \
        if (tid < s && sh_vals[tid + s] > sh_vals[tid]) {                      \
            sh_vals[tid] = sh_vals[tid + s];                                   \
            sh_idxs[tid] = sh_idxs[tid + s];                                  \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
    if (tid == 0) {                                                            \
        out_vals[tgid] = sh_vals[0];                                           \
        out_idxs[tgid] = sh_idxs[0];                                          \
    }                                                                          \
}

DEFINE_REDUCE_MAX_ELEMENT(float)
DEFINE_REDUCE_MAX_ELEMENT(half)
DEFINE_REDUCE_MAX_ELEMENT(int)
DEFINE_REDUCE_MAX_ELEMENT(short)
DEFINE_REDUCE_MAX_ELEMENT(char)
#if defined(__HAVE_BFLOAT__)
DEFINE_REDUCE_MAX_ELEMENT(bfloat)
#endif
)msl";

// Source: src/metal/kernels/normalization.metal
static constexpr const char* kNormalizationMSL = R"msl(
// CTranslate2 Metal normalization kernels — M5.2.
//
// Three kernel families: layer_norm, rms_norm, softmax.
// All use a "one threadgroup per row" design:
//   grid = [num_rows, 1, 1],  threadgroup = [NORM_BLOCK, 1, 1] = [256, 1, 1]
//   threadgroup(0): float[NORM_BLOCK]  — scratch for two-pass reduction
//
// All accumulation is done in float32 for correctness with half/bfloat inputs.
//
// Types: float, half, bfloat (Apple9+ / macOS 14+).

#include <metal_stdlib>
using namespace metal;

constant uint NORM_BLOCK = 256;

// ============================================================
// layer_norm_<T>
//
// One threadgroup per outer row.  Grid = [outer_size].
// Two-pass stable algorithm: pass 1 = mean, pass 2 = variance.
//
// Buffer layout:
//   buffer(0): const T*    x         — input
//   buffer(1): const T*    gamma     — per-channel scale (unused if has_gamma==0)
//   buffer(2): const T*    beta      — per-channel bias  (unused if has_beta==0)
//   buffer(3): T*          y         — output
//   buffer(4): const uint  has_gamma — 1 = apply gamma, 0 = identity scale
//   buffer(5): const uint  has_beta  — 1 = apply beta,  0 = zero bias
//   buffer(6): const uint  N         — axis_size (elements per row)
//   buffer(7): const float eps       — epsilon
// ============================================================
#define DEFINE_LAYER_NORM(T)                                                          \
kernel void layer_norm_##T(                                                           \
    device const T*       x         [[buffer(0)]],                                   \
    device const T*       gamma     [[buffer(1)]],                                   \
    device const T*       beta      [[buffer(2)]],                                   \
    device       T*       y         [[buffer(3)]],                                   \
    constant     uint&    has_gamma [[buffer(4)]],                                   \
    constant     uint&    has_beta  [[buffer(5)]],                                   \
    constant     uint&    N         [[buffer(6)]],                                   \
    constant     float&   eps       [[buffer(7)]],                                   \
    threadgroup  float*   shmem     [[threadgroup(0)]],                              \
    uint tid  [[thread_index_in_threadgroup]],                                        \
    uint tgid [[threadgroup_position_in_grid]])                                       \
{                                                                                     \
    const uint row_off = tgid * N;                                                   \
    /* Pass 1: sum x[i] to compute mean */                                           \
    float s = 0.f;                                                                   \
    for (uint j = tid; j < N; j += NORM_BLOCK) s += (float)x[row_off + j];          \
    shmem[tid] = s;                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint st = NORM_BLOCK >> 1; st > 0; st >>= 1) {                             \
        if (tid < st) shmem[tid] += shmem[tid + st];                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    float mean = shmem[0] / (float)N;                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    /* Pass 2: sum (x[i] - mean)^2 to compute variance */                           \
    float v = 0.f;                                                                   \
    for (uint j = tid; j < N; j += NORM_BLOCK) {                                    \
        float d = (float)x[row_off + j] - mean; v += d * d;                        \
    }                                                                                \
    shmem[tid] = v;                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint st = NORM_BLOCK >> 1; st > 0; st >>= 1) {                             \
        if (tid < st) shmem[tid] += shmem[tid + st];                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    float inv_std = rsqrt(shmem[0] / (float)N + eps);                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    /* Pass 3: normalize, apply optional scale and bias */                           \
    for (uint j = tid; j < N; j += NORM_BLOCK) {                                    \
        float val = ((float)x[row_off + j] - mean) * inv_std;                       \
        float g = has_gamma ? (float)gamma[j] : 1.f;                                \
        float b = has_beta  ? (float)beta[j]  : 0.f;                                \
        y[row_off + j] = (T)(val * g + b);                                          \
    }                                                                                \
}

DEFINE_LAYER_NORM(float)
DEFINE_LAYER_NORM(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_LAYER_NORM(bfloat)
#endif


// ============================================================
// rms_norm_<T>
//
// One threadgroup per batch row.  Grid = [batch_size].
// Single-pass: accumulate sum-of-squares, then normalize.
//
// Buffer layout:
//   buffer(0): const T*    x     — input
//   buffer(1): const T*    gamma — per-channel scale
//   buffer(2): T*          y     — output
//   buffer(3): const uint  N     — depth (elements per row)
//   buffer(4): const float eps   — epsilon
// ============================================================
#define DEFINE_RMS_NORM(T)                                                            \
kernel void rms_norm_##T(                                                             \
    device const T*       x     [[buffer(0)]],                                       \
    device const T*       gamma [[buffer(1)]],                                       \
    device       T*       y     [[buffer(2)]],                                       \
    constant     uint&    N     [[buffer(3)]],                                       \
    constant     float&   eps   [[buffer(4)]],                                       \
    threadgroup  float*   shmem [[threadgroup(0)]],                                  \
    uint tid  [[thread_index_in_threadgroup]],                                        \
    uint tgid [[threadgroup_position_in_grid]])                                       \
{                                                                                     \
    const uint row_off = tgid * N;                                                   \
    /* Accumulate sum-of-squares */                                                  \
    float ss = 0.f;                                                                  \
    for (uint j = tid; j < N; j += NORM_BLOCK) {                                    \
        float v = (float)x[row_off + j]; ss += v * v;                              \
    }                                                                                \
    shmem[tid] = ss;                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint st = NORM_BLOCK >> 1; st > 0; st >>= 1) {                             \
        if (tid < st) shmem[tid] += shmem[tid + st];                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    float rms_inv = rsqrt(shmem[0] / (float)N + eps);                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    /* Normalize and scale */                                                        \
    for (uint j = tid; j < N; j += NORM_BLOCK)                                      \
        y[row_off + j] = (T)((float)x[row_off + j] * rms_inv * (float)gamma[j]);   \
}

DEFINE_RMS_NORM(float)
DEFINE_RMS_NORM(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_RMS_NORM(bfloat)
#endif


// ============================================================
// softmax_<T>
//
// One threadgroup per batch row.  Grid = [batch_size].
// Three passes: max → sum_exp → normalize.
// Supports optional lengths masking and log-softmax mode.
//
// Buffer layout:
//   buffer(0): const T*    x           — input
//   buffer(1): T*          y           — output
//   buffer(2): const int*  lengths     — valid length per row (unused if has_lengths==0)
//   buffer(3): const uint  has_lengths — 1 if lengths buffer is valid, else 0
//   buffer(4): const uint  N           — depth (total elements per row)
//   buffer(5): const uint  log_mode    — 1 for log-softmax, 0 for softmax
// ============================================================
#define DEFINE_SOFTMAX(T)                                                             \
kernel void softmax_##T(                                                              \
    device const T*       x           [[buffer(0)]],                                 \
    device       T*       y           [[buffer(1)]],                                 \
    device const int*     lengths     [[buffer(2)]],                                 \
    constant     uint&    has_lengths [[buffer(3)]],                                 \
    constant     uint&    N           [[buffer(4)]],                                 \
    constant     uint&    log_mode    [[buffer(5)]],                                 \
    threadgroup  float*   shmem       [[threadgroup(0)]],                            \
    uint tid  [[thread_index_in_threadgroup]],                                        \
    uint tgid [[threadgroup_position_in_grid]])                                       \
{                                                                                     \
    const uint row_off  = tgid * N;                                                  \
    const uint active_N = has_lengths ? (uint)lengths[tgid] : N;                    \
    /* When active_N == 0 (fully-masked row):                                        \
     *   Pass 1 leaves max_val = -FLT_MAX.                                           \
     *   Pass 2 sum_e = 0  =>  total_sum = 0.                                        \
     *   log-softmax: log(0) = -inf, but the write loop over [0, active_N) is        \
     *     empty so -inf is never used or written.                                   \
     *   softmax: exp(...) / 0 is never evaluated for the same reason.               \
     *   The zero-fill pass then writes (T)0 to all N output slots.                  \
     *   Result: fully-masked row => all-zeros output.  No NaN is produced. */       \
    /* Pass 1: find max for numerical stability */                                   \
    float mx = -FLT_MAX;                                                             \
    for (uint j = tid; j < active_N; j += NORM_BLOCK) {                             \
        float v = (float)x[row_off + j]; if (v > mx) mx = v;                       \
    }                                                                                \
    shmem[tid] = mx;                                                                 \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint st = NORM_BLOCK >> 1; st > 0; st >>= 1) {                             \
        if (tid < st && shmem[tid + st] > shmem[tid]) shmem[tid] = shmem[tid + st]; \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    float max_val = shmem[0];                                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    /* Pass 2: sum exp(x - max) */                                                   \
    float sum_e = 0.f;                                                               \
    for (uint j = tid; j < active_N; j += NORM_BLOCK)                               \
        sum_e += exp((float)x[row_off + j] - max_val);                              \
    shmem[tid] = sum_e;                                                              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint st = NORM_BLOCK >> 1; st > 0; st >>= 1) {                             \
        if (tid < st) shmem[tid] += shmem[tid + st];                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    float total_sum = shmem[0];                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    /* Pass 3: write output */                                                        \
    if (log_mode) {                                                                  \
        float log_sum = log(total_sum);                                              \
        for (uint j = tid; j < active_N; j += NORM_BLOCK)                           \
            y[row_off + j] = (T)((float)x[row_off + j] - max_val - log_sum);        \
    } else {                                                                         \
        for (uint j = tid; j < active_N; j += NORM_BLOCK)                           \
            y[row_off + j] = (T)(exp((float)x[row_off + j] - max_val) / total_sum); \
    }                                                                                \
    /* Zero-fill out-of-range positions when lengths masking is active */            \
    if (has_lengths) {                                                               \
        for (uint j = active_N + tid; j < N; j += NORM_BLOCK)                       \
            y[row_off + j] = (T)0;                                                  \
    }                                                                                \
}

DEFINE_SOFTMAX(float)
DEFINE_SOFTMAX(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_SOFTMAX(bfloat)
#endif
)msl";

// Source: src/metal/kernels/gather.metal
static constexpr const char* kGatherMSL = R"msl(
// CTranslate2 Metal gather kernel — M5.2.
//
// Kernel: gather_<T>
//
// One thread per output element (gid = slot * copy_size + j).
// For each output position:
//   slot        = gid / copy_size
//   j           = gid % copy_size
//   batch_index = slot / num_indices_per_batch
//   read_index  = indices[slot]
//   dst[gid]    = src[batch_index * batch_stride + read_index * copy_size + j]
//
// This matches the CPU gather loop in gather_cpu.cc:
//   for i in [0, num_indices):
//     batch_index = i / num_indices_per_batch
//     read_index  = indices[i]
//     copy(src + batch_index*batch_stride + read_index*copy_size, dst + i*copy_size, copy_size)
//
// Buffer layout:
//   buffer(0): const T*    src                   — source data
//   buffer(1): T*          dst                   — output
//   buffer(2): const int*  indices               — gather indices (int32)
//   buffer(3): const uint  copy_size             — elements per gather slot
//   buffer(4): const uint  batch_stride          — stride between batches in src
//   buffer(5): const uint  num_indices_per_batch — indices per batch item
//
// Dispatch: grid = [num_indices * copy_size, 1, 1]
//
// Types: float, half, bfloat (Apple9+/macOS 14+), int, short, char.

#include <metal_stdlib>
using namespace metal;

#define DEFINE_GATHER(T)                                                              \
kernel void gather_##T(                                                               \
    device const T*       src                   [[buffer(0)]],                       \
    device       T*       dst                   [[buffer(1)]],                       \
    device const int*     indices               [[buffer(2)]],                       \
    constant     uint&    copy_size             [[buffer(3)]],                       \
    constant     uint&    batch_stride          [[buffer(4)]],                       \
    constant     uint&    num_indices_per_batch [[buffer(5)]],                       \
    uint gid [[thread_position_in_grid]])                                             \
{                                                                                     \
    uint slot        = gid / copy_size;                                              \
    uint j           = gid % copy_size;                                              \
    uint batch_index = slot / num_indices_per_batch;                                 \
    uint read_index  = (uint)indices[slot];                                          \
    dst[gid] = src[batch_index * batch_stride + read_index * copy_size + j];        \
}

DEFINE_GATHER(float)
DEFINE_GATHER(half)
DEFINE_GATHER(int)
DEFINE_GATHER(short)
DEFINE_GATHER(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_GATHER(bfloat)
#endif
)msl";

// Source: src/metal/kernels/sdpa.metal
static constexpr const char* kSdpaMSL = R"msl(
// CTranslate2 Metal SDPA kernels — M6.1.
//
// Causal attention mask: sets scores[gid] = large_neg when the column index
// (gid % seqlen_k) exceeds the row index (gid / seqlen_k) plus `offset`.
//
// Layout: scores[row * seqlen_k + col] (row-major, contiguous).
// Dispatch: MTLSizeMake(seqlen_q * seqlen_k, 1, 1).
// One thread per element.
//
// large_neg values chosen so that exp(x) underflows to zero in softmax:
//   float  : -1e9f    (float32 can represent; exp(-1e9) = 0 in float32)
//   half   : -65504.h (near the float16 minimum; exp(-65504) = 0)
//   bfloat : -1e9f    (bfloat16 has float32-like exponent range)

#include <metal_stdlib>
using namespace metal;

kernel void causal_mask_float(
    device float*  scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = -1e9f;
    }
}

kernel void causal_mask_half(
    device half*   scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = half(-65504.0f);
    }
}

kernel void causal_mask_bfloat(
    device bfloat* scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = bfloat(-1e9f);
    }
}
)msl";

// Source: src/metal/kernels/conv1d.metal
static constexpr const char* kConv1dMSL = R"msl(
// CTranslate2 Metal Conv1D im2col kernel — M8.3.
//
// Transforms input [B, C_in, T_in] into the im2col layout
// [B, T_out, C_in * K] used by the subsequent GEMM.
//
// One thread per output element (B × T_out × C_in × K total threads).
// Out-of-bounds input positions (from padding) produce zeros.
//
// Constraints (M8.3 scope):
//   - groups == 1 only (the Metal compute specialisation throws otherwise)
//   - dilation >= 1 (0 treated as 1 by the caller)
//
// Naming convention:
//   im2col_float, im2col_half, im2col_bfloat
//
// dims0 = { B, C_in, T_in, T_out }
// dims1 = { K, stride, padding, dilation }
//
// gid ∈ [0, B * T_out * C_in * K)
//   decomposed as:  b  = gid / (T_out * C_in * K)
//                   ti = (gid / (C_in * K)) % T_out
//                   c  = (gid / K) % C_in
//                   k  =  gid % K

#include <metal_stdlib>
using namespace metal;

#define DEFINE_IM2COL(T)                                                      \
kernel void im2col_##T(                                                       \
    device const T* input  [[buffer(0)]],                                     \
    device       T* output [[buffer(1)]],                                     \
    constant uint4& dims0  [[buffer(2)]],                                     \
    constant uint4& dims1  [[buffer(3)]],                                     \
    uint gid [[thread_position_in_grid]])                                      \
{                                                                             \
    uint B    = dims0[0];                                                     \
    uint Cin  = dims0[1];                                                     \
    uint Tin  = dims0[2];                                                     \
    uint Tout = dims0[3];                                                     \
    uint K    = dims1[0];                                                     \
    uint strd = dims1[1];                                                     \
    uint pad  = dims1[2];                                                     \
    uint dil  = dims1[3];                                                     \
    uint CK   = Cin * K;                                                      \
    if (gid >= B * Tout * CK) return;                                         \
    uint b  =  gid / (Tout * CK);                                             \
    uint ti = (gid / CK) % Tout;                                              \
    uint c  = (gid % CK) / K;                                                 \
    uint k  =  gid % K;                                                       \
    int  win = (int)(ti * strd) - (int)pad + (int)(k * dil);                 \
    T val = T(0);                                                             \
    if (win >= 0 && win < (int)Tin)                                           \
        val = input[b * Cin * Tin + c * Tin + (uint)win];                     \
    output[gid] = val;                                                        \
}

DEFINE_IM2COL(float)
DEFINE_IM2COL(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_IM2COL(bfloat)
#endif
#undef DEFINE_IM2COL
)msl";

// Source: src/metal/kernels/quantize.metal
static constexpr const char* kQuantizeMSL = R"msl(
// src/metal/kernels/quantize.metal
//
// M9.1 — INT8 Quantize / Dequantize kernels for Metal.
//
// Three kernel families (T = float, half, bfloat):
//
//   quantize_T       — per-row INT8 quantization (one threadgroup per row)
//   dequantize_T     — per-element INT8 dequantization (one thread per element)
//   dequantize_gemm_output_T
//                    — rescale int32 GEMM output to float (one thread per element)
//
// bfloat requires __HAVE_BFLOAT__ (Metal 3.1 / Apple 9+; available on M4).

#include <metal_stdlib>
using namespace metal;

// src/metal/kernels/metal_math.metalh
//
// Shared transcendental math functions for CTranslate2 Metal kernels.
// Included by activation.metal and quantize.metal.

#ifndef CT2_METAL_MATH_H
#define CT2_METAL_MATH_H

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

#endif  // CT2_METAL_MATH_H


// ---------------------------------------------------------------------------
// quantize_T — per-row INT8 quantization
//
// Algorithm:
//   scale[row] = 127 / max(abs(input[row, :]))
//   output[row, i] = char(round(float(input[row, i]) * scale[row]))
//
// Buffer layout:
//   buffer(0): input  [batch_size, depth]  — T
//   buffer(1): output [batch_size, depth]  — char (= int8_t)
//   buffer(2): scales [batch_size]         — float
//   buffer(3): dims   uint2 = {batch_size, depth}
//   threadgroup(0):   float[256]           — scratch for tree reduction
//
// Dispatch: threadgroups=(batch_size,1,1)  threadsPerThreadgroup=(256,1,1)
// ---------------------------------------------------------------------------

#define DEFINE_QUANTIZE(T)                                                    \
kernel void quantize_##T(                                                     \
    device const T*    input  [[buffer(0)]],                                  \
    device char*       output [[buffer(1)]],                                  \
    device float*      scales [[buffer(2)]],                                  \
    constant uint2&    dims   [[buffer(3)]],                                  \
    threadgroup float* sdata  [[threadgroup(0)]],                             \
    uint  tid   [[thread_index_in_threadgroup]],                              \
    uint  blksz [[threads_per_threadgroup]],                                  \
    uint  row   [[threadgroup_position_in_grid]])                             \
{                                                                             \
    uint depth  = dims[1];                                                    \
    device const T* row_in  = input  + row * depth;                          \
    device char*    row_out = output + row * depth;                          \
    /* Step 1: each thread reduces its strided slice to a local abs-max */    \
    float thread_max = 0.f;                                                   \
    for (uint i = tid; i < depth; i += blksz)                                \
        thread_max = max(thread_max, abs(float(row_in[i])));                 \
    sdata[tid] = thread_max;                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    /* Tree reduction to find the row abs-max */                              \
    for (uint s = blksz >> 1; s > 0; s >>= 1) {                             \
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    }                                                                         \
    float amax  = sdata[0];                                                   \
    float scale = (amax != 0.f) ? 127.f / amax : 1.f;                       \
    if (tid == 0) scales[row] = scale;                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                          \
    /* Step 2: scale, round, and cast to int8 */                              \
    for (uint i = tid; i < depth; i += blksz)                                \
        row_out[i] = char(round(float(row_in[i]) * scale));                  \
}

DEFINE_QUANTIZE(float)
DEFINE_QUANTIZE(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_QUANTIZE(bfloat)
#endif

// ---------------------------------------------------------------------------
// dequantize_T — per-element INT8 dequantization
//
// output[row, i] = T(float(input[row, i]) / scales[row])
//
// Buffer layout:
//   buffer(0): input  [batch_size, depth]  — char (= int8_t)
//   buffer(1): scales [batch_size]         — float
//   buffer(2): output [batch_size, depth]  — T
//   buffer(3): dims   uint2 = {batch_size, depth}
//
// Dispatch: dispatchThreads(batch_size * depth, 1, 1)
// ---------------------------------------------------------------------------

#define DEFINE_DEQUANTIZE(T)                                                  \
kernel void dequantize_##T(                                                   \
    device const char*  input  [[buffer(0)]],                                 \
    device const float* scales [[buffer(1)]],                                 \
    device T*           output [[buffer(2)]],                                 \
    constant uint2&     dims   [[buffer(3)]],                                 \
    uint gid [[thread_position_in_grid]])                                     \
{                                                                             \
    uint total = dims[0] * dims[1];                                           \
    if (gid >= total) return;                                                  \
    uint row    = gid / dims[1];                                               \
    output[gid] = T(float(input[gid]) / scales[row]);                        \
}

DEFINE_DEQUANTIZE(float)
DEFINE_DEQUANTIZE(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_DEQUANTIZE(bfloat)
#endif

// ---------------------------------------------------------------------------
// dequantize_gemm_output_T — rescale int32 GEMM output to floating point
//
// y[i, j] = c[i, j] / (a_scales[ta ? j : i] * b_scales[tb ? j : i])
//         + (has_bias ? bias[j] : 0)
// followed by optional activation (act_type).
//
// act_type encoding (0 = none; matches static_cast<uint>(ActivationType) + 1):
//   0=none  1=relu  2=gelu_tanh  3=swish  4=gelu  5=gelu_sigmoid
//   6=tanh  7=sigmoid
//
// Buffer layout:
//   buffer(0): c        [batch, depth]   — int (= int32_t)
//   buffer(1): a_scales [batch or depth] — float
//   buffer(2): b_scales [batch or depth] — float
//   buffer(3): bias     [depth] or dummy — T (bind dummy when has_bias == 0)
//   buffer(4): y        [batch, depth]   — T (output)
//   buffer(5): dims     uint4 = {batch, depth, has_bias, act_type}
//   buffer(6): trans    uint2 = {transpose_a, transpose_b}
//
// Dispatch: dispatchThreads(batch * depth, 1, 1)
// ---------------------------------------------------------------------------

#define DEFINE_DEQUANTIZE_GEMM(T)                                              \
kernel void dequantize_gemm_output_##T(                                        \
    device const int*   c        [[buffer(0)]],                                \
    device const float* a_scales [[buffer(1)]],                                \
    device const float* b_scales [[buffer(2)]],                                \
    device const T*     bias     [[buffer(3)]],                                \
    device T*           y        [[buffer(4)]],                                \
    constant uint4&     dims     [[buffer(5)]],                                \
    constant uint2&     trans    [[buffer(6)]],                                \
    uint gid [[thread_position_in_grid]])                                      \
{                                                                              \
    uint batch = dims[0], depth = dims[1];                                     \
    uint has_bias = dims[2], act_type = dims[3];                               \
    if (gid >= batch * depth) return;                                          \
    uint i = gid / depth, j = gid % depth;                                    \
    float a_s = a_scales[trans[0] ? j : i];                                   \
    float b_s = b_scales[trans[1] ? j : i];                                   \
    float v = float(c[gid]) / (a_s * b_s);                                    \
    if (has_bias) v += float(bias[j]);                                         \
    if (act_type == 1u) {          /* relu */                                  \
        v = max(v, 0.f);                                                       \
    } else if (act_type == 2u) {   /* gelu_tanh */                             \
        float t = v * (1.f + 0.044715f * v * v) * 0.7978845608028654f;        \
        v = v * 0.5f * (1.f + ct2_safe_tanh(t));                              \
    } else if (act_type == 3u) {   /* swish */                                 \
        v = v * (1.f / (1.f + exp(-v)));                                       \
    } else if (act_type == 4u) {   /* gelu */                                  \
        v = v * 0.5f * (1.f + ct2_erf(v * 0.7071067811865476f));              \
    } else if (act_type == 5u) {   /* gelu_sigmoid */                          \
        v = v * (1.f / (1.f + exp(-1.702f * v)));                             \
    } else if (act_type == 6u) {   /* tanh */                                  \
        v = ct2_safe_tanh(v);                                                   \
    } else if (act_type == 7u) {   /* sigmoid */                               \
        v = 1.f / (1.f + exp(-v));                                             \
    }                                                                          \
    y[gid] = T(v);                                                             \
}

DEFINE_DEQUANTIZE_GEMM(float)
DEFINE_DEQUANTIZE_GEMM(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_DEQUANTIZE_GEMM(bfloat)
#endif
)msl";

