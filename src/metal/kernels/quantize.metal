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

// ---------------------------------------------------------------------------
// ct2_erf — Abramowitz & Stegun 7.1.28 polynomial, max error 1.5e-7
// MSL does not guarantee erf() across all targets; this is always safe.
// ---------------------------------------------------------------------------
static inline float ct2_erf(float x) {
    float t = 1.f / (1.f + 0.3275911f * abs(x));
    float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                + t * (-1.453152027f + t * 1.061405429f))));
    float e = 1.f - p * exp(-x * x);
    return x >= 0.f ? e : -e;
}

// ---------------------------------------------------------------------------
// ct2_safe_tanh — clamped tanh to avoid NaN from exp(2x) overflow in Metal
// ---------------------------------------------------------------------------
static inline float ct2_safe_tanh(float x) {
    return tanh(clamp(x, -10.f, 10.f));
}

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
