// CTranslate2 Metal fused LayerNorm/RMSNorm + GEMV kernel.
//
// Fuses normalization with matrix-vector multiplication (GEMV) in a single
// dispatch.  One threadgroup (256 threads) per input row.
//
// Phase 1: Normalize the input row into threadgroup memory (float32).
// Phase 2: Each thread computes ceil(N/256) output columns via dot products.
//
// Weight layout: W[N, K] row-major (trans_b=true convention).
// Threadgroup memory: K floats for the normalized row.
//
// Types: float, half, bfloat (Apple9+ / macOS 14+).

#include <metal_stdlib>
using namespace metal;

constant uint FUSED_BLOCK = 256;

// ============================================================
// fused_layer_norm_gemm_<T>
//
// Buffer layout:
//   buffer(0): const T*    x         — input  [outer_size, K]
//   buffer(1): const T*    gamma     — scale   [K]
//   buffer(2): const T*    beta      — bias    [K] (ignored if has_beta==0)
//   buffer(3): const T*    W         — weight  [N, K] row-major
//   buffer(4): T*          y         — output  [outer_size, N]
//   buffer(5): const uint  K_dim     — input dimension
//   buffer(6): const uint  N_dim     — output dimension
//   buffer(7): const uint  has_beta  — 1 = apply beta, 0 = zero bias
//   buffer(8): const float eps       — epsilon
//   threadgroup(0): float[K_dim]     — normalized row
// ============================================================
#define DEFINE_FUSED_LN_GEMM(T)                                                     \
kernel void fused_layer_norm_gemm_##T(                                              \
    device const T*       x         [[buffer(0)]],                                  \
    device const T*       gamma     [[buffer(1)]],                                  \
    device const T*       beta      [[buffer(2)]],                                  \
    device const T*       W         [[buffer(3)]],                                  \
    device       T*       y         [[buffer(4)]],                                  \
    constant     uint&    K_dim     [[buffer(5)]],                                  \
    constant     uint&    N_dim     [[buffer(6)]],                                  \
    constant     uint&    has_beta  [[buffer(7)]],                                  \
    constant     float&   eps       [[buffer(8)]],                                  \
    threadgroup  float*   norm_row  [[threadgroup(0)]],                             \
    uint tid  [[thread_index_in_threadgroup]],                                       \
    uint tgid [[threadgroup_position_in_grid]])                                      \
{                                                                                    \
    const uint row_off = tgid * K_dim;                                              \
    /* --- Phase 1: LayerNorm into threadgroup memory --- */                         \
    /* Pass 1: compute mean */                                                      \
    float s = 0.f;                                                                  \
    for (uint j = tid; j < K_dim; j += FUSED_BLOCK)                                \
        s += (float)x[row_off + j];                                                \
    /* Reduction for mean */                                                        \
    threadgroup float red[FUSED_BLOCK];                                             \
    red[tid] = s;                                                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint st = FUSED_BLOCK >> 1; st > 0; st >>= 1) {                          \
        if (tid < st) red[tid] += red[tid + st];                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float mean = red[0] / (float)K_dim;                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    /* Pass 2: compute variance */                                                  \
    float v = 0.f;                                                                  \
    for (uint j = tid; j < K_dim; j += FUSED_BLOCK) {                             \
        float d = (float)x[row_off + j] - mean; v += d * d;                       \
    }                                                                               \
    red[tid] = v;                                                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint st = FUSED_BLOCK >> 1; st > 0; st >>= 1) {                          \
        if (tid < st) red[tid] += red[tid + st];                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float inv_std = rsqrt(red[0] / (float)K_dim + eps);                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    /* Pass 3: normalize + gamma/beta -> norm_row (float32) */                      \
    for (uint j = tid; j < K_dim; j += FUSED_BLOCK) {                             \
        float val = ((float)x[row_off + j] - mean) * inv_std;                     \
        float g = (float)gamma[j];                                                 \
        float b = has_beta ? (float)beta[j] : 0.f;                                \
        norm_row[j] = val * g + b;                                                \
    }                                                                               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    /* --- Phase 2: GEMV --- */                                                     \
    const uint out_off = tgid * N_dim;                                             \
    for (uint j = tid; j < N_dim; j += FUSED_BLOCK) {                             \
        float acc = 0.f;                                                           \
        const uint w_row = j * K_dim;                                              \
        for (uint i = 0; i < K_dim; i++)                                           \
            acc += norm_row[i] * (float)W[w_row + i];                              \
        y[out_off + j] = (T)acc;                                                   \
    }                                                                               \
}

DEFINE_FUSED_LN_GEMM(float)
DEFINE_FUSED_LN_GEMM(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_FUSED_LN_GEMM(bfloat)
#endif


// ============================================================
// fused_rms_norm_gemm_<T>
//
// Same as above but RMSNorm (no mean subtraction).
// Buffer layout identical except no beta (buffer(2) unused).
// ============================================================
#define DEFINE_FUSED_RMS_GEMM(T)                                                    \
kernel void fused_rms_norm_gemm_##T(                                                \
    device const T*       x         [[buffer(0)]],                                  \
    device const T*       gamma     [[buffer(1)]],                                  \
    device const T*       W         [[buffer(2)]],                                  \
    device       T*       y         [[buffer(3)]],                                  \
    constant     uint&    K_dim     [[buffer(4)]],                                  \
    constant     uint&    N_dim     [[buffer(5)]],                                  \
    constant     float&   eps       [[buffer(6)]],                                  \
    threadgroup  float*   norm_row  [[threadgroup(0)]],                             \
    uint tid  [[thread_index_in_threadgroup]],                                       \
    uint tgid [[threadgroup_position_in_grid]])                                      \
{                                                                                    \
    const uint row_off = tgid * K_dim;                                              \
    /* --- Phase 1: RMSNorm into threadgroup memory --- */                           \
    float ss = 0.f;                                                                 \
    for (uint j = tid; j < K_dim; j += FUSED_BLOCK) {                             \
        float v = (float)x[row_off + j]; ss += v * v;                             \
    }                                                                               \
    threadgroup float red[FUSED_BLOCK];                                             \
    red[tid] = ss;                                                                  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    for (uint st = FUSED_BLOCK >> 1; st > 0; st >>= 1) {                          \
        if (tid < st) red[tid] += red[tid + st];                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                            \
    }                                                                               \
    float rms_inv = rsqrt(red[0] / (float)K_dim + eps);                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    /* Normalize + gamma -> norm_row */                                              \
    for (uint j = tid; j < K_dim; j += FUSED_BLOCK)                                \
        norm_row[j] = (float)x[row_off + j] * rms_inv * (float)gamma[j];          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                \
    /* --- Phase 2: GEMV --- */                                                     \
    const uint out_off = tgid * N_dim;                                             \
    for (uint j = tid; j < N_dim; j += FUSED_BLOCK) {                             \
        float acc = 0.f;                                                           \
        const uint w_row = j * K_dim;                                              \
        for (uint i = 0; i < K_dim; i++)                                           \
            acc += norm_row[i] * (float)W[w_row + i];                              \
        y[out_off + j] = (T)acc;                                                   \
    }                                                                               \
}

DEFINE_FUSED_RMS_GEMM(float)
DEFINE_FUSED_RMS_GEMM(half)
#if defined(__HAVE_BFLOAT__)
DEFINE_FUSED_RMS_GEMM(bfloat)
#endif
