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
