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
