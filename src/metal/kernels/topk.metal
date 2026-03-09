// CTranslate2 Metal TopK kernel — M11.6.
//
// Kernel: argmax_<T>
//
// Per-row argmax: one threadgroup (TOPK_BLOCK threads) per batch item.
// Each thread scans a strided slice of the row, then a tree reduction
// finds the global maximum value and its index.
//
// Buffer layout:
//   buffer(0): const T*      input   — [batch_size, depth]
//   buffer(1): T*            values  — [batch_size] (output max values)
//   buffer(2): int*          indices — [batch_size] (output argmax indices)
//   buffer(3): const uint    depth   — number of elements per row
//   threadgroup(0): float[TOPK_BLOCK]   — shared values for reduction
//   threadgroup(1): uint[TOPK_BLOCK]    — shared indices for reduction
//
// Dispatch: grid = [batch_size, 1, 1], threadgroup = [TOPK_BLOCK, 1, 1]
//
// Types: float, half, bfloat (Apple9+/macOS 14+).

#include <metal_stdlib>
using namespace metal;

#ifndef TOPK_BLOCK
#define TOPK_BLOCK 256
#endif

#define DEFINE_ARGMAX(T)                                                        \
kernel void argmax_##T(                                                         \
    device const T*       input   [[buffer(0)]],                              \
    device       T*       values  [[buffer(1)]],                              \
    device       int*     indices [[buffer(2)]],                              \
    constant     uint&    depth   [[buffer(3)]],                              \
    threadgroup  float*   sh_vals [[threadgroup(0)]],                         \
    threadgroup  uint*    sh_idxs [[threadgroup(1)]],                         \
    uint batch_id [[threadgroup_position_in_grid]],                            \
    uint tid      [[thread_index_in_threadgroup]],                             \
    uint tgs      [[threads_per_threadgroup]])                                  \
{                                                                               \
    device const T* row = input + batch_id * depth;                           \
                                                                                \
    /* Each thread finds its local max across strided elements */               \
    float best_val = -FLT_MAX;                                                 \
    uint  best_idx = 0;                                                        \
    for (uint i = tid; i < depth; i += tgs) {                                 \
        float v = (float)row[i];                                              \
        if (v > best_val) {                                                   \
            best_val = v;                                                     \
            best_idx = i;                                                     \
        }                                                                     \
    }                                                                          \
                                                                                \
    sh_vals[tid] = best_val;                                                   \
    sh_idxs[tid] = best_idx;                                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                            \
                                                                                \
    /* Tree reduction */                                                        \
    for (uint s = tgs >> 1; s > 0; s >>= 1) {                                \
        if (tid < s && sh_vals[tid + s] > sh_vals[tid]) {                     \
            sh_vals[tid] = sh_vals[tid + s];                                  \
            sh_idxs[tid] = sh_idxs[tid + s];                                  \
        }                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    }                                                                          \
                                                                                \
    if (tid == 0) {                                                            \
        values[batch_id]  = (T)sh_vals[0];                                    \
        indices[batch_id] = (int)sh_idxs[0];                                  \
    }                                                                          \
}

DEFINE_ARGMAX(float)
DEFINE_ARGMAX(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_ARGMAX(bfloat)
#endif

// ---------------------------------------------------------------------------
// Single-pass fused TopK kernel for k > 1 — M11.22.
//
// Algorithm (two phases):
//   Phase 1 — Scan: each thread scans ~N/T elements with strided access,
//     maintaining a sorted (descending) local top-k array in private registers
//     via insertion sort.  Single memory read of the input.
//   Phase 2 — K reduction rounds: each round, every thread offers its current
//     best candidate to shared memory, a tree reduction finds the global best,
//     thread 0 writes the winner to output, and the winning thread advances
//     its rank to offer its next-best candidate in the following round.
//
// Buffer layout:
//   buffer(0): const T*      input   — [batch_size, depth]
//   buffer(1): T*            values  — [batch_size, k] (output top-k values)
//   buffer(2): int*          indices — [batch_size, k] (output top-k indices)
//   buffer(3): const uint    depth   — number of elements per row
//   buffer(4): const uint    k       — number of top elements to find
//   threadgroup(0): float[TOPK_BLOCK]   — shared values for reduction
//   threadgroup(1): uint[TOPK_BLOCK]    — shared indices for reduction
//
// Dispatch: grid = [batch_size, 1, 1], threadgroup = [TOPK_BLOCK, 1, 1]
// ---------------------------------------------------------------------------

#ifndef TOPK_MAX_K
#define TOPK_MAX_K 64
#endif

#define DEFINE_TOPK_FUSED(T)                                                    \
kernel void topk_fused_##T(                                                     \
    device const T*       input   [[buffer(0)]],                              \
    device       T*       values  [[buffer(1)]],                              \
    device       int*     indices [[buffer(2)]],                              \
    constant     uint&    depth   [[buffer(3)]],                              \
    constant     uint&    k       [[buffer(4)]],                              \
    threadgroup  float*   sh_vals [[threadgroup(0)]],                         \
    threadgroup  uint*    sh_idxs [[threadgroup(1)]],                         \
    uint batch_id [[threadgroup_position_in_grid]],                            \
    uint tid      [[thread_index_in_threadgroup]],                             \
    uint tgs      [[threads_per_threadgroup]])                                  \
{                                                                               \
    device const T* row = input + batch_id * depth;                           \
    uint actual_k = min(k, min(depth, (uint)TOPK_MAX_K));                     \
                                                                                \
    /* Phase 1: Scan — build local sorted top-k in private registers */        \
    float priv_vals[TOPK_MAX_K];                                               \
    uint  priv_idxs[TOPK_MAX_K];                                              \
    for (uint j = 0; j < actual_k; j++) {                                     \
        priv_vals[j] = -FLT_MAX;                                              \
        priv_idxs[j] = 0;                                                     \
    }                                                                          \
                                                                                \
    for (uint i = tid; i < depth; i += tgs) {                                 \
        float v = (float)row[i];                                              \
        if (v > priv_vals[actual_k - 1]) {                                    \
            priv_vals[actual_k - 1] = v;                                      \
            priv_idxs[actual_k - 1] = i;                                      \
            /* Insertion sort: bubble up to maintain descending order */       \
            for (int j = (int)actual_k - 2; j >= 0; j--) {                   \
                if (priv_vals[j + 1] > priv_vals[j]) {                        \
                    float tv = priv_vals[j];                                  \
                    priv_vals[j] = priv_vals[j + 1];                          \
                    priv_vals[j + 1] = tv;                                    \
                    uint ti = priv_idxs[j];                                   \
                    priv_idxs[j] = priv_idxs[j + 1];                         \
                    priv_idxs[j + 1] = ti;                                    \
                } else {                                                       \
                    break;                                                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                                \
    /* Phase 2: K rounds of tree reduction */                                  \
    uint priv_rank = 0;                                                        \
    for (uint round = 0; round < actual_k; round++) {                         \
        float my_val = (priv_rank < actual_k)                                 \
                           ? priv_vals[priv_rank] : -FLT_MAX;                 \
        uint  my_idx = (priv_rank < actual_k)                                 \
                           ? priv_idxs[priv_rank] : 0;                        \
                                                                                \
        sh_vals[tid] = my_val;                                                 \
        sh_idxs[tid] = my_idx;                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
                                                                                \
        /* Tree reduction — find global best */                                \
        for (uint s = tgs >> 1; s > 0; s >>= 1) {                            \
            if (tid < s && sh_vals[tid + s] > sh_vals[tid]) {                 \
                sh_vals[tid] = sh_vals[tid + s];                              \
                sh_idxs[tid] = sh_idxs[tid + s];                              \
            }                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                    \
        }                                                                      \
                                                                                \
        if (tid == 0) {                                                        \
            values[batch_id * k + round]  = (T)sh_vals[0];                    \
            indices[batch_id * k + round] = (int)sh_idxs[0];                  \
        }                                                                      \
                                                                                \
        /* Broadcast winner — sh_idxs[0] valid after reduction */             \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        uint winner_idx = sh_idxs[0];                                          \
                                                                                \
        /* Winning thread advances rank for next round */                      \
        if (my_idx == winner_idx && my_val > -FLT_MAX) {                      \
            priv_rank++;                                                       \
        }                                                                      \
    }                                                                          \
}

DEFINE_TOPK_FUSED(float)
DEFINE_TOPK_FUSED(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_TOPK_FUSED(bfloat)
#endif
