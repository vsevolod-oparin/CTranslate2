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
// TopK kernel for k > 1: iterative argmax with excluded-index list.
//
// Runs k sequential argmax passes within a single kernel launch.
// After each pass, the found index is recorded in a threadgroup excluded[]
// array so subsequent passes skip it.
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

#define DEFINE_TOPK_K(T)                                                        \
kernel void topk_k_##T(                                                         \
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
    uint actual_k = min(k, (uint)TOPK_MAX_K);                                 \
    actual_k = min(actual_k, depth);                                           \
                                                                                \
    /* Excluded indices — lives in thread 0's view, broadcast via barrier */  \
    threadgroup uint excluded[TOPK_MAX_K];                                     \
                                                                                \
    for (uint iter = 0; iter < actual_k; iter++) {                             \
        /* Each thread finds its local max, skipping excluded indices */       \
        float best_val = -FLT_MAX;                                            \
        uint  best_idx = 0;                                                   \
        for (uint i = tid; i < depth; i += tgs) {                            \
            /* Check if this index is excluded */                             \
            bool skip = false;                                                \
            for (uint e = 0; e < iter; e++) {                                \
                if (i == excluded[e]) { skip = true; break; }                \
            }                                                                 \
            if (skip) continue;                                               \
            float v = (float)row[i];                                         \
            if (v > best_val) {                                              \
                best_val = v;                                                \
                best_idx = i;                                                \
            }                                                                 \
        }                                                                     \
                                                                                \
        sh_vals[tid] = best_val;                                              \
        sh_idxs[tid] = best_idx;                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                                \
        /* Tree reduction */                                                   \
        for (uint s = tgs >> 1; s > 0; s >>= 1) {                           \
            if (tid < s && sh_vals[tid + s] > sh_vals[tid]) {                \
                sh_vals[tid] = sh_vals[tid + s];                             \
                sh_idxs[tid] = sh_idxs[tid + s];                             \
            }                                                                 \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                     \
                                                                                \
        /* Thread 0 writes result and records excluded index */               \
        if (tid == 0) {                                                       \
            values[batch_id * k + iter]  = (T)sh_vals[0];                    \
            indices[batch_id * k + iter] = (int)sh_idxs[0];                  \
            excluded[iter] = sh_idxs[0];                                      \
        }                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    }                                                                          \
}

DEFINE_TOPK_K(float)
DEFINE_TOPK_K(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_TOPK_K(bfloat)
#endif
