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

// --- M14.3: f32-promoted broadcast variants for half -------------------------
#define DEFINE_BATCH_BROADCAST_F32(name, op, T)                          \
kernel void name##_batch_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& a_size [[buffer(3)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = (T)((float)a[gid % a_size] op (float)b[gid]); }

#define DEFINE_DEPTH_BROADCAST_F32(name, op, T)                          \
kernel void name##_depth_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& depth [[buffer(3)]],                                 \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = (T)((float)a[gid / depth] op (float)b[gid]); }

#define DEFINE_BLOCK_BROADCAST_F32(name, op, T)                          \
kernel void name##_block_broadcast_##T(                                  \
    device const T* a    [[buffer(0)]],                                  \
    device const T* b    [[buffer(1)]],                                  \
    device       T* c    [[buffer(2)]],                                  \
    constant  uint& block  [[buffer(3)]],                                \
    constant  uint& a_size [[buffer(4)]],                                \
    uint gid [[thread_position_in_grid]])                                 \
{ c[gid] = (T)((float)a[(gid / block) % a_size] op (float)b[gid]); }

#define DEFINE_BROADCAST_OPS(T)       \
  DEFINE_BATCH_BROADCAST(add, +, T)   \
  DEFINE_DEPTH_BROADCAST(add, +, T)   \
  DEFINE_BLOCK_BROADCAST(add, +, T)   \
  DEFINE_BATCH_BROADCAST(mul, *, T)

// M14.3: half uses f32-promoted broadcast; other types use native.
#define DEFINE_BROADCAST_OPS_F32(T)       \
  DEFINE_BATCH_BROADCAST_F32(add, +, T)   \
  DEFINE_DEPTH_BROADCAST_F32(add, +, T)   \
  DEFINE_BLOCK_BROADCAST_F32(add, +, T)   \
  DEFINE_BATCH_BROADCAST_F32(mul, *, T)

DEFINE_BROADCAST_OPS(float)
DEFINE_BROADCAST_OPS_F32(half)
DEFINE_BROADCAST_OPS(int)
DEFINE_BROADCAST_OPS(short)
DEFINE_BROADCAST_OPS(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_BROADCAST_OPS(bfloat)
#endif
