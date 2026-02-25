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
