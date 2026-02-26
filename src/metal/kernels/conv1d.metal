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
