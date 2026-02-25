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
