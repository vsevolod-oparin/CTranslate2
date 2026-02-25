// CTranslate2 Metal SDPA kernels — M6.1.
//
// Causal attention mask: sets scores[gid] = large_neg when the column index
// (gid % seqlen_k) exceeds the row index (gid / seqlen_k) plus `offset`.
//
// Layout: scores[row * seqlen_k + col] (row-major, contiguous).
// Dispatch: MTLSizeMake(seqlen_q * seqlen_k, 1, 1).
// One thread per element.
//
// large_neg values chosen so that exp(x) underflows to zero in softmax:
//   float  : -1e9f    (float32 can represent; exp(-1e9) = 0 in float32)
//   half   : -65504.h (near the float16 minimum; exp(-65504) = 0)
//   bfloat : -1e9f    (bfloat16 has float32-like exponent range)

#include <metal_stdlib>
using namespace metal;

kernel void causal_mask_float(
    device float*  scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = -1e9f;
    }
}

kernel void causal_mask_half(
    device half*   scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = half(-65504.0f);
    }
}

kernel void causal_mask_bfloat(
    device bfloat* scores   [[buffer(0)]],
    constant uint& seqlen_k [[buffer(1)]],
    constant uint& offset   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint row = gid / seqlen_k;
    uint col = gid % seqlen_k;
    if (col > row + offset) {
        scores[gid] = bfloat(-1e9f);
    }
}
