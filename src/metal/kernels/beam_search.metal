// CTranslate2 Metal beam-search primitives — M4.7.
//
// This file is the canonical MSL source.  It is also embedded verbatim as a
// raw C++ string (kBeamSearchMSL) in src/metal/primitives.mm and compiled at
// runtime via [MTLDevice newLibraryWithSource:options:error:].
//
// Kernel: penalize_previous_tokens
// ---------------------------------
//   One thread per batch item; iterates sequentially over `length` previous
//   token IDs and applies a repetition penalty to the scores buffer in-place.
//
//   For each position j in [0, length):
//     read_idx  = batch_idx * length + j
//     write_idx = batch_idx * vocab_size + previous_ids[read_idx]
//     score     = previous_scores[read_idx]
//     scores[write_idx] = (score < 0) ? score * penalty : score / penalty
//
//   Sequential iteration within each thread ensures that duplicate token IDs
//   across positions produce a deterministic result (last write wins),
//   matching CPU semantics.
//
// Buffer layout
// -------------
//   buffer(0): T*         scores          — in-place output (logits to penalise)
//   buffer(1): const T*   previous_scores — prior step log-probabilities
//   buffer(2): const int* previous_ids    — token IDs generated so far
//   buffer(3): float      penalty         — repetition penalty scalar (> 1 = stronger)
//   buffer(4): uint       length          — number of previous tokens per batch item
//   buffer(5): uint       vocab_size      — vocabulary size
//
// Dispatch: one thread per batch item (grid = batch_size × 1 × 1).
//
// Types supported: float, half, bfloat (Apple9+ / macOS 14+).

#include <metal_stdlib>
using namespace metal;

#define DEFINE_PENALIZE(T)                                                      \
kernel void penalize_previous_tokens_##T(                                       \
    device       T*       scores          [[buffer(0)]],                        \
    device const T*       previous_scores [[buffer(1)]],                        \
    device const int*     previous_ids    [[buffer(2)]],                        \
    constant     float&   penalty         [[buffer(3)]],                        \
    constant     uint&    length          [[buffer(4)]],                        \
    constant     uint&    vocab_size      [[buffer(5)]],                        \
    uint batch_idx [[thread_position_in_grid]])                                 \
{                                                                               \
  for (uint j = 0; j < length; ++j) {                                          \
    uint read_idx  = batch_idx * length + j;                                    \
    uint write_idx = batch_idx * vocab_size + (uint)previous_ids[read_idx];    \
    float score = (float)previous_scores[read_idx];                             \
    float penalized = (score < 0.f) ? score * penalty : score / penalty;       \
    scores[write_idx] = (T)penalized;                                           \
  }                                                                             \
}

DEFINE_PENALIZE(float)
DEFINE_PENALIZE(half)

#if defined(__HAVE_BFLOAT__)
DEFINE_PENALIZE(bfloat)
#endif

// Kernel: prepare_length_mask
// ---------------------------
//   2D grid: [batch_size, num_heads * num_queries].
//   Each thread writes one element of the output mask.
//
//   buffer(0): const int*  lengths     — per-batch sequence lengths
//   buffer(1): int*        mask        — output mask [batch_size, num_heads * num_queries]
//   buffer(2): uint        num_heads
//   buffer(3): uint        num_queries
//   buffer(4): uint        mask_future — 0 or 1
//   buffer(5): uint        multi_query — 0 or 1
kernel void prepare_length_mask(
    device const int*  lengths     [[buffer(0)]],
    device       int*  mask        [[buffer(1)]],
    constant     uint& num_heads   [[buffer(2)]],
    constant     uint& num_queries [[buffer(3)]],
    constant     uint& mask_future [[buffer(4)]],
    constant     uint& multi_query [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.x;
    uint i = gid.y;
    int length = lengths[b];
    int val;
    if (mask_future) {
        uint idx = multi_query ? (i / num_heads) : (i % num_queries);
        val = min(length, (int)(idx + 1));
    } else {
        val = length;
    }
    mask[b * num_heads * num_queries + i] = val;
}
