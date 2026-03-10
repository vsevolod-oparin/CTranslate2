// indexed_fill.metal — GPU scatter fill kernel.
//
// M11.25: Replaces CPU CT2_COMMIT_AND_WAIT() + loop in indexed_fill.
// Each thread writes x[indices[gid]] = fill_val.  Encode-only — no sync
// required because the GPU-aware allocator (M11.25) prevents buffer
// recycling while there is uncommitted GPU work.

#include <metal_stdlib>
using namespace metal;

#define DEFINE_INDEXED_FILL(T)                                           \
  kernel void indexed_fill_##T(                                          \
      device       T*       x        [[buffer(0)]],                      \
      constant     T&       fill_val [[buffer(1)]],                      \
      device const int*     indices  [[buffer(2)]],                      \
      uint gid [[thread_position_in_grid]])                              \
  { x[indices[gid]] = fill_val; }

DEFINE_INDEXED_FILL(float)
DEFINE_INDEXED_FILL(half)
DEFINE_INDEXED_FILL(int)
DEFINE_INDEXED_FILL(short)
DEFINE_INDEXED_FILL(char)

#if defined(__HAVE_BFLOAT__)
DEFINE_INDEXED_FILL(bfloat)
#endif
