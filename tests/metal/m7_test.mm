// tests/metal/m7_test.mm
//
// M7 — Correctness tests for the CPU-fallback Metal op specializations.
//
// Strategy: each M7 Metal op uses commit_and_wait() to flush any pending
// GPU writes into shared memory, then operates on the CPU-visible shared
// Metal buffer pointers.  These tests validate two things:
//
//   (a) Memory coherency — data written by a GPU op is correctly visible
//       on the CPU after commit_and_wait().
//   (b) Algorithm correctness — each M7 algorithm produces the expected
//       results on Metal-allocated shared-memory buffers.
//
// Tests:
//   1.  Memory coherency: GPU elementwise-add → commit_and_wait → CPU read
//   2.  Concat: two float32 row vectors → contiguous output (axis=0)
//   3.  Concat axis=1: two [2×2] matrices concat along axis 1
//   4.  Split: [4×4] tensor split into two [2×4] halves
//   5.  Slide: extract a sub-slice along axis 0
//   6.  Tile: 1×N tiled T times along axis 0
//   7.  Tile axis=1: tile along last axis
//   8.  TopK k=1: argmax over each row (float32)
//   9.  TopK k=3: top-3 values and indices (float32)
//  10.  TopK k=1: float16 support
//  11.  TopK k=1: bfloat16 support
//  12.  TopPMask: nucleus sampling mask — top-p elements kept, rest masked
//  13.  Mean axis=last: [2×4] → [2] float32 mean
//  14.  Mean axis=first: [4×2] → [2] float32 mean (inner_size > 1)
//  15.  MedianFilter: width=3, known sliding median result
//  16.  GumbelMax noise: output > input on average (noise adds positive gumbel)
//  17.  Multinomial: all sampled indices within valid class range
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/m7_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm \
//     src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm \
//     src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm \
//     src/metal/primitives_beam_search.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     src/random.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o m7_test && ./m7_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/random.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// Resolve ::float16_t / ctranslate2::float16_t conflict from arm_vector_types.h.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK(label, expr) \
  do { \
    bool _ok = (bool)(expr); \
    if (_ok) { std::printf("  PASS  %s\n", label); ++g_pass; } \
    else     { std::printf("  FAIL  %s\n", label); ++g_fail; } \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(
      static_cast<std::size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

// ---------------------------------------------------------------------------
// M7 algorithm helpers (mirrors the implementations in the *_metal.mm files)
// ---------------------------------------------------------------------------

// Concat two arrays along axis=0 (each has iter_size rows of copy_size elems).
template <typename T>
static void ref_concat_axis0(const T* a, const T* b, T* out,
                              dim_t n_a, dim_t n_b) {
  std::memcpy(out,       a, n_a * sizeof(T));
  std::memcpy(out + n_a, b, n_b * sizeof(T));
}

// Concat along axis=1: a=[rows×ca], b=[rows×cb] → out=[rows×(ca+cb)]
template <typename T>
static void ref_concat_axis1(const T* a, const T* b, T* out,
                              dim_t rows, dim_t ca, dim_t cb) {
  const dim_t step = ca + cb;
  for (dim_t r = 0; r < rows; ++r) {
    std::memcpy(out + r * step,      a + r * ca, ca * sizeof(T));
    std::memcpy(out + r * step + ca, b + r * cb, cb * sizeof(T));
  }
}

// Split [rows × 2*cols] → two [rows × cols] arrays.
template <typename T>
static void ref_split_half(const T* in, T* a, T* b,
                            dim_t rows, dim_t cols) {
  const dim_t step = 2 * cols;
  for (dim_t r = 0; r < rows; ++r) {
    std::memcpy(a + r * cols, in + r * step,        cols * sizeof(T));
    std::memcpy(b + r * cols, in + r * step + cols, cols * sizeof(T));
  }
}

// Tile 1×N source T times along axis 0 → T×N output.
template <typename T>
static void ref_tile(const T* src, T* dst, dim_t N, dim_t ntiles) {
  for (dim_t t = 0; t < ntiles; ++t)
    std::memcpy(dst + t * N, src, N * sizeof(T));
}

// TopK k=1 — argmax per row.
template <typename T>
static std::pair<T, int32_t> ref_argmax(const T* row, dim_t depth) {
  int32_t best = 0;
  float best_v = static_cast<float>(row[0]);
  for (dim_t d = 1; d < depth; ++d) {
    float v = static_cast<float>(row[d]);
    if (v > best_v) { best_v = v; best = static_cast<int32_t>(d); }
  }
  return {row[best], best};
}

// Mean along the specified axis.
static float ref_mean_row(const float* src, dim_t N) {
  float s = 0.f;
  for (dim_t i = 0; i < N; ++i) s += src[i];
  return s / static_cast<float>(N);
}

// MedianFilter width=3 with reflect padding.
static void ref_median3(const float* in, float* out, dim_t N) {
  for (dim_t j = 0; j < N; ++j) {
    float w[3];
    for (int k = -1; k <= 1; ++k) {
      dim_t r = std::abs(j + k);
      if (r >= N) r = N - (r - N) - 2;
      w[k + 1] = in[r];
    }
    std::nth_element(w, w + 1, w + 3);
    out[j] = w[1];
  }
}

// ===========================================================================
// Test 1 — Memory coherency via GPU add then CPU read
// ===========================================================================
static void test_memory_coherency() {
  std::printf("\n--- Test 1: Memory coherency ---\n");

  const dim_t N = 8;
  float* a = metal_alloc<float>(N);
  float* b = metal_alloc<float>(N);
  float* c = metal_alloc<float>(N);

  for (dim_t i = 0; i < N; ++i) { a[i] = float(i); b[i] = 1.f; }

  // GPU: c = a + b  (encode-only, not yet committed)
  primitives<Device::METAL>::add(a, b, c, N);

  // commit_and_wait flushes GPU writes into shared memory
  metal::commit_and_wait();

  bool ok = true;
  for (dim_t i = 0; i < N; ++i) {
    if (std::abs(c[i] - (float(i) + 1.f)) > 1e-5f) { ok = false; break; }
  }
  CHECK("GPU write visible on CPU after commit_and_wait", ok);

  metal_free(a); metal_free(b); metal_free(c);
}

// ===========================================================================
// Tests 2–3 — Concat
// ===========================================================================
static void test_concat() {
  std::printf("\n--- Tests 2–3: Concat ---\n");

  // Test 2: Concat two float32 arrays along axis=0 (row concat)
  {
    const dim_t NA = 6, NB = 4;
    float* a   = metal_alloc<float>(NA);
    float* b   = metal_alloc<float>(NB);
    float* out = metal_alloc<float>(NA + NB);

    for (dim_t i = 0; i < NA; ++i) a[i] = float(i + 1);
    for (dim_t i = 0; i < NB; ++i) b[i] = float(NA + i + 1);

    // Simulate M7 concat (commit_and_wait already done; then memcpy)
    metal::commit_and_wait();
    ref_concat_axis0(a, b, out, NA, NB);

    bool ok = true;
    for (dim_t i = 0; i < NA + NB; ++i) {
      if (out[i] != float(i + 1)) { ok = false; break; }
    }
    CHECK("concat axis=0 float32 [6]+[4]=[10]", ok);

    metal_free(a); metal_free(b); metal_free(out);
  }

  // Test 3: Concat along axis=1 — [2×2] + [2×3] → [2×5]
  {
    const dim_t rows = 2, ca = 2, cb = 3;
    float* a   = metal_alloc<float>(rows * ca);
    float* b   = metal_alloc<float>(rows * cb);
    float* out = metal_alloc<float>(rows * (ca + cb));

    // a = [[1,2],[3,4]],  b = [[10,11,12],[13,14,15]]
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    b[0]=10; b[1]=11; b[2]=12; b[3]=13; b[4]=14; b[5]=15;

    metal::commit_and_wait();
    ref_concat_axis1(a, b, out, rows, ca, cb);

    // Expected: row0 = [1,2,10,11,12], row1 = [3,4,13,14,15]
    bool ok = (out[0]==1 && out[1]==2 && out[2]==10 && out[3]==11 && out[4]==12 &&
               out[5]==3 && out[6]==4 && out[7]==13 && out[8]==14 && out[9]==15);
    CHECK("concat axis=1 [2×2]+[2×3]=[2×5]", ok);

    metal_free(a); metal_free(b); metal_free(out);
  }
}

// ===========================================================================
// Tests 4–5 — Split and Slide
// ===========================================================================
static void test_split_slide() {
  std::printf("\n--- Tests 4–5: Split / Slide ---\n");

  // Test 4: Split [4×4] along axis=0 into two [2×4] halves.
  // axis=0 split with iter_size=1, copy_size=8 (2*4): output0 = in[0..7],
  // output1 = in[8..15] (contiguous segments, no interleaving).
  {
    const dim_t total = 16, half = 8;
    float* in = metal_alloc<float>(total);
    float* a  = metal_alloc<float>(half);
    float* b  = metal_alloc<float>(half);

    for (dim_t i = 0; i < total; ++i) in[i] = float(i);

    metal::commit_and_wait();
    // Contiguous axis-0 split: first half then second half.
    std::memcpy(a, in,          half * sizeof(float));
    std::memcpy(b, in + half,   half * sizeof(float));

    bool ok_a = true, ok_b = true;
    for (dim_t i = 0; i < half; ++i) {
      if (a[i] != float(i))        ok_a = false;
      if (b[i] != float(i + half)) ok_b = false;
    }
    CHECK("split [4×4] → 2× [2×4] first half", ok_a);
    CHECK("split [4×4] → 2× [2×4] second half", ok_b);

    metal_free(in); metal_free(a); metal_free(b);
  }

  // Test 5: Slide — extract index=1 from a [3×4] tensor along axis=0
  {
    const dim_t rows = 3, cols = 4;
    float* in  = metal_alloc<float>(rows * cols);
    float* out = metal_alloc<float>(cols);

    for (dim_t i = 0; i < rows * cols; ++i) in[i] = float(i);

    metal::commit_and_wait();
    // Slide index=1, stride_axis=cols (input.stride(0)=cols)
    std::memcpy(out, in + 1 * cols, cols * sizeof(float));

    bool ok = true;
    for (dim_t j = 0; j < cols; ++j) {
      if (out[j] != float(cols + j)) { ok = false; break; }
    }
    CHECK("slide axis=0 index=1 extracts correct row", ok);

    metal_free(in); metal_free(out);
  }
}

// ===========================================================================
// Tests 6–7 — Tile
// ===========================================================================
static void test_tile() {
  std::printf("\n--- Tests 6–7: Tile ---\n");

  // Test 6: Tile 1×N (N=5) along axis=0, 3 times → 3×5
  {
    const dim_t N = 5, ntiles = 3;
    float* src = metal_alloc<float>(N);
    float* dst = metal_alloc<float>(N * ntiles);

    for (dim_t i = 0; i < N; ++i) src[i] = float(i + 1);

    metal::commit_and_wait();
    ref_tile(src, dst, N, ntiles);

    bool ok = true;
    for (dim_t t = 0; t < ntiles; ++t)
      for (dim_t i = 0; i < N; ++i)
        if (dst[t * N + i] != float(i + 1)) { ok = false; break; }
    CHECK("tile 1×5 × 3 → 3×5 (axis=0)", ok);

    metal_free(src); metal_free(dst);
  }

  // Test 7: Tile int32 [3×2] along axis=1, 2 times → [3×4]
  {
    const dim_t rows = 3, cols = 2, ntiles = 2;
    int32_t* src = metal_alloc<int32_t>(rows * cols);
    int32_t* dst = metal_alloc<int32_t>(rows * cols * ntiles);

    for (dim_t i = 0; i < rows * cols; ++i) src[i] = int32_t(i);

    metal::commit_and_wait();
    // Tile along axis=1 means outer_size=rows, inner_size=cols
    int32_t* p = dst;
    for (dim_t r = 0; r < rows; ++r) {
      for (dim_t t = 0; t < ntiles; ++t) {
        std::memcpy(p, src + r * cols, cols * sizeof(int32_t));
        p += cols;
      }
    }

    bool ok = true;
    for (dim_t r = 0; r < rows; ++r)
      for (dim_t t = 0; t < ntiles; ++t)
        for (dim_t c = 0; c < cols; ++c)
          if (dst[r * cols * ntiles + t * cols + c] != int32_t(r * cols + c))
            ok = false;
    CHECK("tile int32 [3×2] × 2 → [3×4] (axis=1)", ok);

    metal_free(src); metal_free(dst);
  }
}

// ===========================================================================
// Tests 8–11 — TopK
// ===========================================================================
static void test_topk() {
  std::printf("\n--- Tests 8–11: TopK ---\n");

  // Test 8: k=1 float32 — argmax per row
  {
    const dim_t batch = 3, depth = 6;
    float* x   = metal_alloc<float>(batch * depth);
    float* val = metal_alloc<float>(batch * 1);
    int32_t* idx = metal_alloc<int32_t>(batch * 1);

    // Row0: [0,5,2,1,3,4] → max at idx=1 (val=5)
    // Row1: [8,2,7,3,1,6] → max at idx=0 (val=8)
    // Row2: [1,1,1,9,1,1] → max at idx=3 (val=9)
    float data[] = {0,5,2,1,3,4, 8,2,7,3,1,6, 1,1,1,9,1,1};
    for (dim_t i = 0; i < batch * depth; ++i) x[i] = data[i];

    metal::commit_and_wait();
    for (dim_t b = 0; b < batch; ++b) {
      auto [v, i] = ref_argmax(x + b * depth, depth);
      val[b] = v;
      idx[b] = i;
    }

    bool ok = (val[0]==5.f && idx[0]==1 &&
               val[1]==8.f && idx[1]==0 &&
               val[2]==9.f && idx[2]==3);
    CHECK("topk k=1 float32: argmax per row", ok);

    metal_free(x); metal_free(val); metal_free(idx);
  }

  // Test 9: k=3 float32 — partial sort
  {
    const dim_t depth = 8, k = 3;
    float* x   = metal_alloc<float>(depth);
    float* val = metal_alloc<float>(k);
    int32_t* idx = metal_alloc<int32_t>(k);

    // x = [3,1,4,1,5,9,2,6]
    // Top3 by value: 9(idx5), 6(idx7), 5(idx4)
    float data[] = {3,1,4,1,5,9,2,6};
    for (dim_t i = 0; i < depth; ++i) x[i] = data[i];

    metal::commit_and_wait();
    std::vector<int32_t> ids(depth);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
        [&x](int32_t a, int32_t b) {
          return static_cast<float>(x[a]) > static_cast<float>(x[b]);
        });
    for (dim_t j = 0; j < k; ++j) {
      idx[j] = ids[j];
      val[j] = x[idx[j]];
    }

    bool ok = (val[0]==9.f && idx[0]==5 &&
               val[1]==6.f && idx[1]==7 &&
               val[2]==5.f && idx[2]==4);
    CHECK("topk k=3 float32: correct top-3 values and indices", ok);

    metal_free(x); metal_free(val); metal_free(idx);
  }

  // Test 10: k=1 float16 — argmax
  {
    const dim_t depth = 4;
    ct2_f16* x = metal_alloc<ct2_f16>(depth);
    float vals[] = {1.f, 7.f, 3.f, 2.f};
    for (dim_t i = 0; i < depth; ++i) x[i] = ct2_f16(vals[i]);

    metal::commit_and_wait();
    auto [v, i] = ref_argmax(x, depth);
    CHECK("topk k=1 float16: argmax at index 1", i == 1);

    metal_free(x);
  }

  // Test 11: k=1 bfloat16 — argmax
  {
    const dim_t depth = 4;
    ct2_bf16* x = metal_alloc<ct2_bf16>(depth);
    float vals[] = {1.f, 2.f, 9.f, 3.f};
    for (dim_t i = 0; i < depth; ++i) x[i] = ct2_bf16(vals[i]);

    metal::commit_and_wait();
    auto [v, i] = ref_argmax(x, depth);
    CHECK("topk k=1 bfloat16: argmax at index 2", i == 2);

    metal_free(x);
  }
}

// ===========================================================================
// Test 12 — TopPMask (nucleus sampling mask)
// ===========================================================================
static void test_topp_mask() {
  std::printf("\n--- Test 12: TopPMask ---\n");

  const dim_t depth = 5;
  // Logits and corresponding uniform probs (sum=1)
  float* x    = metal_alloc<float>(depth);
  float* prob = metal_alloc<float>(depth);
  float* y    = metal_alloc<float>(depth);

  // prob = [0.1, 0.4, 0.3, 0.15, 0.05]; sorted desc: [0.4,0.3,0.15,0.1,0.05]
  // p=0.75: cumsum after 3 → 0.4+0.3+0.15=0.85 > 0.75, so top-3 kept
  // indices 1,2,3 are kept; 0 and 4 are masked
  const float logits[] = {-1.f, 2.f, 1.5f, 0.5f, -2.f};
  const float probs[]  = {0.1f, 0.4f, 0.3f, 0.15f, 0.05f};
  for (dim_t i = 0; i < depth; ++i) { x[i] = logits[i]; prob[i] = probs[i]; }

  metal::commit_and_wait();

  // Simulate TopPMask algorithm
  const float p = 0.75f;
  const float mask_val = -1e9f;
  std::vector<dim_t> ids(depth);
  std::iota(ids.begin(), ids.end(), dim_t(0));
  std::sort(ids.begin(), ids.end(), [&prob](dim_t a, dim_t b) {
    return prob[a] > prob[b];
  });
  float total_p = 0.f;
  for (const auto id : ids) {
    y[id] = total_p < p ? x[id] : mask_val;
    total_p += prob[id];
  }

  // Cumulative prob in sorted order:
  //   id=1: cum=0.00<0.75 → kept, new cum=0.40
  //   id=2: cum=0.40<0.75 → kept, new cum=0.70
  //   id=3: cum=0.70<0.75 → kept, new cum=0.85   (threshold crossed AFTER keeping)
  //   id=0: cum=0.85>=0.75 → masked
  //   id=4: cum=0.95>=0.75 → masked
  bool ok = (y[1] == logits[1] && y[2] == logits[2] && y[3] == logits[3] &&
             y[0] == mask_val  && y[4] == mask_val);
  CHECK("topp_mask p=0.75: top-3 kept, remaining masked", ok);

  metal_free(x); metal_free(prob); metal_free(y);
}

// ===========================================================================
// Tests 13–14 — Mean
// ===========================================================================
static void test_mean() {
  std::printf("\n--- Tests 13–14: Mean ---\n");

  // Test 13: Mean over last axis — [2×4] → [2]
  {
    const dim_t outer = 2, axis = 4, inner = 1;
    float* x = metal_alloc<float>(outer * axis);
    float* y = metal_alloc<float>(outer * inner);

    // row0 = [1,2,3,4], mean=2.5; row1 = [5,6,7,8], mean=6.5
    float data[] = {1,2,3,4,5,6,7,8};
    for (dim_t i = 0; i < outer * axis; ++i) x[i] = data[i];

    metal::commit_and_wait();
    for (dim_t i = 0; i < outer; ++i) {
      for (dim_t j = 0; j < inner; ++j) {
        float sum = 0.f;
        for (dim_t k = 0; k < axis; ++k)
          sum += x[i * axis * inner + k * inner + j];
        y[i * inner + j] = sum / float(axis);
      }
    }

    bool ok = (std::abs(y[0] - 2.5f) < 1e-5f &&
               std::abs(y[1] - 6.5f) < 1e-5f);
    CHECK("mean over last axis [2×4] → [2]", ok);

    metal_free(x); metal_free(y);
  }

  // Test 14: Mean over axis=0 of [4×2] → inner_size=2
  {
    const dim_t outer = 1, axis = 4, inner = 2;
    float* x = metal_alloc<float>(outer * axis * inner);
    float* y = metal_alloc<float>(outer * inner);

    // x = [[1,2],[3,4],[5,6],[7,8]], mean_col0 = (1+3+5+7)/4 = 4.0, col1 = 5.0
    float data[] = {1,2,3,4,5,6,7,8};
    for (dim_t i = 0; i < outer * axis * inner; ++i) x[i] = data[i];

    metal::commit_and_wait();
    for (dim_t i = 0; i < outer; ++i) {
      for (dim_t j = 0; j < inner; ++j) {
        float sum = 0.f;
        for (dim_t k = 0; k < axis; ++k)
          sum += x[i * axis * inner + k * inner + j];
        y[i * inner + j] = sum / float(axis);
      }
    }

    bool ok = (std::abs(y[0] - 4.0f) < 1e-5f &&
               std::abs(y[1] - 5.0f) < 1e-5f);
    CHECK("mean over axis=0 [4×2] → [2] (inner_size=2)", ok);

    metal_free(x); metal_free(y);
  }
}

// ===========================================================================
// Test 15 — MedianFilter
// ===========================================================================
static void test_median_filter() {
  std::printf("\n--- Test 15: MedianFilter ---\n");

  const dim_t N = 7;
  float* x = metal_alloc<float>(N);
  float* y = metal_alloc<float>(N);

  // x = [3, 1, 4, 1, 5, 9, 2]
  float data[] = {3, 1, 4, 1, 5, 9, 2};
  for (dim_t i = 0; i < N; ++i) x[i] = data[i];

  metal::commit_and_wait();

  float ref[7];
  ref_median3(x, ref, N);

  // Apply the same filter
  std::memcpy(y, ref, N * sizeof(float));

  // Verify a few key values (reflect-at-boundary: idx-1 → idx1):
  // j=0: window=[x[1],x[0],x[1]]=[1,3,1] → sorted=[1,1,3], median=1
  // j=1: window=[x[0],x[1],x[2]]=[3,1,4] → sorted=[1,3,4], median=3
  // j=3: window=[x[2],x[3],x[4]]=[4,1,5] → sorted=[1,4,5], median=4
  bool ok = (y[0] == 1.f && y[1] == 3.f && y[3] == 4.f);
  CHECK("median filter width=3: correct median values", ok);

  metal_free(x); metal_free(y);
}

// ===========================================================================
// Test 16 — GumbelMax noise
// ===========================================================================
static void test_gumbel_noise() {
  std::printf("\n--- Test 16: GumbelMax noise ---\n");

  const dim_t N = 1024;
  float* x = metal_alloc<float>(N);
  float* y = metal_alloc<float>(N);

  for (dim_t i = 0; i < N; ++i) x[i] = float(i);

  metal::commit_and_wait();

  // Apply Gumbel noise (same algorithm as gumbel_max_metal.mm)
  auto& generator = get_random_generator();
  std::uniform_real_distribution<float> dist(std::numeric_limits<float>::min(), 1.f);
  for (dim_t i = 0; i < N; ++i) {
    const float z = -std::log(dist(generator));
    y[i] = x[i] + z;
  }

  // Gumbel noise is always positive (-log(U(0,1)) > 0), so y[i] > x[i] always
  bool all_positive_noise = true;
  for (dim_t i = 0; i < N; ++i) {
    if (y[i] <= x[i]) { all_positive_noise = false; break; }
  }
  CHECK("gumbel noise z = -log(U(0,1)) is always positive", all_positive_noise);

  // Average noise should be ~Euler–Mascheroni constant ≈ 0.5772
  float mean_noise = 0.f;
  for (dim_t i = 0; i < N; ++i) mean_noise += (y[i] - x[i]);
  mean_noise /= float(N);
  CHECK("gumbel mean noise in [0.3, 1.5]", mean_noise > 0.3f && mean_noise < 1.5f);

  metal_free(x); metal_free(y);
}

// ===========================================================================
// Test 17 — Multinomial sampling
// ===========================================================================
static void test_multinomial() {
  std::printf("\n--- Test 17: Multinomial ---\n");

  const dim_t class_size = 10, sample_size = 100;
  float* probs = metal_alloc<float>(class_size);
  int32_t* out = metal_alloc<int32_t>(sample_size);

  // Uniform distribution
  for (dim_t i = 0; i < class_size; ++i) probs[i] = 1.f / float(class_size);

  metal::commit_and_wait();

  // Sample (same algorithm as multinomial_metal.mm)
  auto& generator = get_random_generator();
  std::vector<float> weights(probs, probs + class_size);
  std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
  for (dim_t j = 0; j < sample_size; ++j)
    out[j] = dist(generator);

  // All samples must be in [0, class_size)
  bool valid = true;
  for (dim_t j = 0; j < sample_size; ++j) {
    if (out[j] < 0 || out[j] >= int32_t(class_size)) { valid = false; break; }
  }
  CHECK("multinomial: all samples in valid range [0, class_size)", valid);

  // All classes should appear at least once in 100 samples (very likely)
  std::vector<int> counts(class_size, 0);
  for (dim_t j = 0; j < sample_size; ++j) ++counts[out[j]];
  int nonzero = 0;
  for (int c : counts) if (c > 0) ++nonzero;
  CHECK("multinomial: all classes appear in 100 uniform samples", nonzero == class_size);

  metal_free(probs); metal_free(out);
}

// ===========================================================================
// main
// ===========================================================================

int main() {
  std::printf("=== M7 Metal op tests ===\n");

  test_memory_coherency();
  test_concat();
  test_split_slide();
  test_tile();
  test_topk();
  test_topp_mask();
  test_mean();
  test_median_filter();
  test_gumbel_noise();
  test_multinomial();

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
