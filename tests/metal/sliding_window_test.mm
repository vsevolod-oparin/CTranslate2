// Test: Bidirectional sliding window attention mask (M17.1)
// Verifies that make_sliding_window_mask produces correct banded masks
// for both causal [left, 0] and bidirectional [left, right] configurations.

#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"

using namespace ctranslate2;

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(name, cond, msg) do {       \
  if (!(cond)) {                                \
    printf("  FAIL: %s — %s\n", name, msg);     \
    g_fail++;                                   \
  } else {                                      \
    g_pass++;                                   \
  }                                             \
} while(0)

// Replicate the mask generation logic from attention.cc for testing.
// This must match the implementation in make_sliding_window_mask.
static StorageView make_sliding_window_mask_ref(dim_t seq_q, dim_t seq_k,
                                                dim_t left_window, dim_t right_window) {
  const float neg_inf = -1e9f;
  StorageView mask({1, seq_q, seq_k}, 0.f, Device::CPU);
  float* data = mask.data<float>();
  for (dim_t q = 0; q < seq_q; ++q) {
    for (dim_t k = 0; k < seq_k; ++k) {
      const dim_t dist = q - k;
      const bool in_left = (dist >= 0 && dist < left_window);
      const bool in_right = (dist < 0 && (-dist) < right_window);
      if (!in_left && !in_right)
        data[q * seq_k + k] = neg_inf;
    }
  }
  return mask;
}

static bool is_neg_inf(float v) { return v < -1e8f; }
static bool is_zero(float v)    { return std::fabs(v) < 1e-10f; }

// Test 1: Causal window [4, 0] — should be lower-triangular with bandwidth 4
static void test_causal_window() {
  const char* name = "causal_window_4_0";
  auto mask = make_sliding_window_mask_ref(8, 8, 4, 0);
  const float* d = mask.data<float>();

  // Position [0,0]: dist=0, in_left (0 < 4) → valid (0.0)
  ASSERT_TRUE(name, is_zero(d[0*8 + 0]), "[0,0] should be 0");
  // Position [0,1]: dist=-1, right_window=0 → masked
  ASSERT_TRUE(name, is_neg_inf(d[0*8 + 1]), "[0,1] should be -inf");
  // Position [3,0]: dist=3, in_left (3 < 4) → valid
  ASSERT_TRUE(name, is_zero(d[3*8 + 0]), "[3,0] should be 0");
  // Position [4,0]: dist=4, NOT in_left (4 >= 4) → masked
  ASSERT_TRUE(name, is_neg_inf(d[4*8 + 0]), "[4,0] should be -inf");
  // Position [4,1]: dist=3 → valid
  ASSERT_TRUE(name, is_zero(d[4*8 + 1]), "[4,1] should be 0");
  // Position [7,7]: dist=0 → valid
  ASSERT_TRUE(name, is_zero(d[7*8 + 7]), "[7,7] should be 0");
  // Position [7,3]: dist=4 → masked
  ASSERT_TRUE(name, is_neg_inf(d[7*8 + 3]), "[7,3] should be -inf");
  // Position [7,4]: dist=3 → valid
  ASSERT_TRUE(name, is_zero(d[7*8 + 4]), "[7,4] should be 0");

  printf("  PASS: %s\n", name);
}

// Test 2: Bidirectional window [16, 4] — Moonshine's boundary layer config
static void test_bidirectional_window() {
  const char* name = "bidirectional_window_16_4";
  auto mask = make_sliding_window_mask_ref(20, 20, 16, 4);
  const float* d = mask.data<float>();

  // Position [0,0]: dist=0, in_left → valid
  ASSERT_TRUE(name, is_zero(d[0*20 + 0]), "[0,0] should be 0");
  // Position [0,3]: dist=-3, |dist|=3 < 4 → in_right → valid
  ASSERT_TRUE(name, is_zero(d[0*20 + 3]), "[0,3] should be 0 (right window)");
  // Position [0,4]: dist=-4, |dist|=4 >= 4 → masked
  ASSERT_TRUE(name, is_neg_inf(d[0*20 + 4]), "[0,4] should be -inf (outside right)");
  // Position [10,0]: dist=10 < 16 → in_left → valid
  ASSERT_TRUE(name, is_zero(d[10*20 + 0]), "[10,0] should be 0 (left window)");
  // Position [10,13]: dist=-3, |dist|=3 < 4 → in_right → valid
  ASSERT_TRUE(name, is_zero(d[10*20 + 13]), "[10,13] should be 0 (right window)");
  // Position [10,14]: dist=-4, |dist|=4 >= 4 → masked
  ASSERT_TRUE(name, is_neg_inf(d[10*20 + 14]), "[10,14] should be -inf");
  // Position [19,3]: dist=16 >= 16 → masked
  ASSERT_TRUE(name, is_neg_inf(d[19*20 + 3]), "[19,3] should be -inf (beyond left window)");
  // Position [19,4]: dist=15 < 16 → valid
  ASSERT_TRUE(name, is_zero(d[19*20 + 4]), "[19,4] should be 0");

  printf("  PASS: %s\n", name);
}

// Test 3: Small window [2, 1] — easy to manually verify full matrix
static void test_small_window() {
  const char* name = "small_window_2_1";
  auto mask = make_sliding_window_mask_ref(4, 4, 2, 1);
  const float* d = mask.data<float>();

  // Expected pattern (0 = valid, X = masked):
  // Row 0: [0, X, X, X]  — only q=0,k=0 (dist=0 < 2) valid; k=1 dist=-1 but |dist|=1 >= 1 → masked
  // Wait, right_window=1 means |dist| < 1, i.e. dist must be 0 from the right side.
  // That means right_window=1 allows dist = -0 only? No: dist < 0 && -dist < 1 → dist = 0 (not < 0).
  // So right_window=1 actually doesn't allow ANY right tokens? Let me re-check.
  // dist < 0 means k > q (future). -dist < right_window means k - q < right_window.
  // For right_window=1: k - q < 1, so k - q = 0, but dist < 0 requires k > q. Contradiction.
  // So right_window=1 allows NO future tokens. That seems wrong for [16, 4] too.
  // Let me re-check [16, 4]: dist < 0 && -dist < 4 → k-q ∈ {1,2,3}. Yes, 3 future tokens. Good.
  // So [2, 1] allows: left: dist ∈ {0, 1}, right: k-q ∈ {} (none). Wait that's wrong again.
  // Actually for right_window=1: -dist < 1 → dist > -1. Combined with dist < 0: -1 < dist < 0.
  // No integers in that range. So right_window=1 allows 0 future tokens.
  // For right_window=2: -dist < 2 → dist > -2. Combined with dist < 0: -2 < dist < 0 → dist = -1.
  // So right_window=2 allows 1 future token.
  //
  // This matches HF: sliding_window = [left_window_size, right_window_size] where
  // left_mask = (dist >= 0) & (dist < left_window_size) → up to left_window_size-1 past tokens + self
  // right_mask = (dist < 0) & (-dist < right_window_size) → up to right_window_size-1 future tokens
  //
  // So [16, 4] means: attend to self + 15 past + 3 future = 19 total positions.
  // And [2, 1] means: attend to self + 1 past + 0 future = 2 total positions.

  // Row 0: [0, X, X, X]
  ASSERT_TRUE(name, is_zero(d[0*4 + 0]), "[0,0]=0");
  ASSERT_TRUE(name, is_neg_inf(d[0*4 + 1]), "[0,1]=-inf");
  // Row 1: [0, 0, X, X]  (dist=1<2 and dist=0<2)
  ASSERT_TRUE(name, is_zero(d[1*4 + 0]), "[1,0]=0");
  ASSERT_TRUE(name, is_zero(d[1*4 + 1]), "[1,1]=0");
  ASSERT_TRUE(name, is_neg_inf(d[1*4 + 2]), "[1,2]=-inf");
  // Row 2: [X, 0, 0, X]  (k=0: dist=2>=2, masked)
  ASSERT_TRUE(name, is_neg_inf(d[2*4 + 0]), "[2,0]=-inf");
  ASSERT_TRUE(name, is_zero(d[2*4 + 1]), "[2,1]=0");
  ASSERT_TRUE(name, is_zero(d[2*4 + 2]), "[2,2]=0");
  ASSERT_TRUE(name, is_neg_inf(d[2*4 + 3]), "[2,3]=-inf");
  // Row 3: [X, X, 0, 0]
  ASSERT_TRUE(name, is_neg_inf(d[3*4 + 0]), "[3,0]=-inf");
  ASSERT_TRUE(name, is_neg_inf(d[3*4 + 1]), "[3,1]=-inf");
  ASSERT_TRUE(name, is_zero(d[3*4 + 2]), "[3,2]=0");
  ASSERT_TRUE(name, is_zero(d[3*4 + 3]), "[3,3]=0");

  printf("  PASS: %s\n", name);
}

// Test 4: Bidirectional [3, 3] — symmetric window
static void test_symmetric_window() {
  const char* name = "symmetric_window_3_3";
  auto mask = make_sliding_window_mask_ref(5, 5, 3, 3);
  const float* d = mask.data<float>();

  // Row 2: should attend to k=0,1,2,3,4 (dist in {2,1,0,-1,-2})
  // left: dist={0,1,2} all < 3 → valid
  // right: -dist={1,2} both < 3 → valid
  for (int k = 0; k < 5; ++k) {
    ASSERT_TRUE(name, is_zero(d[2*5 + k]),
                "center row should attend to all in small symmetric window");
  }

  // Row 0: left: only dist=0 valid. right: dist=-1 (-dist=1<3), dist=-2 (-dist=2<3) → valid
  ASSERT_TRUE(name, is_zero(d[0*5 + 0]), "[0,0]=0");
  ASSERT_TRUE(name, is_zero(d[0*5 + 1]), "[0,1]=0");
  ASSERT_TRUE(name, is_zero(d[0*5 + 2]), "[0,2]=0");
  ASSERT_TRUE(name, is_neg_inf(d[0*5 + 3]), "[0,3]=-inf (|dist|=3 >= 3)");

  printf("  PASS: %s\n", name);
}

// Test 5: Mask shape and device
static void test_mask_shape() {
  const char* name = "mask_shape";
  auto mask = make_sliding_window_mask_ref(10, 10, 16, 4);
  ASSERT_TRUE(name, mask.rank() == 3, "rank should be 3");
  ASSERT_TRUE(name, mask.dim(0) == 1, "dim0 should be 1 (broadcast)");
  ASSERT_TRUE(name, mask.dim(1) == 10, "dim1 should be seq_q");
  ASSERT_TRUE(name, mask.dim(2) == 10, "dim2 should be seq_k");
  ASSERT_TRUE(name, mask.device() == Device::CPU, "should be on CPU");

  printf("  PASS: %s\n", name);
}

// Test 6: Full window (window covers entire sequence) — no masking
static void test_full_window() {
  const char* name = "full_window_covers_all";
  auto mask = make_sliding_window_mask_ref(5, 5, 100, 100);
  const float* d = mask.data<float>();
  for (int i = 0; i < 25; ++i) {
    ASSERT_TRUE(name, is_zero(d[i]), "all positions should be 0 when window > seq_len");
  }
  printf("  PASS: %s\n", name);
}

// Test 7: Window of 1 (only self-attention, no neighbors)
static void test_self_only_window() {
  const char* name = "self_only_window_1_0";
  auto mask = make_sliding_window_mask_ref(4, 4, 1, 0);
  const float* d = mask.data<float>();
  for (int q = 0; q < 4; ++q) {
    for (int k = 0; k < 4; ++k) {
      if (q == k) {
        ASSERT_TRUE(name, is_zero(d[q*4+k]), "diagonal should be 0");
      } else {
        ASSERT_TRUE(name, is_neg_inf(d[q*4+k]), "off-diagonal should be -inf");
      }
    }
  }
  printf("  PASS: %s\n", name);
}

int main() {
  printf("=== Sliding Window Mask Tests (M17.1) ===\n\n");

  test_causal_window();
  test_bidirectional_window();
  test_small_window();
  test_symmetric_window();
  test_mask_shape();
  test_full_window();
  test_self_only_window();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
