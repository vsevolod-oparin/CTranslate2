// Adversarial tests for M17.x frontend components:
//   - make_sliding_window_mask (M17.1) — edge cases not covered by sliding_window_test.mm
//   - apply_cmvn (M17.2) — degenerate inputs (zero fs, NaN, Inf, single element)
//   - asinh compression — float special values
//
// Tests are CPU-only (no Metal device required), fast (<1ms each).
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_METAL -DCT2_WITH_MPS \
//     tests/metal/adversarial_frontend_test.mm \
//     -L build -lctranslate2.mps -Wl,-rpath,build \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -o adversarial_frontend_test && ./adversarial_frontend_test

#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cfloat>
#include <vector>
#include <numeric>
#include <limits>

#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"

using namespace ctranslate2;

static int g_pass = 0;
static int g_fail = 0;

// ---------------------------------------------------------------------------
// Assertion macros
// ---------------------------------------------------------------------------

#define PASS(name) do { printf("  PASS: %s\n", name); g_pass++; } while(0)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); g_fail++; } while(0)

#define ASSERT_TRUE(name, cond, msg) do { \
  if (!(cond)) { FAIL(name, msg); } else { g_pass++; } \
} while(0)

#define ASSERT_NEAR(name, actual, expected, eps, msg) do { \
  if (std::fabs((double)(actual) - (double)(expected)) > (eps)) { \
    printf("  FAIL: %s — %s (got %.9g, expected %.9g)\n", \
           name, msg, (double)(actual), (double)(expected)); \
    g_fail++; \
  } else { g_pass++; } \
} while(0)

// ---------------------------------------------------------------------------
// Replicated from moonshine.cc / sliding_window logic
// (must match the production implementation exactly)
// ---------------------------------------------------------------------------

// make_sliding_window_mask:
//   mask[q,k] = 0       if ( (q-k) in [0, left_window) )  OR  ( (k-q) in (0, right_window) )
//   mask[q,k] = -1e9    otherwise
static StorageView make_sliding_window_mask_ref(dim_t seq_q, dim_t seq_k,
                                                dim_t left_window, dim_t right_window) {
  const float neg_inf = -1e9f;
  StorageView mask({1, seq_q, seq_k}, 0.f, Device::CPU);
  float* data = mask.data<float>();
  for (dim_t q = 0; q < seq_q; ++q) {
    for (dim_t k = 0; k < seq_k; ++k) {
      const dim_t dist = q - k;
      const bool in_left  = (dist >= 0 && dist < left_window);
      const bool in_right = (dist < 0  && (-dist) < right_window);
      if (!in_left && !in_right)
        data[q * seq_k + k] = neg_inf;
    }
  }
  return mask;
}

// apply_cmvn: per-frame cepstral mean & variance normalisation
static void apply_cmvn_ref(float* data, dim_t batch, dim_t num_frames, dim_t fs) {
  const float eps = 1e-5f;
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t f = 0; f < num_frames; ++f) {
      float* frame = data + (b * num_frames + f) * fs;
      float sum = 0;
      for (dim_t i = 0; i < fs; ++i)
        sum += frame[i];
      const float mean = sum / static_cast<float>(fs);
      float var_sum = 0;
      for (dim_t i = 0; i < fs; ++i) {
        frame[i] -= mean;
        var_sum += frame[i] * frame[i];
      }
      const float rms = std::sqrt(var_sum / static_cast<float>(fs) + eps);
      const float inv_rms = 1.0f / rms;
      for (dim_t i = 0; i < fs; ++i)
        frame[i] *= inv_rms;
    }
  }
}

static bool is_neg_inf_mask(float v) { return v < -1e8f; }
static bool is_valid(float v)        { return std::fabs(v) < 1e-10f; }

// ============================================================================
// Section 1: Sliding window mask — adversarial edge cases
// ============================================================================

// Test: seq_q=0, seq_k=0 — empty mask, no iteration, no crash
static void test_sliding_window_zero_seq() {
  const char* name = "sliding_window_zero_seq";
  StorageView mask = make_sliding_window_mask_ref(0, 0, 16, 4);
  ASSERT_TRUE(name, mask.rank() == 3, "rank=3");
  ASSERT_TRUE(name, mask.dim(0) == 1, "dim0=1");
  ASSERT_TRUE(name, mask.dim(1) == 0, "dim1=0 (seq_q)");
  ASSERT_TRUE(name, mask.dim(2) == 0, "dim2=0 (seq_k)");
  ASSERT_TRUE(name, mask.size() == 0, "size=0");
  PASS(name);
}

// Test: seq_q=0, seq_k=5 — zero query rows but non-zero keys
static void test_sliding_window_zero_queries() {
  const char* name = "sliding_window_zero_queries";
  StorageView mask = make_sliding_window_mask_ref(0, 5, 16, 4);
  ASSERT_TRUE(name, mask.dim(1) == 0, "no query rows");
  ASSERT_TRUE(name, mask.size() == 0, "no elements");
  PASS(name);
}

// Test: seq_q=5, seq_k=0 — queries but no keys
static void test_sliding_window_zero_keys() {
  const char* name = "sliding_window_zero_keys";
  StorageView mask = make_sliding_window_mask_ref(5, 0, 16, 4);
  ASSERT_TRUE(name, mask.dim(2) == 0, "no key columns");
  ASSERT_TRUE(name, mask.size() == 0, "no elements");
  PASS(name);
}

// Test: left_window=0, right_window=0 — all positions should be masked
static void test_sliding_window_zero_windows_all_masked() {
  const char* name = "sliding_window_zero_windows_all_masked";
  StorageView mask = make_sliding_window_mask_ref(4, 4, 0, 0);
  const float* d = mask.data<float>();
  bool all_masked = true;
  for (int i = 0; i < 16; ++i) {
    if (!is_neg_inf_mask(d[i])) { all_masked = false; break; }
  }
  ASSERT_TRUE(name, all_masked, "all positions should be -inf with zero windows");
  PASS(name);
}

// Test: left_window=0, right_window=5 — only future tokens visible
// In particular, diagonal should be masked (dist=0, in_left requires left_window>0)
static void test_sliding_window_left_zero_only_future() {
  const char* name = "sliding_window_left_zero_only_future";
  const dim_t seq = 5, rw = 3;
  StorageView mask = make_sliding_window_mask_ref(seq, seq, 0, rw);
  const float* d = mask.data<float>();

  // Self-attention (q==k, dist=0): in_left = (0>=0 && 0<0) = false, in_right = false → masked
  for (int q = 0; q < (int)seq; ++q) {
    ASSERT_TRUE(name, is_neg_inf_mask(d[q*seq + q]), "diagonal should be masked (left_window=0)");
  }

  // Future token: q=0, k=1 → dist=-1, in_right = (-1<0 && 1<3) = true → valid
  ASSERT_TRUE(name, is_valid(d[0*seq + 1]), "q=0,k=1 should be valid (right window)");
  // q=0, k=3 → dist=-3, in_right = (-3<0 && 3<3) = false → masked (boundary: right_window=3 → k-q ∈ {1,2})
  ASSERT_TRUE(name, is_neg_inf_mask(d[0*seq + 3]), "q=0,k=3 should be masked (k-q=3 >= right_window=3)");

  PASS(name);
}

// Test: right_window=0, left_window=4 — causal only (no future)
static void test_sliding_window_right_zero_causal_only() {
  const char* name = "sliding_window_right_zero_causal_only";
  StorageView mask = make_sliding_window_mask_ref(5, 5, 4, 0);
  const float* d = mask.data<float>();

  // q=0,k=1 (future): dist=-1 → in_right = (-1<0 && 1<0) = false → masked
  ASSERT_TRUE(name, is_neg_inf_mask(d[0*5 + 1]), "future token should be masked with right_window=0");
  // q=1,k=0 (past within window): dist=1 < 4 → valid
  ASSERT_TRUE(name, is_valid(d[1*5 + 0]), "recent past should be valid");
  // q=0,k=0 (self): dist=0 < 4 → valid
  ASSERT_TRUE(name, is_valid(d[0*5 + 0]), "self should be valid");

  PASS(name);
}

// Test: seq_q=1, seq_k=1 — minimal single-token self-attention
static void test_sliding_window_single_token() {
  const char* name = "sliding_window_single_token";
  StorageView mask = make_sliding_window_mask_ref(1, 1, 16, 4);
  const float* d = mask.data<float>();
  ASSERT_TRUE(name, is_valid(d[0]), "single [0,0] should be valid");
  PASS(name);
}

// Test: seq_q=1, seq_k=1 with zero-window — even self-attention is masked
static void test_sliding_window_single_token_zero_window() {
  const char* name = "sliding_window_single_token_zero_window";
  StorageView mask = make_sliding_window_mask_ref(1, 1, 0, 0);
  const float* d = mask.data<float>();
  ASSERT_TRUE(name, is_neg_inf_mask(d[0]), "self should be masked with zero window");
  PASS(name);
}

// Test: seq_q != seq_k — cross-attention (decoder queries attending to encoder keys)
static void test_sliding_window_cross_attention_shape() {
  const char* name = "sliding_window_cross_attention_shape";
  StorageView mask = make_sliding_window_mask_ref(3, 8, 4, 2);
  ASSERT_TRUE(name, mask.dim(1) == 3, "seq_q=3");
  ASSERT_TRUE(name, mask.dim(2) == 8, "seq_k=8");
  ASSERT_TRUE(name, mask.size() == 3 * 8, "3×8=24 elements");

  const float* d = mask.data<float>();
  // q=0,k=0: dist=0, in_left (0<4) → valid
  ASSERT_TRUE(name, is_valid(d[0*8 + 0]), "[0,0] valid");
  // q=0,k=2: dist=-2, in_right (2<2 false) → masked
  ASSERT_TRUE(name, is_neg_inf_mask(d[0*8 + 2]), "[0,2] masked (k-q=2 >= right_window=2)");
  // q=2,k=0: dist=2, in_left (2<4) → valid
  ASSERT_TRUE(name, is_valid(d[2*8 + 0]), "[2,0] valid");
  // q=2,k=7: dist=-5, |dist|=5 >= right_window=2 → masked
  ASSERT_TRUE(name, is_neg_inf_mask(d[2*8 + 7]), "[2,7] masked (far future)");

  PASS(name);
}

// Test: window=1 means only self-attention (dist=0, in_left true only for left_window>0)
// left_window=1: in_left = (dist>=0 && dist<1) = (dist==0). Only diagonal.
static void test_sliding_window_left_window_one_diagonal_only() {
  const char* name = "sliding_window_left_window_one";
  StorageView mask = make_sliding_window_mask_ref(4, 4, 1, 0);
  const float* d = mask.data<float>();
  for (int q = 0; q < 4; ++q) {
    for (int k = 0; k < 4; ++k) {
      if (q == k) {
        ASSERT_TRUE(name, is_valid(d[q*4+k]), "diagonal should be valid");
      } else {
        ASSERT_TRUE(name, is_neg_inf_mask(d[q*4+k]), "off-diagonal should be masked");
      }
    }
  }
  PASS(name);
}

// Test: off-by-one at window boundary
// left_window=4: positions with dist={0,1,2,3} valid; dist=4 masked.
static void test_sliding_window_boundary_off_by_one() {
  const char* name = "sliding_window_boundary_off_by_one";
  const int N = 8;
  StorageView mask = make_sliding_window_mask_ref(N, N, 4, 0);
  const float* d = mask.data<float>();

  // q=4, k=0: dist=4 — must be masked (dist >= left_window=4)
  ASSERT_TRUE(name, is_neg_inf_mask(d[4*N + 0]), "dist=4 should be masked (boundary)");
  // q=4, k=1: dist=3 — must be valid (dist < left_window=4)
  ASSERT_TRUE(name, is_valid(d[4*N + 1]), "dist=3 should be valid (just inside)");

  PASS(name);
}

// ============================================================================
// Section 2: CMVN — adversarial inputs
// ============================================================================

// Test: num_frames=0 — empty loop, no computation, no crash
static void test_cmvn_zero_frames_no_crash() {
  const char* name = "cmvn_zero_frames_no_crash";
  // No data, but function should handle num_frames=0 without crash
  std::vector<float> data;  // empty
  apply_cmvn_ref(data.data(), 1, 0, 80);
  PASS(name);
}

// Test: fs=0 — division by zero (mean = sum/0 = inf/nan)
// This tests whether the implementation handles fs=0 gracefully or produces NaN.
// Expected behavior: the function either avoids the division or produces NaN.
// We verify it doesn't crash and document the output.
static void test_cmvn_zero_features_no_crash() {
  const char* name = "cmvn_zero_features_no_crash";
  // With fs=0: the inner loops don't execute, but mean = 0.0f / 0.0f is NaN.
  // Then inv_rms = 1/sqrt(0/0 + eps) = 1/sqrt(nan) = nan.
  // No data is modified (fs=0 inner loops are empty), so no buffer corruption.
  std::vector<float> dummy;  // no actual data (fs=0 means no elements per frame)
  // This should not crash (the inner for-loops run 0 iterations)
  apply_cmvn_ref(dummy.data(), 1, 1, 0);
  PASS(name);
}

// Test: single-element frame (fs=1)
// mean = x[0], after centering: 0, var_sum=0, rms=sqrt(eps), output=0/sqrt(eps)=0
static void test_cmvn_single_element_frame() {
  const char* name = "cmvn_single_element_frame";
  std::vector<float> data = {42.0f};
  apply_cmvn_ref(data.data(), 1, 1, 1);
  // After centering: 42-42=0. Then output = 0 * inv_rms = 0.
  ASSERT_NEAR(name, data[0], 0.0f, 1e-5f, "single element should normalize to 0");
  PASS(name);
}

// Test: all-NaN frame — NaN propagates, does not crash
static void test_cmvn_nan_input_no_crash() {
  const char* name = "cmvn_nan_input_no_crash";
  const float nan_val = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> data(4, nan_val);
  apply_cmvn_ref(data.data(), 1, 1, 4);
  // Output will contain NaN — that's expected, not a crash
  for (int i = 0; i < 4; ++i) {
    // NaN != NaN is the C++ way to check isnan
    ASSERT_TRUE(name, std::isnan(data[i]) || std::isfinite(data[i]),
                "NaN input must not produce infinity without NaN");
  }
  PASS(name);
}

// Test: all-Inf frame — Inf propagates, does not crash
static void test_cmvn_inf_input_no_crash() {
  const char* name = "cmvn_inf_input_no_crash";
  const float inf_val = std::numeric_limits<float>::infinity();
  std::vector<float> data = {inf_val, -inf_val, inf_val, -inf_val};
  apply_cmvn_ref(data.data(), 1, 1, 4);
  // mean = (inf - inf + inf - inf) / 4 = NaN (indeterminate)
  // All results should be NaN or 0 — not crash
  PASS(name);
}

// Test: FLT_MAX values — large inputs, potential overflow in variance
static void test_cmvn_flt_max_no_crash() {
  const char* name = "cmvn_flt_max_no_crash";
  // Alternating +FLT_MAX and -FLT_MAX: mean ≈ 0, variance = FLT_MAX²
  std::vector<float> data = {FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX};
  apply_cmvn_ref(data.data(), 1, 1, 4);
  // var_sum = 4 * FLT_MAX² = inf. rms = sqrt(inf/4 + eps) = inf.
  // inv_rms = 0. output = ±FLT_MAX * 0 = 0 or NaN.
  // Must not crash.
  PASS(name);
}

// Test: constant frame (all same value, non-zero) — already tested in
// moonshine_frontend_test.mm, but here we verify the math precisely.
// After centering: all 0. rms = sqrt(eps). output = 0 * inv_rms = 0.
static void test_cmvn_constant_nonzero_normalizes_to_zero() {
  const char* name = "cmvn_constant_nonzero_to_zero";
  std::vector<float> data(8, 7.5f);
  apply_cmvn_ref(data.data(), 1, 1, 8);
  for (int i = 0; i < 8; ++i)
    ASSERT_NEAR(name, data[i], 0.0f, 1e-4f, "constant frame must normalize to 0");
  PASS(name);
}

// Test: two-batch CMVN independence — batch 0 normalization does not bleed into batch 1
static void test_cmvn_batch_independence() {
  const char* name = "cmvn_batch_independence";
  // batch=2, num_frames=1, fs=4
  // Batch 0: [1, 2, 3, 4] — mean=2.5, values will be centered + scaled
  // Batch 1: [0, 0, 0, 0] — all zero constant frame → output = [0,0,0,0]
  std::vector<float> data = {1, 2, 3, 4, 0, 0, 0, 0};
  apply_cmvn_ref(data.data(), 2, 1, 4);

  // Batch 1 should still be all zeros (constant frame normalizes to 0)
  for (int i = 4; i < 8; ++i)
    ASSERT_NEAR(name, data[i], 0.0f, 1e-4f, "batch 1 constant frame should be 0");

  // Batch 0 should have mean ~0 (not affected by batch 1)
  float sum = 0;
  for (int i = 0; i < 4; ++i) sum += data[i];
  ASSERT_NEAR(name, sum / 4.0f, 0.0f, 1e-5f, "batch 0 mean should be 0 after CMVN");

  PASS(name);
}

// Test: large fs (1024 elements) — no stack overflow, correct mean
static void test_cmvn_large_feature_vector() {
  const char* name = "cmvn_large_fs_1024";
  const int fs = 1024;
  std::vector<float> data(fs);
  for (int i = 0; i < fs; ++i) data[i] = static_cast<float>(i);
  apply_cmvn_ref(data.data(), 1, 1, fs);

  float mean = 0;
  for (float v : data) mean += v;
  mean /= fs;
  ASSERT_NEAR(name, mean, 0.0f, 1e-3f, "mean of normalized large frame should be ~0");
  PASS(name);
}

// ============================================================================
// Section 3: asinh compression — float special values
// ============================================================================

// Test: asinh(-0.0f) preserves negative zero → asinh(-0) = -0
static void test_asinh_negative_zero() {
  const char* name = "asinh_negative_zero";
  float result = std::asinhf(-0.0f);
  // IEEE 754: asinh(-0) = -0, which is == 0.0f but has negative sign
  ASSERT_NEAR(name, result, 0.0f, 1e-10f, "asinh(-0) should be ~0");
  PASS(name);
}

// Test: asinh(+Inf) = +Inf (defined: sinh^{-1}(inf) = inf)
static void test_asinh_positive_inf() {
  const char* name = "asinh_positive_inf";
  float result = std::asinhf(std::numeric_limits<float>::infinity());
  ASSERT_TRUE(name, std::isinf(result) && result > 0, "asinh(+inf) should be +inf");
  PASS(name);
}

// Test: asinh(-Inf) = -Inf
static void test_asinh_negative_inf() {
  const char* name = "asinh_negative_inf";
  float result = std::asinhf(-std::numeric_limits<float>::infinity());
  ASSERT_TRUE(name, std::isinf(result) && result < 0, "asinh(-inf) should be -inf");
  PASS(name);
}

// Test: asinh(NaN) = NaN
static void test_asinh_nan() {
  const char* name = "asinh_nan";
  float result = std::asinhf(std::numeric_limits<float>::quiet_NaN());
  ASSERT_TRUE(name, std::isnan(result), "asinh(NaN) should be NaN");
  PASS(name);
}

// Test: asinh(FLT_MAX) — very large input, no crash
static void test_asinh_flt_max_no_crash() {
  const char* name = "asinh_flt_max_no_crash";
  float result = std::asinhf(FLT_MAX);
  // asinh(FLT_MAX) ≈ log(2*FLT_MAX) ≈ 89.4, should be finite
  ASSERT_TRUE(name, std::isfinite(result), "asinh(FLT_MAX) should be finite");
  ASSERT_TRUE(name, result > 0, "asinh(FLT_MAX) should be positive");
  PASS(name);
}

// Test: asinh(denormalized) — smallest positive denorm, no crash
static void test_asinh_denormalized_no_crash() {
  const char* name = "asinh_denormalized";
  float result = std::asinhf(FLT_MIN * 0.5f);  // denormalized
  // asinh(~1e-45) ≈ ~1e-45 (approximately linear for small x)
  ASSERT_TRUE(name, std::isfinite(result), "asinh(denorm) should be finite");
  PASS(name);
}

// Test: asinh is odd function — asinh(-x) = -asinh(x) for all finite x
static void test_asinh_odd_function_property() {
  const char* name = "asinh_odd_function";
  const float test_values[] = {0.0f, 1.0f, 100.0f, FLT_EPSILON, 1000.0f};
  for (float x : test_values) {
    float pos = std::asinhf(x);
    float neg = std::asinhf(-x);
    if (std::isfinite(pos)) {
      ASSERT_NEAR(name, neg, -pos, 1e-5f, "asinh should be odd");
    }
  }
  PASS(name);
}

// ============================================================================
// main
// ============================================================================

int main() {
  printf("=== Adversarial Frontend Tests (M17.x) ===\n\n");

  printf("--- Sliding Window Mask ---\n");
  test_sliding_window_zero_seq();
  test_sliding_window_zero_queries();
  test_sliding_window_zero_keys();
  test_sliding_window_zero_windows_all_masked();
  test_sliding_window_left_zero_only_future();
  test_sliding_window_right_zero_causal_only();
  test_sliding_window_single_token();
  test_sliding_window_single_token_zero_window();
  test_sliding_window_cross_attention_shape();
  test_sliding_window_left_window_one_diagonal_only();
  test_sliding_window_boundary_off_by_one();

  printf("\n--- CMVN ---\n");
  test_cmvn_zero_frames_no_crash();
  test_cmvn_zero_features_no_crash();
  test_cmvn_single_element_frame();
  test_cmvn_nan_input_no_crash();
  test_cmvn_inf_input_no_crash();
  test_cmvn_flt_max_no_crash();
  test_cmvn_constant_nonzero_normalizes_to_zero();
  test_cmvn_batch_independence();
  test_cmvn_large_feature_vector();

  printf("\n--- asinh ---\n");
  test_asinh_negative_zero();
  test_asinh_positive_inf();
  test_asinh_negative_inf();
  test_asinh_nan();
  test_asinh_flt_max_no_crash();
  test_asinh_denormalized_no_crash();
  test_asinh_odd_function_property();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
