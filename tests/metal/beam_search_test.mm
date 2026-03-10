// Correctness tests for M4.7 Metal beam-search primitives.
//
// Tests: penalize_previous_tokens (float32, float16, bfloat16),
//        prepare_length_mask (padded batch, causal, multi-query),
//        at() (flush-before-read after GPU write),
//        logsumexp (verify the M4.5 implementation for M4.7 plan acceptance).
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/beam_search_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o beam_search_test && ./beam_search_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// Declare typedef BEFORE any `using namespace ctranslate2` to avoid the
// float16_t name conflict with arm_vector_types.h (pulled in by Metal.h).
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("  FAIL: %s\n", name.c_str());
  }
}

static bool near(float a, float b, float tol = 1e-3f) {
  return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// Read a Metal buffer element (always uses primitives::at which flushes GPU).
template <typename T>
static float read_metal(const T* ptr, dim_t idx) {
  return static_cast<float>(primitives<Device::MPS>::at(ptr, idx));
}

// ---------------------------------------------------------------------------
// penalize_previous_tokens tests
// ---------------------------------------------------------------------------
//
// CPU reference: for each batch i, for each position j:
//   write_idx = i * vocab + previous_ids[i*len + j]
//   score     = previous_scores[i*len + j]
//   scores[write_idx] = score < 0 ? score * penalty : score / penalty
//
// Verify: GPU matches CPU for batch=2, len=8, vocab=100, penalty=1.5

template <typename T>
static void test_penalize_basic(const std::string& tname) {
  const dim_t batch = 2, len = 8, vocab = 100;
  const float penalty = 1.5f;

  // Build CPU reference data.
  std::vector<float> scores_init(batch * vocab, 0.f);
  // Give each score a non-zero value so we can detect changes.
  for (dim_t i = 0; i < batch * vocab; ++i)
    scores_init[i] = (float)(i % 7) - 3.f;  // some negative, some positive

  std::vector<float> prev_scores_f(batch * len);
  std::vector<int32_t> prev_ids(batch * len);
  for (dim_t i = 0; i < batch; ++i) {
    for (dim_t j = 0; j < len; ++j) {
      // prev_scores: mix of positive and negative
      prev_scores_f[i * len + j] = (float)(j % 3) - 1.f;
      // prev_ids: unique per position (avoid conflicts to keep test deterministic)
      prev_ids[i * len + j] = (int32_t)((i * 13 + j * 7) % vocab);
    }
  }

  // CPU reference
  std::vector<float> cpu_scores = scores_init;
  for (dim_t i = 0; i < batch; ++i) {
    for (dim_t j = 0; j < len; ++j) {
      dim_t read_idx  = i * len + j;
      dim_t write_idx = i * vocab + prev_ids[read_idx];
      float s = prev_scores_f[read_idx];
      cpu_scores[write_idx] = (s < 0.f) ? s * penalty : s / penalty;
    }
  }

  // GPU
  T* d_scores         = metal_alloc<T>(batch * vocab);
  T* d_prev_scores    = metal_alloc<T>(batch * len);
  int32_t* d_prev_ids = metal_alloc<int32_t>(batch * len);

  // Initialise from CPU reference values.
  for (dim_t i = 0; i < batch * vocab; ++i)
    d_scores[i] = (T)scores_init[i];
  for (dim_t i = 0; i < batch * len; ++i)
    d_prev_scores[i] = (T)prev_scores_f[i];
  for (dim_t i = 0; i < batch * len; ++i)
    d_prev_ids[i] = prev_ids[i];

  primitives<Device::MPS>::penalize_previous_tokens(
      d_scores, d_prev_scores, d_prev_ids,
      (T)penalty, batch, len, vocab);
  metal::commit_and_wait();  // flush GPU before reading back

  bool all_ok = true;
  for (dim_t i = 0; i < batch * vocab; ++i) {
    float got = (float)d_scores[i];
    float ref = cpu_scores[i];
    if (!near(got, ref, 2e-3f)) {
      std::printf("    [%s] mismatch at i=%lld: got=%g ref=%g\n",
                  tname.c_str(), (long long)i, (double)got, (double)ref);
      all_ok = false;
      break;
    }
  }
  check(all_ok, "penalize_basic<" + tname + ">");

  metal_free(d_scores);
  metal_free(d_prev_scores);
  metal_free(d_prev_ids);
}

// Zero-size edge cases.
template <typename T>
static void test_penalize_zero_size(const std::string& tname) {
  // batch=0
  primitives<Device::MPS>::penalize_previous_tokens(
      (T*)nullptr, (T*)nullptr, (int32_t*)nullptr, (T)1.5f, 0, 8, 100);
  check(true, "penalize_batch0<" + tname + ">");

  // len=0
  primitives<Device::MPS>::penalize_previous_tokens(
      (T*)nullptr, (T*)nullptr, (int32_t*)nullptr, (T)1.5f, 4, 0, 100);
  check(true, "penalize_len0<" + tname + ">");
}

// Test with duplicate token IDs — last write wins (same as CPU semantics).
template <typename T>
static void test_penalize_duplicates(const std::string& tname) {
  // batch=1, len=4, vocab=10 — token_id=5 appears at positions 0 and 2
  const dim_t batch = 1, len = 4, vocab = 10;
  const float penalty = 2.f;

  std::vector<float> scores_init(vocab, 0.f);
  // Distinct initial values so we can tell which write happened
  for (dim_t i = 0; i < vocab; ++i) scores_init[i] = (float)i;

  // prev_ids: [5, 3, 5, 7]  — token 5 appears twice
  int32_t prev_ids_h[4] = {5, 3, 5, 7};
  // prev_scores: [-1.0, 2.0, -3.0, 4.0]
  float prev_scores_h[4] = {-1.f, 2.f, -3.f, 4.f};

  // CPU reference (sequential)
  std::vector<float> cpu_scores = scores_init;
  for (dim_t j = 0; j < len; ++j) {
    dim_t write_idx = prev_ids_h[j];
    float s = prev_scores_h[j];
    cpu_scores[write_idx] = (s < 0.f) ? s * penalty : s / penalty;
  }
  // cpu_scores[5]: first set by j=0 (score=-1 → -1*2=-2),
  //               overwritten by j=2 (score=-3 → -3*2=-6)  → expect -6

  T* d_scores      = metal_alloc<T>(vocab);
  T* d_prev_scores = metal_alloc<T>(len);
  int32_t* d_ids   = metal_alloc<int32_t>(len);

  for (dim_t i = 0; i < vocab; ++i) d_scores[i] = (T)scores_init[i];
  for (dim_t i = 0; i < len;   ++i) d_prev_scores[i] = (T)prev_scores_h[i];
  for (dim_t i = 0; i < len;   ++i) d_ids[i] = prev_ids_h[i];

  primitives<Device::MPS>::penalize_previous_tokens(
      d_scores, d_prev_scores, d_ids, (T)penalty, batch, len, vocab);
  metal::commit_and_wait();

  bool ok = near((float)d_scores[5], cpu_scores[5], 2e-3f);
  check(ok, "penalize_duplicates<" + tname + ">");

  metal_free(d_scores);
  metal_free(d_prev_scores);
  metal_free(d_ids);
}

// ---------------------------------------------------------------------------
// prepare_length_mask tests
// ---------------------------------------------------------------------------
//
// Semantics (matches CPU):
//   if !mask_future: mask[b][h][q] = lengths[b]
//   if  mask_future, !multi_query: mask[b][h][q] = min(lengths[b], q+1)
//   if  mask_future,  multi_query: mask[b][h][q] = min(lengths[b], (h)+1)
//     where position = multi_query ? i/num_heads : i%num_queries

static void test_prepare_mask_padded() {
  // lengths=[3,5], batch=2, heads=2, queries=5, mask_future=false
  const dim_t batch=2, heads=2, queries=5;

  int32_t* d_lengths = metal_alloc<int32_t>(batch);
  int32_t* d_mask    = metal_alloc<int32_t>(batch * heads * queries);

  d_lengths[0] = 3;
  d_lengths[1] = 5;
  std::memset(d_mask, 0, batch * heads * queries * sizeof(int32_t));

  primitives<Device::MPS>::prepare_length_mask(
      d_lengths, batch, heads, queries, /*mask_future=*/false,
      /*multi_query=*/false, d_mask);
  // No GPU work needed — CPU fills mask directly.

  bool ok = true;
  // batch 0: all entries should be 3
  for (dim_t i = 0; i < heads * queries; ++i) {
    if (d_mask[i] != 3) { ok = false; break; }
  }
  // batch 1: all entries should be 5
  for (dim_t i = 0; i < heads * queries; ++i) {
    if (d_mask[heads * queries + i] != 5) { ok = false; break; }
  }
  check(ok, "prepare_mask_padded");

  metal_free(d_lengths);
  metal_free(d_mask);
}

static void test_prepare_mask_causal() {
  // lengths=[4], batch=1, heads=1, queries=4, mask_future=true, multi_query=false
  const dim_t batch=1, heads=1, queries=4;

  int32_t* d_lengths = metal_alloc<int32_t>(batch);
  int32_t* d_mask    = metal_alloc<int32_t>(batch * heads * queries);

  d_lengths[0] = 4;

  primitives<Device::MPS>::prepare_length_mask(
      d_lengths, batch, heads, queries, /*mask_future=*/true,
      /*multi_query=*/false, d_mask);

  // Expected: mask[q] = min(4, q+1) = 1,2,3,4
  bool ok = (d_mask[0]==1 && d_mask[1]==2 && d_mask[2]==3 && d_mask[3]==4);
  check(ok, "prepare_mask_causal");

  metal_free(d_lengths);
  metal_free(d_mask);
}

static void test_prepare_mask_causal_padded() {
  // lengths=[3,5], batch=2, heads=2, queries=5, mask_future=true, multi_query=false
  // For query position q (= i % num_queries): mask = min(length, q+1)
  const dim_t batch=2, heads=2, queries=5;

  int32_t* d_lengths = metal_alloc<int32_t>(batch);
  int32_t* d_mask    = metal_alloc<int32_t>(batch * heads * queries);

  d_lengths[0] = 3;
  d_lengths[1] = 5;

  primitives<Device::MPS>::prepare_length_mask(
      d_lengths, batch, heads, queries, /*mask_future=*/true,
      /*multi_query=*/false, d_mask);

  // Compute CPU reference
  std::vector<int32_t> ref(batch * heads * queries);
  for (dim_t b = 0; b < batch; ++b) {
    int32_t length = d_lengths[b];
    for (dim_t i = 0; i < heads * queries; ++i) {
      // multi_query=false → query index = i % num_queries
      ref[b * heads * queries + i] =
          std::min(length, (int32_t)(i % queries + 1));
    }
  }

  bool ok = true;
  for (dim_t i = 0; i < batch * heads * queries; ++i) {
    if (d_mask[i] != ref[i]) { ok = false; break; }
  }
  check(ok, "prepare_mask_causal_padded");

  metal_free(d_lengths);
  metal_free(d_mask);
}

static void test_prepare_mask_multi_query() {
  // multi_query=true: query index = i / num_heads
  const dim_t batch=1, heads=2, queries=3;

  int32_t* d_lengths = metal_alloc<int32_t>(batch);
  int32_t* d_mask    = metal_alloc<int32_t>(batch * heads * queries);

  d_lengths[0] = 10;  // no truncation from length

  primitives<Device::MPS>::prepare_length_mask(
      d_lengths, batch, heads, queries, /*mask_future=*/true,
      /*multi_query=*/true, d_mask);

  // ref: for i in [0, heads*queries): mask[i] = min(10, i/num_heads + 1)
  // heads=2, queries=3 → 6 elements
  // i=0: i/2+1=1, i=1: 1, i=2: 2, i=3: 2, i=4: 3, i=5: 3
  int32_t expected[] = {1, 1, 2, 2, 3, 3};
  bool ok = true;
  for (dim_t i = 0; i < (dim_t)(heads * queries); ++i) {
    if (d_mask[i] != expected[i]) { ok = false; break; }
  }
  check(ok, "prepare_mask_multi_query");

  metal_free(d_lengths);
  metal_free(d_mask);
}

// ---------------------------------------------------------------------------
// at() test — verify flush-before-read after a GPU write
// ---------------------------------------------------------------------------

static void test_at_after_gpu_write() {
  // Write a value via a GPU kernel (add_scalar), then read it back with at().
  const dim_t n = 4;
  float* d = metal_alloc<float>(n);
  for (dim_t i = 0; i < n; ++i) d[i] = 0.f;

  // Use add(scalar, x, y) GPU kernel to write 42.0 into all elements.
  primitives<Device::MPS>::add(42.f, d, d, n);  // GPU encodes this

  // at() must flush GPU before reading — should return 42.0.
  float v = primitives<Device::MPS>::at(d, 2);
  check(near(v, 42.f, 1e-5f), "at_after_gpu_write");

  metal_free(d);
}

// ---------------------------------------------------------------------------
// logsumexp test — verify M4.5 implementation for M4.7 plan acceptance
// ---------------------------------------------------------------------------

static void test_logsumexp_float() {
  const dim_t n = 3;
  float* d = metal_alloc<float>(n);
  d[0] = 1.f; d[1] = 2.f; d[2] = 3.f;

  float result = primitives<Device::MPS>::logsumexp(d, n);
  // Reference: log(exp(1) + exp(2) + exp(3)) ≈ 3.4076
  float ref = std::log(std::exp(1.f) + std::exp(2.f) + std::exp(3.f));
  check(near(result, ref, 1e-5f), "logsumexp_float");

  metal_free(d);
}

static void test_logsumexp_half() {
  const dim_t n = 3;
  ct2_f16* d = metal_alloc<ct2_f16>(n);
  d[0] = (ct2_f16)1.f; d[1] = (ct2_f16)2.f; d[2] = (ct2_f16)3.f;

  float result = primitives<Device::MPS>::logsumexp(d, n);
  float ref = std::log(std::exp(1.f) + std::exp(2.f) + std::exp(3.f));
  // float16 has ~1e-3 relative precision
  check(near(result, ref, 2e-3f), "logsumexp_half");

  metal_free(d);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.7: Beam-search and attention-mask primitives ===\n\n");

  std::printf("--- penalize_previous_tokens: float32 ---\n");
  test_penalize_basic<float>("float");
  test_penalize_zero_size<float>("float");
  test_penalize_duplicates<float>("float");

  std::printf("--- penalize_previous_tokens: float16 ---\n");
  test_penalize_basic<ct2_f16>("half");
  test_penalize_zero_size<ct2_f16>("half");
  test_penalize_duplicates<ct2_f16>("half");

  std::printf("--- penalize_previous_tokens: bfloat16 ---\n");
  test_penalize_basic<ct2_bf16>("bfloat");
  test_penalize_zero_size<ct2_bf16>("bfloat");
  test_penalize_duplicates<ct2_bf16>("bfloat");

  std::printf("--- prepare_length_mask ---\n");
  test_prepare_mask_padded();
  test_prepare_mask_causal();
  test_prepare_mask_causal_padded();
  test_prepare_mask_multi_query();

  std::printf("--- at() ---\n");
  test_at_after_gpu_write();

  std::printf("--- logsumexp ---\n");
  test_logsumexp_float();
  test_logsumexp_half();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
