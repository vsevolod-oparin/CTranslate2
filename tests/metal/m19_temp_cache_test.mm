// M19: Regression test for F16TempCache / SdpaF16TempCache stale data bug.
//
// Verifies that f16 GEMM produces correct results when dimensions change
// between calls (e.g., beam search with 3000 frames → greedy with 200 frames).
//
// Without the fix, the second (smaller) call reads stale float32 values from
// the previous call's oversized temp buffers, corrupting MPS accumulation.
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/m19_temp_cache_test.mm \
//     -L build -lctranslate2.mps \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -Wl,-rpath,build \
//     -o m19_temp_cache_test && ./m19_temp_cache_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;

using namespace ctranslate2;

static int passed = 0;
static int failed = 0;

#define CHECK(label, expr)                               \
  do {                                                   \
    if (expr) {                                          \
      std::printf("  PASS  %s\n", label);                \
      ++passed;                                          \
    } else {                                             \
      std::printf("  FAIL  %s\n", label);                \
      ++failed;                                          \
    }                                                    \
  } while (0)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

static inline uint16_t float_to_fp16(float f) {
  __fp16 tmp = static_cast<__fp16>(f);
  uint16_t h; std::memcpy(&h, &tmp, 2); return h;
}
static inline float fp16_to_float(uint16_t h) {
  __fp16 tmp;
  std::memcpy(&tmp, &h, 2);
  return static_cast<float>(tmp);
}

// CPU reference GEMM (row-major, no transpose, float32).
static void cpu_gemm_ref(
    int m, int n, int k,
    float alpha,
    const float* a, int lda,
    const float* b, int ldb,
    float* c, int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.f;
      for (int p = 0; p < k; ++p) {
        acc += a[i * lda + p] * b[p * ldb + j];
      }
      c[i * ldc + j] = alpha * acc;
    }
  }
}

// Run an f16 GEMM on Metal and return max error vs CPU reference.
static float run_f16_gemm(int m, int n, int k, unsigned seed) {
  const int lda = k, ldb = n, ldc = n;

  ct2_f16* A = metal_alloc<ct2_f16>(m * k);
  ct2_f16* B = metal_alloc<ct2_f16>(k * n);
  ct2_f16* C = metal_alloc<ct2_f16>(m * n);

  std::vector<float> Af(m * k), Bf(k * n), Cf_ref(m * n, 0.f);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : Af) x = dist(rng);
  for (auto& x : Bf) x = dist(rng);

  // Round-trip through fp16 so CPU reference uses same values.
  for (int i = 0; i < m * k; ++i) {
    uint16_t h = float_to_fp16(Af[i]);
    std::memcpy(&A[i], &h, 2);
    Af[i] = fp16_to_float(h);
  }
  for (int i = 0; i < k * n; ++i) {
    uint16_t h = float_to_fp16(Bf[i]);
    std::memcpy(&B[i], &h, 2);
    Bf[i] = fp16_to_float(h);
  }
  for (int i = 0; i < m * n; ++i) {
    uint16_t zero = float_to_fp16(0.f);
    std::memcpy(&C[i], &zero, 2);
  }

  cpu_gemm_ref(m, n, k, 1.f, Af.data(), lda, Bf.data(), ldb, Cf_ref.data(), n);

  primitives<Device::MPS>::gemm<ct2_f16, ct2_f16>(
      false, false, false, false, m, n, k,
      1.f, A, lda, B, ldb, 0.f, C, ldc);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < m * n; ++i) {
    uint16_t h; std::memcpy(&h, &C[i], 2);
    float cv = fp16_to_float(h);
    float err = std::fabs(cv - Cf_ref[i]);
    if (err > max_err) max_err = err;
  }

  metal_free(A); metal_free(B); metal_free(C);
  return max_err;
}

// ---------------------------------------------------------------------------
// Test: large GEMM followed by small GEMM (the M19 bug pattern)
//
// This simulates beam search (large m) followed by greedy decode (small m)
// with the same n and k (model dimensions stay constant).
// ---------------------------------------------------------------------------
static void test_f16_gemm_large_then_small() {
  const int n = 512, k = 512;  // Typical model hidden dim
  const int m_large = 256;     // Simulates beam search batch (e.g., 5 beams × 50 tokens)
  const int m_small = 4;       // Simulates greedy decode (1 beam × 4 tokens)

  // Step 1: Large GEMM — warms up the F16TempCache with large buffers.
  float err_large = run_f16_gemm(m_large, n, k, 42);
  float tol_large = 0.05f * std::sqrt((float)k);
  char label1[256];
  std::snprintf(label1, sizeof(label1),
      "large_gemm  m=%d n=%d k=%d  max_err=%.2e (tol %.2e)",
      m_large, n, k, (double)err_large, (double)tol_large);
  CHECK(label1, err_large < tol_large);

  // Step 2: Small GEMM — reuses the oversized temp buffers.
  // Without the M19 fix, stale data from step 1 corrupts the result.
  float err_small = run_f16_gemm(m_small, n, k, 99);
  float tol_small = 0.05f * std::sqrt((float)k);
  char label2[256];
  std::snprintf(label2, sizeof(label2),
      "small_gemm_after_large  m=%d n=%d k=%d  max_err=%.2e (tol %.2e)",
      m_small, n, k, (double)err_small, (double)tol_small);
  CHECK(label2, err_small < tol_small);
}

// ---------------------------------------------------------------------------
// Test: multiple dimension changes (stress test)
//
// Cycles through progressively different m values to ensure the cache
// handles repeated reuse correctly.
// ---------------------------------------------------------------------------
static void test_f16_gemm_varying_dimensions() {
  const int n = 256, k = 256;
  const int m_values[] = {512, 8, 256, 1, 128, 32, 512, 4};
  const int num_steps = sizeof(m_values) / sizeof(m_values[0]);

  for (int step = 0; step < num_steps; ++step) {
    const int m = m_values[step];
    float err = run_f16_gemm(m, n, k, 1000 + step);
    float tol = 0.05f * std::sqrt((float)k);
    char label[256];
    std::snprintf(label, sizeof(label),
        "varying_dim  step=%d m=%d n=%d k=%d  max_err=%.2e (tol %.2e)",
        step, m, n, k, (double)err, (double)tol);
    CHECK(label, err < tol);
  }
}

// ---------------------------------------------------------------------------
// Test: n and k also change (covers all 3 temp buffer slots)
//
// A, B, C temp buffers all have different sizes when n and k change.
// ---------------------------------------------------------------------------
static void test_f16_gemm_all_dims_change() {
  struct Config { int m, n, k; };
  const Config configs[] = {
    {256, 1024, 512},  // Large: warms up all 3 cache slots
    {4,   128,  64},   // Small: all 3 slots reuse oversized buffers
    {64,  512,  256},  // Medium: mixed reuse
    {1,   64,   32},   // Minimal: extreme shrink
  };

  for (int i = 0; i < 4; ++i) {
    auto [m, n, k] = configs[i];
    float err = run_f16_gemm(m, n, k, 2000 + i);
    float tol = 0.05f * std::sqrt((float)k);
    char label[256];
    std::snprintf(label, sizeof(label),
        "all_dims_change  step=%d m=%d n=%d k=%d  max_err=%.2e (tol %.2e)",
        i, m, n, k, (double)err, (double)tol);
    CHECK(label, err < tol);
  }
}

// ---------------------------------------------------------------------------
// Test: small then large (reverse direction — should always work,
// but verifies cache growth path is also correct)
// ---------------------------------------------------------------------------
static void test_f16_gemm_small_then_large() {
  const int n = 512, k = 512;

  float err_small = run_f16_gemm(4, n, k, 3000);
  float tol = 0.05f * std::sqrt((float)k);
  char label1[256];
  std::snprintf(label1, sizeof(label1),
      "small_first  m=4 n=%d k=%d  max_err=%.2e", n, k, (double)err_small);
  CHECK(label1, err_small < tol);

  float err_large = run_f16_gemm(256, n, k, 3001);
  char label2[256];
  std::snprintf(label2, sizeof(label2),
      "large_after_small  m=256 n=%d k=%d  max_err=%.2e", n, k, (double)err_large);
  CHECK(label2, err_large < tol);
}

// ---------------------------------------------------------------------------
// Test: m=1 GEMV edge case (may use different kernel path)
// ---------------------------------------------------------------------------
static void test_f16_gemv_after_large() {
  const int n = 512, k = 512;

  // Warm up with large m.
  run_f16_gemm(128, n, k, 4000);

  // m=1: GEMV path. F16TempCache still holds oversized buffers.
  float err = run_f16_gemm(1, n, k, 4001);
  float tol = 0.05f * std::sqrt((float)k);
  char label[256];
  std::snprintf(label, sizeof(label),
      "gemv_after_large  m=1 n=%d k=%d  max_err=%.2e", n, k, (double)err);
  CHECK(label, err < tol);
}

int main() {
  @autoreleasepool {
    std::printf("=== M19 Temp Cache Stale Data Tests ===\n\n");

    std::printf("--- Large then small (the M19 bug pattern) ---\n");
    test_f16_gemm_large_then_small();

    std::printf("\n--- Varying dimensions (stress) ---\n");
    test_f16_gemm_varying_dimensions();

    std::printf("\n--- All dimensions change ---\n");
    test_f16_gemm_all_dims_change();

    std::printf("\n--- Small then large (reverse) ---\n");
    test_f16_gemm_small_then_large();

    std::printf("\n--- GEMV after large GEMM ---\n");
    test_f16_gemv_after_large();

    std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
  }
}
