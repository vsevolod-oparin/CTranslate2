// Regression tests for bugs found in M11 optimization audit.
//
// 1. batch_cpu_gemm_f16 stride bug: widen loop ignored lda/ldb strides
// 2. TopK k>64 guard: GPU kernel silently truncated at TOPK_MAX_K=64
//
// Build:
//   cd /path/to/CTranslate2
//   clang++ -std=c++17 -O2 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/bugfix_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm src/metal/ops_topk.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -framework Accelerate \
//     -o bugfix_test && ./bugfix_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
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

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

static inline uint16_t float_to_fp16(float f) {
  __fp16 tmp = static_cast<__fp16>(f);
  uint16_t h; std::memcpy(&h, &tmp, 2); return h;
}
static inline float fp16_to_float(uint16_t h) {
  __fp16 tmp; std::memcpy(&tmp, &h, 2);
  return static_cast<float>(tmp);
}

// CPU reference: strided GEMM in float32 (supports lda != cols)
static void cpu_gemm_ref(
    bool trans_a, bool trans_b,
    int m, int n, int k,
    float alpha,
    const float* a, int lda,
    const float* b, int ldb,
    float beta,
    float* c, int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.f;
      for (int p = 0; p < k; ++p) {
        float av = trans_a ? a[p * lda + i] : a[i * lda + p];
        float bv = trans_b ? b[j * ldb + p] : b[p * ldb + j];
        acc += av * bv;
      }
      c[i * ldc + j] = alpha * acc + beta * c[i * ldc + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Test 1: FP16 batch_strided GEMM with non-contiguous (padded) strides
//
// This exercises batch_cpu_gemm_f16 with lda > cols_a to verify that
// the strided widen/narrow correctly handles non-contiguous input.
// Before the fix, the widen loop used linear indexing (ai[j]) which
// ignored lda, producing wrong results for lda != cols_a.
// ---------------------------------------------------------------------------
static void test_fp16_gemm_strided_lda() {
  std::printf("\n--- FP16 strided GEMM (non-contiguous lda) ---\n");

  // Use m=1, n=3, k=5 — a tiny decode-style GEMM.
  // lda is padded to 8 (> k=5 for non-transpose, so lda > cols_a).
  // This hits the batch_cpu_gemm_f16 path because:
  //   - m*n = 3 <= 4096 (tiny)
  //   - n=3, float16: 3*2=6 < 16 bytes → needs_padding = true
  //   - m=1, but float16 m=1 stays on cblas (MPS bug)
  const int m = 1, n = 3, k = 5;
  const bool trans_a = false, trans_b = true;
  // For trans_a=false: A is [m, k] with lda >= k.
  // For trans_b=true:  B is [n, k] with ldb >= k.
  const int lda = 8;  // padded: lda > k (non-contiguous!)
  const int ldb = 8;  // padded: ldb > k (non-contiguous!)
  const int ldc = n;  // C is [m, n], contiguous
  const int batch_size = 2;

  // Physical sizes in elements
  const int rows_a = m;       // trans_a=false → rows = m
  const int cols_a = k;       // trans_a=false → cols = k
  const int rows_b = n;       // trans_b=true  → rows = n
  const int cols_b = k;       // trans_b=true  → cols = k
  const int stridea = rows_a * lda;
  const int strideb = rows_b * ldb;
  const int stridec = m * ldc;

  // Allocate Metal buffers large enough for padded layout
  ct2_f16* A = metal_alloc<ct2_f16>(batch_size * stridea);
  ct2_f16* B = metal_alloc<ct2_f16>(batch_size * strideb);
  ct2_f16* C = metal_alloc<ct2_f16>(batch_size * stridec);

  // Fill with random data (including the padding columns)
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  // Fill A: only valid cols [0..cols_a), pad cols [cols_a..lda) with garbage
  std::vector<float> Af(batch_size * stridea, 0.f);
  for (int b = 0; b < batch_size; ++b) {
    for (int r = 0; r < rows_a; ++r) {
      for (int c = 0; c < lda; ++c) {
        float v = (c < cols_a) ? dist(rng) : 999.f;  // garbage in padding
        Af[b * stridea + r * lda + c] = v;
        uint16_t h = float_to_fp16(v);
        std::memcpy(&A[b * stridea + r * lda + c], &h, 2);
      }
    }
  }
  // Fill B similarly
  std::vector<float> Bf(batch_size * strideb, 0.f);
  for (int b = 0; b < batch_size; ++b) {
    for (int r = 0; r < rows_b; ++r) {
      for (int c = 0; c < ldb; ++c) {
        float v = (c < cols_b) ? dist(rng) : 999.f;
        Bf[b * strideb + r * ldb + c] = v;
        uint16_t h = float_to_fp16(v);
        std::memcpy(&B[b * strideb + r * ldb + c], &h, 2);
      }
    }
  }
  // Zero C
  std::memset(C, 0, batch_size * stridec * sizeof(ct2_f16));

  // CPU reference (float32, strided)
  std::vector<float> C_ref(batch_size * stridec, 0.f);
  for (int b = 0; b < batch_size; ++b) {
    cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f,
                 Af.data() + b * stridea, lda,
                 Bf.data() + b * strideb, ldb,
                 0.f,
                 C_ref.data() + b * stridec, ldc);
  }

  // Metal GEMM
  primitives<Device::METAL>::gemm_batch_strided<ct2_f16, ct2_f16>(
      trans_a, trans_b, m, n, k,
      1.f, A, lda, stridea, B, ldb, strideb,
      0.f, C, ldc, stridec, batch_size);
  metal::commit_and_wait();

  // Compare
  float max_err = 0.f;
  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < m * n; ++i) {
      uint16_t h;
      std::memcpy(&h, &C[b * stridec + i], 2);
      float got = fp16_to_float(h);
      float ref = C_ref[b * stridec + i];
      float err = std::fabs(got - ref);
      if (err > max_err) max_err = err;
    }
  }

  char label[256];
  std::snprintf(label, sizeof(label),
      "fp16 strided GEMM (lda=%d > k=%d, ldb=%d > k=%d, batch=%d)  max_err=%.4e",
      lda, k, ldb, k, batch_size, (double)max_err);
  // FP16 tolerance: ~1e-3 for small k
  CHECK(label, max_err < 5e-2f);

  metal_free(A); metal_free(B); metal_free(C);
}

// Same test but with trans_a=false, trans_b=false (different stride pattern)
static void test_fp16_gemm_strided_ldb_no_trans() {
  std::printf("\n--- FP16 strided GEMM (non-contiguous ldb, no transpose) ---\n");

  const int m = 1, n = 3, k = 5;
  const bool trans_a = false, trans_b = false;
  // A is [m, k] with lda = k (contiguous)
  // B is [k, n] with ldb = 8 > n (non-contiguous!)
  const int lda = k;
  const int ldb = 8;  // padded
  const int ldc = n;
  const int batch_size = 3;

  const int rows_a = m, cols_a = k;
  const int rows_b = k, cols_b = n;
  const int stridea = rows_a * lda;
  const int strideb = rows_b * ldb;
  const int stridec = m * ldc;

  ct2_f16* A = metal_alloc<ct2_f16>(batch_size * stridea);
  ct2_f16* B = metal_alloc<ct2_f16>(batch_size * strideb);
  ct2_f16* C = metal_alloc<ct2_f16>(batch_size * stridec);

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> Af(batch_size * stridea), Bf(batch_size * strideb);
  for (int b = 0; b < batch_size; ++b) {
    for (int r = 0; r < rows_a; ++r)
      for (int c = 0; c < lda; ++c) {
        float v = dist(rng);
        Af[b * stridea + r * lda + c] = v;
        uint16_t h = float_to_fp16(v);
        std::memcpy(&A[b * stridea + r * lda + c], &h, 2);
      }
    for (int r = 0; r < rows_b; ++r)
      for (int c = 0; c < ldb; ++c) {
        float v = (c < cols_b) ? dist(rng) : 888.f;
        Bf[b * strideb + r * ldb + c] = v;
        uint16_t h = float_to_fp16(v);
        std::memcpy(&B[b * strideb + r * ldb + c], &h, 2);
      }
  }
  std::memset(C, 0, batch_size * stridec * sizeof(ct2_f16));

  std::vector<float> C_ref(batch_size * stridec, 0.f);
  for (int b = 0; b < batch_size; ++b)
    cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f,
                 Af.data() + b * stridea, lda,
                 Bf.data() + b * strideb, ldb,
                 0.f, C_ref.data() + b * stridec, ldc);

  primitives<Device::METAL>::gemm_batch_strided<ct2_f16, ct2_f16>(
      trans_a, trans_b, m, n, k,
      1.f, A, lda, stridea, B, ldb, strideb,
      0.f, C, ldc, stridec, batch_size);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int b = 0; b < batch_size; ++b)
    for (int i = 0; i < m * n; ++i) {
      uint16_t h;
      std::memcpy(&h, &C[b * stridec + i], 2);
      float err = std::fabs(fp16_to_float(h) - C_ref[b * stridec + i]);
      if (err > max_err) max_err = err;
    }

  char label[256];
  std::snprintf(label, sizeof(label),
      "fp16 strided GEMM (ldb=%d > n=%d, no_trans, batch=%d)  max_err=%.4e",
      ldb, n, batch_size, (double)max_err);
  CHECK(label, max_err < 5e-2f);

  metal_free(A); metal_free(B); metal_free(C);
}

// Test with non-contiguous C (ldc > n)
static void test_fp16_gemm_strided_ldc() {
  std::printf("\n--- FP16 strided GEMM (non-contiguous ldc) ---\n");

  const int m = 2, n = 3, k = 4;
  const bool trans_a = false, trans_b = false;
  const int lda = k;
  const int ldb = n;
  const int ldc = 8;  // padded: ldc > n
  const int batch_size = 2;

  const int rows_a = m, rows_b = k;
  const int stridea = rows_a * lda;
  const int strideb = rows_b * ldb;
  const int stridec = m * ldc;

  ct2_f16* A = metal_alloc<ct2_f16>(batch_size * stridea);
  ct2_f16* B = metal_alloc<ct2_f16>(batch_size * strideb);
  ct2_f16* C = metal_alloc<ct2_f16>(batch_size * stridec);

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> Af(batch_size * stridea), Bf(batch_size * strideb);
  for (int i = 0; i < batch_size * stridea; ++i) {
    float v = dist(rng);
    Af[i] = v;
    uint16_t h = float_to_fp16(v);
    std::memcpy(&A[i], &h, 2);
  }
  for (int i = 0; i < batch_size * strideb; ++i) {
    float v = dist(rng);
    Bf[i] = v;
    uint16_t h = float_to_fp16(v);
    std::memcpy(&B[i], &h, 2);
  }
  // Fill C with garbage in padding columns to detect overwrites
  for (int i = 0; i < batch_size * stridec; ++i) {
    uint16_t h = float_to_fp16(777.f);
    std::memcpy(&C[i], &h, 2);
  }

  std::vector<float> C_ref(batch_size * stridec, 777.f);
  for (int b = 0; b < batch_size; ++b)
    cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f,
                 Af.data() + b * stridea, lda,
                 Bf.data() + b * strideb, ldb,
                 0.f, C_ref.data() + b * stridec, ldc);

  primitives<Device::METAL>::gemm_batch_strided<ct2_f16, ct2_f16>(
      trans_a, trans_b, m, n, k,
      1.f, A, lda, stridea, B, ldb, strideb,
      0.f, C, ldc, stridec, batch_size);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int b = 0; b < batch_size; ++b)
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < n; ++c) {
        uint16_t h;
        std::memcpy(&h, &C[b * stridec + r * ldc + c], 2);
        float err = std::fabs(fp16_to_float(h) - C_ref[b * stridec + r * ldc + c]);
        if (err > max_err) max_err = err;
      }

  char label[256];
  std::snprintf(label, sizeof(label),
      "fp16 strided GEMM (ldc=%d > n=%d, batch=%d)  max_err=%.4e",
      ldc, n, batch_size, (double)max_err);
  CHECK(label, max_err < 5e-2f);

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// Test 2: TopK k>64 throws instead of silent truncation
// ---------------------------------------------------------------------------

// Forward-declare the topk_metal function from ops_topk.mm
namespace ctranslate2 { namespace metal {
  template <typename T>
  void topk_metal(const T* input, T* values, int32_t* indices,
                  dim_t batch_size, dim_t depth, dim_t k = 1);
}}

static void test_topk_k_guard() {
  std::printf("\n--- TopK k>64 guard ---\n");

  const dim_t batch = 1, depth = 128;
  float* input  = metal_alloc<float>(batch * depth);
  float* values = metal_alloc<float>(batch * 65);
  int32_t* indices = metal_alloc<int32_t>(batch * 65);

  // Fill input
  for (int i = 0; i < batch * depth; ++i) input[i] = (float)i;

  // k=5 should work (no throw)
  bool k5_ok = true;
  try {
    metal::topk_metal<float>(input, values, indices, batch, depth, 5);
    metal::commit_and_wait();
  } catch (...) {
    k5_ok = false;
  }
  CHECK("topk k=5 does not throw", k5_ok);

  // k=64 should work (boundary)
  bool k64_ok = true;
  try {
    metal::topk_metal<float>(input, values, indices, batch, depth, 64);
    metal::commit_and_wait();
  } catch (...) {
    k64_ok = false;
  }
  CHECK("topk k=64 does not throw", k64_ok);

  // k=65 should throw
  bool k65_threw = false;
  try {
    metal::topk_metal<float>(input, values, indices, batch, depth, 65);
    metal::commit_and_wait();
  } catch (const std::runtime_error& e) {
    k65_threw = true;
    std::printf("    (caught: %s)\n", e.what());
  }
  CHECK("topk k=65 throws runtime_error", k65_threw);

  metal_free(input); metal_free(values); metal_free(indices);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  @autoreleasepool {
    std::printf("=== M11 Audit Bug-Fix Regression Tests ===\n");

    test_fp16_gemm_strided_lda();
    test_fp16_gemm_strided_ldb_no_trans();
    test_fp16_gemm_strided_ldc();
    test_topk_k_guard();

    std::printf("\n%d/%d passed\n", passed, passed + failed);
    return failed > 0 ? 1 : 0;
  }
}
