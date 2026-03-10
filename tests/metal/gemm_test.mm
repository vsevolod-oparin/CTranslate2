// Standalone tests for M4.4 GEMM: FP32, FP16, BF16.
//
// Verifies correctness of primitives<Device::MPS>::gemm and
// gemm_batch_strided against a CPU reference (std::inner_product).
//
// Run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/gemm_test.mm \
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
//     -o gemm_test && ./gemm_test

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

// arm_vector_types.h (Metal.h) defines ::float16_t as __fp16,
// conflicting with ctranslate2::float16_t.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

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
// Allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// ---------------------------------------------------------------------------
// BF16 / FP16 conversion helpers
// ---------------------------------------------------------------------------

static inline uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u)
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  const uint32_t bias = 0x00007FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}
static inline float bf16_to_float(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
static inline float fp16_to_float(uint16_t h) {
  // IEEE 754 half → float via standard reinterpret
  float f;
  // Use arm intrinsic on Apple Silicon
  __fp16 tmp;
  std::memcpy(&tmp, &h, 2);
  f = static_cast<float>(tmp);
  return f;
}
static inline uint16_t float_to_fp16(float f) {
  __fp16 tmp = static_cast<__fp16>(f);
  uint16_t h; std::memcpy(&h, &tmp, 2); return h;
}

// ---------------------------------------------------------------------------
// CPU reference GEMM (row-major, no transpose)
// C[i][j] = alpha * sum_k A[i][k]*B[k][j] + beta*C[i][j]
// All types computed in float32.
// ---------------------------------------------------------------------------

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
// FP32 GEMM tests
// ---------------------------------------------------------------------------

static void test_gemm_float(bool trans_a, bool trans_b, int m, int n, int k) {
  const int lda = trans_a ? m : k;
  const int ldb = trans_b ? k : n;
  const int ldc = n;

  float* A = metal_alloc<float>(k * m);  // physical: rows × cols
  float* B = metal_alloc<float>(k * n);
  float* C = metal_alloc<float>(m * n);

  // Rows of A = trans_a ? k : m, cols = lda
  const int ra = trans_a ? k : m;
  const int rb = trans_b ? n : k;

  std::mt19937 rng(1234 + m + n + k);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < ra * lda; ++i) A[i] = dist(rng);
  for (int i = 0; i < rb * ldb; ++i) B[i] = dist(rng);
  for (int i = 0; i < m * ldc; ++i) C[i] = 0.f;

  std::vector<float> C_ref(m * n, 0.f);
  cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f, A, lda, B, ldb, 0.f, C_ref.data(), n);

  primitives<Device::MPS>::gemm<float, float>(
      false, false, trans_a, trans_b, m, n, k,
      1.f, A, lda, B, ldb, 0.f, C, ldc);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < m * n; ++i) {
    float err = std::fabs(C[i] - C_ref[i]);
    if (err > max_err) max_err = err;
  }

  char label[128];
  std::snprintf(label, sizeof(label), "gemm<float> m=%d n=%d k=%d ta=%d tb=%d  max_err=%.2e",
                m, n, k, (int)trans_a, (int)trans_b, (double)max_err);
  CHECK(label, max_err < 1e-3f * std::sqrt((float)k));

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// FP16 GEMM tests
// ---------------------------------------------------------------------------

static void test_gemm_fp16(bool trans_a, bool trans_b, int m, int n, int k) {
  const int lda = trans_a ? m : k;
  const int ldb = trans_b ? k : n;
  const int ldc = n;

  const int ra = trans_a ? k : m;
  const int rb = trans_b ? n : k;

  ct2_f16* A = metal_alloc<ct2_f16>(ra * lda);
  ct2_f16* B = metal_alloc<ct2_f16>(rb * ldb);
  ct2_f16* C = metal_alloc<ct2_f16>(m * n);

  std::vector<float> Af(ra * lda), Bf(rb * ldb), Cf_ref(m * n, 0.f);

  std::mt19937 rng(5678 + m + n + k);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : Af) x = dist(rng);
  for (auto& x : Bf) x = dist(rng);

  // Store as FP16 (round-trip so reference uses the same values)
  for (int i = 0; i < ra * lda; ++i) {
    uint16_t h = float_to_fp16(Af[i]);
    std::memcpy(&A[i], &h, 2);
    Af[i] = fp16_to_float(h);
  }
  for (int i = 0; i < rb * ldb; ++i) {
    uint16_t h = float_to_fp16(Bf[i]);
    std::memcpy(&B[i], &h, 2);
    Bf[i] = fp16_to_float(h);
  }
  for (int i = 0; i < m * n; ++i) {
    uint16_t zero = float_to_fp16(0.f);
    std::memcpy(&C[i], &zero, 2);
  }

  cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f,
               Af.data(), lda, Bf.data(), ldb, 0.f, Cf_ref.data(), n);

  primitives<Device::MPS>::gemm<ct2_f16, ct2_f16>(
      false, false, trans_a, trans_b, m, n, k,
      1.f, A, lda, B, ldb, 0.f, C, ldc);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < m * n; ++i) {
    uint16_t h; std::memcpy(&h, &C[i], 2);
    float cv = fp16_to_float(h);
    float err = std::fabs(cv - Cf_ref[i]);
    if (err > max_err) max_err = err;
  }

  char label[128];
  std::snprintf(label, sizeof(label), "gemm<fp16>  m=%d n=%d k=%d ta=%d tb=%d  max_err=%.2e",
                m, n, k, (int)trans_a, (int)trans_b, (double)max_err);
  // FP16 unit-roundoff ≈ 1e-3; tolerance scales with √k for accumulated errors.
  CHECK(label, max_err < 5e-2f * std::sqrt((float)k));

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// BF16 GEMM tests
// ---------------------------------------------------------------------------

static void test_gemm_bf16(bool trans_a, bool trans_b, int m, int n, int k) {
  const int lda = trans_a ? m : k;
  const int ldb = trans_b ? k : n;
  const int ldc = n;

  const int ra = trans_a ? k : m;
  const int rb = trans_b ? n : k;

  ct2_bf16* A = metal_alloc<ct2_bf16>(ra * lda);
  ct2_bf16* B = metal_alloc<ct2_bf16>(rb * ldb);
  ct2_bf16* C = metal_alloc<ct2_bf16>(m * n);

  std::vector<float> Af(ra * lda), Bf(rb * ldb), Cf_ref(m * n, 0.f);

  std::mt19937 rng(9999 + m + n + k);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : Af) x = dist(rng);
  for (auto& x : Bf) x = dist(rng);

  // Store as BF16 and round-trip for reference
  for (int i = 0; i < ra * lda; ++i) {
    uint16_t b = float_to_bf16(Af[i]);
    std::memcpy(&A[i], &b, 2);
    Af[i] = bf16_to_float(b);
  }
  for (int i = 0; i < rb * ldb; ++i) {
    uint16_t b = float_to_bf16(Bf[i]);
    std::memcpy(&B[i], &b, 2);
    Bf[i] = bf16_to_float(b);
  }
  for (int i = 0; i < m * n; ++i) {
    uint16_t zero = float_to_bf16(0.f);
    std::memcpy(&C[i], &zero, 2);
  }

  cpu_gemm_ref(trans_a, trans_b, m, n, k, 1.f,
               Af.data(), lda, Bf.data(), ldb, 0.f, Cf_ref.data(), n);

  primitives<Device::MPS>::gemm<ct2_bf16, ct2_bf16>(
      false, false, trans_a, trans_b, m, n, k,
      1.f, A, lda, B, ldb, 0.f, C, ldc);
  // BF16 GEMM commits internally (MPSGraph path).

  float max_err = 0.f;
  for (int i = 0; i < m * n; ++i) {
    uint16_t b; std::memcpy(&b, &C[i], 2);
    float cv = bf16_to_float(b);
    float err = std::fabs(cv - Cf_ref[i]);
    if (err > max_err) max_err = err;
  }

  char label[128];
  std::snprintf(label, sizeof(label), "gemm<bf16>  m=%d n=%d k=%d ta=%d tb=%d  max_err=%.2e",
                m, n, k, (int)trans_a, (int)trans_b, (double)max_err);
  // BF16 unit-roundoff ≈ 7.8e-3; tolerance scales with √k.
  CHECK(label, max_err < 5e-2f * std::sqrt((float)k));

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// FP32 gemm_batch_strided test
// ---------------------------------------------------------------------------

static void test_gemm_batch_strided_float(int batch, int m, int n, int k) {
  const int lda = k, ldb = n, ldc = n;
  const int stridea = m * k, strideb = k * n, stridec = m * n;

  float* A = metal_alloc<float>(batch * stridea);
  float* B = metal_alloc<float>(batch * strideb);
  float* C = metal_alloc<float>(batch * stridec);

  std::mt19937 rng(3141 + batch + m + n + k);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int i = 0; i < batch * stridea; ++i) A[i] = dist(rng);
  for (int i = 0; i < batch * strideb; ++i) B[i] = dist(rng);
  for (int i = 0; i < batch * stridec; ++i) C[i] = 0.f;

  std::vector<float> C_ref(batch * stridec, 0.f);
  for (int b = 0; b < batch; ++b) {
    cpu_gemm_ref(false, false, m, n, k, 1.f,
                 A + b * stridea, lda,
                 B + b * strideb, ldb,
                 0.f, C_ref.data() + b * stridec, n);
  }

  primitives<Device::MPS>::gemm_batch_strided<float, float>(
      false, false, m, n, k, 1.f,
      A, lda, stridea,
      B, ldb, strideb,
      0.f, C, ldc, stridec, batch);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < batch * stridec; ++i) {
    float err = std::fabs(C[i] - C_ref[i]);
    if (err > max_err) max_err = err;
  }

  char label[128];
  std::snprintf(label, sizeof(label),
                "gemm_batch_strided<float> batch=%d m=%d n=%d k=%d  max_err=%.2e",
                batch, m, n, k, (double)max_err);
  CHECK(label, max_err < 1e-3f * std::sqrt((float)k));

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// BF16 gemm_batch_strided test
// ---------------------------------------------------------------------------

static void test_gemm_batch_strided_bf16(int batch, int m, int n, int k) {
  const int lda = k, ldb = n, ldc = n;
  const int stridea = m * k, strideb = k * n, stridec = m * n;

  ct2_bf16* A = metal_alloc<ct2_bf16>(batch * stridea);
  ct2_bf16* B = metal_alloc<ct2_bf16>(batch * strideb);
  ct2_bf16* C = metal_alloc<ct2_bf16>(batch * stridec);

  std::mt19937 rng(2718 + batch + m + n + k);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  std::vector<float> Af(batch * stridea), Bf(batch * strideb),
                     Cf_ref(batch * stridec, 0.f);
  for (auto& x : Af) x = dist(rng);
  for (auto& x : Bf) x = dist(rng);

  for (int i = 0; i < batch * stridea; ++i) {
    uint16_t b = float_to_bf16(Af[i]);
    std::memcpy(&A[i], &b, 2);
    Af[i] = bf16_to_float(b);
  }
  for (int i = 0; i < batch * strideb; ++i) {
    uint16_t b = float_to_bf16(Bf[i]);
    std::memcpy(&B[i], &b, 2);
    Bf[i] = bf16_to_float(b);
  }
  for (int i = 0; i < batch * stridec; ++i) {
    uint16_t zero = float_to_bf16(0.f);
    std::memcpy(&C[i], &zero, 2);
  }

  for (int b = 0; b < batch; ++b) {
    cpu_gemm_ref(false, false, m, n, k, 1.f,
                 Af.data() + b * stridea, lda,
                 Bf.data() + b * strideb, ldb,
                 0.f, Cf_ref.data() + b * stridec, n);
  }

  primitives<Device::MPS>::gemm_batch_strided<ct2_bf16, ct2_bf16>(
      false, false, m, n, k, 1.f,
      A, lda, stridea,
      B, ldb, strideb,
      0.f, C, ldc, stridec, batch);
  // BF16 batch GEMM commits internally.

  float max_err = 0.f;
  for (int i = 0; i < batch * stridec; ++i) {
    uint16_t b; std::memcpy(&b, &C[i], 2);
    float cv = bf16_to_float(b);
    float err = std::fabs(cv - Cf_ref[i]);
    if (err > max_err) max_err = err;
  }

  char label[128];
  std::snprintf(label, sizeof(label),
                "gemm_batch_strided<bf16>  batch=%d m=%d n=%d k=%d  max_err=%.2e",
                batch, m, n, k, (double)max_err);
  CHECK(label, max_err < 5e-2f * std::sqrt((float)k));

  metal_free(A); metal_free(B); metal_free(C);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.4 GEMM Tests ===\n\n");

  std::printf("--- FP32 gemm ---\n");
  // Non-transposed
  test_gemm_float(false, false, 4,   4,   4);
  test_gemm_float(false, false, 32,  32,  32);
  test_gemm_float(false, false, 128, 64,  256);
  test_gemm_float(false, false, 256, 128, 512);
  // Transpose A
  test_gemm_float(true,  false, 64,  32,  128);
  // Transpose B
  test_gemm_float(false, true,  64,  32,  128);
  // Transpose both
  test_gemm_float(true,  true,  64,  32,  128);

  std::printf("\n--- FP16 gemm ---\n");
  test_gemm_fp16(false, false, 4,   4,   4);
  test_gemm_fp16(false, false, 32,  32,  32);
  test_gemm_fp16(false, false, 128, 64,  256);
  test_gemm_fp16(true,  false, 64,  32,  128);
  test_gemm_fp16(false, true,  64,  32,  128);
  test_gemm_fp16(true,  true,  64,  32,  128);

  std::printf("\n--- BF16 gemm ---\n");
  test_gemm_bf16(false, false, 4,   4,   4);
  test_gemm_bf16(false, false, 32,  32,  32);
  test_gemm_bf16(false, false, 128, 64,  256);
  test_gemm_bf16(true,  false, 64,  32,  128);
  test_gemm_bf16(false, true,  64,  32,  128);
  test_gemm_bf16(true,  true,  64,  32,  128);

  std::printf("\n--- FP32 gemm_batch_strided ---\n");
  test_gemm_batch_strided_float(1,  32, 32, 32);
  test_gemm_batch_strided_float(4,  32, 32, 32);
  test_gemm_batch_strided_float(8,  64, 64, 64);
  test_gemm_batch_strided_float(16, 32, 32, 64);

  std::printf("\n--- BF16 gemm_batch_strided ---\n");
  test_gemm_batch_strided_bf16(1,  32, 32, 32);
  test_gemm_batch_strided_bf16(4,  32, 32, 32);
  test_gemm_batch_strided_bf16(8,  64, 64, 64);

  std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
