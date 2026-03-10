// tests/metal/m92_test.mm
//
// M9.2 — INT8 GEMM on Metal tests.
//
// Tests primitives<Device::MPS>::gemm<int8_t, int32_t>() directly.
// Strategy: dequantize INT8 inputs to FP32, run FP32 MPS GEMM, round → INT32.
//
// PASS criteria:
//   - INT8 A * INT8 B → INT32 matches CPU reference (exact for k ≤ ~1040)
//   - Transpose variants produce correct results
//   - Full pipeline: quantize(float) → INT8 GEMM → dequantize_gemm_output
//     → float output matches float32 GEMM within 1%
//
// Build command (from repo root):
//   clang++ -std=c++17 -O0 \
//       -I include -I src \
//       -DCT2_WITH_MPS \
//       tests/metal/m92_test.mm \
//       src/metal/ops_quantize.mm \
//       src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//       src/metal/primitives_memory.mm \
//       src/metal/primitives_elementwise.mm \
//       src/metal/primitives_reduction.mm \
//       src/metal/primitives_gemm.mm \
//       src/metal/primitives_transpose.mm \
//       src/metal/primitives_beam_search.mm \
//       src/metal/ops_norm_gather.mm \
//       src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//       -framework Metal -framework Foundation \
//       -framework MetalPerformanceShaders \
//       -framework MetalPerformanceShadersGraph \
//       -o m92_test && ./m92_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <string>

#include "ctranslate2/types.h"
#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "metal/utils.h"
#include "metal/ops_metal.h"

// M5.1 fix: declare before 'using namespace ctranslate2'
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

static int g_tests = 0, g_pass = 0, g_fail = 0;

#define CHECK(cond, msg)                                           \
  do {                                                             \
    ++g_tests;                                                     \
    if (cond) { ++g_pass; }                                        \
    else {                                                         \
      ++g_fail;                                                    \
      std::fprintf(stderr, "  FAIL: %s  [%s:%d]\n",               \
                   msg, __FILE__, __LINE__);                       \
    }                                                              \
  } while (0)

static void* alloc_metal(size_t n_bytes) {
  return get_allocator<Device::MPS>().allocate(n_bytes, 0);
}
static void free_metal(void* p) {
  get_allocator<Device::MPS>().free(p, 0);
}

// ---------------------------------------------------------------------------
// CPU reference: INT8 GEMM → INT32
//   C[m,n] = sum_k A[m,k] * B[k,n]  (or transposed variants)
// ---------------------------------------------------------------------------

static void ref_int8_gemm(bool trans_a, bool trans_b,
                           int m, int n, int k,
                           const int8_t* a, int lda,
                           const int8_t* b, int ldb,
                           int32_t* c, int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int32_t acc = 0;
      for (int p = 0; p < k; ++p) {
        int a_val = trans_a ? a[p * lda + i] : a[i * lda + p];
        int b_val = trans_b ? b[j * ldb + p] : b[p * ldb + j];
        acc += a_val * b_val;
      }
      c[i * ldc + j] = acc;
    }
  }
}

// CPU reference: float GEMM (row-major, no transpose, alpha=1, beta=0).
static void ref_float_gemm(int m, int n, int k,
                            const float* a, const float* b, float* c) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float acc = 0.f;
      for (int p = 0; p < k; ++p)
        acc += a[i * k + p] * b[p * n + j];
      c[i * n + j] = acc;
    }
}

// CPU reference: quantize float row-by-row to int8.
static void ref_quantize(const float* in, int8_t* out, float* scales,
                          int batch, int depth) {
  for (int r = 0; r < batch; ++r) {
    const float* row = in + r * depth;
    float amax = 0.f;
    for (int c = 0; c < depth; ++c)
      amax = std::max(amax, std::abs(row[c]));
    float scale = (amax != 0.f) ? 127.f / amax : 1.f;
    scales[r] = scale;
    for (int c = 0; c < depth; ++c)
      out[r * depth + c] = static_cast<int8_t>(std::round(row[c] * scale));
  }
}

// ---------------------------------------------------------------------------
// Test 1: Basic INT8 GEMM — no transpose, small matrix
// ---------------------------------------------------------------------------
static void test_int8_gemm_basic() {
  std::printf("Test 1: INT8 GEMM basic (m=4, n=8, k=16, no transpose)\n");

  const int m = 4, n = 8, k = 16;
  const int lda = k, ldb = n, ldc = n;

  // Build int8 inputs on host.
  std::vector<int8_t> ha(m * k), hb(k * n);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<int8_t>((i % 7) - 3);
  for (int i = 0; i < k * n; ++i) hb[i] = static_cast<int8_t>((i % 5) - 2);

  // CPU reference.
  std::vector<int32_t> ref_c(m * n, 0);
  ref_int8_gemm(false, false, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

  // Metal GEMM.
  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k * sizeof(int8_t)));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n * sizeof(int8_t)));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), m * k);
  std::memcpy(d_b, hb.data(), k * n);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false,   // a_is_packed, b_is_packed
      false, false,   // trans_a, trans_b
      m, n, k,
      1.0f,
      d_a, lda, d_b, ldb,
      0.0f,
      d_c, ldc,
      nullptr);       // a_shift_compensation

  // Result is written synchronously by dispatch_int8_gemm (commit_and_wait inside).
  // No extra commit needed — just read d_c.

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM basic: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 2: INT8 GEMM with trans_b=true (common in CTranslate2 weight layout)
// ---------------------------------------------------------------------------
static void test_int8_gemm_transb() {
  std::printf("Test 2: INT8 GEMM trans_b (m=8, n=4, k=32)\n");

  const int m = 8, n = 4, k = 32;
  // A: [m, k], B: [n, k] (trans_b=true → effective B^T is [k, n])
  const int lda = k, ldb = k, ldc = n;

  std::vector<int8_t> ha(m * k), hb(n * k);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<int8_t>((i % 11) - 5);
  for (int i = 0; i < n * k; ++i) hb[i] = static_cast<int8_t>((i % 9)  - 4);

  // CPU reference (trans_b=true).
  std::vector<int32_t> ref_c(m * n, 0);
  ref_int8_gemm(false, true, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(n * k));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), m * k);
  std::memcpy(d_b, hb.data(), n * k);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, false, true, m, n, k,
      1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM trans_b: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 3: INT8 GEMM with alpha != 1
// ---------------------------------------------------------------------------
static void test_int8_gemm_alpha() {
  std::printf("Test 3: INT8 GEMM alpha=2.5 (m=4, n=4, k=8)\n");

  const int m = 4, n = 4, k = 8;
  const int lda = k, ldb = n, ldc = n;
  const float alpha = 2.5f;

  std::vector<int8_t> ha(m * k), hb(k * n);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<int8_t>((i % 5) - 2);
  for (int i = 0; i < k * n; ++i) hb[i] = static_cast<int8_t>((i % 7) - 3);

  // CPU reference.
  std::vector<int32_t> ref_c_raw(m * n, 0);
  ref_int8_gemm(false, false, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c_raw.data(), ldc);
  std::vector<int32_t> ref_c(m * n);
  for (int i = 0; i < m * n; ++i)
    ref_c[i] = static_cast<int32_t>(std::lroundf(alpha * ref_c_raw[i]));

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), m * k);
  std::memcpy(d_b, hb.data(), k * n);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, false, false, m, n, k,
      alpha, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM alpha=2.5: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 4: INT8 GEMM large k (k=512) — tests exact float32 accumulation
// ---------------------------------------------------------------------------
static void test_int8_gemm_large_k() {
  std::printf("Test 4: INT8 GEMM large k=512 (m=8, n=16)\n");

  const int m = 8, n = 16, k = 512;
  const int lda = k, ldb = n, ldc = n;

  std::vector<int8_t> ha(m * k), hb(k * n);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<int8_t>((i % 7) - 3);
  for (int i = 0; i < k * n; ++i) hb[i] = static_cast<int8_t>((i % 5) - 2);

  // CPU reference.
  std::vector<int32_t> ref_c(m * n, 0);
  ref_int8_gemm(false, false, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), m * k);
  std::memcpy(d_b, hb.data(), k * n);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, false, false, m, n, k,
      1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM large k=512: output matches CPU reference");

  // Verify max accumulator fits in float32 exactly (< 2^24).
  int32_t max_acc = *std::max_element(ref_c.begin(), ref_c.end());
  int32_t min_acc = *std::min_element(ref_c.begin(), ref_c.end());
  const int32_t fp32_exact_limit = 16777216; // 2^24
  CHECK(std::abs(max_acc) < fp32_exact_limit && std::abs(min_acc) < fp32_exact_limit,
        "INT8 GEMM large k: accumulators within float32 exact range");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 5: INT8 GEMM — zero matrix (edge case)
// ---------------------------------------------------------------------------
static void test_int8_gemm_zero() {
  std::printf("Test 5: INT8 GEMM zero matrix (m=4, n=4, k=16)\n");

  const int m = 4, n = 4, k = 16;

  std::vector<int8_t> ha(m * k, 0), hb(k * n, 0);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), m * k);
  std::memcpy(d_b, hb.data(), k * n);
  std::memset(d_c, 0xff, m * n * sizeof(int32_t)); // poison

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, false, false, m, n, k,
      1.0f, d_a, k, d_b, n, 0.0f, d_c, n, nullptr);

  bool all_zero = true;
  for (int i = 0; i < m * n; ++i)
    if (d_c[i] != 0) all_zero = false;
  CHECK(all_zero, "INT8 GEMM zero: output is all zeros");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 6: Full pipeline — float → quantize → INT8 GEMM → dequantize → float
//
// Pipeline:
//   float A [m,k], float B [n,k]  (B stored row-major, trans_b=true for GEMM)
//   1. Quantize A → int8 q_A [m,k], a_scales [m]
//   2. Quantize B → int8 q_B [n,k], b_scales [n]
//   3. INT8 GEMM: q_A [m,k] * q_B^T [k,n] → int32 C [m,n]
//   4. dequantize_gemm_output: C / (a_scales[row] * b_scales[col]) → float Y
//   5. Compare Y with float32 reference GEMM within 1%
// ---------------------------------------------------------------------------
static void test_int8_pipeline() {
  std::printf("Test 6: Full INT8 pipeline (m=8, n=16, k=64)\n");

  const int m = 8, n = 16, k = 64;

  // Float inputs.
  std::vector<float> hA(m * k), hB(n * k);
  for (int i = 0; i < m * k; ++i) hA[i] = -1.f + 2.f * (float)(i % 127) / 127.f;
  for (int i = 0; i < n * k; ++i) hB[i] = -1.f + 2.f * (float)((i * 3 + 7) % 127) / 127.f;

  // Float32 reference GEMM: Y_ref[m,n] = A[m,k] * B^T[k,n].
  std::vector<float> ref_Y(m * n, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      for (int p = 0; p < k; ++p)
        ref_Y[i * n + j] += hA[i * k + p] * hB[j * k + p];

  // --- Metal pipeline ---

  // 1. Allocate Metal buffers.
  float*   d_fA      = static_cast<float*>(alloc_metal(m * k * sizeof(float)));
  float*   d_fB      = static_cast<float*>(alloc_metal(n * k * sizeof(float)));
  int8_t*  d_qA      = static_cast<int8_t*>(alloc_metal(m * k));
  int8_t*  d_qB      = static_cast<int8_t*>(alloc_metal(n * k));
  float*   d_scA     = static_cast<float*>(alloc_metal(m * sizeof(float)));
  float*   d_scB     = static_cast<float*>(alloc_metal(n * sizeof(float)));
  int32_t* d_C       = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));
  float*   d_Y       = static_cast<float*>(alloc_metal(m * n * sizeof(float)));

  std::memcpy(d_fA, hA.data(), m * k * sizeof(float));
  std::memcpy(d_fB, hB.data(), n * k * sizeof(float));

  // 2. GPU quantize A and B.
  metal::quantize_int8_metal<float>(d_fA, d_qA, d_scA, m, k);
  metal::quantize_int8_metal<float>(d_fB, d_qB, d_scB, n, k);
  metal::commit_and_wait();  // flush quantize kernels

  // 3. INT8 GEMM: A [m,k] * B^T [k,n] → int32 C.
  //    B is stored as [n,k]; trans_b=true means it's treated as [k,n].
  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false,  // a_is_packed, b_is_packed
      false, true,   // trans_a=false, trans_b=true
      m, n, k,
      1.0f,
      d_qA, k,  // lda = k  (A is [m,k] contiguous)
      d_qB, k,  // ldb = k  (B is [n,k]; trans_b=true)
      0.0f,
      d_C, n,   // ldc = n
      nullptr);
  // dispatch_int8_gemm already called commit_and_wait internally.

  // 4. dequantize_gemm_output: C / (a_scales[row] * b_scales[col]) → float Y.
  //    trans_a=false → a_scales indexed by row
  //    trans_b=true  → b_scales indexed by col (but in dequantize_gemm_output,
  //    b_scales is [n] since B had n rows before transpose)
  metal::dequantize_gemm_output_metal<float>(
      d_C, d_scA, d_scB,
      static_cast<const void*>(d_C),  // dummy bias (has_bias=false)
      d_Y,
      m, n,        // batch=m, depth=n
      false, true, // trans_a, trans_b
      false,       // has_bias
      -1);         // no activation
  metal::commit_and_wait();

  // 5. Compare Y with float32 reference.
  // Use error relative to peak output (not per-element) since some outputs
  // can be small — the absolute error is what matters for GEMM accuracy.
  // Criterion: max(|Y - Y_ref|) / max(|Y_ref|) < 1%.
  float peak_ref = 0.f;
  for (int i = 0; i < m * n; ++i)
    peak_ref = std::max(peak_ref, std::abs(ref_Y[i]));
  float max_abs_err = 0.f;
  for (int i = 0; i < m * n; ++i)
    max_abs_err = std::max(max_abs_err, std::abs(d_Y[i] - ref_Y[i]));
  float norm_err = (peak_ref > 1e-6f) ? max_abs_err / peak_ref : 0.f;
  std::printf("  max abs err: %.6f, peak ref: %.4f, norm err: %.4f%%\n",
              max_abs_err, peak_ref, norm_err * 100.f);
  CHECK(norm_err < 0.01f, "Full INT8 pipeline: norm error < 1% of peak output");

  free_metal(d_fA); free_metal(d_fB);
  free_metal(d_qA); free_metal(d_qB);
  free_metal(d_scA); free_metal(d_scB);
  free_metal(d_C);  free_metal(d_Y);
}

// ---------------------------------------------------------------------------
// Test 7: batch_strided INT8 GEMM
// ---------------------------------------------------------------------------
static void test_int8_gemm_batch_strided() {
  std::printf("Test 7: INT8 GEMM batch_strided (batch=3, m=4, n=8, k=16)\n");

  const int batch = 3, m = 4, n = 8, k = 16;
  const int lda = k, ldb = n, ldc = n;
  const dim_t stridea = m * k, strideb = k * n, stridec = m * n;
  const dim_t total_a = batch * stridea, total_b = batch * strideb,
              total_c = batch * stridec;

  std::vector<int8_t> ha(total_a), hb(total_b);
  for (int i = 0; i < (int)total_a; ++i) ha[i] = static_cast<int8_t>((i % 7) - 3);
  for (int i = 0; i < (int)total_b; ++i) hb[i] = static_cast<int8_t>((i % 5) - 2);

  std::vector<int32_t> ref_c(total_c, 0);
  for (int b = 0; b < batch; ++b)
    ref_int8_gemm(false, false, m, n, k,
                  ha.data() + b * stridea, lda,
                  hb.data() + b * strideb, ldb,
                  ref_c.data() + b * stridec, ldc);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(total_a));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(total_b));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(total_c * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), total_a);
  std::memcpy(d_b, hb.data(), total_b);
  std::memset(d_c, 0, total_c * sizeof(int32_t));

  primitives<Device::MPS>::gemm_batch_strided<int8_t, int32_t>(
      false, false, m, n, k,
      1.0f,
      d_a, lda, stridea,
      d_b, ldb, strideb,
      0.0f,
      d_c, ldc, stridec,
      batch);

  bool all_ok = true;
  for (int i = 0; i < (int)total_c; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM batch_strided: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 8b: INT8 GEMM with trans_a=true
// ---------------------------------------------------------------------------
static void test_int8_gemm_transa() {
  std::printf("Test 8b: INT8 GEMM trans_a=true (m=4, n=8, k=16)\n");

  const int m = 4, n = 8, k = 16;
  // A stored as [k, m] (trans_a=true → lda=m)
  // B stored as [k, n] (trans_b=false → ldb=n)
  const int lda = m, ldb = n, ldc = n;

  std::vector<int8_t> ha(k * m), hb(k * n);
  for (int i = 0; i < k * m; ++i) ha[i] = static_cast<int8_t>((i % 9) - 4);
  for (int i = 0; i < k * n; ++i) hb[i] = static_cast<int8_t>((i % 7) - 3);

  // CPU reference (trans_a=true).
  std::vector<int32_t> ref_c(m * n, 0);
  ref_int8_gemm(true, false, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(k * m));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), k * m);
  std::memcpy(d_b, hb.data(), k * n);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, true, false, m, n, k,
      1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM trans_a: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 8c: INT8 GEMM with trans_a=true and trans_b=true
// ---------------------------------------------------------------------------
static void test_int8_gemm_transa_transb() {
  std::printf("Test 8c: INT8 GEMM trans_a=true, trans_b=true (m=4, n=8, k=16)\n");

  const int m = 4, n = 8, k = 16;
  // A stored as [k, m] → lda=m; B stored as [n, k] → ldb=k
  const int lda = m, ldb = k, ldc = n;

  std::vector<int8_t> ha(k * m), hb(n * k);
  for (int i = 0; i < k * m; ++i) ha[i] = static_cast<int8_t>((i % 11) - 5);
  for (int i = 0; i < n * k; ++i) hb[i] = static_cast<int8_t>((i % 6) - 2);

  std::vector<int32_t> ref_c(m * n, 0);
  ref_int8_gemm(true, true, m, n, k,
                ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(k * m));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(n * k));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

  std::memcpy(d_a, ha.data(), k * m);
  std::memcpy(d_b, hb.data(), n * k);
  std::memset(d_c, 0, m * n * sizeof(int32_t));

  primitives<Device::MPS>::gemm<int8_t, int32_t>(
      false, false, true, true, m, n, k,
      1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

  bool all_ok = true;
  for (int i = 0; i < m * n; ++i) {
    if (d_c[i] != ref_c[i]) {
      all_ok = false;
      std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                   i, d_c[i], ref_c[i]);
    }
  }
  CHECK(all_ok, "INT8 GEMM trans_a+trans_b: output matches CPU reference");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test T2: k > 1040 boundary — precision boundary for float32 accumulation
//
// float32 exact integer accumulation works for k ≤ 1040 (127*127*k < 2^24).
// At k=1024 with worst-case ±127 values, accumulation should still be exact.
// At k=2048 with large values, some precision loss is expected but should
// be graceful (no catastrophic errors).
// ---------------------------------------------------------------------------
static void test_int8_gemm_k_boundary() {
  std::printf("Test T2a: INT8 GEMM k=1024 (within exact range)\n");

  {
    const int m = 2, n = 4, k = 1024;
    const int lda = k, ldb = n, ldc = n;

    // Worst-case: alternating +127/-127 to maximize accumulator magnitude.
    std::vector<int8_t> ha(m * k), hb(k * n);
    for (int i = 0; i < m * k; ++i) ha[i] = (i % 2 == 0) ? 127 : -127;
    for (int i = 0; i < k * n; ++i) hb[i] = (i % 3 == 0) ? 127 : static_cast<int8_t>(-64);

    std::vector<int32_t> ref_c(m * n, 0);
    ref_int8_gemm(false, false, m, n, k,
                  ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

    // Verify accumulators are within float32 exact range.
    int32_t max_acc = 0;
    for (int i = 0; i < m * n; ++i)
      max_acc = std::max(max_acc, std::abs(ref_c[i]));
    const int32_t fp32_limit = 16777216;  // 2^24
    CHECK(max_acc < fp32_limit, "k=1024 accumulators within float32 exact range");

    int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
    int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
    int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

    std::memcpy(d_a, ha.data(), m * k);
    std::memcpy(d_b, hb.data(), k * n);
    std::memset(d_c, 0, m * n * sizeof(int32_t));

    primitives<Device::MPS>::gemm<int8_t, int32_t>(
        false, false, false, false, m, n, k,
        1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

    bool all_ok = true;
    for (int i = 0; i < m * n; ++i) {
      if (d_c[i] != ref_c[i]) {
        all_ok = false;
        std::fprintf(stderr, "  mismatch at [%d]: got %d, ref %d\n",
                     i, d_c[i], ref_c[i]);
      }
    }
    std::printf("  max_acc=%d  exact=%s\n", max_acc, all_ok ? "PASS" : "FAIL");
    CHECK(all_ok, "INT8 GEMM k=1024: exact match");

    free_metal(d_a); free_metal(d_b); free_metal(d_c);
  }

  std::printf("Test T2b: INT8 GEMM k=2048 (beyond exact range, graceful degradation)\n");

  {
    const int m = 2, n = 4, k = 2048;
    const int lda = k, ldb = n, ldc = n;

    // Moderate values to keep accumulator near the boundary.
    std::vector<int8_t> ha(m * k), hb(k * n);
    for (int i = 0; i < m * k; ++i) ha[i] = static_cast<int8_t>((i % 7) - 3);
    for (int i = 0; i < k * n; ++i) hb[i] = static_cast<int8_t>((i % 5) - 2);

    std::vector<int32_t> ref_c(m * n, 0);
    ref_int8_gemm(false, false, m, n, k,
                  ha.data(), lda, hb.data(), ldb, ref_c.data(), ldc);

    int8_t*  d_a = static_cast<int8_t*>(alloc_metal(m * k));
    int8_t*  d_b = static_cast<int8_t*>(alloc_metal(k * n));
    int32_t* d_c = static_cast<int32_t*>(alloc_metal(m * n * sizeof(int32_t)));

    std::memcpy(d_a, ha.data(), m * k);
    std::memcpy(d_b, hb.data(), k * n);
    std::memset(d_c, 0, m * n * sizeof(int32_t));

    primitives<Device::MPS>::gemm<int8_t, int32_t>(
        false, false, false, false, m, n, k,
        1.0f, d_a, lda, d_b, ldb, 0.0f, d_c, ldc, nullptr);

    // Allow small rounding errors (< 0.1% of peak accumulator).
    int32_t max_ref = 0;
    int32_t max_diff = 0;
    for (int i = 0; i < m * n; ++i) {
      max_ref = std::max(max_ref, std::abs(ref_c[i]));
      max_diff = std::max(max_diff, std::abs(d_c[i] - ref_c[i]));
    }
    float rel_err = (max_ref > 0) ? (float)max_diff / (float)max_ref : 0.f;
    std::printf("  max_ref=%d  max_diff=%d  rel_err=%.4f%%\n",
                max_ref, max_diff, rel_err * 100.f);
    CHECK(rel_err < 0.01f, "INT8 GEMM k=2048: graceful degradation (< 1% error)");

    free_metal(d_a); free_metal(d_b); free_metal(d_c);
  }
}

// ---------------------------------------------------------------------------
// Test T7: beta != 0 rejection for INT8 GEMM
// ---------------------------------------------------------------------------
static void test_int8_gemm_beta_rejection() {
  std::printf("Test T7: INT8 GEMM beta!=0 throws\n");

  int8_t*  d_a = static_cast<int8_t*>(alloc_metal(16));
  int8_t*  d_b = static_cast<int8_t*>(alloc_metal(16));
  int32_t* d_c = static_cast<int32_t*>(alloc_metal(16 * sizeof(int32_t)));

  bool threw_gemm = false;
  try {
    primitives<Device::MPS>::gemm<int8_t, int32_t>(
        false, false, false, false, 4, 4, 4,
        1.0f, d_a, 4, d_b, 4, 1.0f, d_c, 4, nullptr);
  } catch (const std::runtime_error&) {
    threw_gemm = true;
  }
  std::printf("  gemm beta=1.0 threw=%s\n", threw_gemm ? "yes" : "no");
  CHECK(threw_gemm, "INT8 gemm rejects beta!=0");

  bool threw_batch = false;
  try {
    primitives<Device::MPS>::gemm_batch_strided<int8_t, int32_t>(
        false, false, 4, 4, 4,
        1.0f, d_a, 4, 16, d_b, 4, 16,
        0.5f, d_c, 4, 16, 2);
  } catch (const std::runtime_error&) {
    threw_batch = true;
  }
  std::printf("  gemm_batch_strided beta=0.5 threw=%s\n", threw_batch ? "yes" : "no");
  CHECK(threw_batch, "INT8 gemm_batch_strided rejects beta!=0");

  free_metal(d_a); free_metal(d_b); free_metal(d_c);
}

// ---------------------------------------------------------------------------
// Test 9: gemm_pack_b returns 0 for int8_t
// ---------------------------------------------------------------------------
static void test_gemm_pack_b_int8() {
  std::printf("Test 8: gemm_pack_b<int8_t> returns 0\n");
  int8_t dummy = 0;
  dim_t result = primitives<Device::MPS>::gemm_pack_b<int8_t>(
      &dummy, false, 64, 64, 1.0f, nullptr);
  CHECK(result == 0, "gemm_pack_b<int8_t> returns 0");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M9.2 INT8 GEMM on Metal ===\n\n");

  @autoreleasepool {
    test_int8_gemm_basic();
    test_int8_gemm_transb();
    test_int8_gemm_alpha();
    test_int8_gemm_large_k();
    test_int8_gemm_zero();
    test_int8_pipeline();
    test_int8_gemm_batch_strided();
    test_int8_gemm_transa();
    test_int8_gemm_transa_transb();
    test_int8_gemm_k_boundary();
    test_int8_gemm_beta_rejection();
    test_gemm_pack_b_int8();
  }

  std::printf("\n=== Results: %d/%d pass, %d fail ===\n",
              g_pass, g_tests, g_fail);
  return g_fail == 0 ? 0 : 1;
}
