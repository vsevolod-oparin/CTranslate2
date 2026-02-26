// tests/metal/m91_test.mm
//
// M9.1 — INT8 Quantize / Dequantize tests for Metal.
//
// Tests the GPU kernels via metal::quantize_int8_metal<T>() and
// metal::dequantize_int8_metal<T>() directly (no CPU dispatch deps).
//
// PASS criteria:
//   - Quantize → Dequantize round-trip error < 1% for all dtypes
//   - Scale values correct (scale == 127 / max_abs)
//   - dequantize_gemm_output: rescaling, bias, and relu activation correct
//   - compute_u8_compensation: no-op (no throw)
//   - gemm_pack_b: returns 0
//
// Build command (from repo root):
//   clang++ -std=c++17 -O0 \
//       -I include -I src \
//       -DCT2_WITH_METAL \
//       tests/metal/m91_test.mm \
//       src/metal/ops_quantize.mm \
//       src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//       src/metal/primitives_memory.mm \
//       src/metal/primitives_elementwise.mm \
//       src/metal/primitives_reduction.mm \
//       src/metal/primitives_gemm.mm \
//       src/metal/primitives_transpose.mm \
//       src/metal/primitives_beam_search.mm \
//       src/metal/primitives_norm_gather.mm \
//       src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//       -framework Metal -framework Foundation \
//       -framework MetalPerformanceShaders \
//       -framework MetalPerformanceShadersGraph \
//       -o m91_test && ./m91_test

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

// M5.1 fix: declare these before 'using namespace ctranslate2'
// to avoid ARM vector type ambiguity with arm_vector_types.h.
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

// Allocate a Metal-backed buffer via the registered allocator.
static void* alloc_metal(size_t n_bytes) {
  return get_allocator<Device::METAL>().allocate(n_bytes, 0);
}
static void free_metal(void* p) {
  get_allocator<Device::METAL>().free(p, 0);
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

// CPU reference: per-row INT8 quantize.
static void ref_quantize_f32(const float* in, int8_t* out, float* scales,
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

// CPU reference: per-row INT8 dequantize → float.
static void ref_dequantize_f32(const int8_t* in, const float* scales,
                                float* out, int batch, int depth) {
  for (int r = 0; r < batch; ++r)
    for (int c = 0; c < depth; ++c)
      out[r * depth + c] = static_cast<float>(in[r * depth + c]) / scales[r];
}

// CPU reference: dequantize_gemm_output (no bias, no activation).
static void ref_dequantize_gemm(const int32_t* c, const float* a_scales,
                                 const float* b_scales, float* y,
                                 int batch, int depth,
                                 bool ta, bool tb) {
  for (int i = 0; i < batch; ++i)
    for (int j = 0; j < depth; ++j) {
      float as = a_scales[ta ? j : i];
      float bs = b_scales[tb ? j : i];
      y[i * depth + j] = static_cast<float>(c[i * depth + j]) / (as * bs);
    }
}

// ---------------------------------------------------------------------------
// Test 1: float32 round-trip, random values in [-1, 1]
// ---------------------------------------------------------------------------
static void test_quantize_f32_roundtrip() {
  std::printf("Test 1: f32 round-trip (batch=4, depth=128)\n");

  const int B = 4, D = 128;
  const size_t n = (size_t)B * D;

  // Host data: fill with pseudo-random values in [-1, 1].
  std::vector<float> host_in(n);
  for (int i = 0; i < (int)n; ++i)
    host_in[i] = -1.f + 2.f * (float)(i % 127) / 127.f;

  // Allocate Metal buffers.
  float*   d_in     = static_cast<float*>(alloc_metal(n * sizeof(float)));
  int8_t*  d_out8   = static_cast<int8_t*>(alloc_metal(n * sizeof(int8_t)));
  float*   d_scales = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_deq    = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_in, host_in.data(), n * sizeof(float));

  // GPU quantize.
  metal::quantize_int8_metal<float>(d_in, d_out8, d_scales, B, D);
  // GPU dequantize.
  metal::dequantize_int8_metal<float>(d_out8, d_scales, d_deq, B, D);
  metal::commit_and_wait();

  // CPU reference.
  std::vector<int8_t> ref_q(n);
  std::vector<float>  ref_scales(B), ref_deq(n);
  ref_quantize_f32(host_in.data(), ref_q.data(), ref_scales.data(), B, D);
  ref_dequantize_f32(ref_q.data(), ref_scales.data(), ref_deq.data(), B, D);

  // Compare scales.
  float max_scale_err = 0.f;
  for (int r = 0; r < B; ++r)
    max_scale_err = std::max(max_scale_err, std::abs(d_scales[r] - ref_scales[r]));

  // Compare round-trip vs original.
  float max_abs_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_abs_err = std::max(max_abs_err, std::abs(d_deq[i] - host_in[i]));

  float max_rel = max_abs_err;  // input in [-1,1] so abs ≈ rel
  std::printf("  scale_err=%.2e  round-trip_abs_err=%.4f (%.4f%%)  %s\n",
              max_scale_err, max_abs_err, max_rel * 100.f,
              max_rel < 0.01f ? "PASS" : "FAIL");

  CHECK(max_scale_err < 1e-5f, "f32 scales match");
  CHECK(max_rel < 0.01f,       "f32 round-trip < 1%");

  free_metal(d_in); free_metal(d_out8);
  free_metal(d_scales); free_metal(d_deq);
}

// ---------------------------------------------------------------------------
// Test 2: float32, non-uniform data (one row has all zeros → scale stays 1)
// ---------------------------------------------------------------------------
static void test_quantize_f32_zero_row() {
  std::printf("Test 2: f32 zero-row (scale = 1 guard)\n");

  const int B = 3, D = 64;
  const size_t n = (size_t)B * D;

  std::vector<float> host_in(n, 0.f);
  // row 1: all zeros → scale should be 1.
  // row 0 and 2: non-zero.
  for (int c = 0; c < D; ++c) host_in[0 * D + c] = 0.5f;
  // row 1 stays 0
  for (int c = 0; c < D; ++c) host_in[2 * D + c] = -0.25f;

  float*  d_in     = static_cast<float*>(alloc_metal(n * sizeof(float)));
  int8_t* d_out8   = static_cast<int8_t*>(alloc_metal(n * sizeof(int8_t)));
  float*  d_scales = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*  d_deq    = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_in, host_in.data(), n * sizeof(float));
  metal::quantize_int8_metal<float>(d_in, d_out8, d_scales, B, D);
  metal::dequantize_int8_metal<float>(d_out8, d_scales, d_deq, B, D);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_err = std::max(max_err, std::abs(d_deq[i] - host_in[i]));

  bool scale_ok = (std::abs(d_scales[1] - 1.f) < 1e-5f);
  std::printf("  zero-row scale=%.6f  max_err=%.4f  %s\n",
              d_scales[1], max_err,
              (scale_ok && max_err < 0.01f) ? "PASS" : "FAIL");

  CHECK(scale_ok,         "zero-row scale == 1");
  CHECK(max_err < 0.01f,  "f32 zero-row round-trip < 1%");

  free_metal(d_in); free_metal(d_out8);
  free_metal(d_scales); free_metal(d_deq);
}

// ---------------------------------------------------------------------------
// Test 3: float16 round-trip
// ---------------------------------------------------------------------------
static void test_quantize_f16_roundtrip() {
  std::printf("Test 3: f16 round-trip (batch=2, depth=64)\n");

  const int B = 2, D = 64;
  const size_t n = (size_t)B * D;

  std::vector<float> host_f32(n);
  for (int i = 0; i < (int)n; ++i)
    host_f32[i] = -0.5f + (float)(i % 63) / 63.f;

  // Convert to f16 in Metal buffer.
  std::vector<ct2_f16> host_f16(n);
  for (int i = 0; i < (int)n; ++i)
    host_f16[i] = ct2_f16(host_f32[i]);

  ct2_f16* d_in     = static_cast<ct2_f16*>(alloc_metal(n * sizeof(ct2_f16)));
  int8_t*  d_out8   = static_cast<int8_t*>(alloc_metal(n * sizeof(int8_t)));
  float*   d_scales = static_cast<float*>(alloc_metal(B * sizeof(float)));
  ct2_f16* d_deq    = static_cast<ct2_f16*>(alloc_metal(n * sizeof(ct2_f16)));

  std::memcpy(d_in, host_f16.data(), n * sizeof(ct2_f16));
  metal::quantize_int8_metal<ct2_f16>(d_in, d_out8, d_scales, B, D);
  metal::dequantize_int8_metal<ct2_f16>(d_out8, d_scales, d_deq, B, D);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_err = std::max(max_err, std::abs(float(d_deq[i]) - host_f32[i]));

  float rel = max_err / 0.5f;  // normalise by range
  std::printf("  max_abs_err=%.4f  rel=%.4f%%  %s\n",
              max_err, rel * 100.f, rel < 0.01f ? "PASS" : "FAIL");

  CHECK(rel < 0.01f, "f16 round-trip < 1%");

  free_metal(d_in); free_metal(d_out8);
  free_metal(d_scales); free_metal(d_deq);
}

// ---------------------------------------------------------------------------
// Test 4: bfloat16 round-trip
// ---------------------------------------------------------------------------
static void test_quantize_bf16_roundtrip() {
  std::printf("Test 4: bf16 round-trip (batch=2, depth=64)\n");

  const int B = 2, D = 64;
  const size_t n = (size_t)B * D;

  std::vector<float> host_f32(n);
  for (int i = 0; i < (int)n; ++i)
    host_f32[i] = -0.8f + (float)(i % 63) / 63.f * 1.6f;

  std::vector<ct2_bf16> host_bf16(n);
  for (int i = 0; i < (int)n; ++i)
    host_bf16[i] = ct2_bf16(host_f32[i]);

  ct2_bf16* d_in     = static_cast<ct2_bf16*>(alloc_metal(n * sizeof(ct2_bf16)));
  int8_t*   d_out8   = static_cast<int8_t*>(alloc_metal(n * sizeof(int8_t)));
  float*    d_scales = static_cast<float*>(alloc_metal(B * sizeof(float)));
  ct2_bf16* d_deq    = static_cast<ct2_bf16*>(alloc_metal(n * sizeof(ct2_bf16)));

  std::memcpy(d_in, host_bf16.data(), n * sizeof(ct2_bf16));
  metal::quantize_int8_metal<ct2_bf16>(d_in, d_out8, d_scales, B, D);
  metal::dequantize_int8_metal<ct2_bf16>(d_out8, d_scales, d_deq, B, D);
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_err = std::max(max_err, std::abs(float(d_deq[i]) - host_f32[i]));

  float rel = max_err / 0.8f;
  std::printf("  max_abs_err=%.4f  rel=%.4f%%  %s\n",
              max_err, rel * 100.f, rel < 0.01f ? "PASS" : "FAIL");

  CHECK(rel < 0.01f, "bf16 round-trip < 1%");

  free_metal(d_in); free_metal(d_out8);
  free_metal(d_scales); free_metal(d_deq);
}

// ---------------------------------------------------------------------------
// Test 5: larger tensor (batch=8, depth=512)
// ---------------------------------------------------------------------------
static void test_quantize_f32_large() {
  std::printf("Test 5: f32 large (batch=8, depth=512)\n");

  const int B = 8, D = 512;
  const size_t n = (size_t)B * D;

  std::vector<float> host_in(n);
  for (int i = 0; i < (int)n; ++i)
    host_in[i] = std::sin((float)i * 0.1f);

  float*  d_in     = static_cast<float*>(alloc_metal(n * sizeof(float)));
  int8_t* d_out8   = static_cast<int8_t*>(alloc_metal(n * sizeof(int8_t)));
  float*  d_scales = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*  d_deq    = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_in, host_in.data(), n * sizeof(float));
  metal::quantize_int8_metal<float>(d_in, d_out8, d_scales, B, D);
  metal::dequantize_int8_metal<float>(d_out8, d_scales, d_deq, B, D);
  metal::commit_and_wait();

  // Compare GPU scales vs CPU reference.
  std::vector<int8_t> ref_q(n);
  std::vector<float>  ref_sc(B);
  ref_quantize_f32(host_in.data(), ref_q.data(), ref_sc.data(), B, D);

  float max_scale_err = 0.f;
  for (int r = 0; r < B; ++r)
    max_scale_err = std::max(max_scale_err, std::abs(d_scales[r] - ref_sc[r]));

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_err = std::max(max_err, std::abs(d_deq[i] - host_in[i]));

  std::printf("  max_scale_err=%.2e  max_abs_err=%.6f (%.4f%%)  %s\n",
              max_scale_err, max_err, max_err * 100.f,
              max_err < 0.01f ? "PASS" : "FAIL");

  CHECK(max_scale_err < 1e-5f, "large f32 scales match");
  CHECK(max_err < 0.01f,       "large f32 round-trip < 1%");

  free_metal(d_in); free_metal(d_out8);
  free_metal(d_scales); free_metal(d_deq);
}

// ---------------------------------------------------------------------------
// Test 6: dequantize_gemm_output — basic (no bias, no activation)
// ---------------------------------------------------------------------------
static void test_dequantize_gemm_output_basic() {
  std::printf("Test 6: dequantize_gemm_output f32 (no bias, no activation)\n");

  const int B = 4, D = 8;
  const size_t n = (size_t)B * D;

  // Build int32 c matrix and scales.
  std::vector<int32_t> host_c(n);
  std::vector<float>   host_as(B), host_bs(D);
  for (int i = 0; i < (int)n; ++i)  host_c[i]  = (i % 100) - 50;
  for (int i = 0; i < B; ++i)       host_as[i] = 1.f + (float)i * 0.1f;
  for (int j = 0; j < D; ++j)       host_bs[j] = 2.f + (float)j * 0.2f;

  int32_t* d_c   = static_cast<int32_t*>(alloc_metal(n * sizeof(int32_t)));
  float*   d_as  = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_bs  = static_cast<float*>(alloc_metal(D * sizeof(float)));
  float*   d_y   = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_c,  host_c.data(),  n * sizeof(int32_t));
  std::memcpy(d_as, host_as.data(), B * sizeof(float));
  std::memcpy(d_bs, host_bs.data(), D * sizeof(float));

  metal::dequantize_gemm_output_metal<float>(
      d_c, d_as, d_bs,
      static_cast<const void*>(d_c),  // dummy bias (has_bias=false)
      d_y,
      B, D,
      /*transpose_a=*/false, /*transpose_b=*/false,
      /*has_bias=*/false, /*activation_type=*/-1);
  metal::commit_and_wait();

  // CPU reference.
  std::vector<float> ref_y(n);
  ref_dequantize_gemm(host_c.data(), host_as.data(), host_bs.data(),
                      ref_y.data(), B, D, false, false);

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i)
    max_err = std::max(max_err, std::abs(d_y[i] - ref_y[i]));

  std::printf("  max_abs_err=%.2e  %s\n", max_err,
              max_err < 1e-4f ? "PASS" : "FAIL");
  CHECK(max_err < 1e-4f, "dequantize_gemm_output basic");

  free_metal(d_c); free_metal(d_as); free_metal(d_bs); free_metal(d_y);
}

// ---------------------------------------------------------------------------
// Test 7: dequantize_gemm_output — with bias
// ---------------------------------------------------------------------------
static void test_dequantize_gemm_output_bias() {
  std::printf("Test 7: dequantize_gemm_output f32 (with bias)\n");

  const int B = 3, D = 6;
  const size_t n = (size_t)B * D;

  std::vector<int32_t> host_c(n);
  std::vector<float>   host_as(B), host_bs(B), host_bias(D);
  for (int i = 0; i < (int)n; ++i)  host_c[i]      = i * 10;
  for (int i = 0; i < B; ++i)       host_as[i]     = 2.f;
  for (int i = 0; i < B; ++i)       host_bs[i]     = 3.f;
  for (int j = 0; j < D; ++j)       host_bias[j]   = (float)j * 0.5f;

  int32_t* d_c    = static_cast<int32_t*>(alloc_metal(n * sizeof(int32_t)));
  float*   d_as   = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_bs   = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_bias = static_cast<float*>(alloc_metal(D * sizeof(float)));
  float*   d_y    = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_c,    host_c.data(),    n * sizeof(int32_t));
  std::memcpy(d_as,   host_as.data(),   B * sizeof(float));
  std::memcpy(d_bs,   host_bs.data(),   B * sizeof(float));
  std::memcpy(d_bias, host_bias.data(), D * sizeof(float));

  metal::dequantize_gemm_output_metal<float>(
      d_c, d_as, d_bs,
      static_cast<const void*>(d_bias),
      d_y,
      B, D,
      false, false,
      /*has_bias=*/true, /*activation_type=*/-1);
  metal::commit_and_wait();

  // CPU reference.
  float max_err = 0.f;
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < D; ++j) {
      float ref = (float)host_c[i*D+j] / (host_as[i] * host_bs[i])
                + host_bias[j];
      max_err = std::max(max_err, std::abs(d_y[i*D+j] - ref));
    }

  std::printf("  max_abs_err=%.2e  %s\n", max_err,
              max_err < 1e-4f ? "PASS" : "FAIL");
  CHECK(max_err < 1e-4f, "dequantize_gemm_output with bias");

  free_metal(d_c); free_metal(d_as); free_metal(d_bs);
  free_metal(d_bias); free_metal(d_y);
}

// ---------------------------------------------------------------------------
// Test 8: dequantize_gemm_output — with ReLU activation
// ---------------------------------------------------------------------------
static void test_dequantize_gemm_output_relu() {
  std::printf("Test 8: dequantize_gemm_output f32 (relu activation)\n");

  const int B = 4, D = 8;
  const size_t n = (size_t)B * D;

  // Values span positive and negative.
  std::vector<int32_t> host_c(n);
  std::vector<float>   host_as(B), host_bs(B);
  for (int i = 0; i < (int)n; ++i)  host_c[i]  = i * 5 - (int)(n/2);
  for (int i = 0; i < B; ++i)       host_as[i] = 1.f;
  for (int i = 0; i < B; ++i)       host_bs[i] = 1.f;

  int32_t* d_c   = static_cast<int32_t*>(alloc_metal(n * sizeof(int32_t)));
  float*   d_as  = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_bs  = static_cast<float*>(alloc_metal(B * sizeof(float)));
  float*   d_y   = static_cast<float*>(alloc_metal(n * sizeof(float)));

  std::memcpy(d_c,  host_c.data(),  n * sizeof(int32_t));
  std::memcpy(d_as, host_as.data(), B * sizeof(float));
  std::memcpy(d_bs, host_bs.data(), B * sizeof(float));

  // ActivationType::ReLU == 0, so activation_type = 0.
  metal::dequantize_gemm_output_metal<float>(
      d_c, d_as, d_bs,
      static_cast<const void*>(d_c),  // dummy bias
      d_y,
      B, D,
      false, false,
      /*has_bias=*/false, /*activation_type=*/0);  // 0 = ReLU
  metal::commit_and_wait();

  float max_err = 0.f;
  for (int i = 0; i < (int)n; ++i) {
    float ref = std::max(0.f, (float)host_c[i]);  // a_s=b_s=1, no bias
    max_err = std::max(max_err, std::abs(d_y[i] - ref));
  }

  // Also check no negative values.
  float min_y = *std::min_element(d_y, d_y + n);
  std::printf("  max_abs_err=%.2e  min_y=%.4f  %s\n",
              max_err, min_y,
              (max_err < 1e-4f && min_y >= 0.f) ? "PASS" : "FAIL");
  CHECK(max_err < 1e-4f, "dequantize_gemm_output relu values");
  CHECK(min_y >= 0.f,    "dequantize_gemm_output relu non-negative");

  free_metal(d_c); free_metal(d_as); free_metal(d_bs); free_metal(d_y);
}

// ---------------------------------------------------------------------------
// Test 9: gemm_pack_b returns 0 (9.3)
// ---------------------------------------------------------------------------
static void test_gemm_pack_b_zero() {
  std::printf("Test 9: gemm_pack_b returns 0\n");
  dim_t result = primitives<Device::METAL>::gemm_pack_b<float>(
      nullptr, false, 16, 16, 1.f, nullptr);
  std::printf("  gemm_pack_b = %lld  %s\n", (long long)result,
              result == 0 ? "PASS" : "FAIL");
  CHECK(result == 0, "gemm_pack_b returns 0");
}

// ---------------------------------------------------------------------------
// Test 10: compute_u8_compensation is a no-op (9.4)
// ---------------------------------------------------------------------------
static void test_compute_u8_compensation_noop() {
  std::printf("Test 10: compute_u8_compensation is a no-op\n");
  // Allocate a tiny int32 buffer and verify it's unchanged after the call.
  int32_t* buf = static_cast<int32_t*>(alloc_metal(4 * sizeof(int32_t)));
  buf[0] = 42; buf[1] = 43; buf[2] = 44; buf[3] = 45;

  bool threw = false;
  try {
    primitives<Device::METAL>::compute_u8_compensation(
        nullptr, false, 4, 4, 1.f, buf);
  } catch (...) {
    threw = true;
  }

  bool unchanged = (!threw) && buf[0] == 42 && buf[1] == 43;
  std::printf("  threw=%s  buf[0]=%d  %s\n",
              threw ? "yes" : "no", buf[0],
              unchanged ? "PASS" : "FAIL");
  CHECK(!threw,    "compute_u8_compensation does not throw");
  CHECK(unchanged, "compute_u8_compensation does not modify buffer");

  free_metal(buf);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
  std::printf("=== M9.1 Metal Quantize/Dequantize Tests ===\n\n");

  @autoreleasepool {
    test_quantize_f32_roundtrip();
    test_quantize_f32_zero_row();
    test_quantize_f16_roundtrip();
    test_quantize_bf16_roundtrip();
    test_quantize_f32_large();
    test_dequantize_gemm_output_basic();
    test_dequantize_gemm_output_bias();
    test_dequantize_gemm_output_relu();
    test_gemm_pack_b_zero();
    test_compute_u8_compensation_noop();
  }

  std::printf("\nResults: %d/%d pass", g_pass, g_tests);
  if (g_fail) std::printf("  (%d FAIL)", g_fail);
  std::printf("\n");

  return g_fail ? 1 : 0;
}
