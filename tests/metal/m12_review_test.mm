// tests/metal/m12_review_test.mm
//
// M12 Code Review — Test coverage for M3 (INT8 rounding edge cases) and
// M5 (dequantize_gemm_output with activation functions post-M12.6).
//
// M3: Verifies that the GPU float32_round_to_int32_strided kernel uses
//     floor(x+0.5f) = "round half up" (toward +∞) at 0.5 boundaries.
//     This avoids banker's rounding (MSL round()) for positive halves;
//     for negative halves, floor(-0.5+0.5)=0 (rounds toward +∞, not away from zero).
//
// M5: Verifies that dequantize_gemm_output with tanh and gelu_tanh activations
//     produces correct results and no NaN values (regression test for M10.3
//     ct2_safe_tanh fix after M12.6 pipeline changes).
//
// Build command (from repo root):
//   clang++ -std=c++17 -O0 \
//       -I include -I src \
//       -DCT2_WITH_MPS \
//       tests/metal/m12_review_test.mm \
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
//       -framework Accelerate \
//       -o m12_review_test && ./m12_review_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

#include "ctranslate2/types.h"
#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "metal/utils.h"
#include "metal/ops_metal.h"

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

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// =========================================================================
// M3: INT8 GEMM rounding edge cases
//
// Tests that GPU floor(x+0.5f) rounding avoids banker's rounding at 0.5
// boundaries.  For negative halves, floor rounds toward +∞ (not away from 0).
// =========================================================================

static void test_int8_gemm_rounding() {
  std::printf("\n=== M3: INT8 GEMM Rounding Edge Cases ===\n");

  // Strategy: construct a 1×1 INT8 GEMM where the float32 result is exactly
  // at a 0.5 boundary.  GEMM computes: C = alpha * A * B.
  // For int8 GEMM: int32_result = round(float32_gemm_result).
  //
  // floor(x+0.5) behavior vs banker's rounding:
  //   0.5 → banker: 0, floor(1.0)=1   ← differs (key fix)
  //   1.5 → banker: 2, floor(2.0)=2   (agree)
  //   2.5 → banker: 2, floor(3.0)=3   ← differs (key fix)
  //   3.5 → banker: 4, floor(4.0)=4   (agree)
  //  -0.5 → banker: 0, floor(0.0)=0   (agree — round toward +∞)
  //  -2.5 → banker:-2, floor(-2.0)=-2 (agree — round toward +∞)

  // Use 1×1 GEMM with alpha to produce exact .5 values.
  // A=[1], B=[1], alpha=0.5 → result = 0.5 → floor(1.0) = 1
  // A=[1], B=[5], alpha=0.5 → result = 2.5 → floor(3.0) = 3
  // A=[1], B=[-1], alpha=0.5 → result = -0.5 → floor(0.0) = 0

  struct TestCase {
    int8_t a_val;
    int8_t b_val;
    float alpha;
    int32_t expected;  // floor(x + 0.5f) = round-half-up
    const char* label;
  };

  TestCase cases[] = {
    {1,  1, 0.5f,   1, "0.5 → 1 (not banker's 0)"},
    {1,  5, 0.5f,   3, "2.5 → 3 (not banker's 2)"},
    {1, -1, 0.5f,   0, "-0.5 → 0 (round-half-up toward +∞)"},
    {1, -5, 0.5f,  -2, "-2.5 → -2 (round-half-up toward +∞)"},
    {1,  3, 0.5f,   2, "1.5 → 2 (all modes agree)"},
    {1,  7, 0.5f,   4, "3.5 → 4 (all modes agree)"},
    {2,  3, 1.0f,   6, "6.0 → 6 (exact integer)"},
    {10, 10, 1.0f, 100, "100.0 → 100 (larger values)"},
  };

  int case_pass = 0;
  int case_total = sizeof(cases) / sizeof(cases[0]);

  for (auto& tc : cases) {
    int8_t* a = metal_alloc<int8_t>(1);
    int8_t* b = metal_alloc<int8_t>(1);
    int32_t* c = metal_alloc<int32_t>(1);

    std::memcpy(a, &tc.a_val, sizeof(int8_t));
    std::memcpy(b, &tc.b_val, sizeof(int8_t));

    // Run INT8 GEMM: C = alpha * A * B  (m=1, n=1, k=1)
    primitives<Device::MPS>::gemm(
        false, false,  // a_is_packed, b_is_packed
        false, false,  // transpose_a, transpose_b
        1, 1, 1,       // m, n, k
        tc.alpha,
        a, 1,          // a, lda
        b, 1,          // b, ldb
        0.0f,          // beta
        c, 1,          // c, ldc
        static_cast<const int32_t*>(nullptr));  // a_shift_compensation

    metal::commit_and_wait();

    int32_t result;
    std::memcpy(&result, c, sizeof(int32_t));

    if (result == tc.expected) {
      ++case_pass;
    } else {
      std::printf("    MISMATCH %s: got %d, expected %d\n", tc.label, result, tc.expected);
    }

    metal_free(a);
    metal_free(b);
    metal_free(c);
  }

  char buf[128];
  std::snprintf(buf, sizeof(buf), "INT8 GEMM rounding: %d/%d cases", case_pass, case_total);
  CHECK(buf, case_pass == case_total);
}

// =========================================================================
// M5: dequantize_gemm_output with activation functions
//
// Tests that tanh and gelu_tanh activations in dequantize_gemm_output
// produce correct results and no NaN values.
// Regression test for M10.3 ct2_safe_tanh fix.
// =========================================================================

static void test_dequantize_gemm_output_activations() {
  std::printf("\n=== M5: dequantize_gemm_output with Activations ===\n");

  const dim_t batch = 4;
  const dim_t depth = 8;
  const dim_t total = batch * depth;

  // Activation type mapping from ops/activation.h:
  //   1 = ReLU, 2 = GELUTanh, 5 = Tanh, 6 = GELU
  // The enum values:  ReLU=1, GELUSigmoid=2 (approx?), Tanh=5
  // Actually check the enum... Let's use the known values from dequantize kernel.
  // From quantize.metal: act_type 1=relu, 2=gelu_tanh, 5=tanh, ...
  // We test with -1 (none), then with tanh-related ones.

  // Allocate buffers
  int32_t* c_buf = metal_alloc<int32_t>(total);
  float* a_scales = metal_alloc<float>(batch);
  float* b_scales = metal_alloc<float>(depth);
  float* bias = metal_alloc<float>(depth);
  float* y_buf = metal_alloc<float>(total);

  // Fill: c values that would produce large dequantized values (testing tanh clamp).
  // c[i,j] = (i * depth + j) * 100 - 1600 → range [-1600, 1500]
  // a_scales[i] = 1.0, b_scales[j] = 1.0 → dequant = c / (a*b) = c
  // This means dequantized values go up to ±1600, well beyond tanh(±44) NaN range.
  std::vector<int32_t> c_host(total);
  for (dim_t i = 0; i < total; ++i)
    c_host[i] = (int32_t)(i * 100 - 1600);
  std::memcpy(c_buf, c_host.data(), total * sizeof(int32_t));

  std::vector<float> ascales(batch, 1.0f);
  std::vector<float> bscales(depth, 1.0f);
  std::vector<float> bias_host(depth, 0.0f);
  std::memcpy(a_scales, ascales.data(), batch * sizeof(float));
  std::memcpy(b_scales, bscales.data(), depth * sizeof(float));
  std::memcpy(bias, bias_host.data(), depth * sizeof(float));

  // Test 1: No activation — baseline
  {
    metal::dequantize_gemm_output_metal<float>(
        c_buf, a_scales, b_scales, bias, y_buf,
        batch, depth,
        false, false,   // transpose_a, transpose_b
        false,          // has_bias
        -1);            // activation_type: none
    metal::commit_and_wait();

    std::vector<float> y_host(total);
    std::memcpy(y_host.data(), y_buf, total * sizeof(float));

    bool no_nan = true;
    bool values_ok = true;
    for (dim_t i = 0; i < total; ++i) {
      if (std::isnan(y_host[i])) no_nan = false;
      // With unit scales and no bias: y = c / (1*1) = c
      float expected = (float)c_host[i];
      if (std::abs(y_host[i] - expected) > 0.5f) values_ok = false;
    }
    CHECK("dequant no-activation: no NaN", no_nan);
    CHECK("dequant no-activation: values correct", values_ok);
  }

  // Test 2: Tanh activation — values up to ±1600 must not produce NaN
  {
    metal::dequantize_gemm_output_metal<float>(
        c_buf, a_scales, b_scales, bias, y_buf,
        batch, depth,
        false, false,
        false,
        5);             // activation_type: Tanh
    metal::commit_and_wait();

    std::vector<float> y_host(total);
    std::memcpy(y_host.data(), y_buf, total * sizeof(float));

    bool no_nan = true;
    bool in_range = true;
    for (dim_t i = 0; i < total; ++i) {
      if (std::isnan(y_host[i])) {
        no_nan = false;
        std::printf("    NaN at [%lld]: input=%d\n", (long long)i, c_host[i]);
      }
      if (y_host[i] < -1.0f || y_host[i] > 1.0f) in_range = false;
    }
    CHECK("dequant tanh: no NaN (M10.3 regression)", no_nan);
    CHECK("dequant tanh: all outputs in [-1, 1]", in_range);
  }

  // Test 3: GELU (tanh approximation) — large inputs must not produce NaN
  {
    metal::dequantize_gemm_output_metal<float>(
        c_buf, a_scales, b_scales, bias, y_buf,
        batch, depth,
        false, false,
        false,
        2);             // activation_type: GELUTanh
    metal::commit_and_wait();

    std::vector<float> y_host(total);
    std::memcpy(y_host.data(), y_buf, total * sizeof(float));

    bool no_nan = true;
    for (dim_t i = 0; i < total; ++i) {
      if (std::isnan(y_host[i])) {
        no_nan = false;
        std::printf("    NaN at [%lld]: input=%d\n", (long long)i, c_host[i]);
      }
    }
    CHECK("dequant gelu_tanh: no NaN (M10.3 regression)", no_nan);
  }

  // Test 4: Tanh with bias — still no NaN
  {
    // Set bias to large values to push tanh input even further
    std::vector<float> large_bias(depth);
    for (dim_t j = 0; j < depth; ++j)
      large_bias[j] = (j % 2 == 0) ? 1000.0f : -1000.0f;
    std::memcpy(bias, large_bias.data(), depth * sizeof(float));

    metal::dequantize_gemm_output_metal<float>(
        c_buf, a_scales, b_scales, bias, y_buf,
        batch, depth,
        false, false,
        true,           // has_bias
        5);             // Tanh
    metal::commit_and_wait();

    std::vector<float> y_host(total);
    std::memcpy(y_host.data(), y_buf, total * sizeof(float));

    bool no_nan = true;
    for (dim_t i = 0; i < total; ++i) {
      if (std::isnan(y_host[i])) no_nan = false;
    }
    CHECK("dequant tanh+bias: no NaN (extreme inputs)", no_nan);
  }

  metal_free(c_buf);
  metal_free(a_scales);
  metal_free(b_scales);
  metal_free(bias);
  metal_free(y_buf);
}

// =========================================================================

int main() {
  std::printf("M12 Code Review Tests\n");
  std::printf("=====================\n");

  test_int8_gemm_rounding();
  test_dequantize_gemm_output_activations();

  std::printf("\n=====================\n");
  std::printf("%d passed, %d failed\n", passed, failed);
  return failed > 0 ? 1 : 0;
}
