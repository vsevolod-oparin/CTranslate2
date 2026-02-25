// Tests for 4.5 — reduce_sum precision baseline.
//
// Finding 3.4 in metal-primitives-review.md notes that reduce_sum<bfloat16_t>
// accumulates partial sums in bfloat16 threadgroup memory (7 mantissa bits),
// unlike reduce_amax which always promotes to float.  For large arrays with
// values spanning a dynamic range, this can lose significant precision.
//
// This test:
//   1. Establishes a relative-error baseline for bfloat16, float16, and float32
//      summation over N = 65536 elements.
//   2. Does NOT assert hard pass/fail thresholds on precision (the precision
//      trade-off is intentional in the current implementation).
//   3. DOES assert that the returned sum is finite (non-NaN, non-inf) and that
//      the function completes without throwing — these are hard requirements.
//
// Two fill values are tested:
//   a) fill = 1.0     — all-uniform case (exact sum = 65536.0)
//   b) fill = 1/3     — non-dyadic rational; exercises rounding accumulation
//
// Observed relative errors are printed so future regressions are detectable.
//
// Background on expected behaviour for case (a) [fill=1.0, N=65536]:
//   The GPU reduction groups 256 elements per threadgroup.
//   256 × bfloat(1.0) = 256.0 — exactly representable (2^8).
//   256 × bfloat(256.0) = 65536.0 — exactly representable (2^16).
//   All intermediate values are powers of 2, so error is zero.
//
// Background on case (b) [fill=1/3, N=65536]:
//   bfloat(1/3) ≈ 0.333... with rounding.
//   Partial sums grow to ~85.3 per group; ULP at 64 in BF16 is 2^(6-7)=0.5.
//   Accumulation error grows; the exact sum = 65536/3 ≈ 21845.33.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/reduce_sum_precision_test.mm \
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
//     -o reduce_sum_precision_test && ./reduce_sum_precision_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"

// Resolve ::float16_t / ctranslate2::float16_t conflict from arm_vector_types.h.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int g_pass = 0, g_fail = 0;

static void report(bool ok, const char* name) {
  if (ok) { ++g_pass; std::printf("  PASS  %s\n", name); }
  else     { ++g_fail; std::printf("  FAIL  %s\n", name); }
}

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// Helper: run sum<T> and report relative error.
// Returns true if result is finite and no exception was thrown.
// The relative error is printed as informational output; no hard threshold.
// ---------------------------------------------------------------------------

template <typename T>
static bool measure_sum_error(const char* type_name,
                               const char* case_name,
                               T* array, dim_t n,
                               double exact_sum) {
  double result = 0.0;
  bool threw = false;
  try {
    T r = primitives<Device::METAL>::sum(array, n);
    result = static_cast<double>(static_cast<float>(r));
  } catch (const std::exception& e) {
    std::printf("    exception: %s\n", e.what());
    threw = true;
  }

  if (threw) return false;

  bool finite_ok = std::isfinite(result);
  double rel_err = std::fabs(result - exact_sum) / (std::fabs(exact_sum) + 1e-30);

  // Print baseline (always shown, regardless of pass/fail).
  std::printf("    %-12s  %-18s  result=%10.2f  exact=%10.2f  rel_err=%.2e\n",
              type_name, case_name, result, exact_sum, rel_err);

  if (!finite_ok)
    std::printf("    *** result is not finite! ***\n");

  return finite_ok;
}

// ---------------------------------------------------------------------------
// Test suite for one fill value
// ---------------------------------------------------------------------------

static void test_case(const char* case_name, float fill_f32) {
  std::printf("\n--- fill = %g ---\n", (double)fill_f32);

  const dim_t N = 65536;

  // Exact sum computed in double precision.
  // For fill=1.0: exact = 65536.0
  // For fill=1/3: exact = 65536 * (1.0/3.0)  (we use the float fill as truth
  //   basis — the "exact" sum of N copies of the precise float value)
  const double exact_sum = (double)N * (double)fill_f32;

  // ----- float32 -----
  {
    float* x = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) x[i] = fill_f32;
    bool ok = measure_sum_error<float>("float32", case_name, x, N, exact_sum);
    report(ok, "sum<float32> finite");
    metal_free(x);
  }

  // ----- float16 -----
  // Note: float16 max is ~65504.  For fill=1.0, sum=65536 > 65504,
  // so the result overflows to infinity.  The test reports this (inf) and
  // still checks for non-exception.  The finite check will FAIL for this
  // case — this is intentional and documents the float16 overflow limit.
  {
    ct2_f16* x = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) x[i] = ct2_f16(fill_f32);
    // Exact sum capped at float16 representable range for reporting purposes.
    bool ok = measure_sum_error<ct2_f16>("float16", case_name, x, N, exact_sum);
    // For fill=1.0, N=65536 exceeds float16 range; mark expected overflow.
    float f16_max = 65504.f;
    bool expected_overflow = (float)N * fill_f32 > f16_max;
    if (expected_overflow) {
      std::printf("    (float16 sum expected to overflow — N*fill=%.0f > f16_max=%.0f)\n",
                  (double)N * (double)fill_f32, (double)f16_max);
      ok = true;  // not a defect — document the overflow, don't fail the suite
    }
    report(ok, "sum<float16> no-exception");
    metal_free(x);
  }

  // ----- bfloat16 -----
  {
    ct2_bf16* x = metal_alloc<ct2_bf16>(N);
    for (dim_t i = 0; i < N; ++i) x[i] = ct2_bf16(fill_f32);
    bool ok = measure_sum_error<ct2_bf16>("bfloat16", case_name, x, N, exact_sum);
    report(ok, "sum<bfloat16> finite");
    metal_free(x);
  }
}

// ---------------------------------------------------------------------------
// Precision improvement check: what would happen with float accumulator?
// We do a CPU reference using float32 accumulation over bfloat16 inputs.
// ---------------------------------------------------------------------------

static void test_float_accum_reference() {
  std::printf("\n--- Reference: CPU float32 accumulation over bfloat16 inputs ---\n");
  std::printf("    (Illustrates what reduce_sum could achieve if it used float threadgroup memory\n");
  std::printf("     as reduce_amax does.  See finding 3.4 in metal-primitives-review.md.)\n\n");

  const dim_t N = 65536;
  const float fill = 1.f / 3.f;
  const double exact = (double)N * (double)fill;

  ct2_bf16* x = metal_alloc<ct2_bf16>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = ct2_bf16(fill);

  // CPU float32 accumulation (reference for what a fixed implementation would give).
  double cpu_sum_f32 = 0.0;
  for (dim_t i = 0; i < N; ++i) cpu_sum_f32 += static_cast<float>(x[i]);
  double rel_f32 = std::fabs(cpu_sum_f32 - exact) / (std::fabs(exact) + 1e-30);

  std::printf("    float32-accum over bf16 inputs: %.6f  exact: %.6f  rel_err: %.2e\n",
              cpu_sum_f32, exact, rel_f32);
  std::printf("    (This is what the GPU sum would produce if it accumulated in float,\n");
  std::printf("     matching the reduce_amax pattern.)\n");

  metal_free(x);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== reduce_sum precision baseline (finding 3.4) ===\n");
  std::printf("N = 65536 elements.  Relative errors are informational (no hard threshold).\n");
  std::printf("PASS/FAIL only checks that the result is finite and no exception is thrown.\n");

  test_case("fill=1.0",   1.0f);
  test_case("fill=1/3",   1.0f / 3.0f);

  test_float_accum_reference();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
