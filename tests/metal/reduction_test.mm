// Standalone tests for M4.3: sum, max_element, max, amax.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/reduction_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o reduction_test && ./reduction_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// arm_vector_types.h (pulled in by Metal headers) defines ::float16_t as __fp16,
// conflicting with ctranslate2::float16_t when using namespace ctranslate2.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int passed = 0;
static int failed = 0;

#define CHECK(label, expr)                              \
  do {                                                  \
    if (expr) {                                         \
      std::printf("  PASS  %s\n", label);               \
      ++passed;                                         \
    } else {                                            \
      std::printf("  FAIL  %s\n", label);               \
      ++failed;                                         \
    }                                                   \
  } while (0)

#define CHECK_NOTHROW(label, ...)                       \
  do {                                                  \
    bool ok = true;                                     \
    try { __VA_ARGS__; }                                \
    catch (const std::exception& e) {                   \
      std::printf("  FAIL  %s — threw: %s\n",           \
                  label, e.what());                     \
      ok = false;                                       \
    }                                                   \
    if (ok) { std::printf("  PASS  %s\n", label); ++passed; } \
    else     { ++failed; }                              \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// Commit all encoded GPU commands and wait for completion.
static void gpu_sync() {
  metal::commit_and_wait();
}


// ---------------------------------------------------------------------------
// M4.3: sum
// ---------------------------------------------------------------------------

static void test_sum() {
  std::printf("\n--- M4.3: primitives<METAL>::sum ---\n");

  // float32: [1,2,3,4,5,6,7,8] → sum = 36
  {
    const dim_t N = 8;
    float* p = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { p[i] = static_cast<float>(i + 1); }

    float result = 0.f;
    CHECK_NOTHROW("sum<float> — no error",
      result = primitives<Device::MPS>::sum(p, N)
    );
    CHECK("sum<float>: [1..8] == 36", std::fabs(result - 36.f) < 1e-5f);

    metal_free(p);
  }

  // int32: [10,20,30,40] → sum = 100
  {
    const dim_t N = 4;
    int32_t* p = metal_alloc<int32_t>(N);
    p[0] = 10; p[1] = 20; p[2] = 30; p[3] = 40;

    int32_t result = 0;
    CHECK_NOTHROW("sum<int32> — no error",
      result = primitives<Device::MPS>::sum(p, N)
    );
    CHECK("sum<int32>: [10,20,30,40] == 100", result == 100);

    metal_free(p);
  }

  // float16: [1,2,3,4] → sum ≈ 10  (tolerance for fp16 accumulation)
  {
    const dim_t N = 4;
    ct2_f16* p = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) { p[i] = ct2_f16(static_cast<float>(i + 1)); }

    ct2_f16 result(0.f);
    CHECK_NOTHROW("sum<float16> — no error",
      result = primitives<Device::MPS>::sum(p, N)
    );
    CHECK("sum<float16>: [1,2,3,4] ≈ 10",
          std::fabs(static_cast<float>(result) - 10.f) < 0.1f);

    metal_free(p);
  }

  // zero-size: sum of empty → 0
  {
    float* p = metal_alloc<float>(1);
    p[0] = 99.f;
    float result = primitives<Device::MPS>::sum(p, 0);
    CHECK("sum size=0 == 0", std::fabs(result) < 1e-6f);
    metal_free(p);
  }
}


// ---------------------------------------------------------------------------
// M4.3: max_element
// ---------------------------------------------------------------------------

static void test_max_element() {
  std::printf("\n--- M4.3: primitives<METAL>::max_element ---\n");

  // float32: max is at index 3
  {
    const dim_t N = 6;
    float* p = metal_alloc<float>(N);
    p[0] = 1.f; p[1] = 3.f; p[2] = 2.f;
    p[3] = 9.f; p[4] = 4.f; p[5] = 0.f;

    dim_t idx = 0;
    CHECK_NOTHROW("max_element<float> — no error",
      idx = primitives<Device::MPS>::max_element(p, N)
    );
    CHECK("max_element<float>: index == 3", idx == 3);

    metal_free(p);
  }

  // int32: max at last position
  {
    const dim_t N = 4;
    int32_t* p = metal_alloc<int32_t>(N);
    p[0] = -5; p[1] = 0; p[2] = 3; p[3] = 100;

    dim_t idx = primitives<Device::MPS>::max_element(p, N);
    CHECK("max_element<int32>: max at last index (3)", idx == 3);

    metal_free(p);
  }

  // float32: all equal → first occurrence (index 0)
  {
    const dim_t N = 4;
    float* p = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { p[i] = 5.f; }

    dim_t idx = primitives<Device::MPS>::max_element(p, N);
    CHECK("max_element<float>: all equal → index 0", idx == 0);

    metal_free(p);
  }

  // zero-size → 0
  {
    float* p = metal_alloc<float>(1);
    p[0] = 1.f;
    dim_t idx = primitives<Device::MPS>::max_element(p, 0);
    CHECK("max_element size=0 → 0", idx == 0);
    metal_free(p);
  }
}


// ---------------------------------------------------------------------------
// M4.3: max (scalar max value)
// ---------------------------------------------------------------------------

static void test_max() {
  std::printf("\n--- M4.3: primitives<METAL>::max (scalar) ---\n");

  // float32
  {
    const dim_t N = 5;
    float* p = metal_alloc<float>(N);
    p[0] = -3.f; p[1] = 7.f; p[2] = 2.f; p[3] = 1.f; p[4] = 5.f;

    float result = 0.f;
    CHECK_NOTHROW("max<float> — no error",
      result = primitives<Device::MPS>::max(p, N)
    );
    CHECK("max<float>: max == 7.f", std::fabs(result - 7.f) < 1e-5f);

    metal_free(p);
  }

  // int32 with negatives
  {
    const dim_t N = 4;
    int32_t* p = metal_alloc<int32_t>(N);
    p[0] = -10; p[1] = -1; p[2] = -5; p[3] = -2;

    int32_t result = primitives<Device::MPS>::max(p, N);
    CHECK("max<int32>: max of all-negative == -1", result == -1);

    metal_free(p);
  }

  // zero-size → T(0)
  {
    float* p = metal_alloc<float>(1);
    p[0] = 99.f;
    float result = primitives<Device::MPS>::max(p, 0);
    CHECK("max size=0 == 0", std::fabs(result) < 1e-6f);
    metal_free(p);
  }
}


// ---------------------------------------------------------------------------
// M4.3: amax (max of absolute values)
// ---------------------------------------------------------------------------

static void test_amax() {
  std::printf("\n--- M4.3: primitives<METAL>::amax ---\n");

  // float32: all positive
  {
    const dim_t N = 4;
    float* p = metal_alloc<float>(N);
    p[0] = 1.f; p[1] = 3.f; p[2] = 2.f; p[3] = 0.5f;

    float result = 0.f;
    CHECK_NOTHROW("amax<float> all positive — no error",
      result = primitives<Device::MPS>::amax(p, N)
    );
    CHECK("amax<float> all positive == 3.f", std::fabs(result - 3.f) < 1e-5f);

    metal_free(p);
  }

  // float32: negative dominant
  {
    const dim_t N = 4;
    float* p = metal_alloc<float>(N);
    p[0] = 1.f; p[1] = -8.f; p[2] = 2.f; p[3] = 3.f;

    float result = primitives<Device::MPS>::amax(p, N);
    CHECK("amax<float> neg dominant == 8.f", std::fabs(result - 8.f) < 1e-5f);

    metal_free(p);
  }

  // float32: mixed signs, positive dominant
  {
    const dim_t N = 5;
    float* p = metal_alloc<float>(N);
    p[0] = -1.f; p[1] = 4.f; p[2] = -3.f; p[3] = 0.f; p[4] = 2.f;

    float result = primitives<Device::MPS>::amax(p, N);
    CHECK("amax<float> mixed: amax == 4.f", std::fabs(result - 4.f) < 1e-5f);

    metal_free(p);
  }

  // float16: tolerance 0.1
  {
    const dim_t N = 4;
    ct2_f16* p = metal_alloc<ct2_f16>(N);
    p[0] = ct2_f16(1.f); p[1] = ct2_f16(-6.f);
    p[2] = ct2_f16(2.f); p[3] = ct2_f16(3.f);

    ct2_f16 result = primitives<Device::MPS>::amax(p, N);
    CHECK("amax<float16> neg dominant ≈ 6.f",
          std::fabs(static_cast<float>(result) - 6.f) < 0.1f);

    metal_free(p);
  }

  // zero-size → 0
  {
    float* p = metal_alloc<float>(1);
    p[0] = 99.f;
    float result = primitives<Device::MPS>::amax(p, 0);
    CHECK("amax size=0 == 0", std::fabs(result) < 1e-6f);
    metal_free(p);
  }
}


// ---------------------------------------------------------------------------
// M4.3: commit_and_wait behaviour — reductions after GPU add
// ---------------------------------------------------------------------------

static void test_reduction_after_gpu_op() {
  std::printf("\n--- M4.3: reduction after GPU arithmetic (commit_and_wait) ---\n");

  // Encode a GPU add_scalar op (x[i] += 10), then immediately call
  // sum without an explicit synchronize_stream.  The reduction must
  // internally commit_and_wait so the GPU write is visible before the
  // CPU reads.
  const dim_t N = 4;
  float* x   = metal_alloc<float>(N);
  float* out = metal_alloc<float>(N);
  for (dim_t i = 0; i < N; ++i) { x[i] = static_cast<float>(i + 1); }

  // Encode GPU op: out[i] = 10 + x[i] = [11, 12, 13, 14]
  primitives<Device::MPS>::add(10.f, x, out, N);
  // No explicit gpu_sync here — sum() must handle it.

  float s = primitives<Device::MPS>::sum(out, N);
  CHECK("sum after GPU add (no explicit sync): sum([11..14]) == 50",
        std::fabs(s - 50.f) < 1e-4f);

  // Similarly test max after another GPU op
  primitives<Device::MPS>::add(100.f, x, out, N);  // [101,102,103,104]
  float m = primitives<Device::MPS>::max(out, N);
  CHECK("max after GPU add: max([101..104]) == 104",
        std::fabs(m - 104.f) < 1e-4f);

  // amax after GPU sub (result may have negatives)
  primitives<Device::MPS>::sub(
      static_cast<const float*>(out),   // [101..104] still from last sync
      static_cast<const float*>(out),
      out, N);  // out = 0 - 0 = 0 ... wait, sub uses out as both src and dst
  // Actually let's build a cleaner case:
  // out = [-5, -4, -3, -2] after: out[i] = x[i] - 6  where x = [1,2,3,4]
  primitives<Device::MPS>::add(-6.f, x, out, N);  // out = [-5,-4,-3,-2]
  float am = primitives<Device::MPS>::amax(out, N);
  CHECK("amax after GPU add-neg: amax([-5,-4,-3,-2]) == 5",
        std::fabs(am - 5.f) < 1e-4f);

  metal_free(x);
  metal_free(out);
}


int main() {
  std::printf("=== M4.3: Reduction primitives (sum, max_element, max, amax) ===\n");
  test_sum();
  test_max_element();
  test_max();
  test_amax();
  test_reduction_after_gpu_op();
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
