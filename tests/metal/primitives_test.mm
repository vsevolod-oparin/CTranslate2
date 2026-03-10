// Standalone tests for M4.1: fill, strided_fill, indexed_fill, convert.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/primitives_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o primitives_test && ./primitives_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

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

// Allocate N elements of type T from the Metal allocator.
template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}


// ---------------------------------------------------------------------------
// fill<T>
// ---------------------------------------------------------------------------
static void test_fill() {
  std::printf("\n--- M4.1: primitives<METAL>::fill ---\n");
  const dim_t N = 64;

  // float32
  {
    float* p = metal_alloc<float>(N);
    CHECK_NOTHROW("fill<float>(3.14f) — no error",
      primitives<Device::METAL>::fill(p, 3.14f, N)
    );
    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(p[i] - 3.14f) > 1e-6f) { ok = false; break; }
    }
    CHECK("fill<float>: all elements == 3.14f", ok);
    metal_free(p);
  }

  // int32 (zero)
  {
    int32_t* p = metal_alloc<int32_t>(N);
    primitives<Device::METAL>::fill(p, int32_t(0), N);
    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (p[i] != 0) { ok = false; break; }
    }
    CHECK("fill<int32>(0): all elements == 0", ok);
    metal_free(p);
  }

  // int32 (non-zero sentinel)
  {
    int32_t* p = metal_alloc<int32_t>(N);
    primitives<Device::METAL>::fill(p, int32_t(42), N);
    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (p[i] != 42) { ok = false; break; }
    }
    CHECK("fill<int32>(42): all elements == 42", ok);
    metal_free(p);
  }

  // float16 (via half_float)
  {
    float16_t* p = metal_alloc<float16_t>(N);
    primitives<Device::METAL>::fill(p, float16_t(1.5f), N);
    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(p[i]) - 1.5f) > 1e-3f) { ok = false; break; }
    }
    CHECK("fill<float16>(1.5f): all elements ≈ 1.5f", ok);
    metal_free(p);
  }
}


// ---------------------------------------------------------------------------
// strided_fill<T>
// ---------------------------------------------------------------------------
static void test_strided_fill() {
  std::printf("\n--- M4.1: primitives<METAL>::strided_fill ---\n");

  const dim_t N = 8;
  float* p = metal_alloc<float>(N * 2);
  // Zero initialise
  for (dim_t i = 0; i < N * 2; ++i) { p[i] = 0.f; }

  // Fill every other element (stride 2) with 7.f
  primitives<Device::METAL>::strided_fill(p, 7.f, /*inc_x=*/2, N);

  bool even_ok = true, odd_zero = true;
  for (dim_t i = 0; i < N; ++i) {
    if (std::fabs(p[i * 2]     - 7.f) > 1e-6f) { even_ok  = false; }
    if (std::fabs(p[i * 2 + 1] - 0.f) > 1e-6f) { odd_zero = false; }
  }
  CHECK("strided_fill(stride=2): even positions == 7.f", even_ok);
  CHECK("strided_fill(stride=2): odd positions untouched (0.f)", odd_zero);

  metal_free(p);
}


// ---------------------------------------------------------------------------
// indexed_fill<T>
// ---------------------------------------------------------------------------
static void test_indexed_fill() {
  std::printf("\n--- M4.1: primitives<METAL>::indexed_fill ---\n");

  const dim_t N = 8;
  float* p = metal_alloc<float>(N);
  int32_t* idx = metal_alloc<int32_t>(3);

  for (dim_t i = 0; i < N; ++i) { p[i] = 0.f; }
  idx[0] = 1; idx[1] = 3; idx[2] = 5;

  primitives<Device::METAL>::indexed_fill(p, 9.f, idx, 3);
  metal::commit_and_wait();  // M11.25: indexed_fill is now encode-only

  CHECK("indexed_fill: p[1] == 9.f", std::fabs(p[1] - 9.f) < 1e-6f);
  CHECK("indexed_fill: p[3] == 9.f", std::fabs(p[3] - 9.f) < 1e-6f);
  CHECK("indexed_fill: p[5] == 9.f", std::fabs(p[5] - 9.f) < 1e-6f);
  CHECK("indexed_fill: p[0] untouched (0.f)", std::fabs(p[0]) < 1e-6f);
  CHECK("indexed_fill: p[2] untouched (0.f)", std::fabs(p[2]) < 1e-6f);

  metal_free(p);
  metal_free(idx);
}


// ---------------------------------------------------------------------------
// convert<U, V>
// ---------------------------------------------------------------------------
static void test_convert() {
  std::printf("\n--- M4.1: primitives<METAL>::convert ---\n");
  const dim_t N = 8;

  float src_f32[8] = {0.f, 0.5f, 1.f, -1.f, 2.f, -2.f, 100.f, -100.f};

  // float32 → float16 → float32 round-trip
  {
    float16_t* f16 = metal_alloc<float16_t>(N);
    float*     out = metal_alloc<float>(N);

    primitives<Device::METAL>::convert(src_f32, f16, N);
    primitives<Device::METAL>::convert(static_cast<const float16_t*>(f16), out, N);

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - src_f32[i]) > 0.1f) { ok = false; break; }
    }
    CHECK("convert float32→float16→float32 round-trip (tol 0.1)", ok);

    metal_free(f16);
    metal_free(out);
  }

  // float32 → bfloat16 → float32 round-trip
  {
    bfloat16_t* bf16 = metal_alloc<bfloat16_t>(N);
    float*      out  = metal_alloc<float>(N);

    primitives<Device::METAL>::convert(src_f32, bf16, N);
    primitives<Device::METAL>::convert(static_cast<const bfloat16_t*>(bf16), out, N);

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - src_f32[i]) > 0.1f) { ok = false; break; }
    }
    CHECK("convert float32→bfloat16→float32 round-trip (tol 0.1)", ok);

    metal_free(bf16);
    metal_free(out);
  }

  // float16 → bfloat16 → float16 round-trip
  {
    float16_t  f16_src[8];
    for (int i = 0; i < 8; ++i) { f16_src[i] = float16_t(src_f32[i]); }

    bfloat16_t* bf16   = metal_alloc<bfloat16_t>(N);
    float16_t*  f16out = metal_alloc<float16_t>(N);

    primitives<Device::METAL>::convert(
        static_cast<const float16_t*>(f16_src), bf16, N);
    primitives<Device::METAL>::convert(
        static_cast<const bfloat16_t*>(bf16), f16out, N);

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(f16out[i]) - src_f32[i]) > 0.1f) { ok = false; break; }
    }
    CHECK("convert float16→bfloat16→float16 round-trip (tol 0.1)", ok);

    metal_free(bf16);
    metal_free(f16out);
  }
}


int main() {
  std::printf("=== M4.1: Memory primitives (fill, strided_fill, indexed_fill, convert) ===\n");
  test_fill();
  test_strided_fill();
  test_indexed_fill();
  test_convert();
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
