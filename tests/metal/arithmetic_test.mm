// Standalone tests for M4.2: add, sub, mul (scalar and vector forms).
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/arithmetic_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o arithmetic_test && ./arithmetic_test

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

// arm_vector_types.h (pulled in by Metal headers) defines ::float16_t as
// __fp16, which conflicts with ctranslate2::float16_t (= half_float::half)
// when using namespace ctranslate2.  Avoid the name entirely in this file
// by aliasing to ct2_f16 / ct2_bf16 before bringing ctranslate2 into scope.
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
// add(scalar, vector, out)   →  out[i] = scalar + x[i]
// ---------------------------------------------------------------------------
static void test_add_scalar() {
  std::printf("\n--- M4.2: primitives<METAL>::add(scalar, vec, out) ---\n");
  const dim_t N = 16;

  // float32
  {
    float* x   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = static_cast<float>(i); }

    CHECK_NOTHROW("add_scalar float — no error",
      primitives<Device::MPS>::add(10.f, x, out, N)
    );
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - (10.f + static_cast<float>(i))) > 1e-5f) { ok = false; break; }
    }
    CHECK("add_scalar float: out[i] == 10 + i", ok);

    metal_free(x);
    metal_free(out);
  }

  // float16
  {
    ct2_f16* x   = metal_alloc<ct2_f16>(N);
    ct2_f16* out = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = ct2_f16(static_cast<float>(i)); }

    primitives<Device::MPS>::add(ct2_f16(5.f), x, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(out[i]) - (5.f + static_cast<float>(i))) > 0.1f) { ok = false; break; }
    }
    CHECK("add_scalar float16: out[i] ≈ 5 + i", ok);

    metal_free(x);
    metal_free(out);
  }

  // int32
  {
    int32_t* x   = metal_alloc<int32_t>(N);
    int32_t* out = metal_alloc<int32_t>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = static_cast<int32_t>(i); }

    primitives<Device::MPS>::add(int32_t(100), x, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (out[i] != 100 + static_cast<int32_t>(i)) { ok = false; break; }
    }
    CHECK("add_scalar int32: out[i] == 100 + i", ok);

    metal_free(x);
    metal_free(out);
  }
}


// ---------------------------------------------------------------------------
// add(vec, vec, out)   →  out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
static void test_add_vec() {
  std::printf("\n--- M4.2: primitives<METAL>::add(vec, vec, out) ---\n");
  const dim_t N = 16;

  // float32
  {
    float* a   = metal_alloc<float>(N);
    float* b   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { a[i] = static_cast<float>(i); b[i] = 1.f; }

    CHECK_NOTHROW("add_vec float — no error",
      primitives<Device::MPS>::add(
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          out, N)
    );
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - (static_cast<float>(i) + 1.f)) > 1e-5f) { ok = false; break; }
    }
    CHECK("add_vec float: out[i] == i + 1", ok);

    metal_free(a);
    metal_free(b);
    metal_free(out);
  }

  // float16
  {
    ct2_f16* a   = metal_alloc<ct2_f16>(N);
    ct2_f16* b   = metal_alloc<ct2_f16>(N);
    ct2_f16* out = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) {
      a[i] = ct2_f16(static_cast<float>(i));
      b[i] = ct2_f16(2.f);
    }

    primitives<Device::MPS>::add(
        static_cast<const ct2_f16*>(a),
        static_cast<const ct2_f16*>(b),
        out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(out[i]) - (static_cast<float>(i) + 2.f)) > 0.1f) { ok = false; break; }
    }
    CHECK("add_vec float16: out[i] ≈ i + 2", ok);

    metal_free(a);
    metal_free(b);
    metal_free(out);
  }
}


// ---------------------------------------------------------------------------
// sub(vec, vec, out)   →  out[i] = a[i] - b[i]
// ---------------------------------------------------------------------------
static void test_sub_vec() {
  std::printf("\n--- M4.2: primitives<METAL>::sub(vec, vec, out) ---\n");
  const dim_t N = 16;

  // float32
  {
    float* a   = metal_alloc<float>(N);
    float* b   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) {
      a[i] = static_cast<float>(i) * 2.f;
      b[i] = static_cast<float>(i);
    }

    CHECK_NOTHROW("sub_vec float — no error",
      primitives<Device::MPS>::sub(
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          out, N)
    );
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - static_cast<float>(i)) > 1e-5f) { ok = false; break; }
    }
    CHECK("sub_vec float: out[i] == i", ok);

    metal_free(a);
    metal_free(b);
    metal_free(out);
  }

  // Verify sub(a, a) == 0
  {
    float* a   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { a[i] = static_cast<float>(i) + 1.f; }

    primitives<Device::MPS>::sub(
        static_cast<const float*>(a),
        static_cast<const float*>(a),
        out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i]) > 1e-5f) { ok = false; break; }
    }
    CHECK("sub_vec float: a - a == 0", ok);

    metal_free(a);
    metal_free(out);
  }
}


// ---------------------------------------------------------------------------
// mul(scalar, vector, out)   →  out[i] = scalar * x[i]
// ---------------------------------------------------------------------------
static void test_mul_scalar() {
  std::printf("\n--- M4.2: primitives<METAL>::mul(scalar, vec, out) ---\n");
  const dim_t N = 16;

  // float32
  {
    float* x   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = static_cast<float>(i) + 1.f; }

    CHECK_NOTHROW("mul_scalar float — no error",
      primitives<Device::MPS>::mul(3.f, x, out, N)
    );
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - 3.f * (static_cast<float>(i) + 1.f)) > 1e-4f) { ok = false; break; }
    }
    CHECK("mul_scalar float: out[i] == 3*(i+1)", ok);

    metal_free(x);
    metal_free(out);
  }

  // float16
  {
    ct2_f16* x   = metal_alloc<ct2_f16>(N);
    ct2_f16* out = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = ct2_f16(static_cast<float>(i) + 1.f); }

    primitives<Device::MPS>::mul(ct2_f16(2.f), x, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(out[i]) - 2.f * (static_cast<float>(i) + 1.f)) > 0.2f) { ok = false; break; }
    }
    CHECK("mul_scalar float16: out[i] ≈ 2*(i+1)", ok);

    metal_free(x);
    metal_free(out);
  }

  // int32
  {
    int32_t* x   = metal_alloc<int32_t>(N);
    int32_t* out = metal_alloc<int32_t>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = static_cast<int32_t>(i) + 1; }

    primitives<Device::MPS>::mul(int32_t(4), x, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (out[i] != 4 * (static_cast<int32_t>(i) + 1)) { ok = false; break; }
    }
    CHECK("mul_scalar int32: out[i] == 4*(i+1)", ok);

    metal_free(x);
    metal_free(out);
  }
}


// ---------------------------------------------------------------------------
// mul(vec, vec, out)   →  out[i] = a[i] * b[i]
// ---------------------------------------------------------------------------
static void test_mul_vec() {
  std::printf("\n--- M4.2: primitives<METAL>::mul(vec, vec, out) ---\n");
  const dim_t N = 16;

  // float32
  {
    float* a   = metal_alloc<float>(N);
    float* b   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) {
      a[i] = static_cast<float>(i) + 1.f;
      b[i] = 2.f;
    }

    CHECK_NOTHROW("mul_vec float — no error",
      primitives<Device::MPS>::mul(
          static_cast<const float*>(a),
          static_cast<const float*>(b),
          out, N)
    );
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(out[i] - 2.f * (static_cast<float>(i) + 1.f)) > 1e-4f) { ok = false; break; }
    }
    CHECK("mul_vec float: out[i] == 2*(i+1)", ok);

    metal_free(a);
    metal_free(b);
    metal_free(out);
  }

  // float16
  {
    ct2_f16* a   = metal_alloc<ct2_f16>(N);
    ct2_f16* b   = metal_alloc<ct2_f16>(N);
    ct2_f16* out = metal_alloc<ct2_f16>(N);
    for (dim_t i = 0; i < N; ++i) {
      a[i] = ct2_f16(static_cast<float>(i) + 1.f);
      b[i] = ct2_f16(3.f);
    }

    primitives<Device::MPS>::mul(
        static_cast<const ct2_f16*>(a),
        static_cast<const ct2_f16*>(b),
        out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (std::fabs(static_cast<float>(out[i]) - 3.f * (static_cast<float>(i) + 1.f)) > 0.3f) { ok = false; break; }
    }
    CHECK("mul_vec float16: out[i] ≈ 3*(i+1)", ok);

    metal_free(a);
    metal_free(b);
    metal_free(out);
  }
}


// ---------------------------------------------------------------------------
// Edge case: size == 0 (should be a no-op, not a crash)
// ---------------------------------------------------------------------------
static void test_zero_size() {
  std::printf("\n--- M4.2: zero-size edge cases ---\n");

  float* p = metal_alloc<float>(1);
  p[0] = 99.f;

  CHECK_NOTHROW("add_scalar size=0 — no error",
    primitives<Device::MPS>::add(1.f, p, p, 0)
  );
  gpu_sync();
  CHECK("add_scalar size=0: p[0] unchanged", std::fabs(p[0] - 99.f) < 1e-6f);

  CHECK_NOTHROW("mul_vec size=0 — no error",
    primitives<Device::MPS>::mul(
        static_cast<const float*>(p),
        static_cast<const float*>(p),
        p, 0)
  );
  gpu_sync();
  CHECK("mul_vec size=0: p[0] unchanged", std::fabs(p[0] - 99.f) < 1e-6f);

  metal_free(p);
}


int main() {
  std::printf("=== M4.2: Arithmetic primitives (add, sub, mul) ===\n");
  test_add_scalar();
  test_add_vec();
  test_sub_vec();
  test_mul_scalar();
  test_mul_vec();
  test_zero_size();
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
