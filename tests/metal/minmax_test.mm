// Tests for Metal min/max element-wise primitives (M4.2 extension).
//
// Covers all four overloads for float32, float16, bfloat16, int32, int16, int8:
//   min(scalar, vector, out)    y[i] = min(scalar, x[i])
//   min(vector, vector, out)    c[i] = min(a[i], b[i])
//   max(scalar, vector, out)    y[i] = max(scalar, x[i])
//   max(vector, vector, out)    c[i] = max(a[i], b[i])
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/minmax_test.mm \
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
//     -o minmax_test && ./minmax_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
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

// Alias to avoid the ::float16_t / ctranslate2::float16_t conflict from
// arm_vector_types.h (pulled in by Metal.h).  Must be declared BEFORE
// using namespace ctranslate2.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                    \
  do {                                                        \
    if (expr) { std::printf("  PASS  %s\n", label); ++g_passed; } \
    else       { std::printf("  FAIL  %s\n", label); ++g_failed; } \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

static void gpu_sync() { metal::commit_and_wait(); }

// ---------------------------------------------------------------------------
// Helpers: compare GPU result to CPU reference
// ---------------------------------------------------------------------------

// Exact match for integer types; within `tol` for floating-point.
template <typename T>
static bool near(T a, T b, float tol = 1e-4f) {
  return std::fabs((float)a - (float)b) <= tol;
}

// ---------------------------------------------------------------------------
// Generic test body — used by every type specialization
// ---------------------------------------------------------------------------

template <typename T>
static void run_tests(const char* type_name) {
  std::printf("\n--- min/max: %s ---\n", type_name);

  const dim_t N = 32;

  T* a   = metal_alloc<T>(N);
  T* b   = metal_alloc<T>(N);
  T* out = metal_alloc<T>(N);

  // Fill: a[i] = i-8 (spans negative to positive), b[i] = 15-i
  for (dim_t i = 0; i < N; ++i) {
    a[i] = T(static_cast<int>(i) - 8);
    b[i] = T(15 - static_cast<int>(i));
  }

  // -------------------------------------------------------------------
  // 1. min(vector, vector, out)  →  c[i] = min(a[i], b[i])
  // -------------------------------------------------------------------
  {
    primitives<Device::METAL>::min(a, b, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      float fa = (float)a[i], fb = (float)b[i];
      float expected = (fa < fb) ? fa : fb;
      if (!near(out[i], T(expected))) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "min(vec,vec): %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 2. max(vector, vector, out)  →  c[i] = max(a[i], b[i])
  // -------------------------------------------------------------------
  {
    primitives<Device::METAL>::max(a, b, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      float fa = (float)a[i], fb = (float)b[i];
      float expected = (fa > fb) ? fa : fb;
      if (!near(out[i], T(expected))) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "max(vec,vec): %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 3. min(scalar, vector, out)  →  y[i] = min(scalar, x[i])
  //    Clamp from above: scalar = 3  →  out[i] = min(3, a[i])
  // -------------------------------------------------------------------
  {
    T scalar = T(3);
    primitives<Device::METAL>::min(scalar, a, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      float fa = (float)a[i], fs = (float)scalar;
      float expected = (fa < fs) ? fa : fs;
      if (!near(out[i], T(expected))) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "min(scalar,vec): %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 4. max(scalar, vector, out)  →  y[i] = max(scalar, x[i])
  //    Clamp from below (ReLU-like): scalar = 0 → y[i] = max(0, a[i])
  // -------------------------------------------------------------------
  {
    T scalar = T(0);
    primitives<Device::METAL>::max(scalar, a, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      float fa = (float)a[i], fs = (float)scalar;
      float expected = (fa > fs) ? fa : fs;
      if (!near(out[i], T(expected))) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "max(scalar,vec)=relu: %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 5. min(vector, vector) — all elements equal: output == input
  // -------------------------------------------------------------------
  {
    // c[i] = min(a[i], a[i]) should equal a[i]
    primitives<Device::METAL>::min(a, a, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (!near(out[i], a[i])) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "min(a,a)==a: %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 6. max(vector, vector) — all elements equal: output == input
  // -------------------------------------------------------------------
  {
    primitives<Device::METAL>::max(a, a, out, N);
    gpu_sync();

    bool ok = true;
    for (dim_t i = 0; i < N; ++i) {
      if (!near(out[i], a[i])) { ok = false; break; }
    }
    char label[64];
    std::snprintf(label, sizeof(label), "max(a,a)==a: %s", type_name);
    CHECK(label, ok);
  }

  // -------------------------------------------------------------------
  // 7. Zero-size: no crash, no writes
  // -------------------------------------------------------------------
  {
    bool ok = true;
    try {
      primitives<Device::METAL>::min(a, b, out, 0);
      primitives<Device::METAL>::max(a, b, out, 0);
      primitives<Device::METAL>::min(T(0), a, out, 0);
      primitives<Device::METAL>::max(T(0), a, out, 0);
      gpu_sync();
    } catch (...) { ok = false; }
    char label[64];
    std::snprintf(label, sizeof(label), "zero-size no-crash: %s", type_name);
    CHECK(label, ok);
  }

  metal_free(a);
  metal_free(b);
  metal_free(out);
}

// ---------------------------------------------------------------------------
// 4.3 scalar clamp sub-cases (floating-point types only)
//
// The generic run_tests() already covers a "mixed" input for scalar min/max.
// These three focused sub-cases verify the behaviour at the extremes:
//
//   all_below:  every x[i] < scalar  → min(scalar,x)=x (scalar never wins)
//                                       max(scalar,x)=scalar (scalar always wins)
//   all_above:  every x[i] > scalar  → min(scalar,x)=scalar (scalar always wins)
//                                       max(scalar,x)=x (scalar never wins)
//   edge:       scalar == min element → min(scalar,x)=scalar for all i (scalar ≤ x[i])
//                                       max(scalar,x)=x (x[i] ≥ scalar)
//
// For "edge" we choose x[i] = i-8 (range -8..7) and scalar = -8 (minimum value).
// Every element satisfies scalar ≤ x[i], so:
//   min(-8, x[i]) = -8 for all i  (scalar always ≤ x[i])
//   max(-8, x[i]) = x[i] for all i (x[i] ≥ scalar)
// ---------------------------------------------------------------------------

template <typename T>
static void run_scalar_clamp_subcases(const char* type_name) {
  std::printf("\n--- scalar clamp sub-cases: %s ---\n", type_name);

  const dim_t N = 16;
  T* x   = metal_alloc<T>(N);
  T* out = metal_alloc<T>(N);

  // ---------------------------------------------------------------
  // 1. All below scalar: x[i] = i+1 (1..16), scalar = 20
  //    min(20, x) = x      (scalar > every x[i])
  //    max(20, x) = 20     (scalar > every x[i])
  // ---------------------------------------------------------------
  {
    for (dim_t i = 0; i < N; ++i) x[i] = T(static_cast<int>(i) + 1);
    T scalar = T(20);

    primitives<Device::METAL>::min(scalar, x, out, N);
    gpu_sync();
    bool ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], x[i])) { ok = false; break; }
    char label[80];
    std::snprintf(label, sizeof(label), "min(scalar=20, x=1..16)=x [all_below]: %s", type_name);
    CHECK(label, ok);

    primitives<Device::METAL>::max(scalar, x, out, N);
    gpu_sync();
    ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], scalar)) { ok = false; break; }
    std::snprintf(label, sizeof(label), "max(scalar=20, x=1..16)=20 [all_below]: %s", type_name);
    CHECK(label, ok);
  }

  // ---------------------------------------------------------------
  // 2. All above scalar: x[i] = i+1 (1..16), scalar = 0
  //    min(0, x) = 0       (scalar < every x[i])
  //    max(0, x) = x       (scalar < every x[i])
  // ---------------------------------------------------------------
  {
    for (dim_t i = 0; i < N; ++i) x[i] = T(static_cast<int>(i) + 1);
    T scalar = T(0);

    primitives<Device::METAL>::min(scalar, x, out, N);
    gpu_sync();
    bool ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], scalar)) { ok = false; break; }
    char label[80];
    std::snprintf(label, sizeof(label), "min(scalar=0, x=1..16)=0 [all_above]: %s", type_name);
    CHECK(label, ok);

    primitives<Device::METAL>::max(scalar, x, out, N);
    gpu_sync();
    ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], x[i])) { ok = false; break; }
    std::snprintf(label, sizeof(label), "max(scalar=0, x=1..16)=x [all_above]: %s", type_name);
    CHECK(label, ok);
  }

  // ---------------------------------------------------------------
  // 3. Edge: scalar == minimum element.
  //    x[i] = i-8 (range -8..7), scalar = -8 (= x[0], minimum).
  //    Every x[i] >= scalar, so:
  //      min(-8, x[i]) = -8  for all i   (scalar ≤ x[i] always)
  //      max(-8, x[i]) = x[i] for all i  (x[i] ≥ scalar always)
  // ---------------------------------------------------------------
  {
    for (dim_t i = 0; i < N; ++i) x[i] = T(static_cast<int>(i) - 8);
    T scalar = T(-8);  // == x[0], the minimum

    primitives<Device::METAL>::min(scalar, x, out, N);
    gpu_sync();
    bool ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], scalar)) { ok = false; break; }
    char label[80];
    std::snprintf(label, sizeof(label), "min(scalar=min_elem, x)=scalar [edge]: %s", type_name);
    CHECK(label, ok);

    primitives<Device::METAL>::max(scalar, x, out, N);
    gpu_sync();
    ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], x[i])) { ok = false; break; }
    std::snprintf(label, sizeof(label), "max(scalar=min_elem, x)=x [edge]: %s", type_name);
    CHECK(label, ok);
  }

  metal_free(x);
  metal_free(out);
}

// ---------------------------------------------------------------------------
// 4.3 vector min/max sign sub-cases (floating-point types only)
//
// The generic run_tests() uses a crossing pattern (a[i]=i-8, b[i]=15-i) where
// which operand is smaller changes mid-array.  These two sub-cases verify
// uniformly-signed inputs where one operand is always smaller.
//
//   all_positive: a[i] = i+1 (1..16), b[i] = i+2 (2..17)  → a < b for all i
//                 min(a,b) = a,  max(a,b) = b
//
//   all_negative: a[i] = -(i+2) (-3..-18), b[i] = -(i+1) (-2..-17) → a < b for all i
//                 min(a,b) = a,  max(a,b) = b
// ---------------------------------------------------------------------------

template <typename T>
static void run_vector_sign_subcases(const char* type_name) {
  std::printf("\n--- vector sign sub-cases: %s ---\n", type_name);

  const dim_t N = 16;
  T* a   = metal_alloc<T>(N);
  T* b   = metal_alloc<T>(N);
  T* out = metal_alloc<T>(N);

  // ---------------------------------------------------------------
  // All positive: a[i] = i+1, b[i] = i+2  → a[i] < b[i] for all i
  // ---------------------------------------------------------------
  {
    for (dim_t i = 0; i < N; ++i) {
      a[i] = T(static_cast<int>(i) + 1);
      b[i] = T(static_cast<int>(i) + 2);
    }

    primitives<Device::METAL>::min(a, b, out, N);
    gpu_sync();
    bool ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], a[i])) { ok = false; break; }
    char label[80];
    std::snprintf(label, sizeof(label), "min(a,b)=a [all_positive]: %s", type_name);
    CHECK(label, ok);

    primitives<Device::METAL>::max(a, b, out, N);
    gpu_sync();
    ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], b[i])) { ok = false; break; }
    std::snprintf(label, sizeof(label), "max(a,b)=b [all_positive]: %s", type_name);
    CHECK(label, ok);
  }

  // ---------------------------------------------------------------
  // All negative: a[i] = -(i+2), b[i] = -(i+1) → a[i] < b[i] < 0 for all i
  // ---------------------------------------------------------------
  {
    for (dim_t i = 0; i < N; ++i) {
      a[i] = T(-(static_cast<int>(i) + 2));
      b[i] = T(-(static_cast<int>(i) + 1));
    }

    primitives<Device::METAL>::min(a, b, out, N);
    gpu_sync();
    bool ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], a[i])) { ok = false; break; }
    char label[80];
    std::snprintf(label, sizeof(label), "min(a,b)=a [all_negative]: %s", type_name);
    CHECK(label, ok);

    primitives<Device::METAL>::max(a, b, out, N);
    gpu_sync();
    ok = true;
    for (dim_t i = 0; i < N; ++i)
      if (!near(out[i], b[i])) { ok = false; break; }
    std::snprintf(label, sizeof(label), "max(a,b)=b [all_negative]: %s", type_name);
    CHECK(label, ok);
  }

  metal_free(a);
  metal_free(b);
  metal_free(out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.2 Extension: Metal min/max Primitives ===\n");

  // Generic correctness: mixed inputs, all six types (7 cases each = 42 tests).
  run_tests<float>   ("float32");
  run_tests<ct2_f16> ("float16");
  run_tests<ct2_bf16>("bfloat16");
  run_tests<int32_t> ("int32");
  run_tests<int16_t> ("int16");
  run_tests<int8_t>  ("int8");

  // 4.3 scalar clamp sub-cases: all-below, all-above, edge (3 float types only).
  // The bfloat16 pass here also implicitly validates that the MSL kernel uses
  // ternary comparison (not min()/max() overloads) — if the overload were used
  // and absent, the kernel would fail to compile or produce wrong values.
  run_scalar_clamp_subcases<float>   ("float32");
  run_scalar_clamp_subcases<ct2_f16> ("float16");
  run_scalar_clamp_subcases<ct2_bf16>("bfloat16");

  // 4.3 vector sign sub-cases: all-positive, all-negative (3 float types only).
  run_vector_sign_subcases<float>   ("float32");
  run_vector_sign_subcases<ct2_f16> ("float16");
  run_vector_sign_subcases<ct2_bf16>("bfloat16");

  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
