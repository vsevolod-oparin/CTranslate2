// Standalone tests for M4.5: exp, log, cos, sin, tanh, relu, sigmoid, swish,
// gelu, gelu_tanh, gelu_sigmoid — GPU activation/transcendental primitives.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/activation_test.mm \
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
//     -o activation_test && ./activation_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// Avoid arm_vector_types.h ::float16_t conflict.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

static int passed = 0;
static int failed = 0;

#define CHECK(label, expr)                                  \
  do {                                                      \
    if (expr) { std::printf("  PASS  %s\n", label); ++passed; } \
    else       { std::printf("  FAIL  %s\n", label); ++failed; } \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

// Flush pending GPU work and read back y[0].
template <typename T>
static float readback(const T* y) {
  metal::commit_and_wait();
  return static_cast<float>(y[0]);
}

// Run f(x) on the GPU and return the result as float.
// Allocates a single-element Metal buffer for input and output.
template <typename T, typename OpFn>
static float run1(float xval, OpFn op) {
  T* x = metal_alloc<T>(1);
  T* y = metal_alloc<T>(1);
  x[0] = T(xval);
  op(x, y, 1);
  float result = readback(y);
  metal_free(x);
  metal_free(y);
  return result;
}

// Run f(x[0..N-1]) on the GPU, return y[i] as float.
template <typename T, typename OpFn>
static std::vector<float> runN(const std::vector<float>& xvals, OpFn op) {
  dim_t N = (dim_t)xvals.size();
  T* x = metal_alloc<T>(N);
  T* y = metal_alloc<T>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = T(xvals[i]);
  op(x, y, N);
  metal::commit_and_wait();
  std::vector<float> out(N);
  for (dim_t i = 0; i < N; ++i) out[i] = static_cast<float>(y[i]);
  metal_free(x);
  metal_free(y);
  return out;
}

// Tolerance: fp32 → 1e-5, fp16 → 5e-3, bf16 → 1e-2
template <typename T> static float tol();
template <> float tol<float>()    { return 1e-4f; }
template <> float tol<ct2_f16>()  { return 5e-3f; }
template <> float tol<ct2_bf16>() { return 1e-2f; }

static bool near(float a, float b, float eps) {
  return std::fabs(a - b) <= eps;
}

// ---------------------------------------------------------------------------
// Test helpers — one for each op
// ---------------------------------------------------------------------------

template <typename T>
static void test_exp(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::exp(x, y, n); };
  float e = std::exp(1.f);
  std::string l;

  l = std::string("exp<") + tag + "> exp(0) == 1"; CHECK(l.c_str(), near(run1<T>(0.f, op), 1.f, tol<T>()));
  l = std::string("exp<") + tag + "> exp(1) == e"; CHECK(l.c_str(), near(run1<T>(1.f, op), e,   tol<T>()));
  l = std::string("exp<") + tag + "> exp(-1) > 0"; CHECK(l.c_str(), run1<T>(-1.f, op) > 0.f);
}

template <typename T>
static void test_log(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::log(x, y, n); };
  std::string l;

  l = std::string("log<") + tag + "> log(1) == 0";     CHECK(l.c_str(), near(run1<T>(1.f, op), 0.f, tol<T>()));
  l = std::string("log<") + tag + "> log(e) ≈ 1";      CHECK(l.c_str(), near(run1<T>(std::exp(1.f), op), 1.f, tol<T>()));
  l = std::string("log<") + tag + "> log(exp(3)) ≈ 3"; CHECK(l.c_str(), near(run1<T>(std::exp(3.f), op), 3.f, tol<T>()));
}

template <typename T>
static void test_cos(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::cos(x, y, n); };
  std::string l;

  l = std::string("cos<") + tag + "> cos(0) == 1";       CHECK(l.c_str(), near(run1<T>(0.f, op), 1.f, tol<T>()));
  l = std::string("cos<") + tag + "> cos(pi/2) ≈ 0";     CHECK(l.c_str(), near(run1<T>(3.14159265f/2.f, op), 0.f, 1e-3f));
  l = std::string("cos<") + tag + "> cos(pi) ≈ -1";      CHECK(l.c_str(), near(run1<T>(3.14159265f, op), -1.f, tol<T>()));
}

template <typename T>
static void test_sin(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::sin(x, y, n); };
  std::string l;

  l = std::string("sin<") + tag + "> sin(0) == 0";       CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  l = std::string("sin<") + tag + "> sin(pi/2) ≈ 1";     CHECK(l.c_str(), near(run1<T>(3.14159265f/2.f, op), 1.f, tol<T>()));
  l = std::string("sin<") + tag + "> sin(pi) ≈ 0";       CHECK(l.c_str(), near(run1<T>(3.14159265f, op), 0.f, 1e-3f));
}

template <typename T>
static void test_tanh(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::tanh(x, y, n); };
  std::string l;

  l = std::string("tanh<") + tag + "> tanh(0) == 0";   CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  l = std::string("tanh<") + tag + "> tanh(10) ≈ 1";   CHECK(l.c_str(), near(run1<T>(10.f, op), 1.f, tol<T>()));
  l = std::string("tanh<") + tag + "> tanh(-10) ≈ -1"; CHECK(l.c_str(), near(run1<T>(-10.f, op), -1.f, tol<T>()));
  l = std::string("tanh<") + tag + "> tanh(1) in range";
  { float r = run1<T>(1.f, op); CHECK(l.c_str(), r > 0.f && r < 1.f); }
}

template <typename T>
static void test_relu(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::relu(x, y, n); };
  std::string l;

  l = std::string("relu<") + tag + "> relu(-5) == 0";  CHECK(l.c_str(), near(run1<T>(-5.f, op), 0.f, tol<T>()));
  l = std::string("relu<") + tag + "> relu(0) == 0";   CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  l = std::string("relu<") + tag + "> relu(3) == 3";   CHECK(l.c_str(), near(run1<T>(3.f, op), 3.f, tol<T>()));
}

template <typename T>
static void test_sigmoid(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::sigmoid(x, y, n); };
  std::string l;

  l = std::string("sigmoid<") + tag + "> sigmoid(0) == 0.5";
  CHECK(l.c_str(), near(run1<T>(0.f, op), 0.5f, tol<T>()));
  l = std::string("sigmoid<") + tag + "> sigmoid(100) ≈ 1";
  CHECK(l.c_str(), near(run1<T>(100.f, op), 1.f, tol<T>()));
  l = std::string("sigmoid<") + tag + "> sigmoid(-100) ≈ 0";
  CHECK(l.c_str(), near(run1<T>(-100.f, op), 0.f, tol<T>()));
}

template <typename T>
static void test_swish(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::swish(x, y, n); };
  std::string l;

  // swish(0) = 0 * sigmoid(0) = 0
  l = std::string("swish<") + tag + "> swish(0) == 0";
  CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  // swish(x) ≈ x for large x  (sigmoid → 1)
  l = std::string("swish<") + tag + "> swish(10) ≈ 10";
  CHECK(l.c_str(), near(run1<T>(10.f, op), 10.f, 0.1f));
  // swish(-10) ≈ 0  (sigmoid → 0)
  l = std::string("swish<") + tag + "> swish(-10) ≈ 0";
  CHECK(l.c_str(), std::fabs(run1<T>(-10.f, op)) < 0.1f);
  // swish is positive for x > 0 (unlike relu, negative region ~-0.28)
  l = std::string("swish<") + tag + "> swish(1) > 0";
  CHECK(l.c_str(), run1<T>(1.f, op) > 0.f);
}

template <typename T>
static void test_gelu(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::gelu(x, y, n); };
  std::string l;

  // gelu(0) = 0
  l = std::string("gelu<") + tag + "> gelu(0) == 0";
  CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  // gelu(x) ≈ x for large x
  l = std::string("gelu<") + tag + "> gelu(10) ≈ 10";
  CHECK(l.c_str(), near(run1<T>(10.f, op), 10.f, 0.1f));
  // gelu(-10) ≈ 0
  l = std::string("gelu<") + tag + "> gelu(-10) ≈ 0";
  CHECK(l.c_str(), std::fabs(run1<T>(-10.f, op)) < 0.1f);
  // gelu(1) ≈ 0.8413 (CPU reference: 0.5 * 1 * (1 + erf(1/sqrt(2))))
  {
    float ref = 0.5f * 1.f * (1.f + std::erf(1.f * 0.7071067811865475f));
    l = std::string("gelu<") + tag + "> gelu(1) ≈ ref";
    CHECK(l.c_str(), near(run1<T>(1.f, op), ref, tol<T>()));
  }
}

template <typename T>
static void test_gelu_tanh(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::gelu_tanh(x, y, n); };
  std::string l;

  l = std::string("gelu_tanh<") + tag + "> gelu_tanh(0) == 0";
  CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  l = std::string("gelu_tanh<") + tag + "> gelu_tanh(10) ≈ 10";
  CHECK(l.c_str(), near(run1<T>(10.f, op), 10.f, 0.1f));
  l = std::string("gelu_tanh<") + tag + "> gelu_tanh(-10) ≈ 0";
  CHECK(l.c_str(), std::fabs(run1<T>(-10.f, op)) < 0.1f);
  // gelu_tanh(1) CPU ref
  {
    float v = 1.f;
    float ref = 0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v*v*v)));
    l = std::string("gelu_tanh<") + tag + "> gelu_tanh(1) ≈ ref";
    CHECK(l.c_str(), near(run1<T>(1.f, op), ref, tol<T>()));
  }
}

template <typename T>
static void test_gelu_sigmoid(const char* tag) {
  auto op = [](const T* x, T* y, dim_t n){ primitives<Device::METAL>::gelu_sigmoid(x, y, n); };
  std::string l;

  l = std::string("gelu_sigmoid<") + tag + "> gelu_sigmoid(0) == 0";
  CHECK(l.c_str(), near(run1<T>(0.f, op), 0.f, tol<T>()));
  l = std::string("gelu_sigmoid<") + tag + "> gelu_sigmoid(10) ≈ 10";
  CHECK(l.c_str(), near(run1<T>(10.f, op), 10.f, 0.1f));
  l = std::string("gelu_sigmoid<") + tag + "> gelu_sigmoid(-10) ≈ 0";
  CHECK(l.c_str(), std::fabs(run1<T>(-10.f, op)) < 0.1f);
  // gelu_sigmoid(1) = 1 / (1 + exp(-1.702))
  {
    float ref = 1.f / (1.f + std::exp(-1.702f));
    l = std::string("gelu_sigmoid<") + tag + "> gelu_sigmoid(1) ≈ ref";
    CHECK(l.c_str(), near(run1<T>(1.f, op), ref, tol<T>()));
  }
}

// logsumexp — CPU-side implementation
template <typename T>
static void test_logsumexp(const char* tag) {
  std::string l;
  // Single element: log(exp(x)) = x
  {
    T* x = metal_alloc<T>(1);
    x[0] = T(3.f);
    float r = primitives<Device::METAL>::logsumexp(x, 1);
    l = std::string("logsumexp<") + tag + "> single == 3";
    CHECK(l.c_str(), near(r, 3.f, tol<T>()));
    metal_free(x);
  }
  // [1, 2, 3] — CPU reference
  {
    T* x = metal_alloc<T>(3);
    x[0] = T(1.f); x[1] = T(2.f); x[2] = T(3.f);
    float ref = std::log(std::exp(1.f) + std::exp(2.f) + std::exp(3.f));
    float r = primitives<Device::METAL>::logsumexp(x, 3);
    l = std::string("logsumexp<") + tag + "> [1,2,3] ≈ ref";
    CHECK(l.c_str(), near(r, ref, tol<T>()));
    metal_free(x);
  }
  // size=0 → 0
  {
    T* x = metal_alloc<T>(1);
    x[0] = T(99.f);
    float r = primitives<Device::METAL>::logsumexp(x, 0);
    l = std::string("logsumexp<") + tag + "> size=0 == 0";
    CHECK(l.c_str(), near(r, 0.f, 1e-6f));
    metal_free(x);
  }
}

// zero-size no-ops (should not crash)
template <typename T>
static void test_zero_size(const char* tag) {
  T* p = metal_alloc<T>(1);
  p[0] = T(1.f);
  std::string l;
  auto check_nothrow = [&](const char* name, auto fn) {
    bool ok = true;
    try { fn(p, p, 0); metal::commit_and_wait(); }
    catch (...) { ok = false; }
    l = std::string(name) + "<" + tag + "> size=0 no-crash";
    CHECK(l.c_str(), ok);
  };
  check_nothrow("exp",          [](auto* x, auto* y, dim_t n){ primitives<Device::METAL>::exp(x, y, n); });
  check_nothrow("log",          [](auto* x, auto* y, dim_t n){ primitives<Device::METAL>::log(x, y, n); });
  check_nothrow("relu",         [](auto* x, auto* y, dim_t n){ primitives<Device::METAL>::relu(x, y, n); });
  check_nothrow("gelu",         [](auto* x, auto* y, dim_t n){ primitives<Device::METAL>::gelu(x, y, n); });
  check_nothrow("sigmoid",      [](auto* x, auto* y, dim_t n){ primitives<Device::METAL>::sigmoid(x, y, n); });
  metal_free(p);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.5: Activation/Transcendental Primitives ===\n");

  // float32
  std::printf("\n--- float32 ---\n");
  test_exp<float>("float");
  test_log<float>("float");
  test_cos<float>("float");
  test_sin<float>("float");
  test_tanh<float>("float");
  test_relu<float>("float");
  test_sigmoid<float>("float");
  test_swish<float>("float");
  test_gelu<float>("float");
  test_gelu_tanh<float>("float");
  test_gelu_sigmoid<float>("float");
  test_logsumexp<float>("float");
  test_zero_size<float>("float");

  // float16
  std::printf("\n--- float16 ---\n");
  test_exp<ct2_f16>("f16");
  test_log<ct2_f16>("f16");
  test_cos<ct2_f16>("f16");
  test_sin<ct2_f16>("f16");
  test_tanh<ct2_f16>("f16");
  test_relu<ct2_f16>("f16");
  test_sigmoid<ct2_f16>("f16");
  test_swish<ct2_f16>("f16");
  test_gelu<ct2_f16>("f16");
  test_gelu_tanh<ct2_f16>("f16");
  test_gelu_sigmoid<ct2_f16>("f16");
  test_logsumexp<ct2_f16>("f16");
  test_zero_size<ct2_f16>("f16");

  // bfloat16
  std::printf("\n--- bfloat16 ---\n");
  test_exp<ct2_bf16>("bf16");
  test_log<ct2_bf16>("bf16");
  test_cos<ct2_bf16>("bf16");
  test_sin<ct2_bf16>("bf16");
  test_tanh<ct2_bf16>("bf16");
  test_relu<ct2_bf16>("bf16");
  test_sigmoid<ct2_bf16>("bf16");
  test_swish<ct2_bf16>("bf16");
  test_gelu<ct2_bf16>("bf16");
  test_gelu_tanh<ct2_bf16>("bf16");
  test_gelu_sigmoid<ct2_bf16>("bf16");
  test_logsumexp<ct2_bf16>("bf16");
  test_zero_size<ct2_bf16>("bf16");

  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
