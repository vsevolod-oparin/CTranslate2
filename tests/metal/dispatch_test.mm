// Standalone runtime tests for M5.1: DEVICE_AND_FLOAT_DISPATCH guard behaviour.
//
// Verifies:
//   1. float32  + Device::MPS → dispatches, D==METAL, sizeof(T)==4
//   2. float16  + Device::MPS → no throw (M5.1 PASS), D==METAL, sizeof(T)==2
//   3. bfloat16 + Device::MPS → no throw (M5.1 PASS), D==METAL, sizeof(T)==2
//   4. float16  + Device::CPU   → throws std::invalid_argument("FP16 ...")
//   5. bfloat16 + Device::CPU   → throws std::invalid_argument("BF16 ...")
//
// End-to-end Metal float16/bfloat16 adds are confirmed via direct primitive
// calls (not through the dispatch macro) to avoid requiring CPU-primitive
// template instantiations at link time.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/dispatch_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o dispatch_test && ./dispatch_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "dispatch.h"
#include "metal/utils.h"

// arm_vector_types.h (pulled in by Metal headers) defines ::float16_t as
// __fp16, conflicting with ctranslate2::float16_t.  Alias before namespace.
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
static void gpu_sync() { metal::commit_and_wait(); }

static const dim_t N = 8;

// DispatchResult captures the D and sizeof(T) bindings set by
// DEVICE_AND_FLOAT_DISPATCH without calling any device primitives.
// This avoids requiring CPU-primitive symbols at link time (the CPU dead-code
// path in DEVICE_DISPATCH is compiled but never needs to be defined).
struct DispatchResult {
  Device d         = Device::CPU;
  std::size_t sz_T = 0;
};

// ---------------------------------------------------------------------------
// 1. float32 + Device::MPS
// ---------------------------------------------------------------------------
static void test_float32_metal() {
  std::printf("\n--- 1. float32 + Device::MPS ---\n");

  DispatchResult dr;
  CHECK_NOTHROW("float32 + Metal: no throw",
    DEVICE_AND_FLOAT_DISPATCH("test", Device::MPS, DataType::FLOAT32,
      (dr = DispatchResult{D, sizeof(T)})));
  CHECK("float32 + Metal: D == METAL",        dr.d == Device::MPS);
  CHECK("float32 + Metal: sizeof(T) == 4",    dr.sz_T == sizeof(float));

  // End-to-end: confirm the Metal float32 path actually works.
  float* x   = metal_alloc<float>(N);
  float* out = metal_alloc<float>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = float(i);
  primitives<Device::MPS>::add(10.f, x, out, N);
  gpu_sync();
  bool ok = true;
  for (dim_t i = 0; i < N; ++i)
    if (std::fabs(out[i] - (10.f + float(i))) > 1e-5f) { ok = false; break; }
  CHECK("float32 Metal add: out[i] == 10+i", ok);
  metal_free(x); metal_free(out);
}

// ---------------------------------------------------------------------------
// 2. float16 + Device::MPS  — M5.1 primary PASS criterion
// ---------------------------------------------------------------------------
static void test_float16_metal() {
  std::printf("\n--- 2. float16 + Device::MPS (M5.1 PASS) ---\n");

  DispatchResult dr;
  CHECK_NOTHROW("float16 + Metal: no throw",
    DEVICE_AND_FLOAT_DISPATCH("test", Device::MPS, DataType::FLOAT16,
      (dr = DispatchResult{D, sizeof(T)})));
  CHECK("float16 + Metal: D == METAL",        dr.d == Device::MPS);
  CHECK("float16 + Metal: sizeof(T) == 2",    dr.sz_T == 2);

  // End-to-end: Metal float16 add (direct, confirming the FP16 GPU path works).
  ct2_f16* x   = metal_alloc<ct2_f16>(N);
  ct2_f16* out = metal_alloc<ct2_f16>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = ct2_f16(float(i));
  primitives<Device::MPS>::add(ct2_f16(5.f), x, out, N);
  gpu_sync();
  bool ok = true;
  for (dim_t i = 0; i < N; ++i)
    if (std::fabs(static_cast<float>(out[i]) - (5.f + float(i))) > 0.1f) { ok = false; break; }
  CHECK("float16 Metal add: out[i] ≈ 5+i", ok);
  metal_free(x); metal_free(out);
}

// ---------------------------------------------------------------------------
// 3. bfloat16 + Device::MPS  — M5.1 primary PASS criterion
// ---------------------------------------------------------------------------
static void test_bfloat16_metal() {
  std::printf("\n--- 3. bfloat16 + Device::MPS (M5.1 PASS) ---\n");

  DispatchResult dr;
  CHECK_NOTHROW("bfloat16 + Metal: no throw",
    DEVICE_AND_FLOAT_DISPATCH("test", Device::MPS, DataType::BFLOAT16,
      (dr = DispatchResult{D, sizeof(T)})));
  CHECK("bfloat16 + Metal: D == METAL",        dr.d == Device::MPS);
  CHECK("bfloat16 + Metal: sizeof(T) == 2",    dr.sz_T == 2);

  // End-to-end: Metal bfloat16 add (direct, confirming the BF16 GPU path works).
  ct2_bf16* x   = metal_alloc<ct2_bf16>(N);
  ct2_bf16* out = metal_alloc<ct2_bf16>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = ct2_bf16(float(i));
  primitives<Device::MPS>::add(ct2_bf16(3.f), x, out, N);
  gpu_sync();
  bool ok = true;
  for (dim_t i = 0; i < N; ++i)
    if (std::fabs(static_cast<float>(out[i]) - (3.f + float(i))) > 0.2f) { ok = false; break; }
  CHECK("bfloat16 Metal add: out[i] ≈ 3+i", ok);
  metal_free(x); metal_free(out);
}

// ---------------------------------------------------------------------------
// 4. float16 + Device::CPU — guard must throw (preserved from pre-Metal guard)
// ---------------------------------------------------------------------------
static void test_float16_cpu_throws() {
  std::printf("\n--- 4. float16 + Device::CPU: expected throw ---\n");
  bool threw     = false;
  bool right_msg = false;
  try {
    DEVICE_AND_FLOAT_DISPATCH("float16-guard", Device::CPU, DataType::FLOAT16,
      (void)0);
  } catch (const std::invalid_argument& e) {
    threw     = true;
    right_msg = (std::string(e.what()).find("FP16") != std::string::npos);
    if (!right_msg)
      std::printf("    (wrong msg: %s)\n", e.what());
  } catch (...) {}
  CHECK("float16 + CPU: throws invalid_argument containing \"FP16\"",
        threw && right_msg);
}

// ---------------------------------------------------------------------------
// 5. bfloat16 + Device::CPU — guard must throw
// ---------------------------------------------------------------------------
static void test_bfloat16_cpu_throws() {
  std::printf("\n--- 5. bfloat16 + Device::CPU: expected throw ---\n");
  bool threw     = false;
  bool right_msg = false;
  try {
    DEVICE_AND_FLOAT_DISPATCH("bfloat16-guard", Device::CPU, DataType::BFLOAT16,
      (void)0);
  } catch (const std::invalid_argument& e) {
    threw     = true;
    right_msg = (std::string(e.what()).find("BF16") != std::string::npos);
    if (!right_msg)
      std::printf("    (wrong msg: %s)\n", e.what());
  } catch (...) {}
  CHECK("bfloat16 + CPU: throws invalid_argument containing \"BF16\"",
        threw && right_msg);
}

// ---------------------------------------------------------------------------
int main() {
  std::printf("=== M5.1 DEVICE_AND_FLOAT_DISPATCH runtime guard tests ===\n");
  test_float32_metal();
  test_float16_metal();
  test_bfloat16_metal();
  test_float16_cpu_throws();
  test_bfloat16_cpu_throws();
  std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
  return (failed == 0) ? 0 : 1;
}
