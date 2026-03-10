// Correctness tests for M4.6 Metal broadcast primitives.
//
// Tests: add_batch_broadcast, add_depth_broadcast, add_block_broadcast,
//        mul_batch_broadcast — for float32, float16, bfloat16.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/broadcast_test.mm \
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
//     -o broadcast_test && ./broadcast_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
  if (ok) {
    ++g_pass;
  } else {
    ++g_fail;
    std::printf("  FAIL: %s\n", name.c_str());
  }
}

// Approximate equality for float comparisons (handles half/bfloat rounding).
static bool near(float a, float b, float tol = 1e-3f) {
  return std::fabs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// add_batch_broadcast tests
// ---------------------------------------------------------------------------
//
// Semantics: a has size a_size; b and c have size b_size.
// iter = b_size / a_size
// for i in [0, iter): c[i*a_size + j] = a[j] + b[i*a_size + j]
//
// GPU kernel: c[gid] = a[gid % a_size] + b[gid]

template <typename T>
static void test_add_batch_broadcast(const std::string& tname) {
  // a = [1, 2, 3],  b = [10, 20, 30, 40, 50, 60]
  // Expected: [11, 22, 33, 41, 52, 63]
  const dim_t a_size = 3, b_size = 6;
  T* a = metal_alloc<T>(a_size);
  T* b = metal_alloc<T>(b_size);
  T* c = metal_alloc<T>(b_size);

  a[0] = T(1); a[1] = T(2); a[2] = T(3);
  for (dim_t i = 0; i < b_size; ++i) b[i] = T((i + 1) * 10);

  primitives<Device::MPS>::add_batch_broadcast(a, b, c, a_size, b_size);
  metal::commit_and_wait();

  float expected[] = {11.f, 22.f, 33.f, 41.f, 52.f, 63.f};
  bool ok = true;
  for (dim_t i = 0; i < b_size; ++i)
    if (!near(float(c[i]), expected[i])) { ok = false; break; }
  check(ok, "add_batch_broadcast/" + tname + " basic");

  // In-place: b += broadcast(a) with c = b
  T* bip = metal_alloc<T>(b_size);
  for (dim_t i = 0; i < b_size; ++i) bip[i] = T((i + 1) * 10);
  primitives<Device::MPS>::add_batch_broadcast(a, bip, bip, a_size, b_size);
  metal::commit_and_wait();
  bool okip = true;
  for (dim_t i = 0; i < b_size; ++i)
    if (!near(float(bip[i]), expected[i])) { okip = false; break; }
  check(okip, "add_batch_broadcast/" + tname + " in-place");

  // Zero-size: no crash
  primitives<Device::MPS>::add_batch_broadcast(a, b, c, a_size, dim_t(0));
  metal::commit_and_wait();
  check(true, "add_batch_broadcast/" + tname + " zero-size");

  metal_free(a); metal_free(b); metal_free(c); metal_free(bip);
}

// ---------------------------------------------------------------------------
// add_depth_broadcast tests
// ---------------------------------------------------------------------------
//
// Semantics: a has size a_size; b and c have size b_size.
// depth = b_size / a_size
// for i in [0, a_size): c[i*depth + k] = a[i] + b[i*depth + k]
//
// GPU kernel: c[gid] = a[gid / depth] + b[gid]

template <typename T>
static void test_add_depth_broadcast(const std::string& tname) {
  // a = [100, 200],  b = [1, 2, 3, 4, 5, 6],  depth = 3
  // Expected:
  //   i=0: c[0..2] = 100 + [1,2,3] = [101, 102, 103]
  //   i=1: c[3..5] = 200 + [4,5,6] = [204, 205, 206]
  const dim_t a_size = 2, b_size = 6;
  T* a = metal_alloc<T>(a_size);
  T* b = metal_alloc<T>(b_size);
  T* c = metal_alloc<T>(b_size);

  a[0] = T(100); a[1] = T(200);
  for (dim_t i = 0; i < b_size; ++i) b[i] = T(i + 1);

  primitives<Device::MPS>::add_depth_broadcast(a, b, c, a_size, b_size);
  metal::commit_and_wait();

  float expected[] = {101.f, 102.f, 103.f, 204.f, 205.f, 206.f};
  bool ok = true;
  for (dim_t i = 0; i < b_size; ++i)
    if (!near(float(c[i]), expected[i])) { ok = false; break; }
  check(ok, "add_depth_broadcast/" + tname + " basic");

  // depth = 1 (a_size == b_size): c[gid] = a[gid] + b[gid]
  const dim_t N = 4;
  T* a2 = metal_alloc<T>(N);
  T* b2 = metal_alloc<T>(N);
  T* c2 = metal_alloc<T>(N);
  for (dim_t i = 0; i < N; ++i) { a2[i] = T(i + 1); b2[i] = T(10); }
  primitives<Device::MPS>::add_depth_broadcast(a2, b2, c2, N, N);
  metal::commit_and_wait();
  bool ok2 = true;
  for (dim_t i = 0; i < N; ++i)
    if (!near(float(c2[i]), float(i + 1 + 10))) { ok2 = false; break; }
  check(ok2, "add_depth_broadcast/" + tname + " depth=1");

  // Zero-size
  primitives<Device::MPS>::add_depth_broadcast(a, b, c, a_size, dim_t(0));
  metal::commit_and_wait();
  check(true, "add_depth_broadcast/" + tname + " zero-size");

  metal_free(a); metal_free(b); metal_free(c);
  metal_free(a2); metal_free(b2); metal_free(c2);
}

// ---------------------------------------------------------------------------
// add_block_broadcast tests
// ---------------------------------------------------------------------------
//
// Semantics: a has size a_size; b and c have size b_size.
// for i in [0, b_size/block): a_i = a[i % a_size]; c[i*block+k] = a_i + b[i*block+k]
//
// GPU kernel: c[gid] = a[(gid/block) % a_size] + b[gid]

template <typename T>
static void test_add_block_broadcast(const std::string& tname) {
  // block=2, a=[10,20,30] (a_size=3), b=[1..12] (b_size=12)
  // i=0: a[0%3]=10, c[0..1] = 10+[1,2]   = [11, 12]
  // i=1: a[1%3]=20, c[2..3] = 20+[3,4]   = [23, 24]
  // i=2: a[2%3]=30, c[4..5] = 30+[5,6]   = [35, 36]
  // i=3: a[3%3]=10, c[6..7] = 10+[7,8]   = [17, 18]
  // i=4: a[4%3]=20, c[8..9] = 20+[9,10]  = [29, 30]
  // i=5: a[5%3]=30, c[10..11] = 30+[11,12]= [41, 42]
  const dim_t block = 2, a_size = 3, b_size = 12;
  T* a = metal_alloc<T>(a_size);
  T* b = metal_alloc<T>(b_size);
  T* c = metal_alloc<T>(b_size);

  a[0] = T(10); a[1] = T(20); a[2] = T(30);
  for (dim_t i = 0; i < b_size; ++i) b[i] = T(i + 1);

  primitives<Device::MPS>::add_block_broadcast(a, b, c, block, a_size, b_size);
  metal::commit_and_wait();

  float expected[] = {11.f, 12.f, 23.f, 24.f, 35.f, 36.f, 17.f, 18.f, 29.f, 30.f, 41.f, 42.f};
  bool ok = true;
  for (dim_t i = 0; i < b_size; ++i)
    if (!near(float(c[i]), expected[i])) { ok = false; break; }
  check(ok, "add_block_broadcast/" + tname + " basic");

  // block=1: degenerates to add_batch_broadcast
  const dim_t a_size2 = 3, b_size2 = 6;
  T* a3 = metal_alloc<T>(a_size2);
  T* b3 = metal_alloc<T>(b_size2);
  T* c3 = metal_alloc<T>(b_size2);
  a3[0] = T(1); a3[1] = T(2); a3[2] = T(3);
  for (dim_t i = 0; i < b_size2; ++i) b3[i] = T((i + 1) * 10);
  primitives<Device::MPS>::add_block_broadcast(a3, b3, c3, 1, a_size2, b_size2);
  metal::commit_and_wait();
  float exp3[] = {11.f, 22.f, 33.f, 41.f, 52.f, 63.f};
  bool ok3 = true;
  for (dim_t i = 0; i < b_size2; ++i)
    if (!near(float(c3[i]), exp3[i])) { ok3 = false; break; }
  check(ok3, "add_block_broadcast/" + tname + " block=1");

  // Zero-size
  primitives<Device::MPS>::add_block_broadcast(a, b, c, block, a_size, dim_t(0));
  metal::commit_and_wait();
  check(true, "add_block_broadcast/" + tname + " zero-size");

  metal_free(a); metal_free(b); metal_free(c);
  metal_free(a3); metal_free(b3); metal_free(c3);
}

// ---------------------------------------------------------------------------
// mul_batch_broadcast tests
// ---------------------------------------------------------------------------
//
// Semantics: same layout as add_batch_broadcast but multiply.
// c[gid] = a[gid % a_size] * b[gid]

template <typename T>
static void test_mul_batch_broadcast(const std::string& tname) {
  // a=[2,3,4], b=[1,2,3,4,5,6]
  // Expected: [1*2, 2*3, 3*4, 4*2, 5*3, 6*4] = [2, 6, 12, 8, 15, 24]
  const dim_t a_size = 3, b_size = 6;
  T* a = metal_alloc<T>(a_size);
  T* b = metal_alloc<T>(b_size);
  T* c = metal_alloc<T>(b_size);

  a[0] = T(2); a[1] = T(3); a[2] = T(4);
  for (dim_t i = 0; i < b_size; ++i) b[i] = T(i + 1);

  primitives<Device::MPS>::mul_batch_broadcast(a, b, c, a_size, b_size);
  metal::commit_and_wait();

  float expected[] = {2.f, 6.f, 12.f, 8.f, 15.f, 24.f};
  bool ok = true;
  for (dim_t i = 0; i < b_size; ++i)
    if (!near(float(c[i]), expected[i])) { ok = false; break; }
  check(ok, "mul_batch_broadcast/" + tname + " basic");

  // Zero-size
  primitives<Device::MPS>::mul_batch_broadcast(a, b, c, a_size, dim_t(0));
  metal::commit_and_wait();
  check(true, "mul_batch_broadcast/" + tname + " zero-size");

  metal_free(a); metal_free(b); metal_free(c);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.6: Broadcast Primitives ===\n\n");

  std::printf("--- float32 ---\n");
  test_add_batch_broadcast<float>("float32");
  test_add_depth_broadcast<float>("float32");
  test_add_block_broadcast<float>("float32");
  test_mul_batch_broadcast<float>("float32");

  std::printf("--- float16 ---\n");
  test_add_batch_broadcast<ct2_f16>("float16");
  test_add_depth_broadcast<ct2_f16>("float16");
  test_add_block_broadcast<ct2_f16>("float16");
  test_mul_batch_broadcast<ct2_f16>("float16");

  std::printf("--- bfloat16 ---\n");
  test_add_batch_broadcast<ct2_bf16>("bfloat16");
  test_add_depth_broadcast<ct2_bf16>("bfloat16");
  test_add_block_broadcast<ct2_bf16>("bfloat16");
  test_mul_batch_broadcast<ct2_bf16>("bfloat16");

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
