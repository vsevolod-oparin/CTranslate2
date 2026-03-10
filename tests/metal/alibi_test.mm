// tests/metal/alibi_test.mm
//
// M6.4 — metal::alibi_add_metal<T>() correctness tests.
//
// Calls metal::alibi_add_metal<T>() directly (no StorageView dependency).
// CPU reference mirrors the MSL kernel formula exactly.
//
// Tests:
//   1.  float32  basic [1,4,1,8]      alibi_offset=0  (query_length==1, decode shape)
//   2.  float32  multi-query [1,4,4,8] alibi_offset=0  (prefill shape)
//   3.  float32  batched [2,4,1,8]     alibi_offset=0
//   4.  float32  alibi_offset>0 [1,4,1,8] offset=4 (cached_kl=12)
//   5.  float16  basic [1,4,1,8]      alibi_offset=0
//   6.  bfloat16 basic [1,4,1,8]      alibi_offset=0
//   7.  float32  many heads [1,8,1,16] alibi_offset=0
//   8.  float32  large query [1,8,16,32] alibi_offset=0 (full prefill)
//   9.  float32  non-zero offset on multi-query [1,4,4,8] offset=2 (cached_kl=10)
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/alibi_test.mm \
//     src/metal/ops_alibi.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm src/metal/ops_rotary.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o alibi_test && ./alibi_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/ops_metal.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Metal allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static float max_abs_err(const float* ref, const float* got, dim_t n) {
  float err = 0.f;
  for (dim_t i = 0; i < n; ++i) err = std::max(err, std::fabs(ref[i] - got[i]));
  return err;
}

static void check(const char* name, float err, float tol) {
  bool ok = std::isfinite(err) && err <= tol;
  if (ok) { ++g_pass; std::printf("  PASS  %-52s  max_err=%.3e\n", name, err); }
  else     { ++g_fail; std::printf("  FAIL  %-52s  max_err=%.3e  tol=%.3e\n", name, err, tol); }
}

// PRNG
static uint32_t rng_state = 0xABCD1234u;
static float next_float(float lo, float hi) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  float t = static_cast<float>(rng_state) / static_cast<float>(UINT32_MAX);
  return lo + t * (hi - lo);
}

// ---------------------------------------------------------------------------
// CPU reference
//
// Mirrors the MSL kernel formula:
//   vec  = row index in [0, batch*num_heads*query_length)
//   k    = key position in [0, key_length)
//   h    = (vec / query_length) % num_heads
//   output[vec*kl+k] = input[vec*kl+k] + alibi[h*cached_kl + alibi_offset + k]
// ---------------------------------------------------------------------------

static void cpu_alibi_add(
    const float* input, const float* alibi, float* output,
    dim_t batch_size, dim_t num_heads, dim_t query_length, dim_t key_length,
    dim_t cached_key_length, dim_t alibi_offset) {
  const dim_t total_rows = batch_size * num_heads * query_length;
  for (dim_t vec = 0; vec < total_rows; ++vec) {
    const dim_t h = (vec / query_length) % num_heads;
    for (dim_t k = 0; k < key_length; ++k) {
      dim_t in_idx    = vec * key_length + k;
      dim_t alibi_idx = h * cached_key_length + alibi_offset + k;
      output[in_idx]  = input[in_idx] + alibi[alibi_idx];
    }
  }
}

// ---------------------------------------------------------------------------
// Generic test runner
// ---------------------------------------------------------------------------

template <typename T>
static void run_alibi_test(
    const char* name, float tol,
    dim_t batch_size, dim_t num_heads, dim_t query_length, dim_t key_length,
    dim_t cached_key_length, dim_t alibi_offset) {

  rng_state = 0x12345678u ^ (uint32_t)(batch_size * 97 + num_heads * 31
                             + query_length * 17 + key_length * 7);

  const dim_t input_elems = batch_size * num_heads * query_length * key_length;
  const dim_t alibi_elems = num_heads * cached_key_length;  // [1, num_heads, 1, cached_kl]

  // Generate random float32 data
  std::vector<float> input_f(static_cast<size_t>(input_elems));
  std::vector<float> alibi_f(static_cast<size_t>(alibi_elems));
  for (auto& v : input_f) v = next_float(-5.f, 5.f);
  for (auto& v : alibi_f) v = next_float(-2.f, 0.f);   // ALiBi values are typically negative

  // CPU reference (float32)
  std::vector<float> ref_out(static_cast<size_t>(input_elems));
  cpu_alibi_add(input_f.data(), alibi_f.data(), ref_out.data(),
                batch_size, num_heads, query_length, key_length,
                cached_key_length, alibi_offset);

  // --- Metal path ---
  T* in_m    = metal_alloc<T>(input_elems);
  T* alibi_m = metal_alloc<T>(alibi_elems);
  T* out_m   = metal_alloc<T>(input_elems);

  for (dim_t i = 0; i < input_elems; ++i) in_m[i]    = T(input_f[static_cast<size_t>(i)]);
  for (dim_t i = 0; i < alibi_elems; ++i) alibi_m[i] = T(alibi_f[static_cast<size_t>(i)]);

  metal::alibi_add_metal<T>(in_m, alibi_m, out_m,
                            batch_size, num_heads, query_length, key_length,
                            cached_key_length, alibi_offset);
  metal::commit_and_wait();

  std::vector<float> got_f(static_cast<size_t>(input_elems));
  for (dim_t i = 0; i < input_elems; ++i) got_f[static_cast<size_t>(i)] = float(out_m[i]);

  float err = max_abs_err(ref_out.data(), got_f.data(), input_elems);
  check(name, err, tol);

  metal_free(in_m);
  metal_free(alibi_m);
  metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

static void test_f32_basic_decode() {
  // Typical decode: query_length=1, key_length=8, 4 heads
  run_alibi_test<float>(
    "f32 [1,4,1,8]  offset=0  (decode shape)",
    1e-5f, 1, 4, 1, 8, 8, 0);
}

static void test_f32_prefill() {
  // Prefill: query_length=key_length
  run_alibi_test<float>(
    "f32 [1,4,4,8]  offset=0  (prefill, ql==kl)",
    1e-5f, 1, 4, 4, 8, 8, 0);
}

static void test_f32_batched() {
  // Batch > 1
  run_alibi_test<float>(
    "f32 [2,4,1,8]  offset=0  (batch=2)",
    1e-5f, 2, 4, 1, 8, 8, 0);
}

static void test_f32_alibi_offset() {
  // cached_key_length=12, key_length=8, offset=4 (decode after 4 cached tokens)
  run_alibi_test<float>(
    "f32 [1,4,1,8]  offset=4  cached_kl=12",
    1e-5f, 1, 4, 1, 8, 12, 4);
}

static void test_f16_basic() {
  run_alibi_test<ct2_f16>(
    "f16 [1,4,1,8]  offset=0",
    5e-3f, 1, 4, 1, 8, 8, 0);
}

static void test_bf16_basic() {
  run_alibi_test<ct2_bf16>(
    "bf16 [1,4,1,8] offset=0",
    5e-2f, 1, 4, 1, 8, 8, 0);
}

static void test_f32_many_heads() {
  // 8 heads, key_length=16
  run_alibi_test<float>(
    "f32 [1,8,1,16] offset=0  (8 heads)",
    1e-5f, 1, 8, 1, 16, 16, 0);
}

static void test_f32_large_prefill() {
  // Larger prefill: 16 query positions, 32 key positions, 8 heads
  run_alibi_test<float>(
    "f32 [1,8,16,32] offset=0  (large prefill)",
    1e-5f, 1, 8, 16, 32, 32, 0);
}

static void test_f32_offset_multiquery() {
  // query_length > 1 AND alibi_offset > 0 (chunk-prefill with cache)
  // cached_kl=10, key_length=8, offset=2 (2 previously cached tokens)
  run_alibi_test<float>(
    "f32 [1,4,4,8]  offset=2  cached_kl=10",
    1e-5f, 1, 4, 4, 8, 10, 2);
}

// Fix 4.5: large key_length stresses the 2D dispatch threadgroup-y clamping.
static void test_f32_large_kl_decode() {
  // key_length=512: forces tg_kl = min(512, maxTotalThreadsPerThreadgroup).
  run_alibi_test<float>(
    "f32 [1,8,1,512]  offset=0  (large kl decode)",
    1e-5f, 1, 8, 1, 512, 512, 0);
}

static void test_f32_large_kl_prefill() {
  // key_length=1024 with query_length=8: stresses rows*kl = 8*8*1024 = 65536 threads.
  run_alibi_test<float>(
    "f32 [1,8,8,1024] offset=0  (large kl prefill)",
    1e-5f, 1, 8, 8, 1024, 1024, 0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.4 alibi_add_metal<T>() Correctness Tests ===\n\n");

  test_f32_basic_decode();
  test_f32_prefill();
  test_f32_batched();
  test_f32_alibi_offset();
  test_f16_basic();
  test_bf16_basic();
  test_f32_many_heads();
  test_f32_large_prefill();
  test_f32_offset_multiquery();
  test_f32_large_kl_decode();
  test_f32_large_kl_prefill();

  std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
