// tests/metal/rotary_test.mm
//
// M6.3 — Rotary::compute<Device::METAL> correctness tests.
//
// Tests the Metal GPU rotary embedding kernel against the CPU reference.
// Calls metal::rotary_metal<T>() directly (same pattern as kv_cache_test.mm).
// Exercises:
//   1.  float32 non-interleave  is_transposed=false  (FA2 layout, LLaMA-style)
//   2.  float32 non-interleave  is_transposed=true   (std layout)
//   3.  float32 interleave      is_transposed=false  (GPT-NeoX-style)
//   4.  float16 non-interleave  is_transposed=false
//   5.  bfloat16 non-interleave is_transposed=false
//   6.  partial rotation (ndims < depth): only first ndims dims rotate
//   7.  batch=1, time=8, heads=4, hd=64  (typical decode/prefill shape)
//   8.  batch=1, time=16, heads=2, hd=32 (GQA-like)
//   9.  float32 interleave      is_transposed=true   (std layout)
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/rotary_test.mm \
//     src/metal/ops_rotary.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o rotary_test && ./rotary_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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
// Metal allocator helpers (same as kv_cache_test.mm)
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

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
  if (ok) { ++g_pass; std::printf("  PASS  %-50s  max_err=%.3e\n", name, err); }
  else     { ++g_fail; std::printf("  FAIL  %-50s  max_err=%.3e  tol=%.3e\n", name, err, tol); }
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

// Build a standard sin/cos table: sin[i,d] = sin(i * inv_freq[d])
// Shape: [max_time, ndims] (contiguous, float32)
static void make_sin_cos(dim_t max_time, dim_t ndims,
                         std::vector<float>& sin_f, std::vector<float>& cos_f) {
  sin_f.resize(static_cast<size_t>(max_time * ndims));
  cos_f.resize(static_cast<size_t>(max_time * ndims));
  std::vector<float> inv_freq(static_cast<size_t>(ndims));
  for (dim_t d = 0; d < ndims; ++d)
    inv_freq[static_cast<size_t>(d)] = 1.f / std::pow(10000.f, float(d * 2) / float(ndims));
  for (dim_t t = 0; t < max_time; ++t)
    for (dim_t d = 0; d < ndims; ++d) {
      float angle = float(t) * inv_freq[static_cast<size_t>(d)];
      sin_f[static_cast<size_t>(t * ndims + d)] = std::sin(angle);
      cos_f[static_cast<size_t>(t * ndims + d)] = std::cos(angle);
    }
}

// CPU reference: mirrors the CUDA/Metal kernel time-index formula.
//   is_transposed=false: t = vec / head_size  (FA2 layout)
//   is_transposed=true:  t = vec % max_time   (std layout)
static void cpu_rotary(const float* x, const float* sin_t, const float* cos_t,
                       float* y, dim_t total_vecs, dim_t max_time, dim_t head_size,
                       dim_t ndims, dim_t depth, bool interleave, bool is_transposed) {
  const dim_t half = ndims / 2;
  for (dim_t v = 0; v < total_vecs; ++v) {
    dim_t t = is_transposed ? (v % max_time) : (v / head_size);
    const float* sx = x   + v * depth;
    float*       sy = y   + v * depth;
    const float* sc = cos_t + t * ndims;
    const float* ss = sin_t + t * ndims;
    for (dim_t d = 0; d < depth; ++d) {
      if (d >= ndims) { sy[d] = sx[d]; continue; }
      float xi = sx[d], cd = sc[d], sd = ss[d];
      if (!interleave) {
        sy[d] = (d < half) ? xi * cd - sx[d + half] * sd
                           : xi * cd + sx[d - half] * sd;
      } else {
        sy[d] = (d % 2 == 0) ? xi * cd - sx[d + 1] * sd
                              : xi * cd + sx[d - 1] * sd;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Generic test runner
// ---------------------------------------------------------------------------

template <typename T>
static void run_rotary_test(const char* name, float tol,
                            dim_t batch, dim_t time, dim_t heads, dim_t hd,
                            dim_t ndims_in,   // 0 = use hd
                            bool interleave, bool is_transposed) {
  const dim_t ndims      = (ndims_in == 0) ? hd : ndims_in;
  const dim_t total_elems = batch * time * heads * hd;
  const dim_t total_vecs  = total_elems / hd;

  // For the sin/cos lookup:
  //   max_time  = is_transposed ? input.dim(-2) : input.dim(-3)
  //             = is_transposed ? time            : time   (both cases)
  //   head_size = is_transposed ? input.dim(-3) : input.dim(-2)
  //             = is_transposed ? heads           : heads  (both cases)
  const dim_t max_time  = time;
  const dim_t head_size = heads;

  rng_state = 0x11223344u ^ (uint32_t)(batch * 97 + time * 31 + heads * 7 + hd);

  // Generate float32 random input
  std::vector<float> x_f(static_cast<size_t>(total_elems));
  for (auto& v : x_f) v = next_float(-1.f, 1.f);

  // Build sin/cos tables [time, ndims]
  std::vector<float> sin_f, cos_f;
  make_sin_cos(time, ndims, sin_f, cos_f);

  // CPU reference
  std::vector<float> ref_out(static_cast<size_t>(total_elems));
  cpu_rotary(x_f.data(), sin_f.data(), cos_f.data(), ref_out.data(),
             total_vecs, max_time, head_size, ndims, hd, interleave, is_transposed);

  // --- Metal path ---
  // Copy data into Metal-backed buffers.
  const dim_t sin_elems = time * ndims;

  T* x_m   = metal_alloc<T>(total_elems);
  T* sin_m = metal_alloc<T>(sin_elems);
  T* cos_m = metal_alloc<T>(sin_elems);
  T* out_m = metal_alloc<T>(total_elems);

  for (dim_t i = 0; i < total_elems; ++i) x_m[i]   = T(x_f  [static_cast<size_t>(i)]);
  for (dim_t i = 0; i < sin_elems;   ++i) {
    sin_m[i] = T(sin_f[static_cast<size_t>(i)]);
    cos_m[i] = T(cos_f[static_cast<size_t>(i)]);
  }

  // Encode rotary kernel (deferred).
  metal::rotary_metal<T>(x_m, sin_m, cos_m, out_m,
                         total_vecs, hd, ndims, max_time, head_size,
                         interleave, is_transposed);

  // Flush and read back.
  metal::commit_and_wait();

  std::vector<float> got_f(static_cast<size_t>(total_elems));
  for (dim_t i = 0; i < total_elems; ++i) got_f[static_cast<size_t>(i)] = float(out_m[i]);

  float err = max_abs_err(ref_out.data(), got_f.data(), total_elems);
  check(name, err, tol);

  metal_free(x_m);
  metal_free(sin_m);
  metal_free(cos_m);
  metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_float32_non_interleave_fa2() {
  run_rotary_test<float>(
    "f32 non-interleave is_transposed=false (FA2 layout)",
    1e-5f, 1, 8, 4, 64, 0, false, false);
}

static void test_float32_non_interleave_std() {
  run_rotary_test<float>(
    "f32 non-interleave is_transposed=true  (std layout)",
    1e-5f, 1, 8, 4, 64, 0, false, true);
}

static void test_float32_interleave_fa2() {
  run_rotary_test<float>(
    "f32 interleave     is_transposed=false (FA2 layout)",
    1e-5f, 1, 8, 4, 64, 0, true, false);
}

static void test_float32_partial_rot() {
  // ndims=32 < depth=64: first 32 dims rotate, last 32 pass through.
  run_rotary_test<float>(
    "f32 non-interleave partial (ndims=32, hd=64)",
    1e-5f, 1, 8, 4, 64, 32, false, false);
}

static void test_float16_non_interleave() {
  run_rotary_test<ct2_f16>(
    "f16 non-interleave is_transposed=false",
    5e-3f, 1, 8, 4, 64, 0, false, false);
}

static void test_bfloat16_non_interleave() {
  run_rotary_test<ct2_bf16>(
    "bf16 non-interleave is_transposed=false",
    5e-2f, 1, 8, 4, 64, 0, false, false);
}

static void test_longer_sequence() {
  run_rotary_test<float>(
    "f32 non-interleave time=64 heads=8 hd=64",
    1e-5f, 1, 64, 8, 64, 0, false, false);
}

static void test_gqa_shape() {
  run_rotary_test<float>(
    "f32 non-interleave time=16 heads=2 hd=32",
    1e-5f, 1, 16, 2, 32, 0, false, false);
}

static void test_interleave_std_layout() {
  run_rotary_test<float>(
    "f32 interleave     is_transposed=true  (std layout)",
    1e-5f, 1, 8, 4, 64, 0, true, true);
}

// Fix 4.4: interleave + partial rotation (ndims < depth).
// Documents that GPU and CPU agree when both rotate only the first ndims
// dimensions and pass through the rest.
static void test_interleave_partial_fa2() {
  // ndims=32, depth=64: first 32 dims interleave-rotate, last 32 passthrough.
  run_rotary_test<float>(
    "f32 interleave partial ndims=32 hd=64 is_transposed=false",
    1e-5f, 1, 8, 4, 64, 32, true, false);
}

static void test_interleave_partial_std() {
  run_rotary_test<float>(
    "f32 interleave partial ndims=32 hd=64 is_transposed=true",
    1e-5f, 1, 8, 4, 64, 32, true, true);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.3 Rotary::compute<Device::METAL> Correctness Tests ===\n\n");

  test_float32_non_interleave_fa2();
  test_float32_non_interleave_std();
  test_float32_interleave_fa2();
  test_float32_partial_rot();
  test_float16_non_interleave();
  test_bfloat16_non_interleave();
  test_longer_sequence();
  test_gqa_shape();
  test_interleave_std_layout();
  test_interleave_partial_fa2();
  test_interleave_partial_std();

  std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
