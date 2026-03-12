// tests/metal/gpu_decode_rope_test.mm
//
// M12.17 — GPU decode RoPE kernel correctness tests.
//
// Compares metal::decode_rope_metal<T> against a CPU reference (apply_rope_half_ref)
// to verify the GPU kernel produces bit-identical (within tolerance) results.
//
// Tests:
//   1. f32  non-interleave  half_dim=16 depth=32  8 vectors
//   2. f32  interleave      half_dim=16 depth=32  8 vectors
//   3. f16  non-interleave  half_dim=16 depth=32  8 vectors
//   4. bf16 non-interleave  half_dim=16 depth=32  8 vectors
//   5. f32  partial rotation half_dim=8  depth=32  4 vectors (ndims=16 < depth)
//   6. f32  large: half_dim=64 depth=128 32 vectors (LLaMA-like)
//   7. f32  full decode pipeline: GPU RoPE + blit_copy + SDPA vs CPU ref
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_MPS \
//     tests/metal/gpu_decode_rope_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/metal/ops_sdpa.mm src/metal/ops_rotary.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -o gpu_decode_rope_test && ./gpu_decode_rope_test

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
  if (ok) { ++g_pass; std::printf("  PASS  %-60s  max_err=%.3e\n", name, err); }
  else     { ++g_fail; std::printf("  FAIL  %-60s  max_err=%.3e  tol=%.3e\n", name, err, tol); }
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
// CPU reference: apply_rope_half_ref<T>
// Exact replica of the formula in flash_attention_metal.mm.
// ---------------------------------------------------------------------------

template <typename T>
static void apply_rope_half_ref(T* x,
                                 const T* cos_row,
                                 const T* sin_row,
                                 dim_t ndims,
                                 dim_t depth,
                                 bool interleave) {
  const dim_t half = ndims / 2;

  if (!interleave) {
    float tmp[512];
    for (dim_t d = 0; d < half; ++d) {
      const float xd = float(x[d]);
      const float xp = float(x[d + half]);
      const float cd = float(cos_row[d]);
      const float sd = float(sin_row[d]);
      tmp[d]        = xd * cd - xp * sd;
      tmp[d + half] = xp * cd + xd * sd;
    }
    for (dim_t d = 0; d < 2 * half; ++d) x[d] = T(tmp[d]);
  } else {
    for (dim_t i = 0; i < half; ++i) {
      const float xe = float(x[2 * i]);
      const float xo = float(x[2 * i + 1]);
      const float ci = float(cos_row[i]);
      const float si = float(sin_row[i]);
      x[2 * i]     = T(xe * ci - xo * si);
      x[2 * i + 1] = T(xo * ci + xe * si);
    }
  }
}

// ---------------------------------------------------------------------------
// run_gpu_rope_test<T>
//
// Compares metal::decode_rope_metal<T> against apply_rope_half_ref<T>.
//   - Allocates [num_vecs, depth] data in Metal memory.
//   - Builds half-dim cos/sin row (random values).
//   - Runs GPU kernel, syncs, compares element-by-element.
// ---------------------------------------------------------------------------

template <typename T>
static void run_gpu_rope_test(
    const char* name, float tol,
    dim_t num_vecs, dim_t depth, dim_t half_dim,
    bool interleave) {

  rng_state = 0xFACE0000u ^ (uint32_t)(num_vecs * 97 + depth * 31 + half_dim * 7);
  const dim_t ndims = half_dim * 2;
  const dim_t total = num_vecs * depth;

  // Generate random input data.
  std::vector<float> data_f(static_cast<size_t>(total));
  for (auto& v : data_f) v = next_float(-2.f, 2.f);

  // Generate random cos/sin row [half_dim].
  std::vector<float> cos_f(static_cast<size_t>(half_dim));
  std::vector<float> sin_f(static_cast<size_t>(half_dim));
  for (auto& v : cos_f) v = next_float(-1.f, 1.f);
  for (auto& v : sin_f) v = next_float(-1.f, 1.f);

  // --- CPU reference path ---
  std::vector<T> cpu_data(static_cast<size_t>(total));
  for (size_t i = 0; i < cpu_data.size(); ++i) cpu_data[i] = T(data_f[i]);

  std::vector<T> cos_T(static_cast<size_t>(half_dim));
  std::vector<T> sin_T(static_cast<size_t>(half_dim));
  for (size_t i = 0; i < cos_T.size(); ++i) cos_T[i] = T(cos_f[i]);
  for (size_t i = 0; i < sin_T.size(); ++i) sin_T[i] = T(sin_f[i]);

  for (dim_t v = 0; v < num_vecs; ++v) {
    apply_rope_half_ref<T>(cpu_data.data() + v * depth,
                            cos_T.data(), sin_T.data(),
                            ndims, depth, interleave);
  }

  // --- GPU path ---
  T* gpu_data = metal_alloc<T>(total);
  for (dim_t i = 0; i < total; ++i) gpu_data[i] = T(data_f[static_cast<size_t>(i)]);

  T* gpu_cos = metal_alloc<T>(half_dim);
  T* gpu_sin = metal_alloc<T>(half_dim);
  for (dim_t i = 0; i < half_dim; ++i) gpu_cos[i] = cos_T[static_cast<size_t>(i)];
  for (dim_t i = 0; i < half_dim; ++i) gpu_sin[i] = sin_T[static_cast<size_t>(i)];

  metal::decode_rope_metal<T>(
      gpu_data, gpu_cos, gpu_sin,
      num_vecs, depth, half_dim, interleave);
  metal::commit_and_wait();

  // Compare.
  std::vector<float> cpu_f(static_cast<size_t>(total));
  std::vector<float> gpu_f(static_cast<size_t>(total));
  for (dim_t i = 0; i < total; ++i) {
    cpu_f[static_cast<size_t>(i)] = float(cpu_data[static_cast<size_t>(i)]);
    gpu_f[static_cast<size_t>(i)] = float(gpu_data[i]);
  }

  float err = max_abs_err(cpu_f.data(), gpu_f.data(), total);
  check(name, err, tol);

  // Also verify passthrough: elements [ndims, depth) should be unchanged.
  if (ndims < depth) {
    float passthrough_err = 0.f;
    for (dim_t v = 0; v < num_vecs; ++v) {
      for (dim_t d = ndims; d < depth; ++d) {
        float orig = data_f[static_cast<size_t>(v * depth + d)];
        float got  = float(gpu_data[v * depth + d]);
        // Compare via T conversion (original was stored as T).
        float expected = float(T(orig));
        passthrough_err = std::max(passthrough_err, std::fabs(expected - got));
      }
    }
    char pname[128];
    std::snprintf(pname, sizeof(pname), "  passthrough [%lld..%lld)", (long long)ndims, (long long)depth);
    check(pname, passthrough_err, 0.f);
  }

  metal_free(gpu_data);
  metal_free(gpu_cos);
  metal_free(gpu_sin);
}

// ---------------------------------------------------------------------------
// CPU reference SDPA (for pipeline test)
// ---------------------------------------------------------------------------

static void cpu_ref_sdpa(const float* q, const float* k, const float* v, float* out,
                          dim_t batch, dim_t sq, dim_t sk, dim_t nh, dim_t nhk, dim_t hd,
                          float scale) {
  const dim_t q_lda  = nh  * hd;
  const dim_t kv_lda = nhk * hd;
  std::vector<float> scores(static_cast<size_t>(sq * sk));

  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h) {
      const dim_t hk = h / (nh / nhk);
      const float* q0   = q   + (b * sq * nh  + h ) * hd;
      const float* k0   = k   + (b * sk * nhk + hk) * hd;
      const float* v0   = v   + (b * sk * nhk + hk) * hd;
      float*       out0 = out + (b * sq * nh  + h ) * hd;

      for (dim_t s = 0; s < sq; ++s) {
        for (dim_t t = 0; t < sk; ++t) {
          float dot = 0.f;
          for (dim_t d = 0; d < hd; ++d)
            dot += q0[s * q_lda + d] * k0[t * kv_lda + d];
          scores[static_cast<size_t>(s * sk + t)] = scale * dot;
        }
      }
      for (dim_t s = 0; s < sq; ++s) {
        float mx = -1e38f;
        for (dim_t t = 0; t < sk; ++t)
          mx = std::max(mx, scores[static_cast<size_t>(s * sk + t)]);
        float sum = 0.f;
        for (dim_t t = 0; t < sk; ++t)
          sum += std::exp(scores[static_cast<size_t>(s * sk + t)] - mx);
        for (dim_t t = 0; t < sk; ++t)
          scores[static_cast<size_t>(s * sk + t)] =
              std::exp(scores[static_cast<size_t>(s * sk + t)] - mx) / sum;
      }
      for (dim_t s = 0; s < sq; ++s) {
        for (dim_t d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (dim_t t = 0; t < sk; ++t)
            acc += scores[static_cast<size_t>(s * sk + t)] * v0[t * kv_lda + d];
          out0[s * q_lda + d] = acc;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Full pipeline test: GPU RoPE + blit_copy + SDPA
// Mirrors the decode path in flash_attention_metal.mm (M12.17).
// ---------------------------------------------------------------------------

static void run_gpu_rope_pipeline_test(
    const char* name, float tol,
    dim_t batch, dim_t nh, dim_t nhk, dim_t hd,
    dim_t ndims, dim_t offset, dim_t prefill_sk,
    bool interleave) {

  rng_state = 0xDEADBEEFu ^ (uint32_t)(batch * 97 + nh * 31 + hd * 7 + offset);

  const dim_t half       = ndims / 2;
  const dim_t max_cache  = prefill_sk + offset + 16;
  const dim_t row_elems  = nhk * hd;
  const dim_t cache_elems = batch * max_cache * nhk * hd;

  // Build half cos/sin tables.
  const dim_t max_pos = max_cache + 4;
  std::vector<float> cos_table(static_cast<size_t>(max_pos * half));
  std::vector<float> sin_table(static_cast<size_t>(max_pos * half));
  for (dim_t t = 0; t < max_pos; ++t) {
    for (dim_t i = 0; i < half; ++i) {
      float inv_freq = 1.f / std::pow(10000.f, float(2 * i) / float(ndims));
      float angle = float(t) * inv_freq;
      cos_table[static_cast<size_t>(t * half + i)] = std::cos(angle);
      sin_table[static_cast<size_t>(t * half + i)] = std::sin(angle);
    }
  }

  // Allocate caches.
  float* cache_k_cpu = new float[cache_elems]();
  float* cache_v_cpu = new float[cache_elems]();
  float* cache_k_gpu = metal_alloc<float>(cache_elems);
  float* cache_v_gpu = metal_alloc<float>(cache_elems);
  std::memset(cache_k_gpu, 0, cache_elems * sizeof(float));
  std::memset(cache_v_gpu, 0, cache_elems * sizeof(float));

  // Fill prefill slots.
  for (dim_t pos = 0; pos < offset; ++pos) {
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t j = 0; j < row_elems; ++j) {
        float kval = std::sin(float((pos * batch + b) * row_elems + j + 1) * 0.11f);
        float vval = std::cos(float((pos * batch + b) * row_elems + j + 1) * 0.09f);
        dim_t idx = (b * max_cache + pos) * row_elems + j;
        cache_k_cpu[idx] = kval;
        cache_v_cpu[idx] = vval;
        cache_k_gpu[idx] = kval;
        cache_v_gpu[idx] = vval;
      }
    }
  }

  // Generate Q, K_new, V_new.
  const dim_t q_elems  = batch * nh  * hd;
  const dim_t kv_elems = batch * nhk * hd;
  std::vector<float> qf(static_cast<size_t>(q_elems));
  std::vector<float> kf(static_cast<size_t>(kv_elems));
  std::vector<float> vf(static_cast<size_t>(kv_elems));
  for (auto& v : qf) v = next_float(-1.f, 1.f);
  for (auto& v : kf) v = next_float(-1.f, 1.f);
  for (auto& v : vf) v = next_float(-1.f, 1.f);

  // --- CPU reference: apply_rope_half_ref + memcpy + SDPA ---
  std::vector<float> q_rot(qf), k_rot(kf);
  const float* cos_row = cos_table.data() + offset * half;
  const float* sin_row = sin_table.data() + offset * half;

  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h)
      apply_rope_half_ref<float>(q_rot.data() + (b * nh + h) * hd,
                                  cos_row, sin_row, ndims, hd, interleave);
    for (dim_t hk = 0; hk < nhk; ++hk)
      apply_rope_half_ref<float>(k_rot.data() + (b * nhk + hk) * hd,
                                  cos_row, sin_row, ndims, hd, interleave);
    std::memcpy(cache_k_cpu + (b * max_cache + offset) * row_elems,
                k_rot.data() + b * row_elems, row_elems * sizeof(float));
    std::memcpy(cache_v_cpu + (b * max_cache + offset) * row_elems,
                vf.data()   + b * row_elems, row_elems * sizeof(float));
  }

  const dim_t sk_eff = offset + 1;
  std::vector<float> ref_out(static_cast<size_t>(q_elems));
  cpu_ref_sdpa(q_rot.data(), cache_k_cpu, cache_v_cpu, ref_out.data(),
               batch, 1, sk_eff, nh, nhk, hd, 1.f / std::sqrt(float(hd)));

  // --- GPU path: decode_rope_metal + blit_copy + sdpa_metal ---
  float* q_gpu   = metal_alloc<float>(q_elems);
  float* k_gpu   = metal_alloc<float>(kv_elems);
  float* out_gpu = metal_alloc<float>(q_elems);
  for (dim_t i = 0; i < q_elems;  ++i) q_gpu[i] = qf[static_cast<size_t>(i)];
  for (dim_t i = 0; i < kv_elems; ++i) k_gpu[i] = kf[static_cast<size_t>(i)];

  // Allocate GPU cos/sin row.
  float* gpu_cos = metal_alloc<float>(half);
  float* gpu_sin = metal_alloc<float>(half);
  std::memcpy(gpu_cos, cos_row, half * sizeof(float));
  std::memcpy(gpu_sin, sin_row, half * sizeof(float));

  // GPU RoPE on Q.
  metal::decode_rope_metal<float>(
      q_gpu, gpu_cos, gpu_sin,
      batch * nh, hd, half, interleave);

  // GPU RoPE on K_new.
  metal::decode_rope_metal<float>(
      k_gpu, gpu_cos, gpu_sin,
      batch * nhk, hd, half, interleave);

  // GPU blit copy K_new/V_new to cache.
  const size_t row_bytes = 1 * row_elems * sizeof(float);
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = cache_k_gpu + (b * max_cache + offset) * row_elems;
    float* vd = cache_v_gpu + (b * max_cache + offset) * row_elems;
    const float* ks = k_gpu + b * row_elems;
    const float* vs = (float*)vf.data() + b * row_elems;
    // Copy V_new to Metal-registered memory first (blit_copy needs registered ptrs).
    float* v_tmp = metal_alloc<float>(row_elems);
    std::memcpy(v_tmp, vs, row_bytes);
    metal::blit_copy(ks, kd, row_bytes);
    metal::blit_copy(v_tmp, vd, row_bytes);
    metal_free(v_tmp);
  }

  // SDPA.
  metal::sdpa_metal<float>(
      q_gpu, cache_k_gpu, cache_v_gpu, out_gpu,
      batch, 1, sk_eff, nh, nhk, hd,
      1.f / std::sqrt(float(hd)), /*is_causal=*/false,
      max_cache * nhk * hd);
  metal::commit_and_wait();

  // Compare.
  std::vector<float> got_f(static_cast<size_t>(q_elems));
  for (dim_t i = 0; i < q_elems; ++i)
    got_f[static_cast<size_t>(i)] = out_gpu[i];

  float err = max_abs_err(ref_out.data(), got_f.data(), q_elems);
  check(name, err, tol);

  metal_free(q_gpu);
  metal_free(k_gpu);
  metal_free(out_gpu);
  metal_free(gpu_cos);
  metal_free(gpu_sin);
  metal_free(cache_k_gpu);
  metal_free(cache_v_gpu);
  delete[] cache_k_cpu;
  delete[] cache_v_cpu;
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

static void test_f32_noninterleave() {
  run_gpu_rope_test<float>(
    "f32 non-interleave half=16 depth=32 vecs=8",
    1e-6f, 8, 32, 16, /*interleave=*/false);
}

static void test_f32_interleave() {
  run_gpu_rope_test<float>(
    "f32 interleave half=16 depth=32 vecs=8",
    1e-6f, 8, 32, 16, /*interleave=*/true);
}

static void test_f16_noninterleave() {
  run_gpu_rope_test<ct2_f16>(
    "f16 non-interleave half=16 depth=32 vecs=8",
    5e-3f, 8, 32, 16, /*interleave=*/false);
}

static void test_bf16_noninterleave() {
  run_gpu_rope_test<ct2_bf16>(
    "bf16 non-interleave half=16 depth=32 vecs=8",
    5e-2f, 8, 32, 16, /*interleave=*/false);
}

static void test_f32_partial_rotation() {
  // half_dim=8, depth=32 → ndims=16, passthrough [16,32)
  run_gpu_rope_test<float>(
    "f32 partial rotation half=8 depth=32 vecs=4",
    1e-6f, 4, 32, 8, /*interleave=*/false);
}

static void test_f32_large() {
  // LLaMA-like: half_dim=64, depth=128, 32 vectors.
  run_gpu_rope_test<float>(
    "f32 large half=64 depth=128 vecs=32",
    1e-6f, 32, 128, 64, /*interleave=*/false);
}

static void test_pipeline_f32() {
  // Full pipeline: GPU RoPE + blit_copy + SDPA.
  run_gpu_rope_pipeline_test(
    "f32 pipeline: GPU RoPE + blit + SDPA nh=4 nhk=4 hd=32 offset=4",
    1e-4f, 1, 4, 4, 32, 32, 4, 4, /*interleave=*/false);
}

static void test_pipeline_gqa() {
  // GQA pipeline: nh=8, nhk=2.
  run_gpu_rope_pipeline_test(
    "f32 pipeline GQA: nh=8 nhk=2 hd=32 offset=4",
    1e-4f, 1, 8, 2, 32, 32, 4, 4, /*interleave=*/false);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M12.17 GPU Decode RoPE Kernel Correctness Tests ===\n\n");

  // Direct kernel tests.
  test_f32_noninterleave();
  test_f32_interleave();
  test_f16_noninterleave();
  test_bf16_noninterleave();
  test_f32_partial_rotation();
  test_f32_large();

  // Full pipeline tests.
  test_pipeline_f32();
  test_pipeline_gqa();

  std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
