// tests/metal/flash_mha_decode_test.mm
//
// M12.18 — FlashMHA decode path correctness test.
//
// Tests the FlashAttention decode path (offset > 0) on MPS by comparing
// GPU results (blit_copy + sdpa_metal) against a CPU reference.
//
// Tests:
//   1. No-RoPE decode: blit K/V to cache → sdpa_metal vs CPU ref
//   2. With-RoPE decode: decode_rope_metal + blit → sdpa_metal vs CPU ref
//   3. Multi-step decode: 4 steps, accumulating cache, check each step
//   4. GQA decode: num_heads=8, num_heads_k=2
//   5. Larger dimensions (LLaMA-like): hd=64, nh=8, nhk=4
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/flash_mha_decode_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/metal/ops_sdpa.mm src/metal/ops_rotary.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -framework Accelerate \
//     -o flash_mha_decode_test && ./flash_mha_decode_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
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
// Test counters
// ---------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;

static float max_abs_err(const float* ref, const float* got, dim_t n) {
  float err = 0.f;
  for (dim_t i = 0; i < n; ++i)
    err = std::max(err, std::fabs(ref[i] - got[i]));
  return err;
}

static void check(const char* name, float err, float tol) {
  bool ok = std::isfinite(err) && err <= tol;
  if (ok) { ++g_pass; std::printf("  PASS  %-60s  max_err=%.3e\n", name, err); }
  else     { ++g_fail; std::printf("  FAIL  %-60s  max_err=%.3e  tol=%.3e\n", name, err, tol); }
}

// ---------------------------------------------------------------------------
// CPU reference: SDPA (no mask, no causal for sq=1)
// Layout: [batch, seqlen, num_heads, head_dim] — Flash layout
// ---------------------------------------------------------------------------
static void sdpa_cpu_ref(const float* q, const float* k, const float* v,
                         float* out,
                         dim_t batch, dim_t sq, dim_t sk,
                         dim_t nh, dim_t nhk, dim_t hd,
                         float scale,
                         dim_t kv_batch_stride = 0) {
  const dim_t q_lda  = nh  * hd;
  const dim_t kv_lda = nhk * hd;
  const dim_t kv_bstride = (kv_batch_stride > 0)
                            ? kv_batch_stride
                            : sk * nhk * hd;

  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h) {
      const dim_t hk = h % nhk;
      for (dim_t qi = 0; qi < sq; ++qi) {
        const float* qp = q + b * sq * q_lda + qi * q_lda + h * hd;
        float* op       = out + b * sq * q_lda + qi * q_lda + h * hd;

        // Compute scores
        std::vector<float> scores(sk);
        float max_s = -1e30f;
        for (dim_t j = 0; j < sk; ++j) {
          const float* kp = k + b * kv_bstride + j * kv_lda + hk * hd;
          float dot = 0.f;
          for (dim_t d = 0; d < hd; ++d)
            dot += qp[d] * kp[d];
          scores[j] = dot * scale;
          max_s = std::max(max_s, scores[j]);
        }

        // Softmax
        float sum = 0.f;
        for (dim_t j = 0; j < sk; ++j) {
          scores[j] = std::exp(scores[j] - max_s);
          sum += scores[j];
        }
        for (dim_t j = 0; j < sk; ++j) scores[j] /= sum;

        // Weighted sum
        for (dim_t d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (dim_t j = 0; j < sk; ++j) {
            const float* vp = v + b * kv_bstride + j * kv_lda + hk * hd;
            acc += scores[j] * vp[d];
          }
          op[d] = acc;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// CPU RoPE reference (half-table, non-interleave)
// ---------------------------------------------------------------------------
static void rope_cpu_ref(float* x, const float* cos_row, const float* sin_row,
                         dim_t num_vecs, dim_t depth, dim_t half_dim) {
  for (dim_t v = 0; v < num_vecs; ++v) {
    float* xv = x + v * depth;
    for (dim_t d = 0; d < half_dim; ++d) {
      float xd = xv[d];
      float xp = xv[d + half_dim];
      float cd = cos_row[d];
      float sd = sin_row[d];
      xv[d]           = xd * cd - xp * sd;
      xv[d + half_dim] = xp * cd + xd * sd;
    }
  }
}

// ---------------------------------------------------------------------------
// Fill with deterministic pseudorandom values in [-1, 1]
// ---------------------------------------------------------------------------
static void fill_rand(float* p, dim_t n, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (dim_t i = 0; i < n; ++i) p[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// Upload: copy from host to Metal buffer
// ---------------------------------------------------------------------------
static void upload(float* dst, const float* src, dim_t n) {
  std::memcpy(dst, src, n * sizeof(float));
}

// ---------------------------------------------------------------------------
// Test 1: No-RoPE decode — blit K/V to cache, sdpa_metal
//
// Simulates FlashAttention decode path without rotary embeddings.
// This is the code path used when rotary_cos == nullptr.
// ---------------------------------------------------------------------------
static void test_no_rope_decode() {
  std::printf("\n--- Test 1: No-RoPE decode (blit + SDPA) ---\n");

  const dim_t batch = 1, sq = 1, nh = 4, nhk = 4, hd = 16;
  const dim_t prefill_len = 4;  // tokens already in cache
  const dim_t total_cache = prefill_len + 8;  // pre-allocated cache size
  const dim_t offset = prefill_len;
  const dim_t sk_eff = offset + sq;  // effective K sequence length after write
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  // Host data
  const dim_t q_n     = batch * sq * nh * hd;
  const dim_t knew_n  = batch * sq * nhk * hd;
  const dim_t cache_n = batch * total_cache * nhk * hd;
  const dim_t out_n   = q_n;

  std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);

  fill_rand(h_q.data(), q_n, 100);
  fill_rand(h_knew.data(), knew_n, 200);
  fill_rand(h_vnew.data(), knew_n, 300);
  fill_rand(h_kcache.data(), cache_n, 400);  // prefill data (positions 0..prefill_len-1)
  fill_rand(h_vcache.data(), cache_n, 500);

  // --- CPU reference ---
  // Copy knew/vnew into cache at offset, then SDPA over [0, sk_eff)
  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = cpu_kcache.data() + (b * total_cache + offset) * row_elem;
    float* vd = cpu_vcache.data() + (b * total_cache + offset) * row_elem;
    const float* ks = h_knew.data() + b * sq * row_elem;
    const float* vs = h_vnew.data() + b * sq * row_elem;
    std::memcpy(kd, ks, sq * row_elem * sizeof(float));
    std::memcpy(vd, vs, sq * row_elem * sizeof(float));
  }
  std::vector<float> cpu_out(out_n, 0.f);
  sdpa_cpu_ref(h_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
               batch, sq, sk_eff, nh, nhk, hd, scale,
               total_cache * nhk * hd);

  // --- GPU path ---
  float* g_q      = metal_alloc<float>(q_n);
  float* g_knew   = metal_alloc<float>(knew_n);
  float* g_vnew   = metal_alloc<float>(knew_n);
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  float* g_out    = metal_alloc<float>(out_n);

  upload(g_q, h_q.data(), q_n);
  upload(g_knew, h_knew.data(), knew_n);
  upload(g_vnew, h_vnew.data(), knew_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);

  // Flush uploads
  metal::commit_and_wait();

  // Blit copy K_new and V_new into cache at offset
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = g_kcache + (b * total_cache + offset) * row_elem;
    float* vd = g_vcache + (b * total_cache + offset) * row_elem;
    const float* ks = g_knew + b * sq * row_elem;
    const float* vs = g_vnew + b * sq * row_elem;
    metal::blit_copy(ks, kd, sq * row_elem * sizeof(float));
    metal::blit_copy(vs, vd, sq * row_elem * sizeof(float));
  }

  // SDPA over cache[0..sk_eff)
  metal::sdpa_metal<float>(
      g_q, g_kcache, g_vcache, g_out,
      batch, sq, sk_eff, nh, nhk, hd,
      scale, /*is_causal=*/false,
      total_cache * nhk * hd, /*beam_size=*/1);

  metal::commit_and_wait();

  // Compare
  float err = max_abs_err(cpu_out.data(), g_out, out_n);
  check("no-RoPE decode: blit + SDPA", err, 1e-5f);

  // Also verify the cache was written correctly
  std::vector<float> gpu_kcache_readback(cache_n);
  std::memcpy(gpu_kcache_readback.data(), g_kcache, cache_n * sizeof(float));
  float cache_err = max_abs_err(cpu_kcache.data(), gpu_kcache_readback.data(), cache_n);
  check("no-RoPE decode: cache correctness", cache_err, 0.f);

  metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
  metal_free(g_kcache); metal_free(g_vcache); metal_free(g_out);
}

// ---------------------------------------------------------------------------
// Test 2: With-RoPE decode — decode_rope_metal + blit + sdpa_metal
// ---------------------------------------------------------------------------
static void test_rope_decode() {
  std::printf("\n--- Test 2: With-RoPE decode (GPU RoPE + blit + SDPA) ---\n");

  const dim_t batch = 1, sq = 1, nh = 4, nhk = 4, hd = 16;
  const dim_t half_dim = hd / 2;
  const dim_t prefill_len = 4;
  const dim_t total_cache = prefill_len + 8;
  const dim_t offset = prefill_len;
  const dim_t sk_eff = offset + sq;
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  const dim_t q_n     = batch * sq * nh * hd;
  const dim_t knew_n  = batch * sq * nhk * hd;
  const dim_t cache_n = batch * total_cache * nhk * hd;
  const dim_t out_n   = q_n;

  std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);

  fill_rand(h_q.data(), q_n, 110);
  fill_rand(h_knew.data(), knew_n, 210);
  fill_rand(h_vnew.data(), knew_n, 310);
  // Prefill cache: positions 0..prefill_len-1 already have RoPE applied
  fill_rand(h_kcache.data(), cache_n, 410);
  fill_rand(h_vcache.data(), cache_n, 510);

  // Cos/sin table: [max_positions, half_dim]
  const dim_t max_positions = total_cache;
  std::vector<float> h_cos(max_positions * half_dim);
  std::vector<float> h_sin(max_positions * half_dim);
  fill_rand(h_cos.data(), max_positions * half_dim, 600);
  fill_rand(h_sin.data(), max_positions * half_dim, 700);

  // --- CPU reference ---
  std::vector<float> cpu_q(h_q), cpu_knew(h_knew);
  const float* cos_row = h_cos.data() + offset * half_dim;
  const float* sin_row = h_sin.data() + offset * half_dim;

  // Apply RoPE to Q: batch * nh vectors
  rope_cpu_ref(cpu_q.data(), cos_row, sin_row, batch * nh, hd, half_dim);
  // Apply RoPE to K_new: batch * nhk vectors
  rope_cpu_ref(cpu_knew.data(), cos_row, sin_row, batch * nhk, hd, half_dim);

  // Copy rotated K_new and V_new into cache at offset
  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = cpu_kcache.data() + (b * total_cache + offset) * row_elem;
    float* vd = cpu_vcache.data() + (b * total_cache + offset) * row_elem;
    std::memcpy(kd, cpu_knew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
    std::memcpy(vd, h_vnew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
  }

  std::vector<float> cpu_out(out_n, 0.f);
  sdpa_cpu_ref(cpu_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
               batch, sq, sk_eff, nh, nhk, hd, scale,
               total_cache * nhk * hd);

  // --- GPU path ---
  float* g_q      = metal_alloc<float>(q_n);
  float* g_knew   = metal_alloc<float>(knew_n);
  float* g_vnew   = metal_alloc<float>(knew_n);
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  float* g_out    = metal_alloc<float>(out_n);
  float* g_cos    = metal_alloc<float>(max_positions * half_dim);
  float* g_sin    = metal_alloc<float>(max_positions * half_dim);

  upload(g_q, h_q.data(), q_n);
  upload(g_knew, h_knew.data(), knew_n);
  upload(g_vnew, h_vnew.data(), knew_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);
  upload(g_cos, h_cos.data(), max_positions * half_dim);
  upload(g_sin, h_sin.data(), max_positions * half_dim);

  metal::commit_and_wait();

  // GPU RoPE on Q
  const float* g_cos_row = g_cos + offset * half_dim;
  const float* g_sin_row = g_sin + offset * half_dim;
  metal::decode_rope_metal<float>(
      g_q, g_cos_row, g_sin_row,
      batch * nh, hd, half_dim, /*interleave=*/false);

  // GPU RoPE on K_new
  metal::decode_rope_metal<float>(
      g_knew, g_cos_row, g_sin_row,
      batch * nhk, hd, half_dim, /*interleave=*/false);

  // Blit copy K_new/V_new into cache
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = g_kcache + (b * total_cache + offset) * row_elem;
    float* vd = g_vcache + (b * total_cache + offset) * row_elem;
    metal::blit_copy(g_knew + b * sq * row_elem, kd, sq * row_elem * sizeof(float));
    metal::blit_copy(g_vnew + b * sq * row_elem, vd, sq * row_elem * sizeof(float));
  }

  // SDPA
  metal::sdpa_metal<float>(
      g_q, g_kcache, g_vcache, g_out,
      batch, sq, sk_eff, nh, nhk, hd,
      scale, false,
      total_cache * nhk * hd, 1);

  metal::commit_and_wait();

  float err = max_abs_err(cpu_out.data(), g_out, out_n);
  check("with-RoPE decode: RoPE + blit + SDPA", err, 1e-5f);

  // Check Q was rotated correctly
  std::vector<float> gpu_q_readback(q_n);
  std::memcpy(gpu_q_readback.data(), g_q, q_n * sizeof(float));
  float q_err = max_abs_err(cpu_q.data(), gpu_q_readback.data(), q_n);
  check("with-RoPE decode: Q rotation", q_err, 1e-6f);

  metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
  metal_free(g_kcache); metal_free(g_vcache); metal_free(g_out);
  metal_free(g_cos); metal_free(g_sin);
}

// ---------------------------------------------------------------------------
// Test 3: Multi-step decode (4 steps, accumulating cache)
//
// Simulates what Generator does: prefill fills cache[0..3], then 4 decode
// steps fill cache[4], cache[5], cache[6], cache[7].
// ---------------------------------------------------------------------------
static void test_multi_step_decode() {
  std::printf("\n--- Test 3: Multi-step decode (4 steps) ---\n");

  const dim_t batch = 1, nh = 4, nhk = 4, hd = 16;
  const dim_t prefill_len = 4;
  const dim_t total_cache = 16;
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  const dim_t cache_n = batch * total_cache * nhk * hd;

  // Initialize cache with prefill data
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);
  fill_rand(h_kcache.data(), cache_n, 1000);
  fill_rand(h_vcache.data(), cache_n, 1001);

  // GPU cache
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);

  // CPU reference cache (starts same as GPU)
  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);

  bool all_ok = true;

  for (int step = 0; step < 4; ++step) {
    const dim_t offset = prefill_len + step;
    const dim_t sk_eff = offset + 1;

    // Generate new Q, K, V for this step
    const dim_t q_n    = batch * 1 * nh * hd;
    const dim_t knew_n = batch * 1 * nhk * hd;

    std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
    fill_rand(h_q.data(), q_n, 2000 + step * 100);
    fill_rand(h_knew.data(), knew_n, 3000 + step * 100);
    fill_rand(h_vnew.data(), knew_n, 4000 + step * 100);

    // --- CPU reference ---
    for (dim_t b = 0; b < batch; ++b) {
      float* kd = cpu_kcache.data() + (b * total_cache + offset) * row_elem;
      float* vd = cpu_vcache.data() + (b * total_cache + offset) * row_elem;
      std::memcpy(kd, h_knew.data() + b * 1 * row_elem, row_elem * sizeof(float));
      std::memcpy(vd, h_vnew.data() + b * 1 * row_elem, row_elem * sizeof(float));
    }
    std::vector<float> cpu_out(q_n, 0.f);
    sdpa_cpu_ref(h_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
                 batch, 1, sk_eff, nh, nhk, hd, scale,
                 total_cache * nhk * hd);

    // --- GPU path ---
    float* g_q    = metal_alloc<float>(q_n);
    float* g_knew = metal_alloc<float>(knew_n);
    float* g_vnew = metal_alloc<float>(knew_n);
    float* g_out  = metal_alloc<float>(q_n);

    upload(g_q, h_q.data(), q_n);
    upload(g_knew, h_knew.data(), knew_n);
    upload(g_vnew, h_vnew.data(), knew_n);
    metal::commit_and_wait();

    // Blit copy
    for (dim_t b = 0; b < batch; ++b) {
      float* kd = g_kcache + (b * total_cache + offset) * row_elem;
      float* vd = g_vcache + (b * total_cache + offset) * row_elem;
      metal::blit_copy(g_knew + b * row_elem, kd, row_elem * sizeof(float));
      metal::blit_copy(g_vnew + b * row_elem, vd, row_elem * sizeof(float));
    }

    // SDPA
    metal::sdpa_metal<float>(
        g_q, g_kcache, g_vcache, g_out,
        batch, 1, sk_eff, nh, nhk, hd,
        scale, false,
        total_cache * nhk * hd, 1);

    metal::commit_and_wait();

    float err = max_abs_err(cpu_out.data(), g_out, q_n);
    char label[128];
    std::snprintf(label, sizeof(label),
                  "multi-step decode step=%d offset=%d sk_eff=%d",
                  step, (int)offset, (int)sk_eff);
    check(label, err, 1e-5f);
    if (err > 1e-5f) all_ok = false;

    metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
    metal_free(g_out);
  }

  // Final cache comparison
  metal::commit_and_wait();
  std::vector<float> gpu_kcache_rb(cache_n);
  std::memcpy(gpu_kcache_rb.data(), g_kcache, cache_n * sizeof(float));
  float cache_err = max_abs_err(cpu_kcache.data(), gpu_kcache_rb.data(), cache_n);
  check("multi-step: final cache match", cache_err, 0.f);

  metal_free(g_kcache); metal_free(g_vcache);
}

// ---------------------------------------------------------------------------
// Test 4: GQA decode (nh=8, nhk=2)
// ---------------------------------------------------------------------------
static void test_gqa_decode() {
  std::printf("\n--- Test 4: GQA decode (nh=8, nhk=2) ---\n");

  const dim_t batch = 2, sq = 1, nh = 8, nhk = 2, hd = 16;
  const dim_t prefill_len = 6;
  const dim_t total_cache = prefill_len + 4;
  const dim_t offset = prefill_len;
  const dim_t sk_eff = offset + sq;
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  const dim_t q_n     = batch * sq * nh * hd;
  const dim_t knew_n  = batch * sq * nhk * hd;
  const dim_t cache_n = batch * total_cache * nhk * hd;
  const dim_t out_n   = q_n;

  std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);

  fill_rand(h_q.data(), q_n, 5000);
  fill_rand(h_knew.data(), knew_n, 5100);
  fill_rand(h_vnew.data(), knew_n, 5200);
  fill_rand(h_kcache.data(), cache_n, 5300);
  fill_rand(h_vcache.data(), cache_n, 5400);

  // CPU ref
  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = cpu_kcache.data() + (b * total_cache + offset) * row_elem;
    float* vd = cpu_vcache.data() + (b * total_cache + offset) * row_elem;
    std::memcpy(kd, h_knew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
    std::memcpy(vd, h_vnew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
  }
  std::vector<float> cpu_out(out_n, 0.f);
  sdpa_cpu_ref(h_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
               batch, sq, sk_eff, nh, nhk, hd, scale,
               total_cache * nhk * hd);

  // GPU path
  float* g_q      = metal_alloc<float>(q_n);
  float* g_knew   = metal_alloc<float>(knew_n);
  float* g_vnew   = metal_alloc<float>(knew_n);
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  float* g_out    = metal_alloc<float>(out_n);

  upload(g_q, h_q.data(), q_n);
  upload(g_knew, h_knew.data(), knew_n);
  upload(g_vnew, h_vnew.data(), knew_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);
  metal::commit_and_wait();

  for (dim_t b = 0; b < batch; ++b) {
    float* kd = g_kcache + (b * total_cache + offset) * row_elem;
    float* vd = g_vcache + (b * total_cache + offset) * row_elem;
    metal::blit_copy(g_knew + b * sq * row_elem, kd, sq * row_elem * sizeof(float));
    metal::blit_copy(g_vnew + b * sq * row_elem, vd, sq * row_elem * sizeof(float));
  }

  metal::sdpa_metal<float>(
      g_q, g_kcache, g_vcache, g_out,
      batch, sq, sk_eff, nh, nhk, hd,
      scale, false,
      total_cache * nhk * hd, 1);

  metal::commit_and_wait();

  float err = max_abs_err(cpu_out.data(), g_out, out_n);
  check("GQA decode: batch=2 nh=8 nhk=2 hd=16", err, 1e-5f);

  metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
  metal_free(g_kcache); metal_free(g_vcache); metal_free(g_out);
}

// ---------------------------------------------------------------------------
// Test 5: LLaMA-like dimensions (hd=64, nh=8, nhk=4)
// ---------------------------------------------------------------------------
static void test_large_decode() {
  std::printf("\n--- Test 5: LLaMA-like decode (hd=64, nh=8, nhk=4) ---\n");

  const dim_t batch = 1, sq = 1, nh = 8, nhk = 4, hd = 64;
  const dim_t half_dim = hd / 2;
  const dim_t prefill_len = 16;
  const dim_t total_cache = prefill_len + 32;
  const dim_t offset = prefill_len;
  const dim_t sk_eff = offset + sq;
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  const dim_t q_n     = batch * sq * nh * hd;
  const dim_t knew_n  = batch * sq * nhk * hd;
  const dim_t cache_n = batch * total_cache * nhk * hd;
  const dim_t out_n   = q_n;
  const dim_t cos_n   = total_cache * half_dim;

  std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);
  std::vector<float> h_cos(cos_n), h_sin(cos_n);

  fill_rand(h_q.data(), q_n, 6000);
  fill_rand(h_knew.data(), knew_n, 6100);
  fill_rand(h_vnew.data(), knew_n, 6200);
  fill_rand(h_kcache.data(), cache_n, 6300);
  fill_rand(h_vcache.data(), cache_n, 6400);
  fill_rand(h_cos.data(), cos_n, 6500);
  fill_rand(h_sin.data(), cos_n, 6600);

  const float* cos_row = h_cos.data() + offset * half_dim;
  const float* sin_row = h_sin.data() + offset * half_dim;

  // CPU ref
  std::vector<float> cpu_q(h_q), cpu_knew(h_knew);
  rope_cpu_ref(cpu_q.data(), cos_row, sin_row, batch * nh, hd, half_dim);
  rope_cpu_ref(cpu_knew.data(), cos_row, sin_row, batch * nhk, hd, half_dim);

  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = cpu_kcache.data() + (b * total_cache + offset) * row_elem;
    float* vd = cpu_vcache.data() + (b * total_cache + offset) * row_elem;
    std::memcpy(kd, cpu_knew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
    std::memcpy(vd, h_vnew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
  }
  std::vector<float> cpu_out(out_n, 0.f);
  sdpa_cpu_ref(cpu_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
               batch, sq, sk_eff, nh, nhk, hd, scale,
               total_cache * nhk * hd);

  // GPU path
  float* g_q      = metal_alloc<float>(q_n);
  float* g_knew   = metal_alloc<float>(knew_n);
  float* g_vnew   = metal_alloc<float>(knew_n);
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  float* g_out    = metal_alloc<float>(out_n);
  float* g_cos    = metal_alloc<float>(cos_n);
  float* g_sin    = metal_alloc<float>(cos_n);

  upload(g_q, h_q.data(), q_n);
  upload(g_knew, h_knew.data(), knew_n);
  upload(g_vnew, h_vnew.data(), knew_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);
  upload(g_cos, h_cos.data(), cos_n);
  upload(g_sin, h_sin.data(), cos_n);
  metal::commit_and_wait();

  const float* g_cos_row = g_cos + offset * half_dim;
  const float* g_sin_row = g_sin + offset * half_dim;

  metal::decode_rope_metal<float>(
      g_q, g_cos_row, g_sin_row, batch * nh, hd, half_dim, false);
  metal::decode_rope_metal<float>(
      g_knew, g_cos_row, g_sin_row, batch * nhk, hd, half_dim, false);

  for (dim_t b = 0; b < batch; ++b) {
    float* kd = g_kcache + (b * total_cache + offset) * row_elem;
    float* vd = g_vcache + (b * total_cache + offset) * row_elem;
    metal::blit_copy(g_knew + b * sq * row_elem, kd, sq * row_elem * sizeof(float));
    metal::blit_copy(g_vnew + b * sq * row_elem, vd, sq * row_elem * sizeof(float));
  }

  metal::sdpa_metal<float>(
      g_q, g_kcache, g_vcache, g_out,
      batch, sq, sk_eff, nh, nhk, hd,
      scale, false,
      total_cache * nhk * hd, 1);

  metal::commit_and_wait();

  float err = max_abs_err(cpu_out.data(), g_out, out_n);
  check("LLaMA-like: RoPE + blit + SDPA (hd=64 nh=8 nhk=4)", err, 1e-5f);

  metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
  metal_free(g_kcache); metal_free(g_vcache); metal_free(g_out);
  metal_free(g_cos); metal_free(g_sin);
}

// ---------------------------------------------------------------------------
// Test 6: Simulate full FlashAttention::compute decode path
//
// This mirrors flash_attention_metal.mm line 162-246 exactly,
// using the same sequence of operations. The key question: does the
// GPU pipeline (encode-only RoPE + blit + SDPA) produce the same
// result as the CPU reference when ALL operations are in one command buffer?
// ---------------------------------------------------------------------------
static void test_full_flash_compute_path() {
  std::printf("\n--- Test 6: Full FlashAttention::compute decode simulation ---\n");

  const dim_t batch = 2, sq = 1, nh = 4, nhk = 2, hd = 32;
  const dim_t half_dim = hd / 2;
  const dim_t prefill_len = 8;
  const dim_t total_cache = prefill_len + 16;
  const dim_t offset = prefill_len;
  const dim_t sk_eff = offset + sq;
  const dim_t row_elem = nhk * hd;
  const float scale = 1.0f / std::sqrt(float(hd));

  const dim_t q_n     = batch * sq * nh * hd;
  const dim_t knew_n  = batch * sq * nhk * hd;
  const dim_t cache_n = batch * total_cache * nhk * hd;
  const dim_t out_n   = q_n;
  const dim_t cos_n   = total_cache * half_dim;

  std::vector<float> h_q(q_n), h_knew(knew_n), h_vnew(knew_n);
  std::vector<float> h_kcache(cache_n), h_vcache(cache_n);
  std::vector<float> h_cos(cos_n), h_sin(cos_n);

  fill_rand(h_q.data(), q_n, 7000);
  fill_rand(h_knew.data(), knew_n, 7100);
  fill_rand(h_vnew.data(), knew_n, 7200);
  fill_rand(h_kcache.data(), cache_n, 7300);
  fill_rand(h_vcache.data(), cache_n, 7400);
  fill_rand(h_cos.data(), cos_n, 7500);
  fill_rand(h_sin.data(), cos_n, 7600);

  const float* cos_row = h_cos.data() + offset * half_dim;
  const float* sin_row = h_sin.data() + offset * half_dim;

  // CPU ref
  std::vector<float> cpu_q(h_q), cpu_knew(h_knew);
  rope_cpu_ref(cpu_q.data(), cos_row, sin_row, batch * nh, hd, half_dim);
  rope_cpu_ref(cpu_knew.data(), cos_row, sin_row, batch * nhk, hd, half_dim);

  std::vector<float> cpu_kcache(h_kcache), cpu_vcache(h_vcache);
  for (dim_t b = 0; b < batch; ++b) {
    std::memcpy(cpu_kcache.data() + (b * total_cache + offset) * row_elem,
                cpu_knew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
    std::memcpy(cpu_vcache.data() + (b * total_cache + offset) * row_elem,
                h_vnew.data() + b * sq * row_elem, sq * row_elem * sizeof(float));
  }
  std::vector<float> cpu_out(out_n, 0.f);
  sdpa_cpu_ref(cpu_q.data(), cpu_kcache.data(), cpu_vcache.data(), cpu_out.data(),
               batch, sq, sk_eff, nh, nhk, hd, scale,
               total_cache * nhk * hd);

  // --- GPU: mirror flash_attention_metal.mm exactly ---
  float* g_q      = metal_alloc<float>(q_n);
  float* g_knew   = metal_alloc<float>(knew_n);
  float* g_vnew   = metal_alloc<float>(knew_n);
  float* g_kcache = metal_alloc<float>(cache_n);
  float* g_vcache = metal_alloc<float>(cache_n);
  float* g_out    = metal_alloc<float>(out_n);
  float* g_cos    = metal_alloc<float>(cos_n);
  float* g_sin    = metal_alloc<float>(cos_n);

  upload(g_q, h_q.data(), q_n);
  upload(g_knew, h_knew.data(), knew_n);
  upload(g_vnew, h_vnew.data(), knew_n);
  upload(g_kcache, h_kcache.data(), cache_n);
  upload(g_vcache, h_vcache.data(), cache_n);
  upload(g_cos, h_cos.data(), cos_n);
  upload(g_sin, h_sin.data(), cos_n);

  // Single sync to ensure uploads are visible, then NO MORE syncs
  // (mirrors the real pipeline where prior GEMM writes are already encoded)
  metal::commit_and_wait();

  // --- Exact mirror of flash_attention_metal.mm decode path ---
  // Step 1: GPU RoPE on Q
  const float* g_cos_row = g_cos + offset * half_dim;
  const float* g_sin_row = g_sin + offset * half_dim;
  metal::decode_rope_metal<float>(
      g_q, g_cos_row, g_sin_row,
      batch * nh, hd, half_dim, false);

  // Step 2: GPU RoPE on K_new
  metal::decode_rope_metal<float>(
      g_knew, g_cos_row, g_sin_row,
      batch * nhk, hd, half_dim, false);

  // Step 3-4: blit copy K/V to cache
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = g_kcache + (b * total_cache + offset) * row_elem;
    float* vd = g_vcache + (b * total_cache + offset) * row_elem;
    metal::blit_copy(g_knew + b * sq * row_elem, kd, sq * row_elem * sizeof(float));
    metal::blit_copy(g_vnew + b * sq * row_elem, vd, sq * row_elem * sizeof(float));
  }

  // Step 5: SDPA
  const bool eff_causal = false;  // sq == 1 → no causal needed
  metal::sdpa_metal<float>(
      g_q, g_kcache, g_vcache, g_out,
      batch, sq, sk_eff, nh, nhk, hd,
      scale, eff_causal,
      total_cache * nhk * hd, /*beam_size=*/1);

  // Single sync at end (mirrors sampler sync)
  metal::commit_and_wait();

  float err = max_abs_err(cpu_out.data(), g_out, out_n);
  check("full FlashAttention::compute decode simulation", err, 1e-5f);

  // Diagnostic: if it fails, check intermediate results
  if (err > 1e-5f) {
    std::printf("    DIAGNOSTIC: checking intermediate values...\n");

    // Check Q rotation
    std::vector<float> qrb(q_n);
    std::memcpy(qrb.data(), g_q, q_n * sizeof(float));
    float qerr = max_abs_err(cpu_q.data(), qrb.data(), q_n);
    std::printf("    Q rotation error: %.3e\n", qerr);

    // Check K rotation
    std::vector<float> krb(knew_n);
    std::memcpy(krb.data(), g_knew, knew_n * sizeof(float));
    float kerr = max_abs_err(cpu_knew.data(), krb.data(), knew_n);
    std::printf("    K rotation error: %.3e\n", kerr);

    // Check cache
    std::vector<float> kcrb(cache_n);
    std::memcpy(kcrb.data(), g_kcache, cache_n * sizeof(float));
    float cerr = max_abs_err(cpu_kcache.data(), kcrb.data(), cache_n);
    std::printf("    K cache error: %.3e\n", cerr);

    // Print first few output values
    std::vector<float> orb(out_n);
    std::memcpy(orb.data(), g_out, out_n * sizeof(float));
    std::printf("    CPU out[0..3]: %.6f %.6f %.6f %.6f\n",
                cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]);
    std::printf("    GPU out[0..3]: %.6f %.6f %.6f %.6f\n",
                orb[0], orb[1], orb[2], orb[3]);
  }

  metal_free(g_q); metal_free(g_knew); metal_free(g_vnew);
  metal_free(g_kcache); metal_free(g_vcache); metal_free(g_out);
  metal_free(g_cos); metal_free(g_sin);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  @autoreleasepool {
    std::printf("=== M12.18 FlashMHA Decode Path Correctness Tests ===\n");

    test_no_rope_decode();
    test_rope_decode();
    test_multi_step_decode();
    test_gqa_decode();
    test_large_decode();
    test_full_flash_compute_path();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
  }
}
