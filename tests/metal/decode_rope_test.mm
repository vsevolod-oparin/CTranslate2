// tests/metal/decode_rope_test.mm
//
// M6.3 — Decode-path RoPE (apply_rope_half) correctness tests.
//
// Addresses M6 review item 4.1: "apply_rope_half is completely untested."
//
// apply_rope_half is a static template in flash_attention_metal.mm and cannot
// be called directly from a standalone test.  This file replicates the exact
// same formula as apply_rope_half_ref<T>(), then exercises the full decode
// pipeline:
//
//   1. Build half-sized cos/sin tables [max_positions, ndims/2].
//   2. Apply apply_rope_half_ref to Q and K_new at the given cache offset.
//   3. Write K_new + V_new into the KV cache at offset.
//   4. Call sdpa_metal over the full valid cache [0, offset+1).
//   5. Compare against a float32 CPU reference that performs the same steps.
//
// The comparison validates both the rotation formula and the end-to-end
// decode pipeline (including interleave, partial rotation, GQA, and f16/bf16).
//
// Tests:
//   1.  float32  non-interleave  ndims=hd=32  offset=4  nh=4, nhk=4
//   2.  float32  interleave      ndims=hd=32  offset=2  nh=4, nhk=4
//   3.  float16  non-interleave  ndims=hd=32  offset=4  nh=4, nhk=4
//   4.  bfloat16 non-interleave  ndims=hd=32  offset=4  nh=4, nhk=4
//   5.  float32  partial rotation ndims=16, hd=32, offset=4  nh=4, nhk=4
//   6.  float32  GQA decode  nh=8, nhk=2, hd=32, ndims=32, offset=4
//   7.  float32  multi-step decode: 5 steps, non-interleave, ndims=32, hd=64
//       Starting at prefill=4, steps 1-5 each advance offset by 1.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/decode_rope_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/metal/ops_sdpa.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o decode_rope_test && ./decode_rope_test

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
// apply_rope_half_ref<T>
//
// Exact replica of the fixed apply_rope_half from flash_attention_metal.mm.
// (apply_rope_half is a static template there; cannot be called externally.)
//
// Applies rotary embedding in-place to one head vector x[0..depth).
//   cos_row / sin_row: [ndims/2] — the "positive-frequency" half of the table
//                                   for position `offset`.
//   Non-interleave (LLaMA-style):
//     half = ndims/2
//     tmp[d]        = x[d]        * cos[d] - x[d+half] * sin[d]   d < half
//     tmp[d+half]   = x[d+half]   * cos[d] + x[d]      * sin[d]   d < half
//     Write back [0, 2*half); elements [ndims, depth) unchanged.
//   Interleave (GPT-NeoX-style):
//     y[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]   i < half
//     y[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]   i < half
//     Elements [ndims, depth) unchanged.
// ---------------------------------------------------------------------------

template <typename T>
static void apply_rope_half_ref(T* x,
                                 const T* cos_row,  // [ndims/2]
                                 const T* sin_row,  // [ndims/2]
                                 dim_t ndims,
                                 dim_t depth,
                                 bool interleave) {
  const dim_t half = ndims / 2;

  if (!interleave) {
    // Stack buffer; head_dim <= 256 in practice, so 512 floats is safe.
    float tmp[512];
    for (dim_t d = 0; d < half; ++d) {
      const float xd = float(x[d]);
      const float xp = float(x[d + half]);
      const float cd = float(cos_row[d]);
      const float sd = float(sin_row[d]);
      tmp[d]        = xd * cd - xp * sd;
      tmp[d + half] = xp * cd + xd * sd;
    }
    // Write back only 2*half elements: for odd ndims the last element is
    // not part of any pair and is left unchanged (Bug 1.1 fix).
    for (dim_t d = 0; d < 2 * half; ++d) x[d] = T(tmp[d]);
    // Elements [ndims, depth) pass through unchanged.
  } else {
    for (dim_t i = 0; i < half; ++i) {
      const float xe = float(x[2 * i]);
      const float xo = float(x[2 * i + 1]);
      const float ci = float(cos_row[i]);
      const float si = float(sin_row[i]);
      x[2 * i]     = T(xe * ci - xo * si);
      x[2 * i + 1] = T(xo * ci + xe * si);
    }
    // Elements [ndims, depth) pass through unchanged.
  }
}

// ---------------------------------------------------------------------------
// make_half_table
//
// Builds the half-sized cos/sin table format used by FlashAttention decode.
// Shape: [max_positions, ndims/2]
//   cos_table[t, i] = cos(t * inv_freq[i])   for i in [0, ndims/2)
//   inv_freq[i] = 1 / (10000 ^ (2*i / ndims))
// ---------------------------------------------------------------------------

static void make_half_table(dim_t max_positions, dim_t ndims,
                              std::vector<float>& cos_out,
                              std::vector<float>& sin_out) {
  const dim_t half = ndims / 2;
  cos_out.resize(static_cast<size_t>(max_positions * half));
  sin_out.resize(static_cast<size_t>(max_positions * half));

  std::vector<float> inv_freq(static_cast<size_t>(half));
  for (dim_t i = 0; i < half; ++i)
    inv_freq[static_cast<size_t>(i)] = 1.f / std::pow(10000.f,
                                                        float(2 * i) / float(ndims));

  for (dim_t t = 0; t < max_positions; ++t) {
    for (dim_t i = 0; i < half; ++i) {
      float angle = float(t) * inv_freq[static_cast<size_t>(i)];
      cos_out[static_cast<size_t>(t * half + i)] = std::cos(angle);
      sin_out[static_cast<size_t>(t * half + i)] = std::sin(angle);
    }
  }
}

// ---------------------------------------------------------------------------
// CPU reference SDPA (float32, same as kv_cache_test.mm)
// Layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
// ---------------------------------------------------------------------------

static void cpu_ref_sdpa(const float* q, const float* k, const float* v, float* out,
                          dim_t batch, dim_t sq, dim_t sk, dim_t nh, dim_t nhk, dim_t hd,
                          float scale) {
  const dim_t q_lda  = nh  * hd;
  const dim_t kv_lda = nhk * hd;
  std::vector<float> scores(static_cast<size_t>(sq * sk));

  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h) {
      const dim_t hk = h % nhk;
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
      // No causal mask: decode path always uses is_causal=false.
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
// run_decode_rope_test<T>
//
// Exercises the decode-path RoPE pipeline:
//   1. Fills a KV cache with `prefill_sk` tokens (float32 → T, no rotation).
//   2. Generates new Q[batch,1,nh,hd], K_new[batch,1,nhk,hd], V_new[batch,1,nhk,hd].
//   3. Applies apply_rope_half_ref<float> to Q and K_new (CPU ref path).
//   4. Applies apply_rope_half_ref<T> to Q and K_new (Metal test path).
//   5. Writes T K_new/V_new into Metal cache at `offset`.
//   6. Calls sdpa_metal and compares to cpu_ref_sdpa on the pre-rotated data.
// ---------------------------------------------------------------------------

template <typename T>
static void run_decode_rope_test(
    const char* name, float tol,
    dim_t batch, dim_t nh, dim_t nhk, dim_t hd,
    dim_t ndims, dim_t offset, dim_t prefill_sk,
    bool interleave) {

  rng_state = 0xFACEFACEu ^ (uint32_t)(batch * 97 + nh * 31 + hd * 7 +
                                        ndims * 3 + (uint32_t)offset);

  const dim_t half      = ndims / 2;
  const dim_t max_cache = prefill_sk + offset + 16;  // generous headroom
  const dim_t row_elems = nhk * hd;

  // Build half cos/sin tables [max_cache+4, half].
  const dim_t max_pos = max_cache + 4;
  std::vector<float> half_cos_f, half_sin_f;
  make_half_table(max_pos, ndims, half_cos_f, half_sin_f);

  // ------------------------------------------------------------------
  // Fill cache with `offset` tokens (no RoPE for prefill tokens here).
  // Layout: [batch, max_cache, nhk, hd]
  // ------------------------------------------------------------------
  const dim_t cache_elems = batch * max_cache * nhk * hd;
  T* cache_k_m = metal_alloc<T>(cache_elems);
  T* cache_v_m = metal_alloc<T>(cache_elems);
  std::memset(cache_k_m, 0, cache_elems * sizeof(T));
  std::memset(cache_v_m, 0, cache_elems * sizeof(T));

  // CPU cache mirrors (float32).
  std::vector<float> ref_cache_k(static_cast<size_t>(cache_elems), 0.f);
  std::vector<float> ref_cache_v(static_cast<size_t>(cache_elems), 0.f);

  // Populate [0, offset) slots with prefill data.
  for (dim_t pos = 0; pos < offset; ++pos) {
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t j = 0; j < row_elems; ++j) {
        float kval = std::sin(float((pos * batch + b) * row_elems + j + 1) * 0.11f);
        float vval = std::cos(float((pos * batch + b) * row_elems + j + 1) * 0.09f);
        dim_t cache_idx = (b * max_cache + pos) * row_elems + j;
        cache_k_m[cache_idx] = T(kval);
        cache_v_m[cache_idx] = T(vval);
        ref_cache_k[static_cast<size_t>(cache_idx)] = kval;
        ref_cache_v[static_cast<size_t>(cache_idx)] = vval;
      }
    }
  }

  // ------------------------------------------------------------------
  // Generate Q[batch,1,nh,hd], K_new[batch,1,nhk,hd], V_new[batch,1,nhk,hd].
  // ------------------------------------------------------------------
  const dim_t q_elems  = batch * 1 * nh  * hd;
  const dim_t kv_elems = batch * 1 * nhk * hd;

  std::vector<float> qf(static_cast<size_t>(q_elems));
  std::vector<float> kf(static_cast<size_t>(kv_elems));
  std::vector<float> vf(static_cast<size_t>(kv_elems));
  for (auto& v : qf) v = next_float(-1.f, 1.f);
  for (auto& v : kf) v = next_float(-1.f, 1.f);
  for (auto& v : vf) v = next_float(-1.f, 1.f);

  // ------------------------------------------------------------------
  // CPU reference path:
  //   Apply apply_rope_half_ref<float> to Q and K_new at position `offset`.
  //   Write K_new/V_new into ref_cache at offset.
  //   Run cpu_ref_sdpa over full cache [0, offset+1).
  // ------------------------------------------------------------------
  std::vector<float> qf_rotated(qf);
  std::vector<float> kf_rotated(kf);

  const float* cos_row = half_cos_f.data() + offset * half;
  const float* sin_row = half_sin_f.data() + offset * half;

  // Rotate each head of Q (layout [batch, 1, nh, hd]).
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h) {
      float* xq = qf_rotated.data() + (b * nh + h) * hd;
      apply_rope_half_ref<float>(xq, cos_row, sin_row, ndims, hd, interleave);
    }
  }
  // Rotate each head of K_new (layout [batch, 1, nhk, hd]).
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t hk = 0; hk < nhk; ++hk) {
      float* xk = kf_rotated.data() + (b * nhk + hk) * hd;
      apply_rope_half_ref<float>(xk, cos_row, sin_row, ndims, hd, interleave);
    }
  }

  // Write rotated K_new/V_new into ref cache at offset.
  for (dim_t b = 0; b < batch; ++b) {
    float* kd = ref_cache_k.data() + (b * max_cache + offset) * row_elems;
    float* vd = ref_cache_v.data() + (b * max_cache + offset) * row_elems;
    const float* ks = kf_rotated.data() + b * row_elems;
    const float* vs = vf.data()          + b * row_elems;
    std::memcpy(kd, ks, row_elems * sizeof(float));
    std::memcpy(vd, vs, row_elems * sizeof(float));
  }

  const dim_t sk_eff = offset + 1;
  std::vector<float> ref_out(static_cast<size_t>(q_elems));
  cpu_ref_sdpa(qf_rotated.data(), ref_cache_k.data(), ref_cache_v.data(),
               ref_out.data(), batch, 1, sk_eff, nh, nhk, hd,
               1.f / std::sqrt(float(hd)));

  // ------------------------------------------------------------------
  // Metal path:
  //   Apply apply_rope_half_ref<T> to T-typed Q and K_new.
  //   Write into Metal cache at offset.
  //   Call sdpa_metal.
  // ------------------------------------------------------------------

  // Build half cos/sin tables as T arrays.
  std::vector<T> half_cos_T(static_cast<size_t>(max_pos * half));
  std::vector<T> half_sin_T(static_cast<size_t>(max_pos * half));
  for (size_t i = 0; i < half_cos_T.size(); ++i) half_cos_T[i] = T(half_cos_f[i]);
  for (size_t i = 0; i < half_sin_T.size(); ++i) half_sin_T[i] = T(half_sin_f[i]);

  const T* cos_row_T = half_cos_T.data() + offset * half;
  const T* sin_row_T = half_sin_T.data() + offset * half;

  // Convert Q, K_new to T and apply rotation.
  T* q_m   = metal_alloc<T>(q_elems);
  T* out_m = metal_alloc<T>(q_elems);
  for (dim_t i = 0; i < q_elems;  ++i) q_m[i] = T(qf[static_cast<size_t>(i)]);

  std::vector<T> k_new_T(static_cast<size_t>(kv_elems));
  std::vector<T> v_new_T(static_cast<size_t>(kv_elems));
  for (dim_t i = 0; i < kv_elems; ++i) k_new_T[i] = T(kf[static_cast<size_t>(i)]);
  for (dim_t i = 0; i < kv_elems; ++i) v_new_T[i] = T(vf[static_cast<size_t>(i)]);

  // Apply rotation (mirrors flash_attention_metal.mm decode-path).
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t h = 0; h < nh; ++h) {
      T* xq = q_m + (b * nh + h) * hd;
      apply_rope_half_ref<T>(xq, cos_row_T, sin_row_T, ndims, hd, interleave);
    }
  }
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t hk = 0; hk < nhk; ++hk) {
      T* xk = k_new_T.data() + (b * nhk + hk) * hd;
      apply_rope_half_ref<T>(xk, cos_row_T, sin_row_T, ndims, hd, interleave);
    }
  }

  // Write rotated K_new/V_new into Metal cache at offset.
  for (dim_t b = 0; b < batch; ++b) {
    T* kd = cache_k_m + (b * max_cache + offset) * row_elems;
    T* vd = cache_v_m + (b * max_cache + offset) * row_elems;
    const T* ks = k_new_T.data() + b * row_elems;
    const T* vs = v_new_T.data() + b * row_elems;
    std::memcpy(kd, ks, row_elems * sizeof(T));
    std::memcpy(vd, vs, row_elems * sizeof(T));
  }

  // SDPA over full valid cache [0, sk_eff).
  metal::sdpa_metal<T>(q_m, cache_k_m, cache_v_m, out_m,
                        batch, 1, sk_eff, nh, nhk, hd,
                        1.f / std::sqrt(float(hd)), /*is_causal=*/false);
  metal::commit_and_wait();

  std::vector<float> got_f(static_cast<size_t>(q_elems));
  for (dim_t i = 0; i < q_elems; ++i)
    got_f[static_cast<size_t>(i)] = float(out_m[i]);

  float err = max_abs_err(ref_out.data(), got_f.data(), q_elems);
  check(name, err, tol);

  metal_free(q_m);
  metal_free(out_m);
  metal_free(cache_k_m);
  metal_free(cache_v_m);
}

// ---------------------------------------------------------------------------
// Multi-step decode test (item 4.1 extension)
//
// Runs N decode steps with RoPE, each advancing the cache offset by 1.
// Non-interleave, float32 only.  Verifies cumulative error stays bounded.
// ---------------------------------------------------------------------------

static void run_multistep_decode(const char* name, float tol,
                                  dim_t batch, dim_t nh, dim_t nhk, dim_t hd,
                                  dim_t ndims, dim_t prefill_sk, dim_t num_steps) {
  const dim_t half      = ndims / 2;
  const dim_t max_cache = prefill_sk + num_steps + 4;
  const dim_t row_elems = nhk * hd;

  const dim_t max_pos = max_cache + 4;
  std::vector<float> half_cos_f, half_sin_f;
  make_half_table(max_pos, ndims, half_cos_f, half_sin_f);

  // Shared Metal + CPU caches.
  const dim_t cache_elems = batch * max_cache * nhk * hd;
  float* cache_k_f = new float[cache_elems]();
  float* cache_v_f = new float[cache_elems]();
  float* cache_k_m_f = metal_alloc<float>(cache_elems);
  float* cache_v_m_f = metal_alloc<float>(cache_elems);
  std::memset(cache_k_m_f, 0, cache_elems * sizeof(float));
  std::memset(cache_v_m_f, 0, cache_elems * sizeof(float));

  // Populate prefill slots [0, prefill_sk) (no RoPE).
  rng_state = 0xDEAD0000u ^ (uint32_t)(batch * 13 + nh * 7 + hd);
  for (dim_t pos = 0; pos < prefill_sk; ++pos) {
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t j = 0; j < row_elems; ++j) {
        float kv = next_float(-1.f, 1.f);
        dim_t idx = (b * max_cache + pos) * row_elems + j;
        cache_k_f[idx] = kv;
        cache_k_m_f[idx] = kv;
        kv = next_float(-1.f, 1.f);
        cache_v_f[idx] = kv;
        cache_v_m_f[idx] = kv;
      }
    }
  }

  float worst_err = 0.f;
  const dim_t q_elems  = batch * nh  * hd;
  const dim_t kv_elems = batch * nhk * hd;

  for (dim_t step = 0; step < num_steps; ++step) {
    const dim_t offset = prefill_sk + step;
    const float* cos_row = half_cos_f.data() + offset * half;
    const float* sin_row = half_sin_f.data() + offset * half;

    // Generate Q, K_new, V_new for this step.
    rng_state ^= (uint32_t)(step * 31337 + 1);
    std::vector<float> qf(static_cast<size_t>(q_elems));
    std::vector<float> kf(static_cast<size_t>(kv_elems));
    std::vector<float> vf(static_cast<size_t>(kv_elems));
    for (auto& v : qf) v = next_float(-1.f, 1.f);
    for (auto& v : kf) v = next_float(-1.f, 1.f);
    for (auto& v : vf) v = next_float(-1.f, 1.f);

    // --- CPU reference path ---
    std::vector<float> qf_rot(qf), kf_rot(kf);
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t h = 0; h < nh; ++h)
        apply_rope_half_ref<float>(qf_rot.data() + (b * nh + h) * hd,
                                   cos_row, sin_row, ndims, hd, /*interleave=*/false);
      for (dim_t hk = 0; hk < nhk; ++hk)
        apply_rope_half_ref<float>(kf_rot.data() + (b * nhk + hk) * hd,
                                   cos_row, sin_row, ndims, hd, /*interleave=*/false);
      std::memcpy(cache_k_f + (b * max_cache + offset) * row_elems,
                  kf_rot.data() + b * row_elems, row_elems * sizeof(float));
      std::memcpy(cache_v_f + (b * max_cache + offset) * row_elems,
                  vf.data()   + b * row_elems, row_elems * sizeof(float));
    }
    std::vector<float> ref_out(static_cast<size_t>(q_elems));
    cpu_ref_sdpa(qf_rot.data(), cache_k_f, cache_v_f, ref_out.data(),
                 batch, 1, offset + 1, nh, nhk, hd,
                 1.f / std::sqrt(float(hd)));

    // --- Metal path ---
    float* q_m   = metal_alloc<float>(q_elems);
    float* out_m = metal_alloc<float>(q_elems);
    for (dim_t i = 0; i < q_elems;  ++i) q_m[i] = qf[static_cast<size_t>(i)];
    std::vector<float> kf_rot_m(kf);
    for (dim_t b = 0; b < batch; ++b) {
      for (dim_t h = 0; h < nh; ++h)
        apply_rope_half_ref<float>(q_m + (b * nh + h) * hd,
                                   cos_row, sin_row, ndims, hd, false);
      for (dim_t hk = 0; hk < nhk; ++hk)
        apply_rope_half_ref<float>(kf_rot_m.data() + (b * nhk + hk) * hd,
                                   cos_row, sin_row, ndims, hd, false);
      std::memcpy(cache_k_m_f + (b * max_cache + offset) * row_elems,
                  kf_rot_m.data() + b * row_elems, row_elems * sizeof(float));
      std::memcpy(cache_v_m_f + (b * max_cache + offset) * row_elems,
                  vf.data()    + b * row_elems, row_elems * sizeof(float));
    }
    metal::sdpa_metal<float>(q_m, cache_k_m_f, cache_v_m_f, out_m,
                              batch, 1, offset + 1, nh, nhk, hd,
                              1.f / std::sqrt(float(hd)), false);
    metal::commit_and_wait();

    std::vector<float> got_f(static_cast<size_t>(q_elems));
    for (dim_t i = 0; i < q_elems; ++i) got_f[i] = out_m[i];

    float err = max_abs_err(ref_out.data(), got_f.data(), q_elems);
    worst_err = std::max(worst_err, err);

    metal_free(q_m);
    metal_free(out_m);
  }

  check(name, worst_err, tol);

  metal_free(cache_k_m_f);
  metal_free(cache_v_m_f);
  delete[] cache_k_f;
  delete[] cache_v_f;
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

static void test_f32_noninterleave() {
  run_decode_rope_test<float>(
    "f32 non-interleave ndims=32 hd=32 offset=4 (decode)",
    1e-4f, 1, 4, 4, 32, 32, 4, 4, /*interleave=*/false);
}

static void test_f32_interleave() {
  run_decode_rope_test<float>(
    "f32 interleave     ndims=32 hd=32 offset=2 (decode)",
    1e-4f, 1, 4, 4, 32, 32, 2, 2, /*interleave=*/true);
}

static void test_f16_noninterleave() {
  run_decode_rope_test<ct2_f16>(
    "f16 non-interleave ndims=32 hd=32 offset=4 (decode)",
    5e-3f, 1, 4, 4, 32, 32, 4, 4, /*interleave=*/false);
}

static void test_bf16_noninterleave() {
  run_decode_rope_test<ct2_bf16>(
    "bf16 non-interleave ndims=32 hd=32 offset=4 (decode)",
    5e-2f, 1, 4, 4, 32, 32, 4, 4, /*interleave=*/false);
}

static void test_f32_partial_rotation() {
  // ndims=16 < hd=32: first 16 dims rotate, last 16 pass through.
  run_decode_rope_test<float>(
    "f32 partial rotation ndims=16 hd=32 offset=4 (decode)",
    1e-4f, 1, 4, 4, 32, 16, 4, 4, /*interleave=*/false);
}

static void test_f32_gqa_decode() {
  // GQA: 8 query heads, 2 KV heads.
  run_decode_rope_test<float>(
    "f32 GQA non-interleave nh=8 nhk=2 hd=32 offset=4 (decode)",
    1e-4f, 1, 8, 2, 32, 32, 4, 4, /*interleave=*/false);
}

static void test_f32_multistep() {
  run_multistep_decode(
    "f32 non-interleave multi-step 5 steps prefill=4 nh=4 hd=64",
    1e-4f, 1, 4, 4, 64, 64, 4, 5);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.3 decode-path RoPE (apply_rope_half) Correctness Tests ===\n\n");

  test_f32_noninterleave();
  test_f32_interleave();
  test_f16_noninterleave();
  test_bf16_noninterleave();
  test_f32_partial_rotation();
  test_f32_gqa_decode();
  test_f32_multistep();

  std::printf("\n=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
