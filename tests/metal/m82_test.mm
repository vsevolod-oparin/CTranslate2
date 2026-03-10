// tests/metal/m82_test.mm
//
// M8.2 — Transformer Decoder Layer end-to-end integration test.
//
// Tests two elements not covered by M8.1:
//   1. Cross-attention: Q from decoder [sq], KV from encoder [sk], sq != sk.
//   2. KV-cache decode: offset > 0 path (commit_and_wait + CPU memcpy + sdpa).
//
// Architecture under test (Pre-LayerNorm, causal self-attn + cross-attn + ReLU FFN):
//
//   x_dec --+--[LN1]--{Wqs,Wks,Wvs}--[causal self-SDPA]--[Wos]--+--> h1
//           +----------------------------------------------------^
//
//   h1  ----+--[LN2]--{Wqc}--\
//           |                 +--> [cross-SDPA]--[Woc]--+--> h2
//           |  enc_ctx--{Wkc,Wvc}--/                    ^
//           +-------------------------------------------/
//
//   h2  ----+--[LN3]--[W1]--[ReLU]--[W2]--+--> output
//           +------------------------------^
//
// Parameters: B=1, DEC_T=4, ENC_T=6, D=16, NH=2, HD=8, FFN=32, MAX_CACHE=8
//
// Tests:
//   1.  Cross-attention SDPA (sq=4, sk=6, non-causal): Metal vs CPU
//   2.  KV-cache decode step (sq=1, sk=5, offset=4): Metal vs CPU
//   3.  Full decoder layer prefill (DEC_T=4 tokens, 5 checks): Metal vs CPU
//   4.  Full decoder layer decode step (sq=1, offset=4, 4 checks): Metal vs CPU
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/m82_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm \
//     src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm \
//     src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm \
//     src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/metal/ops_sdpa.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o m82_test && ./m82_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "metal/ops_metal.h"
#include "metal/utils.h"

using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK(label, expr) \
  do { \
    bool _ok = (bool)(expr); \
    if (_ok) { std::printf("  PASS  %s\n", label); ++g_pass; } \
    else     { std::printf("  FAIL  %s\n", label); ++g_fail; } \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(
      static_cast<std::size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// Upload host float data into a new Metal Shared buffer.
template <typename T>
static T* metal_from(const std::vector<float>& host) {
  T* buf = metal_alloc<T>(static_cast<dim_t>(host.size()));
  for (std::size_t i = 0; i < host.size(); ++i)
    buf[i] = T(host[i]);
  return buf;
}

// Read a Metal buffer into host float vector (calls commit_and_wait first).
template <typename T>
static std::vector<float> metal_to_host(const T* buf, dim_t n) {
  metal::commit_and_wait();
  std::vector<float> v(static_cast<std::size_t>(n));
  for (dim_t i = 0; i < n; ++i)
    v[static_cast<std::size_t>(i)] = static_cast<float>(buf[i]);
  return v;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float d = 0.f;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

static std::mt19937 g_rng(42);
static std::vector<float> rand_vec(std::size_t n, float lo = -1.f, float hi = 1.f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(g_rng);
  return v;
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

// LayerNorm: y[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) * g[j] + b[j]
static std::vector<float> ref_layer_norm(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    dim_t outer, dim_t D, float eps = 1e-5f)
{
  std::vector<float> y(static_cast<std::size_t>(outer * D));
  for (dim_t i = 0; i < outer; ++i) {
    float mean = 0.f, var = 0.f;
    for (dim_t j = 0; j < D; ++j)
      mean += x[static_cast<std::size_t>(i * D + j)];
    mean /= static_cast<float>(D);
    for (dim_t j = 0; j < D; ++j) {
      float d = x[static_cast<std::size_t>(i * D + j)] - mean;
      var += d * d;
    }
    var /= static_cast<float>(D);
    float inv = 1.f / std::sqrt(var + eps);
    for (dim_t j = 0; j < D; ++j) {
      float xn = (x[static_cast<std::size_t>(i * D + j)] - mean) * inv;
      y[static_cast<std::size_t>(i * D + j)] =
          xn * gamma[static_cast<std::size_t>(j)] + beta[static_cast<std::size_t>(j)];
    }
  }
  return y;
}

// Matrix multiply: C = A × B^T  (A: [m×k], B: [n×k])
static std::vector<float> ref_gemm_bt(
    const std::vector<float>& A, dim_t m, dim_t k,
    const std::vector<float>& B, dim_t n)
{
  std::vector<float> C(static_cast<std::size_t>(m * n), 0.f);
  for (dim_t i = 0; i < m; ++i)
    for (dim_t j = 0; j < n; ++j) {
      float s = 0.f;
      for (dim_t l = 0; l < k; ++l)
        s += A[static_cast<std::size_t>(i * k + l)] *
             B[static_cast<std::size_t>(j * k + l)];
      C[static_cast<std::size_t>(i * n + j)] = s;
    }
  return C;
}

static std::vector<float> ref_relu(const std::vector<float>& x) {
  std::vector<float> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::max(0.f, x[i]);
  return y;
}

static std::vector<float> ref_add(const std::vector<float>& a,
                                   const std::vector<float>& b) {
  std::vector<float> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
  return c;
}

// General SDPA.
// q:   [B, sq, NH, HD]
// k/v: [B, sk, NK, HD]   (NK == NH for MHA; NK < NH for GQA)
// is_causal: mask j > i (only meaningful when sq == sk prefill).
// For decode (sq=1), use is_causal=false — all cached tokens are in the past.
static std::vector<float> ref_sdpa_cross(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    dim_t B, dim_t sq, dim_t sk,
    dim_t NH, dim_t NK, dim_t HD,
    float scale, bool is_causal)
{
  std::vector<float> out(static_cast<std::size_t>(B * sq * NH * HD), 0.f);
  for (dim_t b = 0; b < B; ++b) {
    for (dim_t h = 0; h < NH; ++h) {
      dim_t hk = h % NK;  // GQA head mapping (NK == NH here)
      for (dim_t i = 0; i < sq; ++i) {
        std::vector<float> scores(static_cast<std::size_t>(sk));
        for (dim_t j = 0; j < sk; ++j) {
          if (is_causal && j > i) { scores[static_cast<std::size_t>(j)] = -1e9f; continue; }
          float dot = 0.f;
          for (dim_t d = 0; d < HD; ++d)
            dot += q[static_cast<std::size_t>((b * sq + i) * NH * HD + h * HD + d)]
                 * k[static_cast<std::size_t>((b * sk + j) * NK * HD + hk * HD + d)];
          scores[static_cast<std::size_t>(j)] = dot * scale;
        }
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sums = 0.f;
        for (auto& s : scores) { s = std::exp(s - maxs); sums += s; }
        for (auto& s : scores) s /= sums;
        for (dim_t d = 0; d < HD; ++d) {
          float acc = 0.f;
          for (dim_t j = 0; j < sk; ++j)
            acc += scores[static_cast<std::size_t>(j)]
                 * v[static_cast<std::size_t>((b * sk + j) * NK * HD + hk * HD + d)];
          out[static_cast<std::size_t>((b * sq + i) * NH * HD + h * HD + d)] = acc;
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Test 1 — Cross-attention SDPA: sq=4 (decoder), sk=6 (encoder)
// ---------------------------------------------------------------------------
static void test_cross_attn_sdpa() {
  std::printf("\n--- Test 1: Cross-attention SDPA (sq=4, sk=6) Metal vs CPU ---\n");

  const dim_t B = 1, sq = 4, sk = 6, NH = 2, HD = 8;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));

  auto q_h = rand_vec(static_cast<std::size_t>(B * sq * NH * HD));
  auto k_h = rand_vec(static_cast<std::size_t>(B * sk * NH * HD));
  auto v_h = rand_vec(static_cast<std::size_t>(B * sk * NH * HD));

  auto cpu = ref_sdpa_cross(q_h, k_h, v_h, B, sq, sk, NH, NH, HD, scale, false);

  float* q_m   = metal_from<float>(q_h);
  float* k_m   = metal_from<float>(k_h);
  float* v_m   = metal_from<float>(v_h);
  float* out_m = metal_alloc<float>(B * sq * NH * HD);

  metal::sdpa_metal<float>(q_m, k_m, v_m, out_m,
                            B, sq, sk, NH, NH, HD, scale, false);
  auto metal_out = metal_to_host(out_m, B * sq * NH * HD);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("cross-attn SDPA [1,4,2,8] × [1,6,2,8]: max_abs_diff < 1e-4", err < 1e-4f);

  metal_free(q_m); metal_free(k_m); metal_free(v_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 2 — KV-cache decode step: offset=4, sq=1, sk=5
//
// Pattern mirrors flash_attention_metal.mm M6.2:
//   1. Pre-fill cache positions [0..3] with known K/V.
//   2. commit_and_wait() + memcpy new K/V into position [4].
//   3. sdpa_metal over full cache (sq=1, sk=5).
// ---------------------------------------------------------------------------
static void test_kvcache_decode() {
  std::printf("\n--- Test 2: KV-cache decode step (offset=4, sq=1, sk=5) Metal vs CPU ---\n");

  const dim_t B = 1, DEC_T = 4, NK = 2, HD = 8, MAX_CACHE = 8;
  const dim_t NH = 2;
  const dim_t row_elems = NK * HD;  // elements per cache slot per batch item
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));

  // Pre-fill: DEC_T slots of K/V (shape [DEC_T, NK, HD] per batch).
  auto k_prefill_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems));
  auto v_prefill_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems));

  // Allocate cache [B, MAX_CACHE, NK, HD] in Metal Shared memory.
  float* k_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  float* v_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);

  // Write prefill rows directly (no GPU work pending yet).
  std::memcpy(k_cache, k_prefill_h.data(),
              static_cast<std::size_t>(DEC_T * row_elems) * sizeof(float));
  std::memcpy(v_cache, v_prefill_h.data(),
              static_cast<std::size_t>(DEC_T * row_elems) * sizeof(float));

  // New token at offset=DEC_T: Q [B, sq=1, NH, HD], K/V new [B, 1, NK, HD].
  auto q_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NH * HD));
  auto k_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NK * HD));
  auto v_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NK * HD));

  float* q_m   = metal_from<float>(q_new_h);
  float* out_m = metal_alloc<float>(B * 1 * NH * HD);

  // Metal decode path: flush GPU then CPU-write new K/V into cache.
  metal::commit_and_wait();
  for (dim_t b = 0; b < B; ++b) {
    float* kd = k_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    float* vd = v_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    std::memcpy(kd, k_new_h.data() + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
    std::memcpy(vd, v_new_h.data() + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
  }

  const dim_t sk_eff = DEC_T + 1;  // 5 total tokens in cache
  metal::sdpa_metal<float>(q_m, k_cache, v_cache, out_m,
                            B, 1, sk_eff, NH, NK, HD, scale, false);
  auto metal_out = metal_to_host(out_m, B * 1 * NH * HD);

  // CPU reference: build K_all/V_all [B, sk_eff, NK, HD] = prefill ++ new.
  std::vector<float> k_all(static_cast<std::size_t>(B * sk_eff * row_elems));
  std::vector<float> v_all(static_cast<std::size_t>(B * sk_eff * row_elems));
  for (dim_t b = 0; b < B; ++b) {
    std::size_t base = static_cast<std::size_t>(b * sk_eff * row_elems);
    // Prefill slots [0..DEC_T-1]
    std::copy(k_prefill_h.begin(), k_prefill_h.end(), k_all.begin() + base);
    std::copy(v_prefill_h.begin(), v_prefill_h.end(), v_all.begin() + base);
    // New slot [DEC_T]
    std::size_t new_off = base + static_cast<std::size_t>(DEC_T * row_elems);
    std::copy(k_new_h.begin() + b * row_elems,
              k_new_h.begin() + (b + 1) * row_elems,
              k_all.begin() + new_off);
    std::copy(v_new_h.begin() + b * row_elems,
              v_new_h.begin() + (b + 1) * row_elems,
              v_all.begin() + new_off);
  }

  auto cpu = ref_sdpa_cross(q_new_h, k_all, v_all,
                             B, 1, sk_eff, NH, NK, HD, scale, false);
  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("kv-cache decode [offset=4, sq=1, sk=5]: max_abs_diff < 1e-4", err < 1e-4f);

  metal_free(k_cache); metal_free(v_cache);
  metal_free(q_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 3 — Full decoder layer prefill (DEC_T=4 tokens)
//
// Pipeline:
//   x_dec → LN1 → {Qs,Ks,Vs} → causal self-SDPA → Wos → +x  → h1
//   h1    → LN2 → {Qc}   \
//   enc   →       {Kc,Vc} → cross-SDPA (sq=4, sk=6) → Woc → +h1 → h2
//   h2    → LN3 → W1 → ReLU → W2 → +h2 → output
// ---------------------------------------------------------------------------
static void test_full_decoder_prefill() {
  std::printf("\n--- Test 3: Full Decoder Layer prefill (DEC_T=4, ENC_T=6) Metal vs CPU ---\n");

  const dim_t B = 1, DEC_T = 4, ENC_T = 6, D = 16, NH = 2, HD = 8, FFN = 32;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps   = 1e-5f;

  // Weights [output_dim × input_dim]:
  auto gamma1_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta1_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Ks_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Vs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_os_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto gamma2_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta2_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Kc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Vc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_oc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto gamma3_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta3_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W1_h     = rand_vec(static_cast<std::size_t>(FFN * D), -0.5f, 0.5f);
  auto W2_h     = rand_vec(static_cast<std::size_t>(D * FFN), -0.5f, 0.5f);

  // Inputs:
  auto x_dec_h  = rand_vec(static_cast<std::size_t>(B * DEC_T * D));
  auto enc_ctx_h = rand_vec(static_cast<std::size_t>(B * ENC_T * D));

  // ==========================================================================
  // CPU forward pass
  // ==========================================================================

  // Self-attention sublayer
  auto norm1_cpu  = ref_layer_norm(x_dec_h, gamma1_h, beta1_h, DEC_T, D, eps);
  auto Q_s_cpu    = ref_gemm_bt(norm1_cpu, DEC_T, D, W_Qs_h, D);
  auto K_s_cpu    = ref_gemm_bt(norm1_cpu, DEC_T, D, W_Ks_h, D);
  auto V_s_cpu    = ref_gemm_bt(norm1_cpu, DEC_T, D, W_Vs_h, D);
  auto self_cpu   = ref_sdpa_cross(Q_s_cpu, K_s_cpu, V_s_cpu,
                                    B, DEC_T, DEC_T, NH, NH, HD, scale, true);
  auto proj_s_cpu = ref_gemm_bt(self_cpu, DEC_T, D, W_os_h, D);
  auto h1_cpu     = ref_add(x_dec_h, proj_s_cpu);

  // Cross-attention sublayer
  auto norm2_cpu  = ref_layer_norm(h1_cpu, gamma2_h, beta2_h, DEC_T, D, eps);
  auto Q_c_cpu    = ref_gemm_bt(norm2_cpu, DEC_T, D, W_Qc_h, D);
  auto K_c_cpu    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Kc_h, D);  // [ENC_T, D]
  auto V_c_cpu    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Vc_h, D);
  auto cross_cpu  = ref_sdpa_cross(Q_c_cpu, K_c_cpu, V_c_cpu,
                                    B, DEC_T, ENC_T, NH, NH, HD, scale, false);
  auto proj_c_cpu = ref_gemm_bt(cross_cpu, DEC_T, D, W_oc_h, D);
  auto h2_cpu     = ref_add(h1_cpu, proj_c_cpu);

  // FFN sublayer
  auto norm3_cpu  = ref_layer_norm(h2_cpu, gamma3_h, beta3_h, DEC_T, D, eps);
  auto ffn1_cpu   = ref_gemm_bt(norm3_cpu, DEC_T, D, W1_h, FFN);
  auto ffn1a_cpu  = ref_relu(ffn1_cpu);
  auto ffn2_cpu   = ref_gemm_bt(ffn1a_cpu, DEC_T, FFN, W2_h, D);
  auto out_cpu    = ref_add(h2_cpu, ffn2_cpu);

  // ==========================================================================
  // Metal forward pass
  // ==========================================================================

  // Weights
  float* gamma1_m = metal_from<float>(gamma1_h);
  float* beta1_m  = metal_from<float>(beta1_h);
  float* W_Qs_m   = metal_from<float>(W_Qs_h);
  float* W_Ks_m   = metal_from<float>(W_Ks_h);
  float* W_Vs_m   = metal_from<float>(W_Vs_h);
  float* W_os_m   = metal_from<float>(W_os_h);
  float* gamma2_m = metal_from<float>(gamma2_h);
  float* beta2_m  = metal_from<float>(beta2_h);
  float* W_Qc_m   = metal_from<float>(W_Qc_h);
  float* W_Kc_m   = metal_from<float>(W_Kc_h);
  float* W_Vc_m   = metal_from<float>(W_Vc_h);
  float* W_oc_m   = metal_from<float>(W_oc_h);
  float* gamma3_m = metal_from<float>(gamma3_h);
  float* beta3_m  = metal_from<float>(beta3_h);
  float* W1_m     = metal_from<float>(W1_h);
  float* W2_m     = metal_from<float>(W2_h);

  // Inputs
  float* x_m   = metal_from<float>(x_dec_h);
  float* enc_m = metal_from<float>(enc_ctx_h);

  // Intermediate buffers
  float* norm1_m  = metal_alloc<float>(DEC_T * D);
  float* Q_s_m    = metal_alloc<float>(B * DEC_T * NH * HD);
  float* K_s_m    = metal_alloc<float>(B * DEC_T * NH * HD);
  float* V_s_m    = metal_alloc<float>(B * DEC_T * NH * HD);
  float* self_m   = metal_alloc<float>(B * DEC_T * NH * HD);
  float* proj_s_m = metal_alloc<float>(DEC_T * D);
  float* h1_m     = metal_alloc<float>(DEC_T * D);
  float* norm2_m  = metal_alloc<float>(DEC_T * D);
  float* Q_c_m    = metal_alloc<float>(B * DEC_T * NH * HD);
  float* K_c_m    = metal_alloc<float>(B * ENC_T * NH * HD);
  float* V_c_m    = metal_alloc<float>(B * ENC_T * NH * HD);
  float* cross_m  = metal_alloc<float>(B * DEC_T * NH * HD);
  float* proj_c_m = metal_alloc<float>(DEC_T * D);
  float* h2_m     = metal_alloc<float>(DEC_T * D);
  float* norm3_m  = metal_alloc<float>(DEC_T * D);
  float* ffn1_m   = metal_alloc<float>(DEC_T * FFN);
  float* ffn1a_m  = metal_alloc<float>(DEC_T * FFN);
  float* ffn2_m   = metal_alloc<float>(DEC_T * D);
  float* out_m    = metal_alloc<float>(DEC_T * D);

  // Step 1: Pre-self-attn LayerNorm
  metal::layer_norm_metal<float>(x_m, gamma1_m, beta1_m, norm1_m, DEC_T, D, eps);

  // Step 2: Self-attn Q/K/V projections  (norm1 × W^T)
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      norm1_m, D, W_Qs_m, D, 0.f, Q_s_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      norm1_m, D, W_Ks_m, D, 0.f, K_s_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      norm1_m, D, W_Vs_m, D, 0.f, V_s_m, D);

  // Step 3: Causal self-SDPA [B, DEC_T, NH, HD]
  metal::sdpa_metal<float>(Q_s_m, K_s_m, V_s_m, self_m,
                            B, DEC_T, DEC_T, NH, NH, HD, scale, true);

  // Step 4: Self-attn output projection + residual → h1
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      self_m, D, W_os_m, D, 0.f, proj_s_m, D);
  primitives<Device::MPS>::add<float>(x_m, proj_s_m, h1_m, DEC_T * D);

  // Step 5: Pre-cross-attn LayerNorm
  metal::layer_norm_metal<float>(h1_m, gamma2_m, beta2_m, norm2_m, DEC_T, D, eps);

  // Step 6: Cross-attn Q from decoder, K/V from encoder context
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      norm2_m, D, W_Qc_m, D, 0.f, Q_c_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, ENC_T, D, D, 1.f,
      enc_m, D, W_Kc_m, D, 0.f, K_c_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, ENC_T, D, D, 1.f,
      enc_m, D, W_Vc_m, D, 0.f, V_c_m, D);

  // Step 7: Non-causal cross-SDPA (sq=DEC_T, sk=ENC_T)
  metal::sdpa_metal<float>(Q_c_m, K_c_m, V_c_m, cross_m,
                            B, DEC_T, ENC_T, NH, NH, HD, scale, false);

  // Step 8: Cross-attn output projection + residual → h2
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, D, 1.f,
      cross_m, D, W_oc_m, D, 0.f, proj_c_m, D);
  primitives<Device::MPS>::add<float>(h1_m, proj_c_m, h2_m, DEC_T * D);

  // Step 9: Pre-FFN LayerNorm
  metal::layer_norm_metal<float>(h2_m, gamma3_m, beta3_m, norm3_m, DEC_T, D, eps);

  // Step 10: FFN
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, FFN, D, 1.f,
      norm3_m, D, W1_m, D, 0.f, ffn1_m, FFN);
  primitives<Device::MPS>::relu<float>(ffn1_m, ffn1a_m, DEC_T * FFN);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, DEC_T, D, FFN, 1.f,
      ffn1a_m, FFN, W2_m, FFN, 0.f, ffn2_m, D);
  primitives<Device::MPS>::add<float>(h2_m, ffn2_m, out_m, DEC_T * D);

  // Compare output
  auto out_metal = metal_to_host(out_m, DEC_T * D);
  float err = max_abs_diff(out_cpu, out_metal);
  std::printf("  max_abs_diff = %.2e  (output [%lld × %lld])\n",
              static_cast<double>(err),
              static_cast<long long>(DEC_T), static_cast<long long>(D));
  CHECK("full decoder prefill Metal vs CPU: max_abs_diff < 2e-4", err < 2e-4f);

  // Intermediate checks
  {
    auto v = metal_to_host(self_m, B * DEC_T * NH * HD);
    float e = max_abs_diff(self_cpu, v);
    std::printf("  causal self-attn max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder prefill: causal self-attn Metal vs CPU < 1e-4", e < 1e-4f);
  }
  {
    auto v = metal_to_host(cross_m, B * DEC_T * NH * HD);
    float e = max_abs_diff(cross_cpu, v);
    std::printf("  cross-attn max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder prefill: cross-attn (sq=4, sk=6) Metal vs CPU < 1e-4", e < 1e-4f);
  }
  {
    auto v = metal_to_host(h1_m, DEC_T * D);
    float e = max_abs_diff(h1_cpu, v);
    std::printf("  h1 max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder prefill: h1 (self-attn residual) Metal vs CPU < 1e-4", e < 1e-4f);
  }
  {
    auto v = metal_to_host(h2_m, DEC_T * D);
    float e = max_abs_diff(h2_cpu, v);
    std::printf("  h2 max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder prefill: h2 (cross-attn residual) Metal vs CPU < 1e-4", e < 1e-4f);
  }

  // Free weights
  metal_free(gamma1_m); metal_free(beta1_m);
  metal_free(W_Qs_m);   metal_free(W_Ks_m);   metal_free(W_Vs_m); metal_free(W_os_m);
  metal_free(gamma2_m); metal_free(beta2_m);
  metal_free(W_Qc_m);   metal_free(W_Kc_m);   metal_free(W_Vc_m); metal_free(W_oc_m);
  metal_free(gamma3_m); metal_free(beta3_m);   metal_free(W1_m);   metal_free(W2_m);
  metal_free(x_m);      metal_free(enc_m);

  // Free intermediates
  metal_free(norm1_m);  metal_free(Q_s_m);    metal_free(K_s_m);   metal_free(V_s_m);
  metal_free(self_m);   metal_free(proj_s_m); metal_free(h1_m);
  metal_free(norm2_m);  metal_free(Q_c_m);    metal_free(K_c_m);   metal_free(V_c_m);
  metal_free(cross_m);  metal_free(proj_c_m); metal_free(h2_m);
  metal_free(norm3_m);  metal_free(ffn1_m);   metal_free(ffn1a_m);
  metal_free(ffn2_m);   metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 4 — Full decoder layer decode step (sq=1, offset=DEC_T=4)
//
// State: KV cache has DEC_T=4 tokens from a simulated prefill.
// Action: process one new decoder token.
//
// Key KV-cache decode pattern (mirrors flash_attention_metal.mm M6.2):
//   1. Compute Q_new, K_new, V_new via GEMM on new token.
//   2. commit_and_wait() — flush GPU so CPU can read K_new/V_new.
//   3. memcpy K_new/V_new into cache at position offset=DEC_T.
//   4. sdpa_metal(sq=1, sk=DEC_T+1=5, is_causal=false) over full cache.
//
// Cross-attention uses encoder KV recomputed from enc_ctx (same as prefill).
// ---------------------------------------------------------------------------
static void test_full_decoder_decode() {
  std::printf("\n--- Test 4: Full Decoder Layer decode step (sq=1, offset=4) Metal vs CPU ---\n");

  const dim_t B = 1, DEC_T = 4, ENC_T = 6, D = 16, NH = 2, HD = 8, FFN = 32;
  const dim_t NK = NH;
  const dim_t MAX_CACHE = 8;
  const dim_t row_elems = NK * HD;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps   = 1e-5f;

  // Weights (fresh random, independent of Test 3):
  auto gamma1_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta1_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Ks_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Vs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_os_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto gamma2_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta2_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Kc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_Vc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_oc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto gamma3_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta3_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W1_h     = rand_vec(static_cast<std::size_t>(FFN * D), -0.5f, 0.5f);
  auto W2_h     = rand_vec(static_cast<std::size_t>(D * FFN), -0.5f, 0.5f);

  // Inputs:
  auto x_new_h  = rand_vec(static_cast<std::size_t>(B * 1 * D));  // new decoder token
  auto enc_ctx_h = rand_vec(static_cast<std::size_t>(B * ENC_T * D));

  // Simulated prefill cache: DEC_T rows of K/V from prior tokens.
  auto k_cache_init_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems));
  auto v_cache_init_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems));

  // ==========================================================================
  // CPU reference for decode step
  // ==========================================================================

  // Self-attention on new token (sq=1)
  auto norm1_new = ref_layer_norm(x_new_h, gamma1_h, beta1_h, 1, D, eps);
  auto q_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Qs_h, D);  // [1, D]
  auto k_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Ks_h, D);  // new K [1, D]
  auto v_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Vs_h, D);  // new V [1, D]

  // Build full K/V: [prefill (DEC_T rows), new (1 row)] = [sk_eff, D]
  const dim_t sk_eff = DEC_T + 1;  // 5
  std::vector<float> k_all(static_cast<std::size_t>(sk_eff * row_elems));
  std::vector<float> v_all(static_cast<std::size_t>(sk_eff * row_elems));
  std::copy(k_cache_init_h.begin(), k_cache_init_h.end(), k_all.begin());
  std::copy(v_cache_init_h.begin(), v_cache_init_h.end(), v_all.begin());
  std::copy(k_s_new.begin(), k_s_new.end(),
            k_all.begin() + static_cast<std::size_t>(DEC_T * row_elems));
  std::copy(v_s_new.begin(), v_s_new.end(),
            v_all.begin() + static_cast<std::size_t>(DEC_T * row_elems));

  // All cached tokens are in the past → is_causal=false for decode
  auto self_new   = ref_sdpa_cross(q_s_new, k_all, v_all,
                                    B, 1, sk_eff, NH, NK, HD, scale, false);
  auto proj_s_new = ref_gemm_bt(self_new, 1, D, W_os_h, D);
  auto h1_new     = ref_add(x_new_h, proj_s_new);

  // Cross-attention (sq=1, sk=ENC_T=6)
  auto norm2_new  = ref_layer_norm(h1_new, gamma2_h, beta2_h, 1, D, eps);
  auto q_c_new    = ref_gemm_bt(norm2_new, 1, D, W_Qc_h, D);
  auto K_c_enc    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Kc_h, D);
  auto V_c_enc    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Vc_h, D);
  auto cross_new  = ref_sdpa_cross(q_c_new, K_c_enc, V_c_enc,
                                    B, 1, ENC_T, NH, NK, HD, scale, false);
  auto proj_c_new = ref_gemm_bt(cross_new, 1, D, W_oc_h, D);
  auto h2_new     = ref_add(h1_new, proj_c_new);

  // FFN (sq=1)
  auto norm3_new  = ref_layer_norm(h2_new, gamma3_h, beta3_h, 1, D, eps);
  auto ffn1_new   = ref_gemm_bt(norm3_new, 1, D, W1_h, FFN);
  auto ffn1a_new  = ref_relu(ffn1_new);
  auto ffn2_new   = ref_gemm_bt(ffn1a_new, 1, FFN, W2_h, D);
  auto out_cpu    = ref_add(h2_new, ffn2_new);

  // ==========================================================================
  // Metal decode step
  // ==========================================================================

  // Allocate KV cache [B, MAX_CACHE, NK, HD] and seed with prefill data.
  float* k_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  float* v_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  std::memcpy(k_cache, k_cache_init_h.data(),
              static_cast<std::size_t>(DEC_T * row_elems) * sizeof(float));
  std::memcpy(v_cache, v_cache_init_h.data(),
              static_cast<std::size_t>(DEC_T * row_elems) * sizeof(float));

  // Weights
  float* gamma1_m = metal_from<float>(gamma1_h);
  float* beta1_m  = metal_from<float>(beta1_h);
  float* W_Qs_m   = metal_from<float>(W_Qs_h);
  float* W_Ks_m   = metal_from<float>(W_Ks_h);
  float* W_Vs_m   = metal_from<float>(W_Vs_h);
  float* W_os_m   = metal_from<float>(W_os_h);
  float* gamma2_m = metal_from<float>(gamma2_h);
  float* beta2_m  = metal_from<float>(beta2_h);
  float* W_Qc_m   = metal_from<float>(W_Qc_h);
  float* W_Kc_m   = metal_from<float>(W_Kc_h);
  float* W_Vc_m   = metal_from<float>(W_Vc_h);
  float* W_oc_m   = metal_from<float>(W_oc_h);
  float* gamma3_m = metal_from<float>(gamma3_h);
  float* beta3_m  = metal_from<float>(beta3_h);
  float* W1_m     = metal_from<float>(W1_h);
  float* W2_m     = metal_from<float>(W2_h);

  // Inputs
  float* x_new_m = metal_from<float>(x_new_h);
  float* enc_m   = metal_from<float>(enc_ctx_h);

  // Intermediate buffers (sq=1 throughout)
  float* norm1_m  = metal_alloc<float>(1 * D);
  float* Q_s_m    = metal_alloc<float>(B * 1 * NH * HD);
  float* K_s_m    = metal_alloc<float>(B * 1 * NK * HD);   // new K only
  float* V_s_m    = metal_alloc<float>(B * 1 * NK * HD);   // new V only
  float* self_m   = metal_alloc<float>(B * 1 * NH * HD);
  float* proj_s_m = metal_alloc<float>(1 * D);
  float* h1_m     = metal_alloc<float>(1 * D);
  float* norm2_m  = metal_alloc<float>(1 * D);
  float* Q_c_m    = metal_alloc<float>(B * 1 * NH * HD);
  float* K_c_m    = metal_alloc<float>(B * ENC_T * NH * HD);
  float* V_c_m    = metal_alloc<float>(B * ENC_T * NH * HD);
  float* cross_m  = metal_alloc<float>(B * 1 * NH * HD);
  float* proj_c_m = metal_alloc<float>(1 * D);
  float* h2_m     = metal_alloc<float>(1 * D);
  float* norm3_m  = metal_alloc<float>(1 * D);
  float* ffn1_m   = metal_alloc<float>(1 * FFN);
  float* ffn1a_m  = metal_alloc<float>(1 * FFN);
  float* ffn2_m   = metal_alloc<float>(1 * D);
  float* out_m    = metal_alloc<float>(1 * D);

  // Step 1: Pre-self-attn LayerNorm on new token
  metal::layer_norm_metal<float>(x_new_m, gamma1_m, beta1_m, norm1_m, 1, D, eps);

  // Step 2: Q/K/V projections for new token
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      norm1_m, D, W_Qs_m, D, 0.f, Q_s_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      norm1_m, D, W_Ks_m, D, 0.f, K_s_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      norm1_m, D, W_Vs_m, D, 0.f, V_s_m, D);

  // Step 3: KV-cache update (M6.2 pattern):
  //   flush GPU so K_s_m/V_s_m are readable by CPU, then memcpy into cache.
  metal::commit_and_wait();
  for (dim_t b = 0; b < B; ++b) {
    float* kd = k_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    float* vd = v_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    std::memcpy(kd, K_s_m + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
    std::memcpy(vd, V_s_m + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
  }

  // Step 4: Self-SDPA over full cache (sq=1, sk=sk_eff=5, non-causal)
  metal::sdpa_metal<float>(Q_s_m, k_cache, v_cache, self_m,
                            B, 1, sk_eff, NH, NK, HD, scale, false);

  // Step 5: Self-attn output projection + residual → h1
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      self_m, D, W_os_m, D, 0.f, proj_s_m, D);
  primitives<Device::MPS>::add<float>(x_new_m, proj_s_m, h1_m, 1 * D);

  // Step 6: Pre-cross-attn LayerNorm
  metal::layer_norm_metal<float>(h1_m, gamma2_m, beta2_m, norm2_m, 1, D, eps);

  // Step 7: Cross-attn (Q from decoder sq=1, KV from encoder sk=ENC_T=6)
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      norm2_m, D, W_Qc_m, D, 0.f, Q_c_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, ENC_T, D, D, 1.f,
      enc_m, D, W_Kc_m, D, 0.f, K_c_m, D);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, ENC_T, D, D, 1.f,
      enc_m, D, W_Vc_m, D, 0.f, V_c_m, D);
  metal::sdpa_metal<float>(Q_c_m, K_c_m, V_c_m, cross_m,
                            B, 1, ENC_T, NH, NK, HD, scale, false);

  // Step 8: Cross-attn output projection + residual → h2
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, D, 1.f,
      cross_m, D, W_oc_m, D, 0.f, proj_c_m, D);
  primitives<Device::MPS>::add<float>(h1_m, proj_c_m, h2_m, 1 * D);

  // Step 9: Pre-FFN LayerNorm
  metal::layer_norm_metal<float>(h2_m, gamma3_m, beta3_m, norm3_m, 1, D, eps);

  // Step 10: FFN
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, FFN, D, 1.f,
      norm3_m, D, W1_m, D, 0.f, ffn1_m, FFN);
  primitives<Device::MPS>::relu<float>(ffn1_m, ffn1a_m, 1 * FFN);
  primitives<Device::MPS>::gemm<float, float>(
      false, false, false, true, 1, D, FFN, 1.f,
      ffn1a_m, FFN, W2_m, FFN, 0.f, ffn2_m, D);
  primitives<Device::MPS>::add<float>(h2_m, ffn2_m, out_m, 1 * D);

  // Compare output
  auto out_metal = metal_to_host(out_m, 1 * D);
  float err = max_abs_diff(out_cpu, out_metal);
  std::printf("  max_abs_diff = %.2e  (output [1 × %lld])\n",
              static_cast<double>(err), static_cast<long long>(D));
  CHECK("full decoder decode step Metal vs CPU: max_abs_diff < 2e-4", err < 2e-4f);

  // Intermediate checks
  {
    auto v = metal_to_host(self_m, B * 1 * NH * HD);
    float e = max_abs_diff(self_new, v);
    std::printf("  self-attn decode max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder decode: self-attn (sq=1, sk=5) Metal vs CPU < 1e-4", e < 1e-4f);
  }
  {
    auto v = metal_to_host(cross_m, B * 1 * NH * HD);
    float e = max_abs_diff(cross_new, v);
    std::printf("  cross-attn decode max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder decode: cross-attn (sq=1, sk=6) Metal vs CPU < 1e-4", e < 1e-4f);
  }
  {
    auto v = metal_to_host(h2_m, 1 * D);
    float e = max_abs_diff(h2_new, v);
    std::printf("  h2 decode max_abs_diff = %.2e\n", static_cast<double>(e));
    CHECK("decoder decode: h2 (cross-attn residual) Metal vs CPU < 1e-4", e < 1e-4f);
  }

  // Free
  metal_free(k_cache); metal_free(v_cache);
  metal_free(gamma1_m); metal_free(beta1_m);
  metal_free(W_Qs_m);   metal_free(W_Ks_m);   metal_free(W_Vs_m); metal_free(W_os_m);
  metal_free(gamma2_m); metal_free(beta2_m);
  metal_free(W_Qc_m);   metal_free(W_Kc_m);   metal_free(W_Vc_m); metal_free(W_oc_m);
  metal_free(gamma3_m); metal_free(beta3_m);   metal_free(W1_m);   metal_free(W2_m);
  metal_free(x_new_m);  metal_free(enc_m);
  metal_free(norm1_m);  metal_free(Q_s_m);    metal_free(K_s_m);   metal_free(V_s_m);
  metal_free(self_m);   metal_free(proj_s_m); metal_free(h1_m);
  metal_free(norm2_m);  metal_free(Q_c_m);    metal_free(K_c_m);   metal_free(V_c_m);
  metal_free(cross_m);  metal_free(proj_c_m); metal_free(h2_m);
  metal_free(norm3_m);  metal_free(ffn1_m);   metal_free(ffn1a_m);
  metal_free(ffn2_m);   metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 5 — KV-cache decode with batch > 1 (B=4)
//
// Validates the kv_batch_stride fix for sdpa_metal: when K/V come from a
// pre-allocated cache [B, MAX_CACHE, NK, HD] and seqlen_k_eff < MAX_CACHE,
// batches b >= 1 must use the physical cache stride (total_cache * NK * HD)
// rather than the logical stride (seqlen_k_eff * NK * HD).
//
// Parameters: B=4, offset=3, sq=1, sk_eff=4, MAX_CACHE=8, NH=2, NK=2, HD=8
// ---------------------------------------------------------------------------
static void test_kvcache_decode_batch_gt1() {
  std::printf("\n--- Test 5: KV-cache decode batch>1 (B=4, offset=3, sq=1, sk=4) Metal vs CPU ---\n");

  const dim_t B = 4, OFFSET = 3, NK = 2, HD = 8, MAX_CACHE = 8;
  const dim_t NH = 2;
  const dim_t row_elems = NK * HD;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const dim_t sk_eff = OFFSET + 1;  // 4 total tokens in cache

  // Pre-fill: OFFSET slots of K/V per batch item (each batch gets different data).
  auto k_prefill_h = rand_vec(static_cast<std::size_t>(B * OFFSET * row_elems));
  auto v_prefill_h = rand_vec(static_cast<std::size_t>(B * OFFSET * row_elems));

  // Allocate cache [B, MAX_CACHE, NK, HD] in Metal Shared memory.
  float* k_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  float* v_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);

  // Zero the whole cache to make stale slots deterministic.
  std::memset(k_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(float));
  std::memset(v_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(float));

  // Write prefill rows into each batch's cache slice.
  for (dim_t b = 0; b < B; ++b) {
    float* kd = k_cache + b * MAX_CACHE * row_elems;
    float* vd = v_cache + b * MAX_CACHE * row_elems;
    const float* ks = k_prefill_h.data() + b * OFFSET * row_elems;
    const float* vs = v_prefill_h.data() + b * OFFSET * row_elems;
    std::memcpy(kd, ks, static_cast<std::size_t>(OFFSET * row_elems) * sizeof(float));
    std::memcpy(vd, vs, static_cast<std::size_t>(OFFSET * row_elems) * sizeof(float));
  }

  // New token at position OFFSET: Q [B, 1, NH, HD], K/V new [B, 1, NK, HD].
  auto q_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NH * HD));
  auto k_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NK * HD));
  auto v_new_h = rand_vec(static_cast<std::size_t>(B * 1 * NK * HD));

  float* q_m   = metal_from<float>(q_new_h);
  float* out_m = metal_alloc<float>(B * 1 * NH * HD);

  // Metal decode path: flush GPU then CPU-write new K/V into cache at OFFSET.
  metal::commit_and_wait();
  for (dim_t b = 0; b < B; ++b) {
    float* kd = k_cache + (b * MAX_CACHE + OFFSET) * row_elems;
    float* vd = v_cache + (b * MAX_CACHE + OFFSET) * row_elems;
    std::memcpy(kd, k_new_h.data() + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
    std::memcpy(vd, v_new_h.data() + b * row_elems,
                static_cast<std::size_t>(row_elems) * sizeof(float));
  }

  // SDPA over cache with explicit kv_batch_stride.
  const dim_t kv_bstride = MAX_CACHE * NK * HD;
  metal::sdpa_metal<float>(q_m, k_cache, v_cache, out_m,
                            B, 1, sk_eff, NH, NK, HD, scale, false,
                            kv_bstride);
  auto metal_out = metal_to_host(out_m, B * 1 * NH * HD);

  // CPU reference: build K_all/V_all [B, sk_eff, NK, HD] (tightly packed).
  std::vector<float> k_all(static_cast<std::size_t>(B * sk_eff * row_elems));
  std::vector<float> v_all(static_cast<std::size_t>(B * sk_eff * row_elems));
  for (dim_t b = 0; b < B; ++b) {
    std::size_t dst_base = static_cast<std::size_t>(b * sk_eff * row_elems);
    // Prefill slots [0..OFFSET-1]
    std::copy(k_prefill_h.begin() + b * OFFSET * row_elems,
              k_prefill_h.begin() + (b + 1) * OFFSET * row_elems,
              k_all.begin() + dst_base);
    std::copy(v_prefill_h.begin() + b * OFFSET * row_elems,
              v_prefill_h.begin() + (b + 1) * OFFSET * row_elems,
              v_all.begin() + dst_base);
    // New slot at [OFFSET]
    std::size_t new_off = dst_base + static_cast<std::size_t>(OFFSET * row_elems);
    std::copy(k_new_h.begin() + b * row_elems,
              k_new_h.begin() + (b + 1) * row_elems,
              k_all.begin() + new_off);
    std::copy(v_new_h.begin() + b * row_elems,
              v_new_h.begin() + (b + 1) * row_elems,
              v_all.begin() + new_off);
  }

  auto cpu = ref_sdpa_cross(q_new_h, k_all, v_all,
                             B, 1, sk_eff, NH, NK, HD, scale, false);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("kv-cache decode B=4 [offset=3, sq=1, sk=4]: max_abs_diff < 1e-4", err < 1e-4f);

  // Also verify per-batch: check that each batch produces independently correct output.
  for (dim_t b = 0; b < B; ++b) {
    float batch_err = 0.f;
    for (dim_t i = 0; i < NH * HD; ++i) {
      std::size_t idx = static_cast<std::size_t>(b * NH * HD + i);
      batch_err = std::max(batch_err, std::abs(cpu[idx] - metal_out[idx]));
    }
    char label[128];
    std::snprintf(label, sizeof(label),
                  "kv-cache decode B=4: batch %lld max_abs_diff < 1e-4",
                  static_cast<long long>(b));
    CHECK(label, batch_err < 1e-4f);
  }

  metal_free(k_cache); metal_free(v_cache);
  metal_free(q_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 6/7 — Full decoder layer decode step in fp16 / bf16
//
// Same decode-step pipeline as Test 4 but in reduced precision.
// CPU reference stays float32; tolerances relaxed.
// ---------------------------------------------------------------------------

template <typename T>
static void test_decoder_decode_typed(const char* type_name, float tol) {
  std::printf("\n--- Test: Decoder decode step (%s) Metal vs CPU ---\n", type_name);

  const dim_t B = 1, DEC_T = 4, ENC_T = 6, D = 16, NH = 2, HD = 8, FFN = 32;
  const dim_t NK = NH;
  const dim_t MAX_CACHE = 8;
  const dim_t row_elems = NK * HD;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps   = 1e-5f;

  // Weights (small range to limit reduced-precision error accumulation).
  auto gamma1_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta1_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_Ks_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_Vs_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_os_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto gamma2_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta2_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Qc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_Kc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_Vc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto W_oc_h   = rand_vec(static_cast<std::size_t>(D * D), -0.3f, 0.3f);
  auto gamma3_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta3_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W1_h     = rand_vec(static_cast<std::size_t>(FFN * D), -0.3f, 0.3f);
  auto W2_h     = rand_vec(static_cast<std::size_t>(D * FFN), -0.3f, 0.3f);

  auto x_new_h   = rand_vec(static_cast<std::size_t>(B * 1 * D), -0.5f, 0.5f);
  auto enc_ctx_h = rand_vec(static_cast<std::size_t>(B * ENC_T * D), -0.5f, 0.5f);
  auto k_cache_init_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems), -0.5f, 0.5f);
  auto v_cache_init_h = rand_vec(static_cast<std::size_t>(DEC_T * row_elems), -0.5f, 0.5f);

  // CPU float32 reference (identical to Test 4).
  auto norm1_new = ref_layer_norm(x_new_h, gamma1_h, beta1_h, 1, D, eps);
  auto q_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Qs_h, D);
  auto k_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Ks_h, D);
  auto v_s_new   = ref_gemm_bt(norm1_new, 1, D, W_Vs_h, D);
  const dim_t sk_eff = DEC_T + 1;
  std::vector<float> k_all(static_cast<std::size_t>(sk_eff * row_elems));
  std::vector<float> v_all(static_cast<std::size_t>(sk_eff * row_elems));
  std::copy(k_cache_init_h.begin(), k_cache_init_h.end(), k_all.begin());
  std::copy(v_cache_init_h.begin(), v_cache_init_h.end(), v_all.begin());
  std::copy(k_s_new.begin(), k_s_new.end(),
            k_all.begin() + static_cast<std::size_t>(DEC_T * row_elems));
  std::copy(v_s_new.begin(), v_s_new.end(),
            v_all.begin() + static_cast<std::size_t>(DEC_T * row_elems));
  auto self_new   = ref_sdpa_cross(q_s_new, k_all, v_all,
                                    B, 1, sk_eff, NH, NK, HD, scale, false);
  auto proj_s_new = ref_gemm_bt(self_new, 1, D, W_os_h, D);
  auto h1_new     = ref_add(x_new_h, proj_s_new);
  auto norm2_new  = ref_layer_norm(h1_new, gamma2_h, beta2_h, 1, D, eps);
  auto q_c_new    = ref_gemm_bt(norm2_new, 1, D, W_Qc_h, D);
  auto K_c_enc    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Kc_h, D);
  auto V_c_enc    = ref_gemm_bt(enc_ctx_h, ENC_T, D, W_Vc_h, D);
  auto cross_new  = ref_sdpa_cross(q_c_new, K_c_enc, V_c_enc,
                                    B, 1, ENC_T, NH, NK, HD, scale, false);
  auto proj_c_new = ref_gemm_bt(cross_new, 1, D, W_oc_h, D);
  auto h2_new     = ref_add(h1_new, proj_c_new);
  auto norm3_new  = ref_layer_norm(h2_new, gamma3_h, beta3_h, 1, D, eps);
  auto ffn1_new   = ref_gemm_bt(norm3_new, 1, D, W1_h, FFN);
  auto ffn1a_new  = ref_relu(ffn1_new);
  auto ffn2_new   = ref_gemm_bt(ffn1a_new, 1, FFN, W2_h, D);
  auto out_cpu    = ref_add(h2_new, ffn2_new);

  // Metal decode step in type T.
  T* k_cache = metal_alloc<T>(B * MAX_CACHE * NK * HD);
  T* v_cache = metal_alloc<T>(B * MAX_CACHE * NK * HD);
  std::memset(k_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(T));
  std::memset(v_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(T));
  for (dim_t i = 0; i < DEC_T * row_elems; ++i) {
    k_cache[i] = T(k_cache_init_h[static_cast<std::size_t>(i)]);
    v_cache[i] = T(v_cache_init_h[static_cast<std::size_t>(i)]);
  }

  T* gamma1_m = metal_from<T>(gamma1_h);  T* beta1_m  = metal_from<T>(beta1_h);
  T* W_Qs_m   = metal_from<T>(W_Qs_h);    T* W_Ks_m   = metal_from<T>(W_Ks_h);
  T* W_Vs_m   = metal_from<T>(W_Vs_h);    T* W_os_m   = metal_from<T>(W_os_h);
  T* gamma2_m = metal_from<T>(gamma2_h);   T* beta2_m  = metal_from<T>(beta2_h);
  T* W_Qc_m   = metal_from<T>(W_Qc_h);    T* W_Kc_m   = metal_from<T>(W_Kc_h);
  T* W_Vc_m   = metal_from<T>(W_Vc_h);    T* W_oc_m   = metal_from<T>(W_oc_h);
  T* gamma3_m = metal_from<T>(gamma3_h);   T* beta3_m  = metal_from<T>(beta3_h);
  T* W1_m     = metal_from<T>(W1_h);       T* W2_m     = metal_from<T>(W2_h);
  T* x_new_m  = metal_from<T>(x_new_h);    T* enc_m    = metal_from<T>(enc_ctx_h);

  T* norm1_m  = metal_alloc<T>(1 * D);
  T* Q_s_m    = metal_alloc<T>(B * 1 * NH * HD);
  T* K_s_m    = metal_alloc<T>(B * 1 * NK * HD);
  T* V_s_m    = metal_alloc<T>(B * 1 * NK * HD);
  T* self_m   = metal_alloc<T>(B * 1 * NH * HD);
  T* proj_s_m = metal_alloc<T>(1 * D);
  T* h1_m     = metal_alloc<T>(1 * D);
  T* norm2_m  = metal_alloc<T>(1 * D);
  T* Q_c_m    = metal_alloc<T>(B * 1 * NH * HD);
  T* K_c_m    = metal_alloc<T>(B * ENC_T * NH * HD);
  T* V_c_m    = metal_alloc<T>(B * ENC_T * NH * HD);
  T* cross_m  = metal_alloc<T>(B * 1 * NH * HD);
  T* proj_c_m = metal_alloc<T>(1 * D);
  T* h2_m     = metal_alloc<T>(1 * D);
  T* norm3_m  = metal_alloc<T>(1 * D);
  T* ffn1_m   = metal_alloc<T>(1 * FFN);
  T* ffn1a_m  = metal_alloc<T>(1 * FFN);
  T* ffn2_m   = metal_alloc<T>(1 * D);
  T* out_m    = metal_alloc<T>(1 * D);

  // Self-attention
  metal::layer_norm_metal<T>(x_new_m, gamma1_m, beta1_m, norm1_m, 1, D, eps);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, norm1_m, D, W_Qs_m, D, 0.f, Q_s_m, D);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, norm1_m, D, W_Ks_m, D, 0.f, K_s_m, D);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, norm1_m, D, W_Vs_m, D, 0.f, V_s_m, D);

  // KV-cache update
  metal::commit_and_wait();
  for (dim_t b = 0; b < B; ++b) {
    T* kd = k_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    T* vd = v_cache + (b * MAX_CACHE + DEC_T) * row_elems;
    std::memcpy(kd, K_s_m + b * row_elems, static_cast<std::size_t>(row_elems) * sizeof(T));
    std::memcpy(vd, V_s_m + b * row_elems, static_cast<std::size_t>(row_elems) * sizeof(T));
  }

  const dim_t kv_bstride = MAX_CACHE * NK * HD;
  metal::sdpa_metal<T>(Q_s_m, k_cache, v_cache, self_m,
                        B, 1, sk_eff, NH, NK, HD, scale, false, kv_bstride);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, self_m, D, W_os_m, D, 0.f, proj_s_m, D);
  primitives<Device::MPS>::add<T>(x_new_m, proj_s_m, h1_m, 1 * D);

  // Cross-attention
  metal::layer_norm_metal<T>(h1_m, gamma2_m, beta2_m, norm2_m, 1, D, eps);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, norm2_m, D, W_Qc_m, D, 0.f, Q_c_m, D);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, ENC_T, D, D, 1.f, enc_m, D, W_Kc_m, D, 0.f, K_c_m, D);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, ENC_T, D, D, 1.f, enc_m, D, W_Vc_m, D, 0.f, V_c_m, D);
  metal::sdpa_metal<T>(Q_c_m, K_c_m, V_c_m, cross_m,
                        B, 1, ENC_T, NH, NK, HD, scale, false);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, D, 1.f, cross_m, D, W_oc_m, D, 0.f, proj_c_m, D);
  primitives<Device::MPS>::add<T>(h1_m, proj_c_m, h2_m, 1 * D);

  // FFN
  metal::layer_norm_metal<T>(h2_m, gamma3_m, beta3_m, norm3_m, 1, D, eps);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, FFN, D, 1.f, norm3_m, D, W1_m, D, 0.f, ffn1_m, FFN);
  primitives<Device::MPS>::relu<T>(ffn1_m, ffn1a_m, 1 * FFN);
  primitives<Device::MPS>::gemm<T, T>(
      false, false, false, true, 1, D, FFN, 1.f, ffn1a_m, FFN, W2_m, FFN, 0.f, ffn2_m, D);
  primitives<Device::MPS>::add<T>(h2_m, ffn2_m, out_m, 1 * D);

  auto out_metal = metal_to_host(out_m, 1 * D);
  float err = max_abs_diff(out_cpu, out_metal);
  std::printf("  max_abs_diff = %.2e  (tol = %.2e)\n",
              static_cast<double>(err), static_cast<double>(tol));

  char label[128];
  std::snprintf(label, sizeof(label),
                "decoder decode (%s) Metal vs CPU: max_abs_diff < %.0e",
                type_name, static_cast<double>(tol));
  CHECK(label, err < tol);

  metal_free(k_cache); metal_free(v_cache);
  metal_free(gamma1_m); metal_free(beta1_m);
  metal_free(W_Qs_m); metal_free(W_Ks_m); metal_free(W_Vs_m); metal_free(W_os_m);
  metal_free(gamma2_m); metal_free(beta2_m);
  metal_free(W_Qc_m); metal_free(W_Kc_m); metal_free(W_Vc_m); metal_free(W_oc_m);
  metal_free(gamma3_m); metal_free(beta3_m); metal_free(W1_m); metal_free(W2_m);
  metal_free(x_new_m); metal_free(enc_m);
  metal_free(norm1_m); metal_free(Q_s_m); metal_free(K_s_m); metal_free(V_s_m);
  metal_free(self_m); metal_free(proj_s_m); metal_free(h1_m);
  metal_free(norm2_m); metal_free(Q_c_m); metal_free(K_c_m); metal_free(V_c_m);
  metal_free(cross_m); metal_free(proj_c_m); metal_free(h2_m);
  metal_free(norm3_m); metal_free(ffn1_m); metal_free(ffn1a_m);
  metal_free(ffn2_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 8 — Multi-step decode sequence (offset=0 prefill, then 3 decode steps)
//
// Validates that the KV cache grows correctly across multiple decode steps.
// Each decode step writes new K/V at the next position and attends over
// all accumulated tokens.
//
// Pipeline (self-attention only, no cross-attn, no FFN — isolates cache logic):
//   For step s (offset = PREFILL_T + s):
//     1. LN + Q/K/V projection on new token.
//     2. commit_and_wait + memcpy K/V into cache at position `offset`.
//     3. sdpa_metal(sq=1, sk=offset+1, is_causal=false) over cache.
//     4. output projection + residual.
//   Compare Metal output with CPU reference at each step.
// ---------------------------------------------------------------------------
static void test_multistep_decode_sequence() {
  std::printf("\n--- Test 8: Multi-step decode sequence (prefill=2, 3 steps) Metal vs CPU ---\n");

  const dim_t B = 1, PREFILL_T = 2, NUM_STEPS = 3;
  const dim_t D = 16, NH = 2, HD = 8, NK = NH;
  const dim_t MAX_CACHE = 16;
  const dim_t row_elems = NK * HD;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps   = 1e-5f;

  // Weights
  auto gamma_h  = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta_h   = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);
  auto W_Q_h    = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_K_h    = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_V_h    = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);
  auto W_o_h    = rand_vec(static_cast<std::size_t>(D * D), -0.5f, 0.5f);

  // Pre-fill KV cache with PREFILL_T rows.
  auto k_prefill_h = rand_vec(static_cast<std::size_t>(PREFILL_T * row_elems));
  auto v_prefill_h = rand_vec(static_cast<std::size_t>(PREFILL_T * row_elems));

  // CPU running cache (tightly packed, grows each step).
  std::vector<float> k_cpu_cache(k_prefill_h);
  std::vector<float> v_cpu_cache(v_prefill_h);

  // Metal cache [B, MAX_CACHE, NK, HD].
  float* k_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  float* v_cache = metal_alloc<float>(B * MAX_CACHE * NK * HD);
  std::memset(k_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(float));
  std::memset(v_cache, 0, static_cast<std::size_t>(B * MAX_CACHE * row_elems) * sizeof(float));
  std::memcpy(k_cache, k_prefill_h.data(),
              static_cast<std::size_t>(PREFILL_T * row_elems) * sizeof(float));
  std::memcpy(v_cache, v_prefill_h.data(),
              static_cast<std::size_t>(PREFILL_T * row_elems) * sizeof(float));

  // Metal weights
  float* gamma_m = metal_from<float>(gamma_h);
  float* beta_m  = metal_from<float>(beta_h);
  float* W_Q_m   = metal_from<float>(W_Q_h);
  float* W_K_m   = metal_from<float>(W_K_h);
  float* W_V_m   = metal_from<float>(W_V_h);
  float* W_o_m   = metal_from<float>(W_o_h);

  // Reusable intermediate buffers (sq=1).
  float* norm_m   = metal_alloc<float>(1 * D);
  float* Q_m      = metal_alloc<float>(B * 1 * NH * HD);
  float* K_new_m  = metal_alloc<float>(B * 1 * NK * HD);
  float* V_new_m  = metal_alloc<float>(B * 1 * NK * HD);
  float* attn_m   = metal_alloc<float>(B * 1 * NH * HD);
  float* proj_m   = metal_alloc<float>(1 * D);
  float* out_m    = metal_alloc<float>(1 * D);

  const dim_t kv_bstride = MAX_CACHE * NK * HD;

  bool all_ok = true;
  for (dim_t step = 0; step < NUM_STEPS; ++step) {
    const dim_t offset = PREFILL_T + step;
    const dim_t sk_eff = offset + 1;

    // New token for this step.
    auto x_h = rand_vec(static_cast<std::size_t>(B * 1 * D));
    float* x_m = metal_from<float>(x_h);

    // --- CPU reference ---
    auto norm_cpu = ref_layer_norm(x_h, gamma_h, beta_h, 1, D, eps);
    auto q_cpu    = ref_gemm_bt(norm_cpu, 1, D, W_Q_h, D);
    auto k_new    = ref_gemm_bt(norm_cpu, 1, D, W_K_h, D);
    auto v_new    = ref_gemm_bt(norm_cpu, 1, D, W_V_h, D);

    // Append to CPU cache.
    k_cpu_cache.insert(k_cpu_cache.end(), k_new.begin(), k_new.end());
    v_cpu_cache.insert(v_cpu_cache.end(), v_new.begin(), v_new.end());

    auto sdpa_cpu = ref_sdpa_cross(q_cpu, k_cpu_cache, v_cpu_cache,
                                    B, 1, sk_eff, NH, NK, HD, scale, false);
    auto proj_cpu = ref_gemm_bt(sdpa_cpu, 1, D, W_o_h, D);
    auto out_cpu  = ref_add(x_h, proj_cpu);

    // --- Metal ---
    metal::layer_norm_metal<float>(x_m, gamma_m, beta_m, norm_m, 1, D, eps);
    primitives<Device::MPS>::gemm<float, float>(
        false, false, false, true, 1, D, D, 1.f, norm_m, D, W_Q_m, D, 0.f, Q_m, D);
    primitives<Device::MPS>::gemm<float, float>(
        false, false, false, true, 1, D, D, 1.f, norm_m, D, W_K_m, D, 0.f, K_new_m, D);
    primitives<Device::MPS>::gemm<float, float>(
        false, false, false, true, 1, D, D, 1.f, norm_m, D, W_V_m, D, 0.f, V_new_m, D);

    metal::commit_and_wait();
    std::memcpy(k_cache + offset * row_elems, K_new_m,
                static_cast<std::size_t>(row_elems) * sizeof(float));
    std::memcpy(v_cache + offset * row_elems, V_new_m,
                static_cast<std::size_t>(row_elems) * sizeof(float));

    metal::sdpa_metal<float>(Q_m, k_cache, v_cache, attn_m,
                              B, 1, sk_eff, NH, NK, HD, scale, false, kv_bstride);
    primitives<Device::MPS>::gemm<float, float>(
        false, false, false, true, 1, D, D, 1.f, attn_m, D, W_o_m, D, 0.f, proj_m, D);
    primitives<Device::MPS>::add<float>(x_m, proj_m, out_m, 1 * D);

    auto out_metal = metal_to_host(out_m, 1 * D);
    float err = max_abs_diff(out_cpu, out_metal);
    std::printf("  step %lld (offset=%lld, sk=%lld): max_abs_diff = %.2e\n",
                static_cast<long long>(step),
                static_cast<long long>(offset),
                static_cast<long long>(sk_eff),
                static_cast<double>(err));

    char label[128];
    std::snprintf(label, sizeof(label),
                  "multi-step decode step %lld (offset=%lld, sk=%lld) < 1e-4",
                  static_cast<long long>(step),
                  static_cast<long long>(offset),
                  static_cast<long long>(sk_eff));
    bool ok = err < 1e-4f;
    CHECK(label, ok);
    if (!ok) all_ok = false;

    metal_free(x_m);
  }

  metal_free(k_cache); metal_free(v_cache);
  metal_free(gamma_m); metal_free(beta_m);
  metal_free(W_Q_m); metal_free(W_K_m); metal_free(W_V_m); metal_free(W_o_m);
  metal_free(norm_m); metal_free(Q_m); metal_free(K_new_m); metal_free(V_new_m);
  metal_free(attn_m); metal_free(proj_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M8.2 Transformer Decoder Layer integration tests ===\n");

  @autoreleasepool {
    test_cross_attn_sdpa();
    test_kvcache_decode();
    test_full_decoder_prefill();
    test_full_decoder_decode();
    test_kvcache_decode_batch_gt1();
    test_decoder_decode_typed<ctranslate2::float16_t>("f16", 5e-2f);
    test_decoder_decode_typed<ctranslate2::bfloat16_t>("bf16", 1e-1f);
    test_multistep_decode_sequence();
  }

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
