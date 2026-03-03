// tests/metal/m81_test.mm
//
// M8.1 — Transformer Encoder Layer end-to-end integration test.
//
// Tests the complete encoder layer pipeline using Metal primitives and ops.
// Each stage is tested independently (unit) and then chained into the full
// forward pass (integration).  All results are compared against a float32
// CPU reference.
//
// Architecture under test (Pre-LayerNorm, ReLU FFN):
//
//   x --+--[LayerNorm]--[Q proj]--\
//       |              [K proj]---> SDPA --[out proj]---> + --+-- h1
//       |              [V proj]--/                         ^  |
//       +--------------------------------------------------|  |
//   h1 --+--[LayerNorm]--[FFN W1]--[ReLU]--[FFN W2]--> + --+
//        +--------------------------------------------------^
//
// Parameters: batch=1, seq=4 (T), dim=16 (D), heads=2 (NH), head_dim=8 (HD),
//             ffn_dim=32 (FFN), scale=1/sqrt(8), is_causal=false
//
// Tests:
//   1.  LayerNorm Metal vs CPU (4 rows × 16 dims, with gamma/beta)
//   2.  Linear projection GEMM Metal vs CPU ([4×16] × [16×16]^T = [4×16])
//   3.  ReLU Metal vs CPU ([4×32])
//   4.  Residual Add Metal vs CPU ([4×16] + [4×16])
//   5.  SDPA Metal vs CPU (non-causal, float32, batch=1 sq=4 sk=4 nh=2 hd=8)
//   6.  Full encoder layer: Metal vs CPU, max abs diff < 2e-4
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/m81_test.mm \
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
//     -o m81_test && ./m81_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
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
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(
      static_cast<std::size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

// Copy host data into a Metal buffer (allocate + copy).
template <typename T>
static T* metal_from(const std::vector<float>& host) {
  T* buf = metal_alloc<T>(static_cast<dim_t>(host.size()));
  for (std::size_t i = 0; i < host.size(); ++i)
    buf[i] = T(host[i]);
  return buf;
}

// Read a Metal buffer into host std::vector<float>.
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

// Random number generator with fixed seed for reproducibility.
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

// Layer normalisation: y[i,j] = (x[i,j] - mean_i) / sqrt(var_i + eps) * g[j] + b[j]
static std::vector<float> ref_layer_norm(
    const std::vector<float>& x,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    dim_t outer, dim_t D, float eps = 1e-5f)
{
  std::vector<float> y(static_cast<std::size_t>(outer * D));
  for (dim_t i = 0; i < outer; ++i) {
    float mean = 0.f, var = 0.f;
    for (dim_t j = 0; j < D; ++j) mean += x[static_cast<std::size_t>(i * D + j)];
    mean /= static_cast<float>(D);
    for (dim_t j = 0; j < D; ++j) {
      float d = x[static_cast<std::size_t>(i * D + j)] - mean;
      var += d * d;
    }
    var /= static_cast<float>(D);
    float inv = 1.f / std::sqrt(var + eps);
    for (dim_t j = 0; j < D; ++j) {
      float xn = (x[static_cast<std::size_t>(i * D + j)] - mean) * inv;
      y[static_cast<std::size_t>(i * D + j)] = xn * gamma[static_cast<std::size_t>(j)]
                                               + beta[static_cast<std::size_t>(j)];
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
        s += A[static_cast<std::size_t>(i * k + l)] * B[static_cast<std::size_t>(j * k + l)];
      C[static_cast<std::size_t>(i * n + j)] = s;
    }
  return C;
}

// Element-wise ReLU.
static std::vector<float> ref_relu(const std::vector<float>& x) {
  std::vector<float> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::max(0.f, x[i]);
  return y;
}

// Element-wise add.
static std::vector<float> ref_add(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
  return c;
}

// Scaled dot-product attention (no masking).
// q, k, v: [batch, T, NH, HD] interleaved
static std::vector<float> ref_sdpa(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    dim_t B, dim_t T, dim_t NH, dim_t HD, float scale)
{
  std::vector<float> out(static_cast<std::size_t>(B * T * NH * HD), 0.f);
  for (dim_t b = 0; b < B; ++b) {
    for (dim_t h = 0; h < NH; ++h) {
      for (dim_t i = 0; i < T; ++i) {
        // Compute attention scores for row i.
        std::vector<float> scores(static_cast<std::size_t>(T));
        for (dim_t j = 0; j < T; ++j) {
          float dot = 0.f;
          for (dim_t d = 0; d < HD; ++d) {
            auto qi = static_cast<std::size_t>((b*T + i)*NH*HD + h*HD + d);
            auto kj = static_cast<std::size_t>((b*T + j)*NH*HD + h*HD + d);
            dot += q[qi] * k[kj];
          }
          scores[static_cast<std::size_t>(j)] = dot * scale;
        }
        // Softmax.
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sums = 0.f;
        for (auto& s : scores) { s = std::exp(s - maxs); sums += s; }
        for (auto& s : scores) s /= sums;
        // Weighted sum of V.
        for (dim_t d = 0; d < HD; ++d) {
          float acc = 0.f;
          for (dim_t j = 0; j < T; ++j)
            acc += scores[static_cast<std::size_t>(j)]
                   * v[static_cast<std::size_t>((b*T + j)*NH*HD + h*HD + d)];
          out[static_cast<std::size_t>((b*T + i)*NH*HD + h*HD + d)] = acc;
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Metal primitives helper: flush + copy to host
// ---------------------------------------------------------------------------
static std::vector<float> metal_gemm(
    const std::vector<float>& A_host, dim_t m, dim_t k,
    const std::vector<float>& B_host, dim_t n)
{
  // C = A × B^T  (matches ref_gemm_bt)
  float* A = metal_from<float>(A_host);
  float* B = metal_from<float>(B_host);
  float* C = metal_alloc<float>(m * n);

  primitives<Device::METAL>::gemm<float, float>(
      false, false,     // a_is_packed, b_is_packed
      false, true,      // transpose_a, transpose_b (B^T)
      m, n, k,
      1.f,
      A, k,             // A, lda
      B, k,             // B, ldb  (B stored as [n×k], transposed)
      0.f,
      C, n);            // C, ldc

  auto result = metal_to_host(C, m * n);
  metal_free(A); metal_free(B); metal_free(C);
  return result;
}

// ---------------------------------------------------------------------------
// Test 1 — LayerNorm Metal vs CPU
// ---------------------------------------------------------------------------
static void test_layer_norm() {
  std::printf("\n--- Test 1: LayerNorm Metal vs CPU ---\n");

  const dim_t outer = 4, D = 16;
  auto x_h     = rand_vec(static_cast<std::size_t>(outer * D));
  auto gamma_h = rand_vec(static_cast<std::size_t>(D), 0.5f, 1.5f);
  auto beta_h  = rand_vec(static_cast<std::size_t>(D), -0.1f, 0.1f);

  // CPU reference.
  auto cpu = ref_layer_norm(x_h, gamma_h, beta_h, outer, D);

  // Metal.
  float* x_m     = metal_from<float>(x_h);
  float* gamma_m = metal_from<float>(gamma_h);
  float* beta_m  = metal_from<float>(beta_h);
  float* y_m     = metal_alloc<float>(outer * D);

  metal::layer_norm_metal<float>(x_m, gamma_m, beta_m, y_m, outer, D, 1e-5f);
  auto metal_out = metal_to_host(y_m, outer * D);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("layer_norm Metal vs CPU: max_abs_diff < 1e-4", err < 1e-4f);

  metal_free(x_m); metal_free(gamma_m); metal_free(beta_m); metal_free(y_m);
}

// ---------------------------------------------------------------------------
// Test 2 — GEMM (linear projection) Metal vs CPU
// ---------------------------------------------------------------------------
static void test_gemm() {
  std::printf("\n--- Test 2: GEMM Metal vs CPU ---\n");

  const dim_t m = 4, k = 16, n = 16;
  auto A_h = rand_vec(static_cast<std::size_t>(m * k));
  auto B_h = rand_vec(static_cast<std::size_t>(n * k));  // stored as [n×k]

  auto cpu   = ref_gemm_bt(A_h, m, k, B_h, n);
  auto metal = metal_gemm(A_h, m, k, B_h, n);

  float err = max_abs_diff(cpu, metal);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("gemm [4×16] × [16×16]^T Metal vs CPU: max_abs_diff < 1e-4", err < 1e-4f);
}

// ---------------------------------------------------------------------------
// Test 3 — ReLU Metal vs CPU
// ---------------------------------------------------------------------------
static void test_relu() {
  std::printf("\n--- Test 3: ReLU Metal vs CPU ---\n");

  const dim_t N = 128;
  auto x_h = rand_vec(static_cast<std::size_t>(N), -2.f, 2.f);
  auto cpu  = ref_relu(x_h);

  float* x_m = metal_from<float>(x_h);
  float* y_m = metal_alloc<float>(N);
  primitives<Device::METAL>::relu<float>(x_m, y_m, N);
  auto metal_out = metal_to_host(y_m, N);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("relu [128] Metal vs CPU: max_abs_diff == 0", err == 0.f);

  metal_free(x_m); metal_free(y_m);
}

// ---------------------------------------------------------------------------
// Test 4 — Residual Add Metal vs CPU
// ---------------------------------------------------------------------------
static void test_residual_add() {
  std::printf("\n--- Test 4: Residual Add Metal vs CPU ---\n");

  const dim_t N = 64;
  auto a_h = rand_vec(static_cast<std::size_t>(N));
  auto b_h = rand_vec(static_cast<std::size_t>(N));
  auto cpu  = ref_add(a_h, b_h);

  float* a_m = metal_from<float>(a_h);
  float* b_m = metal_from<float>(b_h);
  float* c_m = metal_alloc<float>(N);
  primitives<Device::METAL>::add<float>(a_m, b_m, c_m, N);
  auto metal_out = metal_to_host(c_m, N);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("add [64] Metal vs CPU: max_abs_diff == 0", err == 0.f);

  metal_free(a_m); metal_free(b_m); metal_free(c_m);
}

// ---------------------------------------------------------------------------
// Test 5 — SDPA Metal vs CPU
// ---------------------------------------------------------------------------
static void test_sdpa() {
  std::printf("\n--- Test 5: SDPA Metal vs CPU ---\n");

  const dim_t B = 1, T = 4, NH = 2, HD = 8;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));

  auto q_h = rand_vec(static_cast<std::size_t>(B * T * NH * HD));
  auto k_h = rand_vec(static_cast<std::size_t>(B * T * NH * HD));
  auto v_h = rand_vec(static_cast<std::size_t>(B * T * NH * HD));

  auto cpu = ref_sdpa(q_h, k_h, v_h, B, T, NH, HD, scale);

  float* q_m   = metal_from<float>(q_h);
  float* k_m   = metal_from<float>(k_h);
  float* v_m   = metal_from<float>(v_h);
  float* out_m = metal_alloc<float>(B * T * NH * HD);

  metal::sdpa_metal<float>(q_m, k_m, v_m, out_m,
                            B, T, T, NH, NH, HD, scale, false);
  auto metal_out = metal_to_host(out_m, B * T * NH * HD);

  float err = max_abs_diff(cpu, metal_out);
  std::printf("  max_abs_diff = %.2e\n", static_cast<double>(err));
  CHECK("sdpa [1,4,2,8] Metal vs CPU: max_abs_diff < 1e-4", err < 1e-4f);

  metal_free(q_m); metal_free(k_m); metal_free(v_m); metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 6 — Full Encoder Layer forward pass: Metal vs CPU
// ---------------------------------------------------------------------------
//
// Pipeline:
//   x  →  norm1  →  {Q,K,V proj}  →  SDPA  →  out_proj  →  + x  →  h1
//   h1 →  norm2  →  ffn_w1  →  relu  →  ffn_w2  →  + h1  →  output

static void test_full_encoder_layer() {
  std::printf("\n--- Test 6: Full Encoder Layer Metal vs CPU ---\n");

  // Hyperparameters
  const dim_t B = 1, T = 4, D = 16, NH = 2, HD = 8, FFN = 32;
  const dim_t N_in = B * T * D;           // 64
  const dim_t N_qkv = B * T * NH * HD;    // 64 = T×D
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps   = 1e-5f;

  // Random weights (all stored as [output × input] = [n × k])
  auto gamma1_h = rand_vec(static_cast<std::size_t>(D),   0.5f, 1.5f);
  auto beta1_h  = rand_vec(static_cast<std::size_t>(D),  -0.1f, 0.1f);
  auto W_q_h    = rand_vec(static_cast<std::size_t>(D*D), -0.5f, 0.5f);  // [D, D]
  auto W_k_h    = rand_vec(static_cast<std::size_t>(D*D), -0.5f, 0.5f);
  auto W_v_h    = rand_vec(static_cast<std::size_t>(D*D), -0.5f, 0.5f);
  auto W_o_h    = rand_vec(static_cast<std::size_t>(D*D), -0.5f, 0.5f);  // [D, D]
  auto gamma2_h = rand_vec(static_cast<std::size_t>(D),   0.5f, 1.5f);
  auto beta2_h  = rand_vec(static_cast<std::size_t>(D),  -0.1f, 0.1f);
  auto W1_h     = rand_vec(static_cast<std::size_t>(FFN*D), -0.5f, 0.5f); // [FFN, D]
  auto W2_h     = rand_vec(static_cast<std::size_t>(D*FFN), -0.5f, 0.5f); // [D, FFN]

  // Random input.
  auto x_h = rand_vec(static_cast<std::size_t>(N_in));

  // ==========================================================================
  // CPU forward pass
  // ==========================================================================
  auto norm1    = ref_layer_norm(x_h, gamma1_h, beta1_h, T, D, eps);
  auto Q_cpu    = ref_gemm_bt(norm1, T, D, W_q_h, D);   // [T, D]
  auto K_cpu    = ref_gemm_bt(norm1, T, D, W_k_h, D);
  auto V_cpu    = ref_gemm_bt(norm1, T, D, W_v_h, D);
  auto attn_cpu = ref_sdpa(Q_cpu, K_cpu, V_cpu, B, T, NH, HD, scale);
  auto proj_cpu = ref_gemm_bt(attn_cpu, T, D, W_o_h, D);
  auto h1_cpu   = ref_add(x_h, proj_cpu);
  auto norm2    = ref_layer_norm(h1_cpu, gamma2_h, beta2_h, T, D, eps);
  auto ffn1_cpu = ref_gemm_bt(norm2, T, D, W1_h, FFN);  // [T, FFN]
  auto ffn1_act = ref_relu(ffn1_cpu);
  auto ffn2_cpu = ref_gemm_bt(ffn1_act, T, FFN, W2_h, D); // [T, D]
  auto out_cpu  = ref_add(h1_cpu, ffn2_cpu);

  // ==========================================================================
  // Metal forward pass
  // ==========================================================================

  // Allocate and upload inputs + weights.
  float* x_m      = metal_from<float>(x_h);
  float* gamma1_m = metal_from<float>(gamma1_h);
  float* beta1_m  = metal_from<float>(beta1_h);
  float* W_q_m    = metal_from<float>(W_q_h);
  float* W_k_m    = metal_from<float>(W_k_h);
  float* W_v_m    = metal_from<float>(W_v_h);
  float* W_o_m    = metal_from<float>(W_o_h);
  float* gamma2_m = metal_from<float>(gamma2_h);
  float* beta2_m  = metal_from<float>(beta2_h);
  float* W1_m     = metal_from<float>(W1_h);
  float* W2_m     = metal_from<float>(W2_h);

  // Intermediate buffers.
  float* norm1_m  = metal_alloc<float>(T * D);
  float* Q_m      = metal_alloc<float>(N_qkv);
  float* K_m      = metal_alloc<float>(N_qkv);
  float* V_m      = metal_alloc<float>(N_qkv);
  float* attn_m   = metal_alloc<float>(N_qkv);
  float* proj_m   = metal_alloc<float>(T * D);
  float* h1_m     = metal_alloc<float>(T * D);
  float* norm2_m  = metal_alloc<float>(T * D);
  float* ffn1_m   = metal_alloc<float>(T * FFN);
  float* ffn1a_m  = metal_alloc<float>(T * FFN);
  float* ffn2_m   = metal_alloc<float>(T * D);
  float* out_m    = metal_alloc<float>(T * D);

  // Step 1: LayerNorm (pre-attention).
  metal::layer_norm_metal<float>(x_m, gamma1_m, beta1_m, norm1_m, T, D, eps);

  // Step 2: Q, K, V projections — norm1 × W^T.
  // B=W stored as [D×D]=[n×k], transpose_b=true → C = A × B^T = [T,D]×[D,D] = [T,D]
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, D, D, 1.f,
      norm1_m, D, W_q_m, D, 0.f, Q_m, D);
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, D, D, 1.f,
      norm1_m, D, W_k_m, D, 0.f, K_m, D);
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, D, D, 1.f,
      norm1_m, D, W_v_m, D, 0.f, V_m, D);

  // Step 3: SDPA (Q/K/V as [B,T,NH,HD]).
  metal::sdpa_metal<float>(Q_m, K_m, V_m, attn_m,
                            B, T, T, NH, NH, HD, scale, false);

  // Step 4: Output projection — attn × W_o^T.
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, D, D, 1.f,
      attn_m, D, W_o_m, D, 0.f, proj_m, D);

  // Step 5: Residual add h1 = x + proj.
  primitives<Device::METAL>::add<float>(x_m, proj_m, h1_m, T * D);

  // Step 6: LayerNorm (pre-FFN).
  metal::layer_norm_metal<float>(h1_m, gamma2_m, beta2_m, norm2_m, T, D, eps);

  // Step 7: FFN W1 — norm2 × W1^T, shape [T, FFN].
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, FFN, D, 1.f,
      norm2_m, D, W1_m, D, 0.f, ffn1_m, FFN);

  // Step 8: ReLU.
  primitives<Device::METAL>::relu<float>(ffn1_m, ffn1a_m, T * FFN);

  // Step 9: FFN W2 — relu_out × W2^T, shape [T, D].
  primitives<Device::METAL>::gemm<float, float>(
      false, false, false, true, T, D, FFN, 1.f,
      ffn1a_m, FFN, W2_m, FFN, 0.f, ffn2_m, D);

  // Step 10: Residual add output = h1 + ffn2.
  primitives<Device::METAL>::add<float>(h1_m, ffn2_m, out_m, T * D);

  // Read output.
  auto out_metal = metal_to_host(out_m, T * D);

  // Compare.
  float err = max_abs_diff(out_cpu, out_metal);
  std::printf("  max_abs_diff = %.2e  (output [%lld × %lld])\n",
              static_cast<double>(err), static_cast<long long>(T), static_cast<long long>(D));
  CHECK("full encoder layer Metal vs CPU: max_abs_diff < 2e-4", err < 2e-4f);

  // Also check intermediate results for diagnosis.
  {
    auto norm1_metal = metal_to_host(norm1_m, T * D);
    float e1 = max_abs_diff(norm1, norm1_metal);
    std::printf("  norm1 max_abs_diff = %.2e\n", static_cast<double>(e1));
    CHECK("encoder: norm1 Metal vs CPU < 1e-4", e1 < 1e-4f);
  }
  {
    auto q_metal = metal_to_host(Q_m, N_qkv);
    float eq = max_abs_diff(Q_cpu, q_metal);
    std::printf("  Q proj max_abs_diff = %.2e\n", static_cast<double>(eq));
    CHECK("encoder: Q projection Metal vs CPU < 1e-4", eq < 1e-4f);
  }
  {
    auto attn_metal = metal_to_host(attn_m, N_qkv);
    float ea = max_abs_diff(attn_cpu, attn_metal);
    std::printf("  SDPA max_abs_diff = %.2e\n", static_cast<double>(ea));
    CHECK("encoder: SDPA output Metal vs CPU < 1e-4", ea < 1e-4f);
  }
  {
    auto h1_metal = metal_to_host(h1_m, T * D);
    float eh = max_abs_diff(h1_cpu, h1_metal);
    std::printf("  h1 (post-attn residual) max_abs_diff = %.2e\n", static_cast<double>(eh));
    CHECK("encoder: h1 Metal vs CPU < 1e-4", eh < 1e-4f);
  }
  {
    auto ffn1_metal = metal_to_host(ffn1a_m, T * FFN);
    float ef = max_abs_diff(ffn1_act, ffn1_metal);
    std::printf("  FFN ReLU max_abs_diff = %.2e\n", static_cast<double>(ef));
    CHECK("encoder: FFN ReLU output Metal vs CPU < 1e-4", ef < 1e-4f);
  }

  // Free all Metal buffers.
  metal_free(x_m);      metal_free(gamma1_m); metal_free(beta1_m);
  metal_free(W_q_m);    metal_free(W_k_m);    metal_free(W_v_m);
  metal_free(W_o_m);    metal_free(gamma2_m); metal_free(beta2_m);
  metal_free(W1_m);     metal_free(W2_m);
  metal_free(norm1_m);  metal_free(Q_m);      metal_free(K_m);
  metal_free(V_m);      metal_free(attn_m);   metal_free(proj_m);
  metal_free(h1_m);     metal_free(norm2_m);  metal_free(ffn1_m);
  metal_free(ffn1a_m);  metal_free(ffn2_m);   metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Test 7/8 — Full Encoder Layer in fp16 / bf16
//
// Templated version: all weights/intermediates are stored in T; CPU reference
// is still float32.  Tolerances are relaxed for reduced precision.
// ---------------------------------------------------------------------------

template <typename T>
static void test_encoder_layer_typed(const char* type_name, float tol) {
  std::printf("\n--- Test: Full Encoder Layer (%s) Metal vs CPU ---\n", type_name);

  const dim_t B = 1, TT = 4, D = 16, NH = 2, HD = 8, FFN = 32;
  const dim_t N_in = B * TT * D;
  const dim_t N_qkv = B * TT * NH * HD;
  const float scale = 1.f / std::sqrt(static_cast<float>(HD));
  const float eps = 1e-5f;

  // Random weights (host float).
  auto gamma1_h = rand_vec(static_cast<std::size_t>(D),   0.5f, 1.5f);
  auto beta1_h  = rand_vec(static_cast<std::size_t>(D),  -0.1f, 0.1f);
  auto W_q_h    = rand_vec(static_cast<std::size_t>(D*D), -0.3f, 0.3f);
  auto W_k_h    = rand_vec(static_cast<std::size_t>(D*D), -0.3f, 0.3f);
  auto W_v_h    = rand_vec(static_cast<std::size_t>(D*D), -0.3f, 0.3f);
  auto W_o_h    = rand_vec(static_cast<std::size_t>(D*D), -0.3f, 0.3f);
  auto gamma2_h = rand_vec(static_cast<std::size_t>(D),   0.5f, 1.5f);
  auto beta2_h  = rand_vec(static_cast<std::size_t>(D),  -0.1f, 0.1f);
  auto W1_h     = rand_vec(static_cast<std::size_t>(FFN*D), -0.3f, 0.3f);
  auto W2_h     = rand_vec(static_cast<std::size_t>(D*FFN), -0.3f, 0.3f);
  auto x_h      = rand_vec(static_cast<std::size_t>(N_in), -0.5f, 0.5f);

  // CPU float32 reference.
  auto norm1    = ref_layer_norm(x_h, gamma1_h, beta1_h, TT, D, eps);
  auto Q_cpu    = ref_gemm_bt(norm1, TT, D, W_q_h, D);
  auto K_cpu    = ref_gemm_bt(norm1, TT, D, W_k_h, D);
  auto V_cpu    = ref_gemm_bt(norm1, TT, D, W_v_h, D);
  auto attn_cpu = ref_sdpa(Q_cpu, K_cpu, V_cpu, B, TT, NH, HD, scale);
  auto proj_cpu = ref_gemm_bt(attn_cpu, TT, D, W_o_h, D);
  auto h1_cpu   = ref_add(x_h, proj_cpu);
  auto norm2    = ref_layer_norm(h1_cpu, gamma2_h, beta2_h, TT, D, eps);
  auto ffn1_cpu = ref_gemm_bt(norm2, TT, D, W1_h, FFN);
  auto ffn1_act = ref_relu(ffn1_cpu);
  auto ffn2_cpu = ref_gemm_bt(ffn1_act, TT, FFN, W2_h, D);
  auto out_cpu  = ref_add(h1_cpu, ffn2_cpu);

  // Metal forward pass in type T.
  T* x_m      = metal_from<T>(x_h);
  T* gamma1_m = metal_from<T>(gamma1_h);
  T* beta1_m  = metal_from<T>(beta1_h);
  T* W_q_m    = metal_from<T>(W_q_h);
  T* W_k_m    = metal_from<T>(W_k_h);
  T* W_v_m    = metal_from<T>(W_v_h);
  T* W_o_m    = metal_from<T>(W_o_h);
  T* gamma2_m = metal_from<T>(gamma2_h);
  T* beta2_m  = metal_from<T>(beta2_h);
  T* W1_m     = metal_from<T>(W1_h);
  T* W2_m     = metal_from<T>(W2_h);

  T* norm1_m  = metal_alloc<T>(TT * D);
  T* Q_m      = metal_alloc<T>(N_qkv);
  T* K_m      = metal_alloc<T>(N_qkv);
  T* V_m      = metal_alloc<T>(N_qkv);
  T* attn_m   = metal_alloc<T>(N_qkv);
  T* proj_m   = metal_alloc<T>(TT * D);
  T* h1_m     = metal_alloc<T>(TT * D);
  T* norm2_m  = metal_alloc<T>(TT * D);
  T* ffn1_m   = metal_alloc<T>(TT * FFN);
  T* ffn1a_m  = metal_alloc<T>(TT * FFN);
  T* ffn2_m   = metal_alloc<T>(TT * D);
  T* out_m    = metal_alloc<T>(TT * D);

  metal::layer_norm_metal<T>(x_m, gamma1_m, beta1_m, norm1_m, TT, D, eps);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, D, D, 1.f,
      norm1_m, D, W_q_m, D, 0.f, Q_m, D);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, D, D, 1.f,
      norm1_m, D, W_k_m, D, 0.f, K_m, D);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, D, D, 1.f,
      norm1_m, D, W_v_m, D, 0.f, V_m, D);
  metal::sdpa_metal<T>(Q_m, K_m, V_m, attn_m,
                        B, TT, TT, NH, NH, HD, scale, false);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, D, D, 1.f,
      attn_m, D, W_o_m, D, 0.f, proj_m, D);
  primitives<Device::METAL>::add<T>(x_m, proj_m, h1_m, TT * D);
  metal::layer_norm_metal<T>(h1_m, gamma2_m, beta2_m, norm2_m, TT, D, eps);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, FFN, D, 1.f,
      norm2_m, D, W1_m, D, 0.f, ffn1_m, FFN);
  primitives<Device::METAL>::relu<T>(ffn1_m, ffn1a_m, TT * FFN);
  primitives<Device::METAL>::gemm<T, T>(
      false, false, false, true, TT, D, FFN, 1.f,
      ffn1a_m, FFN, W2_m, FFN, 0.f, ffn2_m, D);
  primitives<Device::METAL>::add<T>(h1_m, ffn2_m, out_m, TT * D);

  auto out_metal = metal_to_host(out_m, TT * D);
  float err = max_abs_diff(out_cpu, out_metal);
  std::printf("  max_abs_diff = %.2e  (tol = %.2e)\n",
              static_cast<double>(err), static_cast<double>(tol));

  char label[128];
  std::snprintf(label, sizeof(label),
                "full encoder layer (%s) Metal vs CPU: max_abs_diff < %.0e",
                type_name, static_cast<double>(tol));
  CHECK(label, err < tol);

  metal_free(x_m);      metal_free(gamma1_m); metal_free(beta1_m);
  metal_free(W_q_m);    metal_free(W_k_m);    metal_free(W_v_m);
  metal_free(W_o_m);    metal_free(gamma2_m); metal_free(beta2_m);
  metal_free(W1_m);     metal_free(W2_m);
  metal_free(norm1_m);  metal_free(Q_m);      metal_free(K_m);
  metal_free(V_m);      metal_free(attn_m);   metal_free(proj_m);
  metal_free(h1_m);     metal_free(norm2_m);  metal_free(ffn1_m);
  metal_free(ffn1a_m);  metal_free(ffn2_m);   metal_free(out_m);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M8.1 Transformer Encoder Layer integration tests ===\n");

  @autoreleasepool {
    test_layer_norm();
    test_gemm();
    test_relu();
    test_residual_add();
    test_sdpa();
    test_full_encoder_layer();
    test_encoder_layer_typed<ctranslate2::float16_t>("f16", 5e-2f);
    test_encoder_layer_typed<ctranslate2::bfloat16_t>("bf16", 1e-1f);
  }

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
