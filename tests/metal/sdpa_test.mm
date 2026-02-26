// tests/metal/sdpa_test.mm
//
// M6.1 — Correctness tests for metal::sdpa_metal<T>.
//
// Compares Metal SDPA output against a float32 CPU reference for all three
// floating-point types and various configurations.
//
// Tests:
//   1.  float32, non-causal: batch=1, sq=4, sk=4, heads=1, head_dim=8
//   2.  float32, causal:     batch=1, sq=4, sk=4, heads=1, head_dim=8
//   3.  float32, multi-head: batch=2, sq=3, sk=3, heads=2, head_dim=4
//   4.  float32, GQA:        batch=1, sq=4, sk=4, heads=4, num_heads_k=2, head_dim=4
//   5.  float16, non-causal: batch=1, sq=4, sk=4, heads=1, head_dim=8
//   6.  float16, causal:     batch=1, sq=4, sk=4, heads=1, head_dim=8
//   7.  bfloat16, non-causal: batch=1, sq=4, sk=4, heads=1, head_dim=8
//   8.  bfloat16, causal:     batch=1, sq=4, sk=4, heads=1, head_dim=8
//
// Additional tests (M6 review items 4.2 / 4.3):
//   9.  float32, cross-attention: batch=1, sq=4, sk=16, heads=4, nhk=4, hd=64,
//       is_causal=false  — sq != sk, exercises encoder-decoder (Whisper/NLLB) path
//   10. float32, decode sq=1:  batch=1, sq=1, sk=16, heads=4, nhk=4, hd=64
//   11. float16, decode sq=1:  batch=1, sq=1, sk=16, heads=4, nhk=4, hd=64
//   12. bfloat16, decode sq=1: batch=1, sq=1, sk=16, heads=4, nhk=4, hd=64
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/sdpa_test.mm \
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
//     -o sdpa_test && ./sdpa_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
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

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                               \
  do {                                                                   \
    if (expr) { std::printf("  PASS  %s\n", label); ++g_passed; }       \
    else       { std::printf("  FAIL  %s\n", label); ++g_failed; }      \
  } while (0)

template <typename T>
static T* metal_alloc(int n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// float32 CPU reference for SDPA
// ---------------------------------------------------------------------------
//
// Q/K/V layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
// Row stride for Q:    q_lda  = num_heads * head_dim
// Row stride for K/V:  kv_lda = num_heads_k * head_dim
//
// output: same layout as Q.
static void ref_sdpa(const float* q, const float* k, const float* v, float* out,
                     int batch, int sq, int sk, int nh, int nhk, int hd,
                     float scale, bool is_causal) {
  const int q_lda  = nh  * hd;
  const int kv_lda = nhk * hd;
  std::vector<float> scores(sq * sk);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < nh; ++h) {
      const int hk = h % nhk;
      const float* q0  = q   + (b * sq * nh  + h ) * hd;
      const float* k0  = k   + (b * sk * nhk + hk) * hd;
      const float* v0  = v   + (b * sk * nhk + hk) * hd;
      float*       out0 = out + (b * sq * nh  + h ) * hd;

      // scores[s, t] = scale * dot(Q[s], K[t])
      for (int s = 0; s < sq; ++s) {
        for (int t = 0; t < sk; ++t) {
          float dot = 0.f;
          for (int d = 0; d < hd; ++d) {
            dot += q0[s * q_lda + d] * k0[t * kv_lda + d];
          }
          scores[s * sk + t] = scale * dot;
        }
      }

      // Causal mask: col > row → -1e9
      if (is_causal) {
        for (int s = 0; s < sq; ++s) {
          for (int t = s + 1; t < sk; ++t) {
            scores[s * sk + t] = -1e9f;
          }
        }
      }

      // Softmax over each row
      for (int s = 0; s < sq; ++s) {
        float mx = -1e38f;
        for (int t = 0; t < sk; ++t) {
          if (scores[s*sk+t] > mx) {
            mx = scores[s*sk+t];
          }
        }
        float sum = 0.f;
        for (int t = 0; t < sk; ++t) {
          sum += std::exp(scores[s*sk+t] - mx);
        }
        for (int t = 0; t < sk; ++t) {
          scores[s*sk+t] = std::exp(scores[s*sk+t] - mx) / sum;
        }
      }

      // out[s, d] = sum_t(scores[s,t] * V[t, d])
      for (int s = 0; s < sq; ++s) {
        for (int d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (int t = 0; t < sk; ++t) {
            acc += scores[s*sk+t] * v0[t * kv_lda + d];
          }
          out0[s * q_lda + d] = acc;
        }
      }
    }
  }
}

static float max_abs_err(const float* ref, const float* got, int n) {
  float err = 0.f;
  for (int i = 0; i < n; ++i) {
    float e = std::abs(ref[i] - got[i]);
    if (e > err) {
      err = e;
    }
  }
  return err;
}

// ---------------------------------------------------------------------------
// Template test driver
// ---------------------------------------------------------------------------

// Fill a Metal buffer with values from a float32 source (converts T).
template <typename T>
static void fill_from_float(T* dst, const float* src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = T(src[i]);
  }
}

// Convert Metal output T buffer to float32 for comparison.
template <typename T>
static void to_float(float* dst, const T* src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = float(src[i]);
  }
}

// Run one SDPA test: allocate metal buffers, fill, call sdpa_metal, compare.
template <typename T>
static float run_sdpa(const float* qf, const float* kf, const float* vf,
                      int batch, int sq, int sk, int nh, int nhk, int hd,
                      float scale, bool causal) {
  const int q_elems   = batch * sq * nh  * hd;
  const int kv_elems  = batch * sk * nhk * hd;
  const int out_elems = q_elems;

  T* q_m   = metal_alloc<T>(q_elems);
  T* k_m   = metal_alloc<T>(kv_elems);
  T* v_m   = metal_alloc<T>(kv_elems);
  T* out_m = metal_alloc<T>(out_elems);

  fill_from_float(q_m,   qf, q_elems);
  fill_from_float(k_m,   kf, kv_elems);
  fill_from_float(v_m,   vf, kv_elems);

  metal::sdpa_metal<T>(q_m, k_m, v_m, out_m,
                        batch, sq, sk, nh, nhk, hd, scale, causal);
  // Flush deferred CB (FP32/FP16 path is encode-only; BF16 is already sync'd).
  metal::commit_and_wait();

  std::vector<float> got_f(out_elems);
  to_float(got_f.data(), out_m, out_elems);

  metal_free(q_m);
  metal_free(k_m);
  metal_free(v_m);
  metal_free(out_m);

  // Compute float32 reference
  std::vector<float> ref_out(out_elems);
  ref_sdpa(qf, kf, vf, ref_out.data(), batch, sq, sk, nh, nhk, hd, scale, causal);

  return max_abs_err(ref_out.data(), got_f.data(), out_elems);
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

static void test_float32() {
  std::printf("\n--- sdpa float32 ---\n");
  const float scale = 1.f / std::sqrt(8.f);

  // Shared input: batch=1, sq=4, sk=4, heads=1, head_dim=8
  const int B=1, SQ=4, SK=4, NH=1, NHK=1, HD=8;
  const int q_n = B*SQ*NH*HD, kv_n = B*SK*NHK*HD;
  std::vector<float> qf(q_n), kf(kv_n), vf(kv_n);
  // Deterministic: q[i] = sin(i*0.3), k[i] = cos(i*0.2), v[i] = (i % 8) * 0.1
  for (int i = 0; i < q_n;  ++i) { qf[i] = std::sin(float(i) * 0.3f); }
  for (int i = 0; i < kv_n; ++i) { kf[i] = std::cos(float(i) * 0.2f); }
  for (int i = 0; i < kv_n; ++i) { vf[i] = float(i % 8) * 0.1f; }

  // Test 1: non-causal
  float err = run_sdpa<float>(qf.data(), kf.data(), vf.data(),
                               B, SQ, SK, NH, NHK, HD, scale, /*causal=*/false);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("float32 non-causal sq=4 sk=4 h=1 hd=8: err < 1e-4", err < 1e-4f);

  // Test 2: causal
  err = run_sdpa<float>(qf.data(), kf.data(), vf.data(),
                         B, SQ, SK, NH, NHK, HD, scale, /*causal=*/true);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("float32 causal sq=4 sk=4 h=1 hd=8: err < 1e-4", err < 1e-4f);

  // Test 3: multi-head, multi-batch
  {
    const int B2=2, SQ2=3, SK2=3, NH2=2, NHK2=2, HD2=4;
    const float scale2 = 1.f / std::sqrt(4.f);
    const int q2 = B2*SQ2*NH2*HD2, kv2 = B2*SK2*NHK2*HD2;
    std::vector<float> q2f(q2), k2f(kv2), v2f(kv2);
    for (int i = 0; i < q2;  ++i) { q2f[i] = std::sin(float(i+1) * 0.4f); }
    for (int i = 0; i < kv2; ++i) { k2f[i] = std::cos(float(i+1) * 0.15f); }
    for (int i = 0; i < kv2; ++i) { v2f[i] = float((i*3) % 7) * 0.2f; }
    err = run_sdpa<float>(q2f.data(), k2f.data(), v2f.data(),
                           B2, SQ2, SK2, NH2, NHK2, HD2, scale2, /*causal=*/true);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("float32 causal batch=2 sq=3 sk=3 h=2 hd=4: err < 1e-4", err < 1e-4f);
  }

  // Test 4: grouped query attention (GQA): 4 query heads, 2 KV heads
  {
    const int B3=1, SQ3=4, SK3=4, NH3=4, NHK3=2, HD3=4;
    const float scale3 = 1.f / std::sqrt(4.f);
    const int q3 = B3*SQ3*NH3*HD3, kv3 = B3*SK3*NHK3*HD3;
    std::vector<float> q3f(q3), k3f(kv3), v3f(kv3);
    for (int i = 0; i < q3;  ++i) { q3f[i] = std::cos(float(i) * 0.25f); }
    for (int i = 0; i < kv3; ++i) { k3f[i] = std::sin(float(i) * 0.35f); }
    for (int i = 0; i < kv3; ++i) { v3f[i] = float((i % 5) + 1) * 0.15f; }
    err = run_sdpa<float>(q3f.data(), k3f.data(), v3f.data(),
                           B3, SQ3, SK3, NH3, NHK3, HD3, scale3, /*causal=*/false);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("float32 non-causal GQA nh=4 nhk=2 hd=4: err < 1e-4", err < 1e-4f);
  }
}

static void test_float16() {
  std::printf("\n--- sdpa float16 ---\n");
  const int B=1, SQ=4, SK=4, NH=1, NHK=1, HD=8;
  const float scale = 1.f / std::sqrt(8.f);
  const int q_n = B*SQ*NH*HD, kv_n = B*SK*NHK*HD;
  std::vector<float> qf(q_n), kf(kv_n), vf(kv_n);
  for (int i = 0; i < q_n;  ++i) { qf[i] = std::sin(float(i) * 0.3f); }
  for (int i = 0; i < kv_n; ++i) { kf[i] = std::cos(float(i) * 0.2f); }
  for (int i = 0; i < kv_n; ++i) { vf[i] = float(i % 8) * 0.1f; }

  // Test 5: non-causal
  float err = run_sdpa<ct2_f16>(qf.data(), kf.data(), vf.data(),
                                  B, SQ, SK, NH, NHK, HD, scale, false);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("float16 non-causal sq=4 sk=4 h=1 hd=8: err < 0.05", err < 0.05f);

  // Test 6: causal
  err = run_sdpa<ct2_f16>(qf.data(), kf.data(), vf.data(),
                            B, SQ, SK, NH, NHK, HD, scale, true);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("float16 causal sq=4 sk=4 h=1 hd=8: err < 0.05", err < 0.05f);
}

static void test_bfloat16() {
  std::printf("\n--- sdpa bfloat16 ---\n");
  const int B=1, SQ=4, SK=4, NH=1, NHK=1, HD=8;
  const float scale = 1.f / std::sqrt(8.f);
  const int q_n = B*SQ*NH*HD, kv_n = B*SK*NHK*HD;
  std::vector<float> qf(q_n), kf(kv_n), vf(kv_n);
  for (int i = 0; i < q_n;  ++i) { qf[i] = std::sin(float(i) * 0.3f); }
  for (int i = 0; i < kv_n; ++i) { kf[i] = std::cos(float(i) * 0.2f); }
  for (int i = 0; i < kv_n; ++i) { vf[i] = float(i % 8) * 0.1f; }

  // Test 7: non-causal
  float err = run_sdpa<ct2_bf16>(qf.data(), kf.data(), vf.data(),
                                   B, SQ, SK, NH, NHK, HD, scale, false);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("bfloat16 non-causal sq=4 sk=4 h=1 hd=8: err < 0.1", err < 0.1f);

  // Test 8: causal
  err = run_sdpa<ct2_bf16>(qf.data(), kf.data(), vf.data(),
                             B, SQ, SK, NH, NHK, HD, scale, true);
  std::printf("  max abs err = %.2e\n", err);
  CHECK("bfloat16 causal sq=4 sk=4 h=1 hd=8: err < 0.1", err < 0.1f);
}

// ---------------------------------------------------------------------------
// Additional tests: cross-attention (sq != sk) and decode (sq == 1).
//
// These address M6 review items 4.2 and 4.3:
//   4.2 — no test for is_causal=false with sq != sk (encoder-decoder attention)
//   4.3 — dedicated sdpa_test.mm (this file)
// ---------------------------------------------------------------------------

static void test_cross_attention_and_decode() {
  std::printf("\n--- cross-attention (sq!=sk) and decode (sq=1) ---\n");

  // Generate shared float32 data for a larger shape.
  // Q: [1, sq=4, nh=4, hd=64],  K/V: [1, sk=16, nhk=4, hd=64]
  const int B=1, SQ=4, SK=16, NH=4, NHK=4, HD=64;
  const float scale = 1.f / std::sqrt(float(HD));
  const int q_n  = B*SQ*NH*HD;
  const int kv_n = B*SK*NHK*HD;
  std::vector<float> qf(q_n), kf(kv_n), vf(kv_n);
  for (int i = 0; i < q_n;  ++i) qf[i] = std::sin(float(i+1) * 0.17f);
  for (int i = 0; i < kv_n; ++i) kf[i] = std::cos(float(i+1) * 0.13f);
  for (int i = 0; i < kv_n; ++i) vf[i] = std::sin(float(i+1) * 0.09f);

  // Test 9: cross-attention sq=4, sk=16, is_causal=false.
  // All sk=16 key positions are visible to all sq=4 query positions.
  // This tests that dispatch_causal_mask is NOT applied when is_causal=false,
  // and that sq != sk does not corrupt index arithmetic.
  {
    float err = run_sdpa<float>(qf.data(), kf.data(), vf.data(),
                                 B, SQ, SK, NH, NHK, HD, scale, /*causal=*/false);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("float32 cross-attention sq=4 sk=16 nh=4 hd=64 is_causal=false: err < 1e-4",
          err < 1e-4f);
  }

  // Tests 10-12: decode (sq=1, is_causal=false) — mirrors KV-cache decode path.
  // Q: [1, sq=1, nh=4, hd=64],  K/V: [1, sk=16, nhk=4, hd=64]
  {
    const int SQ1=1;
    const int q1  = B*SQ1*NH*HD;
    std::vector<float> q1f(q1);
    for (int i = 0; i < q1; ++i) q1f[i] = std::cos(float(i+1) * 0.21f);

    float err = run_sdpa<float>(q1f.data(), kf.data(), vf.data(),
                                 B, SQ1, SK, NH, NHK, HD, scale, /*causal=*/false);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("float32 decode sq=1 sk=16 nh=4 hd=64 is_causal=false: err < 1e-4", err < 1e-4f);

    err = run_sdpa<ct2_f16>(q1f.data(), kf.data(), vf.data(),
                              B, SQ1, SK, NH, NHK, HD, scale, /*causal=*/false);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("float16 decode sq=1 sk=16 nh=4 hd=64 is_causal=false: err < 0.05", err < 0.05f);

    err = run_sdpa<ct2_bf16>(q1f.data(), kf.data(), vf.data(),
                               B, SQ1, SK, NH, NHK, HD, scale, /*causal=*/false);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("bfloat16 decode sq=1 sk=16 nh=4 hd=64 is_causal=false: err < 0.1", err < 0.1f);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  std::printf("sdpa_test — Metal SDPA M6.1\n");
  try {
    test_float32();
    test_float16();
    test_bfloat16();
    test_cross_attention_and_decode();
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed ? 1 : 0;
}
