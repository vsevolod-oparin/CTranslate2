// tests/metal/kv_cache_test.mm
//
// M6.2 — KV-cache decode correctness tests.
//
// Tests the KV-cache update algorithm implemented in flash_attention_metal.mm:
//   1. commit_and_wait() flushes prior GPU writes.
//   2. CPU memcpy writes new K/V into cached_keys/values at position `offset`.
//   3. sdpa_metal attends Q over the full valid cache [0, offset+seqlen_new).
//
// Each test:
//   - Prefills a cache with sq initial tokens (offset == 0).
//   - Runs N decode steps (offset increases each step by 1).
//   - At each step verifies Metal output matches float32 CPU reference.
//
// Tests:
//   1.  float32 decode: batch=1, nh=4, nh_k=4, hd=32, prefill sq=4, 10 steps
//   2.  float32 GQA decode: batch=1, nh=4, nh_k=2, hd=16, prefill sq=4, 5 steps
//   3.  float32 batch=2 decode: batch=2, nh=2, nh_k=2, hd=16, prefill sq=3, 5 steps
//   4.  float16 decode: batch=1, nh=4, nh_k=4, hd=32, prefill sq=4, 5 steps
//   5.  bfloat16 decode: batch=1, nh=2, nh_k=2, hd=16, prefill sq=3, 3 steps
//   6.  cache contents verified: after each step, cache slot[offset] == K/V_new
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/kv_cache_test.mm \
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
//     -o kv_cache_test && ./kv_cache_test

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

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                               \
  do {                                                                   \
    if (expr) { std::printf("  PASS  %s\n", label); ++g_passed; }       \
    else       { std::printf("  FAIL  %s\n", label); ++g_failed; }      \
  } while (0)

// ---------------------------------------------------------------------------
// Metal allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(int n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// float32 CPU reference SDPA
// Layout: [batch, seqlen, num_heads, head_dim]  (interleaved heads).
// ---------------------------------------------------------------------------
static void ref_sdpa(const float* q, const float* k, const float* v, float* out,
                     int batch, int sq, int sk, int nh, int nhk, int hd,
                     float scale, bool is_causal) {
  const int q_lda  = nh  * hd;
  const int kv_lda = nhk * hd;
  std::vector<float> scores(sq * sk);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < nh; ++h) {
      const int hk = h % nhk;
      const float* q0   = q   + (b * sq * nh  + h ) * hd;
      const float* k0   = k   + (b * sk * nhk + hk) * hd;
      const float* v0   = v   + (b * sk * nhk + hk) * hd;
      float*       out0 = out + (b * sq * nh  + h ) * hd;

      for (int s = 0; s < sq; ++s) {
        for (int t = 0; t < sk; ++t) {
          float dot = 0.f;
          for (int d = 0; d < hd; ++d) {
            dot += q0[s * q_lda + d] * k0[t * kv_lda + d];
          }
          scores[s * sk + t] = scale * dot;
        }
      }
      if (is_causal) {
        for (int s = 0; s < sq; ++s) {
          for (int t = s + 1; t < sk; ++t) {
            scores[s * sk + t] = -1e9f;
          }
        }
      }
      for (int s = 0; s < sq; ++s) {
        float mx = -1e38f;
        for (int t = 0; t < sk; ++t) { if (scores[s*sk+t] > mx) mx = scores[s*sk+t]; }
        float sum = 0.f;
        for (int t = 0; t < sk; ++t) { sum += std::exp(scores[s*sk+t] - mx); }
        for (int t = 0; t < sk; ++t) { scores[s*sk+t] = std::exp(scores[s*sk+t] - mx) / sum; }
      }
      for (int s = 0; s < sq; ++s) {
        for (int d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (int t = 0; t < sk; ++t) { acc += scores[s*sk+t] * v0[t * kv_lda + d]; }
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
    if (e > err) err = e;
  }
  return err;
}

template <typename T>
static void fill_from_float(T* dst, const float* src, int n) {
  for (int i = 0; i < n; ++i) dst[i] = T(src[i]);
}
template <typename T>
static void to_float(float* dst, const T* src, int n) {
  for (int i = 0; i < n; ++i) dst[i] = float(src[i]);
}

// ---------------------------------------------------------------------------
// KV cache update helper (mirrors flash_attention_metal.mm logic).
//
// Writes src[batch, seqlen_new, nhk, hd] into
//        dst[batch, total_cache, nhk, hd] at cache position `offset`.
// ---------------------------------------------------------------------------
template <typename T>
static void update_cache(T* dst, const T* src,
                         int batch, int total_cache, int seqlen_new,
                         int nhk, int hd, int offset) {
  const int row_elems = nhk * hd;
  for (int b = 0; b < batch; ++b) {
    T*       d = dst + (b * total_cache  + offset) * row_elems;
    const T* s = src +  b * seqlen_new * row_elems;
    std::memcpy(d, s, seqlen_new * row_elems * sizeof(T));
  }
}

// ---------------------------------------------------------------------------
// Multi-step decode test
//
// Template on T (float / ct2_f16 / ct2_bf16).
// Returns maximum error across all decode steps.
// ---------------------------------------------------------------------------
template <typename T>
static float run_kv_decode(int batch, int prefill_sq, int num_steps,
                           int nh, int nhk, int hd,
                           float scale, float seed_q, float seed_k,
                           bool verbose = false) {
  const int max_cache  = prefill_sq + num_steps + 4;  // extra room
  const int row_elems  = nhk * hd;
  const int q_elems_pf = batch * prefill_sq * nh  * hd;
  const int kv_elems_pf= batch * prefill_sq * nhk * hd;
  const int q_elems_dc = batch * 1  * nh  * hd;
  const int kv_elems_dc= batch * 1  * nhk * hd;
  const int cache_elems= batch * max_cache * nhk * hd;

  // -----------------------------------------------------------------------
  // Step 0: Prefill (offset == 0, no cache)
  // -----------------------------------------------------------------------
  std::vector<float> qf_pf(q_elems_pf), kf_pf(kv_elems_pf), vf_pf(kv_elems_pf);
  for (int i = 0; i < q_elems_pf;  ++i) qf_pf[i] = std::sin(float(i) * seed_q);
  for (int i = 0; i < kv_elems_pf; ++i) kf_pf[i] = std::cos(float(i) * seed_k);
  for (int i = 0; i < kv_elems_pf; ++i) vf_pf[i] = std::sin(float(i) * (seed_k + 0.1f));

  // Allocate Metal cache buffers (zeroed).
  T* cache_k = metal_alloc<T>(cache_elems);
  T* cache_v = metal_alloc<T>(cache_elems);
  std::memset(cache_k, 0, cache_elems * sizeof(T));
  std::memset(cache_v, 0, cache_elems * sizeof(T));

  // Populate cache slots [0..prefill_sq) with the prefill K/V.
  {
    std::vector<T> kf_pf_T(kv_elems_pf), vf_pf_T(kv_elems_pf);
    fill_from_float(kf_pf_T.data(), kf_pf.data(), kv_elems_pf);
    fill_from_float(vf_pf_T.data(), vf_pf.data(), kv_elems_pf);
    update_cache(cache_k, kf_pf_T.data(), batch, max_cache, prefill_sq, nhk, hd, 0);
    update_cache(cache_v, vf_pf_T.data(), batch, max_cache, prefill_sq, nhk, hd, 0);
  }

  // CPU cache mirrors (float32) for the reference implementation.
  std::vector<float> ref_cache_k(cache_elems, 0.f);
  std::vector<float> ref_cache_v(cache_elems, 0.f);
  for (int i = 0; i < kv_elems_pf; ++i) {
    // ref_cache_k layout: [batch, max_cache, nhk, hd]
    // kf_pf layout:       [batch, prefill_sq, nhk, hd]
    // They share the same row_elems structure; just copy the first prefill_sq rows per batch.
    for (int b = 0; b < batch; ++b) {
      std::memcpy(&ref_cache_k[b * max_cache * row_elems],
                  &kf_pf      [b * prefill_sq * row_elems],
                  prefill_sq * row_elems * sizeof(float));
      std::memcpy(&ref_cache_v[b * max_cache * row_elems],
                  &vf_pf      [b * prefill_sq * row_elems],
                  prefill_sq * row_elems * sizeof(float));
    }
  }

  // -----------------------------------------------------------------------
  // Decode steps
  // -----------------------------------------------------------------------
  float worst_err = 0.f;
  int cur_offset  = prefill_sq;

  for (int step = 0; step < num_steps; ++step) {
    // Generate new Q, K_new, V_new for this decode step.
    std::vector<float> qf(q_elems_dc), kf_new(kv_elems_dc), vf_new(kv_elems_dc);
    const float phase = float(step + 1) * 0.17f;
    for (int i = 0; i < q_elems_dc;  ++i) qf    [i] = std::sin(float(i) * (seed_q + phase));
    for (int i = 0; i < kv_elems_dc; ++i) kf_new[i] = std::cos(float(i) * (seed_k + phase));
    for (int i = 0; i < kv_elems_dc; ++i) vf_new[i] = std::sin(float(i) * (seed_k + phase + 0.2f));

    // Allocate Metal buffers for this step.
    T* q_m   = metal_alloc<T>(q_elems_dc);
    T* k_m   = metal_alloc<T>(kv_elems_dc);
    T* v_m   = metal_alloc<T>(kv_elems_dc);
    T* out_m = metal_alloc<T>(q_elems_dc);

    fill_from_float(q_m,   qf.data(),     q_elems_dc);
    fill_from_float(k_m,   kf_new.data(), kv_elems_dc);
    fill_from_float(v_m,   vf_new.data(), kv_elems_dc);

    // ---- Simulate flash_attention_metal.mm KV-cache update ----
    // 1. GPU flush (commit_and_wait simulates the flush of prior GPU writes).
    metal::commit_and_wait();
    // 2. CPU memcpy: write new K/V into cache at cur_offset.
    update_cache(cache_k, k_m, batch, max_cache, 1, nhk, hd, cur_offset);
    update_cache(cache_v, v_m, batch, max_cache, 1, nhk, hd, cur_offset);
    // 3. SDPA over [0, cur_offset+1).
    const int seqlen_k_eff = cur_offset + 1;
    metal::sdpa_metal<T>(q_m, cache_k, cache_v, out_m,
                          batch, 1, seqlen_k_eff,
                          nh, nhk, hd, scale, /*is_causal=*/false);
    metal::commit_and_wait();

    // Read back output.
    std::vector<float> got_f(q_elems_dc);
    to_float(got_f.data(), out_m, q_elems_dc);

    // ---- CPU reference ----
    // Update ref cache with new K/V.
    for (int b = 0; b < batch; ++b) {
      std::memcpy(&ref_cache_k[(b * max_cache + cur_offset) * row_elems],
                  &kf_new      [b * row_elems],
                  row_elems * sizeof(float));
      std::memcpy(&ref_cache_v[(b * max_cache + cur_offset) * row_elems],
                  &vf_new      [b * row_elems],
                  row_elems * sizeof(float));
    }
    std::vector<float> ref_out(q_elems_dc);
    ref_sdpa(qf.data(), ref_cache_k.data(), ref_cache_v.data(), ref_out.data(),
             batch, 1, seqlen_k_eff, nh, nhk, hd, scale, /*is_causal=*/false);

    float err = max_abs_err(ref_out.data(), got_f.data(), q_elems_dc);
    if (err > worst_err) worst_err = err;
    if (verbose) {
      std::printf("    step %2d (offset=%2d, sk=%2d): err=%.2e\n",
                  step, cur_offset, seqlen_k_eff, err);
    }

    metal_free(q_m);
    metal_free(k_m);
    metal_free(v_m);
    metal_free(out_m);

    ++cur_offset;
  }

  metal_free(cache_k);
  metal_free(cache_v);

  return worst_err;
}

// ---------------------------------------------------------------------------
// Test 6: Verify cache slot contents after update.
//
// After each decode step with float32, read back cache_k[batch, offset, :, :]
// and confirm it matches K_new that was written.
// ---------------------------------------------------------------------------
static void test_cache_contents() {
  std::printf("\n--- cache contents verification (float32) ---\n");
  const int B=1, NHK=2, HD=8, MAX_CACHE=12;
  const int row_elems = NHK * HD;

  float* cache_k = metal_alloc<float>(B * MAX_CACHE * row_elems);
  float* cache_v = metal_alloc<float>(B * MAX_CACHE * row_elems);
  std::memset(cache_k, 0, B * MAX_CACHE * row_elems * sizeof(float));
  std::memset(cache_v, 0, B * MAX_CACHE * row_elems * sizeof(float));

  int all_ok = 1;
  for (int step = 0; step < 8; ++step) {
    // New K/V: distinctive values = 100 * step + element_index
    std::vector<float> k_new(B * row_elems), v_new(B * row_elems);
    for (int i = 0; i < B * row_elems; ++i) {
      k_new[i] = float(100 * (step + 1) + i);
      v_new[i] = float(200 * (step + 1) + i);
    }
    float* k_m = metal_alloc<float>(B * row_elems);
    float* v_m = metal_alloc<float>(B * row_elems);
    std::memcpy(k_m, k_new.data(), B * row_elems * sizeof(float));
    std::memcpy(v_m, v_new.data(), B * row_elems * sizeof(float));

    // Write into cache at offset = step.
    update_cache(cache_k, k_m, B, MAX_CACHE, 1, NHK, HD, step);
    update_cache(cache_v, v_m, B, MAX_CACHE, 1, NHK, HD, step);

    // Verify: cache_k[0, step, :, :] == k_new
    const float* slot_k = cache_k + step * row_elems;
    const float* slot_v = cache_v + step * row_elems;
    for (int i = 0; i < row_elems; ++i) {
      if (slot_k[i] != k_new[i] || slot_v[i] != v_new[i]) {
        all_ok = 0;
        std::printf("    MISMATCH at step=%d elem=%d: k got %.0f want %.0f\n",
                    step, i, slot_k[i], k_new[i]);
      }
    }

    // Verify: earlier slots were NOT overwritten.
    for (int prev = 0; prev < step; ++prev) {
      const float* prev_k = cache_k + prev * row_elems;
      for (int i = 0; i < row_elems; ++i) {
        float expected = float(100 * (prev + 1) + i);
        if (prev_k[i] != expected) {
          all_ok = 0;
          std::printf("    OVERWRITE at step=%d prev=%d: k got %.0f want %.0f\n",
                      step, prev, prev_k[i], expected);
        }
      }
    }

    metal_free(k_m);
    metal_free(v_m);
  }

  CHECK("cache slots written at correct offsets (8 steps)", all_ok == 1);

  metal_free(cache_k);
  metal_free(cache_v);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  std::printf("kv_cache_test — Metal M6.2 KV-cache decode\n");
  try {
    // -----------------------------------------------------------------------
    // Test 1: float32, batch=1, nh=4, nh_k=4, hd=32, prefill=4, 10 steps
    // -----------------------------------------------------------------------
    std::printf("\n--- float32 decode (batch=1, nh=4, hd=32, prefill=4, 10 steps) ---\n");
    {
      float scale = 1.f / std::sqrt(32.f);
      float err = run_kv_decode<float>(
          /*batch=*/1, /*prefill_sq=*/4, /*num_steps=*/10,
          /*nh=*/4, /*nhk=*/4, /*hd=*/32,
          scale, /*seed_q=*/0.3f, /*seed_k=*/0.2f, /*verbose=*/true);
      std::printf("  max err across 10 steps = %.2e\n", err);
      CHECK("float32 decode 10 steps: max err < 1e-4", err < 1e-4f);
    }

    // -----------------------------------------------------------------------
    // Test 2: float32 GQA, nh=4, nh_k=2, prefill=4, 5 steps
    // -----------------------------------------------------------------------
    std::printf("\n--- float32 GQA decode (nh=4, nh_k=2, hd=16, prefill=4, 5 steps) ---\n");
    {
      float scale = 1.f / std::sqrt(16.f);
      float err = run_kv_decode<float>(
          /*batch=*/1, /*prefill_sq=*/4, /*num_steps=*/5,
          /*nh=*/4, /*nhk=*/2, /*hd=*/16,
          scale, /*seed_q=*/0.4f, /*seed_k=*/0.25f, /*verbose=*/true);
      std::printf("  max err across 5 steps = %.2e\n", err);
      CHECK("float32 GQA decode 5 steps: max err < 1e-4", err < 1e-4f);
    }

    // -----------------------------------------------------------------------
    // Test 3: float32 batch=2, prefill=3, 5 steps
    // -----------------------------------------------------------------------
    std::printf("\n--- float32 batch=2 decode (nh=2, hd=16, prefill=3, 5 steps) ---\n");
    {
      float scale = 1.f / std::sqrt(16.f);
      float err = run_kv_decode<float>(
          /*batch=*/2, /*prefill_sq=*/3, /*num_steps=*/5,
          /*nh=*/2, /*nhk=*/2, /*hd=*/16,
          scale, /*seed_q=*/0.35f, /*seed_k=*/0.15f, /*verbose=*/true);
      std::printf("  max err across 5 steps = %.2e\n", err);
      CHECK("float32 batch=2 decode 5 steps: max err < 1e-4", err < 1e-4f);
    }

    // -----------------------------------------------------------------------
    // Test 4: float16 decode, 5 steps
    // -----------------------------------------------------------------------
    std::printf("\n--- float16 decode (batch=1, nh=4, hd=32, prefill=4, 5 steps) ---\n");
    {
      float scale = 1.f / std::sqrt(32.f);
      float err = run_kv_decode<ct2_f16>(
          /*batch=*/1, /*prefill_sq=*/4, /*num_steps=*/5,
          /*nh=*/4, /*nhk=*/4, /*hd=*/32,
          scale, /*seed_q=*/0.3f, /*seed_k=*/0.2f, /*verbose=*/true);
      std::printf("  max err across 5 steps = %.2e\n", err);
      CHECK("float16 decode 5 steps: max err < 0.05", err < 0.05f);
    }

    // -----------------------------------------------------------------------
    // Test 5: bfloat16 decode, 3 steps
    // -----------------------------------------------------------------------
    std::printf("\n--- bfloat16 decode (batch=1, nh=2, hd=16, prefill=3, 3 steps) ---\n");
    {
      float scale = 1.f / std::sqrt(16.f);
      float err = run_kv_decode<ct2_bf16>(
          /*batch=*/1, /*prefill_sq=*/3, /*num_steps=*/3,
          /*nh=*/2, /*nhk=*/2, /*hd=*/16,
          scale, /*seed_q=*/0.25f, /*seed_k=*/0.18f, /*verbose=*/true);
      std::printf("  max err across 3 steps = %.2e\n", err);
      CHECK("bfloat16 decode 3 steps: max err < 0.1", err < 0.1f);
    }

    // -----------------------------------------------------------------------
    // Test 6: cache slot contents verification
    // -----------------------------------------------------------------------
    test_cache_contents();

  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }

  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed ? 1 : 0;
}
