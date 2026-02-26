// tests/metal/m62_bench.mm
//
// M6.2 Metal vs CPU — Accuracy and performance for KV-cache decode step.
//
// Models one decode step at cache position `offset = sk - 1`:
//   • Pre-populate cache with (sk-1) random K/V tokens.
//   • Time one decode step: add the sk-th token and run SDPA over [0, sk).
//
// GPU timing covers the full decode path as implemented in flash_attention_metal.mm:
//   1. commit_and_wait()   — flush prior GPU writes (simulated; instant when CB is empty)
//   2. CPU memcpy          — write new K/V into cache (nhk * hd * sizeof(T) bytes/batch)
//   3. sdpa_metal          — encode SDPA kernel(s) into CB
//   4. commit_and_wait()   — wait for SDPA result
//
// CPU timing (reference):
//   1. memcpy              — write new K/V into CPU-side cache copy
//   2. ref_sdpa            — single-threaded float32 attention over [0, sk)
//
// Shapes cover:
//   - Standard MHA (nh=8, nhk=8, hd=64): sk = 64, 128, 256, 512, 1024, 2048
//   - GQA (nh=16, nhk=4, hd=64): sk = 256, 512, 1024
//   - Small head dim (nh=4, nhk=4, hd=32): sk = 128, 512
//
// NOTE: GPU times include the ~0.4 ms CB submission overhead.
//       In a real pipeline the CB is committed at synchronize_stream(),
//       amortising that cost across many ops.  Standalone timings are
//       conservative — actual pipeline crossover is lower than shown.
//       BF16 path uses synchronous MPSGraph per head (extra overhead at small sk).
//
// Build and run from the repository root (use -O2 for accurate timings):
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/m62_bench.mm \
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
//     -o m62_bench && ./m62_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
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
// Timing
// ---------------------------------------------------------------------------

using Clock  = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

static double now_us() {
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

static volatile float sink_f = 0.f;

template <typename Fn>
static double bench_median_us(int iters, Fn fn) {
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    sink_f = fn();
    double t1 = now_us();
    times.push_back(t1 - t0);
  }
  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// ---------------------------------------------------------------------------
// Metal alloc helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

static uint32_t rng_state = 0xCAFEBABEu;
static float next_float(float lo, float hi) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  float t = static_cast<float>(rng_state) / static_cast<float>(UINT32_MAX);
  return lo + t * (hi - lo);
}

// ---------------------------------------------------------------------------
// Accuracy helpers
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static float max_abs_err(const float* ref, const float* got, dim_t n) {
  float err = 0.f;
  for (dim_t i = 0; i < n; ++i) err = std::max(err, std::fabs(ref[i] - got[i]));
  return err;
}

// ---------------------------------------------------------------------------
// float32 CPU reference SDPA
// Layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
// ---------------------------------------------------------------------------

static void ref_sdpa(const float* q, const float* k, const float* v, float* out,
                     int batch, int sq, int sk, int nh, int nhk, int hd, float scale) {
  const int q_lda  = nh  * hd;
  const int kv_lda = nhk * hd;
  std::vector<float> scores(sq * sk);

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < nh; ++h) {
      const int hk  = h % nhk;
      const float* q0   = q   + (b * sq * nh  + h ) * hd;
      const float* k0   = k   + (b * sk * nhk + hk) * hd;
      const float* v0   = v   + (b * sk * nhk + hk) * hd;
      float*       out0 = out + (b * sq * nh  + h ) * hd;

      for (int s = 0; s < sq; ++s)
        for (int t = 0; t < sk; ++t) {
          float dot = 0.f;
          for (int d = 0; d < hd; ++d) dot += q0[s*q_lda+d] * k0[t*kv_lda+d];
          scores[s*sk+t] = scale * dot;
        }

      // is_causal=false: decode path, all past tokens are visible
      for (int s = 0; s < sq; ++s) {
        float mx = -1e38f;
        for (int t = 0; t < sk; ++t) if (scores[s*sk+t] > mx) mx = scores[s*sk+t];
        float sum = 0.f;
        for (int t = 0; t < sk; ++t) sum += std::exp(scores[s*sk+t] - mx);
        for (int t = 0; t < sk; ++t) scores[s*sk+t] = std::exp(scores[s*sk+t]-mx) / sum;
      }
      for (int s = 0; s < sq; ++s)
        for (int d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (int t = 0; t < sk; ++t) acc += scores[s*sk+t] * v0[t*kv_lda+d];
          out0[s*q_lda+d] = acc;
        }
    }
  }
}

// ---------------------------------------------------------------------------
// KV cache update (mirrors flash_attention_metal.mm offset>0 logic)
// ---------------------------------------------------------------------------

template <typename T>
static void update_cache(T* dst, const T* src,
                         int batch, int total_cache, int nhk, int hd, int offset) {
  const int row = nhk * hd;
  for (int b = 0; b < batch; ++b)
    std::memcpy(dst + (b * total_cache + offset) * row,
                src +  b               * 1       * row,
                row * sizeof(T));
}

// ---------------------------------------------------------------------------
// Decode shape descriptor
// ---------------------------------------------------------------------------

struct DecodeShape {
  int batch, sk, nh, nhk, hd;  // sq is always 1
};

// Shapes: standard MHA, GQA, and smaller head-dim configs.
static const DecodeShape SHAPES[] = {
  // Standard MHA: nh=8, nhk=8, hd=64
  {1,   64, 8, 8, 64},
  {1,  128, 8, 8, 64},
  {1,  256, 8, 8, 64},
  {1,  512, 8, 8, 64},
  {1, 1024, 8, 8, 64},
  {1, 2048, 8, 8, 64},
  // GQA: nh=16, nhk=4, hd=64
  {1,  256, 16, 4, 64},
  {1,  512, 16, 4, 64},
  {1, 1024, 16, 4, 64},
  // Small heads: nh=4, nhk=4, hd=32
  {1,  128, 4, 4, 32},
  {1,  512, 4, 4, 32},
};
static const int N_SHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

// Adaptive iters: fewer for large caches.
static int iter_count(const DecodeShape& s) {
  dim_t work = (dim_t)s.batch * s.nh * s.sk;
  if (work <= 2048)  return 200;
  if (work <= 16384) return 100;
  if (work <= 65536) return 40;
  return 20;
}

static void format_shape(char* buf, int bufsz, const DecodeShape& s) {
  if (s.nhk != s.nh)
    std::snprintf(buf, bufsz, "b%d sq1 sk%d nh%d/%d hd%d",
                  s.batch, s.sk, s.nh, s.nhk, s.hd);
  else
    std::snprintf(buf, bufsz, "b%d sq1 sk%d nh%d hd%d",
                  s.batch, s.sk, s.nh, s.hd);
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

static void print_header() {
  std::printf("  %-30s  %12s  %8s  %10s  %10s  %9s  %s\n",
              "Shape", "max_abs_err", "Status",
              "GPU (µs)", "CPU (µs)", "Speedup", "");
  std::printf("  %s\n", std::string(90, '-').c_str());
}

static void print_row(const char* shape, float abs_err, float tol,
                      double gpu_us, double cpu_us) {
  bool ok = std::isfinite(abs_err) && (abs_err <= tol);
  if (ok) ++g_pass; else ++g_fail;
  double speedup = cpu_us / gpu_us;
  std::printf("  %-30s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
              shape, abs_err, ok ? "PASS" : "FAIL",
              gpu_us, cpu_us, speedup,
              speedup >= 1.0 ? "GPU wins" : "CPU wins");
}

// ---------------------------------------------------------------------------
// Generic bench over all shapes for dtype T
// ---------------------------------------------------------------------------

template <typename T>
static void bench_kv_decode(const char* dtype_name, float tol) {
  if (std::is_same<T, ct2_bf16>::value)
    std::printf("\n=== KV-cache decode (%s) — BF16 uses synchronous MPSGraph per head ===\n", dtype_name);
  else
    std::printf("\n=== KV-cache decode (%s) ===\n", dtype_name);
  print_header();

  for (int si = 0; si < N_SHAPES; ++si) {
    const DecodeShape& s = SHAPES[si];
    const float scale     = 1.f / std::sqrtf((float)s.hd);
    const int   sq        = 1;
    const int   offset    = s.sk - 1;      // position where new token is written
    const int   total_cache = s.sk + 4;    // slight over-alloc (mirrors _offset_free_space)
    const int   row_elems   = s.nhk * s.hd;

    const dim_t q_elems     = (dim_t)s.batch * sq          * s.nh  * s.hd;
    const dim_t kv_new_elems= (dim_t)s.batch * 1           * s.nhk * s.hd;
    const dim_t cache_elems = (dim_t)s.batch * total_cache * s.nhk * s.hd;

    rng_state = 0x55AA55AAu + (uint32_t)si * 0x9E3779B9u;

    // Fill float32 inputs
    std::vector<float> qf(q_elems), kf_hist(cache_elems, 0.f), vf_hist(cache_elems, 0.f);
    std::vector<float> kf_new(kv_new_elems), vf_new(kv_new_elems);
    for (dim_t i = 0; i < q_elems;      ++i) qf   [i] = next_float(-1.f, 1.f);
    for (dim_t i = 0; i < kv_new_elems; ++i) kf_new[i] = next_float(-1.f, 1.f);
    for (dim_t i = 0; i < kv_new_elems; ++i) vf_new[i] = next_float(-1.f, 1.f);
    // History tokens [0..offset-1]
    for (int b = 0; b < s.batch; ++b)
      for (int pos = 0; pos < offset; ++pos) {
        int base = (b * total_cache + pos) * row_elems;
        for (int e = 0; e < row_elems; ++e) kf_hist[base + e] = next_float(-1.f, 1.f);
        for (int e = 0; e < row_elems; ++e) vf_hist[base + e] = next_float(-1.f, 1.f);
      }

    // Metal buffers
    T* q_m       = metal_alloc<T>(q_elems);
    T* kv_new_k  = metal_alloc<T>(kv_new_elems);
    T* kv_new_v  = metal_alloc<T>(kv_new_elems);
    T* cache_k   = metal_alloc<T>(cache_elems);
    T* cache_v   = metal_alloc<T>(cache_elems);
    T* out_m     = metal_alloc<T>(q_elems);

    for (dim_t i = 0; i < q_elems;      ++i) q_m     [i] = T(qf   [i]);
    for (dim_t i = 0; i < kv_new_elems; ++i) kv_new_k[i] = T(kf_new[i]);
    for (dim_t i = 0; i < kv_new_elems; ++i) kv_new_v[i] = T(vf_new[i]);
    for (dim_t i = 0; i < cache_elems;  ++i) { cache_k[i] = T(kf_hist[i]); cache_v[i] = T(vf_hist[i]); }

    // Warmup: compiles PSO / MPSGraph.
    metal::commit_and_wait();
    update_cache(cache_k, kv_new_k, s.batch, total_cache, s.nhk, s.hd, offset);
    update_cache(cache_v, kv_new_v, s.batch, total_cache, s.nhk, s.hd, offset);
    metal::sdpa_metal<T>(q_m, cache_k, cache_v, out_m,
                          s.batch, sq, s.sk, s.nh, s.nhk, s.hd, scale, /*causal=*/false);
    metal::commit_and_wait();

    // ------------------------------------------------------------------
    // Accuracy: Metal output vs float32 CPU reference.
    // ------------------------------------------------------------------
    // Build reference cache (float32) with new token included.
    std::vector<float> ref_cache_k = kf_hist;
    std::vector<float> ref_cache_v = vf_hist;
    for (int b = 0; b < s.batch; ++b) {
      std::memcpy(&ref_cache_k[(b * total_cache + offset) * row_elems],
                  &kf_new[b * row_elems], row_elems * sizeof(float));
      std::memcpy(&ref_cache_v[(b * total_cache + offset) * row_elems],
                  &vf_new[b * row_elems], row_elems * sizeof(float));
    }
    std::vector<float> ref_out(q_elems);
    ref_sdpa(qf.data(), ref_cache_k.data(), ref_cache_v.data(), ref_out.data(),
             s.batch, sq, s.sk, s.nh, s.nhk, s.hd, scale);

    std::vector<float> got_f(q_elems);
    for (dim_t i = 0; i < q_elems; ++i) got_f[i] = float(out_m[i]);
    float err = max_abs_err(ref_out.data(), got_f.data(), q_elems);

    // ------------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------------
    // Each timed iteration writes the same K/V to the same cache slot `offset`
    // and attends over [0, sk).  Because the write is idempotent there is no
    // need to reset the cache between iterations — the result is identical
    // every time.  Excluding the reset gives timings that reflect only the
    // actual decode work (memcpy of nhk*hd elements + SDPA).
    int iters = iter_count(s);

    // GPU: commit_and_wait (flush) + memcpy new K/V into slot + sdpa + commit_and_wait.
    double gpu_us = bench_median_us(iters, [&] {
      metal::commit_and_wait();
      update_cache(cache_k, kv_new_k, s.batch, total_cache, s.nhk, s.hd, offset);
      update_cache(cache_v, kv_new_v, s.batch, total_cache, s.nhk, s.hd, offset);
      metal::sdpa_metal<T>(q_m, cache_k, cache_v, out_m,
                            s.batch, sq, s.sk, s.nh, s.nhk, s.hd, scale, false);
      metal::commit_and_wait();
      return float(out_m[0]);
    });

    // CPU: memcpy new K/V into slot + ref_sdpa.
    // Pre-populate the CPU cache (history + new token already in place from the
    // accuracy step above) and time only the small per-step work.
    std::vector<float> cpu_cache_k = ref_cache_k;
    std::vector<float> cpu_cache_v = ref_cache_v;
    double cpu_us = bench_median_us(iters, [&] {
      for (int b = 0; b < s.batch; ++b) {
        std::memcpy(&cpu_cache_k[(b * total_cache + offset) * row_elems],
                    &kf_new[b * row_elems], row_elems * sizeof(float));
        std::memcpy(&cpu_cache_v[(b * total_cache + offset) * row_elems],
                    &vf_new[b * row_elems], row_elems * sizeof(float));
      }
      ref_sdpa(qf.data(), cpu_cache_k.data(), cpu_cache_v.data(), ref_out.data(),
               s.batch, sq, s.sk, s.nh, s.nhk, s.hd, scale);
      return ref_out[0];
    });

    char shape_buf[56];
    format_shape(shape_buf, sizeof(shape_buf), s);
    print_row(shape_buf, err, tol, gpu_us, cpu_us);

    metal_free(q_m); metal_free(kv_new_k); metal_free(kv_new_v);
    metal_free(cache_k); metal_free(cache_v); metal_free(out_m);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.2 Metal vs CPU: KV-cache Decode Accuracy and Performance ===\n");
  std::printf("One decode step at offset=sk-1 (sq=1, new token added, SDPA over [0,sk)).\n");
  std::printf("GPU time: commit_and_wait + memcpy + sdpa + commit_and_wait.\n");
  std::printf("CPU time: memcpy + single-threaded float32 attention.\n");
  std::printf("NOTE: CB overhead ~0.4 ms is amortised in real pipeline.\n\n");

  bench_kv_decode<float>   ("float32",  1e-5f);
  bench_kv_decode<ct2_f16> ("float16",  0.02f);
  bench_kv_decode<ct2_bf16>("bfloat16", 0.10f);

  std::printf("\n=== Accuracy summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
