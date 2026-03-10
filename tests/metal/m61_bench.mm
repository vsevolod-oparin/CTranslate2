// tests/metal/m61_bench.mm
//
// M6.1 Metal vs CPU — Accuracy and performance for sdpa_metal<T>.
//
// For each shape and dtype (float32, float16, bfloat16), reports:
//   • max absolute error vs float32 CPU reference + PASS/FAIL
//   • GPU time (encode+commit_and_wait) in µs
//   • CPU time (single-threaded float32) in µs
//   • Speedup = CPU_µs / GPU_µs  (>1x means GPU wins)
//
// Shapes cover:
//   - Prefill (sq = sk, causal): typical LLM context-fill
//   - Decode  (sq = 1, sk varies, causal): single-token generation
//   - GQA (nh > nhk): grouped query attention
//
// NOTE: GPU times include the ~0.4 ms CB submission overhead.
//       In a real pipeline the CB is submitted only at synchronize_stream(),
//       amortising overhead across many ops.  Standalone timings are
//       conservative; the true crossover is lower than shown here.
//       BF16 path uses synchronous MPSGraph per head — extra overhead at
//       small shapes but typical for Apple Metal BF16.
//
// Build and run from the repository root (use -O2 for accurate timings):
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/m61_bench.mm \
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
//     -o m61_bench && ./m61_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
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
// Timing
// ---------------------------------------------------------------------------

using Clock  = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

static double now_us() {
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

static volatile float sink_f = 0.f;

// Returns median elapsed time in µs over `iters` calls.
// fn() must return a float stored into sink_f to prevent dead-code elimination.
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
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// xorshift32 PRNG for reproducible random inputs
// ---------------------------------------------------------------------------

static uint32_t rng_state = 0xDEADBEEFu;

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
  for (dim_t i = 0; i < n; ++i) {
    err = std::max(err, std::fabs(ref[i] - got[i]));
  }
  return err;
}

// ---------------------------------------------------------------------------
// float32 CPU reference for SDPA
//
// Q/K/V layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
// Row stride Q:   q_lda  = num_heads   * head_dim
// Row stride K/V: kv_lda = num_heads_k * head_dim
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

      // Causal mask: set scores[s, t] = -1e9 for t > s
      if (is_causal) {
        for (int s = 0; s < sq; ++s) {
          for (int t = s + 1; t < sk; ++t) {
            scores[s * sk + t] = -1e9f;
          }
        }
      }

      // Softmax over each row (numerically stable)
      for (int s = 0; s < sq; ++s) {
        float mx = -1e38f;
        for (int t = 0; t < sk; ++t) {
          if (scores[s * sk + t] > mx) {
            mx = scores[s * sk + t];
          }
        }
        float sum = 0.f;
        for (int t = 0; t < sk; ++t) {
          sum += std::exp(scores[s * sk + t] - mx);
        }
        for (int t = 0; t < sk; ++t) {
          scores[s * sk + t] = std::exp(scores[s * sk + t] - mx) / sum;
        }
      }

      // out[s, d] = sum_t(scores[s, t] * V[t, d])
      for (int s = 0; s < sq; ++s) {
        for (int d = 0; d < hd; ++d) {
          float acc = 0.f;
          for (int t = 0; t < sk; ++t) {
            acc += scores[s * sk + t] * v0[t * kv_lda + d];
          }
          out0[s * q_lda + d] = acc;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Shape descriptor
// ---------------------------------------------------------------------------

struct SDPAShape {
  int batch, sq, sk, nh, nhk, hd;
  bool causal;
};

// Shapes span prefill (sq=sk), decode (sq=1), and GQA (nhk < nh).
static const SDPAShape SHAPES[] = {
  // Prefill: sq = sk, causal=true
  {1,   64,   64,  8,  8,  64, true},
  {1,  128,  128,  8,  8,  64, true},
  {1,  256,  256,  8,  8,  64, true},
  {1,  512,  512,  8,  8,  64, true},
  {1, 1024, 1024,  8,  8,  64, true},
  // Decode: sq=1, sk varies, causal=true
  {1,    1,  256,  8,  8,  64, true},
  {1,    1, 1024,  8,  8,  64, true},
  // GQA prefill: nh=16, nhk=4
  {1,  256,  256, 16,  4,  64, true},
};
static const int N_SHAPES = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));

// Adaptive iteration count: fewer iters for expensive shapes.
static int iter_count(const SDPAShape& s) {
  // Rough work estimate: batch * nh * sq * sk (scores matrix elements)
  dim_t work = (dim_t)s.batch * s.nh * s.sq * s.sk;
  if (work <= 4096) {
    return 200;
  }
  if (work <= 65536) {
    return 80;
  }
  if (work <= 524288) {
    return 20;
  }
  if (work <= 4194304) {
    return 8;
  }
  return 4;
}

static void format_shape(char* buf, int bufsz, const SDPAShape& s) {
  if (s.nhk != s.nh) {
    std::snprintf(buf, bufsz, "b%d sq%d sk%d nh%d/%d hd%d",
                  s.batch, s.sq, s.sk, s.nh, s.nhk, s.hd);
  } else {
    std::snprintf(buf, bufsz, "b%d sq%d sk%d nh%d hd%d",
                  s.batch, s.sq, s.sk, s.nh, s.hd);
  }
}

// ---------------------------------------------------------------------------
// Shared bench header / row printer
// ---------------------------------------------------------------------------

static void print_bench_header() {
  std::printf("  %-28s  %12s  %8s  %10s  %10s  %9s  %s\n",
              "Shape", "max_abs_err", "Status",
              "GPU (µs)", "CPU (µs)", "Speedup", "");
  std::printf("  %s\n", std::string(88, '-').c_str());
}

static void print_bench_row(const char* shape, float abs_err, float tol,
                            double gpu_us, double cpu_us) {
  bool ok = std::isfinite(abs_err) && (abs_err <= tol);
  if (ok) { ++g_pass; } else { ++g_fail; }
  double speedup = cpu_us / gpu_us;
  const char* winner = (speedup >= 1.0) ? "GPU wins" : "CPU wins";
  std::printf("  %-28s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
              shape, abs_err, ok ? "PASS" : "FAIL",
              gpu_us, cpu_us, speedup, winner);
}

// ---------------------------------------------------------------------------
// Generic bench loop for a given dtype T
// ---------------------------------------------------------------------------

template <typename T>
static void bench_sdpa_dtype(const char* dtype_name, float tol) {
  if (std::is_same<T, ct2_bf16>::value) {
    std::printf("\n=== SDPA (%s) — BF16 uses synchronous MPSGraph per head ===\n",
                dtype_name);
  } else {
    std::printf("\n=== SDPA (%s) ===\n", dtype_name);
  }
  print_bench_header();

  for (int si = 0; si < N_SHAPES; ++si) {
    const SDPAShape& s = SHAPES[si];
    const float scale  = 1.f / std::sqrtf((float)s.hd);

    const dim_t q_elems  = (dim_t)s.batch * s.sq * s.nh  * s.hd;
    const dim_t kv_elems = (dim_t)s.batch * s.sk * s.nhk * s.hd;

    rng_state = 0x11223344u + (uint32_t)si * 0x9E3779B9u;

    // Allocate and fill Metal buffers (Shared memory, CPU-writable).
    T* q_m   = metal_alloc<T>(q_elems);
    T* k_m   = metal_alloc<T>(kv_elems);
    T* v_m   = metal_alloc<T>(kv_elems);
    T* out_m = metal_alloc<T>(q_elems);

    std::vector<float> qf(q_elems), kf(kv_elems), vf(kv_elems);
    for (dim_t i = 0; i < q_elems;  ++i) { qf[i] = next_float(-1.f, 1.f); }
    for (dim_t i = 0; i < kv_elems; ++i) { kf[i] = next_float(-1.f, 1.f); }
    for (dim_t i = 0; i < kv_elems; ++i) { vf[i] = next_float(-1.f, 1.f); }

    for (dim_t i = 0; i < q_elems;  ++i) { q_m[i] = T(qf[i]); }
    for (dim_t i = 0; i < kv_elems; ++i) { k_m[i] = T(kf[i]); }
    for (dim_t i = 0; i < kv_elems; ++i) { v_m[i] = T(vf[i]); }

    // Warmup: also compiles PSO and MPSGraph lazily.
    metal::sdpa_metal<T>(q_m, k_m, v_m, out_m,
                         s.batch, s.sq, s.sk, s.nh, s.nhk, s.hd,
                         scale, s.causal);
    metal::commit_and_wait();

    // ------------------------------------------------------------------
    // Accuracy: compare Metal output (converted to float32) vs CPU ref.
    // ------------------------------------------------------------------
    std::vector<float> out_gpu_f(q_elems);
    for (dim_t i = 0; i < q_elems; ++i) {
      out_gpu_f[i] = float(out_m[i]);
    }

    std::vector<float> ref_out(q_elems);
    ref_sdpa(qf.data(), kf.data(), vf.data(), ref_out.data(),
             s.batch, s.sq, s.sk, s.nh, s.nhk, s.hd, scale, s.causal);

    float err = max_abs_err(ref_out.data(), out_gpu_f.data(), q_elems);

    // ------------------------------------------------------------------
    // Timing
    // ------------------------------------------------------------------
    int iters = iter_count(s);

    double gpu_us = bench_median_us(iters, [&] {
      metal::sdpa_metal<T>(q_m, k_m, v_m, out_m,
                           s.batch, s.sq, s.sk, s.nh, s.nhk, s.hd,
                           scale, s.causal);
      metal::commit_and_wait();
      return float(out_m[0]);
    });

    double cpu_us = bench_median_us(iters, [&] {
      ref_sdpa(qf.data(), kf.data(), vf.data(), ref_out.data(),
               s.batch, s.sq, s.sk, s.nh, s.nhk, s.hd, scale, s.causal);
      return ref_out[0];
    });

    char shape_buf[48];
    format_shape(shape_buf, sizeof(shape_buf), s);
    print_bench_row(shape_buf, err, tol, gpu_us, cpu_us);

    metal_free(q_m);
    metal_free(k_m);
    metal_free(v_m);
    metal_free(out_m);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.1 Metal vs CPU: SDPA Accuracy and Performance ===\n");
  std::printf("Timing: median of N timed runs; GPU includes encode+commit_and_wait.\n");
  std::printf("NOTE: CB overhead ~0.4 ms is amortised in real pipeline; standalone GPU\n");
  std::printf("      times are conservative — actual pipeline throughput is higher.\n");
  std::printf("CPU reference: single-threaded float32 SDPA (no BLAS, no SIMD).\n");

  // Tolerances chosen to pass with expected quantisation error per dtype.
  bench_sdpa_dtype<float>  ("float32", 1e-5f);
  bench_sdpa_dtype<ct2_f16>("float16", 0.02f);
  bench_sdpa_dtype<ct2_bf16>("bfloat16", 0.10f);

  std::printf("\n");
  std::printf("=== Accuracy summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
