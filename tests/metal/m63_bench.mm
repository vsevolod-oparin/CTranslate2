// tests/metal/m63_bench.mm
//
// M6.3 — Rotary (RoPE) Metal vs CPU Performance Benchmark
//
// Measures encode-only (deferred) Metal rotary kernel throughput against
// a single-threaded float32 CPU reference for typical LLM prefill shapes.
//
// GPU timing: dispatch rotary_metal<T> (encode-only) + commit_and_wait().
// CPU timing: single-threaded float32 non-interleave rotary loop.
//
// Note: GPU times include one commit_and_wait() per iteration
// (~0.4 ms CB overhead). In production the CB is shared across many ops;
// see the benchmark notes for realistic pipeline context.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/m63_bench.mm \
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
//     -o m63_bench && ./m63_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
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
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;

static double median_us(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---------------------------------------------------------------------------
// CPU reference (float32, non-interleave, is_transposed=false)
// ---------------------------------------------------------------------------

static void cpu_rotary_f32(const float* x, const float* sin_f, const float* cos_f,
                            float* y, dim_t total_vecs, dim_t depth, dim_t ndims,
                            dim_t head_size) {
  const dim_t half = ndims / 2;
  for (dim_t v = 0; v < total_vecs; ++v) {
    dim_t t = v / head_size;
    const float* sx = x   + v * depth;
    float*       sy = y   + v * depth;
    const float* sc = cos_f + t * ndims;
    const float* ss = sin_f + t * ndims;
    for (dim_t d = 0; d < ndims; ++d) {
      float xi = sx[d], cd = sc[d], sd = ss[d];
      sy[d] = (d < half) ? xi * cd - sx[d + half] * sd
                         : xi * cd + sx[d - half] * sd;
    }
    // pass-through for d in [ndims, depth)
    for (dim_t d = ndims; d < depth; ++d) sy[d] = sx[d];
  }
}

// ---------------------------------------------------------------------------
// Accuracy check: compare GPU output to float32 CPU reference
// ---------------------------------------------------------------------------

static int g_acc_pass = 0, g_acc_fail = 0;

template <typename T>
static void accuracy_check(const char* name, float tol,
                            dim_t batch, dim_t time, dim_t heads, dim_t hd,
                            dim_t ndims) {
  const dim_t total_elems = batch * time * heads * hd;
  const dim_t total_vecs  = total_elems / hd;
  const dim_t sin_elems   = time * ndims;

  // Simple ascending fill for reproducibility
  std::vector<float> x_f(static_cast<size_t>(total_elems));
  for (dim_t i = 0; i < total_elems; ++i)
    x_f[static_cast<size_t>(i)] = 0.1f * float(i % 33 - 16);

  std::vector<float> inv_freq(static_cast<size_t>(ndims));
  for (dim_t d = 0; d < ndims; ++d)
    inv_freq[static_cast<size_t>(d)] = 1.f / std::pow(10000.f, float(d * 2) / float(ndims));

  std::vector<float> sin_f(static_cast<size_t>(sin_elems));
  std::vector<float> cos_f(static_cast<size_t>(sin_elems));
  for (dim_t t = 0; t < time; ++t)
    for (dim_t d = 0; d < ndims; ++d) {
      float angle = float(t) * inv_freq[static_cast<size_t>(d)];
      sin_f[static_cast<size_t>(t * ndims + d)] = std::sin(angle);
      cos_f[static_cast<size_t>(t * ndims + d)] = std::cos(angle);
    }

  // CPU reference (float32)
  std::vector<float> ref(static_cast<size_t>(total_elems));
  cpu_rotary_f32(x_f.data(), sin_f.data(), cos_f.data(), ref.data(),
                 total_vecs, hd, ndims, heads);

  // GPU path
  T* x_m   = metal_alloc<T>(total_elems);
  T* sin_m = metal_alloc<T>(sin_elems);
  T* cos_m = metal_alloc<T>(sin_elems);
  T* out_m = metal_alloc<T>(total_elems);

  for (dim_t i = 0; i < total_elems; ++i) x_m[i]   = T(x_f[static_cast<size_t>(i)]);
  for (dim_t i = 0; i < sin_elems;   ++i) {
    sin_m[i] = T(sin_f[static_cast<size_t>(i)]);
    cos_m[i] = T(cos_f[static_cast<size_t>(i)]);
  }

  metal::rotary_metal<T>(x_m, sin_m, cos_m, out_m,
                         total_vecs, hd, ndims, time, heads,
                         false, false);
  metal::commit_and_wait();

  float err = 0.f;
  for (dim_t i = 0; i < total_elems; ++i)
    err = std::max(err, std::fabs(ref[static_cast<size_t>(i)] - float(out_m[i])));

  bool ok = std::isfinite(err) && err <= tol;
  if (ok) { ++g_acc_pass; }
  else     { ++g_acc_fail;
             std::printf("  ACC FAIL  %s  err=%.3e  tol=%.3e\n", name, err, tol); }

  metal_free(x_m);
  metal_free(sin_m);
  metal_free(cos_m);
  metal_free(out_m);
}

// ---------------------------------------------------------------------------
// Benchmark: GPU (encode-only + commit_and_wait) vs CPU
// ---------------------------------------------------------------------------

constexpr int kWarmup = 3;
constexpr int kIter   = 9;

template <typename T>
static void bench(const char* label, float tol,
                  dim_t batch, dim_t time, dim_t heads, dim_t hd,
                  dim_t ndims) {
  // --- accuracy check first ---
  accuracy_check<T>(label, tol, batch, time, heads, hd, ndims);

  const dim_t total_elems = batch * time * heads * hd;
  const dim_t total_vecs  = total_elems / hd;
  const dim_t sin_elems   = time * ndims;

  std::vector<float> x_f(static_cast<size_t>(total_elems), 0.5f);
  std::vector<float> sin_f(static_cast<size_t>(sin_elems), 0.1f);
  std::vector<float> cos_f(static_cast<size_t>(sin_elems), 0.9f);
  std::vector<float> cpu_out(static_cast<size_t>(total_elems));

  T* x_m   = metal_alloc<T>(total_elems);
  T* sin_m = metal_alloc<T>(sin_elems);
  T* cos_m = metal_alloc<T>(sin_elems);
  T* out_m = metal_alloc<T>(total_elems);

  for (dim_t i = 0; i < total_elems; ++i) x_m[i]   = T(x_f[static_cast<size_t>(i)]);
  for (dim_t i = 0; i < sin_elems;   ++i) {
    sin_m[i] = T(sin_f[static_cast<size_t>(i)]);
    cos_m[i] = T(cos_f[static_cast<size_t>(i)]);
  }

  // GPU timing (encode + commit_and_wait per iteration)
  std::vector<double> gpu_times;
  for (int it = 0; it < kWarmup + kIter; ++it) {
    auto t0 = Clock::now();
    metal::rotary_metal<T>(x_m, sin_m, cos_m, out_m,
                           total_vecs, hd, ndims, time, heads,
                           false, false);
    metal::commit_and_wait();
    auto t1 = Clock::now();
    if (it >= kWarmup)
      gpu_times.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  // CPU timing (float32, single-threaded)
  std::vector<double> cpu_times;
  for (int it = 0; it < kWarmup + kIter; ++it) {
    auto t0 = Clock::now();
    cpu_rotary_f32(x_f.data(), sin_f.data(), cos_f.data(), cpu_out.data(),
                   total_vecs, hd, ndims, heads);
    auto t1 = Clock::now();
    if (it >= kWarmup)
      cpu_times.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  double gpu_us = median_us(gpu_times);
  double cpu_us = median_us(cpu_times);
  double speedup = cpu_us / gpu_us;
  const char* winner = (speedup >= 1.0) ? "GPU" : "CPU";

  std::printf("  %-55s  GPU %6.0f µs  CPU %6.0f µs  %5.2fx  %s\n",
              label, gpu_us, cpu_us, speedup, winner);

  metal_free(x_m);
  metal_free(sin_m);
  metal_free(cos_m);
  metal_free(out_m);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.3 Rotary Performance Benchmark (Metal vs CPU) ===\n\n");
  std::printf("  %-55s  %17s  %17s  %7s  %s\n",
              "Shape", "GPU (µs)", "CPU (µs)", "Speedup", "Winner");
  std::printf("  %s\n", std::string(120, '-').c_str());

  // -------------------------------------------------------------------------
  // float32 — typical prefill shapes (FA2 layout: [batch, time, heads, hd])
  // -------------------------------------------------------------------------
  std::printf("\n--- float32 ---\n");

  // Small decode-like (time=1): GPU overhead dominates
  bench<float>("b1 t1  h4  hd64  ndims=64",  1e-5f, 1,  1, 4, 64, 64);
  bench<float>("b1 t8  h4  hd64  ndims=64",  1e-5f, 1,  8, 4, 64, 64);
  bench<float>("b1 t32 h4  hd64  ndims=64",  1e-5f, 1, 32, 4, 64, 64);
  bench<float>("b1 t64 h4  hd64  ndims=64",  1e-5f, 1, 64, 4, 64, 64);
  bench<float>("b1 t128 h4 hd64  ndims=64",  1e-5f, 1,128, 4, 64, 64);
  bench<float>("b1 t256 h8 hd64  ndims=64",  1e-5f, 1,256, 8, 64, 64);
  bench<float>("b1 t512 h8 hd64  ndims=64",  1e-5f, 1,512, 8, 64, 64);
  bench<float>("b1 t1024 h8 hd64 ndims=64",  1e-5f, 1,1024,8, 64, 64);
  bench<float>("b1 t2048 h8 hd64 ndims=64",  1e-5f, 1,2048,8, 64, 64);
  // GQA (nh=16, hd=128, typical for Llama-70B)
  bench<float>("b1 t256 h16 hd128 ndims=128", 1e-5f, 1,256,16,128,128);
  bench<float>("b1 t512 h16 hd128 ndims=128", 1e-5f, 1,512,16,128,128);
  // Partial rotation (ndims < hd)
  bench<float>("b1 t256 h8 hd128 ndims=64 (partial)", 1e-5f, 1,256, 8,128, 64);

  // -------------------------------------------------------------------------
  // float16
  // -------------------------------------------------------------------------
  std::printf("\n--- float16 ---\n");

  bench<ct2_f16>("b1 t8  h4  hd64  ndims=64",  5e-3f, 1,  8, 4, 64, 64);
  bench<ct2_f16>("b1 t64 h4  hd64  ndims=64",  5e-3f, 1, 64, 4, 64, 64);
  bench<ct2_f16>("b1 t256 h8 hd64  ndims=64",  5e-3f, 1,256, 8, 64, 64);
  bench<ct2_f16>("b1 t512 h8 hd64  ndims=64",  5e-3f, 1,512, 8, 64, 64);
  bench<ct2_f16>("b1 t1024 h8 hd64 ndims=64",  5e-3f, 1,1024,8, 64, 64);
  bench<ct2_f16>("b1 t2048 h8 hd64 ndims=64",  5e-3f, 1,2048,8, 64, 64);

  // -------------------------------------------------------------------------
  // bfloat16
  // -------------------------------------------------------------------------
  std::printf("\n--- bfloat16 ---\n");

  bench<ct2_bf16>("b1 t8  h4  hd64  ndims=64",  5e-2f, 1,  8, 4, 64, 64);
  bench<ct2_bf16>("b1 t64 h4  hd64  ndims=64",  5e-2f, 1, 64, 4, 64, 64);
  bench<ct2_bf16>("b1 t256 h8 hd64  ndims=64",  5e-2f, 1,256, 8, 64, 64);
  bench<ct2_bf16>("b1 t512 h8 hd64  ndims=64",  5e-2f, 1,512, 8, 64, 64);
  bench<ct2_bf16>("b1 t1024 h8 hd64 ndims=64",  5e-2f, 1,1024,8, 64, 64);
  bench<ct2_bf16>("b1 t2048 h8 hd64 ndims=64",  5e-2f, 1,2048,8, 64, 64);

  std::printf("\nAccuracy: %d passed, %d failed\n", g_acc_pass, g_acc_fail);
  std::printf("\nNote: GPU times include one commit_and_wait() per iteration\n");
  std::printf("      (~0.4 ms CB overhead). In production the CB is committed\n");
  std::printf("      once per layer (shared with linear ops, norm, SDPA),\n");
  std::printf("      so the effective GPU advantage is much higher.\n");

  return g_acc_fail > 0 ? 1 : 0;
}
