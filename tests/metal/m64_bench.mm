// tests/metal/m64_bench.mm
//
// M6.4 — AlibiAdd Metal vs CPU performance benchmark.
//
// Measures metal::alibi_add_metal<T>() (encode + commit_and_wait) against
// a single-threaded float32 CPU reference (same algorithm as the MSL kernel).
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/m64_bench.mm \
//     src/metal/ops_alibi.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm src/metal/ops_rotary.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o m64_bench && ./m64_bench

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
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// CPU reference (float32, single-threaded)
// ---------------------------------------------------------------------------

static void cpu_alibi_add(
    const float* input, const float* alibi, float* output,
    dim_t batch, dim_t nh, dim_t ql, dim_t kl, dim_t cached_kl, dim_t offset) {
  const dim_t total_rows = batch * nh * ql;
  for (dim_t vec = 0; vec < total_rows; ++vec) {
    const dim_t h = (vec / ql) % nh;
    for (dim_t k = 0; k < kl; ++k) {
      dim_t in_idx    = vec * kl + k;
      dim_t alibi_idx = h * cached_kl + offset + k;
      output[in_idx]  = input[in_idx] + alibi[alibi_idx];
    }
  }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static double now_us() {
  using clock = std::chrono::high_resolution_clock;
  return static_cast<double>(
      clock::now().time_since_epoch().count()) * 1e-3;
}

// ---------------------------------------------------------------------------
// Accuracy check
// ---------------------------------------------------------------------------

static int g_acc_pass = 0, g_acc_fail = 0;

template <typename T>
static void check_accuracy(const char* label,
                            const T* gpu_out, const float* cpu_ref, dim_t n,
                            float tol) {
  float max_err = 0.f;
  for (dim_t i = 0; i < n; ++i)
    max_err = std::max(max_err, std::fabs(float(gpu_out[i]) - cpu_ref[i]));
  bool ok = std::isfinite(max_err) && max_err <= tol;
  if (ok) ++g_acc_pass;
  else   { ++g_acc_fail;
           std::printf("  ACCURACY FAIL  %s  max_err=%.3e tol=%.3e\n", label, max_err, tol); }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

static uint32_t rng_state = 0xDEADBEEFu;
static float next_float(float lo, float hi) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  float t = static_cast<float>(rng_state) / static_cast<float>(UINT32_MAX);
  return lo + t * (hi - lo);
}

template <typename T>
static void bench(const char* label, float acc_tol,
                  dim_t batch, dim_t nh, dim_t ql, dim_t kl,
                  int warmup = 3, int iters = 10) {
  const dim_t cached_kl   = kl;   // no cache offset in benchmark
  const dim_t alibi_offset = 0;
  const dim_t input_elems  = batch * nh * ql * kl;
  const dim_t alibi_elems  = nh * cached_kl;

  rng_state ^= (uint32_t)(batch * 1009 + nh * 313 + ql * 127 + kl * 31);

  // Float32 reference data
  std::vector<float> in_f(static_cast<size_t>(input_elems));
  std::vector<float> alibi_f(static_cast<size_t>(alibi_elems));
  for (auto& v : in_f)    v = next_float(-5.f, 5.f);
  for (auto& v : alibi_f) v = next_float(-2.f, 0.f);

  // CPU reference output
  std::vector<float> cpu_out(static_cast<size_t>(input_elems));
  cpu_alibi_add(in_f.data(), alibi_f.data(), cpu_out.data(),
                batch, nh, ql, kl, cached_kl, alibi_offset);

  // Metal buffers
  T* in_m    = metal_alloc<T>(input_elems);
  T* alibi_m = metal_alloc<T>(alibi_elems);
  T* out_m   = metal_alloc<T>(input_elems);

  for (dim_t i = 0; i < input_elems; ++i) in_m[i]    = T(in_f   [static_cast<size_t>(i)]);
  for (dim_t i = 0; i < alibi_elems; ++i) alibi_m[i] = T(alibi_f[static_cast<size_t>(i)]);

  // GPU warmup
  for (int w = 0; w < warmup; ++w) {
    metal::alibi_add_metal<T>(in_m, alibi_m, out_m, batch, nh, ql, kl, cached_kl, alibi_offset);
    metal::commit_and_wait();
  }

  // Accuracy check after warmup
  check_accuracy(label, out_m, cpu_out.data(), input_elems, acc_tol);

  // GPU timing
  double gpu_total = 0.0;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    metal::alibi_add_metal<T>(in_m, alibi_m, out_m, batch, nh, ql, kl, cached_kl, alibi_offset);
    metal::commit_and_wait();
    double t1 = now_us();
    gpu_total += (t1 - t0);
  }
  double gpu_us = gpu_total / iters;

  // CPU timing
  std::vector<float> cpu_tmp(static_cast<size_t>(input_elems));
  double cpu_total = 0.0;
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    cpu_alibi_add(in_f.data(), alibi_f.data(), cpu_tmp.data(),
                  batch, nh, ql, kl, cached_kl, alibi_offset);
    double t1 = now_us();
    cpu_total += (t1 - t0);
  }
  double cpu_us = cpu_total / iters;

  double speedup = cpu_us / gpu_us;
  const char* winner = (gpu_us < cpu_us) ? "GPU" : "CPU";

  std::printf("  %-44s  GPU %6.0f µs  CPU %5.0f µs  %5.2fx  %s\n",
              label, gpu_us, cpu_us, speedup, winner);

  metal_free(in_m);
  metal_free(alibi_m);
  metal_free(out_m);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M6.4 AlibiAdd Metal vs CPU Benchmark (Apple M4) ===\n");
  std::printf("GPU = alibi_add_metal<T> encode + commit_and_wait()\n");
  std::printf("CPU = single-threaded float32 reference\n\n");

  // ------------------------------------------------------------------
  // float32
  // ------------------------------------------------------------------
  std::printf("--- float32 ---\n");
  //              label                             tol   b   nh   ql    kl
  bench<float>("f32 [1,4,1,8]    decode tiny",     1e-5f, 1,  4,  1,    8);
  bench<float>("f32 [1,4,1,64]   decode short",    1e-5f, 1,  4,  1,   64);
  bench<float>("f32 [1,4,1,256]  decode medium",   1e-5f, 1,  4,  1,  256);
  bench<float>("f32 [1,8,1,512]  decode long",     1e-5f, 1,  8,  1,  512);
  bench<float>("f32 [1,8,1,1024] decode xl",       1e-5f, 1,  8,  1, 1024);
  bench<float>("f32 [1,4,64,64]  prefill mid",     1e-5f, 1,  4, 64,   64);
  bench<float>("f32 [1,4,128,128] prefill 128",    1e-5f, 1,  4,128,  128);
  bench<float>("f32 [1,8,256,256] prefill 256",    1e-5f, 1,  8,256,  256);
  bench<float>("f32 [1,8,512,512] prefill 512",    1e-5f, 1,  8,512,  512);
  bench<float>("f32 [2,8,256,256] batch2 p256",    1e-5f, 2,  8,256,  256);
  bench<float>("f32 [1,16,64,64]  many-heads",     1e-5f, 1, 16, 64,   64);

  // ------------------------------------------------------------------
  // float16
  // ------------------------------------------------------------------
  std::printf("\n--- float16 ---\n");
  bench<ct2_f16>("f16 [1,4,1,64]   decode short",    5e-3f, 1,  4,  1,   64);
  bench<ct2_f16>("f16 [1,8,1,512]  decode long",     5e-3f, 1,  8,  1,  512);
  bench<ct2_f16>("f16 [1,4,128,128] prefill 128",    5e-3f, 1,  4,128,  128);
  bench<ct2_f16>("f16 [1,8,256,256] prefill 256",    5e-3f, 1,  8,256,  256);
  bench<ct2_f16>("f16 [1,8,512,512] prefill 512",    5e-3f, 1,  8,512,  512);

  // ------------------------------------------------------------------
  // bfloat16
  // ------------------------------------------------------------------
  std::printf("\n--- bfloat16 ---\n");
  bench<ct2_bf16>("bf16 [1,4,1,64]   decode short",  5e-2f, 1,  4,  1,   64);
  bench<ct2_bf16>("bf16 [1,8,1,512]  decode long",   5e-2f, 1,  8,  1,  512);
  bench<ct2_bf16>("bf16 [1,4,128,128] prefill 128",  5e-2f, 1,  4,128,  128);
  bench<ct2_bf16>("bf16 [1,8,256,256] prefill 256",  5e-2f, 1,  8,256,  256);
  bench<ct2_bf16>("bf16 [1,8,512,512] prefill 512",  5e-2f, 1,  8,512,  512);

  std::printf("\nAccuracy: %d pass, %d fail\n", g_acc_pass, g_acc_fail);
  return g_acc_fail > 0 ? 1 : 0;
}
