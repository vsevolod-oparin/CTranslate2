// tests/metal/fused_norm_gemm_bench.mm
//
// Benchmark: fused LayerNorm/RMSNorm + GEMV vs separate LN + MPS GEMM.
//
// Measures wall-clock time of:
//   (a) Fused kernel (single dispatch, encode-only + commit_and_wait)
//   (b) Separate: LayerNorm kernel + MPS GEMM (two dispatches + commit_and_wait)
//
// Build (from repo root):
//   clang++ -std=c++17 -O2 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/fused_norm_gemm_bench.mm \
//     src/metal/ops_fused_norm_gemm.mm \
//     src/metal/ops_norm_gather.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -framework Accelerate \
//     -o fused_norm_gemm_bench && ./fused_norm_gemm_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
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

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> rand_vec(size_t n, std::mt19937& rng, float lo = -1.f, float hi = 1.f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

template <typename T>
static std::vector<T> to_type(const std::vector<float>& src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = (T)src[i];
  return dst;
}

template <typename T>
static T* gpu_alloc(const std::vector<T>& src) {
  auto& alloc = get_allocator<Device::METAL>();
  T* ptr = static_cast<T*>(alloc.allocate(src.size() * sizeof(T), 0));
  std::memcpy(ptr, src.data(), src.size() * sizeof(T));
  return ptr;
}

template <typename T>
static T* gpu_alloc_empty(size_t n) {
  auto& alloc = get_allocator<Device::METAL>();
  return static_cast<T*>(alloc.allocate(n * sizeof(T), 0));
}

static void gpu_free(void* ptr) {
  get_allocator<Device::METAL>().free(ptr, 0);
}

using hrc = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Benchmark: fused vs separate
// ---------------------------------------------------------------------------

template <typename T>
static void bench_fused_vs_separate(const char* label, dim_t M, dim_t K, dim_t N,
                                     bool is_rms, int warmup, int iters) {
  std::mt19937 rng(42);
  auto x_f = rand_vec(M * K, rng);
  auto gamma_f = rand_vec(K, rng, 0.5f, 1.5f);
  auto beta_f = rand_vec(K, rng, -0.5f, 0.5f);
  auto W_f = rand_vec(N * K, rng, -0.1f, 0.1f);

  auto x_t = to_type<T>(x_f);
  auto gamma_t = to_type<T>(gamma_f);
  auto beta_t = to_type<T>(beta_f);
  auto W_t = to_type<T>(W_f);

  T* d_x = gpu_alloc(x_t);
  T* d_gamma = gpu_alloc(gamma_t);
  T* d_beta = gpu_alloc(beta_t);
  T* d_W = gpu_alloc(W_t);
  T* d_y = gpu_alloc_empty<T>(M * N);
  T* d_norm = gpu_alloc_empty<T>(M * K);

  float eps = is_rms ? 1e-6f : 1e-5f;

  // --- Fused ---
  for (int i = 0; i < warmup; ++i) {
    if (is_rms)
      metal::fused_rms_norm_gemm_metal(d_x, d_gamma, d_W, d_y, M, K, N, eps);
    else
      metal::fused_layer_norm_gemm_metal(d_x, d_gamma, d_beta, d_W, d_y, M, K, N, eps);
    metal::commit_and_wait();
  }

  auto t0 = hrc::now();
  for (int i = 0; i < iters; ++i) {
    if (is_rms)
      metal::fused_rms_norm_gemm_metal(d_x, d_gamma, d_W, d_y, M, K, N, eps);
    else
      metal::fused_layer_norm_gemm_metal(d_x, d_gamma, d_beta, d_W, d_y, M, K, N, eps);
    metal::commit_and_wait();
  }
  auto t1 = hrc::now();
  double fused_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  // --- Separate: norm + GEMM ---
  for (int i = 0; i < warmup; ++i) {
    if (is_rms)
      metal::rms_norm_metal(d_x, d_gamma, d_norm, M, K, eps);
    else
      metal::layer_norm_metal(d_x, d_gamma, d_beta, d_norm, M, K, eps);
    primitives<Device::METAL>::gemm(false, false, false, true, M, N, K, 1.0f,
                                     d_norm, K, d_W, K, 0.0f, d_y, N);
    metal::commit_and_wait();
  }

  t0 = hrc::now();
  for (int i = 0; i < iters; ++i) {
    if (is_rms)
      metal::rms_norm_metal(d_x, d_gamma, d_norm, M, K, eps);
    else
      metal::layer_norm_metal(d_x, d_gamma, d_beta, d_norm, M, K, eps);
    primitives<Device::METAL>::gemm(false, false, false, true, M, N, K, 1.0f,
                                     d_norm, K, d_W, K, 0.0f, d_y, N);
    metal::commit_and_wait();
  }
  t1 = hrc::now();
  double sep_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

  double speedup = sep_us / fused_us;
  std::printf("  %-45s fused=%7.0f us  separate=%7.0f us  speedup=%.2fx\n",
              label, fused_us, sep_us, speedup);

  gpu_free(d_x);
  gpu_free(d_gamma);
  gpu_free(d_beta);
  gpu_free(d_W);
  gpu_free(d_y);
  gpu_free(d_norm);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  @autoreleasepool {
    std::printf("=== Fused LayerNorm/RMSNorm + GEMV Benchmark ===\n");
    std::printf("(Each measurement: warmup + %d iterations with commit_and_wait)\n\n", 50);

    int warmup = 10, iters = 50;

    // Whisper-base shapes (K=512, N varies)
    std::printf("float32 LayerNorm+GEMV (Whisper-base shapes):\n");
    bench_fused_vs_separate<float>("f32 M=1  K=512  N=1536 (QKV fused)", 1, 512, 1536, false, warmup, iters);
    bench_fused_vs_separate<float>("f32 M=1  K=512  N=2048 (FF1)", 1, 512, 2048, false, warmup, iters);
    bench_fused_vs_separate<float>("f32 M=4  K=512  N=1536", 4, 512, 1536, false, warmup, iters);
    bench_fused_vs_separate<float>("f32 M=16 K=512  N=1536", 16, 512, 1536, false, warmup, iters);

    std::printf("\nfloat16 LayerNorm+GEMV:\n");
    bench_fused_vs_separate<ct2_f16>("f16 M=1  K=512  N=1536", 1, 512, 1536, false, warmup, iters);
    bench_fused_vs_separate<ct2_f16>("f16 M=1  K=512  N=2048", 1, 512, 2048, false, warmup, iters);
    bench_fused_vs_separate<ct2_f16>("f16 M=4  K=512  N=1536", 4, 512, 1536, false, warmup, iters);

    std::printf("\nbfloat16 LayerNorm+GEMV:\n");
    bench_fused_vs_separate<ct2_bf16>("bf16 M=1  K=512  N=1536", 1, 512, 1536, false, warmup, iters);
    bench_fused_vs_separate<ct2_bf16>("bf16 M=1  K=512  N=2048", 1, 512, 2048, false, warmup, iters);

    // RMSNorm variants
    std::printf("\nfloat32 RMSNorm+GEMV:\n");
    bench_fused_vs_separate<float>("f32 RMS M=1  K=1024 N=4096", 1, 1024, 4096, true, warmup, iters);
    bench_fused_vs_separate<float>("f32 RMS M=1  K=2048 N=8192", 1, 2048, 8192, true, warmup, iters);
    bench_fused_vs_separate<float>("f32 RMS M=4  K=1024 N=4096", 4, 1024, 4096, true, warmup, iters);

    // Large K
    std::printf("\nLarge K:\n");
    bench_fused_vs_separate<float>("f32 M=1  K=4096 N=6144", 1, 4096, 6144, false, warmup, iters);
    bench_fused_vs_separate<float>("f32 M=1  K=8192 N=4096", 1, 8192, 4096, false, warmup, iters);

    std::printf("\nDone.\n");
    return 0;
  }
}
