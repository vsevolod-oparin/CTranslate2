// Benchmark: GPU two-pass reduction vs CPU-sequential reduction.
//
// For each input size we time:
//   GPU  — primitives<Device::MPS>::sum  (our new two-pass kernel)
//   CPU  — commit_and_wait() + std::accumulate (the first implementation)
//
// This establishes whether the GPU path actually wins and at what crossover
// size it becomes faster than purely sequential CPU work.
//
// Run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/reduction_bench.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o reduction_bench && ./reduction_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

static double now_us() {
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

// Prevent the compiler from eliminating a computed value.
// Written to a volatile global so the optimizer cannot prove it is dead.
static volatile float  sink_f  = 0.f;
static volatile dim_t  sink_d  = 0;

// Run `fn` `iters` times and return the median elapsed time in microseconds.
// fn must return a value that is stored into `sink_f` (prevents dead-code
// elimination of the computation inside fn).
template <typename Fn>
static double bench_median_us(int iters, Fn fn) {
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    sink_f = static_cast<float>(fn());
    double t1 = now_us();
    times.push_back(t1 - t0);
  }
  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// ---------------------------------------------------------------------------
// CPU-sequential reference (the first M4.3 implementation)
// ---------------------------------------------------------------------------

static float cpu_sum(const float* p, dim_t n) {
  metal::commit_and_wait();          // flush any pending GPU writes
  return std::accumulate(p, p + n, 0.f);
}

static float cpu_amax(const float* p, dim_t n) {
  metal::commit_and_wait();
  float result = 0.f;
  for (dim_t i = 0; i < n; ++i) {
    float f = p[i] < 0.f ? -p[i] : p[i];
    if (f > result) { result = f; }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::MPS>().free(p);
}

// ---------------------------------------------------------------------------
// Benchmark: sum (float32)
// ---------------------------------------------------------------------------

static void bench_sum_float() {
  std::printf("\n=== sum<float32> ===\n");
  std::printf("%-14s  %10s  %10s  %10s  %6s  %s\n",
              "N (elements)", "GPU (μs)", "CPU (μs)", "Ratio", "Match", "");
  std::printf("%s\n", std::string(70, '-').c_str());

  static const dim_t sizes[] = {
    256, 512, 1024, 4096, 16384, 65536, 262144,
    1 << 20,   // 1M
    4 << 20,   // 4M
    16 << 20,  // 16M
  };

  for (dim_t N : sizes) {
    float* p = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { p[i] = 1.f; }  // sum == N

    // Warmup (also compiles the PSO on first call)
    sink_f = primitives<Device::MPS>::sum(p, N);
    sink_f = cpu_sum(p, N);

    // Choose iteration count so total bench time stays reasonable
    int iters = (N <= 4096) ? 200 : (N <= 262144) ? 50 : (N <= (4 << 20)) ? 20 : 10;

    double gpu_us = bench_median_us(iters, [&] {
      return primitives<Device::MPS>::sum(p, N);
    });
    double cpu_us = bench_median_us(iters, [&] {
      return cpu_sum(p, N);
    });

    float gpu_result = primitives<Device::MPS>::sum(p, N);
    float cpu_result = cpu_sum(p, N);
    bool match = std::fabs(gpu_result - cpu_result) < 1e-1f * N;  // relative tol

    double ratio = cpu_us / gpu_us;
    const char* winner = (gpu_us < cpu_us) ? "GPU wins" : "CPU wins";

    std::printf("%-14lld  %10.1f  %10.1f  %9.2fx  %6s  %s\n",
                (long long)N, gpu_us, cpu_us, ratio,
                match ? "OK" : "MISMATCH", winner);

    metal_free(p);
  }
}

// ---------------------------------------------------------------------------
// Benchmark: amax (float32)
// ---------------------------------------------------------------------------

static void bench_amax_float() {
  std::printf("\n=== amax<float32> ===\n");
  std::printf("%-14s  %10s  %10s  %10s  %6s  %s\n",
              "N (elements)", "GPU (μs)", "CPU (μs)", "Ratio", "Match", "");
  std::printf("%s\n", std::string(70, '-').c_str());

  static const dim_t sizes[] = {
    256, 1024, 16384, 65536, 262144,
    1 << 20,
    4 << 20,
    16 << 20,
  };

  for (dim_t N : sizes) {
    float* p = metal_alloc<float>(N);
    // |p[i]| = i+1 for all i, so amax = N (last element has largest magnitude)
    for (dim_t i = 0; i < N; ++i) {
      p[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
    }
    float expected_amax = static_cast<float>(N);  // |p[N-1]| = N

    sink_f = primitives<Device::MPS>::amax(p, N);
    sink_f = cpu_amax(p, N);

    int iters = (N <= 4096) ? 200 : (N <= 262144) ? 50 : (N <= (4 << 20)) ? 20 : 10;

    double gpu_us = bench_median_us(iters, [&] {
      return primitives<Device::MPS>::amax(p, N);
    });
    double cpu_us = bench_median_us(iters, [&] {
      return cpu_amax(p, N);
    });

    float gpu_result = primitives<Device::MPS>::amax(p, N);
    float cpu_result = cpu_amax(p, N);
    bool match = std::fabs(gpu_result - expected_amax) < 1.f
              && std::fabs(cpu_result - expected_amax) < 1.f;

    double ratio = cpu_us / gpu_us;
    const char* winner = (gpu_us < cpu_us) ? "GPU wins" : "CPU wins";

    std::printf("%-14lld  %10.1f  %10.1f  %9.2fx  %6s  %s\n",
                (long long)N, gpu_us, cpu_us, ratio,
                match ? "OK" : "MISMATCH", winner);

    metal_free(p);
  }
}

// ---------------------------------------------------------------------------
// Benchmark: sum — GPU-pipeline scenario
//   Encode a GPU add_scalar before calling sum, so commit_and_wait inside
//   sum must flush real GPU work (not just an empty command buffer).
// ---------------------------------------------------------------------------

static void bench_sum_after_gpu_op() {
  std::printf("\n=== sum<float32> after a pending GPU add (realistic pipeline) ===\n");
  std::printf("%-14s  %10s  %10s  %10s  %s\n",
              "N (elements)", "GPU (μs)", "CPU (μs)", "Ratio", "");
  std::printf("%s\n", std::string(60, '-').c_str());

  static const dim_t sizes[] = {
    256, 4096, 65536, 262144, 1 << 20, 4 << 20, 16 << 20,
  };

  for (dim_t N : sizes) {
    float* x   = metal_alloc<float>(N);
    float* out = metal_alloc<float>(N);
    for (dim_t i = 0; i < N; ++i) { x[i] = 1.f; }

    // Warmup
    primitives<Device::MPS>::add(0.f, x, out, N);
    sink_f = primitives<Device::MPS>::sum(out, N);
    primitives<Device::MPS>::add(0.f, x, out, N);
    sink_f = cpu_sum(out, N);

    int iters = (N <= 4096) ? 200 : (N <= 262144) ? 50 : (N <= (4 << 20)) ? 20 : 10;

    // GPU path: encode add, then sum (sum internally commits both)
    double gpu_us = bench_median_us(iters, [&] {
      primitives<Device::MPS>::add(1.f, x, out, N);
      return primitives<Device::MPS>::sum(out, N);
    });

    // CPU path: encode add, then CPU sum (cpu_sum commits)
    double cpu_us = bench_median_us(iters, [&] {
      primitives<Device::MPS>::add(1.f, x, out, N);
      return cpu_sum(out, N);
    });

    double ratio = cpu_us / gpu_us;
    const char* winner = (gpu_us < cpu_us) ? "GPU wins" : "CPU wins";
    std::printf("%-14lld  %10.1f  %10.1f  %9.2fx  %s\n",
                (long long)N, gpu_us, cpu_us, ratio, winner);

    metal_free(x);
    metal_free(out);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.3 Reduction Benchmark: GPU two-pass vs CPU-sequential ===\n");
  std::printf("Hardware: Apple Silicon (Metal, unified memory)\n");
  std::printf("Metric: median latency over repeated calls (μs)\n");
  std::printf("Ratio: CPU_time / GPU_time  (>1.0 = GPU wins)\n");

  bench_sum_float();
  bench_amax_float();
  bench_sum_after_gpu_op();

  std::printf("\nDone.\n");
  return 0;
}
