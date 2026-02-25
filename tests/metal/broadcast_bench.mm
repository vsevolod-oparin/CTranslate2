// Benchmark + accuracy validation for M4.6 Metal broadcast primitives.
//
// Two sections per operator:
//   1. Accuracy — GPU result vs CPU reference on N=10000 output elements
//      (random float32 a and b); reports max absolute difference and PASS/FAIL.
//   2. Performance — median latency (μs) for GPU (encode+flush) vs CPU
//      sequential loop across several b_size values with a fixed a_size.
//
// Run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/broadcast_bench.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o broadcast_bench && ./broadcast_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

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
// Allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

// ---------------------------------------------------------------------------
// Reproducible random floats (xorshift32)
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
// Accuracy check
// ---------------------------------------------------------------------------

struct AccuracyResult {
  float max_abs_diff;
  float rms_diff;
  bool  all_finite;
};

// MetalOp: fn(const float* a, const float* b, float* c, dim_t a_size, dim_t b_size)
// CpuOp:  fn(const float* a, const float* b, float* c, dim_t a_size, dim_t b_size)
// b_size must be a multiple of a_size (guaranteed by callers below).
using BroadcastFn = std::function<void(const float*, const float*, float*, dim_t, dim_t)>;

static AccuracyResult check_accuracy(dim_t a_size, dim_t b_size,
                                     BroadcastFn metal_op,
                                     BroadcastFn cpu_op) {
  rng_state = 0xDEADBEEFu;

  float* a_host = new float[a_size];
  float* b_host = new float[b_size];
  for (dim_t i = 0; i < a_size; ++i) a_host[i] = next_float(-4.f, 4.f);
  for (dim_t i = 0; i < b_size; ++i) b_host[i] = next_float(-4.f, 4.f);

  // GPU path
  float* a_gpu = metal_alloc<float>(a_size);
  float* b_gpu = metal_alloc<float>(b_size);
  float* c_gpu = metal_alloc<float>(b_size);
  std::memcpy(a_gpu, a_host, a_size * sizeof(float));
  std::memcpy(b_gpu, b_host, b_size * sizeof(float));
  metal_op(a_gpu, b_gpu, c_gpu, a_size, b_size);
  metal::commit_and_wait();

  // CPU reference path
  float* c_cpu = new float[b_size];
  cpu_op(a_host, b_host, c_cpu, a_size, b_size);

  // Statistics
  float max_abs = 0.f, sum_sq = 0.f;
  bool  all_fin = true;
  for (dim_t i = 0; i < b_size; ++i) {
    float g = c_gpu[i], c = c_cpu[i];
    if (!std::isfinite(g) || !std::isfinite(c)) { all_fin = false; continue; }
    float diff = std::fabs(g - c);
    if (diff > max_abs) max_abs = diff;
    sum_sq += diff * diff;
  }
  float rms = std::sqrt(sum_sq / static_cast<float>(b_size));

  metal_free(a_gpu); metal_free(b_gpu); metal_free(c_gpu);
  delete[] a_host; delete[] b_host; delete[] c_cpu;

  return {max_abs, rms, all_fin};
}

// ---------------------------------------------------------------------------
// CPU reference functions (match the CPU primitives.cc formulas exactly)
// ---------------------------------------------------------------------------

static void cpu_add_batch_broadcast(const float* a, const float* b, float* c,
                                    dim_t a_size, dim_t b_size) {
  dim_t iter = b_size / a_size;
  for (dim_t i = 0; i < iter; ++i)
    for (dim_t j = 0; j < a_size; ++j)
      c[i * a_size + j] = a[j] + b[i * a_size + j];
}

static void cpu_add_depth_broadcast(const float* a, const float* b, float* c,
                                    dim_t a_size, dim_t b_size) {
  dim_t depth = b_size / a_size;
  for (dim_t i = 0; i < a_size; ++i)
    for (dim_t k = 0; k < depth; ++k)
      c[i * depth + k] = a[i] + b[i * depth + k];
}

// block_broadcast needs block; we bake it in via a lambda at each call site.
// For the accuracy check we use block = a_size (degenerates to depth_broadcast
// logic) so the reference can share the BroadcastFn signature.
static void cpu_add_block_broadcast(const float* a, const float* b, float* c,
                                    dim_t block, dim_t a_size, dim_t b_size) {
  dim_t num_blocks = b_size / block;
  for (dim_t i = 0; i < num_blocks; ++i) {
    float ai = a[i % a_size];
    for (dim_t k = 0; k < block; ++k)
      c[i * block + k] = ai + b[i * block + k];
  }
}

static void cpu_mul_batch_broadcast(const float* a, const float* b, float* c,
                                    dim_t a_size, dim_t b_size) {
  dim_t iter = b_size / a_size;
  for (dim_t i = 0; i < iter; ++i)
    for (dim_t j = 0; j < a_size; ++j)
      c[i * a_size + j] = a[j] * b[i * a_size + j];
}

// ---------------------------------------------------------------------------
// Accuracy section
// ---------------------------------------------------------------------------

static void run_accuracy() {
  // Accuracy check parameters:
  //   a_size = 100, b_size = 10000  → 100 iterations  (batch/mul broadcasts)
  //   block  = 50, a_size = 20, b_size = 10000         (block broadcast)
  const dim_t a_size  = 100;
  const dim_t b_size  = 10000;  // b_size % a_size == 0
  const dim_t block   = 50;     // b_size % block == 0, (b_size/block) % a_size == 0
  const float tol     = 1e-5f;

  std::printf("=== Accuracy: GPU vs CPU reference (a_size=%lld, b_size=%lld random float32) ===\n",
              (long long)a_size, (long long)b_size);
  std::printf("%-22s  %14s  %14s  %6s\n", "Op", "max_abs_diff", "rms_diff", "Status");
  std::printf("%s\n", std::string(62, '-').c_str());

  struct Spec {
    std::string        name;
    BroadcastFn        metal_op;
    BroadcastFn        cpu_op;
  };

  std::vector<Spec> specs = {
    { "add_batch_broadcast",
      [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
        primitives<Device::METAL>::add_batch_broadcast(a, b, c, as, bs); },
      cpu_add_batch_broadcast },

    { "add_depth_broadcast",
      [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
        primitives<Device::METAL>::add_depth_broadcast(a, b, c, as, bs); },
      cpu_add_depth_broadcast },

    { "add_block_broadcast",
      // block baked in via capture
      [block](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
        primitives<Device::METAL>::add_block_broadcast(a, b, c, block, as, bs); },
      [block](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
        cpu_add_block_broadcast(a, b, c, block, as, bs); } },

    { "mul_batch_broadcast",
      [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
        primitives<Device::METAL>::mul_batch_broadcast(a, b, c, as, bs); },
      cpu_mul_batch_broadcast },
  };

  int pass = 0, fail = 0;
  for (const auto& s : specs) {
    dim_t used_a_size = (s.name == "add_block_broadcast") ? a_size : a_size;
    AccuracyResult r = check_accuracy(used_a_size, b_size, s.metal_op, s.cpu_op);
    bool ok = r.all_finite && (r.max_abs_diff <= tol);
    std::printf("%-22s  %14.3e  %14.3e  %6s\n",
                s.name.c_str(), r.max_abs_diff, r.rms_diff,
                ok ? "PASS" : "FAIL");
    ok ? ++pass : ++fail;
  }
  std::printf("\nAccuracy: %d pass, %d fail\n", pass, fail);
}

// ---------------------------------------------------------------------------
// Performance section
// ---------------------------------------------------------------------------

// BroadcastBenchFn: fn(float* a, float* b, float* c, dim_t a_size, dim_t b_size)
// a_size is fixed; b_size varies.
static void bench_broadcast_op(const std::string& name, dim_t a_size,
                                BroadcastFn metal_op, BroadcastFn cpu_op) {
  // b_size values must be multiples of a_size.
  // We pick a_size = 1024 and b_sizes that are multiples.
  static const dim_t raw_sizes[] = {
    1 << 10,   // 1K
    1 << 12,   // 4K
    1 << 16,   // 64K
    1 << 20,   // 1M
    4 << 20,   // 4M
    16 << 20,  // 16M
  };

  std::printf("\n--- %s (a_size=%lld) ---\n", name.c_str(), (long long)a_size);
  std::printf("%-14s  %10s  %10s  %9s  %s\n",
              "b_size", "GPU (μs)", "CPU (μs)", "Ratio", "");
  std::printf("%s\n", std::string(60, '-').c_str());

  for (dim_t raw : raw_sizes) {
    // Round up to nearest multiple of a_size.
    dim_t b_size = ((raw + a_size - 1) / a_size) * a_size;

    float* a = metal_alloc<float>(a_size);
    float* b = metal_alloc<float>(b_size);
    float* c = metal_alloc<float>(b_size);

    rng_state = 0xABCDEF01u;
    for (dim_t i = 0; i < a_size; ++i) a[i] = next_float(-4.f, 4.f);
    for (dim_t i = 0; i < b_size; ++i) b[i] = next_float(-4.f, 4.f);

    // Warmup + PSO compile
    metal_op(a, b, c, a_size, b_size);
    metal::commit_and_wait();
    cpu_op(a, b, c, a_size, b_size);

    int iters = (b_size <= 4096) ? 200 : (b_size <= 65536) ? 50
              : (b_size <= (1 << 20)) ? 20 : 8;

    double gpu_us = bench_median_us(iters, [&] {
      metal_op(a, b, c, a_size, b_size);
      metal::commit_and_wait();
      return c[0];
    });

    double cpu_us = bench_median_us(iters, [&] {
      cpu_op(a, b, c, a_size, b_size);
      return c[0];
    });

    double ratio   = cpu_us / gpu_us;
    const char* winner = (gpu_us < cpu_us) ? "GPU wins" : "CPU wins";
    std::printf("%-14lld  %10.1f  %10.1f  %8.2fx  %s\n",
                (long long)b_size, gpu_us, cpu_us, ratio, winner);

    metal_free(a); metal_free(b); metal_free(c);
  }
}

static void run_perf() {
  std::printf("\n=== Performance: GPU (encode+flush) vs CPU sequential ===\n");
  std::printf("Ratio = CPU_time / GPU_time  (>1 = GPU wins)\n");

  const dim_t a_size = 1024;   // typical transformer hidden dim
  const dim_t block  = 64;     // typical head dim for add_block_broadcast

  bench_broadcast_op("add_batch_broadcast", a_size,
    [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
      primitives<Device::METAL>::add_batch_broadcast(a, b, c, as, bs); },
    cpu_add_batch_broadcast);

  bench_broadcast_op("add_depth_broadcast", a_size,
    [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
      primitives<Device::METAL>::add_depth_broadcast(a, b, c, as, bs); },
    cpu_add_depth_broadcast);

  // add_block_broadcast with fixed block size; a_size = b_size / block per row
  // Use block=64, a_size=16 so b_size stays a multiple of both.
  bench_broadcast_op("add_block_broadcast", /*a_size=*/16,
    [block](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
      primitives<Device::METAL>::add_block_broadcast(a, b, c, block, as, bs); },
    [block](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
      cpu_add_block_broadcast(a, b, c, block, as, bs); });

  bench_broadcast_op("mul_batch_broadcast", a_size,
    [](const float* a, const float* b, float* c, dim_t as, dim_t bs) {
      primitives<Device::METAL>::mul_batch_broadcast(a, b, c, as, bs); },
    cpu_mul_batch_broadcast);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.6 Broadcast Primitives: Accuracy & Performance ===\n");
  std::printf("Hardware: Apple Silicon (Metal, unified memory)\n");
  std::printf("Accuracy: a_size=100, b_size=10000 random float32 values.\n");
  std::printf("Performance: median latency (μs); GPU includes commit_and_wait.\n");

  run_accuracy();
  run_perf();

  std::printf("\nDone.\n");
  return 0;
}
