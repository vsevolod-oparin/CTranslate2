// Benchmark + accuracy validation for M4.5 activation/transcendental primitives.
//
// Two sections per operator:
//   1. Random-input accuracy — GPU result vs CPU std:: reference on N=10000
//      random values; reports max absolute difference and PASS/FAIL.
//   2. Performance — median latency (μs) for GPU (encode+flush) vs CPU
//      sequential loop across several input sizes.
//
// Run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/activation_bench.mm \
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
//     -o activation_bench && ./activation_bench

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

// Returns median elapsed time in μs over `iters` calls.
// fn() must return a float so the result is stored into sink_f to prevent DCE.
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
// Reproducible random floats  (simple xorshift32 PRNG)
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
// Accuracy check infrastructure
// ---------------------------------------------------------------------------

// Check GPU results against CPU reference on N random inputs.
// MetalOp: fn(const float* x, float* y, N) — calls the Metal primitive.
// CpuRef:  fn(float x) → float               — scalar CPU reference.
// range [lo, hi] for random inputs.
// Returns: {max_abs_diff, rms_diff, all_finite}.
struct AccuracyResult {
  float max_abs_diff;
  float rms_diff;
  bool  all_finite;
};

template <typename MetalOp, typename CpuRef>
static AccuracyResult check_accuracy(dim_t N, float lo, float hi,
                                     MetalOp metal_op, CpuRef cpu_ref) {
  rng_state = 0xDEADBEEFu;  // reproducible

  float* x_host = new float[N];
  for (dim_t i = 0; i < N; ++i)
    x_host[i] = next_float(lo, hi);

  // GPU path
  float* x_gpu = metal_alloc<float>(N);
  float* y_gpu = metal_alloc<float>(N);
  std::memcpy(x_gpu, x_host, N * sizeof(float));
  metal_op(x_gpu, y_gpu, N);
  metal::commit_and_wait();

  // CPU reference path
  float* y_cpu = new float[N];
  for (dim_t i = 0; i < N; ++i)
    y_cpu[i] = cpu_ref(x_host[i]);

  // Compute statistics
  float max_abs = 0.f;
  float sum_sq  = 0.f;
  bool  all_fin = true;
  for (dim_t i = 0; i < N; ++i) {
    float g = y_gpu[i];
    float c = y_cpu[i];
    if (!std::isfinite(g) || !std::isfinite(c)) { all_fin = false; continue; }
    float diff = std::fabs(g - c);
    if (diff > max_abs) max_abs = diff;
    sum_sq += diff * diff;
  }
  float rms = std::sqrt(sum_sq / static_cast<float>(N));

  metal_free(x_gpu);
  metal_free(y_gpu);
  delete[] x_host;
  delete[] y_cpu;

  return {max_abs, rms, all_fin};
}

// ---------------------------------------------------------------------------
// CPU reference functions (matching the kernel formulas exactly)
// ---------------------------------------------------------------------------

static float cpu_relu(float v)         { return std::max(v, 0.f); }
static float cpu_sigmoid(float v)      { return 1.f / (1.f + std::exp(-v)); }
static float cpu_swish(float v)        { return v / (1.f + std::exp(-v)); }
static float cpu_gelu(float v) {
  return 0.5f * v * (1.f + std::erf(v * 0.7071067811865475f));
}
static float cpu_gelu_tanh(float v) {
  return 0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v*v*v)));
}
static float cpu_gelu_sigmoid(float v) { return v / (1.f + std::exp(-1.702f * v)); }

// ---------------------------------------------------------------------------
// Performance benchmark for one op
// ---------------------------------------------------------------------------

// MetalOp: void(const float* x, float* y, dim_t N)
// CpuOp:   void(const float* x, float* y, dim_t N)
static void bench_op(const std::string& name,
                     std::function<void(const float*, float*, dim_t)> metal_op,
                     std::function<void(const float*, float*, dim_t)> cpu_op) {
  static const dim_t sizes[] = {
    256, 4096, 65536,
    1 << 20,    // 1M
    4 << 20,    // 4M
    16 << 20,   // 16M
  };

  std::printf("\n--- %s ---\n", name.c_str());
  std::printf("%-14s  %10s  %10s  %9s  %s\n",
              "N", "GPU (μs)", "CPU (μs)", "Ratio", "");
  std::printf("%s\n", std::string(60, '-').c_str());

  for (dim_t N : sizes) {
    float* x = metal_alloc<float>(N);
    float* y = metal_alloc<float>(N);
    rng_state = 0xABCDEF01u;
    for (dim_t i = 0; i < N; ++i) x[i] = next_float(-4.f, 4.f);

    // Warmup (also compiles PSO on first call)
    metal_op(x, y, N);
    metal::commit_and_wait();
    cpu_op(x, y, N);

    int iters = (N <= 4096) ? 200 : (N <= 65536) ? 50 : (N <= (1 << 20)) ? 20 : 8;

    // GPU: encode + flush
    double gpu_us = bench_median_us(iters, [&] {
      metal_op(x, y, N);
      metal::commit_and_wait();
      return y[0];  // prevent DCE
    });

    // CPU: sequential loop
    double cpu_us = bench_median_us(iters, [&] {
      cpu_op(x, y, N);
      return y[0];
    });

    double ratio = cpu_us / gpu_us;
    const char* winner = (gpu_us < cpu_us) ? "GPU wins" : "CPU wins";
    std::printf("%-14lld  %10.1f  %10.1f  %8.2fx  %s\n",
                (long long)N, gpu_us, cpu_us, ratio, winner);

    metal_free(x);
    metal_free(y);
  }
}

// ---------------------------------------------------------------------------
// Accuracy section
// ---------------------------------------------------------------------------

struct OpSpec {
  std::string name;
  std::function<void(const float*, float*, dim_t)> metal_op;
  std::function<float(float)>                       cpu_ref;
  float lo, hi;         // input range for random values
  float tol_f32;        // acceptable max_abs_diff for float32
};

static void run_accuracy(const std::vector<OpSpec>& ops) {
  const dim_t N = 10000;
  std::printf("\n=== Accuracy: GPU vs CPU reference (N=%lld random inputs) ===\n",
              (long long)N);
  std::printf("%-16s  %14s  %14s  %6s\n",
              "Op", "max_abs_diff", "rms_diff", "Status");
  std::printf("%s\n", std::string(56, '-').c_str());

  int pass = 0, fail = 0;
  for (const auto& op : ops) {
    AccuracyResult r = check_accuracy(N, op.lo, op.hi, op.metal_op, op.cpu_ref);
    bool ok = r.all_finite && (r.max_abs_diff <= op.tol_f32);
    std::printf("%-16s  %14.3e  %14.3e  %6s\n",
                op.name.c_str(), r.max_abs_diff, r.rms_diff,
                ok ? "PASS" : "FAIL");
    ok ? ++pass : ++fail;
  }
  std::printf("\nAccuracy: %d pass, %d fail\n", pass, fail);
}

// ---------------------------------------------------------------------------
// Performance section
// ---------------------------------------------------------------------------

static void run_perf(const std::vector<OpSpec>& ops) {
  std::printf("\n=== Performance: GPU (encode+flush) vs CPU sequential ===\n");
  std::printf("Ratio = CPU_time / GPU_time  (>1 = GPU wins)\n");

  for (const auto& op : ops) {
    // Wrap cpu_ref scalar → vector form
    auto cpu_ref = op.cpu_ref;
    bench_op(op.name,
             op.metal_op,
             [cpu_ref](const float* x, float* y, dim_t N) {
               for (dim_t i = 0; i < N; ++i) y[i] = cpu_ref(x[i]);
             });
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.5 Activation Primitives: Accuracy & Performance ===\n");
  std::printf("Hardware: Apple Silicon (Metal, unified memory)\n");
  std::printf("Accuracy input range: shown per op.  N=10000 random float32 values.\n");
  std::printf("Performance metric: median latency (μs), GPU includes commit_and_wait.\n");

  // clang-format off
  std::vector<OpSpec> ops = {
    { "exp",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::exp(x, y, n); },
      [](float v){ return std::exp(v); },
      -4.f, 4.f, 2e-5f },  // MSL exp rounding vs std::exp can differ by ~1.2e-5
    { "log",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::log(x, y, n); },
      [](float v){ return std::log(v); },
      0.01f, 10.f, 1e-5f },
    { "cos",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::cos(x, y, n); },
      [](float v){ return std::cos(v); },
      -3.14159f, 3.14159f, 1e-5f },
    { "sin",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::sin(x, y, n); },
      [](float v){ return std::sin(v); },
      -3.14159f, 3.14159f, 1e-5f },
    { "tanh",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::tanh(x, y, n); },
      [](float v){ return std::tanh(v); },
      -4.f, 4.f, 1e-5f },
    { "relu",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::relu(x, y, n); },
      cpu_relu,
      -4.f, 4.f, 1e-6f },
    { "sigmoid",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::sigmoid(x, y, n); },
      cpu_sigmoid,
      -4.f, 4.f, 1e-6f },
    { "swish",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::swish(x, y, n); },
      cpu_swish,
      -4.f, 4.f, 1e-5f },
    { "gelu",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::gelu(x, y, n); },
      cpu_gelu,
      -4.f, 4.f, 2e-6f },  // ct2_erf poly error ≤ 1.5e-7; allow extra for fp32 rounding
    { "gelu_tanh",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::gelu_tanh(x, y, n); },
      cpu_gelu_tanh,
      -4.f, 4.f, 1e-5f },
    { "gelu_sigmoid",
      [](const float* x, float* y, dim_t n){ primitives<Device::METAL>::gelu_sigmoid(x, y, n); },
      cpu_gelu_sigmoid,
      -4.f, 4.f, 1e-6f },
  };
  // clang-format on

  run_accuracy(ops);
  run_perf(ops);

  std::printf("\nDone.\n");
  return 0;
}
