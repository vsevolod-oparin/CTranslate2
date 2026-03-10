// tests/metal/m52_bench.mm
//
// M5.2 Metal vs CPU — Accuracy and performance for all five M5.2 ops:
//   LayerNorm, RMSNorm, SoftMax, Gather, BiasAdd.
//
// For each op and shape, reports:
//   • max absolute error (Metal output vs C++ float32 reference) + PASS/FAIL
//   • GPU (encode+commit_and_wait) and CPU (single-threaded C++) latency in µs
//   • Speedup = CPU_µs / GPU_µs  (>1x → GPU wins)
//
// NOTE: GPU times include the ~0.4 ms command-buffer submission overhead.
//       In a real CTranslate2 encoder pipeline the command buffer is submitted
//       only at synchronize_stream(), so the overhead is amortised across many
//       operations.  Standalone GPU times therefore under-represent real
//       pipeline throughput; the true crossover point is lower than shown here.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/m52_bench.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o m52_bench && ./m52_bench

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
#include "ctranslate2/primitives.h"
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

// Adaptive iteration count based on total element count.
static int iter_count(dim_t total) {
  if (total <= 4096)    return 150;
  if (total <= 65536)   return 50;
  if (total <= 1048576) return 20;
  return 8;
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
// Simple xorshift32 PRNG for reproducible random inputs
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
  for (dim_t i = 0; i < n; ++i) err = std::max(err, std::fabs(ref[i] - got[i]));
  return err;
}

// ---------------------------------------------------------------------------
// C++ single-threaded reference implementations
// ---------------------------------------------------------------------------

// LayerNorm: (x - mean) / sqrt(var/N + eps) * gamma + beta
//   gamma/beta may be nullptr → identity scale, zero bias.
static void ref_layer_norm(const float* x, const float* gamma, const float* beta,
                           float* y, int outer, int N, float eps) {
  for (int r = 0; r < outer; ++r) {
    const float* xr = x + r * N;
    float* yr = y + r * N;
    float mean = 0.f;
    for (int j = 0; j < N; ++j) mean += xr[j];
    mean /= N;
    float var = 0.f;
    for (int j = 0; j < N; ++j) { float d = xr[j] - mean; var += d * d; }
    float inv_std = 1.f / std::sqrt(var / N + eps);
    for (int j = 0; j < N; ++j) {
      float g = gamma ? gamma[j] : 1.f;
      float b = beta  ? beta[j]  : 0.f;
      yr[j] = (xr[j] - mean) * inv_std * g + b;
    }
  }
}

// RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
static void ref_rms_norm(const float* x, const float* gamma,
                         float* y, int batch, int depth, float eps) {
  for (int r = 0; r < batch; ++r) {
    const float* xr = x + r * depth;
    float* yr = y + r * depth;
    float ss = 0.f;
    for (int j = 0; j < depth; ++j) ss += xr[j] * xr[j];
    float rms_inv = 1.f / std::sqrt(ss / depth + eps);
    for (int j = 0; j < depth; ++j)
      yr[j] = xr[j] * rms_inv * (gamma ? gamma[j] : 1.f);
  }
}

// SoftMax: subtract max for numerical stability
static void ref_softmax(const float* x, float* y, int batch, int depth, bool log_mode) {
  for (int r = 0; r < batch; ++r) {
    const float* xr = x + r * depth;
    float* yr = y + r * depth;
    float mx = -1e38f;
    for (int j = 0; j < depth; ++j) if (xr[j] > mx) mx = xr[j];
    float sum = 0.f;
    for (int j = 0; j < depth; ++j) sum += std::exp(xr[j] - mx);
    if (log_mode) {
      float log_sum = std::log(sum);
      for (int j = 0; j < depth; ++j) yr[j] = xr[j] - mx - log_sum;
    } else {
      for (int j = 0; j < depth; ++j) yr[j] = std::exp(xr[j] - mx) / sum;
    }
  }
}

// Gather: copy copy_size elements per index slot
static void ref_gather(const float* src, float* dst, const int32_t* indices,
                       dim_t copy_size, dim_t num_indices) {
  for (dim_t i = 0; i < num_indices; ++i)
    std::memcpy(dst + i * copy_size,
                src + (dim_t)indices[i] * copy_size,
                copy_size * sizeof(float));
}

// BiasAdd batch_broadcast: out[i] = x[i] + bias[i % bias_size]
static void ref_bias_batch(const float* x, const float* bias, float* out,
                           dim_t bias_size, dim_t total) {
  for (dim_t i = 0; i < total; ++i) out[i] = x[i] + bias[i % bias_size];
}

// BiasAdd block_broadcast: out[...b*ch*w + c*w + j] = x[...] + bias[c]
//   c = (flat_index / width) % bias_size
static void ref_bias_block(const float* x, const float* bias, float* out,
                           dim_t width, dim_t bias_size, dim_t total) {
  for (dim_t i = 0; i < total; ++i)
    out[i] = x[i] + bias[(i / width) % bias_size];
}

// ---------------------------------------------------------------------------
// Column header shared by all per-op tables
// ---------------------------------------------------------------------------

static void print_bench_header() {
  std::printf("  %-26s  %12s  %10s  %10s  %9s  %s\n",
              "Shape", "max_abs_err", "Status",
              "GPU (µs)", "CPU (µs)", "Speedup");
  std::printf("  %s\n", std::string(80, '-').c_str());
}

static void print_bench_row(const char* shape, float abs_err, float tol,
                            double gpu_us, double cpu_us) {
  bool ok = std::isfinite(abs_err) && (abs_err <= tol);
  ok ? ++g_pass : ++g_fail;
  double speedup = cpu_us / gpu_us;
  const char* winner = (speedup >= 1.0) ? "GPU wins" : "CPU wins";
  std::printf("  %-26s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
              shape, abs_err, ok ? "PASS" : "FAIL",
              gpu_us, cpu_us, speedup, winner);
}

// ---------------------------------------------------------------------------
// 1.  LayerNorm
// ---------------------------------------------------------------------------

static void bench_layer_norm() {
  std::printf("\n=== LayerNorm (float32, gamma+beta) ===\n");
  print_bench_header();

  struct S { int outer, N; };
  const S shapes[] = {{1,64},{4,256},{32,512},{256,1024},{512,4096}};

  for (auto& s : shapes) {
    const dim_t total = (dim_t)s.outer * s.N;
    rng_state = 0x11223344u;

    float* x_m = metal_alloc<float>(total);
    float* g_m = metal_alloc<float>(s.N);
    float* b_m = metal_alloc<float>(s.N);
    float* y_m = metal_alloc<float>(total);
    for (dim_t i = 0; i < total; ++i) x_m[i] = next_float(-2.f, 2.f);
    for (int j = 0; j < s.N; ++j) { g_m[j] = next_float(0.5f, 1.5f); b_m[j] = next_float(-0.1f, 0.1f); }

    // Warmup (also compiles PSO)
    metal::layer_norm_metal<float>(x_m, g_m, b_m, y_m, s.outer, s.N, 1e-5f);
    metal::commit_and_wait();

    // Accuracy: copy Metal input to CPU vectors
    std::vector<float> xc(total), gc(s.N), bc(s.N), yr(total);
    for (dim_t i = 0; i < total; ++i) xc[i] = x_m[i];
    for (int j = 0; j < s.N; ++j) { gc[j] = g_m[j]; bc[j] = b_m[j]; }
    ref_layer_norm(xc.data(), gc.data(), bc.data(), yr.data(), s.outer, s.N, 1e-5f);
    float err = max_abs_err(yr.data(), y_m, total);

    int iters = iter_count(total);
    double gpu_us = bench_median_us(iters, [&] {
      metal::layer_norm_metal<float>(x_m, g_m, b_m, y_m, s.outer, s.N, 1e-5f);
      metal::commit_and_wait();
      return y_m[0];
    });
    double cpu_us = bench_median_us(iters, [&] {
      ref_layer_norm(xc.data(), gc.data(), bc.data(), yr.data(), s.outer, s.N, 1e-5f);
      return yr[0];
    });

    char shape_buf[32];
    std::snprintf(shape_buf, sizeof(shape_buf), "[%d x %d]", s.outer, s.N);
    print_bench_row(shape_buf, err, 1e-5f, gpu_us, cpu_us);

    metal_free(x_m); metal_free(g_m); metal_free(b_m); metal_free(y_m);
  }
}

// ---------------------------------------------------------------------------
// 2.  RMSNorm
// ---------------------------------------------------------------------------

static void bench_rms_norm() {
  std::printf("\n=== RMSNorm (float32, with gamma) ===\n");
  print_bench_header();

  struct S { int batch, depth; };
  const S shapes[] = {{1,64},{4,256},{32,512},{256,1024},{512,4096}};

  for (auto& s : shapes) {
    const dim_t total = (dim_t)s.batch * s.depth;
    rng_state = 0x55667788u;

    float* x_m = metal_alloc<float>(total);
    float* g_m = metal_alloc<float>(s.depth);
    float* y_m = metal_alloc<float>(total);
    for (dim_t i = 0; i < total; ++i) x_m[i] = next_float(-2.f, 2.f);
    for (int j = 0; j < s.depth; ++j) g_m[j] = next_float(0.5f, 1.5f);

    // Warmup
    metal::rms_norm_metal<float>(x_m, g_m, y_m, s.batch, s.depth, 1e-6f);
    metal::commit_and_wait();

    std::vector<float> xc(total), gc(s.depth), yr(total);
    for (dim_t i = 0; i < total; ++i) xc[i] = x_m[i];
    for (int j = 0; j < s.depth; ++j) gc[j] = g_m[j];
    ref_rms_norm(xc.data(), gc.data(), yr.data(), s.batch, s.depth, 1e-6f);
    float err = max_abs_err(yr.data(), y_m, total);

    int iters = iter_count(total);
    double gpu_us = bench_median_us(iters, [&] {
      metal::rms_norm_metal<float>(x_m, g_m, y_m, s.batch, s.depth, 1e-6f);
      metal::commit_and_wait();
      return y_m[0];
    });
    double cpu_us = bench_median_us(iters, [&] {
      ref_rms_norm(xc.data(), gc.data(), yr.data(), s.batch, s.depth, 1e-6f);
      return yr[0];
    });

    char shape_buf[32];
    std::snprintf(shape_buf, sizeof(shape_buf), "[%d x %d]", s.batch, s.depth);
    print_bench_row(shape_buf, err, 1e-4f, gpu_us, cpu_us);

    metal_free(x_m); metal_free(g_m); metal_free(y_m);
  }
}

// ---------------------------------------------------------------------------
// 3.  SoftMax
// ---------------------------------------------------------------------------

static void bench_softmax() {
  std::printf("\n=== SoftMax (float32, no masking) ===\n");
  print_bench_header();

  struct S { int batch, depth; };
  const S shapes[] = {{1,64},{4,256},{32,512},{256,1024},{512,8192}};

  for (auto& s : shapes) {
    const dim_t total = (dim_t)s.batch * s.depth;
    rng_state = 0x99AABBCCu;

    float* x_m = metal_alloc<float>(total);
    float* y_m = metal_alloc<float>(total);
    for (dim_t i = 0; i < total; ++i) x_m[i] = next_float(-3.f, 3.f);

    // Warmup
    metal::softmax_metal<float>(x_m, nullptr, y_m, s.batch, s.depth, false);
    metal::commit_and_wait();

    std::vector<float> xc(total), yr(total);
    for (dim_t i = 0; i < total; ++i) xc[i] = x_m[i];
    ref_softmax(xc.data(), yr.data(), s.batch, s.depth, false);
    float err = max_abs_err(yr.data(), y_m, total);

    int iters = iter_count(total);
    double gpu_us = bench_median_us(iters, [&] {
      metal::softmax_metal<float>(x_m, nullptr, y_m, s.batch, s.depth, false);
      metal::commit_and_wait();
      return y_m[0];
    });
    double cpu_us = bench_median_us(iters, [&] {
      ref_softmax(xc.data(), yr.data(), s.batch, s.depth, false);
      return yr[0];
    });

    char shape_buf[32];
    std::snprintf(shape_buf, sizeof(shape_buf), "[%d x %d]", s.batch, s.depth);
    print_bench_row(shape_buf, err, 1e-5f, gpu_us, cpu_us);

    metal_free(x_m); metal_free(y_m);
  }
}

// ---------------------------------------------------------------------------
// 4.  Gather
// ---------------------------------------------------------------------------

static void bench_gather() {
  std::printf("\n=== Gather (float32) ===\n");
  // Column header with gather-specific shape notation
  std::printf("  %-26s  %12s  %10s  %10s  %10s  %9s  %s\n",
              "idx × copy (src_rows)",
              "max_abs_err", "Status",
              "GPU (µs)", "CPU (µs)", "Speedup", "");
  std::printf("  %s\n", std::string(90, '-').c_str());

  struct S { int num_idx, copy_size, src_rows; };
  const S shapes[] = {
    { 16,   64,   512},   // dst=1K   elements
    {256,  256,  4096},   // dst=64K
    {4096, 512, 32768},   // dst=2M
  };

  for (auto& s : shapes) {
    const dim_t src_total  = (dim_t)s.src_rows  * s.copy_size;
    const dim_t dst_total  = (dim_t)s.num_idx   * s.copy_size;
    rng_state = 0xDDEEFFAAu;

    float*   src = metal_alloc<float>(src_total);
    int32_t* idx = metal_alloc<int32_t>(s.num_idx);
    float*   dst = metal_alloc<float>(dst_total);
    for (dim_t i = 0; i < src_total; ++i) src[i] = next_float(-5.f, 5.f);
    for (int i = 0; i < s.num_idx; ++i)   idx[i]  = int32_t(i % s.src_rows);

    // gather_metal(src, dst, indices, copy_size, batch_stride,
    //              num_indices_per_batch, total_elements)
    // Warmup
    metal::gather_metal<float>(src, dst, idx,
                               s.copy_size, src_total, s.num_idx, dst_total);
    metal::commit_and_wait();

    std::vector<float> dst_cpu(dst_total);
    ref_gather(src, dst_cpu.data(), idx, s.copy_size, s.num_idx);
    float err = max_abs_err(dst_cpu.data(), dst, dst_total);

    int iters = iter_count(dst_total);
    double gpu_us = bench_median_us(iters, [&] {
      metal::gather_metal<float>(src, dst, idx,
                                 s.copy_size, src_total, s.num_idx, dst_total);
      metal::commit_and_wait();
      return dst[0];
    });
    double cpu_us = bench_median_us(iters, [&] {
      ref_gather(src, dst_cpu.data(), idx, s.copy_size, s.num_idx);
      return dst_cpu[0];
    });

    bool ok = std::isfinite(err) && (err == 0.f);
    ok ? ++g_pass : ++g_fail;
    double speedup = cpu_us / gpu_us;
    const char* winner = (speedup >= 1.0) ? "GPU wins" : "CPU wins";
    char shape_buf[64];
    std::snprintf(shape_buf, sizeof(shape_buf),
                  "%d x %d  (src=%d)", s.num_idx, s.copy_size, s.src_rows);
    std::printf("  %-26s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
                shape_buf, err, ok ? "PASS" : "FAIL",
                gpu_us, cpu_us, speedup, winner);

    metal_free(src); metal_free(idx); metal_free(dst);
  }
}

// ---------------------------------------------------------------------------
// 5.  BiasAdd — batch_broadcast and block_broadcast
// ---------------------------------------------------------------------------

static void bench_bias_add() {
  // 5a. batch_broadcast
  {
    std::printf("\n=== BiasAdd batch_broadcast (float32) ===\n");
    std::printf("  %-26s  %12s  %10s  %10s  %10s  %9s  %s\n",
                "total (bias_size)",
                "max_abs_err", "Status",
                "GPU (µs)", "CPU (µs)", "Speedup", "");
    std::printf("  %s\n", std::string(90, '-').c_str());

    struct S { dim_t total, bias_size; };
    const S shapes[] = {
      {    4096,  64},
      {   65536, 128},
      { 1048576, 256},
      { 4194304, 512},
    };

    for (auto& s : shapes) {
      rng_state = 0xAABBCCDDu;
      float* bias = metal_alloc<float>(s.bias_size);
      float* val  = metal_alloc<float>(s.total);
      float* out  = metal_alloc<float>(s.total);
      for (dim_t i = 0; i < s.bias_size; ++i) bias[i] = next_float(-1.f, 1.f);
      for (dim_t i = 0; i < s.total;     ++i) val[i]  = next_float(-5.f, 5.f);

      // Warmup
      primitives<Device::MPS>::add_batch_broadcast(bias, val, out, s.bias_size, s.total);
      metal::commit_and_wait();

      std::vector<float> out_cpu(s.total);
      ref_bias_batch(val, bias, out_cpu.data(), s.bias_size, s.total);
      float err = max_abs_err(out_cpu.data(), out, s.total);

      int iters = iter_count(s.total);
      double gpu_us = bench_median_us(iters, [&] {
        primitives<Device::MPS>::add_batch_broadcast(bias, val, out, s.bias_size, s.total);
        metal::commit_and_wait();
        return out[0];
      });
      double cpu_us = bench_median_us(iters, [&] {
        ref_bias_batch(val, bias, out_cpu.data(), s.bias_size, s.total);
        return out_cpu[0];
      });

      bool ok = std::isfinite(err) && (err < 1e-6f);
      ok ? ++g_pass : ++g_fail;
      double speedup = cpu_us / gpu_us;
      const char* winner = (speedup >= 1.0) ? "GPU wins" : "CPU wins";
      char shape_buf[48];
      std::snprintf(shape_buf, sizeof(shape_buf),
                    "%lld  (bias=%lld)", (long long)s.total, (long long)s.bias_size);
      std::printf("  %-26s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
                  shape_buf, err, ok ? "PASS" : "FAIL",
                  gpu_us, cpu_us, speedup, winner);

      metal_free(bias); metal_free(val); metal_free(out);
    }
  }

  // 5b. block_broadcast
  {
    std::printf("\n=== BiasAdd block_broadcast (float32) ===\n");
    std::printf("  %-26s  %12s  %10s  %10s  %10s  %9s  %s\n",
                "[batch, channels, width]",
                "max_abs_err", "Status",
                "GPU (µs)", "CPU (µs)", "Speedup", "");
    std::printf("  %s\n", std::string(90, '-').c_str());

    struct S { int batch, ch, w; };
    const S shapes[] = {
      {  4,  32,  64},   // total=8K
      { 16,  64, 256},   // total=256K
      { 64, 128, 512},   // total=4M
    };

    for (auto& s : shapes) {
      const dim_t total = (dim_t)s.batch * s.ch * s.w;
      rng_state = 0x11EE22FFu;

      float* bias = metal_alloc<float>(s.ch);
      float* val  = metal_alloc<float>(total);
      float* out  = metal_alloc<float>(total);
      for (int c = 0; c < s.ch;      ++c) bias[c] = next_float(-1.f, 1.f);
      for (dim_t i = 0; i < total;   ++i) val[i]  = next_float(-5.f, 5.f);

      // Warmup
      primitives<Device::MPS>::add_block_broadcast(bias, val, out, s.w, s.ch, total);
      metal::commit_and_wait();

      std::vector<float> out_cpu(total);
      ref_bias_block(val, bias, out_cpu.data(), s.w, s.ch, total);
      float err = max_abs_err(out_cpu.data(), out, total);

      int iters = iter_count(total);
      double gpu_us = bench_median_us(iters, [&] {
        primitives<Device::MPS>::add_block_broadcast(bias, val, out, s.w, s.ch, total);
        metal::commit_and_wait();
        return out[0];
      });
      double cpu_us = bench_median_us(iters, [&] {
        ref_bias_block(val, bias, out_cpu.data(), s.w, s.ch, total);
        return out_cpu[0];
      });

      bool ok = std::isfinite(err) && (err < 1e-6f);
      ok ? ++g_pass : ++g_fail;
      double speedup = cpu_us / gpu_us;
      const char* winner = (speedup >= 1.0) ? "GPU wins" : "CPU wins";
      char shape_buf[48];
      std::snprintf(shape_buf, sizeof(shape_buf), "[%d,%d,%d]", s.batch, s.ch, s.w);
      std::printf("  %-26s  %12.3e  %6s    %10.1f  %10.1f  %8.2fx  %s\n",
                  shape_buf, err, ok ? "PASS" : "FAIL",
                  gpu_us, cpu_us, speedup, winner);

      metal_free(bias); metal_free(val); metal_free(out);
    }
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M5.2 Metal vs CPU: Accuracy and Performance ===\n");
  std::printf("Timing: median of (3 warmup +) N timed runs; GPU includes encode+commit_and_wait.\n");
  std::printf("NOTE: CB overhead ~0.4 ms is amortised in real pipeline; standalone GPU times\n");
  std::printf("      are conservative — actual pipeline throughput is substantially higher.\n");

  bench_layer_norm();
  bench_rms_norm();
  bench_softmax();
  bench_gather();
  bench_bias_add();

  std::printf("\n");
  std::printf("=== Accuracy summary: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
