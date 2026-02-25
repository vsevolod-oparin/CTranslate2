// Benchmark + accuracy validation for M4.8 Metal transpose primitives.
//
// Sections:
//   1. Accuracy — GPU vs CPU float32 for all rank/perm variants.
//   2. Performance — GPU vs CPU across sizes for the critical MHA perm [0,2,1,3]
//      and general 2D/3D/4D permutations.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/transpose_bench.mm \
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
//     -o transpose_bench && ./transpose_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static volatile float g_sink = 0.f;

template <typename Fn>
static double bench_median_us(int iters, Fn fn) {
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    g_sink = fn();
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
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

template <typename T>
static void cpu_transpose_2d(const T* a, const dim_t* dims, T* b) {
  for (dim_t r = 0; r < dims[0]; ++r)
    for (dim_t c = 0; c < dims[1]; ++c)
      b[c * dims[0] + r] = a[r * dims[1] + c];
}

template <typename T>
static void cpu_transpose_3d(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
  dim_t a_stride[3] = {dims[1]*dims[2], dims[2], 1};
  dim_t b_dims[3]   = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
  dim_t b_stride[3] = {b_dims[1]*b_dims[2], b_dims[2], 1};
  dim_t perm_ind[3];
  for (int i = 0; i < 3; ++i) perm_ind[perm[i]] = i;
  dim_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]], b_stride[perm_ind[2]]};
  for (dim_t i0 = 0; i0 < dims[0]; ++i0)
    for (dim_t i1 = 0; i1 < dims[1]; ++i1)
      for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
        dim_t b_i = i0*perm_b_stride[0] + i1*perm_b_stride[1] + i2*perm_b_stride[2];
        dim_t a_i = i0*a_stride[0]      + i1*a_stride[1]      + i2*a_stride[2];
        b[b_i] = a[a_i];
      }
}

template <typename T>
static void cpu_transpose_4d(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
  dim_t a_stride[4] = {dims[1]*dims[2]*dims[3], dims[2]*dims[3], dims[3], 1};
  dim_t b_dims[4]   = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
  dim_t b_stride[4] = {b_dims[1]*b_dims[2]*b_dims[3], b_dims[2]*b_dims[3], b_dims[3], 1};
  dim_t perm_ind[4];
  for (int i = 0; i < 4; ++i) perm_ind[perm[i]] = i;
  dim_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                             b_stride[perm_ind[2]], b_stride[perm_ind[3]]};
  for (dim_t i0 = 0; i0 < dims[0]; ++i0)
    for (dim_t i1 = 0; i1 < dims[1]; ++i1)
      for (dim_t i2 = 0; i2 < dims[2]; ++i2)
        for (dim_t i3 = 0; i3 < dims[3]; ++i3) {
          dim_t b_i = i0*perm_b_stride[0] + i1*perm_b_stride[1]
                    + i2*perm_b_stride[2] + i3*perm_b_stride[3];
          dim_t a_i = i0*a_stride[0] + i1*a_stride[1]
                    + i2*a_stride[2] + i3*a_stride[3];
          b[b_i] = a[a_i];
        }
}

// ---------------------------------------------------------------------------
// Section 1: accuracy
// ---------------------------------------------------------------------------

static void check_accuracy_2d(dim_t rows, dim_t cols) {
  dim_t n = rows * cols;
  dim_t dims[2] = {rows, cols};

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 97) - 48.f;

  primitives<Device::METAL>::transpose_2d(d_a, dims, d_b);
  metal::commit_and_wait();
  cpu_transpose_2d(d_a, dims, cpu_b.data());

  float max_diff = 0.f;
  for (dim_t i = 0; i < n; ++i)
    max_diff = std::max(max_diff, std::fabs(d_b[i] - cpu_b[i]));

  std::printf("  transpose_2d [%lld×%lld]  max_diff=%.1e  %s\n",
              (long long)rows, (long long)cols, (double)max_diff,
              max_diff == 0.f ? "PASS" : "FAIL");
  metal_free(d_a); metal_free(d_b);
}

static void check_accuracy_3d(const dim_t* dims, const dim_t* perm, const char* label) {
  dim_t n = dims[0] * dims[1] * dims[2];
  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 97) - 48.f;

  primitives<Device::METAL>::transpose_3d(d_a, dims, perm, d_b);
  metal::commit_and_wait();
  cpu_transpose_3d(d_a, dims, perm, cpu_b.data());

  float max_diff = 0.f;
  for (dim_t i = 0; i < n; ++i)
    max_diff = std::max(max_diff, std::fabs(d_b[i] - cpu_b[i]));

  std::printf("  transpose_3d %s  max_diff=%.1e  %s\n",
              label, (double)max_diff, max_diff == 0.f ? "PASS" : "FAIL");
  metal_free(d_a); metal_free(d_b);
}

static void check_accuracy_4d(const dim_t* dims, const dim_t* perm, const char* label) {
  dim_t n = dims[0] * dims[1] * dims[2] * dims[3];
  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 97) - 48.f;

  primitives<Device::METAL>::transpose_4d(d_a, dims, perm, d_b);
  metal::commit_and_wait();
  cpu_transpose_4d(d_a, dims, perm, cpu_b.data());

  float max_diff = 0.f;
  for (dim_t i = 0; i < n; ++i)
    max_diff = std::max(max_diff, std::fabs(d_b[i] - cpu_b[i]));

  std::printf("  transpose_4d %s  max_diff=%.1e  %s\n",
              label, (double)max_diff, max_diff == 0.f ? "PASS" : "FAIL");
  metal_free(d_a); metal_free(d_b);
}

static void run_accuracy() {
  std::printf("=== 1. Accuracy: GPU vs CPU float32 ===\n\n");

  check_accuracy_2d(64, 128);
  check_accuracy_2d(512, 512);
  check_accuracy_2d(1, 1024);

  const dim_t dims3[3] = {8, 16, 32};
  const dim_t p021[3]  = {0, 2, 1};
  const dim_t p102[3]  = {1, 0, 2};
  const dim_t p210[3]  = {2, 1, 0};
  const dim_t p120[3]  = {1, 2, 0};
  check_accuracy_3d(dims3, p021, "[8,16,32] perm=[0,2,1]");
  check_accuracy_3d(dims3, p102, "[8,16,32] perm=[1,0,2]");
  check_accuracy_3d(dims3, p210, "[8,16,32] perm=[2,1,0]");
  check_accuracy_3d(dims3, p120, "[8,16,32] perm=[1,2,0]");

  const dim_t dims4a[4] = {2, 8, 32, 64};     // typical MHA: batch, heads, seq, dim
  const dim_t dims4b[4] = {4, 12, 128, 64};   // larger MHA
  const dim_t p0213[4]  = {0, 2, 1, 3};       // MHA head split
  const dim_t p0132[4]  = {0, 1, 3, 2};
  const dim_t p3210[4]  = {3, 2, 1, 0};
  const dim_t p1032[4]  = {1, 0, 3, 2};
  check_accuracy_4d(dims4a, p0213, "[2,8,32,64] perm=[0,2,1,3]");
  check_accuracy_4d(dims4b, p0213, "[4,12,128,64] perm=[0,2,1,3]");
  check_accuracy_4d(dims4a, p0132, "[2,8,32,64] perm=[0,1,3,2]");
  check_accuracy_4d(dims4a, p3210, "[2,8,32,64] perm=[3,2,1,0]");
  check_accuracy_4d(dims4a, p1032, "[2,8,32,64] perm=[1,0,3,2]");

  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 2: performance
// ---------------------------------------------------------------------------

static void bench_2d(const char* label, dim_t rows, dim_t cols, int iters) {
  dim_t n = rows * cols;
  dim_t dims[2] = {rows, cols};

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);
  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)i;

  // Warmup + PSO compile
  primitives<Device::METAL>::transpose_2d(d_a, dims, d_b);
  metal::commit_and_wait();
  cpu_transpose_2d(d_a, dims, cpu_b.data());

  double gpu_us = bench_median_us(iters, [&] {
    primitives<Device::METAL>::transpose_2d(d_a, dims, d_b);
    metal::commit_and_wait();
    return d_b[0];
  });
  double cpu_us = bench_median_us(iters, [&] {
    cpu_transpose_2d(d_a, dims, cpu_b.data());
    return cpu_b[0];
  });

  double ratio = cpu_us / gpu_us;
  std::printf("%-40s  %8.1f  %8.1f  %6.2fx  %s wins\n",
              label, gpu_us, cpu_us, ratio, gpu_us <= cpu_us ? "GPU" : "CPU");
  metal_free(d_a); metal_free(d_b);
}

static void bench_3d(const char* label, const dim_t* dims, const dim_t* perm, int iters) {
  dim_t n = dims[0] * dims[1] * dims[2];
  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);
  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)i;

  primitives<Device::METAL>::transpose_3d(d_a, dims, perm, d_b);
  metal::commit_and_wait();
  cpu_transpose_3d(d_a, dims, perm, cpu_b.data());

  double gpu_us = bench_median_us(iters, [&] {
    primitives<Device::METAL>::transpose_3d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    return d_b[0];
  });
  double cpu_us = bench_median_us(iters, [&] {
    cpu_transpose_3d(d_a, dims, perm, cpu_b.data());
    return cpu_b[0];
  });

  double ratio = cpu_us / gpu_us;
  std::printf("%-40s  %8.1f  %8.1f  %6.2fx  %s wins\n",
              label, gpu_us, cpu_us, ratio, gpu_us <= cpu_us ? "GPU" : "CPU");
  metal_free(d_a); metal_free(d_b);
}

static void bench_4d(const char* label, const dim_t* dims, const dim_t* perm, int iters) {
  dim_t n = dims[0] * dims[1] * dims[2] * dims[3];
  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);
  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)i;

  primitives<Device::METAL>::transpose_4d(d_a, dims, perm, d_b);
  metal::commit_and_wait();
  cpu_transpose_4d(d_a, dims, perm, cpu_b.data());

  double gpu_us = bench_median_us(iters, [&] {
    primitives<Device::METAL>::transpose_4d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    return d_b[0];
  });
  double cpu_us = bench_median_us(iters, [&] {
    cpu_transpose_4d(d_a, dims, perm, cpu_b.data());
    return cpu_b[0];
  });

  double ratio = cpu_us / gpu_us;
  std::printf("%-40s  %8.1f  %8.1f  %6.2fx  %s wins\n",
              label, gpu_us, cpu_us, ratio, gpu_us <= cpu_us ? "GPU" : "CPU");
  metal_free(d_a); metal_free(d_b);
}

static void run_perf() {
  std::printf("=== 2. Performance: GPU (encode+flush) vs CPU ===\n");
  std::printf("Ratio = CPU_time / GPU_time  (>1 = GPU wins)\n\n");
  std::printf("%-40s  %8s  %8s  %7s  %s\n",
              "Config", "GPU(μs)", "CPU(μs)", "Ratio", "Winner");
  std::printf("%s\n", std::string(76, '-').c_str());

  // 2D transpose
  bench_2d("2d [512×512]",         512,   512,  60);
  bench_2d("2d [1024×1024]",      1024,  1024,  30);
  bench_2d("2d [4096×256]",       4096,   256,  30);
  bench_2d("2d [65536×64]",      65536,    64,  20);
  bench_2d("2d [512×32768]",       512, 32768,  20);

  std::printf("\n");

  // 3D transpose
  const dim_t p021[3] = {0, 2, 1};
  const dim_t p210[3] = {2, 1, 0};
  const dim_t dims3a[3] = {8, 512, 64};    // typical: heads, seq, dim
  const dim_t dims3b[3] = {32, 1024, 128};
  const dim_t dims3c[3] = {64, 2048, 256};
  bench_3d("3d [8,512,64]    perm=[0,2,1]", dims3a, p021, 40);
  bench_3d("3d [32,1024,128] perm=[0,2,1]", dims3b, p021, 20);
  bench_3d("3d [64,2048,256] perm=[2,1,0]", dims3c, p210, 10);

  std::printf("\n");

  // 4D transpose — MHA perm [0,2,1,3]
  const dim_t p0213[4] = {0, 2, 1, 3};
  const dim_t p0132[4] = {0, 1, 3, 2};
  const dim_t p3210[4] = {3, 2, 1, 0};

  // Realistic MHA: [batch, heads, seq, head_dim]
  const dim_t mha_small[4]  = {1,  8,  128,  64};
  const dim_t mha_medium[4] = {1,  8,  512,  64};
  const dim_t mha_large[4]  = {1, 32, 2048,  64};
  const dim_t mha_xl[4]     = {4,  8,  512, 128};

  bench_4d("4d [1,8,128,64]  perm=[0,2,1,3]",  mha_small,  p0213, 80);
  bench_4d("4d [1,8,512,64]  perm=[0,2,1,3]",  mha_medium, p0213, 40);
  bench_4d("4d [1,32,2048,64] perm=[0,2,1,3]", mha_large,  p0213, 10);
  bench_4d("4d [4,8,512,128] perm=[0,2,1,3]",  mha_xl,     p0213, 20);

  std::printf("\n");

  // Other 4D perms
  const dim_t dims4g[4] = {2, 8, 256, 64};
  bench_4d("4d [2,8,256,64]  perm=[0,1,3,2]",  dims4g, p0132, 40);
  bench_4d("4d [2,8,256,64]  perm=[3,2,1,0]",  dims4g, p3210, 40);

  std::printf("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.8 Transpose Primitives: Accuracy & Performance ===\n");
  std::printf("Hardware: Apple Silicon (Metal, unified memory)\n");
  std::printf("GPU times include encode + commit_and_wait.\n");
  std::printf("In the full pipeline, encode-only kernels pay no sync overhead.\n\n");

  run_accuracy();
  run_perf();

  std::printf("Done.\n");
  return 0;
}
