// tests/metal/topk_bench.mm
//
// GPU TopK benchmark — GPU iterative argmax vs CPU partial_sort.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/topk_bench.mm \
//     src/metal/ops_topk.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm \
//     src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm \
//     src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm \
//     src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -framework Accelerate \
//     -o topk_bench && ./topk_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
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

static constexpr int kWarmup = 5;
static constexpr int kIters  = 20;

// ---------------------------------------------------------------------------
// CPU reference: partial_sort
// ---------------------------------------------------------------------------

template <typename T>
static double bench_cpu_topk(const T* input, dim_t batch, dim_t depth, dim_t k) {
  std::vector<int32_t> ids(static_cast<std::size_t>(depth));
  std::vector<float> vals(batch * k);
  std::vector<int32_t> idxs(batch * k);

  // Warmup
  for (int w = 0; w < kWarmup; ++w) {
    for (dim_t b = 0; b < batch; ++b) {
      const T* row = input + b * depth;
      std::iota(ids.begin(), ids.end(), int32_t(0));
      std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
          [&row](int32_t i1, int32_t i2) {
            return static_cast<float>(row[i1]) > static_cast<float>(row[i2]);
          });
    }
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < kIters; ++it) {
    for (dim_t b = 0; b < batch; ++b) {
      const T* row = input + b * depth;
      std::iota(ids.begin(), ids.end(), int32_t(0));
      std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
          [&row](int32_t i1, int32_t i2) {
            return static_cast<float>(row[i1]) > static_cast<float>(row[i2]);
          });
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
}

// ---------------------------------------------------------------------------
// GPU: topk_metal with commit_and_wait
// ---------------------------------------------------------------------------

template <typename T>
static double bench_gpu_topk(const T* input, T* values, int32_t* indices,
                              dim_t batch, dim_t depth, dim_t k) {
  // Warmup
  for (int w = 0; w < kWarmup; ++w) {
    metal::topk_metal<T>(input, values, indices, batch, depth, k);
    synchronize_stream(Device::METAL);
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < kIters; ++it) {
    metal::topk_metal<T>(input, values, indices, batch, depth, k);
    synchronize_stream(Device::METAL);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / kIters;
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

template <typename T>
static void run_bench(const char* type_label, dim_t batch, dim_t depth, dim_t k) {
  auto& alloc = get_allocator<Device::METAL>();
  const size_t in_bytes  = batch * depth * sizeof(T);
  const size_t val_bytes = batch * k * sizeof(T);
  const size_t idx_bytes = batch * k * sizeof(int32_t);

  T*       in_ptr  = static_cast<T*>(alloc.allocate(in_bytes, 0));
  T*       val_ptr = static_cast<T*>(alloc.allocate(val_bytes, 0));
  int32_t* idx_ptr = static_cast<int32_t*>(alloc.allocate(idx_bytes, 0));

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (dim_t i = 0; i < batch * depth; ++i)
    in_ptr[i] = static_cast<T>(dist(rng));

  double cpu_us = bench_cpu_topk<T>(in_ptr, batch, depth, k);
  double gpu_us = bench_gpu_topk<T>(in_ptr, val_ptr, idx_ptr, batch, depth, k);
  double speedup = cpu_us / gpu_us;

  std::printf("  %-6s batch=%-2lld depth=%-6lld k=%-3lld | CPU: %8.0f µs | GPU: %8.0f µs | %.2fx\n",
              type_label, (long long)batch, (long long)depth, (long long)k,
              cpu_us, gpu_us, speedup);

  alloc.free(in_ptr, 0);
  alloc.free(val_ptr, 0);
  alloc.free(idx_ptr, 0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  @autoreleasepool {
    std::printf("=== TopK GPU Benchmark (encode + commit_and_wait) ===\n\n");

    // Beam search shapes (Whisper: depth=51865)
    std::printf("--- Whisper vocab (51865) ---\n");
    run_bench<float>("float", 1, 51865, 1);
    run_bench<float>("float", 1, 51865, 5);
    run_bench<float>("float", 1, 51865, 8);
    run_bench<float>("float", 1, 51865, 10);
    run_bench<float>("float", 5, 51865, 5);
    run_bench<float>("float", 5, 51865, 10);

    std::printf("\n--- LLM vocab (32000) ---\n");
    run_bench<float>("float", 1, 32000, 1);
    run_bench<float>("float", 1, 32000, 5);
    run_bench<float>("float", 1, 32000, 10);
    run_bench<float>("float", 1, 32000, 16);

    std::printf("\n--- float16 ---\n");
    run_bench<ct2_f16>("half", 1, 51865, 5);
    run_bench<ct2_f16>("half", 5, 51865, 5);
    run_bench<ct2_f16>("half", 1, 51865, 10);

    std::printf("\n--- bfloat16 ---\n");
    run_bench<ct2_bf16>("bfloat", 1, 51865, 5);
    run_bench<ct2_bf16>("bfloat", 5, 51865, 5);
    run_bench<ct2_bf16>("bfloat", 1, 51865, 10);

    std::printf("\n=== Done ===\n");
    return 0;
  }
}
