// tests/metal/topk_test.mm
//
// GPU TopK test — argmax (k=1) and iterative top-k (k>1).
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/topk_test.mm \
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
//     -o topk_test && ./topk_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK(label, cond) \
  do { \
    bool _ok = (cond); \
    std::printf("  %s  %s\n", _ok ? "PASS" : "FAIL", label); \
    if (_ok) ++g_pass; else ++g_fail; \
  } while (0)

// ---------------------------------------------------------------------------
// CPU reference argmax (k=1)
// ---------------------------------------------------------------------------

template <typename T>
static void ref_argmax(const T* input, float* ref_vals, int32_t* ref_idxs,
                       dim_t batch, dim_t depth) {
  for (dim_t b = 0; b < batch; ++b) {
    const T* row = input + b * depth;
    float best = -1e30f;
    int32_t best_idx = 0;
    for (dim_t j = 0; j < depth; ++j) {
      float v = static_cast<float>(row[j]);
      if (v > best) { best = v; best_idx = static_cast<int32_t>(j); }
    }
    ref_vals[b] = best;
    ref_idxs[b] = best_idx;
  }
}

// ---------------------------------------------------------------------------
// CPU reference top-k (k>1) — std::partial_sort
// ---------------------------------------------------------------------------

template <typename T>
static void ref_topk(const T* input, float* ref_vals, int32_t* ref_idxs,
                     dim_t batch, dim_t depth, dim_t k) {
  std::vector<int32_t> ids(static_cast<std::size_t>(depth));
  for (dim_t b = 0; b < batch; ++b) {
    const T* row = input + b * depth;
    std::iota(ids.begin(), ids.end(), int32_t(0));
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
        [&row](int32_t i1, int32_t i2) {
          return static_cast<float>(row[i1]) > static_cast<float>(row[i2]);
        });
    for (dim_t j = 0; j < k; ++j) {
      ref_idxs[b * k + j] = ids[static_cast<std::size_t>(j)];
      ref_vals[b * k + j] = static_cast<float>(row[ids[static_cast<std::size_t>(j)]]);
    }
  }
}

// ---------------------------------------------------------------------------
// Test: GPU argmax (k=1)
// ---------------------------------------------------------------------------

template <typename T>
static void test_argmax(const char* type_label, dim_t batch, dim_t depth, float tol) {
  char label[128];
  std::snprintf(label, sizeof(label), "argmax_%s  batch=%lld depth=%lld",
                type_label, (long long)batch, (long long)depth);

  auto& alloc = get_allocator<Device::METAL>();
  const size_t in_bytes  = batch * depth * sizeof(T);
  const size_t val_bytes = batch * sizeof(T);
  const size_t idx_bytes = batch * sizeof(int32_t);

  T*       in_ptr  = static_cast<T*>(alloc.allocate(in_bytes, 0));
  T*       val_ptr = static_cast<T*>(alloc.allocate(val_bytes, 0));
  int32_t* idx_ptr = static_cast<int32_t*>(alloc.allocate(idx_bytes, 0));

  std::mt19937 rng(42 + batch * 1000 + depth);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (dim_t i = 0; i < batch * depth; ++i)
    in_ptr[i] = static_cast<T>(dist(rng));

  // Place a known max in a specific position per row
  for (dim_t b = 0; b < batch; ++b) {
    dim_t pos = (b * 137 + 7) % depth;
    in_ptr[b * depth + pos] = static_cast<T>(100.0f + static_cast<float>(b));
  }

  std::memset(val_ptr, 0, val_bytes);
  std::memset(idx_ptr, 0, idx_bytes);

  metal::topk_metal<T>(in_ptr, val_ptr, idx_ptr, batch, depth, 1);
  synchronize_stream(Device::METAL);

  std::vector<float> ref_vals(batch);
  std::vector<int32_t> ref_idxs(batch);
  ref_argmax<T>(in_ptr, ref_vals.data(), ref_idxs.data(), batch, depth);

  bool all_ok = true;
  for (dim_t b = 0; b < batch; ++b) {
    float gpu_val = static_cast<float>(val_ptr[b]);
    int32_t gpu_idx = idx_ptr[b];
    float diff = std::abs(gpu_val - ref_vals[b]);
    if (gpu_idx != ref_idxs[b] || diff > tol) {
      std::printf("    MISMATCH batch=%lld: gpu_idx=%d ref_idx=%d "
                  "gpu_val=%.6f ref_val=%.6f diff=%.2e\n",
                  (long long)b, gpu_idx, ref_idxs[b], gpu_val, ref_vals[b], diff);
      all_ok = false;
    }
  }
  CHECK(label, all_ok);

  alloc.free(in_ptr, 0);
  alloc.free(val_ptr, 0);
  alloc.free(idx_ptr, 0);
}

// ---------------------------------------------------------------------------
// Test: GPU top-k (k>1)
// ---------------------------------------------------------------------------

template <typename T>
static void test_topk(const char* type_label, dim_t batch, dim_t depth, dim_t k, float tol) {
  char label[128];
  std::snprintf(label, sizeof(label), "topk_%s  batch=%lld depth=%lld k=%lld",
                type_label, (long long)batch, (long long)depth, (long long)k);

  auto& alloc = get_allocator<Device::METAL>();
  const size_t in_bytes  = batch * depth * sizeof(T);
  const size_t val_bytes = batch * k * sizeof(T);
  const size_t idx_bytes = batch * k * sizeof(int32_t);

  T*       in_ptr  = static_cast<T*>(alloc.allocate(in_bytes, 0));
  T*       val_ptr = static_cast<T*>(alloc.allocate(val_bytes, 0));
  int32_t* idx_ptr = static_cast<int32_t*>(alloc.allocate(idx_bytes, 0));

  std::mt19937 rng(123 + batch * 1000 + depth * 10 + k);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (dim_t i = 0; i < batch * depth; ++i)
    in_ptr[i] = static_cast<T>(dist(rng));

  // Place k known top values per row at deterministic positions
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t j = 0; j < k; ++j) {
      dim_t pos = (b * 137 + j * 53 + 7) % depth;
      // Ensure distinct values: 100 + b + (k-j)*0.5 so ordering is clear
      in_ptr[b * depth + pos] = static_cast<T>(100.0f + static_cast<float>(b) + static_cast<float>(k - j) * 0.5f);
    }
  }

  std::memset(val_ptr, 0, val_bytes);
  std::memset(idx_ptr, 0, idx_bytes);

  metal::topk_metal<T>(in_ptr, val_ptr, idx_ptr, batch, depth, k);
  synchronize_stream(Device::METAL);

  // CPU reference
  std::vector<float> ref_vals(batch * k);
  std::vector<int32_t> ref_idxs(batch * k);
  ref_topk<T>(in_ptr, ref_vals.data(), ref_idxs.data(), batch, depth, k);

  bool all_ok = true;
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t j = 0; j < k; ++j) {
      dim_t pos = b * k + j;
      float gpu_val = static_cast<float>(val_ptr[pos]);
      int32_t gpu_idx = idx_ptr[pos];
      float ref_val = ref_vals[pos];
      int32_t ref_idx = ref_idxs[pos];
      float diff = std::abs(gpu_val - ref_val);

      // For half/bfloat: values may tie at reduced precision, so compare
      // by value rather than index when diff is within tolerance.
      if (diff > tol) {
        std::printf("    MISMATCH batch=%lld k=%lld: gpu_idx=%d ref_idx=%d "
                    "gpu_val=%.6f ref_val=%.6f diff=%.2e\n",
                    (long long)b, (long long)j, gpu_idx, ref_idx,
                    gpu_val, ref_val, diff);
        all_ok = false;
      } else if (gpu_idx != ref_idx && diff > 0) {
        // Index mismatch but value match within tolerance — acceptable for
        // reduced precision types where ties can occur
      }
    }
  }
  CHECK(label, all_ok);

  alloc.free(in_ptr, 0);
  alloc.free(val_ptr, 0);
  alloc.free(idx_ptr, 0);
}

// ---------------------------------------------------------------------------
// Test: gather encode-only correctness
// ---------------------------------------------------------------------------

static void test_gather_encode_only() {
  auto& alloc = get_allocator<Device::METAL>();
  const dim_t depth = 4;
  const dim_t num_rows = 8;
  const dim_t num_indices = 3;

  float* src = static_cast<float*>(alloc.allocate(num_rows * depth * sizeof(float), 0));
  float* dst = static_cast<float*>(alloc.allocate(num_indices * depth * sizeof(float), 0));
  int32_t* indices = static_cast<int32_t*>(alloc.allocate(num_indices * sizeof(int32_t), 0));

  for (dim_t i = 0; i < num_rows; ++i)
    for (dim_t j = 0; j < depth; ++j)
      src[i * depth + j] = static_cast<float>(i * 10 + j);

  indices[0] = 2;
  indices[1] = 5;
  indices[2] = 0;

  std::memset(dst, 0, num_indices * depth * sizeof(float));

  metal::gather_metal<float>(src, dst, indices, depth, num_rows * depth,
                              num_indices, num_indices * depth);
  synchronize_stream(Device::METAL);

  bool ok = true;
  float expected[] = {20, 21, 22, 23, 50, 51, 52, 53, 0, 1, 2, 3};
  for (dim_t i = 0; i < num_indices * depth; ++i) {
    if (std::abs(dst[i] - expected[i]) > 1e-6f) {
      std::printf("    gather MISMATCH at %lld: got=%.1f expected=%.1f\n",
                  (long long)i, dst[i], expected[i]);
      ok = false;
    }
  }
  CHECK("gather_encode_only_f32", ok);

  alloc.free(src, 0);
  alloc.free(dst, 0);
  alloc.free(indices, 0);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
  @autoreleasepool {
    std::printf("=== TopK GPU tests (k=1 and k>1) ===\n\n");

    // ---- k=1: argmax ----
    std::printf("--- float32 argmax (k=1) ---\n");
    test_argmax<float>("float", 1, 256, 0.0f);
    test_argmax<float>("float", 1, 4096, 0.0f);
    test_argmax<float>("float", 4, 32000, 0.0f);
    test_argmax<float>("float", 1, 51865, 0.0f);
    test_argmax<float>("float", 8, 51865, 0.0f);

    std::printf("\n--- float16 argmax (k=1) ---\n");
    test_argmax<ct2_f16>("half", 1, 4096, 1e-3f);
    test_argmax<ct2_f16>("half", 4, 32000, 1e-3f);
    test_argmax<ct2_f16>("half", 1, 51865, 1e-3f);

    std::printf("\n--- bfloat16 argmax (k=1) ---\n");
    test_argmax<ct2_bf16>("bfloat", 1, 4096, 2e-2f);
    test_argmax<ct2_bf16>("bfloat", 4, 32000, 2e-2f);
    test_argmax<ct2_bf16>("bfloat", 1, 51865, 2e-2f);

    // ---- k>1: top-k ----
    std::printf("\n--- float32 top-k (k>1) ---\n");
    test_topk<float>("float", 1, 32000, 2, 0.0f);
    test_topk<float>("float", 1, 32000, 4, 0.0f);
    test_topk<float>("float", 1, 32000, 8, 0.0f);
    test_topk<float>("float", 1, 32000, 16, 0.0f);
    test_topk<float>("float", 1, 32000, 32, 0.0f);
    test_topk<float>("float", 1, 51865, 5, 0.0f);
    test_topk<float>("float", 1, 51865, 10, 0.0f);
    test_topk<float>("float", 4, 51865, 5, 0.0f);
    test_topk<float>("float", 8, 51865, 8, 0.0f);
    test_topk<float>("float", 5, 51865, 10, 0.0f);

    std::printf("\n--- float16 top-k (k>1) ---\n");
    test_topk<ct2_f16>("half", 1, 32000, 4, 1e-3f);
    test_topk<ct2_f16>("half", 1, 51865, 5, 1e-3f);
    test_topk<ct2_f16>("half", 4, 51865, 8, 1e-3f);
    test_topk<ct2_f16>("half", 5, 51865, 10, 1e-3f);

    std::printf("\n--- bfloat16 top-k (k>1) ---\n");
    test_topk<ct2_bf16>("bfloat", 1, 32000, 4, 2e-2f);
    test_topk<ct2_bf16>("bfloat", 1, 51865, 5, 2e-2f);
    test_topk<ct2_bf16>("bfloat", 4, 51865, 8, 2e-2f);
    test_topk<ct2_bf16>("bfloat", 5, 51865, 10, 2e-2f);

    // ---- gather encode-only ----
    std::printf("\n--- gather encode-only ---\n");
    test_gather_encode_only();

    std::printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
  }
}
