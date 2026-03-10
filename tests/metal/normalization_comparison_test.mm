// tests/metal/normalization_comparison_test.mm
//
// M5.2 review item 4.5 — CPU-reference vs Metal comparison for normalization ops.
//
// Each test computes the expected output in plain C++ float32, then runs the
// corresponding Metal kernel, and asserts that every output element is within
// a small absolute tolerance.  Using random-ish multi-row inputs ensures that
// row-offset bugs (tgid * N) and incorrect reduction bugs would be caught —
// analytical single-row checks cannot detect cross-row contamination.
//
// Tests:
//   1.  layer_norm f32: outer=4, axis_size=8, no gamma/beta — max err < 1e-5
//   2.  layer_norm f32: outer=3, axis_size=16, with gamma/beta — max err < 1e-5
//   3.  rms_norm f32: batch=5, depth=8 — max err < 1e-4
//   4.  softmax f32: batch=3, depth=8 — max err < 1e-5
//   5.  log_softmax f32: batch=2, depth=8 — max err < 1e-5
//   6.  softmax f32 with length masking: batch=3, depth=8 — max err < 1e-5
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/normalization_comparison_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o normalization_comparison_test && ./normalization_comparison_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "metal/ops_metal.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                              \
  do {                                                                  \
    if (expr) { std::printf("  PASS  %s\n", label); ++g_passed; }      \
    else       { std::printf("  FAIL  %s\n", label); ++g_failed; }     \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// C++ reference implementations
// ---------------------------------------------------------------------------

// layer_norm: (x - mean) / sqrt(var + eps) * gamma + beta
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
      float v = (xr[j] - mean) * inv_std;
      float g = gamma ? gamma[j] : 1.f;
      float b = beta  ? beta[j]  : 0.f;
      yr[j] = v * g + b;
    }
  }
}

// rms_norm: x / rms(x) * gamma
static void ref_rms_norm(const float* x, const float* gamma, float* y,
                         int batch, int depth, float eps) {
  for (int r = 0; r < batch; ++r) {
    const float* xr = x + r * depth;
    float* yr = y + r * depth;
    float ss = 0.f;
    for (int j = 0; j < depth; ++j) ss += xr[j] * xr[j];
    float rms_inv = 1.f / std::sqrt(ss / depth + eps);
    for (int j = 0; j < depth; ++j)
      yr[j] = xr[j] * rms_inv * gamma[j];
  }
}

// softmax / log-softmax with optional length masking (lengths may be nullptr)
static void ref_softmax(const float* x, const int* lengths, float* y,
                        int batch, int depth, bool log_mode) {
  for (int r = 0; r < batch; ++r) {
    const float* xr = x + r * depth;
    float* yr = y + r * depth;
    int active = lengths ? lengths[r] : depth;
    if (active == 0) {
      for (int j = 0; j < depth; ++j) yr[j] = 0.f;
      continue;
    }
    float mx = -1e38f;
    for (int j = 0; j < active; ++j) if (xr[j] > mx) mx = xr[j];
    float sum = 0.f;
    for (int j = 0; j < active; ++j) sum += std::exp(xr[j] - mx);
    if (log_mode) {
      float log_sum = std::log(sum);
      for (int j = 0; j < active; ++j)
        yr[j] = xr[j] - mx - log_sum;
    } else {
      for (int j = 0; j < active; ++j)
        yr[j] = std::exp(xr[j] - mx) / sum;
    }
    for (int j = active; j < depth; ++j) yr[j] = 0.f;
  }
}

// Compare Metal output against reference; return max abs error.
static float max_abs_err(const float* ref, const float* got, int n) {
  float err = 0.f;
  for (int i = 0; i < n; ++i) {
    float e = std::abs(ref[i] - got[i]);
    if (e > err) err = e;
  }
  return err;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_layer_norm_comparison() {
  std::printf("\n--- layer_norm comparison ---\n");

  // 1. outer=4, N=8, no gamma/beta
  {
    const int outer = 4, N = 8;
    float* x_m  = metal_alloc<float>(outer * N);
    float* y_m  = metal_alloc<float>(outer * N);
    // deterministic inputs: row r col j = (r*13 + j*7 + 1) * 0.1f
    for (int r = 0; r < outer; ++r)
      for (int j = 0; j < N; ++j)
        x_m[r*N+j] = float(r*13 + j*7 + 1) * 0.1f;

    metal::layer_norm_metal<float>(x_m, nullptr, nullptr, y_m, outer, N, 1e-5f);
    metal::commit_and_wait();

    std::vector<float> x_cpu(outer*N), y_ref(outer*N);
    for (int i = 0; i < outer*N; ++i) x_cpu[i] = x_m[i];
    ref_layer_norm(x_cpu.data(), nullptr, nullptr, y_ref.data(), outer, N, 1e-5f);

    float err = max_abs_err(y_ref.data(), y_m, outer*N);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("layer_norm f32 outer=4 N=8 (no gamma/beta): max err < 1e-5", err < 1e-5f);

    metal_free(x_m); metal_free(y_m);
  }

  // 2. outer=3, N=16, with gamma/beta
  {
    const int outer = 3, N = 16;
    float* x_m = metal_alloc<float>(outer * N);
    float* g_m = metal_alloc<float>(N);
    float* b_m = metal_alloc<float>(N);
    float* y_m = metal_alloc<float>(outer * N);
    for (int r = 0; r < outer; ++r)
      for (int j = 0; j < N; ++j)
        x_m[r*N+j] = float((r+1) * (j+1)) * 0.05f;
    for (int j = 0; j < N; ++j) { g_m[j] = 0.5f + float(j)*0.1f; b_m[j] = float(j)*0.05f; }

    metal::layer_norm_metal<float>(x_m, g_m, b_m, y_m, outer, N, 1e-5f);
    metal::commit_and_wait();

    std::vector<float> xc(outer*N), gc(N), bc(N), yr(outer*N);
    for (int i = 0; i < outer*N; ++i) xc[i] = x_m[i];
    for (int j = 0; j < N; ++j)       { gc[j] = g_m[j]; bc[j] = b_m[j]; }
    ref_layer_norm(xc.data(), gc.data(), bc.data(), yr.data(), outer, N, 1e-5f);

    float err = max_abs_err(yr.data(), y_m, outer*N);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("layer_norm f32 outer=3 N=16 (gamma+beta): max err < 1e-5", err < 1e-5f);

    metal_free(x_m); metal_free(g_m); metal_free(b_m); metal_free(y_m);
  }
}

static void test_rms_norm_comparison() {
  std::printf("\n--- rms_norm comparison ---\n");

  // 3. batch=5, depth=8
  {
    const int batch = 5, depth = 8;
    float* x_m = metal_alloc<float>(batch * depth);
    float* g_m = metal_alloc<float>(depth);
    float* y_m = metal_alloc<float>(batch * depth);
    for (int r = 0; r < batch; ++r)
      for (int j = 0; j < depth; ++j)
        x_m[r*depth+j] = float((r+1) * (j+1)) * 0.3f;
    for (int j = 0; j < depth; ++j) g_m[j] = 0.8f + float(j) * 0.05f;

    metal::rms_norm_metal<float>(x_m, g_m, y_m, batch, depth, 1e-6f);
    metal::commit_and_wait();

    std::vector<float> xc(batch*depth), gc(depth), yr(batch*depth);
    for (int i = 0; i < batch*depth; ++i) xc[i] = x_m[i];
    for (int j = 0; j < depth; ++j)       gc[j] = g_m[j];
    ref_rms_norm(xc.data(), gc.data(), yr.data(), batch, depth, 1e-6f);

    float err = max_abs_err(yr.data(), y_m, batch*depth);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("rms_norm f32 batch=5 depth=8: max err < 1e-4", err < 1e-4f);

    metal_free(x_m); metal_free(g_m); metal_free(y_m);
  }
}

static void test_softmax_comparison() {
  std::printf("\n--- softmax comparison ---\n");

  // 4. softmax batch=3, depth=8
  {
    const int batch = 3, depth = 8;
    float* x_m = metal_alloc<float>(batch * depth);
    float* y_m = metal_alloc<float>(batch * depth);
    for (int r = 0; r < batch; ++r)
      for (int j = 0; j < depth; ++j)
        x_m[r*depth+j] = float(r+1) * float(j+1) * 0.2f;

    metal::softmax_metal<float>(x_m, nullptr, y_m, batch, depth, false);
    metal::commit_and_wait();

    std::vector<float> xc(batch*depth), yr(batch*depth);
    for (int i = 0; i < batch*depth; ++i) xc[i] = x_m[i];
    ref_softmax(xc.data(), nullptr, yr.data(), batch, depth, false);

    float err = max_abs_err(yr.data(), y_m, batch*depth);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("softmax f32 batch=3 depth=8: max err < 1e-5", err < 1e-5f);

    metal_free(x_m); metal_free(y_m);
  }

  // 5. log-softmax batch=2, depth=8
  {
    const int batch = 2, depth = 8;
    float* x_m = metal_alloc<float>(batch * depth);
    float* y_m = metal_alloc<float>(batch * depth);
    for (int r = 0; r < batch; ++r)
      for (int j = 0; j < depth; ++j)
        x_m[r*depth+j] = float(j+1) * (r == 0 ? 0.3f : -0.2f);

    metal::softmax_metal<float>(x_m, nullptr, y_m, batch, depth, true);
    metal::commit_and_wait();

    std::vector<float> xc(batch*depth), yr(batch*depth);
    for (int i = 0; i < batch*depth; ++i) xc[i] = x_m[i];
    ref_softmax(xc.data(), nullptr, yr.data(), batch, depth, true);

    float err = max_abs_err(yr.data(), y_m, batch*depth);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("log_softmax f32 batch=2 depth=8: max err < 1e-5", err < 1e-5f);

    metal_free(x_m); metal_free(y_m);
  }

  // 6. softmax with length masking: batch=3, lengths=[6,3,8]
  {
    const int batch = 3, depth = 8;
    float*   x_m = metal_alloc<float>(batch * depth);
    float*   y_m = metal_alloc<float>(batch * depth);
    int32_t* len = metal_alloc<int32_t>(batch);
    int lens[3] = {6, 3, 8};
    for (int r = 0; r < batch; ++r) {
      len[r] = lens[r];
      for (int j = 0; j < depth; ++j)
        x_m[r*depth+j] = float((r+1) * (j+1)) * 0.25f;
    }

    metal::softmax_metal<float>(x_m, len, y_m, batch, depth, false);
    metal::commit_and_wait();

    std::vector<float> xc(batch*depth), yr(batch*depth);
    for (int i = 0; i < batch*depth; ++i) xc[i] = x_m[i];
    ref_softmax(xc.data(), lens, yr.data(), batch, depth, false);

    float err = max_abs_err(yr.data(), y_m, batch*depth);
    std::printf("  max abs err = %.2e\n", err);
    CHECK("softmax f32 masked batch=3 depth=8: max err < 1e-5", err < 1e-5f);

    metal_free(x_m); metal_free(y_m); metal_free(len);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M5.2 normalization CPU-reference vs Metal comparison ===\n");

  test_layer_norm_comparison();
  test_rms_norm_comparison();
  test_softmax_comparison();

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
