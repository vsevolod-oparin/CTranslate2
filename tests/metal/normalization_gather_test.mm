// tests/metal/normalization_gather_test.mm
//
// M5.2 correctness tests for the Metal normalization and gather kernels.
//
// Tests:
//   1.  layer_norm warmup (library compiles, no exception)
//   2.  layer_norm f32: zero mean, unit variance output
//   3.  layer_norm f32: with gamma scaling and beta bias
//   4.  layer_norm f16: basic correctness
//   5.  rms_norm warmup (library compiles, no exception)
//   6.  rms_norm f32: basic correctness vs reference
//   7.  rms_norm f16: basic correctness
//   8.  softmax warmup (library compiles, no exception)
//   9.  softmax f32: outputs sum to 1 and are all positive
//  10.  log_softmax f32: outputs all <= 0, exp(y) sums to 1
//  11.  softmax f32: length-masked rows — out-of-range slots are zero
//  12.  gather warmup (library compiles, no exception)
//  13.  gather f32: copy_size=1 (single-element copy)
//  14.  gather f32: copy_size=4 (block copy)
//  15.  gather int32: integer element type
//  16.  gather batched (batch_dims=0, batch in src)
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/normalization_gather_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/primitives_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o normalization_gather_test && ./normalization_gather_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "metal/ops_metal.h"
#include "metal/utils.h"

// Resolve ::float16_t / ctranslate2::float16_t conflict from arm_vector_types.h.
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

#define CHECK_NEAR(label, got, want, tol)                               \
  do {                                                                  \
    float _g = (float)(got), _w = (float)(want);                       \
    if (std::abs(_g - _w) <= (tol)) {                                   \
      std::printf("  PASS  %s (got %.5g)\n", label, _g); ++g_passed;   \
    } else {                                                            \
      std::printf("  FAIL  %s (got %.5g, want %.5g)\n", label, _g, _w);\
      ++g_failed;                                                       \
    }                                                                   \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

template <typename F>
static bool no_exception(F&& f) {
  try { f(); return true; }
  catch (const std::exception& e) {
    std::printf("    exception: %s\n", e.what()); return false;
  } catch (...) {
    std::printf("    unknown exception\n"); return false;
  }
}

// Flush GPU and read back a single element.
template <typename T>
static float read_back(const T* ptr) {
  metal::commit_and_wait();
  return (float)(*ptr);
}

// ---------------------------------------------------------------------------
// 1–4: layer_norm
// ---------------------------------------------------------------------------

static void test_layer_norm() {
  std::printf("\n--- layer_norm ---\n");

  // 1. Warmup
  {
    const int N = 4;
    float* x  = metal_alloc<float>(N);
    float* y  = metal_alloc<float>(N);
    for (int i = 0; i < N; ++i) x[i] = float(i + 1);

    bool ok = no_exception([&] {
      metal::layer_norm_metal<float>(x, nullptr, nullptr, y,
                                     /*outer_size=*/1, /*axis_size=*/N,
                                     /*epsilon=*/1e-5f);
      metal::commit_and_wait();
    });
    CHECK("layer_norm warmup (no exception)", ok);

    metal_free(x); metal_free(y);
  }

  // 2. f32 identity (gamma=1, beta=0 omitted): zero mean, unit variance
  {
    // x = [1, 2, 3, 4], mean = 2.5, var = ((-1.5)^2 + (-0.5)^2 + 0.5^2 + 1.5^2)/4 = 1.25
    // inv_std = 1/sqrt(1.25 + 1e-5) ≈ 0.8944
    // y ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
    const int N = 4;
    float* x  = metal_alloc<float>(N);
    float* y  = metal_alloc<float>(N);
    x[0]=1; x[1]=2; x[2]=3; x[3]=4;

    metal::layer_norm_metal<float>(x, nullptr, nullptr, y, 1, N, 1e-5f);
    metal::commit_and_wait();

    // Mean of output should be ~0
    float sum = 0;
    for (int i = 0; i < N; ++i) sum += y[i];
    CHECK_NEAR("layer_norm f32: output mean ≈ 0", sum / N, 0.f, 1e-5f);

    // Variance of output should be ~1
    float var = 0;
    for (int i = 0; i < N; ++i) var += y[i] * y[i];
    CHECK_NEAR("layer_norm f32: output variance ≈ 1", var / N, 1.f, 1e-4f);

    metal_free(x); metal_free(y);
  }

  // 3. f32 with gamma/beta: y[i] = gamma[i]*normalized + beta[i]
  {
    const int N = 4;
    float* x     = metal_alloc<float>(N);
    float* gamma = metal_alloc<float>(N);
    float* beta  = metal_alloc<float>(N);
    float* y     = metal_alloc<float>(N);
    x[0]=1; x[1]=2; x[2]=3; x[3]=4;
    for (int i = 0; i < N; ++i) { gamma[i] = 2.f; beta[i] = 1.f; }

    metal::layer_norm_metal<float>(x, gamma, beta, y, 1, N, 1e-5f);
    metal::commit_and_wait();

    // With gamma=2, beta=1: mean of output ≈ 1 (beta), variance ≈ 4 (gamma^2)
    float sum = 0;
    for (int i = 0; i < N; ++i) sum += y[i];
    CHECK_NEAR("layer_norm f32 gamma/beta: output mean ≈ 1", sum / N, 1.f, 1e-4f);

    float var = 0, m = sum / N;
    for (int i = 0; i < N; ++i) var += (y[i] - m) * (y[i] - m);
    CHECK_NEAR("layer_norm f32 gamma/beta: output variance ≈ 4", var / N, 4.f, 1e-3f);

    metal_free(x); metal_free(gamma); metal_free(beta); metal_free(y);
  }

  // 4. f16 basic: output mean ≈ 0, variance ≈ 1
  {
    const int N = 8;
    ct2_f16* x  = metal_alloc<ct2_f16>(N);
    ct2_f16* y  = metal_alloc<ct2_f16>(N);
    for (int i = 0; i < N; ++i) x[i] = ct2_f16(float(i + 1));

    metal::layer_norm_metal<ct2_f16>(x, nullptr, nullptr, y, 1, N, 1e-5f);
    metal::commit_and_wait();

    float sum = 0, var = 0;
    for (int i = 0; i < N; ++i) sum += (float)y[i];
    float mean = sum / N;
    for (int i = 0; i < N; ++i) var += ((float)y[i] - mean) * ((float)y[i] - mean);
    CHECK_NEAR("layer_norm f16: output mean ≈ 0", mean, 0.f, 1e-3f);
    CHECK_NEAR("layer_norm f16: output variance ≈ 1", var / N, 1.f, 5e-2f);

    metal_free(x); metal_free(y);
  }
}

// ---------------------------------------------------------------------------
// 5–7: rms_norm
// ---------------------------------------------------------------------------

static void test_rms_norm() {
  std::printf("\n--- rms_norm ---\n");

  // 5. Warmup
  {
    const int N = 4;
    float* x     = metal_alloc<float>(N);
    float* gamma = metal_alloc<float>(N);
    float* y     = metal_alloc<float>(N);
    for (int i = 0; i < N; ++i) { x[i] = float(i + 1); gamma[i] = 1.f; }

    bool ok = no_exception([&] {
      metal::rms_norm_metal<float>(x, gamma, y, 1, N, 1e-6f);
      metal::commit_and_wait();
    });
    CHECK("rms_norm warmup (no exception)", ok);

    metal_free(x); metal_free(gamma); metal_free(y);
  }

  // 6. f32 correctness: x=[1,2,3,4], gamma=1
  //    rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
  //    y[0] = 1 / 2.7386 ≈ 0.3651
  {
    const int N = 4;
    float* x     = metal_alloc<float>(N);
    float* gamma = metal_alloc<float>(N);
    float* y     = metal_alloc<float>(N);
    x[0]=1; x[1]=2; x[2]=3; x[3]=4;
    for (int i = 0; i < N; ++i) gamma[i] = 1.f;

    metal::rms_norm_metal<float>(x, gamma, y, 1, N, 1e-6f);
    metal::commit_and_wait();

    // RMS of output should be ≈ 1 (by definition)
    float ss = 0;
    for (int i = 0; i < N; ++i) ss += y[i] * y[i];
    CHECK_NEAR("rms_norm f32: output RMS ≈ 1", std::sqrt(ss / N), 1.f, 1e-4f);

    // y[0] ≈ 1/rms(x)
    float rms_x = std::sqrt((1.f+4.f+9.f+16.f)/4.f);
    CHECK_NEAR("rms_norm f32: y[0] ≈ x[0]/rms(x)", y[0], 1.f/rms_x, 1e-4f);

    metal_free(x); metal_free(gamma); metal_free(y);
  }

  // 7. f16 basic: output RMS ≈ 1
  {
    const int N = 8;
    ct2_f16* x     = metal_alloc<ct2_f16>(N);
    ct2_f16* gamma = metal_alloc<ct2_f16>(N);
    ct2_f16* y     = metal_alloc<ct2_f16>(N);
    for (int i = 0; i < N; ++i) { x[i] = ct2_f16(float(i + 1)); gamma[i] = ct2_f16(1.f); }

    metal::rms_norm_metal<ct2_f16>(x, gamma, y, 1, N, 1e-6f);
    metal::commit_and_wait();

    float ss = 0;
    for (int i = 0; i < N; ++i) ss += (float)y[i] * (float)y[i];
    CHECK_NEAR("rms_norm f16: output RMS ≈ 1", std::sqrt(ss / N), 1.f, 5e-2f);

    metal_free(x); metal_free(gamma); metal_free(y);
  }
}

// ---------------------------------------------------------------------------
// 8–11: softmax
// ---------------------------------------------------------------------------

static void test_softmax() {
  std::printf("\n--- softmax ---\n");

  // 8. Warmup
  {
    const int N = 4;
    float* x = metal_alloc<float>(N);
    float* y = metal_alloc<float>(N);
    for (int i = 0; i < N; ++i) x[i] = float(i);

    bool ok = no_exception([&] {
      metal::softmax_metal<float>(x, nullptr, y, 1, N, false);
      metal::commit_and_wait();
    });
    CHECK("softmax warmup (no exception)", ok);

    metal_free(x); metal_free(y);
  }

  // 9. f32 softmax: outputs sum to 1, all positive
  {
    const int N = 8;
    float* x = metal_alloc<float>(N);
    float* y = metal_alloc<float>(N);
    for (int i = 0; i < N; ++i) x[i] = float(i) * 0.5f;

    metal::softmax_metal<float>(x, nullptr, y, 1, N, false);
    metal::commit_and_wait();

    float sum = 0;
    bool all_positive = true;
    for (int i = 0; i < N; ++i) {
      sum += y[i];
      if (y[i] <= 0.f) all_positive = false;
    }
    CHECK_NEAR("softmax f32: output sums to 1", sum, 1.f, 1e-5f);
    CHECK("softmax f32: all outputs positive", all_positive);

    metal_free(x); metal_free(y);
  }

  // 10. f32 log-softmax: all outputs <= 0, exp(y) sums to 1
  {
    const int N = 8;
    float* x = metal_alloc<float>(N);
    float* y = metal_alloc<float>(N);
    for (int i = 0; i < N; ++i) x[i] = float(i) * 0.5f;

    metal::softmax_metal<float>(x, nullptr, y, 1, N, true);
    metal::commit_and_wait();

    float sum_exp = 0;
    bool all_nonpositive = true;
    for (int i = 0; i < N; ++i) {
      sum_exp += std::exp(y[i]);
      if (y[i] > 0.f) all_nonpositive = false;
    }
    CHECK("log_softmax f32: all outputs <= 0", all_nonpositive);
    CHECK_NEAR("log_softmax f32: exp(y) sums to 1", sum_exp, 1.f, 1e-5f);

    metal_free(x); metal_free(y);
  }

  // 11. f32 length-masked: lengths[0]=4, depth=8, slots [4..7] should be 0
  {
    const int batch = 1;
    const int N = 8;
    float*   x   = metal_alloc<float>(N);
    float*   y   = metal_alloc<float>(N);
    int32_t* len = metal_alloc<int32_t>(batch);
    for (int i = 0; i < N; ++i) x[i] = 1.f;
    len[0] = 4;

    metal::softmax_metal<float>(x, len, y, batch, N, false);
    metal::commit_and_wait();

    // First 4 slots: sum to 1
    float sum = 0;
    for (int i = 0; i < 4; ++i) sum += y[i];
    CHECK_NEAR("softmax masked: active slots sum to 1", sum, 1.f, 1e-5f);

    // Slots 4..7 should be 0
    bool tail_zero = true;
    for (int i = 4; i < N; ++i) if (y[i] != 0.f) tail_zero = false;
    CHECK("softmax masked: inactive slots are zero", tail_zero);

    metal_free(x); metal_free(y); metal_free(len);
  }
}

// ---------------------------------------------------------------------------
// 12–16: gather
// ---------------------------------------------------------------------------

static void test_gather() {
  std::printf("\n--- gather ---\n");

  // 12. Warmup
  {
    const int src_n = 4;
    const int idx_n = 2;
    float*   src = metal_alloc<float>(src_n);
    float*   dst = metal_alloc<float>(idx_n);
    int32_t* idx = metal_alloc<int32_t>(idx_n);
    for (int i = 0; i < src_n; ++i) src[i] = float(i);
    idx[0] = 0; idx[1] = 2;

    bool ok = no_exception([&] {
      // gather with copy_size=1, batch_stride=src_n, num_indices_per_batch=idx_n
      metal::gather_metal<float>(src, dst, idx,
                                  /*copy_size=*/1, /*batch_stride=*/src_n,
                                  /*num_indices_per_batch=*/idx_n,
                                  /*total_elements=*/idx_n);
      metal::commit_and_wait();
    });
    CHECK("gather warmup (no exception)", ok);

    metal_free(src); metal_free(dst); metal_free(idx);
  }

  // 13. f32 copy_size=1: gather single elements
  {
    // src = [10, 20, 30, 40], indices = [3, 1, 0]
    // expected dst = [40, 20, 10]
    float*   src = metal_alloc<float>(4);
    float*   dst = metal_alloc<float>(3);
    int32_t* idx = metal_alloc<int32_t>(3);
    src[0]=10; src[1]=20; src[2]=30; src[3]=40;
    idx[0]=3; idx[1]=1; idx[2]=0;

    metal::gather_metal<float>(src, dst, idx, 1, /*batch_stride=*/4, 3, 3);
    metal::commit_and_wait();

    CHECK_NEAR("gather f32 copy_size=1: dst[0]=40", dst[0], 40.f, 0.f);
    CHECK_NEAR("gather f32 copy_size=1: dst[1]=20", dst[1], 20.f, 0.f);
    CHECK_NEAR("gather f32 copy_size=1: dst[2]=10", dst[2], 10.f, 0.f);

    metal_free(src); metal_free(dst); metal_free(idx);
  }

  // 14. f32 copy_size=4: gather rows of 4 floats
  {
    // src shape: [3, 4] row-major, rows = [0,1,2,3], [10,11,12,13], [20,21,22,23]
    // indices = [2, 0]: gather rows 2 and 0
    // expected dst = [20,21,22,23, 0,1,2,3]
    const int copy_size = 4;
    float*   src = metal_alloc<float>(3 * copy_size);
    float*   dst = metal_alloc<float>(2 * copy_size);
    int32_t* idx = metal_alloc<int32_t>(2);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < copy_size; ++c)
        src[r * copy_size + c] = float(r * 10 + c);
    idx[0] = 2; idx[1] = 0;

    metal::gather_metal<float>(src, dst, idx,
                                /*copy_size=*/copy_size,
                                /*batch_stride=*/3 * copy_size,
                                /*num_indices_per_batch=*/2,
                                /*total_elements=*/2 * copy_size);
    metal::commit_and_wait();

    bool ok = true;
    for (int c = 0; c < copy_size; ++c)
      if (dst[c] != float(20 + c)) { ok = false; break; }
    for (int c = 0; c < copy_size; ++c)
      if (dst[copy_size + c] != float(c)) { ok = false; break; }
    CHECK("gather f32 copy_size=4: rows correct", ok);

    metal_free(src); metal_free(dst); metal_free(idx);
  }

  // 15. int32 gather
  {
    int32_t* src = metal_alloc<int32_t>(5);
    int32_t* dst = metal_alloc<int32_t>(3);
    int32_t* idx = metal_alloc<int32_t>(3);
    for (int i = 0; i < 5; ++i) src[i] = i * 100;
    idx[0] = 4; idx[1] = 0; idx[2] = 2;

    metal::gather_metal<int32_t>(src, dst, idx, 1, 5, 3, 3);
    metal::commit_and_wait();

    CHECK("gather int32: dst[0]=400", dst[0] == 400);
    CHECK("gather int32: dst[1]=0",   dst[1] == 0);
    CHECK("gather int32: dst[2]=200", dst[2] == 200);

    metal_free(src); metal_free(dst); metal_free(idx);
  }

  // 16. Batched gather (2 batches, num_indices_per_batch=2)
  {
    // src shape: [2 batches, 4 rows, 1 elem] = 8 floats
    // batch 0: rows [0,1,2,3] = values [0,1,2,3]
    // batch 1: rows [0,1,2,3] = values [10,11,12,13]
    // indices = [3, 1, 2, 0] (2 per batch)
    // batch 0: gather rows [3,1] → [3,1]
    // batch 1: gather rows [2,0] → [12,10]
    const int copy_size = 1;
    const int rows = 4;
    const int batches = 2;
    const int num_per_batch = 2;
    float*   src = metal_alloc<float>(batches * rows);
    float*   dst = metal_alloc<float>(batches * num_per_batch);
    int32_t* idx = metal_alloc<int32_t>(batches * num_per_batch);
    for (int b = 0; b < batches; ++b)
      for (int r = 0; r < rows; ++r)
        src[b * rows + r] = float(b * 10 + r);
    idx[0]=3; idx[1]=1;  // batch 0 indices
    idx[2]=2; idx[3]=0;  // batch 1 indices

    metal::gather_metal<float>(src, dst, idx,
                                copy_size,
                                /*batch_stride=*/rows,
                                num_per_batch,
                                /*total=*/batches * num_per_batch);
    metal::commit_and_wait();

    CHECK_NEAR("gather batched: dst[0]=3  (b0,row3)", dst[0],  3.f, 0.f);
    CHECK_NEAR("gather batched: dst[1]=1  (b0,row1)", dst[1],  1.f, 0.f);
    CHECK_NEAR("gather batched: dst[2]=12 (b1,row2)", dst[2], 12.f, 0.f);
    CHECK_NEAR("gather batched: dst[3]=10 (b1,row0)", dst[3], 10.f, 0.f);

    metal_free(src); metal_free(dst); metal_free(idx);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M5.2 normalization + gather Metal tests ===\n");

  test_layer_norm();
  test_rms_norm();
  test_softmax();
  test_gather();

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
