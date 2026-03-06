// tests/metal/fused_norm_gemm_test.mm
//
// Fused LayerNorm/RMSNorm + GEMV on Metal: correctness test.
//
// Calls metal::fused_layer_norm_gemm_metal<T>() and
// metal::fused_rms_norm_gemm_metal<T>() directly, comparing against
// a CPU reference (separate LayerNorm + GEMV in float32).
//
// Tests:
//   1-3.   LayerNorm+GEMV f32:  M=1,4,16; K=512; N=2048
//   4-6.   LayerNorm+GEMV f16:  M=1,4,16; K=512; N=2048
//   7-9.   LayerNorm+GEMV bf16: M=1,4,16; K=512; N=2048
//   10-12. RMSNorm+GEMV f32:   M=1,4,16; K=1024; N=512
//   13-15. RMSNorm+GEMV f16:   M=1,4,16; K=1024; N=512
//   16-18. RMSNorm+GEMV bf16:  M=1,4,16; K=1024; N=512
//   19.    LayerNorm+GEMV f32 large: M=1; K=4096; N=6144
//   20.    RMSNorm+GEMV f32 large:  M=1; K=2048; N=4096
//   21.    LayerNorm+GEMV f32 no-beta: M=4; K=512; N=2048
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/fused_norm_gemm_test.mm \
//     src/metal/ops_fused_norm_gemm.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -o fused_norm_gemm_test && ./fused_norm_gemm_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
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
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK_CLOSE(label, got, ref, tol) \
  do { \
    float _max = 0.f; \
    for (size_t _i = 0; _i < (got).size(); ++_i) { \
      float _d = std::abs((float)(got)[_i] - (float)(ref)[_i]); \
      if (_d > _max) _max = _d; \
    } \
    bool _ok = (_max <= (float)(tol)); \
    std::printf("  %s  %s  (max_abs_diff=%.2e, tol=%.2e)\n", \
                _ok ? "PASS" : "FAIL", label, (double)_max, (double)(tol)); \
    if (_ok) ++g_pass; else ++g_fail; \
  } while (0)

// ---------------------------------------------------------------------------
// CPU reference: LayerNorm (float32)
// ---------------------------------------------------------------------------

static void ref_layer_norm_f32(const float* x, const float* gamma, const float* beta,
                                float* y, dim_t M, dim_t K, float eps) {
  for (dim_t m = 0; m < M; ++m) {
    const float* row = x + m * K;
    float* out = y + m * K;
    // mean
    double s = 0;
    for (dim_t j = 0; j < K; ++j) s += row[j];
    float mean = (float)(s / K);
    // variance
    double v = 0;
    for (dim_t j = 0; j < K; ++j) { float d = row[j] - mean; v += d * d; }
    float inv_std = 1.f / std::sqrt((float)(v / K) + eps);
    // normalize
    for (dim_t j = 0; j < K; ++j) {
      float val = (row[j] - mean) * inv_std;
      float g = gamma ? gamma[j] : 1.f;
      float b = beta ? beta[j] : 0.f;
      out[j] = val * g + b;
    }
  }
}

// ---------------------------------------------------------------------------
// CPU reference: RMSNorm (float32)
// ---------------------------------------------------------------------------

static void ref_rms_norm_f32(const float* x, const float* gamma,
                              float* y, dim_t M, dim_t K, float eps) {
  for (dim_t m = 0; m < M; ++m) {
    const float* row = x + m * K;
    float* out = y + m * K;
    double ss = 0;
    for (dim_t j = 0; j < K; ++j) ss += (double)row[j] * row[j];
    float rms_inv = 1.f / std::sqrt((float)(ss / K) + eps);
    for (dim_t j = 0; j < K; ++j)
      out[j] = row[j] * rms_inv * gamma[j];
  }
}

// ---------------------------------------------------------------------------
// CPU reference: GEMV  y[M,N] = norm_out[M,K] * W[N,K]^T  (W is row-major [N,K])
// ---------------------------------------------------------------------------

static void ref_gemv_f32(const float* norm_out, const float* W,
                          float* y, dim_t M, dim_t K, dim_t N) {
  for (dim_t m = 0; m < M; ++m)
    for (dim_t n = 0; n < N; ++n) {
      double acc = 0;
      for (dim_t k = 0; k < K; ++k)
        acc += (double)norm_out[m * K + k] * W[n * K + k];
      y[m * N + n] = (float)acc;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
static std::vector<T> to_type(const std::vector<float>& src) {
  std::vector<T> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = (T)src[i];
  return dst;
}

template <typename T>
static std::vector<float> to_float(const std::vector<T>& src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = (float)src[i];
  return dst;
}

static std::vector<float> rand_vec(size_t n, std::mt19937& rng, float lo = -1.f, float hi = 1.f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}

template <typename T>
static T* gpu_alloc(const std::vector<T>& src) {
  size_t bytes = src.size() * sizeof(T);
  auto& alloc = get_allocator<Device::METAL>();
  T* ptr = static_cast<T*>(alloc.allocate(bytes, 0));
  std::memcpy(ptr, src.data(), bytes);
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

// ---------------------------------------------------------------------------
// Test: fused LayerNorm + GEMV
// ---------------------------------------------------------------------------

template <typename T>
static void test_fused_ln_gemv(const char* label, dim_t M, dim_t K, dim_t N,
                                float tol, bool with_beta = true) {
  std::mt19937 rng(42);
  float eps = 1e-5f;

  auto x_f32     = rand_vec(M * K, rng);
  auto gamma_f32  = rand_vec(K, rng, 0.5f, 1.5f);
  auto beta_f32   = with_beta ? rand_vec(K, rng, -0.5f, 0.5f) : std::vector<float>();
  auto W_f32      = rand_vec(N * K, rng, -0.1f, 0.1f);

  // CPU reference
  std::vector<float> norm_ref(M * K);
  ref_layer_norm_f32(x_f32.data(), gamma_f32.data(),
                      with_beta ? beta_f32.data() : nullptr,
                      norm_ref.data(), M, K, eps);
  std::vector<float> y_ref(M * N);
  ref_gemv_f32(norm_ref.data(), W_f32.data(), y_ref.data(), M, K, N);

  // GPU
  auto x_gpu     = to_type<T>(x_f32);
  auto gamma_gpu = to_type<T>(gamma_f32);
  auto beta_gpu  = with_beta ? to_type<T>(beta_f32) : std::vector<T>();
  auto W_gpu     = to_type<T>(W_f32);
  std::vector<T> y_gpu(M * N, T(0));

  T* d_x     = gpu_alloc(x_gpu);
  T* d_gamma = gpu_alloc(gamma_gpu);
  T* d_beta  = with_beta ? gpu_alloc(beta_gpu) : nullptr;
  T* d_W     = gpu_alloc(W_gpu);
  T* d_y     = gpu_alloc_empty<T>(M * N);

  metal::fused_layer_norm_gemm_metal(d_x, d_gamma, d_beta, d_W, d_y,
                                      M, K, N, eps);
  metal::commit_and_wait();

  std::memcpy(y_gpu.data(), d_y, M * N * sizeof(T));
  auto y_got = to_float(y_gpu);

  CHECK_CLOSE(label, y_got, y_ref, tol);

  gpu_free(d_x);
  gpu_free(d_gamma);
  if (d_beta) gpu_free(d_beta);
  gpu_free(d_W);
  gpu_free(d_y);
}

// ---------------------------------------------------------------------------
// Test: fused RMSNorm + GEMV
// ---------------------------------------------------------------------------

template <typename T>
static void test_fused_rms_gemv(const char* label, dim_t M, dim_t K, dim_t N,
                                 float tol) {
  std::mt19937 rng(123);
  float eps = 1e-6f;

  auto x_f32     = rand_vec(M * K, rng);
  auto gamma_f32  = rand_vec(K, rng, 0.5f, 1.5f);
  auto W_f32      = rand_vec(N * K, rng, -0.1f, 0.1f);

  // CPU reference
  std::vector<float> norm_ref(M * K);
  ref_rms_norm_f32(x_f32.data(), gamma_f32.data(), norm_ref.data(), M, K, eps);
  std::vector<float> y_ref(M * N);
  ref_gemv_f32(norm_ref.data(), W_f32.data(), y_ref.data(), M, K, N);

  // GPU
  auto x_gpu     = to_type<T>(x_f32);
  auto gamma_gpu = to_type<T>(gamma_f32);
  auto W_gpu     = to_type<T>(W_f32);
  std::vector<T> y_gpu(M * N, T(0));

  T* d_x     = gpu_alloc(x_gpu);
  T* d_gamma = gpu_alloc(gamma_gpu);
  T* d_W     = gpu_alloc(W_gpu);
  T* d_y     = gpu_alloc_empty<T>(M * N);

  metal::fused_rms_norm_gemm_metal(d_x, d_gamma, d_W, d_y,
                                    M, K, N, eps);
  metal::commit_and_wait();

  std::memcpy(y_gpu.data(), d_y, M * N * sizeof(T));
  auto y_got = to_float(y_gpu);

  CHECK_CLOSE(label, y_got, y_ref, tol);

  gpu_free(d_x);
  gpu_free(d_gamma);
  gpu_free(d_W);
  gpu_free(d_y);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  @autoreleasepool {
    std::printf("=== Fused LayerNorm/RMSNorm + GEMV test ===\n\n");

    // LayerNorm + GEMV: float32
    std::printf("LayerNorm+GEMV float32:\n");
    test_fused_ln_gemv<float>("f32 M=1  K=512 N=2048",  1, 512, 2048, 1e-4f);
    test_fused_ln_gemv<float>("f32 M=4  K=512 N=2048",  4, 512, 2048, 1e-4f);
    test_fused_ln_gemv<float>("f32 M=16 K=512 N=2048", 16, 512, 2048, 1e-4f);

    // LayerNorm + GEMV: float16
    std::printf("\nLayerNorm+GEMV float16:\n");
    test_fused_ln_gemv<ct2_f16>("f16 M=1  K=512 N=2048",  1, 512, 2048, 1e-1f);
    test_fused_ln_gemv<ct2_f16>("f16 M=4  K=512 N=2048",  4, 512, 2048, 1e-1f);
    test_fused_ln_gemv<ct2_f16>("f16 M=16 K=512 N=2048", 16, 512, 2048, 1e-1f);

    // LayerNorm + GEMV: bfloat16
    std::printf("\nLayerNorm+GEMV bfloat16:\n");
    test_fused_ln_gemv<ct2_bf16>("bf16 M=1  K=512 N=2048",  1, 512, 2048, 5e-1f);
    test_fused_ln_gemv<ct2_bf16>("bf16 M=4  K=512 N=2048",  4, 512, 2048, 5e-1f);
    test_fused_ln_gemv<ct2_bf16>("bf16 M=16 K=512 N=2048", 16, 512, 2048, 5e-1f);

    // RMSNorm + GEMV: float32
    std::printf("\nRMSNorm+GEMV float32:\n");
    test_fused_rms_gemv<float>("f32 M=1  K=1024 N=512",  1, 1024, 512, 1e-4f);
    test_fused_rms_gemv<float>("f32 M=4  K=1024 N=512",  4, 1024, 512, 1e-4f);
    test_fused_rms_gemv<float>("f32 M=16 K=1024 N=512", 16, 1024, 512, 1e-4f);

    // RMSNorm + GEMV: float16
    std::printf("\nRMSNorm+GEMV float16:\n");
    test_fused_rms_gemv<ct2_f16>("f16 M=1  K=1024 N=512",  1, 1024, 512, 1e-1f);
    test_fused_rms_gemv<ct2_f16>("f16 M=4  K=1024 N=512",  4, 1024, 512, 1e-1f);
    test_fused_rms_gemv<ct2_f16>("f16 M=16 K=1024 N=512", 16, 1024, 512, 1e-1f);

    // RMSNorm + GEMV: bfloat16
    std::printf("\nRMSNorm+GEMV bfloat16:\n");
    test_fused_rms_gemv<ct2_bf16>("bf16 M=1  K=1024 N=512",  1, 1024, 512, 5e-1f);
    test_fused_rms_gemv<ct2_bf16>("bf16 M=4  K=1024 N=512",  4, 1024, 512, 5e-1f);
    test_fused_rms_gemv<ct2_bf16>("bf16 M=16 K=1024 N=512", 16, 1024, 512, 5e-1f);

    // Large shapes
    std::printf("\nLarge shapes:\n");
    test_fused_ln_gemv<float>("f32 LN M=1 K=4096 N=6144", 1, 4096, 6144, 5e-4f);
    test_fused_rms_gemv<float>("f32 RMS M=1 K=2048 N=4096", 1, 2048, 4096, 5e-4f);

    // No-beta variant
    std::printf("\nNo-beta:\n");
    test_fused_ln_gemv<float>("f32 LN no-beta M=4 K=512 N=2048", 4, 512, 2048, 1e-4f, false);

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
  }
}
