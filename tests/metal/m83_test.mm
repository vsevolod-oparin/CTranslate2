// tests/metal/m83_test.mm
//
// M8.3 — Conv1D on Metal (im2col + GEMM): end-to-end integration test.
//
// Calls metal::conv1d_metal<T>() directly (following the same pattern as
// m82_test.mm which calls metal::sdpa_metal<T>() directly).  This avoids
// linking CPU primitives and ops dispatch machinery.
//
// CPU reference: ref_conv1d_f32() implemented inline.
//
// Tests:
//   1. float32  stride=1, pad=1          (B=1, C_in=4, T_in=8, C_out=8,  K=3)
//   2. float32  stride=2                 (B=1, C_in=4, T_in=8, C_out=8,  K=3)
//   3. float32  dilation=2, batch=2      (B=2, C_in=4, T_in=12, C_out=4, K=3)
//   4. float32  batch=3                  (B=3, C_in=4, T_in=8, C_out=8,  K=3)
//   5. float32  Whisper-like shape       (B=1, C_in=80, T_in=64, C_out=128, K=3)
//   6. float16  stride=1, pad=1          (B=1, C_in=8, T_in=16, C_out=8, K=3)
//   7. bfloat16 stride=1, pad=1          (B=1, C_in=8, T_in=16, C_out=8, K=3)
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/m83_test.mm \
//     src/metal/ops_conv1d.mm \
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
//     -o m83_test && ./m83_test

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

// Qualify float16/bfloat16 before 'using namespace ctranslate2' to avoid
// ARM vector type header ambiguity (M5.1).
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
// CPU reference: float32 conv1d
//   input:  [B, C_in, T_in]
//   weight: [C_out, C_in, K]
//   output: [B, C_out, T_out]
// ---------------------------------------------------------------------------

static void ref_conv1d_f32(
    const float* input,  dim_t B,  dim_t C_in,  dim_t T_in,
    const float* weight, dim_t C_out, dim_t K,
    float* output, dim_t T_out,
    dim_t stride, dim_t padding, dim_t dilation) {
  for (dim_t b = 0; b < B; ++b)
    for (dim_t co = 0; co < C_out; ++co)
      for (dim_t to = 0; to < T_out; ++to) {
        float sum = 0.f;
        for (dim_t ci = 0; ci < C_in; ++ci)
          for (dim_t k = 0; k < K; ++k) {
            dim_t ti = to * stride - padding + k * dilation;
            if (ti >= 0 && ti < T_in)
              sum += input [b * C_in * T_in + ci * T_in + (size_t)ti]
                   * weight[co * C_in * K  + ci * K    + (size_t)k ];
          }
        output[b * C_out * T_out + co * T_out + to] = sum;
      }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<float> rand_vec(size_t n, unsigned seed, float scale = 1.f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

// Allocate a float32 Metal buffer, copy data in (Shared memory, direct memcpy).
static float* make_f32_buf(const std::vector<float>& data) {
  float* p = static_cast<float*>(
      get_allocator<Device::METAL>().allocate(data.size() * sizeof(float), 0));
  std::memcpy(p, data.data(), data.size() * sizeof(float));
  return p;
}

// Allocate a float32 Metal buffer (output, uninitialized).
static float* alloc_f32_buf(size_t n) {
  return static_cast<float*>(
      get_allocator<Device::METAL>().allocate(n * sizeof(float), 0));
}

// Free a Metal-registered buffer.
static void free_buf(void* p) {
  get_allocator<Device::METAL>().free(p, 0);
}

// Read Metal buffer to std::vector<float> after GPU sync.
static std::vector<float> read_f32(const float* p, size_t n) {
  metal::commit_and_wait();
  std::vector<float> out(n);
  std::memcpy(out.data(), p, n * sizeof(float));
  return out;
}

// ---------------------------------------------------------------------------
// Test 1: float32 stride=1, pad=1
// ---------------------------------------------------------------------------

static void test1_f32_basic() {
  std::printf("\nTest 1: f32 conv1d stride=1 pad=1\n");

  const dim_t B=1, C_in=4, T_in=8, C_out=8, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  42);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 43);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 basic", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 2: float32 stride=2
// ---------------------------------------------------------------------------

static void test2_f32_stride2() {
  std::printf("\nTest 2: f32 conv1d stride=2\n");

  const dim_t B=1, C_in=4, T_in=8, C_out=8, K=3;
  const dim_t stride=2, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  44);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 45);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 stride=2", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 3: float32 dilation=2, batch=2
// ---------------------------------------------------------------------------

static void test3_f32_dilation() {
  std::printf("\nTest 3: f32 conv1d dilation=2 batch=2\n");

  const dim_t B=2, C_in=4, T_in=12, C_out=4, K=3;
  const dim_t stride=1, padding=2, dilation=2;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  46);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 47);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 dilation=2 batch=2", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 4: float32 batch=3
// ---------------------------------------------------------------------------

static void test4_f32_batch3() {
  std::printf("\nTest 4: f32 conv1d batch=3\n");

  const dim_t B=3, C_in=4, T_in=8, C_out=8, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  60);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 61);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 batch=3", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 5: float32, Whisper-like shape
//   Whisper encoder conv1: C_in=80, C_out=128, K=3, stride=1, pad=1
//   Use small T_in=32 to keep test fast.
// ---------------------------------------------------------------------------

static void test5_f32_whisper() {
  std::printf("\nTest 5: f32 conv1d Whisper-like (C_in=80, C_out=128, K=3)\n");

  const dim_t B=1, C_in=80, T_in=32, C_out=128, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  70, 0.1f);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 71, 0.05f);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  // Larger accumulation (C_in*K=240): tolerance 1e-4
  CHECK_CLOSE("f32 Whisper-like", got, ref, 1e-4f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 6: float16
// ---------------------------------------------------------------------------

static void test6_f16() {
  std::printf("\nTest 6: f16 conv1d\n");

  const dim_t B=1, C_in=8, T_in=16, C_out=8, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;
  const size_t N_in  = (size_t)(B*C_in*T_in);
  const size_t N_w   = (size_t)(C_out*C_in*K);
  const size_t N_out = (size_t)(B*C_out*T_out);

  auto x_f32 = rand_vec(N_in,  80, 0.5f);
  auto w_f32 = rand_vec(N_w,   81, 0.5f);

  // CPU float32 reference
  std::vector<float> ref(N_out);
  ref_conv1d_f32(x_f32.data(), B, C_in, T_in, w_f32.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  // Allocate Metal f32 buffers for input, then convert to f16 in-place via a
  // separate f16 buffer.
  float* xf_m  = make_f32_buf(x_f32);
  float* wf_m  = make_f32_buf(w_f32);
  ct2_f16* x_m = static_cast<ct2_f16*>(
      get_allocator<Device::METAL>().allocate(N_in  * sizeof(ct2_f16), 0));
  ct2_f16* w_m = static_cast<ct2_f16*>(
      get_allocator<Device::METAL>().allocate(N_w   * sizeof(ct2_f16), 0));
  ct2_f16* y_m = static_cast<ct2_f16*>(
      get_allocator<Device::METAL>().allocate(N_out * sizeof(ct2_f16), 0));

  primitives<Device::METAL>::convert(xf_m, x_m, (dim_t)N_in);
  primitives<Device::METAL>::convert(wf_m, w_m, (dim_t)N_w);
  metal::commit_and_wait();  // flush f32→f16 conversion before conv1d

  metal::conv1d_metal<ct2_f16>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                                stride, padding, dilation);

  metal::commit_and_wait();
  std::vector<float> got(N_out);
  for (size_t i = 0; i < N_out; ++i)
    got[i] = static_cast<float>(y_m[i]);

  // float16 accumulates ~C_in*K=24 terms; tolerance 2e-2
  CHECK_CLOSE("f16 conv1d", got, ref, 2e-2f);

  free_buf(xf_m); free_buf(wf_m);
  get_allocator<Device::METAL>().free(x_m, 0);
  get_allocator<Device::METAL>().free(w_m, 0);
  get_allocator<Device::METAL>().free(y_m, 0);
}

// ---------------------------------------------------------------------------
// Test 7: bfloat16
// ---------------------------------------------------------------------------

static void test7_bf16() {
  std::printf("\nTest 7: bf16 conv1d\n");

  const dim_t B=1, C_in=8, T_in=16, C_out=8, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;
  const size_t N_in  = (size_t)(B*C_in*T_in);
  const size_t N_w   = (size_t)(C_out*C_in*K);
  const size_t N_out = (size_t)(B*C_out*T_out);

  auto x_f32 = rand_vec(N_in,  90, 0.5f);
  auto w_f32 = rand_vec(N_w,   91, 0.5f);

  // CPU float32 reference
  std::vector<float> ref(N_out);
  ref_conv1d_f32(x_f32.data(), B, C_in, T_in, w_f32.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  // f32 → bf16 via Metal convert
  float* xf_m   = make_f32_buf(x_f32);
  float* wf_m   = make_f32_buf(w_f32);
  ct2_bf16* x_m = static_cast<ct2_bf16*>(
      get_allocator<Device::METAL>().allocate(N_in  * sizeof(ct2_bf16), 0));
  ct2_bf16* w_m = static_cast<ct2_bf16*>(
      get_allocator<Device::METAL>().allocate(N_w   * sizeof(ct2_bf16), 0));
  ct2_bf16* y_m = static_cast<ct2_bf16*>(
      get_allocator<Device::METAL>().allocate(N_out * sizeof(ct2_bf16), 0));

  primitives<Device::METAL>::convert(xf_m, x_m, (dim_t)N_in);
  primitives<Device::METAL>::convert(wf_m, w_m, (dim_t)N_w);
  metal::commit_and_wait();

  metal::conv1d_metal<ct2_bf16>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                                  stride, padding, dilation);

  metal::commit_and_wait();
  std::vector<float> got(N_out);
  for (size_t i = 0; i < N_out; ++i)
    got[i] = static_cast<float>(y_m[i]);

  // bfloat16 has ~2 decimal digits of precision; tolerance 5e-2
  CHECK_CLOSE("bf16 conv1d", got, ref, 5e-2f);

  free_buf(xf_m); free_buf(wf_m);
  get_allocator<Device::METAL>().free(x_m, 0);
  get_allocator<Device::METAL>().free(w_m, 0);
  get_allocator<Device::METAL>().free(y_m, 0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M8.3 Conv1D Metal tests ===\n");

  @autoreleasepool {
    test1_f32_basic();
    test2_f32_stride2();
    test3_f32_dilation();
    test4_f32_batch3();
    test5_f32_whisper();
    test6_f16();
    test7_bf16();
  }

  std::printf("\n=== Results: %d pass, %d fail ===\n", g_pass, g_fail);
  return (g_fail > 0) ? 1 : 0;
}
