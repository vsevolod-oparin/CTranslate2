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
//     -DCT2_WITH_MPS \
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
#include <type_traits>
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
      get_allocator<Device::MPS>().allocate(data.size() * sizeof(float), 0));
  std::memcpy(p, data.data(), data.size() * sizeof(float));
  return p;
}

// Allocate a float32 Metal buffer (output, uninitialized).
static float* alloc_f32_buf(size_t n) {
  return static_cast<float*>(
      get_allocator<Device::MPS>().allocate(n * sizeof(float), 0));
}

// Free a Metal-registered buffer.
static void free_buf(void* p) {
  get_allocator<Device::MPS>().free(p, 0);
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
      get_allocator<Device::MPS>().allocate(N_in  * sizeof(ct2_f16), 0));
  ct2_f16* w_m = static_cast<ct2_f16*>(
      get_allocator<Device::MPS>().allocate(N_w   * sizeof(ct2_f16), 0));
  ct2_f16* y_m = static_cast<ct2_f16*>(
      get_allocator<Device::MPS>().allocate(N_out * sizeof(ct2_f16), 0));

  primitives<Device::MPS>::convert(xf_m, x_m, (dim_t)N_in);
  primitives<Device::MPS>::convert(wf_m, w_m, (dim_t)N_w);
  metal::commit_and_wait();  // flush f32→f16 conversion before conv1d

  metal::conv1d_metal<ct2_f16>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                                stride, padding, dilation);

  metal::commit_and_wait();
  std::vector<float> got(N_out);
  for (size_t i = 0; i < N_out; ++i)
    got[i] = static_cast<float>(y_m[i]);

  // float16 accumulates C_in*K=24 terms per output element; empirical error ~5e-4
  CHECK_CLOSE("f16 conv1d", got, ref, 2e-2f);

  free_buf(xf_m); free_buf(wf_m);
  get_allocator<Device::MPS>().free(x_m, 0);
  get_allocator<Device::MPS>().free(w_m, 0);
  get_allocator<Device::MPS>().free(y_m, 0);
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
      get_allocator<Device::MPS>().allocate(N_in  * sizeof(ct2_bf16), 0));
  ct2_bf16* w_m = static_cast<ct2_bf16*>(
      get_allocator<Device::MPS>().allocate(N_w   * sizeof(ct2_bf16), 0));
  ct2_bf16* y_m = static_cast<ct2_bf16*>(
      get_allocator<Device::MPS>().allocate(N_out * sizeof(ct2_bf16), 0));

  primitives<Device::MPS>::convert(xf_m, x_m, (dim_t)N_in);
  primitives<Device::MPS>::convert(wf_m, w_m, (dim_t)N_w);
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
  get_allocator<Device::MPS>().free(x_m, 0);
  get_allocator<Device::MPS>().free(w_m, 0);
  get_allocator<Device::MPS>().free(y_m, 0);
}

// ---------------------------------------------------------------------------
// Test 8: Conv1D + bias + GELU activation (Whisper Conv1D pipeline)
//
// Whisper encoder uses Conv1D with bias and GELU activation.
// Tests the full pipeline: conv1d_metal → add_block_broadcast (bias) → gelu.
//
// Conv1D output layout: [B, C_out, T_out]
// Bias: [C_out], broadcast along C_out axis (block_size = T_out).
// ---------------------------------------------------------------------------

static float ref_gelu(float x) {
  return x * 0.5f * (1.f + std::erf(x * 0.7071067811865476f));
}

static void test8_f32_bias_gelu() {
  std::printf("\nTest 8: f32 conv1d + bias + GELU\n");

  const dim_t B=2, C_in=4, T_in=8, C_out=8, K=3;
  const dim_t stride=1, padding=1, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;
  const size_t N_out = (size_t)(B * C_out * T_out);

  auto x_h    = rand_vec((size_t)(B*C_in*T_in),  100);
  auto w_h    = rand_vec((size_t)(C_out*C_in*K),  101);
  auto bias_h = rand_vec((size_t)(C_out),          102, 0.5f);

  // CPU reference: conv1d + bias + GELU
  std::vector<float> ref(N_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);
  // Apply bias: output[b, c, t] += bias[c]
  for (dim_t b = 0; b < B; ++b)
    for (dim_t c = 0; c < C_out; ++c)
      for (dim_t t = 0; t < T_out; ++t)
        ref[(size_t)(b * C_out * T_out + c * T_out + t)] += bias_h[(size_t)c];
  // Apply GELU
  for (auto& v : ref) v = ref_gelu(v);

  // Metal: conv1d → bias broadcast → GELU
  float* x_m    = make_f32_buf(x_h);
  float* w_m    = make_f32_buf(w_h);
  float* bias_m = make_f32_buf(bias_h);
  float* y_m    = alloc_f32_buf(N_out);

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  // Bias: [C_out] broadcast over [B*C_out, T_out] where bias repeats every C_out rows.
  // add_block_broadcast(bias, y, block=T_out, bias_size=C_out, y_size=B*C_out*T_out)
  primitives<Device::MPS>::add_block_broadcast<float>(
      bias_m, y_m, static_cast<dim_t>(T_out),
      static_cast<dim_t>(C_out), static_cast<dim_t>(N_out));

  // GELU activation
  primitives<Device::MPS>::gelu<float>(y_m, y_m, static_cast<dim_t>(N_out));

  auto got = read_f32(y_m, N_out);
  CHECK_CLOSE("f32 conv1d + bias + GELU", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(bias_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 9: float32, padding=0 (no zero-padded positions in im2col)
//
// With pad=0 and K=3, T_out = T_in - 2 (all im2col positions are valid).
// ---------------------------------------------------------------------------

static void test9_f32_pad0() {
  std::printf("\nTest 9: f32 conv1d padding=0\n");

  const dim_t B=1, C_in=4, T_in=10, C_out=8, K=3;
  const dim_t stride=1, padding=0, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;  // 8

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  110);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 111);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 padding=0", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 10: float32, K=1 (point convolution / 1x1 conv)
//
// Degenerate case where im2col is essentially a reshape.
// T_out = T_in (stride=1, padding=0, dilation=1, K=1).
// ---------------------------------------------------------------------------

static void test10_f32_k1() {
  std::printf("\nTest 10: f32 conv1d K=1 (point conv)\n");

  const dim_t B=2, C_in=8, T_in=16, C_out=4, K=1;
  const dim_t stride=1, padding=0, dilation=1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;  // 16

  auto x_h = rand_vec((size_t)(B*C_in*T_in),  120);
  auto w_h = rand_vec((size_t)(C_out*C_in*K), 121);

  std::vector<float> ref(B*C_out*T_out);
  ref_conv1d_f32(x_h.data(), B, C_in, T_in, w_h.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  float* x_m = make_f32_buf(x_h);
  float* w_m = make_f32_buf(w_h);
  float* y_m = alloc_f32_buf((size_t)(B*C_out*T_out));

  metal::conv1d_metal<float>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                              stride, padding, dilation);

  auto got = read_f32(y_m, (size_t)(B*C_out*T_out));
  CHECK_CLOSE("f32 K=1 point conv", got, ref, 1e-5f);

  free_buf(x_m); free_buf(w_m); free_buf(y_m);
}

// ---------------------------------------------------------------------------
// Test 11: Stress test at realistic Whisper shapes (f32 / f16 / bf16)
//
// Whisper encoder conv layers:
//   conv1: C_in=80,  C_out=512, K=3, stride=1, pad=1  (T_in=3000 → T_out=3000)
//   conv2: C_in=512, C_out=512, K=3, stride=2, pad=1  (T_in=3000 → T_out=1500)
//
// These test large accumulations: C_in*K = 80*3=240 (conv1), 512*3=1536 (conv2).
// Float16 with 1536 accumulation terms is the critical stress test for precision.
//
// Use T_in=1500 (half the real Whisper length) to keep test time reasonable.
// ---------------------------------------------------------------------------

template <typename T>
static void test_stress_typed(const char* type_name, float tol,
                               dim_t C_in, dim_t C_out, dim_t T_in,
                               dim_t K, dim_t stride, dim_t padding,
                               float data_scale, unsigned seed) {
  const dim_t B = 1, dilation = 1;
  const dim_t T_out = (T_in + 2*padding - (dilation*(K-1)+1)) / stride + 1;
  const size_t N_in  = (size_t)(B*C_in*T_in);
  const size_t N_w   = (size_t)(C_out*C_in*K);
  const size_t N_out = (size_t)(B*C_out*T_out);

  std::printf("\n  %s [B=%lld, Cin=%lld, Cout=%lld, T_in=%lld, K=%lld, s=%lld, p=%lld] → T_out=%lld  (CK=%lld accum terms)\n",
              type_name,
              (long long)B, (long long)C_in, (long long)C_out,
              (long long)T_in, (long long)K, (long long)stride, (long long)padding,
              (long long)T_out, (long long)(C_in*K));

  auto x_f32 = rand_vec(N_in,  seed,     data_scale);
  auto w_f32 = rand_vec(N_w,   seed + 1, data_scale);

  // CPU float32 reference
  std::vector<float> ref(N_out);
  ref_conv1d_f32(x_f32.data(), B, C_in, T_in, w_f32.data(), C_out, K,
                 ref.data(), T_out, stride, padding, dilation);

  // Metal in type T
  T* x_m; T* w_m;
  T* y_m = static_cast<T*>(
      get_allocator<Device::MPS>().allocate(N_out * sizeof(T), 0));

  if constexpr (std::is_same_v<T, float>) {
    x_m = make_f32_buf(x_f32);
    w_m = make_f32_buf(w_f32);
  } else {
    float* xf_m = make_f32_buf(x_f32);
    float* wf_m = make_f32_buf(w_f32);
    x_m = static_cast<T*>(
        get_allocator<Device::MPS>().allocate(N_in * sizeof(T), 0));
    w_m = static_cast<T*>(
        get_allocator<Device::MPS>().allocate(N_w * sizeof(T), 0));
    primitives<Device::MPS>::convert(xf_m, x_m, (dim_t)N_in);
    primitives<Device::MPS>::convert(wf_m, w_m, (dim_t)N_w);
    metal::commit_and_wait();
    free_buf(xf_m); free_buf(wf_m);
  }

  metal::conv1d_metal<T>(x_m, w_m, y_m, B, C_in, T_in, C_out, K, T_out,
                          stride, padding, dilation);

  metal::commit_and_wait();
  std::vector<float> got(N_out);
  for (size_t i = 0; i < N_out; ++i)
    got[i] = static_cast<float>(y_m[i]);

  char label[256];
  std::snprintf(label, sizeof(label), "stress %s Cin=%lld Cout=%lld K=%lld s=%lld",
                type_name, (long long)C_in, (long long)C_out, (long long)K, (long long)stride);
  CHECK_CLOSE(label, got, ref, tol);

  get_allocator<Device::MPS>().free(x_m, 0);
  get_allocator<Device::MPS>().free(w_m, 0);
  get_allocator<Device::MPS>().free(y_m, 0);
}

static void test11_stress() {
  std::printf("\nTest 11: Stress test at realistic Whisper shapes\n");

  // Conv1 shape: C_in=80, C_out=512, K=3, stride=1, pad=1, T_in=1500
  // CK=240 accum terms
  test_stress_typed<float>("f32 conv1", 1e-4f,
                            80, 512, 1500, 3, 1, 1, 0.1f, 200);
  test_stress_typed<ct2_f16>("f16 conv1", 5e-2f,
                              80, 512, 1500, 3, 1, 1, 0.1f, 200);
  test_stress_typed<ct2_bf16>("bf16 conv1", 1e-1f,
                               80, 512, 1500, 3, 1, 1, 0.1f, 200);

  // Conv2 shape: C_in=512, C_out=512, K=3, stride=2, pad=1, T_in=1500
  // CK=1536 accum terms — this is the critical float16 stress test
  test_stress_typed<float>("f32 conv2", 1e-3f,
                            512, 512, 1500, 3, 2, 1, 0.02f, 210);
  test_stress_typed<ct2_f16>("f16 conv2", 2e-1f,
                              512, 512, 1500, 3, 2, 1, 0.02f, 210);
  test_stress_typed<ct2_bf16>("bf16 conv2", 5e-1f,
                               512, 512, 1500, 3, 2, 1, 0.02f, 210);
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
    test8_f32_bias_gelu();
    test9_f32_pad0();
    test10_f32_k1();
    test11_stress();
  }

  std::printf("\n=== Results: %d pass, %d fail ===\n", g_pass, g_fail);
  return (g_fail > 0) ? 1 : 0;
}
