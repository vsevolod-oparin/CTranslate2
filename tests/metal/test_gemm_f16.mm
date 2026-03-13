// Quick test for float16 GEMM with float32 accumulation on Metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cmath>
#include "ctranslate2/types.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/allocator.h"
#include "metal/utils.h"

using ct2_f16 = ctranslate2::float16_t;

int main() {
  constexpr int M = 2, N = 4, K = 3;
  float a_f32[] = {1,2,3, 4,5,6};
  float b_f32[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
  float expected[] = {38,44,50,56, 83,98,113,128};

  auto& alloc = ctranslate2::get_allocator<ctranslate2::Device::MPS>();
  auto* a_ptr = static_cast<ct2_f16*>(alloc.allocate(M * K * sizeof(ct2_f16), 0));
  auto* b_ptr = static_cast<ct2_f16*>(alloc.allocate(K * N * sizeof(ct2_f16), 0));
  auto* c_ptr = static_cast<ct2_f16*>(alloc.allocate(M * N * sizeof(ct2_f16), 0));

  auto* a_raw = reinterpret_cast<__fp16*>(a_ptr);
  auto* b_raw = reinterpret_cast<__fp16*>(b_ptr);
  auto* c_raw = reinterpret_cast<__fp16*>(c_ptr);
  for (int i = 0; i < M*K; i++) a_raw[i] = (__fp16)a_f32[i];
  for (int i = 0; i < K*N; i++) b_raw[i] = (__fp16)b_f32[i];
  for (int i = 0; i < M*N; i++) c_raw[i] = (__fp16)0.0f;

  // Test m=2 GEMM
  ctranslate2::primitives<ctranslate2::Device::MPS>::gemm(
      false, false, false, false,
      M, N, K,
      1.0f,
      a_ptr, K,
      b_ptr, N,
      0.0f,
      c_ptr, N,
      static_cast<const ct2_f16*>(nullptr));

  ctranslate2::metal::commit_and_wait();

  bool ok = true;
  printf("GEMM test: C[%d,%d] = A[%d,%d] * B[%d,%d]\n", M, N, M, K, K, N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float got = (float)c_raw[i * N + j];
      float exp = expected[i * N + j];
      bool match = fabs(got - exp) < 1.0f;
      if (!match) ok = false;
      printf("  C[%d,%d] = %.1f (expected %.1f) %s\n", i, j, got, exp, match ? "OK" : "FAIL");
    }
  }

  // Test m=1 GEMM
  printf("\nm=1 GEMM test:\n");
  auto* a1_ptr = static_cast<ct2_f16*>(alloc.allocate(1 * K * sizeof(ct2_f16), 0));
  auto* c1_ptr = static_cast<ct2_f16*>(alloc.allocate(1 * N * sizeof(ct2_f16), 0));
  auto* a1_raw = reinterpret_cast<__fp16*>(a1_ptr);
  auto* c1_raw = reinterpret_cast<__fp16*>(c1_ptr);
  for (int i = 0; i < K; i++) a1_raw[i] = (__fp16)a_f32[i];
  for (int i = 0; i < N; i++) c1_raw[i] = (__fp16)0.0f;

  ctranslate2::primitives<ctranslate2::Device::MPS>::gemm(
      false, false, false, false,
      1, N, K,
      1.0f,
      a1_ptr, K,
      b_ptr, N,
      0.0f,
      c1_ptr, N,
      static_cast<const ct2_f16*>(nullptr));

  ctranslate2::metal::commit_and_wait();

  for (int j = 0; j < N; j++) {
    float got = (float)c1_raw[j];
    float exp = expected[j];
    bool match = fabs(got - exp) < 1.0f;
    if (!match) ok = false;
    printf("  C[0,%d] = %.1f (expected %.1f) %s\n", j, got, exp, match ? "OK" : "FAIL");
  }

  // Test larger GEMM: M=13, N=512, K=512 (encoder-like)
  printf("\nLarger GEMM test [13x512] = [13x512] * [512x512]:\n");
  constexpr int M2 = 13, N2 = 512, K2 = 512;
  auto* a2 = static_cast<ct2_f16*>(alloc.allocate(M2 * K2 * sizeof(ct2_f16), 0));
  auto* b2 = static_cast<ct2_f16*>(alloc.allocate(K2 * N2 * sizeof(ct2_f16), 0));
  auto* c2 = static_cast<ct2_f16*>(alloc.allocate(M2 * N2 * sizeof(ct2_f16), 0));
  auto* a2r = reinterpret_cast<__fp16*>(a2);
  auto* b2r = reinterpret_cast<__fp16*>(b2);
  auto* c2r = reinterpret_cast<__fp16*>(c2);

  // Fill with small values
  for (int i = 0; i < M2*K2; i++) a2r[i] = (__fp16)(0.01f * (i % 100));
  for (int i = 0; i < K2*N2; i++) b2r[i] = (__fp16)(0.01f * (i % 100));
  for (int i = 0; i < M2*N2; i++) c2r[i] = (__fp16)0.0f;

  // Compute expected on CPU
  std::vector<float> exp2(M2 * N2, 0);
  for (int i = 0; i < M2; i++)
    for (int j = 0; j < N2; j++) {
      float acc = 0;
      for (int p = 0; p < K2; p++)
        acc += (float)a2r[i*K2+p] * (float)b2r[p*N2+j];
      exp2[i*N2+j] = acc;
    }

  ctranslate2::primitives<ctranslate2::Device::MPS>::gemm(
      false, false, false, false,
      M2, N2, K2,
      1.0f, a2, K2, b2, N2,
      0.0f, c2, N2,
      static_cast<const ct2_f16*>(nullptr));

  ctranslate2::metal::commit_and_wait();

  float max_err = 0;
  int fail_count = 0;
  for (int i = 0; i < M2 * N2; i++) {
    float got = (float)c2r[i];
    float exp = exp2[i];
    float err = fabs(got - exp);
    if (err > max_err) max_err = err;
    if (err > 0.5f) fail_count++;
  }
  printf("  max_err = %.4f, fails (>0.5) = %d / %d\n", max_err, fail_count, M2*N2);
  if (fail_count > 0) ok = false;

  alloc.free(a_ptr); alloc.free(b_ptr); alloc.free(c_ptr);
  alloc.free(a1_ptr); alloc.free(c1_ptr);
  alloc.free(a2); alloc.free(b2); alloc.free(c2);

  printf("\n%s\n", ok ? "ALL PASSED" : "FAILURES DETECTED");
  return ok ? 0 : 1;
}
