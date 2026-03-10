// Tests for 4.4 — large tensor transpose.
//
// The existing transpose_test.mm uses small dimensions (e.g., [3,5,7], [2,3,5,7]).
// This file focuses on production-scale shapes where gid spans the full 20-bit
// range, verifying that the MSL index decomposition arithmetic:
//   i0 =  gid / b_s0
//   i1 = (gid / b_s1) % bd1
//   i2 =  gid % b_s1
// remains correct throughout.
//
// Test cases:
//   2D  [1024, 512]          = 524,288 elements   — 2D row/col mix
//   3D  [32, 128, 256]       = 1,048,576 elements — attention-style
//   4D  [4, 32, 64, 128]     = 1,048,576 elements — MHA [batch,heads,seq,dim]
//   4D  [1, 16, 512, 128]    = 1,048,576 elements — decode phase, single batch
//
// All cases compare GPU output against a CPU reference implementation.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/large_transpose_test.mm \
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
//     -o large_transpose_test && ./large_transpose_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// Resolve ::float16_t / ctranslate2::float16_t conflict from arm_vector_types.h.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int g_pass = 0, g_fail = 0;

static void report(bool ok, const char* name) {
  if (ok) { ++g_pass; std::printf("  PASS  %s\n", name); }
  else     { ++g_fail; std::printf("  FAIL  %s\n", name); }
}

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// CPU reference implementations (mirror transpose_test.mm)
// ---------------------------------------------------------------------------

// 2D: a is [dims[0], dims[1]], b is [dims[1], dims[0]] (implicit perm [1,0])
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
  dim_t pbs[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]], b_stride[perm_ind[2]]};
  for (dim_t i0 = 0; i0 < dims[0]; ++i0)
    for (dim_t i1 = 0; i1 < dims[1]; ++i1)
      for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
        b[i0*pbs[0] + i1*pbs[1] + i2*pbs[2]] =
            a[i0*a_stride[0] + i1*a_stride[1] + i2];
      }
}

template <typename T>
static void cpu_transpose_4d(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
  dim_t a_stride[4] = {dims[1]*dims[2]*dims[3], dims[2]*dims[3], dims[3], 1};
  dim_t b_dims[4]   = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
  dim_t b_stride[4] = {b_dims[1]*b_dims[2]*b_dims[3], b_dims[2]*b_dims[3], b_dims[3], 1};
  dim_t perm_ind[4];
  for (int i = 0; i < 4; ++i) perm_ind[perm[i]] = i;
  dim_t pbs[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                  b_stride[perm_ind[2]], b_stride[perm_ind[3]]};
  for (dim_t i0 = 0; i0 < dims[0]; ++i0)
    for (dim_t i1 = 0; i1 < dims[1]; ++i1)
      for (dim_t i2 = 0; i2 < dims[2]; ++i2)
        for (dim_t i3 = 0; i3 < dims[3]; ++i3) {
          b[i0*pbs[0] + i1*pbs[1] + i2*pbs[2] + i3*pbs[3]] =
              a[i0*a_stride[0] + i1*a_stride[1] + i2*a_stride[2] + i3];
        }
}

// ---------------------------------------------------------------------------
// Check helpers
// ---------------------------------------------------------------------------

// For float: use exact equality (inputs are small integers representable in f32).
static bool check_equal_f32(const float* gpu, const float* cpu, dim_t n,
                              const char* label) {
  for (dim_t i = 0; i < n; ++i) {
    if (gpu[i] != cpu[i]) {
      std::printf("    first mismatch at [%lld]: gpu=%.0f cpu=%.0f\n",
                  (long long)i, (double)gpu[i], (double)cpu[i]);
      return false;
    }
  }
  return true;
  (void)label;
}

// ---------------------------------------------------------------------------
// 2D large test — [1024, 512] = 524,288 elements
// ---------------------------------------------------------------------------

static void test_2d_large() {
  std::printf("\n--- 2D large [1024, 512] ---\n");

  const dim_t dims[2] = {1024, 512};
  const dim_t n = dims[0] * dims[1];  // 524,288

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  // Fill with sequential floats — all exactly representable in f32 up to 2^24.
  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)i;

  primitives<Device::MPS>::transpose_2d(d_a, dims, d_b);
  metal::commit_and_wait();

  cpu_transpose_2d(d_a, dims, cpu_b.data());
  report(check_equal_f32(d_b, cpu_b.data(), n, "2d_large"),
         "2d [1024,512] perm=[1,0] vs cpu (f32)");

  metal_free(d_a); metal_free(d_b);
}

// ---------------------------------------------------------------------------
// 3D large tests
// ---------------------------------------------------------------------------

static void test_3d_large() {
  std::printf("\n--- 3D large [32, 128, 256] = 1,048,576 elements ---\n");

  const dim_t dims[3] = {32, 128, 256};
  const dim_t n = dims[0] * dims[1] * dims[2];  // 1,048,576

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 10000);

  // perm [2, 0, 1]: [32,128,256] -> [256,32,128]  (attention-style permutation)
  {
    const dim_t perm[3] = {2, 0, 1};
    primitives<Device::MPS>::transpose_3d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    cpu_transpose_3d(d_a, dims, perm, cpu_b.data());
    report(check_equal_f32(d_b, cpu_b.data(), n, "3d_201"),
           "3d [32,128,256] perm=[2,0,1] vs cpu (f32)");
  }

  // perm [0, 2, 1]: [32,128,256] -> [32,256,128]  (swap last two dims)
  {
    const dim_t perm[3] = {0, 2, 1};
    primitives<Device::MPS>::transpose_3d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    cpu_transpose_3d(d_a, dims, perm, cpu_b.data());
    report(check_equal_f32(d_b, cpu_b.data(), n, "3d_021"),
           "3d [32,128,256] perm=[0,2,1] vs cpu (f32)");
  }

  // perm [1, 0, 2]: [32,128,256] -> [128,32,256]  (swap first two dims)
  {
    const dim_t perm[3] = {1, 0, 2};
    primitives<Device::MPS>::transpose_3d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    cpu_transpose_3d(d_a, dims, perm, cpu_b.data());
    report(check_equal_f32(d_b, cpu_b.data(), n, "3d_102"),
           "3d [32,128,256] perm=[1,0,2] vs cpu (f32)");
  }

  metal_free(d_a); metal_free(d_b);
}

// ---------------------------------------------------------------------------
// 4D large tests
// ---------------------------------------------------------------------------

static void test_4d_large() {
  std::printf("\n--- 4D large [4, 32, 64, 128] = 1,048,576 elements ---\n");

  const dim_t dims[4] = {4, 32, 64, 128};
  const dim_t n = dims[0] * dims[1] * dims[2] * dims[3];  // 1,048,576

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 10000);

  // perm [0,2,1,3]: [4,32,64,128] -> [4,64,32,128]  (MHA heads<->seq swap)
  {
    const dim_t perm[4] = {0, 2, 1, 3};
    primitives<Device::MPS>::transpose_4d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    cpu_transpose_4d(d_a, dims, perm, cpu_b.data());
    report(check_equal_f32(d_b, cpu_b.data(), n, "4d_0213"),
           "4d [4,32,64,128] perm=[0,2,1,3] vs cpu (f32)");
  }

  // perm [3,2,1,0]: full reversal — exercises maximum cross-stride access.
  {
    const dim_t perm[4] = {3, 2, 1, 0};
    primitives<Device::MPS>::transpose_4d(d_a, dims, perm, d_b);
    metal::commit_and_wait();
    cpu_transpose_4d(d_a, dims, perm, cpu_b.data());
    report(check_equal_f32(d_b, cpu_b.data(), n, "4d_3210"),
           "4d [4,32,64,128] perm=[3,2,1,0] vs cpu (f32)");
  }

  metal_free(d_a); metal_free(d_b);
}

// ---------------------------------------------------------------------------
// 4D decode-phase shape — [1, 16, 512, 128] = 1,048,576 elements
// Exercises the "single batch" path where the batch dimension is 1.
// ---------------------------------------------------------------------------

static void test_4d_decode() {
  std::printf("\n--- 4D decode shape [1, 16, 512, 128] = 1,048,576 elements ---\n");

  const dim_t dims[4] = {1, 16, 512, 128};
  const dim_t n = 1 * 16 * 512 * 128;  // 1,048,576

  float* d_a = metal_alloc<float>(n);
  float* d_b = metal_alloc<float>(n);
  std::vector<float> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (float)(i % 10000);

  // perm [0,2,1,3]: [1,16,512,128] -> [1,512,16,128]
  const dim_t perm[4] = {0, 2, 1, 3};
  primitives<Device::MPS>::transpose_4d(d_a, dims, perm, d_b);
  metal::commit_and_wait();
  cpu_transpose_4d(d_a, dims, perm, cpu_b.data());
  report(check_equal_f32(d_b, cpu_b.data(), n, "4d_decode"),
         "4d [1,16,512,128] perm=[0,2,1,3] vs cpu (f32)");

  metal_free(d_a); metal_free(d_b);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== Large Transpose Test (production-scale shapes) ===\n");
  std::printf("Verifies MSL index decomposition arithmetic for gid up to ~1M\n");

  test_2d_large();
  test_3d_large();
  test_4d_large();
  test_4d_decode();

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
