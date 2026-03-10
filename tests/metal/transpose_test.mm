// Correctness tests for M4.8 Metal transpose primitives.
//
// Tests: transpose_2d, transpose_3d, transpose_4d for float32, float16,
//        bfloat16, int32 — covering:
//          * vs CPU reference
//          * round-trip (permute then inverse-permute = identity)
//          * the critical perm=[0,2,1,3] (multi-head attention)
//          * zero-size edge cases
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/transpose_test.mm \
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
//     -o transpose_test && ./transpose_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& name) {
  if (ok) { ++g_pass; }
  else {
    ++g_fail;
    std::printf("  FAIL: %s\n", name.c_str());
  }
}

// ---------------------------------------------------------------------------
// Allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// CPU reference implementations (mirror src/cpu/primitives.cc)
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
  // perm_b_stride[k] = b_stride at output position corresponding to input dim k
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

// Compute inverse of a permutation.
static void invert_perm(const dim_t* perm, dim_t rank, dim_t* inv) {
  for (dim_t i = 0; i < rank; ++i) inv[perm[i]] = i;
}

// ---------------------------------------------------------------------------
// Generic compare helper
// ---------------------------------------------------------------------------

template <typename T>
static bool arrays_equal(const T* a, const T* b, dim_t n) {
  for (dim_t i = 0; i < n; ++i)
    if ((float)a[i] != (float)b[i]) return false;
  return true;
}

// ---------------------------------------------------------------------------
// 2D tests
// ---------------------------------------------------------------------------

template <typename T>
static void test_2d(const std::string& tname) {
  // --- vs CPU reference ---
  {
    const dim_t dims[2] = {5, 7};  // 5 rows, 7 cols
    const dim_t n = dims[0] * dims[1];

    T* d_a = metal_alloc<T>(n);
    T* d_b = metal_alloc<T>(n);
    std::vector<T> cpu_b(n);

    for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)i;

    primitives<Device::MPS>::transpose_2d(d_a, dims, d_b);
    metal::commit_and_wait();

    cpu_transpose_2d(d_a, dims, cpu_b.data());

    check(arrays_equal(d_b, cpu_b.data(), n), "2d_vs_cpu<" + tname + ">");
    metal_free(d_a); metal_free(d_b);
  }

  // --- round-trip (transpose twice = identity) ---
  {
    const dim_t dims[2] = {6, 9};
    const dim_t dims_t[2] = {9, 6};  // transposed shape
    const dim_t n = 6 * 9;

    T* d_a = metal_alloc<T>(n);
    T* d_b = metal_alloc<T>(n);
    T* d_c = metal_alloc<T>(n);

    for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)(i + 1);

    primitives<Device::MPS>::transpose_2d(d_a, dims, d_b);
    primitives<Device::MPS>::transpose_2d(d_b, dims_t, d_c);
    metal::commit_and_wait();

    check(arrays_equal(d_a, d_c, n), "2d_roundtrip<" + tname + ">");
    metal_free(d_a); metal_free(d_b); metal_free(d_c);
  }

  // --- zero-size ---
  {
    const dim_t dims[2] = {0, 5};
    primitives<Device::MPS>::transpose_2d((T*)nullptr, dims, (T*)nullptr);
    check(true, "2d_zero<" + tname + ">");
  }
}

// ---------------------------------------------------------------------------
// 3D tests
// ---------------------------------------------------------------------------

template <typename T>
static void test_3d_perm(const dim_t* dims, const dim_t* perm, const std::string& name) {
  dim_t n = dims[0] * dims[1] * dims[2];
  T* d_a = metal_alloc<T>(n);
  T* d_b = metal_alloc<T>(n);
  std::vector<T> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)i;

  primitives<Device::MPS>::transpose_3d(d_a, dims, perm, d_b);
  metal::commit_and_wait();

  cpu_transpose_3d(d_a, dims, perm, cpu_b.data());

  check(arrays_equal(d_b, cpu_b.data(), n), name);
  metal_free(d_a); metal_free(d_b);
}

template <typename T>
static void test_3d_roundtrip(const dim_t* dims, const dim_t* perm, const std::string& name) {
  dim_t n = dims[0] * dims[1] * dims[2];
  dim_t inv_perm[3]; invert_perm(perm, 3, inv_perm);

  // dims of permuted tensor
  const dim_t dims_b[3] = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};

  T* d_a = metal_alloc<T>(n);
  T* d_b = metal_alloc<T>(n);
  T* d_c = metal_alloc<T>(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)(i + 1);

  primitives<Device::MPS>::transpose_3d(d_a, dims,   perm,     d_b);
  primitives<Device::MPS>::transpose_3d(d_b, dims_b, inv_perm, d_c);
  metal::commit_and_wait();

  check(arrays_equal(d_a, d_c, n), name);
  metal_free(d_a); metal_free(d_b); metal_free(d_c);
}

template <typename T>
static void test_3d(const std::string& tname) {
  const dim_t dims[3] = {3, 5, 7};

  const dim_t perm_021[3] = {0, 2, 1};
  const dim_t perm_102[3] = {1, 0, 2};
  const dim_t perm_120[3] = {1, 2, 0};
  const dim_t perm_201[3] = {2, 0, 1};
  const dim_t perm_210[3] = {2, 1, 0};

  test_3d_perm<T>(dims, perm_021, "3d_021_vs_cpu<" + tname + ">");
  test_3d_perm<T>(dims, perm_102, "3d_102_vs_cpu<" + tname + ">");
  test_3d_perm<T>(dims, perm_210, "3d_210_vs_cpu<" + tname + ">");

  test_3d_roundtrip<T>(dims, perm_021, "3d_021_roundtrip<" + tname + ">");
  test_3d_roundtrip<T>(dims, perm_120, "3d_120_roundtrip<" + tname + ">");
  test_3d_roundtrip<T>(dims, perm_201, "3d_201_roundtrip<" + tname + ">");

  // zero-size
  const dim_t dims_z[3] = {0, 5, 7};
  primitives<Device::MPS>::transpose_3d((T*)nullptr, dims_z, perm_021, (T*)nullptr);
  check(true, "3d_zero<" + tname + ">");
}

// ---------------------------------------------------------------------------
// 4D tests
// ---------------------------------------------------------------------------

template <typename T>
static void test_4d_perm(const dim_t* dims, const dim_t* perm, const std::string& name) {
  dim_t n = dims[0] * dims[1] * dims[2] * dims[3];
  T* d_a = metal_alloc<T>(n);
  T* d_b = metal_alloc<T>(n);
  std::vector<T> cpu_b(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)i;

  primitives<Device::MPS>::transpose_4d(d_a, dims, perm, d_b);
  metal::commit_and_wait();

  cpu_transpose_4d(d_a, dims, perm, cpu_b.data());

  check(arrays_equal(d_b, cpu_b.data(), n), name);
  metal_free(d_a); metal_free(d_b);
}

template <typename T>
static void test_4d_roundtrip(const dim_t* dims, const dim_t* perm, const std::string& name) {
  dim_t n = dims[0] * dims[1] * dims[2] * dims[3];
  dim_t inv_perm[4]; invert_perm(perm, 4, inv_perm);
  const dim_t dims_b[4] = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};

  T* d_a = metal_alloc<T>(n);
  T* d_b = metal_alloc<T>(n);
  T* d_c = metal_alloc<T>(n);

  for (dim_t i = 0; i < n; ++i) d_a[i] = (T)(float)(i + 1);

  primitives<Device::MPS>::transpose_4d(d_a, dims,   perm,     d_b);
  primitives<Device::MPS>::transpose_4d(d_b, dims_b, inv_perm, d_c);
  metal::commit_and_wait();

  check(arrays_equal(d_a, d_c, n), name);
  metal_free(d_a); metal_free(d_b); metal_free(d_c);
}

template <typename T>
static void test_4d(const std::string& tname) {
  const dim_t dims[4] = {2, 3, 5, 7};

  // MHA permutation: [0,2,1,3] — the most performance-critical case.
  // Input: [batch, heads, seq, dim] → Output: [batch, seq, heads, dim]
  const dim_t perm_0213[4] = {0, 2, 1, 3};
  // Realistic MHA shape: batch=1, heads=8, seq=16, head_dim=64
  const dim_t mha_dims[4] = {1, 8, 16, 64};

  test_4d_perm<T>(dims,     perm_0213, "4d_0213_vs_cpu<" + tname + ">");
  test_4d_perm<T>(mha_dims, perm_0213, "4d_0213_mha_vs_cpu<" + tname + ">");

  // Arbitrary permutations vs CPU
  const dim_t perm_0132[4] = {0, 1, 3, 2};
  const dim_t perm_1032[4] = {1, 0, 3, 2};
  const dim_t perm_3210[4] = {3, 2, 1, 0};
  test_4d_perm<T>(dims, perm_0132, "4d_0132_vs_cpu<" + tname + ">");
  test_4d_perm<T>(dims, perm_1032, "4d_1032_vs_cpu<" + tname + ">");
  test_4d_perm<T>(dims, perm_3210, "4d_3210_vs_cpu<" + tname + ">");

  // Round-trips
  test_4d_roundtrip<T>(dims, perm_0213, "4d_0213_roundtrip<" + tname + ">");
  test_4d_roundtrip<T>(dims, perm_0132, "4d_0132_roundtrip<" + tname + ">");
  test_4d_roundtrip<T>(dims, perm_3210, "4d_3210_roundtrip<" + tname + ">");

  // zero-size
  const dim_t dims_z[4] = {0, 3, 5, 7};
  primitives<Device::MPS>::transpose_4d((T*)nullptr, dims_z, perm_0213, (T*)nullptr);
  check(true, "4d_zero<" + tname + ">");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.8: Transpose Primitives ===\n\n");

  std::printf("--- float32 ---\n");
  test_2d<float>("float");
  test_3d<float>("float");
  test_4d<float>("float");

  std::printf("--- float16 ---\n");
  test_2d<ct2_f16>("half");
  test_3d<ct2_f16>("half");
  test_4d<ct2_f16>("half");

  std::printf("--- bfloat16 ---\n");
  test_2d<ct2_bf16>("bfloat");
  test_3d<ct2_bf16>("bfloat");
  test_4d<ct2_bf16>("bfloat");

  std::printf("--- int32 ---\n");
  test_2d<int32_t>("int");
  test_3d<int32_t>("int");
  test_4d<int32_t>("int");

  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
