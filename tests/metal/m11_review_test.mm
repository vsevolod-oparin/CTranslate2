// M11 Code Review tests: buffer_for_ptr, protect_buffer, indexed_fill f16,
// MPS GEMM cache.
//
// Run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src -DCT2_WITH_METAL \
//     tests/metal/m11_review_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -framework Accelerate \
//     -o m11_review_test && ./m11_review_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "ctranslate2/primitives.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

static int passed = 0;
static int failed = 0;

#define CHECK(label, expr)                              \
  do {                                                  \
    if (expr) {                                         \
      std::printf("  PASS  %s\n", label);               \
      ++passed;                                         \
    } else {                                            \
      std::printf("  FAIL  %s\n", label);               \
      ++failed;                                         \
    }                                                   \
  } while (0)

#define CHECK_NOTHROW(label, ...)                       \
  do {                                                  \
    bool ok = true;                                     \
    try { __VA_ARGS__; }                                \
    catch (const std::exception& e) {                   \
      std::printf("  FAIL  %s — threw: %s\n",           \
                  label, e.what());                     \
      ok = false;                                       \
    }                                                   \
    if (ok) { std::printf("  PASS  %s\n", label); ++passed; } \
    else    { ++failed; }                               \
  } while (0)

#define CHECK_THROWS(label, ...)                        \
  do {                                                  \
    bool threw = false;                                 \
    try { __VA_ARGS__; } catch (...) { threw = true; }  \
    CHECK(label, threw);                                \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t count) {
  return static_cast<T*>(
      get_allocator<Device::METAL>().allocate(count * sizeof(T)));
}

template <typename T>
static void metal_free(T* ptr) {
  get_allocator<Device::METAL>().free(ptr);
}


// =========================================================================
// TEST-1: buffer_for_ptr O(log N) correctness
// =========================================================================
static void test_buffer_for_ptr() {
  std::printf("\n--- TEST-1: buffer_for_ptr O(log N) ---\n");
  Allocator& alloc = get_allocator<Device::METAL>();

  // Allocate 3 buffers of different sizes.
  const size_t sz_a = 256 * sizeof(float);
  const size_t sz_b = 1024 * sizeof(float);
  const size_t sz_c = 512 * sizeof(float);
  float* a = static_cast<float*>(alloc.allocate(sz_a));
  float* b = static_cast<float*>(alloc.allocate(sz_b));
  float* c = static_cast<float*>(alloc.allocate(sz_c));

  // 1. Base pointer lookup.
  NSUInteger off = 999;
  id<MTLBuffer> buf_a = metal_buffer_for_ptr(a, &off);
  CHECK("buffer_for_ptr(a base): non-nil", buf_a != nil);
  CHECK("buffer_for_ptr(a base): offset=0", off == 0);

  // 2. Sub-pointer lookup (middle of buffer).
  off = 999;
  id<MTLBuffer> buf_b = metal_buffer_for_ptr(b + 100, &off);
  CHECK("buffer_for_ptr(b+100): non-nil", buf_b != nil);
  CHECK("buffer_for_ptr(b+100): offset=400", off == 100 * sizeof(float));

  // 3. Last valid byte.
  off = 999;
  const uint8_t* last_byte = reinterpret_cast<const uint8_t*>(c) + sz_c - 1;
  id<MTLBuffer> buf_c = metal_buffer_for_ptr(last_byte, &off);
  CHECK("buffer_for_ptr(c last byte): non-nil", buf_c != nil);
  CHECK("buffer_for_ptr(c last byte): offset=sz-1", off == sz_c - 1);

  // 4. offset_out=nullptr doesn't crash.
  CHECK_NOTHROW("buffer_for_ptr(a, nullptr) — no crash",
    metal_buffer_for_ptr(a, nullptr)
  );

  // 5. Pointer outside any allocation throws.
  float stack_var = 0.f;
  CHECK_THROWS("buffer_for_ptr(stack ptr) — throws",
    metal_buffer_for_ptr(&stack_var, &off)
  );

  // 6. After free, pointer no longer found.
  alloc.free(b);
  CHECK_THROWS("buffer_for_ptr(freed ptr) — throws",
    metal_buffer_for_ptr(b + 100, &off)
  );

  // 7. Other buffers still found after one is freed.
  CHECK_NOTHROW("buffer_for_ptr(a) after b freed — ok",
    metal_buffer_for_ptr(a, &off)
  );
  CHECK_NOTHROW("buffer_for_ptr(c) after b freed — ok",
    metal_buffer_for_ptr(c, &off)
  );

  alloc.free(a);
  alloc.free(c);
  alloc.clear_cache();
}


// =========================================================================
// TEST-2: protect_buffer + flush_pending_frees lifecycle
// =========================================================================
static void test_protect_buffer() {
  std::printf("\n--- TEST-2: protect_buffer lifecycle ---\n");
  Allocator& alloc = get_allocator<Device::METAL>();

  // Allocate a buffer and protect it.
  const size_t sz = 128 * sizeof(float);
  float* p = static_cast<float*>(alloc.allocate(sz));
  float* original_ptr = p;

  // Write some data so we can verify later.
  for (int i = 0; i < 128; ++i) p[i] = static_cast<float>(i);

  // Protect via sub-pointer (tests O(log N) protect_buffer path).
  metal::protect_buffer(p + 64);

  // Free the buffer — should go to pending_free, NOT pool.
  alloc.free(p);

  // Allocate same size — should NOT return the protected buffer.
  float* q = static_cast<float*>(alloc.allocate(sz));
  CHECK("protected buffer NOT reused immediately", q != original_ptr);
  alloc.free(q);

  // Flush pending frees — buffer should now return to pool.
  metal::flush_pending_frees();

  // Now allocate same size — should get the original buffer back.
  float* r = static_cast<float*>(alloc.allocate(sz));
  CHECK("after flush: buffer returned to pool", r == original_ptr);
  alloc.free(r);

  // Test protect_buffer_by_base (O(log N) with exact base).
  float* s = static_cast<float*>(alloc.allocate(sz));
  metal::protect_buffer_by_base(s);
  alloc.free(s);
  float* t = static_cast<float*>(alloc.allocate(sz));
  CHECK("protect_by_base: buffer NOT reused immediately", t != s);
  alloc.free(t);
  metal::flush_pending_frees();

  alloc.clear_cache();
}


// =========================================================================
// TEST-3: indexed_fill f16 encode-only correctness
// =========================================================================
static void test_indexed_fill_f16() {
  std::printf("\n--- TEST-3: indexed_fill float16 encode-only ---\n");

  const dim_t N = 16;
  auto* x = metal_alloc<ct2_f16>(N);
  auto* indices = metal_alloc<int32_t>(4);

  // Initialize x to zeros.
  for (dim_t i = 0; i < N; ++i) x[i] = ct2_f16(0.0f);

  // Set up scatter indices.
  indices[0] = 2;
  indices[1] = 5;
  indices[2] = 11;
  indices[3] = 15;

  // Call indexed_fill (should be encode-only for f16).
  primitives<Device::METAL>::indexed_fill(x, ct2_f16(7.5f), indices, 4);

  // Must commit to see results.
  metal::commit_and_wait();

  CHECK("f16 indexed_fill: x[2] == 7.5",   std::fabs(float(x[2]) - 7.5f) < 0.01f);
  CHECK("f16 indexed_fill: x[5] == 7.5",   std::fabs(float(x[5]) - 7.5f) < 0.01f);
  CHECK("f16 indexed_fill: x[11] == 7.5",  std::fabs(float(x[11]) - 7.5f) < 0.01f);
  CHECK("f16 indexed_fill: x[15] == 7.5",  std::fabs(float(x[15]) - 7.5f) < 0.01f);
  CHECK("f16 indexed_fill: x[0] untouched", std::fabs(float(x[0])) < 0.01f);
  CHECK("f16 indexed_fill: x[1] untouched", std::fabs(float(x[1])) < 0.01f);
  CHECK("f16 indexed_fill: x[3] untouched", std::fabs(float(x[3])) < 0.01f);

  metal_free(x);
  metal_free(indices);
}

// f16 indexed_fill interleaved with GPU writes (encode-only chain)
static void test_indexed_fill_f16_chain() {
  std::printf("\n--- TEST-3b: indexed_fill f16 after GPU fill ---\n");

  const dim_t N = 32;
  auto* x = metal_alloc<ct2_f16>(N);

  // GPU fill (encode-only) — fill all with 1.0
  primitives<Device::METAL>::fill(x, ct2_f16(1.0f), N);

  // Now scatter -1.0 at specific indices.
  auto* idx = metal_alloc<int32_t>(3);
  idx[0] = 0;
  idx[1] = 10;
  idx[2] = 31;

  primitives<Device::METAL>::indexed_fill(x, ct2_f16(-1.0f), idx, 3);
  metal::commit_and_wait();

  CHECK("f16 chain: x[0] == -1.0",   std::fabs(float(x[0]) + 1.0f) < 0.01f);
  CHECK("f16 chain: x[10] == -1.0",  std::fabs(float(x[10]) + 1.0f) < 0.01f);
  CHECK("f16 chain: x[31] == -1.0",  std::fabs(float(x[31]) + 1.0f) < 0.01f);
  CHECK("f16 chain: x[1] == 1.0",    std::fabs(float(x[1]) - 1.0f) < 0.01f);
  CHECK("f16 chain: x[15] == 1.0",   std::fabs(float(x[15]) - 1.0f) < 0.01f);

  metal_free(x);
  metal_free(idx);
}

// f32 indexed_fill (requires pre-sync)
static void test_indexed_fill_f32() {
  std::printf("\n--- TEST-3c: indexed_fill float32 (with sync) ---\n");

  const dim_t N = 16;
  auto* x = metal_alloc<float>(N);
  auto* indices = metal_alloc<int32_t>(3);

  for (dim_t i = 0; i < N; ++i) x[i] = 0.f;
  indices[0] = 1;
  indices[1] = 7;
  indices[2] = 14;

  primitives<Device::METAL>::indexed_fill(x, 3.14f, indices, 3);
  metal::commit_and_wait();

  CHECK("f32 indexed_fill: x[1] == 3.14",  std::fabs(x[1] - 3.14f) < 0.001f);
  CHECK("f32 indexed_fill: x[7] == 3.14",  std::fabs(x[7] - 3.14f) < 0.001f);
  CHECK("f32 indexed_fill: x[14] == 3.14", std::fabs(x[14] - 3.14f) < 0.001f);
  CHECK("f32 indexed_fill: x[0] untouched", std::fabs(x[0]) < 0.001f);

  metal_free(x);
  metal_free(indices);
}

// Zero-length indexed_fill (early return)
static void test_indexed_fill_zero() {
  std::printf("\n--- TEST-3d: indexed_fill zero-length ---\n");

  auto* x = metal_alloc<float>(4);
  for (dim_t i = 0; i < 4; ++i) x[i] = 42.f;

  CHECK_NOTHROW("indexed_fill(0 indices) — no crash",
    primitives<Device::METAL>::indexed_fill(x, 0.f, nullptr, 0)
  );

  // x should be unchanged.
  CHECK("zero-length: x[0] still 42", std::fabs(x[0] - 42.f) < 0.001f);

  metal_free(x);
}


// bf16 indexed_fill (verify pre-sync fires for bfloat16 too)
static void test_indexed_fill_bf16() {
  std::printf("\n--- TEST-3e: indexed_fill bfloat16 ---\n");

  // Check BF16 GPU support at runtime.
  id<MTLDevice> dev = metal::get_metal_device();
  if (![dev supportsFamily:MTLGPUFamilyApple9]) {
    std::printf("  SKIP  bfloat16 not supported on this GPU\n");
    return;
  }

  const dim_t N = 16;
  auto* x = metal_alloc<ct2_bf16>(N);
  auto* indices = metal_alloc<int32_t>(3);

  for (dim_t i = 0; i < N; ++i) x[i] = ct2_bf16(0.0f);
  indices[0] = 0;
  indices[1] = 8;
  indices[2] = 15;

  primitives<Device::METAL>::indexed_fill(x, ct2_bf16(5.0f), indices, 3);
  metal::commit_and_wait();

  CHECK("bf16 indexed_fill: x[0] == 5.0",  std::fabs(float(x[0]) - 5.0f) < 0.1f);
  CHECK("bf16 indexed_fill: x[8] == 5.0",  std::fabs(float(x[8]) - 5.0f) < 0.1f);
  CHECK("bf16 indexed_fill: x[15] == 5.0", std::fabs(float(x[15]) - 5.0f) < 0.1f);
  CHECK("bf16 indexed_fill: x[1] untouched", std::fabs(float(x[1])) < 0.1f);

  metal_free(x);
  metal_free(indices);
}


// =========================================================================
// TEST-4: MPS GEMM cache correctness
// =========================================================================
static void test_gemm_cache() {
  std::printf("\n--- TEST-4: MPS GEMM cache correctness ---\n");

  // Small GEMM: C = A × B, where A[4×8], B[8×6], C[4×6].
  const dim_t M = 4, N = 6, K = 8;
  auto* a = metal_alloc<float>(M * K);
  auto* b = metal_alloc<float>(K * N);
  auto* c = metal_alloc<float>(M * N);

  // Fill A=1.0, B=1.0 → C should be K*1.0 = 8.0.
  for (dim_t i = 0; i < M * K; ++i) a[i] = 1.0f;
  for (dim_t i = 0; i < K * N; ++i) b[i] = 1.0f;
  for (dim_t i = 0; i < M * N; ++i) c[i] = 0.0f;

  // First GEMM — cache miss.
  primitives<Device::METAL>::gemm<float, float>(false, false, false, false,
      M, N, K, 1.0f, a, K, b, N, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool first_ok = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(c[i] - 8.0f) > 0.01f) { first_ok = false; break; }
  }
  CHECK("GEMM cache miss: C = A×B correct", first_ok);

  // Second GEMM with same shape — cache hit. Use different values.
  for (dim_t i = 0; i < M * K; ++i) a[i] = 2.0f;
  for (dim_t i = 0; i < M * N; ++i) c[i] = 0.0f;

  primitives<Device::METAL>::gemm<float, float>(false, false, false, false,
      M, N, K, 1.0f, a, K, b, N, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool second_ok = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(c[i] - 16.0f) > 0.01f) { second_ok = false; break; }
  }
  CHECK("GEMM cache hit: C = 2A×B correct", second_ok);

  // Third GEMM with alpha=0.5 — different cache key.
  for (dim_t i = 0; i < M * N; ++i) c[i] = 0.0f;

  primitives<Device::METAL>::gemm<float, float>(false, false, false, false,
      M, N, K, 0.5f, a, K, b, N, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool alpha_ok = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(c[i] - 8.0f) > 0.01f) { alpha_ok = false; break; }
  }
  CHECK("GEMM alpha=0.5: C = 0.5*2A×B correct", alpha_ok);

  // Fourth GEMM: transpose_b, B^T[6×8] → C = A × B^T, same result.
  auto* bt = metal_alloc<float>(N * K);
  // Transpose B manually: bt[j][i] = b[i][j]
  for (dim_t i = 0; i < K; ++i)
    for (dim_t j = 0; j < N; ++j)
      bt[j * K + i] = b[i * N + j];
  for (dim_t i = 0; i < M * N; ++i) c[i] = 0.0f;

  primitives<Device::METAL>::gemm<float, float>(false, false, false, true,
      M, N, K, 1.0f, a, K, bt, K, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool trans_ok = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(c[i] - 16.0f) > 0.01f) { trans_ok = false; break; }
  }
  CHECK("GEMM transpose_b: C = A×B^T correct", trans_ok);

  metal_free(a);
  metal_free(b);
  metal_free(c);
  metal_free(bt);
}

// GEMM with float16 — verify cache works for f16 dtype too.
static void test_gemm_cache_f16() {
  std::printf("\n--- TEST-4b: MPS GEMM cache float16 ---\n");

  const dim_t M = 4, N = 8, K = 16;
  auto* a = metal_alloc<ct2_f16>(M * K);
  auto* b = metal_alloc<ct2_f16>(K * N);
  auto* c = metal_alloc<ct2_f16>(M * N);

  for (dim_t i = 0; i < M * K; ++i) a[i] = ct2_f16(1.0f);
  for (dim_t i = 0; i < K * N; ++i) b[i] = ct2_f16(0.5f);
  for (dim_t i = 0; i < M * N; ++i) c[i] = ct2_f16(0.0f);

  // C = A × B = K * 0.5 = 8.0
  primitives<Device::METAL>::gemm<ct2_f16, ct2_f16>(false, false, false, false,
      M, N, K, 1.0f, a, K, b, N, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool ok = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(float(c[i]) - 8.0f) > 0.1f) { ok = false; break; }
  }
  CHECK("f16 GEMM cache: C = A×B correct", ok);

  // Second call — cache hit.
  for (dim_t i = 0; i < M * K; ++i) a[i] = ct2_f16(2.0f);
  for (dim_t i = 0; i < M * N; ++i) c[i] = ct2_f16(0.0f);

  primitives<Device::METAL>::gemm<ct2_f16, ct2_f16>(false, false, false, false,
      M, N, K, 1.0f, a, K, b, N, 0.0f, c, N, nullptr);
  metal::commit_and_wait();

  bool ok2 = true;
  for (dim_t i = 0; i < M * N; ++i) {
    if (std::fabs(float(c[i]) - 16.0f) > 0.1f) { ok2 = false; break; }
  }
  CHECK("f16 GEMM cache hit: C = 2A×B correct", ok2);

  metal_free(a);
  metal_free(b);
  metal_free(c);
}


// =========================================================================
// TEST-5: protect_buffer race simulation
// =========================================================================
// Simulates the M10.1 pattern: encode-only GPU op reads a buffer,
// buffer is freed (protected), new alloc must NOT get that buffer.
static void test_protect_buffer_race() {
  std::printf("\n--- TEST-5: protect_buffer race simulation ---\n");

  Allocator& alloc = get_allocator<Device::METAL>();

  // Allocate source and destination.
  const dim_t N = 64;
  auto* src = metal_alloc<float>(N);
  auto* dst = metal_alloc<float>(N);

  for (dim_t i = 0; i < N; ++i) {
    src[i] = static_cast<float>(i);
    dst[i] = 0.f;
  }

  // Encode a GPU copy (encode-only).
  metal::blit_copy(src, dst, N * sizeof(float));

  // Protect src — simulates "GPU still reading this buffer".
  metal::protect_buffer(src);

  // Free src — should go to pending, NOT pool.
  alloc.free(src);

  // Allocate same size — must NOT get the protected buffer.
  auto* new_buf = metal_alloc<float>(N * sizeof(float));
  // Overwrite new_buf with garbage.
  for (dim_t i = 0; i < N; ++i) new_buf[i] = -999.f;

  // Now commit — the GPU copy reads src (which is in pending, not overwritten).
  metal::commit_and_wait();

  // Verify dst has correct data (from original src, not from garbage new_buf).
  bool ok = true;
  for (dim_t i = 0; i < N; ++i) {
    if (std::fabs(dst[i] - static_cast<float>(i)) > 0.001f) {
      std::printf("    dst[%d] = %f, expected %f\n",
                  (int)i, dst[i], static_cast<float>(i));
      ok = false;
      break;
    }
  }
  CHECK("protect_buffer race: GPU read correct data", ok);

  metal_free(dst);
  metal_free(new_buf);
  alloc.clear_cache();
}


int main() {
  std::printf("=== M11 Code Review Tests ===\n");

  test_buffer_for_ptr();
  test_protect_buffer();
  test_indexed_fill_f16();
  test_indexed_fill_f16_chain();
  test_indexed_fill_f32();
  test_indexed_fill_zero();
  test_indexed_fill_bf16();
  test_gemm_cache();
  test_gemm_cache_f16();
  test_protect_buffer_race();

  std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
