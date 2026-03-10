// Tests for primitives<Device::MPS>::convert<U,V>
//
// Primary concern (Fix 1.1 from metal-primitives-review.md):
//   convert() performs a CPU std::copy from a Metal (shared-memory) buffer.
//   If the GPU has written to that buffer since the last commit, the CPU would
//   read stale data unless a flush is performed first.
//
//   The fix adds   metal::commit_and_wait()   at the top of convert(), matching
//   the same guard already used by at(), logsumexp(), and prepare_length_mask().
//
// Test strategy:
//   1. Allocate a Metal float buffer d_src[N] = {1, 1, ..., 1}
//   2. Run a GPU kernel (add_scalar) that sets d_src[i] = 1 + 5 = 6  (without
//      an explicit sync, so the write is still pending in the command buffer).
//   3. Immediately call convert<float, float16_t>(d_src, d_dst, N) — no
//      manual gpu_sync() in between.
//   4. Verify d_dst[i] == 6.0 (post-GPU value), not 1.0 (stale pre-GPU value).
//      Without the fix, this would return stale 1.0 on fast GPUs where
//      commit_and_wait() was never called before std::copy.
//
// Additional correctness tests (type-pair coverage):
//   float  → float16
//   float  → bfloat16
//   float16 → float
//   bfloat16 → float
//   float16 → bfloat16
//   bfloat16 → float16
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/convert_test.mm \
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
//     -o convert_test && ./convert_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

// Alias to avoid the ::float16_t / ctranslate2::float16_t conflict from
// arm_vector_types.h (pulled in by Metal.h).  Must be declared BEFORE
// using namespace ctranslate2.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;

using namespace ctranslate2;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(label, expr)                                          \
  do {                                                              \
    if (expr) { std::printf("  PASS  %s\n", label); ++g_passed; }  \
    else       { std::printf("  FAIL  %s\n", label); ++g_failed; } \
  } while (0)

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// Fix 1.1 regression test
//
// GPU writes d_src, then convert() is called with NO explicit sync between
// the write and the convert.  convert() must flush before reading.
// ---------------------------------------------------------------------------

static void test_convert_flushes_gpu_writes() {
  std::printf("\n--- convert: flushes pending GPU writes (Fix 1.1) ---\n");

  const dim_t N = 64;
  const float initial = 1.0f;
  const float addend  = 5.0f;
  const float expected = initial + addend;  // 6.0

  float*   d_src = metal_alloc<float>(N);
  ct2_f16* d_dst = metal_alloc<ct2_f16>(N);

  // Initialise source on CPU.
  for (dim_t i = 0; i < N; ++i) d_src[i] = initial;

  // GPU: encode add_scalar (d_src[i] += addend) — intentionally NOT synced.
  primitives<Device::MPS>::add(addend, d_src, d_src, N);
  // No gpu_sync() / commit_and_wait() here — write is still pending.

  // convert must internally commit_and_wait() before reading d_src.
  primitives<Device::MPS>::convert(d_src, d_dst, N);

  // Verify: all elements should be expected (6.0), NOT stale initial (1.0).
  bool ok = true;
  for (dim_t i = 0; i < N; ++i) {
    float got = static_cast<float>(d_dst[i]);
    if (std::fabs(got - expected) > 0.01f) {
      std::printf("    index %lld: got %.4f, expected %.4f\n", (long long)i, got, expected);
      ok = false;
      break;
    }
  }
  CHECK("convert flushes GPU writes (f32→f16)", ok);

  metal_free(d_src);
  metal_free(d_dst);
}

// ---------------------------------------------------------------------------
// Type-pair correctness tests (CPU-initiated, no pending GPU writes)
// ---------------------------------------------------------------------------

template <typename U, typename V>
static void test_pair(const char* label) {
  const dim_t N = 32;

  U* src = metal_alloc<U>(N);
  V* dst = metal_alloc<V>(N);

  // Fill src with small integers that are exactly representable in all types.
  for (dim_t i = 0; i < N; ++i) src[i] = U(static_cast<int>(i) + 1);

  // No pending GPU ops — convert should just std::copy after a no-op flush.
  primitives<Device::MPS>::convert(src, dst, N);

  bool ok = true;
  for (dim_t i = 0; i < N; ++i) {
    float expected = static_cast<float>(src[i]);
    float got      = static_cast<float>(dst[i]);
    if (std::fabs(got - expected) > 0.1f) { ok = false; break; }
  }
  CHECK(label, ok);

  metal_free(src);
  metal_free(dst);
}

// ---------------------------------------------------------------------------
// Zero-size: no crash, no writes
// ---------------------------------------------------------------------------

static void test_zero_size() {
  std::printf("\n--- convert: zero-size no-crash ---\n");
  float*   d_src = metal_alloc<float>(1);
  ct2_f16* d_dst = metal_alloc<ct2_f16>(1);
  bool ok = true;
  try {
    primitives<Device::MPS>::convert(d_src, d_dst, 0);
  } catch (...) { ok = false; }
  CHECK("zero-size no-crash", ok);
  metal_free(d_src);
  metal_free(d_dst);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== Metal convert<U,V> tests ===\n");

  // Fix 1.1 regression: must come first to test the flush path.
  test_convert_flushes_gpu_writes();

  // Type-pair correctness.
  std::printf("\n--- convert: type-pair correctness ---\n");
  test_pair<float,    ct2_f16>("f32 → f16");
  test_pair<float,    ct2_bf16>("f32 → bf16");
  test_pair<ct2_f16,  float>  ("f16 → f32");
  test_pair<ct2_bf16, float>  ("bf16 → f32");
  test_pair<ct2_f16,  ct2_bf16>("f16 → bf16");
  test_pair<ct2_bf16, ct2_f16> ("bf16 → f16");

  // Zero-size.
  test_zero_size();

  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
