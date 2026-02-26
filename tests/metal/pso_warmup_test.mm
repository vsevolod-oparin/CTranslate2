// Tests for 4.2 — PSO warmup: verify all eight MSL libraries compile cleanly.
//
// Each of the eight kernel groups (split across primitives_*.mm files) has its
// own MSL source string that is compiled lazily on first use via
// `newLibraryWithSource:`.  Syntax errors in any embedded string surface only
// at runtime (during the first transformer layer forward pass), not at C++
// compile time.
//
// This test triggers one kernel from every library group and asserts that no
// std::exception is thrown (which is what compile_library_once and make_pso
// propagate on failure).
//
// Library groups under test:
//   1. elementwise   — add / sub / mul (arithmetic ops)
//   2. activation    — relu / gelu / tanh / erf-based (transcendentals)
//   3. broadcast     — add_batch/depth/block_broadcast, mul_batch_broadcast
//   4. beam_search   — penalize_previous_tokens
//   5. transpose     — transpose_2d/3d/4d
//   6. reduction     — reduce_sum / reduce_max / reduce_amax / max_element
//   7. normalization — layer_norm / rms_norm / softmax
//   8. gather        — gather
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/pso_warmup_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives_memory.mm \
//     src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm \
//     src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm \
//     src/metal/primitives_beam_search.mm \
//     src/metal/ops_norm_gather.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o pso_warmup_test && ./pso_warmup_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdint>
#include <exception>
#include <string>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/ops_metal.h"
#include "metal/utils.h"

// Resolve ::float16_t / ctranslate2::float16_t conflict from arm_vector_types.h.
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
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::METAL>().free(p); }

// ---------------------------------------------------------------------------
// Helper: run a callable; return true if it completes without exception.
// ---------------------------------------------------------------------------

template <typename F>
static bool no_exception(F&& f) {
  try {
    f();
    return true;
  } catch (const std::exception& e) {
    std::printf("    exception: %s\n", e.what());
    return false;
  } catch (...) {
    std::printf("    unknown exception\n");
    return false;
  }
}

// ---------------------------------------------------------------------------
// 1. Elementwise library — triggered by add(vec, vec, out, N)
// ---------------------------------------------------------------------------

static void test_elementwise_warmup() {
  std::printf("\n--- warmup: elementwise library ---\n");

  float* a = metal_alloc<float>(4);
  float* b = metal_alloc<float>(4);
  float* c = metal_alloc<float>(4);
  for (int i = 0; i < 4; ++i) { a[i] = 1.f; b[i] = 2.f; }

  bool ok = no_exception([&] {
    primitives<Device::METAL>::add(a, b, c, 4);
    metal::commit_and_wait();
  });
  CHECK("elementwise library compiles (add f32)", ok);

  // Also test f16 variant — same library, different kernel name.
  ct2_f16* x16 = metal_alloc<ct2_f16>(4);
  ct2_f16* y16 = metal_alloc<ct2_f16>(4);
  ct2_f16* z16 = metal_alloc<ct2_f16>(4);
  for (int i = 0; i < 4; ++i) { x16[i] = ct2_f16(1.f); y16[i] = ct2_f16(2.f); }

  ok = no_exception([&] {
    primitives<Device::METAL>::add(x16, y16, z16, 4);
    metal::commit_and_wait();
  });
  CHECK("elementwise library compiles (add f16)", ok);

  metal_free(a); metal_free(b); metal_free(c);
  metal_free(x16); metal_free(y16); metal_free(z16);
}

// ---------------------------------------------------------------------------
// 2. Activation library — triggered by relu(x, y, N)
// ---------------------------------------------------------------------------

static void test_activation_warmup() {
  std::printf("\n--- warmup: activation library ---\n");

  float* x = metal_alloc<float>(4);
  float* y = metal_alloc<float>(4);
  for (int i = 0; i < 4; ++i) x[i] = float(i) - 1.5f;  // mix of neg and pos

  bool ok = no_exception([&] {
    primitives<Device::METAL>::relu(x, y, 4);
    metal::commit_and_wait();
  });
  CHECK("activation library compiles (relu f32)", ok);

  // GELU exercises the ct2_erf polynomial path.
  ok = no_exception([&] {
    primitives<Device::METAL>::gelu(x, y, 4);
    metal::commit_and_wait();
  });
  CHECK("activation library compiles (gelu f32, ct2_erf path)", ok);

  ct2_bf16* xbf = metal_alloc<ct2_bf16>(4);
  ct2_bf16* ybf = metal_alloc<ct2_bf16>(4);
  for (int i = 0; i < 4; ++i) xbf[i] = ct2_bf16(float(i) - 1.5f);

  ok = no_exception([&] {
    primitives<Device::METAL>::relu(xbf, ybf, 4);
    metal::commit_and_wait();
  });
  CHECK("activation library compiles (relu bf16)", ok);

  metal_free(x); metal_free(y);
  metal_free(xbf); metal_free(ybf);
}

// ---------------------------------------------------------------------------
// 3. Broadcast library — triggered by add_batch_broadcast
// ---------------------------------------------------------------------------

static void test_broadcast_warmup() {
  std::printf("\n--- warmup: broadcast library ---\n");

  // a_size=2, b_size=4: a is broadcast over 4-element b.
  float* a = metal_alloc<float>(2);
  float* b = metal_alloc<float>(4);
  float* c = metal_alloc<float>(4);
  a[0] = 1.f; a[1] = 2.f;
  for (int i = 0; i < 4; ++i) b[i] = float(i);

  bool ok = no_exception([&] {
    primitives<Device::METAL>::add_batch_broadcast(a, b, c, 2, 4);
    metal::commit_and_wait();
  });
  CHECK("broadcast library compiles (add_batch_broadcast f32)", ok);

  ct2_f16* af = metal_alloc<ct2_f16>(2);
  ct2_f16* bf = metal_alloc<ct2_f16>(4);
  ct2_f16* cf = metal_alloc<ct2_f16>(4);
  af[0] = ct2_f16(1.f); af[1] = ct2_f16(2.f);
  for (int i = 0; i < 4; ++i) bf[i] = ct2_f16(float(i));

  ok = no_exception([&] {
    primitives<Device::METAL>::add_batch_broadcast(af, bf, cf, 2, 4);
    metal::commit_and_wait();
  });
  CHECK("broadcast library compiles (add_batch_broadcast f16)", ok);

  metal_free(a); metal_free(b); metal_free(c);
  metal_free(af); metal_free(bf); metal_free(cf);
}

// ---------------------------------------------------------------------------
// 4. Beam search library — triggered by penalize_previous_tokens
// ---------------------------------------------------------------------------

static void test_beam_search_warmup() {
  std::printf("\n--- warmup: beam_search library ---\n");

  const dim_t vocab = 8;
  float* scores   = metal_alloc<float>(vocab);
  float* prev_scr = metal_alloc<float>(vocab);
  int32_t* prev_ids = metal_alloc<int32_t>(1);
  for (dim_t i = 0; i < vocab; ++i) { scores[i] = 0.f; prev_scr[i] = 0.f; }
  prev_ids[0] = 0;

  bool ok = no_exception([&] {
    primitives<Device::METAL>::penalize_previous_tokens(
        scores, prev_scr, prev_ids,
        /*penalty=*/0.9f,
        /*batch_size=*/1, /*length=*/1, /*vocabulary_size=*/vocab);
    metal::commit_and_wait();
  });
  CHECK("beam_search library compiles (penalize_previous_tokens f32)", ok);

  metal_free(scores);
  metal_free(prev_scr);
  metal_free(prev_ids);
}

// ---------------------------------------------------------------------------
// 5. Transpose library — triggered by transpose_2d
// ---------------------------------------------------------------------------

static void test_transpose_warmup() {
  std::printf("\n--- warmup: transpose library ---\n");

  // Transpose a 2×4 matrix.
  const dim_t dims[2] = {2, 4};
  float* a = metal_alloc<float>(8);
  float* b = metal_alloc<float>(8);
  for (int i = 0; i < 8; ++i) a[i] = float(i);

  bool ok = no_exception([&] {
    primitives<Device::METAL>::transpose_2d(a, dims, b);
    metal::commit_and_wait();
  });
  CHECK("transpose library compiles (transpose_2d f32)", ok);

  ct2_f16* af = metal_alloc<ct2_f16>(8);
  ct2_f16* bf = metal_alloc<ct2_f16>(8);
  for (int i = 0; i < 8; ++i) af[i] = ct2_f16(float(i));

  ok = no_exception([&] {
    primitives<Device::METAL>::transpose_2d(af, dims, bf);
    metal::commit_and_wait();
  });
  CHECK("transpose library compiles (transpose_2d f16)", ok);

  metal_free(a); metal_free(b);
  metal_free(af); metal_free(bf);
}

// ---------------------------------------------------------------------------
// 6. Reduction library — triggered by sum(x, N)
// ---------------------------------------------------------------------------

static void test_reduction_warmup() {
  std::printf("\n--- warmup: reduction library ---\n");

  const dim_t N = 16;
  float* x = metal_alloc<float>(N);
  for (dim_t i = 0; i < N; ++i) x[i] = 1.f;

  bool ok = no_exception([&] {
    float result = primitives<Device::METAL>::sum(x, N);
    (void)result;  // value correctness is not the focus here
  });
  CHECK("reduction library compiles (sum f32)", ok);

  // max_element exercises a distinct kernel with two output buffers.
  ok = no_exception([&] {
    primitives<Device::METAL>::max_element(x, N);
  });
  CHECK("reduction library compiles (max_element f32)", ok);

  metal_free(x);
}

// ---------------------------------------------------------------------------
// 7. Normalization library — layer_norm / rms_norm / softmax
// ---------------------------------------------------------------------------

static void test_normalization_warmup() {
  std::printf("\n--- warmup: normalization library ---\n");

  const int N = 8;
  float*   xf  = metal_alloc<float>(N);
  float*   gf  = metal_alloc<float>(N);
  float*   yf  = metal_alloc<float>(N);
  ct2_f16* xh  = metal_alloc<ct2_f16>(N);
  ct2_f16* yh  = metal_alloc<ct2_f16>(N);
  ct2_bf16* xb = metal_alloc<ct2_bf16>(N);
  ct2_bf16* gb = metal_alloc<ct2_bf16>(N);
  ct2_bf16* yb = metal_alloc<ct2_bf16>(N);
  for (int i = 0; i < N; ++i) {
    xf[i] = float(i + 1);  gf[i] = 1.f;
    xh[i] = ct2_f16(float(i + 1));
    xb[i] = ct2_bf16(float(i + 1)); gb[i] = ct2_bf16(1.f);
  }

  bool ok = no_exception([&] {
    metal::layer_norm_metal<float>(xf, nullptr, nullptr, yf, 1, N, 1e-5f);
    metal::commit_and_wait();
  });
  CHECK("normalization library compiles (layer_norm float)", ok);

  ok = no_exception([&] {
    metal::softmax_metal<ct2_f16>(xh, nullptr, yh, 1, N, false);
    metal::commit_and_wait();
  });
  CHECK("normalization library compiles (softmax half)", ok);

  ok = no_exception([&] {
    metal::rms_norm_metal<ct2_bf16>(xb, gb, yb, 1, N, 1e-6f);
    metal::commit_and_wait();
  });
  CHECK("normalization library compiles (rms_norm bfloat)", ok);

  metal_free(xf); metal_free(gf); metal_free(yf);
  metal_free(xh); metal_free(yh);
  metal_free(xb); metal_free(gb); metal_free(yb);
}

// ---------------------------------------------------------------------------
// 8. Gather library — gather kernel for float and int32
// ---------------------------------------------------------------------------

static void test_gather_warmup() {
  std::printf("\n--- warmup: gather library ---\n");

  float*   src_f = metal_alloc<float>(4);
  float*   dst_f = metal_alloc<float>(2);
  int32_t* src_i = metal_alloc<int32_t>(4);
  int32_t* dst_i = metal_alloc<int32_t>(2);
  int32_t* idx   = metal_alloc<int32_t>(2);
  for (int i = 0; i < 4; ++i) { src_f[i] = float(i); src_i[i] = i * 10; }
  idx[0] = 0; idx[1] = 2;

  bool ok = no_exception([&] {
    metal::gather_metal<float>(src_f, dst_f, idx, 1, 4, 2, 2);
    metal::commit_and_wait();
  });
  CHECK("gather library compiles (gather float)", ok);

  ok = no_exception([&] {
    metal::gather_metal<int32_t>(src_i, dst_i, idx, 1, 4, 2, 2);
    metal::commit_and_wait();
  });
  CHECK("gather library compiles (gather int32)", ok);

  metal_free(src_f); metal_free(dst_f);
  metal_free(src_i); metal_free(dst_i);
  metal_free(idx);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== PSO warmup: all eight MSL libraries ===\n");

  test_elementwise_warmup();
  test_activation_warmup();
  test_broadcast_warmup();
  test_beam_search_warmup();
  test_transpose_warmup();
  test_reduction_warmup();
  test_normalization_warmup();
  test_gather_warmup();

  std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
