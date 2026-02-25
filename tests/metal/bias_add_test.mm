// tests/metal/bias_add_test.mm
//
// M5.2 review item 4.6 — Metal BiasAdd correctness tests.
//
// BiasAdd::compute<Device::METAL> (src/ops/bias_add_metal.mm) is a thin
// routing layer: it calls primitives<D>::add_batch_broadcast for last-axis
// bias and primitives<D>::add_block_broadcast for other axes.  Both
// primitives are tested here directly, which exercises the same Metal GPU
// code paths that BiasAdd::compute invokes at runtime.
//
// Tests:
//   1.  last-axis bias f32: value[i] + bias[i % bias_size] — single row
//   2.  last-axis bias f32: batched (3 rows × 4 cols, bias size 4)
//   3.  last-axis bias f16: basic correctness
//   4.  mid-axis bias f32: add_block_broadcast [batch=2, ch=3, w=4], bias=ch
//   5.  residual addition f32: broadcast then element-wise add
//
// Note: The activation-fusion path (get_activation_op) and the full
// BiasAdd::operator() dispatch (which requires StorageView + cpu primitives)
// are tested via the CMake build.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/bias_add_test.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
//     src/metal/primitives_norm_gather.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o bias_add_test && ./bias_add_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

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

// ---------------------------------------------------------------------------
// 1. Last-axis bias f32: single row
//    add_batch_broadcast(bias, value, out, bias_size, value_size)
//    out[i] = bias[i % bias_size] + value[i]
// ---------------------------------------------------------------------------

static void test_last_axis_single_row() {
  std::printf("\n--- last-axis bias f32 (single row) ---\n");

  // value = [10, 20, 30, 40, 50, 60, 70, 80]
  // bias  = [1, 2, 3, 4]
  // expected = [11, 22, 33, 44, 51, 62, 73, 84]
  const int bias_size = 4, value_size = 8;
  float* bias = metal_alloc<float>(bias_size);
  float* val  = metal_alloc<float>(value_size);
  float* out  = metal_alloc<float>(value_size);
  for (int i = 0; i < bias_size;  ++i) bias[i] = float(i + 1);
  for (int i = 0; i < value_size; ++i) val[i]  = float((i + 1) * 10);

  primitives<Device::METAL>::add_batch_broadcast(bias, val, out, bias_size, value_size);
  metal::commit_and_wait();

  // out[i] = val[i] + bias[i % 4]
  bool ok = true;
  for (int i = 0; i < value_size; ++i) {
    float want = val[i] + bias[i % bias_size];
    if (std::abs(out[i] - want) > 1e-6f) { ok = false; break; }
  }
  CHECK("last-axis bias f32: out[i] = val[i] + bias[i % B]", ok);
  // Spot-check
  CHECK_NEAR("last-axis bias f32: out[0]=11", out[0], 11.f, 0.f);
  CHECK_NEAR("last-axis bias f32: out[4]=51", out[4], 51.f, 0.f);

  metal_free(bias); metal_free(val); metal_free(out);
}

// ---------------------------------------------------------------------------
// 2. Last-axis bias f32: batched (3 rows × 4 cols, bias_size=4)
// ---------------------------------------------------------------------------

static void test_last_axis_batched() {
  std::printf("\n--- last-axis bias f32 (batched) ---\n");

  // value shape [3, 4], bias shape [4]
  // value[r][c] = r*100 + c*10,  bias[c] = c+1
  // expected[r][c] = r*100 + c*10 + (c+1)
  const int rows = 3, cols = 4;
  const int bias_size = cols, value_size = rows * cols;
  float* bias = metal_alloc<float>(bias_size);
  float* val  = metal_alloc<float>(value_size);
  float* out  = metal_alloc<float>(value_size);
  for (int c = 0; c < cols; ++c) bias[c] = float(c + 1);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      val[r*cols+c] = float(r*100 + c*10);

  primitives<Device::METAL>::add_batch_broadcast(bias, val, out, bias_size, value_size);
  metal::commit_and_wait();

  bool ok = true;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      float want = val[r*cols+c] + bias[c];
      if (std::abs(out[r*cols+c] - want) > 1e-6f) { ok = false; break; }
    }
  CHECK("last-axis bias f32 batched: each element correct", ok);

  metal_free(bias); metal_free(val); metal_free(out);
}

// ---------------------------------------------------------------------------
// 3. Last-axis bias f16
// ---------------------------------------------------------------------------

static void test_last_axis_f16() {
  std::printf("\n--- last-axis bias f16 ---\n");

  // Small integers exact in f16
  const int bias_size = 4, value_size = 8;
  ct2_f16* bias = metal_alloc<ct2_f16>(bias_size);
  ct2_f16* val  = metal_alloc<ct2_f16>(value_size);
  ct2_f16* out  = metal_alloc<ct2_f16>(value_size);
  for (int i = 0; i < bias_size;  ++i) bias[i] = ct2_f16(float(i + 1));
  for (int i = 0; i < value_size; ++i) val[i]  = ct2_f16(float((i % 4) * 2));

  primitives<Device::METAL>::add_batch_broadcast(bias, val, out, bias_size, value_size);
  metal::commit_and_wait();

  bool ok = true;
  for (int i = 0; i < value_size; ++i) {
    float want = (float)val[i] + (float)bias[i % bias_size];
    if (std::abs((float)out[i] - want) > 1e-3f) { ok = false; break; }
  }
  CHECK("last-axis bias f16: out[i] ≈ val[i] + bias[i % B]", ok);

  metal_free(bias); metal_free(val); metal_free(out);
}

// ---------------------------------------------------------------------------
// 4. Mid-axis bias f32: add_block_broadcast
//    Tensor shape [2, 3, 4]: batch=2, channels=3, width=4.
//    Bias shape [3] (channel bias).  width=4.
//    add_block_broadcast(bias, value, out, width=4, bias_size=3, value_size=24)
//    out[b*12 + c*4 + w] = val[...] + bias[c]
// ---------------------------------------------------------------------------

static void test_mid_axis_bias() {
  std::printf("\n--- mid-axis bias f32 (block broadcast) ---\n");

  const int batch = 2, channels = 3, width = 4;
  const int bias_size  = channels;
  const int value_size = batch * channels * width;  // 24
  float* bias = metal_alloc<float>(bias_size);
  float* val  = metal_alloc<float>(value_size);
  float* out  = metal_alloc<float>(value_size);
  for (int c = 0; c < channels; ++c) bias[c] = float(c + 1) * 10.f;  // 10, 20, 30
  for (int i = 0; i < value_size; ++i) val[i] = float(i);

  // add_block_broadcast(bias, val, out, width, bias_size, value_size)
  primitives<Device::METAL>::add_block_broadcast(bias, val, out, width, bias_size, value_size);
  metal::commit_and_wait();

  // Expected: out[b*ch*w + c*w + w_idx] = val[...] + bias[c]
  bool ok = true;
  for (int b = 0; b < batch; ++b)
    for (int c = 0; c < channels; ++c)
      for (int w = 0; w < width; ++w) {
        int idx = b * channels * width + c * width + w;
        float want = val[idx] + bias[c];
        if (std::abs(out[idx] - want) > 1e-6f) { ok = false; break; }
      }
  CHECK("mid-axis bias f32: out[b,c,w] = val[b,c,w] + bias[c]", ok);
  // Spot-check channel 1 (bias=20)
  // Slot [0,1,0]: idx=4, val=4, bias=20 → 24
  CHECK_NEAR("mid-axis bias f32: out[0,1,0]=24", out[4], 24.f, 0.f);
  // Slot [1,2,3]: idx=1*12+2*4+3=23, val=23, bias=30 → 53
  CHECK_NEAR("mid-axis bias f32: out[1,2,3]=53", out[23], 53.f, 0.f);

  metal_free(bias); metal_free(val); metal_free(out);
}

// ---------------------------------------------------------------------------
// 5. Residual addition f32
//    BiasAdd with residual calls: output = broadcast(bias,value) then output += residual
//    We test: broadcast + add (the two primitives BiasAdd uses).
// ---------------------------------------------------------------------------

static void test_residual_add() {
  std::printf("\n--- residual f32 (broadcast + add) ---\n");

  // value = [1,2,3,4,5,6,7,8],  bias = [10,20,30,40]
  // residual = [100,200,300,400,500,600,700,800]
  // expected = value + bias_broadcast + residual
  const int bias_size = 4, value_size = 8;
  float* bias = metal_alloc<float>(bias_size);
  float* val  = metal_alloc<float>(value_size);
  float* res  = metal_alloc<float>(value_size);
  float* out  = metal_alloc<float>(value_size);
  for (int i = 0; i < bias_size;  ++i) bias[i] = float((i + 1) * 10);
  for (int i = 0; i < value_size; ++i) { val[i] = float(i + 1); res[i] = float((i+1)*100); }

  // Step 1: broadcast
  primitives<Device::METAL>::add_batch_broadcast(bias, val, out, bias_size, value_size);
  // Step 2: add residual (in-place)
  primitives<Device::METAL>::add(out, res, out, value_size);
  metal::commit_and_wait();

  bool ok = true;
  for (int i = 0; i < value_size; ++i) {
    float want = val[i] + bias[i % bias_size] + res[i];
    if (std::abs(out[i] - want) > 1e-5f) { ok = false; break; }
  }
  CHECK("residual f32: out[i] = val[i] + bias[i%B] + residual[i]", ok);
  // Spot-check: i=0: 1 + 10 + 100 = 111
  CHECK_NEAR("residual f32: out[0]=111", out[0], 111.f, 0.f);
  // i=5: 6 + 20 + 600 = 626  (5 % 4 = 1, bias[1]=20)
  CHECK_NEAR("residual f32: out[5]=626", out[5], 626.f, 0.f);

  metal_free(bias); metal_free(val); metal_free(res); metal_free(out);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M5.2 BiasAdd Metal tests ===\n");

  test_last_axis_single_row();
  test_last_axis_batched();
  test_last_axis_f16();
  test_mid_axis_bias();
  test_residual_add();

  std::printf("\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed > 0 ? 1 : 0;
}
