// Standalone tests for M3.2:
//   - cross_device_primitives<CPU,METAL> and <METAL,CPU>  (memcpy over unified memory)
//   - primitives<METAL>::at and ::copy  (real implementations)
//   - storage_view.cc copy_from path is exercised via direct allocator + primitives calls
//     that replicate what StorageView::copy_from does internally; full StorageView
//     integration is verified by the CMake build where cpu/primitives.cc is available.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/storage_view_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o storage_view_test && ./storage_view_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"

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


// ---------------------------------------------------------------------------
// M3.2a — cross_device_primitives<CPU,METAL> and <METAL,CPU>
// ---------------------------------------------------------------------------

static void test_cross_device_primitives() {
  std::printf("\n--- M3.2a: cross_device_primitives (CPU↔Metal) ---\n");

  Allocator& alloc = get_allocator<Device::MPS>();
  const dim_t N = 8;
  float* metal_ptr = static_cast<float*>(alloc.allocate(N * sizeof(float)));
  CHECK("Metal alloc non-null", metal_ptr != nullptr);

  // 1. CPU → Metal
  float src[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  CHECK_NOTHROW("CPU→Metal copy — no error",
    (cross_device_primitives<Device::CPU, Device::MPS>::copy(src, metal_ptr, N))
  );
  bool ok = true;
  for (dim_t i = 0; i < N; ++i) {
    if (metal_ptr[i] != src[i]) { ok = false; break; }
  }
  CHECK("CPU→Metal: values visible via Metal ptr (unified memory)", ok);

  // 2. Metal → CPU
  float dst[8] = {};
  CHECK_NOTHROW("Metal→CPU copy — no error",
    (cross_device_primitives<Device::MPS, Device::CPU>::copy(
        static_cast<const float*>(metal_ptr), dst, N))
  );
  ok = true;
  for (dim_t i = 0; i < N; ++i) {
    if (dst[i] != src[i]) { ok = false; break; }
  }
  CHECK("Metal→CPU: values correct", ok);

  // 3. int32 — verify the template works for non-float types too
  int32_t* metal_i32 = static_cast<int32_t*>(
      alloc.allocate(4 * sizeof(int32_t)));
  int32_t src_i32[4] = {10, 20, 30, 40};
  cross_device_primitives<Device::CPU, Device::MPS>::copy(src_i32, metal_i32, 4);
  int32_t dst_i32[4] = {};
  cross_device_primitives<Device::MPS, Device::CPU>::copy(
      static_cast<const int32_t*>(metal_i32), dst_i32, 4);
  CHECK("int32 round-trip CPU↔Metal", std::memcmp(src_i32, dst_i32, 16) == 0);

  alloc.free(metal_ptr);
  alloc.free(metal_i32);
}


// ---------------------------------------------------------------------------
// M3.2b — primitives<METAL>::at and ::copy  (real implementations)
// ---------------------------------------------------------------------------

static void test_metal_primitives_at_copy() {
  std::printf("\n--- M3.2b: primitives<METAL>::at and ::copy ---\n");

  Allocator& alloc = get_allocator<Device::MPS>();
  const dim_t N = 4;
  float* src_m = static_cast<float*>(alloc.allocate(N * sizeof(float)));
  float* dst_m = static_cast<float*>(alloc.allocate(N * sizeof(float)));

  // Seed src via CPU write (unified memory)
  src_m[0] = 1.f; src_m[1] = 2.f; src_m[2] = 3.f; src_m[3] = 4.f;

  // primitives<METAL>::at — direct CPU read of shared-memory pointer
  CHECK("primitives<METAL>::at(0) == 1.f",
        primitives<Device::MPS>::at(src_m, 0) == 1.f);
  CHECK("primitives<METAL>::at(3) == 4.f",
        primitives<Device::MPS>::at(src_m, 3) == 4.f);

  // primitives<METAL>::copy — memcpy between two Metal (shared) buffers
  CHECK_NOTHROW("primitives<METAL>::copy — no error",
    primitives<Device::MPS>::copy(
        static_cast<const float*>(src_m), dst_m, N)
  );
  bool copy_ok = true;
  for (dim_t i = 0; i < N; ++i) {
    if (dst_m[i] != src_m[i]) { copy_ok = false; break; }
  }
  CHECK("primitives<METAL>::copy: values match", copy_ok);

  alloc.free(src_m);
  alloc.free(dst_m);
}


// ---------------------------------------------------------------------------
// M3.2c — synchronize_stream(METAL) before Metal→CPU read (correctness fence)
// ---------------------------------------------------------------------------

static void test_metal_sync_fence() {
  std::printf("\n--- M3.2c: synchronize_stream fence before Metal→CPU read ---\n");

  // Simulate what storage_view.cc::copy_from does for Metal→CPU:
  //   1. synchronize_stream(METAL)  — commit_and_wait(), flushes GPU writes
  //   2. cross_device_primitives<METAL,CPU>::copy
  Allocator& alloc = get_allocator<Device::MPS>();
  float* metal_ptr = static_cast<float*>(alloc.allocate(4 * sizeof(float)));
  metal_ptr[0] = 9.f; metal_ptr[1] = 8.f; metal_ptr[2] = 7.f; metal_ptr[3] = 6.f;

  CHECK_NOTHROW("synchronize_stream(METAL) + Metal→CPU — no error",
    synchronize_stream(Device::MPS);
    float dst[4] = {};
    cross_device_primitives<Device::MPS, Device::CPU>::copy(
        static_cast<const float*>(metal_ptr), dst, 4);
    CHECK("fence+copy: dst[0] == 9.f", dst[0] == 9.f);
    CHECK("fence+copy: dst[3] == 6.f", dst[3] == 6.f);
  );

  alloc.free(metal_ptr);
}


int main() {
  std::printf("=== M3.2: CPU↔Metal primitives tests ===\n");
  test_cross_device_primitives();
  test_metal_primitives_at_copy();
  test_metal_sync_fence();
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
