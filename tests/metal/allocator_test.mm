// Standalone tests for M3.1 (MetalAllocator).
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/allocator_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o allocator_test && ./allocator_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "metal/utils.h"

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


int main() {
  std::printf("=== M3.1: MetalAllocator tests ===\n\n");

  Allocator& alloc = get_allocator<Device::MPS>();

  // 1. allocate() returns a non-null pointer.
  float* ptr = nullptr;
  CHECK_NOTHROW("allocate(1024 * sizeof(float)) — no error",
    ptr = static_cast<float*>(alloc.allocate(1024 * sizeof(float)))
  );
  CHECK("allocate() returns non-null", ptr != nullptr);

  // 2. Returned pointer is CPU-writable (shared memory).
  CHECK_NOTHROW("CPU write to allocated buffer — no error",
    ptr[0] = 42.f; ptr[1] = -1.f; ptr[1023] = 7.f
  );

  // 3. CPU reads back the values just written.
  CHECK("CPU read-back [0] == 42.f",   ptr[0]    == 42.f);
  CHECK("CPU read-back [1] == -1.f",   ptr[1]    == -1.f);
  CHECK("CPU read-back [1023] == 7.f", ptr[1023] == 7.f);

  // 4. free() does not throw.
  CHECK_NOTHROW("free() — no error",
    alloc.free(ptr)
  );

  // 5. Buffer pool: second allocate of the same size returns the same
  //    underlying pointer (pool hit — same MTLBuffer contents address).
  float* ptr2 = static_cast<float*>(alloc.allocate(1024 * sizeof(float)));
  CHECK("pool hit: second alloc same size returns same ptr", ptr2 == ptr);
  alloc.free(ptr2);

  // 6. Different sizes get distinct allocations (no cross-size pool bleed).
  float* small = static_cast<float*>(alloc.allocate(16 * sizeof(float)));
  float* large = static_cast<float*>(alloc.allocate(2048 * sizeof(float)));
  CHECK("different sizes: small != large", small != large);
  alloc.free(small);
  alloc.free(large);

  // 7. clear_cache() does not throw.
  CHECK_NOTHROW("clear_cache() — no error",
    alloc.clear_cache()
  );

  // 8. Allocate after clear_cache() still works.
  float* ptr3 = nullptr;
  CHECK_NOTHROW("allocate after clear_cache() — no error",
    ptr3 = static_cast<float*>(alloc.allocate(64 * sizeof(float)))
  );
  CHECK("allocate after clear_cache() non-null", ptr3 != nullptr);
  alloc.free(ptr3);

  // 9. freeing nullptr is a no-op (should not throw).
  CHECK_NOTHROW("free(nullptr) — no error",
    alloc.free(nullptr)
  );

  // 10. freeing an unknown pointer throws.
  float stack_var = 0.f;
  CHECK_THROWS("free(unknown ptr) — throws",
    alloc.free(&stack_var)
  );

  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
