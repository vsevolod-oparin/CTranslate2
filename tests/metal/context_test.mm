// Standalone test for M2.1: Metal context module.
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/context_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o context_test && ./context_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstdio>
#include <stdexcept>

#include "metal/utils.h"

using namespace ctranslate2::metal;

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

#define CHECK_THROWS(label, expr)                       \
  do {                                                  \
    bool threw = false;                                 \
    try { (void)(expr); } catch (...) { threw = true; } \
    CHECK(label, threw);                                \
  } while (0)

int main() {
  std::printf("=== Metal context tests ===\n");

  // 1. Device singleton is non-nil.
  id<MTLDevice> dev = get_metal_device();
  CHECK("get_metal_device() != nil", dev != nil);

  // 2. Calling again returns the same pointer (singleton).
  CHECK("get_metal_device() is singleton", get_metal_device() == dev);

  // 3. Command queue is non-nil.
  id<MTLCommandQueue> queue = get_metal_command_queue();
  CHECK("get_metal_command_queue() != nil", queue != nil);

  // 4. Command queue is per-thread singleton.
  CHECK("get_metal_command_queue() is per-thread singleton",
        get_metal_command_queue() == queue);

  // 5. Command buffer is non-nil.
  id<MTLCommandBuffer> buf = get_current_command_buffer();
  CHECK("get_current_command_buffer() != nil", buf != nil);

  // 6. Same buffer returned before commit.
  CHECK("get_current_command_buffer() stable before commit",
        get_current_command_buffer() == buf);

  // 7. After commit_command_buffer(), a new buffer is issued.
  commit_command_buffer();
  id<MTLCommandBuffer> buf2 = get_current_command_buffer();
  CHECK("new buffer issued after commit_command_buffer()", buf2 != nil);
  CHECK("new buffer differs from committed buffer", buf2 != buf);

  // 8. commit_and_wait() with no encoded commands completes without error.
  bool no_throw = true;
  try {
    commit_and_wait();
  } catch (const std::exception& e) {
    std::printf("  commit_and_wait() threw: %s\n", e.what());
    no_throw = false;
  }
  CHECK("commit_and_wait() with empty buffer — no error", no_throw);

  // 9. After commit_and_wait(), a fresh buffer is ready.
  id<MTLCommandBuffer> buf3 = get_current_command_buffer();
  CHECK("fresh buffer ready after commit_and_wait()", buf3 != nil);

  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
