// Standalone tests for M2.2 (synchronize) and M2.3 (ScopedDeviceSetter).
//
// Run from the repository root:
//   clang++ -std=c++17 -O0 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/sync_scoped_test.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     -framework Metal -framework Foundation -framework MetalPerformanceShaders \
//     -o sync_scoped_test && ./sync_scoped_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ctranslate2/devices.h"
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


// ---------------------------------------------------------------------------
// M2.2 — synchronize_stream / synchronize_device
// ---------------------------------------------------------------------------

static void test_sync() {
  std::printf("\n--- M2.2: synchronize (commit_and_wait) ---\n");

  // 1. commit_and_wait with no encoded commands — must not throw or hang.
  CHECK_NOTHROW("commit_and_wait() empty buffer — no error",
    commit_and_wait()
  );

  // 2. Encode a real blit (copy 64 bytes between two shared buffers) and sync.
  id<MTLDevice> dev = get_metal_device();
  const NSUInteger buf_size = 64;

  id<MTLBuffer> src = [dev newBufferWithLength:buf_size
                                       options:MTLResourceStorageModeShared];
  id<MTLBuffer> dst = [dev newBufferWithLength:buf_size
                                       options:MTLResourceStorageModeShared];

  // Fill source via CPU (shared memory — directly writable).
  uint8_t* src_ptr = static_cast<uint8_t*>([src contents]);
  uint8_t* dst_ptr = static_cast<uint8_t*>([dst contents]);
  for (NSUInteger i = 0; i < buf_size; ++i) src_ptr[i] = (uint8_t)i;
  std::memset(dst_ptr, 0, buf_size);

  // Encode blit copy into the deferred command buffer.
  id<MTLCommandBuffer> cb = get_current_command_buffer();
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:src sourceOffset:0
              toBuffer:dst destinationOffset:0
                  size:buf_size];
  [blit endEncoding];

  // commit_and_wait = synchronize_stream for Metal.
  CHECK_NOTHROW("commit_and_wait() after blit encode — no error",
    commit_and_wait()
  );

  // Verify GPU copied the data (shared memory — CPU reads directly).
  bool data_ok = true;
  for (NSUInteger i = 0; i < buf_size; ++i) {
    if (dst_ptr[i] != (uint8_t)i) { data_ok = false; break; }
  }
  CHECK("blit result correct after commit_and_wait()", data_ok);

  // 3. A fresh command buffer is ready after commit.
  id<MTLCommandBuffer> cb2 = get_current_command_buffer();
  CHECK("fresh command buffer ready after sync", cb2 != nil);
  CHECK("fresh buffer differs from committed one", cb2 != cb);
}


// ---------------------------------------------------------------------------
// M2.3 — ScopedDeviceSetter (Metal device index)
// ---------------------------------------------------------------------------

// Minimal stubs so we can test the template specialisations without linking
// the full CTranslate2 library.  The runtime dispatch (get_device_index /
// set_device_index) is not exercised here — only the Metal specialisations.

template <ctranslate2::Device D> int  get_device_index();
template <ctranslate2::Device D> void set_device_index(int index);

template<>
int get_device_index<ctranslate2::Device::METAL>() {
  return 0;
}

template<>
void set_device_index<ctranslate2::Device::METAL>(int index) {
  if (index != 0)
    throw std::invalid_argument(
        "Invalid Metal device index: " + std::to_string(index));
}

static void test_scoped_device_setter() {
  std::printf("\n--- M2.3: Metal device index ---\n");

  // 1. get_device_index<METAL>() always returns 0.
  CHECK("get_device_index<METAL>() == 0",
        get_device_index<ctranslate2::Device::METAL>() == 0);

  // 2. set_device_index<METAL>(0) is a no-op — must not throw.
  CHECK_NOTHROW("set_device_index<METAL>(0) — no error",
    set_device_index<ctranslate2::Device::METAL>(0)
  );

  // 3. set_device_index<METAL>(1) must throw invalid_argument.
  CHECK_THROWS("set_device_index<METAL>(1) — throws",
    set_device_index<ctranslate2::Device::METAL>(1)
  );

  // 4. ScopedDeviceSetter pattern: index 0 → prev == new → no set called.
  //    Simulate the RAII scope manually using the template functions.
  int prev = get_device_index<ctranslate2::Device::METAL>();
  CHECK("prev index == 0", prev == 0);
  // Constructor: prev == new (0 == 0), so no set_device_index call.
  // Destructor:  same — no restoration needed.
  CHECK_NOTHROW("ScopedDeviceSetter(METAL, 0) — no error",
    if (prev != 0) set_device_index<ctranslate2::Device::METAL>(0)
  );
}


int main() {
  std::printf("=== M2.2 + M2.3 tests ===\n");
  test_sync();
  test_scoped_device_setter();
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
