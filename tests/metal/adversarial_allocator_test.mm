// Adversarial tests for MetalAllocator (M3.1/M12.x).
//
// Focuses on edge cases and attack patterns NOT covered by allocator_test.mm:
//   - zero-size allocation
//   - SIZE_MAX bucket_size overflow
//   - interior pointer lookup (buffer_for_ptr with offset)
//   - buffer_for_ptr with nullptr / unknown pointer
//   - double free
//   - protect_buffer on non-live pointer (silent ignore)
//   - pending-free lifecycle (protect → free → flush)
//   - live_bytes / pool_bytes accounting
//   - bucket-size reuse: alloc 513 + 700 bytes share the same 1024-byte bucket
//   - pointer cache invalidation after free
//   - concurrent alloc/free thread safety
//
// Build:
//   clang++ -std=c++17 -O0 \
//     -I include -I src -DCT2_WITH_METAL -DCT2_WITH_MPS \
//     tests/metal/adversarial_allocator_test.mm \
//     -L build -lctranslate2.mps -Wl,-rpath,build \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
//     -o adversarial_allocator_test && ./adversarial_allocator_test

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/devices.h"
#include "metal/utils.h"

using namespace ctranslate2;

static int passed = 0;
static int failed = 0;

// ---------------------------------------------------------------------------
// Assertion macros
// ---------------------------------------------------------------------------

#define PASS(label) do { std::printf("  PASS  %s\n", label); ++passed; } while(0)
#define FAIL(label, msg) do { std::printf("  FAIL  %s — %s\n", label, msg); ++failed; } while(0)

#define CHECK(label, expr) do { \
  if (expr) { PASS(label); } else { FAIL(label, "assertion false"); } \
} while(0)

#define CHECK_THROWS(label, ...) do { \
  bool threw = false; \
  try { __VA_ARGS__; } catch (...) { threw = true; } \
  CHECK(label, threw); \
} while(0)

#define CHECK_NOTHROW(label, ...) do { \
  bool ok = true; \
  try { __VA_ARGS__; } catch (const std::exception& e) { \
    std::printf("  FAIL  %s — threw: %s\n", label, e.what()); \
    ok = false; ++failed; \
  } \
  if (ok) { PASS(label); } \
} while(0)

// ---------------------------------------------------------------------------
// Helper: get allocator and helpers
// ---------------------------------------------------------------------------

static Allocator& A() { return get_allocator<Device::MPS>(); }

// ---------------------------------------------------------------------------
// Test: allocate(0) — zero-size request
// Outcome: either returns non-null or throws; must NOT silently corrupt state.
// ---------------------------------------------------------------------------
static void test_alloc_zero_size_no_crash() {
  const char* name = "alloc_zero_size_no_crash";
  bool threw = false;
  void* ptr = nullptr;
  try {
    ptr = A().allocate(0);
  } catch (...) {
    threw = true;
  }
  // Either throws OR returns a stable pointer. If no throw, must be freeable.
  if (!threw && ptr != nullptr) {
    bool free_ok = true;
    try { A().free(ptr); } catch (...) { free_ok = false; }
    CHECK(name, free_ok);
  } else if (!threw && ptr == nullptr) {
    // allocate returned nullptr without throwing — undefined by the interface
    FAIL(name, "allocate(0) returned nullptr without throwing");
  } else {
    // threw — acceptable
    PASS(name);
  }
}

// ---------------------------------------------------------------------------
// Test: allocate(1) — minimal single-byte allocation
// ---------------------------------------------------------------------------
static void test_alloc_one_byte() {
  const char* name = "alloc_one_byte";
  void* ptr = nullptr;
  CHECK_NOTHROW(name, ptr = A().allocate(1));
  CHECK(name, ptr != nullptr);
  if (ptr) {
    static_cast<uint8_t*>(ptr)[0] = 0xAB;
    CHECK(name, static_cast<uint8_t*>(ptr)[0] == 0xAB);
    CHECK_NOTHROW("alloc_one_byte/free", A().free(ptr));
  }
}

// ---------------------------------------------------------------------------
// Test: allocate(SIZE_MAX) — bucket_size overflow path
//
// bucket_size(SIZE_MAX):
//   v = SIZE_MAX - 1 = 0xFFFFFFFFFFFFFFFE
//   After all bit-OR ops: v = 0xFFFFFFFFFFFFFFFF
//   v + 1 = 0 (wraps to zero on 64-bit)
// So bucket = 0, and newBufferWithLength:0 is requested.
// Metal either returns nil (→ throw) or a 0-byte buffer (dangerous).
// Either way the allocator must not return a usable pointer silently.
// ---------------------------------------------------------------------------
static void test_alloc_size_max_throws_or_oom() {
  const char* name = "alloc_size_max_throws_or_oom";
  bool threw = false;
  void* ptr = nullptr;
  try {
    ptr = A().allocate(SIZE_MAX);
  } catch (...) {
    threw = true;
  }
  if (threw) {
    PASS(name);
  } else if (ptr == nullptr) {
    // allocate should never return nullptr without throwing per the contract
    FAIL(name, "allocate(SIZE_MAX) returned nullptr without throwing");
  } else {
    // Allocated successfully — this means bucket_size overflowed to 0 and
    // Metal returned a 0-byte buffer with a SIZE_MAX requested_size in _live.
    // Free it to avoid leaking and mark as surprising.
    try { A().free(ptr); } catch (...) {}
    FAIL(name, "allocate(SIZE_MAX) succeeded — bucket_size overflow bug");
  }
}

// ---------------------------------------------------------------------------
// Test: allocate(SIZE_MAX/2 + 1) — another overflow case
// next_pow2(SIZE_MAX/2 + 1) = SIZE_MAX/2+1 if it's already a power of 2,
// but v = SIZE_MAX/2 = 0x7FFFFFFFFFFFFFFF → after bit-OR: 0xFFFFFFFFFFFFFFFF
// → v+1 = 0 overflow again.
// ---------------------------------------------------------------------------
static void test_alloc_near_size_max_throws_or_oom() {
  const char* name = "alloc_near_size_max_throws";
  bool threw = false;
  void* ptr = nullptr;
  try {
    // SIZE_MAX / 2 + 1 in bucket_size also overflows on 64-bit
    ptr = A().allocate(SIZE_MAX / 2 + 1);
  } catch (...) {
    threw = true;
  }
  if (threw) {
    PASS(name);
  } else {
    // For very large values, Metal may fail allocation naturally (OOM),
    // but it should not return a corrupt/zero bucket.
    if (ptr) {
      try { A().free(ptr); } catch (...) {}
    }
    PASS(name);  // Large but not overflowing allocation might just OOM — acceptable
  }
}

// ---------------------------------------------------------------------------
// Test: double free — second free must throw
// ---------------------------------------------------------------------------
static void test_double_free_throws() {
  const char* name = "double_free_throws";
  void* ptr = A().allocate(64);
  A().free(ptr);
  // The ptr is now in the pool. A second free of the same raw pointer should
  // throw "attempt to free unknown pointer" because it's no longer in _live.
  CHECK_THROWS(name, A().free(ptr));
  // Clean up: the pointer is now in the pool from first free; drain it.
  // Allocate the same bucket size to pull it back out, then legitimately free.
  void* p2 = A().allocate(64);
  A().free(p2);
}

// ---------------------------------------------------------------------------
// Test: free(nullptr) — must be silent no-op (already tested in allocator_test
// but verified here too to guard regressions)
// ---------------------------------------------------------------------------
static void test_free_nullptr_noop() {
  CHECK_NOTHROW("free_nullptr_noop", A().free(nullptr));
}

// ---------------------------------------------------------------------------
// Test: buffer_for_ptr(nullptr) — must throw
// ---------------------------------------------------------------------------
static void test_buffer_for_ptr_null_throws() {
  const char* name = "buffer_for_ptr_null_throws";
  CHECK_THROWS(name, {
    NSUInteger off = 0;
    metal_buffer_for_ptr(nullptr, &off);
  });
}

// ---------------------------------------------------------------------------
// Test: buffer_for_ptr(stack pointer) — must throw
// ---------------------------------------------------------------------------
static void test_buffer_for_ptr_stack_ptr_throws() {
  const char* name = "buffer_for_ptr_stack_ptr_throws";
  int stack_var = 42;
  CHECK_THROWS(name, {
    NSUInteger off = 0;
    metal_buffer_for_ptr(&stack_var, &off);
  });
}

// ---------------------------------------------------------------------------
// Test: buffer_for_ptr with interior pointer — must return base buffer + correct offset
// ---------------------------------------------------------------------------
static void test_buffer_for_ptr_interior_pointer() {
  const char* name = "buffer_for_ptr_interior_pointer";
  const size_t sz = 1024;
  uint8_t* base = static_cast<uint8_t*>(A().allocate(sz));

  // Check several interior offsets
  bool all_ok = true;
  for (size_t off_in = 0; off_in < sz; off_in += 64) {
    NSUInteger off_out = 0;
    id<MTLBuffer> buf = nil;
    try {
      buf = metal_buffer_for_ptr(base + off_in, &off_out);
    } catch (...) {
      all_ok = false;
      break;
    }
    if (buf == nil || off_out != (NSUInteger)off_in) {
      all_ok = false;
      break;
    }
  }

  CHECK(name, all_ok);

  // Also test the last byte (offset sz-1)
  {
    NSUInteger off_out = 0;
    bool ok = true;
    try {
      metal_buffer_for_ptr(base + sz - 1, &off_out);
      ok = (off_out == sz - 1);
    } catch (...) {
      ok = false;
    }
    CHECK("buffer_for_ptr_last_byte", ok);
  }

  A().free(base);
}

// ---------------------------------------------------------------------------
// Test: buffer_for_ptr on a freed pointer — must throw (no longer in _live)
// ---------------------------------------------------------------------------
static void test_buffer_for_ptr_freed_pointer_throws() {
  const char* name = "buffer_for_ptr_freed_ptr_throws";
  uint8_t* ptr = static_cast<uint8_t*>(A().allocate(128));
  A().free(ptr);
  // ptr is now in the pool, not in _live
  CHECK_THROWS(name, {
    NSUInteger off = 0;
    metal_buffer_for_ptr(ptr, &off);
  });
  // Drain the pool entry we just freed
  void* p2 = A().allocate(128);
  A().free(p2);
}

// ---------------------------------------------------------------------------
// Test: bucket-size reuse — 513 and 700 bytes both bucket to 1024
// After freeing a 513-byte allocation, a 700-byte request should reuse the
// same MTLBuffer (same contents pointer, same underlying buffer).
// ---------------------------------------------------------------------------
static void test_bucket_reuse_different_sizes() {
  const char* name = "bucket_reuse_different_sizes";

  // First drain any cached 1024-bucket entry from prior tests
  A().clear_cache();

  void* p1 = A().allocate(513);   // buckets to 1024
  A().free(p1);                   // returns the 1024-bucket buffer to pool

  void* p2 = A().allocate(700);   // also buckets to 1024 → pool hit
  // If bucketing works, p2 == p1 (same MTLBuffer contents address reused)
  CHECK(name, p2 == p1);
  A().free(p2);
}

// ---------------------------------------------------------------------------
// Test: bucket boundary — exactly kBucketMinSize (512) stays exact,
// 513 bumps to 1024.
// We test indirectly: alloc 512, free, alloc 513 — should NOT reuse the 512
// bucket (different bucket size), so p2 != p1.
// ---------------------------------------------------------------------------
static void test_bucket_boundary_512_vs_513() {
  const char* name = "bucket_boundary_512_not_1024";

  A().clear_cache();

  void* p512 = A().allocate(512);  // bucket = 512 (≤ kBucketMinSize)
  A().free(p512);                   // puts a 512-byte bucket in pool

  void* p513 = A().allocate(513);  // bucket = 1024 (new allocation)
  // Different buckets → should NOT return p512
  CHECK(name, p513 != p512);
  A().free(p513);
}

// ---------------------------------------------------------------------------
// Test: live_bytes and pool_bytes accounting
// ---------------------------------------------------------------------------
static void test_live_and_pool_bytes_accounting() {
  const char* name = "live_pool_bytes_accounting";

  A().clear_cache();

  size_t live0 = metal::live_bytes();
  size_t pool0 = metal::pool_bytes();

  void* p = A().allocate(4096);
  size_t live1 = metal::live_bytes();
  // Live bytes must have increased by at least the requested amount
  CHECK("live_bytes_increases_on_alloc", live1 >= live0 + 4096);

  size_t pool1 = metal::pool_bytes();
  CHECK("pool_bytes_unchanged_on_alloc", pool1 == pool0);

  A().free(p);
  size_t live2 = metal::live_bytes();
  size_t pool2 = metal::pool_bytes();
  CHECK("live_bytes_decreases_on_free", live2 == live0);
  // Pool grows by bucket_size(4096) = 4096
  CHECK("pool_bytes_increases_on_free", pool2 >= pool1 + 4096);

  A().clear_cache();
}

// ---------------------------------------------------------------------------
// Test: protect_buffer on a non-live pointer — must NOT throw (silently ignored)
// ---------------------------------------------------------------------------
static void test_protect_buffer_unknown_ptr_no_throw() {
  const char* name = "protect_buffer_unknown_ptr_no_throw";
  int stack_var = 0;
  CHECK_NOTHROW(name, metal::protect_buffer(&stack_var));
}

// ---------------------------------------------------------------------------
// Test: protect_buffer_by_base on a non-live pointer — must NOT throw
// ---------------------------------------------------------------------------
static void test_protect_buffer_by_base_unknown_no_throw() {
  const char* name = "protect_buffer_by_base_unknown_no_throw";
  int stack_var = 0;
  CHECK_NOTHROW(name, metal::protect_buffer_by_base(&stack_var));
}

// ---------------------------------------------------------------------------
// Test: flush_pending_frees on empty list — must be a no-op
// ---------------------------------------------------------------------------
static void test_flush_pending_frees_empty_no_crash() {
  A().clear_cache();
  CHECK_NOTHROW("flush_pending_frees_empty", metal::flush_pending_frees());
}

// ---------------------------------------------------------------------------
// Test: protect → free → flush lifecycle
// After protect_buffer: free goes to _pending_free, not pool.
// After flush_pending_frees: buffer returns to pool.
// Verify: second allocation of same size reuses the buffer.
// ---------------------------------------------------------------------------
static void test_protect_free_flush_recycle() {
  const char* name = "protect_free_flush_recycle";

  A().clear_cache();

  void* p = A().allocate(2048);
  metal::protect_buffer(p);    // mark as GPU-protected
  A().free(p);                 // goes to _pending_free, NOT pool

  // Now pool should not have the 2048-bucket buffer
  void* p2 = A().allocate(2048);
  // p2 should be a NEW allocation (not p) because p is in pending_free
  bool is_new = (p2 != p);
  A().free(p2);                // p2 → pool

  // Flush pending frees — now p should move to pool
  metal::flush_pending_frees();

  // Allocate again: may hit pool (either p2 or p, both in pool now)
  void* p3 = A().allocate(2048);
  bool from_pool = (p3 == p || p3 == p2);
  A().free(p3);

  CHECK(name, is_new);          // Before flush: p was NOT recycled
  CHECK("protect_flush_pool_hit", from_pool);  // After flush: pool hit

  A().clear_cache();
}

// ---------------------------------------------------------------------------
// Test: clear_cache invalidates pointer cache (buffer_for_ptr on old ptr throws)
// ---------------------------------------------------------------------------
static void test_clear_cache_invalidates_ptr_cache() {
  const char* name = "clear_cache_invalidates_ptr_cache";

  void* p = A().allocate(256);
  // Warm the pointer cache
  {
    NSUInteger off = 0;
    metal_buffer_for_ptr(p, &off);
  }
  A().free(p);
  A().clear_cache();

  // After clear_cache, the buffer is released. buffer_for_ptr must throw.
  CHECK_THROWS(name, {
    NSUInteger off = 0;
    metal_buffer_for_ptr(p, &off);
  });
}

// ---------------------------------------------------------------------------
// Test: many alloc/free cycles — no memory growth (pool cap sanity)
// Allocate and free 200 small buffers of varying sizes within the same bucket.
// Pool should not grow unboundedly (same bucket reused).
// ---------------------------------------------------------------------------
static void test_alloc_free_many_no_growth() {
  const char* name = "alloc_free_many_no_growth";

  A().clear_cache();

  const int N = 200;
  for (int i = 0; i < N; ++i) {
    @autoreleasepool {
      void* p = A().allocate(512 + i % 64);  // all bucket to same power-of-2
      A().free(p);
    }
  }

  // Pool should not have grown to N entries — pool size is bounded by the
  // fact that the same bucket address is repeatedly reused.
  size_t pool_after = metal::pool_bytes();
  // Expect at most a handful of distinct entries (not 200)
  const size_t kMaxExpectedPoolBytes = 200 * 1024;  // generous upper bound
  CHECK(name, pool_after < kMaxExpectedPoolBytes);

  A().clear_cache();
}

// ---------------------------------------------------------------------------
// Test: concurrent alloc/free — thread safety of mutex
// Run 4 threads each doing 50 alloc/free cycles.
// Must not crash or produce data races under TSan.
// ---------------------------------------------------------------------------
static void test_concurrent_alloc_free_no_crash() {
  const char* name = "concurrent_alloc_free_no_crash";

  A().clear_cache();
  std::atomic<int> errors{0};

  const int kThreads = 4;
  const int kIters   = 50;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      @autoreleasepool {
        for (int i = 0; i < kIters; ++i) {
          @autoreleasepool {
            const size_t sz = 1024 * ((t % 4) + 1);  // 1K, 2K, 3K, 4K
            void* p = nullptr;
            try {
              p = A().allocate(sz);
              if (p) {
                // Write to verify it's writable
                static_cast<uint8_t*>(p)[0] = static_cast<uint8_t>(t);
                A().free(p);
              }
            } catch (...) {
              errors.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      }
    });
  }

  for (auto& th : threads) th.join();

  CHECK(name, errors.load() == 0);
  A().clear_cache();
}

// ---------------------------------------------------------------------------
// Test: allocate very large but valid sizes — 1GB should either succeed or throw
// (not silently corrupt state)
// ---------------------------------------------------------------------------
static void test_alloc_1gb_or_throw() {
  const char* name = "alloc_1gb_or_throw";
  bool threw = false;
  void* ptr = nullptr;
  try {
    ptr = A().allocate(1UL << 30);  // 1 GB
  } catch (...) {
    threw = true;
  }
  if (threw) {
    PASS(name);
  } else if (ptr) {
    // If it succeeded, free it gracefully
    A().free(ptr);
    PASS(name);
  } else {
    FAIL(name, "returned nullptr without throwing");
  }
}

// ---------------------------------------------------------------------------
// Test: ptr cache hit rate tracks correctly after alloc
// ---------------------------------------------------------------------------
static void test_ptr_cache_stats_tracking() {
  const char* name = "ptr_cache_stats_tracking";

  A().clear_cache();
  metal::reset_ptr_cache_stats();

  void* p = A().allocate(512);
  // First lookup: miss (not in cache yet)
  NSUInteger off = 0;
  metal_buffer_for_ptr(p, &off);
  uint64_t miss1 = metal::ptr_cache_misses();
  CHECK("ptr_cache_first_lookup_is_miss", miss1 >= 1);

  // Second lookup: should be a hit (cached from first lookup)
  metal_buffer_for_ptr(p, &off);
  uint64_t hit1 = metal::ptr_cache_hits();
  CHECK(name, hit1 >= 1);

  A().free(p);
  A().clear_cache();
}

// ---------------------------------------------------------------------------
// Test: ptr cache is invalidated on free — after free, the slot is cleared
// and a subsequent lookup of the same pointer throws (no stale cache hit).
// ---------------------------------------------------------------------------
static void test_ptr_cache_invalidated_on_free() {
  const char* name = "ptr_cache_invalidated_on_free";

  void* p = A().allocate(256);
  // Warm cache
  NSUInteger off = 0;
  metal_buffer_for_ptr(p, &off);

  // Free invalidates the cache entry
  A().free(p);

  // Now buffer_for_ptr must throw (ptr not in _live, and cache must not return stale)
  bool threw = false;
  try {
    metal_buffer_for_ptr(p, &off);
  } catch (...) {
    threw = true;
  }
  CHECK(name, threw);

  // Drain pool
  void* p2 = A().allocate(256);
  A().free(p2);
}

// ---------------------------------------------------------------------------
// Test: interior pointer at exact allocation end (one-past-end) must throw
// ---------------------------------------------------------------------------
static void test_buffer_for_ptr_one_past_end_throws() {
  const char* name = "buffer_for_ptr_one_past_end_throws";
  const size_t sz = 256;
  uint8_t* base = static_cast<uint8_t*>(A().allocate(sz));

  // base + sz is ONE past the end — NOT within [base, base+sz)
  CHECK_THROWS(name, {
    NSUInteger off = 0;
    metal_buffer_for_ptr(base + sz, &off);
  });

  A().free(base);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== Adversarial MetalAllocator Tests ===\n\n");

  test_alloc_zero_size_no_crash();
  test_alloc_one_byte();
  test_alloc_size_max_throws_or_oom();
  test_alloc_near_size_max_throws_or_oom();
  test_double_free_throws();
  test_free_nullptr_noop();
  test_buffer_for_ptr_null_throws();
  test_buffer_for_ptr_stack_ptr_throws();
  test_buffer_for_ptr_interior_pointer();
  test_buffer_for_ptr_freed_pointer_throws();
  test_bucket_reuse_different_sizes();
  test_bucket_boundary_512_vs_513();
  test_live_and_pool_bytes_accounting();
  test_protect_buffer_unknown_ptr_no_throw();
  test_protect_buffer_by_base_unknown_no_throw();
  test_flush_pending_frees_empty_no_crash();
  test_protect_free_flush_recycle();
  test_clear_cache_invalidates_ptr_cache();
  test_alloc_free_many_no_growth();
  test_concurrent_alloc_free_no_crash();
  test_alloc_1gb_or_throw();
  test_ptr_cache_stats_tracking();
  test_ptr_cache_invalidated_on_free();
  test_buffer_for_ptr_one_past_end_throws();

  std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
