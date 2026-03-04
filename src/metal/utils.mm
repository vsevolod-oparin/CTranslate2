#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>

#include "utils.h"

namespace ctranslate2 {
  namespace metal {
    namespace {

      // Per-thread command queue.
      // thread_local with ARC strong ObjC pointer is supported by AppleClang
      // in ObjC++ mode.  Minor leak on thread exit is acceptable for
      // long-lived inference threads.
      thread_local id<MTLCommandQueue>  _thread_queue  = nil;

      // Per-thread active command buffer.  Ops encode into this; only
      // commit_and_wait() / commit_command_buffer() may commit it.
      thread_local id<MTLCommandBuffer> _thread_buffer = nil;

      // Counter for commit_and_wait() calls (profiling).
      thread_local uint64_t _commit_count = 0;

    }  // namespace


    id<MTLDevice> get_metal_device() {
      // C++11 function-local static initialisation is thread-safe.
      // MTLCreateSystemDefaultDevice() never throws, so the static is
      // initialised exactly once even if the result is nil.
      static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        throw std::runtime_error("Metal: no Metal-capable device found");
      }
      return device;
    }


    id<MTLCommandQueue> get_metal_command_queue() {
      if (_thread_queue == nil) {
        _thread_queue = [get_metal_device() newCommandQueue];
        CT2_METAL_CHECK_OBJ(_thread_queue, "MTLCommandQueue");
      }
      return _thread_queue;
    }


    id<MTLCommandBuffer> get_current_command_buffer() {
      if (_thread_buffer == nil) {
        _thread_buffer = [get_metal_command_queue() commandBuffer];
        CT2_METAL_CHECK_OBJ(_thread_buffer, "MTLCommandBuffer");
      }
      return _thread_buffer;
    }


    void commit_command_buffer() {
      if (_thread_buffer == nil) {
        return;
      }
      [_thread_buffer commit];
      _thread_buffer = nil;
    }


    void commit_and_wait() {
      if (_thread_buffer == nil) {
        return;
      }
      // Capture a strong reference before resetting the thread-local slot.
      id<MTLCommandBuffer> buf = _thread_buffer;
      commit_command_buffer();
      [buf waitUntilCompleted];
      CT2_METAL_CHECK_BUFFER(buf);
      ++_commit_count;
    }

    uint64_t commit_count() { return _commit_count; }
    void reset_commit_count() { _commit_count = 0; }

    // M11.2: global PSO cache statistics (atomics for thread safety).
    namespace {
      std::atomic<uint64_t> _pso_hits{0};
      std::atomic<uint64_t> _pso_misses{0};
    }

    uint64_t pso_hit_count() { return _pso_hits.load(std::memory_order_relaxed); }
    uint64_t pso_miss_count() { return _pso_misses.load(std::memory_order_relaxed); }
    void reset_pso_stats() {
      _pso_hits.store(0, std::memory_order_relaxed);
      _pso_misses.store(0, std::memory_order_relaxed);
    }
    void increment_pso_hits() { _pso_hits.fetch_add(1, std::memory_order_relaxed); }
    void increment_pso_misses() { _pso_misses.fetch_add(1, std::memory_order_relaxed); }

  }  // namespace metal
}  // namespace ctranslate2
