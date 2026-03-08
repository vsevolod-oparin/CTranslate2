#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>

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

      // Counter for commit_and_wait() calls (global atomic for cross-thread visibility).
      std::atomic<uint64_t> _commit_count{0};

      // M11.4: Accumulated GPU execution time (seconds) since last reset.
      thread_local double _gpu_time_elapsed = 0.0;

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


    // Debug tracing: global map of caller → commit count.
    static std::mutex _trace_mutex;
    static std::unordered_map<std::string, uint64_t> _trace_counts;
    static bool _trace_enabled = false;

    void enable_commit_trace(bool on) { _trace_enabled = on; }

    void commit_and_wait_impl(const char* caller) {
      if (_thread_buffer == nil) {
        return;
      }
      // Auto-enable trace via env var on first real commit.
      static std::once_flag _env_flag;
      std::call_once(_env_flag, [] {
        if (std::getenv("CT2_METAL_TRACE")) {
          _trace_enabled = true;
          std::atexit([] { dump_commit_trace(); });
        }
      });
      // Capture a strong reference before resetting the thread-local slot.
      id<MTLCommandBuffer> buf = _thread_buffer;
      commit_command_buffer();
      [buf waitUntilCompleted];
      CT2_METAL_CHECK_BUFFER(buf);
      _commit_count.fetch_add(1, std::memory_order_relaxed);
      // M11.4: Accumulate GPU execution time.
      _gpu_time_elapsed += (buf.GPUEndTime - buf.GPUStartTime);
      // M11.18: Now that all GPU work has completed, recycle deferred-free
      // buffers back to the allocator pool for reuse.
      flush_pending_frees();
      if (_trace_enabled && caller) {
        std::lock_guard<std::mutex> lk(_trace_mutex);
        _trace_counts[caller]++;
      }
    }

    void commit_and_wait() {
      commit_and_wait_impl("unknown");
    }

    void dump_commit_trace() {
      std::lock_guard<std::mutex> lk(_trace_mutex);
      // Sort by count descending
      std::vector<std::pair<std::string, uint64_t>> sorted(_trace_counts.begin(), _trace_counts.end());
      std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
      fprintf(stderr, "=== commit_and_wait trace ===\n");
      for (auto& [k, v] : sorted)
        fprintf(stderr, "  %5llu  %s\n", (unsigned long long)v, k.c_str());
      fprintf(stderr, "============================\n");
    }

    void reset_commit_trace() {
      std::lock_guard<std::mutex> lk(_trace_mutex);
      _trace_counts.clear();
    }

    void blit_copy(const void* src, void* dst, size_t bytes) {
      if (bytes == 0) return;
      NSUInteger src_off = 0, dst_off = 0;
      id<MTLBuffer> src_buf = metal_buffer_for_ptr(src, &src_off);
      id<MTLBuffer> dst_buf = metal_buffer_for_ptr(dst, &dst_off);
      id<MTLCommandBuffer> cmd = get_current_command_buffer();
      id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
      [blit copyFromBuffer:src_buf sourceOffset:src_off
                  toBuffer:dst_buf destinationOffset:dst_off
                      size:static_cast<NSUInteger>(bytes)];
      [blit endEncoding];
    }

    uint64_t commit_count() { return _commit_count.load(std::memory_order_relaxed); }
    void reset_commit_count() { _commit_count.store(0, std::memory_order_relaxed); }

    double gpu_time_elapsed() { return _gpu_time_elapsed; }
    void reset_gpu_time() { _gpu_time_elapsed = 0.0; }

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
