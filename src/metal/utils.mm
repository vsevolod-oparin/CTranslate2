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

      // Per-thread MTLSharedEvent for GPU-side cross-CB ordering.
      // Signaled in commit_command_buffer() before each non-blocking commit.
      // encode_barrier() encodes a wait on the latest signal value into the
      // current CB, ensuring prior CB's GPU work completes before new encodes
      // execute — without blocking the CPU.
      thread_local id<MTLSharedEvent> _thread_event = nil;
      thread_local uint64_t           _event_counter = 0;
      // Tracks the highest event value for which commit_and_wait() completed.
      // encode_barrier() is a no-op when _event_counter <= _last_waited,
      // because the CPU wait already guarantees all prior GPU work finished.
      thread_local uint64_t           _last_waited = 0;

      // Counter for commit_and_wait() calls (global atomic for cross-thread visibility).
      std::atomic<uint64_t> _commit_count{0};

      // M11.4: Accumulated GPU execution time (seconds) since last reset.
      // Global (not thread-local) for cross-thread visibility, matching commit_count.
      std::atomic<double> _gpu_time_elapsed{0.0};

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
        // commandBuffer returns autoreleased (+0).  We retain to own it
        // in the thread-local slot; released in commit_command_buffer().
        // The @autoreleasepool drains the autorelease reference immediately
        // so it doesn't leak on Python threads (which have no pool).
        @autoreleasepool {
          _thread_buffer = [[get_metal_command_queue() commandBuffer] retain];
        }
        CT2_METAL_CHECK_OBJ(_thread_buffer, "MTLCommandBuffer");
      }
      return _thread_buffer;
    }


    // Internal: commit with optional event signaling.
    // commit_and_wait_impl passes signal=false because waitUntilCompleted
    // already provides a full CPU+GPU barrier — the extra signal would be
    // redundant overhead on every sync point.
    static void commit_command_buffer_impl(bool signal_event) {
      if (_thread_buffer == nil) {
        return;
      }
      if (signal_event) {
        // Signal shared event so encode_barrier() in a later CB can
        // wait for this CB's GPU work to complete (no CPU block).
        if (_thread_event == nil) {
          _thread_event = [get_metal_device() newSharedEvent];
        }
        [_thread_buffer encodeSignalEvent:_thread_event value:++_event_counter];
      }
      [_thread_buffer commit];
      [_thread_buffer release];
      _thread_buffer = nil;
    }

    void commit_command_buffer() {
      commit_command_buffer_impl(/*signal_event=*/true);
    }


    // Debug tracing: global map of caller → commit count.
    // Note: _trace_mutex is accessed from an atexit handler (dump_commit_trace).
    // Static destruction order is technically unspecified relative to atexit,
    // but std::mutex is trivially destructible on Apple/glibc, so safe in practice.
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
      // Retain the buffer independently — commit_command_buffer() releases
      // the thread-local slot.  We need buf alive for waitUntilCompleted
      // and GPUEndTime/GPUStartTime access.
      id<MTLCommandBuffer> buf = [_thread_buffer retain];
      commit_command_buffer_impl(/*signal_event=*/false);
      @autoreleasepool {
        // Drain autoreleased ObjC temporaries (compute encoders,
        // descriptors, etc.) that accumulated since the last drain.
        [buf waitUntilCompleted];
        CT2_METAL_CHECK_BUFFER(buf);
      }
      _commit_count.fetch_add(1, std::memory_order_relaxed);
      // M11.4: Accumulate GPU execution time (atomic add via CAS loop).
      {
        double delta = buf.GPUEndTime - buf.GPUStartTime;
        double old_val = _gpu_time_elapsed.load(std::memory_order_relaxed);
        while (!_gpu_time_elapsed.compare_exchange_weak(
            old_val, old_val + delta, std::memory_order_relaxed)) {}
      }
      [buf release];
      // All prior GPU work (including any signaled events) is now complete.
      _last_waited = _event_counter;
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
      id<MTLBlitCommandEncoder> blit;
      @autoreleasepool {
        blit = [[cmd blitCommandEncoder] retain];
      }
      [blit copyFromBuffer:src_buf sourceOffset:src_off
                  toBuffer:dst_buf destinationOffset:dst_off
                      size:static_cast<NSUInteger>(bytes)];
      [blit endEncoding];
      [blit release];
    }

    id<MTLComputeCommandEncoder> create_compute_encoder() {
      id<MTLCommandBuffer> cmd = get_current_command_buffer();
      id<MTLComputeCommandEncoder> enc;
      @autoreleasepool {
        enc = [[cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
      }
      return enc;
    }

    void encode_barrier() {
      // GPU-side barrier: ensures all prior committed command buffers have
      // finished executing before any subsequent encodes in the current CB
      // begin on the GPU.  Unlike commit_and_wait(), this does NOT block the
      // CPU — it only serializes GPU execution across CB boundaries.
      //
      // No-op if:
      //   - No prior commit_command_buffer() calls occurred (same CB), or
      //   - commit_and_wait() already waited past the latest event value
      //     (all prior GPU work guaranteed complete).
      if (_event_counter <= _last_waited || _thread_event == nil) {
        return;
      }
      id<MTLCommandBuffer> cb = get_current_command_buffer();
      [cb encodeWaitForEvent:_thread_event value:_event_counter];
    }

    uint64_t commit_count() { return _commit_count.load(std::memory_order_relaxed); }
    void reset_commit_count() { _commit_count.store(0, std::memory_order_relaxed); }

    double gpu_time_elapsed() { return _gpu_time_elapsed.load(std::memory_order_relaxed); }
    void reset_gpu_time() { _gpu_time_elapsed.store(0.0, std::memory_order_relaxed); }

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
