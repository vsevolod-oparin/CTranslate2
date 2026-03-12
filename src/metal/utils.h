#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "ctranslate2/types.h"

// ---------------------------------------------------------------------------
// C++ interface — callable from plain .cc files
// ---------------------------------------------------------------------------

namespace ctranslate2 {
  namespace metal {

    // Commit the current thread's command buffer and block until the GPU
    // finishes all encoded commands.
    // This is the implementation of synchronize_stream(Device::MPS) and
    // synchronize_device(Device::MPS).
    // No-op if no commands have been encoded since the last commit.
    void commit_and_wait();

// Traced variant — records caller for debugging.
#define CT2_STRINGIFY2(x) #x
#define CT2_STRINGIFY(x) CT2_STRINGIFY2(x)
#define CT2_COMMIT_AND_WAIT() \
    ctranslate2::metal::commit_and_wait_impl(__FILE__ ":" CT2_STRINGIFY(__LINE__))

    // Profiling: thread-local count of commit_and_wait() calls.
    uint64_t commit_count();
    void reset_commit_count();

    // M11.4: GPU-native timing from Metal command buffers.
    // Accumulated GPU execution time (seconds) since last reset.
    double gpu_time_elapsed();
    void reset_gpu_time();

    // Debug: commit tracing.
    void commit_and_wait_impl(const char* caller);
    void enable_commit_trace(bool on);
    void dump_commit_trace();
    void reset_commit_trace();

    // M11.18: Deferred-free mechanism for encode-only GPU kernels.
    //
    // protect_buffer() marks a live allocation as GPU-referenced.  When that
    // buffer is later freed, it goes to a deferred queue instead of the reuse
    // pool.  flush_pending_frees() moves deferred buffers back to the pool
    // after commit_and_wait() completes all prior GPU work.
    //
    // This prevents encode-only GPU kernels from reading/writing a buffer
    // that has been recycled and overwritten by a subsequent allocate().
    void protect_buffer(const void* ptr);
    // O(1) version — caller supplies the base pointer (e.g. [buf contents]).
    void protect_buffer_by_base(void* base_ptr);
    void flush_pending_frees();

    // GPU-side barrier: encodes a wait in the current command buffer for all
    // prior committed CBs to complete.  Unlike commit_and_wait(), this does
    // NOT block the CPU.  Use when an encode-only kernel needs to read data
    // written by a prior CB (e.g. indexed_fill after a padded GEMM that
    // split command buffers).
    void encode_barrier();

    // Release all cached MPSMatrixMultiplication objects (MPS GEMM cache).
    // Called from MetalAllocator::clear_cache() to prevent unbounded growth.
    void clear_gemm_cache();

    // Release cached SDPA MPSMatrixMultiplication objects (P1 cache).
    void clear_sdpa_gemm_cache();

    // Allocator memory stats (bytes).
    size_t pool_bytes();   // Cached but unused (reclaimable via clear_cache).
    size_t live_bytes();   // Currently in use by StorageView/tensors.

    // M12.3: Pointer cache profiling counters.
    uint64_t ptr_cache_hits();
    uint64_t ptr_cache_misses();
    void reset_ptr_cache_stats();

    // M11.2: Global PSO cache hit/miss counters.
    // Incremented by PSOCache::get() in primitives_infra.h.
    uint64_t pso_hit_count();
    uint64_t pso_miss_count();
    void reset_pso_stats();
    void increment_pso_hits();
    void increment_pso_misses();

    // Fused timestamp check + disable (M11.17).
    // Performs should_sample_timestamps reduction AND writes -inf
    // directly to logits[batch_id][0..num_text) on the GPU.
    // Encode-only — no CT2_COMMIT_AND_WAIT().
    template <typename T>
    void fuse_timestamp_check_and_disable_metal(
        const T* log_probs, T* logits, dim_t vocab_size,
        dim_t num_text_tokens, dim_t num_ts_tokens,
        const std::vector<dim_t>& batch_ids);

  }  // namespace metal
}  // namespace ctranslate2


// ---------------------------------------------------------------------------
// ObjC++ interface — only visible when compiling .mm files
// ---------------------------------------------------------------------------

#ifdef __OBJC__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace ctranslate2 {
  namespace metal {

    // Returns the process-wide MTLDevice singleton (lazy, thread-safe).
    id<MTLDevice> get_metal_device();

    // Returns the calling thread's MTLCommandQueue (created on first call).
    id<MTLCommandQueue> get_metal_command_queue();

    // Returns the calling thread's active MTLCommandBuffer.
    // A new buffer is created automatically after each commit_command_buffer().
    // Metal ops must encode into this buffer — never commit it themselves.
    id<MTLCommandBuffer> get_current_command_buffer();

    // Commits the current thread's command buffer and resets the thread-local
    // slot to nil.  The committed buffer object remains valid for the caller
    // to call [buf waitUntilCompleted] on.
    // Only commit_and_wait() should call this directly.
    void commit_command_buffer();

    // Encode a GPU-side buffer copy into the current command buffer.
    // No commit — the copy executes when the CB is eventually committed.
    // Both src and dst must be within MetalAllocator-managed buffers.
    void blit_copy(const void* src, void* dst, size_t bytes);

    // Create a serial compute encoder from the current command buffer.
    // Returns a +1 retained encoder — caller MUST call [enc endEncoding]
    // followed by [enc release].  The @autoreleasepool inside drains the
    // autoreleased reference so it doesn't leak on Python threads.
    id<MTLComputeCommandEncoder> create_compute_encoder();

  }  // namespace metal

  // ---------------------------------------------------------------------------
  // Buffer lookup (ObjC++ only — needs id<MTLBuffer>)
  // ---------------------------------------------------------------------------

  // Returns the MTLBuffer that contains ptr (which may be any byte offset
  // within a MetalAllocator allocation) and fills *offset_out with the byte
  // offset of ptr within that buffer.  Throws if ptr was not allocated by
  // MetalAllocator.
  id<MTLBuffer> metal_buffer_for_ptr(const void* ptr, NSUInteger* offset_out);

}  // namespace ctranslate2


// ---------------------------------------------------------------------------
// Error-check macros (ObjC++ only)
// ---------------------------------------------------------------------------

// Check command buffer status AFTER [buf waitUntilCompleted].
#define CT2_MPS_CHECK_BUFFER(buf)                                           \
  do {                                                                        \
    if ((buf).status == MTLCommandBufferStatusError) {                        \
      throw std::runtime_error(                                               \
          std::string("Metal command buffer error: ") +                       \
          [(buf).error.localizedDescription UTF8String]);                     \
    }                                                                         \
  } while (0)

// Check that an Objective-C object was successfully allocated (non-nil).
#define CT2_MPS_CHECK_OBJ(obj, name)                                        \
  do {                                                                        \
    if ((obj) == nil) {                                                       \
      throw std::runtime_error("Metal: failed to create " name);              \
    }                                                                         \
  } while (0)

#endif  // __OBJC__
