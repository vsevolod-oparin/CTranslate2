#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ctranslate2/allocator.h"
#include "metal/utils.h"

namespace ctranslate2 {
  namespace metal {

    // Caching Metal allocator backed by MTLResourceStorageModeShared.
    //
    // On Apple Silicon all memory is physically unified — CPU and GPU share the
    // same DRAM.  A Shared-mode MTLBuffer gives a stable void* via [buf contents]
    // that is simultaneously valid for CPU reads/writes and GPU access.
    // StorageView stores that pointer directly; no cudaMemcpy equivalent is needed.
    //
    // Buffer lifetime:
    //   _live  : ptr → {requested_size, MTLBuffer}   (currently in use)
    //   _pool  : requested_size → [MTLBuffer, ...]    (available for reuse)
    //
    // We key the pool on the *requested* size (not buf.length which Metal may
    // round up) so that allocate() finds the exact-size bucket it inserted into.
    //
    // Thread safety: a single mutex guards both maps.  Allocation is not on the
    // hot path (buffers are long-lived in CTranslate2's StorageView).
    //
    // Memory management: ARC is NOT enabled for this TU (thread-local
    // id<MTLCommandBuffer/Queue> in utils.mm prevent it).  MTLBuffer objects
    // are manually released.  newBufferWithLength: returns +1 retained;
    // we release when evicting from pool or on destruction.

    class MetalAllocator : public Allocator {
    public:
      ~MetalAllocator() {
        for (auto& [sz, bufs] : _pool)
          for (id<MTLBuffer> buf : bufs)
            [buf release];
        for (auto& entry : _pending_free)
          [entry.buffer release];
        for (auto& [ptr, entry] : _live)
          [entry.buffer release];
      }


      // Returns the MTLBuffer that contains ptr and sets *offset_out to the
      // byte offset of ptr within that buffer.  Needed by compute encoders
      // (which take id<MTLBuffer> + offset, not raw void*).
      //
      // Uses a sorted map (std::map) with upper_bound for O(log n) lookup
      // instead of O(n) linear scan.  Called for every setBuffer: dispatch.
      id<MTLBuffer> buffer_for_ptr(const void* ptr, NSUInteger* offset_out) {
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
        std::lock_guard<std::mutex> lock(_mutex);

        // Find the first entry with base > byte_ptr, then step back one.
        auto it = _live.upper_bound(byte_ptr);
        if (it != _live.begin()) {
          --it;
          if (byte_ptr < it->first + it->second.requested_size) {
            if (offset_out) {
              *offset_out = static_cast<NSUInteger>(byte_ptr - it->first);
            }
            return it->second.buffer;
          }
        }
        throw std::runtime_error("Metal: pointer not in any live allocation");
      }


      void* allocate(size_t size, int /*device_index*/) override {
        std::lock_guard<std::mutex> lock(_mutex);

        // Check pool for a cached buffer of the same size.
        auto pool_it = _pool.find(size);
        if (pool_it != _pool.end() && !pool_it->second.empty()) {
          id<MTLBuffer> buf = pool_it->second.back();
          pool_it->second.pop_back();
          uint8_t* ptr = static_cast<uint8_t*>([buf contents]);
          _live[ptr] = {size, buf};
          return ptr;
        }

        // No cached buffer — allocate a new one.
        id<MTLBuffer> buf = [get_metal_device()
            newBufferWithLength:size
                        options:MTLResourceStorageModeShared];
        if (buf == nil) {
          throw std::runtime_error(
              "Metal: failed to allocate MTLBuffer of size " + std::to_string(size));
        }

        uint8_t* ptr = static_cast<uint8_t*>([buf contents]);
        _live[ptr] = {size, buf};
        return ptr;
      }

      void free(void* ptr, int /*device_index*/) override {
        if (!ptr) {
          return;
        }
        std::lock_guard<std::mutex> lock(_mutex);

        auto live_it = _live.find(static_cast<uint8_t*>(ptr));
        if (live_it == _live.end()) {
          throw std::runtime_error("Metal: attempt to free unknown pointer");
        }

        const size_t     sz  = live_it->second.requested_size;
        id<MTLBuffer>    buf = live_it->second.buffer;
        const bool  is_protected = live_it->second.gpu_protected;
        _live.erase(live_it);

        if (is_protected) {
          // M11.18: This buffer is referenced by a pending encode-only GPU
          // kernel (e.g. MPS padded GEMM row_copy-back, indexed_fill indices).
          // Defer recycling until commit_and_wait() completes all prior GPU work.
          _pending_free.push_back({sz, buf, false});
        } else {
          // Return to pool immediately (safe — not GPU-referenced).
          _pool[sz].push_back(buf);
        }
      }

      // Mark a live buffer as GPU-protected: its reclamation will be deferred
      // when free() is called, until after the next commit_and_wait().
      //
      // O(log n) via sorted map upper_bound.
      // Prefer protect_buffer_by_base() when the base pointer is already known.
      void protect_buffer(const void* ptr) {
        std::lock_guard<std::mutex> lock(_mutex);
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
        auto it = _live.upper_bound(byte_ptr);
        if (it != _live.begin()) {
          --it;
          if (byte_ptr < it->first + it->second.requested_size) {
            it->second.gpu_protected = true;
            return;
          }
        }
        // Not found — might already be freed or not from this allocator.
      }

      // O(log n) version — caller supplies the base pointer of the allocation
      // (e.g. from [MTLBuffer contents] after metal_buffer_for_ptr()).
      void protect_buffer_by_base(void* base_ptr) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _live.find(static_cast<uint8_t*>(base_ptr));
        if (it != _live.end()) {
          it->second.gpu_protected = true;
        }
      }

      // Move all pending-free buffers to the pool for reuse.
      // Called after commit_and_wait() ensures prior GPU work has completed.
      void flush_pending_frees() {
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& entry : _pending_free) {
          _pool[entry.requested_size].push_back(entry.buffer);
        }
        _pending_free.clear();
      }

      void clear_cache() override {
        // Flush any in-flight GPU work so _pending_free buffers are no longer
        // referenced by uncommitted command buffers.
        // Note: clear_cache() is NOT thread-safe.  Callers (ReplicaPool::clear_cache,
        // unload_model) must ensure no concurrent GPU work or allocations.
        metal::commit_and_wait();

        // Release cached MPS GEMM objects (WEAK-1: prevents unbounded growth).
        metal::clear_gemm_cache();

        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& [sz, bufs] : _pool)
          for (id<MTLBuffer> buf : bufs)
            [buf release];
        _pool.clear();
        // After commit_and_wait(), pending_free buffers have already been
        // flushed to pool by flush_pending_frees().  Clear any stragglers.
        for (auto& entry : _pending_free)
          [entry.buffer release];
        _pending_free.clear();
      }

      // Return total bytes held in pool (not live — available for reuse).
      size_t pool_bytes() {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t total = 0;
        for (const auto& [sz, bufs] : _pool)
          total += sz * bufs.size();
        return total;
      }

      // Return total bytes in live allocations.
      size_t live_bytes() {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t total = 0;
        for (const auto& [ptr, entry] : _live)
          total += entry.requested_size;
        return total;
      }

    private:
      struct LiveEntry {
        size_t        requested_size;
        id<MTLBuffer> buffer;
        bool          gpu_protected = false;  // M11.18: deferred free
      };

      std::mutex                                               _mutex;
      std::map<const uint8_t*, LiveEntry>                      _live;   // sorted for O(log n) range lookup
      std::unordered_map<size_t, std::vector<id<MTLBuffer>>>   _pool;
      std::vector<LiveEntry>                                   _pending_free;
    };

  }  // namespace metal


  template<>
  Allocator& get_allocator<Device::METAL>() {
    static metal::MetalAllocator allocator;
    return allocator;
  }

  // Free function callable from primitives.mm (ObjC++ TU).
  // Returns the MTLBuffer that contains ptr and fills *offset_out with the
  // byte offset of ptr within that buffer.
  id<MTLBuffer> metal_buffer_for_ptr(const void* ptr, NSUInteger* offset_out) {
    return static_cast<metal::MetalAllocator&>(
        get_allocator<Device::METAL>())
        .buffer_for_ptr(ptr, offset_out);
  }

  namespace metal {
    void flush_pending_frees() {
      static_cast<MetalAllocator&>(
          get_allocator<Device::METAL>())
          .flush_pending_frees();
    }

    void protect_buffer(const void* ptr) {
      static_cast<MetalAllocator&>(
          get_allocator<Device::METAL>())
          .protect_buffer(ptr);
    }

    void protect_buffer_by_base(void* base_ptr) {
      static_cast<MetalAllocator&>(
          get_allocator<Device::METAL>())
          .protect_buffer_by_base(base_ptr);
    }

    size_t pool_bytes() {
      return static_cast<MetalAllocator&>(
          get_allocator<Device::METAL>())
          .pool_bytes();
    }

    size_t live_bytes() {
      return static_cast<MetalAllocator&>(
          get_allocator<Device::METAL>())
          .live_bytes();
    }
  }  // namespace metal

}  // namespace ctranslate2
