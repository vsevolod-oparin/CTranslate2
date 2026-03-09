#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

    class MetalAllocator : public Allocator {
    public:
      // Returns the MTLBuffer that contains ptr and sets *offset_out to the
      // byte offset of ptr within that buffer.  Needed by compute encoders
      // (which take id<MTLBuffer> + offset, not raw void*).
      //
      // Iterates _live to find the enclosing allocation.  O(n) where n is
      // the number of live allocations — typically a few dozen in inference.
      id<MTLBuffer> buffer_for_ptr(const void* ptr, NSUInteger* offset_out) {
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
        std::lock_guard<std::mutex> lock(_mutex);
        for (auto& [base, entry] : _live) {
          const uint8_t* base_ptr = static_cast<const uint8_t*>(base);
          if (byte_ptr >= base_ptr && byte_ptr < base_ptr + entry.requested_size) {
            if (offset_out) {
              *offset_out = static_cast<NSUInteger>(byte_ptr - base_ptr);
            }
            return entry.buffer;
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
          void* ptr = [buf contents];
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

        void* ptr = [buf contents];
        _live[ptr] = {size, buf};
        return ptr;
      }

      void free(void* ptr, int /*device_index*/) override {
        if (!ptr) {
          return;
        }
        std::lock_guard<std::mutex> lock(_mutex);

        auto live_it = _live.find(ptr);
        if (live_it == _live.end()) {
          throw std::runtime_error("Metal: attempt to free unknown pointer");
        }

        const size_t     sz  = live_it->second.requested_size;
        id<MTLBuffer>    buf = live_it->second.buffer;
        const bool  is_protected = live_it->second.gpu_protected;
        _live.erase(live_it);

        if (is_protected) {
          // M11.18: This buffer is referenced by a pending encode-only GPU
          // kernel (e.g. MPS padded GEMM row_copy-back).  Defer recycling
          // until commit_and_wait() completes all prior GPU work.
          _pending_free.push_back({sz, buf, false});
        } else {
          // Return to pool immediately (safe — not GPU-referenced).
          _pool[sz].push_back(buf);
        }
      }

      // Mark a live buffer as GPU-protected: its reclamation will be deferred
      // when free() is called, until after the next commit_and_wait().
      //
      // O(n) scan version — finds the enclosing allocation for any sub-pointer.
      // Prefer protect_buffer_by_base() when the base pointer is already known.
      void protect_buffer(const void* ptr) {
        std::lock_guard<std::mutex> lock(_mutex);
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(ptr);
        for (auto& [base, entry] : _live) {
          const uint8_t* base_ptr = static_cast<const uint8_t*>(base);
          if (byte_ptr >= base_ptr && byte_ptr < base_ptr + entry.requested_size) {
            entry.gpu_protected = true;
            return;
          }
        }
        // Not found in _live — might already be freed or not from this allocator.
        // Silently ignore (the buffer might be a temp or stack allocation).
      }

      // O(1) version — caller supplies the base pointer of the allocation
      // (e.g. from [MTLBuffer contents] after metal_buffer_for_ptr()).
      void protect_buffer_by_base(void* base_ptr) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _live.find(base_ptr);
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
        std::lock_guard<std::mutex> lock(_mutex);
        _pool.clear();  // ARC releases all pooled MTLBuffers.
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
      std::unordered_map<void*, LiveEntry>                     _live;
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
