// src/metal/primitives_infra.h
//
// Internal header shared by all primitives_*.mm translation units.
// NOT for external consumption — include only from Metal .mm files.
//
// Provides:
//   - All required ObjC/Metal framework imports and C++ std headers
//   - compile_library_once / make_pso / PSOCache (lazy PSO infrastructure)
//   - ct2_u32            (checked dim_t → uint32_t narrowing)
//   - alloc_temp_buffer  (temporary MTLBuffer helper)
//   - MetalTypeName<T>   (C++ type → MSL type name string)
//   - kKernelNameBufSize (snprintf buffer size for kernel name formatting)
//   - METAL_STUB         (throw for unimplemented primitives)

#pragma once

// ---------------------------------------------------------------------------
// ObjC / Metal framework imports
// ---------------------------------------------------------------------------

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// ---------------------------------------------------------------------------
// C++ standard headers
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ---------------------------------------------------------------------------
// CTranslate2 headers
// ---------------------------------------------------------------------------

#include "ctranslate2/types.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/allocator.h"
#include "metal/utils.h"
#include "type_dispatch.h"
#include "metal/msl_strings.h"

// ---------------------------------------------------------------------------
// Lazy MSL library compilation (compile once per TU, thread-safe)
//
// `flag` and `lib_out` are static locals owned by the calling library getter.
// ---------------------------------------------------------------------------

static inline id<MTLLibrary> compile_library_once(std::once_flag& flag,
                                                    id<MTLLibrary>& lib_out,
                                                    const char* msl_src,
                                                    const char* label,
                                                    MTLCompileOptions* opts = nil) {
  std::call_once(flag, [&] {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:msl_src];
    lib_out = [ctranslate2::metal::get_metal_device()
        newLibraryWithSource:src options:opts error:&err];
    if (lib_out == nil) {
      std::string msg = std::string("Metal: failed to compile ") + label + " library";
      if (err)
        msg += std::string(": ") + [err.localizedDescription UTF8String];
      throw std::runtime_error(msg);
    }
  });
  return lib_out;
}

// ---------------------------------------------------------------------------
// PSO creation helper
// ---------------------------------------------------------------------------

static inline id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib,
                                                    const char* name) {
  NSString* nsname = [NSString stringWithUTF8String:name];
  id<MTLFunction> fn = [lib newFunctionWithName:nsname];
  if (fn == nil)
    throw std::runtime_error(std::string("Metal: kernel not found: ") + name);
  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [ctranslate2::metal::get_metal_device()
          newComputePipelineStateWithFunction:fn error:&err];
  if (pso == nil) {
    std::string msg = std::string("Metal: PSO creation failed for ") + name;
    if (err)
      msg += std::string(": ") + [err.localizedDescription UTF8String];
    throw std::runtime_error(msg);
  }
  return pso;
}

// ---------------------------------------------------------------------------
// Thread-safe PSO cache
//
// Each kernel group owns one static PSOCache instance.
// Keys are kernel function names; populated lazily on first use.
// ---------------------------------------------------------------------------

struct PSOCache {
  std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  std::mutex mtx;

  template <typename LibFn>
  id<MTLComputePipelineState> get(LibFn lib_fn, const char* name) {
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(name);
    if (it != cache.end()) {
      ctranslate2::metal::increment_pso_hits();
      return it->second;
    }
    ctranslate2::metal::increment_pso_misses();
    return cache[name] = make_pso(lib_fn(), name);
  }
};

// ---------------------------------------------------------------------------
// Checked narrowing: dim_t (int64_t) → uint32_t
//
// Used wherever a GPU kernel argument is declared `uint` in MSL.
// Throws rather than silently truncating values above 2^32-1.
// ---------------------------------------------------------------------------

static inline uint32_t ct2_u32(ctranslate2::dim_t v) {
  constexpr ctranslate2::dim_t kMax =
      static_cast<ctranslate2::dim_t>(std::numeric_limits<uint32_t>::max());
  if (v < 0 || v > kMax)
    throw std::runtime_error(
        "Metal: dimension value " + std::to_string(v) +
        " overflows uint32_t (Metal kernel argument limit)");
  return static_cast<uint32_t>(v);
}

// ---------------------------------------------------------------------------
// Temporary MTLBuffer allocation (Shared mode, ARC-managed)
// ---------------------------------------------------------------------------

static inline id<MTLBuffer> alloc_temp_buffer(NSUInteger bytes) {
  id<MTLBuffer> buf = [ctranslate2::metal::get_metal_device()
      newBufferWithLength:bytes
                 options:MTLResourceStorageModeShared];
  if (buf == nil)
    throw std::runtime_error("Metal: failed to allocate temporary buffer");
  return buf;
}

// ---------------------------------------------------------------------------
// RAII allocator-registered temporary buffer.
//
// Buffers allocated via get_allocator<Device::METAL>() are tracked in
// MetalAllocator::_live, so metal_buffer_for_ptr() can find them.
// Use this (instead of alloc_temp_buffer) when GEMM or other ops need to
// locate the backing MTLBuffer via metal_buffer_for_ptr().
// ---------------------------------------------------------------------------

struct MetalTempBuf {
  void* ptr = nullptr;

  MetalTempBuf() = default;

  explicit MetalTempBuf(size_t n_bytes) {
    ptr = ctranslate2::get_allocator<ctranslate2::Device::METAL>().allocate(n_bytes, 0);
  }

  ~MetalTempBuf() {
    if (ptr) {
      ctranslate2::get_allocator<ctranslate2::Device::METAL>().free(ptr, 0);
    }
  }

  MetalTempBuf(const MetalTempBuf&) = delete;
  MetalTempBuf& operator=(const MetalTempBuf&) = delete;

  MetalTempBuf(MetalTempBuf&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  MetalTempBuf& operator=(MetalTempBuf&& o) noexcept {
    if (this != &o) {
      if (ptr) ctranslate2::get_allocator<ctranslate2::Device::METAL>().free(ptr, 0);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  template <typename T>
  T* as() { return static_cast<T*>(ptr); }
};

// ---------------------------------------------------------------------------
// MetalTypeName<T> — maps C++ scalar type to its MSL type name string
// ---------------------------------------------------------------------------

template <typename T> struct MetalTypeName;
template<> struct MetalTypeName<float>                              { static constexpr const char* value = "float";  };
template<> struct MetalTypeName<ctranslate2::float16_t>             { static constexpr const char* value = "half";   };
template<> struct MetalTypeName<ctranslate2::bfloat16_t>            { static constexpr const char* value = "bfloat"; };
template<> struct MetalTypeName<int8_t>                             { static constexpr const char* value = "char";   };
template<> struct MetalTypeName<int16_t>                            { static constexpr const char* value = "short";  };
template<> struct MetalTypeName<int32_t>                            { static constexpr const char* value = "int";    };

// Generous upper bound for any formatted kernel name (e.g. "mul_scalar_bfloat").
static constexpr size_t kKernelNameBufSize = 64;

// ---------------------------------------------------------------------------
// METAL_STUB — throw for unimplemented primitives<Device::METAL> methods
// ---------------------------------------------------------------------------

#define METAL_STUB(name) \
  throw std::runtime_error("primitives<METAL>::" #name ": not yet implemented")
