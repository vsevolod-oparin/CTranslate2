// src/metal/primitives_transpose.mm
//
// M4.8 — Transpose primitives for Device::METAL.
//
// Algorithm: one GPU thread per output element.
//   - Decompose flat output index → multi-index via output strides.
//   - Map to input via permuted input strides: a_ps[k] = input_stride[perm[k]].
//   - All shapes and strides stored as uint32_t in argument structs passed via
//     setBytes: (no extra buffer allocation; no padding issues since all fields
//     are uint32_t, naturally aligned).
//
// Three kernel variants per element type: 2D (implicit [1,0] perm), 3D, 4D.

#include "metal/primitives_infra.h"

namespace {

// Argument structs — layout must exactly match the MSL struct definitions.

struct TransposeArgs2D {
  uint32_t rows, cols;
};

struct TransposeArgs3D {
  uint32_t a_ps0, a_ps1, a_ps2;  // permuted input strides
  uint32_t b_s0,  b_s1;          // output strides (b_s2 = 1, implicit)
  uint32_t bd1;                   // output dim[1] (for % decomposition)
};

struct TransposeArgs4D {
  uint32_t a_ps0, a_ps1, a_ps2, a_ps3;
  uint32_t b_s0,  b_s1,  b_s2;
  uint32_t bd1,   bd2;
};

static id<MTLLibrary> get_transpose_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kTransposeMSL, "transpose");
}

static id<MTLComputePipelineState> get_transpose_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_transpose_library, name);
}

// Shared dispatch: 2 data buffers + 1 args struct via setBytes:.
// n = total output elements.
static void dispatch_transpose(const char* kname,
                                const void* a, void* b, ctranslate2::dim_t n,
                                const void* args, size_t args_size) {
  if (n == 0) return;
  (void)ct2_u32(n);
  id<MTLComputePipelineState> pso = get_transpose_pso(kname);
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  NSUInteger off_a = 0, off_b = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(a, &off_a) offset:off_a atIndex:0];
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(b, &off_b) offset:off_b atIndex:1];
  [enc setBytes:args length:args_size atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(n));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

}  // anonymous namespace

namespace ctranslate2 {

  // transpose_2d: implicit perm = [1, 0].
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_2d(const T* a,
                                                const dim_t* dims,
                                                T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_2d_%s", MetalTypeName<T>::value);
    TransposeArgs2D args{ ct2_u32(dims[0]), ct2_u32(dims[1]) };
    dispatch_transpose(kname, a, b, dims[0] * dims[1], &args, sizeof(args));
  }

  // transpose_3d: arbitrary 3D permutation.
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_3d(const T* a,
                                                const dim_t* dims,
                                                const dim_t* perm,
                                                T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_3d_%s", MetalTypeName<T>::value);
    const uint32_t a_stride[3] = {
      ct2_u32(dims[1] * dims[2]),
      ct2_u32(dims[2]),
      1u
    };
    const uint32_t bd1 = ct2_u32(dims[perm[1]]);
    const uint32_t bd2 = ct2_u32(dims[perm[2]]);
    TransposeArgs3D args{
      a_stride[perm[0]], a_stride[perm[1]], a_stride[perm[2]],
      bd1 * bd2, bd2,   // b_s0, b_s1
      bd1
    };
    dispatch_transpose(kname, a, b, dims[0] * dims[1] * dims[2],
                       &args, sizeof(args));
  }

  // transpose_4d: arbitrary 4D permutation.
  template<>
  template <typename T>
  void primitives<Device::METAL>::transpose_4d(const T* a,
                                                const dim_t* dims,
                                                const dim_t* perm,
                                                T* b) {
    char kname[kKernelNameBufSize];
    std::snprintf(kname, sizeof(kname), "transpose_4d_%s", MetalTypeName<T>::value);
    const uint32_t a_stride[4] = {
      ct2_u32(dims[1] * dims[2] * dims[3]),
      ct2_u32(dims[2] * dims[3]),
      ct2_u32(dims[3]),
      1u
    };
    const uint32_t bd1 = ct2_u32(dims[perm[1]]);
    const uint32_t bd2 = ct2_u32(dims[perm[2]]);
    const uint32_t bd3 = ct2_u32(dims[perm[3]]);
    TransposeArgs4D args{
      a_stride[perm[0]], a_stride[perm[1]], a_stride[perm[2]], a_stride[perm[3]],
      bd1 * bd2 * bd3, bd2 * bd3, bd3,   // b_s0, b_s1, b_s2
      bd1, bd2
    };
    dispatch_transpose(kname, a, b,
                       dims[0] * dims[1] * dims[2] * dims[3],
                       &args, sizeof(args));
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

#define DECLARE_IMPL(T)                                                          \
  template void                                                                  \
  primitives<Device::METAL>::transpose_2d(const T* a,                           \
                                           const dim_t* dims,                   \
                                           T* b);                               \
  template void                                                                  \
  primitives<Device::METAL>::transpose_3d(const T* a,                           \
                                           const dim_t* dims,                   \
                                           const dim_t* perm,                   \
                                           T* b);                               \
  template void                                                                  \
  primitives<Device::METAL>::transpose_4d(const T* a,                           \
                                           const dim_t* dims,                   \
                                           const dim_t* perm,                   \
                                           T* b);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#undef DECLARE_IMPL

}  // namespace ctranslate2
