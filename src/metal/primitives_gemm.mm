// src/metal/primitives_gemm.mm
//
// M4.4 — GEMM primitives for Device::METAL.
//
// Three paths depending on element type:
//
//   Path A (FP32 / FP16): MPSMatrixMultiplication.
//     Encodes into the per-thread command buffer (encode-only).
//     When the natural rowBytes is below the MPS hardware minimum (small
//     matrices), inputs are copied to row-padded temporary buffers.
//
//   Path B (BF16): MPSGraph matrixMultiplicationWithPrimaryTensor.
//     MPSMatrixMultiplication asserts at runtime for BF16 (verified in M0.2).
//     MPSGraph runs synchronously on its own queue — pending GPU work is
//     flushed with commit_and_wait() before each graph execution.
//
//   Path C (INT8 → INT32): dequantize INT8 inputs to FP32 on CPU, run FP32
//     MPS GEMM, then round FP32 → INT32 on CPU.  Metal MPS has no native
//     INT8 matmul as of macOS 14.  FP32 represents the accumulator exactly
//     for k ≤ ~1040 (max int8 product 127*127*k < 2^24 = 16,777,216).
//
// Critical: get_current_command_buffer() must be called OUTSIDE
// @autoreleasepool{} to avoid use-after-free of the thread-local CB.

#include "metal/primitives_infra.h"

#include <Accelerate/Accelerate.h>

namespace {

// ---------------------------------------------------------------------------
// Vectorized int8 → float32 conversion using vDSP (Accelerate framework)
// ---------------------------------------------------------------------------
static inline void int8_to_float32(float* dst, const int8_t* src, NSUInteger count) {
  vDSP_vflt8(reinterpret_cast<const char*>(src), 1, dst, 1, (vDSP_Length)count);
}

// ---------------------------------------------------------------------------
// MPS data type mapping (FP32 and FP16 only — BF16 uses MPSGraph)
// ---------------------------------------------------------------------------

template <typename T> struct MPS_Dtype;
template<> struct MPS_Dtype<float>                  { static const MPSDataType value = MPSDataTypeFloat32; };
template<> struct MPS_Dtype<ctranslate2::float16_t> { static const MPSDataType value = MPSDataTypeFloat16; };

// ---------------------------------------------------------------------------
// Path A — FP32 / FP16 GEMM via MPSMatrixMultiplication
//
// Physical layout of A in memory:
//   !transpose_a → rows = m, cols = k, rowBytes = lda * sizeof(T)
//    transpose_a → rows = k, cols = m, rowBytes = lda * sizeof(T)
// (Same logic for B.)
//
// MPSMatrix requires rowBytes >= [MPSMatrixDescriptor rowBytesForColumns:...].
// When the natural stride is below this minimum, we copy to a row-padded
// temporary buffer.  For the output we encode a GPU row_copy back to c.
// ---------------------------------------------------------------------------

// Forward declaration — defined after row_copy MSL infrastructure.
static void dispatch_row_copy(id<MTLBuffer> src_buf, NSUInteger src_off,
                              NSUInteger src_rb, NSUInteger src_mb,
                              id<MTLBuffer> dst_buf, NSUInteger dst_off,
                              NSUInteger dst_rb, NSUInteger dst_mb,
                              NSUInteger copy_bytes, NSUInteger rows,
                              NSUInteger batch_size);

template <typename T>
static void dispatch_mps_gemm(bool transpose_a, bool transpose_b,
                               ctranslate2::dim_t m,
                               ctranslate2::dim_t n,
                               ctranslate2::dim_t k,
                               float alpha,
                               const T* a, ctranslate2::dim_t lda,
                               const T* b, ctranslate2::dim_t ldb,
                               float beta,
                               T* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;

  constexpr NSUInteger elem = sizeof(T);
  const MPSDataType dtype = MPS_Dtype<T>::value;

  const NSUInteger rows_a = transpose_a ? (NSUInteger)k : (NSUInteger)m;
  const NSUInteger cols_a = transpose_a ? (NSUInteger)m : (NSUInteger)k;
  const NSUInteger rows_b = transpose_b ? (NSUInteger)n : (NSUInteger)k;
  const NSUInteger cols_b = transpose_b ? (NSUInteger)k : (NSUInteger)n;
  const NSUInteger rows_c = (NSUInteger)m;
  const NSUInteger cols_c = (NSUInteger)n;

  const NSUInteger nat_rb_a = (NSUInteger)lda * elem;
  const NSUInteger nat_rb_b = (NSUInteger)ldb * elem;
  const NSUInteger nat_rb_c = (NSUInteger)ldc * elem;

  // Query MPS minimum rowBytes inside an autorelease pool (may allocate ObjC temps).
  NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
  @autoreleasepool {
    mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a dataType:dtype];
    mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b dataType:dtype];
    mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:cols_c dataType:dtype];
  }

  const bool pad_a = (nat_rb_a < mps_rb_a);
  const bool pad_b = (nat_rb_b < mps_rb_b);
  const bool pad_c = (nat_rb_c < mps_rb_c);

  // -----------------------------------------------------------------------
  // CPU GEMM fast-path for tiny padded matrices (e.g. 3×3 attention scores).
  // For these, cblas is faster than MPS kernel launch + sync overhead.
  //
  // NOTE: This threshold is also a **correctness boundary** for float16.
  // MPS batched GEMM produces incorrect results for float16 with m=1 and
  // small n (verified in M11.18).  For float16, m=1 is intercepted earlier
  // by the custom GEMV kernel (M11.19), but this threshold remains as a
  // safety net for any other tiny float16 padded GEMMs.
  // -----------------------------------------------------------------------
  constexpr NSUInteger kCpuGemmThresh = 4096;
  if ((pad_a || pad_b || pad_c) && (rows_c * cols_c <= kCpuGemmThresh)) {
    CT2_COMMIT_AND_WAIT();

    if constexpr (std::is_same_v<T, float>) {
      cblas_sgemm(CblasRowMajor,
                  transpose_a ? CblasTrans : CblasNoTrans,
                  transpose_b ? CblasTrans : CblasNoTrans,
                  (int)m, (int)n, (int)k,
                  alpha, a, (int)lda, b, (int)ldb,
                  beta, c, (int)ldc);
    } else {
      // float16: widen to float32, cblas_sgemm, narrow back.
      constexpr NSUInteger kStackThresh = 4096;
      float sa[kStackThresh], sb[kStackThresh], sc[kStackThresh];
      const NSUInteger elems_a = rows_a * cols_a;
      const NSUInteger elems_b = rows_b * cols_b;
      const NSUInteger elems_c = rows_c * cols_c;
      float* fa = (elems_a <= kStackThresh) ? sa : new float[elems_a];
      float* fb = (elems_b <= kStackThresh) ? sb : new float[elems_b];
      float* fc = (elems_c <= kStackThresh) ? sc : new float[elems_c];

      for (NSUInteger r = 0; r < rows_a; ++r)
        for (NSUInteger ci = 0; ci < cols_a; ++ci)
          fa[r * cols_a + ci] = float(a[r * (NSUInteger)lda + ci]);
      for (NSUInteger r = 0; r < rows_b; ++r)
        for (NSUInteger ci = 0; ci < cols_b; ++ci)
          fb[r * cols_b + ci] = float(b[r * (NSUInteger)ldb + ci]);
      if (beta != 0.0f) {
        for (NSUInteger r = 0; r < rows_c; ++r)
          for (NSUInteger ci = 0; ci < cols_c; ++ci)
            fc[r * cols_c + ci] = float(c[r * (NSUInteger)ldc + ci]);
      }

      cblas_sgemm(CblasRowMajor,
                  transpose_a ? CblasTrans : CblasNoTrans,
                  transpose_b ? CblasTrans : CblasNoTrans,
                  (int)m, (int)n, (int)k,
                  alpha, fa, (int)cols_a, fb, (int)cols_b,
                  beta, fc, (int)cols_c);

      for (NSUInteger r = 0; r < rows_c; ++r)
        for (NSUInteger ci = 0; ci < cols_c; ++ci)
          c[r * (NSUInteger)ldc + ci] = T(fc[r * cols_c + ci]);

      if (fa != sa) delete[] fa;
      if (fb != sb) delete[] fb;
      if (fc != sc) delete[] fc;
    }
    return;
  }

  // -----------------------------------------------------------------------
  // MPS GEMM path — handles non-padded and large padded matrices.
  // For large matrices with alignment padding (e.g. logits n=51865),
  // MPS is faster than cblas. Uses temp buffers when row stride < MPS minimum.
  // -----------------------------------------------------------------------

  // Flush pending GPU work if we need to CPU-read A/B for padding,
  // or C for beta != 0 padding.
  if (pad_a || pad_b || (pad_c && beta != 0.0f))
    CT2_COMMIT_AND_WAIT();

  id<MTLBuffer> buf_a = nil, buf_b = nil, buf_c = nil;
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> tmp_a = nil, tmp_b = nil, tmp_c = nil;

  if (pad_a) {
    tmp_a = alloc_temp_buffer(rows_a * mps_rb_a);
    auto* dst = static_cast<uint8_t*>([tmp_a contents]);
    auto* src = reinterpret_cast<const uint8_t*>(a);
    for (NSUInteger r = 0; r < rows_a; ++r)
      std::memcpy(dst + r * mps_rb_a, src + r * nat_rb_a, nat_rb_a);
    buf_a = tmp_a;
  } else {
    buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  }

  if (pad_b) {
    tmp_b = alloc_temp_buffer(rows_b * mps_rb_b);
    auto* dst = static_cast<uint8_t*>([tmp_b contents]);
    auto* src = reinterpret_cast<const uint8_t*>(b);
    for (NSUInteger r = 0; r < rows_b; ++r)
      std::memcpy(dst + r * mps_rb_b, src + r * nat_rb_b, nat_rb_b);
    buf_b = tmp_b;
  } else {
    buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  }

  if (pad_c) {
    tmp_c = alloc_temp_buffer(rows_c * mps_rb_c);
    auto* dst = static_cast<uint8_t*>([tmp_c contents]);
    if (beta != 0.0f) {
      auto* src = reinterpret_cast<const uint8_t*>(c);
      for (NSUInteger r = 0; r < rows_c; ++r)
        std::memcpy(dst + r * mps_rb_c, src + r * nat_rb_c, nat_rb_c);
    } else {
      std::memset(dst, 0, rows_c * mps_rb_c);
    }
    buf_c = tmp_c;
  } else {
    buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  }

  const NSUInteger rb_a = pad_a ? mps_rb_a : nat_rb_a;
  const NSUInteger rb_b = pad_b ? mps_rb_b : nat_rb_b;
  const NSUInteger rb_c = pad_c ? mps_rb_c : nat_rb_c;

  // Fetch the command buffer BEFORE @autoreleasepool to avoid use-after-free.
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();

  @autoreleasepool {
    MPSMatrixDescriptor* descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                              columns:cols_a
                                             rowBytes:rb_a
                                             dataType:dtype];
    MPSMatrixDescriptor* descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                              columns:cols_b
                                             rowBytes:rb_b
                                             dataType:dtype];
    MPSMatrixDescriptor* descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_c
                                              columns:cols_c
                                             rowBytes:rb_c
                                             dataType:dtype];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:off_a descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:off_b descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:off_c descriptor:descC];

    id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
    MPSMatrixMultiplication* gemm_op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:(BOOL)transpose_a
                                          transposeRight:(BOOL)transpose_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:(double)beta];

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
    [gemm_op release];
  }

  // GPU unpack: copy rows from padded tmp_c back to tightly-packed C.
  if (pad_c) {
    NSUInteger dst_off_c = 0;
    id<MTLBuffer> dst_buf = ctranslate2::metal_buffer_for_ptr(c, &dst_off_c);
    dispatch_row_copy(tmp_c, 0,
                      mps_rb_c, rows_c * mps_rb_c,
                      dst_buf, dst_off_c,
                      nat_rb_c, rows_c * nat_rb_c,
                      nat_rb_c, rows_c, 1);
  }

  // Release temp buffers (command buffer retains them until GPU completes).
  if (tmp_a) [tmp_a release];
  if (tmp_b) [tmp_b release];
  if (tmp_c) [tmp_c release];
}

// ---------------------------------------------------------------------------
// Path B — BF16 GEMM via MPSGraph
//
// Four graph instances cached by (trans_a × trans_b).
// nil-shape placeholders allow MPSGraph to JIT-compile per shape on first use.
// The transpose is baked as a fused transposeTensor node.
//
// Thread safety: MPSGraph is not thread-safe.  CTranslate2 runs single-threaded
// per translator, so concurrent use is not expected.
// ---------------------------------------------------------------------------

struct Bf16GemmEntry {
  MPSGraph*       graph;
  MPSGraphTensor* ph_a;    // placeholder for A (physical layout, before transpose)
  MPSGraphTensor* ph_b;    // placeholder for B (physical layout, before transpose)
  MPSGraphTensor* result;  // output of the matmul node
};

static Bf16GemmEntry& get_bf16_graph(bool trans_a, bool trans_b) {
  static Bf16GemmEntry entries[4];
  static bool initialized[4] = {};
  static std::mutex mtx;

  const int idx = (trans_a ? 2 : 0) | (trans_b ? 1 : 0);
  std::lock_guard<std::mutex> lk(mtx);
  if (!initialized[idx]) {
    MPSGraph* g = [[MPSGraph alloc] init];
    MPSGraphTensor* pA = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"A"];
    MPSGraphTensor* pB = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"B"];
    MPSGraphTensor* opA = trans_a
        ? [g transposeTensor:pA dimension:0 withDimension:1 name:@"AT"] : pA;
    MPSGraphTensor* opB = trans_b
        ? [g transposeTensor:pB dimension:0 withDimension:1 name:@"BT"] : pB;
    MPSGraphTensor* tC = [g matrixMultiplicationWithPrimaryTensor:opA
                                                  secondaryTensor:opB
                                                             name:@"C"];
    entries[idx] = { g, pA, pB, tC };
    initialized[idx] = true;
  }
  return entries[idx];
}

// Inner BF16 GEMM — synchronous MPSGraph execution.
// Only contiguous matrices supported (lda/ldb/ldc = natural column count);
// the ops layer always passes contiguous matrices so this is not a limitation.
static void run_bf16_gemm_inner(bool trans_a, bool trans_b,
                                ctranslate2::dim_t m,
                                ctranslate2::dim_t n,
                                ctranslate2::dim_t k,
                                const ctranslate2::bfloat16_t* a, ctranslate2::dim_t lda,
                                const ctranslate2::bfloat16_t* b, ctranslate2::dim_t ldb,
                                ctranslate2::bfloat16_t* c, ctranslate2::dim_t ldc) {
  const ctranslate2::dim_t exp_lda = trans_a ? m : k;
  const ctranslate2::dim_t exp_ldb = trans_b ? k : n;
  if (lda != exp_lda || ldb != exp_ldb || ldc != n) {
    throw std::runtime_error(
        "Metal BF16 GEMM: only contiguous matrices are supported "
        "(lda/ldb/ldc must equal the physical column count)");
  }

  const Bf16GemmEntry& entry = get_bf16_graph(trans_a, trans_b);
  id<MTLCommandQueue> queue = ctranslate2::metal::get_metal_command_queue();

  NSArray<NSNumber*>* shape_a =
      trans_a ? @[@((int)k), @((int)m)] : @[@((int)m), @((int)k)];
  NSArray<NSNumber*>* shape_b =
      trans_b ? @[@((int)n), @((int)k)] : @[@((int)k), @((int)n)];

  const size_t bytes_a = (size_t)m * k * sizeof(ctranslate2::bfloat16_t);
  const size_t bytes_b = (size_t)k * n * sizeof(ctranslate2::bfloat16_t);

  // MPSGraphTensorData has no byte-offset parameter.  Copy to zero-offset
  // temp when the input has a non-zero offset within its MTLBuffer (occurs
  // for non-first batch elements).
  NSUInteger off_a = 0, off_b = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

  id<MTLBuffer> tmp_a = nil, tmp_b = nil;
  if (off_a != 0) {
    tmp_a = alloc_temp_buffer(bytes_a);
    std::memcpy([tmp_a contents], a, bytes_a);
    buf_a = tmp_a;
  }
  if (off_b != 0) {
    tmp_b = alloc_temp_buffer(bytes_b);
    std::memcpy([tmp_b contents], b, bytes_b);
    buf_b = tmp_b;
  }

  @autoreleasepool {
    MPSGraphTensorData* tdA = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buf_a shape:shape_a dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData* tdB = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buf_b shape:shape_b dataType:MPSDataTypeBFloat16];

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [entry.graph runWithMTLCommandQueue:queue
                                      feeds:@{entry.ph_a: tdA, entry.ph_b: tdB}
                              targetTensors:@[entry.result]
                           targetOperations:nil];

    if (!results || !results[entry.result])
      throw std::runtime_error("Metal BF16 GEMM: graph execution returned nil");

    // c is a Shared MTLBuffer contents pointer — valid as CPU destination.
    [[results[entry.result] mpsndarray] readBytes:c strideBytes:nil];
    [tdA release];
    [tdB release];
  }

  // Release temp buffers (MPSGraph ran synchronously, GPU is done).
  if (tmp_a) [tmp_a release];
  if (tmp_b) [tmp_b release];
}

// Flush pending GPU work and run one BF16 GEMM.
// MPSGraph BF16 matmul does not support alpha/beta natively.
// Strategy: C = A*B, then C *= alpha if alpha != 1.  beta must be 0.
static void dispatch_bf16_gemm(bool trans_a, bool trans_b,
                                ctranslate2::dim_t m,
                                ctranslate2::dim_t n,
                                ctranslate2::dim_t k,
                                float alpha, float beta,
                                const ctranslate2::bfloat16_t* a, ctranslate2::dim_t lda,
                                const ctranslate2::bfloat16_t* b, ctranslate2::dim_t ldb,
                                ctranslate2::bfloat16_t* c, ctranslate2::dim_t ldc) {
  if (beta != 0.0f)
    throw std::runtime_error(
        "Metal BF16 GEMM: only beta=0.0 is supported");
  if (m == 0 || n == 0 || k == 0) return;
  // MPSGraph uses its own queue; flush the deferred CB first.
  CT2_COMMIT_AND_WAIT();
  run_bf16_gemm_inner(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc);
  // Apply alpha scaling post-GEMM if needed.
  if (alpha != 1.0f) {
    const ctranslate2::dim_t size = m * n;
    ctranslate2::primitives<ctranslate2::Device::METAL>::mul(
        static_cast<ctranslate2::bfloat16_t>(alpha), c, c, size);
  }
}

// ---------------------------------------------------------------------------
// Path C — INT8 → INT32 GEMM
//
// Low-level MPS GEMM helper: takes id<MTLBuffer> arguments directly.
// Used by dispatch_int8_gemm because temp buffers from alloc_temp_buffer
// are not registered in the MetalAllocator's live map (so metal_buffer_for_ptr
// would throw); we pass the MTLBuffers directly instead.
// ---------------------------------------------------------------------------

static void dispatch_mps_gemm_buf(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    id<MTLBuffer> buf_a, NSUInteger off_a, NSUInteger rb_a,
    NSUInteger rows_a, NSUInteger cols_a,
    id<MTLBuffer> buf_b, NSUInteger off_b, NSUInteger rb_b,
    NSUInteger rows_b, NSUInteger cols_b,
    id<MTLBuffer> buf_c, NSUInteger off_c, NSUInteger rb_c,
    MPSDataType dtype) {
  // Fetch command buffer BEFORE @autoreleasepool to avoid use-after-free.
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  @autoreleasepool {
    MPSMatrixDescriptor* descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                              columns:cols_a
                                             rowBytes:rb_a
                                             dataType:dtype];
    MPSMatrixDescriptor* descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                              columns:cols_b
                                             rowBytes:rb_b
                                             dataType:dtype];
    MPSMatrixDescriptor* descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                              columns:(NSUInteger)n
                                             rowBytes:rb_c
                                             dataType:dtype];
    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:off_a descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:off_b descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:off_c descriptor:descC];

    id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
    MPSMatrixMultiplication* gemm_op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:(BOOL)transpose_a
                                          transposeRight:(BOOL)transpose_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:0.0];
    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
    [gemm_op release];
  }
}

// INT8 GEMM: convert int8 A and B to float32, run float32 MPS GEMM,
// round float32 result to int32.  Only beta=0 is supported.
static void dispatch_int8_gemm(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const int8_t* a, ctranslate2::dim_t lda,
    const int8_t* b, ctranslate2::dim_t ldb,
    float beta,
    int32_t* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;
  if (beta != 0.0f)
    throw std::runtime_error("Metal INT8 GEMM: only beta=0 is supported");

  // Physical layout of A and B in memory:
  //   !transpose_a → rows_a = m, cols_a = k
  //    transpose_a → rows_a = k, cols_a = m
  // (Same for B with transpose_b, n, k.)
  const NSUInteger rows_a = (NSUInteger)(transpose_a ? k : m);
  const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
  const NSUInteger rows_b = (NSUInteger)(transpose_b ? n : k);
  const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);

  // Flush pending GPU work before CPU reads from a/b.
  CT2_COMMIT_AND_WAIT();

  // Query MPS minimum rowBytes for float32.
  NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
  @autoreleasepool {
    mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a
                                              dataType:MPSDataTypeFloat32];
    mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b
                                              dataType:MPSDataTypeFloat32];
    mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:(NSUInteger)n
                                              dataType:MPSDataTypeFloat32];
  }

  // Use padded row bytes to satisfy MPS alignment requirements.
  const NSUInteger rb_a = std::max((NSUInteger)lda * sizeof(float), mps_rb_a);
  const NSUInteger rb_b = std::max((NSUInteger)ldb * sizeof(float), mps_rb_b);
  const NSUInteger rb_c = std::max((NSUInteger)n   * sizeof(float), mps_rb_c);

  // Allocate float32 temporary buffers (Shared mode, CPU+GPU coherent).
  id<MTLBuffer> tmp_a = alloc_temp_buffer(rows_a * rb_a);
  id<MTLBuffer> tmp_b = alloc_temp_buffer(rows_b * rb_b);
  id<MTLBuffer> tmp_c = alloc_temp_buffer((NSUInteger)m * rb_c);

  // CPU: convert int8 → float32 row by row, respecting lda / ldb strides.
  for (NSUInteger r = 0; r < rows_a; ++r) {
    float*        dst = reinterpret_cast<float*>(
                      static_cast<uint8_t*>([tmp_a contents]) + r * rb_a);
    const int8_t* src = a + r * (NSUInteger)lda;
    int8_to_float32(dst, src, cols_a);
  }
  for (NSUInteger r = 0; r < rows_b; ++r) {
    float*        dst = reinterpret_cast<float*>(
                      static_cast<uint8_t*>([tmp_b contents]) + r * rb_b);
    const int8_t* src = b + r * (NSUInteger)ldb;
    int8_to_float32(dst, src, cols_b);
  }
  std::memset([tmp_c contents], 0, (NSUInteger)m * rb_c);

  // Encode float32 MPS GEMM using temp buffers directly.
  dispatch_mps_gemm_buf(
      transpose_a, transpose_b, m, n, k, alpha,
      tmp_a, 0, rb_a, rows_a, cols_a,
      tmp_b, 0, rb_b, rows_b, cols_b,
      tmp_c, 0, rb_c, MPSDataTypeFloat32);

  // Wait for GPU, then round float32 → int32 (handles ldc stride).
  CT2_COMMIT_AND_WAIT();
  for (ctranslate2::dim_t row = 0; row < m; ++row) {
    const float* src = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>([tmp_c contents]) + (NSUInteger)row * rb_c);
    int32_t* dst = c + row * ldc;
    for (ctranslate2::dim_t col = 0; col < n; ++col)
      dst[col] = static_cast<int32_t>(std::lroundf(src[col]));
  }

  // Release temp buffers (GPU completed, CPU readback done).
  [tmp_a release];
  [tmp_b release];
  [tmp_c release];
}

// ---------------------------------------------------------------------------
// Batched MPS GEMM for contiguous strided layouts.
//
// Uses MPS batch matrix descriptors (matrixDescriptorWithRows:...matrices:...)
// to encode all batch elements in a single MPSMatrixMultiplication call,
// eliminating per-element ObjC allocations and enabling MPS-internal
// GPU scheduling optimizations.
//
// Falls back to the per-element loop when MPS batch constraints are not met
// (matrixBytes must be a multiple of rowBytes and >= rows * rowBytes).
// ---------------------------------------------------------------------------

template <typename T>
static bool dispatch_mps_gemm_batched(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const T* a, ctranslate2::dim_t lda, ctranslate2::dim_t stridea,
    const T* b, ctranslate2::dim_t ldb, ctranslate2::dim_t strideb,
    float beta,
    T* c, ctranslate2::dim_t ldc, ctranslate2::dim_t stridec,
    ctranslate2::dim_t batch_size) {
  if (batch_size <= 0 || m == 0 || n == 0 || k == 0) return true;

  constexpr NSUInteger elem = sizeof(T);
  const MPSDataType dtype = MPS_Dtype<T>::value;

  const NSUInteger rows_a = transpose_a ? (NSUInteger)k : (NSUInteger)m;
  const NSUInteger cols_a = transpose_a ? (NSUInteger)m : (NSUInteger)k;
  const NSUInteger rows_b = transpose_b ? (NSUInteger)n : (NSUInteger)k;
  const NSUInteger cols_b = transpose_b ? (NSUInteger)k : (NSUInteger)n;

  const NSUInteger rb_a = (NSUInteger)lda * elem;
  const NSUInteger rb_b = (NSUInteger)ldb * elem;
  const NSUInteger rb_c = (NSUInteger)ldc * elem;

  const NSUInteger mb_a = (NSUInteger)stridea * elem;
  const NSUInteger mb_b = (NSUInteger)strideb * elem;
  const NSUInteger mb_c = (NSUInteger)stridec * elem;

  // MPS batch API requires:
  //   matrixBytes % rowBytes == 0
  //   matrixBytes >= rows * rowBytes
  if (mb_a % rb_a != 0 || mb_a < rows_a * rb_a ||
      mb_b % rb_b != 0 || mb_b < rows_b * rb_b ||
      mb_c % rb_c != 0 || mb_c < (NSUInteger)m * rb_c)
    return false;  // fall back to per-element loop

  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);

  // Fetch the command buffer BEFORE @autoreleasepool to avoid use-after-free.
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();

  @autoreleasepool {
    MPSMatrixDescriptor* descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                              columns:cols_a
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:rb_a
                                          matrixBytes:mb_a
                                             dataType:dtype];
    MPSMatrixDescriptor* descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                              columns:cols_b
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:rb_b
                                          matrixBytes:mb_b
                                             dataType:dtype];
    MPSMatrixDescriptor* descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                              columns:(NSUInteger)n
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:rb_c
                                          matrixBytes:mb_c
                                             dataType:dtype];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:off_a descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:off_b descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:off_c descriptor:descC];

    id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
    MPSMatrixMultiplication* gemm_op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:(BOOL)transpose_a
                                          transposeRight:(BOOL)transpose_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:(double)beta];
    gemm_op.batchSize = (NSUInteger)batch_size;
    gemm_op.batchStart = 0;

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
    [gemm_op release];
  }

  return true;  // successfully dispatched
}

// ---------------------------------------------------------------------------
// Batched MPS GEMM with padded temp buffers and GPU row_copy kernel.
//
// Like dispatch_mps_gemm_batched but handles cases where natural row bytes
// are below MPS alignment requirements.  Uses an MSL compute kernel to
// copy rows between tightly-packed and MPS-padded layouts — completely
// encode-only, zero commit_and_wait() calls.
//
// For pad_a/pad_b: GPU row_copy from src to padded temp (encode-only)
// For pad_c:       MPS GEMM writes to padded temp, GPU row_copy back
//                  (encode-only, zero syncs)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU row-copy kernel: copies rows between layouts with different row strides.
// One dispatch replaces thousands of individual blit commands.
// Grid: [total_rows_across_all_batches, 1, 1], threads per group: [256, 1, 1]
// Each thread copies a strided chunk of one row (256 threads × uint4 = 4KB/iter).
// ---------------------------------------------------------------------------
static const char* kRowCopyMSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void row_copy(
    device const char* src  [[buffer(0)]],
    device       char* dst  [[buffer(1)]],
    constant     uint& copy_bytes   [[buffer(2)]],  // bytes to copy per row
    constant     uint& src_rb       [[buffer(3)]],  // source row stride (bytes)
    constant     uint& dst_rb       [[buffer(4)]],  // dest row stride (bytes)
    constant     uint& rows_per_mat [[buffer(5)]],  // rows per batch element
    constant     uint& src_mb       [[buffer(6)]],  // source matrixBytes
    constant     uint& dst_mb       [[buffer(7)]],  // dest matrixBytes
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    // gid = global row index across all batches
    uint batch = gid / rows_per_mat;
    uint row   = gid % rows_per_mat;
    uint s_off = batch * src_mb + row * src_rb;
    uint d_off = batch * dst_mb + row * dst_rb;
    // Each thread copies a strided chunk of the row.
    // Use uint (4-byte) copies for alignment.
    device const uint* s = (device const uint*)(src + s_off);
    device       uint* d = (device       uint*)(dst + d_off);
    uint n_uint = copy_bytes / 4;
    for (uint i = tid; i < n_uint; i += tgs)
        d[i] = s[i];
    // Handle remainder bytes (< 4).
    if (tid == 0) {
        uint rem_start = n_uint * 4;
        for (uint i = rem_start; i < copy_bytes; ++i)
            dst[d_off + i] = src[s_off + i];
    }
}
)";

static id<MTLLibrary> get_row_copy_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kRowCopyMSL, "row_copy");
}

static id<MTLComputePipelineState> get_row_copy_pso() {
  static PSOCache cache;
  return cache.get(get_row_copy_library, "row_copy");
}

// GPU row-copy dispatch: encode-only, no sync.
// Copies rows between different row strides for pad/unpad operations.
static void dispatch_row_copy(id<MTLBuffer> src_buf, NSUInteger src_off,
                               NSUInteger src_rb, NSUInteger src_mb,
                               id<MTLBuffer> dst_buf, NSUInteger dst_off,
                               NSUInteger dst_rb, NSUInteger dst_mb,
                               NSUInteger copy_bytes, NSUInteger rows,
                               NSUInteger batch_size) {
  const NSUInteger total_rows = rows * batch_size;
  if (total_rows == 0 || copy_bytes == 0) return;

  uint32_t copy_u32   = static_cast<uint32_t>(copy_bytes);
  uint32_t src_rb_u32 = static_cast<uint32_t>(src_rb);
  uint32_t dst_rb_u32 = static_cast<uint32_t>(dst_rb);
  uint32_t rows_u32   = static_cast<uint32_t>(rows);
  uint32_t src_mb_u32 = static_cast<uint32_t>(src_mb);
  uint32_t dst_mb_u32 = static_cast<uint32_t>(dst_mb);

  id<MTLComputePipelineState> pso = get_row_copy_pso();
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_off atIndex:0];
  [enc setBuffer:dst_buf offset:dst_off atIndex:1];
  [enc setBytes:&copy_u32   length:sizeof(uint32_t) atIndex:2];
  [enc setBytes:&src_rb_u32 length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&dst_rb_u32 length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&rows_u32   length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&src_mb_u32 length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&dst_mb_u32 length:sizeof(uint32_t) atIndex:7];

  NSUInteger threads_per_group = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreadgroups:MTLSizeMake(total_rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// ---------------------------------------------------------------------------
// M11.19: Custom float16 GEMV kernel for m=1 decode attention GEMMs.
//
// MPS batched GEMM produces garbled output for float16 m=1 (verified M11.18).
// This custom MSL kernel avoids MPS entirely.  Encode-only — zero syncs.
// Uses protect_buffer (M11.18) to prevent buffer reuse before GPU execution.
//
// Grid: [batch_size, 1, 1], threads per group: [256, 1, 1]
// Each threadgroup computes one batch element's C[1,N] = alpha * A[1,K] * B + beta * C.
// Each thread handles ceil(N/256) output columns, accumulating in float32.
// ---------------------------------------------------------------------------
static const char* kGemvF16MSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void gemv_half(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device       half*  C       [[buffer(2)]],
    constant     uint&  K       [[buffer(3)]],
    constant     uint&  N       [[buffer(4)]],
    constant     uint&  stridea [[buffer(5)]],
    constant     uint&  strideb [[buffer(6)]],
    constant     uint&  stridec [[buffer(7)]],
    constant     uint&  ldb_val [[buffer(8)]],
    constant     float& alpha   [[buffer(9)]],
    constant     float& beta    [[buffer(10)]],
    constant     uint&  tb      [[buffer(11)]],
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tgs      [[threads_per_threadgroup]])
{
    device const half* a_row = A + batch_id * stridea;
    device const half* b_mat = B + batch_id * strideb;
    device       half* c_row = C + batch_id * stridec;

    for (uint j = tid; j < N; j += tgs) {
        float acc = 0.0f;
        if (tb) {
            device const half* b_row = b_mat + j * ldb_val;
            for (uint i = 0; i < K; ++i)
                acc += (float)a_row[i] * (float)b_row[i];
        } else {
            for (uint i = 0; i < K; ++i)
                acc += (float)a_row[i] * (float)b_mat[i * ldb_val + j];
        }
        float old_c = (beta != 0.0f) ? (float)c_row[j] : 0.0f;
        c_row[j] = (half)(alpha * acc + beta * old_c);
    }
}
)";

static id<MTLLibrary> get_gemv_f16_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kGemvF16MSL, "gemv_f16");
}

static id<MTLComputePipelineState> get_gemv_f16_pso() {
  static PSOCache cache;
  return cache.get(get_gemv_f16_library, "gemv_half");
}

static void dispatch_gemv_f16_batched(
    bool transpose_b,
    ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha, float beta,
    const ctranslate2::float16_t* a, ctranslate2::dim_t lda, ctranslate2::dim_t stridea,
    const ctranslate2::float16_t* b, ctranslate2::dim_t ldb, ctranslate2::dim_t strideb,
    ctranslate2::float16_t* c, ctranslate2::dim_t ldc, ctranslate2::dim_t stridec,
    ctranslate2::dim_t batch_size) {
  if (batch_size <= 0 || n == 0 || k == 0) return;

  id<MTLComputePipelineState> pso = get_gemv_f16_pso();
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  [enc setBuffer:buf_a offset:off_a atIndex:0];
  [enc setBuffer:buf_b offset:off_b atIndex:1];
  [enc setBuffer:buf_c offset:off_c atIndex:2];

  uint32_t K_u32 = ct2_u32(k);
  uint32_t N_u32 = ct2_u32(n);
  uint32_t sa_u32 = ct2_u32(stridea);
  uint32_t sb_u32 = ct2_u32(strideb);
  uint32_t sc_u32 = ct2_u32(stridec);
  uint32_t ldb_u32 = ct2_u32(ldb);
  uint32_t tb_u32 = transpose_b ? 1u : 0u;
  [enc setBytes:&K_u32   length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&N_u32   length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&sa_u32  length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&sb_u32  length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&sc_u32  length:sizeof(uint32_t) atIndex:7];
  [enc setBytes:&ldb_u32 length:sizeof(uint32_t) atIndex:8];
  [enc setBytes:&alpha   length:sizeof(float)    atIndex:9];
  [enc setBytes:&beta    length:sizeof(float)    atIndex:10];
  [enc setBytes:&tb_u32  length:sizeof(uint32_t) atIndex:11];

  NSUInteger tgs = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
  [enc endEncoding];
  [enc release];

  // M11.19: Protect original buffers from premature reuse (encode-only).
  // Use O(1) protect_buffer_by_base — base pointers already obtained above.
  ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_b contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_c contents]);
}

template <typename T>
static void dispatch_mps_gemm_batched_padded(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const T* a, ctranslate2::dim_t lda, ctranslate2::dim_t stridea,
    const T* b, ctranslate2::dim_t ldb, ctranslate2::dim_t strideb,
    float beta,
    T* c, ctranslate2::dim_t ldc, ctranslate2::dim_t stridec,
    ctranslate2::dim_t batch_size) {
  if (batch_size <= 0 || m == 0 || n == 0 || k == 0) return;

  constexpr NSUInteger elem = sizeof(T);
  const MPSDataType dtype = MPS_Dtype<T>::value;

  const NSUInteger rows_a = transpose_a ? (NSUInteger)k : (NSUInteger)m;
  const NSUInteger cols_a = transpose_a ? (NSUInteger)m : (NSUInteger)k;
  const NSUInteger rows_b = transpose_b ? (NSUInteger)n : (NSUInteger)k;
  const NSUInteger cols_b = transpose_b ? (NSUInteger)k : (NSUInteger)n;

  const NSUInteger nat_rb_a = (NSUInteger)lda * elem;
  const NSUInteger nat_rb_b = (NSUInteger)ldb * elem;
  const NSUInteger nat_rb_c = (NSUInteger)ldc * elem;

  NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
  @autoreleasepool {
    mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a dataType:dtype];
    mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b dataType:dtype];
    mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:(NSUInteger)n dataType:dtype];
  }

  const bool pad_a = (nat_rb_a < mps_rb_a);
  const bool pad_b = (nat_rb_b < mps_rb_b);
  const bool pad_c = (nat_rb_c < mps_rb_c);

  // matrixBytes for the padded layout: rows * padded_rowBytes.
  const NSUInteger mb_a = rows_a * mps_rb_a;
  const NSUInteger mb_b = rows_b * mps_rb_b;
  const NSUInteger mb_c = (NSUInteger)m * mps_rb_c;

  id<MTLBuffer> buf_a = nil, buf_b = nil, buf_c = nil;
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> tmp_a = nil, tmp_b = nil, tmp_c = nil;

  // GPU row_copy for A/B/C packing — encode-only, zero CPU waits.
  // Row_copy and MPS GEMM are on separate command buffers (non-blocking commit
  // between them); the serial command queue guarantees execution order.

  if (pad_a) {
    tmp_a = alloc_temp_buffer(mb_a * (NSUInteger)batch_size);
    NSUInteger src_off_a = 0;
    id<MTLBuffer> src_buf_a = ctranslate2::metal_buffer_for_ptr(a, &src_off_a);
    dispatch_row_copy(src_buf_a, src_off_a,
                      nat_rb_a, (NSUInteger)stridea * elem,
                      tmp_a, 0,
                      mps_rb_a, mb_a,
                      nat_rb_a, rows_a, (NSUInteger)batch_size);
    buf_a = tmp_a;
  } else {
    buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  }

  if (pad_b) {
    tmp_b = alloc_temp_buffer(mb_b * (NSUInteger)batch_size);
    NSUInteger src_off_b = 0;
    id<MTLBuffer> src_buf_b = ctranslate2::metal_buffer_for_ptr(b, &src_off_b);
    dispatch_row_copy(src_buf_b, src_off_b,
                      nat_rb_b, (NSUInteger)strideb * elem,
                      tmp_b, 0,
                      mps_rb_b, mb_b,
                      nat_rb_b, rows_b, (NSUInteger)batch_size);
    buf_b = tmp_b;
  } else {
    buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  }

  if (pad_c) {
    tmp_c = alloc_temp_buffer(mb_c * (NSUInteger)batch_size);
    if (beta != 0.0f) {
      NSUInteger src_off_c = 0;
      id<MTLBuffer> src_buf_c = ctranslate2::metal_buffer_for_ptr(c, &src_off_c);
      dispatch_row_copy(src_buf_c, src_off_c,
                        nat_rb_c, (NSUInteger)stridec * elem,
                        tmp_c, 0,
                        mps_rb_c, mb_c,
                        nat_rb_c, (NSUInteger)m, (NSUInteger)batch_size);
    } else {
      std::memset(static_cast<uint8_t*>([tmp_c contents]), 0,
                  mb_c * (NSUInteger)batch_size);
    }
    buf_c = tmp_c;
  } else {
    buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  }

  // Non-blocking commit: flush row_copy encoders to a separate CB so MPS
  // GEMM gets a fresh CB.  The serial queue ensures row_copy finishes first.
  if (pad_a || pad_b || (pad_c && beta != 0.0f))
    ctranslate2::metal::commit_command_buffer();

  // Compute final layout parameters for the MPS batch descriptor.
  const NSUInteger final_mb_a = pad_a ? mb_a : (NSUInteger)stridea * elem;
  const NSUInteger final_mb_b = pad_b ? mb_b : (NSUInteger)strideb * elem;
  const NSUInteger final_mb_c = pad_c ? mb_c : (NSUInteger)stridec * elem;
  const NSUInteger final_rb_a = pad_a ? mps_rb_a : nat_rb_a;
  const NSUInteger final_rb_b = pad_b ? mps_rb_b : nat_rb_b;
  const NSUInteger final_rb_c = pad_c ? mps_rb_c : nat_rb_c;

  // Verify MPS batch constraints for the padded layout.
  if (final_mb_a % final_rb_a != 0 || final_mb_a < rows_a * final_rb_a ||
      final_mb_b % final_rb_b != 0 || final_mb_b < rows_b * final_rb_b ||
      final_mb_c % final_rb_c != 0 || final_mb_c < (NSUInteger)m * final_rb_c) {
    // Fallback: per-element MPS for safety.
    for (ctranslate2::dim_t i = 0; i < batch_size; ++i)
      dispatch_mps_gemm<T>(transpose_a, transpose_b, m, n, k,
                           alpha, a + i * stridea, lda,
                                  b + i * strideb, ldb,
                           beta,  c + i * stridec, ldc);
    return;
  }

  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();

  @autoreleasepool {
    MPSMatrixDescriptor* descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                              columns:cols_a
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:final_rb_a
                                          matrixBytes:final_mb_a
                                             dataType:dtype];
    MPSMatrixDescriptor* descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                              columns:cols_b
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:final_rb_b
                                          matrixBytes:final_mb_b
                                             dataType:dtype];
    MPSMatrixDescriptor* descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                              columns:(NSUInteger)n
                                             matrices:(NSUInteger)batch_size
                                             rowBytes:final_rb_c
                                          matrixBytes:final_mb_c
                                             dataType:dtype];

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:off_a descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:off_b descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:off_c descriptor:descC];

    id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
    MPSMatrixMultiplication* gemm_op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                           transposeLeft:(BOOL)transpose_a
                                          transposeRight:(BOOL)transpose_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:(double)beta];
    gemm_op.batchSize = (NSUInteger)batch_size;
    gemm_op.batchStart = 0;

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
    [gemm_op release];
  }

  // GPU unpack: copy rows from padded tmp_c back to tightly-packed C.
  if (pad_c) {
    NSUInteger dst_off_c = 0;
    id<MTLBuffer> dst_buf = ctranslate2::metal_buffer_for_ptr(c, &dst_off_c);
    dispatch_row_copy(tmp_c, 0,
                      mps_rb_c, mb_c,
                      dst_buf, dst_off_c,
                      nat_rb_c, (NSUInteger)stridec * elem,
                      nat_rb_c, (NSUInteger)m, (NSUInteger)batch_size);
  }

  // Release temp buffers (command buffer retains them until GPU completes).
  if (tmp_a) [tmp_a release];
  if (tmp_b) [tmp_b release];
  if (tmp_c) [tmp_c release];

}

}  // anonymous namespace

namespace ctranslate2 {

  template<>
  template <typename T>
  dim_t primitives<Device::METAL>::gemm_pack_b(
      const T*, bool, dim_t, dim_t, float, T*) {
    return 0;  // Packing not supported on Metal.
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::METAL>::gemm(
      bool a_is_packed, bool b_is_packed,
      bool transpose_a, bool transpose_b,
      dim_t m, dim_t n, dim_t k,
      float alpha,
      const In* a, dim_t lda,
      const In* b, dim_t ldb,
      float beta,
      Out* c, dim_t ldc,
      const Out* a_shift_compensation) {
    (void)a_is_packed; (void)b_is_packed; (void)a_shift_compensation;
    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      dispatch_mps_gemm<float>(transpose_a, transpose_b, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
      dispatch_mps_gemm<float16_t>(transpose_a, transpose_b, m, n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same_v<In, bfloat16_t> && std::is_same_v<Out, bfloat16_t>) {
      dispatch_bf16_gemm(transpose_a, transpose_b, m, n, k,
                         alpha, beta, a, lda, b, ldb, c, ldc);
    } else if constexpr (std::is_same_v<In, int8_t> && std::is_same_v<Out, int32_t>) {
      dispatch_int8_gemm(transpose_a, transpose_b, m, n, k,
                         alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
      METAL_STUB(gemm);
    }
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::METAL>::gemm_batch_strided(
      bool transpose_a, bool transpose_b,
      dim_t m, dim_t n, dim_t k,
      float alpha,
      const In* a, dim_t lda, dim_t stridea,
      const In* b, dim_t ldb, dim_t strideb,
      float beta,
      Out* c, dim_t ldc, dim_t stridec,
      dim_t batch_size) {
    // Check if batch elements need tiny-matrix CPU GEMM fallback.
    // Only for truly tiny columns (≤4) where MPS overhead dominates.
    // Alignment-only padding (large matrices) is handled inside dispatch_mps_gemm
    // via GPU-side temp buffer + blit copy (zero syncs).
    auto needs_padding = [&]() -> bool {
      constexpr NSUInteger elem_sz = sizeof(In);
      const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
      const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);
      const NSUInteger cols_c = (NSUInteger)n;
      NSUInteger mps_a, mps_b, mps_c;
      MPSDataType dt = (elem_sz == 4) ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
      @autoreleasepool {
        mps_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a dataType:dt];
        mps_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b dataType:dt];
        mps_c = [MPSMatrixDescriptor rowBytesForColumns:cols_c dataType:dt];
      }
      return ((NSUInteger)lda * elem_sz < mps_a) ||
             ((NSUInteger)ldb * elem_sz < mps_b) ||
             ((NSUInteger)ldc * elem_sz < mps_c);
    };

    // Helper: single-sync CPU cblas loop for a batch of tiny padded float32 GEMMs.
    auto batch_cpu_gemm_f32 = [&](const float* ba, const float* bb, float* bc) {
      CT2_COMMIT_AND_WAIT();
      for (dim_t i = 0; i < batch_size; ++i)
        cblas_sgemm(CblasRowMajor,
                    transpose_a ? CblasTrans : CblasNoTrans,
                    transpose_b ? CblasTrans : CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    alpha,
                    ba + i * stridea, (int)lda,
                    bb + i * strideb, (int)ldb,
                    beta,
                    bc + i * stridec, (int)ldc);
    };

    // Helper: single-sync CPU cblas loop for a batch of tiny padded float16 GEMMs.
    // Widens to float32, runs cblas_sgemm, narrows back.
    auto batch_cpu_gemm_f16 = [&](const float16_t* ba, const float16_t* bb, float16_t* bc) {
      CT2_COMMIT_AND_WAIT();
      const dim_t rows_a = transpose_a ? k : m;
      const dim_t cols_a = transpose_a ? m : k;
      const dim_t rows_b = transpose_b ? n : k;
      const dim_t cols_b = transpose_b ? k : n;
      const dim_t elems_a = rows_a * cols_a;
      const dim_t elems_b = rows_b * cols_b;
      const dim_t elems_c = m * n;
      // Stack-allocate for small buffers, heap for large.
      constexpr dim_t kStackMax = 4096;
      float sa[kStackMax], sb[kStackMax], sc[kStackMax];
      float* fa = (elems_a <= kStackMax) ? sa : new float[elems_a];
      float* fb = (elems_b <= kStackMax) ? sb : new float[elems_b];
      float* fc = (elems_c <= kStackMax) ? sc : new float[elems_c];
      for (dim_t i = 0; i < batch_size; ++i) {
        const auto* ai = ba + i * stridea;
        const auto* bi = bb + i * strideb;
        auto* ci = bc + i * stridec;
        // Strided widen: lda/ldb may differ from cols when input is non-contiguous.
        for (dim_t r = 0; r < rows_a; ++r)
          for (dim_t c = 0; c < cols_a; ++c)
            fa[r * cols_a + c] = static_cast<float>(ai[r * lda + c]);
        for (dim_t r = 0; r < rows_b; ++r)
          for (dim_t c = 0; c < cols_b; ++c)
            fb[r * cols_b + c] = static_cast<float>(bi[r * ldb + c]);
        if (beta != 0.0f)
          for (dim_t r = 0; r < m; ++r)
            for (dim_t c = 0; c < n; ++c)
              fc[r * n + c] = static_cast<float>(ci[r * ldc + c]);
        // cblas uses the contiguous widened layout (lda=cols_a, ldb=cols_b, ldc=n).
        cblas_sgemm(CblasRowMajor,
                    transpose_a ? CblasTrans : CblasNoTrans,
                    transpose_b ? CblasTrans : CblasNoTrans,
                    (int)m, (int)n, (int)k,
                    alpha, fa, (int)cols_a, fb, (int)cols_b,
                    beta, fc, (int)n);
        for (dim_t r = 0; r < m; ++r)
          for (dim_t c = 0; c < n; ++c)
            ci[r * ldc + c] = static_cast<float16_t>(fc[r * n + c]);
      }
      if (fa != sa) delete[] fa;
      if (fb != sb) delete[] fb;
      if (fc != sc) delete[] fc;
    };

    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      if (batch_size > 0 && needs_padding()) {
        if (m * n > 4096 || m == 1) {
          // M11.18: Route m=1 decode attention GEMMs through MPS padded path
          // (encode-only, zero syncs) instead of cblas (1 sync per call).
          dispatch_mps_gemm_batched_padded<float>(
              transpose_a, transpose_b, m, n, k,
              alpha, a, lda, stridea, b, ldb, strideb,
              beta, c, ldc, stridec, batch_size);
          if (m == 1) {
            // M11.18: Protect original buffers from premature reuse.
            // Only needed for m=1 decode attention GEMMs where the caller
            // may free A/B/C before the encode-only GPU work completes.
            // Use O(1) protect_buffer_by_base via metal_buffer_for_ptr.
            NSUInteger off_tmp;
            ctranslate2::metal::protect_buffer_by_base(
                [ctranslate2::metal_buffer_for_ptr(a, &off_tmp) contents]);
            ctranslate2::metal::protect_buffer_by_base(
                [ctranslate2::metal_buffer_for_ptr(b, &off_tmp) contents]);
            ctranslate2::metal::protect_buffer_by_base(
                [ctranslate2::metal_buffer_for_ptr(c, &off_tmp) contents]);
          }
        } else {
          batch_cpu_gemm_f32(a, b, c);
        }
      } else {
        if (!dispatch_mps_gemm_batched<float>(
                transpose_a, transpose_b, m, n, k,
                alpha, a, lda, stridea, b, ldb, strideb,
                beta, c, ldc, stridec, batch_size)) {
          for (dim_t i = 0; i < batch_size; ++i)
            dispatch_mps_gemm<float>(transpose_a, transpose_b, m, n, k,
                                     alpha, a + i * stridea, lda,
                                            b + i * strideb, ldb,
                                     beta,  c + i * stridec, ldc);
        }
      }
    } else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
      // M11.19: Route ALL m=1 float16 GEMMs through custom GEMV (encode-only, zero syncs).
      // MPS batched GEMM produces garbled output for float16 m=1 (MPS bug verified in M11.18).
      // Custom MSL kernel avoids MPS entirely; protect_buffer prevents buffer-reuse crashes.
      if (m == 1 && batch_size > 0) {
        dispatch_gemv_f16_batched(transpose_b, n, k, alpha, beta,
                                  a, lda, stridea, b, ldb, strideb,
                                  c, ldc, stridec, batch_size);
        return;
      }
      if (batch_size > 0 && needs_padding()) {
        if (m * n > 4096) {
          dispatch_mps_gemm_batched_padded<float16_t>(
              transpose_a, transpose_b, m, n, k,
              alpha, a, lda, stridea, b, ldb, strideb,
              beta, c, ldc, stridec, batch_size);
        } else {
          batch_cpu_gemm_f16(a, b, c);
        }
      } else {
        if (!dispatch_mps_gemm_batched<float16_t>(
                transpose_a, transpose_b, m, n, k,
                alpha, a, lda, stridea, b, ldb, strideb,
                beta, c, ldc, stridec, batch_size)) {
          for (dim_t i = 0; i < batch_size; ++i)
            dispatch_mps_gemm<float16_t>(transpose_a, transpose_b, m, n, k,
                                         alpha, a + i * stridea, lda,
                                                b + i * strideb, ldb,
                                         beta,  c + i * stridec, ldc);
        }
      }
    } else if constexpr (std::is_same_v<In, bfloat16_t> && std::is_same_v<Out, bfloat16_t>) {
      if (beta != 0.0f)
        throw std::runtime_error(
            "Metal BF16 GEMM: only beta=0.0 is supported");
      CT2_COMMIT_AND_WAIT();  // flush once before the batch loop
      for (dim_t i = 0; i < batch_size; ++i)
        run_bf16_gemm_inner(transpose_a, transpose_b, m, n, k,
                            a + i * stridea, lda,
                            b + i * strideb, ldb,
                            c + i * stridec, ldc);
      // Apply alpha scaling post-GEMM if needed.
      if (alpha != 1.0f) {
        const dim_t total = batch_size * m * n;
        primitives<Device::METAL>::mul(
            static_cast<bfloat16_t>(alpha), c, c, total);
      }
    } else if constexpr (std::is_same_v<In, int8_t> && std::is_same_v<Out, int32_t>) {
      if (batch_size == 0 || m == 0 || n == 0 || k == 0) return;
      if (beta != 0.0f)
        throw std::runtime_error("Metal INT8 GEMM: only beta=0 is supported");
      if (batch_size == 1) {
        dispatch_int8_gemm(transpose_a, transpose_b, m, n, k,
                           alpha, a, lda, b, ldb, beta, c, ldc);
      } else {
        // Amortized path: 2 syncs total instead of 2*B.
        // Phase 1: flush pending GPU work so CPU can read int8 inputs.
        CT2_COMMIT_AND_WAIT();

        const NSUInteger rows_a = (NSUInteger)(transpose_a ? k : m);
        const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
        const NSUInteger rows_b = (NSUInteger)(transpose_b ? n : k);
        const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);

        NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
        @autoreleasepool {
          mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a
                                                    dataType:MPSDataTypeFloat32];
          mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b
                                                    dataType:MPSDataTypeFloat32];
          mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:(NSUInteger)n
                                                    dataType:MPSDataTypeFloat32];
        }
        const NSUInteger rb_a = std::max((NSUInteger)lda * sizeof(float), mps_rb_a);
        const NSUInteger rb_b = std::max((NSUInteger)ldb * sizeof(float), mps_rb_b);
        const NSUInteger rb_c = std::max((NSUInteger)n   * sizeof(float), mps_rb_c);

        const NSUInteger bytes_a = rows_a * rb_a;
        const NSUInteger bytes_b = rows_b * rb_b;
        const NSUInteger bytes_c = (NSUInteger)m * rb_c;

        // Allocate temp buffers for all batches.
        id<MTLBuffer> tmp_a = alloc_temp_buffer(bytes_a * (NSUInteger)batch_size);
        id<MTLBuffer> tmp_b = alloc_temp_buffer(bytes_b * (NSUInteger)batch_size);
        id<MTLBuffer> tmp_c = alloc_temp_buffer(bytes_c * (NSUInteger)batch_size);
        std::memset([tmp_c contents], 0, bytes_c * (NSUInteger)batch_size);

        // Phase 2: CPU convert all batches int8→float32 (vectorized via vDSP).
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          const int8_t* src_a = a + bi * stridea;
          const int8_t* src_b = b + bi * strideb;
          for (NSUInteger r = 0; r < rows_a; ++r) {
            float* dst = reinterpret_cast<float*>(
                static_cast<uint8_t*>([tmp_a contents]) + bi * bytes_a + r * rb_a);
            int8_to_float32(dst, src_a + r * (NSUInteger)lda, cols_a);
          }
          for (NSUInteger r = 0; r < rows_b; ++r) {
            float* dst = reinterpret_cast<float*>(
                static_cast<uint8_t*>([tmp_b contents]) + bi * bytes_b + r * rb_b);
            int8_to_float32(dst, src_b + r * (NSUInteger)ldb, cols_b);
          }
        }

        // Phase 3: Encode all B MPS GEMMs (encode-only, no sync).
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          dispatch_mps_gemm_buf(
              transpose_a, transpose_b, m, n, k, alpha,
              tmp_a, bi * bytes_a, rb_a, rows_a, cols_a,
              tmp_b, bi * bytes_b, rb_b, rows_b, cols_b,
              tmp_c, bi * bytes_c, rb_c, MPSDataTypeFloat32);
        }

        // Phase 4: Wait for all GPU GEMMs.
        CT2_COMMIT_AND_WAIT();

        // Phase 5: CPU round all batches float32→int32.
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          int32_t* dst_c = c + bi * stridec;
          for (dim_t row = 0; row < m; ++row) {
            const float* src = reinterpret_cast<const float*>(
                static_cast<const uint8_t*>([tmp_c contents])
                + bi * bytes_c + (NSUInteger)row * rb_c);
            int32_t* dst = dst_c + row * ldc;
            for (dim_t col = 0; col < n; ++col)
              dst[col] = static_cast<int32_t>(std::lroundf(src[col]));
          }
        }

        // Release temp buffers (GPU completed, CPU readback done).
        [tmp_a release];
        [tmp_b release];
        [tmp_c release];
      }
    } else {
      METAL_STUB(gemm_batch_strided);
    }
  }

  // -------------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------------

  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const float*, bool, dim_t, dim_t, float, float*);
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const float16_t*, bool, dim_t, dim_t, float, float16_t*);
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const bfloat16_t*, bool, dim_t, dim_t, float, bfloat16_t*);
  template dim_t primitives<Device::METAL>::gemm_pack_b(
      const int8_t*, bool, dim_t, dim_t, float, int8_t*);

  template void primitives<Device::METAL>::gemm<float, float>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, const float*, dim_t,
      float, float*, dim_t, const float*);
  template void primitives<Device::METAL>::gemm<float16_t, float16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, const float16_t*, dim_t,
      float, float16_t*, dim_t, const float16_t*);
  template void primitives<Device::METAL>::gemm<bfloat16_t, bfloat16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, const bfloat16_t*, dim_t,
      float, bfloat16_t*, dim_t, const bfloat16_t*);
  template void primitives<Device::METAL>::gemm<int8_t, int32_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, int32_t*, dim_t, const int32_t*);

  template void primitives<Device::METAL>::gemm_batch_strided<float, float>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, dim_t, const float*, dim_t, dim_t,
      float, float*, dim_t, dim_t, dim_t);
  template void primitives<Device::METAL>::gemm_batch_strided<float16_t, float16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, dim_t, const float16_t*, dim_t, dim_t,
      float, float16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::METAL>::gemm_batch_strided<bfloat16_t, bfloat16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, dim_t, const bfloat16_t*, dim_t, dim_t,
      float, bfloat16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::METAL>::gemm_batch_strided<int8_t, int32_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, dim_t, const int8_t*, dim_t, dim_t,
      float, int32_t*, dim_t, dim_t, dim_t);

}  // namespace ctranslate2
