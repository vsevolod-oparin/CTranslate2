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

namespace {

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
// temporary buffer.  For the output we flush the GPU and unpack back to c.
// ---------------------------------------------------------------------------

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

  // Prepare buffers — copy to row-padded temps when stride is too small.
  //
  // Coherency note (2.6): CPU memcpy to freshly-allocated Shared buffers is
  // immediately visible to the GPU on Apple Silicon unified memory.
  //
  // Padding hazard: when pad_a or pad_b is true the CPU must memcpy from A/B,
  // which may have been written by a previous GPU kernel (e.g. layernorm,
  // transpose).  On unified memory the CPU only sees the GPU's writes after
  // the command buffer has been committed and completed.  Flush before reading.
  // (pad_c reads C only when beta≠0, same reasoning.)
  if (pad_a || pad_b || (pad_c && beta != 0.0f))
    ctranslate2::metal::commit_and_wait();

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
  }

  // If C was routed to a padded temp buffer, flush GPU and unpack back to c.
  if (pad_c) {
    ctranslate2::metal::commit_and_wait();
    const auto* src = static_cast<const uint8_t*>([tmp_c contents]);
    auto* dst = reinterpret_cast<uint8_t*>(c);
    for (NSUInteger r = 0; r < rows_c; ++r)
      std::memcpy(dst + r * nat_rb_c, src + r * mps_rb_c, nat_rb_c);
  }
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
  }
}

// Flush pending GPU work and run one BF16 GEMM.
// Throws if alpha != 1.0 or beta != 0.0 (not supported by the MPSGraph path).
static void dispatch_bf16_gemm(bool trans_a, bool trans_b,
                                ctranslate2::dim_t m,
                                ctranslate2::dim_t n,
                                ctranslate2::dim_t k,
                                float alpha, float beta,
                                const ctranslate2::bfloat16_t* a, ctranslate2::dim_t lda,
                                const ctranslate2::bfloat16_t* b, ctranslate2::dim_t ldb,
                                ctranslate2::bfloat16_t* c, ctranslate2::dim_t ldc) {
  if (alpha != 1.0f || beta != 0.0f)
    throw std::runtime_error(
        "Metal BF16 GEMM: only alpha=1.0 and beta=0.0 are supported");
  if (m == 0 || n == 0 || k == 0) return;
  // MPSGraph uses its own queue; flush the deferred CB first.
  ctranslate2::metal::commit_and_wait();
  run_bf16_gemm_inner(trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc);
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
  ctranslate2::metal::commit_and_wait();

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
    for (NSUInteger ci = 0; ci < cols_a; ++ci)
      dst[ci] = static_cast<float>(src[ci]);
  }
  for (NSUInteger r = 0; r < rows_b; ++r) {
    float*        dst = reinterpret_cast<float*>(
                      static_cast<uint8_t*>([tmp_b contents]) + r * rb_b);
    const int8_t* src = b + r * (NSUInteger)ldb;
    for (NSUInteger ci = 0; ci < cols_b; ++ci)
      dst[ci] = static_cast<float>(src[ci]);
  }
  std::memset([tmp_c contents], 0, (NSUInteger)m * rb_c);

  // Encode float32 MPS GEMM using temp buffers directly.
  dispatch_mps_gemm_buf(
      transpose_a, transpose_b, m, n, k, alpha,
      tmp_a, 0, rb_a, rows_a, cols_a,
      tmp_b, 0, rb_b, rows_b, cols_b,
      tmp_c, 0, rb_c, MPSDataTypeFloat32);

  // Wait for GPU, then round float32 → int32 (handles ldc stride).
  ctranslate2::metal::commit_and_wait();
  for (ctranslate2::dim_t row = 0; row < m; ++row) {
    const float* src = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>([tmp_c contents]) + (NSUInteger)row * rb_c);
    int32_t* dst = c + row * ldc;
    for (ctranslate2::dim_t col = 0; col < n; ++col)
      dst[col] = static_cast<int32_t>(std::lroundf(src[col]));
  }
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
    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      for (dim_t i = 0; i < batch_size; ++i)
        dispatch_mps_gemm<float>(transpose_a, transpose_b, m, n, k,
                                 alpha, a + i * stridea, lda,
                                        b + i * strideb, ldb,
                                 beta,  c + i * stridec, ldc);
    } else if constexpr (std::is_same_v<In, float16_t> && std::is_same_v<Out, float16_t>) {
      for (dim_t i = 0; i < batch_size; ++i)
        dispatch_mps_gemm<float16_t>(transpose_a, transpose_b, m, n, k,
                                     alpha, a + i * stridea, lda,
                                            b + i * strideb, ldb,
                                     beta,  c + i * stridec, ldc);
    } else if constexpr (std::is_same_v<In, bfloat16_t> && std::is_same_v<Out, bfloat16_t>) {
      if (alpha != 1.0f || beta != 0.0f)
        throw std::runtime_error(
            "Metal BF16 GEMM: only alpha=1.0 and beta=0.0 are supported");
      metal::commit_and_wait();  // flush once before the batch loop
      for (dim_t i = 0; i < batch_size; ++i)
        run_bf16_gemm_inner(transpose_a, transpose_b, m, n, k,
                            a + i * stridea, lda,
                            b + i * strideb, ldb,
                            c + i * stridec, ldc);
    } else if constexpr (std::is_same_v<In, int8_t> && std::is_same_v<Out, int32_t>) {
      for (dim_t i = 0; i < batch_size; ++i)
        dispatch_int8_gemm(transpose_a, transpose_b, m, n, k,
                           alpha,
                           a + i * stridea, lda,
                           b + i * strideb, ldb,
                           beta,
                           c + i * stridec, ldc);
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
