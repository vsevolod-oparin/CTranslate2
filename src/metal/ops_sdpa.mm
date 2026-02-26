// src/metal/ops_sdpa.mm
//
// M6.1 — Scaled Dot-Product Attention for Device::METAL.
//
// Algorithm per (batch b, query head h):
//   hk = h % num_heads_k               (grouped query attention)
//   S  = scale * Q[b,h] @ K[b,hk]^T   [seqlen_q × seqlen_k]
//   if (is_causal) S[col > row] = large_neg
//   A  = softmax(S, dim=-1)
//   O[b,h] = A @ V[b,hk]               [seqlen_q × head_dim]
//
// Q/K/V layout: [batch, seqlen, num_heads, head_dim] (interleaved heads).
//   Row stride for Q:    q_lda  = num_heads   * head_dim
//   Row stride for K/V:  kv_lda = num_heads_k * head_dim
//
// FP32 / FP16 path: MPSMatrixMultiplication (encode-only).
//   Non-contiguous strides handled via MPS rowBytes parameter.
//
// BF16 path: MPSGraph matmul (synchronous, commits_and_waits around each GEMM).
//   Q and K slices are packed row-by-row to contiguous allocator-registered
//   buffers before each GEMM (MPSGraph requires contiguous inputs).
//   V slice is packed before the second GEMM.
//
// Scores buffer: always allocator-registered (not alloc_temp_buffer), so
//   metal_buffer_for_ptr() can locate it for the causal-mask and softmax kernels.

#include "metal/primitives_infra.h"
#include "metal/ops_metal.h"
#include "ctranslate2/allocator.h"

namespace {

// ---------------------------------------------------------------------------
// RAII allocator-registered temporary buffer.
//
// Buffers allocated via get_allocator<Device::METAL>() are tracked in
// MetalAllocator::_live, so metal_buffer_for_ptr() can find them.
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

  template <typename T>
  T* as() { return static_cast<T*>(ptr); }
};

// ---------------------------------------------------------------------------
// SDPA MSL library (causal_mask_float / causal_mask_half / causal_mask_bfloat)
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_sdpa_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kSdpaMSL, "sdpa");
}

static id<MTLComputePipelineState> get_sdpa_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_sdpa_library, name);
}

// Encode the causal mask into the scores buffer (one thread per element).
//   scores layout: [seqlen_q, seqlen_k] contiguous.
//   Sets scores[row*seqlen_k + col] = large_neg  when  col > row.
//   (offset == 0 for M6.1; the MSL kernel takes it as a uniform)
template <typename T>
static void dispatch_causal_mask(T* scores,
                                  ctranslate2::dim_t seqlen_q,
                                  ctranslate2::dim_t seqlen_k) {
  if (seqlen_q == 0 || seqlen_k == 0) {
    return;
  }
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "causal_mask_%s", MetalTypeName<T>::value);
  id<MTLComputePipelineState> pso = get_sdpa_pso(kname);
  const ctranslate2::dim_t total = seqlen_q * seqlen_k;
  uint32_t sk     = ct2_u32(seqlen_k);
  uint32_t offset = 0u;
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc =
      [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc setComputePipelineState:pso];
  NSUInteger off = 0;
  [enc setBuffer:ctranslate2::metal_buffer_for_ptr(scores, &off) offset:off atIndex:0];
  [enc setBytes:&sk     length:sizeof(uint32_t) atIndex:1];
  [enc setBytes:&offset length:sizeof(uint32_t) atIndex:2];
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup,
                                       static_cast<NSUInteger>(total));
  [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(total), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
}

// ---------------------------------------------------------------------------
// MPS GEMM — FP32 / FP16 only (encode-only, handles non-contiguous strides)
//
// Computes C = alpha * A * B^(trans_b).
//   A: [m, k], rowStride lda  (not transposed)
//   B: [k, n] if !trans_b, or [n, k] if trans_b, rowStride ldb
//   C: [m, n], rowStride ldc
// Handles non-contiguous rows via MPS rowBytes (same approach as
// dispatch_mps_gemm in primitives_gemm.mm).
// ---------------------------------------------------------------------------

template <typename T> struct SdpaMPSDtype;
template <> struct SdpaMPSDtype<float>
{ static const MPSDataType v = MPSDataTypeFloat32; };
template <> struct SdpaMPSDtype<ctranslate2::float16_t>
{ static const MPSDataType v = MPSDataTypeFloat16; };

template <typename T>
static void sdpa_mps_gemm(bool trans_b,
                            ctranslate2::dim_t m,
                            ctranslate2::dim_t n,
                            ctranslate2::dim_t k,
                            float alpha,
                            const T* a, ctranslate2::dim_t lda,
                            const T* b, ctranslate2::dim_t ldb,
                            T* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  const MPSDataType dtype = SdpaMPSDtype<T>::v;
  constexpr NSUInteger elem = sizeof(T);

  const NSUInteger rows_a = (NSUInteger)m, cols_a = (NSUInteger)k;
  const NSUInteger rows_b = trans_b ? (NSUInteger)n : (NSUInteger)k;
  const NSUInteger cols_b = trans_b ? (NSUInteger)k : (NSUInteger)n;
  const NSUInteger rows_c = (NSUInteger)m, cols_c = (NSUInteger)n;

  const NSUInteger nat_rb_a = static_cast<NSUInteger>(ct2_u32(lda)) * elem;
  const NSUInteger nat_rb_b = static_cast<NSUInteger>(ct2_u32(ldb)) * elem;
  const NSUInteger nat_rb_c = static_cast<NSUInteger>(ct2_u32(ldc)) * elem;

  NSUInteger mps_rb_a, mps_rb_b, mps_rb_c;
  @autoreleasepool {
    mps_rb_a = [MPSMatrixDescriptor rowBytesForColumns:cols_a dataType:dtype];
    mps_rb_b = [MPSMatrixDescriptor rowBytesForColumns:cols_b dataType:dtype];
    mps_rb_c = [MPSMatrixDescriptor rowBytesForColumns:cols_c dataType:dtype];
  }

  const bool pad_a = (nat_rb_a < mps_rb_a);
  const bool pad_b = (nat_rb_b < mps_rb_b);
  const bool pad_c = (nat_rb_c < mps_rb_c);

  id<MTLBuffer> buf_a = nil, buf_b = nil, buf_c = nil;
  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> tmp_a = nil, tmp_b = nil, tmp_c = nil;

  if (pad_a) {
    tmp_a = alloc_temp_buffer(rows_a * mps_rb_a);
    auto* dst = static_cast<uint8_t*>([tmp_a contents]);
    const auto* src = reinterpret_cast<const uint8_t*>(a);
    for (NSUInteger r = 0; r < rows_a; ++r) {
      std::memcpy(dst + r * mps_rb_a, src + r * nat_rb_a, nat_rb_a);
    }
    buf_a = tmp_a;
  } else {
    buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  }

  if (pad_b) {
    tmp_b = alloc_temp_buffer(rows_b * mps_rb_b);
    auto* dst = static_cast<uint8_t*>([tmp_b contents]);
    const auto* src = reinterpret_cast<const uint8_t*>(b);
    for (NSUInteger r = 0; r < rows_b; ++r) {
      std::memcpy(dst + r * mps_rb_b, src + r * nat_rb_b, nat_rb_b);
    }
    buf_b = tmp_b;
  } else {
    buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  }

  if (pad_c) {
    tmp_c = alloc_temp_buffer(rows_c * mps_rb_c);
    std::memset([tmp_c contents], 0, rows_c * mps_rb_c);
    buf_c = tmp_c;
  } else {
    buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  }

  const NSUInteger rb_a = pad_a ? mps_rb_a : nat_rb_a;
  const NSUInteger rb_b = pad_b ? mps_rb_b : nat_rb_b;
  const NSUInteger rb_c = pad_c ? mps_rb_c : nat_rb_c;

  // Fetch command buffer BEFORE @autoreleasepool (avoids use-after-free).
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
                                           transposeLeft:NO
                                          transposeRight:(BOOL)trans_b
                                             resultRows:(NSUInteger)m
                                          resultColumns:(NSUInteger)n
                                        interiorColumns:(NSUInteger)k
                                                  alpha:(double)alpha
                                                   beta:0.0];
    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
  }

  // If C required padding: flush, then unpack the padded temp back to c.
  if (pad_c) {
    ctranslate2::metal::commit_and_wait();
    const auto* src = static_cast<const uint8_t*>([tmp_c contents]);
    auto* dst = reinterpret_cast<uint8_t*>(c);
    for (NSUInteger r = 0; r < rows_c; ++r) {
      std::memcpy(dst + r * nat_rb_c, src + r * mps_rb_c, nat_rb_c);
    }
  }
}

// ---------------------------------------------------------------------------
// BF16 GEMM — MPSGraph (synchronous)
// ---------------------------------------------------------------------------

struct SdpaBf16Entry {
  MPSGraph*       graph  = nil;
  MPSGraphTensor* ph_a   = nil;
  MPSGraphTensor* ph_b   = nil;
  MPSGraphTensor* result = nil;
};

// Two graph variants: (trans_b=false) for S@V, (trans_b=true) for Q@K^T.
static SdpaBf16Entry& get_sdpa_bf16_entry(bool trans_b) {
  static SdpaBf16Entry entries[2];
  static bool initialized[2] = {};
  static std::mutex mtx;
  const int idx = trans_b ? 1 : 0;
  std::lock_guard<std::mutex> lk(mtx);
  if (!initialized[idx]) {
    MPSGraph* g = [[MPSGraph alloc] init];
    MPSGraphTensor* pA = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"A"];
    MPSGraphTensor* pB = [g placeholderWithShape:nil
                                        dataType:MPSDataTypeBFloat16
                                            name:@"B"];
    MPSGraphTensor* opB = trans_b
        ? [g transposeTensor:pB dimension:0 withDimension:1 name:@"BT"] : pB;
    MPSGraphTensor* tC  = [g matrixMultiplicationWithPrimaryTensor:pA
                                                   secondaryTensor:opB
                                                              name:@"C"];
    entries[idx] = { g, pA, pB, tC };
    initialized[idx] = true;
  }
  return entries[idx];
}

// Run BF16 GEMM: c = a * b^(trans_b), alpha must be 1.0 (baked into the call
// site by scaling a before calling).
// a: [m, k] contiguous; b: [k,n] or [n,k] contiguous; c: [m,n] contiguous.
// All pointers must be allocator-registered (live in MetalAllocator::_live).
// Flushes the deferred CB before running the graph (MPSGraph is synchronous).
static void sdpa_bf16_gemm(bool trans_b,
                            ctranslate2::dim_t m,
                            ctranslate2::dim_t n,
                            ctranslate2::dim_t k,
                            const ctranslate2::bfloat16_t* a,
                            const ctranslate2::bfloat16_t* b,
                            ctranslate2::bfloat16_t* c) {
  if (m == 0 || n == 0 || k == 0) {
    return;
  }
  // Flush any pending GPU work (causal_mask, softmax) before MPSGraph runs.
  ctranslate2::metal::commit_and_wait();

  SdpaBf16Entry& entry = get_sdpa_bf16_entry(trans_b);
  id<MTLCommandQueue> queue = ctranslate2::metal::get_metal_command_queue();

  NSArray<NSNumber*>* shape_a = @[@((int)m), @((int)k)];
  NSArray<NSNumber*>* shape_b = trans_b
      ? @[@((int)n), @((int)k)] : @[@((int)k), @((int)n)];

  NSUInteger off_a = 0, off_b = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

  // MPSGraphTensorData has no byte-offset; copy to zero-offset temp if needed.
  id<MTLBuffer> tmp_a = nil, tmp_b = nil;
  if (off_a != 0) {
    const size_t bytes = (size_t)m * k * sizeof(ctranslate2::bfloat16_t);
    tmp_a = alloc_temp_buffer(bytes);
    std::memcpy([tmp_a contents], a, bytes);
    buf_a = tmp_a;
  }
  if (off_b != 0) {
    const size_t rows_b = trans_b ? (size_t)n : (size_t)k;
    const size_t cols_b = trans_b ? (size_t)k : (size_t)n;
    const size_t bytes  = rows_b * cols_b * sizeof(ctranslate2::bfloat16_t);
    tmp_b = alloc_temp_buffer(bytes);
    std::memcpy([tmp_b contents], b, bytes);
    buf_b = tmp_b;
  }

  @autoreleasepool {
    MPSGraphTensorData* tdA =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_a
                                                shape:shape_a
                                             dataType:MPSDataTypeBFloat16];
    MPSGraphTensorData* tdB =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:buf_b
                                                shape:shape_b
                                             dataType:MPSDataTypeBFloat16];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [entry.graph runWithMTLCommandQueue:queue
                                      feeds:@{entry.ph_a: tdA, entry.ph_b: tdB}
                              targetTensors:@[entry.result]
                           targetOperations:nil];
    if (!results || !results[entry.result]) {
      throw std::runtime_error("Metal SDPA BF16 GEMM: graph execution returned nil");
    }
    // c is a Shared-mode MTLBuffer contents pointer — directly CPU-writable.
    [[results[entry.result] mpsndarray] readBytes:c strideBytes:nil];
  }
}

// ---------------------------------------------------------------------------
// Head-level SDPA — FP32 / FP16 path (encode-only)
// ---------------------------------------------------------------------------

template <typename T>
static void sdpa_head_mps(const T* q_row0, const T* k_row0,
                           const T* v_row0, T* out_row0,
                           ctranslate2::dim_t q_lda,
                           ctranslate2::dim_t kv_lda,
                           ctranslate2::dim_t seqlen_q,
                           ctranslate2::dim_t seqlen_k,
                           ctranslate2::dim_t head_dim,
                           float scale, bool is_causal) {
  // scores[seqlen_q, seqlen_k] — allocator-registered for metal_buffer_for_ptr.
  MetalTempBuf scores_buf(static_cast<size_t>(seqlen_q) * seqlen_k * sizeof(T));
  T* scores = scores_buf.as<T>();

  // Step 1: scores = scale * Q[b,h] @ K[b,hk]^T   (encode-only MPS GEMM)
  sdpa_mps_gemm<T>(/*trans_b=*/true,
                   seqlen_q, seqlen_k, head_dim, scale,
                   q_row0, q_lda,
                   k_row0, kv_lda,
                   scores, seqlen_k);

  // Step 2: causal mask (encode-only GPU kernel, if requested)
  if (is_causal) {
    dispatch_causal_mask(scores, seqlen_q, seqlen_k);
  }

  // Step 3: softmax over last dim  (encode-only GPU kernel)
  ctranslate2::metal::softmax_metal<T>(scores, nullptr, scores,
                                        seqlen_q, seqlen_k, /*log_mode=*/false);

  // Flush pending CB before step 4 if the scores row stride is narrower than
  // the MPS hardware minimum rowBytes.  In that case sdpa_mps_gemm would
  // perform a CPU memcpy of 'scores' (pad_a path), which must see the GPU-
  // written softmax output rather than the stale Q@K^T values.
  // For typical seqlen_k >= 8 (float32) / >= 16 (float16) this is a no-op.
  {
    NSUInteger mps_min;
    @autoreleasepool {
      mps_min = [MPSMatrixDescriptor rowBytesForColumns:(NSUInteger)seqlen_k
                                              dataType:SdpaMPSDtype<T>::v];
    }
    if ((NSUInteger)seqlen_k * sizeof(T) < mps_min) {
      ctranslate2::metal::commit_and_wait();
    }
  }

  // Step 4: out = scores @ V[b,hk]   (encode-only MPS GEMM)
  //   output row stride = q_lda (same as Q, since output shares layout with Q)
  sdpa_mps_gemm<T>(/*trans_b=*/false,
                   seqlen_q, head_dim, seqlen_k, 1.0f,
                   scores, seqlen_k,
                   v_row0, kv_lda,
                   out_row0, q_lda);
}

// ---------------------------------------------------------------------------
// Head-level SDPA — BF16 path (synchronous MPSGraph GEMMs)
// ---------------------------------------------------------------------------

static void sdpa_head_bf16(const ctranslate2::bfloat16_t* q_row0,
                            const ctranslate2::bfloat16_t* k_row0,
                            const ctranslate2::bfloat16_t* v_row0,
                            ctranslate2::bfloat16_t* out_row0,
                            ctranslate2::dim_t q_lda,
                            ctranslate2::dim_t kv_lda,
                            ctranslate2::dim_t seqlen_q,
                            ctranslate2::dim_t seqlen_k,
                            ctranslate2::dim_t head_dim,
                            float scale, bool is_causal) {
  using BF = ctranslate2::bfloat16_t;

  // Pack Q[b,h] → q_cont [seqlen_q, head_dim], applying scale during copy.
  // (MPSGraph requires contiguous inputs; MPS rowBytes cannot handle BF16.)
  MetalTempBuf q_buf(static_cast<size_t>(seqlen_q) * head_dim * sizeof(BF));
  BF* q_cont = q_buf.as<BF>();
  for (ctranslate2::dim_t r = 0; r < seqlen_q; ++r) {
    const BF* src = q_row0 + r * q_lda;
    BF* dst = q_cont + r * head_dim;
    for (ctranslate2::dim_t j = 0; j < head_dim; ++j) {
      dst[j] = BF(float(src[j]) * scale);
    }
  }

  // Pack K[b,hk] → k_cont [seqlen_k, head_dim] (contiguous).
  MetalTempBuf k_buf(static_cast<size_t>(seqlen_k) * head_dim * sizeof(BF));
  BF* k_cont = k_buf.as<BF>();
  for (ctranslate2::dim_t r = 0; r < seqlen_k; ++r) {
    std::memcpy(k_cont + r * head_dim, k_row0 + r * kv_lda,
                static_cast<size_t>(head_dim) * sizeof(BF));
  }

  // scores[seqlen_q, seqlen_k] — allocator-registered for metal_buffer_for_ptr.
  MetalTempBuf scores_buf(static_cast<size_t>(seqlen_q) * seqlen_k * sizeof(BF));
  BF* scores = scores_buf.as<BF>();

  // Step 1: scores = q_cont @ k_cont^T  (BF16 MPSGraph, synchronous)
  // alpha=1.0: scale was already folded into q_cont during the pack above.
  sdpa_bf16_gemm(/*trans_b=*/true, seqlen_q, seqlen_k, head_dim,
                  q_cont, k_cont, scores);

  // After MPSGraph returns, scores is in CPU-accessible Shared memory.
  // Encode causal mask and softmax into the next CB (GPU kernels).
  if (is_causal) {
    dispatch_causal_mask(scores, seqlen_q, seqlen_k);
  }

  ctranslate2::metal::softmax_metal<BF>(scores, nullptr, scores,
                                         seqlen_q, seqlen_k, /*log_mode=*/false);

  // Pack V[b,hk] → v_cont [seqlen_k, head_dim] (contiguous).
  MetalTempBuf v_buf(static_cast<size_t>(seqlen_k) * head_dim * sizeof(BF));
  BF* v_cont = v_buf.as<BF>();
  for (ctranslate2::dim_t r = 0; r < seqlen_k; ++r) {
    std::memcpy(v_cont + r * head_dim, v_row0 + r * kv_lda,
                static_cast<size_t>(head_dim) * sizeof(BF));
  }

  // Contiguous output buffer [seqlen_q, head_dim].
  MetalTempBuf out_buf(static_cast<size_t>(seqlen_q) * head_dim * sizeof(BF));
  BF* out_cont = out_buf.as<BF>();

  // Step 4: out_cont = scores @ v_cont  (BF16 MPSGraph, synchronous)
  // sdpa_bf16_gemm calls commit_and_wait() first, executing causal_mask +
  // softmax on the GPU, then runs the MPSGraph with the updated scores.
  sdpa_bf16_gemm(/*trans_b=*/false, seqlen_q, head_dim, seqlen_k,
                  scores, v_cont, out_cont);

  // Unpack out_cont → output[b,h] (non-contiguous: row stride = q_lda).
  for (ctranslate2::dim_t r = 0; r < seqlen_q; ++r) {
    std::memcpy(out_row0 + r * q_lda, out_cont + r * head_dim,
                static_cast<size_t>(head_dim) * sizeof(BF));
  }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point — declared in src/metal/ops_metal.h
// ---------------------------------------------------------------------------

namespace ctranslate2 {
  namespace metal {

    template <typename T>
    void sdpa_metal(const T* q, const T* k, const T* v, T* output,
                    dim_t batch_size, dim_t seqlen_q, dim_t seqlen_k,
                    dim_t num_heads, dim_t num_heads_k, dim_t head_dim,
                    float scale, bool is_causal) {
      const dim_t q_lda  = num_heads   * head_dim;
      const dim_t kv_lda = num_heads_k * head_dim;

      for (dim_t b = 0; b < batch_size; ++b) {
        for (dim_t h = 0; h < num_heads; ++h) {
          const dim_t hk = h % num_heads_k;

          const T* q_row0 = q      + (b * seqlen_q * num_heads   + h ) * head_dim;
          const T* k_row0 = k      + (b * seqlen_k * num_heads_k + hk) * head_dim;
          const T* v_row0 = v      + (b * seqlen_k * num_heads_k + hk) * head_dim;
          T*     out_row0 = output + (b * seqlen_q * num_heads   + h ) * head_dim;

          if constexpr (!std::is_same_v<T, bfloat16_t>) {
            sdpa_head_mps<T>(q_row0, k_row0, v_row0, out_row0,
                              q_lda, kv_lda, seqlen_q, seqlen_k, head_dim,
                              scale, is_causal);
          } else {
            sdpa_head_bf16(q_row0, k_row0, v_row0, out_row0,
                            q_lda, kv_lda, seqlen_q, seqlen_k, head_dim,
                            scale, is_causal);
          }
        }
      }
    }

    // Explicit instantiations — inside namespace ctranslate2::metal so that
    // unqualified float16_t / bfloat16_t resolve to ctranslate2:: versions.
    template void sdpa_metal<float>(
        const float*, const float*, const float*, float*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool);
    template void sdpa_metal<float16_t>(
        const float16_t*, const float16_t*, const float16_t*, float16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool);
    template void sdpa_metal<bfloat16_t>(
        const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool);

  }  // namespace metal
}  // namespace ctranslate2
