// src/metal/ops_sdpa.mm
//
// M6.1 — Scaled Dot-Product Attention for Device::MPS.
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

namespace {

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
//   Sets scores[row*seqlen_k + col] = large_neg  when  col > row + causal_offset.
//   H3: causal_offset is a parameter for future chunk-prefill support.
//   Currently always 0 (full prefill), but parameterized to avoid hardcoding.
template <typename T>
static void dispatch_causal_mask(T* scores,
                                  ctranslate2::dim_t seqlen_q,
                                  ctranslate2::dim_t seqlen_k,
                                  uint32_t causal_offset = 0u) {
  if (seqlen_q == 0 || seqlen_k == 0) {
    return;
  }
  char kname[kKernelNameBufSize];
  std::snprintf(kname, sizeof(kname), "causal_mask_%s", MetalTypeName<T>::value);
  id<MTLComputePipelineState> pso = get_sdpa_pso(kname);
  const ctranslate2::dim_t total = seqlen_q * seqlen_k;
  uint32_t sk     = ct2_u32(seqlen_k);
  uint32_t offset = causal_offset;
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
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
  [enc release];
}

// ---------------------------------------------------------------------------
// P2: Cached rowBytesForColumns — avoids ObjC message send per call.
// Same pattern as primitives_gemm.mm (M12.2).
// ---------------------------------------------------------------------------

static NSUInteger sdpa_cached_row_bytes(NSUInteger cols, MPSDataType dtype) {
  struct Key {
    NSUInteger cols;
    MPSDataType dtype;
    bool operator==(const Key& o) const { return cols == o.cols && dtype == o.dtype; }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<NSUInteger>()(k.cols) ^ (std::hash<uint32_t>()(k.dtype) << 16);
    }
  };
  static std::unordered_map<Key, NSUInteger, KeyHash> cache;

  Key key{cols, dtype};
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  NSUInteger rb;
  @autoreleasepool {
    rb = [MPSMatrixDescriptor rowBytesForColumns:cols dataType:dtype];
  }
  cache[key] = rb;
  return rb;
}

// ---------------------------------------------------------------------------
// P1: Cached MPSMatrixMultiplication — avoids ~15µs alloc+init per call.
// Same pattern as primitives_gemm.mm (M11.27).  SDPA only uses trans_a=false
// and beta=0, so the key is simpler.
// ---------------------------------------------------------------------------

struct SdpaGemmKey {
  bool transpose_b;
  NSUInteger m, n, k;
  uint64_t alpha_bits;

  bool operator==(const SdpaGemmKey& o) const {
    return transpose_b == o.transpose_b && m == o.m && n == o.n && k == o.k
        && alpha_bits == o.alpha_bits;
  }
};

struct SdpaGemmKeyHash {
  size_t operator()(const SdpaGemmKey& key) const {
    size_t h = 14695981039346656037ULL;
    h ^= std::hash<bool>()(key.transpose_b); h *= 1099511628211ULL;
    h ^= std::hash<NSUInteger>()(key.m);     h *= 1099511628211ULL;
    h ^= std::hash<NSUInteger>()(key.n);     h *= 1099511628211ULL;
    h ^= std::hash<NSUInteger>()(key.k);     h *= 1099511628211ULL;
    h ^= std::hash<uint64_t>()(key.alpha_bits); h *= 1099511628211ULL;
    return h;
  }
};

static std::unordered_map<SdpaGemmKey, MPSMatrixMultiplication*, SdpaGemmKeyHash> g_sdpa_gemm_cache;

static MPSMatrixMultiplication* get_cached_sdpa_gemm(
    bool transpose_b, NSUInteger m, NSUInteger n, NSUInteger k, double alpha) {
  uint64_t alpha_bits;
  std::memcpy(&alpha_bits, &alpha, sizeof(double));
  SdpaGemmKey key{transpose_b, m, n, k, alpha_bits};

  auto it = g_sdpa_gemm_cache.find(key);
  if (it != g_sdpa_gemm_cache.end()) return it->second;

  id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
  MPSMatrixMultiplication* op =
      [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                         transposeLeft:NO
                                        transposeRight:(BOOL)transpose_b
                                           resultRows:m
                                        resultColumns:n
                                      interiorColumns:k
                                                alpha:alpha
                                                 beta:0.0];
  g_sdpa_gemm_cache[key] = op;
  return op;
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

}  // close anonymous namespace for extern declarations

// Forward declarations for f16 conversion kernels (defined in primitives_gemm.mm).
// These are encode-only GPU kernels — no CPU/GPU sync required.
extern void encode_half_to_float32(
    id<MTLBuffer> src_buf, NSUInteger src_off, NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off, NSUInteger out_stride,
    uint32_t rows, uint32_t cols);
extern void encode_float32_to_half(
    id<MTLBuffer> src_buf, NSUInteger src_off, NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off, NSUInteger out_stride,
    uint32_t rows, uint32_t cols);

namespace {  // reopen anonymous namespace

// ---------------------------------------------------------------------------
// M14.2: Per-thread f32 temp buffer cache for SDPA f16 promoted GEMM.
// Separate from main GEMM cache since SDPA has different matrix sizes.
// ---------------------------------------------------------------------------
struct SdpaF16TempCache {
  id<MTLBuffer> buf[3] = {nil, nil, nil};   // A, B, C temps
  NSUInteger    cap[3] = {0, 0, 0};

  id<MTLBuffer> get(int idx, NSUInteger bytes) {
    if (bytes <= cap[idx]) {
      // M19: Zero stale data — same fix as F16TempCache in primitives_gemm.mm.
      memset([buf[idx] contents], 0, cap[idx]);
      return buf[idx];
    }
    if (buf[idx]) [buf[idx] release];
    buf[idx] = alloc_temp_buffer(bytes);
    cap[idx] = [buf[idx] length];
    return buf[idx];
  }
};

static thread_local SdpaF16TempCache _sdpa_f16_temp_cache;

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

  if constexpr (std::is_same_v<T, ctranslate2::float16_t>) {
    // -------------------------------------------------------------------
    // M14.2: Float16 SDPA GEMM with f32 accumulation.
    //
    // Matches CUDA Tensor Core behavior: f16 inputs → f32 accum → f16 output.
    // Three-phase encode-only pipeline (zero CPU/GPU syncs):
    //   1. GPU half→f32 conversion
    //   2. MPS f32 GEMM (hardware-optimized, f32 accumulation)
    //   3. GPU f32→half conversion
    // -------------------------------------------------------------------
    const NSUInteger rows_a = (NSUInteger)m, cols_a = (NSUInteger)k;
    const NSUInteger rows_b = trans_b ? (NSUInteger)n : (NSUInteger)k;
    const NSUInteger cols_b = trans_b ? (NSUInteger)k : (NSUInteger)n;

    // MPS row-byte alignment for float32 matrices.
    const NSUInteger mps_rb_a = sdpa_cached_row_bytes(cols_a, MPSDataTypeFloat32);
    const NSUInteger mps_rb_b = sdpa_cached_row_bytes(cols_b, MPSDataTypeFloat32);
    const NSUInteger mps_rb_c = sdpa_cached_row_bytes((NSUInteger)n, MPSDataTypeFloat32);

    const NSUInteger rb_a = std::max((NSUInteger)lda * sizeof(float), mps_rb_a);
    const NSUInteger rb_b = std::max((NSUInteger)ldb * sizeof(float), mps_rb_b);
    const NSUInteger rb_c = std::max((NSUInteger)n   * sizeof(float), mps_rb_c);

    // Per-thread f32 temp buffers (reused across GEMM calls).
    id<MTLBuffer> tmp_a = _sdpa_f16_temp_cache.get(0, rows_a * rb_a);
    id<MTLBuffer> tmp_b = _sdpa_f16_temp_cache.get(1, rows_b * rb_b);
    id<MTLBuffer> tmp_c = _sdpa_f16_temp_cache.get(2, (NSUInteger)m * rb_c);

    // Resolve input half buffers.
    NSUInteger off_a = 0, off_b = 0;
    id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
    id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

    // Protect input half buffers from premature reuse.
    ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
    ctranslate2::metal::protect_buffer_by_base([buf_b contents]);

    // Phase 1: GPU half→f32 conversion (encode-only).
    encode_half_to_float32(buf_a, off_a, (NSUInteger)lda,
                            tmp_a, 0, rb_a / sizeof(float),
                            ct2_u32(rows_a), ct2_u32(cols_a));
    encode_half_to_float32(buf_b, off_b, (NSUInteger)ldb,
                            tmp_b, 0, rb_b / sizeof(float),
                            ct2_u32(rows_b), ct2_u32(cols_b));

    // Phase 2: MPS f32 GEMM (encode-only, hardware-optimized).
    id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
    @autoreleasepool {
      MPSMatrixDescriptor* descA =
          [MPSMatrixDescriptor matrixDescriptorWithRows:rows_a
                                                columns:cols_a
                                               rowBytes:rb_a
                                               dataType:MPSDataTypeFloat32];
      MPSMatrixDescriptor* descB =
          [MPSMatrixDescriptor matrixDescriptorWithRows:rows_b
                                                columns:cols_b
                                               rowBytes:rb_b
                                               dataType:MPSDataTypeFloat32];
      MPSMatrixDescriptor* descC =
          [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                                columns:(NSUInteger)n
                                               rowBytes:rb_c
                                               dataType:MPSDataTypeFloat32];
      MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:tmp_a offset:0 descriptor:descA];
      MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:tmp_b offset:0 descriptor:descB];
      MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:tmp_c offset:0 descriptor:descC];

      MPSMatrixMultiplication* gemm_op =
          get_cached_sdpa_gemm(trans_b, (NSUInteger)m, (NSUInteger)n, (NSUInteger)k, (double)alpha);
      [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
      [matA release];
      [matB release];
      [matC release];
    }

    // Phase 3: GPU f32→half conversion (encode-only).
    NSUInteger off_c = 0;
    id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
    encode_float32_to_half(tmp_c, 0, rb_c / sizeof(float),
                            buf_c, off_c, (NSUInteger)ldc,
                            ct2_u32(m), ct2_u32(n));

    // Protect output buffer from premature reuse.
    ctranslate2::metal::protect_buffer_by_base([buf_c contents]);

    // Temp buffers owned by _sdpa_f16_temp_cache — NOT released here.

  } else {
    // -------------------------------------------------------------------
    // Float32 path: native MPS GEMM (already f32 accumulation).
    // -------------------------------------------------------------------
    const MPSDataType dtype = SdpaMPSDtype<T>::v;
    constexpr NSUInteger elem = sizeof(T);

    const NSUInteger rows_a = (NSUInteger)m, cols_a = (NSUInteger)k;
    const NSUInteger rows_b = trans_b ? (NSUInteger)n : (NSUInteger)k;
    const NSUInteger cols_b = trans_b ? (NSUInteger)k : (NSUInteger)n;
    const NSUInteger rows_c = (NSUInteger)m, cols_c = (NSUInteger)n;

    const NSUInteger nat_rb_a = static_cast<NSUInteger>(ct2_u32(lda)) * elem;
    const NSUInteger nat_rb_b = static_cast<NSUInteger>(ct2_u32(ldb)) * elem;
    const NSUInteger nat_rb_c = static_cast<NSUInteger>(ct2_u32(ldc)) * elem;

    // P2: Use cached rowBytesForColumns (avoids ObjC message send per call).
    const NSUInteger mps_rb_a = sdpa_cached_row_bytes(cols_a, dtype);
    const NSUInteger mps_rb_b = sdpa_cached_row_bytes(cols_b, dtype);
    const NSUInteger mps_rb_c = sdpa_cached_row_bytes(cols_c, dtype);

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
      // P1: Use cached MPSMatrixMultiplication (saves ~15µs alloc per call).
      MPSMatrixMultiplication* gemm_op =
          get_cached_sdpa_gemm(trans_b, (NSUInteger)m, (NSUInteger)n, (NSUInteger)k, (double)alpha);
      [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
      [matA release];
      [matB release];
      [matC release];
      // gemm_op is cached — do NOT release.
    }

    // If C required padding: flush, then unpack the padded temp back to c.
    if (pad_c) {
      CT2_COMMIT_AND_WAIT();
      const auto* src = static_cast<const uint8_t*>([tmp_c contents]);
      auto* dst = reinterpret_cast<uint8_t*>(c);
      for (NSUInteger r = 0; r < rows_c; ++r) {
        std::memcpy(dst + r * nat_rb_c, src + r * mps_rb_c, nat_rb_c);
      }
    }

    // Release temp buffers (command buffer retains them until GPU completes).
    if (tmp_a) [tmp_a release];
    if (tmp_b) [tmp_b release];
    if (tmp_c) [tmp_c release];
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
  CT2_COMMIT_AND_WAIT();

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
    [tdA release];
    [tdB release];
  }

  // Release temp buffers (MPSGraph ran synchronously, GPU is done).
  if (tmp_a) [tmp_a release];
  if (tmp_b) [tmp_b release];
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

  // H2: Protect scores buffer from premature pool reuse.  The MetalTempBuf
  // destructor returns it to the pool when this function exits, but the
  // encode-only GPU work (GEMM → mask → softmax → GEMM) hasn't executed yet.
  ctranslate2::metal::protect_buffer_by_base(scores);

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
    NSUInteger mps_min = sdpa_cached_row_bytes((NSUInteger)seqlen_k, SdpaMPSDtype<T>::v);
    if ((NSUInteger)seqlen_k * sizeof(T) < mps_min) {
      CT2_COMMIT_AND_WAIT();
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

// ---------------------------------------------------------------------------
// Fused SDPA decode — single MSL kernel for sq==1 (FP32 / FP16).
//
// Replaces the per-head MPS GEMM loop (2 GEMMs × num_heads per layer) with
// one kernel dispatch for all (batch, head) pairs.  Reduces ObjC overhead
// from 1408 MPS GEMM calls/step to 22 kernel dispatches (one per layer).
//
// The kernel uses threadgroup memory for softmax scores (always float32
// regardless of T), limiting max seqlen_k to 32768/sizeof(float) = 8192.
// Falls back to the per-head MPS GEMM path for larger sequences.
// ---------------------------------------------------------------------------

// FusedSdpaDecodeParams must match the MSL struct layout in kSdpaMSL.
struct FusedSdpaDecodeParams {
  uint32_t heads_per_kv;
  uint32_t head_dim;
  uint32_t seqlen_k;
  float    scale;
  uint32_t q_row_elems;
  uint32_t kv_row_elems;
  uint32_t kv_batch_stride;
  uint32_t beam_size;
};

template <typename T>
static void dispatch_fused_sdpa_decode(
    const T* q, const T* k, const T* v, T* output,
    ctranslate2::dim_t batch_size,
    ctranslate2::dim_t seqlen_k,
    ctranslate2::dim_t num_heads,
    ctranslate2::dim_t num_heads_k,
    ctranslate2::dim_t head_dim,
    float scale,
    ctranslate2::dim_t kv_batch_stride,
    ctranslate2::dim_t beam_size) {

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, ctranslate2::float16_t>,
                "Fused SDPA decode only supports float32 and float16");

  const char* kname = std::is_same_v<T, float>
      ? "fused_sdpa_decode_float" : "fused_sdpa_decode_half";
  id<MTLComputePipelineState> pso = get_sdpa_pso(kname);

  FusedSdpaDecodeParams params;
  params.heads_per_kv    = ct2_u32(num_heads / num_heads_k);
  params.head_dim        = ct2_u32(head_dim);
  params.seqlen_k        = ct2_u32(seqlen_k);
  params.scale           = scale;
  params.q_row_elems     = ct2_u32(num_heads * head_dim);
  params.kv_row_elems    = ct2_u32(num_heads_k * head_dim);
  params.kv_batch_stride = ct2_u32(kv_batch_stride);
  params.beam_size       = ct2_u32(beam_size);

  NSUInteger off_q = 0, off_k = 0, off_v = 0, off_o = 0;
  id<MTLBuffer> buf_q = ctranslate2::metal_buffer_for_ptr(q, &off_q);
  id<MTLBuffer> buf_k = ctranslate2::metal_buffer_for_ptr(k, &off_k);
  id<MTLBuffer> buf_v = ctranslate2::metal_buffer_for_ptr(v, &off_v);
  id<MTLBuffer> buf_o = ctranslate2::metal_buffer_for_ptr(output, &off_o);

  // H1: Protect input buffers from premature reuse (encode-only dispatch).
  ctranslate2::metal::protect_buffer_by_base([buf_q contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_k contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_v contents]);

  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:buf_q offset:off_q atIndex:0];
  [enc setBuffer:buf_k offset:off_k atIndex:1];
  [enc setBuffer:buf_v offset:off_v atIndex:2];
  [enc setBuffer:buf_o offset:off_o atIndex:3];
  [enc setBytes:&params length:sizeof(params) atIndex:4];

  // Threadgroup memory index 0: softmax scores (seqlen_k floats).
  // Scores are always float in the kernel, regardless of T.
  // Metal requires threadgroup memory length to be a multiple of 16 bytes.
  NSUInteger tg_scores_bytes = static_cast<NSUInteger>(seqlen_k) * sizeof(float);
  tg_scores_bytes = (tg_scores_bytes + 15) & ~NSUInteger(15);
  [enc setThreadgroupMemoryLength:tg_scores_bytes atIndex:0];

  // Threadgroup memory index 1: reduction scratch (kTgSize floats).
  // Must match [[threadgroup(1)]] in the MSL kernel.
  constexpr NSUInteger kTgSize = 256;
  NSUInteger tg_reduce_bytes = kTgSize * sizeof(float);
  [enc setThreadgroupMemoryLength:tg_reduce_bytes atIndex:1];
  [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(batch_size),
                                         static_cast<NSUInteger>(num_heads), 1)
       threadsPerThreadgroup:MTLSizeMake(kTgSize, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// Maximum seqlen_k for fused decode (threadgroup memory limit).
static constexpr ctranslate2::dim_t kFusedSdpaMaxSk = 8192;

// ---------------------------------------------------------------------------
// CPU fast-path for small SDPA.
//
// Handles any sq/sk. Per (batch, head, query_pos i):
//   score[j] = scale * dot(Q[i], K[j])   j in [0, sk)
//   if (is_causal && j > i) score[j] = -inf
//   prob     = softmax(score)
//   out[i]   = prob @ V
//
// This avoids MPS GEMM entirely — no padding, no command buffers, no commits.
// CPU SDPA is fast for small sq*sk (e.g. encoder with 3 tokens: sq=sk=3).
// MPS GEMM with tiny matrices triggers commit_and_wait per head (~0.4 ms).
// ---------------------------------------------------------------------------

template <typename T>
static void sdpa_cpu(const T* q, const T* k, const T* v, T* output,
                     ctranslate2::dim_t batch_size,
                     ctranslate2::dim_t seqlen_q,
                     ctranslate2::dim_t seqlen_k,
                     ctranslate2::dim_t num_heads,
                     ctranslate2::dim_t num_heads_k,
                     ctranslate2::dim_t head_dim,
                     float scale, bool is_causal,
                     ctranslate2::dim_t kv_batch_stride,
                     ctranslate2::dim_t beam_size = 1) {
  const ctranslate2::dim_t q_lda  = num_heads   * head_dim;
  const ctranslate2::dim_t kv_lda = num_heads_k * head_dim;
  const ctranslate2::dim_t kv_bstride = (kv_batch_stride > 0)
                                        ? kv_batch_stride
                                        : seqlen_k * num_heads_k * head_dim;

  // Stack-allocated score buffer. For sk > 8192, fall back to heap.
  constexpr ctranslate2::dim_t kStackLimit = 8192;
  float stack_scores[kStackLimit];
  std::unique_ptr<float[]> heap_scores;
  float* scores = stack_scores;
  if (seqlen_k > kStackLimit) {
    heap_scores.reset(new float[seqlen_k]);
    scores = heap_scores.get();
  }

  for (ctranslate2::dim_t b = 0; b < batch_size; ++b) {
    const ctranslate2::dim_t kv_b = b / beam_size;  // K/V batch broadcasting
    const ctranslate2::dim_t heads_per_kv = num_heads / num_heads_k;
    for (ctranslate2::dim_t h = 0; h < num_heads; ++h) {
      const ctranslate2::dim_t hk = h / heads_per_kv;

      const T* k_base = k + kv_b * kv_bstride + hk * head_dim;
      const T* v_base = v + kv_b * kv_bstride + hk * head_dim;

      for (ctranslate2::dim_t qi = 0; qi < seqlen_q; ++qi) {
        // Q layout: [batch, sq, num_heads, head_dim]
        const T* q_ptr = q + (b * seqlen_q * q_lda) + qi * q_lda + h * head_dim;
        T* out_ptr = output + (b * seqlen_q * q_lda) + qi * q_lda + h * head_dim;

        // Step 1: scores = scale * Q[qi] · K^T
        float max_score = -1e30f;
        for (ctranslate2::dim_t j = 0; j < seqlen_k; ++j) {
          const T* k_row = k_base + j * kv_lda;
          float dot = 0.0f;
          for (ctranslate2::dim_t d = 0; d < head_dim; ++d)
            dot += float(q_ptr[d]) * float(k_row[d]);
          float s = dot * scale;
          if (is_causal && j > qi)
            s = -1e30f;
          scores[j] = s;
          if (s > max_score) max_score = s;
        }

        // Step 2: softmax
        float sum_exp = 0.0f;
        for (ctranslate2::dim_t j = 0; j < seqlen_k; ++j) {
          scores[j] = std::exp(scores[j] - max_score);
          sum_exp += scores[j];
        }
        const float inv_sum = 1.0f / sum_exp;
        for (ctranslate2::dim_t j = 0; j < seqlen_k; ++j)
          scores[j] *= inv_sum;

        // Step 3: out = prob @ V
        for (ctranslate2::dim_t d = 0; d < head_dim; ++d) {
          float acc = 0.0f;
          for (ctranslate2::dim_t j = 0; j < seqlen_k; ++j)
            acc += scores[j] * float(v_base[j * kv_lda + d]);
          out_ptr[d] = T(acc);
        }
      }
    }
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
                    float scale, bool is_causal,
                    dim_t kv_batch_stride,
                    dim_t beam_size) {
      // CPU fast-path: avoids per-head MPS GEMM overhead.
      // For decode (sq=1), CPU SDPA with float32 accumulation is ~100x
      // faster than 32 per-head MPS GEMMs (each with alloc/encode overhead).
      // The CT2_COMMIT_AND_WAIT cost (~0.4ms × 22 layers) is far less than
      // the per-head GPU loop (32 × 22 × ~50µs GEMM overhead).
      // Also covers short prefill (sq*sk ≤ 32).
      //
      // Exception: for float32, CPU SDPA produces slightly different results
      // from GPU GEMM due to different accumulation order.  These differences
      // compound through 22 transformer layers and cause token divergence
      // vs the standard attention path (which uses GPU GEMM).  For f16,
      // rounding masks the differences.  So f32 always uses GPU SDPA to
      // match the standard path's numerical behavior.
      // CPU fast-path for bf16 and small prefill (not f32/f16 decode, which
      // uses the fused GPU kernel below).
      constexpr dim_t kCpuSdpaThresh = 32;
      const bool is_f32_or_f16 = std::is_same_v<T, float> || std::is_same_v<T, float16_t>;
      if (!is_f32_or_f16 && (seqlen_q == 1 || seqlen_q * seqlen_k <= kCpuSdpaThresh)) {
        // Flush any pending GPU writes so CPU can read Q/K/V.
        CT2_COMMIT_AND_WAIT();
        sdpa_cpu<T>(q, k, v, output, batch_size, seqlen_q, seqlen_k,
                    num_heads, num_heads_k, head_dim,
                    scale, is_causal, kv_batch_stride, beam_size);
        return;
      }

      const dim_t kv_bstride = (kv_batch_stride > 0)
                                   ? kv_batch_stride
                                   : seqlen_k * num_heads_k * head_dim;

      // Fused decode kernel (f32/f16): single dispatch for all (batch, head)
      // pairs.  Replaces per-head MPS GEMM loop (1408 ObjC calls/step → 22
      // kernel dispatches) and CPU SDPA (22 commits/step → 0 commits).
      // The kernel uses float32 accumulation for both f32 and f16 types.
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float16_t>) {
        if (seqlen_q == 1 && seqlen_k <= kFusedSdpaMaxSk) {
          dispatch_fused_sdpa_decode<T>(
              q, k, v, output,
              batch_size, seqlen_k, num_heads, num_heads_k, head_dim,
              scale, kv_bstride, beam_size);
          return;
        }
      }

      // Fallback: per-head SDPA (prefill, large sk, or bf16).
      const dim_t q_lda  = num_heads   * head_dim;
      const dim_t kv_lda = num_heads_k * head_dim;

      // Periodic flush for f32/f16: each sdpa_head_mps encodes ~4 GPU command
      // encoders (2 MPS GEMMs + causal_mask + softmax).  Metal's resource
      // tracking becomes unreliable beyond ~128 encoders per command buffer,
      // causing silent data corruption (manifests as BLEU degradation for
      // batch_size > 4 with 8 heads).  Flush every kSdpaFlushInterval heads
      // to keep encoder count safely under the limit.
      // BF16 is unaffected — sdpa_bf16_gemm already flushes per GEMM.
      constexpr dim_t kSdpaFlushInterval = 16;  // ~64 encoders between flushes
      dim_t sdpa_head_count = 0;

      for (dim_t b = 0; b < batch_size; ++b) {
        const dim_t kv_b = b / beam_size;  // K/V batch broadcasting
        const dim_t heads_per_kv = num_heads / num_heads_k;
        for (dim_t h = 0; h < num_heads; ++h) {
          const dim_t hk = h / heads_per_kv;

          const T* q_row0 = q      + (b * seqlen_q * num_heads   + h ) * head_dim;
          const T* k_row0 = k      + kv_b * kv_bstride + hk * head_dim;
          const T* v_row0 = v      + kv_b * kv_bstride + hk * head_dim;
          T*     out_row0 = output + (b * seqlen_q * num_heads   + h ) * head_dim;

          if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float16_t>) {
            sdpa_head_mps<T>(q_row0, k_row0, v_row0, out_row0,
                              q_lda, kv_lda, seqlen_q, seqlen_k, head_dim,
                              scale, is_causal);
            ++sdpa_head_count;
            if (sdpa_head_count >= kSdpaFlushInterval) {
              CT2_COMMIT_AND_WAIT();
              sdpa_head_count = 0;
            }
          } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            sdpa_head_bf16(q_row0, k_row0, v_row0, out_row0,
                            q_lda, kv_lda, seqlen_q, seqlen_k, head_dim,
                            scale, is_causal);
          } else {
            throw std::runtime_error("sdpa_metal: integer types are not supported");
          }
        }
      }
    }

    // Explicit instantiations — inside namespace ctranslate2::metal so that
    // unqualified float16_t / bfloat16_t resolve to ctranslate2:: versions.
    template void sdpa_metal<float>(
        const float*, const float*, const float*, float*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);
    template void sdpa_metal<float16_t>(
        const float16_t*, const float16_t*, const float16_t*, float16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);
    template void sdpa_metal<bfloat16_t>(
        const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);
    // TYPE_DISPATCH in flash_attention_metal.mm generates branches for all types.
    // These int instantiations satisfy the linker but throw at runtime — SDPA only
    // supports float/float16/bfloat16.  Changing this would require modifying the
    // TYPE_DISPATCH macro infrastructure, which is shared across all ops.
    template void sdpa_metal<int8_t>(
        const int8_t*, const int8_t*, const int8_t*, int8_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);
    template void sdpa_metal<int16_t>(
        const int16_t*, const int16_t*, const int16_t*, int16_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);
    template void sdpa_metal<int32_t>(
        const int32_t*, const int32_t*, const int32_t*, int32_t*,
        dim_t, dim_t, dim_t, dim_t, dim_t, dim_t, float, bool, dim_t, dim_t);

    void clear_sdpa_gemm_cache() {
      for (auto& [key, op] : g_sdpa_gemm_cache)
        [op release];
      g_sdpa_gemm_cache.clear();
    }

  }  // namespace metal
}  // namespace ctranslate2
