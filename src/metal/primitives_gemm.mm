// src/metal/primitives_gemm.mm
//
// M4.4 — GEMM primitives for Device::MPS.
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
// M12.6: GPU kernels for INT8 GEMM — eliminates 2 CPU/GPU syncs per GEMM.
//
// int8_to_float32_strided:
//   Reads int8 matrix [rows, cols] with element stride in_stride,
//   writes float32 matrix with element stride out_stride.
//
// float32_round_to_int32_strided:
//   Reads float32 matrix [rows, cols] with element stride in_stride,
//   rounds each element to nearest int32, writes with element stride out_stride.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// M12.14: Fused INT8 GEMV kernel — reads int8 A and B directly, accumulates
// in int32 (exact for k ≤ 133K, vs f32 exact only for k ≤ 1040), outputs int32.
//
// For decode (m=1, trans_b=true):
//   A is [1, k] int8 (row vector)
//   B is [n, k] int8 (weight matrix, transposed)
//   C is [1, n] int32
//   c[j] = round(alpha * float(sum_i(int(a[i]) * int(b[j, i]))))
//
// Uses char4 vectorized reads for 4× bandwidth efficiency.
// Each thread computes one output element.
// ---------------------------------------------------------------------------

static constexpr const char* kFusedInt8GemvMSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void fused_int8_gemv(
    device const char*  a      [[buffer(0)]],   // [1, k] int8
    device const char*  b      [[buffer(1)]],   // [n, k] int8 (row-major)
    device int*         c      [[buffer(2)]],   // [1, n] int32 output
    constant uint2&     params [[buffer(3)]],   // {n, k}
    constant float&     alpha  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint k = params[1];
    if (gid >= n) return;

    // Pointer to B row for this output element
    device const char* b_row = b + gid * k;

    // Vectorized accumulation using char4 (4 int8 values at once)
    int acc = 0;
    const uint k4 = k / 4;
    device const char4* a4 = (device const char4*)a;
    device const char4* b4 = (device const char4*)b_row;

    for (uint i = 0; i < k4; ++i) {
        char4 av = a4[i];
        char4 bv = b4[i];
        acc += int(av[0]) * int(bv[0])
             + int(av[1]) * int(bv[1])
             + int(av[2]) * int(bv[2])
             + int(av[3]) * int(bv[3]);
    }

    // Handle remaining elements (k not divisible by 4)
    for (uint i = k4 * 4; i < k; ++i) {
        acc += int(a[i]) * int(b_row[i]);
    }

    c[gid] = int(floor(float(acc) * alpha + 0.5f));
}
)";

static id<MTLLibrary> get_fused_int8_gemv_library() {
  static id<MTLLibrary>  lib  = nil;
  static std::once_flag  flag;
  return compile_library_once(flag, lib, kFusedInt8GemvMSL, "fused_int8_gemv");
}

static id<MTLComputePipelineState> get_fused_int8_gemv_pso() {
  static PSOCache cache;
  return cache.get(get_fused_int8_gemv_library, "fused_int8_gemv");
}

// Dispatch the fused INT8 GEMV kernel for m=1, trans_b=true.
// Reads int8 A[1,k] and B[n,k] directly — no f32 temp buffers needed.
// ~7× less memory bandwidth than the 3-kernel path for large weight matrices.
static void dispatch_fused_int8_gemv(
    ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const int8_t* a, ctranslate2::dim_t lda,
    const int8_t* b, ctranslate2::dim_t ldb,
    int32_t* c, ctranslate2::dim_t ldc) {
  id<MTLComputePipelineState> pso = get_fused_int8_gemv_pso();
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);

  // Protect A and B from premature reuse (encode-only).
  ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_b contents]);

  [enc setBuffer:buf_a offset:off_a atIndex:0];
  [enc setBuffer:buf_b offset:off_b atIndex:1];
  [enc setBuffer:buf_c offset:off_c atIndex:2];

  const uint32_t params[2] = { ct2_u32(n), ct2_u32(k) };
  [enc setBytes:params length:sizeof(params) atIndex:3];
  [enc setBytes:&alpha length:sizeof(alpha) atIndex:4];

  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

static constexpr const char* kInt8GemmHelperMSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void int8_to_float32_strided(
    device const char* input  [[buffer(0)]],
    device float*      output [[buffer(1)]],
    constant uint4&    params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rows = params[0], cols = params[1];
    uint in_stride = params[2], out_stride = params[3];
    if (gid >= rows * cols) return;
    uint row = gid / cols, col = gid % cols;
    output[row * out_stride + col] = float(input[row * in_stride + col]);
}

kernel void float32_round_to_int32_strided(
    device const float* input  [[buffer(0)]],
    device int*         output [[buffer(1)]],
    constant uint4&     params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rows = params[0], cols = params[1];
    uint in_stride = params[2], out_stride = params[3];
    if (gid >= rows * cols) return;
    uint row = gid / cols, col = gid % cols;
    output[row * out_stride + col] = int(floor(input[row * in_stride + col] + 0.5f));
}
)";

static id<MTLLibrary> get_int8_gemm_helper_library() {
  static id<MTLLibrary>  lib  = nil;
  static std::once_flag  flag;
  return compile_library_once(flag, lib, kInt8GemmHelperMSL, "int8_gemm_helper");
}

static id<MTLComputePipelineState> get_int8_helper_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_int8_gemm_helper_library, name);
}

// Encode int8→float32 conversion on GPU (encode-only, no sync).
// src_buf/src_off: MTLBuffer + byte offset for int8 data.
// dst_buf: MTLBuffer for float32 output (offset 0 or specified).
static void encode_int8_to_float32(
    id<MTLBuffer> src_buf, NSUInteger src_off,
    NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off,
    NSUInteger out_stride,
    uint32_t rows, uint32_t cols) {
  id<MTLComputePipelineState> pso =
      get_int8_helper_pso("int8_to_float32_strided");
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_off atIndex:0];
  [enc setBuffer:dst_buf offset:dst_off atIndex:1];
  // M12 review M2: Use ct2_u32() for checked narrowing (project convention).
  const uint32_t params[4] = { rows, cols, ct2_u32(in_stride), ct2_u32(out_stride) };
  [enc setBytes:params length:sizeof(params) atIndex:2];
  const NSUInteger total = (NSUInteger)rows * cols;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// Encode float32→int32 rounding on GPU (encode-only, no sync).
static void encode_float32_to_int32(
    id<MTLBuffer> src_buf, NSUInteger src_off,
    NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off,
    NSUInteger out_stride,
    uint32_t rows, uint32_t cols) {
  id<MTLComputePipelineState> pso =
      get_int8_helper_pso("float32_round_to_int32_strided");
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_off atIndex:0];
  [enc setBuffer:dst_buf offset:dst_off atIndex:1];
  const uint32_t params[4] = { rows, cols, ct2_u32(in_stride), ct2_u32(out_stride) };
  [enc setBytes:params length:sizeof(params) atIndex:2];
  const NSUInteger total = (NSUInteger)rows * cols;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

}  // close anonymous namespace — f16 conversion functions need external linkage
   // for ops_sdpa.mm to call encode_half_to_float32 / encode_float32_to_half.

// ---------------------------------------------------------------------------
// M13: GPU kernels for FP16 precision support.
//
// 1. half_to_float32_strided / float32_to_half_strided:
//    Conversion kernels for the float16 promotion path.  Used by both
//    dispatch_f16_promoted_gemm (main GEMM) and ops_sdpa.mm (SDPA path).
//
// 2. gemm_f16_acc32:
//    Custom MSL GEMM kernel (RETAINED for potential future use).
//    The main f16 GEMM path now uses dispatch_f16_promoted_gemm which
//    promotes half→f32, runs MPS GEMM, demotes f32→half — all encode-only,
//    zero syncs.  This follows the proven INT8 GEMM pattern.
// ---------------------------------------------------------------------------

static constexpr const char* kF16ConvertMSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void half_to_float32_strided(
    device const half* input  [[buffer(0)]],
    device float*      output [[buffer(1)]],
    constant uint4&    params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rows = params[0], cols = params[1];
    uint in_stride = params[2], out_stride = params[3];
    if (gid >= rows * cols) return;
    uint row = gid / cols, col = gid % cols;
    output[row * out_stride + col] = float(input[row * in_stride + col]);
}

kernel void float32_to_half_strided(
    device const float* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    constant uint4&     params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rows = params[0], cols = params[1];
    uint in_stride = params[2], out_stride = params[3];
    if (gid >= rows * cols) return;
    uint row = gid / cols, col = gid % cols;
    output[row * out_stride + col] = half(input[row * in_stride + col]);
}
)";

// ---------------------------------------------------------------------------
// M13: Tiled GEMM — half inputs, float32 accumulation, half output.
//
// C[M,N] = alpha * op(A) * op(B) + beta * C
// where op(X) = X or X^T depending on transpose flags.
//
// Uses simdgroup_matrix for hardware-accelerated 8×8 matrix multiply-
// accumulate with float32 accumulation.  Each SIMD group computes an 8×8
// tile of C.  4 SIMD groups per threadgroup → 16×16 output tile.
//
// Handles arbitrary M, N, K (boundary masking for non-8-aligned dims).
// ---------------------------------------------------------------------------
static constexpr const char* kGemmF16Acc32MSL = R"(
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TILE = 8;  // simdgroup_matrix tile size

// Tiled GEMM: 4 SIMD groups per threadgroup, each computing 8×8 of C.
// Threadgroup output tile: 16×16 (2×2 arrangement of 8×8 tiles).
// Dispatch: threadgroups = (ceil(N/16), ceil(M/16), 1),
//           threadsPerThreadgroup = 128 (4 simdgroups × 32).
// params: M, N, K, lda, ldb, ldc, ta, tb
kernel void gemm_f16_acc32(
    device const half*  A       [[buffer(0)]],
    device const half*  B       [[buffer(1)]],
    device       half*  C       [[buffer(2)]],
    constant     uint*  params  [[buffer(3)]],
    constant     float& alpha   [[buffer(4)]],
    constant     float& beta    [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint  sg_id    [[simdgroup_index_in_threadgroup]],
    uint  lane     [[thread_index_in_simdgroup]])
{
    uint M   = params[0], N   = params[1], K   = params[2];
    uint lda = params[3], ldb = params[4], ldc = params[5];
    uint ta  = params[6], tb  = params[7];

    // This SIMD group's 8×8 output tile position.
    uint sg_row = sg_id / 2;   // 0 or 1
    uint sg_col = sg_id % 2;   // 0 or 1
    uint base_row = group_id.y * 16 + sg_row * TILE;
    uint base_col = group_id.x * 16 + sg_col * TILE;

    // Skip if fully out of bounds.
    if (base_row >= M || base_col >= N) return;

    simdgroup_matrix<float, 8, 8> acc(0);

    // Aligned K loop: process 8 columns at a time.
    uint K_aligned = K & ~7u;
    for (uint kk = 0; kk < K_aligned; kk += TILE) {
        simdgroup_matrix<half, 8, 8> a_mat, b_mat;

        // Load op(A)[base_row:+8, kk:+8]
        if (!ta)
            simdgroup_load(a_mat, A + (ulong)base_row * lda + kk, lda);
        else
            simdgroup_load(a_mat, A + (ulong)kk * lda + base_row, lda,
                           ulong2(0,0), true);

        // Load op(B)[kk:+8, base_col:+8]
        if (!tb)
            simdgroup_load(b_mat, B + (ulong)kk * ldb + base_col, ldb);
        else
            simdgroup_load(b_mat, B + (ulong)base_col * ldb + kk, ldb,
                           ulong2(0,0), true);

        simdgroup_multiply_accumulate(acc, a_mat, b_mat, acc);
    }

    // Remainder K (K % 8 != 0): scalar fallback per thread.
    if (K_aligned < K) {
        // Each of the 32 lanes handles ~2 of the 64 output elements.
        for (uint i = lane; i < TILE * TILE; i += 32) {
            uint r = i / TILE;
            uint c = i % TILE;
            uint out_r = base_row + r;
            uint out_c = base_col + c;
            if (out_r < M && out_c < N) {
                float partial = 0.0f;
                for (uint kk = K_aligned; kk < K; ++kk) {
                    float a_val = ta
                        ? (float)A[(ulong)kk * lda + out_r]
                        : (float)A[(ulong)out_r * lda + kk];
                    float b_val = tb
                        ? (float)B[(ulong)out_c * ldb + kk]
                        : (float)B[(ulong)kk * ldb + out_c];
                    partial += a_val * b_val;
                }
                // Add to accumulator via threadgroup memory.
                // For simplicity, just add directly — the simdgroup_store
                // below will pick up the full acc.  We handle the
                // remainder in the store phase instead.
            }
        }
        // Note: for the K-remainder case, we accumulate the remainder
        // contribution during the store phase below.
    }

    // Store: extract from acc, apply alpha/beta, write to C.
    // Use threadgroup memory as intermediate for the 8×8 float tile.
    threadgroup float tg_acc[4][TILE * TILE];  // one per SIMD group
    simdgroup_store(acc, &tg_acc[sg_id][0], TILE);

    // Each of the 32 lanes writes ~2 elements.
    for (uint i = lane; i < TILE * TILE; i += 32) {
        uint r = i / TILE;
        uint c = i % TILE;
        uint out_r = base_row + r;
        uint out_c = base_col + c;
        if (out_r < M && out_c < N) {
            float val = tg_acc[sg_id][i];

            // Add K-remainder contribution if needed.
            if (K_aligned < K) {
                float partial = 0.0f;
                for (uint kk = K_aligned; kk < K; ++kk) {
                    float a_val = ta
                        ? (float)A[(ulong)kk * lda + out_r]
                        : (float)A[(ulong)out_r * lda + kk];
                    float b_val = tb
                        ? (float)B[(ulong)out_c * ldb + kk]
                        : (float)B[(ulong)kk * ldb + out_c];
                    partial += a_val * b_val;
                }
                val += partial;
            }

            val *= alpha;
            if (beta != 0.0f)
                val += beta * (float)C[(ulong)out_r * ldc + out_c];
            C[(ulong)out_r * ldc + out_c] = (half)val;
        }
    }
}
)";

static id<MTLLibrary> get_f16_convert_library() {
  static id<MTLLibrary>  lib  = nil;
  static std::once_flag  flag;
  return compile_library_once(flag, lib, kF16ConvertMSL, "f16_convert");
}

static id<MTLComputePipelineState> get_f16_convert_pso(const char* name) {
  static PSOCache cache;
  return cache.get(get_f16_convert_library, name);
}

static id<MTLLibrary> get_gemm_f16_acc32_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kGemmF16Acc32MSL, "gemm_f16_acc32");
}

static id<MTLComputePipelineState> get_gemm_f16_acc32_pso() {
  static PSOCache cache;
  return cache.get(get_gemm_f16_acc32_library, "gemm_f16_acc32");
}

// Encode half→float32 conversion on GPU (encode-only, no sync).
// Not static — also called from ops_sdpa.mm for SDPA float16 promotion.
void encode_half_to_float32(
    id<MTLBuffer> src_buf, NSUInteger src_off,
    NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off,
    NSUInteger out_stride,
    uint32_t rows, uint32_t cols) {
  id<MTLComputePipelineState> pso =
      get_f16_convert_pso("half_to_float32_strided");
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_off atIndex:0];
  [enc setBuffer:dst_buf offset:dst_off atIndex:1];
  const uint32_t params[4] = { rows, cols, ct2_u32(in_stride), ct2_u32(out_stride) };
  [enc setBytes:params length:sizeof(params) atIndex:2];
  const NSUInteger total = (NSUInteger)rows * cols;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

// Encode float32→half conversion on GPU (encode-only, no sync).
// Not static — also called from ops_sdpa.mm for SDPA float16 promotion.
void encode_float32_to_half(
    id<MTLBuffer> src_buf, NSUInteger src_off,
    NSUInteger in_stride,
    id<MTLBuffer> dst_buf, NSUInteger dst_off,
    NSUInteger out_stride,
    uint32_t rows, uint32_t cols) {
  id<MTLComputePipelineState> pso =
      get_f16_convert_pso("float32_to_half_strided");
  id<MTLComputeCommandEncoder> enc =
      ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];
  [enc setBuffer:src_buf offset:src_off atIndex:0];
  [enc setBuffer:dst_buf offset:dst_off atIndex:1];
  const uint32_t params[4] = { rows, cols, ct2_u32(in_stride), ct2_u32(out_stride) };
  [enc setBytes:params length:sizeof(params) atIndex:2];
  const NSUInteger total = (NSUInteger)rows * cols;
  const NSUInteger tpg = std::min((NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreads:MTLSizeMake(total, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];
  [enc release];
}

namespace {  // reopen anonymous namespace

// ---------------------------------------------------------------------------
// MPS data type mapping (FP32 and FP16 only — BF16 uses MPSGraph)
// ---------------------------------------------------------------------------

template <typename T> struct MPS_Dtype;
template<> struct MPS_Dtype<float>                  { static const MPSDataType value = MPSDataTypeFloat32; };
template<> struct MPS_Dtype<ctranslate2::float16_t> { static const MPSDataType value = MPSDataTypeFloat16; };

// ---------------------------------------------------------------------------
// M12.2: Cached rowBytesForColumns — avoids repeated ObjC message sends.
//
// rowBytesForColumns:dataType: is a pure function of (cols, dtype) for a
// given Metal device.  Caching eliminates ~15 ObjC calls per GEMM dispatch
// and the associated @autoreleasepool blocks.
// ---------------------------------------------------------------------------

static NSUInteger cached_row_bytes(NSUInteger cols, MPSDataType dtype) {
  // Small inline cache: most GEMM shapes reuse a handful of (cols, dtype) pairs.
  // Single-threaded GPU work means no contention on the mutex.
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
  static std::mutex mtx;

  Key key{cols, dtype};
  std::lock_guard<std::mutex> lk(mtx);
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
// M11.27 — MPSMatrixMultiplication cache
//
// MPSMatrixMultiplication alloc+init performs kernel selection internally,
// costing ~10-20µs per call.  With 76K+ GEMM dispatches per whisper inference,
// this adds up.  The object only depends on (transpose, m, n, k, alpha, beta)
// and can be reused across encodes with different buffers.
//
// For batched GEMMs, batchSize is included in the key to avoid mutation races
// if multiple threads share the cache.
// ---------------------------------------------------------------------------

struct MpsGemmKey {
  bool transpose_a;
  bool transpose_b;
  NSUInteger m, n, k;
  uint64_t alpha_bits;
  uint64_t beta_bits;
  NSUInteger batch_size;  // 0 for non-batched

  bool operator==(const MpsGemmKey& o) const {
    return transpose_a == o.transpose_a && transpose_b == o.transpose_b &&
           m == o.m && n == o.n && k == o.k &&
           alpha_bits == o.alpha_bits && beta_bits == o.beta_bits &&
           batch_size == o.batch_size;
  }
};

struct MpsGemmKeyHash {
  size_t operator()(const MpsGemmKey& key) const {
    // FNV-1a inspired mixing
    size_t h = 14695981039346656037ULL;
    h ^= (size_t)key.transpose_a; h *= 1099511628211ULL;
    h ^= (size_t)key.transpose_b; h *= 1099511628211ULL;
    h ^= key.m;                   h *= 1099511628211ULL;
    h ^= key.n;                   h *= 1099511628211ULL;
    h ^= key.k;                   h *= 1099511628211ULL;
    h ^= key.alpha_bits;          h *= 1099511628211ULL;
    h ^= key.beta_bits;           h *= 1099511628211ULL;
    h ^= key.batch_size;          h *= 1099511628211ULL;
    return h;
  }
};

// Returns a cached (or newly created) MPSMatrixMultiplication.
// The returned pointer is owned by the cache — caller must NOT release it.
//
// Thread safety: the cache uses a global mutex for lookup/insert, but the
// returned MPSMatrixMultiplication* is used WITHOUT a lock for encoding.
// Apple documents that MPSKernel subclasses are NOT thread-safe for
// concurrent encodeToCommandBuffer: calls.  CTranslate2 runs single-threaded
// per translator for GPU work, so concurrent encode on the same object does
// not occur.  A thread_local cache was tested but adds ~90ms overhead per
// inference due to TLS initialization costs — not worth it for a latent issue.

static std::unordered_map<MpsGemmKey, MPSMatrixMultiplication*, MpsGemmKeyHash> g_mps_gemm_cache;
static std::mutex g_mps_gemm_mutex;

static MPSMatrixMultiplication* get_cached_mps_gemm(
    bool transpose_a, bool transpose_b,
    NSUInteger m, NSUInteger n, NSUInteger k,
    double alpha, double beta,
    NSUInteger batch_size = 0) {
  uint64_t alpha_bits, beta_bits;
  std::memcpy(&alpha_bits, &alpha, sizeof(double));
  std::memcpy(&beta_bits, &beta, sizeof(double));

  MpsGemmKey key{transpose_a, transpose_b, m, n, k,
                 alpha_bits, beta_bits, batch_size};

  std::lock_guard<std::mutex> lk(g_mps_gemm_mutex);
  auto it = g_mps_gemm_cache.find(key);
  if (it != g_mps_gemm_cache.end()) {
    return it->second;
  }

  id<MTLDevice> dev = ctranslate2::metal::get_metal_device();
  MPSMatrixMultiplication* op =
      [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                         transposeLeft:(BOOL)transpose_a
                                        transposeRight:(BOOL)transpose_b
                                           resultRows:m
                                        resultColumns:n
                                      interiorColumns:k
                                                alpha:alpha
                                                 beta:beta];
  if (batch_size > 0) {
    op.batchSize = batch_size;
    op.batchStart = 0;
  }
  g_mps_gemm_cache[key] = op;
  return op;
}

// Release all cached MPSMatrixMultiplication objects and clear the cache.
// Called from MetalAllocator::clear_cache() via metal::clear_gemm_cache().
static void clear_mps_gemm_cache() {
  std::lock_guard<std::mutex> lk(g_mps_gemm_mutex);
  for (auto& [key, op] : g_mps_gemm_cache)
    [op release];
  g_mps_gemm_cache.clear();
}

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

  // M12.2: Use cached rowBytesForColumns (avoids ObjC message send per call).
  const NSUInteger mps_rb_a = cached_row_bytes(cols_a, dtype);
  const NSUInteger mps_rb_b = cached_row_bytes(cols_b, dtype);
  const NSUInteger mps_rb_c = cached_row_bytes(cols_c, dtype);

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

    // M11.27: Use cached MPSMatrixMultiplication (caller must NOT release).
    MPSMatrixMultiplication* gemm_op =
        get_cached_mps_gemm(transpose_a, transpose_b,
                            (NSUInteger)m, (NSUInteger)n, (NSUInteger)k,
                            (double)alpha, (double)beta);

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
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
// M13: Float16 GEMM via float32 promotion.
//
// MPSMatrixMultiplication with float16 inputs accumulates in float16,
// causing catastrophic precision loss for K >= 512.  This wrapper:
//   1. GPU-converts A and B from half → float32 (encode-only)
//   2. Runs MPS GEMM in float32 (float32 accumulation)
//   3. GPU-converts C from float32 → half (encode-only)
//
// Zero CPU/GPU syncs — all conversions are encode-only GPU kernels.
// Memory cost: 2× for A, B, C temp buffers (released after encoding).
// Performance: ~1.5-2× slower than native float16 MPS GEMM, but correct.
// ---------------------------------------------------------------------------

// Forward declaration — dispatch_mps_gemm_buf is defined later in this file.
static void dispatch_mps_gemm_buf(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    id<MTLBuffer> buf_a, NSUInteger off_a, NSUInteger rb_a,
    NSUInteger rows_a, NSUInteger cols_a,
    id<MTLBuffer> buf_b, NSUInteger off_b, NSUInteger rb_b,
    NSUInteger rows_b, NSUInteger cols_b,
    id<MTLBuffer> buf_c, NSUInteger off_c, NSUInteger rb_c,
    MPSDataType dtype,
    float beta);

// ---------------------------------------------------------------------------
// M13: Custom MSL GEMM — half inputs, float32 accumulation, half output.
//
// Used for small-m decode GEMMs (m ≤ kF16DirectThreshold) where the
// overhead of temp buffer allocation in the promotion path dominates.
//
// Note: the pre-existing race condition in Metal's resource tracking
// (duplicated first tokens in beam search) affects f32 equally — it's
// NOT specific to f16 or the SIMD kernel.  Periodic CT2_COMMIT_AND_WAIT
// is NOT used here because it was shown to not improve correctness
// beyond the f32 baseline, while adding significant overhead.
// ---------------------------------------------------------------------------
static constexpr ctranslate2::dim_t kF16DirectThreshold = 32;

static void dispatch_f16_gemm_direct(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const ctranslate2::float16_t* a, ctranslate2::dim_t lda,
    const ctranslate2::float16_t* b, ctranslate2::dim_t ldb,
    float beta,
    ctranslate2::float16_t* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;

  id<MTLComputePipelineState> pso = get_gemm_f16_acc32_pso();
  id<MTLComputeCommandEncoder> enc = ctranslate2::metal::create_compute_encoder();
  [enc setComputePipelineState:pso];

  NSUInteger off_a = 0, off_b = 0, off_c = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  [enc setBuffer:buf_a offset:off_a atIndex:0];
  [enc setBuffer:buf_b offset:off_b atIndex:1];
  [enc setBuffer:buf_c offset:off_c atIndex:2];

  uint32_t params[8] = {
    ct2_u32(m), ct2_u32(n), ct2_u32(k),
    ct2_u32(lda), ct2_u32(ldb), ct2_u32(ldc),
    transpose_a ? 1u : 0u, transpose_b ? 1u : 0u
  };
  [enc setBytes:params length:sizeof(params) atIndex:3];
  [enc setBytes:&alpha length:sizeof(float)  atIndex:4];
  [enc setBytes:&beta  length:sizeof(float)  atIndex:5];

  // Tiled dispatch: 4 SIMD groups (128 threads) per threadgroup,
  // each threadgroup computes 16×16 output tile.
  NSUInteger tg_x = ((NSUInteger)n + 15) / 16;
  NSUInteger tg_y = ((NSUInteger)m + 15) / 16;
  [enc dispatchThreadgroups:MTLSizeMake(tg_x, tg_y, 1)
       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [enc endEncoding];
  [enc release];

  // Protect buffers from premature reuse (encode-only, no immediate sync).
  ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_b contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_c contents]);
}

// ---------------------------------------------------------------------------
// M13: Thread-local temp buffer cache for f16 promotion.
//
// Caches the last A, B, C float32 temp buffers per thread.  Since model
// dimensions are fixed, the same sizes are requested on every GEMM call —
// the cache hits 100% after warmup, eliminating per-GEMM ObjC alloc.
//
// Safety: encoders within a command buffer execute sequentially (Metal
// guarantees encoder ordering within a CB).  Buffer reuse across GEMMs is
// safe because by the time encoder N reads tmp_a, encoder N-1 (which wrote
// tmp_a) has already completed on the GPU.
//
// Memory savings: reduces live temp buffers from 3 × num_promoted_GEMMs
// to just 3, eliminating memory pressure during batch prefill.
// ---------------------------------------------------------------------------
struct F16TempCache {
  id<MTLBuffer> buf[3] = {nil, nil, nil};   // A, B, C temps
  NSUInteger    cap[3] = {0, 0, 0};

  id<MTLBuffer> get(int idx, NSUInteger bytes) {
    if (bytes <= cap[idx]) return buf[idx];
    if (buf[idx]) [buf[idx] release];
    buf[idx] = alloc_temp_buffer(bytes);
    cap[idx] = [buf[idx] length];
    return buf[idx];
  }
};

static thread_local F16TempCache _f16_temp_cache;

// ---------------------------------------------------------------------------
// M13: Float16 GEMM via MPS f32 promotion — following the INT8 GEMM pattern.
//
// Pipeline (all encode-only, zero syncs):
//   1. GPU encode: half→float32 conversion for A and B
//   2. GPU encode: MPS float32 GEMM (hardware optimized)
//   3. GPU encode: float32→half conversion for C
//
// Temp f32 buffers are cached per-thread (same sizes every call for a given
// model).  Alpha applied by MPS GEMM.  Beta=0 only (all transformer GEMMs).
// ---------------------------------------------------------------------------
static void dispatch_f16_promoted_gemm(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const ctranslate2::float16_t* a, ctranslate2::dim_t lda,
    const ctranslate2::float16_t* b, ctranslate2::dim_t ldb,
    float beta,
    ctranslate2::float16_t* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;
  if (beta != 0.0f)
    throw std::runtime_error("Metal F16 promoted GEMM: only beta=0 supported");

  // Physical layout of A and B in memory.
  const NSUInteger rows_a = (NSUInteger)(transpose_a ? k : m);
  const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
  const NSUInteger rows_b = (NSUInteger)(transpose_b ? n : k);
  const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);

  // MPS row-byte alignment for float32 matrices.
  const NSUInteger mps_rb_a = cached_row_bytes(cols_a, MPSDataTypeFloat32);
  const NSUInteger mps_rb_b = cached_row_bytes(cols_b, MPSDataTypeFloat32);
  const NSUInteger mps_rb_c = cached_row_bytes((NSUInteger)n, MPSDataTypeFloat32);

  const NSUInteger rb_a = std::max((NSUInteger)lda * sizeof(float), mps_rb_a);
  const NSUInteger rb_b = std::max((NSUInteger)ldb * sizeof(float), mps_rb_b);
  const NSUInteger rb_c = std::max((NSUInteger)n   * sizeof(float), mps_rb_c);

  // Get float32 temp buffers from per-thread cache (same sizes every call).
  id<MTLBuffer> tmp_a = _f16_temp_cache.get(0, rows_a * rb_a);
  id<MTLBuffer> tmp_b = _f16_temp_cache.get(1, rows_b * rb_b);
  id<MTLBuffer> tmp_c = _f16_temp_cache.get(2, (NSUInteger)m * rb_c);

  // Resolve input half buffers.
  NSUInteger off_a = 0, off_b = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

  // Protect input buffers from premature reuse (encode-only).
  ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_b contents]);

  // GPU: encode half→float32 conversion (encode-only).
  encode_half_to_float32(buf_a, off_a, (NSUInteger)lda,
                          tmp_a, 0, rb_a / sizeof(float),
                          ct2_u32(rows_a), ct2_u32(cols_a));
  encode_half_to_float32(buf_b, off_b, (NSUInteger)ldb,
                          tmp_b, 0, rb_b / sizeof(float),
                          ct2_u32(rows_b), ct2_u32(cols_b));

  // GPU: encode float32 MPS GEMM (encode-only, hardware optimized).
  dispatch_mps_gemm_buf(
      transpose_a, transpose_b, m, n, k, alpha,
      tmp_a, 0, rb_a, rows_a, cols_a,
      tmp_b, 0, rb_b, rows_b, cols_b,
      tmp_c, 0, rb_c, MPSDataTypeFloat32, 0.0f);

  // GPU: encode float32→half conversion (encode-only).
  // No sync needed — INT8 GEMM uses the same pattern (MPS → custom encoder)
  // without sync successfully.
  NSUInteger off_c = 0;
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  encode_float32_to_half(tmp_c, 0, rb_c / sizeof(float),
                          buf_c, off_c, (NSUInteger)ldc,
                          ct2_u32(m), ct2_u32(n));

  // Protect output buffer from premature reuse.
  ctranslate2::metal::protect_buffer_by_base([buf_c contents]);

  // Temp buffers owned by _f16_temp_cache — NOT released here.
  // Reuse is safe: encoders execute sequentially within the CB.
}

// ---------------------------------------------------------------------------
// M13: Combined f16 GEMM dispatch.
//
// Dispatch logic:
//   m ≤ kF16DirectThreshold (32): SIMD kernel with f32 accumulation
//     — avoids MPS float16 accumulation precision loss during autoregressive
//     decode where errors compound across steps.
//   m > 32: Direct MPS float16 GEMM (fast, hardware optimized)
//     — acceptable precision for single-pass computation (encoder prefill,
//     first decode step).  Errors don't compound because these are
//     non-autoregressive passes.
// ---------------------------------------------------------------------------
static void dispatch_f16_gemm(
    bool transpose_a, bool transpose_b,
    ctranslate2::dim_t m, ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha,
    const ctranslate2::float16_t* a, ctranslate2::dim_t lda,
    const ctranslate2::float16_t* b, ctranslate2::dim_t ldb,
    float beta,
    ctranslate2::float16_t* c, ctranslate2::dim_t ldc) {
  if (m == 0 || n == 0 || k == 0) return;

  if (m <= kF16DirectThreshold) {
    // Decode path: custom MSL SIMD kernel with f32 accumulation.
    dispatch_f16_gemm_direct(transpose_a, transpose_b, m, n, k,
                             alpha, a, lda, b, ldb, beta, c, ldc);
  } else {
    // Encode/prefill path: promote half→f32, run MPS f32 GEMM, demote f32→half.
    // Matches CUDA Tensor Core behavior (f32 accumulation).  ~1.5x slower than
    // native MPS f16 GEMM but eliminates the ~5 BLEU loss from f16 accumulation.
    dispatch_f16_promoted_gemm(transpose_a, transpose_b, m, n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
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
    ctranslate2::primitives<ctranslate2::Device::MPS>::mul(
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
    MPSDataType dtype,
    float beta = 0.0f) {
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

    // M11.27: Use cached MPSMatrixMultiplication.
    MPSMatrixMultiplication* gemm_op =
        get_cached_mps_gemm(transpose_a, transpose_b,
                            (NSUInteger)m, (NSUInteger)n, (NSUInteger)k,
                            (double)alpha, (double)beta);
    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
  }
}

// INT8 GEMM: convert int8 A and B to float32 on GPU, run float32 MPS GEMM,
// round float32 result to int32 on GPU.  All encode-only, no CPU/GPU syncs.
// Only beta=0 is supported.
//
// M12.6: Replaced CPU vDSP conversions + 2 commit_and_wait() with GPU kernels.
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

  // M12.14: Fused INT8 GEMV for decode (m=1, trans_b=true, !trans_a).
  // Reads int8 A and B directly — no f32 temp buffers, ~7× less bandwidth.
  // Requires contiguous layout: lda==k (A row-major) and ldb==k (B row-major).
  if (m == 1 && !transpose_a && transpose_b && lda == k && ldb == k) {
    dispatch_fused_int8_gemv(n, k, alpha, a, lda, b, ldb, c, ldc);
    return;
  }

  // Physical layout of A and B in memory:
  //   !transpose_a → rows_a = m, cols_a = k
  //    transpose_a → rows_a = k, cols_a = m
  const NSUInteger rows_a = (NSUInteger)(transpose_a ? k : m);
  const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
  const NSUInteger rows_b = (NSUInteger)(transpose_b ? n : k);
  const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);

  // M12.2: Use cached rowBytesForColumns for float32.
  const NSUInteger mps_rb_a = cached_row_bytes(cols_a, MPSDataTypeFloat32);
  const NSUInteger mps_rb_b = cached_row_bytes(cols_b, MPSDataTypeFloat32);
  const NSUInteger mps_rb_c = cached_row_bytes((NSUInteger)n, MPSDataTypeFloat32);

  // Use padded row bytes to satisfy MPS alignment requirements.
  const NSUInteger rb_a = std::max((NSUInteger)lda * sizeof(float), mps_rb_a);
  const NSUInteger rb_b = std::max((NSUInteger)ldb * sizeof(float), mps_rb_b);
  const NSUInteger rb_c = std::max((NSUInteger)n   * sizeof(float), mps_rb_c);

  // Allocate float32 temporary buffers (Shared mode).
  id<MTLBuffer> tmp_a = alloc_temp_buffer(rows_a * rb_a);
  id<MTLBuffer> tmp_b = alloc_temp_buffer(rows_b * rb_b);
  id<MTLBuffer> tmp_c = alloc_temp_buffer((NSUInteger)m * rb_c);

  // M12.6: All-GPU path — encode-only, no CPU/GPU syncs.
  NSUInteger off_a = 0, off_b = 0;
  id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(a, &off_a);
  id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(b, &off_b);

  // M12 review H1: Protect input buffers from premature reuse.
  // GPU kernels read A and B in encode-only mode; if the caller frees them
  // before commit_and_wait(), the allocator could reuse the MTLBuffer.
  ctranslate2::metal::protect_buffer_by_base([buf_a contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_b contents]);

  encode_int8_to_float32(buf_a, off_a, (NSUInteger)lda,
                          tmp_a, 0, rb_a / sizeof(float),
                          ct2_u32(rows_a), ct2_u32(cols_a));
  encode_int8_to_float32(buf_b, off_b, (NSUInteger)ldb,
                          tmp_b, 0, rb_b / sizeof(float),
                          ct2_u32(rows_b), ct2_u32(cols_b));

  // GPU: encode float32 MPS GEMM (encode-only).
  dispatch_mps_gemm_buf(
      transpose_a, transpose_b, m, n, k, alpha,
      tmp_a, 0, rb_a, rows_a, cols_a,
      tmp_b, 0, rb_b, rows_b, cols_b,
      tmp_c, 0, rb_c, MPSDataTypeFloat32);

  // GPU: encode float32 → int32 rounding (encode-only).
  NSUInteger off_c = 0;
  id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(c, &off_c);
  encode_float32_to_int32(tmp_c, 0, rb_c / sizeof(float),
                           buf_c, off_c, (NSUInteger)ldc,
                           ct2_u32(m), ct2_u32(n));

  // Release temp buffers — CB retains them until GPU execution completes.
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

    // M11.27: Use cached MPSMatrixMultiplication with batch_size in key.
    MPSMatrixMultiplication* gemm_op =
        get_cached_mps_gemm(transpose_a, transpose_b,
                            (NSUInteger)m, (NSUInteger)n, (NSUInteger)k,
                            (double)alpha, (double)beta,
                            (NSUInteger)batch_size);

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
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

// ---------------------------------------------------------------------------
// M12.2: Custom float32 GEMV kernel for m=1 decode GEMMs.
//
// Mirrors the float16 GEMV (M11.19) but for float32.  During autoregressive
// decoding, all GEMMs have m=1 (one query token per step).  For float16,
// the custom GEMV already bypasses MPS entirely (M11.19).  For float32,
// decode GEMMs still went through the full MPS pipeline:
//   MPSMatrixDescriptor × 3 + MPSMatrix alloc/init/release × 3
//   + MPSMatrixMultiplication encode + @autoreleasepool
// This custom kernel eliminates all of that overhead.
//
// Grid: [batch_size, 1, 1], threads per group: [256, 1, 1]
// Each threadgroup computes C[1,N] = alpha * A[1,K] * B + beta * C.
// No float32→float32 conversion needed (unlike f16 which promotes to f32).
// ---------------------------------------------------------------------------
static const char* kGemvF32MSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void gemv_float(
    device const float*  A       [[buffer(0)]],
    device const float*  B       [[buffer(1)]],
    device       float*  C       [[buffer(2)]],
    constant     uint&   K       [[buffer(3)]],
    constant     uint&   N       [[buffer(4)]],
    constant     uint&   stridea [[buffer(5)]],
    constant     uint&   strideb [[buffer(6)]],
    constant     uint&   stridec [[buffer(7)]],
    constant     uint&   ldb_val [[buffer(8)]],
    constant     float&  alpha   [[buffer(9)]],
    constant     float&  beta    [[buffer(10)]],
    constant     uint&   tb      [[buffer(11)]],
    uint batch_id [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tgs      [[threads_per_threadgroup]])
{
    device const float* a_row = A + batch_id * stridea;
    device const float* b_mat = B + batch_id * strideb;
    device       float* c_row = C + batch_id * stridec;

    for (uint j = tid; j < N; j += tgs) {
        float acc = 0.0f;
        if (tb) {
            device const float* b_row = b_mat + j * ldb_val;
            for (uint i = 0; i < K; ++i)
                acc += a_row[i] * b_row[i];
        } else {
            for (uint i = 0; i < K; ++i)
                acc += a_row[i] * b_mat[i * ldb_val + j];
        }
        float old_c = (beta != 0.0f) ? c_row[j] : 0.0f;
        c_row[j] = alpha * acc + beta * old_c;
    }
}
)";

static id<MTLLibrary> get_gemv_f32_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kGemvF32MSL, "gemv_f32");
}

static id<MTLComputePipelineState> get_gemv_f32_pso() {
  static PSOCache cache;
  return cache.get(get_gemv_f32_library, "gemv_float");
}

static void dispatch_gemv_f32_batched(
    bool transpose_b,
    ctranslate2::dim_t n, ctranslate2::dim_t k,
    float alpha, float beta,
    const float* a, ctranslate2::dim_t lda, ctranslate2::dim_t stridea,
    const float* b, ctranslate2::dim_t ldb, ctranslate2::dim_t strideb,
    float* c, ctranslate2::dim_t ldc, ctranslate2::dim_t stridec,
    ctranslate2::dim_t batch_size) {
  if (batch_size <= 0 || n == 0 || k == 0) return;

  id<MTLComputePipelineState> pso = get_gemv_f32_pso();
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

  // M12.2: Protect original buffers from premature reuse (encode-only).
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

  // M12.2: Use cached rowBytesForColumns.
  const NSUInteger mps_rb_a = cached_row_bytes(cols_a, dtype);
  const NSUInteger mps_rb_b = cached_row_bytes(cols_b, dtype);
  const NSUInteger mps_rb_c = cached_row_bytes((NSUInteger)n, dtype);

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
    // Release temp buffers before fallback (ARC is off — must not leak).
    if (tmp_a) [tmp_a release];
    if (tmp_b) [tmp_b release];
    if (tmp_c) [tmp_c release];
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

    // M11.27: Use cached MPSMatrixMultiplication with batch_size in key.
    MPSMatrixMultiplication* gemm_op =
        get_cached_mps_gemm(transpose_a, transpose_b,
                            (NSUInteger)m, (NSUInteger)n, (NSUInteger)k,
                            (double)alpha, (double)beta,
                            (NSUInteger)batch_size);

    [gemm_op encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [matA release];
    [matB release];
    [matC release];
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
  dim_t primitives<Device::MPS>::gemm_pack_b(
      const T*, bool, dim_t, dim_t, float, T*) {
    return 0;  // Packing not supported on Metal.
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::MPS>::gemm(
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
      // M13: Float16 GEMM with float32 accumulation to match CUDA precision.
      // MPS native f16 GEMM accumulates in f16, causing ~5 BLEU loss on
      // encoder-decoder models (OPUS-MT) due to precision loss at padding
      // boundaries.  CUDA Tensor Cores accumulate in f32 by default.
      //   m <= 32: custom MSL SIMD kernel (f32 accum, decode path)
      //   m >  32: half→f32 promotion + MPS f32 GEMM + f32→half (encode path)
      dispatch_f16_gemm(transpose_a, transpose_b, m, n, k,
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
  void primitives<Device::MPS>::gemm_batch_strided(
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
    // M12.2: Use cached rowBytesForColumns in needs_padding check.
    auto needs_padding = [&]() -> bool {
      constexpr NSUInteger elem_sz = sizeof(In);
      const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
      const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);
      const NSUInteger cols_c = (NSUInteger)n;
      MPSDataType dt = (elem_sz == 4) ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
      NSUInteger mps_a = cached_row_bytes(cols_a, dt);
      NSUInteger mps_b = cached_row_bytes(cols_b, dt);
      NSUInteger mps_c = cached_row_bytes(cols_c, dt);
      return ((NSUInteger)lda * elem_sz < mps_a) ||
             ((NSUInteger)ldb * elem_sz < mps_b) ||
             ((NSUInteger)ldc * elem_sz < mps_c);
    };

    if constexpr (std::is_same_v<In, float> && std::is_same_v<Out, float>) {
      if (batch_size > 0 && needs_padding()) {
        // M11.26: Always route padded f32 GEMMs through MPS (encode-only,
        // zero syncs).  The old m*n>4096 threshold fell back to CPU cblas,
        // requiring CT2_COMMIT_AND_WAIT before each call.
        dispatch_mps_gemm_batched_padded<float>(
            transpose_a, transpose_b, m, n, k,
            alpha, a, lda, stridea, b, ldb, strideb,
            beta, c, ldc, stridec, batch_size);
        if (m == 1) {
          // M11.18: Protect original buffers from premature reuse.
          // Only needed for m=1 decode attention GEMMs where the caller
          // may free A/B/C before the encode-only GPU work completes.
          NSUInteger off_tmp;
          ctranslate2::metal::protect_buffer_by_base(
              [ctranslate2::metal_buffer_for_ptr(a, &off_tmp) contents]);
          ctranslate2::metal::protect_buffer_by_base(
              [ctranslate2::metal_buffer_for_ptr(b, &off_tmp) contents]);
          ctranslate2::metal::protect_buffer_by_base(
              [ctranslate2::metal_buffer_for_ptr(c, &off_tmp) contents]);
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
      // M13: m>1 float16 batched GEMMs — same code path as f32.
      if (batch_size > 0 && needs_padding()) {
        dispatch_mps_gemm_batched_padded<float16_t>(
            transpose_a, transpose_b, m, n, k,
            alpha, a, lda, stridea, b, ldb, strideb,
            beta, c, ldc, stridec, batch_size);
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
      // Scale per batch element to respect stridec (may differ from m*n).
      if (alpha != 1.0f) {
        const dim_t elems_per_batch = m * n;
        for (dim_t i = 0; i < batch_size; ++i)
          primitives<Device::MPS>::mul(
              static_cast<bfloat16_t>(alpha),
              c + i * stridec, c + i * stridec, elems_per_batch);
      }
    } else if constexpr (std::is_same_v<In, int8_t> && std::is_same_v<Out, int32_t>) {
      if (batch_size == 0 || m == 0 || n == 0 || k == 0) return;
      if (beta != 0.0f)
        throw std::runtime_error("Metal INT8 GEMM: only beta=0 is supported");
      if (batch_size == 1) {
        dispatch_int8_gemm(transpose_a, transpose_b, m, n, k,
                           alpha, a, lda, b, ldb, beta, c, ldc);
      } else {
        // M12.6: All-GPU batched path — 0 syncs (all encode-only).
        const NSUInteger rows_a = (NSUInteger)(transpose_a ? k : m);
        const NSUInteger cols_a = (NSUInteger)(transpose_a ? m : k);
        const NSUInteger rows_b = (NSUInteger)(transpose_b ? n : k);
        const NSUInteger cols_b = (NSUInteger)(transpose_b ? k : n);

        NSUInteger mps_rb_a = cached_row_bytes(cols_a, MPSDataTypeFloat32);
        NSUInteger mps_rb_b = cached_row_bytes(cols_b, MPSDataTypeFloat32);
        NSUInteger mps_rb_c = cached_row_bytes((NSUInteger)n, MPSDataTypeFloat32);
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

        const NSUInteger out_stride_a = rb_a / sizeof(float);
        const NSUInteger out_stride_b = rb_b / sizeof(float);
        const NSUInteger out_stride_c = rb_c / sizeof(float);

        // M12 review M1: Protect input A and B buffers from premature reuse.
        // All batch elements may share the same underlying MTLBuffer (e.g. weight matrix B),
        // so protect once using the base pointer from the first element.
        {
          NSUInteger off_tmp = 0;
          ctranslate2::metal::protect_buffer_by_base(
              [ctranslate2::metal_buffer_for_ptr(a, &off_tmp) contents]);
          ctranslate2::metal::protect_buffer_by_base(
              [ctranslate2::metal_buffer_for_ptr(b, &off_tmp) contents]);
        }

        // Phase 1: GPU encode int8→float32 for all batches (encode-only).
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          const int8_t* src_a = a + bi * stridea;
          const int8_t* src_b = b + bi * strideb;
          NSUInteger off_a = 0, off_b = 0;
          id<MTLBuffer> buf_a = ctranslate2::metal_buffer_for_ptr(src_a, &off_a);
          id<MTLBuffer> buf_b = ctranslate2::metal_buffer_for_ptr(src_b, &off_b);
          encode_int8_to_float32(buf_a, off_a, (NSUInteger)lda,
                                  tmp_a, bi * bytes_a, out_stride_a,
                                  ct2_u32(rows_a), ct2_u32(cols_a));
          encode_int8_to_float32(buf_b, off_b, (NSUInteger)ldb,
                                  tmp_b, bi * bytes_b, out_stride_b,
                                  ct2_u32(rows_b), ct2_u32(cols_b));
        }

        // Phase 2: GPU encode all MPS GEMMs (encode-only).
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          dispatch_mps_gemm_buf(
              transpose_a, transpose_b, m, n, k, alpha,
              tmp_a, bi * bytes_a, rb_a, rows_a, cols_a,
              tmp_b, bi * bytes_b, rb_b, rows_b, cols_b,
              tmp_c, bi * bytes_c, rb_c, MPSDataTypeFloat32);
        }

        // Phase 3: GPU encode float32→int32 for all batches (encode-only).
        for (dim_t bi = 0; bi < batch_size; ++bi) {
          int32_t* dst_c = c + bi * stridec;
          NSUInteger off_c = 0;
          id<MTLBuffer> buf_c = ctranslate2::metal_buffer_for_ptr(dst_c, &off_c);
          encode_float32_to_int32(tmp_c, bi * bytes_c, out_stride_c,
                                   buf_c, off_c, (NSUInteger)ldc,
                                   ct2_u32(m), ct2_u32(n));
        }

        // Release temp buffers — CB retains them until GPU execution.
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

  template dim_t primitives<Device::MPS>::gemm_pack_b(
      const float*, bool, dim_t, dim_t, float, float*);
  template dim_t primitives<Device::MPS>::gemm_pack_b(
      const float16_t*, bool, dim_t, dim_t, float, float16_t*);
  template dim_t primitives<Device::MPS>::gemm_pack_b(
      const bfloat16_t*, bool, dim_t, dim_t, float, bfloat16_t*);
  template dim_t primitives<Device::MPS>::gemm_pack_b(
      const int8_t*, bool, dim_t, dim_t, float, int8_t*);

  template void primitives<Device::MPS>::gemm<float, float>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, const float*, dim_t,
      float, float*, dim_t, const float*);
  template void primitives<Device::MPS>::gemm<float16_t, float16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, const float16_t*, dim_t,
      float, float16_t*, dim_t, const float16_t*);
  template void primitives<Device::MPS>::gemm<bfloat16_t, bfloat16_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, const bfloat16_t*, dim_t,
      float, bfloat16_t*, dim_t, const bfloat16_t*);
  template void primitives<Device::MPS>::gemm<int8_t, int32_t>(
      bool, bool, bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, const int8_t*, dim_t,
      float, int32_t*, dim_t, const int32_t*);

  template void primitives<Device::MPS>::gemm_batch_strided<float, float>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float*, dim_t, dim_t, const float*, dim_t, dim_t,
      float, float*, dim_t, dim_t, dim_t);
  template void primitives<Device::MPS>::gemm_batch_strided<float16_t, float16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const float16_t*, dim_t, dim_t, const float16_t*, dim_t, dim_t,
      float, float16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::MPS>::gemm_batch_strided<bfloat16_t, bfloat16_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const bfloat16_t*, dim_t, dim_t, const bfloat16_t*, dim_t, dim_t,
      float, bfloat16_t*, dim_t, dim_t, dim_t);
  template void primitives<Device::MPS>::gemm_batch_strided<int8_t, int32_t>(
      bool, bool, dim_t, dim_t, dim_t,
      float, const int8_t*, dim_t, dim_t, const int8_t*, dim_t, dim_t,
      float, int32_t*, dim_t, dim_t, dim_t);

  namespace metal {
    void clear_gemm_cache() {
      clear_mps_gemm_cache();
      clear_sdpa_gemm_cache();  // P1: also clear SDPA's local cache
    }
  }  // namespace metal

}  // namespace ctranslate2
