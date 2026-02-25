/**
 * tools/metal_poc/bf16_poc.mm
 *
 * Milestone 0.2 – Validate BF16 availability on Apple Silicon
 *
 * KEY FINDING from first attempt:
 *   MPSMatrixMultiplication asserts on MPSDataTypeBFloat16 — it only accepts
 *   Float32, Float16, Int8, Int16.  BF16 matrix multiply must use MPSGraph
 *   (matrixMultiplicationWithPrimaryTensor:secondaryTensor:).
 *   This confirms the plan's guidance: "Use MPSGraph only where MPS doesn't
 *   have a single-op equivalent."
 *
 * This file therefore tests:
 *   1. GPU family check: MTLGPUFamilyApple9+ for BF16 hardware support.
 *   2. MPSGraph BF16 matmul: correctness vs FP32 cblas reference.
 *   3. Performance: BF16 MPSGraph vs FP32 MPSMatrixMultiplication (M0.1 baseline).
 *
 * Build (macOS 14+ required for MPSDataTypeBFloat16):
 *   clang++ -std=c++17 -O2 -mmacosx-version-min=14.0 \
 *       -DACCELERATE_NEW_LAPACK \
 *       -o bf16_poc tools/metal_poc/bf16_poc.mm \
 *       -framework Metal -framework Foundation \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph \
 *       -framework Accelerate
 *   ./bf16_poc
 *
 * PASS criteria (APPLE_M4_METAL_PLAN.md §0.2):
 *   1. Device reports BF16 support (Apple9+ GPU family).
 *   2. BF16 MPSGraph matmul result is within 1e-2 relative tolerance of
 *      FP32 cblas_sgemm reference (BF16 unit-roundoff ≈ 4e-3).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <Accelerate/Accelerate.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static constexpr int M = 4096;
static constexpr int K = 4096;
static constexpr int N = 4096;

static constexpr int WARMUP_RUNS   = 3;
static constexpr int TIMED_RUNS    = 10;
static constexpr int CHECK_SAMPLES = 2048;

// BF16 unit-roundoff ≈ 2^-7 ≈ 7.8e-3.
// 1e-2 gives comfortable margin for both input and output quantisation.
static constexpr float BF16_REL_TOLERANCE = 1e-2f;

// FP32 Metal baseline from M0.1 (Apple M4, same machine)
static constexpr double FP32_BASELINE_MS = 46.64;

// ---------------------------------------------------------------------------
// BF16 ↔ FP32 conversion (round-to-nearest-even)
// ---------------------------------------------------------------------------

static inline uint16_t float_to_bf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u)
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);  // quiet NaN
  const uint32_t bias = 0x00007FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + bias) >> 16);
}

static inline float bf16_to_float(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f; memcpy(&f, &bits, sizeof(f)); return f;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double elapsed_ms(std::chrono::steady_clock::time_point t0,
                         std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int highest_apple_gpu_family(id<MTLDevice> device) {
  static const MTLGPUFamily kFamilies[] = {
    MTLGPUFamilyApple10, MTLGPUFamilyApple9,
    MTLGPUFamilyApple8,  MTLGPUFamilyApple7,
  };
  static const int kTiers[] = { 10, 9, 8, 7 };
  for (int i = 0; i < 4; ++i) {
    if ([device supportsFamily:kFamilies[i]]) return kTiers[i];
  }
  return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  // ── 1. Device & BF16 capability ─────────────────────────────────────────
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { fprintf(stderr, "FAIL: No Metal device.\n"); return 1; }

  printf("Metal device      : %s\n", [[device name] UTF8String]);
  const int  gpu_family        = highest_apple_gpu_family(device);
  const bool bf16_hw_supported = (gpu_family >= 9);
  printf("Highest GPU family: Apple%d\n", gpu_family);
  printf("BF16 HW support   : %s  (Apple9+ required)\n\n",
         bf16_hw_supported ? "YES" : "NO");

  if (!bf16_hw_supported) {
    printf("RESULT: FAIL — BF16 hardware not available.\n");
    return 1;
  }

  // ── 2. Allocate shared BF16 buffers ─────────────────────────────────────
  id<MTLCommandQueue> queue = [device newCommandQueue];

  const size_t bytesA = static_cast<size_t>(M) * K * sizeof(uint16_t);
  const size_t bytesB = static_cast<size_t>(K) * N * sizeof(uint16_t);

  id<MTLBuffer> bufA = [device newBufferWithLength:bytesA
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [device newBufferWithLength:bytesB
                                           options:MTLResourceStorageModeShared];
  // No output buffer needed: MPSGraph allocates its own; we read back via
  // MPSNDArray.readBytes:strideBytes: after each run.

  if (!bufA || !bufB) {
    fprintf(stderr, "FAIL: MTLBuffer allocation failed.\n"); return 1;
  }

  // ── 3. Fill with random data (FP32 → BF16) ──────────────────────────────
  std::vector<float> fp32A(M * K), fp32B(K * N);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : fp32A) x = dist(rng);
    for (auto& x : fp32B) x = dist(rng);
  }
  auto* pA = static_cast<uint16_t*>([bufA contents]);
  auto* pB = static_cast<uint16_t*>([bufB contents]);
  for (int i = 0; i < M * K; ++i) pA[i] = float_to_bf16(fp32A[i]);
  for (int i = 0; i < K * N; ++i) pB[i] = float_to_bf16(fp32B[i]);

  // ── 4. Build MPSGraph for BF16 matmul ───────────────────────────────────
  //
  // MPSMatrixMultiplication does NOT support BF16 (asserts at runtime).
  // MPSGraph matrixMultiplicationWithPrimaryTensor: does support BF16.
  // We compile the graph once; MPSGraph caches the compiled kernel.
  MPSGraph* graph = [[MPSGraph alloc] init];

  NSArray<NSNumber*>* shapeA = @[@(M), @(K)];
  NSArray<NSNumber*>* shapeB = @[@(K), @(N)];

  MPSGraphTensor* tA = [graph placeholderWithShape:shapeA
                                          dataType:MPSDataTypeBFloat16
                                              name:@"A"];
  MPSGraphTensor* tB = [graph placeholderWithShape:shapeB
                                          dataType:MPSDataTypeBFloat16
                                              name:@"B"];
  MPSGraphTensor* tC = [graph matrixMultiplicationWithPrimaryTensor:tA
                                                    secondaryTensor:tB
                                                               name:@"C"];

  // Wrap MTLBuffers in MPSGraphTensorData for feed/result
  MPSGraphTensorData* tdA =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA
                                              shape:shapeA
                                           dataType:MPSDataTypeBFloat16];
  MPSGraphTensorData* tdB =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB
                                              shape:shapeB
                                           dataType:MPSDataTypeBFloat16];

  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{tA: tdA, tB: tdB};

  // Run one graph execution synchronously.
  // captureInto: if non-null, copies the BF16 output into that buffer.
  auto run_bf16_gemm = [&](uint16_t* captureInto) -> bool {
    @autoreleasepool {
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
          [graph runWithMTLCommandQueue:queue
                                  feeds:feeds
                          targetTensors:@[tC]
                       targetOperations:nil];
      if (!results || !results[tC]) return false;
      if (captureInto) {
        // readBytes:strideBytes: copies data from device to the supplied CPU buffer.
        // nil strides = packed (contiguous) layout.
        [[results[tC] mpsndarray] readBytes:captureInto strideBytes:nil];
      }
      return true;
    }
  };

  // ── 5. Warmup (triggers graph compilation & kernel caching) ─────────────
  printf("Warming up BF16 MPSGraph (%d runs — includes compilation)...\n",
         WARMUP_RUNS);
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    if (!run_bf16_gemm(nullptr)) {
      fprintf(stderr, "FAIL: BF16 warmup run %d failed.\n", i); return 1;
    }
  }

  // ── 6. Timed BF16 MPSGraph runs ─────────────────────────────────────────
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < TIMED_RUNS; ++i) {
    if (!run_bf16_gemm(nullptr)) {
      fprintf(stderr, "FAIL: BF16 timed run %d failed.\n", i); return 1;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  const double bf16_avg_ms = elapsed_ms(t0, t1) / TIMED_RUNS;

  // Capture result once for correctness check
  std::vector<uint16_t> bf16Result(M * N);
  if (!run_bf16_gemm(bf16Result.data())) {
    fprintf(stderr, "FAIL: BF16 result capture run failed.\n"); return 1;
  }

  // ── 7. CPU reference using BF16-quantised inputs ─────────────────────────
  //
  // Compare Metal BF16 output against cblas_sgemm on the SAME BF16-quantised
  // inputs (round-tripped through BF16 → FP32).  This removes input-
  // quantisation noise from the error metric and isolates hardware BF16
  // accumulation precision.  Comparing against the raw fp32A/fp32B inflates
  // apparent error by up to ~7% for K=4096 due to input rounding alone.
  std::vector<float> refA(M * K), refB(K * N);
  for (int i = 0; i < M * K; ++i) refA[i] = bf16_to_float(pA[i]);
  for (int i = 0; i < K * N; ++i) refB[i] = bf16_to_float(pB[i]);

  std::vector<float> cpuResult(M * N, 0.0f);
  // Warmup
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, 1.0f, refA.data(), K, refB.data(), N,
              0.0f, cpuResult.data(), N);
  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < TIMED_RUNS; ++i) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, refA.data(), K, refB.data(), N,
                0.0f, cpuResult.data(), N);
  }
  auto t3 = std::chrono::steady_clock::now();
  const double fp32_cpu_avg_ms = elapsed_ms(t2, t3) / TIMED_RUNS;

  // ── 8. Correctness ───────────────────────────────────────────────────────
  float max_rel_err = 0.0f, max_abs_err = 0.0f;
  int mismatches = 0;
  const int stride = (M * N) / CHECK_SAMPLES;
  for (int s = 0; s < CHECK_SAMPLES; ++s) {
    const int   idx     = s * stride;
    const float cpu_v   = cpuResult[idx];
    const float metal_v = bf16_to_float(bf16Result[idx]);
    const float abs_d   = std::abs(metal_v - cpu_v);
    const float rel_d   = abs_d / (std::abs(cpu_v) + 1.0f);
    if (abs_d > max_abs_err) max_abs_err = abs_d;
    if (rel_d > max_rel_err) max_rel_err = rel_d;
    if (rel_d > BF16_REL_TOLERANCE) ++mismatches;
  }

  // ── 9. Report ─────────────────────────────────────────────────────────────
  const double bf16_tflops      = 2.0 * M * N * K / (bf16_avg_ms * 1e-3) / 1e12;
  const double vs_fp32_metal    = FP32_BASELINE_MS / bf16_avg_ms;
  const double vs_fp32_cpu      = fp32_cpu_avg_ms  / bf16_avg_ms;

  printf("\n");
  printf("=== BF16 MPSGraph GEMM POC  (%dx%dx%d) ===\n", M, N, K);
  printf("  BF16 MPSGraph avg : %8.2f ms   (%.2f TFLOPS)\n",
         bf16_avg_ms, bf16_tflops);
  printf("  FP32 Metal (M0.1) : %8.2f ms   (2.95 TFLOPS)\n", FP32_BASELINE_MS);
  printf("  BF16 vs FP32 Metal: %.2fx\n", vs_fp32_metal);
  printf("  FP32 CPU          : %8.2f ms\n", fp32_cpu_avg_ms);
  printf("  BF16 vs FP32 CPU  : %.1fx\n", vs_fp32_cpu);
  printf("\n");
  printf("  Correctness (sampled %d / %d elements):\n", CHECK_SAMPLES, M * N);
  printf("    max abs diff : %.4e\n", max_abs_err);
  printf("    max rel diff : %.4e  (tolerance %.0e)\n",
         max_rel_err, static_cast<double>(BF16_REL_TOLERANCE));
  printf("    mismatches   : %d / %d\n", mismatches, CHECK_SAMPLES);
  printf("\n");
  printf("ARCHITECTURE NOTE:\n");
  printf("  MPSMatrixMultiplication does NOT support BF16 (Float32/16/Int8/16 only).\n");
  printf("  BF16 matmul requires MPSGraph — aligns with plan recommendation\n");
  printf("  to use MPSGraph only where MPS ops lack support.\n");
  printf("\n");

  // ── 10. Pass/Fail ─────────────────────────────────────────────────────────
  const bool pass_support     = bf16_hw_supported;
  const bool pass_correctness = (mismatches == 0);
  const bool pass_all         = pass_support && pass_correctness;

  printf("PASS/FAIL:\n");
  printf("  BF16 HW support (Apple9+)   : %s\n", pass_support     ? "PASS" : "FAIL");
  printf("  Correctness (rel < %.0e)   : %s\n",
         static_cast<double>(BF16_REL_TOLERANCE),
         pass_correctness ? "PASS" : "FAIL");
  printf("  BF16 vs FP32 Metal speedup  : %.2fx  (informational)\n", vs_fp32_metal);
  printf("\nOVERALL: %s\n\n", pass_all ? "PASS" : "FAIL");

  return pass_all ? 0 : 1;
}
