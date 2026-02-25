/**
 * tools/metal_poc/gemm_poc.mm
 *
 * Milestone 0.1 – Standalone MPS GEMM benchmark
 *
 * Validates that MPSMatrixMultiplication on Apple Silicon is fast enough and
 * numerically correct before writing any CTranslate2 infrastructure.
 *
 * Build (standalone – no CMake):
 *   clang++ -std=c++17 -O2 -o gemm_poc tools/metal_poc/gemm_poc.mm \
 *       -framework Metal -framework Foundation -framework MetalPerformanceShaders \
 *       -framework Accelerate
 *   ./gemm_poc
 *
 * PASS criteria (from APPLE_M4_METAL_PLAN.md §0.1):
 *   1. Metal result matches CPU (Accelerate cblas_sgemm) within relative
 *      tolerance 1e-4 (i.e. |metal - cpu| / (|cpu| + 1) < 1e-4).
 *   2. Metal is ≥ 2× faster than CPU on the timed runs.
 *
 * Notes on tolerance:
 *   For a 4096×4096×4096 FP32 matmul with U[-1,1] inputs, typical output
 *   magnitudes are ~21, so relative tolerance 1e-4 ≈ absolute 2e-3.
 *   FP32 round-trip accumulation error for K=4096 is ~K·ε ≈ 5e-4, well
 *   within that bound.  MPS may use mixed-precision internally but must
 *   produce output close enough to the cblas reference.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

static constexpr int M = 4096;   // rows of A, rows of C
static constexpr int K = 4096;   // cols of A, rows of B (interior)
static constexpr int N = 4096;   // cols of B, cols of C

static constexpr int WARMUP_RUNS  = 3;
static constexpr int TIMED_RUNS   = 10;
static constexpr int CHECK_SAMPLES = 2048;  // random elements sampled for correctness
static constexpr float REL_TOLERANCE = 1e-4f;
static constexpr double MIN_SPEEDUP  = 2.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double elapsed_ms(std::chrono::steady_clock::time_point t0,
                         std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Run one Metal GEMM: encode → commit → waitUntilCompleted.
// Returns false if the command buffer encountered an error.
static bool run_metal_gemm(id<MTLCommandQueue>        queue,
                           MPSMatrixMultiplication*   kernel,
                           MPSMatrix*                 matA,
                           MPSMatrix*                 matB,
                           MPSMatrix*                 matC) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) return false;
    [kernel encodeToCommandBuffer:cb
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.status == MTLCommandBufferStatusCompleted);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  // ── 1. Device ────────────────────────────────────────────────────────────
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    fprintf(stderr, "FAIL: MTLCreateSystemDefaultDevice() returned nil – "
                    "no Metal-capable device found.\n");
    return 1;
  }
  printf("Metal device : %s\n", [[device name] UTF8String]);
  printf("Matrix size  : %d x %d x %d (FP32, %.1f GB total)\n\n",
         M, K, N,
         3.0 * M * N * sizeof(float) / 1e9);

  id<MTLCommandQueue> queue = [device newCommandQueue];

  // ── 2. Allocate shared buffers ──────────────────────────────────────────
  //
  // MTLResourceStorageModeShared: buffer is accessible by both CPU and GPU
  // without explicit copy — the correct choice for unified-memory Apple Silicon.
  const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
  const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
  const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

  id<MTLBuffer> bufA = [device newBufferWithLength:bytesA
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [device newBufferWithLength:bytesB
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [device newBufferWithLength:bytesC
                                           options:MTLResourceStorageModeShared];

  if (!bufA || !bufB || !bufC) {
    fprintf(stderr, "FAIL: MTLBuffer allocation failed (out of memory?).\n");
    return 1;
  }

  // ── 3. Fill inputs with reproducible random data ────────────────────────
  float* pA = static_cast<float*>([bufA contents]);
  float* pB = static_cast<float*>([bufB contents]);
  float* pC = static_cast<float*>([bufC contents]);

  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) pA[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) pB[i] = dist(rng);
  }
  memset(pC, 0, bytesC);

  // ── 4. Set up MPSMatrixMultiplication ───────────────────────────────────
  //
  // MPSMatrixDescriptor expects row-major layout.
  // rowBytes must be >= columns * sizeof(element), aligned to 32 bytes.
  // For our packed layout (no padding), rowBytes = columns * sizeof(float).
  MPSMatrixDescriptor* descA =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                            columns:(NSUInteger)K
                                           rowBytes:(NSUInteger)(K * sizeof(float))
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* descB =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)K
                                            columns:(NSUInteger)N
                                           rowBytes:(NSUInteger)(N * sizeof(float))
                                           dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* descC =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                            columns:(NSUInteger)N
                                           rowBytes:(NSUInteger)(N * sizeof(float))
                                           dataType:MPSDataTypeFloat32];

  MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
  MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
  MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

  // C = 1·A·B + 0·C  (standard GEMM)
  MPSMatrixMultiplication* gemm =
      [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          resultRows:(NSUInteger)M
                                       resultColumns:(NSUInteger)N
                                     interiorColumns:(NSUInteger)K];

  // ── 5. Warmup (GPU pipeline compilation, caches) ────────────────────────
  printf("Warming up Metal (%d runs)...\n", WARMUP_RUNS);
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    if (!run_metal_gemm(queue, gemm, matA, matB, matC)) {
      fprintf(stderr, "FAIL: Metal warmup run %d failed.\n", i);
      return 1;
    }
  }

  // ── 6. Timed Metal runs ─────────────────────────────────────────────────
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < TIMED_RUNS; ++i) {
    if (!run_metal_gemm(queue, gemm, matA, matB, matC)) {
      fprintf(stderr, "FAIL: Metal timed run %d failed.\n", i);
      return 1;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  const double metal_avg_ms = elapsed_ms(t0, t1) / TIMED_RUNS;

  // Save Metal result for comparison (copy contents; buffer is Shared so CPU
  // can read directly – but take a snapshot after all runs complete).
  std::vector<float> metalResult(M * N);
  memcpy(metalResult.data(), pC, bytesC);

  // ── 7. CPU reference: Accelerate cblas_sgemm ───────────────────────────
  std::vector<float> cpuResult(M * N, 0.0f);

  // Warmup
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K, 1.0f,
              pA, K,
              pB, N,
              0.0f, cpuResult.data(), N);

  auto t2 = std::chrono::steady_clock::now();
  for (int i = 0; i < TIMED_RUNS; ++i) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                pA, K,
                pB, N,
                0.0f, cpuResult.data(), N);
  }
  auto t3 = std::chrono::steady_clock::now();
  const double cpu_avg_ms = elapsed_ms(t2, t3) / TIMED_RUNS;

  // ── 8. Correctness check (sampled) ─────────────────────────────────────
  //
  // Relative error: |metal[i] - cpu[i]| / (|cpu[i]| + 1.0)
  // Using (|cpu| + 1.0) as denominator prevents division-by-zero for
  // near-zero elements while still being strict for large magnitudes.
  float max_rel_err = 0.0f;
  float max_abs_err = 0.0f;
  int   mismatches  = 0;

  {
    // Deterministic strided sample to cover the full matrix
    const int total = M * N;
    const int stride = total / CHECK_SAMPLES;
    for (int s = 0; s < CHECK_SAMPLES; ++s) {
      const int idx = s * stride;
      const float cpu_val   = cpuResult[idx];
      const float metal_val = metalResult[idx];
      const float abs_diff  = std::abs(metal_val - cpu_val);
      const float rel_diff  = abs_diff / (std::abs(cpu_val) + 1.0f);

      if (abs_diff > max_abs_err) max_abs_err = abs_diff;
      if (rel_diff > max_rel_err) max_rel_err = rel_diff;
      if (rel_diff > REL_TOLERANCE) ++mismatches;
    }
  }

  // ── 9. Report ───────────────────────────────────────────────────────────
  const double tflops   = 2.0 * M * N * K / (metal_avg_ms * 1e-3) / 1e12;
  const double speedup  = cpu_avg_ms / metal_avg_ms;

  printf("\n");
  printf("=== MPS GEMM POC  (%dx%dx%d  FP32) ===\n", M, N, K);
  printf("  Metal  avg : %8.2f ms   (%.2f TFLOPS)\n", metal_avg_ms, tflops);
  printf("  CPU    avg : %8.2f ms\n", cpu_avg_ms);
  printf("  Speedup    : %.1fx\n", speedup);
  printf("\n");
  printf("  Correctness (sampled %d / %d elements):\n", CHECK_SAMPLES, M * N);
  printf("    max abs diff  : %.4e\n", max_abs_err);
  printf("    max rel diff  : %.4e  (tolerance %.0e)\n",
         max_rel_err, static_cast<double>(REL_TOLERANCE));
  printf("    mismatches    : %d / %d\n", mismatches, CHECK_SAMPLES);
  printf("\n");

  // ── 10. Pass/Fail ───────────────────────────────────────────────────────
  const bool pass_correctness = (mismatches == 0);
  const bool pass_speed       = (speedup >= MIN_SPEEDUP);
  const bool pass_all         = pass_correctness && pass_speed;

  printf("PASS/FAIL:\n");
  printf("  Correctness (rel err < %.0e) : %s\n",
         static_cast<double>(REL_TOLERANCE),
         pass_correctness ? "PASS" : "FAIL");
  printf("  Speed (>= %.0fx)              : %s\n",
         MIN_SPEEDUP,
         pass_speed ? "PASS" : "FAIL");
  printf("\nOVERALL: %s\n\n", pass_all ? "PASS" : "FAIL");

  if (!pass_correctness) {
    fprintf(stderr,
            "NOTE: Correctness FAIL — MPS may be using reduced-precision internally.\n"
            "      Consider re-evaluating MPS vs custom FP32 shaders (see §0.1 FAIL path).\n");
  }
  if (!pass_speed) {
    fprintf(stderr,
            "NOTE: Speed FAIL — Metal speedup %.1fx < %.0fx threshold.\n"
            "      Re-evaluate MPS vs custom shaders before proceeding with M1+.\n",
            speedup, MIN_SPEEDUP);
  }

  return pass_all ? 0 : 1;
}
