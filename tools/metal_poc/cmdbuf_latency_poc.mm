/**
 * tools/metal_poc/cmdbuf_latency_poc.mm
 *
 * Milestone 0.3 – Command buffer latency test
 *
 * Measures:
 *   1. Pure scheduling overhead: empty command buffer (create → commit → wait, no ops)
 *   2. Per-op latency: 1 GEMM per command buffer × many buffers
 *   3. Amortised latency: N GEMMs per command buffer (N = 2, 5, 10, 20)
 *
 * Uses a 512×512×512 FP32 GEMM as the unit op — representative of a
 * transformer attention projection head — small enough that command-buffer
 * overhead is a measurable fraction of per-op time.
 *
 * Build:
 *   clang++ -std=c++17 -O2 -DACCELERATE_NEW_LAPACK \
 *       -o cmdbuf_latency_poc tools/metal_poc/cmdbuf_latency_poc.mm \
 *       -framework Metal -framework Foundation -framework MetalPerformanceShaders
 *   ./cmdbuf_latency_poc
 *
 * PASS criteria (APPLE_M4_METAL_PLAN.md §0.3):
 *   Multi-op batching (10 ops/buffer) is measurably faster per-op
 *   than single-op commits.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Matrix dimensions for the unit op: 512×512×512 FP32 GEMM.
// At ~0.09 ms compute time, overhead is clearly visible.
static constexpr int DIM = 512;

static constexpr int WARMUP_SUBMISSIONS = 5;   // command-buffer submissions for warmup
static constexpr int TIMED_SUBMISSIONS  = 50;  // submissions per configuration

// Ops-per-buffer sweep
static constexpr int OPS_COUNTS[] = { 1, 2, 5, 10, 20, 25, 40 };
static constexpr int N_OPS_CONFIGS = 7;

static constexpr int EMPTY_CB_RUNS = 200;  // more runs for stable empty-overhead estimate

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double elapsed_ms(std::chrono::steady_clock::time_point t0,
                         std::chrono::steady_clock::time_point t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Encode `ops_per_buffer` GEMMs into a single command buffer, commit, and wait.
// Returns false on error.
static bool submit_batch(id<MTLCommandQueue>       queue,
                         MPSMatrixMultiplication*  gemm,
                         MPSMatrix*                matA,
                         MPSMatrix*                matB,
                         MPSMatrix*                matC,
                         int                       ops_per_buffer) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) return false;
    for (int i = 0; i < ops_per_buffer; ++i) {
      [gemm encodeToCommandBuffer:cb
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];
    }
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.status == MTLCommandBufferStatusCompleted);
  }
}

// Submit an empty command buffer (no ops) and wait — measures pure scheduling overhead.
static bool submit_empty(id<MTLCommandQueue> queue) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) return false;
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.status == MTLCommandBufferStatusCompleted);
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  // ── 1. Device setup ───────────────────────────────────────────────────────
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) { fprintf(stderr, "FAIL: No Metal device.\n"); return 1; }
  id<MTLCommandQueue> queue = [device newCommandQueue];

  printf("Metal device : %s\n", [[device name] UTF8String]);
  printf("Unit op      : %dx%dx%d FP32 GEMM\n\n", DIM, DIM, DIM);

  // ── 2. Allocate shared buffers ────────────────────────────────────────────
  const size_t sz = static_cast<size_t>(DIM) * DIM * sizeof(float);

  id<MTLBuffer> bufA = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];

  if (!bufA || !bufB || !bufC) {
    fprintf(stderr, "FAIL: MTLBuffer allocation failed.\n"); return 1;
  }

  // Fill inputs
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto* pA = static_cast<float*>([bufA contents]);
    auto* pB = static_cast<float*>([bufB contents]);
    for (int i = 0; i < DIM * DIM; ++i) { pA[i] = dist(rng); pB[i] = dist(rng); }
  }
  memset([bufC contents], 0, sz);

  // ── 3. Set up MPS GEMM ────────────────────────────────────────────────────
  const NSUInteger rowBytes = static_cast<NSUInteger>(DIM * sizeof(float));
  MPSMatrixDescriptor* desc =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)DIM
                                            columns:(NSUInteger)DIM
                                           rowBytes:rowBytes
                                           dataType:MPSDataTypeFloat32];
  MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
  MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
  MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];

  MPSMatrixMultiplication* gemm =
      [[MPSMatrixMultiplication alloc] initWithDevice:device
                                          resultRows:(NSUInteger)DIM
                                       resultColumns:(NSUInteger)DIM
                                     interiorColumns:(NSUInteger)DIM];

  // ── 4. Warmup ─────────────────────────────────────────────────────────────
  printf("Warming up (%d submissions)...\n", WARMUP_SUBMISSIONS);
  for (int i = 0; i < WARMUP_SUBMISSIONS; ++i) {
    submit_batch(queue, gemm, matA, matB, matC, 1);
  }
  for (int i = 0; i < WARMUP_SUBMISSIONS; ++i) {
    submit_empty(queue);
  }

  // ── 5. Empty command buffer overhead ─────────────────────────────────────
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < EMPTY_CB_RUNS; ++i) {
    submit_empty(queue);
  }
  auto t1 = std::chrono::steady_clock::now();
  const double empty_ms = elapsed_ms(t0, t1) / EMPTY_CB_RUNS;

  // ── 6. Per-op latency sweep (ops per buffer: 1, 2, 5, 10, 20) ───────────
  struct BenchResult {
    int    ops_per_buf;
    double total_submission_ms;  // wall time per buffer submission
    double per_op_ms;            // total_submission_ms / ops_per_buf
    double overhead_ms;          // extra cost vs pure compute (per submission)
  };

  std::vector<BenchResult> results;
  results.reserve(N_OPS_CONFIGS);

  for (int ci = 0; ci < N_OPS_CONFIGS; ++ci) {
    const int ops = OPS_COUNTS[ci];

    // Warmup for this config
    for (int i = 0; i < WARMUP_SUBMISSIONS; ++i) {
      submit_batch(queue, gemm, matA, matB, matC, ops);
    }

    auto ta = std::chrono::steady_clock::now();
    for (int i = 0; i < TIMED_SUBMISSIONS; ++i) {
      submit_batch(queue, gemm, matA, matB, matC, ops);
    }
    auto tb = std::chrono::steady_clock::now();

    const double submission_ms = elapsed_ms(ta, tb) / TIMED_SUBMISSIONS;
    const double per_op_ms     = submission_ms / ops;

    results.push_back({ops, submission_ms, per_op_ms,
                       submission_ms - ops * (per_op_ms)});  // filled below
  }

  // Pure compute time per op = per_op time at highest ops-per-buffer
  // (overhead approaches zero as ops/buffer → ∞)
  const double compute_per_op_ms = results.back().per_op_ms;
  for (auto& r : results) {
    // overhead per submission ≈ submission_ms - ops * compute_per_op_ms
    r.overhead_ms = r.total_submission_ms - r.ops_per_buf * compute_per_op_ms;
  }

  // ── 7. Report ─────────────────────────────────────────────────────────────
  const double flops_per_op = 2.0 * DIM * DIM * DIM;

  printf("\n");
  printf("=== Command Buffer Latency  (%dx%dx%d FP32 GEMM) ===\n\n", DIM, DIM, DIM);

  printf("  Empty CB overhead (create→commit→wait, no ops): %.3f ms\n\n", empty_ms);

  printf("  %-14s  %-16s  %-14s  %-16s  %-10s\n",
         "ops/buffer", "submission(ms)", "per-op(ms)", "overhead(ms)", "TFLOPS");
  printf("  %s\n", std::string(78, '-').c_str());

  for (const auto& r : results) {
    const double tflops = (flops_per_op * r.ops_per_buf)
                          / (r.total_submission_ms * 1e-3) / 1e12;
    printf("  %-14d  %-16.3f  %-14.3f  %-16.3f  %-10.3f\n",
           r.ops_per_buf,
           r.total_submission_ms,
           r.per_op_ms,
           r.overhead_ms,
           tflops);
  }

  // Speedup: per-op at 1 op/buf vs per-op at 10 ops/buf
  const double per_op_1  = results[0].per_op_ms;
  const double per_op_10 = results[3].per_op_ms;  // OPS_COUNTS[3] == 10
  const double speedup_10x = per_op_1 / per_op_10;

  printf("\n");
  printf("  Speedup (10 ops/buf vs 1 op/buf): %.2fx per-op\n", speedup_10x);
  printf("  Pure compute time / op (≈ %d ops/buf): %.3f ms  (%.2f TFLOPS)\n",
         OPS_COUNTS[N_OPS_CONFIGS - 1],
         compute_per_op_ms,
         flops_per_op / (compute_per_op_ms * 1e-3) / 1e12);

  // ── 8. Pass / Fail ────────────────────────────────────────────────────────
  // "Measurably faster" = at least 10% per-op improvement at 10 ops/buf vs 1 op/buf
  const bool pass = (speedup_10x > 1.10);

  printf("\n");
  printf("PASS/FAIL:\n");
  printf("  Multi-op batching measurably faster (>10%% at 10 ops/buf): %s\n",
         pass ? "PASS" : "FAIL");
  printf("\nOVERALL: %s\n\n", pass ? "PASS" : "FAIL");

  if (!pass) {
    fprintf(stderr,
            "NOTE: Per-op speedup %.2fx is below 1.10× — command buffer overhead\n"
            "      may be negligible for this op size. Re-run with smaller matrix.\n",
            speedup_10x);
  }

  return pass ? 0 : 1;
}
