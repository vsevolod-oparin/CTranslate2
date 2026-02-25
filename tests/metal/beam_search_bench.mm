// Benchmark + accuracy validation for M4.7 Metal beam-search primitives.
//
// Sections:
//   1. penalize_previous_tokens — accuracy (GPU vs CPU float32) and
//      performance (GPU vs CPU across realistic beam-search configurations).
//   2. prepare_length_mask — accuracy (CPU-side impl vs CPU reference) and
//      performance for typical attention configurations.
//   3. at<float> — latency after a GPU write.
//   4. logsumexp — accuracy and latency for float32 and float16 inputs.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_METAL \
//     tests/metal/beam_search_bench.mm \
//     src/metal/device.mm \
//     src/metal/utils.mm \
//     src/metal/allocator.mm \
//     src/metal/primitives.mm \
//     src/allocator.cc \
//     src/devices.cc \
//     src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o beam_search_bench && ./beam_search_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "ctranslate2/allocator.h"
#include "ctranslate2/primitives.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "metal/utils.h"

typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock  = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

static double now_us() {
  return std::chrono::duration_cast<Micros>(Clock::now().time_since_epoch()).count();
}

static volatile float g_sink = 0.f;

template <typename Fn>
static double bench_median_us(int iters, Fn fn) {
  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us();
    g_sink = fn();
    double t1 = now_us();
    times.push_back(t1 - t0);
  }
  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// ---------------------------------------------------------------------------
// Allocator helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::METAL>().allocate(n * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) {
  get_allocator<Device::METAL>().free(p);
}

// ---------------------------------------------------------------------------
// Reproducible RNG (xorshift32)
// ---------------------------------------------------------------------------

static uint32_t rng_state = 0xDEADBEEFu;

static float next_float(float lo, float hi) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  float t = static_cast<float>(rng_state) / static_cast<float>(UINT32_MAX);
  return lo + t * (hi - lo);
}

static int32_t next_int(int32_t lo, int32_t hi) {  // [lo, hi)
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return lo + (int32_t)(rng_state % (uint32_t)(hi - lo));
}

// ---------------------------------------------------------------------------
// CPU reference — penalize_previous_tokens
// ---------------------------------------------------------------------------

static void cpu_penalize(float* scores,
                          const float* previous_scores,
                          const int32_t* previous_ids,
                          float penalty,
                          dim_t batch_size, dim_t length, dim_t vocab_size) {
  for (dim_t i = 0; i < batch_size; ++i) {
    for (dim_t j = 0; j < length; ++j) {
      dim_t read_idx  = i * length + j;
      dim_t write_idx = i * vocab_size + previous_ids[read_idx];
      float s = previous_scores[read_idx];
      scores[write_idx] = (s < 0.f) ? s * penalty : s / penalty;
    }
  }
}

// ---------------------------------------------------------------------------
// Section 1: penalize_previous_tokens — accuracy
// ---------------------------------------------------------------------------

static void penalize_accuracy() {
  std::printf("=== 1. penalize_previous_tokens — Accuracy (GPU vs CPU, float32) ===\n");
  std::printf("Config: batch=%d, len=%d, vocab=%d, penalty=1.5\n\n", 4, 64, 32000);

  const dim_t batch = 4, len = 64, vocab = 32000;
  const float penalty = 1.5f;

  rng_state = 0xDEADBEEFu;

  // Initialise scores (same for both paths)
  std::vector<float> scores_init(batch * vocab);
  for (auto& v : scores_init) v = next_float(-10.f, 10.f);

  // previous_scores: log-probs — mix of negative and small-positive
  std::vector<float> prev_scores(batch * len);
  for (auto& v : prev_scores) v = next_float(-6.f, 1.f);

  // previous_ids: distinct within each batch item to avoid write conflicts
  std::vector<int32_t> prev_ids(batch * len);
  for (dim_t b = 0; b < batch; ++b) {
    // shuffle [0, vocab) and take first `len` entries
    std::vector<int32_t> pool(vocab);
    std::iota(pool.begin(), pool.end(), 0);
    // Fisher-Yates with our rng
    for (int32_t k = (int32_t)vocab - 1; k > 0; --k) {
      int32_t j = next_int(0, k + 1);
      std::swap(pool[k], pool[j]);
    }
    for (dim_t j = 0; j < len; ++j)
      prev_ids[b * len + j] = pool[j];
  }

  // --- GPU path ---
  float*    d_scores     = metal_alloc<float>(batch * vocab);
  float*    d_prev_scr   = metal_alloc<float>(batch * len);
  int32_t*  d_prev_ids   = metal_alloc<int32_t>(batch * len);

  std::memcpy(d_scores,   scores_init.data(), batch * vocab * sizeof(float));
  std::memcpy(d_prev_scr, prev_scores.data(), batch * len   * sizeof(float));
  std::memcpy(d_prev_ids, prev_ids.data(),    batch * len   * sizeof(int32_t));

  primitives<Device::METAL>::penalize_previous_tokens(
      d_scores, d_prev_scr, d_prev_ids, penalty, batch, len, vocab);
  metal::commit_and_wait();

  // --- CPU path ---
  std::vector<float> cpu_scores = scores_init;
  cpu_penalize(cpu_scores.data(), prev_scores.data(), prev_ids.data(),
               penalty, batch, len, vocab);

  // --- Diff ---
  float max_abs = 0.f, sum_sq = 0.f;
  for (dim_t i = 0; i < batch * vocab; ++i) {
    float diff = std::fabs(d_scores[i] - cpu_scores[i]);
    if (diff > max_abs) max_abs = diff;
    sum_sq += diff * diff;
  }
  float rms = std::sqrt(sum_sq / (float)(batch * vocab));

  bool pass = (max_abs < 1e-5f);
  std::printf("max_abs_diff = %.3e   rms_diff = %.3e   %s\n\n",
              (double)max_abs, (double)rms, pass ? "PASS" : "FAIL");

  metal_free(d_scores); metal_free(d_prev_scr); metal_free(d_prev_ids);
}

// ---------------------------------------------------------------------------
// Section 2: penalize_previous_tokens — performance
// ---------------------------------------------------------------------------
//
// Benchmark across realistic beam-search configurations.
// The key dimensions are batch_size (= beam_width, typically 4–8),
// length (= number of tokens generated so far, 1 → max_seq_len),
// and vocab_size (32k–250k for common LLMs).
//
// GPU: encode + commit_and_wait (includes per-submission overhead).
// CPU: sequential nested loop (-O2).

static void penalize_perf_row(const char* label,
                               dim_t batch, dim_t len, dim_t vocab,
                               int iters) {
  const float penalty = 1.5f;

  rng_state = 0xABCDEF01u + (uint32_t)(batch * len);

  std::vector<float>   scores_init(batch * vocab);
  std::vector<float>   prev_scores(batch * len);
  std::vector<int32_t> prev_ids(batch * len);

  for (auto& v : scores_init) v = next_float(-10.f, 10.f);
  for (auto& v : prev_scores) v = next_float(-6.f, 1.f);
  // Use modulo for speed; potential conflicts are OK for perf benchmark
  for (dim_t b = 0; b < batch; ++b)
    for (dim_t j = 0; j < len; ++j)
      prev_ids[b * len + j] = (int32_t)((b * 7919 + j * 131) % vocab);

  float*   d_scores   = metal_alloc<float>(batch * vocab);
  float*   d_prev_scr = metal_alloc<float>(batch * len);
  int32_t* d_prev_ids = metal_alloc<int32_t>(batch * len);
  std::vector<float> cpu_scores(batch * vocab);

  // Warmup + PSO compile
  std::memcpy(d_scores, scores_init.data(), batch * vocab * sizeof(float));
  std::memcpy(d_prev_scr, prev_scores.data(), batch * len * sizeof(float));
  std::memcpy(d_prev_ids, prev_ids.data(), batch * len * sizeof(int32_t));
  primitives<Device::METAL>::penalize_previous_tokens(
      d_scores, d_prev_scr, d_prev_ids, penalty, batch, len, vocab);
  metal::commit_and_wait();

  // GPU benchmark: reinitialise scores each iter so the kernel actually writes
  double gpu_us = bench_median_us(iters, [&] {
    std::memcpy(d_scores, scores_init.data(), batch * vocab * sizeof(float));
    primitives<Device::METAL>::penalize_previous_tokens(
        d_scores, d_prev_scr, d_prev_ids, penalty, batch, len, vocab);
    metal::commit_and_wait();
    return d_scores[0];
  });

  // CPU benchmark
  double cpu_us = bench_median_us(iters, [&] {
    std::memcpy(cpu_scores.data(), scores_init.data(), batch * vocab * sizeof(float));
    cpu_penalize(cpu_scores.data(), prev_scores.data(), prev_ids.data(),
                 penalty, batch, len, vocab);
    return cpu_scores[0];
  });

  double ratio = cpu_us / gpu_us;
  const char* winner = (gpu_us <= cpu_us) ? "GPU" : "CPU";
  std::printf("%-32s  %8.1f  %8.1f  %6.2fx  %s wins\n",
              label, gpu_us, cpu_us, ratio, winner);

  metal_free(d_scores); metal_free(d_prev_scr); metal_free(d_prev_ids);
}

static void penalize_perf() {
  std::printf("=== 2. penalize_previous_tokens — Performance ===\n");
  std::printf("GPU time includes encode + commit_and_wait + score reinit.\n");
  std::printf("CPU time includes memcpy reinit + nested loop.\n\n");
  std::printf("%-32s  %8s  %8s  %7s  %s\n",
              "Config", "GPU(μs)", "CPU(μs)", "Ratio", "Winner");
  std::printf("%s\n", std::string(72, '-').c_str());

  // Vary len (decode step depth) with typical beam=4, vocab=32k
  penalize_perf_row("batch=4 len=1   vocab=32k",  4,   1, 32000, 100);
  penalize_perf_row("batch=4 len=16  vocab=32k",  4,  16, 32000,  80);
  penalize_perf_row("batch=4 len=64  vocab=32k",  4,  64, 32000,  60);
  penalize_perf_row("batch=4 len=256 vocab=32k",  4, 256, 32000,  40);
  penalize_perf_row("batch=4 len=512 vocab=32k",  4, 512, 32000,  20);

  std::printf("\n");

  // Vary vocab size (larger vocab = larger scores scatter write)
  penalize_perf_row("batch=4 len=64 vocab=8k",    4,  64,  8000,  60);
  penalize_perf_row("batch=4 len=64 vocab=32k",   4,  64, 32000,  60);
  penalize_perf_row("batch=4 len=64 vocab=128k",  4,  64,128000,  40);
  penalize_perf_row("batch=4 len=64 vocab=250k",  4,  64,250000,  30);

  std::printf("\n");

  // Vary batch size (beam width)
  penalize_perf_row("batch=1  len=64 vocab=32k",  1,  64, 32000,  60);
  penalize_perf_row("batch=4  len=64 vocab=32k",  4,  64, 32000,  60);
  penalize_perf_row("batch=8  len=64 vocab=32k",  8,  64, 32000,  60);
  penalize_perf_row("batch=16 len=64 vocab=32k", 16,  64, 32000,  40);

  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 3: prepare_length_mask — accuracy + performance
// ---------------------------------------------------------------------------

static void prepare_mask_cpu_ref(const int32_t* lengths,
                                  dim_t batch, dim_t heads, dim_t queries,
                                  bool mask_future, bool multi_query,
                                  int32_t* mask) {
  for (dim_t b = 0; b < batch; ++b) {
    const int32_t length = lengths[b];
    auto* bm = mask + b * heads * queries;
    for (dim_t i = 0; i < heads * queries; ++i) {
      bm[i] = (mask_future
               ? std::min(length, (int32_t)((multi_query ? i / heads
                                                         : i % queries) + 1))
               : length);
    }
  }
}

static void mask_accuracy_and_perf() {
  std::printf("=== 3. prepare_length_mask — Accuracy + Performance ===\n\n");

  // --- Accuracy ---
  {
    const dim_t batch=4, heads=8, queries=512;
    int32_t* d_lengths = metal_alloc<int32_t>(batch);
    int32_t* d_mask    = metal_alloc<int32_t>(batch * heads * queries);

    rng_state = 0x12345678u;
    for (dim_t b = 0; b < batch; ++b)
      d_lengths[b] = next_int(1, (int32_t)queries + 1);

    std::vector<int32_t> ref_mask(batch * heads * queries);
    prepare_mask_cpu_ref(d_lengths, batch, heads, queries,
                         /*mask_future=*/true, /*multi_query=*/false,
                         ref_mask.data());

    primitives<Device::METAL>::prepare_length_mask(
        d_lengths, batch, heads, queries,
        /*mask_future=*/true, /*multi_query=*/false, d_mask);

    int mismatches = 0;
    for (dim_t i = 0; i < batch * heads * queries; ++i)
      if (d_mask[i] != ref_mask[i]) ++mismatches;

    std::printf("Accuracy (batch=%lld, heads=%lld, queries=%lld, mask_future=true):\n",
                (long long)batch, (long long)heads, (long long)queries);
    std::printf("  mismatches = %d   %s\n\n",
                mismatches, mismatches == 0 ? "PASS" : "FAIL");

    metal_free(d_lengths); metal_free(d_mask);
  }

  // --- Performance ---
  std::printf("Performance (median latency μs):\n");
  std::printf("%-36s  %10s\n", "Config", "Latency(μs)");
  std::printf("%s\n", std::string(50, '-').c_str());

  struct MaskSpec {
    const char* label;
    dim_t batch, heads, queries;
    bool  mask_future, multi_query;
    int   iters;
  };

  std::vector<MaskSpec> specs = {
    {"batch=1  h=8  q=128  causal",    1,  8,  128, true,  false, 500},
    {"batch=4  h=8  q=512  causal",    4,  8,  512, true,  false, 200},
    {"batch=8  h=8  q=512  causal",    8,  8,  512, true,  false, 200},
    {"batch=4  h=32 q=2048 causal",    4, 32, 2048, true,  false,  80},
    {"batch=8  h=32 q=2048 causal",    8, 32, 2048, true,  false,  50},
    {"batch=4  h=8  q=512  padded",    4,  8,  512, false, false, 200},
    {"batch=4  h=8  q=512  mquery",    4,  8,  512, true,  true,  200},
  };

  for (const auto& s : specs) {
    int32_t* d_lengths = metal_alloc<int32_t>(s.batch);
    int32_t* d_mask    = metal_alloc<int32_t>(s.batch * s.heads * s.queries);

    rng_state = 0xFACEFACEu;
    for (dim_t b = 0; b < s.batch; ++b)
      d_lengths[b] = next_int(1, (int32_t)s.queries + 1);

    // Warmup
    primitives<Device::METAL>::prepare_length_mask(
        d_lengths, s.batch, s.heads, s.queries,
        s.mask_future, s.multi_query, d_mask);

    double lat_us = bench_median_us(s.iters, [&] {
      primitives<Device::METAL>::prepare_length_mask(
          d_lengths, s.batch, s.heads, s.queries,
          s.mask_future, s.multi_query, d_mask);
      return (float)d_mask[0];
    });

    std::printf("%-36s  %10.2f\n", s.label, lat_us);

    metal_free(d_lengths); metal_free(d_mask);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 4: at<float> — latency after GPU write
// ---------------------------------------------------------------------------

static void at_perf() {
  std::printf("=== 4. at<float> — Latency after GPU write ===\n\n");

  const dim_t n = 256;
  float* d = metal_alloc<float>(n);
  for (dim_t i = 0; i < n; ++i) d[i] = (float)i;

  // Warmup
  primitives<Device::METAL>::add(1.f, d, d, n);  // GPU encode
  float v = primitives<Device::METAL>::at(d, 0);  // flush + read

  double gpu_write_then_at = bench_median_us(200, [&] {
    primitives<Device::METAL>::add(1.f, d, d, n);  // GPU encode (increments by 1)
    float val = primitives<Device::METAL>::at(d, 0);  // flush + CPU read
    return val;
  });

  // Baseline: at() with no pending GPU work (just commit empty buffer + read)
  metal::commit_and_wait();
  double at_no_gpu_work = bench_median_us(200, [&] {
    return (float)primitives<Device::METAL>::at(d, 0);
  });

  std::printf("at() after GPU add kernel:  %.1f μs  (flush + read)\n",
              gpu_write_then_at);
  std::printf("at() with no pending work:  %.1f μs  (empty flush + read)\n\n",
              at_no_gpu_work);

  metal_free(d);
}

// ---------------------------------------------------------------------------
// Section 5: logsumexp — accuracy + latency
// ---------------------------------------------------------------------------

static void logsumexp_bench() {
  std::printf("=== 5. logsumexp — Accuracy + Latency ===\n\n");

  // --- Accuracy ---
  const dim_t n = 1024;
  rng_state = 0xCAFEBABEu;

  float*   d_f32 = metal_alloc<float>(n);
  ct2_f16* d_f16 = metal_alloc<ct2_f16>(n);
  std::vector<float> host(n);

  for (dim_t i = 0; i < n; ++i) {
    host[i] = next_float(-3.f, 3.f);
    d_f32[i] = host[i];
    d_f16[i] = (ct2_f16)host[i];
  }

  // CPU reference (numerically stable)
  float max_h = *std::max_element(host.begin(), host.end());
  float sum_h = 0.f;
  for (float x : host) sum_h += std::exp(x - max_h);
  float ref = std::log(sum_h) + max_h;

  float res_f32 = primitives<Device::METAL>::logsumexp(d_f32, n);
  float res_f16 = primitives<Device::METAL>::logsumexp(d_f16, n);

  std::printf("Accuracy (n=%lld random floats in [-3, 3]):\n", (long long)n);
  std::printf("  CPU ref   = %.6f\n", (double)ref);
  std::printf("  GPU f32   = %.6f   abs_diff=%.2e   %s\n",
              (double)res_f32, (double)std::fabs(res_f32 - ref),
              std::fabs(res_f32 - ref) < 1e-4f ? "PASS" : "FAIL");
  std::printf("  GPU f16   = %.6f   abs_diff=%.2e   %s\n\n",
              (double)res_f16, (double)std::fabs(res_f16 - ref),
              std::fabs(res_f16 - ref) < 2e-2f ? "PASS" : "FAIL");

  // --- Latency vs size ---
  std::printf("Latency (CPU-side after flush, median μs):\n");
  std::printf("%-10s  %12s  %12s\n", "n", "float32(μs)", "float16(μs)");
  std::printf("%s\n", std::string(38, '-').c_str());

  for (dim_t sz : {32, 256, 1024, 4096, 16384, 65536}) {
    float*   d32 = metal_alloc<float>(sz);
    ct2_f16* d16 = metal_alloc<ct2_f16>(sz);
    for (dim_t i = 0; i < sz; ++i) { d32[i] = (float)i * 0.001f; d16[i] = (ct2_f16)d32[i]; }
    // Warmup
    (void)primitives<Device::METAL>::logsumexp(d32, sz);

    int iters = (sz <= 1024) ? 500 : (sz <= 16384) ? 100 : 40;

    double lat32 = bench_median_us(iters, [&] {
      return primitives<Device::METAL>::logsumexp(d32, sz);
    });
    double lat16 = bench_median_us(iters, [&] {
      return primitives<Device::METAL>::logsumexp(d16, sz);
    });

    std::printf("%-10lld  %12.2f  %12.2f\n", (long long)sz, lat32, lat16);
    metal_free(d32); metal_free(d16);
  }
  std::printf("\n");

  metal_free(d_f32); metal_free(d_f16);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M4.7 Beam-search Primitives: Accuracy & Performance ===\n");
  std::printf("Hardware: Apple Silicon (Metal, unified memory)\n");
  std::printf("GPU times include commit_and_wait (worst-case standalone latency).\n");
  std::printf("In the full inference pipeline, encode-only operations pay zero\n");
  std::printf("sync overhead — cost is hidden inside the GEMM command buffer.\n\n");

  penalize_accuracy();
  penalize_perf();
  mask_accuracy_and_perf();
  at_perf();
  logsumexp_bench();

  std::printf("Done.\n");
  return 0;
}
