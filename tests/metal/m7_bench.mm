// tests/metal/m7_bench.mm
//
// M7 Remaining Ops — Metal path vs CPU benchmark.
//
// All M7 ops use commit_and_wait() + CPU algorithm on Metal shared memory
// (MTLResourceStorageModeShared — no device copy needed on Apple Silicon).
//
//   "Metal" = commit_and_wait() [flush GPU] + algorithm on Metal-backed buffers
//   "CPU"   = same algorithm on the same Metal-backed buffers, no sync overhead
//
// Both paths use Metal-allocated I/O buffers (realistic: all CTranslate2
// tensors on Device::MPS are Metal-backed).  The only difference is the
// commit_and_wait() call in the Metal path.
//
// Key insight: commit_and_wait() adds ~0.4 ms fixed overhead.
// For small ops this dominates; for large ops the algorithm cost dominates.
//
// Ops covered: Concat, Split, Tile, TopK, TopPMask, Mean, MedianFilter,
//              GumbelMax, Multinomial.
//
// Build and run from the repository root:
//   clang++ -std=c++17 -O2 \
//     -I include -I src \
//     -DCT2_WITH_MPS \
//     tests/metal/m7_bench.mm \
//     src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
//     src/metal/primitives_memory.mm \
//     src/metal/primitives_elementwise.mm \
//     src/metal/primitives_reduction.mm \
//     src/metal/primitives_gemm.mm \
//     src/metal/primitives_transpose.mm \
//     src/metal/primitives_beam_search.mm \
//     src/allocator.cc src/devices.cc src/cpu/allocator.cc \
//     -framework Metal -framework Foundation \
//     -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph \
//     -o m7_bench && ./m7_bench

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "ctranslate2/allocator.h"
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
  return std::chrono::duration_cast<Micros>(
      Clock::now().time_since_epoch()).count();
}

template <typename Fn>
static double bench_median_us(int iters, Fn fn) {
  std::vector<double> times;
  times.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    double t0 = now_us(); fn(); double t1 = now_us();
    times.push_back(t1 - t0);
  }
  std::sort(times.begin(), times.end());
  return times[static_cast<size_t>(iters / 2)];
}

static int iter_count(dim_t total) {
  if (total <=    32768) return 200;
  if (total <=   524288) return  50;
  if (total <= 4194304)  return  20;
  return 8;
}

// ---------------------------------------------------------------------------
// Metal alloc helpers
// ---------------------------------------------------------------------------

template <typename T>
static T* metal_alloc(dim_t n) {
  return static_cast<T*>(get_allocator<Device::MPS>().allocate(
      static_cast<size_t>(n) * sizeof(T)));
}
template <typename T>
static void metal_free(T* p) { get_allocator<Device::MPS>().free(p); }

// ---------------------------------------------------------------------------
// Simple xorshift32 PRNG for reproducible data
// ---------------------------------------------------------------------------

static uint32_t g_rng = 0xDEADBEEFu;

static float next_float(float lo = -1.f, float hi = 1.f) {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  float t = static_cast<float>(g_rng) / static_cast<float>(UINT32_MAX);
  return lo + t * (hi - lo);
}

// Stateful mt19937 for GumbelMax/Multinomial (RNG cost is what we measure).
static std::mt19937& local_mt() {
  static std::mt19937 rng(42);
  return rng;
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

static void print_row(const char* label, double metal_us, double cpu_us) {
  double ratio  = cpu_us / metal_us;
  const char* w = (metal_us <= cpu_us) ? "Metal" : "CPU  ";
  std::printf("  %-56s  Metal %6.0f µs  CPU %5.0f µs  %4.2fx  %s\n",
              label, metal_us, cpu_us, ratio, w);
}

// ---------------------------------------------------------------------------
// Section 0: commit_and_wait baseline
// ---------------------------------------------------------------------------

static void bench_commit_baseline() {
  std::printf("--- commit_and_wait() baseline ---\n");
  double us = bench_median_us(100, [] { metal::commit_and_wait(); });
  std::printf("  %-56s  %6.0f µs\n", "empty commit (no pending GPU work)", us);
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 1: Concat  (axis=0 contiguous and axis=1 row-interleaved)
// ---------------------------------------------------------------------------

static void bench_concat() {
  std::printf("--- Concat ---\n");

  // axis=0: two inputs of equal size concatenated end-to-end
  {
    struct S { dim_t n; const char* label; };
    S shapes[] = {
      {   16384, "axis=0  [32×512]+[32×512]=[64×512]      (32K+32K)"},
      {  131072, "axis=0  [256×512]+[256×512]=[512×512]   (128K+128K)"},
      {  524288, "axis=0  [1024×512]+[1024×512]           (512K+512K)"},
      { 2097152, "axis=0  [1024×2048]+[1024×2048]         (2M+2M)"},
    };
    for (auto& s : shapes) {
      float* a   = metal_alloc<float>(s.n);
      float* b   = metal_alloc<float>(s.n);
      float* out = metal_alloc<float>(s.n * 2);
      for (dim_t i = 0; i < s.n; ++i) { a[i] = next_float(); b[i] = next_float(); }

      int iters = iter_count(s.n * 2);
      double metal_us = bench_median_us(iters, [&]{
        metal::commit_and_wait();
        std::memcpy(out,       a, static_cast<size_t>(s.n) * sizeof(float));
        std::memcpy(out + s.n, b, static_cast<size_t>(s.n) * sizeof(float));
      });
      double cpu_us = bench_median_us(iters, [&]{
        std::memcpy(out,       a, static_cast<size_t>(s.n) * sizeof(float));
        std::memcpy(out + s.n, b, static_cast<size_t>(s.n) * sizeof(float));
      });
      print_row(s.label, metal_us, cpu_us);
      metal_free(a); metal_free(b); metal_free(out);
    }
  }

  // axis=1: row-interleaved (copy wa then wb per row into wout-wide output)
  {
    struct S { dim_t nrows, wa, wb; const char* label; };
    S shapes[] = {
      {  256, 512,  512, "axis=1  [256×512]+[256×512]=[256×1024]  (interleaved)"},
      { 1024, 512, 1024, "axis=1  [1024×512]+[1024×1024]          (interleaved)"},
    };
    for (auto& s : shapes) {
      dim_t wout = s.wa + s.wb;
      float* a   = metal_alloc<float>(s.nrows * s.wa);
      float* b   = metal_alloc<float>(s.nrows * s.wb);
      float* out = metal_alloc<float>(s.nrows * wout);
      for (dim_t i = 0; i < s.nrows * s.wa; ++i) a[i] = next_float();
      for (dim_t i = 0; i < s.nrows * s.wb; ++i) b[i] = next_float();

      auto do_concat = [&]{
        for (dim_t r = 0; r < s.nrows; ++r) {
          std::memcpy(out + r * wout,        a + r * s.wa, static_cast<size_t>(s.wa) * sizeof(float));
          std::memcpy(out + r * wout + s.wa, b + r * s.wb, static_cast<size_t>(s.wb) * sizeof(float));
        }
      };

      int iters = iter_count(s.nrows * wout);
      double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_concat(); });
      double cpu_us   = bench_median_us(iters, [&]{ do_concat(); });
      print_row(s.label, metal_us, cpu_us);
      metal_free(a); metal_free(b); metal_free(out);
    }
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 2: Split  (axis=0 — two equal halves)
// ---------------------------------------------------------------------------

static void bench_split() {
  std::printf("--- Split (axis=0, two equal halves) ---\n");

  dim_t halfs[] = { 16384, 131072, 524288, 2097152 };
  const char* labels[] = {
    "split [64×512]→2×[32×512]          (32K+32K)",
    "split [512×512]→2×[256×512]        (128K+128K)",
    "split [2048×512]→2×[1024×512]      (512K+512K)",
    "split [2048×2048]→2×[1024×2048]    (2M+2M)",
  };

  for (int si = 0; si < 4; ++si) {
    dim_t half  = halfs[si];
    dim_t total = half * 2;
    float* in = metal_alloc<float>(total);
    float* oa = metal_alloc<float>(half);
    float* ob = metal_alloc<float>(half);
    for (dim_t i = 0; i < total; ++i) in[i] = next_float();

    auto do_split = [&]{
      std::memcpy(oa, in,        static_cast<size_t>(half) * sizeof(float));
      std::memcpy(ob, in + half, static_cast<size_t>(half) * sizeof(float));
    };

    int iters = iter_count(total);
    double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_split(); });
    double cpu_us   = bench_median_us(iters, [&]{ do_split(); });
    print_row(labels[si], metal_us, cpu_us);
    metal_free(in); metal_free(oa); metal_free(ob);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 3: Tile
// ---------------------------------------------------------------------------

static void bench_tile() {
  std::printf("--- Tile ---\n");

  struct S { dim_t outer, inner, tiles; const char* label; };
  S shapes[] = {
    {    1,  1024, 64, "tile [1×1024] ×64 → [64×1024]          (64K out)"},
    {  256,  1024,  4, "tile [256×1024] ×4 → [1024×1024]       (1M out)"},
    {    1, 65536, 16, "tile [1×65536] ×16 → [16×65536]        (1M out)"},
    { 1024,  2048,  4, "tile [1024×2048] ×4 → [4096×2048]      (8M out)"},
  };

  for (auto& s : shapes) {
    dim_t in_n  = s.outer * s.inner;
    dim_t out_n = s.outer * s.inner * s.tiles;
    float* in  = metal_alloc<float>(in_n);
    float* out = metal_alloc<float>(out_n);
    for (dim_t i = 0; i < in_n; ++i) in[i] = next_float();

    auto do_tile = [&]{
      float* dst = out;
      for (dim_t i = 0; i < s.outer; ++i) {
        const float* src = in + i * s.inner;
        for (dim_t t = 0; t < s.tiles; ++t) {
          std::memcpy(dst, src, static_cast<size_t>(s.inner) * sizeof(float));
          dst += s.inner;
        }
      }
    };

    int iters = iter_count(out_n);
    double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_tile(); });
    double cpu_us   = bench_median_us(iters, [&]{ do_tile(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(out);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 4: TopK
// ---------------------------------------------------------------------------

static void bench_topk() {
  std::printf("--- TopK ---\n");

  // k=1 (greedy decode — linear argmax)
  std::printf("  [k=1 — greedy decode: linear argmax]\n");
  struct K1 { dim_t batch, depth; const char* label; };
  K1 k1s[] = {
    { 1,  32768, "k=1  batch=1  vocab=32768  (greedy decode)"},
    { 1,  50257, "k=1  batch=1  vocab=50257  (GPT-2)"},
    { 1, 100352, "k=1  batch=1  vocab=100352 (Qwen/Llama-3)"},
    { 4,  32768, "k=1  batch=4  vocab=32768  (batch greedy)"},
  };
  for (auto& s : k1s) {
    dim_t n     = s.batch * s.depth;
    float*   in   = metal_alloc<float>(n);
    float*   vals = metal_alloc<float>(s.batch);
    int32_t* inds = metal_alloc<int32_t>(s.batch);
    for (dim_t i = 0; i < n; ++i) in[i] = next_float();

    auto do_topk1 = [&]{
      for (dim_t i = 0; i < s.batch; ++i) {
        const float* row = in + i * s.depth;
        auto it = std::max_element(row, row + s.depth);
        vals[i] = *it;
        inds[i] = static_cast<int32_t>(std::distance(row, it));
      }
    };

    double metal_us = bench_median_us(50, [&]{ metal::commit_and_wait(); do_topk1(); });
    double cpu_us   = bench_median_us(50, [&]{ do_topk1(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(vals); metal_free(inds);
  }

  // k>1 (beam search — partial sort)
  std::printf("  [k>1 — beam search: partial sort]\n");
  struct KN { dim_t batch, depth, k; const char* label; };
  KN kns[] = {
    { 1,  32768,  3, "k=3   batch=1  vocab=32768  (beam=3)"},
    { 1,  32768,  5, "k=5   batch=1  vocab=32768  (beam=5)"},
    { 1,  65536, 10, "k=10  batch=1  vocab=65536  (wide beam)"},
    { 4,  32768,  5, "k=5   batch=4  vocab=32768  (batch beam)"},
  };
  for (auto& s : kns) {
    dim_t n     = s.batch * s.depth;
    float*   in   = metal_alloc<float>(n);
    float*   vals = metal_alloc<float>(s.batch * s.k);
    int32_t* inds = metal_alloc<int32_t>(s.batch * s.k);
    for (dim_t i = 0; i < n; ++i) in[i] = next_float();
    std::vector<int32_t> ids(static_cast<size_t>(s.depth));

    auto do_topk = [&]{
      for (dim_t i = 0; i < s.batch; ++i) {
        const float* row = in + i * s.depth;
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + s.k, ids.end(),
            [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        for (dim_t j = 0; j < s.k; ++j) {
          inds[i * s.k + j] = ids[static_cast<size_t>(j)];
          vals[i * s.k + j] = row[ids[static_cast<size_t>(j)]];
        }
      }
    };

    double metal_us = bench_median_us(50, [&]{ metal::commit_and_wait(); do_topk(); });
    double cpu_us   = bench_median_us(50, [&]{ do_topk(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(vals); metal_free(inds);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 5: TopPMask
// ---------------------------------------------------------------------------

// Sort by prob descending; accumulate cumsum; mask once cumsum >= p.
static void bench_topp_mask() {
  std::printf("--- TopPMask ---\n");

  struct S { dim_t batch, depth; float p; const char* label; };
  S shapes[] = {
    { 1,  32768, 0.90f, "topp  p=0.90  batch=1  vocab=32768"},
    { 1,  50257, 0.95f, "topp  p=0.95  batch=1  vocab=50257  (GPT-2)"},
    { 4,  32768, 0.90f, "topp  p=0.90  batch=4  vocab=32768"},
    { 1, 100352, 0.95f, "topp  p=0.95  batch=1  vocab=100352"},
  };

  for (auto& s : shapes) {
    dim_t n = s.batch * s.depth;
    float* in  = metal_alloc<float>(n);
    float* out = metal_alloc<float>(n);
    // Softmax-like probabilities summing to ~1 per row
    for (dim_t bi = 0; bi < s.batch; ++bi) {
      float sum = 0.f;
      for (dim_t j = 0; j < s.depth; ++j) {
        float v = std::exp(next_float(-3.f, 3.f));
        in[bi * s.depth + j] = v; sum += v;
      }
      for (dim_t j = 0; j < s.depth; ++j) in[bi * s.depth + j] /= sum;
    }

    std::vector<int32_t> ids(static_cast<size_t>(s.depth));
    auto do_topp = [&]{
      for (dim_t bi = 0; bi < s.batch; ++bi) {
        const float* row = in  + bi * s.depth;
        float*       dst = out + bi * s.depth;
        std::copy(row, row + s.depth, dst);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(),
            [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        float cum = 0.f;
        for (dim_t j = 0; j < s.depth; ++j) {
          int32_t id = ids[static_cast<size_t>(j)];
          if (cum >= s.p) dst[id] = -1e9f;
          else            cum += row[id];
        }
      }
    };

    int iters = (n <= 131072) ? 30 : 10;
    double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_topp(); });
    double cpu_us   = bench_median_us(iters, [&]{ do_topp(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(out);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 6: Mean
// ---------------------------------------------------------------------------

// Input: [outer, axis_size, inner], Output: [outer, inner]
static void bench_mean() {
  std::printf("--- Mean ---\n");

  struct S { dim_t outer, axis, inner; const char* label; };
  S shapes[] = {
    {   1,    512,    1, "mean [1×512]        → [1]        (sequence mean)"},
    { 256,   1024,    1, "mean [256×1024]     → [256]      (batch last-axis)"},
    {   1,    256,  512, "mean [1×256×512]    → [1×512]    (axis=1)"},
    { 256,    512,  128, "mean [256×512×128]  → [256×128]  (mid-axis)"},
  };

  for (auto& s : shapes) {
    dim_t in_n  = s.outer * s.axis * s.inner;
    dim_t out_n = s.outer * s.inner;
    float* in  = metal_alloc<float>(in_n);
    float* out = metal_alloc<float>(out_n);
    for (dim_t i = 0; i < in_n; ++i) in[i] = next_float();

    float inv = 1.f / static_cast<float>(s.axis);
    auto do_mean = [&]{
      for (dim_t o = 0; o < s.outer; ++o)
        for (dim_t i = 0; i < s.inner; ++i) {
          float acc = 0.f;
          for (dim_t a = 0; a < s.axis; ++a)
            acc += in[(o * s.axis + a) * s.inner + i];
          out[o * s.inner + i] = acc * inv;
        }
    };

    int iters = iter_count(in_n);
    double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_mean(); });
    double cpu_us   = bench_median_us(iters, [&]{ do_mean(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(out);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 7: MedianFilter
// ---------------------------------------------------------------------------

// Sliding window median with reflect-at-boundary padding.
static void bench_median_filter() {
  std::printf("--- MedianFilter ---\n");

  struct S { dim_t batch, depth, width; const char* label; };
  S shapes[] = {
    {  1, 1500,  3, "median  width=3  [1×1500]        (Whisper 30s mel, 1 band)"},
    {  1, 1500,  7, "median  width=7  [1×1500]        (Whisper mel)"},
    { 80, 1500,  3, "median  width=3  [80×1500]       (full mel spectrogram)"},
    { 80, 3000,  5, "median  width=5  [80×3000]       (Whisper large mel)"},
  };

  for (auto& s : shapes) {
    dim_t n = s.batch * s.depth;
    float* in  = metal_alloc<float>(n);
    float* out = metal_alloc<float>(n);
    for (dim_t i = 0; i < n; ++i) in[i] = next_float();
    long rank = static_cast<long>(s.width / 2);
    std::vector<float> window(static_cast<size_t>(s.width));

    auto do_median = [&]{
      for (dim_t i = 0; i < s.batch; ++i) {
        const float* src = in  + i * s.depth;
        float*       dst = out + i * s.depth;
        for (dim_t j = 0; j < s.depth; ++j) {
          for (dim_t k = 0; k < s.width; ++k) {
            long rd = std::labs(static_cast<long>(j) + static_cast<long>(k) - rank);
            dim_t read = static_cast<dim_t>(rd);
            if (read >= s.depth) read = s.depth - (read - s.depth) - 2;
            window[k] = src[read];
          }
          std::nth_element(window.begin(), window.begin() + rank, window.end());
          dst[j] = window[static_cast<size_t>(rank)];
        }
      }
    };

    int iters = iter_count(n);
    double metal_us = bench_median_us(iters, [&]{ metal::commit_and_wait(); do_median(); });
    double cpu_us   = bench_median_us(iters, [&]{ do_median(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(out);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 8: GumbelMax  (add_gumbel_noise)
// ---------------------------------------------------------------------------

// Gumbel noise: z = -log(-log(U(eps,1))) > 0 always.
static void bench_gumbel_max() {
  std::printf("--- GumbelMax (add_gumbel_noise) ---\n");

  struct S { dim_t n; const char* label; };
  S shapes[] = {
    {  32768, "gumbel  batch=1  vocab=32768"},
    {  50257, "gumbel  batch=1  vocab=50257  (GPT-2)"},
    { 100352, "gumbel  batch=1  vocab=100352"},
  };

  for (auto& s : shapes) {
    float* x = metal_alloc<float>(s.n);
    for (dim_t i = 0; i < s.n; ++i) x[i] = next_float(-5.f, 5.f);

    std::uniform_real_distribution<float> dist(1e-7f, 1.f);
    auto& rng = local_mt();

    auto do_gumbel = [&]{
      for (dim_t i = 0; i < s.n; ++i)
        x[i] += -std::log(-std::log(dist(rng)));
    };

    double metal_us = bench_median_us(30, [&]{ metal::commit_and_wait(); do_gumbel(); });
    double cpu_us   = bench_median_us(30, [&]{ do_gumbel(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(x);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// Section 9: Multinomial
// ---------------------------------------------------------------------------

static void bench_multinomial() {
  std::printf("--- Multinomial ---\n");

  struct S { dim_t batch, depth, samples; const char* label; };
  S shapes[] = {
    { 1,  32768, 1, "multinomial  batch=1  vocab=32768   samples=1"},
    { 1,  50257, 1, "multinomial  batch=1  vocab=50257   samples=1  (GPT-2)"},
    { 4,  32768, 1, "multinomial  batch=4  vocab=32768   samples=1"},
    { 1,  32768, 4, "multinomial  batch=1  vocab=32768   samples=4"},
  };

  for (auto& s : shapes) {
    dim_t in_n  = s.batch * s.depth;
    dim_t out_n = s.batch * s.samples;
    float*   in  = metal_alloc<float>(in_n);
    int32_t* out = metal_alloc<int32_t>(out_n);
    // Uniform probability
    float uniform_p = 1.f / static_cast<float>(s.depth);
    for (dim_t i = 0; i < in_n; ++i) in[i] = uniform_p;

    std::vector<float> weights(static_cast<size_t>(s.depth));
    auto& rng = local_mt();

    auto do_multinomial = [&]{
      for (dim_t b = 0; b < s.batch; ++b) {
        const float* row = in + b * s.depth;
        std::copy(row, row + s.depth, weights.begin());
        std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
        for (dim_t ss = 0; ss < s.samples; ++ss)
          out[b * s.samples + ss] = dist(rng);
      }
    };

    double metal_us = bench_median_us(20, [&]{ metal::commit_and_wait(); do_multinomial(); });
    double cpu_us   = bench_median_us(20, [&]{ do_multinomial(); });
    print_row(s.label, metal_us, cpu_us);
    metal_free(in); metal_free(out);
  }
  std::printf("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  std::printf("=== M7 Remaining Ops — Metal Path vs CPU Benchmark (Apple M4) ===\n");
  std::printf("Metal = commit_and_wait() + CPU algorithm on Metal shared memory\n");
  std::printf("CPU   = same algorithm on the same Metal buffers, no sync overhead\n");
  std::printf("Note: CPU wins for small ops (commit overhead ~0.4 ms dominates);\n");
  std::printf("      Metal ≈ CPU for large ops (algorithm cost dominates).\n\n");

  bench_commit_baseline();
  bench_concat();
  bench_split();
  bench_tile();
  bench_topk();
  bench_topp_mask();
  bench_mean();
  bench_median_filter();
  bench_gumbel_max();
  bench_multinomial();

  return 0;
}
