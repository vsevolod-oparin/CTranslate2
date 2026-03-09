// src/ops/multinomial_metal.mm
//
// M11.20 — GPU Multinomial Sampling for Metal.
//
// Replaces the M7 CPU-side std::discrete_distribution with an encode-only
// GPU kernel, eliminating ~756 commit_and_wait() syncs during Whisper decode.
//
// Algorithm (sample_size == 1):
//   Each threadgroup handles one batch row.  Threads cooperatively scan the
//   probability vector, accumulating partial sums.  A threadgroup-level
//   exclusive prefix sum converts per-thread partial sums to cumulative
//   offsets.  Each thread then re-scans its chunk with the offset and the
//   first thread whose cumulative sum exceeds the random threshold writes
//   the sampled index.  Ties are broken by the lowest index (left-most).
//
// For sample_size > 1 (used only by GumbelMax path, not Multinomial in
// practice), falls back to the CPU path.

#include "ctranslate2/ops/multinomial.h"

#include <random>

#include "ctranslate2/random.h"
#include "metal/primitives_infra.h"

namespace {

// ---------------------------------------------------------------------------
// MSL kernel for multinomial sampling (sample_size == 1).
//
// Supports float and half inputs via template specialization at PSO level.
// Output is always int (mapped to int32_t on host).
//
// Parameters:
//   probs:      input probability array  [batch_size × class_size]
//   output:     output index array       [batch_size × 1]
//   class_size: number of classes (vocabulary size)
//   rand_vals:  per-batch uniform random values in [0, 1)
//
// Grid:  threadgroups = batch_size,  threads_per_tg = TGS (e.g. 256)
// ---------------------------------------------------------------------------

static const char* kMultinomialMSL = R"(
#include <metal_stdlib>
using namespace metal;

// Threadgroup scratch for prefix sum.  TGS ≤ 1024.
constant uint TGS_CONST [[function_constant(0)]];

kernel void multinomial_float(
    device const float*  probs      [[buffer(0)]],
    device       int*    output     [[buffer(1)]],
    constant     uint&   class_size [[buffer(2)]],
    device const float*  rand_vals  [[buffer(3)]],
    uint  batch_id [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  tgs      [[threads_per_threadgroup]])
{
    // Pointer to this batch row's probabilities.
    device const float* row = probs + batch_id * class_size;
    float threshold = rand_vals[batch_id];

    // --- Phase 1: Each thread sums its chunk ---
    // Thread tid handles indices [tid, tid+tgs, tid+2*tgs, ...] (strided).
    float local_sum = 0.0f;
    for (uint i = tid; i < class_size; i += tgs)
        local_sum += row[i];

    // --- Phase 2: Threadgroup exclusive prefix sum of local_sums ---
    threadgroup float shared_sums[1024];
    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple sequential prefix sum (tgs ≤ 1024; done by thread 0).
    if (tid == 0) {
        // Scale threshold by total sum (handles unnormalized distributions).
        float total = 0.0f;
        for (uint i = 0; i < tgs; ++i)
            total += shared_sums[i];
        // Now do exclusive prefix sum in-place.
        float running = 0.0f;
        for (uint i = 0; i < tgs; ++i) {
            float val = shared_sums[i];
            shared_sums[i] = running;
            running += val;
        }
        // Store scaled threshold in slot after prefix sums.
        // total should be ~1.0 for softmax output; multiply threshold.
        shared_sums[tgs] = threshold * total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float my_offset = shared_sums[tid];
    float scaled_threshold = shared_sums[tgs];

    // --- Phase 3: Each thread scans its chunk with offset ---
    // The winning thread is the one whose cumsum first exceeds threshold.
    float cumsum = my_offset;
    int my_result = -1;  // No match found.
    for (uint i = tid; i < class_size; i += tgs) {
        cumsum += row[i];
        if (cumsum > scaled_threshold) {
            my_result = int(i);
            break;
        }
    }

    // --- Phase 4: Find the thread with the smallest winning index ---
    // Write each thread's result to shared memory; thread 0 picks the min.
    threadgroup int shared_results[1024];
    shared_results[tid] = my_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int best = int(class_size) - 1;  // Fallback: last class.
        for (uint i = 0; i < tgs; ++i) {
            int r = shared_results[i];
            if (r >= 0 && r < best)
                best = r;
        }
        output[batch_id] = best;
    }
}

kernel void multinomial_half(
    device const half*   probs      [[buffer(0)]],
    device       int*    output     [[buffer(1)]],
    constant     uint&   class_size [[buffer(2)]],
    device const float*  rand_vals  [[buffer(3)]],
    uint  batch_id [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  tgs      [[threads_per_threadgroup]])
{
    device const half* row = probs + batch_id * class_size;
    float threshold = rand_vals[batch_id];

    float local_sum = 0.0f;
    for (uint i = tid; i < class_size; i += tgs)
        local_sum += float(row[i]);

    threadgroup float shared_sums[1024];
    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < tgs; ++i)
            total += shared_sums[i];
        float running = 0.0f;
        for (uint i = 0; i < tgs; ++i) {
            float val = shared_sums[i];
            shared_sums[i] = running;
            running += val;
        }
        shared_sums[tgs] = threshold * total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float my_offset = shared_sums[tid];
    float scaled_threshold = shared_sums[tgs];

    float cumsum = my_offset;
    int my_result = -1;
    for (uint i = tid; i < class_size; i += tgs) {
        cumsum += float(row[i]);
        if (cumsum > scaled_threshold) {
            my_result = int(i);
            break;
        }
    }

    threadgroup int shared_results[1024];
    shared_results[tid] = my_result;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int best = int(class_size) - 1;
        for (uint i = 0; i < tgs; ++i) {
            int r = shared_results[i];
            if (r >= 0 && r < best)
                best = r;
        }
        output[batch_id] = best;
    }
}
)";

// ---------------------------------------------------------------------------
// Library / PSO getters
// ---------------------------------------------------------------------------

static id<MTLLibrary> get_multinomial_library() {
  static id<MTLLibrary> lib = nil;
  static std::once_flag flag;
  return compile_library_once(flag, lib, kMultinomialMSL, "multinomial");
}

static PSOCache _multinomial_pso_cache;

static id<MTLComputePipelineState> get_multinomial_pso(const char* name) {
  return _multinomial_pso_cache.get(get_multinomial_library, name);
}

// ---------------------------------------------------------------------------
// GPU dispatch for sample_size == 1
// ---------------------------------------------------------------------------

template <typename T>
static void dispatch_multinomial_gpu(const T* probs,
                                     int32_t* output,
                                     ctranslate2::dim_t batch_size,
                                     ctranslate2::dim_t class_size,
                                     float* rand_vals_host) {
  // Select kernel by type.
  const char* kernel_name;
  if constexpr (std::is_same_v<T, float>) {
    kernel_name = "multinomial_float";
  } else if constexpr (std::is_same_v<T, ctranslate2::float16_t>) {
    kernel_name = "multinomial_half";
  } else {
    // BF16 falls back to CPU path (bfloat in MSL requires separate kernel).
    throw std::runtime_error("GPU multinomial: bfloat16 not supported");
  }

  id<MTLComputePipelineState> pso = get_multinomial_pso(kernel_name);

  // Get Metal buffers for probs and output.
  NSUInteger off_probs = 0, off_output = 0;
  id<MTLBuffer> buf_probs  = ctranslate2::metal_buffer_for_ptr(probs, &off_probs);
  id<MTLBuffer> buf_output = ctranslate2::metal_buffer_for_ptr(output, &off_output);

  // Create a small temporary buffer for random values.
  // batch_size is typically 1–5, so this is tiny.
  id<MTLBuffer> buf_rand = [ctranslate2::metal::get_metal_device()
      newBufferWithBytes:rand_vals_host
                  length:batch_size * sizeof(float)
                 options:MTLResourceStorageModeShared];

  uint32_t cs = ctranslate2::dim_t(class_size);

  // Encode.
  id<MTLCommandBuffer> cmd = ctranslate2::metal::get_current_command_buffer();
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];

  [enc setComputePipelineState:pso];
  [enc setBuffer:buf_probs  offset:off_probs  atIndex:0];
  [enc setBuffer:buf_output offset:off_output atIndex:1];
  [enc setBytes:&cs length:sizeof(cs) atIndex:2];
  [enc setBuffer:buf_rand offset:0 atIndex:3];

  NSUInteger tgs = std::min<NSUInteger>(256, pso.maxTotalThreadsPerThreadgroup);
  [enc dispatchThreadgroups:MTLSizeMake(batch_size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
  [enc endEncoding];

  // Protect input/output buffers from deferred-free reuse.
  ctranslate2::metal::protect_buffer_by_base([buf_probs contents]);
  ctranslate2::metal::protect_buffer_by_base([buf_output contents]);
  // buf_rand is ARC-managed and will live until autorelease pool drain,
  // which happens after commit_and_wait(). Safe.
}

}  // namespace


namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      const dim_t class_size  = input.dim(-1);
      const dim_t batch_size  = input.size() / class_size;

      // M11.20: GPU path for sample_size == 1 (the only case used by
      // RandomSampler).  Encode-only — zero commit_and_wait() syncs.
      if (_sample_size == 1
          && !std::is_same_v<T, bfloat16_t>
          && batch_size > 0) {
        // Generate random values on CPU.
        auto& generator = get_random_generator();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> rand_vals(batch_size);
        for (dim_t i = 0; i < batch_size; ++i)
          rand_vals[i] = dist(generator);

        dispatch_multinomial_gpu(input.data<T>(),
                                 output.data<int32_t>(),
                                 batch_size,
                                 class_size,
                                 rand_vals.data());
        return;
      }

      // Fallback: CPU path (sample_size > 1 or bfloat16).
      CT2_COMMIT_AND_WAIT();

      const T*      inp = input.data<T>();
      int32_t*      out = output.data<int32_t>();

      auto& generator = get_random_generator();

      for (dim_t i = 0; i < batch_size; ++i) {
        const T*     row_in  = inp + i * class_size;
        int32_t*     row_out = out + i * _sample_size;

        std::discrete_distribution<int32_t> dist(row_in, row_in + class_size);
        for (dim_t j = 0; j < _sample_size; ++j)
          row_out[j] = dist(generator);
      }
    }

#define DECLARE_IMPL(T)                                                   \
    template void                                                         \
    Multinomial::compute<Device::METAL, T>(const StorageView& input,     \
                                            StorageView& output) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(ctranslate2::float16_t)
    DECLARE_IMPL(ctranslate2::bfloat16_t)
#undef DECLARE_IMPL

  }  // namespace ops
}  // namespace ctranslate2
