// Test: MoonshineAudioFrontend components (M17.2)
// Tests CMVN normalization, asinh compression, and causal padding logic.
// Does NOT test the full pipeline (requires model weights).

#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <numeric>

#include "ctranslate2/storage_view.h"
#include "ctranslate2/ops/concat.h"

using namespace ctranslate2;

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(name, cond, msg) do {       \
  if (!(cond)) {                                \
    printf("  FAIL: %s — %s\n", name, msg);     \
    g_fail++;                                   \
  } else {                                      \
    g_pass++;                                   \
  }                                             \
} while(0)

#define ASSERT_NEAR(name, actual, expected, eps, msg) do { \
  if (std::fabs((actual) - (expected)) > (eps)) {          \
    printf("  FAIL: %s — %s (got %f, expected %f)\n",      \
           name, msg, (double)(actual), (double)(expected));\
    g_fail++;                                              \
  } else {                                                 \
    g_pass++;                                              \
  }                                                        \
} while(0)

// ---- Test CMVN logic (replicated from moonshine.cc) ----

static void apply_cmvn_ref(float* data, dim_t batch, dim_t num_frames, dim_t fs) {
  const float eps = 1e-5f;
  for (dim_t b = 0; b < batch; ++b) {
    for (dim_t f = 0; f < num_frames; ++f) {
      float* frame = data + (b * num_frames + f) * fs;
      float sum = 0;
      for (dim_t i = 0; i < fs; ++i)
        sum += frame[i];
      const float mean = sum / static_cast<float>(fs);
      float var_sum = 0;
      for (dim_t i = 0; i < fs; ++i) {
        frame[i] -= mean;
        var_sum += frame[i] * frame[i];
      }
      const float rms = std::sqrt(var_sum / static_cast<float>(fs) + eps);
      const float inv_rms = 1.0f / rms;
      for (dim_t i = 0; i < fs; ++i)
        frame[i] *= inv_rms;
    }
  }
}

static void test_cmvn_zero_mean() {
  const char* name = "cmvn_zero_mean";
  // After CMVN, each frame should have mean ~0
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};  // 1 batch, 1 frame, 8 elements
  apply_cmvn_ref(data.data(), 1, 1, 8);

  float sum = 0;
  for (float v : data) sum += v;
  ASSERT_NEAR(name, sum, 0.0f, 1e-5f, "mean should be ~0 after CMVN");
  printf("  PASS: %s\n", name);
}

static void test_cmvn_unit_variance() {
  const char* name = "cmvn_unit_variance";
  // After CMVN, each frame should have RMS ~1
  std::vector<float> data = {10, 20, 30, 40};
  apply_cmvn_ref(data.data(), 1, 1, 4);

  float sq_sum = 0;
  for (float v : data) sq_sum += v * v;
  float rms = std::sqrt(sq_sum / 4.0f);
  ASSERT_NEAR(name, rms, 1.0f, 1e-4f, "RMS should be ~1 after CMVN");
  printf("  PASS: %s\n", name);
}

static void test_cmvn_multi_frame() {
  const char* name = "cmvn_multi_frame";
  // 2 frames, each independently normalized
  std::vector<float> data = {
    1, 2, 3, 4,     // frame 0
    100, 200, 300, 400  // frame 1 (much larger values)
  };
  apply_cmvn_ref(data.data(), 1, 2, 4);

  // Frame 0 and frame 1 should both have mean ~0, RMS ~1
  float mean0 = 0, mean1 = 0;
  for (int i = 0; i < 4; ++i) { mean0 += data[i]; mean1 += data[4+i]; }
  ASSERT_NEAR(name, mean0 / 4, 0.0f, 1e-5f, "frame 0 mean ~0");
  ASSERT_NEAR(name, mean1 / 4, 0.0f, 1e-5f, "frame 1 mean ~0");

  // Both frames should produce identical output (same pattern, just scaled)
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(name, data[i], data[4+i], 1e-5f,
                "frames with same pattern should normalize identically");
  }
  printf("  PASS: %s\n", name);
}

static void test_cmvn_constant_frame() {
  const char* name = "cmvn_constant_frame";
  // Constant frame: mean=5, var=0 → output should be all 0 (eps prevents div by zero)
  std::vector<float> data = {5, 5, 5, 5};
  apply_cmvn_ref(data.data(), 1, 1, 4);
  for (float v : data) {
    ASSERT_NEAR(name, v, 0.0f, 1e-2f, "constant frame should normalize to ~0");
  }
  printf("  PASS: %s\n", name);
}

// ---- Test asinh logic ----

static void test_asinh_identity() {
  const char* name = "asinh_identity";
  // With log_k = 0, scale = exp(0) = 1, so output = asinh(x)
  float x = 1.0f;
  float result = std::asinhf(x * std::exp(0.0f));
  float expected = std::asinhf(1.0f);  // ~0.8814
  ASSERT_NEAR(name, result, expected, 1e-6f, "asinh(1) should be ~0.8814");
  printf("  PASS: %s\n", name);
}

static void test_asinh_with_scale() {
  const char* name = "asinh_with_scale";
  // With log_k = 0.75, scale = exp(0.75) ≈ 2.117
  float log_k = 0.75f;
  float scale = std::exp(log_k);
  float x = 1.0f;
  float result = std::asinhf(x * scale);
  float expected = std::asinhf(2.117f);  // ≈ 1.504
  ASSERT_NEAR(name, result, expected, 1e-2f, "asinh(2.117) ≈ 1.504");
  printf("  PASS: %s\n", name);
}

static void test_asinh_zero() {
  const char* name = "asinh_zero";
  float result = std::asinhf(0.0f);
  ASSERT_NEAR(name, result, 0.0f, 1e-7f, "asinh(0) = 0");
  printf("  PASS: %s\n", name);
}

static void test_asinh_negative() {
  const char* name = "asinh_negative";
  // asinh is odd: asinh(-x) = -asinh(x)
  float x = 3.0f;
  float pos = std::asinhf(x);
  float neg = std::asinhf(-x);
  ASSERT_NEAR(name, neg, -pos, 1e-6f, "asinh should be odd function");
  printf("  PASS: %s\n", name);
}

// ---- Test causal padding logic ----

static void test_causal_pad() {
  const char* name = "causal_pad";
  // Input: [1, 2, 3] (batch=1, channels=2, time=3)
  StorageView input({1, 2, 3}, std::vector<float>{
    1, 2, 3,    // channel 0
    4, 5, 6     // channel 1
  });

  // Pad left by 4 → output should be [1, 2, 7]
  StorageView padding({1, 2, 4}, 0.f, Device::CPU);
  StorageView output(DataType::FLOAT32, Device::CPU);
  const ops::Concat concat_op(2);
  concat_op({&padding, &input}, output);

  ASSERT_TRUE(name, output.dim(0) == 1, "batch=1");
  ASSERT_TRUE(name, output.dim(1) == 2, "channels=2");
  ASSERT_TRUE(name, output.dim(2) == 7, "time=3+4=7");

  const float* d = output.data<float>();
  // Channel 0: [0, 0, 0, 0, 1, 2, 3]
  ASSERT_NEAR(name, d[0], 0.0f, 1e-7f, "pad[0]=0");
  ASSERT_NEAR(name, d[3], 0.0f, 1e-7f, "pad[3]=0");
  ASSERT_NEAR(name, d[4], 1.0f, 1e-7f, "data[0]=1");
  ASSERT_NEAR(name, d[6], 3.0f, 1e-7f, "data[2]=3");
  // Channel 1: [0, 0, 0, 0, 4, 5, 6]
  ASSERT_NEAR(name, d[7], 0.0f, 1e-7f, "ch1 pad[0]=0");
  ASSERT_NEAR(name, d[11], 4.0f, 1e-7f, "ch1 data[0]=4");
  ASSERT_NEAR(name, d[13], 6.0f, 1e-7f, "ch1 data[2]=6");

  printf("  PASS: %s\n", name);
}

// ---- Test framing logic ----

static void test_framing() {
  const char* name = "framing";
  // 240 samples → 3 frames of 80
  const dim_t frame_size = 80;
  const dim_t samples = 240;
  const dim_t num_frames = samples / frame_size;

  ASSERT_TRUE(name, num_frames == 3, "240/80 = 3 frames");

  // Create input [1, 240]
  std::vector<float> data(samples);
  for (int i = 0; i < samples; ++i) data[i] = static_cast<float>(i);
  StorageView audio({1, samples}, data);

  // Reshape to [1, 3, 80]
  audio.reshape({1, num_frames, frame_size});
  ASSERT_TRUE(name, audio.dim(0) == 1, "batch=1");
  ASSERT_TRUE(name, audio.dim(1) == 3, "frames=3");
  ASSERT_TRUE(name, audio.dim(2) == 80, "frame_size=80");

  // Verify frame boundaries
  const float* d = audio.data<float>();
  ASSERT_NEAR(name, d[0], 0.0f, 1e-7f, "frame0[0]=0");
  ASSERT_NEAR(name, d[79], 79.0f, 1e-7f, "frame0[79]=79");
  ASSERT_NEAR(name, d[80], 80.0f, 1e-7f, "frame1[0]=80");
  ASSERT_NEAR(name, d[160], 160.0f, 1e-7f, "frame2[0]=160");

  printf("  PASS: %s\n", name);
}

static void test_framing_truncation() {
  const char* name = "framing_truncation";
  // 250 samples → 3 frames of 80, 10 samples truncated
  const dim_t frame_size = 80;
  const dim_t samples = 250;
  const dim_t num_frames = samples / frame_size;
  ASSERT_TRUE(name, num_frames == 3, "250/80 = 3 frames (truncate 10)");
  printf("  PASS: %s\n", name);
}

// ---- Test output shape calculation ----

static void test_output_shape() {
  const char* name = "output_shape";
  // Conv1d with stride=2 halves the time dimension.
  // Two CausalConv1d with stride=2 → time/4.
  // For 16kHz, 5ms frames: 1 second = 200 frames.
  // After 2x stride-2 conv: 200/4 = 50 frames → 50Hz output.
  const dim_t sample_rate = 16000;
  const dim_t frame_size = 80;  // 5ms
  const dim_t audio_duration_s = 5;
  const dim_t total_samples = sample_rate * audio_duration_s;
  const dim_t num_frames = total_samples / frame_size;

  ASSERT_TRUE(name, num_frames == 1000, "5s at 16kHz/80 = 1000 frames");

  // After causal pad (4) + conv1d(k=5, s=2): output_len = (1000+4 - 5)/2 + 1 = 500
  // After causal pad (4) + conv1d(k=5, s=2): output_len = (500+4 - 5)/2 + 1 = 250
  const dim_t after_conv1 = (num_frames + 4 - 5) / 2 + 1;
  const dim_t after_conv2 = (after_conv1 + 4 - 5) / 2 + 1;
  ASSERT_TRUE(name, after_conv1 == 500, "after conv1: 500");
  ASSERT_TRUE(name, after_conv2 == 250, "after conv2: 250 (50Hz)");

  // 250 frames in 5s = 50Hz ✓
  printf("  PASS: %s\n", name);
}

// ---- Test CMVN with realistic audio-like data ----

static void test_cmvn_realistic() {
  const char* name = "cmvn_realistic";
  // Simulate a frame of 80 samples with DC offset and varying amplitude
  const int fs = 80;
  std::vector<float> data(fs);
  for (int i = 0; i < fs; ++i)
    data[i] = 1000.0f + 50.0f * std::sin(2.0f * M_PI * i / fs);

  apply_cmvn_ref(data.data(), 1, 1, fs);

  // Check mean ≈ 0
  float mean = 0;
  for (float v : data) mean += v;
  mean /= fs;
  ASSERT_NEAR(name, mean, 0.0f, 1e-5f, "mean should be ~0");

  // Check RMS ≈ 1
  float sq_sum = 0;
  for (float v : data) sq_sum += v * v;
  float rms = std::sqrt(sq_sum / fs);
  ASSERT_NEAR(name, rms, 1.0f, 1e-4f, "RMS should be ~1");

  printf("  PASS: %s\n", name);
}

int main() {
  printf("=== Moonshine Audio Frontend Tests (M17.2) ===\n\n");

  // CMVN tests
  test_cmvn_zero_mean();
  test_cmvn_unit_variance();
  test_cmvn_multi_frame();
  test_cmvn_constant_frame();
  test_cmvn_realistic();

  // Asinh tests
  test_asinh_identity();
  test_asinh_with_scale();
  test_asinh_zero();
  test_asinh_negative();

  // Causal padding test
  test_causal_pad();

  // Framing tests
  test_framing();
  test_framing_truncation();

  // Output shape test
  test_output_shape();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
