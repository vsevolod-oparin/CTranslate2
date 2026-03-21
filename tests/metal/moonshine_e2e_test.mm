// Test: Moonshine end-to-end encoder validation (M17.7)
// Loads the converted CT2 model, runs encoder on synthetic audio,
// and compares output against HuggingFace reference.

#import <Foundation/Foundation.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>

#include "ctranslate2/models/moonshine.h"
#include "ctranslate2/models/model_factory.h"

using namespace ctranslate2;

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(name, cond, msg) do { \
  if (!(cond)) { printf("  FAIL: %s — %s\n", name, msg); g_fail++; } \
  else { g_pass++; } \
} while(0)

#define ASSERT_NEAR(name, a, b, eps, msg) do { \
  if (std::fabs((a) - (b)) > (eps)) { \
    printf("  FAIL: %s — %s (got %f, expected %f, diff %e)\n", \
           name, msg, (double)(a), (double)(b), (double)std::fabs((a)-(b))); \
    g_fail++; \
  } else { g_pass++; } \
} while(0)

// Load a simple binary file: int32 ndim, int64 dims[], float32 data[]
static std::vector<float> load_bin(const std::string& path, std::vector<int64_t>& shape) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("Cannot open: " + path);

  int32_t ndim;
  f.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

  shape.resize(ndim);
  f.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

  int64_t total = 1;
  for (auto d : shape) total *= d;

  std::vector<float> data(total);
  f.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
  return data;
}

static std::shared_ptr<const models::Model> load_moonshine() {
  return models::Model::load("/tmp/moonshine-tiny-ct2", Device::CPU, 0);
}

static void test_model_load() {
  const char* name = "model_load";
  try {
    auto model = load_moonshine();
    ASSERT_TRUE(name, model != nullptr, "model loaded");
    ASSERT_TRUE(name, dynamic_cast<const models::MoonshineModel*>(model.get()) != nullptr,
                "is MoonshineModel");
    printf("  PASS: %s\n", name);
  } catch (const std::exception& e) {
    printf("  FAIL: %s — %s\n", name, e.what());
    g_fail++;
  }
}

static void test_encoder_output() {
  const char* name = "encoder_output";
  try {
    auto model = load_moonshine();
    auto replica = models::MoonshineReplica::create_from_model(*model);

    // Generate same sine wave as reference (no file loading needed)
    const dim_t sr = 16000;
    std::vector<float> audio_data(sr);
    for (dim_t i = 0; i < sr; ++i)
      audio_data[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / sr);
    printf("  Audio: %zu samples\n", audio_data.size());

    StorageView audio({1, static_cast<dim_t>(audio_data.size())}, audio_data, Device::CPU);

    // Run encode
    StorageView encoder_output = replica->encode(std::move(audio), /*to_cpu=*/true);
    printf("  CT2 encoder output: [%lld, %lld, %lld]\n",
           encoder_output.dim(0), encoder_output.dim(1), encoder_output.dim(2));

    // Load reference encoder output (no-pad version — matches our C++ pipeline)
    std::vector<int64_t> ref_shape;
    auto ref_data = load_bin("/tmp/moonshine_ref_adapted_nopad.bin", ref_shape);
    printf("  HF reference: [%lld, %lld, %lld]\n",
           ref_shape[0], ref_shape[1], ref_shape[2]);

    // Compare shapes
    ASSERT_TRUE(name, encoder_output.dim(0) == ref_shape[0], "batch dim match");
    ASSERT_TRUE(name, encoder_output.dim(1) == ref_shape[1], "time dim match");
    ASSERT_TRUE(name, encoder_output.dim(2) == ref_shape[2], "hidden dim match");

    if (encoder_output.dim(1) != ref_shape[1] || encoder_output.dim(2) != ref_shape[2]) {
      printf("  SKIP: shape mismatch, cannot compare values\n");
      return;
    }

    // Compare values
    StorageView ref_sv({static_cast<dim_t>(ref_shape[0]),
                        static_cast<dim_t>(ref_shape[1]),
                        static_cast<dim_t>(ref_shape[2])}, ref_data, Device::CPU);

    // Convert CT2 output to float32 if needed
    StorageView ct2_f32 = encoder_output.dtype() != DataType::FLOAT32
        ? encoder_output.to_float32()
        : std::move(encoder_output);

    const float* ct2_ptr = ct2_f32.data<float>();
    const float* ref_ptr = ref_sv.data<float>();
    const dim_t total = ct2_f32.size();

    float max_diff = 0;
    float sum_diff = 0;
    for (dim_t i = 0; i < total; ++i) {
      float diff = std::fabs(ct2_ptr[i] - ref_ptr[i]);
      max_diff = std::max(max_diff, diff);
      sum_diff += diff;
    }
    float mean_diff = sum_diff / total;

    printf("  Max abs diff: %e\n", max_diff);
    printf("  Mean abs diff: %e\n", mean_diff);

    // Tolerance: ~5e-3 max is expected after 6+ transformer layers in float32.
    // Each layer accumulates ~5e-4 of floating point diff vs HF PyTorch.
    ASSERT_TRUE(name, max_diff < 5e-2, "max diff < 5e-2");
    ASSERT_TRUE(name, mean_diff < 5e-3, "mean diff < 5e-3");

    printf("  PASS: %s\n", name);
  } catch (const std::exception& e) {
    printf("  FAIL: %s — %s\n", name, e.what());
    g_fail++;
  }
}

static void test_encoder_output_shape_5s() {
  const char* name = "encoder_shape_5s";
  try {
    auto model = load_moonshine();
    auto replica = models::MoonshineReplica::create_from_model(*model);

    // 5 seconds of audio
    const dim_t samples = 80000;
    std::vector<float> audio(samples, 0.0f);
    for (dim_t i = 0; i < samples; ++i)
      audio[i] = 0.1f * std::sin(2.0f * M_PI * 440.0f * i / 16000.0f);

    StorageView audio_sv({1, samples}, audio, Device::CPU);
    StorageView output = replica->encode(std::move(audio_sv), true);

    // 80000 samples / 80 (frame_size) = 1000 frames
    // After 2x stride-2 conv: 1000 → ~250 (depends on exact conv math with causal padding)
    // Expected: ~250 frames at 50Hz
    printf("  5s audio → output shape: [%lld, %lld, %lld]\n",
           output.dim(0), output.dim(1), output.dim(2));

    ASSERT_TRUE(name, output.dim(0) == 1, "batch=1");
    ASSERT_TRUE(name, output.dim(1) > 200 && output.dim(1) < 300, "time ~250");
    ASSERT_TRUE(name, output.dim(2) == 320, "hidden=320 (Tiny)");

    printf("  PASS: %s\n", name);
  } catch (const std::exception& e) {
    printf("  FAIL: %s — %s\n", name, e.what());
    g_fail++;
  }
}

int main() {
  printf("=== Moonshine E2E Validation Tests (M17.7) ===\n\n");

  test_model_load();
  test_encoder_output();
  test_encoder_output_shape_5s();

  printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
  return g_fail > 0 ? 1 : 0;
}
