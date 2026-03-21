#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    // Moonshine Streaming audio frontend.
    // Converts raw 16kHz waveform to encoder input features.
    //
    // Pipeline:
    //   raw audio [batch, samples]
    //     → frame into [batch, num_frames, 80]  (5ms frames at 16kHz)
    //     → per-frame CMVN (mean subtraction + RMS normalization)
    //     → asinh compression with learned log_k parameter
    //     → Linear projection [80 → hidden_size]
    //     → SiLU activation
    //     → CausalConv1d(hidden → hidden*2, k=5, s=2)
    //     → CausalConv1d(hidden*2 → hidden, k=5, s=2)
    //     → output [batch, num_frames/4, hidden_size]
    class MoonshineAudioFrontend : public Layer {
    public:
      MoonshineAudioFrontend(const models::Model& model, const std::string& scope);

      // Input:  raw waveform [batch, samples] (float32, 16kHz)
      // Output: encoder features [batch, time, hidden_size]
      void operator()(const StorageView& audio, StorageView& output) const;

      DataType output_type() const override;
      dim_t output_size() const override;

    private:
      static constexpr dim_t _frame_size = 80;  // 5ms at 16kHz

      const StorageView& _log_k;        // learned asinh compression parameter (scalar)
      const Dense _linear;               // Linear(frame_size → hidden_size)
      const Conv1D _conv1;               // CausalConv1d(hidden → hidden*2, k=5, s=2)
      const Conv1D _conv2;               // CausalConv1d(hidden*2 → hidden, k=5, s=2)
      const ops::Transpose _to_time_major;   // [batch, channels, time] → [batch, time, channels]
      const ops::Transpose _to_chan_major;    // [batch, time, channels] → [batch, channels, time]

      // Apply per-frame CMVN: (x - mean) / sqrt(var + eps)
      // Operates on [batch, num_frames, frame_size] in-place.
      void apply_cmvn(StorageView& frames) const;

      // Apply asinh compression: asinh(x * exp(log_k))
      // Operates element-wise in-place.
      void apply_asinh(StorageView& x) const;

      // Left-pad input on time dimension (dim 2) for causal convolution.
      // Input: [batch, channels, time]
      // Output: [batch, channels, time + pad_size]
      static void causal_pad(const StorageView& input, dim_t pad_size, StorageView& output);
    };

  }
}
