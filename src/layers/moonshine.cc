#include "ctranslate2/layers/moonshine.h"

#include <cmath>
#include <stdexcept>

#include "ctranslate2/ops/activation.h"
#include "ctranslate2/ops/concat.h"

namespace ctranslate2 {
  namespace layers {

    MoonshineAudioFrontend::MoonshineAudioFrontend(const models::Model& model,
                                                   const std::string& scope)
      : _log_k(model.get_variable(scope + "/log_k"))
      , _linear(model, scope + "/linear")
      , _conv1(model, scope + "/conv1", /*stride=*/2, /*padding=*/0)
      , _conv2(model, scope + "/conv2", /*stride=*/2, /*padding=*/0)
      , _to_time_major({0, 2, 1})
      , _to_chan_major({0, 2, 1})
    {
    }

    DataType MoonshineAudioFrontend::output_type() const {
      return _conv2.output_type();
    }

    dim_t MoonshineAudioFrontend::output_size() const {
      return _conv2.output_size();
    }

    void MoonshineAudioFrontend::apply_cmvn(StorageView& frames) const {
      // Per-frame CMVN: (x - mean) / sqrt(var + eps)
      // frames shape: [batch, num_frames, frame_size]
      // Operates on CPU — flush GPU first, then work on shared memory.
      const float eps = 1e-5f;

      if (frames.device() != Device::CPU)
        synchronize_stream(frames.device());

      float* data = frames.data<float>();
      const dim_t batch = frames.dim(0);
      const dim_t num_frames = frames.dim(1);
      const dim_t fs = frames.dim(2);

      for (dim_t b = 0; b < batch; ++b) {
        for (dim_t f = 0; f < num_frames; ++f) {
          if (fs == 0) continue;  // empty frame — avoid 0/0 NaN

          float* frame = data + (b * num_frames + f) * fs;

          // Compute mean
          float sum = 0;
          for (dim_t i = 0; i < fs; ++i)
            sum += frame[i];
          const float mean = sum / static_cast<float>(fs);

          // Subtract mean and compute variance
          float var_sum = 0;
          for (dim_t i = 0; i < fs; ++i) {
            frame[i] -= mean;
            var_sum += frame[i] * frame[i];
          }
          const float rms = std::sqrt(var_sum / static_cast<float>(fs) + eps);

          // Normalize
          const float inv_rms = 1.0f / rms;
          for (dim_t i = 0; i < fs; ++i)
            frame[i] *= inv_rms;
        }
      }
    }

    void MoonshineAudioFrontend::apply_asinh(StorageView& x) const {
      // asinh(x * exp(log_k)) where log_k is a learned scalar.
      // Operates on CPU — the data is small (num_frames * frame_size).

      if (x.device() != Device::CPU)
        synchronize_stream(x.device());

      // Read log_k from model weight (may be on GPU, copy to CPU).
      float log_k_val;
      if (_log_k.device() == Device::CPU) {
        log_k_val = _log_k.data<float>()[0];
      } else {
        StorageView log_k_cpu = _log_k.to(Device::CPU);
        log_k_val = log_k_cpu.data<float>()[0];
      }
      const float scale = std::exp(log_k_val);

      float* data = x.data<float>();
      const dim_t size = x.size();
      for (dim_t i = 0; i < size; ++i) {
        data[i] = std::asinhf(data[i] * scale);
      }
    }

    void MoonshineAudioFrontend::causal_pad(const StorageView& input,
                                            dim_t pad_size,
                                            StorageView& output) {
      // Left-pad the time dimension (dim 2) with zeros for causal convolution.
      // input: [batch, channels, time]
      // output: [batch, channels, time + pad_size]
      const dim_t batch = input.dim(0);
      const dim_t channels = input.dim(1);

      StorageView padding({batch, channels, pad_size}, 0.f, input.device());
      if (input.dtype() != DataType::FLOAT32)
        padding = padding.to(input.dtype());

      const ops::Concat concat_op(2);
      concat_op({&padding, &input}, output);
    }

    void MoonshineAudioFrontend::operator()(const StorageView& audio,
                                            StorageView& output) const {
      PROFILE("MoonshineAudioFrontend");

      if (audio.rank() != 2)
        throw std::invalid_argument(
            "MoonshineAudioFrontend: expected input shape [batch, samples], got rank "
            + std::to_string(audio.rank()));

      const dim_t batch = audio.dim(0);
      const dim_t samples = audio.dim(1);
      const dim_t num_frames = samples / _frame_size;

      if (num_frames == 0)
        throw std::invalid_argument(
            "MoonshineAudioFrontend: input too short — need at least "
            + std::to_string(_frame_size) + " samples (5ms at 16kHz), got "
            + std::to_string(samples));

      // Step 1: Frame the raw waveform into [batch, num_frames, frame_size].
      // This is a reshape (no copy) — samples must be a multiple of frame_size.
      // If not exact multiple, truncate trailing samples (matching HF behavior).
      StorageView frames(audio.dtype(), audio.device());
      const dim_t usable_samples = num_frames * _frame_size;
      if (usable_samples < samples) {
        // Truncate trailing samples that don't fill a full frame.
        StorageView truncated({batch, usable_samples}, audio.dtype(), audio.device());
        const ops::Slide slice_op(1, 0, usable_samples);
        slice_op(audio, truncated);
        frames = std::move(truncated);
      } else {
        frames.shallow_copy(const_cast<StorageView&>(audio));
      }
      frames.reshape({batch, num_frames, _frame_size});

      // Step 2: Per-frame CMVN normalization.
      // Need a mutable copy since we modify in-place.
      StorageView normalized(frames);
      apply_cmvn(normalized);

      // Step 3: Asinh compression with learned scale.
      apply_asinh(normalized);

      // Step 4: Linear projection [batch, num_frames, 80] → [batch, num_frames, hidden_size]
      // + Step 5: SiLU activation (applied via Dense activation parameter — but Dense
      //   doesn't take activation in CT2's Layer wrapper. Apply separately.)
      StorageView projected(output_type(), audio.device());
      _linear(normalized, projected);

      // Apply SiLU (Swish) activation.
      const auto& swish_op = ops::get_activation_op(ops::ActivationType::Swish);
      swish_op(projected, projected);

      // Step 6: Transpose to channel-major for Conv1d: [batch, time, hidden] → [batch, hidden, time]
      StorageView chan_major(projected.dtype(), projected.device());
      _to_chan_major(projected, chan_major);

      // Step 7: CausalConv1d #1 (hidden → hidden*2, kernel=5, stride=2) + SiLU
      // Left-pad by (kernel_size - 1) = 4
      StorageView padded(chan_major.dtype(), chan_major.device());
      causal_pad(chan_major, 4, padded);
      StorageView conv1_out(chan_major.dtype(), chan_major.device());
      _conv1(padded, conv1_out);

      // SiLU activation between conv layers (matches HF forward pass)
      swish_op(conv1_out, conv1_out);

      // Step 8: CausalConv1d #2 (hidden*2 → hidden, kernel=5, stride=2) — no activation after
      // Left-pad by (kernel_size - 1) = 4
      causal_pad(conv1_out, 4, padded);
      StorageView conv2_out(chan_major.dtype(), chan_major.device());
      _conv2(padded, conv2_out);

      // Step 9: Transpose back to time-major: [batch, hidden, time/4] → [batch, time/4, hidden]
      _to_time_major(conv2_out, output);
    }

  }
}
