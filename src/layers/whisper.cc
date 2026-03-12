#include "ctranslate2/layers/whisper.h"

#include "ctranslate2/devices.h"
#include "ctranslate2/ops/concat.h"

namespace ctranslate2 {
  namespace layers {

    static const ops::ActivationType activation_type = ops::ActivationType::GELU;

    WhisperEncoder::WhisperEncoder(const models::Model& model, const std::string& scope)
      : _conv1(model, scope + "/conv1", /*stride=*/1, /*padding=*/1,
               /*dilation=*/1, /*groups=*/1, &activation_type)
      , _conv2(model, scope + "/conv2", /*stride=*/2, /*padding=*/1,
               /*dilation=*/1, /*groups=*/1, &activation_type)
      , _transpose({0, 2, 1})
      , _position_embedding(model, scope + "/position_encodings")
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _layers(build_layers_list<const TransformerEncoderLayer>(model,
                                                                 scope + "/layer",
                                                                 _num_heads,
                                                                 /*pre_norm=*/true,
                                                                 ops::ActivationType::GELU))
      , _output_norm(model, scope + "/layer_norm")
    {
    }

    void WhisperEncoder::operator()(const StorageView& features, StorageView& output) {
      PROFILE("WhisperEncoder");

      if (features.rank() != 3)
        throw std::invalid_argument("Expected input features to have 3 dimensions, but got "
                                    + std::to_string(features.rank())
                                    + " dimension(s) instead");

      if (features.dim(1) != input_size() || features.dim(2) > max_input_time())
        throw std::invalid_argument("Invalid input features shape: expected an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(input_size())
                                    + ", "
                                    + std::to_string(std::min(features.dim(2), max_input_time()))
                                    + "), but got an input with shape ("
                                    + std::to_string(features.dim(0))
                                    + ", "
                                    + std::to_string(features.dim(1))
                                    + ", "
                                    + std::to_string(features.dim(2))
                                    + ") instead");

      StorageView input(output_type(), features.device());

      _conv1(features, input);
      _conv2(input, output);
      _transpose(output, input);
      _position_embedding(input);

      for (const auto& layer : _layers) {
        (*layer)(input, nullptr, output);
        input = std::move(output);
      }

      _output_norm(input, output);
    }


    void WhisperDecoder::forward_prompt(const StorageView& prompt,
                                        DecoderState& state,
                                        StorageView* outputs) {
#ifdef CT2_WITH_MPS
      // M12.8: On MPS with deep decoders (32 layers), processing multiple prompt
      // tokens at once causes numerical divergence in the KV cache. Process tokens
      // one at a time (iterative decoding) to match the single-token path that
      // detect_language uses successfully.
      if (prompt.device() == Device::MPS && prompt.dim(1) > 1) {
        const dim_t batch_size = prompt.dim(0);
        const dim_t prompt_length = prompt.dim(1);

        std::vector<StorageView> all_outputs;

        for (dim_t t = 0; t < prompt_length; ++t) {
          // Use rank-1 ids like detect_language (proven working path)
          StorageView token_ids({batch_size}, DataType::INT32, prompt.device());
          for (dim_t b = 0; b < batch_size; ++b)
            token_ids.at<int32_t>(b) = prompt.at<int32_t>({b, t});

          if (outputs) {
            StorageView step_output(output_type(), prompt.device());
            decode(token_ids,
                   /*lengths=*/nullptr,
                   /*step=*/t,
                   state,
                   &step_output,
                   /*attention=*/nullptr,
                   /*return_logits=*/false);
            // decode with rank-1 ids produces {batch, d_model}; unsqueeze for concat
            step_output.expand_dims(1);
            all_outputs.push_back(std::move(step_output));
          } else {
            decode(token_ids,
                   /*lengths=*/nullptr,
                   /*step=*/t,
                   state,
                   /*outputs=*/nullptr,
                   /*attention=*/nullptr,
                   /*return_logits=*/false);
          }

          // Flush GPU work between prompt steps to ensure KV cache consistency
          synchronize_stream(prompt.device());
        }

        // Reconstruct {batch, prompt_length, d_model} output for compute_logits_for_steps
        if (outputs && !all_outputs.empty()) {
          std::vector<const StorageView*> ptrs;
          ptrs.reserve(all_outputs.size());
          for (const auto& o : all_outputs)
            ptrs.push_back(&o);
          ops::Concat(1)(ptrs, *outputs);
        }
        return;
      }
#endif
      decode(prompt,
             /*lengths=*/nullptr,
             /*step=*/0,
             state,
             outputs,
             /*attention=*/nullptr,
             /*return_logits=*/false);
    }

    void WhisperDecoder::compute_logits_for_steps(const StorageView& outputs,
                                                  const StorageView& steps,
                                                  StorageView& logits) {
      StorageView step_outputs(outputs.dtype(), outputs.device());
      ops::Gather(1, 1)(outputs, steps, step_outputs);
      _proj(step_outputs, logits);
    }

  }
}
