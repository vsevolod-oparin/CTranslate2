#include "ctranslate2/models/moonshine.h"

#include <algorithm>

#include "ctranslate2/decoding.h"
#include "ctranslate2/ops/ops.h"

#include "dispatch.h"

#ifdef CT2_WITH_MPS
#  include "metal/utils.h"
#endif

namespace ctranslate2 {
  namespace models {

    // ---- MoonshineModel ----

    const Vocabulary& MoonshineModel::get_vocabulary() const {
      return *_vocabulary;
    }

    size_t MoonshineModel::current_spec_revision() const {
      return 1;
    }

    void MoonshineModel::initialize(ModelReader& model_reader) {
      VocabularyInfo vocab_info;
      vocab_info.unk_token = "<unk>";
      vocab_info.bos_token = "<s>";
      vocab_info.eos_token = "</s>";

      _vocabulary = load_vocabulary(model_reader, "vocabulary", std::move(vocab_info));
      if (!_vocabulary)
        throw std::runtime_error("Cannot load the vocabulary from the model directory");
    }

    bool MoonshineModel::is_quantizable(const std::string& variable_name) const {
      return Model::is_quantizable(variable_name);
    }

    bool MoonshineModel::is_linear_weight(const std::string& variable_name) const {
      return is_quantizable(variable_name)
        && variable_name.find("embeddings") == std::string::npos;
    }

    std::unique_ptr<Model> MoonshineModel::clone() const {
      return std::make_unique<MoonshineModel>(*this);
    }

    // ---- MoonshineReplica ----

    std::unique_ptr<MoonshineReplica>
    MoonshineReplica::create_from_model(const Model& model) {
      if (!dynamic_cast<const MoonshineModel*>(&model))
        throw std::invalid_argument("The model is not a Moonshine model");

      const auto scoped_device_setter = model.get_scoped_device_setter();
      const auto model_ptr = model.shared_from_this();
      const auto concrete_model = std::static_pointer_cast<const MoonshineModel>(model_ptr);
      return std::make_unique<MoonshineReplica>(concrete_model);
    }

    MoonshineReplica::MoonshineReplica(const std::shared_ptr<const MoonshineModel>& model)
      : ModelReplica(model)
      , _model(model)
      , _frontend(std::make_unique<layers::MoonshineAudioFrontend>(*model, "encoder/frontend"))
      , _encoder(std::make_unique<layers::TransformerEncoder>(*model, "encoder"))
      , _decoder(std::make_unique<layers::TransformerDecoder>(*model, "decoder"))
      , _adapter_pos_emb(model->get_variable_if_exists("adapter/position_embeddings/weight"))
      , _adapter_proj_weight(model->get_variable_if_exists("adapter/projection/weight"))
    {
      const auto& vocabulary = model->get_vocabulary();
      _bos_id = vocabulary.bos_id();
      _eos_id = vocabulary.eos_id();
    }

    StorageView MoonshineReplica::apply_adapter(const StorageView& encoder_output) const {
      // Adapter: add position embeddings → optional linear projection.
      // encoder_output: [batch, time, enc_hidden]
      const Device device = encoder_output.device();
      const DataType dtype = encoder_output.dtype();
      const dim_t time = encoder_output.dim(1);

      // Copy encoder output — we'll modify in-place by adding pos embeddings.
      StorageView result(encoder_output);

      // Add learned position embeddings.
      if (_adapter_pos_emb) {
        StorageView pos_emb(dtype, device);

        // Slice to [time, hidden] if needed.
        if (_adapter_pos_emb->dim(0) >= time) {
          const ops::Slide slice_op(0, 0, time);
          slice_op(*_adapter_pos_emb, pos_emb);
        } else {
          pos_emb.shallow_copy(const_cast<StorageView&>(*_adapter_pos_emb));
        }

        // Broadcast add: [batch, time, hidden] += [1, time, hidden]
        pos_emb.reshape({1, pos_emb.dim(0), pos_emb.dim(1)});
        DEVICE_AND_TYPE_DISPATCH(device, dtype,
                                 primitives<D>::add_batch_broadcast(
                                     pos_emb.data<T>(),
                                     result.data<T>(),
                                     pos_emb.size(),
                                     result.size()));
      }

      // Linear projection (enc_hidden → dec_hidden) if dimensions differ.
      if (_adapter_proj_weight) {
        StorageView projected(dtype, device);
        const ops::MatMul matmul;
        matmul(result, *_adapter_proj_weight, projected);
        return projected;
      }

      return result;
    }

    StorageView MoonshineReplica::encode(StorageView audio, bool to_cpu) {
      PROFILE("MoonshineReplica::encode");

      const auto scoped_device_setter = _model->get_scoped_device_setter();
      const Device device = _model->device();
      const DataType dtype = _frontend->output_type();

      audio.move_to(device, DataType::FLOAT32);  // Frontend expects float32 input.

      // Step 1: Audio frontend → [batch, time/4, hidden]
      StorageView frontend_output(dtype, device);
      (*_frontend)(audio, frontend_output);

      // Step 2: Transformer encoder → [batch, time/4, hidden]
      // The encoder expects [batch_size, features, time] for embedding input,
      // but our MoonshineEncoder uses the TransformerEncoder which expects
      // vector<StorageView> ids input... Actually, TransformerEncoder calls
      // _embeddings first. Moonshine doesn't use that — the frontend replaces it.
      // We need to feed frontend_output directly through the encoder layers.
      //
      // Since TransformerEncoder expects embedding input (token IDs), we can't use
      // it directly. Instead, feed frontend_output through the encoder layers manually.
      StorageView encoder_output(dtype, device);
      {
        StorageView input = std::move(frontend_output);
        const auto& layers = _encoder->get_layers();
        for (const auto& layer : layers) {
          (*layer)(input, nullptr, encoder_output);
          input = std::move(encoder_output);
        }
        // Final layer norm
        _encoder->get_output_norm()(input, encoder_output);
      }

      // Step 3: Adapter → [batch, time/4, dec_hidden]
      StorageView adapted = apply_adapter(encoder_output);

      if (to_cpu && device != Device::CPU)
        adapted = adapted.to(Device::CPU);

      synchronize_stream(device);
      return adapted;
    }

    StorageView MoonshineReplica::maybe_encode(StorageView audio) {
      const Device device = _model->device();
      const DataType dtype = _frontend->output_type();
      audio.move_to(device, DataType::FLOAT32);

      // Check if already encoded (3D with last dim = decoder output size).
      if (audio.rank() == 3 && audio.dim(2) == _decoder->output_size())
        return audio;

      StorageView frontend_output(dtype, device);
      (*_frontend)(audio, frontend_output);

      StorageView encoder_output(dtype, device);
      {
        StorageView input = std::move(frontend_output);
        const auto& layers = _encoder->get_layers();
        for (const auto& layer : layers) {
          (*layer)(input, nullptr, encoder_output);
          input = std::move(encoder_output);
        }
        _encoder->get_output_norm()(input, encoder_output);
      }

      return apply_adapter(encoder_output);
    }

    std::vector<MoonshineResult>
    MoonshineReplica::generate(StorageView audio,
                               const std::vector<std::vector<std::string>>& prompts,
                               const MoonshineOptions& options) {
      const auto& vocabulary = _model->get_vocabulary();
      return generate(std::move(audio), vocabulary.to_ids(prompts), options);
    }

    std::vector<MoonshineResult>
    MoonshineReplica::generate(StorageView audio,
                               const std::vector<std::vector<size_t>>& prompts,
                               const MoonshineOptions& options) {
      PROFILE("MoonshineReplica::generate");
      if (prompts.empty())
        return {};

      const auto& vocabulary = _model->get_vocabulary();
      const auto scoped_device_setter = _model->get_scoped_device_setter();

      layers::DecoderState state = _decoder->initial_state();
      state.emplace("memory", maybe_encode(std::move(audio)));

#ifdef CT2_WITH_MPS
      if (_model->device() == Device::MPS)
        synchronize_stream(_model->device());
#endif

      _decoder->update_output_layer(_model->preferred_size_multiple());

      // Moonshine uses simple prompts: just [BOS] as start token.
      // No timestamp tokens, no language tokens.
      const std::vector<std::vector<size_t>>& start_tokens = prompts;

      DecodingOptions decoding_options;
      decoding_options.beam_size = options.beam_size;
      decoding_options.patience = options.patience;
      decoding_options.length_penalty = options.length_penalty;
      decoding_options.repetition_penalty = options.repetition_penalty;
      decoding_options.no_repeat_ngram_size = options.no_repeat_ngram_size;
      decoding_options.max_length = options.max_length;
      decoding_options.sampling_topk = options.sampling_topk;
      decoding_options.sampling_temperature = options.sampling_temperature;
      decoding_options.num_hypotheses = options.num_hypotheses;
      decoding_options.return_scores = options.return_scores;

      std::vector<DecodingResult> results = decode(
          *_decoder,
          state,
          start_tokens,
          {_eos_id},
          decoding_options);

      std::vector<MoonshineResult> final_results;
      final_results.reserve(results.size());

      for (auto& result : results) {
        MoonshineResult moonshine_result;
        moonshine_result.sequences_ids = std::move(result.hypotheses);
        moonshine_result.scores = std::move(result.scores);

        // Convert token IDs to strings.
        moonshine_result.sequences.reserve(moonshine_result.sequences_ids.size());
        for (const auto& ids : moonshine_result.sequences_ids) {
          std::vector<std::string> tokens;
          tokens.reserve(ids.size());
          for (const auto id : ids)
            tokens.push_back(vocabulary.to_token(id));
          moonshine_result.sequences.push_back(std::move(tokens));
        }

        final_results.push_back(std::move(moonshine_result));
      }

      return final_results;
    }

    // ---- Moonshine pool ----

    std::future<StorageView>
    Moonshine::encode(const StorageView& audio, bool to_cpu) {
      return post<StorageView>(
        [audio = StorageView(audio), to_cpu](MoonshineReplica& replica) mutable {
          return replica.encode(std::move(audio), to_cpu);
        });
    }

    std::vector<std::future<MoonshineResult>>
    Moonshine::generate(const StorageView& audio,
                        std::vector<std::vector<size_t>> prompts,
                        MoonshineOptions options) {
      const size_t batch_size = audio.dim(0);
      return post_batch<MoonshineResult>(
        [audio = StorageView(audio),
         prompts = std::move(prompts),
         options = std::move(options)]
        (MoonshineReplica& replica) mutable {
          return replica.generate(std::move(audio), prompts, options);
        },
        batch_size);
    }

    std::vector<std::future<MoonshineResult>>
    Moonshine::generate(const StorageView& audio,
                        std::vector<std::vector<std::string>> prompts,
                        MoonshineOptions options) {
      const size_t batch_size = audio.dim(0);
      return post_batch<MoonshineResult>(
        [audio = StorageView(audio),
         prompts = std::move(prompts),
         options = std::move(options)]
        (MoonshineReplica& replica) mutable {
          return replica.generate(std::move(audio), prompts, options);
        },
        batch_size);
    }

  }
}
