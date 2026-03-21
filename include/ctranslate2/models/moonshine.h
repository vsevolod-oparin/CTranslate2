#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/moonshine.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"

namespace ctranslate2 {
  namespace models {

    struct MoonshineOptions {
      // Beam size (1 = greedy).
      size_t beam_size = 5;
      float patience = 1;
      float length_penalty = 1;
      float repetition_penalty = 1;
      size_t no_repeat_ngram_size = 0;

      // Maximum generation length.
      size_t max_length = 448;

      // Sampling parameters.
      size_t sampling_topk = 1;
      float sampling_temperature = 1;

      // Result options.
      size_t num_hypotheses = 1;
      bool return_scores = false;
    };

    struct MoonshineResult {
      std::vector<std::vector<std::string>> sequences;
      std::vector<std::vector<size_t>> sequences_ids;
      std::vector<float> scores;

      size_t num_sequences() const {
        return sequences.size();
      }

      bool has_scores() const {
        return !scores.empty();
      }
    };

    class MoonshineModel : public Model {
    public:
      const Vocabulary& get_vocabulary() const;

      size_t current_spec_revision() const override;
      bool is_quantizable(const std::string& variable_name) const override;
      bool is_linear_weight(const std::string& variable_name) const override;
      std::unique_ptr<Model> clone() const override;

      bool use_global_int16_scale() const override {
        return false;
      }

    protected:
      void initialize(ModelReader& model_reader) override;

    private:
      std::shared_ptr<const Vocabulary> _vocabulary;
    };

    class MoonshineReplica : public ModelReplica {
    public:
      static std::unique_ptr<MoonshineReplica> create_from_model(const Model& model);

      MoonshineReplica(const std::shared_ptr<const MoonshineModel>& model);

      // Encode raw audio waveform → encoder output (after adapter).
      StorageView encode(StorageView audio, bool to_cpu = false);

      // Generate text from audio.
      std::vector<MoonshineResult>
      generate(StorageView audio,
               const std::vector<std::vector<size_t>>& prompts,
               const MoonshineOptions& options);

      std::vector<MoonshineResult>
      generate(StorageView audio,
               const std::vector<std::vector<std::string>>& prompts,
               const MoonshineOptions& options);

    private:
      const std::shared_ptr<const MoonshineModel> _model;
      const std::unique_ptr<layers::MoonshineAudioFrontend> _frontend;
      const std::unique_ptr<layers::TransformerEncoder> _encoder;
      const std::unique_ptr<layers::TransformerDecoder> _decoder;

      // Adapter weights (loaded manually, not via a Layer subclass)
      const StorageView* _adapter_pos_emb;    // [max_pos, dec_hidden]
      const StorageView* _adapter_proj_weight; // [enc_hidden, dec_hidden] or nullptr

      size_t _bos_id;
      size_t _eos_id;

      StorageView maybe_encode(StorageView audio);
      StorageView apply_adapter(const StorageView& encoder_output) const;
    };

    class Moonshine : public ReplicaPool<MoonshineReplica> {
    public:
      using ReplicaPool::ReplicaPool;

      std::future<StorageView> encode(const StorageView& audio, bool to_cpu = false);

      std::vector<std::future<MoonshineResult>>
      generate(const StorageView& audio,
               std::vector<std::vector<size_t>> prompts,
               MoonshineOptions options = {});

      std::vector<std::future<MoonshineResult>>
      generate(const StorageView& audio,
               std::vector<std::vector<std::string>> prompts,
               MoonshineOptions options = {});
    };

  }
}
