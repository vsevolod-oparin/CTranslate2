#include "module.h"

#include <ctranslate2/models/moonshine.h>

#include "replica_pool.h"

namespace ctranslate2 {
  namespace python {

    class MoonshineWrapper : public ReplicaPoolHelper<models::Moonshine> {
    public:
      using ReplicaPoolHelper::ReplicaPoolHelper;

      StorageView encode(const StorageView& audio, const bool to_cpu) {
        std::shared_lock lock(_mutex);
        assert_model_is_ready();
        return _pool->encode(audio, to_cpu).get();
      }

      std::vector<models::MoonshineResult>
      generate(const StorageView& audio,
               std::variant<BatchTokens, BatchIds> prompts,
               size_t beam_size,
               float patience,
               size_t num_hypotheses,
               float length_penalty,
               float repetition_penalty,
               size_t no_repeat_ngram_size,
               size_t max_length,
               bool return_scores,
               size_t sampling_topk,
               float sampling_temperature) {
        std::vector<std::future<models::MoonshineResult>> futures;

        models::MoonshineOptions options;
        options.beam_size = beam_size;
        options.patience = patience;
        options.length_penalty = length_penalty;
        options.repetition_penalty = repetition_penalty;
        options.no_repeat_ngram_size = no_repeat_ngram_size;
        options.sampling_topk = sampling_topk;
        options.sampling_temperature = sampling_temperature;
        options.max_length = max_length;
        options.num_hypotheses = num_hypotheses;
        options.return_scores = return_scores;

        std::shared_lock lock(_mutex);
        assert_model_is_ready();

        if (prompts.index() == 0)
          futures = _pool->generate(audio, std::get<BatchTokens>(prompts), options);
        else
          futures = _pool->generate(audio, std::get<BatchIds>(prompts), options);

        return wait_on_futures(std::move(futures));
      }
    };


    void register_moonshine(py::module& m) {
      py::class_<models::MoonshineResult>(m, "MoonshineResult",
                                          "A generation result from the Moonshine model.")

        .def_readonly("sequences", &models::MoonshineResult::sequences,
                      "Generated sequences of tokens.")
        .def_readonly("sequences_ids", &models::MoonshineResult::sequences_ids,
                      "Generated sequences of token IDs.")
        .def_readonly("scores", &models::MoonshineResult::scores,
                      "Score of each sequence (empty if :obj:`return_scores` was disabled).")

        .def("__repr__", [](const models::MoonshineResult& result) {
          return "MoonshineResult(sequences=" + std::string(py::repr(py::cast(result.sequences)))
            + ", sequences_ids=" + std::string(py::repr(py::cast(result.sequences_ids)))
            + ", scores=" + std::string(py::repr(py::cast(result.scores)))
            + ")";
        })
        ;

      py::class_<MoonshineWrapper>(
        m, "Moonshine",
        R"pbdoc(
            Implements the Moonshine Streaming speech recognition model.

            See Also:
               https://github.com/moonshine-ai/moonshine
        )pbdoc")

        .def(py::init<const std::string&, const std::string&, const std::variant<int, std::vector<int>>&, const StringOrMap&, size_t, size_t, long, bool, bool, py::object>(),
             py::arg("model_path"),
             py::arg("device")="cpu",
             py::kw_only(),
             py::arg("device_index")=0,
             py::arg("compute_type")="default",
             py::arg("inter_threads")=1,
             py::arg("intra_threads")=0,
             py::arg("max_queued_batches")=0,
             py::arg("flash_attention")=false,
             py::arg("tensor_parallel")=false,
             py::arg("files")=py::none(),
             R"pbdoc(
                 Initializes a Moonshine model from a converted model.

                 Arguments:
                   model_path: Path to the CTranslate2 model directory.
                   device: Device to use (possible values are: cpu, cuda, auto).
                   device_index: Device IDs where to place this model on.
                   compute_type: Model computation type or a dictionary mapping a device name
                     to the computation type.
                   inter_threads: Number of workers.
                   intra_threads: Number of OpenMP threads per worker (0 for default).
                   max_queued_batches: Maximum queued batches (-1 unlimited, 0 auto).
                   flash_attention: Use flash attention for self-attention.
                   tensor_parallel: Use tensor parallel mode.
                   files: Load model files from memory (dict of filename -> content).
             )pbdoc")

        .def_property_readonly("device", &MoonshineWrapper::device,
                               "Device this model is running on.")
        .def_property_readonly("device_index", &MoonshineWrapper::device_index,
                               "List of device IDs where this model is running on.")
        .def_property_readonly("compute_type", &MoonshineWrapper::compute_type,
                               "Computation type used by the model.")
        .def_property_readonly("num_workers", &MoonshineWrapper::num_replicas,
                               "Number of model workers.")
        .def_property_readonly("num_queued_batches", &MoonshineWrapper::num_queued_batches,
                               "Number of batches waiting to be processed.")
        .def_property_readonly("model_is_loaded", &MoonshineWrapper::model_is_loaded,
                               "Whether the model is loaded and ready.")

        .def("encode", &MoonshineWrapper::encode,
             py::arg("audio"),
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the input audio waveform.

                 Arguments:
                   audio: Raw audio waveform as a float array with shape
                     ``[batch_size, num_samples]`` (16kHz, float32).
                   to_cpu: Copy the encoder output to the CPU before returning.

                 Returns:
                   The encoder output.
             )pbdoc")

        .def("generate", &MoonshineWrapper::generate,
             py::arg("audio"),
             py::arg("prompts"),
             py::kw_only(),
             py::arg("beam_size")=5,
             py::arg("patience")=1,
             py::arg("num_hypotheses")=1,
             py::arg("length_penalty")=1,
             py::arg("repetition_penalty")=1,
             py::arg("no_repeat_ngram_size")=0,
             py::arg("max_length")=448,
             py::arg("return_scores")=false,
             py::arg("sampling_topk")=1,
             py::arg("sampling_temperature")=1,
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
                 Encodes the audio and generates text tokens.

                 Arguments:
                   audio: Raw audio waveform as a float array with shape
                     ``[batch_size, num_samples]`` (16kHz, float32).
                   prompts: Batch of initial token IDs (typically just ``[[bos_id]]``).
                   beam_size: Beam size (1 for greedy search).
                   patience: Beam search patience factor.
                   num_hypotheses: Number of hypotheses to return.
                   length_penalty: Exponential penalty applied to length during beam search.
                   repetition_penalty: Penalty for previously generated tokens (set > 1).
                   no_repeat_ngram_size: Prevent repetitions of ngrams with this size (0 to disable).
                   max_length: Maximum generation length.
                   return_scores: Include scores in the output.
                   sampling_topk: Randomly sample from top K candidates.
                   sampling_temperature: Sampling temperature.

                 Returns:
                   A list of MoonshineResult objects.
             )pbdoc")

        .def("unload_model", &MoonshineWrapper::unload_model,
             py::arg("to_cpu")=false,
             py::call_guard<py::gil_scoped_release>(),
             "Unloads the model from the device.")

        .def("load_model", &MoonshineWrapper::load_model,
             py::arg("keep_cache")=false,
             py::call_guard<py::gil_scoped_release>(),
             "Loads the model back to the initial device.")
        ;
    }

  }
}
