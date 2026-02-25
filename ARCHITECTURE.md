# CTranslate2 Architecture Diagram

This document provides a comprehensive visual guide to navigate the CTranslate2 codebase.

## Legend

```
[Box]            = Class/Module
(Box: Subtype)  = Specialization
<Function>       = Method/Function
→                = Calls/Uses
⬇                = Inherits/Implements
→*               = Multiple calls
📦               = Directory
```

---

## 1. High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Python API  │    │   CLI Tool   │    │  C++ API  │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│         ↓                     ↓                    ↓             │
│  ┌──────────────────────────────────────────────────┐           │
│  │      Python Bindings (pybind11)          │           │
│  └──────────────────────────────────────────────────┘           │
│         ↓                                                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │           CORE INFERENCE ENGINE            │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Entry Points and Binding Layer

### Python API Entry Point
```
📦 python/ctranslate2/__init__.py
│
├─ Imports from ctranslate2._ext (compiled extension)
│  └─ pybind11 module in python/cpp/module.cc
│
└─ Exposes:
   ├─ Translator        → python/cpp/translator.cc
   ├─ Generator         → python/cpp/generator.cc
   ├─ Encoder          → python/cpp/encoder.cc
   ├─ StorageView      → python/cpp/storage_view.cc
   └─ Utility functions (get_cuda_device_count, etc.)
```

### CLI Entry Point
```
📦 cli/translator.cc
│
├─ Uses cxxopts (📦 third_party/cxxopts/)
│  ├─ Parse command line arguments
│  ├─ Configure: device, compute_type, batch_size
│  └─ Set translation options (beam_size, etc.)
│
└─ Calls:
   └─ ctranslate2::Translator::translate_text_file()
```

### Python Binding Module
```
📦 python/cpp/module.cc
│
└─ PYBIND11_MODULE(_ext, m)
   ├─ register_storage_view(m)   → python/cpp/storage_view.cc
   ├─ register_translator(m)     → python/cpp/translator.cc
   ├─ register_generator(m)      → python/cpp/generator.cc
   ├─ register_encoder(m)        → python/cpp/encoder.cc
   ├─ register_whisper(m)        → python/cpp/whisper.cc
   ├─ register_wav2vec2(m)       → python/cpp/wav2vec2.cc
   ├─ register_wav2vec2bert(m)   → python/cpp/wav2vec2bert.cc
   ├─ register_extensions(m)     → python/cpp/extensions.cc
   ├─ register_mpi(m)            → python/cpp/mpi.cc
   └─ Register utility functions (execution stats, logging, etc.)
```

---

## 3. Core Abstraction Hierarchy

Levels increase from most fundamental (1) to most abstract (6).
Each level uses (→ calls) the level below it.

```
                    ┌───────────────────────────┐
                    │  Replica Pool  (Level 6)  │  Thread pool of replicas
                    │  Manages parallel execution│  (ReplicaPool<T>)
                    └───────────────────────────┘
                                 ↓ uses
                    ┌───────────────────────────┐
                    │  Replicas      (Level 5)  │  Runnable model instances
                    │                           │  (SequenceToSequenceReplica)
                    └───────────────────────────┘
                                 ↓ uses
                    ┌───────────────────────────┐
                    │  Models        (Level 4)  │  Complete architectures
                    │                           │  (Transformer, Whisper)
                    └───────────────────────────┘
                                 ↓ uses
                    ┌───────────────────────────┐
                    │  Layers        (Level 3)  │  Neural network components
                    │                           │  (Decoder, Attention, FFN)
                    └───────────────────────────┘
                                 ↓ uses
                    ┌───────────────────────────┐
                    │  Ops           (Level 2)  │  Atomic operations
                    │                           │  (Gemm, Softmax, LayerNorm)
                    └───────────────────────────┘
                                 ↓ uses
                    ┌───────────────────────────┐
                    │  Primitives    (Level 1)  │  Vector/matrix kernels
                    │                           │  (cpu/primitives.cc,
                    │                           │   cuda/primitives.cu)
                    └───────────────────────────┘
```

---

## 4. Detailed Component Breakdown

### 4.1 StorageView (Central Tensor Abstraction)

```
┌──────────────────────────────────────────────────────────────┐
│                StorageView Class                      │
│  📦 include/ctranslate2/storage_view.h              │
│  📦 src/storage_view.cc                              │
└──────────────────────────────────────────────────────────────┘
│
├─ Core Data:
│  ├─ Shape _shape              → std::vector<dim_t>
│  ├─ DataType _dtype           → FLOAT32, INT8, etc.
│  ├─ Device _device            → CPU or CUDA
│  ├─ void* _buffer           → Allocated memory (64-byte aligned)
│  └─ Allocator* _allocator    → Device-specific allocator
│
├─ Key Methods:
│  ├─ resize(Shape)           → Reuses buffer if smaller
│  ├─ reshape(Shape)          → Zero-copy view change
│  ├─ to(Device)              → Copy to device
│  ├─ to(DataType)            → Type conversion
│  └─ view(void*, Shape)       → View external buffer (zero-copy)
│
└─ Python Bindings (📦 python/cpp/storage_view.cc):
   ├─ from_array(np.array)     → NumPy array interface
   ├─ from_array(torch.tensor)  → PyTorch array interface
   ├─ to(dtype)              → Type conversion
   └─ Properties: dtype, shape, device, device_index
```

### StorageView States

```
┌─────────────────────────────────────────────────────┐
│  StorageView Lifecycle                            │
└─────────────────────────────────────────────────────┘

Created → Resized → Used → Released
    ↓          ↓        ↓        ↓
  Empty    Owned    View    Freed
 (buffer)  (alloc) (ref)   (cache)
```

**Memory Ownership:**
- **Owned**: StorageView allocates buffer (uses allocator)
- **View**: StorageView references external buffer (zero-copy)
- **Copy**: StorageView owns copied data

**Optimization:**
- `resize()` reuses existing buffer if larger
- `view()` creates reference without allocation
- `to(Device)` uses allocator for cross-device copies

### 4.1b ComputeType (Quantization Policy)

```
┌──────────────────────────────────────────────────────────────┐
│                ComputeType Enum                       │
│  📦 include/ctranslate2/types.h                      │
└──────────────────────────────────────────────────────────────┘
│
├─ Values:
│  ├─ DEFAULT      → Use model's stored weight type
│  ├─ AUTO         → Select best type for device at runtime
│  ├─ FLOAT32      → Full precision
│  ├─ FLOAT16      → Half precision (GPU)
│  ├─ BFLOAT16     → Brain float (GPU, modern CPUs)
│  ├─ INT8         → INT8 weights + FLOAT32 accumulation
│  ├─ INT8_FLOAT16 → INT8 weights + FLOAT16 accumulation
│  ├─ INT8_BFLOAT16→ INT8 weights + BFLOAT16 accumulation
│  └─ INT16        → INT16 weights (CPU only)
│
├─ Resolution:
│  └─ resolve_compute_type(requested, model_type, device)
│     └─ Falls back gracefully if device lacks support
│
└─ Key helpers:
   ├─ mayiuse_float16(device)   → runtime capability check
   ├─ mayiuse_bfloat16(device)  → runtime capability check
   └─ mayiuse_int8(device)      → runtime capability check
```

### 4.2 Operations (Ops)

```
┌──────────────────────────────────────────────────────────────┐
│                   Ops Layer                         │
│  📦 include/ctranslate2/ops/*.h                     │
│  📦 src/ops/*.cc (dispatcher)                     │
│  📦 src/ops/*_cpu.cc (CPU impl)                  │
│  📦 src/ops/*_gpu.cu (CUDA impl)                   │
└──────────────────────────────────────────────────────────────┘
│
├─ Base Classes (📦 include/ctranslate2/ops/op.h):
│  ├─ Op                          → Base class
│  ├─ UnaryOp                     → operator()(input, output)
│  ├─ BinaryOp                    → operator()(input1, input2, output)
│  └─ TernaryOp                   → operator()(input1, input2, input3, output)
│
├─ Operation Examples:
│  ├─ LayerNorm (📦 src/ops/layer_norm.cc)
│  │  ├─ Checks input dimensions
│  │  ├─ DEVICE_AND_FLOAT_DISPATCH() → Select kernel
│  │  ├─ Calls: compute<D,T>() (CPU or GPU)
│  │  └─ Uses: primitives::add(), primitives::mul()
│  │
│  ├─ RMSNorm (📦 src/ops/rms_norm.cc)
│  │  └─ Used by modern LLMs: LLaMA, Qwen, Gemma, etc.
│  │
│  ├─ Gemm (📦 src/ops/gemm.cc)
│  │  ├─ Dispatches by device and data type
│  │  └─ Backend-specific: MKL, cuBLAS, etc.
│  │
│  ├─ Flash Attention (📦 src/ops/flash_attention.cc)
│  │  ├─ CUDA: uses pre-compiled SM80 kernels
│  │  │  └─ 📦 src/ops/flash-attention/ (fp16/bf16, hdim 32–256)
│  │  └─ CPU: fallback implementation
│  │
│  ├─ Rotary (📦 src/ops/rotary.cc)
│  │  └─ Rotary Position Embeddings (RoPE)
│  │
│  ├─ ALiBiAdd (📦 src/ops/alibi_add.cc)
│  │  └─ ALiBi positional bias added to attention scores
│  │
│  ├─ AWQ ops (📦 src/ops/awq/)
│  │  ├─ dequantize.cc  → AWQ weight dequantization
│  │  ├─ gemm.cc        → AWQ fused GEMM
│  │  ├─ gemv.cc        → AWQ fused GEMV (single-token decode)
│  │  └─ NOTE: disabled when compiled with ROCm/HIP (#ifndef CT2_USE_HIP)
│  │
│  ├─ Conv1d (📦 src/ops/conv1d.cc)
│  │  └─ Used by audio models (Wav2Vec2, Whisper encoder)
│  │
│  ├─ Quantize / Dequantize (📦 src/ops/quantize.cc / dequantize.cc)
│  │  └─ Dynamic INT8 quantization during inference
│  │
│  ├─ Activation functions:
│  │  ├─ GELU (📦 src/ops/gelu.cc)
│  │  ├─ SiLU/Swish (📦 src/ops/swish.cc)
│  │  ├─ ReLU, Sigmoid, Tanh, Log, Cos, Sin
│  │  └─ Dispatched as UnaryOp specializations
│  │
│  ├─ Tensor ops:
│  │  ├─ Gather, Concat, Split, Transpose, Tile
│  │  ├─ Squeeze, Unsqueeze          → rank manipulation
│  │  ├─ TopK, TopPMask              → sampling/beam search
│  │  ├─ GumbelMax, Multinomial      → stochastic sampling
│  │  ├─ BiasAdd, Add, Sub, Mul, Sum, Mean, MinMax
│  │  └─ Slide                       → sliding window on sequences
│  │
│  ├─ MedianFilter (📦 src/ops/median_filter_*.cc)
│  │  └─ Sliding-window median; used for Whisper timestamp smoothing
│  │
│  ├─ NCCL collective ops (📦 include/ctranslate2/ops/nccl_ops.h)
│  │  ├─ ReduceAll  → all-reduce (SUM/PROD/MIN/MAX/AVG across ranks)
│  │  └─ GatherAll  → all-gather across ranks
│  │     (both compiled only with CT2_WITH_TENSOR_PARALLEL)
│  │
│  └─ Softmax (📦 src/ops/softmax.cc)
│
└─ Dispatch Pattern:
   DEVICE_AND_TYPE_DISPATCH(device, dtype,
     (kernel_function<D, T>(...)));
```

### 4.3 Layers

```
┌──────────────────────────────────────────────────────────────┐
│                  Layers Layer                        │
│  📦 include/ctranslate2/layers/*.h                   │
│  📦 src/layers/*.cc                                  │
└──────────────────────────────────────────────────────────────┘
│
├─ Key Layer Types:
│  │
│  ├─ Decoder (📦 include/ctranslate2/layers/decoder.h)
│  │  ├─ Manages: DecoderState (unordered_map of StorageViews)
│  │  ├─ Methods:
│  │  │  ├─ initial_state(iterative_decoding)
│  │  │  ├─ operator()(step, ids, state, logits)
│  │  │  ├─ update_state(state, beam_indices)
│  │  │  └─ replicate_state(state, beam_size)
│  │  └─ Uses Ops: LayerNorm, MultiHeadAttention, FeedForward
│  │
│  ├─ AttentionLayer
│  │  ├─ MultiHeadAttention (self and cross attention)
│  │  ├─ Uses Ops: MatMul, Add, LayerNorm, Scale
│  │  └─ Optionally uses Flash Attention op when available
│  │
│  ├─ FlashAttentionLayer (📦 src/layers/flash_attention.cc)
│  │  └─ Fused SDPA for long sequences; CUDA-only on SM80+
│  │
│  ├─ TransformerEncoderLayer
│  │  └─ Self-attention + feed-forward network
│  │
│  └─ TransformerDecoderLayer
│     └─ Cross-attention + feed-forward network
│
├─ Positional Encoding Variants (📦 include/ctranslate2/layers/common.h,
│                                     include/ctranslate2/layers/attention_layer.h)
│  ├─ PositionEncoder (abstract base)
│  ├─ PositionEmbedding         → learned absolute positions (loaded from model)
│  ├─ SinusoidalPositionEncoder → fixed sinusoidal encoding (original Transformer)
│  ├─ RotaryEmbeddings          → RoPE; supports linear/dynamic/yarn scaling
│  └─ Alibi                     → ALiBi bias added to attention logits
│
├─ Whisper-specific architecture:
│  └─ WhisperEncoder has a CNN frontend before the transformer stack:
│     Conv1D(stride=1) → Conv1D(stride=2) → Transpose → PositionEmbedding
│     → N × TransformerEncoderLayer
│     (halves the time dimension, handles raw mel-spectrogram input)
│
└─ Layer Composition:
   Layer objects own weights (StorageViews)
   Layer methods call Ops (which dispatch to kernels)
```

### 4.4 Models

```
┌──────────────────────────────────────────────────────────────┐
│                  Models Layer                        │
│  📦 include/ctranslate2/models/*.h                   │
│  📦 src/models/*.cc                                  │
└──────────────────────────────────────────────────────────────┘
│
├─ Base Model (📦 include/ctranslate2/models/model.h):
│  │
│  ├─ class Model:
│  │  ├─ virtual ~Model()
│  │  ├─ nlohmann::json config
│  │  ├─ Device device() const
│  │  └─ ComputeType effective_compute_type() const
│  │
│  └─ virtual initialize(ModelReader& model_reader)
│
├─ Specialized Models:
│  │
│  ├─ SequenceToSequenceModel (📦 include/ctranslate2/models/sequence_to_sequence.h)
│  │  ├─ For encoder-decoder translation
│  │  ├─ Components:
│  │  │  ├─ Encoder (multiple encoder layers)
│  │  │  ├─ Decoder (multiple decoder layers)
│  │  │  └─ Projected embeddings (source/target)
│  │  └─ Methods:
│  │     ├─ encode(source) → encoder_output, encoder_state
│  │     └─ decode(target_prefix, encoder_state, step) → logits
│  │
│  ├─ TransformerModel
│  │  └─ Standard Transformer encoder-decoder
│  │
│  ├─ LanguageModel (📦 include/ctranslate2/models/language_model.h)
│  │  ├─ For decoder-only models (GPT, LLaMA, etc.)
│  │  └─ Methods:
│  │     └─ decode(target_prefix, step) → logits
│  │
│  └─ Specialized Models:
│     ├─ WhisperModel (📦 include/ctranslate2/models/whisper.h)
│     ├─ Wav2Vec2Model (📦 include/ctranslate2/models/wav2vec2.h)
│     └─ Wav2Vec2BertModel
│
└─ Model Loading (📦 src/models/model_reader.cc):
   ├─ ModelFileReader → Reads from filesystem
   ├─ ModelMemoryReader → Reads from memory
   └─ load_vocabulary() → Vocabulary object
```

### 4.5 Replica Pool (Parallel Execution)

```
┌──────────────────────────────────────────────────────────────┐
│              Replica Pool Layer                      │
│  📦 include/ctranslate2/replica_pool.h               │
└──────────────────────────────────────────────────────────────┘
│
├─ Template Class:
│  │
│  └─ ReplicaPool<ModelReplicaType>
│     ├─ std::vector<std::unique_ptr<ModelReplicaType>> _replicas
│     ├─ ThreadPool (📦 src/thread_pool.cc — custom JobQueue/Job impl)
│     ├─ std::queue<Batch> _work_queue
│     └─ std::condition_variable _cv
│
├─ Key Methods:
│  ├─ load_model(ModelLoader&)
│  │  ├─ Create replicas (one per worker thread)
│  │  └─ Initialize each replica with model
│  │
│  ├─ post(Batch) → std::future<Result>
│  │  └─ Submit batch to work queue
│  │
│  ├─ post_batch(std::vector<Batch>)
│  │  └─ Submit multiple batches
│  │
│  └─ _worker_thread()
│     ├─ Loop: wait for work
│     ├─ Pop batch from queue
│     ├─ Execute: replica->operator()(batch)
│     └─ Return result via promise
│
└─ Concrete Implementations:
   ├─ ReplicaPool<SequenceToSequenceReplica> → Translator
   ├─ ReplicaPool<LanguageModelReplica> → Generator
   └─ ReplicaPool<EncoderReplica> → Encoder
```

### ReplicaPoolConfig

```
ReplicaPoolConfig {
  num_threads_per_replica  → intra-op threads per model replica (0 = auto)
  max_queued_batches       → backpressure limit on the work queue (0 = unlimited)
  cpu_core_offset          → first CPU core for thread affinity (-1 = disabled)
}
```

### Parallel Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│            Batch Processing Timeline                   │
├─────────────────────────────────────────────────────────┤
│                                                    │
│  Batch A  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━          │
│  Batch B        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        │
│  Batch C              ━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  Batch D                    ━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                    │
│  └─ Each batch processed on different thread      │
│  └─ Multiple replicas = parallel processing      │
│  └─ Worker pool size = num_inter_threads          │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Complete Translation/Generation Flow

### 5.1 From Python to C++

```
Python Code:
┌─────────────────────────────────────────────────────────┐
│ translator = ctranslate2.Translator(model_path) │
│ results = translator.translate_batch(tokens)         │
└─────────────────────────────────────────────────────────┘
           ↓
Python Bindings (📦 python/cpp/translator.cc):
┌─────────────────────────────────────────────────────────┐
│ class TranslatorWrapper : ReplicaPoolHelper<Translator>│
│                                                    │
│ ├─ translate_batch(source, target_prefix, options)│
│ │  ├─ Convert Python lists to C++ vectors       │
│ │  ├─ Finalize optional batches                 │
│ │  └─ Call: _pool->translate_batch_async()    │
│ │                                           │
│ └─ Lock/unlock mutex for thread safety          │
└─────────────────────────────────────────────────────────┘
           ↓
C++ Translator (📦 include/ctranslate2/translator.h):
┌─────────────────────────────────────────────────────────┐
│ class Translator : ReplicaPool<SequenceToSequenceReplica>│
│                                                    │
│ └─ translate_batch_async()                          │
    ├─ Create futures from post_batch()               │
    └─ Return: std::vector<std::future<Result>>   │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Batch Processing Pipeline

```
translate_batch_async()
    ↓
┌──────────────────────────────────────────────────────────┐
│              Batch Processing                      │
│  📦 include/ctranslate2/batch_reader.h              │
└──────────────────────────────────────────────────────────┘
    │
    ├─ TextLineReader (if input is text)
    │  ├─ Read lines from input file
    │  ├─ Tokenize (if tokenizer provided)
    │  └─ Returns: std::vector<std::vector<std::string>>
    │
    ├─ load_examples()
    │  ├─ Convert tokens to StorageViews
    │  └─ Create Example objects
    │
    ├─ create_batches(max_batch_size, batch_type)
    │  ├─ BatchType::Examples → Batch by example count
    │  └─ BatchType::Tokens → Batch by token count
    │
    └─ pad_sequences() (📦 include/ctranslate2/padder.h)
       └─ Pad to max length in batch
```

### 5.3 Inference Execution

```
Batch submitted to worker thread
    ↓
┌──────────────────────────────────────────────────────────┐
│      SequenceToSequenceReplica::operator()        │
│  📦 include/ctranslate2/models/sequence_to_sequence.h│
└──────────────────────────────────────────────────────────┘
    │
    ├─ Encode source
    │  └─ encoder(source_ids, lengths)
    │     ├─ Embeddings lookup
    │     ├─ Pass through N encoder layers
    │     │  └─ Each layer: self-attention + feed-forward
    │     └─ Return: encoder_output, encoder_state
    │
    └─ Decode target (iterative loop)
       ├─ Initialize: decoder_state = decoder->initial_state()
       ├─ Loop until max_length or EOS:
       │  │
       │  ├─ decoder->operator()(step, ids, decoder_state, logits)
       │  │  ├─ Look up target embeddings
       │  │  ├─ Pass through N decoder layers
       │  │  │  ├─ Self-attention (to previous tokens)
       │  │  │  └─ Cross-attention (to encoder_output)
       │  │  ├─ Final layer norm and projection
       │  │  └─ Return: logits for next token
       │  │
       │  └─ Search strategy (beam/greedy)
       │     ├─ Apply: length_penalty, coverage_penalty, repetition_penalty
       │     ├─ Sample: topk, topp, temperature
       │     └─ Select next tokens (beam_size or 1)
       │
       └─ Update decoder_state for next iteration
       │
    └─ Return: TranslationResult (hypotheses, scores, attention)
```

### 5.4 Search Strategy

```
┌──────────────────────────────────────────────────────────┐
│         Decoding / Search Strategy              │
│  📦 include/ctranslate2/decoding.h                 │
└──────────────────────────────────────────────────────────┘
    │
    ├─ BeamSearch
    │  ├─ beam_size: Number of hypotheses
    │  ├─ patience: Early stopping factor
    │  ├─ length_penalty: Penalize long sequences
    │  ├─ coverage_penalty: Encourage coverage
    │  ├─ repetition_penalty: Penalize repetitions
    │  └─ Returns: top beam_size hypotheses
    │
    ├─ GreedyDecoding
    │  └─ beam_size = 1, always pick best token
    │
    └─ Sampling Variants
       ├─ sampling_topk: Sample from top K
       ├─ sampling_topp: Nucleus sampling
       ├─ sampling_temperature: Softmax temperature
       └─ Can be combined with beam search
```

---

## 6. Device and Type Dispatch System

```
┌──────────────────────────────────────────────────────────┐
│          Dispatch Macros (📦 src/dispatch.h)     │
└──────────────────────────────────────────────────────────┘
    │
    ├─ DEVICE_DISPATCH(device, statements)
    │  └─ switch(device):
    │     ├─ Device::CPU  → execute CPU statements
    │     └─ Device::CUDA → execute CUDA statements
    │                        (ROCm/HIP uses this same enum value;
    │                         code compiled with hipcc for AMD GPUs — added v4.7.0)
    │
    ├─ TYPE_DISPATCH(type, statements)
    │  └─ switch(type):
    │     ├─ float → float32_t
    │     ├─ int8 → int8_t
    │     ├─ int16 → int16_t
    │     ├─ float16 → float16_t
    │     └─ bfloat16 → bfloat16_t
    │
    ├─ DEVICE_AND_TYPE_DISPATCH(device, type, statements)
    │  └─ Nested switch (device × type combinations)
    │     └─ Generates ~12 code paths
    │
    ├─ DEVICE_AND_FLOAT_DISPATCH(device, type, statements)
    │  └─ Optimized for float types (GPU uses special kernels)
    │
    └─ CPU_ISA_DISPATCH (📦 src/cpu/cpu_isa.h)
       ├─ A third dispatch axis applied inside CPU kernels
       ├─ x86_64: GENERIC → AVX → AVX2 → AVX512
       ├─ ARM64:  GENERIC → NEON
       └─ Detected at runtime via CPUID; override: CT2_FORCE_CPU_ISA=AVX2
          (vec.h, vec_avx.h, vec_avx512.h, vec_neon.h implement SIMD wrappers)
```

### CPU GEMM Backend Selection

```
📦 src/cpu/backend.h — GemmBackend enum

Selection order (first available wins):
  MKL → DNNL (oneDNN) → Accelerate (Apple) → OpenBLAS → Ruy → NONE

├─ MKL        Intel Math Kernel Library      (CT2_WITH_MKL)
├─ DNNL       Intel oneDNN                  (CT2_WITH_DNNL)
├─ Accelerate Apple Accelerate / BLAS        (CT2_WITH_ACCELERATE)
├─ OpenBLAS   Open source BLAS               (CT2_WITH_OPENBLAS)
└─ Ruy        Google lightweight GEMM        (CT2_WITH_RUY)

Override: CT2_USE_MKL=0 to disable MKL at runtime
```

### CPU Backend Selection Decision Tree

**Runtime selection (first available wins):**
```
                   ┌─────────────┐
                   │   Start     │
                   └──────┬──────┘
                          │
                CT2_WITH_MKL=ON?
              ┌─────────────┴─────────────┐
              │ YES                      │ NO
              ↓                          ↓
        ┌──────────┐           CT2_WITH_DNNL=ON?
        │ Use MKL  │         ┌─────────────┴─────────────┐
        └──────────┘         │ YES                      │ NO
                             ↓                          ↓
                      ┌──────────┐           Apple Silicon?
                      │ Use DNNL │         ┌─────────────┴─────────────┐
                      └──────────┘         │ YES                      │ NO
                                               ↓                          ↓
                                        ┌──────────┐              CT2_WITH_OPENBLAS=ON?
                                        │Accelerate│            ┌─────────────┴─────────────┐
                                        └──────────┘            │ YES                      │ NO
                                                                 ↓                          ↓
                                                          ┌──────────┐              CT2_WITH_RUY=ON?
                                                          │OpenBLAS  │            ┌─────────────┴─────────────┐
                                                          └──────────┘            │ YES                      │ NO
                                                                                   ↓                          ↓
                                                                            ┌──────────┐              ┌──────────┐
                                                                            │   Ruy    │              │  NONE    │
                                                                            └──────────┘              └──────────┘
```

### Dispatch Example

```
Op: LayerNorm (📦 src/ops/layer_norm.cc)

void LayerNorm::operator()(StorageView& input, StorageView& output)
{
  DEVICE_AND_FLOAT_DISPATCH("LayerNorm",
                         input.device(),
                         input.dtype(),
    (compute<D, T>(beta, gamma, input, output)));
  │
  └─ Generates code for each device-type combo:
     ├─ Device::CPU + float → compute<Device::CPU, float>()
     ├─ Device::CPU + float16 → compute<Device::CPU, float16_t>()
     ├─ Device::CUDA + float → compute<Device::CUDA, float>()
     └─ Device::CUDA + float16 → compute<Device::CUDA, float16_t>()
}
```

### Dispatch Decision Tree

**When an Op is called:**
```
                       Is dtype float type?
                      ┌─────────────┐
                      │    NO       │ → DEVICE_AND_TYPE_DISPATCH
                      └──────┬──────┘
                             │ YES
                             ↓
                   ┌──────────────────┐
                   │ DEVICE_AND_     │
                   │ FLOAT_DISPATCH   │
                   └────────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              ↓                           ↓
        Device::CPU                Device::CUDA
              │                           │
         Check CPU ISA               Check GPU arch
              │                           │
         AVX512? AVX2?              SM80+?  SM60+
              │                           │
    CPU_ISA_DISPATCH               Special kernels
```

---

## 7. Model Conversion Pipeline

```
┌──────────────────────────────────────────────────────────┐
│          Python Model Converters                 │
│  📦 python/ctranslate2/converters/*.py          │
└──────────────────────────────────────────────────────────┘
    │
    ├─ Step 1: Load source model
    │  ├─ TransformersConverter: Load Hugging Face model   (transformers.py)
    │  ├─ OpenNMTPyConverter:    Load OpenNMT-py checkpoint (opennmt_py.py)
    │  ├─ OpenNMTTFConverter:    Load OpenNMT-TF checkpoint (opennmt_tf.py)
    │  ├─ OpenAIGPT2Converter:   Load OpenAI GPT-2 weights  (openai_gpt2.py)
    │  ├─ FairseqConverter:      Load Fairseq checkpoint    (fairseq.py)
    │  ├─ MarianConverter:       Load Marian model          (marian.py / opus_mt.py)
    │  └─ EoleCT2Converter:      Load Eole checkpoint       (eole_ct2.py)
    │
    ├─ Step 2: Create model spec
    │  └─ Instantiate: TransformerSpec, WhisperSpec, etc.
    │     ├─ Define expected layers (EncoderLayer, DecoderLayer, etc.)
    │     ├─ Define expected variables (weight names, shapes)
    │     └─ Register: spec.register_variable(name, array)
    │
    ├─ Step 3: Copy weights to spec
    │  └─ Iterate model state dict:
    │     └─ spec.set_weight(name, weight_array)
    │
    ├─ Step 4: Optimize
    │  ├─ spec.optimize(quantization)
    │  │  ├─ Fuse operations (if applicable)
    │  │  ├─ Convert data types (float16, int8, etc.)
    │  │  └─ Align weights (64-byte alignment)
    │  └─ spec.validate()
    │
    └─ Step 5: Save
       └─ Write to output_dir:
          ├─ model.bin (binary weights)
          └─ config.json (model configuration)
```

### Model Specs

```
┌──────────────────────────────────────────────────────────┐
│            Model Specifications                  │
│  📦 python/ctranslate2/specs/model_spec.py          │
│  📦 python/ctranslate2/specs/transformer_spec.py   │
└──────────────────────────────────────────────────────────┘
    │
    ├─ LayerSpec (base class)
    │  ├─ Frozen (immutable after creation)
    │  ├─ Declares expected weights
    │  └─ Methods:
    │     ├─ register_variable(name, value)
    │     ├─ set_variable(name, value)
    │     └─ get_variable(name) → StorageView
    │
    ├─ TransformerSpec
    │  ├─ Encoder (list of TransformerEncoderLayer)
    │  ├─ Decoder (list of TransformerDecoderLayer)
    │  └─ Embeddings (source, target)
    │
    └─ Specialized Specs
       ├─ WhisperSpec
       ├─ Wav2Vec2Spec
       └─ Wav2Vec2BertSpec
```

---

## 8. Third-Party Dependencies

```
┌──────────────────────────────────────────────────────────┐
│           Third-Party Libraries              │
│  📦 third_party/                                       │
└──────────────────────────────────────────────────────────┘
    │
    ├─ spdlog/ (header-only)
    │  └─ Logging library
    │
    ├─ cxxopts/
    │  └─ Command-line option parsing (CLI only)
    │
    ├─ cpu_features/
    │  └─ CPU feature detection (x86_64 only)
    │
    ├─ ruy/
    │  └─ Google's matrix multiplication library (optional)
    │
    ├─ thrust/ + cub/
    │  └─ CUDA parallel primitives (used with CUDA)
    │
    ├─ cutlass/
    │  └─ CUDA template library (for advanced INT8 Tensor Core kernels)
    │
    ├─ nlohmann/json.hpp
    │  └─ JSON parsing (model config)
    │
    ├─ half_float/
    │  └─ Half-precision floating point types (float16_t)
    │
    ├─ BS_thread_pool_light.hpp
    │  └─ Reference thread-pool header (not the primary thread pool;
    │     CTranslate2 uses its own JobQueue in src/thread_pool.cc)
    │
    ├─ avx_mathfun.h / avx512_mathfun.h / neon_mathfun.h
    │  └─ SIMD vectorized transcendental math (sin, cos, exp, log)
    │     for AVX, AVX-512, and ARM NEON backends
    │
    └─ googletest/
       └─ C++ unit test framework (📦 tests/)
```

---

## 9. Memory Management

```
┌──────────────────────────────────────────────────────────┐
│            Memory Allocators                   │
│  📦 include/ctranslate2/allocator.h                 │
│  📦 src/allocator.cc                                 │
└──────────────────────────────────────────────────────────┘
    │
    ├─ Allocator (base class)
    │  └─ virtual allocate(size, alignment) → void*
    │
    ├─ CPUAllocator
    │  ├─ Uses: malloc/aligned_alloc
    │  ├─ Caching: Reuses freed buffers
    │  └─ Thread-safe
    │
    ├─ CUDAAllocator
    │  ├─ Uses: cudaMalloc
    │  ├─ Caching: CUDA pool
    │  └─ ScopedDeviceSetter (sets current CUDA device)
    │
    └─ StorageView Integration:
       ├─ Owned StorageViews use allocator
       └─ View StorageViews reference external memory
```

---

## 10. Tensor Parallelism and Distributed Inference

```
┌──────────────────────────────────────────────────────────────┐
│            Tensor Parallel / MPI                     │
│  📦 include/ctranslate2/devices.h (ScopedMPISetter)  │
│  📦 src/cuda/mpi_stub.cc / nccl_stub.cc              │
│  📦 python/cpp/mpi.cc                                │
└──────────────────────────────────────────────────────────────┘
│
├─ Build flag: -DCT2_WITH_TENSOR_PARALLEL
│
├─ ScopedMPISetter (📦 include/ctranslate2/devices.h)
│  ├─ Initialises MPI (multi-process / multi-node)
│  ├─ Tracks: my_rank, local_rank, n_ranks
│  └─ Manages NCCL communicators for GPU all-reduce
│
├─ Workflow:
│  ├─ Each rank loads a shard of the model
│  ├─ Tensor-parallel ops (GEMM) split across ranks
│  ├─ NCCL all-reduce synchronises outputs
│  └─ Rank 0 gathers and returns final results
│
└─ Stubs (compile without MPI/NCCL installed):
   ├─ src/cuda/mpi_stub.cc  → no-op MPI symbols
   └─ src/cuda/nccl_stub.cc → no-op NCCL symbols
```

---

## 11. Auxiliary Components

### Dynamic Time Warping (DTW)
```
📦 src/dtw.cc / src/dtw.h

Used by Whisper for word-level timestamp alignment:
├─ Computes optimal alignment between encoder frames and decoder tokens
└─ Returns per-token start/end timestamps in the audio
```

### Buffered Translation Wrapper
```
📦 src/buffered_translation_wrapper.cc
📦 include/ctranslate2/buffered_translation_wrapper.h

Wraps Translator to support streaming / incremental output:
├─ Buffers partial translations internally
└─ Flushes complete sentences as they are ready
```

### Sampling
```
📦 src/sampling.cc
📦 include/ctranslate2/sampling.h

Standalone sampling utilities used by decoding:
├─ Top-K sampling
├─ Nucleus (top-P) sampling
└─ Temperature scaling
```

### VocabularyMap
```
📦 src/vocabulary_map.cc
📦 include/ctranslate2/vocabulary_map.h

Maps source tokens to allowed target tokens for constrained decoding:
└─ Restricts logits to a subset of the vocabulary per source context
```

---

## 12. Runtime Configuration and Profiling

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CT2_VERBOSE` | `0` | Log verbosity level (0=off, 1=info, 2=debug) |
| `CT2_FORCE_CPU_ISA` | _(auto)_ | Force CPU ISA: `GENERIC`, `AVX`, `AVX2`, `AVX512`, `NEON` |
| `CT2_USE_MKL` | _(auto)_ | `1`/`0` to force-enable or force-disable Intel MKL |
| `CT2_USE_EXPERIMENTAL_PACKED_GEMM` | `0` | Pre-pack GEMM weights for repeated small batches |
| `CT2_CUDA_ALLOCATOR` | `caching` | CUDA memory allocator: `caching` or `cuda_malloc` |
| `CT2_CUDA_TRUE_FP16_GEMM` | `1` | Use native FP16 cuBLAS GEMM (vs FP32 accumulation) |
| `CT2_CUDA_ALLOW_FP16` | `0` | Allow FP16 compute on GPUs without native support |
| `CT2_CUDA_ALLOW_BF16` | `0` | Allow BF16 compute on older CUDA devices |
| `OMP_NUM_THREADS` | _(nproc)_ | OpenMP thread count for CPU parallelism |

### Profiling

```
📦 include/ctranslate2/profiler.h
📦 src/profiler.cc

Build flag: -DCT2_ENABLE_PROFILING

├─ ScopeProfiler  → RAII scope timer, accumulates across threads by name
├─ PROFILE("name") macro → zero-overhead no-op when profiling is disabled
├─ init_profiling(device, num_threads)  → start recording
└─ dump_profiling(ostream)              → print accumulated timings

CLI shortcut: ct2-translator --log_profiling ...
```

---

## 13. Key File Navigation Map

### Entry Points
```
Python API:
  python/ctranslate2/__init__.py          → Main imports
  python/cpp/module.cc                   → pybind11 module registration

CLI:
  cli/translator.cc                       → Main executable
  cli/CMakeLists.txt                     → Build config

C++ Public API:
  include/ctranslate2/translator.h       → Translation interface
  include/ctranslate2/generator.h        → Generation interface
  include/ctranslate2/encoder.h          → Encoder interface
```

### Core Types
```
include/ctranslate2/storage_view.h          → Tensor abstraction
include/ctranslate2/types.h                → Data types (Device, DataType, etc.)
include/ctranslate2/allocator.h             → Memory allocation
include/ctranslate2/vocabulary.h           → Vocabulary handling
```

### Dispatch
```
src/dispatch.h                          → Combined dispatch macros
src/device_dispatch.h                    → Device selection
src/type_dispatch.h                      → Type selection
```

### Ops Layer
```
include/ctranslate2/ops/op.h              → Base op classes
include/ctranslate2/ops/*.h               → Op interfaces
src/ops/*.cc                            → Op dispatchers
src/ops/*_cpu.cc                        → CPU implementations
src/ops/*_gpu.cu                        → CUDA implementations
src/cuda/primitives.cu                  → CUDA primitives
src/cpu/primitives.cc                     → CPU primitives
```

### Layers Layer
```
include/ctranslate2/layers/*.h            → Layer interfaces
src/layers/*.cc                          → Layer implementations
include/ctranslate2/layers/decoder.h      → Core decoder logic
include/ctranslate2/layers/attention.h   → Attention mechanisms
```

### Models Layer
```
include/ctranslate2/models/model.h        → Base model
include/ctranslate2/models/sequence_to_sequence.h  → Enc-dec models
include/ctranslate2/models/language_model.h   → Decoder-only models
include/ctranslate2/models/transformer.h     → Transformer
include/ctranslate2/models/whisper.h          → Whisper
include/ctranslate2/models/wav2vec2.h        ├─── Wav2Vec2 variants
include/ctranslate2/models/wav2vec2bert.h   ┘
src/models/*.cc                          → Model implementations
src/models/model_reader.cc                → Model loading
```

### Execution Layer
```
include/ctranslate2/replica_pool.h                  → Parallel execution
include/ctranslate2/decoding.h                       → Search strategies
include/ctranslate2/translation.h                    → Result types
include/ctranslate2/generation.h                     → Generation types
include/ctranslate2/scoring.h                        → Scoring types
include/ctranslate2/sampling.h                       → Sampling utilities
include/ctranslate2/buffered_translation_wrapper.h   → Streaming translation
include/ctranslate2/vocabulary_map.h                 → Constrained decoding
src/dtw.h                                            → DTW for Whisper alignment
```

### Python Bindings
```
python/cpp/replica_pool.h       → Pool wrapper
python/cpp/translator.cc        → Translator wrapper
python/cpp/generator.cc         → Generator wrapper
python/cpp/encoder.cc           → Encoder wrapper
python/cpp/whisper.cc           → Whisper wrapper
python/cpp/wav2vec2.cc          → Wav2Vec2 wrapper
python/cpp/wav2vec2bert.cc      → Wav2Vec2Bert wrapper
python/cpp/storage_view.cc      → StorageView bindings
python/cpp/extensions.cc        → Extension registration
python/cpp/mpi.cc               → MPI / tensor parallel bindings
python/cpp/execution_stats.cc   → Execution stats bindings
```

### Conversion Layer
```
python/ctranslate2/specs/model_spec.py    → Base spec
python/ctranslate2/specs/transformer_spec.py → Transformer
python/ctranslate2/converters/*.py     → All converters
python/ctranslate2/models/*.py          → Python model classes
```

### Cross-Reference Map

**Concept → Documentation Location**
- Device Dispatch            → Section 6
- CPU ISA Dispatch           → Section 6
- CPU GEMM Backend           → Section 6
- Type Dispatch              → Section 6
- StorageView Memory         → Section 4.1 / Section 9
- ComputeType / Quantization → Section 4.1b
- Ops Implementation         → Section 4.2
- Layers Composition         → Section 4.3
- Positional Encoding        → Section 4.3
- Parallel Execution         → Section 4.5
- Model Loading              → Section 7
- Third-Party Deps           → Section 8
- Memory Allocators          → Section 9
- Tensor Parallel / MPI      → Section 10
- DTW / Streaming / Sampling → Section 11
- Environment Variables      → Section 12
- Profiling                  → Section 12
- Batch Processing           → Section 5.2
- Search Strategies          → Section 5.4
- Testing                    → Section 16
- Performance Guidelines     → Section 17

---

## 14. Complete Data Flow Example

### Python Translation

```
┌────────────────────────────────────────────────────────────────┐
│  User Code                                         │
│                                                     │
│  translator = Translator("model_path")                  │
│  results = translator.translate_batch([tokens])            │
└────────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│  Python Bindings (python/cpp/translator.cc)              │
│                                                     │
│  TranslatorWrapper::translate_batch(source, options)      │
│  ├─ Convert Python lists → C++ vectors               │
│  ├─ Lock mutex                                      │
│  └─ _pool->translate_batch_async()                  │
└────────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│  C++ Translator (include/ctranslate2/translator.h)   │
│                                                     │
│  translate_batch_async()                               │
│  ├─ Create batches                                   │
│  └─ post_batch(batch) → future<Result>              │
└────────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│  Replica Pool (include/ctranslate2/replica_pool.h)      │
│                                                     │
│  post_batch() → _work_queue.push(batch)                │
│  Worker thread:                                         │
│  ├─ pop batch                                          │
│  ├─ replica->operator()(batch)                           │
│  └─ future.set_value(result)                          │
└────────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│  Model Replica (include/ctranslate2/models/              │
│                sequence_to_sequence.h)                   │
│                                                     │
│  operator()(batch) → TranslationResult                   │
│  ├─ For each example in batch:                        │
│  │  ├─ encode(source_ids) → encoder_output           │
│  │  │  ├─ embeddings lookup                          │
│  │  │  └─ encoder layers (N × (self-attn + ff)) │
│  │  │                                                │
│  │  └─ decode(target_prefix, encoder_state) → results  │
│  │     ├─ Initialize decoder state                      │
│  │     ├─ Loop until max_length or EOS:                │
│  │     │  ├─ decoder->operator()(step, ids, state, logits)
│  │     │  │  ├─ embeddings lookup                     │
│  │     │  │  ├─ decoder layers (N × (self-attn +     │
│  │     │  │  │  cross-attn + ff))              │
│  │     │  │  └─ projection layer               │
│  │     │  ├─ apply_search_strategy(logits, step)      │
│  │     │  │  ├─ beam search or greedy       │
│  │     │  │  ├─ apply penalties                │
│  │     │  │  └─ sample next tokens            │
│  │     │  └─ update decoder state                  │
│  │     └─ return best hypotheses                     │
│  │                                                │
│  └─ Return: std::vector<TranslationResult>            │
│                                                     │
│  Throughout execution:                                    │
│  ├─ All tensors are StorageView objects                 │
│  ├─ Ops use DEVICE_AND_TYPE_DISPATCH()                │
│  ├─ Memory reused via allocators                     │
│  └─ No unnecessary copies (zero-copy views)            │
└────────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────────┐
│  Back to Python                                      │
│                                                     │
│  results = wait_on_futures(futures)                    │
│  └─ Convert C++ results → Python objects               │
└────────────────────────────────────────────────────────────────┘
```

---

## 15. Quick Reference for Common Tasks

### Add New Op
```
1. Create header:  include/ctranslate2/ops/my_op.h
   └─ Declare MyOp : public UnaryOp/BinaryOp

2. Create dispatcher: src/ops/my_op.cc
   └─ Use DEVICE_AND_TYPE_DISPATCH()

3. Create CPU impl: src/ops/my_op_cpu.cc

4. Create GPU impl: src/ops/my_op_gpu.cu

5. Update CMakeLists.txt: Add source files

6. Create tests: tests/ops_test.cc
```

### Add New Model Type
```
1. Create spec: python/ctranslate2/specs/my_model_spec.py
   └─ Inherit from ModelSpec

2. Create C++ model: include/ctranslate2/models/my_model.h
   └─ Inherit from Model or SequenceToSequenceModel

3. Implement model: src/models/my_model.cc
   └─ Implement initialize() and forward methods

4. Register in factory: src/models/model_factory.cc
   └─ REGISTER_MODEL(MyModel, "my_model_type")

5. Create converter: python/ctranslate2/converters/my_converter.py
   └─ Implement Converter base class

6. Create Python wrapper: python/ctranslate2/models/my_model.py
   └─ Expose via _ext imports
```

### Debug Performance Issues
```
1. Enable profiling:
   ct2-translator --log_profiling --model <path> <input>

2. Check device/type dispatch:
   └─ Verify correct kernel is being used

3. Check allocator cache:
   └─ Look for excessive allocations

4. Use benchmark tools:
   └─ tools/benchmark/benchmark.py
```

### Run Tests
```
C++ tests:
  cd build && ./tests/ctranslate2_test tests/data

Python tests:
  cd python && pytest tests/

Specific test file:
  pytest tests/test_translator.py -v -k "test_beam_search"
```

### Compute Type Selection Guide
```
Use case → Recommended Compute Type
─────────────────────────────────────────
Maximum accuracy (CPU)       → FLOAT32
Maximum accuracy (GPU)       → FLOAT32 / BFLOAT16 (if supported)
GPU with memory constraints → FLOAT16 / INT8
CPU with memory constraints  → INT8 / INT16
Mobile/Edge devices        → INT8
─────────────────────────────────────────

Resolution order:
1. Explicit compute_type parameter
2. AUTO: Select based on device capabilities
3. DEFAULT: Use model's stored type
4. Fallback to FLOAT32
```

---

This diagram provides a complete navigation map of the CTranslate2 codebase. For more detailed information, refer to the specific source files listed in each section.

---

## 16. Testing Architecture

### C++ Tests

```
📦 tests/
├─ ops_test.cc           → Unit tests for all ops
├─ layers_test.cc        → Layer-level tests
├─ model_test.cc         → Model integration tests
├─ attention_test.cc     → Attention mechanism tests
├─ batching_test.cc      → Batch processing tests
├─ decoding_test.cc      → Beam search / greedy tests
├─ translator_test.cc    → End-to-end translation tests
├─ storage_view_test.cc  → Tensor abstraction tests
├─ primitives_test.cc    → Low-level primitive tests
├─ benchmark_ops.cc      → Op microbenchmarks
└─ test_utils.h          → Test utilities (expect_storage_eq, etc.)

Testing approach:
└─ Google Test framework; run with: ./tests/ctranslate2_test tests/data
```

### Python Tests

```
📦 python/tests/
├─ conftest.py           → pytest fixtures (model paths, etc.)
├─ test_translator.py    → Translator API tests
├─ test_transformers.py  → HuggingFace converter tests
├─ test_opennmt_py.py    → OpenNMT-py converter tests
├─ test_opennmt_tf.py    → OpenNMT-TF converter tests
├─ test_fairseq.py       → Fairseq converter tests
├─ test_marian.py        → Marian converter tests
├─ test_spec.py          → Model spec tests
├─ test_storage_view.py  → StorageView Python API tests
├─ test_misc.py          → Miscellaneous tests
└─ test_utils.py         → Test utilities
```

### Test Data

```
📦 tests/data/
└─ models/                → Small test models (downloaded in CI)
```

### CI Testing

```yaml
GitHub Actions runs:
├─ C++ tests: MKL, DNNL, AddressSanitizer
├─ Python tests: from built wheels
├─ ARM64: OpenBLAS + Ruy
├─ Docker: CUDA and ROCm
└─ Style: black, isort, flake8
```

---

## 17. Performance Optimization Guidelines

### Memory Optimization

```
DO:
├─ Use view() instead of copies when possible
├─ Reuse StorageView objects in loops
├─ Batch small sequences together
├─ Use appropriate compute_type (INT8 for large models)
└─ Let allocator cache buffers

DON'T:
├─ Unnecessary to(Device) calls
├─ Create StorageView for temporary operations
├─ Copy large tensors unnecessarily
└─ Disable allocator caching
```

### Computation Optimization

```
DO:
├─ Use Flash Attention when available (CUDA SM80+)
├─ Enable quantization for large models
├─ Tune batch_size for throughput
├─ Use num_inter_threads for CPU
└─ Enable experimental packed GEMM for small batches

DON'T:
├─ Force higher precision than needed
├─ Use single-threaded for large batches
└─ Disable ISA dispatch optimizations
```

### Profiling Tips

```
1. Use --log_profiling to identify bottlenecks
2. Check allocator stats for memory churn
3. Verify correct dispatch (CPU ISA, GPU arch)
4. Monitor throughput with --log_throughput
5. Profile with perf/VTune on CPU, Nsight on GPU
```

### Common Performance Issues

```
Symptom               → Likely Cause          → Solution
──────────────────────────────────────────────────────────
Slow CPU              → Wrong backend          → Check MKL/DNNL
Low GPU utilization   → Small batches          → Increase batch_size
High memory usage     → No quantization       → Use INT8/INT16
Stuttering latency    → Single-threaded       → Use thread pool
Frequent allocations  → No caching             → Check allocator
```