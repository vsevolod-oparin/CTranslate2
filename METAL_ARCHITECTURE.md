# Metal Backend Architecture

**Branch:** `metal-backend` | **Last updated:** 2026-03-10 | **Status:** M11.29 complete (22.1x speedup)

---

## 1. Execution Model

### Command-buffer lifecycle

Every Metal op **encodes only** — it never commits its own command buffer.

```
  encode phase  →  commit_and_wait()  →  CPU reads
       │                  │
  GPU encodes          commits +
  kernels into         waits for
  per-thread CB        completion
```

- **One MTLDevice** per process (singleton, lazy init in `device.mm`).
- **One MTLCommandQueue per thread** (`thread_local`, created on first use in `utils.mm`).
- **One MTLCommandBuffer per thread** (also `thread_local`; reset after each commit).
- **One MTLSharedEvent per thread** (`thread_local`, for GPU-side cross-CB ordering — M11.29).
- `synchronize_stream(Device::METAL)` / `synchronize_device(Device::METAL)` → calls `commit_and_wait()`.
- `metal::commit_and_wait()` must be called before any CPU reads from Metal buffers.

### GPU-side barriers (M11.29)

When a non-blocking `commit_command_buffer()` splits the command buffer (e.g. MPS padded GEMM
CB split in `primitives_gemm.mm`), subsequent encode-only kernels in the **new** CB may need to
read data written by the **prior** CB. Two mechanisms handle this:

- **`commit_and_wait()`** (CPU-blocking): full barrier, guarantees all prior GPU work complete.
  Required for f32 `indexed_fill` due to MPS driver coherency issue.
- **`encode_barrier()`** (GPU-side, non-blocking): encodes `[cb encodeWaitForEvent:value:]` into
  the current CB. GPU serializes; CPU continues encoding freely. Sufficient for f16/bf16 paths.

`commit_command_buffer()` signals a `MTLSharedEvent` with an incrementing counter. `encode_barrier()`
encodes a wait on the latest counter. A `_last_waited` tracker makes `encode_barrier()` a no-op when
`commit_and_wait()` already drained all prior GPU work.

### Deferred-free mechanism (M11.18)

Encode-only GPU kernels may read buffers that are freed before `commit_and_wait()` runs.
`protect_buffer(ptr)` marks a live allocation as GPU-referenced. When freed, the buffer goes to
a deferred queue instead of the reuse pool. `flush_pending_frees()` (called from `commit_and_wait()`)
moves deferred buffers back to the pool after all GPU work completes.

### Memory model

All allocations use `MTLResourceStorageModeShared`:
- Buffer contents are simultaneously valid for CPU and GPU on Apple Silicon.
- No explicit copy between CPU and GPU is ever needed.
- After a GPU write, `commit_and_wait()` is the only synchronisation needed before CPU reads.

### M7 "commit_and_wait + CPU" pattern

Ops that lack a GPU kernel (Concat, Split, Tile, TopK, etc.) call `commit_and_wait()` at the
start of `compute<METAL>`, then run the algorithm directly on the shared-memory CPU pointer.
The result is immediately visible to the next GPU op encoded into the fresh command buffer.

---

## 2. Directory Structure

```
src/metal/
├── device.mm                  # MTLDevice singleton, init/teardown
├── utils.mm / utils.h         # get_metal_device/queue/cmd_buffer, commit_and_wait,
│                              #   encode_barrier (MTLSharedEvent), commit tracing
├── allocator.mm               # MetalAllocator: newBufferWithLength (Shared mode), free,
│                              #   protect_buffer, flush_pending_frees, buffer_for_ptr O(log N)
├── primitives_infra.h         # Shared internal header: compile_library_once, PSOCache,
│                              #   ct2_u32, alloc_temp_buffer, MetalTypeName, METAL_STUB
├── msl_strings.h              # AUTO-GENERATED — all MSL kernels as C++ string constants
│                              #   (do not edit; run tools/gen_msl_strings.py to regenerate)
│
├── ops_conv1d.mm              # metal::conv1d_metal (im2col + GEMM)
├── ops_quantize.mm            # metal::quantize_int8_metal, dequantize_int8_metal,
│                              #   dequantize_gemm_output_metal
├── primitives_memory.mm       # at, fill, copy, convert, indexed_fill (GPU), cross_device_copy
├── primitives_elementwise.mm  # add/sub/mul/min/max (scalar+vec and vec+vec),
│                              #   activations (relu/gelu/gelu_tanh/gelu_sigmoid/sigmoid/swish),
│                              #   broadcast variants (batch/depth/block)
├── primitives_reduction.mm    # sum, max, amax, max_element, logsumexp
├── primitives_gemm.mm         # gemm, gemm_batch_strided (MPS for f32/f16; MPSGraph for bf16),
│                              #   MPSMatrixMultiplication cache, f16 GEMV kernel
├── primitives_transpose.mm    # transpose_2d/3d/4d (GPU kernels, 6 types each)
├── primitives_beam_search.mm  # penalize_previous_tokens (GPU), prepare_length_mask (GPU)
├── ops_norm_gather.mm         # metal:: wrappers: layer_norm, rms_norm, softmax, gather
├── ops_sdpa.mm                # metal::sdpa_metal (MPS + MPSGraph; handles KV-cache offset)
├── ops_rotary.mm              # metal::rotary_metal (GPU prefill kernel)
├── ops_alibi.mm               # metal::alibi_add_metal (GPU kernel)
│
└── kernels/                   # Canonical MSL source files (input to gen_msl_strings.py)
    ├── elementwise.metal
    ├── activation.metal
    ├── broadcast.metal
    ├── beam_search.metal
    ├── transpose.metal
    ├── reduction.metal
    ├── normalization.metal
    ├── gather.metal
    ├── sdpa.metal
    ├── conv1d.metal
    ├── quantize.metal
    └── indexed_fill.metal

src/ops/                       # Op::compute<Device::METAL> specializations
├── normalization_metal.mm     # LayerNorm, RMSNorm, SoftMax → delegates to ops_norm_gather
├── gather_metal.mm            # Gather → delegates to ops_norm_gather
├── bias_add_metal.mm          # BiasAdd (add_batch_broadcast / add_block_broadcast)
├── flash_attention_metal.mm   # FlashAttention (SDPA + decode RoPE CPU path)
├── rotary_metal.mm            # Rotary → delegates to ops_rotary
├── alibi_add_metal.mm         # AlibiAdd → delegates to ops_alibi
├── concat_split_slide_metal.mm # Concat, Split, Slide (commit_and_wait + memcpy)
├── tile_metal.mm              # Tile (commit_and_wait + memcpy)
├── topk_metal.mm              # TopK (commit_and_wait + partial_sort / max_element)
├── topp_mask_metal.mm         # TopPMask + max_num_classes<METAL> (no vocab limit)
├── gumbel_max_metal.mm        # GumbelMax (commit_and_wait + CPU RNG)
├── multinomial_metal.mm       # Multinomial (commit_and_wait + discrete_distribution)
├── mean_metal.mm              # Mean (commit_and_wait + 3-loop CPU)
├── median_filter_metal.mm     # MedianFilter (commit_and_wait + nth_element)
├── conv1d_metal.mm            # Conv1D → delegates to metal::conv1d_metal
├── quantize_metal.mm          # Quantize → delegates to metal::quantize_int8_metal
└── dequantize_metal.mm        # Dequantize → delegates to metal::dequantize_int8_metal /
                               #   dequantize_gemm_output_metal

tools/
└── gen_msl_strings.py         # Reads kernels/*.metal → msl_strings.h
                               #   python3 tools/gen_msl_strings.py          (regenerate)
                               #   python3 tools/gen_msl_strings.py --check  (CI verify)
```

---

## 3. Two-Layer Design

```
Op layer:  Op::compute<Device::METAL, T>(StorageView&, ...)
              src/ops/*_metal.mm
                   │
                   ▼
Primitives: primitives<Device::METAL>::gemm / add / relu / ...
  + Metal:  metal::layer_norm_metal / sdpa_metal / ...
              src/metal/primitives_*.mm
              src/metal/ops_*.mm
```

- **Op layer** (`src/ops/*_metal.mm`): receives `StorageView`, extracts raw pointers and
  shape info, calls into the primitives/Metal layer. One file per op or op group.
- **Primitives layer** (`src/metal/primitives_*.mm`): receives raw typed pointers + dim_t sizes.
  Encodes GPU kernels or calls `commit_and_wait()` + CPU algorithm on shared memory.
- **Metal ops** (`src/metal/ops_*.mm`): free functions (no StorageView) implementing complex
  ops (SDPA, LayerNorm, RoPE, ALiBi) — tested standalone.

---

## 4. GEMM Dispatch

| dtype | API used | Notes |
|-------|---------|-------|
| float32 | `MPSMatrixMultiplication` | Eager, encode-only |
| float16 | `MPSMatrixMultiplication` (or custom GEMV for m=1) | Eager, encode-only; cached MPSMatrixMultiplication objects (M11.27) |
| bfloat16 | `MPSGraph` matmul | **Commits immediately** (MPSGraph limitation) |
| int8→int32 | CPU int8→f32 + `MPSMatrixMultiplication` + CPU f32→int32 | No native MPS INT8; dequantize-before-GEMM workaround (M9.2) |

`MPSMatrixMultiplication` requires `rowBytes ≥ rowBytesForColumns:` (MPS minimum alignment).
`primitives_gemm.mm` auto-pads into a temporary buffer when needed.

`get_current_command_buffer()` must be called **outside** any `@autoreleasepool {}` block to
avoid use-after-free when the autorelease pool drains.

---

## 5. MSL Kernel Inventory

Each kernel group has one MSL source file and one `PSOCache` + `get_*_library()` getter in the
corresponding `primitives_*.mm` or `ops_*.mm` file.

| Kernel group | Source | Key kernels |
|---|---|---|
| `kElementwiseMSL` | `elementwise.metal` | `add_T`, `sub_T`, `mul_T`, `min_T`, `max_T` (binary), `*_scalar_T` (scalar) |
| `kActivationMSL` | `activation.metal` | `relu_T`, `gelu_T`, `gelu_tanh_T`, `gelu_sigmoid_T`, `sigmoid_T`, `swish_T`, `exp_T`, `log_T`, `cos_T`, `sin_T`, `tanh_T` |
| `kBroadcastMSL` | `broadcast.metal` | `add_batch_broadcast_T`, `add_depth_broadcast_T`, `add_block_broadcast_T`, `mul_batch_broadcast_T`, `mul_block_broadcast_T` |
| `kBeamSearchMSL` | `beam_search.metal` | `penalize_previous_tokens_T`, `prepare_length_mask` |
| `kTransposeMSL` | `transpose.metal` | `transpose_2d_T`, `transpose_3d_T`, `transpose_4d_T` (all 6 types) |
| `kReductionMSL` | `reduction.metal` | `reduce_sum_T`, `reduce_max_T`, `reduce_amax_T`, `reduce_max_element_T` |
| `kNormalizationMSL` | `normalization.metal` | `layer_norm_T`, `rms_norm_T`, `softmax_T` |
| `kGatherMSL` | `gather.metal` | `gather_T` |
| `kSdpaMSL` | `sdpa.metal` | `causal_mask_float/half/bfloat`, MPS-driven SDPA (no MSL matmul) |
| `kConv1dMSL` | `conv1d.metal` | `im2col_float`, `im2col_half`, `im2col_bfloat` |
| `kQuantizeMSL` | `quantize.metal` | `quantize_T`, `dequantize_T`, `dequantize_gemm_output_T` |
| `kIndexedFillMSL` | `indexed_fill.metal` | `indexed_fill_T` (GPU scatter) |
| `kRotaryMSL` (inline) | `ops_rotary.mm` | `rotary_T` — not in gen_msl_strings.py |
| `kAlibiMSL` (inline) | `ops_alibi.mm` | `alibi_add_T` — not in gen_msl_strings.py |

`T` expands to: `float`, `half`, `bfloat`, `int`, `short`, `char`
(bfloat requires Metal 3.1 / Apple9+; available on M4).

---

## 6. Op Specialization Pattern

```cpp
// src/ops/foo_metal.mm
#include "ctranslate2/ops/foo.h"
#include "metal/utils.h"
// ... other includes

// IMPORTANT: qualify float16/bfloat16 before 'using namespace ctranslate2'
// to avoid ambiguity with ARM vector type headers.
typedef ctranslate2::float16_t  ct2_f16;
typedef ctranslate2::bfloat16_t ct2_bf16;
using namespace ctranslate2;

namespace ctranslate2 { namespace ops {

  template <Device D, typename T>
  void Foo::compute(...) const {
    // For GPU ops: encode into get_current_command_buffer(); no commit.
    // For CPU-fallback: metal::commit_and_wait(); then use raw pointers.
  }

  #define DECLARE_IMPL(T) \
      template void Foo::compute<Device::METAL, T>(...) const;
  DECLARE_ALL_TYPES(DECLARE_IMPL)   // or list explicit types
  #undef DECLARE_IMPL

}}
```

`DECLARE_ALL_TYPES` expands to: `float`, `int8_t`, `int16_t`, `int32_t`,
`ctranslate2::float16_t`, `ctranslate2::bfloat16_t`.

---

## 7. Type Name Convention

```cpp
// MetalTypeName<T>::value maps C++ → MSL type string (used for kernel name formatting)
float         → "float"
float16_t     → "half"
bfloat16_t    → "bfloat"
int32_t       → "int"
int16_t       → "short"
int8_t        → "char"
```

Kernel names follow: `<operation>_<type>`, e.g. `add_float`, `relu_half`, `gather_bfloat`.

---

## 8. Dispatch Macro Patterns

```cpp
// Comma protection inside DEVICE_AND_FLOAT_DISPATCH — wrap multi-arg in parens:
DEVICE_AND_FLOAT_DISPATCH("Op", device, dtype, (compute<D, T>(a, b)));

// Comma protection inside DEVICE_DISPATCH — use SINGLE_ARG wrapper:
DEVICE_DISPATCH(device, SINGLE_ARG(result = foo<D, T>()));
```

Only `float`, `float16_t`, `bfloat16_t` are dispatched by `DEVICE_AND_FLOAT_DISPATCH`.
`DEVICE_DISPATCH` dispatches all types including int.

---

## 9. Milestone History (one-line per milestone)

| Milestone | What was added |
|-----------|---------------|
| M0 | Feasibility: MPS GEMM, BF16 via MPSGraph, CB overhead measurement (~0.4 ms) |
| M1 | `Device::METAL` enum, CMake `CT2_WITH_METAL`, Python exposure |
| M2 | `metal::get_metal_device/queue/cmd_buffer`, `commit_and_wait`, error handling |
| M3 | `MetalAllocator` (Shared mode), `StorageView` Metal device support |
| M4.1 | Memory primitives: `at`, `fill`, `copy`, `convert` |
| M4.2 | Elementwise: `add/sub/mul/min/max`, MSL + PSO infrastructure |
| M4.3 | Reduction: `sum/max/amax/max_element`, two-pass GPU+CPU |
| M4.4 | GEMM: MPS f32/f16, MPSGraph bf16, batch-strided |
| M4.5 | Activations: `relu/gelu/gelu_tanh/gelu_sigmoid/sigmoid/swish/exp/log/cos/sin/tanh` |
| M4.6 | Broadcast: `add_batch/depth/block_broadcast`, `mul_batch/block_broadcast` |
| M4.7 | Beam search: `penalize_previous_tokens` (GPU), `prepare_length_mask` (CPU) |
| M4.8 | Transpose: GPU kernels for 2D/3D/4D, all 6 types |
| M5.1 | Fixed `DEVICE_AND_FLOAT_DISPATCH` for Metal ARM header BF16/F16 name collision |
| M5.2 | Op specializations: LayerNorm, RMSNorm, SoftMax, Gather, BiasAdd |
| M6.1 | SDPA: `metal::sdpa_metal` via MPS (f32/f16) and MPSGraph (bf16) |
| M6.2 | KV-cache: offset-based decode in `FlashAttention::compute<METAL>` |
| M6.3 | RoPE: GPU prefill kernel, CPU decode path in `flash_attention_metal.mm` |
| M6.4 | ALiBi: `metal::alibi_add_metal` GPU kernel, `AlibiAdd::compute<METAL>` |
| M7 | CPU-fallback ops: Concat, Split, Slide, Tile, TopK, TopPMask, GumbelMax, Multinomial, Mean, MedianFilter |
| M8.1 | Integration validation: full encoder layer pipeline, 11/11 pass, errors ~1e-7 |
| M8.2 | Integration validation: full decoder layer (cross-attn sq≠sk, KV-cache decode), 11/11 pass |
| M8.3 | Conv1D: im2col MSL kernel + Metal GEMM; f32/f16/bf16; 7/7 pass, errors ~1e-7 |
| M9.1 | INT8 Quantize/Dequantize: GPU kernels (quantize/dequantize/dequantize_gemm_output); 15/15 pass |
| M9.2 | INT8 GEMM: CPU int8→f32, float32 MPS GEMM, CPU f32→int32; 9/9 pass, norm error 0.43% |
| M9.3 | gemm_pack_b returns 0 (was already done in M4.4) |
| M9.4 | compute_u8_compensation changed from METAL_STUB to no-op |
| M10 | E2E validation: Seq2Seq, LM, Whisper; beam_size>1 gather fix; GEMM padding hazard fix |
| M11.1–4 | CB batching, PSO caching, BF16 inference (M3+), GPU-native profiling |
| M11.5–11 | General optimization: CPU GEMM tiny, MPS batched/padded GEMM, GPU blit, flash cross-attn, fused LN+GEMM, GPU TopK |
| M11.12–17 | Pad-C sync elim, fused timestamp rules, batch beam gather, GPU prepare_length_mask |
| M11.18–21 | Encode-only MPS padded GEMM, f16 GEMV kernel, GPU multinomial, gather sync elimination |
| M11.22 | **Memory leak fix**: 6 MPS object leak categories; RSS 5GB→800MB; 2.7x speedup |
| M11.23–25 | GPU fused TopK (268 syncs→0), GPU indexed_fill scatter kernel |
| M11.26 | **Sync elimination**: encode-only patterns, batched sampling sync; 2,977→1,941ms = 1.53x |
| M11.27–28 | MPSMatrixMultiplication cache, indexed_fill pre-sync elimination (f16) |
| M11.29 | **MTLSharedEvent encode_barrier**: hybrid sync (f32 CPU-block, f16/bf16 GPU-barrier); **final: 1,860ms = 22.1x** |
