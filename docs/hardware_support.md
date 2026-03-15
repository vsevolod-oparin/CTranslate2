# Hardware support

```{attention}
The information below is only valid for the prebuilt binaries. If you compiled the project from the sources, the supported hardware will depend on the selected backend and compilation flags.
```

## CPU

* x86-64 processors supporting at least SSE 4.1
* AArch64/ARM64 processors

On x86-64, prebuilt binaries are configured to automatically select the best backend and instruction set architecture for the platform (AVX, AVX2, or AVX512). In particular, they are compiled with both [Intel MKL](https://software.intel.com/en-us/mkl) and [oneDNN](https://github.com/oneapi-src/oneDNN) so that Intel MKL is only used on Intel processors where it performs best, whereas oneDNN is used on other x86-64 processors such as AMD.

```{tip}
See the [environment variables](environment_variables.md) `CT2_USE_MKL` and `CT2_FORCE_CPU_ISA` to control this behavior.
```

## GPU (NVIDIA)

* NVIDIA GPUs with a Compute Capability greater or equal to 3.5

The driver requirement depends on the CUDA version. See the [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

## GPU (Apple Silicon / Metal)

```{attention}
The Metal backend is not available in prebuilt binaries. You must build CTranslate2 from source with `-DWITH_METAL=ON`.
```

### Requirements

* Apple Silicon Mac (M1 or later)
* macOS 14 (Sonoma) or later
* CMake build with `-DWITH_METAL=ON`

### Supported compute types

| Compute type | Support | Notes |
|-------------|---------|-------|
| `float32` | Full | Default compute type on Metal |
| `float16` | Full | Typically 1.5x faster than float32 for single inference |
| `bfloat16` | M3 or later | Requires `MTLGPUFamilyApple9` (auto-detected at runtime) |
| `int8` | Supported | CPU dequantize followed by GPU GEMM |
| `int8_float16` | Supported | Hybrid INT8 quantization with FP16 GEMM |
| `int8_bfloat16` | Supported | Hybrid INT8 quantization with BF16 GEMM (M3 or later) |

### Known limitations

* **AWQ quantization** is not supported (no native INT8 matrix multiplication on Metal)
* `gemm_pack_b` always returns 0 (weight pre-packing is not supported on Metal)
* **RMSNorm with residual output** path is not supported, which blocks models that require it (e.g., Gemma)
* **BF16** requires macOS 14+ and an M3 or later chip; on older hardware it is automatically disabled

```{tip}
See the [environment variables](environment_variables.md) page for Metal-specific settings:

* `CT2_MPS_ALLOW_BF16` -- override BF16 hardware detection
* `CT2_METAL_POOL_MAX_MB` -- cap the Metal buffer pool size (useful for memory-constrained workloads)
* `CT2_MPS_TRACE` -- enable Metal command buffer tracing for debugging
```
