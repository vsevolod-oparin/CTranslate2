# CTranslate2 Agent Guide

This guide helps agents work effectively in the CTranslate2 codebase.

## Project Overview

CTranslate2 is a high-performance C++ and Python library for efficient inference with Transformer models. It's a **low-level, performance-critical codebase** where efficient memory usage and computation speed are primary concerns.

### Key Characteristics

- **Performance-focused**: Custom runtime with optimizations like weight quantization, layer fusion, batch reordering
- **Multi-backend**: Supports CPU backends (MKL, DNNL, OpenBLAS, Ruy, Accelerate) and GPU backends (CUDA, ROCm/HIP)
- **Multi-language**: Core in C++17, Python bindings via pybind11
- **Production-oriented**: Maintains backward compatibility guarantees

## Essential Commands

### C++ Build

```bash
# Basic build
mkdir build && cd build
cmake ..
make -j$(nproc) install

# Build with tests
cmake -DBUILD_TESTS=ON ..
make -j$(nproc) install

# Run C++ tests
./tests/ctranslate2_test tests/data

# Build with specific backend
cmake -DWITH_MKL=ON -DBUILD_TESTS=ON .
cmake -DWITH_DNNL=ON -DBUILD_TESTS=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP .
cmake -DWITH_OPENBLAS=ON -DWITH_RUY=ON -DBUILD_TESTS=ON -DWITH_MKL=OFF .

# Build with CUDA support
cmake -DWITH_CUDA=ON -DBUILD_TESTS=ON .
```

### Python Build and Test

```bash
cd python

# Install in development mode
pip install -r install_requirements.txt
pip install -e .

# Run Python tests
pytest tests/

# Format and lint code
black .
isort .
flake8 .
```

### Build Python Wheels

```bash
# Build wheels (uses cibuildwheel)
cd python
pip install -r install_requirements.txt
python -m build

# Prepare build environment (scripts in python/tools/)
bash python/tools/prepare_build_environment_linux.sh
bash python/tools/prepare_build_environment_macos.sh
bash python/tools/prepare_build_environment_windows.sh
```

### Docker Build

```bash
# Build Docker image
./docker/build_all.sh latest 0 cuda  # build only
./docker/build_all.sh 4.7.1 1 cuda  # build and push tag
```

## Code Organization

### Directory Structure

```
├── include/ctranslate2/     # C++ public headers
├── src/                      # C++ implementation
│   ├── ops/                 # Neural network operations
│   ├── layers/               # Neural network layers (stateful)
│   ├── models/               # Model implementations
│   ├── cuda/                # CUDA-specific code
│   ├── cpu/                 # CPU-specific code
│   └── dispatch.h           # Device/type dispatching
├── python/
│   ├── ctranslate2/          # Python package
│   │   ├── converters/       # Model format converters
│   │   ├── specs/           # Model specifications
│   │   └── models/          # Python model interfaces
│   ├── cpp/                 # Python bindings (pybind11)
│   └── tests/              # Python tests
├── cli/                     # Command-line tools
├── tests/                   # C++ tests (Google Test)
└── third_party/             # Vendored dependencies
```

### C++ Architecture Levels

From lowest to highest (as documented in CONTRIBUTING.md):

1. **Kernels**: Low-level compute functions (e.g., CUDA implementation of Softmax)
2. **Primitives**: Basic vector/matrix processing (e.g., addition of C arrays)
3. **Ops**: Neural network operations (e.g., Softmax, Gemm)
4. **Layers**: Stateful neural network layers (e.g., Dense, LayerNorm)
5. **Models**: Collection of layers and weights (e.g., Transformer)
6. **Replicas**: Runnable instances of a model
7. **Replicas pool**: Thread pool of model replicas

### Op Implementation Pattern

Each op typically requires multiple source files:

```
include/ctranslate2/ops/my_op.h    # Op interface
src/ops/my_op.cc                 # Input checks and device/type dispatch
src/ops/my_op_cpu.cc             # CPU-specific implementation
src/ops/my_op_gpu.cu             # CUDA-specific implementation
```

**Important**: No compilation flags should be used in header files to make the library easy to use as a dependency.

## Code Conventions

### C++

- **C++17** standard required
- **Row-major** storage order (encapsulated in `StorageView` class)
- **RAII** for resource management
- **Type dispatch** for handling different data types (see `dispatch.h`)
- **Device dispatch** for CPU vs GPU code paths
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes

### Python

- **Python 3.9+** support
- **Type hints** required
- **Docstrings** for public APIs
- **Testing**: pytest with fixtures in `conftest.py`
- **Formatting**: black (line length 88 chars), isort, flake8

## Key Classes and Concepts

### StorageView

Lightweight tensor wrapper around allocated buffer:

- Row-major storage with shape information
- Type-safe with dynamic type dispatch
- Device-aware (CPU/GPU)
- Allocation-aligned to 64 bytes by default
- Minimizes allocations: no reallocation for smaller resizes, uses caching allocators

**Critical pattern**: Avoid unnecessary copies. Use `view()` or move semantics when possible.

### Model Spec System

Located in `python/ctranslate2/specs/`:

- `ModelSpec`: Base class for model specifications
- `LayerSpec`: Frozen (immutable) layer weight specifications
- `visit_spec()`: Recursive traversal of spec hierarchy
- Model converters populate specs by calling `spec.register_variable()`

### Device and Type Dispatch

The codebase uses macro-based dispatch:

- `DEVICE_DISPATCH(device, code)`: Dispatch by device type (CPU/CUDA)
- `TYPE_DISPATCH(type, code)`: Dispatch by data type (float32, int8, etc.)
- `DEVICE_AND_TYPE_DISPATCH(device, type, code)`: Combined dispatch
- `DEVICE_AND_FLOAT_DISPATCH(device, type, code)`: For float types only

## Testing Patterns

### C++ Tests

- Uses **Google Test** framework
- Tests located in `tests/` directory
- Test data in `tests/data/`
- Use helper macros from `tests/test_utils.h`:
  - `expect_storage_eq()` for comparing StorageViews
  - `ASSERT_RAISES()` for exception testing
  - `expect_vector_eq()` for vector comparison

### Python Tests

- Uses **pytest**
- Tests in `python/tests/`
- Shared test utilities in `test_utils.py`
- Test fixtures in `conftest.py`
- Mark tests with `@pytest.mark.parametrize` for parameterized tests
- Test data uses models from `tests/data/models/`

**Test data requirements**: Tests may need to download models (see CI workflow for examples).

## Build Configuration Options

### CMake Options

Key options from `CMakeLists.txt`:

- `WITH_MKL=ON/OFF` (default ON): Intel MKL backend
- `WITH_DNNL=ON/OFF` (default OFF): Intel oneDNN backend
- `WITH_ACCELERATE=ON/OFF` (default OFF): Apple Accelerate backend
- `WITH_OPENBLAS=ON/OFF` (default OFF): OpenBLAS backend
- `WITH_RUY=ON/OFF` (default OFF): Google Ruy backend
- `WITH_CUDA=ON/OFF` (default OFF): CUDA support
- `WITH_CUDNN=ON/OFF` (default OFF): cuDNN support
- `WITH_HIP=ON/OFF` (default OFF): AMD ROCm/HIP support
- `ENABLE_CPU_DISPATCH=ON/OFF` (default ON): Compile for multiple CPU ISAs
- `ENABLE_PROFILING=ON/OFF` (default OFF): Enable profiling
- `BUILD_TESTS=ON/OFF` (default OFF): Build C++ tests
- `BUILD_SHARED_LIBS=ON/OFF` (default ON): Build shared libraries

### Platform-Specific Notes

**x86_64**: With `ENABLE_CPU_DISPATCH=ON`, compiles kernels for AVX, AVX2, AVX512 and dispatches at runtime

**ARM64**: Compiles NEON kernels when enabled

**Windows**: MSVC-specific flags apply; static builds use different runtime linking

## Important Gotchas

### Performance-Critical Code

This is **not** a typical ML framework:

- Every memory allocation and pointer access matters
- LLM-generated code without deep understanding will be declined (see CONTRIBUTING.md)
- Contributors must understand their code's purpose and performance impact
- Focus on cache locality, memory reuse, and vectorization

### Memory Management

- Use caching allocators (CPU and GPU) to reuse buffers
- Avoid dynamic allocations in hot paths
- Prefer `StorageView::resize()` to allocation/deallocation cycles
- Move semantics preferred over copying

### Multi-Backend Support

Code must work across all backends:

- Avoid backend-specific optimizations that break other backends
- Test with at least two backends (e.g., MKL and OpenBLAS)
- Backend detection is at runtime for CPU dispatch builds

### GPU Code

- CUDA code uses `.cu` extension
- HIP (ROCm) code shares CUDA implementation with `set_source_files_properties`
- Separate GPU implementations for ops: `*_gpu.cu` files
- Some ops have cuDNN-specific versions: `*_cudnn_gpu.cu`

### Testing Without Models

Many tests require model files in `tests/data/models/`:

- Small test models are downloaded in CI workflow
- For local testing, download models manually or skip dependent tests
- See `.github/workflows/ci.yml` for download commands

### Python Bindings

- Bindings in `python/cpp/*.cc` use pybind11
- C++ library must be built first (or use `CTRANSLATE2_ROOT`)
- On macOS, minimum deployment target is 10.14 (for std::visit)

## Version Management

- Version defined in `python/ctranslate2/version.py`
- Format: `__version__ = "X.Y.Z"`
- Release process documented in `CONTRIBUTING.md`

## Performance Measurement

### Throughput

Use `--log_throughput` CLI flag:

```bash
ct2-translator --model /path/to/model --log_throughput < input.txt
```

Reports tokens generated per second (higher is better).

### Profiling

Use `--log_profiling` CLI flag to get execution profile:

```bash
ct2-translator --model /path/to/model --log_profiling < input.txt
```

Output format:
```
%self  %total  %cum  function_name  time_ms
```

- %self: Time in function excluding callees
- %total: Time in function including callees
- %cum: Cumulative time percentage so far
- time_ms: Absolute time in milliseconds

## Common Development Tasks

### Adding a New Model

1. Create model spec in `python/ctranslate2/specs/`
2. Create C++ model class in `src/models/` and `include/ctranslate2/models/`
3. Add converter in `python/ctranslate2/converters/`
4. Add tests in `python/tests/test_<framework>.py`
5. Register converter entry point in `python/setup.py`

### Adding a New Op

1. Create header: `include/ctranslate2/ops/my_op.h`
2. Create dispatcher: `src/ops/my_op.cc`
3. Create CPU impl: `src/ops/my_op_cpu.cc`
4. Create GPU impl: `src/ops/my_op_gpu.cu`
5. Update `CMakeLists.txt` to include new source files
6. Add tests to `tests/ops_test.cc` or create new test file
7. Add Python bindings if op needs Python exposure

### Updating Dependencies

- **oneDNN/MKL**: Update version in CI workflow and test across platforms
- **CUDA**: Ensure CUDA 11+ compatibility; Python wheels support CUDA 12.x
- **Python packages**: Update in `python/setup.py` and `python/install_requirements.txt`

Search commit history for examples of dependency updates.

## Style and Quality Requirements

### C++

- Wall/Wextra warnings enabled
- MSVC uses `/W4` with `/d2FH4-` (no flow analysis)
- AddressSanitizer testing in CI for Debug builds
- No exceptions in hot paths (use error codes or assertions)

### Python

- **Black** (22.x): Auto-formatting
- **isort** (5.x): Import sorting
- **flake8** (3.8.x): Linting
- All three must pass before PR merge

## CI Pipeline

The CI pipeline (`.github/workflows/ci.yml`) runs:

1. C++ tests with MKL backend
2. C++ tests with DNNL backend
3. C++ tests with AddressSanitizer (Debug build)
4. C++ tests on ARM64 (OpenBLAS + Ruy on Ubuntu, Ruy on macOS)
5. Python wheel builds (manylinux, macOS, Windows) via cibuildwheel
6. Python tests from built wheels
7. Python style checks (black, isort, flake8)
8. Docker builds (CUDA and ROCm)
9. Documentation build and deployment

**Important**: CI builds wheels with `cibuildwheel` which handles cross-platform dependencies.

## Documentation

- Docs in `docs/` using Sphinx
- Generated with `python generate.py python` then `sphinx-build`
- Guides for specific frameworks in `docs/guides/`
- API docs auto-generated from docstrings

## Contributing Guidelines Summary

From `CONTRIBUTING.md`:

1. **Deep understanding required**: Must understand code and justify design choices
2. **AI tool disclosure**: Explicitly state AI assistance; AI-generated code without review is declined
3. **Contribute within expertise**: Focus on documentation, examples if unfamiliar with core codebase
4. **Performance testing**: Ensure changes don't negatively impact performance
5. **Search TODO comments**: Look for `TODO` in codebase for small tasks
6. **Use the forum**: Ask questions on https://forum.opennmt.net tagged with `ctranslate2`
