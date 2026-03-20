#!/usr/bin/env bash
# build_mps_wheels.sh — Build ctranslate2-mps Python wheels for multiple Python versions
#
# Prerequisites: conda (or mamba), cmake, C++17 compiler (Xcode/CommandLineTools)
# Platform: macOS arm64 only
#
# Exit codes:
#   0  All wheels built successfully
#   1  General error (missing dependency, build failure, etc.)
#   2  Invalid arguments
#   3  Platform not supported

set -Eeuo pipefail
# inherit_errexit requires Bash 4.4+; skip gracefully on older versions
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )); then
  shopt -s inherit_errexit
fi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly PYTHON_DIR="$PROJECT_ROOT/python"
readonly VERSION_FILE="$PYTHON_DIR/ctranslate2/version.py"
readonly BUILD_DIR="$PROJECT_ROOT/build"
readonly INSTALL_DIR="$BUILD_DIR/install"

readonly DEFAULT_PYTHON_VERSIONS="3.9 3.10 3.11 3.12 3.13 3.14"
readonly ENV_PREFIX="ct2-build-py"
readonly BUILD_DEPS=(pybind11 setuptools wheel numpy)

# Populated during init
DYLIB_VERSION=""
DYLIB_SOVERSION=""
DYLIB_NAME=""
DYLIB_SONAME=""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log_info() {
  printf "[INFO]  %s  %s\n" "$(date +%H:%M:%S)" "$*" >&2
}

log_warn() {
  printf "[WARN]  %s  %s\n" "$(date +%H:%M:%S)" "$*" >&2
}

log_error() {
  printf "[ERROR] %s  %s\n" "$(date +%H:%M:%S)" "$*" >&2
}

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
on_error() {
  log_error "Failed at line $1 (exit code $2)"
  exit 1
}
trap 'on_error ${LINENO} $?' ERR

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<'USAGE'
Usage: build_mps_wheels.sh [OPTIONS]

Build ctranslate2-mps Python wheels for multiple Python versions on macOS arm64.

Options:
  --python-versions "X.Y ..."  Python versions to build (default: "3.9 3.10 3.11 3.12 3.13 3.14")
  --skip-cmake                 Skip C++ library build (use existing build/)
  --output-dir DIR             Wheel output directory (default: dist/)
  --build-dir DIR              CMake build directory (default: build/)
  --jobs N                     Parallel build jobs (default: auto-detect)
  --trace                      Enable bash tracing (set -x)
  -h, --help                   Show this help message

Environment variables:
  CONDA_EXE    Path to conda executable (auto-detected if not set)

Examples:
  # Build wheels for all supported Python versions
  ./tools/build_mps_wheels.sh

  # Build only for 3.11 and 3.12, skip C++ rebuild
  ./tools/build_mps_wheels.sh --skip-cmake --python-versions "3.11 3.12"

  # Custom output directory
  ./tools/build_mps_wheels.sh --output-dir /tmp/wheels
USAGE
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
PYTHON_VERSIONS="$DEFAULT_PYTHON_VERSIONS"
SKIP_CMAKE=0
OUTPUT_DIR="$PROJECT_ROOT/dist"
USER_BUILD_DIR=""
JOBS=""

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --python-versions)
        [[ $# -ge 2 ]] || { log_error "--python-versions requires an argument"; exit 2; }
        PYTHON_VERSIONS="$2"
        shift 2
        ;;
      --skip-cmake)
        SKIP_CMAKE=1
        shift
        ;;
      --output-dir)
        [[ $# -ge 2 ]] || { log_error "--output-dir requires an argument"; exit 2; }
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --build-dir)
        [[ $# -ge 2 ]] || { log_error "--build-dir requires an argument"; exit 2; }
        USER_BUILD_DIR="$2"
        shift 2
        ;;
      --jobs)
        [[ $# -ge 2 ]] || { log_error "--jobs requires an argument"; exit 2; }
        JOBS="$2"
        shift 2
        ;;
      --trace)
        set -x
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        log_error "Unknown option: $1"
        usage >&2
        exit 2
        ;;
      *)
        log_error "Unexpected argument: $1"
        usage >&2
        exit 2
        ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
validate_platform() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    log_error "This script is macOS-only (Metal/MPS backend)"
    exit 3
  fi
  if [[ "$(uname -m)" != "arm64" ]]; then
    log_warn "Running on $(uname -m); Metal backend targets arm64 (Apple Silicon)"
  fi
}

detect_conda() {
  if [[ -n "${CONDA_EXE:-}" ]] && [[ -x "$CONDA_EXE" ]]; then
    return 0
  fi
  if command -v conda &>/dev/null; then
    CONDA_EXE="$(command -v conda)"
    return 0
  fi
  if command -v mamba &>/dev/null; then
    CONDA_EXE="$(command -v mamba)"
    return 0
  fi
  log_error "conda or mamba not found. Install Miniforge or Anaconda first."
  exit 1
}

detect_version() {
  if [[ ! -f "$VERSION_FILE" ]]; then
    log_error "Version file not found: $VERSION_FILE"
    exit 1
  fi

  # Extract __version__ = "X.Y.Z" from version.py
  local raw_version
  raw_version="$(grep -oE '__version__\s*=\s*"[^"]+"' "$VERSION_FILE" | grep -oE '"[^"]+"' | tr -d '"')"

  if [[ -z "$raw_version" ]]; then
    log_error "Could not parse version from $VERSION_FILE"
    exit 1
  fi

  DYLIB_VERSION="$raw_version"
  # SOVERSION is the major version component
  DYLIB_SOVERSION="${DYLIB_VERSION%%.*}"
  DYLIB_NAME="libctranslate2.mps.${DYLIB_VERSION}.dylib"
  DYLIB_SONAME="libctranslate2.mps.${DYLIB_SOVERSION}.dylib"

  log_info "Detected version: $DYLIB_VERSION (soversion: $DYLIB_SOVERSION)"
  log_info "Dylib: $DYLIB_NAME  Soname link: $DYLIB_SONAME"
}

validate_prerequisites() {
  local missing=()
  for cmd in cmake make; do
    if ! command -v "$cmd" &>/dev/null; then
      missing+=("$cmd")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    log_error "Missing required commands: ${missing[*]}"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Conda environment management
# ---------------------------------------------------------------------------
env_name_for_version() {
  local pyver="$1"
  # 3.10 -> ct2-build-py310
  printf '%s%s\n' "$ENV_PREFIX" "${pyver//./}"
}

ensure_conda_env() {
  local pyver="$1"
  local env_name
  env_name="$(env_name_for_version "$pyver")"

  # Check if environment already exists
  if conda info --envs 2>/dev/null | grep -qE "^${env_name}[[:space:]]"; then
    log_info "Conda env '$env_name' already exists"
  else
    log_info "Creating conda env '$env_name' (Python $pyver)..."
    conda create -y -n "$env_name" "python=${pyver}" --quiet 2>&1 | tail -1 || {
      log_error "Failed to create conda env for Python $pyver"
      return 1
    }
  fi

  # Install build dependencies if missing
  log_info "Ensuring build dependencies in '$env_name'..."
  conda run -n "$env_name" \
    pip install --quiet "${BUILD_DEPS[@]}" 2>&1 | tail -3 || {
    log_error "Failed to install build deps in '$env_name'"
    return 1
  }
}

# ---------------------------------------------------------------------------
# C++ library build
# ---------------------------------------------------------------------------
build_cpp_library() {
  local jobs="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || printf '4')}"
  local build_dir="${USER_BUILD_DIR:-$BUILD_DIR}"

  log_info "Building C++ library (jobs=$jobs, build_dir=$build_dir)..."

  if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    log_info "No existing CMake cache found; running cmake configure..."
    cmake -S "$PROJECT_ROOT" -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_METAL=ON \
      -DBUILD_CLI=OFF \
      -DCMAKE_INSTALL_PREFIX="$build_dir/install"
  fi

  cmake --build "$build_dir" --target ctranslate2 "-j${jobs}"
  cmake --install "$build_dir" --prefix "$build_dir/install"

  log_info "C++ library built and installed to $build_dir/install"
}

# ---------------------------------------------------------------------------
# Dylib staging
# ---------------------------------------------------------------------------
stage_dylib() {
  local build_dir="${USER_BUILD_DIR:-$BUILD_DIR}"
  local dest_dir="$PYTHON_DIR/ctranslate2"
  local src_dylib="$build_dir/$DYLIB_NAME"

  # Also check install dir
  if [[ ! -f "$src_dylib" ]]; then
    src_dylib="$build_dir/install/lib/$DYLIB_NAME"
  fi

  if [[ ! -f "$src_dylib" ]]; then
    log_error "Dylib not found: tried $build_dir/$DYLIB_NAME and $build_dir/install/lib/$DYLIB_NAME"
    exit 1
  fi

  log_info "Staging dylib: $src_dylib -> $dest_dir/"
  cp -- "$src_dylib" "$dest_dir/$DYLIB_NAME"
  ln -sf "$DYLIB_NAME" "$dest_dir/$DYLIB_SONAME"

  # Verify the files exist
  if [[ ! -f "$dest_dir/$DYLIB_NAME" ]]; then
    log_error "Failed to stage dylib"
    exit 1
  fi

  log_info "Dylib staged: $DYLIB_NAME + symlink $DYLIB_SONAME"
}

# ---------------------------------------------------------------------------
# Wheel building
# ---------------------------------------------------------------------------
build_wheel_for_version() {
  local pyver="$1"
  local env_name
  env_name="$(env_name_for_version "$pyver")"
  local install_prefix="${USER_BUILD_DIR:-$BUILD_DIR}/install"

  log_info "Building wheel for Python $pyver (env: $env_name)..."

  conda run -n "$env_name" --cwd "$PYTHON_DIR" \
    env CT2_MPS_BUILD=1 "CTRANSLATE2_ROOT=$install_prefix" \
    pip wheel . \
      --no-deps \
      --no-build-isolation \
      --wheel-dir "$OUTPUT_DIR" \
    2>&1 | while IFS= read -r line; do
      printf "  [py%s] %s\n" "$pyver" "$line" >&2
    done

  # Verify a wheel was created
  local pattern="ctranslate2_mps-*-cp${pyver//./}*.whl"
  local found=0
  for whl in "$OUTPUT_DIR"/$pattern; do
    if [[ -f "$whl" ]]; then
      found=1
      break
    fi
  done

  if [[ $found -eq 0 ]]; then
    log_warn "No wheel matching $pattern found in $OUTPUT_DIR (build may have failed)"
    return 1
  fi

  return 0
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
  local -a versions=("$@")
  local total=${#versions[@]}
  local success=0
  local failed=0

  printf "\n" >&2
  printf "====================================================================\n" >&2
  printf "  ctranslate2-mps Wheel Build Summary (v%s)\n" "$DYLIB_VERSION" >&2
  printf "====================================================================\n" >&2
  printf "  %-12s %-15s %s\n" "Python" "Status" "Wheel" >&2
  printf "  %-12s %-15s %s\n" "------" "------" "-----" >&2

  for pyver in "${versions[@]}"; do
    local pattern="ctranslate2_mps-*-cp${pyver//./}*.whl"
    local wheel_path=""
    for whl in "$OUTPUT_DIR"/$pattern; do
      if [[ -f "$whl" ]]; then
        wheel_path="$whl"
        break
      fi
    done

    if [[ -n "$wheel_path" ]]; then
      local wheel_name
      wheel_name="$(basename -- "$wheel_path")"
      local wheel_size
      wheel_size="$(du -h "$wheel_path" | cut -f1 | tr -d ' ')"
      printf "  %-12s %-15s %s (%s)\n" "$pyver" "OK" "$wheel_name" "$wheel_size" >&2
      (( success++ ))
    else
      printf "  %-12s %-15s %s\n" "$pyver" "FAILED" "-" >&2
      (( failed++ ))
    fi
  done

  printf "  %-12s %-15s %s\n" "------" "------" "-----" >&2
  printf "  Total: %d  Success: %d  Failed: %d\n" "$total" "$success" "$failed" >&2
  printf "  Output: %s\n" "$OUTPUT_DIR" >&2
  printf "====================================================================\n" >&2

  if [[ $failed -gt 0 ]]; then
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  parse_args "$@"
  validate_platform
  validate_prerequisites
  detect_conda
  detect_version

  # Parse version list into array
  local -a py_versions
  read -ra py_versions <<< "$PYTHON_VERSIONS"

  if [[ ${#py_versions[@]} -eq 0 ]]; then
    log_error "No Python versions specified"
    exit 2
  fi

  log_info "Python versions: ${py_versions[*]}"
  log_info "Output directory: $OUTPUT_DIR"
  log_info "Skip CMake: $SKIP_CMAKE"

  # Create output directory
  mkdir -p -- "$OUTPUT_DIR"

  # Step 1: Build C++ library
  if [[ $SKIP_CMAKE -eq 0 ]]; then
    build_cpp_library
  else
    log_info "Skipping C++ build (--skip-cmake)"
    local install_prefix="${USER_BUILD_DIR:-$BUILD_DIR}/install"
    if [[ ! -d "$install_prefix/include" ]] || [[ ! -d "$install_prefix/lib" ]]; then
      log_error "Install prefix missing (no include/ or lib/ in $install_prefix). Run without --skip-cmake first."
      exit 1
    fi
  fi

  # Step 2: Stage dylib
  stage_dylib

  # Step 3: Ensure conda envs and build wheels
  local -a succeeded=()
  local -a failed_versions=()

  for pyver in "${py_versions[@]}"; do
    if ! ensure_conda_env "$pyver"; then
      log_warn "Skipping Python $pyver (env setup failed)"
      failed_versions+=("$pyver")
      continue
    fi

    if build_wheel_for_version "$pyver"; then
      succeeded+=("$pyver")
    else
      failed_versions+=("$pyver")
    fi
  done

  # Step 4: Summary
  print_summary "${py_versions[@]}"
  local summary_rc=$?

  # Cleanup staged dylib (leave it; harmless and .gitignored)
  log_info "Done. Wheels are in $OUTPUT_DIR/"

  return $summary_rc
}

main "$@"
