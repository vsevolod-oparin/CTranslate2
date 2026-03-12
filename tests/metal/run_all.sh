#!/usr/bin/env bash
# tests/metal/run_all.sh
#
# Build and run every Apple Metal unit test, then print an aggregate summary.
#
# Usage (from the repository root or any subdirectory):
#   bash tests/metal/run_all.sh [options]
#
# Options:
#   --clean       Delete the build directory before starting.
#   --build-only  Compile all tests but do not run them.
#   --no-rebuild  Skip compilation when the binary already exists.
#   --list        Print test names and exit (no building or running).
#
# Output:
#   Each test's output is printed in full, indented by two spaces.
#   A summary table follows showing per-test and total pass/fail counts.
#   Exit code is 0 if all tests pass, 1 otherwise.

set -uo pipefail

# ---------------------------------------------------------------------------
# Resolve paths relative to this script, regardless of working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TESTS_DIR="${SCRIPT_DIR}"
BUILD_DIR="${REPO_ROOT}/build/metal_tests"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
OPT_CLEAN=0
OPT_BUILD_ONLY=0
OPT_NO_REBUILD=0
OPT_LIST=0

for _arg in "$@"; do
  case "${_arg}" in
    --clean)       OPT_CLEAN=1 ;;
    --build-only)  OPT_BUILD_ONLY=1 ;;
    --no-rebuild)  OPT_NO_REBUILD=1 ;;
    --list)        OPT_LIST=1 ;;
    -*)
      printf 'run_all.sh: unknown option: %s\n' "${_arg}" >&2
      printf 'Usage: bash tests/metal/run_all.sh [--clean] [--build-only] [--no-rebuild] [--list]\n' >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Source-file groups (space-separated paths relative to REPO_ROOT).
# Kept on single lines to avoid embedded newlines or backslash continuations.
# ---------------------------------------------------------------------------

# Full stack: context + allocator + primitives + ops + C++ allocator/device glue
# utils.mm → allocator.mm → primitives_gemm.mm → ops_sdpa.mm dependency chain
# means even "minimal" tests now need the full source set.
PRIM_SRCS="src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm"
OPS_SRCS="src/metal/ops_norm_gather.mm src/metal/ops_sdpa.mm"
FULL_SRCS="src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm ${PRIM_SRCS} ${OPS_SRCS} src/allocator.cc src/devices.cc src/cpu/allocator.cc"

# ---------------------------------------------------------------------------
# Compiler / linker flags
# ---------------------------------------------------------------------------
CFLAGS_COMMON="-std=c++17 -I${REPO_ROOT}/include -I${REPO_ROOT}/src -DCT2_WITH_MPS"

# Base Metal frameworks (required by all tests)
FW_BASE="-framework Metal -framework Foundation -framework MetalPerformanceShaders"

# Full Metal frameworks (required by tests that use primitives_gemm.mm, which
# includes MPSGraph for the BF16 GEMM path + Accelerate for CBLAS)
FW_GRAPH="${FW_BASE} -framework MetalPerformanceShadersGraph -framework Accelerate"

# ---------------------------------------------------------------------------
# Test registry.
#
# Format of each entry (four pipe-delimited fields):
#   stem | opt_flags | source_list | framework_flags
#
# stem         : filename without .mm extension; also used as binary name
# opt_flags    : optimisation flag (-O0 or -O2)
# source_list  : space-separated relative paths (expanded at array creation)
# framework_flags : -framework ... flags (expanded at array creation)
#
# Ordering: context → allocator → storage → primitives → arithmetic
#           → reduction → gemm → activation → broadcast → dispatch (M5.1)
#           → beam_search → transpose → convert → truncation → minmax
#           → pso_warmup → large_transpose → reduce_sum_precision
#           → normalization_gather (M5.2)
#           → normalization_comparison → bias_add (M5.2 review 4.5, 4.6)
# ---------------------------------------------------------------------------
TESTS=(
  "context_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "sync_scoped_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "allocator_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "storage_view_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "primitives_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "arithmetic_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "reduction_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "gemm_test|-O2|${FULL_SRCS}|${FW_GRAPH}"
  "activation_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "broadcast_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "dispatch_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "beam_search_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "transpose_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "convert_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "truncation_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "minmax_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "pso_warmup_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "large_transpose_test|-O2|${FULL_SRCS}|${FW_GRAPH}"
  "reduce_sum_precision_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "normalization_gather_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "normalization_comparison_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
  "bias_add_test|-O0|${FULL_SRCS}|${FW_GRAPH}"
)
TOTAL=${#TESTS[@]}

# ---------------------------------------------------------------------------
# --list: print test names and exit
# ---------------------------------------------------------------------------
if [ "${OPT_LIST}" -eq 1 ]; then
  for _spec in "${TESTS[@]}"; do
    IFS='|' read -r _stem _ _ _ <<< "${_spec}"
    printf '%s\n' "${_stem}"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if [ "${OPT_CLEAN}" -eq 1 ]; then
  printf 'Removing %s\n' "${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# build_one  stem  opt_flags  src_list  fw_flags
#
# Compiles tests/metal/<stem>.mm plus the given sources into BUILD_DIR/<stem>.
# Returns 0 on success, 1 on failure.
# Compiler output is suppressed on success; printed to stderr on failure.
# With --no-rebuild, skips compilation when the binary already exists.
# ---------------------------------------------------------------------------
build_one() {
  local stem="${1}" opt_flags="${2}" src_list="${3}" fw_flags="${4}"
  local binary="${BUILD_DIR}/${stem}"

  if [ "${OPT_NO_REBUILD}" -eq 1 ] && [ -x "${binary}" ]; then
    return 0
  fi

  # Convert space-separated relative paths to absolute paths.
  local abs_srcs=""
  local _s
  for _s in ${src_list}; do   # intentional unquoted word split
    abs_srcs="${abs_srcs} ${REPO_ROOT}/${_s}"
  done

  local build_log
  # shellcheck disable=SC2086  # word splitting on opt_flags / abs_srcs / fw_flags is intentional
  if build_log=$(
      clang++ ${CFLAGS_COMMON} ${opt_flags} \
        "${TESTS_DIR}/${stem}.mm" \
        ${abs_srcs} \
        ${fw_flags} \
        -o "${binary}" 2>&1
  ); then
    return 0
  else
    printf '\n--- BUILD FAILED: %s ---\n' "${stem}" >&2
    printf '%s\n' "${build_log}" >&2
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------
printf '=== Metal Test Suite (%d tests) ===\n' "${TOTAL}"
printf 'Repository : %s\n' "${REPO_ROOT}"
printf 'Build dir  : %s\n\n' "${BUILD_DIR}"
printf 'Building...\n'

build_failed=""   # space-separated list of stems that failed to build
_i=0
for _spec in "${TESTS[@]}"; do
  _i=$((_i + 1))
  IFS='|' read -r _stem _opt _srcs _fws <<< "${_spec}"
  printf '  [%2d/%2d] %-36s' "${_i}" "${TOTAL}" "${_stem}"
  if build_one "${_stem}" "${_opt}" "${_srcs}" "${_fws}"; then
    printf 'OK\n'
  else
    printf 'FAILED\n'
    build_failed="${build_failed} ${_stem}"
  fi
done

if [ -n "${build_failed}" ]; then
  printf '\nBuild failures:%s\n' "${build_failed}"
fi

if [ "${OPT_BUILD_ONLY}" -eq 1 ]; then
  printf '\n--build-only: binaries are in %s\n' "${BUILD_DIR}"
  [ -z "${build_failed}" ] && exit 0 || exit 1
fi

# ---------------------------------------------------------------------------
# Run phase
# ---------------------------------------------------------------------------
printf '\nRunning tests...\n'

total_pass=0
total_fail=0

# Parallel arrays to accumulate per-test results for the summary table.
RESULT_STEMS=()
RESULT_PASS=()
RESULT_FAIL=()
RESULT_STATUS=()

_i=0
for _spec in "${TESTS[@]}"; do
  _i=$((_i + 1))
  IFS='|' read -r _stem _ _ _ <<< "${_spec}"
  binary="${BUILD_DIR}/${_stem}"

  printf '\n[%2d/%2d] %s\n' "${_i}" "${TOTAL}" "${_stem}"

  # --- binary did not build -----------------------------------------------
  if [ ! -x "${binary}" ]; then
    printf '  SKIPPED (binary not built)\n'
    RESULT_STEMS+=("${_stem}")
    RESULT_PASS+=("0")
    RESULT_FAIL+=("1")
    RESULT_STATUS+=("BUILD-FAIL")
    total_fail=$((total_fail + 1))
    continue
  fi

  # --- run the binary and capture output + exit code ----------------------
  run_exit=0
  test_output=$("${binary}" 2>&1) || run_exit=$?

  # Print indented test output.
  printf '%s\n' "${test_output}" | sed 's/^/  /'

  # --- parse "N passed, M failed" (or "=== Results: N passed, M failed ===")
  # Strategy: find the summary line, extract the first two numbers from it.
  summary_line=$(printf '%s\n' "${test_output}" | grep -E 'passed.*failed' | tail -1)

  t_pass=""
  t_fail=""
  if [ -n "${summary_line}" ]; then
    # Extract all integers from the line; first = passed, last = failed.
    t_pass=$(printf '%s' "${summary_line}" | grep -oE '[0-9]+' | head -1)
    t_fail=$(printf '%s' "${summary_line}" | grep -oE '[0-9]+' | tail -1)
  fi

  # Validate numeric; fall back to 0.
  case "${t_pass}" in ''|*[!0-9]*) t_pass=0 ;; esac
  case "${t_fail}" in ''|*[!0-9]*) t_fail=0 ;; esac

  # If the process crashed (non-zero exit) and we couldn't parse a fail count,
  # count it as one failure so the crash is visible in the summary.
  if [ "${run_exit}" -ne 0 ] && [ "${t_fail}" -eq 0 ]; then
    t_fail=1
  fi

  total_pass=$((total_pass + t_pass))
  total_fail=$((total_fail + t_fail))

  status="pass"
  if [ "${t_fail}" -gt 0 ] || [ "${run_exit}" -ne 0 ]; then
    status="FAIL"
  fi

  RESULT_STEMS+=("${_stem}")
  RESULT_PASS+=("${t_pass}")
  RESULT_FAIL+=("${t_fail}")
  RESULT_STATUS+=("${status}")
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
printf '\n'
printf '%s\n' "======================================================="
printf '%-36s  %6s  %6s  %s\n' "Test" "Passed" "Failed" "Result"
printf '%s\n' "-------------------------------------------------------"

_n=${#RESULT_STEMS[@]}
_idx=0
while [ "${_idx}" -lt "${_n}" ]; do
  printf '%-36s  %6s  %6s  %s\n' \
    "${RESULT_STEMS[${_idx}]}" \
    "${RESULT_PASS[${_idx}]}"  \
    "${RESULT_FAIL[${_idx}]}"  \
    "${RESULT_STATUS[${_idx}]}"
  _idx=$((_idx + 1))
done

printf '%s\n' "-------------------------------------------------------"
printf '%-36s  %6d  %6d\n' "TOTAL" "${total_pass}" "${total_fail}"
printf '%s\n' "======================================================="

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
if [ "${total_fail}" -eq 0 ] && [ -z "${build_failed}" ]; then
  printf '\nRESULT: ALL PASS  (%d tests, %d assertions passed)\n' \
    "${TOTAL}" "${total_pass}"
  exit 0
else
  printf '\nRESULT: FAILURES DETECTED  (%d passed, %d failed)\n' \
    "${total_pass}" "${total_fail}"
  exit 1
fi
