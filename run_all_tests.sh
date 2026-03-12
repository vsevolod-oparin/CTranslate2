#!/bin/bash
cd /Users/smileijp/projects/branch/CTranslate2

BASE_SRCS="src/metal/device.mm src/metal/utils.mm src/metal/allocator.mm \
src/metal/primitives_memory.mm src/metal/primitives_elementwise.mm \
src/metal/primitives_reduction.mm src/metal/primitives_gemm.mm \
src/metal/primitives_transpose.mm src/metal/primitives_beam_search.mm \
src/metal/ops_norm_gather.mm \
src/allocator.cc src/devices.cc src/cpu/allocator.cc"

FRAMEWORKS="-framework Metal -framework Foundation \
-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph"

COMMON="clang++ -std=c++17 -O0 -I include -I src -DCT2_WITH_METAL"

# Extra sources that tests may need
SDPA="src/metal/ops_sdpa.mm"
ROTARY="src/metal/ops_rotary.mm"
ALIBI="src/metal/ops_alibi.mm"
TOPK="src/metal/ops_topk.mm"
QUANTIZE="src/metal/ops_quantize.mm"
FUSED="src/metal/ops_fused_norm_gemm.mm"
CONV1D="src/metal/ops_conv1d.mm"

compile_and_run() {
    local test_file="$1"
    local name=$(basename "$test_file" .mm)
    local extra="$2"
    local outfile="/tmp/ct2_test_${name}"

    echo "=== COMPILING: $name ==="
    local cmd="$COMMON $test_file $BASE_SRCS $extra $FRAMEWORKS -o $outfile"

    local compile_out
    compile_out=$(eval $cmd 2>&1)
    local compile_rc=$?

    if [ $compile_rc -ne 0 ]; then
        echo "COMPILE_FAIL: $name"
        echo "$compile_out" | tail -20
        return 1
    fi

    echo "=== RUNNING: $name ==="
    local run_out
    run_out=$(timeout 120 $outfile 2>&1)
    local run_rc=$?

    if [ $run_rc -ne 0 ]; then
        echo "RUN_FAIL: $name (exit code $run_rc)"
        echo "$run_out" | tail -30
        return 2
    fi

    echo "PASS: $name"
    echo "$run_out" | tail -5
    return 0
}

# Results file
RESULTS="/tmp/ct2_test_results.txt"
> "$RESULTS"

run_test() {
    local test_file="$1"
    local extra="$2"
    local name=$(basename "$test_file" .mm)
    local outfile="/tmp/ct2_test_${name}"

    # Compile
    local cmd="$COMMON $test_file $BASE_SRCS $extra $FRAMEWORKS -o $outfile"
    local compile_out
    compile_out=$(eval $cmd 2>&1)
    local compile_rc=$?

    if [ $compile_rc -ne 0 ]; then
        echo "$name|COMPILE_FAIL|$(echo "$compile_out" | grep -E "error:" | head -3 | tr '\n' ' ')" >> "$RESULTS"
        return
    fi

    # Run
    local run_out
    run_out=$(timeout 120 $outfile 2>&1)
    local run_rc=$?

    if [ $run_rc -ne 0 ]; then
        echo "$name|RUN_FAIL|exit=$run_rc $(echo "$run_out" | tail -5 | tr '\n' ' ')" >> "$RESULTS"
        return
    fi

    # Check for test failures in output
    if echo "$run_out" | grep -qi "FAIL\|failed\|assertion\|abort"; then
        echo "$name|TEST_FAIL|$(echo "$run_out" | grep -iE "FAIL|failed|assertion" | head -3 | tr '\n' ' ')" >> "$RESULTS"
    else
        echo "$name|PASS|$(echo "$run_out" | tail -3 | tr '\n' ' ')" >> "$RESULTS"
    fi
}

echo "Starting all tests..."

# Simple tests (base sources only)
for t in activation_test allocator_test arithmetic_test beam_search_test broadcast_test \
         context_test convert_test dispatch_test large_transpose_test minmax_test \
         primitives_test pso_warmup_test reduce_sum_precision_test reduction_test \
         storage_view_test sync_scoped_test transpose_test truncation_test; do
    run_test "tests/metal/${t}.mm" ""
done

# Tests needing SDPA
for t in sdpa_test flash_mha_decode_test kv_cache_test; do
    run_test "tests/metal/${t}.mm" "$SDPA"
done

# Tests needing rotary
for t in rotary_test decode_rope_test gpu_decode_rope_test; do
    run_test "tests/metal/${t}.mm" "$ROTARY"
done

# Tests needing alibi
run_test "tests/metal/alibi_test.mm" "$ALIBI"

# Tests needing topk
run_test "tests/metal/topk_test.mm" "$TOPK"

# Tests needing fused norm gemm
run_test "tests/metal/fused_norm_gemm_test.mm" "$FUSED"

# Normalization tests
for t in normalization_comparison_test normalization_gather_test; do
    run_test "tests/metal/${t}.mm" ""
done

# Bias add test
run_test "tests/metal/bias_add_test.mm" ""

# Bugfix test - may need various extras
run_test "tests/metal/bugfix_test.mm" ""

# Gemm test
run_test "tests/metal/gemm_test.mm" ""

# Milestone tests - try base first
for t in m7_test m81_test m82_test m83_test m91_test m92_test m11_review_test m12_review_test; do
    run_test "tests/metal/${t}.mm" ""
done

echo ""
echo "============================================"
echo "RESULTS SUMMARY"
echo "============================================"
cat "$RESULTS"
