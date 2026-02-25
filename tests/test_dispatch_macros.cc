// Compile-only test for dispatch macros (M1.1 verification).
// Tests all four combinations of CT2_WITH_CUDA / CT2_WITH_METAL.
//
// Run from the repository root:
//   while IFS='|' read -r label flags; do
//     printf "%-20s " "$label:";
//     clang++ -std=c++17 -fsyntax-only -I include -I src $flags tests/test_dispatch_macros.cc \
//       2>&1 && echo "OK" || echo "FAIL";
//   done <<'EOF'
//   CPU-only            |
//   Metal-only          |-DCT2_WITH_METAL
//   CUDA-only           |-DCT2_WITH_CUDA
//   CUDA+Metal          |-DCT2_WITH_CUDA -DCT2_WITH_METAL
//   EOF

#include <stdexcept>
#include "ctranslate2/devices.h"
#include "device_dispatch.h"
#include "dispatch.h"
#include "type_dispatch.h"

using namespace ctranslate2;

// Helper templates to verify D and T are correct types.
template <Device D>
void check_device() { (void)D; }

template <Device D, typename T>
void check_device_and_type() { (void)D; (void)(T*)nullptr; }

// Test 1: DEVICE_DISPATCH sets constexpr Device D correctly.
void test_device_dispatch(Device d) {
  DEVICE_DISPATCH(d, check_device<D>());
}

// Test 2: DEVICE_AND_FLOAT_DISPATCH sets D and typedef T correctly.
// The outer parens around the call protect the D,T comma from the preprocessor.
void test_float_dispatch(Device d, DataType t) {
  DEVICE_AND_FLOAT_DISPATCH("test", d, t,
    (check_device_and_type<D, T>()));
}
