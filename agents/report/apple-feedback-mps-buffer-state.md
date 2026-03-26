# Apple Feedback Assistant — Draft

**Product:** macOS
**Area:** Metal / Metal Performance Shaders
**Type:** Incorrect/Unexpected Behavior

---

## Title

MPSMatrixMultiplication produces incorrect results when an MTLBuffer is reused with different MPSMatrixDescriptor dimensions

## Description

`MPSMatrixMultiplication` produces incorrect (all-zero) results when encoding a GEMM where one of the `MPSMatrix` arguments wraps an `MTLBuffer` that was previously used in a different `MPSMatrixMultiplication` encoding with different matrix dimensions.

The issue appears to be that MPS caches internal optimization state per `MTLBuffer` object. When the same buffer is reused at different dimensions (different `rows`, `columns`, or `rowBytes` in the `MPSMatrixDescriptor`), the cached state becomes invalid and corrupts the computation.

**Key observations:**
- Zeroing the buffer contents does NOT fix the issue — the problem is per-object MPS state, not data contents
- Releasing the `MTLBuffer` and allocating a fresh one DOES fix the issue — new buffer object has no cached state
- The corruption is dimension-specific: it depends on the relationship between old and new matrix dimensions (alignment/tiling boundaries)
- Only affects certain dimension transitions (e.g., odd column counts after large even counts)

**Environment:**
- Apple M4 (MacBook Pro)
- macOS 15.x (Sequoia)
- Metal Performance Shaders framework
- `MTLResourceStorageModeShared` buffers

## Steps to Reproduce

Minimal Objective-C++ reproduction (single file, no dependencies beyond Metal/MPS frameworks):

```objc
// mps_buffer_reuse_bug.mm
// Build: clang++ -std=c++17 -framework Metal -framework Foundation \
//        -framework MetalPerformanceShaders -o mps_bug mps_buffer_reuse_bug.mm
// Run:   ./mps_bug

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Step 1: Allocate a large shared buffer (simulates pool allocator)
        const NSUInteger large_size = 4 * 1024 * 1024;  // 4 MB
        id<MTLBuffer> buf_c = [device newBufferWithLength:large_size
                                                  options:MTLResourceStorageModeShared];

        // Step 2: Use buf_c as output for a LARGE GEMM (m=256, n=512)
        {
            const NSUInteger m = 256, n = 512, k = 128;
            id<MTLBuffer> buf_a = [device newBufferWithLength:m * k * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_b = [device newBufferWithLength:k * n * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

            // Fill A and B with 1.0
            float* a = (float*)[buf_a contents];
            float* b = (float*)[buf_b contents];
            for (NSUInteger i = 0; i < m * k; ++i) a[i] = 1.0f;
            for (NSUInteger i = 0; i < k * n; ++i) b[i] = 1.0f;

            NSUInteger rb_a = [MPSMatrixDescriptor rowBytesForColumns:k dataType:MPSDataTypeFloat32];
            NSUInteger rb_b = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];
            NSUInteger rb_c = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];

            // Use natural stride if it meets MPS minimum
            rb_a = (k * sizeof(float) >= rb_a) ? k * sizeof(float) : rb_a;
            rb_b = (n * sizeof(float) >= rb_b) ? n * sizeof(float) : rb_b;
            rb_c = (n * sizeof(float) >= rb_c) ? n * sizeof(float) : rb_c;

            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:rb_a dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:rb_b dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:rb_c dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:0 descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:0 descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:0 descriptor:descC];

            MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:NO
                resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            [gemm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmd commit];
            [cmd waitUntilCompleted];

            // Verify: C[0][0] should be k = 128
            float* c = (float*)[buf_c contents];
            std::printf("Large GEMM: C[0][0] = %.1f (expected %.1f)\n", c[0], (float)k);

            [matA release]; [matB release]; [matC release];
            [gemm release];
            [buf_a release]; [buf_b release];
            // buf_c is NOT released — simulates pool returning it
        }

        // Step 3: Reuse buf_c as output for a SMALL GEMM (m=8, n=103)
        // This is the dimension that triggers the bug in practice.
        {
            const NSUInteger m = 8, n = 103, k = 128;
            id<MTLBuffer> buf_a = [device newBufferWithLength:m * k * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_b = [device newBufferWithLength:k * n * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

            float* a = (float*)[buf_a contents];
            float* b = (float*)[buf_b contents];
            for (NSUInteger i = 0; i < m * k; ++i) a[i] = 1.0f;
            for (NSUInteger i = 0; i < k * n; ++i) b[i] = 1.0f;

            // Zero buf_c completely (proves the issue is not stale data)
            memset([buf_c contents], 0, large_size);

            NSUInteger rb_a = [MPSMatrixDescriptor rowBytesForColumns:k dataType:MPSDataTypeFloat32];
            NSUInteger rb_b = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];
            NSUInteger rb_c = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];

            rb_a = (k * sizeof(float) >= rb_a) ? k * sizeof(float) : rb_a;
            rb_b = (n * sizeof(float) >= rb_b) ? n * sizeof(float) : rb_b;
            rb_c = (n * sizeof(float) >= rb_c) ? n * sizeof(float) : rb_c;

            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:rb_a dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:rb_b dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:rb_c dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:0 descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:0 descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c offset:0 descriptor:descC];

            // New MPSMatrixMultiplication object (different dimensions)
            MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:NO
                resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            [gemm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmd commit];
            [cmd waitUntilCompleted];

            float* c = (float*)[buf_c contents];
            float expected = (float)k;  // 128.0
            bool all_zero = true;
            float max_err = 0;
            for (NSUInteger i = 0; i < m; ++i) {
                // Row i, column 0: C[i][0] should be k
                NSUInteger idx = i * (rb_c / sizeof(float));
                if (c[idx] != 0) all_zero = false;
                float err = std::fabs(c[idx] - expected);
                if (err > max_err) max_err = err;
            }

            std::printf("Small GEMM (reused buf_c): C[0][0] = %.1f (expected %.1f) max_err=%.2e %s\n",
                        c[0], expected, max_err,
                        all_zero ? "*** ALL ZERO — BUG ***" : "OK");

            [matA release]; [matB release]; [matC release];
            [gemm release]; [buf_a release]; [buf_b release];
        }

        // Step 4: Same small GEMM but with a FRESH buf_c (proves the fix)
        {
            id<MTLBuffer> fresh_buf_c = [device newBufferWithLength:large_size
                                                            options:MTLResourceStorageModeShared];
            const NSUInteger m = 8, n = 103, k = 128;
            id<MTLBuffer> buf_a = [device newBufferWithLength:m * k * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_b = [device newBufferWithLength:k * n * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

            float* a = (float*)[buf_a contents];
            float* b = (float*)[buf_b contents];
            for (NSUInteger i = 0; i < m * k; ++i) a[i] = 1.0f;
            for (NSUInteger i = 0; i < k * n; ++i) b[i] = 1.0f;

            NSUInteger rb_c = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];
            NSUInteger rb_a = [MPSMatrixDescriptor rowBytesForColumns:k dataType:MPSDataTypeFloat32];
            NSUInteger rb_b = [MPSMatrixDescriptor rowBytesForColumns:n dataType:MPSDataTypeFloat32];
            rb_a = (k * sizeof(float) >= rb_a) ? k * sizeof(float) : rb_a;
            rb_b = (n * sizeof(float) >= rb_b) ? n * sizeof(float) : rb_b;
            rb_c = (n * sizeof(float) >= rb_c) ? n * sizeof(float) : rb_c;

            MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:rb_a dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:rb_b dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:rb_c dataType:MPSDataTypeFloat32];

            MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a offset:0 descriptor:descA];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b offset:0 descriptor:descB];
            MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:fresh_buf_c offset:0 descriptor:descC];

            MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
                initWithDevice:device transposeLeft:NO transposeRight:NO
                resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            [gemm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
            [cmd commit];
            [cmd waitUntilCompleted];

            float* c = (float*)[fresh_buf_c contents];
            std::printf("Small GEMM (fresh buf_c): C[0][0] = %.1f (expected %.1f)\n",
                        c[0], (float)k);

            [matA release]; [matB release]; [matC release];
            [gemm release]; [buf_a release]; [buf_b release];
            [fresh_buf_c release];
        }

        [buf_c release];
        [queue release];
    }
    return 0;
}
```

## Expected Results

All three GEMMs should produce C[0][0] = 128.0 (since A is all-ones, B is all-ones, k=128).

## Actual Results

The second GEMM (reused buf_c at different dimensions) produces all-zero output, despite:
- The buffer being zeroed before the GEMM
- A new `MPSMatrixMultiplication` object being created
- Correct `MPSMatrixDescriptor` dimensions

The third GEMM (fresh buffer, same dimensions) produces the correct result.

## Impact

This affects any application that uses a buffer pool (allocate once, reuse across calls) with MPS matrix operations at varying dimensions — a standard pattern in ML inference frameworks. The workaround is to never reuse MTLBuffers across MPS GEMM calls with different dimensions, which eliminates the performance benefit of buffer pooling (~3-5% regression).

## Workaround

Release and reallocate MTLBuffers instead of reusing them from a pool when the buffer will be used with `MPSMatrixMultiplication` at different dimensions.
