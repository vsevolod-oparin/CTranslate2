// Minimal test: compile kSdpaMSL and report any errors.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Include the MSL string
#include "metal/msl_strings.h"

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "No Metal device\n");
      return 1;
    }
    printf("Device: %s\n", [[device name] UTF8String]);

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (@available(macOS 14.0, *)) {
      opts.languageVersion = MTLLanguageVersion3_1;
      printf("Using MSL 3.1\n");
    } else {
      printf("Using default MSL version (pre-macOS 14)\n");
    }

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kSdpaMSL];
    printf("MSL source length: %lu chars\n", (unsigned long)[src length]);

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    [opts release];

    if (lib == nil) {
      fprintf(stderr, "COMPILE FAILED: %s\n",
              [[err localizedDescription] UTF8String]);
      return 1;
    }

    printf("Library compiled successfully!\n");
    NSArray<NSString*>* names = [lib functionNames];
    printf("Functions (%lu):\n", (unsigned long)[names count]);
    for (NSString* name in names) {
      printf("  - %s\n", [name UTF8String]);
    }

    // Try to create PSO for fused_sdpa_decode_half
    id<MTLFunction> fn = [lib newFunctionWithName:@"fused_sdpa_decode_half"];
    if (!fn) {
      fprintf(stderr, "Function 'fused_sdpa_decode_half' NOT FOUND in library\n");
      [lib release];
      return 1;
    }

    NSError* psoErr = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&psoErr];
    [fn release];
    if (!pso) {
      fprintf(stderr, "PSO creation FAILED: %s\n",
              [[psoErr localizedDescription] UTF8String]);
      [lib release];
      return 1;
    }

    printf("PSO for fused_sdpa_decode_half created successfully!\n");
    printf("  maxTotalThreadsPerThreadgroup: %lu\n",
           (unsigned long)[pso maxTotalThreadsPerThreadgroup]);
    printf("  threadExecutionWidth: %lu\n",
           (unsigned long)[pso threadExecutionWidth]);

    [pso release];
    [lib release];
    return 0;
  }
}
