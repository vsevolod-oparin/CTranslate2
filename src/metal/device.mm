#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device.h"

namespace ctranslate2 {
  namespace metal {

    int get_device_count() {
      return MTLCreateSystemDefaultDevice() != nil ? 1 : 0;
    }

    // M11.3: BF16 requires MTLGPUFamilyApple9 (M3/A17 Pro generation and newer).
    // MTLGPUFamilyApple9 corresponds to devices that natively support BFloat16
    // in Metal Shading Language and MPSGraph.
    // supportsFamily: is upward-compatible: M4 (family > Apple9) returns YES.
    bool gpu_supports_bfloat16() {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) return false;
      if (@available(macOS 14.0, *)) {
        return [device supportsFamily:MTLGPUFamilyApple9];
      }
      return false;
    }

    // FP16 is universally supported on all Apple GPUs that support Metal
    // (Apple7+ / all M-series Macs). MPSMatrixMultiplication accepts Float16.
    bool gpu_supports_float16() {
      return MTLCreateSystemDefaultDevice() != nil;
    }

  }
}
