#pragma once

namespace ctranslate2 {
  namespace metal {

    // Returns 1 if a Metal-capable device is present, 0 otherwise.
    // Implemented in device.mm using MTLCreateSystemDefaultDevice().
    int get_device_count();

    // M11.3: Runtime GPU capability detection.
    // BF16 requires MTLGPUFamilyApple9 (M3/A17 Pro and newer, macOS 14+).
    // FP16 is universally supported on all Apple7+ GPUs (all M-series Macs).
    bool gpu_supports_bfloat16();
    bool gpu_supports_float16();

  }
}
