#pragma once

namespace ctranslate2 {
  namespace metal {

    // Returns 1 if a Metal-capable device is present, 0 otherwise.
    // Implemented in device.mm using MTLCreateSystemDefaultDevice().
    int get_device_count();

  }
}
