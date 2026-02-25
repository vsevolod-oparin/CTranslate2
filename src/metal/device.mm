#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "device.h"

namespace ctranslate2 {
  namespace metal {

    int get_device_count() {
      return MTLCreateSystemDefaultDevice() != nil ? 1 : 0;
    }

  }
}
