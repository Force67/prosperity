#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>
#include <base/containers/vector.h>

namespace gfx {
class frameBase {
public:
  virtual ~frameBase() = default;
  virtual void toggleFullscreen() = 0;
  virtual void takeScreenshot(base::Vector<uint8_t> &data, uint32_t sizeX,
                              uint32_t sizeY) = 0;
};
}
