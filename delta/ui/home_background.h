#pragma once

#include "base/arch.h"

namespace ui {

// A uniform draw from [0, 767] gives the anniversary scene a 1-in-256 chance.
constexpr u32 ChooseHomeBackground(u32 draw) {
  return draw % 256 == 0 ? 4 : 1 + draw / 256;
}

}  // namespace ui
