#pragma once

#include "base/arch.h"

namespace guest {
struct ControllerState {
  u32 buttons = 0;
  u8 lx = 128, ly = 128, rx = 128, ry = 128;
  u8 l2 = 0, r2 = 0;
  u64 timestamp = 0;
};
}  // namespace guest
