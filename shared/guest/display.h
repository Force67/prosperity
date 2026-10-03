#pragma once

#include "base/arch.h"
#include "base/functional/function.h"
#include "options/options.h"

namespace guest::display {

DELTA_OPTION_INLINE(bool, kEmulateTiming, "DELTA_DISPLAY_TIMING", true);

u64 VblankCount();
u64 VblankTimeNs();
void WaitVblank();
void SetFlipRate(int rate);
bool QueueFlip(int mode, base::Function<void()> complete);
bool WaitFlip(int mode, base::Function<void()> complete);

}  // namespace guest::display
