#pragma once

#include "base/arch.h"
#include <options/options.h>


namespace krnl::ps4 {

enum class HardwareMode { base, neo };

struct HardwareModeProfile {
  HardwareMode mode;
  u32 mainSocId;
};

// DELTA_PS4_NEO selects the emulated hardware. A title only enters enhanced
// Neo mode when its param.sfo ATTRIBUTE also advertises Neo support.
extern base::Option<bool> kNeoMode;
const HardwareModeProfile &hardwareModeProfile();

void setTitleAttributes(u32 attributes);
u32 titleAttributes();
u32 cpuMode();
bool isNeoMode();
const char *gnmDriverModule();

} // namespace krnl::ps4
