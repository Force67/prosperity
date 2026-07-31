#pragma once

#include <cstdint>
#include <utl/options.h>


namespace krnl::ps4 {

enum class HardwareMode { base, neo };

struct HardwareModeProfile {
  HardwareMode mode;
  uint32_t mainSocId;
};

// DELTA_PS4_NEO selects the emulated hardware. A title only enters enhanced
// Neo mode when its param.sfo ATTRIBUTE also advertises Neo support.
extern base::Option<bool> kNeoMode;
const HardwareModeProfile &hardwareModeProfile();

void setTitleAttributes(uint32_t attributes);
uint32_t titleAttributes();
uint32_t cpuMode();
bool isNeoMode();
const char *gnmDriverModule();

} // namespace krnl::ps4
