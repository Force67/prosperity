#pragma once

#include "base/arch.h"
#include "options/options.h"

namespace kern::ps4 {

enum class HardwareMode { kBase, kNeo };

struct HardwareModeProfile {
  HardwareMode mode;
  u32 main_soc_id;
};

// DELTA_PS4_NEO selects the emulated hardware. A title only enters enhanced
// Neo mode when its param.sfo ATTRIBUTE also advertises Neo support.
extern base::Option<bool> kNeoMode;
const HardwareModeProfile& GetHardwareModeProfile();

void SetTitleAttributes(u32 attributes);
u32 TitleAttributes();
u32 CpuMode();
bool IsNeoMode();
const char* GnmDriverModule();

}  // namespace kern::ps4
