#include "kern/ps4/hardware_mode.h"
#include "base/arch.h"

#include <cstdlib>
#include "base/atomic.h"
#include "options/options.h"

namespace kern::ps4 {

DELTA_OPTION(bool, kNeoMode, "DELTA_PS4_NEO", false);

namespace {

constexpr HardwareModeProfile kBaseProfile{HardwareMode::kBase, 0x710f10};
constexpr HardwareModeProfile kNeoProfile{HardwareMode::kNeo, 0x740f30};
base::Atomic<u32> g_title_attributes{0};

}  // namespace

const HardwareModeProfile& GetHardwareModeProfile() {
  return kNeoMode ? kNeoProfile : kBaseProfile;
}

void SetTitleAttributes(u32 attributes) {
  g_title_attributes.store(attributes, base::memory_order_release);
}

u32 TitleAttributes() {
  return g_title_attributes.load(base::memory_order_acquire);
}

u32 CpuMode() {
  const u32 attributes = TitleAttributes();
  const bool six_cpu = attributes & (1u << 15);
  const bool seven_cpu = attributes & (1u << 16);
  if (six_cpu && seven_cpu)
    return 2;
  return seven_cpu ? 5 : 0;
}

bool IsNeoMode() {
  return GetHardwareModeProfile().mode == HardwareMode::kNeo &&
         (TitleAttributes() & (1u << 23));
}

const char* GnmDriverModule() {
  return IsNeoMode() ? "libSceGnmDriverForNeoMode" : "libSceGnmDriver";
}

}  // namespace kern::ps4
