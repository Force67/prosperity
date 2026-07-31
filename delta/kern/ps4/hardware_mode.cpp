#include "hardware_mode.h"

#include <atomic>
#include <cstdlib>
#include <utl/options.h>

namespace krnl::ps4 {

DELTA_OPTION(bool, kNeoMode, "DELTA_PS4_NEO", false);

namespace {

constexpr HardwareModeProfile kBaseProfile{HardwareMode::base, 0x710f10};
constexpr HardwareModeProfile kNeoProfile{HardwareMode::neo, 0x740f30};
std::atomic<uint32_t> g_titleAttributes{0};

} // namespace

const HardwareModeProfile &hardwareModeProfile() {
  static const HardwareModeProfile *const profile =
      kNeoMode ? &kNeoProfile : &kBaseProfile;
  return *profile;
}

void setTitleAttributes(uint32_t attributes) {
  g_titleAttributes.store(attributes, std::memory_order_release);
}

uint32_t titleAttributes() {
  return g_titleAttributes.load(std::memory_order_acquire);
}

uint32_t cpuMode() {
  const uint32_t attributes = titleAttributes();
  const bool sixCpu = attributes & (1u << 15);
  const bool sevenCpu = attributes & (1u << 16);
  if (sixCpu && sevenCpu)
    return 2;
  return sevenCpu ? 5 : 0;
}

bool isNeoMode() {
  return hardwareModeProfile().mode == HardwareMode::neo &&
         (titleAttributes() & (1u << 23));
}

const char *gnmDriverModule() {
  return isNeoMode() ? "libSceGnmDriverForNeoMode" : "libSceGnmDriver";
}

} // namespace krnl::ps4
