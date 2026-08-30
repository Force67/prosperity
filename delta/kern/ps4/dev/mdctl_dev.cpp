#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstring>

#include "file_dev.h"
#include "mdctl_dev.h"

namespace krnl {
namespace {
// md_ioctl commands (FreeBSD MDIOC*).
constexpr u32 kMdAttach = 0xc1c06d00;
constexpr u32 kMdDetach = 0xc1c06d01;
constexpr u32 kMdQuery  = 0xc1c06d02;
constexpr u32 kMdList   = 0xc1c06d03;
} // namespace

mdctlDevice::mdctlDevice(objectTable &objects) : device(objects) {}

i32 mdctlDevice::ioctl(u32 cmd, void *data) {
  auto *args = static_cast<u8 *>(data);
  if (!args)
    return -SysError::eFAULT;

  // md_ioctl layout: version u32@0 (must be 0), unit u32@4, type u32@8,
  // file u64@16, mediasize u64@24, sectorsize u32@32, flags u32@36.
  if (*reinterpret_cast<u32 *>(args) != 0)
    return -SysError::eINVAL;

  switch (cmd) {
  case kMdAttach: {
    // No memory-disk backing in the emulator; report unsupported so a caller
    // that expects a /dev/md%d afterwards fails cleanly instead of hanging.
    BASE_LOGI("mdctl", "ATTACH unsupported (no memory-disk backing)");
    return -SysError::eOPNOTSUPP;
  }
  case kMdDetach: {
    if (reinterpret_cast<u64 *>(args)[3] != 0)
      return -SysError::eINVAL;
    const u32 flags = reinterpret_cast<u32 *>(args)[9];
    if ((flags & 0xffffffdfu) != 0)
      return -SysError::eINVAL;
    // No disks attached: unit not found.
    return -SysError::eNOENT;
  }
  case kMdQuery: {
    // No disks attached: unit not found.
    return -SysError::eNOENT;
  }
  case kMdList: {
    // No disks: write the empty-unit count and succeed.
    *reinterpret_cast<u32 *>(args + 56) = 0;
    return 0;
  }
  default:
    // Unknown mdctl command: soft-succeed, zeroing any OUT payload so the
    // caller reads a benign result instead of stale stack bytes.
    BASE_LOGI("mdctl", "UNHANDLED ioctl({:#x})", cmd);
    if (cmd & 0x40000000u) {
      const u32 len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(args, 0, len);
    }
    return 0;
  }
}

i64 mdctlDevice::lseek(i64, int) { return 0; }

int mdctlDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
