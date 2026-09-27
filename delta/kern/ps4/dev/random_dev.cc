/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"
#include "base/random/random.h"

#include "base/math/value_bounds.h"
#include "kern/ps4/dev/file_dev.h"  // FillStat / kSceFileMode*
#include "kern/ps4/dev/random_dev.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kArndZero, "DELTA_ARND_ZERO", false);
}  // namespace

namespace krnl {
namespace {
// DELTA_ARND_ZERO already exists for the sysctl entropy path; honour it here
// too so a run can be made deterministic end to end.
bool ZeroEntropy() {
  return kArndZero;
}
}  // namespace

RandomDevice::RandomDevice(ObjectTable& objects) : Device(objects) {}

i64 RandomDevice::Read(void* buf, size_t len) {
  if (!buf)
    return -SysError::eFAULT;
  if (ZeroEntropy()) {
    std::memset(buf, 0, len);
    return static_cast<i64>(len);
  }
  // xorshift64*, seeded per thread from the OS entropy source.
  static thread_local u64 state = base::SourceTrueRandomSeed() | 1;
  auto* out = static_cast<u8*>(buf);
  size_t done = 0;
  while (done < len) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const u64 v = state * 0x2545F4914F6CDD1Dull;
    const size_t n = base::Min(sizeof(v), len - done);
    std::memcpy(out + done, &v, n);
    done += n;
  }
  return static_cast<i64>(len);
}

// /dev/rng hands out entropy through IOR('S', 2, 0x44) instead of read(): the
// reply is a leading status word followed by the bytes, and the caller copies
// from offset 4. Everything else falls through to the device default.
i32 RandomDevice::Ioctl(u32 command, void* args) {
  const u32 group = (command >> 8) & 0xff, num = command & 0xff;
  const u32 len = (command >> 16) & 0x1fff;
  if (group != 'S' || num != 2 || !args || len <= 4)
    return Device::Ioctl(command, args);
  auto* out = static_cast<u8*>(args);
  std::memset(out, 0, 4);
  Read(out + 4, len - 4);
  return 0;
}

// A character device has no position; seeks succeed and stay at 0 so a caller
// that rewinds before reading doesn't error out.
i64 RandomDevice::Lseek(i64, int) {
  return 0;
}

int RandomDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000 /*S_IFCHR*/, 0);
  return 0;
}
}  // namespace krnl
