/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <base.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "file_dev.h" // fillStat / kSceFileMode*
#include "random_dev.h"

namespace krnl {
namespace {
// DELTA_ARND_ZERO already exists for the sysctl entropy path; honour it here too
// so a run can be made deterministic end to end.
bool zeroEntropy() {
  static const bool on = std::getenv("DELTA_ARND_ZERO") != nullptr;
  return on;
}
} // namespace

randomDevice::randomDevice(proc *p) : device(p) {}

int64_t randomDevice::read(void *buf, size_t len) {
  if (!buf)
    return -SysError::eFAULT;
  if (zeroEntropy()) {
    std::memset(buf, 0, len);
    return static_cast<int64_t>(len);
  }
  static thread_local std::random_device rd;
  static thread_local std::mt19937_64 gen(rd());
  auto *out = static_cast<uint8_t *>(buf);
  size_t done = 0;
  while (done < len) {
    const uint64_t v = gen();
    const size_t n = std::min(sizeof(v), len - done);
    std::memcpy(out + done, &v, n);
    done += n;
  }
  return static_cast<int64_t>(len);
}

// A character device has no position; seeks succeed and stay at 0 so a caller
// that rewinds before reading doesn't error out.
int64_t randomDevice::lseek(int64_t, int) { return 0; }

int randomDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000 /*S_IFCHR*/, 0);
  return 0;
}
} // namespace krnl
