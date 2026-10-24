/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero clock syscalls. The PS5 kernel accepts the FreeBSD CPU-clock ids
 * (negative, from clock_getcpuclockid2) that it refuses a PS4 process.
 */

#include <cstring>
#include <ctime>
#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_thread.h"
#include "kern/lv2/sys_time.h"

namespace kern {
namespace {
// FreeBSD's CPUCLOCK encoding: bit 31 marks a CPU clock, bit 30 a process one.
constexpr u32 kCpuClockBit = 0x80000000;
constexpr u32 kCpuClockProcessBit = 0x40000000;
constexpr u32 kCpuClockIdMask = 0x3fffffff;

enum CpuClockWhich { kWhichPid = 0, kWhichTid = 1 };

// The host clock behind a guest CPU-clock id, or false for an id naming no
// thread or process of ours (the kernel answers EINVAL then).
bool HostCpuClock(u32 id, clockid_t& host) {
  const u32 target = id & kCpuClockIdMask;
  if (id & kCpuClockProcessBit) {
    if (static_cast<int>(target) != sys_getpid())
      return false;
    host = CLOCK_PROCESS_CPUTIME_ID;
    return true;
  }
  if (target == *CurrentGuestTidPtr()) {
    host = CLOCK_THREAD_CPUTIME_ID;
    return true;
  }
  const long tid = HostTidForGuest(target);
  if (!tid)
    return false;
  // Linux MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED).
  host = static_cast<clockid_t>((~static_cast<unsigned long>(tid) << 3) | 6);
  return true;
}
}  // namespace

int PS4ABI Ps5ClockGettime(int clock_id, void* tp) {
  if (clock_id >= 0)
    return sys_clock_gettime(static_cast<u32>(clock_id),
                             static_cast<sce_timespec*>(tp));
  clockid_t host;
  if (!HostCpuClock(static_cast<u32>(clock_id), host))
    return -SysError::eINVAL;
  timespec ts{};
  if (::clock_gettime(host, &ts) != 0)
    return -SysError::eINVAL;
  auto* out = static_cast<sce_timespec*>(tp);
  out->tv_sec = ts.tv_sec;
  out->tv_nsec = ts.tv_nsec;
  return 0;
}

int PS4ABI Ps5ClockGetres(int clock_id, void* res) {
  if (clock_id >= 0)
    return sys_clock_getres(static_cast<u32>(clock_id),
                            static_cast<sce_timespec*>(res));
  clockid_t host;
  if (!HostCpuClock(static_cast<u32>(clock_id), host))
    return -SysError::eINVAL;
  if (res) {
    auto* out = static_cast<sce_timespec*>(res);
    out->tv_sec = 0;
    out->tv_nsec = 1;
  }
  return 0;
}

int PS4ABI Ps5ClockGetcpuclockid2(i64 id, int which, int* clock_id) {
  u32 out;
  if (which == kWhichTid) {
    out = kCpuClockBit | static_cast<u32>(id ? id : *CurrentGuestTidPtr());
  } else if (which == kWhichPid) {
    if (id && id != sys_getpid())
      return -SysError::eSRCH;
    out = kCpuClockBit | kCpuClockProcessBit | static_cast<u32>(sys_getpid());
  } else {
    return -SysError::eINVAL;
  }
  if (!clock_id)
    return -SysError::eFAULT;
  *clock_id = static_cast<int>(out);
  return 0;
}

// The guest runs in UTC: no offset and no daylight saving, so the 16-byte
// zone record and the dst flag are all zero.
int PS4ABI Ps5UtcToLocaltime(i64 time, i64* local, void* tz, int* dst) {
  if (local)
    *local = time;
  if (tz)
    std::memset(tz, 0, 16);
  if (dst)
    *dst = 0;
  return 0;
}

int PS4ABI
Ps5LocaltimeToUtc(i64 time, int /*type*/, i64* utc, void* tz, int* dst) {
  return Ps5UtcToLocaltime(time, utc, tz, dst);
}

}  // namespace kern
