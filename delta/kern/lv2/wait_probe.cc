/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Wait probe. A thread blocked in a syscall burns no cycles, so a profiler
 * cannot see it: a title stalled on a wait looks exactly like an idle one.
 * This records which wait each guest thread is parked in and for how long, and
 * reports the ones that never come back.
 */

#include "kern/lv2/wait_probe.h"

#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/syscall.h>
#include <unistd.h>
#include "base/logging.h"
#include "options/options.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/crash.h"

namespace krnl {
const u32* CurrentGuestTidPtr();  // sys_thread.cc: this thread's guest tid
}

namespace {
DELTA_OPTION(bool, kWaitProbe, "DELTA_WAIT_PROBE", false);
}  // namespace

namespace krnl {
namespace {

struct Parked {
  const char* what;
  unsigned gtid;
  long a0, a1;
  base::TimeTicks since;
  uintptr_t gsp;  // guest stack at the syscall, walked by the reporter
  bool traced;    // this wait has had its guest stack reported
};

base::Mutex g_mtx;
base::HashMap<long, Parked> g_parked;

long SelfTid() {
  return static_cast<long>(::syscall(SYS_gettid));
}

unsigned CurrentGuestTid() {
  const u32* p = krnl::CurrentGuestTidPtr();
  return p ? *p : 0;
}

bool ProbeOn() {
  return kWaitProbe;
}

// The host thread's comm name (set from the guest's sys_mname stack tag, see
// thread_names.cc) read fresh at report time, so late renames still show.
void ThreadComm(long tid, char* buf, size_t len) {
  buf[0] = '\0';
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", tid);
  if (std::FILE* f = std::fopen(path, "r")) {
    if (std::fgets(buf, static_cast<int>(len), f)) {
      if (char* nl = std::strchr(buf, '\n'))
        *nl = '\0';
    }
    std::fclose(f);
  }
}

void StartReporter() {
  static const bool kStarted = [] {
    base::SpawnDetachedThread("wait_probe", [] {
      for (;;) {
        base::SleepForMilliseconds((5) * 1000);
        const auto now = base::TimeTicks::Now();
        base::LockGuard<base::Mutex> lk(g_mtx);
        bool any = false;
        for (auto& [tid, p] : g_parked) {
          const auto secs = (now - p.since).InSeconds();
          if (secs < 2)
            continue;
          if (!any) {
            BASE_LOGI("waitprobe", "threads parked > 2s:");
            any = true;
          }
          char comm[32];
          ThreadComm(tid, comm, sizeof(comm));
          BASE_LOGI("waitprobe",
                    "  tid={} ({:<15}) {:<16} {}s a0={:#x} a1={:#x}", tid, comm,
                    p.what, static_cast<long long>(secs), p.a0, p.a1);
          // Once per wait, not once per thread: a thread that gets past one
          // wait and parks in the next one has a different story to tell.
          if (!p.traced) {
            p.traced = true;
            GuestStackTraceFrom(p.gsp, "waitprobe", 20, tid);
          }
        }
      }
    });
    return true;
  }();
  (void)kStarted;
}

}  // namespace

void WaitProbeEnter(const char* what, long a0, long a1) {
  if (!ProbeOn())
    return;
  StartReporter();
  const long tid = SelfTid();
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_parked[tid] = {what, CurrentGuestTid(),      a0,
                   a1,   base::TimeTicks::Now(), GuestStackScanBase(),
                   false};
}

bool WaitProbeDescribeGuest(unsigned gtid, char* out, unsigned long len) {
  if (!out || !len)
    return false;
  out[0] = '\0';
  if (!ProbeOn() || !gtid)
    return false;
  const auto now = base::TimeTicks::Now();
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (const auto& [tid, p] : g_parked) {
    if (p.gtid != gtid)
      continue;
    const long secs = static_cast<long>((now - p.since).InSeconds());
    std::snprintf(out, len, "%s(%#lx) for %lds", p.what, p.a0, secs);
    // The owner's own stack is the other half of the cycle: it says which of
    // its calls is waiting, not just which primitive.
    GuestStackTraceFrom(p.gsp, "umtxstall-owner", 12, tid);
    return true;
  }
  return false;
}

void WaitProbeExit() {
  if (!ProbeOn())
    return;
  const long tid = SelfTid();
  base::LockGuard<base::Mutex> lk(g_mtx);
  g_parked.erase(tid);
}

WaitProbe::WaitProbe(const char* what, long a0, long a1) : on(ProbeOn()) {
  if (on)
    WaitProbeEnter(what, a0, a1);
}
WaitProbe::~WaitProbe() {
  if (on)
    WaitProbeExit();
}

}  // namespace krnl
