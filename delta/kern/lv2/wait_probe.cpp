/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Wait probe. A thread blocked in a syscall burns no cycles, so a profiler
 * cannot see it: a title stalled on a wait looks exactly like an idle one.
 * This records which wait each guest thread is parked in and for how long, and
 * reports the ones that never come back.
 */

#include "wait_probe.h"

#include "base/arch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <base/logging.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <utl/options.h>

#include "kern/crash.h"

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid
}

namespace {
DELTA_OPTION(bool, kWaitProbe, "DELTA_WAIT_PROBE", false);
}  // namespace

namespace krnl {
namespace {

struct Parked {
  const char *what;
  unsigned gtid;
  long a0, a1;
  std::chrono::steady_clock::time_point since;
  uintptr_t gsp;   // guest stack at the syscall, walked by the reporter
  bool traced;     // this wait has had its guest stack reported
};

std::mutex g_mtx;
std::unordered_map<long, Parked> g_parked;

long selfTid() { return static_cast<long>(::syscall(SYS_gettid)); }

unsigned currentGuestTid() {
  const u32 *p = krnl::currentGuestTidPtr();
  return p ? *p : 0;
}

bool probeOn() {
  return kWaitProbe;
}

// The host thread's comm name (set from the guest's sys_mname stack tag, see
// thread_names.cpp) read fresh at report time, so late renames still show.
void threadComm(long tid, char *buf, size_t len) {
  buf[0] = '\0';
  char path[64];
  std::snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", tid);
  if (std::FILE *f = std::fopen(path, "r")) {
    if (std::fgets(buf, static_cast<int>(len), f)) {
      if (char *nl = std::strchr(buf, '\n'))
        *nl = '\0';
    }
    std::fclose(f);
  }
}

void startReporter() {
  static const bool started = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(g_mtx);
        bool any = false;
        for (auto &[tid, p] : g_parked) {
          const auto secs =
              std::chrono::duration_cast<std::chrono::seconds>(now - p.since)
                  .count();
          if (secs < 2)
            continue;
          if (!any) {
            BASE_LOGI("waitprobe", "threads parked > 2s:");
            any = true;
          }
          char comm[32];
          threadComm(tid, comm, sizeof(comm));
          BASE_LOGI("waitprobe",
                    "  tid={} ({:<15}) {:<16} {}s a0={:#x} a1={:#x}", tid, comm,
                    p.what, static_cast<long long>(secs), p.a0, p.a1);
          // Once per wait, not once per thread: a thread that gets past one
          // wait and parks in the next one has a different story to tell.
          if (!p.traced) {
            p.traced = true;
            guestStackTraceFrom(p.gsp, "waitprobe", 20, tid);
          }
        }
      }
    }).detach();
    return true;
  }();
  (void)started;
}

}  // namespace

void waitProbeEnter(const char *what, long a0, long a1) {
  if (!probeOn())
    return;
  startReporter();
  const long tid = selfTid();
  std::lock_guard<std::mutex> lk(g_mtx);
  g_parked[tid] = {what, currentGuestTid(), a0, a1,
                   std::chrono::steady_clock::now(), guestStackScanBase(),
                   false};
}

bool waitProbeDescribeGuest(unsigned gtid, char *out, unsigned long len) {
  if (!out || !len)
    return false;
  out[0] = '\0';
  if (!probeOn() || !gtid)
    return false;
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(g_mtx);
  for (const auto &[tid, p] : g_parked) {
    if (p.gtid != gtid)
      continue;
    const long secs = static_cast<long>(
        std::chrono::duration_cast<std::chrono::seconds>(now - p.since).count());
    std::snprintf(out, len, "%s(%#lx) for %lds", p.what, p.a0, secs);
    return true;
  }
  return false;
}

void waitProbeExit() {
  if (!probeOn())
    return;
  const long tid = selfTid();
  std::lock_guard<std::mutex> lk(g_mtx);
  g_parked.erase(tid);
}

WaitProbe::WaitProbe(const char *what, long a0, long a1) : on(probeOn()) {
  if (on)
    waitProbeEnter(what, a0, a1);
}
WaitProbe::~WaitProbe() {
  if (on)
    waitProbeExit();
}

}  // namespace krnl
