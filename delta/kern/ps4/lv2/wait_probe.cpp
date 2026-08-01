/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Wait probe. A thread blocked in a syscall burns no cycles, so a profiler
 * cannot see it: a title stalled on a wait looks exactly like an idle one.
 * This records which wait each guest thread is parked in and for how long, and
 * reports the ones that never come back.
 */

#include "wait_probe.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <unistd.h>
#include <sys/syscall.h>
#include <utl/options.h>

#include "kern/crash.h"

namespace {
DELTA_OPTION(bool, kWaitProbe, "DELTA_WAIT_PROBE", false);
}  // namespace

namespace krnl {
namespace {

struct Parked {
  const char *what;
  long a0, a1;
  std::chrono::steady_clock::time_point since;
  uintptr_t gsp;  // guest stack at the syscall, walked by the reporter
};

std::mutex g_mtx;
std::unordered_map<long, Parked> g_parked;

long selfTid() { return static_cast<long>(::syscall(SYS_gettid)); }

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
      std::unordered_set<long> once;  // one guest trace per thread, not per tick
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(g_mtx);
        bool any = false;
        for (const auto &[tid, p] : g_parked) {
          const auto secs =
              std::chrono::duration_cast<std::chrono::seconds>(now - p.since)
                  .count();
          if (secs < 2)
            continue;
          if (!any) {
            std::fprintf(stderr, "[waitprobe] threads parked > 2s:\n");
            any = true;
          }
          char comm[32];
          threadComm(tid, comm, sizeof(comm));
          std::fprintf(stderr,
                       "[waitprobe]   tid=%ld (%-15s) %-16s %llds a0=%#lx a1=%#lx\n",
                       tid, comm, p.what, static_cast<long long>(secs), p.a0,
                       p.a1);
          if (once.insert(tid).second)
            guestStackTraceFrom(p.gsp, "waitprobe", 20, tid);
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
  g_parked[tid] = {what, a0, a1, std::chrono::steady_clock::now(),
                   guestStackScanBase()};
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
