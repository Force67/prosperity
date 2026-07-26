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

#include <unistd.h>
#include <sys/syscall.h>

namespace krnl {
namespace {

struct Parked {
  const char *what;
  long a0, a1;
  std::chrono::steady_clock::time_point since;
};

std::mutex g_mtx;
std::unordered_map<long, Parked> g_parked;

long selfTid() { return static_cast<long>(::syscall(SYS_gettid)); }

bool probeOn() {
  static const bool on = std::getenv("DELTA_WAIT_PROBE") != nullptr;
  return on;
}

void startReporter() {
  static const bool started = [] {
    std::thread([] {
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
          std::fprintf(stderr, "[waitprobe]   tid=%ld %-16s %llds a0=%#lx a1=%#lx\n",
                       tid, p.what, static_cast<long long>(secs), p.a0, p.a1);
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
  g_parked[tid] = {what, a0, a1, std::chrono::steady_clock::now()};
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
