
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "../proc.h"
#include "cpu/cpu_backend.h"
#include "error_table.h"
#include "sys_thread_ext.h"

namespace krnl {

struct ts64 {
  int64_t tv_sec;
  int64_t tv_nsec;
};

struct sched_param64 {
  int32_t sched_priority;
};

// FreeBSD stores a nonzero value at *state and wakes the umtx waiter the joiner
// spins on, then tears the thread down. We must not call exit(): that would kill
// the whole emulated process. Publishing the state word and returning lets the
// guest thread function return so cpu::runGuestThread unwinds the host thread.
int PS4ABI sys_thr_exit(int64_t *state) {
  if (state)
    *state = 1;
  std::printf("[thr_exit] state=%p -> terminating guest thread\n",
              (void *)state);
  // thr_exit must never return to the caller: FreeBSD destroys the thread
  // in-kernel, and libkernel's pthread trampoline aborts ("thr_exit() returned")
  // if it does. Leave the JIT now (FEX longjmps out of ExecuteThread; native is
  // a no-op and the entry returns naturally).
  cpu::exitGuestThread();
  return 0;
}

// No guest signal machinery, so inter-thread kills are accepted and logged.
int PS4ABI sys_thr_kill(uint32_t tid, int sig) {
  std::printf("[thr_kill] tid=%u sig=%d (ignored)\n", tid, sig);
  return 0;
}

int PS4ABI sys_thr_kill2(uint32_t pid, uint32_t tid, int sig) {
  std::printf("[thr_kill2] pid=%u tid=%u sig=%d (ignored)\n", pid, tid, sig);
  return 0;
}

// We can't freeze another host thread without a suspension protocol, so sleep
// the caller briefly (honouring the timeout if given, capped) and return. Most
// callers poll, so a short delay is enough.
int PS4ABI sys_thr_suspend(const void *timeout) {
  using namespace std::chrono;
  nanoseconds dur = milliseconds(1);
  if (timeout) {
    auto *ts = static_cast<const ts64 *>(timeout);
    auto req = seconds(ts->tv_sec) + nanoseconds(ts->tv_nsec);
    dur = std::min<nanoseconds>(req, milliseconds(50));
  }
  std::this_thread::sleep_for(dur);
  return 0;
}

int PS4ABI sys_thr_wake(uint32_t tid) { return 0; }

int PS4ABI sys_thr_set_name(uint32_t tid, const char *name) {
  std::printf("[thr_set_name] tid=%u name=%s\n", tid, name ? name : "(null)");
  return 0;
}

int PS4ABI sys_thr_get_name(uint32_t tid, char *buf) {
  if (buf)
    std::strcpy(buf, "thr");
  return 0;
}

// A CPU "relax" hint for a spin-wait: no syscall, no context switch. The guest
// scheduler (e.g. Doom64's KEX parallel-job manager) busy-waits for a worker by
// calling scePthreadYield in a tight loop -- millions of times per frame. Mapping
// that to the host sched_yield forced a context-switch each call; with ~35 guest
// threads runnable the host scheduler kept parking the waiter, so the worker's
// result was only seen after tens of microseconds of scheduling latency PER spin,
// summing to ~1s/frame (~1fps). On our many-core host the workers run on their own
// cores regardless, so a `pause` (the worker's write is seen within microseconds)
// is both correct and ~100x faster here.
static inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

int PS4ABI sys_yield() {
  cpuRelax();
  return 0;
}
int PS4ABI sys_sched_yield() {
  cpuRelax();
  return 0;
}

// PS4 pthread priorities run 256 (lowest) to 767 (highest); report that band so
// guest libpthread maps into the range it expects.
int PS4ABI sys_sched_get_priority_max(int policy) { return 767; }
int PS4ABI sys_sched_get_priority_min(int policy) { return 256; }

int PS4ABI sys_sched_setparam(int pid, const void *param) { return 0; }

// Report a fixed mid/high priority so read-modify-write callers see a sane value.
int PS4ABI sys_sched_getparam(int pid, void *param) {
  if (param)
    static_cast<sched_param64 *>(param)->sched_priority = 700;
  return 0;
}

int PS4ABI sys_sched_setscheduler(int pid, int policy, const void *param) {
  return 0;
}
// Report SCHED_OTHER (the timesharing class). 0 isn't a valid FreeBSD policy
// (FIFO=1, OTHER=2, RR=3), so a guest that validates the result would reject it.
int PS4ABI sys_sched_getscheduler(int pid) { return 2; }

int PS4ABI sys_sched_rr_get_interval(int pid, void *interval) {
  if (interval) {
    auto *ts = static_cast<ts64 *>(interval);
    ts->tv_sec = 0;
    ts->tv_nsec = 10'000'000; // 10 ms
  }
  return 0;
}

// All guest threads run on the host scheduler with no pinning, so cpuset
// operations are accepted and ignored.
int PS4ABI sys_cpuset_setid(int which, int level, void *id, int setid) {
  return 0;
}

int PS4ABI sys_cpuset_getid(int level, int which, int64_t id, int *setid) {
  if (setid)
    *setid = 0;
  return 0;
}

int PS4ABI sys_cpuset_setaffinity(int level, int which, int64_t id,
                                  size_t cpusetsize, const void *mask) {
  return 0;
}

// We can't safely snapshot a JITed guest thread's registers, so suspend/resume
// no-op and the get/set context calls report "not supported".
int PS4ABI sys_thr_suspend_ucontext(uint32_t tid) { return 0; }
int PS4ABI sys_thr_resume_ucontext(uint32_t tid) { return 0; }

int PS4ABI sys_thr_get_ucontext(uint32_t tid, void *ucontext) {
  std::printf("[thr_get_ucontext] tid=%u (unsupported)\n", tid);
  return -SysError::eOPNOTSUPP;
}
int PS4ABI sys_thr_set_ucontext(uint32_t tid, void *ucontext) {
  std::printf("[thr_set_ucontext] tid=%u (unsupported)\n", tid);
  return -SysError::eOPNOTSUPP;
}

} // namespace krnl
