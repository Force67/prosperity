#include "cpu/aarch64/live_threads.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <FEXCore/Debug/InternalThreadState.h>

#include "cpu/backend.h"
#include "kern/lv2/sys_thread.h"

namespace cpu::fex {
namespace {

base::Mutex g_live_mutex;
base::Vector<LiveThread> g_live;
base::Atomic<u32> g_live_seq{0};

base::Atomic<u64> g_sample_gen{0};
thread_local base::Atomic<u64> t_sample_gen{0};
thread_local base::Atomic<u64> t_sample_rip{0};
thread_local base::Atomic<u64> t_sample_ns{0};

}  // namespace

u32 AddCurrentLiveThread(FEXCore::Core::InternalThreadState* thread) {
  const u32 id = g_live_seq.fetch_add(1);
  base::LockGuard lk(g_live_mutex);
  g_live.push_back({thread, id, ThreadTraceRing(), ThreadTracePos(),
                    kern::CurrentGuestTidPtr(), ThreadInSyscall(),
                    &t_sample_gen, &t_sample_rip, &t_sample_ns,
                    static_cast<pid_t>(::syscall(SYS_gettid))});
  return id;
}

void RemoveLiveThread(FEXCore::Core::InternalThreadState* thread) {
  base::LockGuard lk(g_live_mutex);
  for (size_t i = 0; i < g_live.size(); i++)
    if (g_live[i].thread == thread) {
      g_live.erase(g_live.begin() + i);
      break;
    }
}

u64 BeginRipSampleRound() {
  return g_sample_gen.fetch_add(1, base::memory_order_relaxed) + 1;
}

void RecordRipSample(u64 rip, u64 ns) {
  t_sample_ns.store(ns, base::memory_order_relaxed);
  t_sample_rip.store(rip, base::memory_order_relaxed);
  t_sample_gen.store(g_sample_gen.load(base::memory_order_relaxed),
                     base::memory_order_release);
}

namespace internal {

base::Mutex& LiveThreadMutex() {
  return g_live_mutex;
}

const base::Vector<LiveThread>& LiveThreadList() {
  return g_live;
}

}  // namespace internal
}  // namespace cpu::fex

namespace cpu {

void GuestThreadFsBases(base::Vector<u64>& out) {
  out.clear();
  fex::VisitLiveThreads([&](const base::Vector<fex::LiveThread>& live) {
    out.reserve(live.size());
    for (auto& t : live)
      if (t.thread && t.thread->CurrentFrame)
        out.push_back(t.thread->CurrentFrame->State.fs_cached);
  });
}

}  // namespace cpu
