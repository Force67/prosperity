#pragma once

#include <sys/types.h>

#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "cpu/aarch64/crossing_trace.h"

namespace FEXCore::Core {
struct InternalThreadState;
}

namespace cpu::fex {

struct LiveThread {
  FEXCore::Core::InternalThreadState* thread;
  u32 id;
  const TraceEvt* trace = nullptr;
  const u32* trace_pos = nullptr;
  const u32* gtid = nullptr;
  const bool* in_syscall = nullptr;
  base::Atomic<u64>* sample_gen = nullptr;
  base::Atomic<u64>* sample_rip = nullptr;
  base::Atomic<u64>* sample_ns = nullptr;
  pid_t host_tid = 0;
};

u32 AddCurrentLiveThread(FEXCore::Core::InternalThreadState* thread);
void RemoveLiveThread(FEXCore::Core::InternalThreadState* thread);

template <typename Fn>
void VisitLiveThreads(Fn&& fn);

u64 BeginRipSampleRound();
void RecordRipSample(u64 rip, u64 ns);

namespace internal {
base::Mutex& LiveThreadMutex();
const base::Vector<LiveThread>& LiveThreadList();
}  // namespace internal

template <typename Fn>
void VisitLiveThreads(Fn&& fn) {
  base::LockGuard lk(internal::LiveThreadMutex());
  fn(internal::LiveThreadList());
}

}  // namespace cpu::fex
