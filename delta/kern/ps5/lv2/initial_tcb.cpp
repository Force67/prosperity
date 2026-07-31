#include "initial_tcb.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kProcparamTrace, "DELTA_PROCPARAM_TRACE", false);
}  // namespace

namespace krnl::ps5 {
namespace {

// amd64 TCB, as libkernel reads it through fs.
constexpr size_t kTcbSelf = 0x00;    // struct tcb *
constexpr size_t kTcbDtv = 0x08;     // void *
constexpr size_t kTcbThread = 0x10;  // struct pthread *
constexpr size_t kTcbSize = 0x40;

// Stand-in for libkernel's `struct pthread`. Only what the mutex path touches
// before libkernel installs its own thread matters: the id at +0 (which becomes
// the mutex owner word via `lock cmpxchg`) and the owned-mutex queues, which are
// TAILQs -- an empty one has tqh_last pointing at its own tqh_first, so a zeroed
// head faults on the first insert. Offsets read off libkernel's own thread
// initializer (fw 01.14.00 libkernel+0x37940..0x37992), which TAILQ_INITs
// exactly these two heads.
constexpr size_t kThreadSize = 0x1000;
constexpr uint32_t kInitialThreadId = 1;
constexpr size_t kMutexQueues[] = {0x1a0, 0x1b0};

} // namespace

// The Prospero kernel hands a new process's initial thread a TCB before it jumps
// to libkernel's entry, and libkernel relies on that: pthread_mutex_lock loads
// the current thread from fs:0x10 with no null check, on its very first
// instruction. That first lock comes from libc's init -- which libkernel runs
// BEFORE the sysarch(AMD64_SET_FSBASE) that installs its own TCB -- so entering
// the guest with a zero fs base faults any title whose libc init locks, e.g. one
// installing a sceLibcParam malloc replacement.
//
// Provide the same starting state: a TCB with room below it for static TLS
// (which grows down from the thread pointer) and a zeroed stand-in thread.
// libkernel replaces both as soon as its own thread init runs.
uint64_t makeInitialTcb() {
  size_t tls = 0x10000;  // slack for modules whose PT_TLS we have not seen yet
  if (auto *p = proc::getActive())
    for (auto &m : p->getModuleList())
      if (m)
        tls += (m->getInfo().tlsSizeMem + 0xFFF) & ~size_t(0xFFF);

  uint8_t *block = krnl::allocLowGuest(tls + kTcbSize + kThreadSize);
  if (!block)
    return 0;
  std::memset(block, 0, tls + kTcbSize + kThreadSize);

  uint8_t *tcb = block + tls;
  uint8_t *thread = tcb + kTcbSize;
  *reinterpret_cast<uint64_t *>(tcb + kTcbSelf) =
      reinterpret_cast<uint64_t>(tcb);
  *reinterpret_cast<uint64_t *>(tcb + kTcbDtv) = 0;
  *reinterpret_cast<uint64_t *>(tcb + kTcbThread) =
      reinterpret_cast<uint64_t>(thread);
  *reinterpret_cast<uint32_t *>(thread) = kInitialThreadId;
  for (size_t q : kMutexQueues) {  // TAILQ_INIT
    *reinterpret_cast<uint64_t *>(thread + q) = 0;
    *reinterpret_cast<uint64_t *>(thread + q + 8) =
        reinterpret_cast<uint64_t>(thread + q);
  }

  if (kProcparamTrace)
    std::fprintf(stderr,
                 "[tcb] initial thread: tcb=%p thread=%p static-tls=%#zx\n",
                 static_cast<void *>(tcb), static_cast<void *>(thread), tls);
  return reinterpret_cast<uint64_t>(tcb);
}

} // namespace krnl::ps5
