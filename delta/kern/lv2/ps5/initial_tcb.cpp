#include "initial_tcb.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/logging.h>

#include "kern/proc.h"
#include "kern/lv2/sys_mem.h"
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

// Stand-in for libkernel's `struct pthread`: only what the mutex path touches before
// libkernel installs its own thread. The id at +0 becomes the mutex owner word; the
// owned-mutex queues are TAILQs whose empty head has tqh_last -> &tqh_first, so a
// zeroed head faults on the first insert. Offsets from libkernel's thread initializer
// (fw 01.14.00, +0x37940..0x37992).
constexpr size_t kThreadSize = 0x1000;
constexpr u32 kInitialThreadId = 1;
constexpr size_t kMutexQueues[] = {0x1a0, 0x1b0};

} // namespace

// The Prospero kernel hands the initial thread a TCB before libkernel's entry, and
// libkernel relies on it: pthread_mutex_lock loads fs:0x10 with no null check on its
// first instruction, and libc's init locks before the sysarch(AMD64_SET_FSBASE) that
// installs libkernel's own TCB, so a zero fs base faults any title whose libc init
// locks (e.g. a sceLibcParam malloc replacement). Provide TCB + static-TLS room below
// it + a zeroed stand-in thread; libkernel replaces both at its own init.
u64 makeInitialTcb() {
  size_t tls = 0x10000;  // slack for modules whose PT_TLS we have not seen yet
  if (auto *p = proc::getActive())
    for (auto &m : p->getModuleList())
      if (m)
        tls += (m->getInfo().tlsSizeMem + 0xFFF) & ~size_t(0xFFF);

  u8 *block = krnl::allocLowGuest(tls + kTcbSize + kThreadSize);
  if (!block)
    return 0;
  std::memset(block, 0, tls + kTcbSize + kThreadSize);

  u8 *tcb = block + tls;
  u8 *thread = tcb + kTcbSize;
  *reinterpret_cast<u64 *>(tcb + kTcbSelf) =
      reinterpret_cast<u64>(tcb);
  *reinterpret_cast<u64 *>(tcb + kTcbDtv) = 0;
  *reinterpret_cast<u64 *>(tcb + kTcbThread) =
      reinterpret_cast<u64>(thread);
  *reinterpret_cast<u32 *>(thread) = kInitialThreadId;
  for (size_t q : kMutexQueues) {  // TAILQ_INIT
    *reinterpret_cast<u64 *>(thread + q) = 0;
    *reinterpret_cast<u64 *>(thread + q + 8) =
        reinterpret_cast<u64>(thread + q);
  }

  if (kProcparamTrace)
    BASE_LOGI("tcb", "initial thread: tcb={:p} thread={:p} static-tls={:#x}",
              static_cast<void *>(tcb), static_cast<void *>(thread), tls);
  return reinterpret_cast<u64>(tcb);
}

} // namespace krnl::ps5
