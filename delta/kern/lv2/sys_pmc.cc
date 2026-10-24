/*
 * PS4Delta : PS4 emulation and research project
 *
 * opmc_*: the core performance counters a title programs for itself
 * (orbis_pmc). Six counters per thread take AMD PERF_CTL event selects, which
 * the host's Zen PMU counts as Linux raw perf events. The uncore controls
 * (set_hw/get_hw) are checked and accepted; nothing reads the uncore back.
 */

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include "base/arch.h"
#include "base/containers/hash_map.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_debug.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_thread.h"

namespace kern {
namespace {
constexpr u32 kCounters = 6;
// The PERF_CTL bits a title may not set: the high bits, and the user, OS,
// interrupt and enable bits the kernel owns. A title's counters count user
// mode while opmc_enable has them on.
constexpr u64 kCtlReserved = 0xfffffff0007b0000;

struct Counter {
  u64 ctl = 0;
  i64 bias = 0;  // set_ctr's value less what the host counted by then
  int fd = -1;
};

struct ThreadCounters {
  Counter c[kCounters];
  bool enabled = false;
};

base::Mutex g_mutex;
base::HashMap<u32, ThreadCounters> g_threads;

const guest::SessionReset g_reset([] {
  base::LockGuard<base::Mutex> lk(g_mutex);
  for (auto& [tid, t] : g_threads)
    for (auto& c : t.c)
      if (c.fd >= 0)
        ::close(c.fd);
  guest::ResetResource(g_threads);
});

u64 Read(const Counter& c) {
  u64 v = 0;
  if (c.fd >= 0 && ::read(c.fd, &v, sizeof(v)) != sizeof(v))
    v = 0;
  return v;
}

void Close(Counter& c) {
  if (c.fd < 0)
    return;
  c.bias += static_cast<i64>(Read(c));
  ::close(c.fd);
  c.fd = -1;
}

// Counts `c.ctl` on host thread `host_tid` from now on. An unprogrammed
// counter stays closed: the host has fewer free counters than six, and an idle
// event would multiplex the real ones out.
void Open(Counter& c, long host_tid) {
  if (!c.ctl)
    return;
  perf_event_attr attr{};
  attr.size = sizeof(attr);
  attr.type = PERF_TYPE_RAW;
  attr.config = c.ctl;
  attr.exclude_kernel = 1;
  attr.exclude_hv = 1;
  c.fd = static_cast<int>(
      ::syscall(SYS_perf_event_open, &attr, host_tid, -1, -1, 0));
}

// The counters of guest thread `id` (0: the caller) and its host thread. The
// process as a target is for system processes.
int Resolve(u32 id, ThreadCounters** out, long* host_tid) {
  if (static_cast<int>(id) == sys_getpid())
    return -SysError::eSRCH;
  const u32 tid = id ? id : *CurrentGuestTidPtr();
  if (tid == *CurrentGuestTidPtr()) {
    *host_tid = 0;  // perf's "this thread"
  } else {
    *host_tid = HostTidForGuest(tid);
    if (!*host_tid)
      return -SysError::eSRCH;
  }
  *out = &g_threads[tid];
  return 0;
}
}  // namespace

int PS4ABI sys_opmc_enable(u32 id) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  ThreadCounters* t;
  long host;
  if (const int err = Resolve(id, &t, &host))
    return err;
  if (!t->enabled)
    for (auto& c : t->c)
      Open(c, host);
  t->enabled = true;
  return 0;
}

int PS4ABI sys_opmc_disable(u32 id) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  ThreadCounters* t;
  long host;
  if (const int err = Resolve(id, &t, &host))
    return err;
  for (auto& c : t->c)
    Close(c);
  t->enabled = false;
  return 0;
}

int PS4ABI sys_opmc_set_ctl(u32 id, u32 index, u64 ctl) {
  if (index >= kCounters || (ctl & kCtlReserved))
    return -SysError::eINVAL;
  // Two event selects drop the unit-mask bits the kernel reserves for them.
  const u64 event = ctl & 0xf000000ffull;
  if (event == 0x100000070ull || event == 0x6e)
    ctl &= 0xfff8400ffull;
  base::LockGuard<base::Mutex> lk(g_mutex);
  ThreadCounters* t;
  long host;
  if (const int err = Resolve(id, &t, &host))
    return err;
  Counter& c = t->c[index];
  Close(c);
  c.ctl = ctl;
  if (t->enabled)
    Open(c, host);
  return 0;
}

int PS4ABI sys_opmc_set_ctr(u32 id, u32 index, u64 value) {
  if (index >= kCounters)
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(g_mutex);
  ThreadCounters* t;
  long host;
  if (const int err = Resolve(id, &t, &host))
    return err;
  Counter& c = t->c[index];
  c.bias = static_cast<i64>(value) - static_cast<i64>(Read(c));
  return 0;
}

int PS4ABI sys_opmc_get_ctr(u32 id, u32 index, u64* value) {
  if (index >= kCounters)
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(value, sizeof(u64)))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  ThreadCounters* t;
  long host;
  if (const int err = Resolve(id, &t, &host))
    return err;
  const Counter& c = t->c[index];
  *value = static_cast<u64>(static_cast<i64>(Read(c)) + c.bias);
  return 0;
}

// Uncore controls: 23 operations taking {op, a, b, c}.
int PS4ABI sys_opmc_set_hw(u32 op, u64 /*a*/, u32 /*b*/, u64 /*c*/) {
  return op >= 1 && op <= 23 ? 0 : -SysError::eINVAL;
}

// Six uncore reads, each a u64 through `out`; the uncore here counts nothing.
int PS4ABI sys_opmc_get_hw(u32 op, u64 /*a*/, u32 /*index*/, u64* out) {
  if (op < 1 || op > 6)
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(out, sizeof(u64)))
    return -SysError::eFAULT;
  *out = 0;
  return 0;
}

}  // namespace kern
