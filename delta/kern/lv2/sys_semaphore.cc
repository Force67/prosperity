/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "base/logging.h"
#include "guest_abi.h"

#include <cstdio>
#include "kern/lv2/wait_probe.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_semaphore.h"
#include "kern/lv2/sys_thread.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
// DELTA_OSEM_TRACE=<id>: every wait/post on one semaphore, with the guest tid.
// A title parked on a semaphore is waiting for a post that either never comes
// or came before it arrived, and only the pairing tells the two apart.
DELTA_OPTION(u32, kOsemTrace, "DELTA_OSEM_TRACE", 0);
// DELTA_OSEM_MAXWAIT=<ms>: cap an untimed wait. Diagnostic: a title parked
// forever on a semaphore tells you nothing, but one that is let go says what it
// does next, usually an assert naming what it was waiting for.
DELTA_OPTION(u32, kOsemMaxWaitMs, "DELTA_OSEM_MAXWAIT", 0);
}  // namespace

namespace kern {

static void OsemTrace(const char* what, int id, int n, int count) {
  // 1 traces every semaphore; any other value traces just that id.
  if (!kOsemTrace || (kOsemTrace != 1 && static_cast<u32>(id) != kOsemTrace))
    return;
  BASE_LOGI("osem", "{} id={} n={} count={} gtid={}", what, id, n, count,
            *CurrentGuestTidPtr());
}

// Named semaphores, so osem_open(name) finds the one osem_create(name) made.
static base::Mutex g_sem_reg_m;
static base::HashMap<base::String, Semaphore*> g_sem_by_name;

Semaphore::Semaphore(ObjectTable& objects, const char* nm, int init, int max)
    : Object(objects, Type::kSemaphore),
      count_(init),
      max_count_(max),
      init_count_(init) {
  if (nm && *nm) {
    name_ = nm;
    base::LockGuard<base::Mutex> lk(g_sem_reg_m);
    g_sem_by_name[nm] = this;
  }
}

int Semaphore::Wait(int need, u32* timeout_us) {
  if (need <= 0)
    return -SysError::eINVAL;
  base::UniqueLock<base::Mutex> lk(m_);
  // A request larger than the ceiling can never succeed: the kernel rejects it
  // outright with EINVAL rather than parking the thread forever.
  if (max_count_ > 0 && need > max_count_)
    return -SysError::eINVAL;
  auto enough = [&] { return count_ >= need; };
  if (!enough()) {
    waiters_++;
    bool ok = true;
    if (timeout_us)
      ok = cv_.WaitFor(lk, base::Microseconds(*timeout_us), enough);
    else
      cv_.Wait(lk, enough);
    waiters_--;
    if (!ok)
      return -SysError::eTIMEDOUT;
  }
  count_ -= need;
  return 0;
}

int Semaphore::Trywait(int need) {
  if (need <= 0)
    return -SysError::eINVAL;
  base::UniqueLock<base::Mutex> lk(m_);
  if (count_ < need) {
    // Distinguish an impossible request (need > ceiling => EINVAL) from a
    // momentarily-unavailable one (=> EBUSY).
    if (max_count_ > 0 && need > max_count_)
      return -SysError::eINVAL;
    return -SysError::eBUSY;
  }
  count_ -= need;
  return 0;
}

int Semaphore::Post(int n) {
  if (n <= 0)
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(m_);
  // The kernel rejects a post that would push the count past maxCount and
  // leaves the count untouched (returns EINVAL).
  if (max_count_ > 0 && count_ + n > max_count_)
    return -SysError::eINVAL;
  count_ += n;
  cv_.NotifyAll();
  return 0;
}

int Semaphore::Cancel(int set_count, int* num_waiters) {
  base::LockGuard<base::Mutex> lk(m_);
  if (max_count_ > 0 && set_count > max_count_)
    return -SysError::eINVAL;
  // Report the waiter count before waking: each woken thread will decrement it
  // itself as it returns from cv.wait.
  if (num_waiters)
    *num_waiters = waiters_;
  if (set_count < 0)
    count_ = init_count_;  // negative => reset to the create-time value
  else
    count_ = set_count;
  cv_.NotifyAll();
  return 0;
}

static Semaphore* FromId(int id) {
  auto* obj = Process::GetActive()->GetObjTable().Get(id);
  if (!obj || obj->type() != Object::Type::kSemaphore)
    return nullptr;
  return static_cast<Semaphore*>(obj);
}

int PS4ABI sys_osem_create(const char* name, u32 attr, int init, int max) {
  auto* s = new Semaphore(Process::GetActive()->GetObjTable(), name, init, max);
  BASE_LOGI("osem", "create '{}' attr={:#x} init={} max={} -> id={}",
            name ? name : "", attr, init, max, s->handle());
  return s->handle();
}

int PS4ABI sys_osem_open(const char* name) {
  {
    base::LockGuard<base::Mutex> lk(g_sem_reg_m);
    auto it = name ? g_sem_by_name.find(name) : g_sem_by_name.end();
    if (it != g_sem_by_name.end())
      return it->second->handle();
  }
  // Auto-create unknown named semaphores (a system service makes them on real
  // hw); creating on first open gives producer+consumer a shared one.
  auto* s =
      new Semaphore(Process::GetActive()->GetObjTable(), name, 0, 0x7fffffff);
  BASE_LOGI("osem", "open '{}' (auto-created) -> id={}", name ? name : "",
            s->handle());
  return s->handle();
}

int PS4ABI sys_osem_delete(int id) {
  auto* s = FromId(id);
  if (!s)
    return -SysError::eSRCH;
  {
    base::LockGuard<base::Mutex> lk(g_sem_reg_m);
    if (!s->fname().empty())
      g_sem_by_name.erase(s->fname().c_str());
  }
  Process::GetActive()->GetObjTable().Release(id);
  return 0;
}

int PS4ABI sys_osem_close(int id) {
  return sys_osem_delete(id);
}

int PS4ABI sys_osem_wait(int id, int need, u32* timeout_us) {
  WaitProbe wp("osem_wait", (long)id, (long)need);
  auto* s = FromId(id);
  if (!s)
    return -SysError::eSRCH;
  OsemTrace("wait", id, need, s->value());
  u32 cap_us = kOsemMaxWaitMs * 1000;
  if (!timeout_us && cap_us)
    timeout_us = &cap_us;
  // The doorbell of a service we do not host: nothing in this process will ever
  // ring it, so an untimed wait parks the caller for the rest of the run (Tomb
  // Raider's sceNpCheckCallback sat on 'SceNpTpip 0' forever). Give it the
  // answer an idle channel gives: wait a beat, then time out, so the caller
  // polls on instead of blocking, without spinning a core.
  if (IsAbsentServiceChannel(s->fname().c_str())) {
    u32 idle_us = 100 * 1000;
    if (timeout_us && *timeout_us < idle_us)
      idle_us = *timeout_us;
    const int r = s->Wait(need, &idle_us);
    return r == 0 ? 0 : -SysError::eTIMEDOUT;
  }
  return s->Wait(need, timeout_us);
}

int PS4ABI sys_osem_trywait(int id, int need) {
  auto* s = FromId(id);
  if (!s)
    return -SysError::eSRCH;
  return s->Trywait(need);
}

int PS4ABI sys_osem_post(int id, int count) {
  auto* s = FromId(id);
  if (!s)
    return -SysError::eSRCH;
  const int r = s->Post(count);
  OsemTrace("post", id, count, s->value());
  return r;
}

int PS4ABI sys_osem_cancel(int id, int set_count, int* num_waiters) {
  auto* s = FromId(id);
  if (!s)
    return -SysError::eSRCH;
  return s->Cancel(set_count, num_waiters);
}
}  // namespace kern
