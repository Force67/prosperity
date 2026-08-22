/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>

#include "wait_probe.h"
#include <chrono>
#include <cstdio>
#include <unordered_map>

#include "error_table.h"
#include "kern/proc.h"
#include "sys_mem.h"
#include "sys_semaphore.h"
#include <utl/options.h>

namespace {
// DELTA_OSEM_TRACE=<id>: every wait/post on one semaphore, with the guest tid.
// A title parked on a semaphore is waiting for a post that either never comes
// or came before it arrived, and only the pairing tells the two apart.
DELTA_OPTION(u32, kOsemTrace, "DELTA_OSEM_TRACE", 0);
}  // namespace

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid

static void osemTrace(const char *what, int id, int n, int count) {
  // 1 traces every semaphore; any other value traces just that id.
  if (!kOsemTrace || (kOsemTrace != 1 && static_cast<u32>(id) != kOsemTrace))
    return;
  BASE_LOGI("osem", "{} id={} n={} count={} gtid={}", what, id, n, count,
            *currentGuestTidPtr());
}

// Named semaphores, so osem_open(name) finds the one osem_create(name) made.
static std::mutex g_semRegM;
static std::unordered_map<std::string, semaphore *> g_semByName;

semaphore::semaphore(proc *p, const char *nm, int init, int max)
    : kObject(p, oType::semaphore), count(init), maxCount(max), initCount(init) {
  if (nm && *nm) {
    name = nm;
    std::lock_guard<std::mutex> lk(g_semRegM);
    g_semByName[nm] = this;
  }
}

int semaphore::wait(int need, u32 *timeoutUs) {
  if (need <= 0)
    return -SysError::eINVAL;
  std::unique_lock<std::mutex> lk(m);
  // A request larger than the ceiling can never succeed: the kernel rejects it
  // outright with EINVAL rather than parking the thread forever.
  if (maxCount > 0 && need > maxCount)
    return -SysError::eINVAL;
  auto enough = [&] { return count >= need; };
  if (!enough()) {
    waiters++;
    bool ok = true;
    if (timeoutUs)
      ok = cv.wait_for(lk, std::chrono::microseconds(*timeoutUs), enough);
    else
      cv.wait(lk, enough);
    waiters--;
    if (!ok)
      return -SysError::eTIMEDOUT;
  }
  count -= need;
  return 0;
}

int semaphore::trywait(int need) {
  if (need <= 0)
    return -SysError::eINVAL;
  std::unique_lock<std::mutex> lk(m);
  if (count < need) {
    // Distinguish an impossible request (need > ceiling => EINVAL) from a
    // momentarily-unavailable one (=> EBUSY).
    if (maxCount > 0 && need > maxCount)
      return -SysError::eINVAL;
    return -SysError::eBUSY;
  }
  count -= need;
  return 0;
}

int semaphore::post(int n) {
  if (n <= 0)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(m);
  // The kernel rejects a post that would push the count past maxCount and leaves
  // the count untouched (returns EINVAL).
  if (maxCount > 0 && count + n > maxCount)
    return -SysError::eINVAL;
  count += n;
  cv.notify_all();
  return 0;
}

int semaphore::cancel(int setCount, int *numWaiters) {
  std::lock_guard<std::mutex> lk(m);
  if (maxCount > 0 && setCount > maxCount)
    return -SysError::eINVAL;
  // Report the waiter count before waking: each woken thread will decrement it
  // itself as it returns from cv.wait.
  if (numWaiters)
    *numWaiters = waiters;
  if (setCount < 0)
    count = initCount;  // negative => reset to the create-time value
  else
    count = setCount;
  cv.notify_all();
  return 0;
}

static semaphore *fromId(int id) {
  auto *obj = proc::getActive()->getObjTable().get(id);
  if (!obj || obj->type() != kObject::oType::semaphore)
    return nullptr;
  return static_cast<semaphore *>(obj);
}

int PS4ABI sys_osem_create(const char *name, u32 attr, int init, int max) {
  auto *s = new semaphore(proc::getActive(), name, init, max);
  BASE_LOGI("osem", "create '{}' attr={:#x} init={} max={} -> id={}",
            name ? name : "", attr, init, max, s->handle());
  return s->handle();
}

int PS4ABI sys_osem_open(const char *name) {
  {
    std::lock_guard<std::mutex> lk(g_semRegM);
    auto it = name ? g_semByName.find(name) : g_semByName.end();
    if (it != g_semByName.end())
      return it->second->handle();
  }
  // Auto-create unknown named semaphores (a system service makes them on real
  // hw); creating on first open gives producer+consumer a shared one.
  auto *s = new semaphore(proc::getActive(), name, 0, 0x7fffffff);
  BASE_LOGI("osem", "open '{}' (auto-created) -> id={}", name ? name : "",
            s->handle());
  return s->handle();
}

int PS4ABI sys_osem_delete(int id) {
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  {
    std::lock_guard<std::mutex> lk(g_semRegM);
    if (!s->fname().empty())
      g_semByName.erase(s->fname().c_str());
  }
  proc::getActive()->getObjTable().release(id);
  return 0;
}

int PS4ABI sys_osem_close(int id) { return sys_osem_delete(id); }

int PS4ABI sys_osem_wait(int id, int need, u32 *timeoutUs) {
  WaitProbe _wp("osem_wait", (long)id, (long)need);
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  osemTrace("wait", id, need, s->value());
  // The doorbell of a service we do not host: nothing in this process will ever
  // ring it, so an untimed wait parks the caller for the rest of the run (Tomb
  // Raider's sceNpCheckCallback sat on 'SceNpTpip 0' forever). Give it the
  // answer an idle channel gives -- wait a beat, then time out -- so the caller
  // polls on instead of blocking, without spinning a core.
  if (isAbsentServiceChannel(s->fname().c_str())) {
    u32 idleUs = 100 * 1000;
    if (timeoutUs && *timeoutUs < idleUs)
      idleUs = *timeoutUs;
    const int r = s->wait(need, &idleUs);
    return r == 0 ? 0 : -SysError::eTIMEDOUT;
  }
  return s->wait(need, timeoutUs);
}

int PS4ABI sys_osem_trywait(int id, int need) {
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  return s->trywait(need);
}

int PS4ABI sys_osem_post(int id, int count) {
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  const int r = s->post(count);
  osemTrace("post", id, count, s->value());
  return r;
}

int PS4ABI sys_osem_cancel(int id, int setCount, int *numWaiters) {
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  return s->cancel(setCount, numWaiters);
}
}  // namespace krnl
