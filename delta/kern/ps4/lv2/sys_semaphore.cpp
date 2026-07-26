/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

#include "wait_probe.h"
#include <chrono>
#include <cstdio>
#include <unordered_map>

#include "error_table.h"
#include "kern/proc.h"
#include "sys_semaphore.h"

namespace krnl {
// Named semaphores, so osem_open(name) finds the one osem_create(name) made.
static std::mutex g_semRegM;
static std::unordered_map<std::string, semaphore *> g_semByName;

semaphore::semaphore(proc *p, const char *nm, int init, int max)
    : kObject(p, oType::semaphore), count(init), maxCount(max) {
  if (nm && *nm) {
    name = nm;
    std::lock_guard<std::mutex> lk(g_semRegM);
    g_semByName[nm] = this;
  }
}

int semaphore::wait(int need, uint32_t *timeoutUs) {
  if (need <= 0)
    return -SysError::eINVAL;
  std::unique_lock<std::mutex> lk(m);
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
  if (count < need)
    return -SysError::eBUSY;
  count -= need;
  return 0;
}

int semaphore::post(int n) {
  if (n <= 0)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(m);
  // Cap at the declared maximum (SCE rejects an over-post; clamping is the safe
  // approximation (never deadlock a waiter, never grow unbounded).
  count += n;
  if (maxCount > 0 && count > maxCount)
    count = maxCount;
  cv.notify_all();
  return 0;
}

int semaphore::cancel(int setCount, int *numWaiters) {
  std::lock_guard<std::mutex> lk(m);
  if (numWaiters)
    *numWaiters = waiters;
  if (setCount >= 0)
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

int PS4ABI sys_osem_create(const char *name, uint32_t attr, int init, int max) {
  auto *s = new semaphore(proc::getActive(), name, init, max);
  std::printf("[osem] create '%s' attr=%#x init=%d max=%d -> id=%u\n",
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
  std::printf("[osem] open '%s' (auto-created) -> id=%u\n", name ? name : "",
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

int PS4ABI sys_osem_wait(int id, int need, uint32_t *timeoutUs) {
  WaitProbe _wp("osem_wait", (long)id, (long)need);
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
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
  return s->post(count);
}

int PS4ABI sys_osem_cancel(int id, int setCount, int *numWaiters) {
  auto *s = fromId(id);
  if (!s)
    return -SysError::eSRCH;
  return s->cancel(setCount, numWaiters);
}
}  // namespace krnl
