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
#include <unordered_map>

#include "kern/proc.h"
#include "sys_event_flag.h"

namespace krnl {
// SCE_KERNEL_EVF_WAITMODE_*
enum { kEvfAnd = 0x01, kEvfOr = 0x02, kEvfClearAll = 0x10, kEvfClearPat = 0x20 };

// Named event flags, so evf_open(name) finds the one evf_create(name) made.
static std::mutex g_efRegM;
static std::unordered_map<std::string, eventFlag *> g_efByName;

eventFlag::eventFlag(proc *p, const char *nm, uint64_t init)
    : kObject(p, oType::eventflag), bits(init) {
  if (nm && *nm) {
    name = nm;
    std::lock_guard<std::mutex> lk(g_efRegM);
    g_efByName[nm] = this;
  }
}

bool eventFlag::satisfied(uint64_t pattern, uint32_t mode) const {
  return (mode & kEvfOr) ? (bits & pattern) != 0 : (bits & pattern) == pattern;
}

int eventFlag::take(uint64_t pattern, uint32_t mode, uint64_t *result) {
  if (result)
    *result = bits;
  if (mode & kEvfClearAll)
    bits = 0;
  else if (mode & kEvfClearPat)
    bits &= ~pattern;
  return 0;
}

int eventFlag::wait(uint64_t pattern, uint32_t mode, uint64_t *result,
                    uint32_t *timeoutUs) {
  std::unique_lock<std::mutex> lk(m);
  if (satisfied(pattern, mode))
    return take(pattern, mode, result);
  if (timeoutUs) {
    if (!cv.wait_for(lk, std::chrono::microseconds(*timeoutUs),
                     [&] { return satisfied(pattern, mode); }))
      return -SysError::eTIMEDOUT;
  } else {
    cv.wait(lk, [&] { return satisfied(pattern, mode); });
  }
  return take(pattern, mode, result);
}

int eventFlag::trywait(uint64_t pattern, uint32_t mode, uint64_t *result) {
  std::unique_lock<std::mutex> lk(m);
  if (!satisfied(pattern, mode))
    return -SysError::eBUSY;
  return take(pattern, mode, result);
}

void eventFlag::set(uint64_t b) {
  std::lock_guard<std::mutex> lk(m);
  bits |= b;
  cv.notify_all();
}

void eventFlag::clear(uint64_t b) {
  std::lock_guard<std::mutex> lk(m);
  bits &= b;  // SCE clear keeps the bits set in b
}

static eventFlag *fromId(int id) {
  auto *obj = proc::getActive()->getObjTable().get(id);
  if (!obj || obj->type() != kObject::oType::eventflag)
    return nullptr;
  return static_cast<eventFlag *>(obj);
}

int PS4ABI sys_evf_create(const char *name, uint32_t attr,
                          uint64_t initPattern) {
  auto *ef = new eventFlag(proc::getActive(), name, initPattern);
  std::printf("[evf] create '%s' attr=%#x init=%#llx -> id=%u\n",
              name ? name : "", attr, (unsigned long long)initPattern,
              ef->handle());
  return ef->handle();
}

// Some system-service event flags gate the game on state the ShellCore would
// publish (app focus granted, power normal, system running). With no ShellCore
// the flag stays 0 and the game's "wait for focus/ready" (EVF OR-wait for any
// bit) blocks forever. Seed those flags as "focused/ready" so the game proceeds.
static uint64_t systemFlagInit(const char *name) {
  if (!name)
    return 0;
  base::StringRef n(name);
  if (n.find("AppFocus", 0, 8) != base::StringRef::npos ||
      n.find("CtrlFocus", 0, 9) != base::StringRef::npos ||
      n.find("PowerControl", 0, 12) != base::StringRef::npos ||
      n.find("SystemStateMgr", 0, 14) != base::StringRef::npos)
    return 0x1;  // bit0 = focused / powered / running
  return 0;
}

int PS4ABI sys_evf_open(const char *name) {
  {
    std::lock_guard<std::mutex> lk(g_efRegM);
    auto it = name ? g_efByName.find(name) : g_efByName.end();
    if (it != g_efByName.end())
      return it->second->handle();
  }
  // Auto-create unknown named flags: on real hw a system service creates them;
  // here both producer and consumer just open by name, so creating on first
  // open gives them a shared flag and the sync actually works.
  auto *ef = new eventFlag(proc::getActive(), name, systemFlagInit(name));
  std::printf("[evf] open '%s' (auto-created) -> id=%u\n", name ? name : "",
              ef->handle());
  return ef->handle();
}

int PS4ABI sys_evf_delete(int id) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  {
    std::lock_guard<std::mutex> lk(g_efRegM);
    if (!ef->fname().empty())
      g_efByName.erase(ef->fname().c_str());
  }
  proc::getActive()->getObjTable().release(id);
  return 0;
}

int PS4ABI sys_evf_close(int id) { return sys_evf_delete(id); }

int PS4ABI sys_evf_wait(int id, uint64_t pattern, uint32_t mode,
                        uint64_t *result, uint32_t *timeoutUs) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  return ef->wait(pattern, mode, result, timeoutUs);
}

int PS4ABI sys_evf_trywait(int id, uint64_t pattern, uint32_t mode,
                           uint64_t *result) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  return ef->trywait(pattern, mode, result);
}

int PS4ABI sys_evf_set(int id, uint64_t bits) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->set(bits);
  return 0;
}

int PS4ABI sys_evf_clear(int id, uint64_t bits) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->clear(bits);
  return 0;
}

int PS4ABI sys_evf_cancel(int id, uint64_t pattern, int *numWaiters) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->set(pattern);  // wake waiters with the pattern (best-effort)
  if (numWaiters)
    *numWaiters = 0;
  return 0;
}
}  // namespace krnl
