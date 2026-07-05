/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include "error_table.h"
#include "kern/proc.h"
#include "sys_ksem.h"

namespace krnl {
// A single FreeBSD POSIX kernel semaphore. Its own mutex/cv so different ksems
// never contend on a global lock while a thread sits blocked in wait().
struct KSem {
  std::mutex m;
  std::condition_variable cv;
  int value;
  explicit KSem(int v) : value(v) {}
};

// Registry. g_ksemMutex guards only the two maps (id->KSem*, name->id); it is
// never held while blocking on a KSem's own cv, so a sleeping waiter can't wedge
// other ksem operations.
static std::mutex g_ksemMutex;
static std::unordered_map<int, KSem *> g_ksemById;
static std::unordered_map<std::string, int> g_ksemByName;
static std::atomic<int> g_nextKsemId{1};

// Resolve an id to its KSem under the registry lock, then release it: callers
// operate on the returned object's own mutex, so we never hold both at once.
static KSem *fromId(int id) {
  std::lock_guard<std::mutex> lk(g_ksemMutex);
  auto it = g_ksemById.find(id);
  return it == g_ksemById.end() ? nullptr : it->second;
}

// FreeBSD oflag bits relevant to ksem_open.
static constexpr int kO_CREAT = 0x0200;
static constexpr int kO_EXCL = 0x0800;

int PS4ABI sys_ksem_init(int *idp, unsigned value) {
  auto *s = new KSem(static_cast<int>(value));
  int id = g_nextKsemId.fetch_add(1);
  {
    std::lock_guard<std::mutex> lk(g_ksemMutex);
    g_ksemById[id] = s;
  }
  if (idp)
    *idp = id;
  std::printf("[ksem] init value=%u -> id=%d\n", value, id);
  return 0;
}

int PS4ABI sys_ksem_open(int *idp, const char *name, int oflag, uint16_t mode,
                         unsigned value) {
  std::string key = name ? name : "";
  std::lock_guard<std::mutex> lk(g_ksemMutex);
  auto it = g_ksemByName.find(key);
  if (it != g_ksemByName.end()) {
    // Name exists. O_CREAT|O_EXCL together demand exclusive creation -> fail.
    if ((oflag & kO_CREAT) && (oflag & kO_EXCL))
      return -SysError::eEXIST;
    if (idp)
      *idp = it->second;
    std::printf("[ksem] open '%s' (existing) -> id=%d\n", key.c_str(),
                it->second);
    return 0;
  }
  // Not found: without O_CREAT this is an error.
  if (!(oflag & kO_CREAT))
    return -SysError::eNOENT;
  auto *s = new KSem(static_cast<int>(value));
  int id = g_nextKsemId.fetch_add(1);
  g_ksemById[id] = s;
  g_ksemByName[key] = id;
  if (idp)
    *idp = id;
  std::printf("[ksem] open '%s' (created) value=%u mode=%#x -> id=%d\n",
              key.c_str(), value, mode, id);
  return 0;
}

int PS4ABI sys_ksem_unlink(const char *name) {
  std::string key = name ? name : "";
  std::lock_guard<std::mutex> lk(g_ksemMutex);
  auto it = g_ksemByName.find(key);
  if (it == g_ksemByName.end())
    return -SysError::eNOENT;
  // Only drop the name binding; the KSem object stays alive for any process that
  // still holds the id (POSIX unlink semantics: future opens won't find it, but
  // existing references keep working).
  g_ksemByName.erase(it);
  return 0;
}

int PS4ABI sys_ksem_close(int id) {
  // We intentionally keep the object: an unnamed ksem could still be in use, and
  // tracking per-handle refcounts isn't worth it here; everything is reclaimed
  // at process exit. So close is a no-op success.
  (void)id;
  return 0;
}

int PS4ABI sys_ksem_post(int id) {
  KSem *s = fromId(id);
  if (!s)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(s->m);
  s->value++;
  s->cv.notify_one();
  return 0;
}

int PS4ABI sys_ksem_wait(int id) {
  KSem *s = fromId(id);
  if (!s)
    return -SysError::eINVAL;
  std::unique_lock<std::mutex> lk(s->m);
  s->cv.wait(lk, [&] { return s->value > 0; });
  s->value--;
  return 0;
}

int PS4ABI sys_ksem_trywait(int id) {
  KSem *s = fromId(id);
  if (!s)
    return -SysError::eINVAL;
  std::lock_guard<std::mutex> lk(s->m);
  if (s->value == 0)
    return -SysError::eAGAIN;
  s->value--;
  return 0;
}

int PS4ABI sys_ksem_timedwait(int id, const struct ksem_timespec *abstime) {
  KSem *s = fromId(id);
  if (!s)
    return -SysError::eINVAL;
  // Null deadline -> block forever, same as ksem_wait.
  if (!abstime)
    return sys_ksem_wait(id);

  // abstime is an ABSOLUTE CLOCK_REALTIME deadline. Convert it to a relative
  // duration against now and feed cv.wait_for; computing a relative timeout (vs.
  // a system_clock::time_point) avoids tangling guest epoch assumptions with the
  // host clock representation.
  auto now = std::chrono::system_clock::now().time_since_epoch();
  auto deadline = std::chrono::seconds(abstime->tv_sec) +
                  std::chrono::nanoseconds(abstime->tv_nsec);
  auto rel = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline) -
             std::chrono::duration_cast<std::chrono::nanoseconds>(now);
  if (rel.count() < 0)
    rel = std::chrono::nanoseconds(0);  // already expired -> poll once

  std::unique_lock<std::mutex> lk(s->m);
  if (!s->cv.wait_for(lk, rel, [&] { return s->value > 0; }))
    return -SysError::eTIMEDOUT;
  s->value--;
  return 0;
}

int PS4ABI sys_ksem_getvalue(int id, int *val) {
  KSem *s = fromId(id);
  if (!s)
    return -SysError::eINVAL;
  if (val) {
    std::lock_guard<std::mutex> lk(s->m);
    *val = s->value;
  }
  return 0;
}

int PS4ABI sys_ksem_destroy(int id) {
  KSem *s = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_ksemMutex);
    auto it = g_ksemById.find(id);
    if (it == g_ksemById.end())
      return -SysError::eINVAL;
    s = it->second;
    g_ksemById.erase(it);
    // Drop any name binding still pointing at this id so the slot is fully gone.
    for (auto nit = g_ksemByName.begin(); nit != g_ksemByName.end();) {
      if (nit->second == id)
        nit = g_ksemByName.erase(nit);
      else
        ++nit;
    }
  }
  // FreeBSD returns EBUSY if waiters remain; we skip that check and destroy
  // unconditionally. A blocked waiter holds s->m, so taking it here serializes
  // against an in-flight wait before we free, avoiding a use-after-free.
  {
    std::lock_guard<std::mutex> lk(s->m);
  }
  delete s;
  return 0;
}
}  // namespace krnl
