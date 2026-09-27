/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Guest thread naming. Titles rarely call sys_thr_set_name; the common
 * convention is tagging the thread's STACK via sys_mname ("RenderFlip",
 * "Game Render", ...). Carry those tags onto the host threads so every
 * host-side view of the process (the wait probe, /proc, gdb, perf)
 * shows which guest thread is which.
 *
 * Two directions, because the tag and the thread can start in either order:
 *  - RegisterGuestThreadStack: a starting guest thread names itself from the
 *    VMA tag covering its stack, if one landed already.
 *  - NameThreadsForRange: a later sys_mname tag covering a live registered
 *    stack renames that thread.
 */

#include "kern/thread_names.h"
#include "base/arch.h"

#include <cstddef>
#include <cstring>

#include <pthread.h>

#include "base/containers/vector.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/process.h"

namespace krnl {
namespace {

struct StackEntry {
  const void* stack;
  size_t size;
  pthread_t thread;
};

base::Mutex g_mtx;
base::Vector<StackEntry> g_stacks;

void SetName(pthread_t t, const char* name) {
  if (!name || !*name)
    return;
  char buf[16];  // pthread_setname_np limit: 15 chars + NUL
  std::strncpy(buf, name, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  pthread_setname_np(t, buf);
}

}  // namespace

void RegisterGuestThreadStack(const void* stack, size_t size) {
  if (!stack || !size)
    return;
  {
    base::LockGuard<base::Mutex> lk(g_mtx);
    g_stacks.push_back({stack, size, pthread_self()});
  }
  // The creating thread usually tags the stack before starting the thread;
  // pick that tag up now.
  if (auto* proc = Proc::GetActive()) {
    auto* info =
        proc->GetVma().Get(const_cast<u8*>(static_cast<const u8*>(stack)));
    if (info && info->name)
      SetName(pthread_self(), info->name);
  }
}

void UnregisterGuestThreadStack() {
  const pthread_t self = pthread_self();
  base::LockGuard<base::Mutex> lk(g_mtx);
  for (auto it = g_stacks.begin(); it != g_stacks.end(); ++it) {
    if (pthread_equal(it->thread, self)) {
      g_stacks.erase(it);
      return;
    }
  }
}

void NameThreadsForRange(const void* ptr, size_t len, const char* name) {
  if (!ptr || !len || !name || !*name)
    return;
  // "stack guard" tags land adjacent to the real stack tag; naming a thread
  // after its guard page would overwrite the useful name.
  if (std::strcmp(name, "stack guard") == 0)
    return;
  const auto* lo = static_cast<const u8*>(ptr);
  const auto* hi = lo + len;
  base::LockGuard<base::Mutex> lk(g_mtx);
  // CONTAINMENT, not overlap. A title tags a region that can span several
  // guest stacks (SotC's "Resource Loading" tag covers a range overlapping the
  // FIOS and job-worker stacks), and renaming on any overlap gave four
  // unrelated threads the same name, which is worse than no name, because it
  // makes a wait-probe report look like one subsystem is wedged four times.
  // Only rename a thread whose whole stack lies inside the tagged range.
  for (const auto& e : g_stacks) {
    const auto* slo = static_cast<const u8*>(e.stack);
    const auto* shi = slo + e.size;
    if (slo >= lo && shi <= hi)
      SetName(e.thread, name);
  }
}

}  // namespace krnl
