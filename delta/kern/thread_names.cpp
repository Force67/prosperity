/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Guest thread naming. Titles rarely call sys_thr_set_name; the common
 * convention is tagging the thread's STACK via sys_mname ("RenderFlip",
 * "Game Render", ...). Carry those tags onto the host threads so every
 * host-side view of the process -- the wait probe, /proc, gdb, perf --
 * shows which guest thread is which.
 *
 * Two directions, because the tag and the thread can start in either order:
 *  - registerGuestThreadStack: a starting guest thread names itself from the
 *    VMA tag covering its stack, if one landed already.
 *  - nameThreadsForRange: a later sys_mname tag covering a live registered
 *    stack renames that thread.
 */

#include "thread_names.h"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

#include <pthread.h>

#include "proc.h"

namespace krnl {
namespace {

struct StackEntry {
  const void *stack;
  size_t size;
  pthread_t thread;
};

std::mutex g_mtx;
std::vector<StackEntry> g_stacks;

void setName(pthread_t t, const char *name) {
  if (!name || !*name)
    return;
  char buf[16];  // pthread_setname_np limit: 15 chars + NUL
  std::strncpy(buf, name, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  pthread_setname_np(t, buf);
}

}  // namespace

void registerGuestThreadStack(const void *stack, size_t size) {
  if (!stack || !size)
    return;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_stacks.push_back({stack, size, pthread_self()});
  }
  // The creating thread usually tags the stack before starting the thread;
  // pick that tag up now.
  if (auto *proc = proc::getActive()) {
    auto *info = proc->getVma().get(
        const_cast<uint8_t *>(static_cast<const uint8_t *>(stack)));
    if (info && info->name)
      setName(pthread_self(), info->name);
  }
}

void unregisterGuestThreadStack() {
  const pthread_t self = pthread_self();
  std::lock_guard<std::mutex> lk(g_mtx);
  for (auto it = g_stacks.begin(); it != g_stacks.end(); ++it) {
    if (pthread_equal(it->thread, self)) {
      g_stacks.erase(it);
      return;
    }
  }
}

void nameThreadsForRange(const void *ptr, size_t len, const char *name) {
  if (!ptr || !len || !name || !*name)
    return;
  // "stack guard" tags land adjacent to the real stack tag; naming a thread
  // after its guard page would overwrite the useful name.
  if (std::strcmp(name, "stack guard") == 0)
    return;
  const auto *lo = static_cast<const uint8_t *>(ptr);
  const auto *hi = lo + len;
  std::lock_guard<std::mutex> lk(g_mtx);
  // CONTAINMENT, not overlap. A title tags a region that can span several
  // guest stacks (SotC's "Resource Loading" tag covers a range overlapping the
  // FIOS and job-worker stacks), and renaming on any overlap gave four
  // unrelated threads the same name -- which is worse than no name, because it
  // makes a wait-probe report look like one subsystem is wedged four times.
  // Only rename a thread whose whole stack lies inside the tagged range.
  for (const auto &e : g_stacks) {
    const auto *slo = static_cast<const uint8_t *>(e.stack);
    const auto *shi = slo + e.size;
    if (slo >= lo && shi <= hi)
      setName(e.thread, name);
  }
}

}  // namespace krnl
