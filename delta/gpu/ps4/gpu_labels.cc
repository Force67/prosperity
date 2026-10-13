/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/gpu_labels.h"
#include "guest/session.h"

#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/render/renderer.h"
#include "options/options.h"

namespace gpu::ps4 {
namespace {
DELTA_OPTION(bool, kLabelsAtGpu, "DELTA_GPU_LABELS_AT_GPU", false);

struct Held {
  u64 submission = 0;
  u64 address = 0;
  u32 bytes = 0;
  u64 value = 0;
  base::Function<void()> write;
};

base::Mutex g_lock;
base::ConditionVariable g_wake;
base::Vector<Held> g_held;
size_t g_head = 0;
size_t g_unsubmitted = 0;
bool g_started = false;

void Run() {
  for (;;) {
    u64 submission = 0;
    base::Function<void()> write;
    {
      base::UniqueLock<base::Mutex> lock(g_lock);
      if (!guest::Wait(g_wake, lock, [] {
            return g_head < g_unsubmitted && g_held[g_head].submission;
          }))
        return;
      submission = g_held[g_head].submission;
      write = base::move(g_held[g_head].write);
    }
    // A lost device still writes; a title waiting on the fence would hang.
    render::WaitSubmission(submission);
    write();
    base::LockGuard<base::Mutex> lock(g_lock);
    if (++g_head == g_held.size()) {
      g_held.clear();
      g_head = g_unsubmitted = 0;
    }
  }
}
}  // namespace

void WriteAfterGpu(base::Function<void()> write,
                   u64 address,
                   u32 bytes,
                   u64 value) {
  if (!kLabelsAtGpu) {
    write();
    return;
  }
  base::LockGuard<base::Mutex> lock(g_lock);
  Held& held = g_held.emplace_back();
  held.address = address;
  held.bytes = bytes;
  held.value = value;
  held.write = base::move(write);
  if (!g_started) {
    g_started = true;
    guest::SpawnThread("gnm-labels", [] { Run(); });
  }
}

void SubmitHeldWrites() {
  {
    base::LockGuard<base::Mutex> lock(g_lock);
    if (g_unsubmitted == g_held.size())
      return;
  }
  const u64 submission = render::SubmitRecordedWork();
  {
    base::LockGuard<base::Mutex> lock(g_lock);
    for (; g_unsubmitted < g_held.size(); g_unsubmitted++)
      g_held[g_unsubmitted].submission = submission ? submission : 1;
  }
  g_wake.NotifyAll();
}

bool HeldWriteValue(u64 address, u32* value) {
  base::LockGuard<base::Mutex> lock(g_lock);
  bool found = false;
  for (size_t i = g_head; i < g_held.size(); i++) {
    const Held& h = g_held[i];
    if (h.bytes && h.bytes <= 8 && h.address <= address &&
        address + 4 <= h.address + h.bytes) {
      *value = static_cast<u32>(h.value >> ((address - h.address) * 8));
      found = true;
    }
  }
  return found;
}

bool HeldWriteOverlaps(u64 address, u64 bytes) {
  base::LockGuard<base::Mutex> lock(g_lock);
  for (size_t i = g_head; i < g_held.size(); i++)
    if (g_held[i].address < address + bytes &&
        address < g_held[i].address + base::Max<u64>(g_held[i].bytes, 1))
      return true;
  return false;
}

namespace {
const guest::SessionReset g_session_reset([] {
  g_held = {};
  g_head = g_unsubmitted = 0;
  g_started = false;
});
}  // namespace
}  // namespace gpu::ps4
