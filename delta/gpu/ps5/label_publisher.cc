#include "gpu/ps5/label_publisher.h"
#include "guest/session.h"

#include <cstring>

#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "gpu/render/renderer.h"

namespace gpu::ps5 {
namespace {
struct Held {
  u64 batch = 0;  // 0: nothing of its own to wait for
  u64 lo = 0, hi = 0;     // the guest bytes it writes
  base::Vector<u8> data;  // what it writes there, when known
  base::Function<void()> write;
};

base::Mutex g_lock;
base::ConditionVariable g_wake;  // a write was held
base::ConditionVariable g_done;  // a held write ran
base::Vector<Held> g_held;
size_t g_head = 0;  // g_held[g_head..] have not run

bool g_started = false;
base::Mutex g_event_lock;
base::ConditionVariable g_event;
base::Atomic<u64> g_event_seq{0};

void Run() {
  for (;;) {
    Held item;
    {
      base::UniqueLock<base::Mutex> lock(g_lock);
      if (!guest::Wait(g_wake, lock, [] { return g_head < g_held.size(); }))
        return;
      item.batch = g_held[g_head].batch;
      item.write = base::move(g_held[g_head].write);
    }
    if (item.batch)
      render::WaitBatch(item.batch);
    item.write();
    {
      base::LockGuard<base::Mutex> lock(g_lock);
      // Counted as run only now, so a caller that finds the queue empty is
      // ordered after this write.
      if (++g_head == g_held.size()) {
        g_held.clear();
        g_head = 0;
      }
    }
    g_done.NotifyAll();
    NoteLabelWritten();
  }
}
}  // namespace

u64 LabelSequence() {
  return g_event_seq.load(base::memory_order_acquire);
}

void NoteLabelWritten() {
  {
    base::LockGuard<base::Mutex> lock(g_event_lock);
    g_event_seq.fetch_add(1, base::memory_order_acq_rel);
  }
  g_event.NotifyAll();
}

void WaitLabelWritten(u64 seen, base::TimeDelta timeout) {
  base::UniqueLock<base::Mutex> lock(g_event_lock);
  g_event.WaitFor(lock, timeout, [&] {
    return g_event_seq.load(base::memory_order_acquire) != seen;
  });
}

void PublishLabel(base::Function<void()> write,
                  u64 base,
                  u64 bytes,
                  const void* data) {
  // Staged shader outputs must reach guest memory before their completion
  // labels let the CPU reuse or free those buffers.
  render::FlushCsWrites(render::DefaultRenderer(), "label");
  const u64 point = render::GuestWritePublishBatch();
  {
    base::LockGuard<base::Mutex> lock(g_lock);
    if (point || g_head < g_held.size()) {
      Held& held = g_held.emplace_back();
      held.batch = point;
      held.lo = base;
      held.hi = base + bytes < base ? ~0ull : base + bytes;
      if (data)
        held.data.assign(static_cast<const u8*>(data),
                         static_cast<const u8*>(data) + bytes);
      held.write = base::move(write);
      if (!g_started) {
        g_started = true;
        guest::SpawnThread("agc-labels", [] { Run(); });
      }
      write = nullptr;
    }
  }
  if (write) {
    write();
    NoteLabelWritten();
  } else {
    g_wake.NotifyAll();
  }
}

void ReportHeldLabels() {
  base::LockGuard<base::Mutex> lock(g_lock);
  base::String line;
  for (size_t i = g_head; i < g_held.size() && i < g_head + 8; i++)
    base::FormatTo(line, " {}", g_held[i].batch);
  BASE_LOGW("agc", "held labels: {} waiting on batches [{} ]",
            g_held.size() - g_head, line.c_str());
  render::ReportBatchState();
}

bool OverlayHeld(u64 base, u32 bytes, u8* out) {
  base::LockGuard<base::Mutex> lock(g_lock);
  for (size_t i = g_head; i < g_held.size(); i++) {
    const Held& h = g_held[i];
    if (h.hi <= base || base + bytes <= h.lo)
      continue;
    if (h.data.empty())
      return false;
    const u64 lo = base::Max(base, h.lo), hi = base::Min(base + bytes, h.hi);
    std::memcpy(out + (lo - base), h.data.data() + (lo - h.lo), hi - lo);
  }
  return true;
}

bool LabelPending(u64 base, u64 bytes) {
  bool pending = false;
  {
    base::LockGuard<base::Mutex> lock(g_lock);
    for (size_t i = g_head; i < g_held.size() && !pending; i++)
      pending = g_held[i].lo < base + bytes && base < g_held[i].hi;
  }
  // The held write may wait for the batch this thread has not submitted yet.
  if (pending)
    render::SubmitForPublishedLabels();
  return pending;
}
namespace {
const bool kSessionReset = [] {
  guest::RegisterSessionReset([] {
    g_held = {};
    g_head = 0;
    g_started = false;
    g_event_seq.store(0);
  });
  return true;
}();
}  // namespace
}  // namespace gpu::ps5
