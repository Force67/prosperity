#include "gpu/ps5/label_publisher.h"

#include "base/containers/vector.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "gpu/render/renderer.h"

namespace gpu::ps5 {
namespace {
struct Held {
  u64 batch = 0;  // 0: nothing of its own to wait for
  base::Function<void()> write;
};

base::Mutex g_lock;
base::ConditionVariable g_wake;  // a write was held
base::ConditionVariable g_done;  // a held write ran
base::Vector<Held> g_held;
size_t g_head = 0;  // g_held[g_head..] have not run
base::UniquePointer<base::Thread> g_thread;

void Run() {
  for (;;) {
    Held item;
    {
      base::UniqueLock<base::Mutex> lock(g_lock);
      g_wake.Wait(lock, [] { return g_head < g_held.size(); });
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
  }
}
}  // namespace

void PublishLabel(base::Function<void()> write) {
  const u64 point = render::GuestWritePublishBatch();
  {
    base::LockGuard<base::Mutex> lock(g_lock);
    if (point || g_head < g_held.size()) {
      g_held.push_back({point, base::move(write)});
      if (!g_thread)
        g_thread = base::MakeUnique<base::Thread>(
            "agc-labels", [] { Run(); }, /*start_now=*/true);
      write = nullptr;
    }
  }
  if (write)
    write();
  else
    g_wake.NotifyAll();
}

void DrainLabels() {
  // The held labels may wait for the batch this thread has not submitted yet.
  render::SubmitForPublishedLabels();
  base::UniqueLock<base::Mutex> lock(g_lock);
  g_done.Wait(lock, [] { return g_head == g_held.size(); });
}
}  // namespace gpu::ps5
