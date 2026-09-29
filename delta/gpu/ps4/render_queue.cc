/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/render_queue.h"

#include "gpu/render/renderer.h"

#include <pthread.h>

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/map.h"
#include "base/functional/function.h"
#include "base/logging.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "options/options.h"
#include "profile/profile.h"

namespace gpu::ps4 {

namespace {
DELTA_OPTION(bool, kDrainTrace, "DELTA_GPU_DRAINTRACE", false);
}

RenderQueue::~RenderQueue() {
  if (!running_)
    return;
  stop_.store(true);
  Wake(render_wake_);
  thread_->Join();
}

void RenderQueue::Start(render::Renderer& renderer) {
  if (running_)
    return;
  renderer_ = &renderer;
  commands_ = base::MakeUnique<Command[]>(kCommands);
  draws_ = base::MakeUnique<render::DrawInfo[]>(kDrawSlots);
  thread_ = base::MakeUnique<base::Thread>(
      "gpu-render", [this] { Run(); },
      /*start_now=*/true);
  running_ = true;
}

render::DrawInfo& RenderQueue::NextDraw() {
  if (draw_head_ >= kDrawSlots)
    WaitDone(draws_done_, draw_head_ - kDrawSlots + 1, "draw-slot");
  render::DrawInfo& d = draws_[draw_head_ % kDrawSlots];
  d.Reset();
  return d;
}

RenderQueue::Command& RenderQueue::Claim() {
  const u64 head = head_.load(base::memory_order_relaxed);
  if (head >= kCommands)
    WaitDone(done_, head - kCommands + 1, "command-slot");
  return commands_[head % kCommands];
}

void RenderQueue::Publish() {
  head_.fetch_add(1, base::memory_order_release);
  Wake(render_wake_);
}

void RenderQueue::PushDraw() {
  Command& c = Claim();
  c.kind = Kind::kDraw;
  c.arg = draw_head_++ % kDrawSlots;
  Publish();
}

void RenderQueue::PushBeginFrame() {
  Claim().kind = Kind::kBeginFrame;
  Publish();
}

void RenderQueue::PushEndFrame(u64 scanout_base) {
  Command& c = Claim();
  c.kind = Kind::kEndFrame;
  c.arg = scanout_base;
  Publish();
  const u64 previous = last_end_frame_;
  last_end_frame_ = head_.load(base::memory_order_relaxed);
  WaitDone(done_, previous, "end-frame");
}

void RenderQueue::PushCall(base::Function<void()> fn) {
  Command& c = Claim();
  c.kind = Kind::kCall;
  c.fn = base::move(fn);
  Publish();
}

void RenderQueue::Drain(const char* why) {
  DELTA_ZONE("gpuq.drain");
  // The renderer's own thread already owns it (a replay it runs may ask).
  if (!running_ || base::IsCurrentThread(thread_->handle()))
    return;
  const u64 head = head_.load(base::memory_order_relaxed);
  if (kDrainTrace && done_.load(base::memory_order_acquire) != head) {
    static base::Map<const char*, u64> counts;
    static u64 n = 0;
    counts[why]++;
    if (++n % 2000 == 0) {
      base::String line;
      for (const auto& [k, v] : counts)
        base::FormatTo(line, " {}={}", k, v);
      BASE_LOGI("drain", "waited{}", line.c_str());
    }
  }
  WaitDone(done_, head, why);
}

void RenderQueue::Wake(base::ConditionVariable& cv) {
  // Through the lock: a waiter between its check and its sleep holds it, so
  // the change it missed cannot also miss it.
  {
    base::LockGuard<base::Mutex> lock(wake_mutex_);
  }
  cv.NotifyAll();
}

void RenderQueue::WaitDone(const base::Atomic<u64>& done,
                           u64 target,
                           const char* what) {
  DELTA_ZONE("gpuq.wait_done");
  u64 seen = done.load(base::memory_order_acquire);
  if (seen >= target)
    return;
  const auto start = base::TimeTicks::Now();
  {
    base::UniqueLock<base::Mutex> lock(wake_mutex_);
    walk_wake_.Wait(
        lock, [&] { return done.load(base::memory_order_acquire) >= target; });
  }
  const auto waited = base::TimeTicks::Now() - start;
  if (waited > base::Milliseconds(500)) {
    static const char* const kKinds[] = {"draw", "begin-frame", "end-frame",
                                         "call"};
    BASE_LOGW("gpuq", "walk waited {} ms for {}; renderer thread last ran {}",
              (waited).InMilliseconds(), what,
              kKinds[running_kind_.load() & 3]);
  }
}

void RenderQueue::Run() {
  for (u64 tail = 0;; tail++) {
    if (head_.load(base::memory_order_acquire) == tail) {
      base::UniqueLock<base::Mutex> lock(wake_mutex_);
      render_wake_.Wait(lock, [&] {
        return stop_.load() || head_.load(base::memory_order_acquire) != tail;
      });
      if (head_.load(base::memory_order_acquire) == tail)
        return;  // stopped
    }
    Command& c = commands_[tail % kCommands];
    running_kind_.store(static_cast<u8>(c.kind), base::memory_order_relaxed);
    const auto started = base::TimeTicks::Now();
    switch (c.kind) {
      case Kind::kDraw:
        render::Draw(*renderer_, draws_[c.arg]);
        draws_done_.fetch_add(1, base::memory_order_release);
        Wake(walk_wake_);
        break;
      case Kind::kBeginFrame:
        render::BeginFrame(*renderer_);
        break;
      case Kind::kEndFrame:
        render::EndFrame(*renderer_, c.arg);
        break;
      case Kind::kCall:
        c.fn();
        c.fn = nullptr;
        break;
    }
    const auto took = base::TimeTicks::Now() - started;
    if (took > base::Milliseconds(500))
      BASE_LOGW("gpuq", "renderer thread spent {} ms in one {} command",
                (took).InMilliseconds(), static_cast<int>(c.kind));
    done_.store(tail + 1, base::memory_order_release);
    Wake(walk_wake_);
  }
}

void RenderQueue::NotePendingWrite(u64 base, u64 bytes) {
  if (running_ && bytes)
    pending_writes_.push_back(
        {head_.load(base::memory_order_relaxed), base, base + bytes});
}

bool RenderQueue::PendingWriteOverlaps(u64 base, u64 bytes) {
  if (pending_writes_.empty())
    return false;
  const u64 done = done_.load(base::memory_order_acquire);
  base::EraseIf(pending_writes_,
                [done](const PendingWrite& w) { return w.command < done; });
  const u64 end = base + bytes;
  for (const PendingWrite& w : pending_writes_)
    if (w.first < end && base < w.end) {
      if (kDrainTrace) {
        static int shown = 0;
        if ((shown++ % 64) == 0)
          BASE_LOGI("drain", "read {:#x}+{:#x} hits pending {:#x}+{:#x}", base,
                    bytes, w.first, w.end - w.first);
      }
      return true;
    }
  return false;
}

RenderQueue& GuestRenderQueue() {
  // Never destroyed: its thread may still be inside the renderer at exit.
  static RenderQueue* queue = new RenderQueue;
  return *queue;
}

void FlushForWalkRead(render::Renderer& renderer,
                      u64 base,
                      u64 bytes,
                      const char* why) {
  // Only dispatches make a range dirty, and they run on the walk's side of
  // the queue, so a clear filter cannot be contradicted by queued work.
  RenderQueue& queue = GuestRenderQueue();
  if (queue.running() && queue.PendingWriteOverlaps(base, bytes))
    OwnRenderer("pending-write");
  if (queue.running() && !render::CsRangeMaybeDirty(base, bytes))
    return;
  OwnRenderer(why);
  render::FlushCsWritesRange(renderer, base, bytes, why);
}

}  // namespace gpu::ps4
