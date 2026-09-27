/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/render_queue.h"

#include "gpu/render/renderer.h"

#include <chrono>
#include <map>
#include <pthread.h>

#include <base/logging.h>
#include <base/strings/format.h>
#include <utl/options.h>

namespace gpu::ps4 {

namespace {
DELTA_OPTION(bool, kDrainTrace, "DELTA_GPU_DRAINTRACE", false);
}

RenderQueue::~RenderQueue() {
  if (!running_)
    return;
  stop_.store(true);
  head_.notify_all();
  thread_.join();
}

void RenderQueue::Start(render::Renderer& renderer) {
  if (running_)
    return;
  renderer_ = &renderer;
  commands_ = std::make_unique<Command[]>(kCommands);
  draws_ = std::make_unique<render::DrawInfo[]>(kDrawSlots);
  thread_ = std::thread([this] { Run(); });
  pthread_setname_np(thread_.native_handle(), "gpu-render");
  running_ = true;
}

render::DrawInfo& RenderQueue::NextDraw() {
  if (draw_head_ >= kDrawSlots)
    WaitDone(draws_done_, draw_head_ - kDrawSlots + 1, "draw-slot");
  render::DrawInfo& d = draws_[draw_head_ % kDrawSlots];
  d = render::DrawInfo{};
  return d;
}

RenderQueue::Command& RenderQueue::Claim() {
  const u64 head = head_.load(std::memory_order_relaxed);
  if (head >= kCommands)
    WaitDone(done_, head - kCommands + 1, "command-slot");
  return commands_[head % kCommands];
}

void RenderQueue::Publish() {
  head_.fetch_add(1, std::memory_order_release);
  head_.notify_one();
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
  last_end_frame_ = head_.load(std::memory_order_relaxed);
  WaitDone(done_, previous, "end-frame");
}

void RenderQueue::PushCall(std::function<void()> fn) {
  Command& c = Claim();
  c.kind = Kind::kCall;
  c.fn = std::move(fn);
  Publish();
}

void RenderQueue::Drain(const char* why) {
  // The renderer's own thread already owns it (a replay it runs may ask).
  if (!running_ || std::this_thread::get_id() == thread_.get_id())
    return;
  const u64 head = head_.load(std::memory_order_relaxed);
  if (kDrainTrace && done_.load(std::memory_order_acquire) != head) {
    static std::map<const char*, u64> counts;
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

void RenderQueue::WaitDone(const std::atomic<u64>& done,
                           u64 target,
                           const char* what) {
  u64 seen = done.load(std::memory_order_acquire);
  if (seen >= target)
    return;
  const auto start = std::chrono::steady_clock::now();
  for (; seen < target; seen = done.load(std::memory_order_acquire))
    done.wait(seen, std::memory_order_acquire);
  const auto waited = std::chrono::steady_clock::now() - start;
  if (waited > std::chrono::milliseconds(500)) {
    static const char* const kKinds[] = {"draw", "begin-frame", "end-frame",
                                         "call"};
    BASE_LOGW("gpuq", "walk waited {} ms for {}; renderer thread last ran {}",
              std::chrono::duration_cast<std::chrono::milliseconds>(waited)
                  .count(),
              what, kKinds[running_kind_.load() & 3]);
  }
}

void RenderQueue::Run() {
  for (u64 tail = 0;; tail++) {
    for (u64 head = head_.load(std::memory_order_acquire); head == tail;
         head = head_.load(std::memory_order_acquire)) {
      if (stop_.load())
        return;
      head_.wait(head, std::memory_order_acquire);
    }
    Command& c = commands_[tail % kCommands];
    running_kind_.store(static_cast<u8>(c.kind), std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    switch (c.kind) {
      case Kind::kDraw:
        render::Draw(*renderer_, draws_[c.arg]);
        draws_done_.fetch_add(1, std::memory_order_release);
        draws_done_.notify_all();
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
    const auto took = std::chrono::steady_clock::now() - started;
    if (took > std::chrono::milliseconds(500))
      BASE_LOGW("gpuq", "renderer thread spent {} ms in one {} command",
                std::chrono::duration_cast<std::chrono::milliseconds>(took)
                    .count(),
                static_cast<int>(c.kind));
    done_.store(tail + 1, std::memory_order_release);
    done_.notify_all();
  }
}

void RenderQueue::NotePendingWrite(u64 base, u64 bytes) {
  if (running_ && bytes)
    pending_writes_.push_back(
        {head_.load(std::memory_order_relaxed), base, base + bytes});
}

bool RenderQueue::PendingWriteOverlaps(u64 base, u64 bytes) {
  if (pending_writes_.empty())
    return false;
  const u64 done = done_.load(std::memory_order_acquire);
  std::erase_if(pending_writes_,
                [done](const PendingWrite& w) { return w.command < done; });
  const u64 end = base + bytes;
  for (const PendingWrite& w : pending_writes_)
    if (w.first < end && base < w.end)
      return true;
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
