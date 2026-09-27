#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The renderer's own thread. The command processor walks PM4 and resolves
 * each draw's descriptors on the submitting guest thread, then hands the
 * finished DrawInfo here; this thread stages its buffers, resolves textures
 * and records it, in submission order, while the walk moves on.
 *
 * Renderer state belongs to one thread at a time. The walk may touch it only
 * after Drain(), which returns once this thread has run everything queued and
 * gone idle. Label writes are queued like draws, so the guest never learns
 * that the GPU passed a point before the draws ahead of it read their guest
 * memory.
 */

#include "base/arch.h"
#include "gpu/render/command.h"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace gpu::render {
class Renderer;
}

namespace gpu::ps4 {

class RenderQueue {
 public:
  RenderQueue() = default;
  RenderQueue(const RenderQueue&) = delete;
  RenderQueue& operator=(const RenderQueue&) = delete;
  ~RenderQueue();

  void Start(render::Renderer& renderer);
  bool running() const { return running_; }

  // The slot the next PushDraw hands over, reset to a default DrawInfo. Owned
  // by the caller until then; a draw that is not pushed leaves it for the next.
  render::DrawInfo& NextDraw();
  void PushDraw();
  void PushBeginFrame();
  // Also waits for the previous frame's end, so the walk runs at most one
  // frame ahead of what the renderer has presented.
  void PushEndFrame(u64 scanout_base);
  void PushCall(std::function<void()> fn);

  // Returns once every pushed command has run. `why` names the caller in the
  // DELTA_GPU_DRAINTRACE report.
  void Drain(const char* why = "other");

  // The next pushed command writes [base, base+bytes) of guest memory (a DMA
  // copy, a WRITE_DATA); a walk read overlapping it must Drain first.
  void NotePendingWrite(u64 base, u64 bytes);
  bool PendingWriteOverlaps(u64 base, u64 bytes);

 private:
  enum class Kind : u8 { kDraw, kBeginFrame, kEndFrame, kCall };
  struct Command {
    Kind kind = Kind::kCall;
    u64 arg = 0;
    std::function<void()> fn;
  };
  static constexpr u64 kCommands = 8192;
  static constexpr u64 kDrawSlots = 512;

  Command& Claim();
  void Publish();
  void Run();
  // Blocks until `done` reaches `target`; reports a wait long enough to
  // matter (the title's hang detector fires at 10 s).
  void WaitDone(const std::atomic<u64>& done, u64 target, const char* what);

  render::Renderer* renderer_ = nullptr;
  bool running_ = false;
  std::unique_ptr<Command[]> commands_;
  std::unique_ptr<render::DrawInfo[]> draws_;
  // Producer side: advanced under the command processor's lock.
  u64 draw_head_ = 0;
  u64 last_end_frame_ = 0;
  struct PendingWrite {
    u64 command, first, end;
  };
  std::vector<PendingWrite> pending_writes_;
  std::atomic<u64> head_{0};
  // Consumer side: commands and draws fully run.
  std::atomic<u64> done_{0};
  std::atomic<u64> draws_done_{0};
  std::atomic<bool> stop_{false};
  // What the renderer thread is running, for the long-wait report.
  std::atomic<u8> running_kind_{0};
  std::thread thread_;
};

// The queue the PS4 command processor feeds; not running when the renderer
// thread is off (DELTA_GPU_RENDER_THREAD=0), and then every call runs inline.
RenderQueue& GuestRenderQueue();

// For the walk, before it touches renderer state itself.
inline void OwnRenderer(const char* why = "other") {
  GuestRenderQueue().Drain(why);
}

// Make [base, base+bytes) current in guest memory before the walk reads it:
// compute results the renderer still holds are written back first.
void FlushForWalkRead(render::Renderer& renderer,
                      u64 base,
                      u64 bytes,
                      const char* why);

}  // namespace gpu::ps4
