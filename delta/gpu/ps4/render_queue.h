#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "gpu/render/command.h"

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/condition_variable.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"

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
  void Stop();
  bool running() const { return running_; }

  render::DrawInfo& NextDraw();
  void PushDraw();
  void PushBeginFrame();
  void PushEndFrame(u64 scanout_base);
  void PushCall(base::Function<void()> fn);

  void Drain(const char* why = "other");

  void NotePendingWrite(u64 base, u64 bytes);
  void NotePendingValue(u64 base, u32 bytes, u64 value);
  bool PendingWriteOverlaps(u64 base, u64 bytes);
  bool PendingWriteValue(u64 address, u32* value);
  bool PendingWritesKnown(u64 base, u64 bytes);
  void OverlayPendingValues(u64 base, u64 bytes, u8* out);

 private:
  enum class Kind : u8 { kDraw, kBeginFrame, kEndFrame, kCall };
  struct Command {
    Kind kind = Kind::kCall;
    u64 arg = 0;
    base::Function<void()> fn;
  };
  static constexpr u64 kCommands = 8192;
  static constexpr u64 kDrawSlots = 512;

  Command& Claim();
  void Publish();
  void Run();
  void WaitDone(const base::Atomic<u64>& done, u64 target, const char* what);
  void Wake(base::ConditionVariable& cv);

  render::Renderer* renderer_ = nullptr;
  bool running_ = false;
  base::UniquePointer<Command[]> commands_;
  base::UniquePointer<render::DrawInfo[]> draws_;
  u64 draw_head_ = 0;
  u64 last_end_frame_ = 0;
  struct PendingWrite {
    u64 command, first, end;
    u64 value = 0;
    u32 value_bytes = 0;
  };
  void AddPendingWrite(const PendingWrite& w);
  void ErasePendingWritesDone();
  bool PendingMayOverlap(u64 base, u64 bytes) const;
  base::Vector<PendingWrite> pending_writes_;
  size_t pending_head_ = 0;
  base::HashMap<u64, u32> pending_pages_;
  u32 pending_big_ = 0;
  base::Atomic<u64> head_{0};
  base::Atomic<u64> done_{0};
  base::Atomic<u64> draws_done_{0};
  base::Atomic<bool> stop_{false};
  base::Atomic<u8> running_kind_{0};
  base::Mutex wake_mutex_;
  base::ConditionVariable walk_wake_;
  base::ConditionVariable render_wake_;
  base::UniquePointer<base::Thread> thread_;
};

RenderQueue& GuestRenderQueue();

inline void OwnRenderer(const char* why = "other") {
  GuestRenderQueue().Drain(why);
}

void FlushForWalkRead(render::Renderer& renderer,
                      u64 base,
                      u64 bytes,
                      const char* why);
void SpeculateWalkRead(render::Renderer& renderer, u64 base, u64 bytes);
bool WalkReadWouldWait(u64 base, u64 bytes);

}  // namespace gpu::ps4
