/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_present.h"
#include "base/arch.h"

#include "gfx/gfx.h"
#include "gpu/gpu_perf.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu::vk {

// gfx::present blocks on the window swapchain (previous-present fence, vsync /
// compositor pacing) and, on a software Vulkan driver, rasterizes the blit on
// the CPU -- ~10ms+ that used to sit on the frame loop. A dedicated presenter
// thread owns the window (creation, event pump, and present all happen on it)
// and always shows the newest completed frame; the frame loop just snapshots
// the pixels and signals. DELTA_GPU_SYNCPRESENT=1 restores the inline call.
LatestFramePresenter::~LatestFramePresenter() {
  Stop();
}

void LatestFramePresenter::StartLocked() {
  if (!thread_.joinable())
    thread_ = std::thread(&LatestFramePresenter::Run, this);
}

void LatestFramePresenter::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<u8> local;
  while (true) {
    ready_.wait(lock, [this] { return pending_ || stopping_; });
    if (stopping_)
      return;
    const u32 w = width_;
    const u32 h = height_;
    const gfx::PixelFormat fmt = pending_fmt_;
    // A lent buffer is presented in place. Staging it into `local` first would
    // be a second 33 MB copy of a 4K scanout on top of the one gfx::present
    // already does into its upload buffer -- 6.5 ms for nothing. The borrow is
    // held until present returns; the renderer waits for that in BeginFrame
    // before it lets the GPU write the buffer again.
    const u8* src = pending_src_;
    if (!src) {
      // Steal the pending buffer (the allocations ping-pong, no per-frame
      // alloc).
      local.swap(pending_pixels_);
      src = local.data();
    }
    pending_ = false;
    lock.unlock();
    const u64 _tp = NowNs();
    if (gfx::ensure("prosperity", w, h) && gfx::pumpEvents())
      gfx::present(src, w, h, w * 4, fmt);
    g_ns_gfx_present += NowNs() - _tp;
    lock.lock();
    if (pending_src_ == src) {
      pending_src_ = nullptr;
      released_.notify_all();
    }
  }
}

void LatestFramePresenter::Present(const u8* pixels,
                                   u32 w,
                                   u32 h,
                                   gfx::PixelFormat fmt) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_src_ = pixels;
  width_ = w;
  height_ = h;
  pending_fmt_ = fmt;
  pending_ = true;
  ready_.notify_one();
}

void LatestFramePresenter::Present(std::vector<u8>&& pixels,
                                   u32 w,
                                   u32 h,
                                   gfx::PixelFormat fmt) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_pixels_.swap(pixels);
  pending_src_ = nullptr;
  released_.notify_all();
  width_ = w;
  height_ = h;
  pending_fmt_ = fmt;
  pending_ = true;
  ready_.notify_one();
}

void LatestFramePresenter::WaitForBorrowed() {
  const u64 _t0 = NowNs();
  std::unique_lock<std::mutex> lock(mutex_);
  released_.wait(lock, [this] { return !pending_src_ || stopping_; });
  g_ns_borrow_wait += NowNs() - _t0;
}

void LatestFramePresenter::Stop() {
  std::thread thread;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    pending_ = false;
    pending_src_ = nullptr;
    thread = std::move(thread_);
  }
  released_.notify_all();
  if (thread.joinable())
    gfx::requestPresentStop();
  ready_.notify_one();
  if (thread.joinable())
    thread.join();
}

}  // namespace gpu::vk
