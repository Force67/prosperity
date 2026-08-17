/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_present.h"
#include "base/arch.h"

#include "gfx/gfx.h"

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
    // Steal the pending buffer (the allocations ping-pong, no per-frame alloc).
    local.swap(pending_pixels_);
    const u32 w = width_;
    const u32 h = height_;
    const gfx::PixelFormat fmt = pending_fmt_;
    pending_ = false;
    lock.unlock();
    if (gfx::ensure("prosperity", w, h) && gfx::pumpEvents())
      gfx::present(local.data(), w, h, w * 4, fmt);
    lock.lock();
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
  pending_pixels_.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
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
  width_ = w;
  height_ = h;
  pending_fmt_ = fmt;
  pending_ = true;
  ready_.notify_one();
}

void LatestFramePresenter::Stop() {
  std::thread thread;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    pending_ = false;
    thread = std::move(thread_);
  }
  if (thread.joinable())
    gfx::requestPresentStop();
  ready_.notify_one();
  if (thread.joinable())
    thread.join();
}

}  // namespace gpu::vk
