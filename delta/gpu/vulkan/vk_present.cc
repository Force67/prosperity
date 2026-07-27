/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_present.h"

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
  std::vector<uint8_t> local;
  while (true) {
    ready_.wait(lock, [this] { return pending_ || stopping_; });
    if (stopping_)
      return;
    // Steal the pending buffer (the allocations ping-pong, no per-frame alloc).
    local.swap(pending_pixels_);
    const uint32_t w = width_;
    const uint32_t h = height_;
    pending_ = false;
    lock.unlock();
    if (gfx::ensure("prosperity", w, h) && gfx::pumpEvents())
      gfx::present(local.data(), w, h, w * 4, gfx::PixelFormat::bgra8);
    lock.lock();
  }
}

void LatestFramePresenter::Present(const uint8_t* pixels,
                                   uint32_t w,
                                   uint32_t h) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_pixels_.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
  width_ = w;
  height_ = h;
  pending_ = true;
  ready_.notify_one();
}

void LatestFramePresenter::Present(std::vector<uint8_t>&& pixels,
                                   uint32_t w,
                                   uint32_t h) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_pixels_.swap(pixels);
  width_ = w;
  height_ = h;
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
