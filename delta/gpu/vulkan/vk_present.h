/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Handing a finished frame to the window. gfx::present blocks on the window
// swapchain and, on a software driver, rasterizes the blit on the CPU, so a
// dedicated thread owns the window and always shows the newest complete frame.

#include <condition_variable>
#include "base/arch.h"
#include <mutex>
#include <thread>
#include <vector>

#include "gfx/gfx.h"

namespace gpu::vk {

class LatestFramePresenter {
 public:
  LatestFramePresenter() = default;
  ~LatestFramePresenter();

  LatestFramePresenter(const LatestFramePresenter&) = delete;
  LatestFramePresenter& operator=(const LatestFramePresenter&) = delete;

  void Present(const u8* pixels, u32 w, u32 h,
               gfx::PixelFormat fmt = gfx::PixelFormat::bgra8);
  void Present(std::vector<u8>&& pixels, u32 w, u32 h,
               gfx::PixelFormat fmt = gfx::PixelFormat::bgra8);
  // Block until a lent buffer has been copied out, for a caller that is about
  // to write over the one it lent.
  void WaitForBorrowed();
  void Stop();

 private:
  void StartLocked();
  void Run();

  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable released_;  // a lent buffer has been copied out
  std::vector<u8> pending_pixels_;  // tight pitch, pending_fmt_; latest wins
  // Set instead of pending_pixels_ when the caller lends us its buffer: the
  // presenter thread copies out of it, under the lock, before releasing it.
  const u8* pending_src_ = nullptr;
  gfx::PixelFormat pending_fmt_ = gfx::PixelFormat::bgra8;
  u32 width_ = 0;
  u32 height_ = 0;
  bool pending_ = false;
  bool stopping_ = false;
};

}  // namespace gpu::vk
