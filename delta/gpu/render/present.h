/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Handing a finished frame to the window. host::Present blocks on the window
// swapchain and, on a software driver, rasterizes the blit on the CPU, so a
// dedicated thread owns the window and always shows the newest complete frame.

#include "base/arch.h"

#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/condition_variable.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "host/window.h"

namespace gpu::render {

class LatestFramePresenter {
 public:
  LatestFramePresenter() = default;
  ~LatestFramePresenter();

  LatestFramePresenter(const LatestFramePresenter&) = delete;
  LatestFramePresenter& operator=(const LatestFramePresenter&) = delete;

  void Present(const u8* pixels,
               u32 w,
               u32 h,
               host::PixelFormat fmt = host::PixelFormat::kBgra8);
  void Present(base::Vector<u8>&& pixels,
               u32 w,
               u32 h,
               host::PixelFormat fmt = host::PixelFormat::kBgra8);
  // Block until a lent buffer has been copied out, for a caller that is about
  // to write over the one it lent.
  void WaitForBorrowed();
  // Keep the window usable when handing presentation to the exit animation.
  void Stop(bool interrupt_present = true);

 private:
  void StartLocked();
  void Run();

  base::UniquePointer<base::Thread> thread_;
  base::Mutex mutex_;
  base::ConditionVariable ready_;
  base::ConditionVariable released_;  // a lent buffer has been copied out
  base::Vector<u8> pending_pixels_;   // tight pitch, pending_fmt_; latest wins
  // Set instead of pending_pixels_ when the caller lends us its buffer: the
  // presenter thread copies out of it, under the lock, before releasing it.
  const u8* pending_src_ = nullptr;
  host::PixelFormat pending_fmt_ = host::PixelFormat::kBgra8;
  u32 width_ = 0;
  u32 height_ = 0;
  bool pending_ = false;
  bool stopping_ = false;
};

}  // namespace gpu::render
