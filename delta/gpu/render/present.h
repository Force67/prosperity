/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "base/atomic.h"
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
  void PresentShared(const void* token,
                     u32 id,
                     int fd,
                     u64 bytes,
                     u32 w,
                     u32 h,
                     host::PixelFormat fmt);
  bool SharedWorks() const {
    return shared_works_.load(base::memory_order_acquire);
  }
  void WaitForBorrowed();
  void Stop(bool interrupt_present = true);

 private:
  void StartLocked();
  void Run();

  base::UniquePointer<base::Thread> thread_;
  base::Mutex mutex_;
  base::ConditionVariable ready_;
  base::ConditionVariable released_;
  base::Vector<u8> pending_pixels_;
  const u8* pending_src_ = nullptr;
  struct Shared {
    u32 id = 0;
    int fd = -1;
    u64 bytes = 0;
  };
  Shared pending_shared_;
  bool pending_is_shared_ = false;
  base::Atomic<bool> shared_works_{true};
  host::PixelFormat pending_fmt_ = host::PixelFormat::kBgra8;
  u32 width_ = 0;
  u32 height_ = 0;
  bool pending_ = false;
  bool stopping_ = false;
};

}  // namespace gpu::render
