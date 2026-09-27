/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/present.h"
#include "base/arch.h"

#include "host/window.h"
#include "gpu/gpu_perf.h"

#include <base/containers/vector.h>
#include <base/memory/move.h>
#include <base/threading/lock_guard.h>
#include <base/threading/mutex.h>

namespace gpu::render {

// host::Present blocks on the window swapchain (previous-present fence, vsync /
// compositor pacing) and, on a software Vulkan driver, rasterizes the blit on
// the CPU, ~10ms+ that used to sit on the frame loop. A dedicated presenter
// thread owns the window (creation, event pump, and present all happen on it)
// and always shows the newest completed frame; the frame loop just snapshots
// the pixels and signals. DELTA_GPU_SYNCPRESENT=1 restores the inline call.
LatestFramePresenter::~LatestFramePresenter() {
  Stop();
}

void LatestFramePresenter::StartLocked() {
  if (!thread_)
    thread_ = base::MakeUnique<base::Thread>("present", [this] { Run(); },
                                             true);
}

void LatestFramePresenter::Run() {
  base::UniqueLock<base::Mutex> lock(mutex_);
  base::Vector<u8> local;
  while (true) {
    ready_.Wait(lock, [this] { return pending_ || stopping_; });
    if (stopping_)
      return;
    const u32 w = width_;
    const u32 h = height_;
    const host::PixelFormat fmt = pending_fmt_;
    // A lent buffer is presented in place. Staging it into `local` first would
    // be a second 33 MB copy of a 4K scanout on top of the one host::Present
    // already does into its upload buffer, 6.5 ms for nothing. The borrow is
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
    if (host::Ensure("prosperity", w, h) && host::PumpEvents())
      host::Present(src, w, h, w * 4, fmt);
    g_ns_gfx_present += NowNs() - _tp;
    lock.lock();
    if (pending_src_ == src) {
      pending_src_ = nullptr;
      released_.NotifyAll();
    }
  }
}

void LatestFramePresenter::Present(const u8* pixels,
                                   u32 w,
                                   u32 h,
                                   host::PixelFormat fmt) {
  base::LockGuard<base::Mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_src_ = pixels;
  width_ = w;
  height_ = h;
  pending_fmt_ = fmt;
  pending_ = true;
  ready_.NotifyOne();
}

void LatestFramePresenter::Present(base::Vector<u8>&& pixels,
                                   u32 w,
                                   u32 h,
                                   host::PixelFormat fmt) {
  base::LockGuard<base::Mutex> lock(mutex_);
  if (stopping_)
    return;
  StartLocked();
  pending_pixels_.swap(pixels);
  pending_src_ = nullptr;
  released_.NotifyAll();
  width_ = w;
  height_ = h;
  pending_fmt_ = fmt;
  pending_ = true;
  ready_.NotifyOne();
}

void LatestFramePresenter::WaitForBorrowed() {
  const u64 _t0 = NowNs();
  base::UniqueLock<base::Mutex> lock(mutex_);
  released_.Wait(lock, [this] { return !pending_src_ || stopping_; });
  g_ns_borrow_wait += NowNs() - _t0;
}

void LatestFramePresenter::Stop() {
  base::UniquePointer<base::Thread> thread;
  {
    base::LockGuard<base::Mutex> lock(mutex_);
    stopping_ = true;
    pending_ = false;
    pending_src_ = nullptr;
    thread = base::move(thread_);
  }
  released_.NotifyAll();
  if (thread)
    host::RequestPresentStop();
  ready_.NotifyOne();
  if (thread)
    thread->Join();
}

}  // namespace gpu::render
