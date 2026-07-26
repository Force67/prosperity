/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_present.h"

#include "gfx/gfx.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace gpu::vk {
namespace {

// gfx::present blocks on the window swapchain (previous-present fence, vsync /
// compositor pacing) and, on a software Vulkan driver, rasterizes the blit on
// the CPU -- ~10ms+ that used to sit on the frame loop. A dedicated presenter
// thread owns the window (creation, event pump, and present all happen on it)
// and always shows the newest completed frame; the frame loop just snapshots
// the pixels and signals. DELTA_GPU_SYNCPRESENT=1 restores the inline call.
struct Presenter {
  std::thread th;
  std::mutex mtx;
  std::condition_variable cv;
  std::vector<uint8_t> buf;  // pending frame (BGRA, tight pitch); latest wins
  uint32_t w = 0, h = 0;
  bool pending = false;
  bool started = false;
};

Presenter g_presenter;

void presenterLoop() {
  std::unique_lock<std::mutex> lk(g_presenter.mtx);
  std::vector<uint8_t> local;
  while (true) {
    g_presenter.cv.wait(lk, [] { return g_presenter.pending; });
    // Steal the pending buffer (the allocations ping-pong, no per-frame alloc).
    local.swap(g_presenter.buf);
    const uint32_t w = g_presenter.w, h = g_presenter.h;
    g_presenter.pending = false;
    lk.unlock();
    if (gfx::ensure("prosperity", w, h) && gfx::pumpEvents())
      gfx::present(local.data(), w, h, w * 4, gfx::PixelFormat::bgra8);
    lk.lock();
  }
}

}  // namespace

void presentAsync(const uint8_t *pixels, uint32_t w, uint32_t h) {
  Presenter &p = g_presenter;
  if (!p.started) {
    p.started = true;
    p.th = std::thread(presenterLoop);
    p.th.detach();
  }
  std::lock_guard<std::mutex> lk(p.mtx);
  p.buf.assign(pixels, pixels + (size_t)w * h * 4);
  p.w = w;
  p.h = h;
  p.pending = true;
  p.cv.notify_one();
}

}  // namespace gpu::vk
