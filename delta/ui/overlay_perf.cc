#include "ui/overlay.h"

#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"

namespace ui {
namespace {
base::Mutex g_perf_mtx;
float g_fps = 0, g_gpu_ms = 0, g_frame_ms = 0;
}  // namespace

void OverlaySetPerf(float fps, float gpu_ms, float frame_ms) {
  base::LockGuard<base::Mutex> lk(g_perf_mtx);
  g_fps = fps;
  g_gpu_ms = gpu_ms;
  g_frame_ms = frame_ms;
}

}  // namespace ui
