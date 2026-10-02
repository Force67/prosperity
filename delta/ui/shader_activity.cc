#include "ui/shader_activity.h"

#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"

namespace ui {
namespace {
base::Mutex g_mutex;
ShaderActivity g_activity;
}  // namespace

ShaderActivity GetShaderActivity() {
  base::LockGuard<base::Mutex> lock(g_mutex);
  return g_activity;
}

ShaderCompilation::ShaderCompilation() : started_ns_(base::TickClock::NowNs()) {
  base::LockGuard<base::Mutex> lock(g_mutex);
  if (g_activity.active++ == 0)
    g_activity.active_since_ns = started_ns_;
}

ShaderCompilation::~ShaderCompilation() {
  const u64 now = base::TickClock::NowNs();
  base::LockGuard<base::Mutex> lock(g_mutex);
  --g_activity.active;
  if (now - started_ns_ >= 40'000'000)
    g_activity.last_slow_compile_ns = now;
}

}  // namespace ui
