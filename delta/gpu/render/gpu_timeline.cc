/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/gpu_timeline.h"
#include "base/arch.h"
#include "guest/session.h"

#include <cstring>

#include "base/containers/hash_map.h"
#include "base/strings/xstring.h"
#include "gpu/render/frame.h"

#if defined(DELTA_TRACY)
#include "tracy/TracyC.h"
#endif

namespace gpu::render::timeline {

#if defined(DELTA_TRACY)
namespace {
constexpr u8 kContext = 0;
constexpr u8 kTypeVulkan = 2;  // tracy::GpuContextType::Vulkan
bool g_active = false;
u16 g_next_query = 0;

// A GPU timestamp taken now, to pair with the CPU clock for the new context.
bool GpuNow(u64* ticks) {
  static rhi::CommandList* list = nullptr;
  static rhi::TimestampPool* pool = nullptr;
  static const guest::SessionReset reset([] {
    list = nullptr;
    pool = nullptr;
    g_active = false;
    g_next_query = 0;
  });
  if (!list)
    list = Device().CreateCommandList();
  if (!pool)
    pool = Device().CreateTimestampPool(1);
  if (!list || !pool)
    return false;
  list->Begin();
  list->ResetTimestamps(pool, 0, 1);
  list->WriteTimestamp(pool, 0, false);
  list->End();
  return Device().Wait(Device().Submit(&list, 1)) &&
         Device().ReadTimestamps(pool, 0, 1, ticks);
}
}  // namespace

void Refresh() {
  const bool connected = ___tracy_connected() != 0;
  if (!connected || !Device().caps().timestamps) {
    g_active = false;
    return;
  }
  if (g_active)
    return;
  u64 ticks = 0;
  if (!GpuNow(&ticks))
    return;
  ___tracy_gpu_new_context_data context{};
  context.gpuTime = static_cast<int64_t>(ticks);
  context.period = static_cast<float>(Device().caps().timestamp_period_ns);
  context.context = kContext;
  context.type = kTypeVulkan;
  ___tracy_emit_gpu_new_context_serial(context);
  static const char kName[] = "guest GPU";
  ___tracy_emit_gpu_context_name_serial({kContext, kName, sizeof(kName) - 1});
  g_next_query = 0;
  g_active = true;
}

bool Active() {
  return g_active;
}

u16 Begin(const char* name) {
  // One lasting source location per name: there are only as many as the
  // title has targets and shaders, and a fresh one per zone makes the viewer
  // resolve each of them.
  static base::HashMap<base::String, ___tracy_source_location_data*> where;
  auto it = where.find(base::String(name));
  if (it == where.end()) {
    auto* location = new ___tracy_source_location_data{};
    location->name = strdup(name);
    location->function = location->name;
    location->file = __FILE__;
    location->line = __LINE__;
    it = where.emplace(base::String(name), location).first;
  }
  const u16 query = g_next_query;
  g_next_query += 2;
  ___tracy_emit_gpu_zone_begin_serial(
      {reinterpret_cast<uint64_t>(it->second), query, kContext});
  return query;
}

void End(u16 query) {
  ___tracy_emit_gpu_zone_end_serial({static_cast<u16>(query + 1), kContext});
}

void Time(u16 query, u64 gpu_ticks) {
  ___tracy_emit_gpu_time_serial(
      {static_cast<int64_t>(gpu_ticks), query, kContext});
}
#else
void Refresh() {}
bool Active() {
  return false;
}
u16 Begin(const char*) {
  return 0;
}
void End(u16) {}
void Time(u16, u64) {}
#endif

}  // namespace gpu::render::timeline
