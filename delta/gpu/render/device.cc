/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/device.h"

#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include "base/threading/thread.h"

#include "base/logging.h"
#include "options/options.h"

#include "base/strings/xstring.h"
#include "gpu/render/backend.h"
#include "gpu/render/frame.h"
#include "gpu/render/labels.h"
#include "gpu/render/renderer.h"
#include "gpu/render/renderer_state.h"
#include "gpu/render/trace.h"
#include "gpu/render/upload_ring.h"

namespace {
DELTA_OPTION(const char*, kBackend, "DELTA_GPU_BACKEND", "vulkan");
DELTA_OPTION(const char*, kVkGpu, "DELTA_VK_GPU", nullptr);
DELTA_OPTION(bool, kCheckpoints, "DELTA_GPU_CHECKPOINTS", false);
DELTA_OPTION(bool, kShaderCacheOn, "DELTA_GPU_SHADER_CACHE", true);
DELTA_OPTION(const char*,
             kShaderCacheDirOpt,
             "DELTA_GPU_SHADER_CACHE_DIR",
             nullptr);
}  // namespace

namespace gpu::render {

namespace {

// Where driver pipeline caches live. Same convention as the SPIR-V cache
// (DELTA_GPU_SHADER_CACHE_DIR, else $XDG_CACHE_HOME/ps4delta, else
// ~/.cache/ps4delta), and disabled by the same DELTA_GPU_SHADER_CACHE=0.
base::String PipelineCacheDir() {
  if (!kShaderCacheOn)
    return base::String();
  base::String d;
  if (kShaderCacheDirOpt && *kShaderCacheDirOpt)
    d = kShaderCacheDirOpt;
  else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    d = base::String(xdg) + "/ps4delta";
  else if (const char* home = std::getenv("HOME"); home && *home)
    d = base::String(home) + "/.cache/ps4delta";
  else
    return base::String();
  for (size_t i = 1; i <= d.size(); i++)
    if (i == d.size() || d[i] == '/')
      ::mkdir(d.substr(0, i).c_str(), 0755);
  return d;
}

bool ParseBackend(const char* name, rhi::Backend& out) {
  for (rhi::Backend b : render::CompiledBackends())
    if (name && !std::strcmp(name, rhi::BackendName(b))) {
      out = b;
      return true;
    }
  return false;
}

}  // namespace

bool CreateDevice() {
  rhi::Backend backend = rhi::Backend::kVulkan;
  if (!ParseBackend(kBackend, backend))
    BASE_LOGI("gpu", "backend '{}' is not compiled in; using vulkan",
              kBackend ? kBackend : "");
  render::BackendOptions options;
  options.gpu_filter = kVkGpu;
  options.debug_labels = WantDebugUtils() || trace::WantValidation();
  options.validation = trace::WantValidation();
  options.sync_validation = trace::WantSyncValidation();
  options.checkpoints = kCheckpoints;
  options.on_message = trace::OnDeviceMessage;
  const base::String cache_dir = PipelineCacheDir();
  options.cache_dir = cache_dir.empty() ? nullptr : cache_dir.c_str();
  // Lives for the process: tearing a device down during static destruction
  // races the driver's own exit handlers.
  auto owned = render::CreateBackendDevice(backend, options);
  rhi::Device* device = owned.Get_UseOnlyIfYouKnowWhatYouareDoing();
  owned.ResetUnchecked_UseOnlyIfYouKnowWhatYouareDoing();
  if (!device)
    return false;
  g_backend.device = device;
  BASE_LOGI("gpu", "backend: {} on {}", rhi::BackendName(device->backend()),
            device->caps().device_name);
  return CreateFrameSlots() && CreateUploadRings();
}

void DrawCheckpoint(rhi::CommandList* list, u32 frame, u32 draw, bool after) {
  if (kCheckpoints)
    list->Checkpoint(((u64(frame) + 1) << 32) | (u64(draw) << 1) | u32(after));
}

// Bit 63 tells a dispatch marker from a draw marker.
void DispatchCheckpoint(rhi::CommandList* list, u64 cs_addr, bool after) {
  if (kCheckpoints)
    list->Checkpoint((1ull << 63) | ((cs_addr << 1) & ~(1ull << 63)) |
                     u32(after));
}

}  // namespace gpu::render

namespace gpu::render {

bool Init(Renderer& renderer) {
  if (renderer.available())
    return true;
  // A partial creation cannot safely be initialized over: its pools and
  // caches still contain objects from that attempt.
  if (g_backend.device)
    return false;
  // Create the device from a clean host thread: Init() is reached on a FEX
  // guest thread (guest stack / TLS), where the NVIDIA ICD's
  // vk_icdGetInstanceProcAddr silently fails and enumeration falls back to
  // llvmpipe, a ~30ms/frame software rasteriser on a box with a real GPU.
  bool ok = false;
  base::Thread init_thread("gpu-init", [&ok] { ok = CreateDevice(); }, true);
  init_thread.Join();
  if (!ok) {
    BASE_LOGI("gpu", "no usable graphics device; gpu disabled");
    return false;
  }
  // Registered after all static initialization, so the worker is joined before
  // BackendState and gfx translation-unit globals begin destruction.
  std::atexit([] { g_backend.presenter.Stop(); });
  renderer.state = &g_backend;
  return true;
}

}  // namespace gpu::render
