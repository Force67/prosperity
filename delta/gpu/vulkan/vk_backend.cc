/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_backend.h"

#include "gpu/rhi/renderer.h"

namespace gpu::vk {

rhi::BackendState g_backend;

// Transitional aliases: the one place the per-subsystem names bind to the
// backend state. Everything is initialized within this translation unit
// (g_backend first, in declaration order), so the references are never read
// unbound; no other TU touches these during static initialization.
DeviceState& g_dev = g_backend.device;
FrameState& g_frame = g_backend.frame;
UploadRings& g_ring = g_backend.rings;
QuadPipelines& g_quad = g_backend.quad;
TextureBindings& g_tex = g_backend.tex;
RenderRegion& g_region = g_backend.region;
std::unordered_map<uint64_t, RTarget>& g_rts = g_backend.rts;
std::unordered_map<uint64_t, DepthTarget>& g_depths = g_backend.depths;
std::unordered_map<uint64_t, std::vector<uint64_t>>& g_rt_pages =
    g_backend.rt_pages;
PFN_vkCmdBeginRenderingKHR& g_cmd_begin_rendering =
    g_backend.cmd_begin_rendering;
PFN_vkCmdEndRenderingKHR& g_cmd_end_rendering = g_backend.cmd_end_rendering;

}  // namespace gpu::vk

namespace gpu::rhi {

Renderer& DefaultRenderer() {
  static Renderer renderer;
  return renderer;
}

}  // namespace gpu::rhi
