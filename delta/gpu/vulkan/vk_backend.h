/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The whole Vulkan backend state as one value. rhi::Renderer::state points at
// this; rhi::BackendState is opaque to everyone outside gpu/vulkan.
//
// The backend is single-instance today: g_backend is the one BackendState, and
// the per-subsystem names the implementation uses (g_dev, g_frame, ...) are
// reference aliases into it (vk_backend.cc). New code should take the state it
// needs from the BackendState& / Renderer& it is handed instead of naming an
// alias; the aliases exist so the existing code can migrate incrementally.

#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_memory.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_present.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gpu::rhi {

struct BackendState {
  vk::DeviceState device;
  vk::FrameState frame;
  vk::UploadRings rings;
  vk::QuadPipelines quad;
  vk::RecompiledPipelineCache recompiled_pipelines;
  vk::ImageMemoryPool image_memory;
  vk::TextureBindings tex;

  // Render targets keyed by guest address + the guest-page -> target index.
  vk::RenderRegion region;
  std::unordered_map<uint64_t, vk::RTarget> rts;
  std::unordered_map<uint64_t, vk::DepthTarget> depths;
  std::unordered_map<uint64_t, std::vector<uint64_t>> rt_pages;

  // Dynamic rendering entry points (core in 1.3, KHR on older drivers).
  PFN_vkCmdBeginRenderingKHR cmd_begin_rendering = nullptr;
  PFN_vkCmdEndRenderingKHR cmd_end_rendering = nullptr;

  // Declared last so its worker stops before the device-owned state above is
  // destroyed during process teardown.
  vk::LatestFramePresenter presenter;
};

}  // namespace gpu::rhi

namespace gpu::vk {

// The single backend instance (see the header comment).
extern rhi::BackendState g_backend;

}  // namespace gpu::vk
