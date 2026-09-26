/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The whole Vulkan backend state as one value. render::Renderer::state points at
// this; render::BackendState is opaque to everyone outside gpu/vulkan.
//
// The backend is single-instance today: g_backend is the one BackendState, and
// the per-subsystem names the implementation uses (g_dev, g_frame, ...) are
// reference aliases into it (vk_backend.cc). New code should take the state it
// needs from the BackendState& / Renderer& it is handed instead of naming an
// alias; the aliases exist so the existing code can migrate incrementally.

#include "gpu/rhi/device.h"
#include "gpu/vulkan/vk_device.h"
#include "base/arch.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_present.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <unordered_map>
#include <vector>

namespace gpu::render {

struct BackendState {
  rhi::Device* device = nullptr;
  vk::FrameState frame;
  vk::UploadRings rings;
  vk::QuadPipelines quad;
  vk::RecompiledPipelineCache recompiled_pipelines;
  vk::TextureBindings tex;

  // Render targets keyed by guest address + the guest-page -> target index.
  vk::RenderRegion region;
  std::unordered_map<u64, vk::RTarget> rts;
  std::unordered_map<u64, vk::DepthTarget> depths;
  std::unordered_map<u64, std::vector<u64>> rt_pages;

  // Declared last so its worker stops before the device-owned state above is
  // destroyed during process teardown.
  vk::LatestFramePresenter presenter;
};

}  // namespace gpu::render

namespace gpu::vk {

// The single backend instance (see the header comment).
extern render::BackendState g_backend;

}  // namespace gpu::vk
