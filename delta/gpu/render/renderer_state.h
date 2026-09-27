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
#include "gpu/render/device.h"
#include "base/arch.h"
#include "gpu/render/frame.h"
#include "gpu/render/pipeline_cache.h"
#include "gpu/render/present.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/upload_ring.h"
#include <base/containers/map.h>
#include <base/containers/vector.h>
#include <base/containers/hash_map.h>


namespace gpu::render {

struct BackendState {
  rhi::Device* device = nullptr;
  FrameState frame;
  UploadRings rings;
  QuadPipelines quad;
  RecompiledPipelineCache recompiled_pipelines;
  TextureBindings tex;

  // Render targets keyed by guest address + the guest-page -> target index.
  RenderRegion region;
  base::HashMap<u64, RTarget> rts;
  base::HashMap<u64, DepthTarget> depths;
  base::HashMap<u64, base::Vector<u64>> rt_pages;

  // Declared last so its worker stops before the device-owned state above is
  // destroyed during process teardown.
  LatestFramePresenter presenter;
};

}  // namespace gpu::render

namespace gpu::render {

// The single backend instance (see the header comment).
extern render::BackendState g_backend;

}  // namespace gpu::render
