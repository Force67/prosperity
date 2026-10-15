/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/draw/pipeline_cache.h"
#include "gpu/render/present.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/render/memory/upload_ring.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct BackendState {
  rhi::Device* device = nullptr;
  FrameState frame;
  UploadRings rings;
  RecompiledPipelineCache recompiled_pipelines;
  TextureBindings tex;

  RenderRegion region;
  base::HashMap<u64, RTarget> rts;
  base::HashMap<u64, DepthTarget> depths;
  base::HashMap<u64, base::Vector<u64>> rt_pages;

  LatestFramePresenter presenter;
};

}  // namespace gpu::render

namespace gpu::render {

extern render::BackendState g_backend;

}  // namespace gpu::render
