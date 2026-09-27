/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/renderer_state.h"
#include "base/arch.h"

#include "gpu/render/renderer.h"
#include <base/containers/map.h>
#include <base/containers/vector.h>
#include <base/containers/hash_map.h>

namespace gpu::render {

render::BackendState g_backend;

// Transitional aliases: the one place the per-subsystem names bind to the
// backend state. Everything is initialized within this translation unit
// (g_backend first, in declaration order), so the references are never read
// unbound; no other TU touches these during static initialization.
FrameState& g_frame = g_backend.frame;
UploadRings& g_ring = g_backend.rings;
QuadPipelines& g_quad = g_backend.quad;
RecompiledPipelineCache& g_recomp_cache = g_backend.recompiled_pipelines;
TextureBindings& g_tex = g_backend.tex;
RenderRegion& g_region = g_backend.region;
base::HashMap<u64, RTarget>& g_rts = g_backend.rts;
base::HashMap<u64, DepthTarget>& g_depths = g_backend.depths;
base::HashMap<u64, base::Vector<u64>>& g_rt_pages =
    g_backend.rt_pages;

}  // namespace gpu::render

namespace gpu::render {

Renderer& DefaultRenderer() {
  static Renderer renderer;
  return renderer;
}

}  // namespace gpu::render
