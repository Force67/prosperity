/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Graphics pipelines for the recompiled shaders, cached by the guest state that
// shapes them (shader pair + blend + vertex layout + depth/raster state).

#include "base/arch.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/xstring.h"
#include "base/atomic.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// The heuristic quad path: a coloured and a textured pipeline, plus the
// per-blend-state variants built on demand.
// A graphics pipeline compiling on a shader worker. The worker owns the inputs
// until it publishes the result.
struct PendingPipeline {
  base::Vector<u32> vertex, geometry, fragment;
  base::String name;
  rhi::GraphicsPipelineDesc desc;
  rhi::Pipeline* pipe = nullptr;
  base::Atomic<bool> done{false};
};

struct RecompPipe {
  // Null while `pending` compiles: draws that need it are skipped meanwhile.
  rhi::Pipeline* pipe = nullptr;
  base::SharedPointer<PendingPipeline> pending;
  bool failed = false;  // an async build that produced nothing
  rhi::PipelineLayout* layout = nullptr;
  rhi::BindGroupLayout* tex_set_layout = nullptr;
  // Bindings tex_set_layout declares. The shader may declare more samplers than
  // the draw resolved textures for, and every declared binding has to be
  // written or the draw reads a descriptor that was never updated.
  u32 tex_bindings = 0;
  bool textured = false;
  bool multi_tex = false;  // custom set 0 for multiple and/or storage images
  u32 cbuf_count = 0;
  u64 cbuf_mask = 0;
  // The shader pair reads raw buffers with MUBUF, so the layout carries set 2
  // (the raw-buffer ring). Shaders that do not are unaffected: their layout
  // ends at set 1 exactly as before.
  bool raw_bufs = false;
  bool shared_lds = false;  // the VS backs its LDS with the set-3 scratch
};

class RecompiledPipelineCache {
 public:
  RecompPipe* Find(u64 key);
  RecompPipe* Store(u64 key, RecompPipe pipeline);

 private:
  base::HashMap<u64, RecompPipe> pipelines_;
};

// Transitional alias into render::BackendState while pipeline creation is
// migrated to receive its cache explicitly.
extern RecompiledPipelineCache& g_recomp_cache;

RecompPipe* GetRecompPipe(const render::DrawInfo& d);

}  // namespace gpu::render
