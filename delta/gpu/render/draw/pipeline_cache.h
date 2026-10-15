/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/xstring.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct PendingPipeline {
  base::Vector<u32> vertex, geometry, fragment;
  base::String name;
  rhi::GraphicsPipelineDesc desc;
  rhi::Pipeline* pipe = nullptr;
  base::Atomic<bool> done{false};
};

struct RecompPipe {
  rhi::Pipeline* pipe = nullptr;
  base::SharedPointer<PendingPipeline> pending;
  bool failed = false;
  rhi::PipelineLayout* layout = nullptr;
  rhi::BindGroupLayout* tex_set_layout = nullptr;
  u32 tex_bindings = 0;
  bool textured = false;
  bool multi_tex = false;
  u32 cbuf_count = 0;
  u64 cbuf_mask = 0;
  u32 raw_count = 0;
  u32 raw_mask = 0;
  bool shared_lds = false;
};

class RecompiledPipelineCache {
 public:
  RecompPipe* Find(u64 key);
  RecompPipe* Store(u64 key, RecompPipe pipeline);

 private:
  base::HashMap<u64, RecompPipe> pipelines_;
};

extern RecompiledPipelineCache& g_recomp_cache;

RecompPipe* GetRecompPipe(const render::DrawInfo& d);

}  // namespace gpu::render
