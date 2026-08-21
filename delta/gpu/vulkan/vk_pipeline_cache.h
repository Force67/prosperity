/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Graphics pipelines, cached by the guest state that shapes them: the heuristic
// quad pipelines (blend state + colour format) and the recompiled-shader
// pipelines (shader pair + blend + vertex layout + depth/raster state).

#include <vulkan/vulkan.h>
#include "base/arch.h"

#include <unordered_map>

#include "gpu/rhi/command.h"

namespace gpu::vk {

// The heuristic quad path: a coloured and a textured pipeline, plus the
// per-blend-state variants built on demand.
struct QuadPipelines {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout tex_layout = VK_NULL_HANDLE;
  VkPipeline tex_pipeline = VK_NULL_HANDLE;
  // Keyed by (textured<<0, enable<<1, blend_control<<2), mixed with the format.
  std::unordered_map<u64, VkPipeline> cache;
};

extern QuadPipelines& g_quad;

bool CreatePipeline();
bool CreateTexPipeline();
VkPipeline GetPipeline(bool textured,
                       u32 bc,
                       bool en,
                       VkFormat color_format);

struct RecompPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout tex_set_layout = VK_NULL_HANDLE;
  // Bindings tex_set_layout declares. The shader may declare more samplers than
  // the draw resolved textures for, and every declared binding has to be
  // written or the draw reads a descriptor that was never updated.
  u32 tex_bindings = 0;
  bool textured = false;
  bool multi_tex = false;  // custom set 0 for multiple and/or storage images
  // The shader pair reads raw buffers with MUBUF, so the layout carries set 2
  // (the raw-buffer ring). Shaders that do not are unaffected: their layout
  // ends at set 1 exactly as before.
  bool raw_bufs = false;
};

class RecompiledPipelineCache {
 public:
  RecompPipe* Find(u64 key);
  RecompPipe* Store(u64 key, RecompPipe pipeline);

 private:
  std::unordered_map<u64, RecompPipe> pipelines_;
};

// Transitional alias into rhi::BackendState while pipeline creation is
// migrated to receive its cache explicitly.
extern RecompiledPipelineCache& g_recomp_cache;

RecompPipe* GetRecompPipe(const rhi::DrawInfo& d);

}  // namespace gpu::vk
