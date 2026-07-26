/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Graphics pipelines, cached by the guest state that shapes them: the heuristic
// quad pipelines (blend state + colour format) and the recompiled-shader
// pipelines (shader pair + blend + vertex layout + depth/raster state).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>

#include "rhi/command.h"

namespace gpu::vk {

// The heuristic quad path: a coloured and a textured pipeline, plus the
// per-blend-state variants built on demand.
struct QuadPipelines {
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout texLayout = VK_NULL_HANDLE;
  VkPipeline texPipeline = VK_NULL_HANDLE;
  // Keyed by (textured<<0, enable<<1, blendControl<<2), mixed with the format.
  std::unordered_map<uint64_t, VkPipeline> cache;
};

extern QuadPipelines g_quad;

bool createPipeline();
bool createTexPipeline();
VkPipeline getPipeline(bool textured, uint32_t bc, bool en,
                       VkFormat colorFormat);

struct RecompPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout texSetLayout = VK_NULL_HANDLE;
  bool textured = false;
  bool multiTex = false;  // custom set 0 for multiple and/or storage images
};

RecompPipe *getRecompPipe(const rhi::DrawInfo &d);

}  // namespace gpu::vk
