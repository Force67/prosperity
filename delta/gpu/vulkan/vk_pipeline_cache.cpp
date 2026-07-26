/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_pipeline_cache.h"

#include "gcn/gcn_translate.h"
#include "shaders/quad_frag_spv.h"
#include "shaders/quad_vert_spv.h"
#include "shaders/tex_frag_spv.h"
#include "shaders/tex_vert_spv.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_format.h"
#include "vulkan/vk_hash.h"
#include "vulkan/vk_render_target.h"
#include "vulkan/vk_texture_cache.h"
#include "vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gpu::vk {

using rhi::DrawInfo;

QuadPipelines g_quad;
std::unordered_map<uint64_t, RecompPipe> g_recompPipes;

// Build a graphics pipeline for the colored (textured=false) or textured quad
// with the given colour-blend attachment. Shaders + layout selected by `textured`.
VkPipeline buildPipeline(bool textured, VkPipelineColorBlendAttachmentState cba,
                         VkFormat colorFormat) {
  VkShaderModule vs = makeModule(textured ? tex_vert_spv : quad_vert_spv,
                                 textured ? sizeof(tex_vert_spv) : sizeof(quad_vert_spv));
  VkShaderModule fs = makeModule(textured ? tex_frag_spv : quad_frag_spv,
                                 textured ? sizeof(tex_frag_spv) : sizeof(quad_frag_spv));
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs; stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs; stages[1].pName = "main";

  // Interleaved repacked vertex: pos.xy@0, color.rgba@8, uv.xy@24, stride 32.
  VkVertexInputBindingDescription bind{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[3] = {
      {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, 24},
  };
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 3;
  vi.pVertexAttributeDescriptions = attrs;
  // GNM draws are indexed triangle LISTS (VGT_PRIMITIVE_TYPE 4); the previous
  // hardcoded strip connected separate sprites into long diagonal triangles.
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1; cb.pAttachments = &cba;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;
  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &colorFormat;
  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci; pi.stageCount = 2; pi.pStages = stages;
  pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
  pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
  pi.layout = textured ? g_quad.texLayout : g_quad.layout;
  VkPipeline p = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(g_dev.device, VK_NULL_HANDLE, 1, &pi, nullptr, &p);
  vkDestroyShaderModule(g_dev.device, vs, nullptr);
  vkDestroyShaderModule(g_dev.device, fs, nullptr);
  return p;
}

// Pipeline for a draw's blend state, cached. Returns the default src-alpha pipeline
// when the per-state build fails so a draw never silently drops.
VkPipeline getPipeline(bool textured, uint32_t bc, bool en, VkFormat colorFormat) {
  uint64_t key = (textured ? 1ull : 0) | (en ? 2ull : 0) |
                 ((uint64_t)(en ? (bc & 0x7FFFFFFFu) : 0u) << 2);
  key = hashWord(key, colorFormat);
  auto it = g_quad.cache.find(key);
  if (it != g_quad.cache.end())
    return it->second;
  VkPipeline p = buildPipeline(textured, blendAttachment(bc, en), colorFormat);
  if (!p && colorFormat == kDefaultRtFormat) p = textured ? g_quad.texPipeline : g_quad.pipeline;
  g_quad.cache[key] = p;
  return p;
}

bool createPipeline() {
  if (g_quad.pipeline)
    return true;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};  // mat4
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g_dev.device, &li, nullptr, &g_quad.layout));
  // Default colored pipeline: classic src-alpha (used as the fallback / for draws
  // that don't enable blend the cache builds an opaque one on demand).
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g_quad.pipeline = buildPipeline(false, cba, kDefaultRtFormat);
  if (!g_quad.pipeline) { std::fprintf(stderr, "[gpuvk] pipeline failed\n"); return false; }
  return true;
}

bool createTexPipeline() {
  if (g_quad.texPipeline)
    return true;
  if (!createTextureDescriptors())
    return false;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 68};  // mat4 + clipUV flag
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1;
  li.pSetLayouts = &g_tex.dsLayout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g_dev.device, &li, nullptr, &g_quad.texLayout));

  // Default textured pipeline: src-alpha over (the common sprite blend). Per-draw
  // blend states build their own pipeline on demand via getPipeline().
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g_quad.texPipeline = buildPipeline(true, cba, kDefaultRtFormat);
  if (!g_quad.texPipeline) { std::fprintf(stderr, "[gpuvk] tex pipeline failed\n"); return false; }
  return true;
}

// Build (or fetch) the pipeline for a recompiled draw, keyed by the shader pair
// + blend state + vertex layout.
RecompPipe *getRecompPipe(const DrawInfo &d) {
  if (d.recomp->ps_texs.size() > kMaxTex)
    return nullptr;
  uint32_t mrtN = std::min(d.mrtCount, 8u);
  // Depth + primitive-setup state folded into the pipeline key (mixed through
  // an FNV prime so it spreads across the whole 64-bit space, away from the
  // blend/stride bits).
  uint32_t dstate = (d.depthBase ? 1u : 0u) | (d.depthTestEnable ? 2u : 0u) |
                    (d.depthWriteEnable ? 4u : 0u) | ((d.depthFunc & 7u) << 3) |
                    ((d.primType & 0x1Fu) << 6) | ((d.cullMode & 3u) << 11) |
                    (d.frontCCW ? 0u : (1u << 13));
  uint64_t key =
      d.vsAddr * 0x9e3779b97f4a7c15ull ^ d.psAddr ^
      ((uint64_t)(d.blendEnable ? (d.blendControl & 0x7FFFFFFFu) : 0) << 1) ^
      ((uint64_t)d.vertexStride << 33) ^ ((uint64_t)mrtN << 60) ^
      ((uint64_t)dstate * 0x100000001b3ull);
  key = hashWord(key, d.nvattrs);
  for (uint32_t i = 0; i < mrtN; i++)
    key = hashWord(key, colorTargetFormat(d.mrtInfo[i]));
  // The vertex-input layout (binding count + per-binding strides + per-attr
  // binding assignment) is baked into the pipeline, so it must be part of the key
  // or a later multi-stream draw would reuse a single-stream pipeline (or vice
  // versa) for the same shader pair.
  key = hashWord(key, d.nvbufs);
  for (uint32_t j = 0; j < d.nvbufs; j++) key = hashWord(key, d.vbufs[j].stride);
  for (uint32_t i = 0; i < d.nvattrs; i++) {
    key = hashWord(key, d.vattrs[i].location);
    key = hashWord(key, d.vattrs[i].binding);
    key = hashWord(key, d.vattrs[i].offset);
    key = hashWord(key, d.vattrs[i].num_comps);
    key = hashWord(key, d.vattrs[i].dfmt);
    key = hashWord(key, d.vattrs[i].nfmt);
  }
  auto it = g_recompPipes.find(key);
  if (it != g_recompPipes.end()) return &it->second;
  RecompPipe rp;
  rp.textured = !d.recomp->ps_texs.empty();
  const bool hasStorage = std::any_of(
      d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
      [](const gcn::ShaderTex &tex) { return tex.storage; });
  rp.multiTex = d.recomp->ps_texs.size() > 1 || hasStorage;

  // set 0 = texture(s) (or an empty layout when untextured), set 1 = cbuffer UBO.
  // Multi/storage shaders use an exact per-binding descriptor layout;
  // single-sampler shaders retain the shared one-binding layout.
  VkDescriptorSetLayout set0 = !rp.textured ? g_ring.emptyLayout : g_tex.dsLayout;
  if (rp.multiTex) {
    VkDescriptorSetLayoutBinding bindings[kMaxTex];
    for (uint32_t i = 0; i < d.recomp->ps_texs.size(); i++) {
      bindings[i] = {i,
                     d.recomp->ps_texs[i].storage
                         ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                         : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                     1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo sl{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sl.bindingCount = static_cast<uint32_t>(d.recomp->ps_texs.size());
    sl.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr,
                                    &rp.texSetLayout) != VK_SUCCESS)
      return nullptr;
    set0 = rp.texSetLayout;
  }
  VkDescriptorSetLayout sls[2] = {set0, g_ring.uboLayout};
  VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT |
                               VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, 128};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 2;
  li.pSetLayouts = sls;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(g_dev.device, &li, nullptr, &rp.layout) != VK_SUCCESS) return nullptr;

  VkShaderModule vs = makeModuleVec(d.recomp->vs_spirv);
  VkShaderModule fs = makeModuleVec(d.recomp->fs_spirv);
  // RECTLIST is primitive type 17 on GFX7 but 7 on gfx10.3 (PrimitiveType::
  // kRectList; 17 is kRectListLegacy there). Missing the gfx10 number rendered
  // every PS5 fullscreen pass as a single triangle covering half the rect.
  const bool isRectList = d.primType == 17 || d.primType == 7;
  bool rectList = isRectList && g_dev.geometryShader && !d.recomp->gs_spirv.empty();
  VkShaderModule gs = rectList ? makeModuleVec(d.recomp->gs_spirv) : VK_NULL_HANDLE;
  VkPipelineShaderStageCreateInfo stages[3]{};
  uint32_t stageCount = 0;
  stages[stageCount] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[stageCount].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[stageCount].module = vs; stages[stageCount++].pName = "main";
  if (rectList) {
    stages[stageCount] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[stageCount].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[stageCount].module = gs; stages[stageCount++].pName = "main";
  }
  stages[stageCount] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[stageCount].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[stageCount].module = fs; stages[stageCount++].pName = "main";

  // One Vulkan binding per resolved vertex buffer (single-stream draws stay a
  // single binding, identical to before); attributes reference their binding.
  uint32_t nbind = d.nvattrs ? std::min(d.nvbufs, 8u) : 0;
  VkVertexInputBindingDescription binds[8];
  for (uint32_t j = 0; j < nbind; j++)
    binds[j] = {j, d.vbufs[j].stride, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[8];
  for (uint32_t i = 0; i < d.nvattrs; i++)
    attrs[i] = {d.vattrs[i].location, d.vattrs[i].binding,
                vfmt(d.vattrs[i].dfmt, d.vattrs[i].nfmt), d.vattrs[i].offset};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = nbind;
  vi.pVertexBindingDescriptions = nbind ? binds : nullptr;
  vi.vertexAttributeDescriptionCount = d.nvattrs; vi.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = vkTopology(d.primType);
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  // Face culling from PA_SU_SC_MODE_CNTL (CULL_FRONT[0]/CULL_BACK[1] map 1:1 onto the
  // Vulkan cull-mode bits). The render region uses a negative-height (y-up) viewport
  // to match GCN rasterisation, which flips triangle winding in framebuffer space, so
  // the guest's front-face sense is inverted here to compensate. Culling is opt-in
  // (DELTA_GPU_CULL=1) until the winding can be validated against visible 3D geometry:
  // Doom64's world textures are compute-built (unimplemented) so its geometry is not
  // yet visible, and depth already resolves occlusion, so the default stays cull-none
  // to avoid dropping correctly-drawn faces (some HUD draws set cull bits).
  static const bool doCull = std::getenv("DELTA_GPU_CULL") != nullptr;
  rs.cullMode = doCull ? (VkCullModeFlags)(d.cullMode & 0x3) : VK_CULL_MODE_NONE;
  rs.frontFace = d.frontCCW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  // Depth test/write from DB_DEPTH_CONTROL (only when the draw bound a Z buffer; 2D
  // draws leave depthBase 0 so this stays fully disabled, unchanged from before).
  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  if (d.depthBase) {
    dss.depthTestEnable = d.depthTestEnable ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable = d.depthWriteEnable ? VK_TRUE : VK_FALSE;
    dss.depthCompareOp = (VkCompareOp)(d.depthFunc & 0x7);  // ZFUNC maps 1:1
  }
  // One blend attachment per bound MRT target, each from its own CB_BLENDn_CONTROL
  // (mrtBlend[i] / mrtBlendMask bit i); target 0 mirrors blendControl/blendEnable so the
  // single-RT path is unchanged. Targets the PS does not export to are write-masked off
  // so they keep their loaded content.
  VkPipelineColorBlendAttachmentState cbAtt[8];
  for (uint32_t i = 0; i < mrtN; i++) {
    uint32_t bc = i == 0 ? d.blendControl : d.mrtBlend[i];
    bool en = i == 0 ? d.blendEnable : ((d.mrtBlendMask >> i) & 1u);
    cbAtt[i] = blendAttachment(bc, en);
    // Mask attachments the PS does not export to. A PS with no color export
    // at all (depth-only / buffer-store passes) writes nothing -- previously a
    // white fallback was painted, which poisoned multi-pass chains (PT).
    static const bool noMaskDiag = std::getenv("DELTA_GPU_NOMASK") != nullptr;
    if (!noMaskDiag && !(d.recomp->ps_mrt_mask & (1u << i)))
      cbAtt[i].colorWriteMask = 0;
  }
  // DELTA_GPU_PIPETRACE: the colour-blend state a pipeline is actually built
  // with, next to the PS's export mask -- the two have to agree or an attachment
  // is silently write-masked off (or written unblended).
  if (std::getenv("DELTA_GPU_PIPETRACE")) {
    static int n = 0;
    if (n++ < 24)
      std::fprintf(stderr,
                    "[pipe] ps=%#lx mrtN=%u psMrtMask=%#x att0: en=%u src=%d dst=%d "
                   "srcA=%d dstA=%d writeMask=%#x\n",
                    (unsigned long)d.psAddr, mrtN, d.recomp->ps_mrt_mask,
                   cbAtt[0].blendEnable, (int)cbAtt[0].srcColorBlendFactor,
                   (int)cbAtt[0].dstColorBlendFactor, (int)cbAtt[0].srcAlphaBlendFactor,
                   (int)cbAtt[0].dstAlphaBlendFactor, cbAtt[0].colorWriteMask);
  }
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = mrtN; cb.pAttachments = cbAtt;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;
  VkFormat fmts[8];
  for (uint32_t i = 0; i < mrtN; i++)
    fmts[i] = colorTargetFormat(d.mrtInfo[i]);
  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = mrtN; rci.pColorAttachmentFormats = fmts;
  if (d.depthBase) rci.depthAttachmentFormat = kDepthFormat;
  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci; pi.stageCount = stageCount; pi.pStages = stages;
  pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
  pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb; pi.pDynamicState = &dy; pi.layout = rp.layout;
  VkResult r = vkCreateGraphicsPipelines(g_dev.device, VK_NULL_HANDLE, 1, &pi, nullptr, &rp.pipe);
  vkDestroyShaderModule(g_dev.device, vs, nullptr);
  if (gs) vkDestroyShaderModule(g_dev.device, gs, nullptr);
  vkDestroyShaderModule(g_dev.device, fs, nullptr);
  if (r != VK_SUCCESS) { std::fprintf(stderr, "[gpuvk] recomp pipeline failed: %d\n", (int)r);
    return nullptr; }
  g_recompPipes[key] = rp;
  return &g_recompPipes[key];
}

}  // namespace gpu::vk
