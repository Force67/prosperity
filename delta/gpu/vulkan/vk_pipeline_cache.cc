/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_pipeline_cache.h"

#include "gpu/ps4/gcn/gcn_translate.h"
#include "gpu/shaders/quad_frag_spv.h"
#include "gpu/shaders/quad_vert_spv.h"
#include "gpu/shaders/tex_frag_spv.h"
#include "gpu/shaders/tex_vert_spv.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_hash.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gpu::vk {

using rhi::DrawInfo;

std::unordered_map<uint64_t, RecompPipe> g_recomp_pipes;

// Build a graphics pipeline for the colored (textured=false) or textured quad
// with the given colour-blend attachment. Shaders + layout selected by
// `textured`.
VkPipeline BuildPipeline(bool textured,
                         VkPipelineColorBlendAttachmentState cba,
                         VkFormat color_format) {
  VkShaderModule vs =
      MakeModule(textured ? tex_vert_spv : quad_vert_spv,
                 textured ? sizeof(tex_vert_spv) : sizeof(quad_vert_spv));
  VkShaderModule fs =
      MakeModule(textured ? tex_frag_spv : quad_frag_spv,
                 textured ? sizeof(tex_frag_spv) : sizeof(quad_frag_spv));
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  // Interleaved repacked vertex: pos.xy@0, color.rgba@8, uv.xy@24, stride 32.
  VkVertexInputBindingDescription bind{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[3] = {
      {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, 24},
  };
  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 3;
  vi.pVertexAttributeDescriptions = attrs;
  // GNM draws are indexed triangle LISTS (VGT_PRIMITIVE_TYPE 4); the previous
  // hardcoded strip connected separate sprites into long diagonal triangles.
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo dss{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                            VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyns;
  VkPipelineRenderingCreateInfo rci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = 1;
  rci.pColorAttachmentFormats = &color_format;
  VkGraphicsPipelineCreateInfo pi{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci;
  pi.stageCount = 2;
  pi.pStages = stages;
  pi.pVertexInputState = &vi;
  pi.pInputAssemblyState = &ia;
  pi.pViewportState = &vp;
  pi.pRasterizationState = &rs;
  pi.pMultisampleState = &ms;
  pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb;
  pi.pDynamicState = &dy;
  pi.layout = textured ? g_quad.tex_layout : g_quad.layout;
  VkPipeline p = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(g_dev.device, g_dev.pipeline_cache, 1, &pi, nullptr,
                            &p);
  vkDestroyShaderModule(g_dev.device, vs, nullptr);
  vkDestroyShaderModule(g_dev.device, fs, nullptr);
  return p;
}

// Pipeline for a draw's blend state, cached. Returns the default src-alpha
// pipeline when the per-state build fails so a draw never silently drops.
VkPipeline GetPipeline(bool textured,
                       uint32_t bc,
                       bool en,
                       VkFormat color_format) {
  uint64_t key = (textured ? 1ull : 0) | (en ? 2ull : 0) |
                 ((uint64_t)(en ? (bc & 0x7FFFFFFFu) : 0u) << 2);
  key = HashWord(key, color_format);
  auto it = g_quad.cache.find(key);
  if (it != g_quad.cache.end())
    return it->second;
  VkPipeline p = BuildPipeline(textured, BlendAttachment(bc, en), color_format);
  if (!p && color_format == kDefaultRtFormat)
    p = textured ? g_quad.tex_pipeline : g_quad.pipeline;
  g_quad.cache[key] = p;
  return p;
}

bool CreatePipeline() {
  if (g_quad.pipeline)
    return true;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};  // mat4
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g_dev.device, &li, nullptr, &g_quad.layout));
  // Default colored pipeline: classic src-alpha (used as the fallback / for
  // draws that don't enable blend the cache builds an opaque one on demand).
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g_quad.pipeline = BuildPipeline(false, cba, kDefaultRtFormat);
  if (!g_quad.pipeline) {
    std::fprintf(stderr, "[gpuvk] pipeline failed\n");
    return false;
  }
  return true;
}

bool CreateTexPipeline() {
  if (g_quad.tex_pipeline)
    return true;
  if (!CreateTextureDescriptors())
    return false;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0,
                          68};  // mat4 + clipUV flag
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1;
  li.pSetLayouts = &g_tex.ds_layout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g_dev.device, &li, nullptr, &g_quad.tex_layout));

  // Default textured pipeline: src-alpha over (the common sprite blend).
  // Per-draw blend states build their own pipeline on demand via GetPipeline().
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  g_quad.tex_pipeline = BuildPipeline(true, cba, kDefaultRtFormat);
  if (!g_quad.tex_pipeline) {
    std::fprintf(stderr, "[gpuvk] tex pipeline failed\n");
    return false;
  }
  return true;
}

// Build (or fetch) the pipeline for a recompiled draw, keyed by the shader pair
// + blend state + vertex layout.
RecompPipe* GetRecompPipe(const DrawInfo& d) {
  if (d.recomp->ps_texs.size() > kMaxTex)
    return nullptr;
  uint32_t mrt_n = std::min(d.mrt_count, 8u);
  // Depth + primitive-setup state folded into the pipeline key (mixed through
  // an FNV prime so it spreads across the whole 64-bit space, away from the
  // blend/stride bits).
  uint32_t dstate = (d.depth_base ? 1u : 0u) | (d.depth_test_enable ? 2u : 0u) |
                    (d.depth_write_enable ? 4u : 0u) |
                    ((d.depth_func & 7u) << 3) | ((d.prim_type & 0x1Fu) << 6) |
                    ((d.cull_mode & 3u) << 11) |
                    (d.front_ccw ? 0u : (1u << 13));
  uint64_t key =
      d.vs_addr * 0x9e3779b97f4a7c15ull ^ d.ps_addr ^
      ((uint64_t)(d.blend_enable ? (d.blend_control & 0x7FFFFFFFu) : 0) << 1) ^
      ((uint64_t)d.vertex_stride << 33) ^ ((uint64_t)mrt_n << 60) ^
      ((uint64_t)dstate * 0x100000001b3ull);
  key = HashWord(key, d.num_vattrs);
  for (uint32_t i = 0; i < mrt_n; i++)
    key = HashWord(key, ColorTargetFormat(d.mrt_info[i]));
  // The vertex-input layout (binding count + per-binding strides + per-attr
  // binding assignment) is baked into the pipeline, so it must be part of the
  // key or a later multi-stream draw would reuse a single-stream pipeline (or
  // vice versa) for the same shader pair.
  key = HashWord(key, d.num_vbufs);
  for (uint32_t j = 0; j < d.num_vbufs; j++)
    key = HashWord(key, d.vbufs[j].stride);
  for (uint32_t i = 0; i < d.num_vattrs; i++) {
    key = HashWord(key, d.vattrs[i].location);
    key = HashWord(key, d.vattrs[i].binding);
    key = HashWord(key, d.vattrs[i].offset);
    key = HashWord(key, d.vattrs[i].num_comps);
    key = HashWord(key, d.vattrs[i].dfmt);
    key = HashWord(key, d.vattrs[i].nfmt);
  }
  auto it = g_recomp_pipes.find(key);
  if (it != g_recomp_pipes.end())
    return &it->second;
  RecompPipe rp;
  rp.textured = !d.recomp->ps_texs.empty();
  const bool has_storage =
      std::any_of(d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                  [](const gcn::ShaderTex& tex) { return tex.storage; });
  rp.multi_tex = d.recomp->ps_texs.size() > 1 || has_storage;

  // set 0 = texture(s) (or an empty layout when untextured), set 1 = cbuffer
  // UBO. Multi/storage shaders use an exact per-binding descriptor layout;
  // single-sampler shaders retain the shared one-binding layout.
  VkDescriptorSetLayout set0 =
      !rp.textured ? g_ring.empty_layout : g_tex.ds_layout;
  if (rp.multi_tex) {
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
                                    &rp.tex_set_layout) != VK_SUCCESS)
      return nullptr;
    set0 = rp.tex_set_layout;
  }
  VkDescriptorSetLayout sls[2] = {set0, g_ring.ubo_layout};
  VkPushConstantRange push{
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 2;
  li.pSetLayouts = sls;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(g_dev.device, &li, nullptr, &rp.layout) !=
      VK_SUCCESS)
    return nullptr;

  VkShaderModule vs = MakeModuleVec(d.recomp->vs_spirv);
  VkShaderModule fs = MakeModuleVec(d.recomp->fs_spirv);
  // RECTLIST is primitive type 17 on GFX7 but 7 on gfx10.3 (PrimitiveType::
  // kRectList; 17 is kRectListLegacy there). Missing the gfx10 number rendered
  // every PS5 fullscreen pass as a single triangle covering half the rect.
  const bool is_rect_list = d.prim_type == 17 || d.prim_type == 7;
  bool rect_list =
      is_rect_list && g_dev.geometry_shader && !d.recomp->gs_spirv.empty();
  VkShaderModule gs =
      rect_list ? MakeModuleVec(d.recomp->gs_spirv) : VK_NULL_HANDLE;
  VkPipelineShaderStageCreateInfo stages[3]{};
  uint32_t stage_count = 0;
  stages[stage_count] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[stage_count].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[stage_count].module = vs;
  stages[stage_count++].pName = "main";
  if (rect_list) {
    stages[stage_count] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[stage_count].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[stage_count].module = gs;
    stages[stage_count++].pName = "main";
  }
  stages[stage_count] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[stage_count].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[stage_count].module = fs;
  stages[stage_count++].pName = "main";

  // One Vulkan binding per resolved vertex buffer (single-stream draws stay a
  // single binding, identical to before); attributes reference their binding.
  uint32_t nbind = d.num_vattrs ? std::min(d.num_vbufs, 8u) : 0;
  VkVertexInputBindingDescription binds[8];
  for (uint32_t j = 0; j < nbind; j++)
    binds[j] = {j, d.vbufs[j].stride, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[8];
  for (uint32_t i = 0; i < d.num_vattrs; i++)
    attrs[i] = {d.vattrs[i].location, d.vattrs[i].binding,
                VertexFormat(d.vattrs[i].dfmt, d.vattrs[i].nfmt),
                d.vattrs[i].offset};
  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = nbind;
  vi.pVertexBindingDescriptions = nbind ? binds : nullptr;
  vi.vertexAttributeDescriptionCount = d.num_vattrs;
  vi.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = PrimitiveTopology(d.prim_type);
  VkPipelineViewportStateCreateInfo vp{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  // Face culling from PA_SU_SC_MODE_CNTL (CULL_FRONT[0]/CULL_BACK[1] map 1:1
  // onto the Vulkan cull-mode bits). The render region uses a negative-height
  // (y-up) viewport to match GCN rasterisation, which flips triangle winding in
  // framebuffer space, so the guest's front-face sense is inverted here to
  // compensate. Culling is opt-in (DELTA_GPU_CULL=1) until the winding can be
  // validated against visible 3D geometry: Doom64's world textures are
  // compute-built (unimplemented) so its geometry is not yet visible, and depth
  // already resolves occlusion, so the default stays cull-none to avoid
  // dropping correctly-drawn faces (some HUD draws set cull bits).
  static const bool kDoCull = std::getenv("DELTA_GPU_CULL") != nullptr;
  rs.cullMode =
      kDoCull ? (VkCullModeFlags)(d.cull_mode & 0x3) : VK_CULL_MODE_NONE;
  rs.frontFace =
      d.front_ccw ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  // Depth test/write from DB_DEPTH_CONTROL (only when the draw bound a Z
  // buffer; 2D draws leave depth_base 0 so this stays fully disabled, unchanged
  // from before).
  VkPipelineDepthStencilStateCreateInfo dss{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  if (d.depth_base) {
    dss.depthTestEnable = d.depth_test_enable ? VK_TRUE : VK_FALSE;
    dss.depthWriteEnable = d.depth_write_enable ? VK_TRUE : VK_FALSE;
    dss.depthCompareOp = (VkCompareOp)(d.depth_func & 0x7);  // ZFUNC maps 1:1
  }
  // One blend attachment per bound MRT target, each from its own
  // CB_BLENDn_CONTROL (mrt_blend[i] / mrt_blend_mask bit i); target 0 mirrors
  // blend_control/blend_enable so the single-RT path is unchanged. Targets the
  // PS does not export to are write-masked off so they keep their loaded
  // content.
  VkPipelineColorBlendAttachmentState cb_att[8];
  for (uint32_t i = 0; i < mrt_n; i++) {
    uint32_t bc = i == 0 ? d.blend_control : d.mrt_blend[i];
    bool en = i == 0 ? d.blend_enable : ((d.mrt_blend_mask >> i) & 1u);
    cb_att[i] = BlendAttachment(bc, en);
    // Mask attachments the PS does not export to. A PS with no color export
    // at all (depth-only / buffer-store passes) writes nothing -- previously a
    // white fallback was painted, which poisoned multi-pass chains (PT).
    static const bool kNoMaskDiag = std::getenv("DELTA_GPU_NOMASK") != nullptr;
    if (!kNoMaskDiag && !(d.recomp->ps_mrt_mask & (1u << i)))
      cb_att[i].colorWriteMask = 0;
  }
  // DELTA_GPU_PIPETRACE: the colour-blend state a pipeline is actually built
  // with, next to the PS's export mask -- the two have to agree or an
  // attachment is silently write-masked off (or written unblended).
  if (std::getenv("DELTA_GPU_PIPETRACE")) {
    static int n = 0;
    if (n++ < 24)
      std::fprintf(
          stderr,
          "[pipe] ps=%#lx mrtN=%u psMrtMask=%#x att0: en=%u src=%d dst=%d "
          "src_a=%d dst_a=%d writeMask=%#x\n",
          (unsigned long)d.ps_addr, mrt_n, d.recomp->ps_mrt_mask,
          cb_att[0].blendEnable, (int)cb_att[0].srcColorBlendFactor,
          (int)cb_att[0].dstColorBlendFactor,
          (int)cb_att[0].srcAlphaBlendFactor,
          (int)cb_att[0].dstAlphaBlendFactor, cb_att[0].colorWriteMask);
  }
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = mrt_n;
  cb.pAttachments = cb_att;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                            VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyns;
  VkFormat fmts[8];
  for (uint32_t i = 0; i < mrt_n; i++)
    fmts[i] = ColorTargetFormat(d.mrt_info[i]);
  VkPipelineRenderingCreateInfo rci{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = mrt_n;
  rci.pColorAttachmentFormats = fmts;
  if (d.depth_base)
    rci.depthAttachmentFormat = kDepthFormat;
  VkGraphicsPipelineCreateInfo pi{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci;
  pi.stageCount = stage_count;
  pi.pStages = stages;
  pi.pVertexInputState = &vi;
  pi.pInputAssemblyState = &ia;
  pi.pViewportState = &vp;
  pi.pRasterizationState = &rs;
  pi.pMultisampleState = &ms;
  pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb;
  pi.pDynamicState = &dy;
  pi.layout = rp.layout;
  VkResult r = vkCreateGraphicsPipelines(g_dev.device, g_dev.pipeline_cache, 1,
                                         &pi, nullptr, &rp.pipe);
  vkDestroyShaderModule(g_dev.device, vs, nullptr);
  if (gs)
    vkDestroyShaderModule(g_dev.device, gs, nullptr);
  vkDestroyShaderModule(g_dev.device, fs, nullptr);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[gpuvk] recomp pipeline failed: %d\n", (int)r);
    return nullptr;
  }
  g_recomp_pipes[key] = rp;
  return &g_recomp_pipes[key];
}

}  // namespace gpu::vk
