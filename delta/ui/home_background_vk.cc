#include "ui/home_background_vk.h"

#include "ui/shaders/home_background_frag.h"
#include "ui/shaders/home_background_vert.h"

namespace ui {
namespace {
VkDevice g_device = VK_NULL_HANDLE;
VkPipeline g_pipeline = VK_NULL_HANDLE;
VkPipelineLayout g_layout = VK_NULL_HANDLE;

struct BackgroundPush {
  float width;
  float height;
  float time;
  float style;
  float pulse;
};

VkShaderModule MakeShader(const u32* code, mem_size size) {
  VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  info.codeSize = size;
  info.pCode = code;
  VkShaderModule shader = VK_NULL_HANDLE;
  vkCreateShaderModule(g_device, &info, nullptr, &shader);
  return shader;
}
}  // namespace

bool HomeBackgroundVkInit(VkDevice device, VkRenderPass pass) {
  g_device = device;
  VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(BackgroundPush)};
  VkPipelineLayoutCreateInfo layout{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layout.pushConstantRangeCount = 1;
  layout.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(device, &layout, nullptr, &g_layout) != VK_SUCCESS)
    return false;
  VkShaderModule vert =
      MakeShader(kHomeBackgroundVert, sizeof(kHomeBackgroundVert));
  VkShaderModule frag =
      MakeShader(kHomeBackgroundFrag, sizeof(kHomeBackgroundFrag));
  if (!vert || !frag) {
    if (vert)
      vkDestroyShaderModule(device, vert, nullptr);
    if (frag)
      vkDestroyShaderModule(device, frag, nullptr);
    HomeBackgroundVkShutdown();
    return false;
  }
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = stages[1].sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[0].module = vert;
  stages[1].module = frag;
  stages[0].pName = stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vertices{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo viewport{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  viewport.viewportCount = viewport.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo raster{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  raster.polygonMode = VK_POLYGON_MODE_FILL;
  raster.cullMode = VK_CULL_MODE_NONE;
  raster.lineWidth = 1;
  VkPipelineMultisampleStateCreateInfo samples{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineColorBlendAttachmentState attachment{};
  attachment.colorWriteMask = 0xf;
  VkPipelineColorBlendStateCreateInfo blend{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  blend.attachmentCount = 1;
  blend.pAttachments = &attachment;
  const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                    VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo state{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  state.dynamicStateCount = 2;
  state.pDynamicStates = dynamic;
  VkGraphicsPipelineCreateInfo pipeline{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pipeline.stageCount = 2;
  pipeline.pStages = stages;
  pipeline.pVertexInputState = &vertices;
  pipeline.pInputAssemblyState = &assembly;
  pipeline.pViewportState = &viewport;
  pipeline.pRasterizationState = &raster;
  pipeline.pMultisampleState = &samples;
  pipeline.pColorBlendState = &blend;
  pipeline.pDynamicState = &state;
  pipeline.layout = g_layout;
  pipeline.renderPass = pass;
  const auto result = vkCreateGraphicsPipelines(
      device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &g_pipeline);
  vkDestroyShaderModule(device, vert, nullptr);
  vkDestroyShaderModule(device, frag, nullptr);
  if (result != VK_SUCCESS)
    HomeBackgroundVkShutdown();
  return result == VK_SUCCESS;
}

void HomeBackgroundVkDraw(VkCommandBuffer cmd,
                          u32 width,
                          u32 height,
                          float time,
                          u32 style,
                          float pulse) {
  if (!g_pipeline)
    return;
  VkViewport viewport{0, 0, float(width), float(height), 0, 1};
  VkRect2D scissor{{0, 0}, {width, height}};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);
  const BackgroundPush push{float(width), float(height), time, float(style),
                            pulse};
  vkCmdPushConstants(cmd, g_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(push), &push);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

void HomeBackgroundVkShutdown() {
  if (g_pipeline)
    vkDestroyPipeline(g_device, g_pipeline, nullptr);
  if (g_layout)
    vkDestroyPipelineLayout(g_device, g_layout, nullptr);
  g_pipeline = VK_NULL_HANDLE;
  g_layout = VK_NULL_HANDLE;
  g_device = VK_NULL_HANDLE;
}

}  // namespace ui
