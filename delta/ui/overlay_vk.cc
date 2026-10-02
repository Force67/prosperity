/*
 * PS4Delta : PS4 emulation and research project
 *
 * Vulkan backend for the Dear ImGui overlay. See overlay_vk.h. A small pipeline
 * (embedded SPIR-V, ImDrawVert layout, alpha blend, dynamic viewport/scissor)
 * draws the overlay's ImDrawData into the swapchain image through a LOAD render
 * pass that also transitions the image from TRANSFER_DST (the frame blit) to
 * PRESENT_SRC, so gfx_vk's present path just calls OverlayVkRender.
 */

#include "ui/overlay_vk.h"
#include "base/arch.h"

#include <cstdio>

#include "base/logging.h"
#include "base/memory/mem_ops.h"

#include "base/containers/vector.h"
#include "imgui.h"
#include "ui/overlay.h"
#include "ui/overlay_vk_shaders.h"

namespace ui {
namespace {

struct Frame {
  VkBuffer vtx = VK_NULL_HANDLE, idx = VK_NULL_HANDLE;
  VkDeviceMemory vtx_mem = VK_NULL_HANDLE, idx_mem = VK_NULL_HANDLE;
  VkDeviceSize vtx_cap = 0, idx_cap = 0;
  void *vtx_map = nullptr, *idx_map = nullptr;
};

struct {
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};

  VkRenderPass pass = VK_NULL_HANDLE;
  VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool = VK_NULL_HANDLE;
  VkDescriptorSet desc_set = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;

  VkImage font_img = VK_NULL_HANDLE;
  VkDeviceMemory font_mem = VK_NULL_HANDLE;
  VkImageView font_view = VK_NULL_HANDLE;

  base::Vector<VkImageView> views;
  base::Vector<VkFramebuffer> fbs;
  base::Vector<Frame> frames;
  bool ready = false;
} g_vk;

u32 MemType(u32 bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &mp);
  for (u32 i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

bool MakeBuffer(VkDeviceSize size,
                VkBufferUsageFlags usage,
                VkMemoryPropertyFlags props,
                VkBuffer& buf,
                VkDeviceMemory& mem) {
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(g_vk.device, &bi, nullptr, &buf) != VK_SUCCESS)
    return false;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(g_vk.device, buf, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = MemType(req.memoryTypeBits, props);
  if (vkAllocateMemory(g_vk.device, &ai, nullptr, &mem) != VK_SUCCESS)
    return false;
  vkBindBufferMemory(g_vk.device, buf, mem, 0);
  return true;
}

VkShaderModule MakeModule(const u32* code, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = code;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(g_vk.device, &ci, nullptr, &m);
  return m;
}

bool CreatePipeline() {
  VkAttachmentDescription att{};
  att.format = g_vk.format;
  att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // preserve the blitted frame
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;  // after the blit
  att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub{};
  sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  sub.colorAttachmentCount = 1;
  sub.pColorAttachments = &ref;
  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass = 0;
  dep.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  dep.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rp.attachmentCount = 1;
  rp.pAttachments = &att;
  rp.subpassCount = 1;
  rp.pSubpasses = &sub;
  rp.dependencyCount = 1;
  rp.pDependencies = &dep;
  if (vkCreateRenderPass(g_vk.device, &rp, nullptr, &g_vk.pass) != VK_SUCCESS)
    return false;

  VkDescriptorSetLayoutBinding b{};
  b.binding = 0;
  b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  b.descriptorCount = 1;
  b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  VkDescriptorSetLayoutCreateInfo dl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 1;
  dl.pBindings = &b;
  vkCreateDescriptorSetLayout(g_vk.device, &dl, nullptr, &g_vk.desc_layout);

  VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 4};
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &g_vk.desc_layout;
  pl.pushConstantRangeCount = 1;
  pl.pPushConstantRanges = &pc;
  vkCreatePipelineLayout(g_vk.device, &pl, nullptr, &g_vk.pipe_layout);

  VkShaderModule vs = MakeModule(kImguiVertSpv, sizeof(kImguiVertSpv));
  VkShaderModule fs = MakeModule(kImguiFragSpv, sizeof(kImguiFragSpv));
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = vs;
  st[0].pName = "main";
  st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = fs;
  st[1].pName = "main";

  VkVertexInputBindingDescription bind{0, sizeof(ImDrawVert),
                                       VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attr[3]{
      {0, 0, VK_FORMAT_R32G32_SFLOAT, (u32)IM_OFFSETOF(ImDrawVert, pos)},
      {1, 0, VK_FORMAT_R32G32_SFLOAT, (u32)IM_OFFSETOF(ImDrawVert, uv)},
      {2, 0, VK_FORMAT_R8G8B8A8_UNORM, (u32)IM_OFFSETOF(ImDrawVert, col)}};
  VkPipelineVertexInputStateCreateInfo vi{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 3;
  vi.pVertexAttributeDescriptions = attr;
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
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;
  VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo ds{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  ds.dynamicStateCount = 2;
  ds.pDynamicStates = dyn;
  VkGraphicsPipelineCreateInfo gp{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gp.stageCount = 2;
  gp.pStages = st;
  gp.pVertexInputState = &vi;
  gp.pInputAssemblyState = &ia;
  gp.pViewportState = &vp;
  gp.pRasterizationState = &rs;
  gp.pMultisampleState = &ms;
  gp.pColorBlendState = &cb;
  gp.pDynamicState = &ds;
  gp.layout = g_vk.pipe_layout;
  gp.renderPass = g_vk.pass;
  VkResult r = vkCreateGraphicsPipelines(g_vk.device, VK_NULL_HANDLE, 1, &gp,
                                         nullptr, &g_vk.pipe);
  vkDestroyShaderModule(g_vk.device, vs, nullptr);
  vkDestroyShaderModule(g_vk.device, fs, nullptr);
  return r == VK_SUCCESS;
}

bool UploadFont() {
  ImGuiIO& io = ImGui::GetIO();
  unsigned char* px = nullptr;
  int w = 0, h = 0;
  io.Fonts->GetTexDataAsRGBA32(&px, &w, &h);
  const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;

  VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ic.imageType = VK_IMAGE_TYPE_2D;
  ic.format = VK_FORMAT_R8G8B8A8_UNORM;
  ic.extent = {(u32)w, (u32)h, 1};
  ic.mipLevels = 1;
  ic.arrayLayers = 1;
  ic.samples = VK_SAMPLE_COUNT_1_BIT;
  ic.tiling = VK_IMAGE_TILING_OPTIMAL;
  ic.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(g_vk.device, &ic, nullptr, &g_vk.font_img) != VK_SUCCESS)
    return false;
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(g_vk.device, g_vk.font_img, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex =
      MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g_vk.device, &ai, nullptr, &g_vk.font_mem);
  vkBindImageMemory(g_vk.device, g_vk.font_img, g_vk.font_mem, 0);

  VkBuffer stg;
  VkDeviceMemory stg_mem;
  MakeBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
             stg, stg_mem);
  void* map = nullptr;
  vkMapMemory(g_vk.device, stg_mem, 0, bytes, 0, &map);
  base::MemCopy(map, px, bytes);
  vkUnmapMemory(g_vk.device, stg_mem);

  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_vk.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(g_vk.device, &ca, &cmd);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  auto barrier = [&](VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                     VkAccessFlags da, VkPipelineStageFlags ss,
                     VkPipelineStageFlags dds) {
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.oldLayout = from;
    bar.newLayout = to;
    bar.srcAccessMask = sa;
    bar.dstAccessMask = da;
    bar.image = g_vk.font_img;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd, ss, dds, 0, 0, nullptr, 0, nullptr, 1, &bar);
  };
  barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {(u32)w, (u32)h, 1};
  vkCmdCopyBufferToImage(cmd, stg, g_vk.font_img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
  barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(g_vk.queue);
  vkFreeCommandBuffers(g_vk.device, g_vk.pool, 1, &cmd);
  vkDestroyBuffer(g_vk.device, stg, nullptr);
  vkFreeMemory(g_vk.device, stg_mem, nullptr);

  VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  iv.image = g_vk.font_img;
  iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
  iv.format = VK_FORMAT_R8G8B8A8_UNORM;
  iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(g_vk.device, &iv, nullptr, &g_vk.font_view);

  VkSamplerCreateInfo sm{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sm.magFilter = sm.minFilter = VK_FILTER_LINEAR;
  sm.addressModeU = sm.addressModeV = sm.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  vkCreateSampler(g_vk.device, &sm, nullptr, &g_vk.sampler);

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = 1;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &ps;
  vkCreateDescriptorPool(g_vk.device, &dp, nullptr, &g_vk.desc_pool);
  VkDescriptorSetAllocateInfo da{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = g_vk.desc_pool;
  da.descriptorSetCount = 1;
  da.pSetLayouts = &g_vk.desc_layout;
  vkAllocateDescriptorSets(g_vk.device, &da, &g_vk.desc_set);
  VkDescriptorImageInfo dii{g_vk.sampler, g_vk.font_view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = g_vk.desc_set;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(g_vk.device, 1, &wr, 0, nullptr);
  io.Fonts->SetTexID((ImTextureID)(intptr_t)g_vk.desc_set);
  return true;
}

void DestroyMappedBuffer(VkBuffer& buffer, VkDeviceMemory& memory, void*& map) {
  if (map)
    vkUnmapMemory(g_vk.device, memory);
  if (buffer)
    vkDestroyBuffer(g_vk.device, buffer, nullptr);
  if (memory)
    vkFreeMemory(g_vk.device, memory, nullptr);
  buffer = VK_NULL_HANDLE;
  memory = VK_NULL_HANDLE;
  map = nullptr;
}

void DestroyFrame(Frame& frame) {
  DestroyMappedBuffer(frame.vtx, frame.vtx_mem, frame.vtx_map);
  DestroyMappedBuffer(frame.idx, frame.idx_mem, frame.idx_map);
  frame = {};
}

void EnsureFrameCapacity(Frame& f,
                         VkDeviceSize vtx_bytes,
                         VkDeviceSize idx_bytes) {
  if (f.vtx_cap < vtx_bytes) {
    DestroyMappedBuffer(f.vtx, f.vtx_mem, f.vtx_map);
    VkDeviceSize cap = vtx_bytes + 4096;
    MakeBuffer(cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               f.vtx, f.vtx_mem);
    vkMapMemory(g_vk.device, f.vtx_mem, 0, cap, 0, &f.vtx_map);
    f.vtx_cap = cap;
  }
  if (f.idx_cap < idx_bytes) {
    DestroyMappedBuffer(f.idx, f.idx_mem, f.idx_map);
    VkDeviceSize cap = idx_bytes + 4096;
    MakeBuffer(cap, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               f.idx, f.idx_mem);
    vkMapMemory(g_vk.device, f.idx_mem, 0, cap, 0, &f.idx_map);
    f.idx_cap = cap;
  }
}

void DestroyFramebuffers() {
  for (VkFramebuffer fb : g_vk.fbs)
    vkDestroyFramebuffer(g_vk.device, fb, nullptr);
  for (VkImageView iv : g_vk.views)
    vkDestroyImageView(g_vk.device, iv, nullptr);
  g_vk.fbs.clear();
  g_vk.views.clear();
}

}  // namespace

bool OverlayVkInit(VkPhysicalDevice phys,
                   VkDevice device,
                   VkQueue queue,
                   u32 queue_family,
                   VkCommandPool pool,
                   VkFormat swap_format) {
  if (g_vk.ready)
    return true;
  g_vk.phys = phys;
  g_vk.device = device;
  g_vk.queue = queue;
  g_vk.pool = pool;
  g_vk.format = swap_format;
  OverlayEnsureImGui();
  if (!CreatePipeline() || !UploadFont()) {
    BASE_LOGI("overlay", "Vulkan backend init failed");
    return false;
  }
  g_vk.ready = true;
  return true;
}

void OverlayVkSetSwapchain(const base::Vector<VkImage>& images,
                           VkExtent2D extent,
                           VkFormat format) {
  if (!g_vk.device)
    return;
  DestroyFramebuffers();
  for (Frame& frame : g_vk.frames)
    DestroyFrame(frame);
  g_vk.frames.clear();
  g_vk.extent = extent;
  g_vk.format = format;
  g_vk.frames.resize(images.size());
  for (VkImage img : images) {
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.image = img;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = format;
    iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(g_vk.device, &iv, nullptr, &view);
    g_vk.views.push_back(view);
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = g_vk.pass;
    fb.attachmentCount = 1;
    fb.pAttachments = &view;
    fb.width = extent.width;
    fb.height = extent.height;
    fb.layers = 1;
    VkFramebuffer f = VK_NULL_HANDLE;
    vkCreateFramebuffer(g_vk.device, &fb, nullptr, &f);
    g_vk.fbs.push_back(f);
  }
}

bool OverlayVkRender(VkCommandBuffer cmd, u32 image_index) {
  if (!g_vk.ready || image_index >= g_vk.fbs.size() ||
      image_index >= g_vk.frames.size())
    return false;
  const ImDrawData* dd = ImGui::GetDrawData();
  const int fb_w = g_vk.extent.width, fb_h = g_vk.extent.height;
  Frame& frame = g_vk.frames[image_index];

  if (dd && dd->TotalVtxCount > 0) {
    EnsureFrameCapacity(frame, dd->TotalVtxCount * sizeof(ImDrawVert),
                        dd->TotalIdxCount * sizeof(ImDrawIdx));
    auto* vtx = static_cast<ImDrawVert*>(frame.vtx_map);
    auto* idx = static_cast<ImDrawIdx*>(frame.idx_map);
    for (int i = 0; i < dd->CmdListsCount; i++) {
      const ImDrawList* cl = dd->CmdLists[i];
      base::MemCopy(vtx, cl->VtxBuffer.Data,
                    cl->VtxBuffer.Size * sizeof(ImDrawVert));
      base::MemCopy(idx, cl->IdxBuffer.Data,
                    cl->IdxBuffer.Size * sizeof(ImDrawIdx));
      vtx += cl->VtxBuffer.Size;
      idx += cl->IdxBuffer.Size;
    }
  }

  VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rp.renderPass = g_vk.pass;
  rp.framebuffer = g_vk.fbs[image_index];
  rp.renderArea.extent = g_vk.extent;
  vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

  if (dd && dd->TotalVtxCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            g_vk.pipe_layout, 0, 1, &g_vk.desc_set, 0, nullptr);
    VkViewport vpt{0, 0, (float)fb_w, (float)fb_h, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vpt);
    float push[4] = {2.0f / fb_w, 2.0f / fb_h, -1.0f, -1.0f};
    vkCmdPushConstants(cmd, g_vk.pipe_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(push), push);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &frame.vtx, &off);
    vkCmdBindIndexBuffer(
        cmd, frame.idx, 0,
        sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    int vtx_off = 0, idx_off = 0;
    for (int i = 0; i < dd->CmdListsCount; i++) {
      const ImDrawList* cl = dd->CmdLists[i];
      for (const ImDrawCmd& c : cl->CmdBuffer) {
        VkRect2D sc{};
        sc.offset.x = (i32)(c.ClipRect.x > 0 ? c.ClipRect.x : 0);
        sc.offset.y = (i32)(c.ClipRect.y > 0 ? c.ClipRect.y : 0);
        sc.extent.width = (u32)(c.ClipRect.z - sc.offset.x);
        sc.extent.height = (u32)(c.ClipRect.w - sc.offset.y);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdDrawIndexed(cmd, c.ElemCount, 1, c.IdxOffset + idx_off,
                         c.VtxOffset + vtx_off, 0);
      }
      vtx_off += cl->VtxBuffer.Size;
      idx_off += cl->IdxBuffer.Size;
    }
  }
  vkCmdEndRenderPass(cmd);
  return true;
}

void OverlayVkShutdown() {
  if (!g_vk.device)
    return;
  vkDeviceWaitIdle(g_vk.device);
  DestroyFramebuffers();
  for (Frame& frame : g_vk.frames)
    DestroyFrame(frame);
  if (g_vk.pipe)
    vkDestroyPipeline(g_vk.device, g_vk.pipe, nullptr);
  if (g_vk.pipe_layout)
    vkDestroyPipelineLayout(g_vk.device, g_vk.pipe_layout, nullptr);
  if (g_vk.desc_pool)
    vkDestroyDescriptorPool(g_vk.device, g_vk.desc_pool, nullptr);
  if (g_vk.desc_layout)
    vkDestroyDescriptorSetLayout(g_vk.device, g_vk.desc_layout, nullptr);
  if (g_vk.sampler)
    vkDestroySampler(g_vk.device, g_vk.sampler, nullptr);
  if (g_vk.font_view)
    vkDestroyImageView(g_vk.device, g_vk.font_view, nullptr);
  if (g_vk.font_img)
    vkDestroyImage(g_vk.device, g_vk.font_img, nullptr);
  if (g_vk.font_mem)
    vkFreeMemory(g_vk.device, g_vk.font_mem, nullptr);
  if (g_vk.pass)
    vkDestroyRenderPass(g_vk.device, g_vk.pass, nullptr);
  g_vk = {};
}

bool OverlayVkReady() {
  return g_vk.ready && !g_vk.fbs.empty();
}

}  // namespace ui
