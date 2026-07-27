/*
 * PS4Delta : PS4 emulation and research project
 *
 * Vulkan backend for the Dear ImGui overlay. See overlay_vk.h. A small pipeline
 * (embedded SPIR-V, ImDrawVert layout, alpha blend, dynamic viewport/scissor)
 * draws the overlay's ImDrawData into the swapchain image through a LOAD render
 * pass that also transitions the image from TRANSFER_DST (the frame blit) to
 * PRESENT_SRC, so gfx_vk's present path just calls overlayVkRender.
 */
#ifndef __ANDROID__

#include "overlay_vk.h"

#include <cstdio>
#include <cstring>

#include "imgui.h"
#include "overlay.h"
#include "overlay_vk_shaders.h"

namespace gfx {
namespace {

struct Frame {
  VkBuffer vtx = VK_NULL_HANDLE, idx = VK_NULL_HANDLE;
  VkDeviceMemory vtxMem = VK_NULL_HANDLE, idxMem = VK_NULL_HANDLE;
  VkDeviceSize vtxCap = 0, idxCap = 0;
  void *vtxMap = nullptr, *idxMap = nullptr;
};

struct {
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};

  VkRenderPass pass = VK_NULL_HANDLE;
  VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
  VkPipeline pipe = VK_NULL_HANDLE;
  VkDescriptorPool descPool = VK_NULL_HANDLE;
  VkDescriptorSet descSet = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;

  VkImage fontImg = VK_NULL_HANDLE;
  VkDeviceMemory fontMem = VK_NULL_HANDLE;
  VkImageView fontView = VK_NULL_HANDLE;

  std::vector<VkImageView> views;
  std::vector<VkFramebuffer> fbs;
  std::vector<Frame> frames;
  bool ready = false;
} v;

uint32_t memType(uint32_t bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(v.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

bool makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                VkMemoryPropertyFlags props, VkBuffer &buf,
                VkDeviceMemory &mem) {
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(v.device, &bi, nullptr, &buf) != VK_SUCCESS)
    return false;
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(v.device, buf, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = memType(req.memoryTypeBits, props);
  if (vkAllocateMemory(v.device, &ai, nullptr, &mem) != VK_SUCCESS)
    return false;
  vkBindBufferMemory(v.device, buf, mem, 0);
  return true;
}

VkShaderModule makeModule(const uint32_t *code, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = code;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(v.device, &ci, nullptr, &m);
  return m;
}

bool createPipeline() {
  VkAttachmentDescription att{};
  att.format = v.format;
  att.samples = VK_SAMPLE_COUNT_1_BIT;
  att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // preserve the blitted frame
  att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  att.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; // after the blit
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
  if (vkCreateRenderPass(v.device, &rp, nullptr, &v.pass) != VK_SUCCESS)
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
  vkCreateDescriptorSetLayout(v.device, &dl, nullptr, &v.descLayout);

  VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 4};
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &v.descLayout;
  pl.pushConstantRangeCount = 1;
  pl.pPushConstantRanges = &pc;
  vkCreatePipelineLayout(v.device, &pl, nullptr, &v.pipeLayout);

  VkShaderModule vs = makeModule(kImguiVertSpv, sizeof(kImguiVertSpv));
  VkShaderModule fs = makeModule(kImguiFragSpv, sizeof(kImguiFragSpv));
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
      {0, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)IM_OFFSETOF(ImDrawVert, pos)},
      {1, 0, VK_FORMAT_R32G32_SFLOAT, (uint32_t)IM_OFFSETOF(ImDrawVert, uv)},
      {2, 0, VK_FORMAT_R8G8B8A8_UNORM, (uint32_t)IM_OFFSETOF(ImDrawVert, col)}};
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
  gp.layout = v.pipeLayout;
  gp.renderPass = v.pass;
  VkResult r = vkCreateGraphicsPipelines(v.device, VK_NULL_HANDLE, 1, &gp,
                                         nullptr, &v.pipe);
  vkDestroyShaderModule(v.device, vs, nullptr);
  vkDestroyShaderModule(v.device, fs, nullptr);
  return r == VK_SUCCESS;
}

bool uploadFont() {
  ImGuiIO &io = ImGui::GetIO();
  unsigned char *px = nullptr;
  int w = 0, h = 0;
  io.Fonts->GetTexDataAsRGBA32(&px, &w, &h);
  const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;

  VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ic.imageType = VK_IMAGE_TYPE_2D;
  ic.format = VK_FORMAT_R8G8B8A8_UNORM;
  ic.extent = {(uint32_t)w, (uint32_t)h, 1};
  ic.mipLevels = 1;
  ic.arrayLayers = 1;
  ic.samples = VK_SAMPLE_COUNT_1_BIT;
  ic.tiling = VK_IMAGE_TILING_OPTIMAL;
  ic.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(v.device, &ic, nullptr, &v.fontImg) != VK_SUCCESS)
    return false;
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(v.device, v.fontImg, &req);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex =
      memType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(v.device, &ai, nullptr, &v.fontMem);
  vkBindImageMemory(v.device, v.fontImg, v.fontMem, 0);

  VkBuffer stg;
  VkDeviceMemory stgMem;
  makeBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
             stg, stgMem);
  void *map = nullptr;
  vkMapMemory(v.device, stgMem, 0, bytes, 0, &map);
  std::memcpy(map, px, bytes);
  vkUnmapMemory(v.device, stgMem);

  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = v.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(v.device, &ca, &cmd);
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
    bar.image = v.fontImg;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd, ss, dds, 0, 0, nullptr, 0, nullptr, 1, &bar);
  };
  barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
  vkCmdCopyBufferToImage(cmd, stg, v.fontImg,
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
  vkQueueSubmit(v.queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(v.queue);
  vkFreeCommandBuffers(v.device, v.pool, 1, &cmd);
  vkDestroyBuffer(v.device, stg, nullptr);
  vkFreeMemory(v.device, stgMem, nullptr);

  VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  iv.image = v.fontImg;
  iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
  iv.format = VK_FORMAT_R8G8B8A8_UNORM;
  iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(v.device, &iv, nullptr, &v.fontView);

  VkSamplerCreateInfo sm{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sm.magFilter = sm.minFilter = VK_FILTER_LINEAR;
  sm.addressModeU = sm.addressModeV = sm.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  vkCreateSampler(v.device, &sm, nullptr, &v.sampler);

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = 1;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &ps;
  vkCreateDescriptorPool(v.device, &dp, nullptr, &v.descPool);
  VkDescriptorSetAllocateInfo da{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = v.descPool;
  da.descriptorSetCount = 1;
  da.pSetLayouts = &v.descLayout;
  vkAllocateDescriptorSets(v.device, &da, &v.descSet);
  VkDescriptorImageInfo dii{v.sampler, v.fontView,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = v.descSet;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(v.device, 1, &wr, 0, nullptr);
  io.Fonts->SetTexID((ImTextureID)(intptr_t)v.descSet);
  return true;
}

void destroyMappedBuffer(VkBuffer &buffer, VkDeviceMemory &memory, void *&map) {
  if (map)
    vkUnmapMemory(v.device, memory);
  if (buffer)
    vkDestroyBuffer(v.device, buffer, nullptr);
  if (memory)
    vkFreeMemory(v.device, memory, nullptr);
  buffer = VK_NULL_HANDLE;
  memory = VK_NULL_HANDLE;
  map = nullptr;
}

void destroyFrame(Frame &frame) {
  destroyMappedBuffer(frame.vtx, frame.vtxMem, frame.vtxMap);
  destroyMappedBuffer(frame.idx, frame.idxMem, frame.idxMap);
  frame = {};
}

void ensureFrameCapacity(Frame &f, VkDeviceSize vtxBytes,
                         VkDeviceSize idxBytes) {
  if (f.vtxCap < vtxBytes) {
    destroyMappedBuffer(f.vtx, f.vtxMem, f.vtxMap);
    VkDeviceSize cap = vtxBytes + 4096;
    makeBuffer(cap, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               f.vtx, f.vtxMem);
    vkMapMemory(v.device, f.vtxMem, 0, cap, 0, &f.vtxMap);
    f.vtxCap = cap;
  }
  if (f.idxCap < idxBytes) {
    destroyMappedBuffer(f.idx, f.idxMem, f.idxMap);
    VkDeviceSize cap = idxBytes + 4096;
    makeBuffer(cap, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               f.idx, f.idxMem);
    vkMapMemory(v.device, f.idxMem, 0, cap, 0, &f.idxMap);
    f.idxCap = cap;
  }
}

void destroyFramebuffers() {
  for (VkFramebuffer fb : v.fbs)
    vkDestroyFramebuffer(v.device, fb, nullptr);
  for (VkImageView iv : v.views)
    vkDestroyImageView(v.device, iv, nullptr);
  v.fbs.clear();
  v.views.clear();
}

} // namespace

bool overlayVkInit(VkPhysicalDevice phys, VkDevice device, VkQueue queue,
                   uint32_t queueFamily, VkCommandPool pool,
                   VkFormat swapFormat) {
  if (v.ready)
    return true;
  v.phys = phys;
  v.device = device;
  v.queue = queue;
  v.pool = pool;
  v.format = swapFormat;
  overlayEnsureImGui();
  if (!createPipeline() || !uploadFont()) {
    std::fprintf(stderr, "[overlay] Vulkan backend init failed\n");
    return false;
  }
  v.ready = true;
  return true;
}

void overlayVkSetSwapchain(const std::vector<VkImage> &images,
                           VkExtent2D extent, VkFormat format) {
  if (!v.device)
    return;
  destroyFramebuffers();
  for (Frame &frame : v.frames)
    destroyFrame(frame);
  v.frames.clear();
  v.extent = extent;
  v.format = format;
  v.frames.resize(images.size());
  for (VkImage img : images) {
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.image = img;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = format;
    iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(v.device, &iv, nullptr, &view);
    v.views.push_back(view);
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass = v.pass;
    fb.attachmentCount = 1;
    fb.pAttachments = &view;
    fb.width = extent.width;
    fb.height = extent.height;
    fb.layers = 1;
    VkFramebuffer f = VK_NULL_HANDLE;
    vkCreateFramebuffer(v.device, &fb, nullptr, &f);
    v.fbs.push_back(f);
  }
}

bool overlayVkRender(VkCommandBuffer cmd, uint32_t imageIndex) {
  if (!v.ready || imageIndex >= v.fbs.size() || imageIndex >= v.frames.size())
    return false;
  const ImDrawData *dd = ImGui::GetDrawData();
  const int fbW = v.extent.width, fbH = v.extent.height;
  Frame &frame = v.frames[imageIndex];

  if (dd && dd->TotalVtxCount > 0) {
    ensureFrameCapacity(frame, dd->TotalVtxCount * sizeof(ImDrawVert),
                        dd->TotalIdxCount * sizeof(ImDrawIdx));
    auto *vtx = static_cast<ImDrawVert *>(frame.vtxMap);
    auto *idx = static_cast<ImDrawIdx *>(frame.idxMap);
    for (int i = 0; i < dd->CmdListsCount; i++) {
      const ImDrawList *cl = dd->CmdLists[i];
      std::memcpy(vtx, cl->VtxBuffer.Data,
                  cl->VtxBuffer.Size * sizeof(ImDrawVert));
      std::memcpy(idx, cl->IdxBuffer.Data,
                  cl->IdxBuffer.Size * sizeof(ImDrawIdx));
      vtx += cl->VtxBuffer.Size;
      idx += cl->IdxBuffer.Size;
    }
  }

  VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  rp.renderPass = v.pass;
  rp.framebuffer = v.fbs[imageIndex];
  rp.renderArea.extent = v.extent;
  vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

  if (dd && dd->TotalVtxCount > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, v.pipeLayout,
                            0, 1, &v.descSet, 0, nullptr);
    VkViewport vpt{0, 0, (float)fbW, (float)fbH, 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vpt);
    float push[4] = {2.0f / fbW, 2.0f / fbH, -1.0f, -1.0f};
    vkCmdPushConstants(cmd, v.pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(push), push);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &frame.vtx, &off);
    vkCmdBindIndexBuffer(cmd, frame.idx, 0,
                         sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16
                                                : VK_INDEX_TYPE_UINT32);
    int vtxOff = 0, idxOff = 0;
    for (int i = 0; i < dd->CmdListsCount; i++) {
      const ImDrawList *cl = dd->CmdLists[i];
      for (const ImDrawCmd &c : cl->CmdBuffer) {
        VkRect2D sc{};
        sc.offset.x = (int32_t)(c.ClipRect.x > 0 ? c.ClipRect.x : 0);
        sc.offset.y = (int32_t)(c.ClipRect.y > 0 ? c.ClipRect.y : 0);
        sc.extent.width = (uint32_t)(c.ClipRect.z - sc.offset.x);
        sc.extent.height = (uint32_t)(c.ClipRect.w - sc.offset.y);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdDrawIndexed(cmd, c.ElemCount, 1, c.IdxOffset + idxOff,
                         c.VtxOffset + vtxOff, 0);
      }
      vtxOff += cl->VtxBuffer.Size;
      idxOff += cl->IdxBuffer.Size;
    }
  }
  vkCmdEndRenderPass(cmd);
  return true;
}

void overlayVkShutdown() {
  if (!v.device)
    return;
  vkDeviceWaitIdle(v.device);
  destroyFramebuffers();
  for (Frame &frame : v.frames)
    destroyFrame(frame);
  if (v.pipe)
    vkDestroyPipeline(v.device, v.pipe, nullptr);
  if (v.pipeLayout)
    vkDestroyPipelineLayout(v.device, v.pipeLayout, nullptr);
  if (v.descPool)
    vkDestroyDescriptorPool(v.device, v.descPool, nullptr);
  if (v.descLayout)
    vkDestroyDescriptorSetLayout(v.device, v.descLayout, nullptr);
  if (v.sampler)
    vkDestroySampler(v.device, v.sampler, nullptr);
  if (v.fontView)
    vkDestroyImageView(v.device, v.fontView, nullptr);
  if (v.fontImg)
    vkDestroyImage(v.device, v.fontImg, nullptr);
  if (v.fontMem)
    vkFreeMemory(v.device, v.fontMem, nullptr);
  if (v.pass)
    vkDestroyRenderPass(v.device, v.pass, nullptr);
  v = {};
}

bool overlayVkReady() { return v.ready && !v.fbs.empty(); }

} // namespace gfx

#endif // !__ANDROID__
