/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless Vulkan renderer. See vk_render.h. Renders the decoded PM4 draws as
 * MVP-transformed quads into an offscreen render target that mirrors the guest
 * scanout, then reads it back (presented to a window when a display exists).
 */

#include "vk_render.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gfx/gfx.h"
#include "shaders/quad_vert_spv.h"
#include "shaders/quad_frag_spv.h"

namespace gpu::vk {
namespace {

#define VKOK(x)                                                                \
  do {                                                                         \
    VkResult _r = (x);                                                         \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gpuvk] %s failed: %d\n", #x, (int)_r);            \
      return false;                                                            \
    }                                                                          \
  } while (0)

constexpr VkDeviceSize kVbRing = 16ull * 1024 * 1024;  // per-frame vertex ring

struct State {
  bool ready = false;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t qfam = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkImage rt = VK_NULL_HANDLE;
  VkDeviceMemory rtMem = VK_NULL_HANDLE;
  VkImageView rtView = VK_NULL_HANDLE;
  uint32_t rtW = 0, rtH = 0;
  VkFormat rtFormat = VK_FORMAT_B8G8R8A8_UNORM;

  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;

  // Pipeline (textured/colored quad).
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  // Per-frame vertex ring (host-visible).
  VkBuffer vb = VK_NULL_HANDLE;
  VkDeviceMemory vbMem = VK_NULL_HANDLE;
  uint8_t *vbMap = nullptr;
  VkDeviceSize vbOffset = 0;

  uint32_t frameDraws = 0;
  bool recording = false;
} g;

const bool g_dump = std::getenv("DELTA_GPU_DUMP") != nullptr;
int g_dumpedFrames = 0;

PFN_vkCmdBeginRenderingKHR p_vkCmdBeginRendering = nullptr;
PFN_vkCmdEndRenderingKHR p_vkCmdEndRendering = nullptr;

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

void imageBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = srcA;
  b.dstAccessMask = dstA;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

bool createDevice() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_3;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  VKOK(vkCreateInstance(&ic, nullptr, &g.instance));

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
  if (!n) { std::fprintf(stderr, "[gpuvk] no device\n"); return false; }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(g.instance, &n, devs.data());
  g.phys = devs[0];

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, qprops.data());
  bool found = false;
  for (uint32_t i = 0; i < qn; i++)
    if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g.qfam = i; found = true; break; }
  if (!found) { std::fprintf(stderr, "[gpuvk] no gfx queue\n"); return false; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qc.queueFamilyIndex = g.qfam;
  qc.queueCount = 1;
  qc.pQueuePriorities = &prio;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  f13.dynamicRendering = VK_TRUE;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dc.pNext = &f13;
  dc.queueCreateInfoCount = 1;
  dc.pQueueCreateInfos = &qc;
  VKOK(vkCreateDevice(g.phys, &dc, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.qfam, 0, &g.queue);

  p_vkCmdBeginRendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdBeginRendering");
  p_vkCmdEndRendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdEndRendering");
  if (!p_vkCmdBeginRendering) {
    p_vkCmdBeginRendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdBeginRenderingKHR");
    p_vkCmdEndRendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(g.device, "vkCmdEndRenderingKHR");
  }
  if (!p_vkCmdBeginRendering) { std::fprintf(stderr, "[gpuvk] no dynamic rendering\n"); return false; }

  VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pc.queueFamilyIndex = g.qfam;
  VKOK(vkCreateCommandPool(g.device, &pc, nullptr, &g.pool));
  VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VKOK(vkAllocateCommandBuffers(g.device, &ca, &g.cmd));
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VKOK(vkCreateFence(g.device, &fc, nullptr, &g.fence));

  // Vertex ring.
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = kVbRing;
  bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VKOK(vkCreateBuffer(g.device, &bi, nullptr, &g.vb));
  VkMemoryRequirements vr;
  vkGetBufferMemoryRequirements(g.device, g.vb, &vr);
  VkMemoryAllocateInfo va{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  va.allocationSize = vr.size;
  va.memoryTypeIndex = findMemoryType(vr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g.device, &va, nullptr, &g.vbMem));
  VKOK(vkBindBufferMemory(g.device, g.vb, g.vbMem, 0));
  VKOK(vkMapMemory(g.device, g.vbMem, 0, kVbRing, 0, (void **)&g.vbMap));

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g.phys, &props);
  std::fprintf(stderr, "[gpuvk] device: %s\n", props.deviceName);
  return true;
}

VkShaderModule makeModule(const uint32_t *spv, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = spv;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(g.device, &ci, nullptr, &m);
  return m;
}

bool createPipeline() {
  if (g.pipeline)
    return true;
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};  // mat4
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g.device, &li, nullptr, &g.layout));

  VkShaderModule vs = makeModule(quad_vert_spv, sizeof(quad_vert_spv));
  VkShaderModule fs = makeModule(quad_frag_spv, sizeof(quad_frag_spv));
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs;
  stages[1].pName = "main";

  // Tight repacked vertex: float2 position, stride 8.
  VkVertexInputBindingDescription bind{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 1;
  vi.pVertexAttributeDescriptions = &attr;

  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  dss.depthTestEnable = VK_FALSE;
  dss.depthWriteEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;

  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyns;

  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = 1;
  rci.pColorAttachmentFormats = &g.rtFormat;

  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
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
  pi.layout = g.layout;
  VkResult r = vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &g.pipeline);
  vkDestroyShaderModule(g.device, vs, nullptr);
  vkDestroyShaderModule(g.device, fs, nullptr);
  if (r != VK_SUCCESS) { std::fprintf(stderr, "[gpuvk] pipeline failed: %d\n", (int)r); return false; }
  return true;
}

bool ensureRT(uint32_t w, uint32_t h, VkFormat fmt) {
  if (g.rt && g.rtW == w && g.rtH == h && g.rtFormat == fmt)
    return true;
  vkDeviceWaitIdle(g.device);
  if (g.rtView) vkDestroyImageView(g.device, g.rtView, nullptr);
  if (g.rt) vkDestroyImage(g.device, g.rt, nullptr);
  if (g.rtMem) vkFreeMemory(g.device, g.rtMem, nullptr);
  if (g.readback) vkDestroyBuffer(g.device, g.readback, nullptr);
  if (g.readbackMem) vkFreeMemory(g.device, g.readbackMem, nullptr);
  g.rt = VK_NULL_HANDLE; g.readback = VK_NULL_HANDLE; g.rtView = VK_NULL_HANDLE;
  g.rtW = w; g.rtH = h; g.rtFormat = fmt;

  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = fmt;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VKOK(vkCreateImage(g.device, &ii, nullptr, &g.rt));
  VkMemoryRequirements ir;
  vkGetImageMemoryRequirements(g.device, g.rt, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex = findMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VKOK(vkAllocateMemory(g.device, &ia, nullptr, &g.rtMem));
  VKOK(vkBindImageMemory(g.device, g.rt, g.rtMem, 0));

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = g.rt;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VKOK(vkCreateImageView(g.device, &vci, nullptr, &g.rtView));

  g.readbackSize = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = g.readbackSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VKOK(vkCreateBuffer(g.device, &bi, nullptr, &g.readback));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g.device, g.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g.device, &ba, nullptr, &g.readbackMem));
  VKOK(vkBindBufferMemory(g.device, g.readback, g.readbackMem, 0));
  VKOK(vkMapMemory(g.device, g.readbackMem, 0, g.readbackSize, 0, &g.readbackMap));
  std::fprintf(stderr, "[gpuvk] render target %ux%u fmt=%d\n", w, h, (int)fmt);
  return true;
}

void dumpPpm(const uint8_t *bgra, uint32_t w, uint32_t h) {
  if (g_dumpedFrames >= 4) return;
  char path[128];
  std::snprintf(path, sizeof(path), "/tmp/gpu_frame_%d.ppm", g_dumpedFrames++);
  FILE *f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; i++) {
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
  }
  std::fclose(f);
  std::fprintf(stderr, "[gpuvk] dumped %s\n", path);
}

}  // namespace

bool init() {
  if (g.ready) return true;
  if (!createDevice()) {
    std::fprintf(stderr, "[gpuvk] headless Vulkan unavailable; gpu disabled\n");
    return false;
  }
  g.ready = true;
  return true;
}

bool available() { return g.ready; }

void beginFrame(uint64_t, uint32_t width, uint32_t height, uint32_t, uint32_t) {
  if (!g.ready || !width || !height) return;
  if (!ensureRT(width, height, g.rtFormat)) return;
  if (!createPipeline()) return;
  g.frameDraws = 0;
  g.vbOffset = 0;

  vkResetCommandBuffer(g.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g.cmd, &bi);

  imageBarrier(g.cmd, g.rt, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = g.rtView;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {width, height}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  p_vkCmdBeginRendering(g.cmd, &ri);

  vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeline);
  VkViewport vpt{0, 0, (float)width, (float)height, 0, 1};
  vkCmdSetViewport(g.cmd, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {width, height}};
  vkCmdSetScissor(g.cmd, 0, 1, &sc);
  g.recording = true;
}

void draw(const DrawInfo &d) {
  if (!g.recording || !d.vertexData || d.vertexCount < 3 || !d.vertexStride)
    return;
  uint32_t nv = d.vertexCount;
  VkDeviceSize need = (VkDeviceSize)nv * 8;
  if (g.vbOffset + need > kVbRing)
    return;  // ring full this frame
  // Repack the float2 positions tightly into the vertex ring.
  auto *src = static_cast<const uint8_t *>(d.vertexData) + d.posOffset;
  auto *dst = reinterpret_cast<float *>(g.vbMap + g.vbOffset);
  for (uint32_t v = 0; v < nv; v++) {
    auto *p = reinterpret_cast<const float *>(src + (size_t)v * d.vertexStride);
    dst[v * 2 + 0] = p[0];
    dst[v * 2 + 1] = p[1];
  }
  static int dbg = 0;
  if (g_dump && dbg < 6) {
    dbg++;
    const float *m = d.mvp;
    std::fprintf(stderr, "[gpuvk] draw nv=%u mvp=[%g %g %g %g | %g %g %g %g | "
                 "%g %g %g %g | %g %g %g %g]\n", nv, m[0],m[1],m[2],m[3],
                 m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15]);
    for (uint32_t v = 0; v < nv && v < 4; v++) {
      float x = dst[v*2], y = dst[v*2+1];
      // clip = mvp(column-major) * (x,y,0,1)
      float cx = m[0]*x + m[4]*y + m[12];
      float cy = m[1]*x + m[5]*y + m[13];
      float cw = m[3]*x + m[7]*y + m[15];
      std::fprintf(stderr, "[gpuvk]   v%u pos=(%g,%g) clip=(%g,%g) w=%g ndc=(%g,%g)\n",
                   v, x, y, cx, cy, cw, cw!=0?cx/cw:0, cw!=0?cy/cw:0);
    }
  }
  vkCmdPushConstants(g.cmd, g.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, d.mvp);
  VkDeviceSize off = g.vbOffset;
  vkCmdBindVertexBuffers(g.cmd, 0, 1, &g.vb, &off);
  vkCmdDraw(g.cmd, nv, 1, 0, 0);
  g.vbOffset += need;
  g.frameDraws++;
}

void endFrame() {
  if (!g.ready || !g.recording) return;
  g.recording = false;
  p_vkCmdEndRendering(g.cmd);

  imageBarrier(g.cmd, g.rt, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {g.rtW, g.rtH, 1};
  vkCmdCopyImageToBuffer(g.cmd, g.rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         g.readback, 1, &copy);
  vkEndCommandBuffer(g.cmd);

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g.cmd;
  vkResetFences(g.device, 1, &g.fence);
  vkQueueSubmit(g.queue, 1, &si, g.fence);
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);

  static int frameNum = 0;
  ++frameNum;
  auto *pixels = static_cast<const uint8_t *>(g.readbackMap);
  // Dump frames well into the run (past the empty loading frames) where the menu
  // actually issues draws.
  if (g_dump && frameNum >= 400 && g.frameDraws > 0) dumpPpm(pixels, g.rtW, g.rtH);
  if (g_dump && frameNum % 200 == 0)
    std::fprintf(stderr, "[gpuvk] frame %d draws=%u\n", frameNum, g.frameDraws);
  // Present to the window. Gated by env so headless testing (PPM dump) doesn't
  // hit the windowing/swapchain path (which needs a real display).
  static const bool present = std::getenv("DELTA_GPU_PRESENT") != nullptr;
  if (present && gfx::pumpEvents())
    gfx::present(pixels, g.rtW, g.rtH, g.rtW * 4, gfx::PixelFormat::bgra8);
}

}  // namespace gpu::vk
