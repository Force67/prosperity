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
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gfx/gfx.h"
#include "shaders/quad_vert_spv.h"
#include "shaders/quad_frag_spv.h"
#include "shaders/tex_vert_spv.h"
#include "shaders/tex_frag_spv.h"

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

  VkFormat rtFormat = VK_FORMAT_B8G8R8A8_UNORM;

  // Shared readback buffer (sized for the largest RT presented).
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;

  uint64_t curRt = 0;   // RT currently in a dynamic-rendering region (0 = none)
  uint64_t lastRt = 0;  // last RT rendered to (present fallback)
  uint64_t busiestRt = 0;
  uint32_t busiestRtDraws = 0;

  // Pipeline (textured/colored quad).
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  // Per-frame vertex ring (host-visible). Holds interleaved pos+uv (vec4) per vert.
  VkBuffer vb = VK_NULL_HANDLE;
  VkDeviceMemory vbMem = VK_NULL_HANDLE;
  uint8_t *vbMap = nullptr;
  VkDeviceSize vbOffset = 0;

  // Textured pipeline + texture cache.
  VkPipeline texPipeline = VK_NULL_HANDLE;
  VkPipelineLayout texLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
  VkDescriptorPool dsPool = VK_NULL_HANDLE;
  VkSampler sampler = VK_NULL_HANDLE;

  uint32_t frameDraws = 0;
  int frameNum = 0;
  bool recording = false;
} g;

struct TexEntry {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  uint32_t w = 0, h = 0;
  uint64_t hash = 0;  // cheap fingerprint of guest pixels; re-upload when it changes
};
std::unordered_map<uint64_t, TexEntry> g_texCache;

// Cheap content fingerprint of a guest texture: sample a spread of dwords. Lets
// us re-upload dynamic textures (room art composed/loaded after first sample)
// instead of serving the stale first upload.
uint64_t texHash(uint64_t base, uint32_t w, uint32_t h) {
  const uint32_t *p = reinterpret_cast<const uint32_t *>(base);
  uint64_t count = (uint64_t)w * h;            // dwords (RGBA8)
  uint64_t step = count > 512 ? count / 512 : 1;
  uint64_t hsh = 1469598103934665603ull;       // FNV-ish
  for (uint64_t i = 0; i < count; i += step) {
    hsh = (hsh ^ p[i]) * 1099511628211ull;
  }
  return hsh ^ (count << 1);
}

// A render target, keyed by its guest CB_COLOR0 address. Doubles as a sampleable
// texture (render-to-texture) for composite passes.
struct RTarget {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;  // for sampling this RT as a texture
  uint32_t w = 0, h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool usedThisFrame = false;
  uint32_t draws = 0;  // draws into this RT this frame
};
std::unordered_map<uint64_t, RTarget> g_rts;

const bool g_dump = std::getenv("DELTA_GPU_DUMP") != nullptr;
int g_dumpedFrames = 0;

// Directory frame dumps go to. Defaults to /tmp; Android has no /tmp, so the
// runner sets DELTA_GPU_DUMP_DIR to a writable path (e.g. the cwd under
// /data/local/tmp). Returned without a trailing slash.
const char *dumpDir() {
  const char *d = std::getenv("DELTA_GPU_DUMP_DIR");
  return (d && *d) ? d : "/tmp";
}

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

  // Prefer a real GPU over the llvmpipe software rasteriser (reported as type
  // CPU): discrete > integrated > virtual > CPU. The loader can enumerate both
  // a discrete GPU and llvmpipe on the same box, so picking devs[0] blindly may
  // land on software. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char *want = std::getenv("DELTA_VK_GPU");
  int best = -1;
  g.phys = VK_NULL_HANDLE;
  for (VkPhysicalDevice d : devs) {
    uint32_t dqn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, nullptr);
    std::vector<VkQueueFamilyProperties> dq(dqn);
    vkGetPhysicalDeviceQueueFamilyProperties(d, &dqn, dq.data());
    bool gfx = false;
    for (auto &q : dq)
      if (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) { gfx = true; break; }
    if (!gfx) continue;
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(d, &p);
    int score;
    switch (p.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 4; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 2; break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 0; break;  // llvmpipe
      default:                                     score = 1; break;
    }
    if (want && std::strstr(p.deviceName, want)) score = 100;
    if (score > best) { best = score; g.phys = d; }
  }
  if (g.phys == VK_NULL_HANDLE) { std::fprintf(stderr, "[gpuvk] no gfx device\n"); return false; }

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

  // Interleaved repacked vertex: pos.xy@0, color.rgba@8, uv.xy@24, stride 32
  // (shared layout with the textured pipeline).
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

bool createTexPipeline() {
  if (g.texPipeline)
    return true;
  // descriptor set layout: binding 0 = combined image sampler (fragment).
  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                 VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 1;
  dl.pBindings = &b;
  VKOK(vkCreateDescriptorSetLayout(g.device, &dl, nullptr, &g.dsLayout));

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = 4096;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &ps;
  VKOK(vkCreateDescriptorPool(g.device, &dp, nullptr, &g.dsPool));

  VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sc.magFilter = sc.minFilter = VK_FILTER_LINEAR;
  sc.addressModeU = sc.addressModeV = sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VKOK(vkCreateSampler(g.device, &sc, nullptr, &g.sampler));

  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, 68};  // mat4 + clipUV flag
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1;
  li.pSetLayouts = &g.dsLayout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  VKOK(vkCreatePipelineLayout(g.device, &li, nullptr, &g.texLayout));

  VkShaderModule vs = makeModule(tex_vert_spv, sizeof(tex_vert_spv));
  VkShaderModule fs = makeModule(tex_frag_spv, sizeof(tex_frag_spv));
  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs; stages[0].pName = "main";
  stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fs; stages[1].pName = "main";

  VkVertexInputBindingDescription bind{0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription attrs[3] = {
      {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},        // pos
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 8},  // color
      {2, 0, VK_FORMAT_R32G32_SFLOAT, 24},       // uv
  };
  VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &bind;
  vi.vertexAttributeDescriptionCount = 3;
  vi.pVertexAttributeDescriptions = attrs;

  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1; vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  VkPipelineColorBlendAttachmentState cba{};
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1; cb.pAttachments = &cba;
  VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2; dy.pDynamicStates = dyns;
  VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  rci.colorAttachmentCount = 1; rci.pColorAttachmentFormats = &g.rtFormat;
  VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  pi.pNext = &rci; pi.stageCount = 2; pi.pStages = stages;
  pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
  pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &dss;
  pi.pColorBlendState = &cb; pi.pDynamicState = &dy; pi.layout = g.texLayout;
  VkResult r = vkCreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &pi, nullptr, &g.texPipeline);
  vkDestroyShaderModule(g.device, vs, nullptr);
  vkDestroyShaderModule(g.device, fs, nullptr);
  if (r != VK_SUCCESS) { std::fprintf(stderr, "[gpuvk] tex pipeline failed: %d\n", (int)r); return false; }
  return true;
}

// Copy guest pixels (linear 32bpp RGBA) into `img` via a staging buffer.
void uploadTexPixels(VkImage img, uint64_t base, uint32_t w, uint32_t h) {
  VkDeviceSize sz = (VkDeviceSize)w * h * 4;
  VkBuffer stg; VkDeviceMemory stgMem; void *map;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = sz; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCreateBuffer(g.device, &bi, nullptr, &stg);
  VkMemoryRequirements br; vkGetBufferMemoryRequirements(g.device, stg, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g.device, &ba, nullptr, &stgMem);
  vkBindBufferMemory(g.device, stg, stgMem, 0);
  vkMapMemory(g.device, stgMem, 0, sz, 0, &map);
  std::memcpy(map, reinterpret_cast<const void *>(base), sz);
  vkUnmapMemory(g.device, stgMem);

  VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
  VkCommandBuffer c; vkAllocateCommandBuffers(g.device, &ca, &c);
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(c, &cbi);
  imageBarrier(c, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               0, VK_ACCESS_TRANSFER_WRITE_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(c, stg, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
  imageBarrier(c, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT);
  vkEndCommandBuffer(c);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &c;
  vkResetFences(g.device, 1, &g.fence);
  vkQueueSubmit(g.queue, 1, &si, g.fence);
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
  vkFreeCommandBuffers(g.device, g.pool, 1, &c);
  vkDestroyBuffer(g.device, stg, nullptr);
  vkFreeMemory(g.device, stgMem, nullptr);
}

// Upload a guest texture (linear 32bpp RGBA) and return a descriptor set bound
// to it. Cached by guest base; re-uploaded when the guest pixels change (the
// room art is composed/loaded into the same buffer after the first sample, so a
// once-only cache would serve a stale black frame).
VkDescriptorSet getTexture(uint64_t base, uint32_t w, uint32_t h) {
  if (!w || !h) return VK_NULL_HANDLE;
  uint64_t hsh = texHash(base, w, h);
  auto it = g_texCache.find(base);
  if (it != g_texCache.end()) {
    TexEntry &e = it->second;
    if (e.w == w && e.h == h) {
      if (e.hash != hsh) {  // dynamic texture changed -> re-upload pixels
        uploadTexPixels(e.image, base, w, h);
        e.hash = hsh;
      }
      return e.set;
    }
    // size changed (buffer reused for a different texture): fall through to recreate
  }
  if (g_texCache.size() > 3000) return VK_NULL_HANDLE;
  TexEntry e; e.w = w; e.h = h; e.hash = hsh;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = VK_FORMAT_R8G8B8A8_UNORM;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g.device, &ii, nullptr, &e.image) != VK_SUCCESS) return VK_NULL_HANDLE;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g.device, e.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g.device, &ai, nullptr, &e.mem);
  vkBindImageMemory(g.device, e.image, e.mem, 0);

  uploadTexPixels(e.image, base, w, h);

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = e.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = VK_FORMAT_R8G8B8A8_UNORM;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(g.device, &vci, nullptr, &e.view);

  VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = g.dsPool; da.descriptorSetCount = 1; da.pSetLayouts = &g.dsLayout;
  if (vkAllocateDescriptorSets(g.device, &da, &e.set) != VK_SUCCESS) return VK_NULL_HANDLE;
  VkDescriptorImageInfo dii{g.sampler, e.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = e.set; wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(g.device, 1, &wr, 0, nullptr);

  g_texCache[base] = e;
  return e.set;
}

void ensureReadback(uint32_t w, uint32_t h) {
  VkDeviceSize need = (VkDeviceSize)w * h * 4;
  if (g.readback && need <= g.readbackSize)
    return;
  vkDeviceWaitIdle(g.device);
  if (g.readbackMap) vkUnmapMemory(g.device, g.readbackMem);
  if (g.readback) vkDestroyBuffer(g.device, g.readback, nullptr);
  if (g.readbackMem) vkFreeMemory(g.device, g.readbackMem, nullptr);
  g.readbackSize = need;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = need; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(g.device, &bi, nullptr, &g.readback);
  VkMemoryRequirements br; vkGetBufferMemoryRequirements(g.device, g.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g.device, &ba, nullptr, &g.readbackMem);
  vkBindBufferMemory(g.device, g.readback, g.readbackMem, 0);
  vkMapMemory(g.device, g.readbackMem, 0, need, 0, &g.readbackMap);
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget *getRT(uint64_t base, uint32_t w, uint32_t h) {
  auto it = g_rts.find(base);
  if (it != g_rts.end())
    return &it->second;
  if (g_rts.size() > 64 || !w || !h)
    return nullptr;
  RTarget t; t.w = w; t.h = h;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = g.rtFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g.device, &ii, nullptr, &t.image) != VK_SUCCESS) return nullptr;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g.device, t.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g.device, &ai, nullptr, &t.mem);
  vkBindImageMemory(g.device, t.image, t.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = g.rtFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(g.device, &vci, nullptr, &t.view);
  // descriptor set so this RT can be sampled (render-to-texture).
  if (g.dsPool) {
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = g.dsPool; da.descriptorSetCount = 1; da.pSetLayouts = &g.dsLayout;
    if (vkAllocateDescriptorSets(g.device, &da, &t.set) == VK_SUCCESS) {
      VkDescriptorImageInfo dii{g.sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set; wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new RT %#lx %ux%u\n", (unsigned long)base, w, h);
  g_rts[base] = t;
  return &g_rts[base];
}

// End the current dynamic-rendering region (if any), leaving its RT readable.
void endRegion() {
  if (!g.curRt) return;
  p_vkCmdEndRendering(g.cmd);
  auto &rt = g_rts[g.curRt];
  imageBarrier(g.cmd, rt.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  rt.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  g.curRt = 0;
}

// Begin a dynamic-rendering region on `rt` (clear on first use this frame).
void beginRegion(uint64_t base, RTarget &rt) {
  VkImageLayout from = rt.layout;
  imageBarrier(g.cmd, rt.image, from, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
  rt.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = rt.view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = rt.usedThisFrame ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {rt.w, rt.h}};
  ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &color;
  p_vkCmdBeginRendering(g.cmd, &ri);
  VkViewport vpt{0, 0, (float)rt.w, (float)rt.h, 0, 1};
  vkCmdSetViewport(g.cmd, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {rt.w, rt.h}};
  vkCmdSetScissor(g.cmd, 0, 1, &sc);
  rt.usedThisFrame = true;
  g.curRt = base;
  g.lastRt = base;
}

void writePpm(const char *path, const uint8_t *bgra, uint32_t w, uint32_t h) {
  FILE *f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; i++) {
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
  }
  std::fclose(f);
}

void dumpPpm(const uint8_t *bgra, uint32_t w, uint32_t h) {
  if (g_dumpedFrames >= 4) return;
  char path[256];
  std::snprintf(path, sizeof(path), "%s/gpu_frame_%d.ppm", dumpDir(), g_dumpedFrames++);
  writePpm(path, bgra, w, h);
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

void beginFrame() {
  if (!g.ready) return;
  if (!createPipeline()) return;
  createTexPipeline();  // best-effort; colored path still works without it
  g.frameDraws = 0;
  g.frameNum++;
  g.vbOffset = 0;
  g.curRt = 0;
  g.lastRt = 0;
  g.busiestRt = 0;
  g.busiestRtDraws = 0;
  for (auto &kv : g_rts) {
    kv.second.usedThisFrame = false;
    kv.second.draws = 0;
  }

  vkResetCommandBuffer(g.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g.cmd, &bi);
  g.recording = true;
}

void draw(const DrawInfo &d) {
  if (!g.recording || !d.vertexData || d.vertexCount < 3 || !d.vertexStride)
    return;
  uint32_t nv = d.vertexCount;
  if (nv > 4096) return;  // sane cap (Isaac quads are tiny)
  VkDeviceSize need = (VkDeviceSize)nv * 32;  // pos.xy + color.rgba + uv.xy
  if (g.vbOffset + need > kVbRing)
    return;  // ring full this frame
  // Repack pos / color / uv interleaved into the vertex ring (stride 32).
  auto *base = static_cast<const uint8_t *>(d.vertexData);
  auto *dst = reinterpret_cast<float *>(g.vbMap + g.vbOffset);
  for (uint32_t v = 0; v < nv; v++) {
    const uint8_t *vert = base + (size_t)v * d.vertexStride;
    auto *p = reinterpret_cast<const float *>(vert + d.posOffset);
    dst[v * 8 + 0] = p[0];
    dst[v * 8 + 1] = p[1];
    if (d.colorOffset != 0xFFFFFFFFu) {
      auto *c = reinterpret_cast<const float *>(vert + d.colorOffset);
      dst[v * 8 + 2] = c[0];
      dst[v * 8 + 3] = c[1];
      dst[v * 8 + 4] = c[2];
      dst[v * 8 + 5] = 1.0f;
    } else {
      dst[v * 8 + 2] = dst[v * 8 + 3] = dst[v * 8 + 4] = dst[v * 8 + 5] = 1.0f;
    }
    if (d.uvData && d.uvStride) {
      auto *u = reinterpret_cast<const float *>(vert + d.uvOffset);
      dst[v * 8 + 6] = u[0];
      dst[v * 8 + 7] = u[1];
    } else {
      dst[v * 8 + 6] = dst[v * 8 + 7] = 0.0f;
    }
  }

  if (g_dump && g.frameNum == 450) {
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (uint32_t v = 0; v < nv; v++) {
      float x = dst[v * 8], y = dst[v * 8 + 1];
      minx = x < minx ? x : minx; maxx = x > maxx ? x : maxx;
      miny = y < miny ? y : miny; maxy = y > maxy ? y : maxy;
    }
    std::fprintf(stderr, "[gpuvk] f450 draw#%u nv=%u pos[%.0f,%.0f..%.0f,%.0f] "
                 "rt=%#lx tex=%#lx rtAsTex=%d col=(%.2f,%.2f,%.2f) uv[%.3f,%.3f "
                 "%.3f,%.3f %.3f,%.3f %.3f,%.3f]\n", g.frameDraws,
                 nv, minx, miny, maxx, maxy, (unsigned long)d.rtBase,
                 (unsigned long)d.texBase,
                 (d.texBase && d.texBase != d.rtBase && g_rts.count(d.texBase)) ? 1 : 0,
                 dst[2], dst[3], dst[4],
                 dst[6], dst[7], dst[14], dst[15], dst[22], dst[23], dst[30], dst[31]);
    if (d.texBase) std::fprintf(stderr, "[gpuvk]      texW=%u texH=%u\n", d.texW, d.texH);
  }

  // Experiment: skip late fullscreen-covering draws (the post/composite passes
  // that, without multi-RT, cover the scene). Reveals the sprite scene.
  static const bool noComposite = std::getenv("DELTA_GPU_NOCOMPOSITE") != nullptr;
  if (noComposite && g.frameDraws >= 4) {
    const float *m = d.mvp;
    float nx0 = 1e9f, ny0 = 1e9f, nx1 = -1e9f, ny1 = -1e9f;
    for (uint32_t v = 0; v < nv; v++) {
      float x = dst[v * 8], y = dst[v * 8 + 1];
      float cw = m[3] * x + m[7] * y + m[15];
      if (cw == 0) cw = 1;
      float ndx = (m[0] * x + m[4] * y + m[12]) / cw;
      float ndy = (m[1] * x + m[5] * y + m[13]) / cw;
      nx0 = ndx < nx0 ? ndx : nx0; nx1 = ndx > nx1 ? ndx : nx1;
      ny0 = ndy < ny0 ? ndy : ny0; ny1 = ndy > ny1 ? ndy : ny1;
    }
    if ((nx1 - nx0) >= 1.8f && (ny1 - ny0) >= 1.8f) {  // covers ~all of NDC
      g.frameDraws++;
      return;
    }
  }

  // Is this a render-to-texture sample (the draw samples another render target)?
  bool rtAsTex = d.texBase && d.texBase != d.rtBase && g_rts.count(d.texBase);
  // A genuine fullscreen composite samples a near-fullscreen source RT (the scene
  // buffer copied to the scanout, or a post pass). Those use screen-space UVs and
  // must land opaquely. Sprite/effect draws that merely sample a smaller RT keep
  // their real vertex UVs and blend (clipUV off), so dim overlays/glows tint
  // instead of being forced opaque-white over the scene.
  bool fsComposite = rtAsTex && g_rts[d.texBase].w >= 1280;

  // Targeted draw trace (DELTA_GPU_DTRACE): dump every draw on a couple of frames
  // so the scanout composite can be inspected (which RT it samples, geometry).
  static const bool dtrace = std::getenv("DELTA_GPU_DTRACE") != nullptr;
  static const int dtframe = [] {
    const char *e = std::getenv("DELTA_GPU_DTFRAME");
    return e ? std::atoi(e) : 800;
  }();
  if (dtrace && (g.frameNum == dtframe || g.frameNum == dtframe + 1)) {
    float minx=1e9f,miny=1e9f,maxx=-1e9f,maxy=-1e9f;
    for (uint32_t v=0; v<nv; v++){float x=dst[v*8],y=dst[v*8+1];
      minx=x<minx?x:minx;maxx=x>maxx?x:maxx;miny=y<miny?y:miny;maxy=y>maxy?y:maxy;}
    std::fprintf(stderr,"[dt] f%d d%u rt=%#lx %ux%u tex=%#lx %ux%u rtAsTex=%d fsC=%d nv=%u "
                 "pos[%.0f,%.0f..%.0f,%.0f] uv0=%.3f,%.3f col=%.2f,%.2f,%.2f\n",
                 g.frameNum,g.frameDraws,(unsigned long)d.rtBase,d.rtW,d.rtH,
                 (unsigned long)d.texBase,d.texW,d.texH,rtAsTex?1:0,fsComposite?1:0,nv,
                 minx,miny,maxx,maxy,dst[6],dst[7],dst[2],dst[3],dst[4]);
  }
  // Upload guest texture (independent of the render region) if not RT-as-texture.
  VkDescriptorSet texSet = VK_NULL_HANDLE;
  if (d.texBase && g.texPipeline && !rtAsTex)
    texSet = getTexture(d.texBase, d.texW, d.texH);

  // Switch render target if this draw targets a different RT than the open region.
  if (g.curRt != d.rtBase) {
    endRegion();
    if (rtAsTex) {  // make the sampled RT shader-readable before we render
      auto &src = g_rts[d.texBase];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        imageBarrier(g.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    RTarget *rt = getRT(d.rtBase, d.rtW, d.rtH);
    if (!rt) { g.frameDraws++; return; }
    beginRegion(d.rtBase, *rt);
  }
  if (rtAsTex && g_rts[d.texBase].layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    texSet = g_rts[d.texBase].set;

  VkDeviceSize off = g.vbOffset;
  if (texSet) {
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.texPipeline);
    // Push mat4 MVP + a clipUV flag. Render-to-texture composites (sampling a
    // scene RT) are fullscreen blits: the shader derives uv from clip position
    // rather than the per-vertex attribute (whose format/offset varies per draw).
    float pc[17];
    std::memcpy(pc, d.mvp, 64);
    reinterpret_cast<uint32_t *>(pc)[16] = fsComposite ? 1u : 0u;
    vkCmdPushConstants(g.cmd, g.texLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 68, pc);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.texLayout, 0,
                            1, &texSet, 0, nullptr);
  } else {
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.pipeline);
    vkCmdPushConstants(g.cmd, g.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, d.mvp);
  }
  vkCmdBindVertexBuffers(g.cmd, 0, 1, &g.vb, &off);
  vkCmdDraw(g.cmd, nv, 1, 0, 0);
  g.vbOffset += need;
  g.frameDraws++;
  if (g.curRt) {
    auto &rt = g_rts[g.curRt];
    if (++rt.draws > g.busiestRtDraws) { g.busiestRtDraws = rt.draws; g.busiestRt = g.curRt; }
  }
}

void endFrame(uint64_t scanoutBase) {
  if (!g.ready || !g.recording) return;
  g.recording = false;
  endRegion();  // close any open region

  // Present the scanout RT (the flip buffer); fall back to the last RT rendered.
  uint64_t presentBase = g_rts.count(scanoutBase) ? scanoutBase : g.lastRt;
  // Debug: present the busiest RT (the scene) instead of the composited scanout.
  static const bool presentScene = std::getenv("DELTA_GPU_PRESENT_SCENE") != nullptr;
  if (presentScene && g.busiestRt) presentBase = g.busiestRt;
  auto it = g_rts.find(presentBase);
  if (it == g_rts.end()) {  // nothing rendered this frame
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g.cmd;
    vkResetFences(g.device, 1, &g.fence);
    vkQueueSubmit(g.queue, 1, &si, g.fence);
    vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    return;
  }
  RTarget &rt = it->second;
  ensureReadback(rt.w, rt.h);
  imageBarrier(g.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkBufferImageCopy copy{};
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {rt.w, rt.h, 1};
  vkCmdCopyImageToBuffer(g.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         g.readback, 1, &copy);
  vkEndCommandBuffer(g.cmd);

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g.cmd;
  vkResetFences(g.device, 1, &g.fence);
  vkQueueSubmit(g.queue, 1, &si, g.fence);
  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);

  // Orientation correction. DELTA_GPU_FLIP selects the readback transform so the
  // correct one can be determined empirically: 0=none 1=Y(rows) 2=X(cols)
  // 3=XY(180, default). The guest scanout's handedness/origin mismatch is uniform
  // across every draw, so one pass on the final scanout fixes it with no
  // render-to-texture sampling interactions.
  static const int flipMode = [] {
    const char *e = std::getenv("DELTA_GPU_FLIP");
    return e ? std::atoi(e) : 1;
  }();
  static std::vector<uint8_t> flipped;
  flipped.resize(static_cast<size_t>(rt.w) * rt.h * 4);
  const auto *rb = static_cast<const uint8_t *>(g.readbackMap);
  for (uint32_t y = 0; y < rt.h; y++) {
    uint32_t sy = (flipMode & 1) ? (rt.h - 1 - y) : y;
    const uint32_t *srow = reinterpret_cast<const uint32_t *>(rb + (size_t)sy * rt.w * 4);
    uint32_t *drow = reinterpret_cast<uint32_t *>(flipped.data() + (size_t)y * rt.w * 4);
    if (flipMode & 2)
      for (uint32_t x = 0; x < rt.w; x++) drow[x] = srow[rt.w - 1 - x];
    else
      std::memcpy(drow, srow, (size_t)rt.w * 4);
  }
  const uint8_t *pixels = flipped.data();
  if (g_dump && g.frameNum >= 1000 && g.frameNum % 2000 == 0 && g.frameDraws > 0)
    dumpPpm(pixels, rt.w, rt.h);
  // Rolling latest-frame capture (uncapped) so late transitions (menu/gameplay)
  // can be inspected from a long headless run without knowing the frame number.
  if (g_dump && g.frameNum % 300 == 0 && g.frameDraws > 0) {
    char latest[256];
    std::snprintf(latest, sizeof(latest), "%s/gpu_latest.ppm", dumpDir());
    writePpm(latest, pixels, rt.w, rt.h);
  }
  if (g_dump && g.frameNum % 200 == 0) {
    std::fprintf(stderr, "[gpuvk] frame %d draws=%u rt=%#lx %ux%u  scanout=%#lx\n",
                 g.frameNum, g.frameDraws, (unsigned long)presentBase, rt.w, rt.h,
                 (unsigned long)scanoutBase);
    for (auto &kv : g_rts)
      if (kv.second.usedThisFrame)
        std::fprintf(stderr, "[gpuvk]    RT %#lx %ux%u draws=%u%s\n",
                     (unsigned long)kv.first, kv.second.w, kv.second.h,
                     kv.second.draws,
                     kv.first == scanoutBase ? " <-SCANOUT" : "");
  }
  // Present the rendered scanout into the window the VideoOut HLE opened. When
  // there is no display (headless) the window was never created, so we skip
  // present and rely on the readback/PPM path. DELTA_GPU_NOPRESENT forces that
  // headless path even on a display.
  static const bool noPresent = std::getenv("DELTA_GPU_NOPRESENT") != nullptr;
  // Bring the window up on the first presentable frame: the videoout HLE only
  // creates it from its own scanout-present path, which the GPU (Gnm) title
  // never takes, so the renderer owns window creation here. ensure() is
  // idempotent and runs on this (the presenting) thread.
  if (!noPresent && gfx::ensure("prosperity", rt.w, rt.h) && gfx::pumpEvents())
    gfx::present(pixels, rt.w, rt.h, rt.w * 4, gfx::PixelFormat::bgra8);
}

}  // namespace gpu::vk
