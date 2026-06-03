/*
 * PS4Delta : PS4 emulation and research project
 *
 * Headless Vulkan renderer. See vk_render.h. This first stage stands up the
 * device + render-target image and presents the cleared/rendered RT each frame
 * (read back to a linear buffer); the GCN-shader draw pipeline builds on top.
 */

#include "vk_render.h"

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gfx/gfx.h"

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

  // Render target.
  VkImage rt = VK_NULL_HANDLE;
  VkDeviceMemory rtMem = VK_NULL_HANDLE;
  uint32_t rtW = 0, rtH = 0;
  VkFormat rtFormat = VK_FORMAT_B8G8R8A8_UNORM;

  // Host-visible readback buffer.
  VkBuffer readback = VK_NULL_HANDLE;
  VkDeviceMemory readbackMem = VK_NULL_HANDLE;
  void *readbackMap = nullptr;
  VkDeviceSize readbackSize = 0;

  uint32_t frameDraws = 0;
  bool recording = false;
} g;

const bool g_dump = std::getenv("DELTA_GPU_DUMP") != nullptr;
int g_dumpedFrames = 0;

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
  app.apiVersion = VK_API_VERSION_1_2;
  app.pApplicationName = "prosperity-gpu";
  VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ic.pApplicationInfo = &app;
  VKOK(vkCreateInstance(&ic, nullptr, &g.instance));

  uint32_t n = 0;
  vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
  if (!n) {
    std::fprintf(stderr, "[gpuvk] no Vulkan physical device\n");
    return false;
  }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(g.instance, &n, devs.data());
  g.phys = devs[0];

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qprops(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, qprops.data());
  bool found = false;
  for (uint32_t i = 0; i < qn; i++)
    if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      g.qfam = i;
      found = true;
      break;
    }
  if (!found) {
    std::fprintf(stderr, "[gpuvk] no graphics queue family\n");
    return false;
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qc.queueFamilyIndex = g.qfam;
  qc.queueCount = 1;
  qc.pQueuePriorities = &prio;
  VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dc.queueCreateInfoCount = 1;
  dc.pQueueCreateInfos = &qc;
  VKOK(vkCreateDevice(g.phys, &dc, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.qfam, 0, &g.queue);

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

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g.phys, &props);
  std::fprintf(stderr, "[gpuvk] device: %s\n", props.deviceName);
  return true;
}

bool ensureRT(uint32_t w, uint32_t h, VkFormat fmt) {
  if (g.rt && g.rtW == w && g.rtH == h && g.rtFormat == fmt)
    return true;
  vkDeviceWaitIdle(g.device);
  if (g.rt) vkDestroyImage(g.device, g.rt, nullptr);
  if (g.rtMem) vkFreeMemory(g.device, g.rtMem, nullptr);
  if (g.readback) vkDestroyBuffer(g.device, g.readback, nullptr);
  if (g.readbackMem) vkFreeMemory(g.device, g.readbackMem, nullptr);
  g.rt = VK_NULL_HANDLE; g.readback = VK_NULL_HANDLE;
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

  g.readbackSize = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = g.readbackSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKOK(vkCreateBuffer(g.device, &bi, nullptr, &g.readback));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g.device, g.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VKOK(vkAllocateMemory(g.device, &ba, nullptr, &g.readbackMem));
  VKOK(vkBindBufferMemory(g.device, g.readback, g.readbackMem, 0));
  VKOK(vkMapMemory(g.device, g.readbackMem, 0, g.readbackSize, 0, &g.readbackMap));
  std::fprintf(stderr, "[gpuvk] render target %ux%u fmt=%d\n", w, h, (int)fmt);
  return true;
}

void dumpPpm(const uint8_t *bgra, uint32_t w, uint32_t h) {
  if (g_dumpedFrames >= 4)
    return;
  char path[128];
  std::snprintf(path, sizeof(path), "/tmp/gpu_frame_%d.ppm", g_dumpedFrames++);
  FILE *f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; i++) {
    // stored BGRA -> write RGB
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
  }
  std::fclose(f);
  std::fprintf(stderr, "[gpuvk] dumped %s\n", path);
}

}  // namespace

bool init() {
  if (g.ready)
    return true;
  if (!createDevice()) {
    std::fprintf(stderr, "[gpuvk] headless Vulkan unavailable; gpu disabled\n");
    return false;
  }
  g.ready = true;
  return true;
}

bool available() { return g.ready; }

void beginFrame(uint64_t rtAddr, uint32_t width, uint32_t height, uint32_t pitch,
                uint32_t cbInfo) {
  if (!g.ready || !width || !height)
    return;
  if (!ensureRT(width, height, g.rtFormat))
    return;
  g.frameDraws = 0;

  vkResetCommandBuffer(g.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g.cmd, &bi);

  // Clear the RT (until real draws fill it). A dim blue clear is recognizable.
  imageBarrier(g.cmd, g.rt, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT);
  VkClearColorValue cc{};
  cc.float32[0] = 0.05f; cc.float32[1] = 0.05f; cc.float32[2] = 0.12f; cc.float32[3] = 1.0f;
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(g.cmd, g.rt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1,
                       &range);
  g.recording = true;
}

void draw(const DrawInfo &d) {
  if (!g.recording)
    return;
  // TODO(gpu): recompile d.vsAddr/d.psAddr (GCN->SPIR-V), build a pipeline from
  // the register state, bind the V#/T#/S# resources from the user-data SGPRs,
  // and vkCmdDraw into the RT. For now just count.
  g.frameDraws++;
}

void endFrame() {
  if (!g.ready || !g.recording)
    return;
  g.recording = false;

  // RT -> readback buffer.
  imageBarrier(g.cmd, g.rt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_TRANSFER_READ_BIT);
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

  auto *pixels = static_cast<const uint8_t *>(g.readbackMap);
  if (g_dump)
    dumpPpm(pixels, g.rtW, g.rtH);
  // Present to the window if a display/swapchain is up.
  if (gfx::pumpEvents())
    gfx::present(pixels, g.rtW, g.rtH, g.rtW * 4, gfx::PixelFormat::bgra8);
}

}  // namespace gpu::vk
