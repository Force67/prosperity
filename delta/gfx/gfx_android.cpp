/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen Vulkan present for the Android app (DELTA_ANDROID_APP). Same scheme
 * as the desktop gfx_vk.cpp (CPU framebuffer -> staging buffer -> device image
 * -> blit into the acquired swapchain image -> present), but the window is an
 * ANativeWindow handed in by the NativeActivity loop (android_main) and the
 * surface comes from VK_KHR_android_surface. All Vulkan calls run on the guest
 * renderer thread (the only caller of ensure()/present()); android_main only
 * publishes the window handle and the touch-derived pad state.
 */
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include "gfx.h"
#include "gfx_android.h"

namespace gfx {
namespace {

#define VK_CHECK(expr)                                                         \
  do {                                                                         \
    VkResult _r = (expr);                                                      \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gfx-android] %s failed: VkResult=%d\n", #expr,    \
                   _r);                                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct State {
  ANativeWindow *window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D swapExtent{};
  std::vector<VkImage> swapImages;

  VkCommandPool cmdPool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkSemaphore acquireSem = VK_NULL_HANDLE;
  VkSemaphore renderSem = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  uint32_t fbW = 0, fbH = 0;
  VkFormat fbFormat = VK_FORMAT_R8G8B8A8_UNORM;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  void *stagingMap = nullptr;
  VkImage frameImg = VK_NULL_HANDLE;
  VkDeviceMemory frameMem = VK_NULL_HANDLE;

  bool needRecreate = false;
};
State g;

// Window handle + pad state published by android_main (other thread).
std::mutex g_inMutex;
ANativeWindow *g_pendingWindow = nullptr;
bool g_windowChanged = false;
PadKeys g_pad;

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return UINT32_MAX;
}

void imageBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA,
                  VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = srcA;
  b.dstAccessMask = dstA;
  vkCmdPipelineBarrier(c, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

bool createSwapchain() {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.phys, g.surface, &caps);

  uint32_t nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(nfmt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, fmts.data());
  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto &f : fmts)
    if (f.format == VK_FORMAT_R8G8B8A8_UNORM ||
        f.format == VK_FORMAT_B8G8R8A8_UNORM)
      chosen = f;
  g.swapFormat = chosen.format;

  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    ext.width = (uint32_t)ANativeWindow_getWidth(g.window);
    ext.height = (uint32_t)ANativeWindow_getHeight(g.window);
  }
  if (ext.width == 0 || ext.height == 0)
    return false;
  g.swapExtent = ext;

  uint32_t imgCount = caps.minImageCount + 1;
  if (caps.maxImageCount && imgCount > caps.maxImageCount)
    imgCount = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sc{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sc.surface = g.surface;
  sc.minImageCount = imgCount;
  sc.imageFormat = chosen.format;
  sc.imageColorSpace = chosen.colorSpace;
  sc.imageExtent = ext;
  sc.imageArrayLayers = 1;
  sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sc.preTransform = caps.currentTransform;
  sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sc.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sc.clipped = VK_TRUE;
  sc.oldSwapchain = g.swapchain;

  VkSwapchainKHR newSwap = VK_NULL_HANDLE;
  VK_CHECK(vkCreateSwapchainKHR(g.device, &sc, nullptr, &newSwap));
  if (g.swapchain)
    vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  g.swapchain = newSwap;

  uint32_t n = 0;
  vkGetSwapchainImagesKHR(g.device, g.swapchain, &n, nullptr);
  g.swapImages.resize(n);
  vkGetSwapchainImagesKHR(g.device, g.swapchain, &n, g.swapImages.data());
  g.needRecreate = false;
  return true;
}

void destroyFrameResources() {
  if (g.stagingMap) {
    vkUnmapMemory(g.device, g.stagingMem);
    g.stagingMap = nullptr;
  }
  if (g.staging) vkDestroyBuffer(g.device, g.staging, nullptr);
  if (g.stagingMem) vkFreeMemory(g.device, g.stagingMem, nullptr);
  if (g.frameImg) vkDestroyImage(g.device, g.frameImg, nullptr);
  if (g.frameMem) vkFreeMemory(g.device, g.frameMem, nullptr);
  g.staging = VK_NULL_HANDLE;
  g.stagingMem = VK_NULL_HANDLE;
  g.frameImg = VK_NULL_HANDLE;
  g.frameMem = VK_NULL_HANDLE;
}

bool ensureFrameResources(uint32_t w, uint32_t h, VkFormat fmt) {
  if (g.fbW == w && g.fbH == h && g.fbFormat == fmt && g.staging)
    return true;
  vkDeviceWaitIdle(g.device);
  destroyFrameResources();
  g.fbW = w;
  g.fbH = h;
  g.fbFormat = fmt;

  VkDeviceSize size = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(g.device, &bi, nullptr, &g.staging));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g.device, g.staging, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ba, nullptr, &g.stagingMem));
  VK_CHECK(vkBindBufferMemory(g.device, g.staging, g.stagingMem, 0));
  VK_CHECK(vkMapMemory(g.device, g.stagingMem, 0, size, 0, &g.stagingMap));

  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = fmt;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(g.device, &ii, nullptr, &g.frameImg));
  VkMemoryRequirements ir;
  vkGetImageMemoryRequirements(g.device, g.frameImg, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex =
      findMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ia, nullptr, &g.frameMem));
  VK_CHECK(vkBindImageMemory(g.device, g.frameImg, g.frameMem, 0));
  return true;
}

// Full bring-up against the current g.window: instance, android surface, device,
// command/sync objects and the swapchain.
bool bringUp() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "prosperity";
  app.apiVersion = VK_API_VERSION_1_1;
  const char *exts[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = exts;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

  VkAndroidSurfaceCreateInfoKHR si{
      VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
  si.window = g.window;
  VK_CHECK(vkCreateAndroidSurfaceKHR(g.instance, &si, nullptr, &g.surface));

  uint32_t nphys = 0;
  vkEnumeratePhysicalDevices(g.instance, &nphys, nullptr);
  if (!nphys) {
    std::fprintf(stderr, "[gfx-android] no Vulkan physical devices\n");
    return false;
  }
  std::vector<VkPhysicalDevice> phs(nphys);
  vkEnumeratePhysicalDevices(g.instance, &nphys, phs.data());
  bool found = false;
  for (auto pd : phs) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    for (uint32_t i = 0; i < nq; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g.surface, &present);
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
        g.phys = pd;
        g.queueFamily = i;
        found = true;
        break;
      }
    }
    if (found)
      break;
  }
  if (!found) {
    std::fprintf(stderr, "[gfx-android] no graphics+present queue\n");
    return false;
  }
  {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(g.phys, &pp);
    std::printf("[gfx-android] device: %s\n", pp.deviceName);
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g.queueFamily;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  const char *devExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = devExts;
  VK_CHECK(vkCreateDevice(g.phys, &dci, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.queueFamily, 0, &g.queue);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g.queueFamily;
  VK_CHECK(vkCreateCommandPool(g.device, &pci, nullptr, &g.cmdPool));
  VkCommandBufferAllocateInfo cbi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbi.commandPool = g.cmdPool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(g.device, &cbi, &g.cmd));

  VkSemaphoreCreateInfo si2{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VK_CHECK(vkCreateSemaphore(g.device, &si2, nullptr, &g.acquireSem));
  VK_CHECK(vkCreateSemaphore(g.device, &si2, nullptr, &g.renderSem));
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VK_CHECK(vkCreateFence(g.device, &fi, nullptr, &g.fence));

  if (!createSwapchain())
    return false;
  std::printf("[gfx-android] swapchain %ux%u, %u images\n", g.swapExtent.width,
              g.swapExtent.height, (uint32_t)g.swapImages.size());
  return true;
}

// Tear everything down (window lost). The guest GPU renderer has its own Vulkan
// device, so dropping ours only stops presentation; it resumes on re-init.
void teardown() {
  if (g.device)
    vkDeviceWaitIdle(g.device);
  destroyFrameResources();
  if (g.fence) vkDestroyFence(g.device, g.fence, nullptr);
  if (g.acquireSem) vkDestroySemaphore(g.device, g.acquireSem, nullptr);
  if (g.renderSem) vkDestroySemaphore(g.device, g.renderSem, nullptr);
  if (g.cmdPool) vkDestroyCommandPool(g.device, g.cmdPool, nullptr);
  if (g.swapchain) vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  if (g.device) vkDestroyDevice(g.device, nullptr);
  if (g.surface) vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
  if (g.instance) vkDestroyInstance(g.instance, nullptr);
  ANativeWindow *keep = g.window;
  g = State{};
  g.window = keep;
}

}  // namespace

// --- public gfx API ---------------------------------------------------------

bool init(const char *, uint32_t, uint32_t) {
  return g.swapchain != VK_NULL_HANDLE || (g.window && bringUp());
}

bool available() {
  return g.window != nullptr && g.swapchain != VK_NULL_HANDLE;
}

bool ensure(const char *, uint32_t, uint32_t) {
  // Adopt any window change published by android_main (this thread owns Vulkan).
  {
    std::lock_guard<std::mutex> lk(g_inMutex);
    if (g_windowChanged) {
      g_windowChanged = false;
      if (g_pendingWindow != g.window) {
        if (g.instance)
          teardown();  // resets g, preserves g.window
        g.window = g_pendingWindow;
      }
    }
  }
  if (!g.window)
    return false;
  if (!g.instance)
    return bringUp();
  if (g.needRecreate)
    createSwapchain();
  return available();
}

void present(const void *pixels, uint32_t w, uint32_t h, uint32_t srcPitch,
             PixelFormat fmt) {
  if (!g.device || !g.swapchain || !pixels || !w || !h)
    return;
  if (srcPitch == 0)
    srcPitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::bgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                               : VK_FORMAT_R8G8B8A8_UNORM;
  if (!ensureFrameResources(w, h, vkfmt))
    return;

  auto *dst = static_cast<uint8_t *>(g.stagingMap);
  auto *src = static_cast<const uint8_t *>(pixels);
  for (uint32_t y = 0; y < h; y++)
    std::memcpy(dst + (size_t)y * w * 4, src + (size_t)y * srcPitch, w * 4);

  vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);

  if (g.needRecreate && !createSwapchain())
    return;

  uint32_t idx = 0;
  VkResult ar = vkAcquireNextImageKHR(g.device, g.swapchain, UINT64_MAX,
                                      g.acquireSem, VK_NULL_HANDLE, &idx);
  if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_SUBOPTIMAL_KHR) {
    g.needRecreate = true;
    return;
  }
  if (ar != VK_SUCCESS)
    return;

  vkResetFences(g.device, 1, &g.fence);
  vkResetCommandBuffer(g.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g.cmd, &bi);

  imageBarrier(g.cmd, g.frameImg, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(g.cmd, g.staging, g.frameImg,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

  imageBarrier(g.cmd, g.frameImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  imageBarrier(g.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[1] = {(int32_t)g.swapExtent.width,
                        (int32_t)g.swapExtent.height, 1};
  vkCmdBlitImage(g.cmd, g.frameImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blit, VK_FILTER_LINEAR);

  imageBarrier(g.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  vkEndCommandBuffer(g.cmd);

  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo subi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  subi.waitSemaphoreCount = 1;
  subi.pWaitSemaphores = &g.acquireSem;
  subi.pWaitDstStageMask = &waitStage;
  subi.commandBufferCount = 1;
  subi.pCommandBuffers = &g.cmd;
  subi.signalSemaphoreCount = 1;
  subi.pSignalSemaphores = &g.renderSem;
  vkQueueSubmit(g.queue, 1, &subi, g.fence);

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &g.renderSem;
  pi.swapchainCount = 1;
  pi.pSwapchains = &g.swapchain;
  pi.pImageIndices = &idx;
  VkResult pr = vkQueuePresentKHR(g.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
    g.needRecreate = true;
}

// Lifecycle is driven by android_main's looper; nothing to pump here.
bool pumpEvents() { return true; }

bool pollKeyboardPad(PadKeys &out) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  out = g_pad;
  return true;
}

void shutdown() { teardown(); }

void setAndroidWindow(ANativeWindow *window) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  g_pendingWindow = window;
  g_windowChanged = true;
}

void setAndroidPad(const PadKeys &keys) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  g_pad = keys;
}

}  // namespace gfx

#endif  // __ANDROID__ && DELTA_ANDROID_APP
