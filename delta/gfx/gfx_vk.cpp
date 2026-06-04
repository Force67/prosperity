/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// SDL3 window with a Vulkan swapchain. A CPU framebuffer is copied into a
// host-visible staging buffer, then into a device-local image, then blitted
// (scaling) into the acquired swapchain image and presented. The buffer-to-image
// copy plus image-to-image blit avoids host-writes-to-image layout constraints
// and lets the window be any size relative to the framebuffer.

// SDL3 is not available on Android; that build uses the headless gfx stub
// (gfx_headless.cpp) and the GPU renderer dumps frames instead of presenting.
#ifndef __ANDROID__

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "gfx.h"
#include "overlay.h"

namespace gfx {
namespace {

#define VK_CHECK(expr)                                                         \
  do {                                                                         \
    VkResult _r = (expr);                                                      \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gfx] %s failed: VkResult=%d\n", #expr, _r);       \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct State {
  SDL_Window *window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
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

  // Per-framebuffer-size upload resources (staging buffer + device frame image).
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

  // Choose a format (prefer BGRA8 unorm).
  uint32_t nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(nfmt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, fmts.data());
  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto &f : fmts)
    if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      chosen = f;
  g.swapFormat = chosen.format;

  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g.window, &w, &h);
    ext.width = (uint32_t)w;
    ext.height = (uint32_t)h;
  }
  if (ext.width == 0 || ext.height == 0)
    return false;  // minimised; try again later
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
  sc.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // always supported
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

}  // namespace

bool init(const char *title, uint32_t width, uint32_t height) {
  if (available())
    return true;  // already up; init is idempotent
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::fprintf(stderr, "[gfx] SDL_Init failed: %s\n", SDL_GetError());
    return false;
  }
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    std::fprintf(stderr, "[gfx] SDL_Vulkan_LoadLibrary failed: %s\n",
                 SDL_GetError());
    return false;
  }
  g.window = SDL_CreateWindow(title, (int)width, (int)height,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
  if (!g.window) {
    std::fprintf(stderr, "[gfx] SDL_CreateWindow failed: %s\n", SDL_GetError());
    return false;
  }

  // Instance: SDL-required extensions + optional validation.
  uint32_t nExt = 0;
  const char *const *sdlExt = SDL_Vulkan_GetInstanceExtensions(&nExt);
  std::vector<const char *> exts(sdlExt, sdlExt + nExt);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = title;
  app.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char *> layers;
  if (SDL_getenv("DELTA_VK_VALIDATE")) {
    uint32_t nl = 0;
    vkEnumerateInstanceLayerProperties(&nl, nullptr);
    std::vector<VkLayerProperties> lp(nl);
    vkEnumerateInstanceLayerProperties(&nl, lp.data());
    for (auto &l : lp)
      if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = (uint32_t)exts.size();
  ici.ppEnabledExtensionNames = exts.data();
  ici.enabledLayerCount = (uint32_t)layers.size();
  ici.ppEnabledLayerNames = layers.data();
  VK_CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

  if (!SDL_Vulkan_CreateSurface(g.window, g.instance, nullptr, &g.surface)) {
    std::fprintf(stderr, "[gfx] SDL_Vulkan_CreateSurface failed: %s\n",
                 SDL_GetError());
    return false;
  }

  // Physical device + a queue family that does graphics AND present.
  uint32_t nphys = 0;
  vkEnumeratePhysicalDevices(g.instance, &nphys, nullptr);
  if (!nphys) {
    std::fprintf(stderr, "[gfx] no Vulkan physical devices\n");
    return false;
  }
  std::vector<VkPhysicalDevice> phs(nphys);
  vkEnumeratePhysicalDevices(g.instance, &nphys, phs.data());
  // Prefer a real GPU over the llvmpipe software rasteriser (type CPU) among the
  // devices that can both render and present; discrete > integrated > virtual >
  // CPU. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char *want = SDL_getenv("DELTA_VK_GPU");
  bool found = false;
  int best = -1;
  for (auto pd : phs) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    uint32_t fam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g.surface, &present);
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) { fam = i; break; }
    }
    if (fam == UINT32_MAX)
      continue;  // can't both render and present
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(pd, &pp);
    int score;
    switch (pp.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 4; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 3; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 2; break;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 0; break;  // llvmpipe
      default:                                     score = 1; break;
    }
    if (want && std::strstr(pp.deviceName, want)) score = 100;
    if (score > best) { best = score; g.phys = pd; g.queueFamily = fam; found = true; }
  }
  if (!found) {
    std::fprintf(stderr, "[gfx] no graphics+present queue\n");
    return false;
  }
  {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(g.phys, &pp);
    std::printf("[gfx] device: %s\n", pp.deviceName);
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

  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VK_CHECK(vkCreateSemaphore(g.device, &si, nullptr, &g.acquireSem));
  VK_CHECK(vkCreateSemaphore(g.device, &si, nullptr, &g.renderSem));
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VK_CHECK(vkCreateFence(g.device, &fi, nullptr, &g.fence));

  if (!createSwapchain())
    return false;
  std::printf("[gfx] swapchain %ux%u, %u images\n", g.swapExtent.width,
              g.swapExtent.height, (uint32_t)g.swapImages.size());
  return true;
}

void present(const void *pixels, uint32_t w, uint32_t h, uint32_t srcPitch,
             PixelFormat fmt) {
  if (!g.device || !pixels || !w || !h)
    return;
  if (srcPitch == 0)
    srcPitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::bgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                               : VK_FORMAT_R8G8B8A8_UNORM;
  if (!ensureFrameResources(w, h, vkfmt))
    return;

  // Upload rows into the staging buffer (tightly packed w*4).
  auto *dst = static_cast<uint8_t *>(g.stagingMap);
  auto *src = static_cast<const uint8_t *>(pixels);
  for (uint32_t y = 0; y < h; y++)
    std::memcpy(dst + (size_t)y * w * 4, src + (size_t)y * srcPitch, w * 4);

  // Composite the keyboard->DualSense legend over the frame (toggle with F1).
  overlayDraw(dst, w, h, fmt == PixelFormat::bgra8);

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

  // staging buffer -> frame image (TRANSFER_DST)
  imageBarrier(g.cmd, g.frameImg, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(g.cmd, g.staging, g.frameImg,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

  // frame image -> TRANSFER_SRC ; swapchain image -> TRANSFER_DST
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

  // swapchain image -> PRESENT
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

bool available() { return g.window != nullptr && g.swapchain != VK_NULL_HANDLE; }

// Idempotent bring-up: create the window/swapchain on the first call, then just
// report availability. Safe to call every frame from the presenting thread;
// after a failed attempt it stops retrying so a no-display run doesn't spam.
bool ensure(const char *title, uint32_t width, uint32_t height) {
  if (available())
    return true;
  static bool failed = false;
  if (failed)
    return false;
  if (!init(title, width, height)) {
    failed = true;
    return false;
  }
  return true;
}

bool pumpEvents() {
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_EVENT_QUIT)
      return false;
    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
        e.type == SDL_EVENT_WINDOW_RESIZED)
      g.needRecreate = true;
    if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
        e.key.scancode == SDL_SCANCODE_F1)
      overlayToggle();
  }
  return true;
}

// Keyboard->DS4 adapter, laid out for two-handed keyboard play: the left hand
// moves (WASD) and works the action keys, the right hand aims (arrow keys).
// Both hands reach a shoulder pair via the Shift keys. Keep this in sync with
// the on-screen legend (overlay.cpp).
bool pollKeyboardPad(PadKeys &out) {
  if (!g.window)
    return false;
  const bool *k = SDL_GetKeyboardState(nullptr);
  if (!k)
    return false;
  auto down = [&](SDL_Scancode s) { return k[s]; };

  // Movement on the left stick (and the d-pad, for menus).
  out.left = down(SDL_SCANCODE_A);
  out.right = down(SDL_SCANCODE_D);
  out.up = down(SDL_SCANCODE_W);
  out.down = down(SDL_SCANCODE_S);
  out.lx = out.left ? 0 : (out.right ? 255 : 128);
  out.ly = out.up ? 0 : (out.down ? 255 : 128);
  // Aim / shoot on the right stick (arrow keys).
  out.rx = down(SDL_SCANCODE_LEFT) ? 0 : (down(SDL_SCANCODE_RIGHT) ? 255 : 128);
  out.ry = down(SDL_SCANCODE_UP) ? 0 : (down(SDL_SCANCODE_DOWN) ? 255 : 128);

  out.cross = down(SDL_SCANCODE_SPACE);                                  // confirm / accept
  out.circle = down(SDL_SCANCODE_ESCAPE) || down(SDL_SCANCODE_BACKSPACE);// cancel / back
  out.square = down(SDL_SCANCODE_F);                                     // use card / pill
  out.triangle = down(SDL_SCANCODE_R);                                   // pick up / swap
  out.l1 = down(SDL_SCANCODE_Q);
  out.r1 = down(SDL_SCANCODE_E);
  out.l2 = down(SDL_SCANCODE_LSHIFT);
  out.r2 = down(SDL_SCANCODE_RSHIFT);
  out.options = down(SDL_SCANCODE_RETURN) || down(SDL_SCANCODE_P);       // start / pause
  out.touchpad = down(SDL_SCANCODE_TAB);                                 // map / select
  return true;
}

void shutdown() {
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
  if (g.window) SDL_DestroyWindow(g.window);
  SDL_Quit();
  g = State{};
}

}  // namespace gfx

#endif  // !__ANDROID__
