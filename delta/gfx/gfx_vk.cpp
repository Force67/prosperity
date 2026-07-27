/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// SDL3 window with a Vulkan swapchain. A CPU framebuffer is copied into a
// host-visible staging buffer, then into a device-local image, then blitted
// (scaling) into the acquired swapchain image and presented. The
// buffer-to-image copy plus image-to-image blit avoids host-writes-to-image
// layout constraints and lets the window be any size relative to the
// framebuffer.

// SDL3 is not available on Android; that build uses the headless gfx stub
// (gfx_headless.cpp) and the GPU renderer dumps frames instead of presenting.
#ifndef __ANDROID__

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#if defined(__linux__)
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "prosperity_logo.h"
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "gfx.h"
#include <string>

#include "overlay.h"
#include "overlay_vk.h"

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

// Window title set by the boot path (title id + platform). The renderer and the
// videoout HLE race to bring the window up and each passes its own generic
// title, so whoever wins uses this instead when it is set.
std::string g_title;
std::vector<uint8_t> g_iconPng;

constexpr uint32_t kFrameSlotCount = 2;

struct FrameSlot {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkSemaphore acquireSem = VK_NULL_HANDLE;
  VkFence acquireFence = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  void *stagingMap = nullptr;
  VkImage frameImg = VK_NULL_HANDLE;
  VkDeviceMemory frameMem = VK_NULL_HANDLE;
};

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
  std::array<FrameSlot, kFrameSlotCount> slots;
  std::vector<VkSemaphore> renderSems;
  // Semaphores of a replaced swapchain. vkDeviceWaitIdle does not cover the
  // presentation engine's pending semaphore waits (that needs
  // VK_EXT_swapchain_maintenance1), so a retired swapchain's semaphores rest
  // here for one whole swapchain generation before being destroyed.
  std::vector<VkSemaphore> retiredRenderSems;
  uint32_t nextSlot = 0;

  // Framebuffer dimensions shared by the per-slot upload resources.
  uint32_t fbW = 0, fbH = 0;
  VkFormat fbFormat = VK_FORMAT_R8G8B8A8_UNORM;

  SDL_Gamepad *gamepad =
      nullptr; // first connected controller (for input + rumble)

  bool needRecreate = false;
  bool hasMemBudget = false;
};

State g;
std::atomic_bool g_canPresent{true};
constexpr uint64_t kPresentWaitSliceNs = 50'000'000;
constexpr size_t kMaxIconSize = 16u << 20;
constexpr int kMaxIconDimension = 4096;

#if defined(__linux__)
void drawBadge(uint8_t *pixels, int width, int height, const uint8_t *logo,
               int logoWidth, int logoHeight) {
  const int size = std::max(1, std::min(width, height) * 3 / 8);
  const int left = width - size;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const int px = left + x;
      uint8_t *rgba = pixels + (static_cast<size_t>(y) * width + px) * 4;
      const uint8_t *badge =
          logo + (static_cast<size_t>(y * logoHeight / size) * logoWidth +
                  x * logoWidth / size) *
                     4;
      const uint32_t alpha = badge[3];
      const uint32_t dstAlpha = rgba[3];
      const uint32_t outAlpha = alpha * 255 + dstAlpha * (255 - alpha);
      if (outAlpha) {
        for (int channel = 0; channel < 3; ++channel)
          rgba[channel] = static_cast<uint8_t>(
              (badge[channel] * alpha * 255 +
               rgba[channel] * dstAlpha * (255 - alpha) + outAlpha / 2) /
              outAlpha);
      }
      rgba[3] = static_cast<uint8_t>((outAlpha + 127) / 255);
    }
  }
}

void applyWindowIcon() {
  if (!g.window || g_iconPng.empty() || g_iconPng.size() > kMaxIconSize)
    return;
  int width = 0;
  int height = 0;
  int channels = 0;
  if (!stbi_info_from_memory(g_iconPng.data(),
                             static_cast<int>(g_iconPng.size()), &width,
                             &height, &channels) ||
      width > kMaxIconDimension || height > kMaxIconDimension)
    return;
  stbi_uc *pixels = stbi_load_from_memory(
      g_iconPng.data(), static_cast<int>(g_iconPng.size()), &width, &height,
      &channels, STBI_rgb_alpha);
  if (!pixels) {
    stbi_image_free(pixels);
    return;
  }
  int logoWidth = 0;
  int logoHeight = 0;
  stbi_uc *logo =
      stbi_load_from_memory(kProsperityLogoPng, sizeof(kProsperityLogoPng),
                            &logoWidth, &logoHeight, nullptr, STBI_rgb_alpha);
  if (!logo) {
    stbi_image_free(pixels);
    return;
  }
  drawBadge(pixels, width, height, logo, logoWidth, logoHeight);
  SDL_Surface *surface = SDL_CreateSurfaceFrom(
      width, height, SDL_PIXELFORMAT_RGBA32, pixels, width * STBI_rgb_alpha);
  if (surface) {
    SDL_SetWindowIcon(g.window, surface);
    SDL_DestroySurface(surface);
  }
  stbi_image_free(logo);
  stbi_image_free(pixels);
}
#endif

void stopPresenting(const char *operation, VkResult result) {
  std::fprintf(stderr, "[gfx] %s failed: VkResult=%d\n", operation, result);
  g_canPresent.store(false, std::memory_order_release);
}

bool waitForPresentFence(VkFence fence, const char *operation) {
  while (g_canPresent.load(std::memory_order_acquire)) {
    const VkResult result =
        vkWaitForFences(g.device, 1, &fence, VK_TRUE, kPresentWaitSliceNs);
    if (result == VK_SUCCESS)
      return true;
    if (result != VK_TIMEOUT) {
      stopPresenting(operation, result);
      return false;
    }
  }
  return false;
}

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

void destroyRenderSemaphores() {
  for (VkSemaphore sem : g.retiredRenderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.retiredRenderSems.clear();
  for (VkSemaphore sem : g.renderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.renderSems.clear();
}

// Park the current semaphores instead of destroying them: the presentation
// engine may still wait on one after vkDeviceWaitIdle returns. Whatever was
// parked by the previous recreation is destroyed now -- by then a full
// swapchain generation (plus another idle) has passed.
void retireRenderSemaphores() {
  for (VkSemaphore sem : g.retiredRenderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.retiredRenderSems = std::move(g.renderSems);
  g.renderSems.clear();
}

bool createRenderSemaphores(uint32_t count,
                            std::vector<VkSemaphore> &semaphores) {
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphores.resize(count);
  for (VkSemaphore &sem : semaphores) {
    if (vkCreateSemaphore(g.device, &si, nullptr, &sem) != VK_SUCCESS) {
      for (VkSemaphore created : semaphores) {
        if (created)
          vkDestroySemaphore(g.device, created, nullptr);
      }
      semaphores.clear();
      return false;
    }
  }
  return true;
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
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g.window, &w, &h);
    ext.width = (uint32_t)w;
    ext.height = (uint32_t)h;
  }
  if (ext.width == 0 || ext.height == 0)
    return false; // minimised; try again later

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
  // Present mode: FIFO (vsync) is always supported but double-buffer-misses to
  // half the refresh when a frame lands late, and adds latency. Prefer MAILBOX
  // (triple- buffer: latest frame wins, no tearing, no half-rate drop).
  // DELTA_GPU_VSYNC=0 forces IMMEDIATE (uncapped, for benchmarking); =1 forces
  // FIFO.
  {
    uint32_t npm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.phys, g.surface, &npm, nullptr);
    std::vector<VkPresentModeKHR> pms(npm);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g.phys, g.surface, &npm,
                                              pms.data());
    auto has = [&](VkPresentModeKHR m) {
      for (auto p : pms)
        if (p == m)
          return true;
      return false;
    };
    const char *vs = std::getenv("DELTA_GPU_VSYNC");
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (vs && vs[0] == '0' && has(VK_PRESENT_MODE_IMMEDIATE_KHR))
      mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if (!(vs && vs[0] == '1') && has(VK_PRESENT_MODE_MAILBOX_KHR))
      mode = VK_PRESENT_MODE_MAILBOX_KHR;
    sc.presentMode = mode;
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR && imgCount < 3) {
      imgCount = 3; // mailbox wants >=3 images to actually triple-buffer
      if (caps.maxImageCount && imgCount > caps.maxImageCount)
        imgCount = caps.maxImageCount;
      sc.minImageCount = imgCount;
    }
  }
  sc.clipped = VK_TRUE;
  const VkSwapchainKHR oldSwapchain = g.swapchain;
  sc.oldSwapchain = oldSwapchain;

  // Swapchain replacement is exceptional. Idle once here so every old image,
  // semaphore, and overlay attachment can be torn down together.
  if (g.swapchain)
    vkDeviceWaitIdle(g.device);

  auto discardRetiredSwapchain = [&] {
    if (!oldSwapchain)
      return;
    overlayVkSetSwapchain({}, {}, chosen.format);
    retireRenderSemaphores();
    vkDestroySwapchainKHR(g.device, oldSwapchain, nullptr);
    g.swapchain = VK_NULL_HANDLE;
    g.swapImages.clear();
  };

  VkSwapchainKHR newSwap = VK_NULL_HANDLE;
  const VkResult createResult =
      vkCreateSwapchainKHR(g.device, &sc, nullptr, &newSwap);
  if (createResult != VK_SUCCESS) {
    discardRetiredSwapchain();
    std::fprintf(stderr, "[gfx] vkCreateSwapchainKHR failed: VkResult=%d\n",
                 createResult);
    return false;
  }

  uint32_t n = 0;
  vkGetSwapchainImagesKHR(g.device, newSwap, &n, nullptr);
  std::vector<VkImage> newImages(n);
  vkGetSwapchainImagesKHR(g.device, newSwap, &n, newImages.data());
  std::vector<VkSemaphore> newRenderSems;
  if (!createRenderSemaphores(n, newRenderSems)) {
    discardRetiredSwapchain();
    vkDestroySwapchainKHR(g.device, newSwap, nullptr);
    return false;
  }

  // This destroys framebuffers and views for the old images before their
  // swapchain is destroyed, then creates attachments for the replacement.
  overlayVkSetSwapchain(newImages, ext, chosen.format); // no-op pre-init
  retireRenderSemaphores();
  if (g.swapchain)
    vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);

  g.swapchain = newSwap;
  g.swapFormat = chosen.format;
  g.swapExtent = ext;
  g.swapImages.swap(newImages);
  g.renderSems.swap(newRenderSems);
  g.needRecreate = false;
  return true;
}

void destroyFrameResources(FrameSlot &slot) {
  if (slot.stagingMap) {
    vkUnmapMemory(g.device, slot.stagingMem);
    slot.stagingMap = nullptr;
  }
  if (slot.staging)
    vkDestroyBuffer(g.device, slot.staging, nullptr);
  if (slot.stagingMem)
    vkFreeMemory(g.device, slot.stagingMem, nullptr);
  if (slot.frameImg)
    vkDestroyImage(g.device, slot.frameImg, nullptr);
  if (slot.frameMem)
    vkFreeMemory(g.device, slot.frameMem, nullptr);
  slot.staging = VK_NULL_HANDLE;
  slot.stagingMem = VK_NULL_HANDLE;
  slot.frameImg = VK_NULL_HANDLE;
  slot.frameMem = VK_NULL_HANDLE;
}

bool createFrameResources(FrameSlot &slot, uint32_t w, uint32_t h,
                          VkFormat fmt) {
  VkDeviceSize size = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(g.device, &bi, nullptr, &slot.staging));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g.device, slot.staging, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ba, nullptr, &slot.stagingMem));
  VK_CHECK(vkBindBufferMemory(g.device, slot.staging, slot.stagingMem, 0));
  VK_CHECK(
      vkMapMemory(g.device, slot.stagingMem, 0, size, 0, &slot.stagingMap));

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
  VK_CHECK(vkCreateImage(g.device, &ii, nullptr, &slot.frameImg));
  VkMemoryRequirements ir;
  vkGetImageMemoryRequirements(g.device, slot.frameImg, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex =
      findMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ia, nullptr, &slot.frameMem));
  VK_CHECK(vkBindImageMemory(g.device, slot.frameImg, slot.frameMem, 0));
  return true;
}

bool ensureFrameResources(uint32_t w, uint32_t h, VkFormat fmt) {
  bool ready = true;
  for (const FrameSlot &slot : g.slots)
    ready &= slot.staging != VK_NULL_HANDLE && slot.frameImg != VK_NULL_HANDLE;
  if (g.fbW == w && g.fbH == h && g.fbFormat == fmt && ready)
    return true;
  vkDeviceWaitIdle(g.device);
  for (FrameSlot &slot : g.slots)
    destroyFrameResources(slot);
  g.fbW = w;
  g.fbH = h;
  g.fbFormat = fmt;
  for (FrameSlot &slot : g.slots)
    if (!createFrameResources(slot, w, h, fmt))
      return false;
  return true;
}

} // namespace

bool init(const char *title, uint32_t width, uint32_t height) {
  if (available())
    return true; // already up; init is idempotent
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    std::fprintf(stderr, "[gfx] SDL_Init failed: %s\n", SDL_GetError());
    return false;
  }
  // Open the first connected controller (if any) for input + rumble. Hotplug is
  // handled in pumpEvents(); keyboard play works regardless.
  {
    int n = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    if (ids) {
      if (n > 0)
        g.gamepad = SDL_OpenGamepad(ids[0]);
      SDL_free(ids);
    }
  }
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    std::fprintf(stderr, "[gfx] SDL_Vulkan_LoadLibrary failed: %s\n",
                 SDL_GetError());
    return false;
  }
  g.window =
      SDL_CreateWindow(g_title.empty() ? title : g_title.c_str(), (int)width,
                       (int)height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
  if (!g.window) {
    std::fprintf(stderr, "[gfx] SDL_CreateWindow failed: %s\n", SDL_GetError());
    return false;
  }
#if defined(__linux__)
  applyWindowIcon();
#endif

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
  // Prefer a real GPU over the llvmpipe software rasteriser (type CPU) among
  // the devices that can both render and present; discrete > integrated >
  // virtual > CPU. DELTA_VK_GPU=<name-substring> forces a specific device.
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
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
        fam = i;
        break;
      }
    }
    if (fam == UINT32_MAX)
      continue; // can't both render and present
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(pd, &pp);
    int score;
    switch (pp.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      score = 4;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      score = 3;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      score = 2;
      break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      score = 0;
      break; // llvmpipe
    default:
      score = 1;
      break;
    }
    if (want && std::strstr(pp.deviceName, want))
      score = 100;
    if (score > best) {
      best = score;
      g.phys = pd;
      g.queueFamily = fam;
      found = true;
    }
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
  std::vector<const char *> devExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  { // VK_EXT_memory_budget (optional): powers the overlay VRAM gauge.
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ext(ne);
    vkEnumerateDeviceExtensionProperties(g.phys, nullptr, &ne, ext.data());
    for (auto &e : ext)
      if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        devExts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        g.hasMemBudget = true;
      }
  }
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (uint32_t)devExts.size();
  dci.ppEnabledExtensionNames = devExts.data();
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
  cbi.commandBufferCount = kFrameSlotCount;
  std::array<VkCommandBuffer, kFrameSlotCount> commands;
  VK_CHECK(vkAllocateCommandBuffers(g.device, &cbi, commands.data()));

  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFenceCreateInfo acquireFi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (uint32_t i = 0; i < kFrameSlotCount; i++) {
    g.slots[i].cmd = commands[i];
    VK_CHECK(vkCreateSemaphore(g.device, &si, nullptr, &g.slots[i].acquireSem));
    VK_CHECK(
        vkCreateFence(g.device, &acquireFi, nullptr, &g.slots[i].acquireFence));
    VK_CHECK(vkCreateFence(g.device, &fi, nullptr, &g.slots[i].fence));
  }

  if (!createSwapchain())
    return false;
  std::printf("[gfx] swapchain %ux%u, %u images\n", g.swapExtent.width,
              g.swapExtent.height, (uint32_t)g.swapImages.size());
  overlayVkInit(g.phys, g.device, g.queue, g.queueFamily, g.cmdPool,
                g.swapFormat);
  overlayVkSetSwapchain(g.swapImages, g.swapExtent, g.swapFormat);
  return true;
}

void queryVram(uint64_t &used, uint64_t &total) {
  used = total = 0;
  VkPhysicalDeviceMemoryProperties2 mp2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  if (g.hasMemBudget)
    mp2.pNext = &budget;
  vkGetPhysicalDeviceMemoryProperties2(g.phys, &mp2);
  const VkPhysicalDeviceMemoryProperties &mp = mp2.memoryProperties;
  for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
    if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += g.hasMemBudget ? budget.heapBudget[i] : mp.memoryHeaps[i].size;
      if (g.hasMemBudget)
        used += budget.heapUsage[i];
    }
}

void present(const void *pixels, uint32_t w, uint32_t h, uint32_t srcPitch,
             PixelFormat fmt) {
  if (!g_canPresent.load(std::memory_order_acquire) || !g.device || !pixels ||
      !w || !h)
    return;
  if (g.needRecreate && !createSwapchain())
    return;
  if (srcPitch == 0)
    srcPitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::bgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                               : VK_FORMAT_R8G8B8A8_UNORM;
  if (!ensureFrameResources(w, h, vkfmt))
    return;

  FrameSlot &slot = g.slots[g.nextSlot];
  // The previous submission may still be reading this slot's mapped buffer.
  // Host writes must not begin until its fence signals.
  if (!waitForPresentFence(slot.fence, "vkWaitForFences(submit)"))
    return;

  // Upload rows into the staging buffer (tightly packed w*4).
  auto *dst = static_cast<uint8_t *>(slot.stagingMap);
  auto *src = static_cast<const uint8_t *>(pixels);
  for (uint32_t y = 0; y < h; y++)
    std::memcpy(dst + (size_t)y * w * 4, src + (size_t)y * srcPitch, w * 4);

  uint32_t idx = 0;
  VkResult result = vkResetFences(g.device, 1, &slot.acquireFence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetFences(acquire)", result);
    return;
  }
  VkResult ar;
  do {
    ar = vkAcquireNextImageKHR(g.device, g.swapchain, kPresentWaitSliceNs,
                               slot.acquireSem, slot.acquireFence, &idx);
  } while (ar == VK_TIMEOUT && g_canPresent.load(std::memory_order_acquire));
  if (!g_canPresent.load(std::memory_order_acquire))
    return;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    g.needRecreate = true;
    return;
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
    stopPresenting("vkAcquireNextImageKHR", ar);
    return;
  }
  if (ar == VK_SUBOPTIMAL_KHR)
    g.needRecreate = true;
  if (!waitForPresentFence(slot.acquireFence, "vkWaitForFences(acquire)"))
    return;

  uint64_t vramUsed = 0, vramTotal = 0;
  queryVram(vramUsed, vramTotal);
  overlayBuildFrame(g.swapExtent.width, g.swapExtent.height, vramUsed,
                    vramTotal);

  result = vkResetCommandBuffer(slot.cmd, 0);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetCommandBuffer", result);
    return;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(slot.cmd, &bi);
  if (result != VK_SUCCESS) {
    stopPresenting("vkBeginCommandBuffer", result);
    return;
  }

  // staging buffer -> frame image (TRANSFER_DST)
  imageBarrier(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(slot.cmd, slot.staging, slot.frameImg,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

  // frame image -> TRANSFER_SRC ; swapchain image -> TRANSFER_DST
  imageBarrier(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  imageBarrier(slot.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[1] = {(int32_t)g.swapExtent.width,
                        (int32_t)g.swapExtent.height, 1};
  vkCmdBlitImage(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blit, VK_FILTER_LINEAR);

  // The overlay's LOAD render pass draws over the blitted frame and transitions
  // the image to PRESENT_SRC; without it, do that transition directly.
  if (!overlayVkRender(slot.cmd, idx))
    imageBarrier(
        slot.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  result = vkEndCommandBuffer(slot.cmd);
  if (result != VK_SUCCESS) {
    stopPresenting("vkEndCommandBuffer", result);
    return;
  }

  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo subi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  subi.waitSemaphoreCount = 1;
  subi.pWaitSemaphores = &slot.acquireSem;
  subi.pWaitDstStageMask = &waitStage;
  subi.commandBufferCount = 1;
  subi.pCommandBuffers = &slot.cmd;
  subi.signalSemaphoreCount = 1;
  subi.pSignalSemaphores = &g.renderSems[idx];
  result = vkResetFences(g.device, 1, &slot.fence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetFences(submit)", result);
    return;
  }
  result = vkQueueSubmit(g.queue, 1, &subi, slot.fence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkQueueSubmit", result);
    return;
  }
  g.nextSlot = (g.nextSlot + 1) % kFrameSlotCount;

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &g.renderSems[idx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &g.swapchain;
  pi.pImageIndices = &idx;
  VkResult pr = vkQueuePresentKHR(g.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
    g.needRecreate = true;
  else if (pr != VK_SUCCESS)
    stopPresenting("vkQueuePresentKHR", pr);
}

void setTitle(const char *title) {
  g_title = title ? title : "";
  if (g.window && !g_title.empty())
    SDL_SetWindowTitle(g.window, g_title.c_str());
}

void setIcon(const uint8_t *png, size_t size) {
#if defined(__linux__)
  if (size > kMaxIconSize)
    return;
  g_iconPng.assign(png, png + size);
  applyWindowIcon();
#else
  (void)png;
  (void)size;
#endif
}

bool available() {
  return g.window != nullptr && g.swapchain != VK_NULL_HANDLE;
}

bool canPresent() { return g_canPresent.load(std::memory_order_acquire); }

void requestPresentStop() {
  g_canPresent.store(false, std::memory_order_release);
}

// Idempotent bring-up: create the window/swapchain on the first call, then just
// report availability. Safe to call every frame from the presenting thread;
// after a failed attempt it stops retrying so a no-display run doesn't spam.
bool ensure(const char *title, uint32_t width, uint32_t height) {
  if (available())
    return true;
  if (!g_canPresent.load(std::memory_order_acquire))
    return false;
  if (g.device)
    return createSwapchain();
  if (!init(title, width, height)) {
    g_canPresent.store(false, std::memory_order_release);
    return false;
  }
  return true;
}

bool pumpEvents() {
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_EVENT_QUIT) {
      g_canPresent.store(false, std::memory_order_release);
      return false;
    }
    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
        e.type == SDL_EVENT_WINDOW_RESIZED)
      g.needRecreate = true;
    if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
        e.key.scancode == SDL_SCANCODE_F1)
      overlayToggle();
    if (e.type == SDL_EVENT_GAMEPAD_ADDED && !g.gamepad)
      g.gamepad = SDL_OpenGamepad(e.gdevice.which);
    if (e.type == SDL_EVENT_GAMEPAD_REMOVED && g.gamepad &&
        e.gdevice.which == SDL_GetGamepadID(g.gamepad)) {
      SDL_CloseGamepad(g.gamepad);
      g.gamepad = nullptr;
    }
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

  out.cross = down(SDL_SCANCODE_SPACE); // confirm / accept
  out.circle = down(SDL_SCANCODE_ESCAPE) ||
               down(SDL_SCANCODE_BACKSPACE); // cancel / back
  out.square = down(SDL_SCANCODE_F);         // use card / pill
  out.triangle = down(SDL_SCANCODE_R);       // pick up / swap
  out.l1 = down(SDL_SCANCODE_Q);
  out.r1 = down(SDL_SCANCODE_E);
  out.l2 = down(SDL_SCANCODE_LSHIFT);
  out.r2 = down(SDL_SCANCODE_RSHIFT);
  out.options =
      down(SDL_SCANCODE_RETURN) || down(SDL_SCANCODE_P); // start / pause
  out.touchpad = down(SDL_SCANCODE_TAB);                 // map / select

  // Overlay a real controller when one is connected (it takes precedence over
  // keyboard for any button/axis it actively asserts).
  if (g.gamepad) {
    auto b = [&](SDL_GamepadButton n) {
      return SDL_GetGamepadButton(g.gamepad, n);
    };
    out.cross |= b(SDL_GAMEPAD_BUTTON_SOUTH);
    out.circle |= b(SDL_GAMEPAD_BUTTON_EAST);
    out.square |= b(SDL_GAMEPAD_BUTTON_WEST);
    out.triangle |= b(SDL_GAMEPAD_BUTTON_NORTH);
    out.up |= b(SDL_GAMEPAD_BUTTON_DPAD_UP);
    out.down |= b(SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    out.left |= b(SDL_GAMEPAD_BUTTON_DPAD_LEFT);
    out.right |= b(SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
    out.l1 |= b(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    out.r1 |= b(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    out.options |= b(SDL_GAMEPAD_BUTTON_START);
    out.touchpad |= b(SDL_GAMEPAD_BUTTON_TOUCHPAD);
    // Sticks: map [-32768,32767] to [0,255]; only override the centred keyboard
    // value.
    auto axis = [&](SDL_GamepadAxis n) -> int {
      int v = SDL_GetGamepadAxis(g.gamepad, n);
      return (v + 32768) * 255 / 65535;
    };
    int lx = axis(SDL_GAMEPAD_AXIS_LEFTX), ly = axis(SDL_GAMEPAD_AXIS_LEFTY);
    int rx = axis(SDL_GAMEPAD_AXIS_RIGHTX), ry = axis(SDL_GAMEPAD_AXIS_RIGHTY);
    if (std::abs(lx - 128) > 12)
      out.lx = (uint8_t)lx;
    if (std::abs(ly - 128) > 12)
      out.ly = (uint8_t)ly;
    if (std::abs(rx - 128) > 12)
      out.rx = (uint8_t)rx;
    if (std::abs(ry - 128) > 12)
      out.ry = (uint8_t)ry;
    // Triggers -> L2/R2 (and the analog buttons via the bit, see fillPadState).
    if (SDL_GetGamepadAxis(g.gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 8000)
      out.l2 = true;
    if (SDL_GetGamepadAxis(g.gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 8000)
      out.r2 = true;
  }
  return true;
}

void setRumble(uint8_t largeMotor, uint8_t smallMotor) {
  if (!g.gamepad)
    return;
  // DS4 motors are 0..255; SDL rumble is 0..65535. Large = low-freq, small =
  // high-freq. Duration 0 means "until the next call"; the game re-issues
  // continuously.
  SDL_RumbleGamepad(g.gamepad, (uint16_t)(largeMotor * 257),
                    (uint16_t)(smallMotor * 257), 0);
}

void shutdown() {
  if (g.device)
    vkDeviceWaitIdle(g.device);
  overlayVkShutdown();
  for (FrameSlot &slot : g.slots) {
    destroyFrameResources(slot);
    if (slot.fence)
      vkDestroyFence(g.device, slot.fence, nullptr);
    if (slot.acquireFence)
      vkDestroyFence(g.device, slot.acquireFence, nullptr);
    if (slot.acquireSem)
      vkDestroySemaphore(g.device, slot.acquireSem, nullptr);
  }
  destroyRenderSemaphores();
  if (g.cmdPool)
    vkDestroyCommandPool(g.device, g.cmdPool, nullptr);
  if (g.swapchain)
    vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  if (g.device)
    vkDestroyDevice(g.device, nullptr);
  if (g.surface)
    vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
  if (g.instance)
    vkDestroyInstance(g.instance, nullptr);
  if (g.window)
    SDL_DestroyWindow(g.window);
  SDL_Quit();
  g = State{};
  g_canPresent.store(true, std::memory_order_release);
}

} // namespace gfx

#endif // !__ANDROID__
