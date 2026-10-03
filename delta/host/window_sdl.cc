/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

// SDL3 window + Vulkan swapchain. A CPU framebuffer goes to a host-visible
// staging buffer, then a device-local image, then blits (scaling) into the
// swapchain image. The copy+blit route sidesteps host-writes-to-image layout
// constraints and allows any window size relative to the framebuffer.

// SDL3 is not available on Android; those builds use window_android.cc or the
// headless stub (window_headless.cc).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"

#if defined(__linux__)
#include <stb_image.h>

#include "prosperity_logo.h"
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include "host/window.h"

#include "base/logging.h"

#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/mem_ops.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "guest/pause.h"
#include "guest/session.h"
#include "host/audio_output.h"
#include "options/options.h"
#include "ui/home_screen.h"
#include "ui/input_sdl.h"
#include "ui/mouse_look.h"
#include "ui/overlay.h"
#include "ui/overlay_vk.h"
#include "ui/pause_menu.h"

namespace {
DELTA_OPTION(bool, kVkValidate, "DELTA_VK_VALIDATE", false);
DELTA_OPTION(const char*, kVkGpu, "DELTA_VK_GPU", nullptr);
DELTA_OPTION(const char*, kVsync, "DELTA_GPU_VSYNC", nullptr);
}  // namespace

namespace host {
namespace {

#define VK_CHECK(expr)                                            \
  do {                                                            \
    VkResult _r = (expr);                                         \
    if (_r != VK_SUCCESS) {                                       \
      BASE_LOGI("gfx", "{} failed: VkResult={}", #expr, (int)_r); \
      return false;                                               \
    }                                                             \
  } while (0)

// Window title set by the boot path (title id + platform). The renderer and the
// videoout HLE race to bring the window up and each passes its own generic
// title, so whoever wins uses this instead when it is set.
base::String g_title;
base::Vector<u8> g_icon_png;

constexpr u32 kFrameSlotCount = 2;

struct FrameSlot {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkSemaphore acquire_sem = VK_NULL_HANDLE;
  VkFence acquire_fence = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  void* staging_map = nullptr;
  VkImage frame_img = VK_NULL_HANDLE;
  VkDeviceMemory frame_mem = VK_NULL_HANDLE;
};

struct State {
  SDL_Window* window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  u32 queue_family = 0;
  VkQueue queue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D swap_extent{};
  base::Vector<VkImage> swap_images;

  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  base::Array<FrameSlot, kFrameSlotCount> slots;
  base::Vector<VkSemaphore> render_sems;
  // Semaphores of a replaced swapchain. vkDeviceWaitIdle does not cover the
  // presentation engine's pending semaphore waits (that needs
  // VK_EXT_swapchain_maintenance1), so a retired swapchain's semaphores rest
  // here for one whole swapchain generation before being destroyed.
  base::Vector<VkSemaphore> retired_render_sems;
  u32 next_slot = 0;
  u32 last_frame_slot = kFrameSlotCount;

  // Framebuffer dimensions shared by the per-slot upload resources.
  u32 fb_w = 0, fb_h = 0;
  VkFormat fb_format = VK_FORMAT_R8G8B8A8_UNORM;

  SDL_Gamepad* gamepad =
      nullptr;  // first connected controller (for input + rumble)

  bool need_recreate = false;
  bool has_mem_budget = false;
};

State g_window;
base::Atomic<bool> g_can_present{true};
base::Atomic<bool> g_suppress_pad{false};
// The splash thread and the videoout HLE may both bring the window up.
base::Mutex g_init_mutex;
base::Mutex g_ui_mutex;
base::Atomic<bool> g_exit_animation{false};
base::UniquePointer<base::Thread> g_exit_thread;
thread_local bool t_exit_thread = false;

void StopExitAnimation() {
  if (!g_exit_thread)
    return;
  g_exit_animation.store(false, base::memory_order_release);
  g_exit_thread->Join();
  g_exit_thread = {};
}
// 0: no splash, 1: the splash thread owns the window, 2: asked to hand it over.
base::Atomic<int> g_splash{0};
thread_local bool t_splash_thread = false;

// The first present or event pump from anyone else takes the window over.
void StopSplash() {
  if (t_splash_thread || g_splash.load(base::memory_order_acquire) == 0)
    return;
  int running = 1;
  g_splash.compare_exchange_strong(running, 2);
  while (g_splash.load(base::memory_order_acquire) != 0)
    base::YieldCurrentThread();
}
constexpr u64 kPresentWaitSliceNs = 50'000'000;
constexpr size_t kMaxIconSize = 16u << 20;
constexpr int kMaxIconDimension = 4096;

#if defined(__linux__)
void DrawBadge(u8* pixels,
               int width,
               int height,
               const u8* logo,
               int logo_width,
               int logo_height) {
  const int size = base::Max(1, base::Min(width, height) * 3 / 4);
  const int left = width - size;
  // The logo art carries ~11% transparent margin, so a top-anchored badge reads
  // as floating below the edge. Lift it by that margin; the rows that fall off
  // the top are the empty ones.
  const int top = -(size / 8);
  for (int y = 0; y < size; ++y) {
    const int py = top + y;
    if (py < 0 || py >= height)
      continue;
    for (int x = 0; x < size; ++x) {
      const int px = left + x;
      u8* rgba = pixels + (static_cast<size_t>(py) * width + px) * 4;
      const u8* badge =
          logo + (static_cast<size_t>(y * logo_height / size) * logo_width +
                  x * logo_width / size) *
                     4;
      const u32 alpha = badge[3];
      const u32 dst_alpha = rgba[3];
      const u32 out_alpha = alpha * 255 + dst_alpha * (255 - alpha);
      if (out_alpha) {
        for (int channel = 0; channel < 3; ++channel)
          rgba[channel] = static_cast<u8>(
              (badge[channel] * alpha * 255 +
               rgba[channel] * dst_alpha * (255 - alpha) + out_alpha / 2) /
              out_alpha);
      }
      rgba[3] = static_cast<u8>((out_alpha + 127) / 255);
    }
  }
}

// Blue frame around the artwork, so the icon reads as ours at taskbar size.
void DrawBorder(u8* pixels, int width, int height) {
  constexpr u8 kFrame[4] = {0x18, 0x60, 0xCC, 0xFF};
  const int thickness = base::Max(2, base::Min(width, height) / 24);
  for (int y = 0; y < height; ++y) {
    const bool edge_row = y < thickness || y >= height - thickness;
    for (int x = 0; x < width; ++x) {
      if (!edge_row && x >= thickness && x < width - thickness)
        continue;
      std::memcpy(pixels + (static_cast<size_t>(y) * width + x) * 4, kFrame, 4);
    }
  }
}

void ApplyWindowIcon() {
  if (!g_window.window || g_icon_png.empty() ||
      g_icon_png.size() > kMaxIconSize)
    return;
  int width = 0;
  int height = 0;
  int channels = 0;
  if (!stbi_info_from_memory(g_icon_png.data(),
                             static_cast<int>(g_icon_png.size()), &width,
                             &height, &channels) ||
      width > kMaxIconDimension || height > kMaxIconDimension)
    return;
  stbi_uc* pixels = stbi_load_from_memory(
      g_icon_png.data(), static_cast<int>(g_icon_png.size()), &width, &height,
      &channels, STBI_rgb_alpha);
  if (!pixels) {
    stbi_image_free(pixels);
    return;
  }
  int logo_width = 0;
  int logo_height = 0;
  stbi_uc* logo =
      stbi_load_from_memory(kProsperityLogoPng, sizeof(kProsperityLogoPng),
                            &logo_width, &logo_height, nullptr, STBI_rgb_alpha);
  if (!logo) {
    stbi_image_free(pixels);
    return;
  }
  DrawBadge(pixels, width, height, logo, logo_width, logo_height);
  DrawBorder(pixels, width, height);
  SDL_Surface* surface = SDL_CreateSurfaceFrom(
      width, height, SDL_PIXELFORMAT_RGBA32, pixels, width * STBI_rgb_alpha);
  if (surface) {
    SDL_SetWindowIcon(g_window.window, surface);
    SDL_DestroySurface(surface);
  }
  stbi_image_free(logo);
  stbi_image_free(pixels);
}
#endif

void StopPresenting(const char* operation, VkResult result) {
  BASE_LOGI("gfx", "{} failed: VkResult={}", operation, (int)result);
  g_can_present.store(false, base::memory_order_release);
}

bool WaitForPresentFence(VkFence fence, const char* operation) {
  while (g_can_present.load(base::memory_order_acquire)) {
    const VkResult result = vkWaitForFences(g_window.device, 1, &fence, VK_TRUE,
                                            kPresentWaitSliceNs);
    if (result == VK_SUCCESS)
      return true;
    if (result != VK_TIMEOUT) {
      StopPresenting(operation, result);
      return false;
    }
  }
  return false;
}

u32 FindMemoryType(u32 type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_window.phys, &mp);
  for (u32 i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return UINT32_MAX;
}

void ImageBarrier(VkCommandBuffer c,
                  VkImage img,
                  VkImageLayout from,
                  VkImageLayout to,
                  VkAccessFlags src_a,
                  VkAccessFlags dst_a,
                  VkPipelineStageFlags src_s,
                  VkPipelineStageFlags dst_s) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  vkCmdPipelineBarrier(c, src_s, dst_s, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void DestroyRenderSemaphores() {
  for (VkSemaphore sem : g_window.retired_render_sems)
    vkDestroySemaphore(g_window.device, sem, nullptr);
  g_window.retired_render_sems.clear();
  for (VkSemaphore sem : g_window.render_sems)
    vkDestroySemaphore(g_window.device, sem, nullptr);
  g_window.render_sems.clear();
}

// Park the current semaphores instead of destroying them: the presentation
// engine may still wait on one after vkDeviceWaitIdle returns. Whatever was
// parked by the previous recreation is destroyed now, and by then a full
// swapchain generation (plus another idle) has passed.
void RetireRenderSemaphores() {
  for (VkSemaphore sem : g_window.retired_render_sems)
    vkDestroySemaphore(g_window.device, sem, nullptr);
  g_window.retired_render_sems = base::move(g_window.render_sems);
  g_window.render_sems.clear();
}

bool CreateRenderSemaphores(u32 count, base::Vector<VkSemaphore>& semaphores) {
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphores.resize(count);
  for (VkSemaphore& sem : semaphores) {
    if (vkCreateSemaphore(g_window.device, &si, nullptr, &sem) != VK_SUCCESS) {
      for (VkSemaphore created : semaphores) {
        if (created)
          vkDestroySemaphore(g_window.device, created, nullptr);
      }
      semaphores.clear();
      return false;
    }
  }
  return true;
}

bool CreateSwapchain() {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_window.phys, g_window.surface,
                                            &caps);

  // Choose a format (prefer BGRA8 unorm).
  u32 nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_window.phys, g_window.surface, &nfmt,
                                       nullptr);
  base::Vector<VkSurfaceFormatKHR> fmts(nfmt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_window.phys, g_window.surface, &nfmt,
                                       fmts.data());
  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto& f : fmts)
    if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      chosen = f;
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(g_window.window, &w, &h);
    ext.width = (u32)w;
    ext.height = (u32)h;
  }
  if (ext.width == 0 || ext.height == 0)
    return false;  // minimised; try again later

  u32 img_count = caps.minImageCount + 1;
  if (caps.maxImageCount && img_count > caps.maxImageCount)
    img_count = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sc{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sc.surface = g_window.surface;
  sc.minImageCount = img_count;
  sc.imageFormat = chosen.format;
  sc.imageColorSpace = chosen.colorSpace;
  sc.imageExtent = ext;
  sc.imageArrayLayers = 1;
  // The frame arrives as a blit (TRANSFER_DST), but the overlay renders into
  // the same images through a render pass, which needs COLOR_ATTACHMENT. Asking
  // only for TRANSFER_DST left every overlay framebuffer and render-pass begin
  // using an image without the usage its layout requires.
  sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    sc.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sc.preTransform = caps.currentTransform;
  sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  // FIFO always works but half-rates on a late frame; prefer MAILBOX
  // (triple-buffer, latest wins, no tearing). DELTA_GPU_VSYNC=0 forces
  // IMMEDIATE (benchmarking), =1 FIFO.
  {
    u32 npm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_window.phys, g_window.surface,
                                              &npm, nullptr);
    base::Vector<VkPresentModeKHR> pms(npm);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_window.phys, g_window.surface,
                                              &npm, pms.data());
    auto has = [&](VkPresentModeKHR m) {
      for (auto p : pms)
        if (p == m)
          return true;
      return false;
    };
    const char* vs = kVsync;
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (vs && vs[0] == '0' && has(VK_PRESENT_MODE_IMMEDIATE_KHR))
      mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if (!(vs && vs[0] == '1') && has(VK_PRESENT_MODE_MAILBOX_KHR))
      mode = VK_PRESENT_MODE_MAILBOX_KHR;
    sc.presentMode = mode;
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR && img_count < 3) {
      img_count = 3;  // mailbox wants >=3 images to actually triple-buffer
      if (caps.maxImageCount && img_count > caps.maxImageCount)
        img_count = caps.maxImageCount;
      sc.minImageCount = img_count;
    }
  }
  sc.clipped = VK_TRUE;
  const VkSwapchainKHR old_swapchain = g_window.swapchain;
  sc.oldSwapchain = old_swapchain;

  // Swapchain replacement is exceptional. Idle once here so every old image,
  // semaphore, and overlay attachment can be torn down together.
  if (g_window.swapchain)
    vkDeviceWaitIdle(g_window.device);

  auto discard_retired_swapchain = [&] {
    if (!old_swapchain)
      return;
    ui::OverlayVkSetSwapchain({}, {}, chosen.format);
    RetireRenderSemaphores();
    vkDestroySwapchainKHR(g_window.device, old_swapchain, nullptr);
    g_window.swapchain = VK_NULL_HANDLE;
    g_window.swap_images.clear();
  };

  VkSwapchainKHR new_swap = VK_NULL_HANDLE;
  const VkResult create_result =
      vkCreateSwapchainKHR(g_window.device, &sc, nullptr, &new_swap);
  if (create_result != VK_SUCCESS) {
    discard_retired_swapchain();
    BASE_LOGI("gfx", "vkCreateSwapchainKHR failed: VkResult={}",
              (int)create_result);
    return false;
  }

  u32 n = 0;
  vkGetSwapchainImagesKHR(g_window.device, new_swap, &n, nullptr);
  base::Vector<VkImage> new_images(n);
  vkGetSwapchainImagesKHR(g_window.device, new_swap, &n, new_images.data());
  base::Vector<VkSemaphore> new_render_sems;
  if (!CreateRenderSemaphores(n, new_render_sems)) {
    discard_retired_swapchain();
    vkDestroySwapchainKHR(g_window.device, new_swap, nullptr);
    return false;
  }

  // This destroys framebuffers and views for the old images before their
  // swapchain is destroyed, then creates attachments for the replacement.
  ui::OverlayVkSetSwapchain(new_images, ext, chosen.format);  // no-op pre-init
  RetireRenderSemaphores();
  if (g_window.swapchain)
    vkDestroySwapchainKHR(g_window.device, g_window.swapchain, nullptr);

  g_window.swapchain = new_swap;
  g_window.swap_format = chosen.format;
  g_window.swap_extent = ext;
  g_window.swap_images.swap(new_images);
  g_window.render_sems.swap(new_render_sems);
  g_window.need_recreate = false;
  return true;
}

void DestroyFrameResources(FrameSlot& slot) {
  if (slot.staging_map) {
    vkUnmapMemory(g_window.device, slot.staging_mem);
    slot.staging_map = nullptr;
  }
  if (slot.staging)
    vkDestroyBuffer(g_window.device, slot.staging, nullptr);
  if (slot.staging_mem)
    vkFreeMemory(g_window.device, slot.staging_mem, nullptr);
  if (slot.frame_img)
    vkDestroyImage(g_window.device, slot.frame_img, nullptr);
  if (slot.frame_mem)
    vkFreeMemory(g_window.device, slot.frame_mem, nullptr);
  slot.staging = VK_NULL_HANDLE;
  slot.staging_mem = VK_NULL_HANDLE;
  slot.frame_img = VK_NULL_HANDLE;
  slot.frame_mem = VK_NULL_HANDLE;
}

bool CreateFrameResources(FrameSlot& slot, u32 w, u32 h, VkFormat fmt) {
  VkDeviceSize size = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(g_window.device, &bi, nullptr, &slot.staging));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g_window.device, slot.staging, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = FindMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(g_window.device, &ba, nullptr, &slot.staging_mem));
  VK_CHECK(
      vkBindBufferMemory(g_window.device, slot.staging, slot.staging_mem, 0));
  VK_CHECK(vkMapMemory(g_window.device, slot.staging_mem, 0, size, 0,
                       &slot.staging_map));

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
  VK_CHECK(vkCreateImage(g_window.device, &ii, nullptr, &slot.frame_img));
  VkMemoryRequirements ir;
  vkGetImageMemoryRequirements(g_window.device, slot.frame_img, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex =
      FindMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(g_window.device, &ia, nullptr, &slot.frame_mem));
  VK_CHECK(
      vkBindImageMemory(g_window.device, slot.frame_img, slot.frame_mem, 0));
  return true;
}

bool EnsureFrameResources(u32 w, u32 h, VkFormat fmt) {
  bool ready = true;
  for (const FrameSlot& slot : g_window.slots)
    ready &= slot.staging != VK_NULL_HANDLE && slot.frame_img != VK_NULL_HANDLE;
  if (g_window.fb_w == w && g_window.fb_h == h && g_window.fb_format == fmt &&
      ready)
    return true;
  vkDeviceWaitIdle(g_window.device);
  for (FrameSlot& slot : g_window.slots)
    DestroyFrameResources(slot);
  g_window.last_frame_slot = kFrameSlotCount;
  g_window.fb_w = w;
  g_window.fb_h = h;
  g_window.fb_format = fmt;
  for (FrameSlot& slot : g_window.slots)
    if (!CreateFrameResources(slot, w, h, fmt))
      return false;
  return true;
}

}  // namespace

bool Init(const char* title, u32 width, u32 height) {
  base::LockGuard<base::Mutex> lock(g_init_mutex);
  if (Available())
    return true;  // already up; init is idempotent
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
    BASE_LOGI("gfx", "SDL_Init failed: {}", SDL_GetError());
    return false;
  }
  // Open the first connected controller (if any) for input + rumble. Hotplug is
  // handled in PumpEvents(); keyboard play works regardless.
  {
    int n = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&n);
    if (ids) {
      if (n > 0)
        g_window.gamepad = SDL_OpenGamepad(ids[0]);
      SDL_free(ids);
    }
  }
  if (!SDL_Vulkan_LoadLibrary(nullptr)) {
    BASE_LOGI("gfx", "SDL_Vulkan_LoadLibrary failed: {}", SDL_GetError());
    return false;
  }
  g_window.window =
      SDL_CreateWindow(g_title.empty() ? title : g_title.c_str(), (int)width,
                       (int)height, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
  if (!g_window.window) {
    BASE_LOGI("gfx", "SDL_CreateWindow failed: {}", SDL_GetError());
    return false;
  }
#if defined(__linux__)
  ApplyWindowIcon();
#endif

  // Instance: SDL-required extensions + optional validation.
  u32 n_ext = 0;
  const char* const* sdl_ext = SDL_Vulkan_GetInstanceExtensions(&n_ext);
  base::Vector<const char*> exts(sdl_ext, sdl_ext + n_ext);

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = title;
  app.apiVersion = VK_API_VERSION_1_1;

  base::Vector<const char*> layers;
  if (kVkValidate) {
    u32 nl = 0;
    vkEnumerateInstanceLayerProperties(&nl, nullptr);
    base::Vector<VkLayerProperties> lp(nl);
    vkEnumerateInstanceLayerProperties(&nl, lp.data());
    for (auto& l : lp)
      if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        layers.push_back("VK_LAYER_KHRONOS_validation");
  }

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = (u32)exts.size();
  ici.ppEnabledExtensionNames = exts.data();
  ici.enabledLayerCount = (u32)layers.size();
  ici.ppEnabledLayerNames = layers.data();
  VK_CHECK(vkCreateInstance(&ici, nullptr, &g_window.instance));

  if (!SDL_Vulkan_CreateSurface(g_window.window, g_window.instance, nullptr,
                                &g_window.surface)) {
    BASE_LOGI("gfx", "SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
    return false;
  }

  // Physical device + a queue family that does graphics AND present.
  u32 nphys = 0;
  vkEnumeratePhysicalDevices(g_window.instance, &nphys, nullptr);
  if (!nphys) {
    BASE_LOGI("gfx", "no Vulkan physical devices");
    return false;
  }
  base::Vector<VkPhysicalDevice> phs(nphys);
  vkEnumeratePhysicalDevices(g_window.instance, &nphys, phs.data());
  // Prefer a real GPU over the llvmpipe software rasteriser (type CPU) among
  // the devices that can both render and present; discrete > integrated >
  // virtual > CPU. DELTA_VK_GPU=<name-substring> forces a specific device.
  const char* want = kVkGpu;
  bool found = false;
  int best = -1;
  for (auto pd : phs) {
    u32 nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    base::Vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    u32 fam = UINT32_MAX;
    for (u32 i = 0; i < nq; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g_window.surface, &present);
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
        fam = i;
        break;
      }
    }
    if (fam == UINT32_MAX)
      continue;  // can't both render and present
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
        break;  // llvmpipe
      default:
        score = 1;
        break;
    }
    if (want && std::strstr(pp.deviceName, want))
      score = 100;
    if (score > best) {
      best = score;
      g_window.phys = pd;
      g_window.queue_family = fam;
      found = true;
    }
  }
  if (!found) {
    BASE_LOGI("gfx", "no graphics+present queue");
    return false;
  }
  {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(g_window.phys, &pp);
    BASE_LOGI("gfx", "device: {}", pp.deviceName);
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g_window.queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  base::Vector<const char*> dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  {  // VK_EXT_memory_budget (optional): powers the overlay VRAM gauge.
    u32 ne = 0;
    vkEnumerateDeviceExtensionProperties(g_window.phys, nullptr, &ne, nullptr);
    base::Vector<VkExtensionProperties> ext(ne);
    vkEnumerateDeviceExtensionProperties(g_window.phys, nullptr, &ne,
                                         ext.data());
    for (auto& e : ext)
      if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        g_window.has_mem_budget = true;
      }
  }
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = (u32)dev_exts.size();
  dci.ppEnabledExtensionNames = dev_exts.data();
  VK_CHECK(vkCreateDevice(g_window.phys, &dci, nullptr, &g_window.device));
  vkGetDeviceQueue(g_window.device, g_window.queue_family, 0, &g_window.queue);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g_window.queue_family;
  VK_CHECK(
      vkCreateCommandPool(g_window.device, &pci, nullptr, &g_window.cmd_pool));
  VkCommandBufferAllocateInfo cbi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbi.commandPool = g_window.cmd_pool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = kFrameSlotCount;
  base::Array<VkCommandBuffer, kFrameSlotCount> commands;
  VK_CHECK(vkAllocateCommandBuffers(g_window.device, &cbi, commands.data()));

  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFenceCreateInfo acquire_fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (u32 i = 0; i < kFrameSlotCount; i++) {
    g_window.slots[i].cmd = commands[i];
    VK_CHECK(vkCreateSemaphore(g_window.device, &si, nullptr,
                               &g_window.slots[i].acquire_sem));
    VK_CHECK(vkCreateFence(g_window.device, &acquire_fi, nullptr,
                           &g_window.slots[i].acquire_fence));
    VK_CHECK(
        vkCreateFence(g_window.device, &fi, nullptr, &g_window.slots[i].fence));
  }

  if (!CreateSwapchain())
    return false;
  BASE_LOGI("gfx", "swapchain {}x{}, {} images", g_window.swap_extent.width,
            g_window.swap_extent.height, (u32)g_window.swap_images.size());
  ui::OverlayVkInit(g_window.phys, g_window.device, g_window.queue,
                    g_window.queue_family, g_window.cmd_pool,
                    g_window.swap_format);
  ui::OverlayVkSetSwapchain(g_window.swap_images, g_window.swap_extent,
                            g_window.swap_format);
  return true;
}

base::Vector<base::String> GraphicsDevices() {
  base::Vector<base::String> result;
  if (!g_window.instance)
    return result;
  u32 count = 0;
  if (vkEnumeratePhysicalDevices(g_window.instance, &count, nullptr) !=
      VK_SUCCESS)
    return result;
  base::Vector<VkPhysicalDevice> devices(count);
  if (vkEnumeratePhysicalDevices(g_window.instance, &count, devices.data()) !=
      VK_SUCCESS)
    return result;
  for (u32 i = 0; i < count; ++i) {
    u32 family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count,
                                             nullptr);
    base::Vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count,
                                             families.data());
    bool presentable = false;
    for (u32 family = 0; family < family_count; ++family) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], family, g_window.surface,
                                           &present);
      presentable |=
          present && (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    }
    if (!presentable)
      continue;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(devices[i], &properties);
    result.emplace_back(properties.deviceName);
  }
  return result;
}

void ReloadOverlay() {
  ui::OverlayVkInit(g_window.phys, g_window.device, g_window.queue,
                    g_window.queue_family, g_window.cmd_pool,
                    g_window.swap_format);
  ui::OverlayVkSetSwapchain(g_window.swap_images, g_window.swap_extent,
                            g_window.swap_format);
}

void QueryVram(u64& used, u64& total) {
  used = total = 0;
  // Callers outside the present path (the GPU perf overlay) can ask before the
  // window exists, or in a headless run where it never will.
  if (g_window.phys == VK_NULL_HANDLE)
    return;
  VkPhysicalDeviceMemoryProperties2 mp2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  if (g_window.has_mem_budget)
    mp2.pNext = &budget;
  vkGetPhysicalDeviceMemoryProperties2(g_window.phys, &mp2);
  const VkPhysicalDeviceMemoryProperties& mp = mp2.memoryProperties;
  for (u32 i = 0; i < mp.memoryHeapCount; i++)
    if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += g_window.has_mem_budget ? budget.heapBudget[i]
                                       : mp.memoryHeaps[i].size;
      if (g_window.has_mem_budget)
        used += budget.heapUsage[i];
    }
}

static void PresentFrame(const void* pixels,
                         u32 w,
                         u32 h,
                         u32 src_pitch,
                         PixelFormat fmt,
                         bool frame_stalled) {
  if (!g_can_present.load(base::memory_order_acquire) || !g_window.device ||
      !w || !h || (!pixels && g_window.last_frame_slot == kFrameSlotCount))
    return;
  if (g_window.need_recreate && !CreateSwapchain())
    return;
  if (src_pitch == 0)
    src_pitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::kBgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                                : VK_FORMAT_R8G8B8A8_UNORM;
  if (pixels && !EnsureFrameResources(w, h, vkfmt))
    return;

  FrameSlot& slot = g_window.slots[g_window.next_slot];
  // The previous submission may still be reading this slot's mapped buffer.
  // Host writes must not begin until its fence signals.
  if (!WaitForPresentFence(slot.fence, "vkWaitForFences(submit)"))
    return;

  // Upload rows into the staging buffer (tightly packed w*4).
  if (pixels) {
    auto* dst = static_cast<u8*>(slot.staging_map);
    auto* src = static_cast<const u8*>(pixels);
    for (u32 y = 0; y < h; y++)
      base::MemCopy(dst + (size_t)y * w * 4, src + (size_t)y * src_pitch,
                    w * 4);
  }

  u32 idx = 0;
  VkResult result = vkResetFences(g_window.device, 1, &slot.acquire_fence);
  if (result != VK_SUCCESS) {
    StopPresenting("vkResetFences(acquire)", result);
    return;
  }
  VkResult ar;
  do {
    ar = vkAcquireNextImageKHR(g_window.device, g_window.swapchain,
                               kPresentWaitSliceNs, slot.acquire_sem,
                               slot.acquire_fence, &idx);
  } while (ar == VK_TIMEOUT && g_can_present.load(base::memory_order_acquire));
  if (!g_can_present.load(base::memory_order_acquire))
    return;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    g_window.need_recreate = true;
    return;
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
    StopPresenting("vkAcquireNextImageKHR", ar);
    return;
  }
  if (ar == VK_SUBOPTIMAL_KHR)
    g_window.need_recreate = true;
  if (!WaitForPresentFence(slot.acquire_fence, "vkWaitForFences(acquire)"))
    return;

  u64 vram_used = 0, vram_total = 0;
  QueryVram(vram_used, vram_total);
  ui::OverlayBuildFrame(g_window.swap_extent.width, g_window.swap_extent.height,
                        vram_used, vram_total, frame_stalled);

  result = vkResetCommandBuffer(slot.cmd, 0);
  if (result != VK_SUCCESS) {
    StopPresenting("vkResetCommandBuffer", result);
    return;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(slot.cmd, &bi);
  if (result != VK_SUCCESS) {
    StopPresenting("vkBeginCommandBuffer", result);
    return;
  }

  const VkImage frame_image =
      pixels ? slot.frame_img
             : g_window.slots[g_window.last_frame_slot].frame_img;
  if (pixels) {
    ImageBarrier(slot.cmd, frame_image, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy cp{};
    cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(slot.cmd, slot.staging, frame_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    ImageBarrier(slot.cmd, frame_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
  }
  ImageBarrier(slot.cmd, g_window.swap_images[idx], VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {(i32)w, (i32)h, 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[1] = {(i32)g_window.swap_extent.width,
                        (i32)g_window.swap_extent.height, 1};
  vkCmdBlitImage(slot.cmd, frame_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 g_window.swap_images[idx],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                 VK_FILTER_LINEAR);

  // The overlay's LOAD render pass draws over the blitted frame and transitions
  // the image to PRESENT_SRC; without it, do that transition directly.
  if (!ui::OverlayVkRender(slot.cmd, idx))
    ImageBarrier(slot.cmd, g_window.swap_images[idx],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT,
                 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  result = vkEndCommandBuffer(slot.cmd);
  if (result != VK_SUCCESS) {
    StopPresenting("vkEndCommandBuffer", result);
    return;
  }

  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo subi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  subi.waitSemaphoreCount = 1;
  subi.pWaitSemaphores = &slot.acquire_sem;
  subi.pWaitDstStageMask = &wait_stage;
  subi.commandBufferCount = 1;
  subi.pCommandBuffers = &slot.cmd;
  subi.signalSemaphoreCount = 1;
  subi.pSignalSemaphores = &g_window.render_sems[idx];
  result = vkResetFences(g_window.device, 1, &slot.fence);
  if (result != VK_SUCCESS) {
    StopPresenting("vkResetFences(submit)", result);
    return;
  }
  result = vkQueueSubmit(g_window.queue, 1, &subi, slot.fence);
  if (result != VK_SUCCESS) {
    StopPresenting("vkQueueSubmit", result);
    return;
  }
  if (pixels)
    g_window.last_frame_slot = g_window.next_slot;
  g_window.next_slot = (g_window.next_slot + 1) % kFrameSlotCount;

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &g_window.render_sems[idx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &g_window.swapchain;
  pi.pImageIndices = &idx;
  VkResult pr = vkQueuePresentKHR(g_window.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR)
    g_window.need_recreate = true;
  else if (pr != VK_SUCCESS)
    StopPresenting("vkQueuePresentKHR", pr);
}

void Present(const void* pixels, u32 w, u32 h, u32 src_pitch, PixelFormat fmt) {
  StopSplash();
  base::LockGuard<base::Mutex> lock(g_ui_mutex);
  if (g_exit_animation.load(base::memory_order_acquire) && !t_exit_thread)
    return;
  if (pixels && !t_splash_thread && !ui::HomeScreenActive()) {
    ui::LaunchTransitionGameReady();
    ui::PauseMenuGameReady();
  }
  if (guest::Paused()) {
    PresentFrame(nullptr, g_window.fb_w, g_window.fb_h, 0,
                 PixelFormat::kRgba8, false);
    return;
  }
  if (pixels)
    PresentFrame(pixels, w, h, src_pitch, fmt, false);
}

void RefreshFrame(bool frame_stalled) {
  base::LockGuard<base::Mutex> lock(g_ui_mutex);
  if (g_exit_animation.load(base::memory_order_acquire) && !t_exit_thread)
    return;
  PresentFrame(nullptr, g_window.fb_w, g_window.fb_h, 0, PixelFormat::kRgba8,
               frame_stalled);
}

void SetTitle(const char* title) {
  g_title = title ? title : "";
  if (g_window.window && !g_title.empty())
    SDL_SetWindowTitle(g_window.window, g_title.c_str());
}

void SetIcon(const u8* png, size_t size) {
#if defined(__linux__)
  if (size > kMaxIconSize)
    return;
  g_icon_png.assign(png, png + size);
  ApplyWindowIcon();
#else
  (void)png;
  (void)size;
#endif
}

void ShowSplash(base::Vector<u8> png) {
#if defined(__linux__)
  if (png.size() > kMaxIconSize || !CanPresent())
    return;
  StopSplash();
  g_splash.store(1, base::memory_order_release);
  guest::SpawnThread("splash", [png = base::move(png)] {
    t_splash_thread = true;
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels =
        png.empty()
            ? nullptr
            : stbi_load_from_memory(png.data(), static_cast<int>(png.size()),
                                    &w, &h, &channels, STBI_rgb_alpha);
    if ((pixels || png.empty()) && Init("prosperity", 1920, 1080)) {
      if (pixels)
        BASE_LOGI("gfx", "splash {}x{} until the first frame", w, h);
      bool shown = false;
      while (g_splash.load(base::memory_order_acquire) == 1 && CanPresent()) {
        // Again after a resize: the swapchain was rebuilt without it.
        if (pixels && (!shown || g_window.need_recreate)) {
          Present(pixels, static_cast<u32>(w), static_cast<u32>(h), 0,
                  PixelFormat::kRgba8);
          shown = true;
        } else {
          RefreshFrame(false);
        }
        PumpEvents();
        base::SleepForMicroseconds(16667);
      }
    }
    stbi_image_free(pixels);
    g_splash.store(0, base::memory_order_release);
  });
#else
  (void)png;
#endif
}

bool Available() {
  return g_window.window != nullptr && g_window.swapchain != VK_NULL_HANDLE;
}

bool CanPresent() {
  return g_can_present.load(base::memory_order_acquire);
}

void RequestPresentStop() {
  g_can_present.store(false, base::memory_order_release);
}

// Idempotent bring-up: create the window/swapchain on the first call, then just
// report availability. Safe to call every frame from the presenting thread;
// after a failed attempt it stops retrying so a no-display run doesn't spam.
bool Ensure(const char* title, u32 width, u32 height) {
  if (Available())
    return true;
  if (!g_can_present.load(base::memory_order_acquire))
    return false;
  if (g_window.device)
    return CreateSwapchain();
  if (!Init(title, width, height)) {
    g_can_present.store(false, base::memory_order_release);
    return false;
  }
  return true;
}

static bool DrainEvents() {
  ui::SyncMouseLook(g_window.window);
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (e.type == SDL_EVENT_QUIT ||
        (e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
         e.window.windowID == SDL_GetWindowID(g_window.window))) {
      ui::PauseMenuRequestExit();
      g_can_present.store(false, base::memory_order_release);
      return false;
    }
    if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
        e.type == SDL_EVENT_WINDOW_RESIZED)
      g_window.need_recreate = true;
    ui::ProcessEvent(e, g_window.window, g_window.swap_extent.width,
                     g_window.swap_extent.height);
    if (e.type == SDL_EVENT_GAMEPAD_ADDED && !g_window.gamepad)
      g_window.gamepad = SDL_OpenGamepad(e.gdevice.which);
    if (e.type == SDL_EVENT_GAMEPAD_REMOVED && g_window.gamepad &&
        e.gdevice.which == SDL_GetGamepadID(g_window.gamepad)) {
      SDL_CloseGamepad(g_window.gamepad);
      g_window.gamepad = nullptr;
    }
  }
  ui::SyncMouseLook(g_window.window);
  ui::UpdateMouseLook();
  return true;
}

bool PumpEvents() {
  StopSplash();
  base::LockGuard<base::Mutex> lock(g_ui_mutex);
  if (g_exit_animation.load(base::memory_order_acquire) && !t_exit_thread)
    return false;
  if (guest::Stopping() && guest::OnGuestThread())
    return false;
  bool alive = DrainEvents();
  SetAudioPaused(guest::Paused());
  if (guest::Paused())
    g_suppress_pad.store(true);
  if (guest::OnGuestThread()) {
    while (alive && guest::Paused() && !guest::Stopping()) {
      PresentFrame(nullptr, g_window.fb_w, g_window.fb_h, 0,
                   PixelFormat::kRgba8, false);
      base::SleepForMilliseconds(16);
      alive = DrainEvents();
      SetAudioPaused(guest::Paused());
    }
  }
  if (!alive)
    guest::SetPaused(false);
  return alive;
}

// Keyboard->DS4 adapter, laid out for two-handed keyboard play: the left hand
// moves (WASD) and works the action keys, the right hand aims (arrow keys).
// Both hands reach a shoulder pair via the Shift keys. Keep this in sync with
// the on-screen legend (overlay.cc).
bool PollKeyboardPad(PadKeys& out) {
  if (guest::Paused() || ui::HomeScreenActive() || ui::PauseMenuVisible() ||
      (g_window.window &&
       !(SDL_GetWindowFlags(g_window.window) & SDL_WINDOW_INPUT_FOCUS))) {
    out = {};
    return true;
  }
  if (!g_window.window)
    return false;
  const bool* k = SDL_GetKeyboardState(nullptr);
  if (!k)
    return false;
  auto down = [&](SDL_Scancode s) { return k[s]; };
  if (g_suppress_pad.load()) {
    const bool held =
        down(SDL_SCANCODE_LCTRL) || down(SDL_SCANCODE_RCTRL) ||
        down(SDL_SCANCODE_RETURN) || down(SDL_SCANCODE_ESCAPE) ||
        (g_window.gamepad &&
         (SDL_GetGamepadButton(g_window.gamepad, SDL_GAMEPAD_BUTTON_SOUTH) ||
          SDL_GetGamepadButton(g_window.gamepad, SDL_GAMEPAD_BUTTON_EAST)));
    if (held) {
      out = {};
      return true;
    }
    g_suppress_pad.store(false);
  }

  // Sending both stick and d-pad directions makes menus navigate twice.
  out.lx = down(SDL_SCANCODE_A) ? 0 : (down(SDL_SCANCODE_D) ? 255 : 128);
  out.ly = down(SDL_SCANCODE_W) ? 0 : (down(SDL_SCANCODE_S) ? 255 : 128);
  // Aim / shoot on the right stick (arrow keys).
  out.rx = down(SDL_SCANCODE_LEFT) ? 0 : (down(SDL_SCANCODE_RIGHT) ? 255 : 128);
  out.ry = down(SDL_SCANCODE_UP) ? 0 : (down(SDL_SCANCODE_DOWN) ? 255 : 128);

  out.cross = down(SDL_SCANCODE_SPACE);  // confirm / accept
  out.circle = down(SDL_SCANCODE_ESCAPE) ||
               down(SDL_SCANCODE_BACKSPACE);  // cancel / back
  out.square = down(SDL_SCANCODE_F);          // use card / pill
  out.triangle = down(SDL_SCANCODE_R);        // pick up / swap
  out.l1 = down(SDL_SCANCODE_Q);
  out.r1 = down(SDL_SCANCODE_E);
  out.l2 = down(SDL_SCANCODE_LSHIFT);
  out.r2 = down(SDL_SCANCODE_RSHIFT);
  out.options =
      down(SDL_SCANCODE_RETURN) || down(SDL_SCANCODE_P);  // start / pause
  out.touchpad = down(SDL_SCANCODE_TAB);                  // map / select

  u8 mouse_x, mouse_y;
  bool mouse_left, mouse_right;
  if (ui::PollMouseLook(mouse_x, mouse_y, mouse_left, mouse_right)) {
    if (mouse_x != 128)
      out.rx = mouse_x;
    if (mouse_y != 128)
      out.ry = mouse_y;
    out.r2 |= mouse_left;
    out.l2 |= mouse_right;
  }

  // Overlay a real controller when one is connected (it takes precedence over
  // keyboard for any button/axis it actively asserts).
  if (g_window.gamepad) {
    auto b = [&](SDL_GamepadButton n) {
      return SDL_GetGamepadButton(g_window.gamepad, n);
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
      int v = SDL_GetGamepadAxis(g_window.gamepad, n);
      return (v + 32768) * 255 / 65535;
    };
    int lx = axis(SDL_GAMEPAD_AXIS_LEFTX), ly = axis(SDL_GAMEPAD_AXIS_LEFTY);
    int rx = axis(SDL_GAMEPAD_AXIS_RIGHTX), ry = axis(SDL_GAMEPAD_AXIS_RIGHTY);
    if (std::abs(lx - 128) > 12)
      out.lx = (u8)lx;
    if (std::abs(ly - 128) > 12)
      out.ly = (u8)ly;
    if (std::abs(rx - 128) > 12)
      out.rx = (u8)rx;
    if (std::abs(ry - 128) > 12)
      out.ry = (u8)ry;
    // Triggers -> L2/R2 (and the analog buttons via the bit, see fillPadState).
    if (SDL_GetGamepadAxis(g_window.gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) >
        8000)
      out.l2 = true;
    if (SDL_GetGamepadAxis(g_window.gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) >
        8000)
      out.r2 = true;
  }
  return true;
}

void SetRumble(u8 large_motor, u8 small_motor) {
  if (!g_window.gamepad)
    return;
  // DS4 motors are 0..255; SDL rumble is 0..65535. Large = low-freq, small =
  // high-freq. Duration 0 means "until the next call"; the game re-issues
  // continuously.
  SDL_RumbleGamepad(g_window.gamepad, (u16)(large_motor * 257),
                    (u16)(small_motor * 257), 0);
}

void ShowExitTransition() {
#if defined(__linux__)
  StopSplash();
  if (!Available() || !CanPresent() || g_exit_thread)
    return;
  {
    // Take ownership after the game presenter and any inline window calls
    // have returned.
    base::LockGuard<base::Mutex> lock(g_ui_mutex);
    g_exit_animation.store(true, base::memory_order_release);
  }
  g_exit_thread = base::MakeUnique<base::Thread>(
      "exit-ui",
      [] {
        t_exit_thread = true;
        while (g_exit_animation.load(base::memory_order_acquire) &&
               CanPresent()) {
          if (!PumpEvents())
            break;
          RefreshFrame(false);
          base::SleepForMilliseconds(16);
        }
      },
      true);
  if (!g_exit_thread->good()) {
    g_exit_thread = {};
    g_exit_animation.store(false, base::memory_order_release);
  }
#endif
}

void ResetGuest() {
  StopExitAnimation();
  StopSplash();
  if (g_window.device)
    vkDeviceWaitIdle(g_window.device);
  for (auto& slot : g_window.slots)
    DestroyFrameResources(slot);
  g_window.fb_w = g_window.fb_h = 0;
  g_window.last_frame_slot = kFrameSlotCount;
  ui::ResetMouseLook(g_window.window);
  SetRumble(0, 0);
  g_suppress_pad.store(true);
  g_can_present.store(true, base::memory_order_release);
  SetInGameplay(false);
#if defined(__linux__)
  g_icon_png = {};
#endif
  SetTitle("Prosperity");
}

void Shutdown() {
  StopExitAnimation();
  ui::ResetMouseLook(g_window.window);
  guest::SetPaused(false);
  g_suppress_pad.store(false);
  SetAudioPaused(false);
  StopSplash();
  if (g_window.device)
    vkDeviceWaitIdle(g_window.device);
  ui::OverlayVkShutdown();
  ui::OverlayShutdownImGui();
  for (FrameSlot& slot : g_window.slots) {
    DestroyFrameResources(slot);
    if (slot.fence)
      vkDestroyFence(g_window.device, slot.fence, nullptr);
    if (slot.acquire_fence)
      vkDestroyFence(g_window.device, slot.acquire_fence, nullptr);
    if (slot.acquire_sem)
      vkDestroySemaphore(g_window.device, slot.acquire_sem, nullptr);
  }
  DestroyRenderSemaphores();
  if (g_window.cmd_pool)
    vkDestroyCommandPool(g_window.device, g_window.cmd_pool, nullptr);
  if (g_window.swapchain)
    vkDestroySwapchainKHR(g_window.device, g_window.swapchain, nullptr);
  if (g_window.device)
    vkDestroyDevice(g_window.device, nullptr);
  if (g_window.surface)
    vkDestroySurfaceKHR(g_window.instance, g_window.surface, nullptr);
  if (g_window.instance)
    vkDestroyInstance(g_window.instance, nullptr);
  if (g_window.window)
    SDL_DestroyWindow(g_window.window);
  SDL_Quit();
  g_window = State{};
  g_can_present.store(true, base::memory_order_release);
}

}  // namespace host
