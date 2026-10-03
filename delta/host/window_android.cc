/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen Vulkan present for the Android app (DELTA_ANDROID_APP). Same scheme
 * as the desktop window_sdl.cc (CPU framebuffer -> staging buffer -> device
 * image
 * -> blit into the acquired swapchain image -> present), but the window is an
 * ANativeWindow handed in by the NativeActivity loop (android_main) and the
 * surface comes from VK_KHR_android_surface. All Vulkan calls run on the guest
 * renderer thread (the only caller of ensure()/present()); android_main only
 * publishes the window handle and the touch-derived pad state.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "base/arch.h"

#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/native_window.h>
#include <vulkan/vulkan.h>
#include "base/logging.h"

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host/window.h"
#include "host/window_android.h"

namespace host {
namespace {

#define VK_CHECK(expr)                                                    \
  do {                                                                    \
    VkResult _r = (expr);                                                 \
    if (_r != VK_SUCCESS) {                                               \
      BASE_LOGI("gfx-android", "{} failed: VkResult={}", #expr, (int)_r); \
      return false;                                                       \
    }                                                                     \
  } while (0)

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
  ANativeWindow* window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  u32 queue_family = 0;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swap_format = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D swap_extent{};
  VkSurfaceTransformFlagBitsKHR pre_transform =
      VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  base::Vector<VkImage> swap_images;

  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  base::Array<FrameSlot, kFrameSlotCount> slots;
  base::Vector<VkSemaphore> render_sems;
  // Semaphores of a replaced swapchain; see RetireRenderSemaphores.
  base::Vector<VkSemaphore> retired_render_sems;
  u32 next_slot = 0;

  // Framebuffer dimensions shared by the per-slot upload resources.
  u32 fb_w = 0, fb_h = 0;
  VkFormat fb_format = VK_FORMAT_R8G8B8A8_UNORM;

  bool need_recreate = false;
  bool ready = false;
};
State g_window;
base::Atomic<bool> g_present_failed{false};
base::Atomic<bool> g_present_stop_requested{false};
constexpr u64 kPresentWaitSliceNs = 50'000'000;

void StopPresenting(const char* operation, VkResult result) {
  BASE_LOGI("gfx-android", "{} failed: VkResult={}", operation, (int)result);
  g_present_failed.store(true, base::memory_order_release);
}

bool WaitForPresentFence(VkFence fence, const char* operation) {
  while (!g_present_failed.load(base::memory_order_acquire) &&
         !g_present_stop_requested.load(base::memory_order_acquire)) {
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

// Window handle + touch state published by android_main (other thread).
base::Mutex g_in_mutex;
ANativeWindow* g_pending_window = nullptr;
bool g_window_changed = false;
constexpr int kMaxTouch = 8;
Touch g_touches[kMaxTouch];
int g_touch_count = 0;
u32 g_surface_w = 0, g_surface_h = 0;

// On-screen virtual gamepad, in normalised [0,1] surface coords. The present
// blit stretches the whole game framebuffer across the whole surface, so a
// point at (fx,fy) draws and hit-tests at the same place. Buttons use a square
// zone of half-size `r` (in height units); sticks use a generous side region.
struct Btn {
  float cx, cy, r;
};
const Btn kCross{0.70f, 0.86f, 0.06f};    // use item / confirm
const Btn kCircle{0.80f, 0.66f, 0.06f};   // cancel
const Btn kBomb{0.70f, 0.62f, 0.06f};     // bomb (R1)
const Btn kOptions{0.95f, 0.10f, 0.05f};  // pause / start
struct Stick {
  float cx, cy, r;
};
const Stick kLStick{0.12f, 0.70f, 0.14f};  // move
const Stick kRStick{0.88f, 0.70f, 0.14f};  // aim / shoot

bool InBtn(float nx, float ny, const Btn& b, float aspect) {
  // square zone; scale x by aspect so the zone is square on screen, not
  // stretched
  return std::fabs((nx - b.cx) * aspect) <= b.r && std::fabs(ny - b.cy) <= b.r;
}

// Map the down touches to a DS4 pad against the layout above.
PadKeys ComputePad() {
  PadKeys k;  // neutral (sticks centred at 128)
  if (!g_surface_w || !g_surface_h)
    return k;
  float aspect = float(g_surface_h) / float(g_surface_w);
  for (int i = 0; i < g_touch_count; i++) {
    float nx = g_touches[i].x / g_surface_w;
    float ny = g_touches[i].y / g_surface_h;
    if (InBtn(nx, ny, kOptions, aspect)) {
      k.options = true;
      continue;
    }
    if (InBtn(nx, ny, kCross, aspect)) {
      k.cross = true;
      continue;
    }
    if (InBtn(nx, ny, kCircle, aspect)) {
      k.circle = true;
      continue;
    }
    if (InBtn(nx, ny, kBomb, aspect)) {
      k.r1 = true;
      continue;
    }
    // Sticks: left half moves, right half aims. Deflection from the ring
    // centre.
    const Stick& s = (nx < 0.5f) ? kLStick : kRStick;
    float dx = (nx - s.cx) / s.r, dy = (ny - s.cy) / s.r;
    dx = base::Clamp(dx, -1.0f, 1.0f);
    dy = base::Clamp(dy, -1.0f, 1.0f);
    u8 vx = u8(base::Clamp(128.0f + dx * 127.0f, 0.0f, 255.0f));
    u8 vy = u8(base::Clamp(128.0f + dy * 127.0f, 0.0f, 255.0f));
    if (nx < 0.5f) {
      k.lx = vx;
      k.ly = vy;
      k.left = dx < -0.4f;
      k.right = dx > 0.4f;
      k.up = dy < -0.4f;
      k.down = dy > 0.4f;  // d-pad for menus
    } else {
      k.rx = vx;
      k.ry = vy;
    }
  }
  return k;
}

// --- helper overlay (CPU alpha-blend into the present framebuffer) ----------
// Drawn in the game framebuffer, which the present blit stretches across the
// whole surface, so normalised (fx,fy) lands at the same on-screen spot. Radii
// are corrected by the surface aspect so glyphs stay round on screen.

inline void BlendPx(u8* buf,
                    int w,
                    int h,
                    bool bgra,
                    int x,
                    int y,
                    u8 r,
                    u8 g,
                    u8 b,
                    float a) {
  if (x < 0 || y < 0 || x >= w || y >= h || a <= 0.0f)
    return;
  u8* p = buf + (size_t)(y * w + x) * 4;
  int ri = bgra ? 2 : 0, bi = bgra ? 0 : 2;
  p[ri] = u8(p[ri] * (1 - a) + r * a);
  p[1] = u8(p[1] * (1 - a) + g * a);
  p[bi] = u8(p[bi] * (1 - a) + b * a);
}

// Round-on-screen ellipse in the framebuffer. band<=0 => filled disc, else a
// ring of that normalised-radius thickness.
void Glyph(u8* buf,
           int w,
           int h,
           bool bgra,
           float xr,
           float yr,
           float fx,
           float fy,
           float sr,
           float band,
           u8 r,
           u8 g,
           u8 b,
           float a) {
  float cx = fx * w, cy = fy * h;
  float rx = sr * xr, ry = sr * yr;  // px radii (round on screen)
  int x0 = base::Max(0, int(cx - rx - 1)),
      x1 = base::Min(w - 1, int(cx + rx + 1));
  int y0 = base::Max(0, int(cy - ry - 1)),
      y1 = base::Min(h - 1, int(cy + ry + 1));
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++) {
      float ex = (x - cx) / rx, ey = (y - cy) / ry;
      float d = std::sqrt(ex * ex + ey * ey);
      if (band <= 0.0f ? (d <= 1.0f) : (d <= 1.0f && d >= 1.0f - band))
        BlendPx(buf, w, h, bgra, x, y, r, g, b, a);
    }
}

void DrawOverlay(u8* buf, u32 w, u32 h, bool bgra) {
  Touch t[kMaxTouch];
  int n;
  u32 sw, sh;
  {
    base::LockGuard<base::Mutex> lk(g_in_mutex);
    n = g_touch_count;
    std::memcpy(t, g_touches, sizeof(Touch) * (n < kMaxTouch ? n : kMaxTouch));
    sw = g_surface_w;
    sh = g_surface_h;
  }
  if (!sw || !sh)
    return;
  // Per-unit px radii so a glyph stays round after the fb is stretched to the
  // surface: screen radius sr*sh maps to sr*h px in y and sr*(sh*w/sw) px in x.
  float yr = float(h);
  float xr = float(sh) * float(w) / float(sw);
  float aspect = float(sh) / float(sw);

  auto pressed = [&](const Btn& btn) {
    for (int i = 0; i < n; i++)
      if (InBtn(t[i].x / sw, t[i].y / sh, btn, aspect))
        return true;
    return false;
  };

  // Face buttons: PS4 colours, brighter when held.
  struct BC {
    const Btn& b;
    u8 r, g, bl;
  } bcs[] = {{kCross, 70, 130, 255},
             {kCircle, 240, 70, 70},
             {kBomb, 220, 220, 220},
             {kOptions, 200, 200, 200}};
  for (auto& c : bcs) {
    float a = pressed(c.b) ? 0.85f : 0.40f;
    Glyph(buf, w, h, bgra, xr, yr, c.b.cx, c.b.cy, c.b.r, 0.0f, c.r, c.g, c.bl,
          a);
    Glyph(buf, w, h, bgra, xr, yr, c.b.cx, c.b.cy, c.b.r, 0.18f, 255, 255, 255,
          0.6f);
  }

  // Sticks: outer ring + a dot at the current deflection.
  const Stick* sticks[] = {&kLStick, &kRStick};
  for (auto* s : sticks) {
    Glyph(buf, w, h, bgra, xr, yr, s->cx, s->cy, s->r, 0.10f, 255, 255, 255,
          0.5f);
    float dotx = s->cx, doty = s->cy;
    for (int i = 0; i < n; i++) {
      float nx = t[i].x / sw, ny = t[i].y / sh;
      bool mine = (s == &kLStick) ? (nx < 0.5f) : (nx >= 0.5f);
      if (!mine)
        continue;
      float dx = base::Clamp((nx - s->cx) / s->r, -1.0f, 1.0f);
      float dy = base::Clamp((ny - s->cy) / s->r, -1.0f, 1.0f);
      dotx = s->cx + dx * s->r;
      doty = s->cy + dy * s->r;
    }
    Glyph(buf, w, h, bgra, xr, yr, dotx, doty, s->r * 0.4f, 0.0f, 255, 255, 255,
          0.8f);
  }
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
// engine may still wait on one after vkDeviceWaitIdle returns (that guarantee
// needs VK_EXT_swapchain_maintenance1). Whatever the previous recreation
// parked is destroyed now, a full swapchain generation later.
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

  u32 nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_window.phys, g_window.surface, &nfmt,
                                       nullptr);
  base::Vector<VkSurfaceFormatKHR> fmts(nfmt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g_window.phys, g_window.surface, &nfmt,
                                       fmts.data());
  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto& f : fmts)
    if (f.format == VK_FORMAT_R8G8B8A8_UNORM ||
        f.format == VK_FORMAT_B8G8R8A8_UNORM)
      chosen = f;
  // Opt out of Android pre-rotation: phones report currentTransform=ROTATE_90/
  // 270 for a landscape window (the panel is natively portrait) and expect the
  // app to render rotated. We present via a plain blit (can't rotate), so we
  // ask the compositor to do the rotation by choosing IDENTITY. currentExtent
  // is in the pre-rotated basis, so swap W/H to get the identity (display)
  // extent.
  const VkSurfaceTransformFlagsKHR rotated =
      VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR |
      VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
  bool pre_rotated = (caps.currentTransform & rotated) != 0;
  g_window.pre_transform =
      (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : caps.currentTransform;
  BASE_LOGI("gfx-android",
            "surface transform={:#x} supported={:#x} -> using {:#x}",
            (u32)caps.currentTransform, (u32)caps.supportedTransforms,
            (u32)g_window.pre_transform);

  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    ext.width = (u32)ANativeWindow_getWidth(g_window.window);
    ext.height = (u32)ANativeWindow_getHeight(g_window.window);
  }
  if (g_window.pre_transform == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR &&
      pre_rotated)
    base::Swap(ext.width, ext.height);
  if (ext.width == 0 || ext.height == 0)
    return false;
  g_window.swap_extent = ext;
  {
    // Input + overlay work in the on-screen (landscape) space, which may differ
    // from the swapchain extent when the compositor is rotating for us.
    base::LockGuard<base::Mutex> lk(g_in_mutex);
    g_surface_w = base::Max(ext.width, ext.height);
    g_surface_h = base::Min(ext.width, ext.height);
  }

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
  sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sc.preTransform = g_window.pre_transform;
  sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sc.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sc.clipped = VK_TRUE;
  const VkSwapchainKHR old_swapchain = g_window.swapchain;
  sc.oldSwapchain = old_swapchain;

  // Replacement is exceptional. Idle once so old images and their render-
  // finished semaphores can be destroyed together after the new set is ready.
  if (g_window.swapchain)
    vkDeviceWaitIdle(g_window.device);

  auto discard_retired_swapchain = [&] {
    if (!old_swapchain)
      return;
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
    BASE_LOGI("gfx-android", "vkCreateSwapchainKHR failed: VkResult={}",
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

  RetireRenderSemaphores();
  if (g_window.swapchain)
    vkDestroySwapchainKHR(g_window.device, g_window.swapchain, nullptr);

  g_window.swapchain = new_swap;
  g_window.swap_format = chosen.format;
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
  g_window.fb_w = w;
  g_window.fb_h = h;
  g_window.fb_format = fmt;
  for (FrameSlot& slot : g_window.slots)
    if (!CreateFrameResources(slot, w, h, fmt))
      return false;
  return true;
}

// Full bring-up against the current g.window: instance, android surface,
// device, command/sync objects and the swapchain.
bool BringUp() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "prosperity";
  app.apiVersion = VK_API_VERSION_1_1;
  const char* exts[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = exts;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &g_window.instance));

  VkAndroidSurfaceCreateInfoKHR si{
      VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
  si.window = g_window.window;
  VK_CHECK(vkCreateAndroidSurfaceKHR(g_window.instance, &si, nullptr,
                                     &g_window.surface));

  u32 nphys = 0;
  vkEnumeratePhysicalDevices(g_window.instance, &nphys, nullptr);
  if (!nphys) {
    BASE_LOGI("gfx-android", "no Vulkan physical devices");
    return false;
  }
  base::Vector<VkPhysicalDevice> phs(nphys);
  vkEnumeratePhysicalDevices(g_window.instance, &nphys, phs.data());
  bool found = false;
  for (auto pd : phs) {
    u32 nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    base::Vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    for (u32 i = 0; i < nq; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g_window.surface, &present);
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
        g_window.phys = pd;
        g_window.queue_family = i;
        found = true;
        break;
      }
    }
    if (found)
      break;
  }
  if (!found) {
    BASE_LOGI("gfx-android", "no graphics+present queue");
    return false;
  }
  {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(g_window.phys, &pp);
    BASE_LOGI("gfx-android", "device: {}", pp.deviceName);
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g_window.queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  const char* dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = dev_exts;
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
  base::Array<VkCommandBuffer, kFrameSlotCount> commands{};
  VK_CHECK(vkAllocateCommandBuffers(g_window.device, &cbi, commands.data()));

  VkSemaphoreCreateInfo si2{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFenceCreateInfo acquire_fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (u32 i = 0; i < kFrameSlotCount; i++) {
    g_window.slots[i].cmd = commands[i];
    VK_CHECK(vkCreateSemaphore(g_window.device, &si2, nullptr,
                               &g_window.slots[i].acquire_sem));
    VK_CHECK(vkCreateFence(g_window.device, &acquire_fi, nullptr,
                           &g_window.slots[i].acquire_fence));
    VK_CHECK(
        vkCreateFence(g_window.device, &fi, nullptr, &g_window.slots[i].fence));
  }

  if (!CreateSwapchain())
    return false;
  BASE_LOGI("gfx-android", "swapchain {}x{}, {} images",
            g_window.swap_extent.width, g_window.swap_extent.height,
            (u32)g_window.swap_images.size());
  g_window.ready = true;
  return true;
}

// Tear everything down (window lost). The guest GPU renderer has its own Vulkan
// device, so dropping ours only stops presentation; it resumes on re-init.
void Teardown() {
  if (g_window.device)
    vkDeviceWaitIdle(g_window.device);
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
  ANativeWindow* keep = g_window.window;
  g_window = State{};
  g_window.window = keep;
  g_present_failed.store(false, base::memory_order_release);
}

}  // namespace

// --- public window API ------------------------------------------------------

bool Init(const char*, u32, u32) {
  if (available())
    return true;
  if (g_window.ready)
    return CreateSwapchain();
  if (g_window.instance)
    Teardown();
  return g_window.window && BringUp();
}

bool Available() {
  return g_window.ready && g_window.window != nullptr &&
         g_window.swapchain != VK_NULL_HANDLE;
}

bool CanPresent() {
  base::LockGuard<base::Mutex> lk(g_in_mutex);
  return !g_present_stop_requested.load(base::memory_order_acquire) &&
         (g_window_changed ||
          !g_present_failed.load(base::memory_order_acquire)) &&
         (g_pending_window != nullptr || g_window.window != nullptr);
}

void RequestPresentStop() {
  g_present_stop_requested.store(true, base::memory_order_release);
}

bool Ensure(const char*, u32, u32) {
  // Adopt any window change published by android_main (this thread owns
  // Vulkan).
  {
    base::LockGuard<base::Mutex> lk(g_in_mutex);
    if (g_window_changed) {
      g_window_changed = false;
      if (g_pending_window != g_window.window ||
          g_present_failed.load(base::memory_order_acquire)) {
        if (g_window.instance)
          Teardown();  // resets g, preserves g.window
        else
          g_present_failed.store(false, base::memory_order_release);
        g_window.window = g_pending_window;
      }
    }
  }
  if (g_present_failed.load(base::memory_order_acquire) ||
      g_present_stop_requested.load(base::memory_order_acquire))
    return false;
  if (!g_window.window)
    return false;
  if (!g_window.instance)
    return BringUp();
  if (!g_window.ready) {
    Teardown();
    return g_window.window && BringUp();
  }
  if (g_window.need_recreate)
    CreateSwapchain();
  return Available();
}

void RefreshFrame(bool) {}

void Present(const void* pixels, u32 w, u32 h, u32 src_pitch, PixelFormat fmt) {
  if (g_present_failed.load(base::memory_order_acquire) ||
      g_present_stop_requested.load(base::memory_order_acquire) ||
      !g_window.device || !g_window.swapchain || !pixels || !w || !h)
    return;
  if (g_window.need_recreate && !CreateSwapchain())
    return;
  if (src_pitch == 0)
    src_pitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::bgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                               : VK_FORMAT_R8G8B8A8_UNORM;
  if (!EnsureFrameResources(w, h, vkfmt))
    return;

  FrameSlot& slot = g_window.slots[g_window.next_slot];
  // The previous submission may still be reading this slot's mapped buffer.
  // Host writes and the CPU overlay must wait for that use to finish.
  if (!WaitForPresentFence(slot.fence, "vkWaitForFences"))
    return;

  auto* dst = static_cast<u8*>(slot.staging_map);
  auto* src = static_cast<const u8*>(pixels);
  for (u32 y = 0; y < h; y++)
    std::memcpy(dst + (size_t)y * w * 4, src + (size_t)y * src_pitch, w * 4);

  // Composite the virtual-gamepad helper over the frame.
  DrawOverlay(dst, w, h, fmt == PixelFormat::bgra8);

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
  } while (ar == VK_TIMEOUT &&
           !g_present_failed.load(base::memory_order_acquire) &&
           !g_present_stop_requested.load(base::memory_order_acquire));
  if (g_present_stop_requested.load(base::memory_order_acquire))
    return;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    g_window.need_recreate = true;
    return;
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
    StopPresenting("vkAcquireNextImageKHR", ar);
    return;
  }
  if (!WaitForPresentFence(slot.acquire_fence, "vkWaitForFences(acquire)"))
    return;

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

  ImageBarrier(slot.cmd, slot.frame_img, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(slot.cmd, slot.staging, slot.frame_img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

  ImageBarrier(slot.cmd, slot.frame_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
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
  vkCmdBlitImage(slot.cmd, slot.frame_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 g_window.swap_images[idx],
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                 VK_FILTER_LINEAR);

  ImageBarrier(
      slot.cmd, g_window.swap_images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
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
    StopPresenting("vkResetFences", result);
    return;
  }
  result = vkQueueSubmit(g_window.queue, 1, &subi, slot.fence);
  if (result != VK_SUCCESS) {
    StopPresenting("vkQueueSubmit", result);
    return;
  }
  g_window.next_slot = (g_window.next_slot + 1) % kFrameSlotCount;

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &g_window.render_sems[idx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &g_window.swapchain;
  pi.pImageIndices = &idx;
  VkResult pr = vkQueuePresentKHR(g_window.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR)
    g_window.need_recreate = true;
  else if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR)
    StopPresenting("vkQueuePresentKHR", pr);
}

// Lifecycle is driven by android_main's looper; nothing to pump here.
bool PumpEvents() {
  return true;
}

bool PollKeyboardPad(PadKeys& out) {
  base::LockGuard<base::Mutex> lk(g_in_mutex);
  out = ComputePad();
  return true;
}

// Haptics on the touch-gamepad build would need a JNI call to the Vibrator
// service; no-op until that's wired through the NativeActivity.
void SetRumble(u8, u8) {}

void ResetGuest() {}

void Shutdown() {
  Teardown();
}

void QueryVram(u64& used, u64& total) {
  used = total = 0;
}

void SetAndroidWindow(ANativeWindow* window) {
  base::LockGuard<base::Mutex> lk(g_in_mutex);
  g_pending_window = window;
  g_window_changed = true;
}

void SetAndroidTouches(const Touch* pts, int count) {
  base::LockGuard<base::Mutex> lk(g_in_mutex);
  g_touch_count = count < 0 ? 0 : (count > kMaxTouch ? kMaxTouch : count);
  for (int i = 0; i < g_touch_count; i++)
    g_touches[i] = pts[i];
}

}  // namespace host
