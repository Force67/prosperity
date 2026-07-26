/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_frame.h"

#include "gfx/gfx.h"
#include "rhi/renderer.h"
#include "vulkan/vk_capture.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_draw_recomp.h"
#include "vulkan/vk_format.h"
#include "vulkan/vk_perf.h"
#include "vulkan/vk_pipeline_cache.h"
#include "vulkan/vk_present.h"
#include "vulkan/vk_render_target.h"
#include "vulkan/vk_texture_cache.h"
#include "vulkan/vk_upload_ring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

namespace gpu::vk {

FrameState g_frame;

bool createFrameSlots() {
  VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (auto &slot : g_frame.slots) {
    VKOK(vkAllocateCommandBuffers(g_dev.device, &ca, &slot.cmd));
    VKOK(vkCreateFence(g_dev.device, &fc, nullptr, &slot.fence));
  }
  g_frame.cmd = g_frame.slots[0].cmd;
  return true;
}

// Pipelined by default; DELTA_GPU_SYNC=1 restores the submit-and-wait frame.
// DELTA_GPU_RTSTAT also forces sync: its readback reuses the active slot's
// buffer mid-flight, which pipelining would present a frame later.
bool framePipelined() {
  static const bool sync = [] {
    const char *e = std::getenv("DELTA_GPU_SYNC");
    return (e && e[0] && e[0] != '0') ||
           std::getenv("DELTA_GPU_RTSTAT") != nullptr;
  }();
  return !sync;
}

void ensureReadback(uint32_t w, uint32_t h, VkFormat fmt) {
  VkDeviceSize need = (VkDeviceSize)w * h * formatBytes(fmt);
  if (g_frame.readback && need <= g_frame.readbackSize)
    return;
  vkDeviceWaitIdle(g_dev.device);
  if (g_frame.readbackMap) vkUnmapMemory(g_dev.device, g_frame.readbackMem);
  if (g_frame.readback) vkDestroyBuffer(g_dev.device, g_frame.readback, nullptr);
  if (g_frame.readbackMem) vkFreeMemory(g_dev.device, g_frame.readbackMem, nullptr);
  g_frame.readbackSize = need;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = need; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(g_dev.device, &bi, nullptr, &g_frame.readback);
  VkMemoryRequirements br; vkGetBufferMemoryRequirements(g_dev.device, g_frame.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  // CPU reads this buffer every frame (the flip) -> prefer HOST_CACHED so reads hit
  // cache instead of streaming from write-combined memory (the dominant frame cost).
  ba.memoryTypeIndex = findMemoryTypePref(br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g_dev.device, &ba, nullptr, &g_frame.readbackMem);
  vkBindBufferMemory(g_dev.device, g_frame.readback, g_frame.readbackMem, 0);
  vkMapMemory(g_dev.device, g_frame.readbackMem, 0, need, 0, &g_frame.readbackMap);
}

namespace {

// DELTA_RDOC_FRAME=N: bracket frame N's guest rendering with a RenderDoc capture.
// The guest draws run on this device, which owns no swapchain (the compositor
// presents the read-back pixels from its own device), so a capture taken at a
// present boundary only ever catches that final blit. The frame has to be marked
// explicitly, and the instance named, or RenderDoc picks the wrong device.
// RENDERDOC_API_1_0_0 is append-only, so these entry indices hold in every
// version; the ones in between are options and keybind setters we do not use.
struct RdocApi {
  void *entry0[19];
  void (*StartFrameCapture)(void *dev, void *wnd);
  uint32_t (*IsFrameCapturing)();
  uint32_t (*EndFrameCapture)(void *dev, void *wnd);
};

RdocApi *rdocApi() {
  static RdocApi *api = [] () -> RdocApi * {
    using GetApi = int (*)(uint32_t version, void **out);
    void *lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    auto get = lib ? reinterpret_cast<GetApi>(dlsym(lib, "RENDERDOC_GetAPI")) : nullptr;
    RdocApi *a = nullptr;
    if (!get || get(10000, reinterpret_cast<void **>(&a)) != 1) {
      std::fprintf(stderr, "[rdoc] no capture layer attached\n");
      return nullptr;
    }
    return a;
  }();
  return api;
}

int rdocFrame() {
  static const int f = [] {
    const char *e = std::getenv("DELTA_RDOC_FRAME"); return e ? std::atoi(e) : 0;
  }();
  return f;
}

// RenderDoc identifies a Vulkan device by its instance's dispatch pointer.
void *rdocDevice() { return *reinterpret_cast<void **>(g_dev.instance); }
// DELTA_GPU_RTSTAT: every 200th frame, read back each render target used this
// frame and report how many sampled texels are non-zero. RTSTAT_FRAME selects a
// single early frame instead. DELTA_GPU_RTDUMP also writes the selected targets.
void reportRtContents() {
  static const bool enabled = std::getenv("DELTA_GPU_RTSTAT") != nullptr;
  static const bool dump = std::getenv("DELTA_GPU_RTDUMP") != nullptr;
  static const int reportFrame = [] {
    const char *e = std::getenv("DELTA_GPU_RTSTAT_FRAME");
    return e ? std::atoi(e) : 0;
  }();
  // DELTA_GPU_RTSTAT_EVERY=<n>: sample every n frames instead of every 200, so a
  // per-frame flicker can be told apart from a slow animation.
  static const int every = [] {
    const char *e = std::getenv("DELTA_GPU_RTSTAT_EVERY");
    return e ? std::max(1, std::atoi(e)) : 200;
  }();
  if (!enabled || (reportFrame ? g_frame.num != reportFrame
                               : g_frame.num % every != 0))
    return;
  int reported = 0;
  for (auto &kv : g_rts) {
    RTarget &rt = kv.second;
    if (!rt.usedThisFrame || reported >= 32) continue;
    reported++;
    ensureReadback(rt.w, rt.h, rt.fmt);
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    VkCommandBuffer c; vkAllocateCommandBuffers(g_dev.device, &ca, &c);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(c, &cbi);
    imageBarrier(c, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {rt.w, rt.h, 1};
    vkCmdCopyImageToBuffer(c, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           g_frame.readback, 1, &copy);
    vkEndCommandBuffer(c);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &c;
    vkResetFences(g_dev.device, 1, &g_dev.fence);
    vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence);
    vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX);
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
    const uint32_t *px = static_cast<const uint32_t *>(g_frame.readbackMap);
    const uint64_t n = static_cast<uint64_t>(rt.w) * rt.h;
    const uint64_t step = n > 16384 ? n / 16384 : 1;
    uint64_t nz = 0, rgb_nz = 0, samples = 0, lumaSum = 0;
    uint32_t distinct[4] = {};
    uint32_t num_distinct = 0;
    for (uint64_t i = 0; i < n; i += step, samples++) {
      const uint32_t v = px[i];
      // Mean brightness of the sampled grid: a count of non-zero pixels cannot
      // show a target drifting brighter frame over frame, which is what a
      // runaway exposure or an accumulating pass looks like.
      lumaSum += ((v & 0xFF) + ((v >> 8) & 0xFF) + ((v >> 16) & 0xFF)) / 3;
      if (v) nz++;
      if (v & 0x00FFFFFFu) rgb_nz++;  // ignores an opaque-black alpha channel
      bool seen = false;
      for (uint32_t k = 0; k < num_distinct; k++) seen |= distinct[k] == v;
      if (!seen && num_distinct < 4) distinct[num_distinct++] = v;
    }
    std::fprintf(stderr,
                 "[rtstat] f%d RT %#lx %ux%u draws=%u nz=%lu rgbnz=%lu/%lu "
                 "mean=%lu vals=%08x %08x %08x %08x\n",
                 g_frame.num, (unsigned long)kv.first, rt.w, rt.h, rt.draws,
                 (unsigned long)nz, (unsigned long)rgb_nz,
                 (unsigned long)samples,
                 (unsigned long)(samples ? lumaSum / samples : 0), distinct[0],
                 distinct[1], distinct[2], distinct[3]);
    if (dump) {
      std::vector<uint8_t> bgra(n * 4);
      const auto *src = static_cast<const uint8_t *>(g_frame.readbackMap);
      const uint32_t srcBytes = formatBytes(rt.fmt);
      for (uint64_t i = 0; i < n; i++)
        readbackPixelBgra(src + i * srcBytes, rt.fmt, bgra.data() + i * 4);
      char path[256];
      std::snprintf(path, sizeof(path), "%s/rt_f%d_%#lx_%ux%u.ppm", dumpDir(),
                    g_frame.num, (unsigned long)kv.first, rt.w, rt.h);
      writePpm(path, bgra.data(), rt.w, rt.h);
    }
  }
}

}  // namespace
}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

void beginFrame() {
  if (!g_dev.ready) return;
  // Objects retired two frames ago are past every in-flight command buffer
  // (see releaseRetiredTextures) and safe to destroy now.
  releaseRetiredTextures();
  if (!createPipeline()) return;
  createTexPipeline();  // best-effort; colored path still works without it
  g_frame.draws = 0;
  g_frame.heuristic = 0;
  g_frame.maxIdx = 0;
  g_frame.num++;
  // DELTA_GPU_FORCECLEAR=<rt address>: clear that target at the top of every
  // frame. Diagnostic for a target the title clears by a means we do not see --
  // additive passes into it otherwise accumulate frame over frame.
  static const uint64_t forceClear = [] {
    const char *e = std::getenv("DELTA_GPU_FORCECLEAR");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  if (forceClear) {
    auto it = g_rts.find(forceClear);
    if (it != g_rts.end()) it->second.clearPending = true;
  }
  if (rdocFrame() && g_frame.num == rdocFrame() && rdocApi()) {
    rdocApi()->StartFrameCapture(rdocDevice(), nullptr);
    std::fprintf(stderr, "[rdoc] capturing frame %d\n", g_frame.num);
  }
  // Bind the active frame slot: its command buffer + readback aliases, and its
  // half of each host-visible ring (the other half may still be read by the
  // in-flight previous frame).
  FrameSlot &slot = g_frame.slots[g_frame.slotIdx];
  g_frame.cmd = slot.cmd;
  g_frame.readback = slot.readback;
  g_frame.readbackMem = slot.readbackMem;
  g_frame.readbackMap = slot.readbackMap;
  g_frame.readbackSize = slot.readbackSize;
  const VkDeviceSize vbBase = g_frame.slotIdx * (kVbRing / 2);
  const VkDeviceSize ibBase = g_frame.slotIdx * (kIbRing / 2);
  const VkDeviceSize uboBase = g_frame.slotIdx * (kUboRing / 2);
  g_ring.vbOffset = vbBase; g_ring.vbEnd = vbBase + kVbRing / 2;
  g_ring.ibOffset = ibBase; g_ring.ibEnd = ibBase + kIbRing / 2;
  g_ring.uboOffset = uboBase; g_ring.uboEnd = uboBase + kUboRing / 2;
  // Window 0 of the cbuffer ring is a permanently-zero window: every binding a
  // draw does not use points there (dynamic offset 0), so drawRecomp only
  // writes the windows it actually fills instead of zeroing 8 windows per
  // draw. Slot 0's usable range starts after it; nothing ever writes it again.
  if (g_ring.uboMap) {
    static bool zeroWindowInit = false;
    if (!zeroWindowInit) {
      zeroWindowInit = true;
      std::memset(g_ring.uboMap, 0, kCbufWindow);
    }
    if (g_frame.slotIdx == 0)
      g_ring.uboOffset =
          (kCbufWindow + g_ring.uboAlign - 1) & ~(VkDeviceSize)(g_ring.uboAlign - 1);
  }
  g_region.curRt = 0;
  g_region.curDepth = 0;
  g_region.open = false;
  g_region.lastRt = 0;
  g_region.busiestRt = 0;
  g_region.busiestRtDraws = 0;
  g_frame.hadRoom = false;
  g_frame.roomBake = false;
  for (auto &kv : g_rts) {
    kv.second.usedThisFrame = false;
    kv.second.draws = 0;
    // An orphaned lazy clear must not wipe persistent content when an unrelated
    // incremental draw touches this RT in a later frame.
    kv.second.clearPending = false;
  }
  for (auto &kv : g_depths)
    kv.second.usedThisFrame = false;

  vkResetCommandBuffer(g_frame.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g_frame.cmd, &bi);
  g_frame.recording = true;
}

void endFrame(uint64_t scanoutBase) {
  if (!g_dev.ready || !g_frame.recording) return;
  flushCsWrites();  // bound CS-write staleness for guest CPU readers
  g_frame.recording = false;
  reportFps();
  ScopeNs _t(&g_nsEnd);
  endRegion();  // close any open region

  // Present the scanout RT (the flip buffer); fall back to the last RT rendered.
  uint64_t presentBase = g_rts.count(scanoutBase) ? scanoutBase : g_region.lastRt;
  // Debug: present the busiest RT (the scene) instead of the composited scanout.
  static const bool presentScene = std::getenv("DELTA_GPU_PRESENT_SCENE") != nullptr;
  if (presentScene && g_region.busiestRt) presentBase = g_region.busiestRt;
  static const bool presentFirst =
      std::getenv("DELTA_GPU_PRESENT_FIRST_RT") != nullptr;
  if (presentFirst && g_region.firstRt) presentBase = g_region.firstRt;
  // Debug: present the first RT matching DELTA_GPU_PRESENT_RTW x RTH (inspect a
  // specific render target, e.g. the 832x512 room buffer).
  static const int wantW = [] { const char *e = std::getenv("DELTA_GPU_PRESENT_RTW"); return e ? std::atoi(e) : 0; }();
  static const int wantH = [] { const char *e = std::getenv("DELTA_GPU_PRESENT_RTH"); return e ? std::atoi(e) : 0; }();
  if (wantW && wantH) {
    int best = -1000000;  // pick the FRESHEST match (room buffers cycle addresses)
    for (auto &kv : g_rts)
      if ((int)kv.second.w == wantW && (int)kv.second.h == wantH &&
          kv.second.lastFrame > best) { best = kv.second.lastFrame; presentBase = kv.first; }
  }
  // Debug: present a specific RT by guest address (addresses are stable per build).
  static const uint64_t wantAddr = [] {
    const char *e = std::getenv("DELTA_GPU_PRESENT_ADDR");
    return e ? strtoull(e, nullptr, 0) : 0ull;
  }();
  if (wantAddr && g_rts.count(wantAddr)) presentBase = wantAddr;
  static const bool presentTrace = std::getenv("DELTA_PRESENT_TRACE") != nullptr;
  if (presentTrace)
    std::fprintf(stderr, "[present] f%d scanout=%#lx -> present=%#lx%s\n",
                 g_frame.num, (unsigned long)scanoutBase, (unsigned long)presentBase,
                 (scanoutBase && presentBase == scanoutBase) ? "" : " (fallback lastRt)");
  auto it = g_rts.find(presentBase);

  // Record the presented RT's readback copy into this frame's slot, submit it,
  // and DON'T wait: the (software) GPU rasterizes this frame while the guest
  // emulates the next one. The fence is waited one endFrame later, where the
  // slot's pixels are presented (one frame of latency). DELTA_GPU_SYNC=1
  // restores wait-here (framePipelined()).
  FrameSlot &cur = g_frame.slots[g_frame.slotIdx];
  cur.presentable = false;
  if (it != g_rts.end()) {
    RTarget &rt = it->second;
    ensureReadback(rt.w, rt.h, rt.fmt);
    static const bool clearRedTransfer =
        std::getenv("DELTA_GPU_CLEARRED") != nullptr;
    if (clearRedTransfer) {
      VkAccessFlags srcAccess =
          rt.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
              ? VK_ACCESS_SHADER_READ_BIT
          : rt.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
              ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
          : rt.layout == VK_IMAGE_LAYOUT_GENERAL
              ? VK_ACCESS_SHADER_WRITE_BIT
              : 0;
      imageBarrier(g_frame.cmd, rt.image, rt.layout,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, srcAccess,
                   VK_ACCESS_TRANSFER_WRITE_BIT);
      rt.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
      VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(g_frame.cmd, rt.image, rt.layout, &red, 1, &range);
    }
    const VkAccessFlags presentSrc =
        rt.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            ? VK_ACCESS_TRANSFER_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier(g_frame.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 presentSrc, VK_ACCESS_TRANSFER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {rt.w, rt.h, 1};
    vkCmdCopyImageToBuffer(g_frame.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           g_frame.readback, 1, &copy);
    cur.presentable = true;
    cur.w = rt.w; cur.h = rt.h; cur.fmt = rt.fmt;
  }
  {
    ScopeNs _ts(&g_nsSubmit);
    ScopeNs _tsf(&g_frSubmit);
    const VkResult endResult = vkEndCommandBuffer(g_frame.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_frame.cmd;
    vkResetFences(g_dev.device, 1, &cur.fence);
    const VkResult submitResult = vkQueueSubmit(g_dev.queue, 1, &si, cur.fence);
    if (endResult != VK_SUCCESS || submitResult != VK_SUCCESS)
      std::fprintf(stderr, "[gpuvk] frame submit failed: end=%d submit=%d\n",
                   (int)endResult, (int)submitResult);
  }
  cur.submitted = true;
  cur.frameNum = g_frame.num;
  cur.frameDraws = g_frame.draws;
  cur.frameMaxIdx = g_frame.maxIdx;
  cur.frameHadRoom = g_frame.hadRoom;
  cur.presentBase = presentBase;
  cur.scanoutBase = scanoutBase;
  // ensureReadback may have (re)created the aliased buffer; store it back.
  cur.readback = g_frame.readback;
  cur.readbackMem = g_frame.readbackMem;
  cur.readbackMap = g_frame.readbackMap;
  cur.readbackSize = g_frame.readbackSize;

  // Gameplay latches judge the just-recorded frame's command stream (no pixels
  // involved): sustained room frames with real draw counts, or a huge-index 3D
  // draw, mean a run is underway -- stop the headless autoskip mashing menus.
  static int roomStreak = 0;
  if (g_frame.hadRoom && g_frame.draws > 20 && ++roomStreak >= 4)
    gfx::setInGameplay(true);  // latch fast, before the autoskip re-pauses
  if (g_frame.maxIdx >= 1500)
    gfx::setInGameplay(true);

  // Finish a completed frame: the previous slot when pipelined (its raster ran
  // while this frame recorded), this frame's own when synchronous.
  const uint32_t finishIdx = framePipelined() ? (g_frame.slotIdx ^ 1) : g_frame.slotIdx;
  if (framePipelined()) g_frame.slotIdx ^= 1;
  FrameSlot &fin = g_frame.slots[finishIdx];
  const bool waited = fin.submitted;
  if (fin.submitted) {
    uint64_t _tr0 = nowNs();
    const VkResult finWait =
        vkWaitForFences(g_dev.device, 1, &fin.fence, VK_TRUE, UINT64_MAX);
    if (finWait != VK_SUCCESS) {
      std::fprintf(stderr,
                   "[gpuvk] frame %d fence DEVICE FAULT: wait=%d draws=%u\n",
                   fin.frameNum, (int)finWait, fin.frameDraws);
      reportDeviceFault(g_dev.device);
    }
    uint64_t dt = nowNs() - _tr0;
    g_nsReadback += dt;
    g_frWait += dt;
    fin.submitted = false;
  }
  pushStageSample();
  if (!waited || !fin.presentable) return;

  // Readback transform (DELTA_GPU_FLIP: 0=none 1=Y 2=X 3=XY). Default 0 (none): the
  // y-up (negative-height) viewport already stores render-target content upright, and
  // the scene->scanout copy runs the game's real recompiled shader (which samples that
  // upright content correctly), so the presented image needs no flip. (The old default
  // Y-flip existed only to undo the heuristic composite's upside-down output.)
  static const int flipMode = [] {
    const char *e = std::getenv("DELTA_GPU_FLIP");
    return e ? std::atoi(e) : 0;
  }();
  static std::vector<uint8_t> flipped;
  auto *rb = static_cast<uint8_t *>(fin.readbackMap);
  uint8_t *pixels;
  if (flipMode == 0 && fin.fmt == VK_FORMAT_B8G8R8A8_UNORM) {
    // Common case: the readback is already BGRA8 in presentation order; the
    // consumers below (writePpm/present) read it in place, so skip the 8 MB
    // per-pixel convert-and-copy entirely. reportRtContents (the only other
    // readback-buffer user) runs after the last consumer.
    pixels = rb;
  } else {
    flipped.resize(static_cast<size_t>(fin.w) * fin.h * 4);
    const uint32_t srcBytes = formatBytes(fin.fmt);
    const uint32_t srcStride = fin.w * srcBytes;
    for (uint32_t y = 0; y < fin.h; y++) {
      uint32_t sy = (flipMode & 1) ? (fin.h - 1 - y) : y;
      const uint8_t *srow = rb + static_cast<size_t>(sy) * srcStride;
      uint8_t *drow = flipped.data() + static_cast<size_t>(y) * fin.w * 4;
      for (uint32_t x = 0; x < fin.w; x++) {
        uint32_t sx = (flipMode & 2) ? (fin.w - 1 - x) : x;
        readbackPixelBgra(srow + static_cast<size_t>(sx) * srcBytes,
                          fin.fmt, drow + static_cast<size_t>(x) * 4);
      }
    }
    pixels = flipped.data();
  }
  // Minimal single-shot capture (DELTA_GPU_SNAP=N): write ONE ppm of the presented
  // scanout to <dumpdir>/gpu_snap.ppm at the first drawing frame >= N, then never
  // again. For verifying gfx without the rolling DELTA_GPU_DUMP firehose (hundreds
  // of MB per run). DELTA_GPU_SNAP_ROOM=1 waits for a gameplay-room frame.
  static const int snapAt = [] { const char *e = std::getenv("DELTA_GPU_SNAP"); return e ? std::atoi(e) : 0; }();
  static const bool snapRoom = std::getenv("DELTA_GPU_SNAP_ROOM") != nullptr;
  // Wait for a frame with at least this many draws before capturing, so a busy
  // scene frame is grabbed instead of a sparse HUD/transition frame (e.g. Doom64
  // gameplay where only some frames carry the full level geometry).
  static const int snapMinDraws = [] { const char *e = std::getenv("DELTA_GPU_SNAP_MINDRAWS"); return e ? std::atoi(e) : 0; }();
  // DELTA_GPU_SNAP_MININDICES: require a frame to contain a draw with at least this
  // many indices (3D level geometry, e.g. Doom64 with ~2400-index draws) instead of
  // counting draws -- a level frame can have few draws but huge index counts that a
  // draw-count gate (snapMinDraws/snapBest) misses.
  static const int snapMinIdx = [] { const char *e = std::getenv("DELTA_GPU_SNAP_MININDICES"); return e ? std::atoi(e) : 0; }();
  // DELTA_GPU_SNAP_BEST: instead of capturing the first qualifying frame, keep
  // re-capturing whenever this frame has more draws than any seen so far (after
  // snapAt). The final gpu_snap.ppm is then the busiest frame of the run -- a real
  // scene frame, not a sparse HUD/transition one, without guessing a frame number.
  static const bool snapBest = std::getenv("DELTA_GPU_SNAP_BEST") != nullptr;
  static int snapBestDraws = 0;
  static bool snapped = false;
  bool snapNow = snapAt && fin.frameNum >= snapAt && fin.frameDraws > 0 &&
                 (int)fin.frameDraws >= snapMinDraws &&
                 (int)fin.frameMaxIdx >= snapMinIdx &&
                 (!snapRoom || (fin.frameHadRoom && fin.frameDraws > 20));
  // With a min-indices gate, "best" tracks the largest index count seen (the busiest
  // 3D frame) rather than the draw count.
  if (snapBest && snapMinIdx)
    snapNow = snapNow && (int)fin.frameMaxIdx > snapBestDraws;
  else if (snapBest)
    snapNow = snapNow && (int)fin.frameDraws > snapBestDraws;
  else
    snapNow = snapNow && !snapped;
  if (snapNow) {
    snapBestDraws = snapMinIdx ? (int)fin.frameMaxIdx : (int)fin.frameDraws;
    char p[256]; std::snprintf(p, sizeof p, "%s/gpu_snap.ppm", dumpDir());
    writePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr,
                 "[snap] wrote %s (f%d %ux%u draws=%u rt=%#lx scanout=%#lx)\n",
                 p, fin.frameNum, fin.w, fin.h, fin.frameDraws,
                 (unsigned long)fin.presentBase,
                 (unsigned long)fin.scanoutBase);
    snapped = true;
  }
  // Sequence capture (DELTA_GPU_SNAPSEQ=K): write up to K numbered gameplay-room
  // frames, one every ~250 frames, to seq_NN.ppm. Bounded (K*6MB, cleaned up after);
  // lets a long explore run be inspected for non-start rooms without the firehose.
  static const int snapSeqN = [] { const char *e = std::getenv("DELTA_GPU_SNAPSEQ"); return e ? std::atoi(e) : 0; }();
  static int seqDone = 0, seqLastFrame = -10000;
  if (snapSeqN && seqDone < snapSeqN && fin.frameHadRoom && fin.frameDraws > 20 &&
      fin.frameNum - seqLastFrame >= 250) {
    char p[256]; std::snprintf(p, sizeof p, "%s/seq_%02d.ppm", dumpDir(), seqDone);
    writePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr, "[snapseq] %d -> f%d draws=%u\n", seqDone, fin.frameNum, fin.frameDraws);
    seqDone++; seqLastFrame = fin.frameNum;
  }

  // Deterministic room capture: whenever this frame sampled a room RT, roll the
  // presented image to /tmp/gpu_room.ppm (atomic). The last write is guaranteed a
  // gameplay frame regardless of when the flaky autoskip enters/leaves a run. Skip
  // sparse transition frames (few draws) so the capture is representative gameplay.
  if (g_dump && fin.frameHadRoom && fin.frameDraws > 20) {
    char p[256], tmp[256];
    std::snprintf(p, sizeof(p), "%s/gpu_room.ppm", dumpDir());
    std::snprintf(tmp, sizeof(tmp), "%s/gpu_room.tmp", dumpDir());
    writePpm(tmp, pixels, fin.w, fin.h);
    std::rename(tmp, p);
  }
  if (g_dump && fin.frameNum >= 1000 && fin.frameNum % 2000 == 0 && fin.frameDraws > 0)
    dumpPpm(pixels, fin.w, fin.h);
  // Rolling latest-frame capture (uncapped) so late transitions (menu/gameplay)
  // can be inspected from a long headless run without knowing the frame number.
  static const int latestEvery = [] { const char *e = std::getenv("DELTA_GPU_LATEST_EVERY");
    return e ? std::atoi(e) : 300; }();
  if (g_dump && fin.frameNum % latestEvery == 0 && fin.frameDraws > 0) {
    char latest[256];
    std::snprintf(latest, sizeof(latest), "%s/gpu_latest.ppm", dumpDir());
    writePpm(latest, pixels, fin.w, fin.h);
  }
  if ((g_dump || g_declines) && fin.frameNum % 30 == 0) {
    reportDeclines();
  }
  if (g_dump && fin.frameNum % 200 == 0) {
    std::fprintf(stderr, "[gpuvk] frame %d draws=%u heuristic=%u rt=%#lx %ux%u  scanout=%#lx\n",
                 fin.frameNum, fin.frameDraws, g_frame.heuristic,
                 (unsigned long)fin.presentBase, fin.w, fin.h,
                 (unsigned long)fin.scanoutBase);
    for (auto &kv : g_rts)
      if (kv.second.usedThisFrame)
        std::fprintf(stderr, "[gpuvk]    RT %#lx %ux%u draws=%u%s\n",
                     (unsigned long)kv.first, kv.second.w, kv.second.h,
                     kv.second.draws,
                     kv.first == fin.scanoutBase ? " <-SCANOUT" : "");
  }
  // Perf overlay, drawn into the presented buffer only -- the PPM capture
  // paths above already consumed `pixels`, so dumps stay clean.
  drawPerfOverlay(pixels, fin.w, fin.h);
  // DELTA_GPU_OVERLAY_DUMP: one post-overlay ppm (visual check of the overlay
  // itself, which the clean capture paths above deliberately exclude).
  static const bool overlayDump = std::getenv("DELTA_GPU_OVERLAY_DUMP") != nullptr;
  static bool overlayDumped = false;
  if (overlayDump && !overlayDumped && fin.frameNum >= 600) {
    overlayDumped = true;
    char p[256];
    std::snprintf(p, sizeof p, "%s/gpu_overlay.ppm", dumpDir());
    writePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr, "[overlay] wrote %s\n", p);
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
  if (!noPresent) {
    ScopeNs _tp(&g_nsPresent);
    ScopeNs _tpf(&g_frPresent);
    static const bool syncPresent = [] {
      const char *e = std::getenv("DELTA_GPU_SYNCPRESENT");
      return e && e[0] && e[0] != '0';
    }();
    if (syncPresent) {
      if (gfx::ensure("prosperity", fin.w, fin.h) && gfx::pumpEvents())
        gfx::present(pixels, fin.w, fin.h, fin.w * 4, gfx::PixelFormat::bgra8);
    } else {
      presentAsync(pixels, fin.w, fin.h);
    }
  }

  // Runs last: reuses (and clobbers) the readback buffer the present path
  // above already consumed.
  reportRtContents();

  if (rdocFrame() && rdocApi() && rdocApi()->IsFrameCapturing()) {
    uint32_t ok = rdocApi()->EndFrameCapture(rdocDevice(), nullptr);
    std::fprintf(stderr, "[rdoc] capture %s\n", ok ? "written" : "FAILED");
  }
}

}  // namespace gpu::rhi
