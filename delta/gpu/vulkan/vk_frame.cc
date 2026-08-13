/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_frame.h"

#include "gfx/gfx.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/guest_memory.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_backend.h"
#include "gpu/vulkan/vk_capture.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_draw_recomp.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_perf.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_present.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_trace.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGpuSync, "DELTA_GPU_SYNC", false);
DELTA_OPTION(int, kRdocFrame, "DELTA_RDOC_FRAME", 0);
DELTA_OPTION(bool, kFbDump, "DELTA_GPU_FBDUMP", false);
DELTA_OPTION(uint64_t, kMemWatch, "DELTA_GPU_MEMWATCH", 0);
DELTA_OPTION(bool, kRdocExit, "DELTA_RDOC_EXIT", false);
DELTA_OPTION(int, kReportFrame, "DELTA_GPU_RTSTAT_FRAME", 0);
DELTA_OPTION(int, kRtStatEvery, "DELTA_GPU_RTSTAT_EVERY", 200);
DELTA_OPTION(uint64_t, kForceClear, "DELTA_GPU_FORCECLEAR", 0);
DELTA_OPTION(int, kWantW, "DELTA_GPU_PRESENT_RTW", 0);
DELTA_OPTION(int, kWantH, "DELTA_GPU_PRESENT_RTH", 0);
DELTA_OPTION(uint64_t, kWantAddr, "DELTA_GPU_PRESENT_ADDR", 0);
DELTA_OPTION(int, kFlipMode, "DELTA_GPU_FLIP", 0);
DELTA_OPTION(bool, kCsLazyFlush, "DELTA_GPU_CS_LAZY_FLUSH", false);
DELTA_OPTION(int, kSnapAt, "DELTA_GPU_SNAP", 0);
DELTA_OPTION(int, kSnapMinDraws, "DELTA_GPU_SNAP_MINDRAWS", 0);
DELTA_OPTION(int, kSnapMinIdx, "DELTA_GPU_SNAP_MININDICES", 0);
DELTA_OPTION(int, kSnapSeqN, "DELTA_GPU_SNAPSEQ", 0);
DELTA_OPTION(int, kSnapEvery, "DELTA_GPU_SNAPEVERY", 0);
DELTA_OPTION(int, kSnapEveryMax, "DELTA_GPU_SNAPEVERY_MAX", 40);
DELTA_OPTION(int, kLatestEvery, "DELTA_GPU_LATEST_EVERY", 300);
DELTA_OPTION(bool, kSyncPresent, "DELTA_GPU_SYNCPRESENT", false);
DELTA_OPTION(bool, kSnapExit, "DELTA_GPU_SNAP_EXIT", false);
DELTA_OPTION(bool, kClearRedTransfer, "DELTA_GPU_CLEARRED", false);
DELTA_OPTION(bool, kDumpRaw, "DELTA_GPU_RTDUMP_RAW", false);
DELTA_OPTION(bool, kGpuRtdump, "DELTA_GPU_RTDUMP", false);
DELTA_OPTION(bool, kGpuRtstat, "DELTA_GPU_RTSTAT", false);
// Depth scoring is a SEPARATE knob from RTSTAT on purpose: reading a depth
// image needs its own barriers and submits, and doing that inside the colour
// report perturbed the frame it was reporting on -- the scene stopped
// rendering whenever RTSTAT was on, which silently invalidated every colour
// measurement taken alongside it. Opt in when you want depth, and know that
// the numbers you get come at that cost.
// DELTA_GPU_DBSTAT=1 scores every depth target; =<base> scores only that one.
// Each score costs a synchronous submit and a full-image readback, and this
// runs with the frame already submitted -- doing it for a 2048x2048 shadow
// map alongside everything else is enough to take the guest down with it.
DELTA_OPTION(uint64_t, kGpuDbStat, "DELTA_GPU_DBSTAT", 0);
DELTA_OPTION(bool, kGpuRtstatAll, "DELTA_GPU_RTSTAT_ALL", false);
DELTA_OPTION(bool, kNanDis, "DELTA_GPU_RTSTAT_DIS", false);
DELTA_OPTION(bool, kNoPresent, "DELTA_GPU_NOPRESENT", false);
DELTA_OPTION(bool, kOverlayDump, "DELTA_GPU_OVERLAY_DUMP", false);
DELTA_OPTION(bool, kPresentFirst, "DELTA_GPU_PRESENT_FIRST_RT", false);
DELTA_OPTION(bool, kPresentScene, "DELTA_GPU_PRESENT_SCENE", false);
DELTA_OPTION(bool, kPresentTrace, "DELTA_PRESENT_TRACE", false);
DELTA_OPTION(bool, kSnapBest, "DELTA_GPU_SNAP_BEST", false);
DELTA_OPTION(bool, kSnapRoom, "DELTA_GPU_SNAP_ROOM", false);
DELTA_OPTION(bool, kWantOffscreen, "DELTA_GPU_PRESENT_OFFSCREEN", false);
}  // namespace

namespace gpu::vk {

bool CreateFrameSlots() {
  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  uint32_t slot_index = 0;
  for (auto& slot : g_frame.slots) {
    VKOK(vkAllocateCommandBuffers(g_dev.device, &ca, &slot.cmd));
    VKOK(vkCreateFence(g_dev.device, &fc, nullptr, &slot.fence));
    NameObject(VK_OBJECT_TYPE_COMMAND_BUFFER, (uint64_t)slot.cmd,
               "frame slot %u", slot_index);
    NameObject(VK_OBJECT_TYPE_FENCE, (uint64_t)slot.fence, "frame fence %u",
               slot_index);
    slot_index++;
    if (g_dev.timestamp_valid_bits) {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = 2;
      if (vkCreateQueryPool(g_dev.device, &qi, nullptr, &slot.timestamps) !=
          VK_SUCCESS)
        slot.timestamps = VK_NULL_HANDLE;
    }
  }
  g_frame.cmd = g_frame.slots[0].cmd;
  return true;
}

// Pipelined by default; DELTA_GPU_SYNC=1 restores the submit-and-wait frame.
// DELTA_GPU_RTSTAT also forces sync: its readback reuses the active slot's
// buffer mid-flight, which pipelining would present a frame later.
bool FramePipelined() {
  return !(kGpuSync || kGpuRtstat);
}

void EnsureReadback(uint32_t w, uint32_t h, VkFormat fmt) {
  VkDeviceSize need = (VkDeviceSize)w * h * FormatBytes(fmt);
  if (g_frame.readback && need <= g_frame.readback_size)
    return;
  // DELTA_GPU_RBTRACE: every (re)allocation of the readback buffer, with the
  // map it is replacing. This buffer is per-frame-slot, but EnsureReadback only
  // ever updates the CURRENTLY bound slot's copy of the handles -- so a growth
  // here unmaps a pointer the other slot is still holding.
  static const bool kRbTrace = std::getenv("DELTA_GPU_RBTRACE") != nullptr;
  if (kRbTrace)
    std::fprintf(stderr,
                 "[rb] realloc for %ux%u fmt=%d need=%llu have=%llu "
                 "old_map=%p old_buf=%p\n",
                 w, h, (int)fmt, (unsigned long long)need,
                 (unsigned long long)g_frame.readback_size,
                 g_frame.readback_map, (void*)g_frame.readback);
  vkDeviceWaitIdle(g_dev.device);
  if (g_frame.readback_map)
    vkUnmapMemory(g_dev.device, g_frame.readback_mem);
  if (g_frame.readback)
    vkDestroyBuffer(g_dev.device, g_frame.readback, nullptr);
  if (g_frame.readback_mem)
    vkFreeMemory(g_dev.device, g_frame.readback_mem, nullptr);
  g_frame.readback_size = need;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = need;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(g_dev.device, &bi, nullptr, &g_frame.readback);
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g_dev.device, g_frame.readback, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  // CPU reads this buffer every frame (the flip) -> prefer HOST_CACHED so reads
  // hit cache instead of streaming from write-combined memory (the dominant
  // frame cost).
  ba.memoryTypeIndex = FindMemoryTypePref(
      br.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(g_dev.device, &ba, nullptr, &g_frame.readback_mem);
  vkBindBufferMemory(g_dev.device, g_frame.readback, g_frame.readback_mem, 0);
  vkMapMemory(g_dev.device, g_frame.readback_mem, 0, need, 0,
              &g_frame.readback_map);
}

namespace {

// DELTA_RDOC_FRAME=N: bracket frame N's guest rendering with a RenderDoc
// capture. DELTA_RDOC_EXIT=1 exits once the capture has been written. The guest
// draws run on this device, which owns no swapchain (the compositor presents the
// read-back pixels from its own device), so a capture taken at a present
// boundary only ever catches that final blit. The frame has to be marked
// explicitly, and the instance named, or RenderDoc picks the wrong device.
// RENDERDOC_API_1_0_0 is append-only, so these entry indices hold in every
// version; the ones in between are options and keybind setters we do not use.
struct RdocApi {
  void* entry0[19];
  void (*StartFrameCapture)(void* dev, void* wnd);
  uint32_t (*IsFrameCapturing)();
  uint32_t (*EndFrameCapture)(void* dev, void* wnd);
};

RdocApi* GetRdocApi() {
  static RdocApi* api = []() -> RdocApi* {
    using GetApi = int (*)(uint32_t version, void** out);
    void* lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    auto get = lib ? reinterpret_cast<GetApi>(dlsym(lib, "RENDERDOC_GetAPI"))
                   : nullptr;
    RdocApi* a = nullptr;
    if (!get || get(10000, reinterpret_cast<void**>(&a)) != 1) {
      std::fprintf(stderr, "[rdoc] no capture layer attached\n");
      return nullptr;
    }
    return a;
  }();
  return api;
}

int RdocFrame() {
  return kRdocFrame;
}

// RenderDoc identifies a Vulkan device by its instance's dispatch pointer.
void* RdocDevice() {
  return *reinterpret_cast<void**>(g_dev.instance);
}
// DELTA_GPU_RTSTAT: every 200th frame, read back each render target used this
// frame and report how many sampled texels are non-zero. RTSTAT_FRAME selects a
// single early frame instead. DELTA_GPU_RTDUMP also writes the selected
// targets.
bool ReportRtContents(FrameSlot& owner) {
  // DELTA_GPU_RTSTAT_EVERY=<n>: sample every n frames instead of every 200, so
  // a per-frame flicker can be told apart from a slow animation.
  const int every = std::max(1, kRtStatEvery.get());
  if (!kGpuRtstat ||
      (kReportFrame ? g_frame.num != kReportFrame : g_frame.num % every != 0))
    return true;
  int reported = 0;
  for (auto& kv : g_rts) {
    RTarget& rt = kv.second;
    if ((!rt.used_this_frame && !(kGpuRtstatAll && rt.ever_rendered)) || reported >= 32)
      continue;
    reported++;
    EnsureReadback(rt.w, rt.h, rt.fmt);
    owner.readback = g_frame.readback;
    owner.readback_mem = g_frame.readback_mem;
    owner.readback_map = g_frame.readback_map;
    owner.readback_size = g_frame.readback_size;
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    VkCommandBuffer c = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &c) != VK_SUCCESS)
      return false;
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c, &cbi) != VK_SUCCESS) {
      vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
      return false;
    }
    const VkImageLayout old_layout = rt.layout;
    ImageBarrier(c, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 ColorImageAccess(rt.layout), VK_ACCESS_TRANSFER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {rt.w, rt.h, 1};
    vkCmdCopyImageToBuffer(c, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           g_frame.readback, 1, &copy);
    const VkResult end_result = vkEndCommandBuffer(c);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c;
    VkResult submit_result = end_result;
    if (submit_result == VK_SUCCESS)
      submit_result = vkResetFences(g_dev.device, 1, &g_dev.fence);
    if (submit_result == VK_SUCCESS)
      submit_result = vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence);
    const VkResult wait_result =
        submit_result == VK_SUCCESS
            ? vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE,
                              UINT64_MAX)
            : submit_result;
    if (wait_result != VK_SUCCESS) {
      if (submit_result != VK_SUCCESS) {
        vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
        rt.layout = old_layout;
      }
      std::fprintf(
          stderr, "[rtstat] readback submit failed: end=%d submit=%d wait=%d\n",
          (int)end_result, (int)submit_result, (int)wait_result);
      return false;
    }
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
    // One texel is FormatBytes(fmt) wide, which is 8 for the RGBA16F targets
    // P.T. lights into. Indexing a uint32_t* by TEXEL walked half a texel at a
    // time, so every other sample read (B,A) where it meant (R,G). Alpha on a
    // light volume is its coverage term and sits at 1.0 wherever the volume
    // covers; counting that as luminance is what produced the "P.T. has no
    // midtones" reading -- a 50% pile-up just under 1.0 with an empty midrange,
    // in a buffer that had neither.
    const auto* px8 = static_cast<const uint8_t*>(g_frame.readback_map);
    const uint32_t bpp = std::max(1u, (uint32_t)FormatBytes(rt.fmt));
    const bool is_h4 = rt.fmt == VK_FORMAT_R16G16B16A16_SFLOAT;
    const bool is_h2 = rt.fmt == VK_FORMAT_R16G16_SFLOAT;
    const bool is_half = is_h4 || is_h2;
    const uint32_t chans = is_h4 ? 4u : (is_h2 ? 2u : (bpp >= 4 ? 4u : bpp));
    const uint64_t n = static_cast<uint64_t>(rt.w) * rt.h;
    const uint64_t step = n > 16384 ? n / 16384 : 1;
    uint64_t nz = 0, rgb_nz = 0, samples = 0, nan_half = 0, inf_half = 0,
             hot = 0;
    double luma_sum = 0.0;
    // The HDR magnitude, which the buckets cannot show: they all collapse into
    // ">=1" and a target peaking at 1.5 reads identically to one peaking at
    // 1000. That difference is exactly "this pass is too bright" vs "something
    // downstream amplifies a correct pass".
    float lum_max = 0.f;
    uint32_t max_x = 0, max_y = 0;
    uint64_t hi = 0;  // samples far above any displayable value
    // Where the runaway texels SIT. A contiguous block means a pass is not
    // covering that region and it keeps stale content; scattered means the
    // values are being computed. The two need completely different fixes and
    // no aggregate can tell them apart.
    uint32_t hx0 = 0xffffffffu, hy0 = 0xffffffffu, hx1 = 0, hy1 = 0;
    // Alpha AT the runaway texels. P.T.'s feedback pass divides by (1-alpha),
    // so which band alpha sits in decides whether a texel resets or compounds;
    // a whole-buffer alpha average cannot answer that, only alpha conditioned
    // on the texels that ran away.
    float a_at_max = -1.f;
    double hi_a_sum = 0.0;
    float hi_a_min = 1e30f, hi_a_max = -1e30f;
    std::vector<float> lums;
    uint64_t tone[8] = {};
    uint32_t distinct[4] = {};
    uint32_t num_distinct = 0;
    const auto half_val = [](uint32_t h) -> float {
      const uint32_t e = (h >> 10) & 0x1F, mn = h & 0x3FF;
      float f;
      if (e == 0)
        f = std::ldexp((float)mn, -24);
      else if (e != 0x1F)
        f = std::ldexp(1.f + (float)mn / 1024.f, (int)e - 15);
      else
        f = mn ? std::numeric_limits<float>::quiet_NaN()
               : std::numeric_limits<float>::infinity();
      return (h & 0x8000u) ? -f : f;
    };
    for (uint64_t i = 0; i < n; i += step, samples++) {
      const uint8_t* t = px8 + i * bpp;
      uint32_t v = 0;
      std::memcpy(&v, t, std::min<uint32_t>(bpp, 4));
      // Channel values in display units: [0,1] is the displayable range for a
      // unorm target and for a float one alike.
      float ch[4] = {0.f, 0.f, 0.f, 1.f};
      bool finite = true;
      for (uint32_t c = 0; c < chans; c++) {
        if (is_half) {
          const uint32_t h =
              (uint32_t)t[2 * c] | ((uint32_t)t[2 * c + 1] << 8);
          // Inf and NaN share the exponent; only the mantissa tells them apart,
          // so counting NaN alone reports nothing for a buffer full of Inf --
          // and an Inf in an HDR target reads downstream as a clipped
          // highlight, not as an error.
          if ((h & 0x7C00u) == 0x7C00u) {
            ((h & 0x03FFu) ? nan_half : inf_half)++;
            finite = false;
          }
          ch[c] = half_val(h);
        } else {
          ch[c] = (float)t[c] / 255.f;
        }
      }
      bool any = false, any_rgb = false;
      for (uint32_t c = 0; c < chans; c++) {
        any |= ch[c] != 0.f;
        any_rgb |= c < 3 && ch[c] != 0.f;
      }
      nz += any;
      rgb_nz += any_rgb;  // ignores an opaque-black alpha channel
      // Luminance is RGB only. Alpha is a coverage or fade term on most of
      // these targets and says nothing about how bright the frame looks.
      const float lum = std::max({ch[0], ch[1], ch[2]});
      if (finite) {
        luma_sum += std::min(1.f, std::max(0.f, lum));
        if (lum > lum_max) {
          lum_max = lum;
          max_x = (uint32_t)(i % rt.w);
          max_y = (uint32_t)(i / rt.w);
          a_at_max = ch[3];
        }
        if (lum > 100.f) {
          hi++;
          const uint32_t hx = (uint32_t)(i % rt.w), hy = (uint32_t)(i / rt.w);
          hx0 = std::min(hx0, hx);
          hy0 = std::min(hy0, hy);
          hx1 = std::max(hx1, hx);
          hy1 = std::max(hy1, hy);
          hi_a_sum += ch[3];
          hi_a_min = std::min(hi_a_min, ch[3]);
          hi_a_max = std::max(hi_a_max, ch[3]);
        }
        lums.push_back(lum);
        // Eight buckets, not four: the coarse form showed a "gap" that four
        // buckets cannot distinguish from a lit region that simply sits high.
        int b = 7;
        if (lum < 0.0625f)
          b = 0;
        else if (lum < 0.125f)
          b = 1;
        else if (lum < 0.25f)
          b = 2;
        else if (lum < 0.375f)
          b = 3;
        else if (lum < 0.5f)
          b = 4;
        else if (lum < 0.75f)
          b = 5;
        else if (lum < 1.0f)
          b = 6;
        tone[b]++;
        // Clipped at the top of the displayable range. On a FLOAT target that
        // means strictly above 1.0: a buffer of exact 1.0s is a mask or a
        // normalised term, not a blown highlight, and counting those made an
        // ordinary ambient buffer read as 99.9% saturated. On a UNORM target
        // 1.0 IS the clip -- nothing can exceed it -- so the same test there
        // would report zero however blown the frame is.
        if (is_half ? lum > 1.f : lum >= 1.f)
          hot++;
      }
      bool seen = false;
      for (uint32_t k = 0; k < num_distinct; k++)
        seen |= distinct[k] == v;
      if (!seen && num_distinct < 4)
        distinct[num_distinct++] = v;
    }
    // Same picture for the TARGET itself, from the readback the stats above
    // already made -- so the feedback copy and the image it was copied from can
    // be compared directly, and a defect in one told apart from a defect in the
    // other.
    if (kFbDump && is_h4) {
      char rp2[256];
      std::snprintf(rp2, sizeof(rp2), "%s/rt_%lx.ppm", DumpDir(),
                    (unsigned long)kv.first);
      if (FILE* rf2 = std::fopen(rp2, "wb")) {
        std::fprintf(rf2, "P6\n%u %u\n255\n", rt.w, rt.h);
        for (uint64_t q = 0; q < (uint64_t)rt.w * rt.h; q++) {
          const uint8_t* t8 = px8 + q * bpp;
          float c4[4];
          for (uint32_t k = 0; k < 4; k++)
            c4[k] = half_val((uint32_t)t8[2 * k] |
                             ((uint32_t)t8[2 * k + 1] << 8));
          const float l = std::max({c4[0], c4[1], c4[2]});
          uint8_t px3[3];
          if (l > 100.f) {
            px3[0] = 255; px3[1] = 0; px3[2] = 0;
          } else {
            const float g = l <= 0.f ? 0.f : std::log10(1.f + l * 9.f);
            const int v8 = (int)(std::min(1.f, g) * 255.f);
            px3[0] = px3[1] = px3[2] = (uint8_t)v8;
          }
          std::fwrite(px3, 1, 3, rf2);
        }
        std::fclose(rf2);
      }
    }
    // The FEEDBACK image holds exactly what a self-sampling pass read this
    // frame, mid-frame -- not the end-of-frame state every other number here
    // reports. For P.T.'s un-premultiply pass (ps=0x80b54b4000, out.rgb =
    // rgb/(1-alpha)) that mid-frame alpha is the number that decides whether a
    // texel resets or amplifies, and end-of-frame alpha cannot stand in for it:
    // the 33 light draws run in between and only ever lower it.
    if (rt.feedback_image && is_h4) {
      VkCommandBuffer fc = VK_NULL_HANDLE;
      VkCommandBufferAllocateInfo fa{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      fa.commandPool = g_dev.pool;
      fa.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      fa.commandBufferCount = 1;
      if (vkAllocateCommandBuffers(g_dev.device, &fa, &fc) == VK_SUCCESS) {
        VkCommandBufferBeginInfo fbi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        fbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(fc, &fbi);
        const VkImageLayout fold = rt.feedback_layout;
        ImageBarrier(fc, rt.feedback_image, fold,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy fcp{};
        fcp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        fcp.imageExtent = {rt.w, rt.h, 1};
        vkCmdCopyImageToBuffer(fc, rt.feedback_image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               g_frame.readback, 1, &fcp);
        ImageBarrier(fc, rt.feedback_image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, fold,
                     VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
        if (vkEndCommandBuffer(fc) == VK_SUCCESS) {
          VkSubmitInfo fsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
          fsi.commandBufferCount = 1;
          fsi.pCommandBuffers = &fc;
          if (vkResetFences(g_dev.device, 1, &g_dev.fence) == VK_SUCCESS &&
              vkQueueSubmit(g_dev.queue, 1, &fsi, g_dev.fence) == VK_SUCCESS) {
            vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX);
            uint64_t fhi = 0, below = 0;
            float fa_min = 1e30f, fa_max = -1e30f, fmax = 0.f;
            double fa_sum = 0.0;
            for (uint64_t i = 0; i < n; i += step) {
              const uint8_t* t8 = px8 + i * bpp;
              float c4[4];
              for (uint32_t k = 0; k < 4; k++)
                c4[k] = half_val((uint32_t)t8[2 * k] |
                                 ((uint32_t)t8[2 * k + 1] << 8));
              const float l = std::max({c4[0], c4[1], c4[2]});
              fmax = std::max(fmax, l);
              if (l > 100.f) {
                fhi++;
                fa_sum += c4[3];
                fa_min = std::min(fa_min, c4[3]);
                fa_max = std::max(fa_max, c4[3]);
                if (c4[3] < 0.89990f)
                  below++;
              }
            }
            // AT the texel that ran away in the target. The aggregate above
            // is conditioned on feedback texels over 100, and there are none,
            // so it says nothing about the one texel that matters. This is the
            // number that decides reset (alpha >= 0.89990) vs amplify.
            float at[4] = {0, 0, 0, 0};
            {
              const uint64_t idx =
                  (uint64_t)max_y * rt.w + (uint64_t)max_x;
              if (idx < n) {
                const uint8_t* t8 = px8 + idx * bpp;
                for (uint32_t k = 0; k < 4; k++)
                  at[k] = half_val((uint32_t)t8[2 * k] |
                                   ((uint32_t)t8[2 * k + 1] << 8));
              }
            }
            // A PICTURE of where the runaway texels are. Every aggregate so
            // far (count, bounding box, percentiles) is compatible with several
            // different mechanisms; the arrangement usually is not. Written
            // from the feedback readback, which is already non-perturbing --
            // DELTA_GPU_RTDUMP is not (see the profile).
            if (kFbDump) {
              char fp[256];
              std::snprintf(fp, sizeof(fp), "%s/fb_%lx.ppm", DumpDir(),
                            (unsigned long)kv.first);
              if (FILE* pf = std::fopen(fp, "wb")) {
                std::fprintf(pf, "P6\n%u %u\n255\n", rt.w, rt.h);
                for (uint64_t q = 0; q < (uint64_t)rt.w * rt.h; q++) {
                  const uint8_t* t8 = px8 + q * bpp;
                  float c4[4];
                  for (uint32_t k = 0; k < 4; k++)
                    c4[k] = half_val((uint32_t)t8[2 * k] |
                                     ((uint32_t)t8[2 * k + 1] << 8));
                  const float l = std::max({c4[0], c4[1], c4[2]});
                  uint8_t px3[3];
                  if (l > 100.f) {  // runaway: flag red
                    px3[0] = 255; px3[1] = 0; px3[2] = 0;
                  } else {  // else log-scaled grey so structure is visible
                    const float g = l <= 0.f ? 0.f
                                             : std::log10(1.f + l * 9.f);
                    const int v8 = (int)(std::min(1.f, g) * 255.f);
                    px3[0] = px3[1] = px3[2] = (uint8_t)v8;
                  }
                  std::fwrite(px3, 1, 3, pf);
                }
                std::fclose(pf);
                std::fprintf(stderr, "[rtstat-fb] wrote %s\n", fp);
              }
            }
            std::fprintf(stderr,
                         "[rtstat-fb] %#lx max=%.4g hi100=%lu "
                         "fbA=%.4g/%.4g/%.4g below0.8999=%lu "
                         "at(%u,%u)=%.4g,%.4g,%.4g a=%.4g %s\n",
                         (unsigned long)kv.first, fmax, (unsigned long)fhi,
                         fhi ? fa_min : 0.f,
                         fhi ? fa_sum / (double)fhi : 0.0, fhi ? fa_max : 0.f,
                         (unsigned long)below, max_x, max_y, at[0], at[1],
                         at[2], at[3],
                         at[3] >= 0.89990f ? "RESET" : "AMPLIFY");
          }
        }
        vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &fc);
      }
    }
    // Non-zero bytes in the target's GUEST memory, sampled. A render target's
    // contents live in its VkImage and nothing ever publishes them back, so a
    // title that reads one with the CPU reads whatever was there before. This
    // says whether there is anything there at all.
    uint64_t guest_nz = 0, guest_bytes = 0;
    {
      const uint64_t fp = RtByteSize(rt);
      if (fp && fp <= (64u << 20) && gpu::IsReadableRange(kv.first, fp)) {
        const auto* gp = reinterpret_cast<const uint8_t*>(kv.first);
        const uint64_t gstep = fp > 65536 ? fp / 65536 : 1;
        for (uint64_t k = 0; k < fp; k += gstep, guest_bytes++)
          guest_nz += gp[k] != 0;
      }
    }
    std::sort(lums.begin(), lums.end());
    const auto pct = [&](double q) -> float {
      if (lums.empty())
        return 0.f;
      size_t i = (size_t)(q * (double)(lums.size() - 1));
      return lums[i];
    };
    std::fprintf(
        stderr,
        "[rtstat] f%d RT %#lx %ux%u draws=%u nz=%lu rgbnz=%lu/%lu "
        "mean=%lu tone=%lu/%lu/%lu/%lu/%lu/%lu/%lu/%lu hot=%lu nan=%lu inf=%lu "
        "max=%.4g@%u,%u(a=%.4g) hi100=%lu hibox=%u,%u-%u,%u hiA=%.4g/%.4g/%.4g p99=%.4g p999=%.4g guestnz=%lu/%lu vs=%#lx ps=%#lx "
        "cb=%#x rb=%#x vals=%08x %08x %08x %08x\n",
        g_frame.num, (unsigned long)kv.first, rt.w, rt.h, rt.draws,
        (unsigned long)nz, (unsigned long)rgb_nz, (unsigned long)samples,
        (unsigned long)(samples ? luma_sum / (double)samples * 255.0 : 0.0),
        (unsigned long)tone[0], (unsigned long)tone[1], (unsigned long)tone[2],
        (unsigned long)tone[3], (unsigned long)tone[4], (unsigned long)tone[5],
        (unsigned long)tone[6], (unsigned long)tone[7], (unsigned long)hot,
        (unsigned long)nan_half,
        (unsigned long)inf_half, lum_max, max_x, max_y, a_at_max, (unsigned long)hi,
        hi ? hx0 : 0, hi ? hy0 : 0, hx1, hy1,
        hi ? hi_a_min : 0.f, hi ? hi_a_sum / (double)hi : 0.0,
        hi ? hi_a_max : 0.f, pct(0.99), pct(0.999),
        (unsigned long)guest_nz, (unsigned long)guest_bytes,
        (unsigned long)rt.last_vs,
        (unsigned long)rt.last_ps, rt.last_cbuf_mask, rt.last_rawbuf_mask,
        distinct[0], distinct[1], distinct[2], distinct[3]);
    // DELTA_GPU_RTSTAT_DIS: disassemble the pixel shader that produced a
    // NaN-poisoned half-float target. Guest shader addresses differ from run to
    // run, so the shader has to be named in the same run that observed the NaN.
    if (kNanDis && nan_half && rt.last_ps) {
      static std::vector<uint64_t> seen;
      if (std::find(seen.begin(), seen.end(), rt.last_ps) == seen.end()) {
        seen.push_back(rt.last_ps);
        gcn::DisassembleAt(rt.last_ps, "nan.PS");
      }
    }
    if (kGpuRtdump) {
      std::vector<uint8_t> bgra(n * 4);
      const auto* src = static_cast<const uint8_t*>(g_frame.readback_map);
      const uint32_t src_bytes = FormatBytes(rt.fmt);
      for (uint64_t i = 0; i < n; i++)
        ReadbackPixelBgra(src + i * src_bytes, rt.fmt, bgra.data() + i * 4);
      char path[256];
      std::snprintf(path, sizeof(path), "%s/rt_f%d_%#lx_%ux%u.ppm", DumpDir(),
                    g_frame.num, (unsigned long)kv.first, rt.w, rt.h);
      WritePpm(path, bgra.data(), rt.w, rt.h);
      // DELTA_GPU_RTDUMP_RAW: also write the untouched readback bytes. The
      // PPM conversion clamps HDR targets to 8-bit (a 16F scene at
      // luminance ~1500 dumps as solid white); the raw file keeps the halfs
      // for offline exposure/tonemapping.
      if (kDumpRaw) {
        std::snprintf(path, sizeof(path), "%s/rt_f%d_%#lx_%ux%u_fmt%d.raw",
                      DumpDir(), g_frame.num, (unsigned long)kv.first, rt.w,
                      rt.h, (int)rt.fmt);
        if (std::FILE* rf = std::fopen(path, "wb")) {
          std::fwrite(src, src_bytes, n, rf);
          std::fclose(rf);
        }
      }
    }
  }
  // Depth targets were never scored, only colour ones -- so "does the pass
  // that samples the depth read anything?" had no answer at all, and a whole
  // chain of black post-process targets could not be told apart from a black
  // depth buffer. Read the Z plane back the same way.
  // Score parked variants too: the live one at a base is whichever geometry
  // ran last, which is routinely the small post-process pass rather than the
  // full-resolution one the scene was drawn into.
  std::vector<std::pair<uint64_t, DepthTarget*>> depth_list;
  for (auto& kv : g_depths)
    depth_list.emplace_back(kv.first, &kv.second);
  for (auto& kv : g_depth_variants)
    for (DepthTarget& v : kv.second)
      depth_list.emplace_back(kv.first, &v);
  if (!kGpuDbStat)
    depth_list.clear();
  else if (kGpuDbStat != 1)
    depth_list.erase(
        std::remove_if(depth_list.begin(), depth_list.end(),
                       [](const std::pair<uint64_t, DepthTarget*>& e) {
                         return e.first != (uint64_t)kGpuDbStat;
                       }),
        depth_list.end());
  for (auto& entry : depth_list) {
    DepthTarget& d = *entry.second;
    // Never touch a target still in UNDEFINED: a barrier out of that layout
    // is allowed to DISCARD the image, so reading one to report on it would
    // destroy the very thing being measured -- and it did, every RTSTAT run,
    // which quietly invalidated depth readings taken with it.
    if (!d.image || !d.w || !d.h || d.layout == VK_IMAGE_LAYOUT_UNDEFINED ||
        (!d.used_this_frame && !kGpuRtstatAll))
      continue;
    EnsureReadback(d.w, d.h, VK_FORMAT_R32_SFLOAT);
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    VkCommandBuffer c = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &c) != VK_SUCCESS)
      continue;
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(c, &cbi) != VK_SUCCESS) {
      vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
      continue;
    }
    const VkImageLayout old_layout = d.layout;
    DepthBarrier(c, d.image, d.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT);
    d.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    copy.imageExtent = {d.w, d.h, 1};
    vkCmdCopyImageToBuffer(c, d.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           g_frame.readback, 1, &copy);
    VkSubmitInfo dsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    dsi.commandBufferCount = 1;
    dsi.pCommandBuffers = &c;
    if (vkEndCommandBuffer(c) != VK_SUCCESS ||
        vkResetFences(g_dev.device, 1, &g_dev.fence) != VK_SUCCESS ||
        vkQueueSubmit(g_dev.queue, 1, &dsi, g_dev.fence) != VK_SUCCESS ||
        vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX) !=
            VK_SUCCESS) {
      vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
      d.layout = old_layout;
      continue;
    }
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
    // Put it back where the frame left it: this is a diagnostic, and a
    // diagnostic that moves the pipeline's state is a diagnostic that lies.
    {
      VkCommandBuffer rc2 = VK_NULL_HANDLE;
      if (vkAllocateCommandBuffers(g_dev.device, &ca, &rc2) == VK_SUCCESS) {
        VkCommandBufferBeginInfo rbi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        rbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(rc2, &rbi) == VK_SUCCESS) {
          DepthBarrier(rc2, d.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       old_layout, VK_ACCESS_TRANSFER_READ_BIT,
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
          if (vkEndCommandBuffer(rc2) == VK_SUCCESS) {
            VkSubmitInfo rsi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            rsi.commandBufferCount = 1;
            rsi.pCommandBuffers = &rc2;
            if (vkResetFences(g_dev.device, 1, &g_dev.fence) == VK_SUCCESS &&
                vkQueueSubmit(g_dev.queue, 1, &rsi, g_dev.fence) == VK_SUCCESS)
              vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE,
                              UINT64_MAX);
          }
        }
        vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &rc2);
      }
      d.layout = old_layout;
    }
    const float* z = static_cast<const float*>(g_frame.readback_map);
    const uint64_t n = static_cast<uint64_t>(d.w) * d.h;
    const uint64_t step = n > 16384 ? n / 16384 : 1;
    uint64_t samples = 0, zero = 0, one = 0;
    float lo = 1e30f, hi = -1e30f;
    double sum = 0;
    for (uint64_t i = 0; i < n; i += step, samples++) {
      const float v = z[i];
      zero += v == 0.f;
      one += v == 1.f;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
      sum += v;
    }
    std::fprintf(stderr,
                 "[dbstat] f%d DEPTH %#lx %ux%u min=%g max=%g mean=%g "
                 "zero=%lu/%lu one=%lu\n",
                 g_frame.num, (unsigned long)entry.first, d.w, d.h, lo, hi,
                 samples ? sum / samples : 0.0, (unsigned long)zero,
                 (unsigned long)samples, (unsigned long)one);
    // Under RTDUMP, write the Z plane too: "75% of it is exactly zero" is a
    // number that fits several very different pictures, and which one it is
    // decides where to look next.
    if (kGpuRtdump && hi > lo) {
      std::vector<uint8_t> bgra(n * 4);
      for (uint64_t i = 0; i < n; i++) {
        const float t = (z[i] - lo) / (hi - lo);
        const uint8_t g = static_cast<uint8_t>(
            std::min(255.f, std::max(0.f, t * 255.f)));
        bgra[i * 4 + 0] = g;
        bgra[i * 4 + 1] = g;
        bgra[i * 4 + 2] = g;
        bgra[i * 4 + 3] = 255;
      }
      char path[256];
      std::snprintf(path, sizeof(path), "%s/depth_f%d_%#lx_%ux%u.ppm",
                    DumpDir(), g_frame.num, (unsigned long)entry.first, d.w,
                    d.h);
      WritePpm(path, bgra.data(), d.w, d.h);
    }
  }
  return true;
}

}  // namespace
}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

void BeginFrame(Renderer& renderer) {
  if (!renderer.available())
    return;
  // Objects retired two frames ago are past every in-flight command buffer
  // (see ReleaseRetiredTextures) and safe to destroy now.
  ReleaseRetiredTextures();
  if (!CreatePipeline())
    return;
  CreateTexPipeline();  // best-effort; colored path still works without it
  // Frame debugger (DELTA_GPU_CAPTURE*): armed before the counters reset, so
  // the busy trigger can still see the frame that just ended.
  trace::FrameBegin(g_frame.num + 1);
  g_frame.draws = 0;
  g_frame.heuristic = 0;
  g_frame.max_idx = 0;
  g_frame.num++;
  // DELTA_GPU_FORCECLEAR=<rt address>: clear that target at the top of every
  // frame. Diagnostic for a target the title clears by a means we do not see --
  // additive passes into it otherwise accumulate frame over frame.
  if (kForceClear) {
    auto it = g_rts.find(kForceClear);
    if (it != g_rts.end())
      it->second.clear_pending = true;
      it->second.clear_src = "forceclear-knob";
  }
  if (RdocFrame() && g_frame.num == RdocFrame() && GetRdocApi()) {
    GetRdocApi()->StartFrameCapture(RdocDevice(), nullptr);
    std::fprintf(stderr, "[rdoc] capturing frame %d\n", g_frame.num);
  }
  // Bind the active frame slot: its command buffer + readback aliases, and its
  // half of each host-visible ring (the other half may still be read by the
  // in-flight previous frame).
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  g_frame.cmd = slot.cmd;
  g_frame.readback = slot.readback;
  g_frame.readback_mem = slot.readback_mem;
  g_frame.readback_map = slot.readback_map;
  g_frame.readback_size = slot.readback_size;
  ResetTextureUploads(g_frame.slot_idx);
  // DELTA_GPU_RINGHWM: what the frame that just ended actually consumed from
  // each per-frame upload ring half, against that half's capacity. A draw
  // declined for ring space is invisible in a draw count, so the only way to
  // tell "the ring is the limit" from "the ring is not the limit" is to see the
  // high-water mark next to the cap. Printed before the reset below, so the
  // offsets still hold the finished frame's totals.
  static const bool kRingHwm = std::getenv("DELTA_GPU_RINGHWM") != nullptr;
  if (kRingHwm) {
    static VkDeviceSize prev_vb = 0, prev_ib = 0, prev_ubo = 0, prev_sbo = 0;
    static VkDeviceSize peak_vb = 0, peak_ib = 0, peak_ubo = 0, peak_sbo = 0;
    const VkDeviceSize used_vb = g_ring.vb_offset - prev_vb;
    const VkDeviceSize used_ib = g_ring.ib_offset - prev_ib;
    const VkDeviceSize used_ubo = g_ring.ubo_offset - prev_ubo;
    const VkDeviceSize used_sbo =
        g_ring.sbo_map ? g_ring.sbo_offset - prev_sbo : 0;
    peak_vb = std::max(peak_vb, used_vb);
    peak_ib = std::max(peak_ib, used_ib);
    peak_ubo = std::max(peak_ubo, used_ubo);
    peak_sbo = std::max(peak_sbo, used_sbo);
    if (g_frame.num % 10 == 0)
      std::fprintf(stderr,
                   "[ringhwm] f%d draws=%u vb=%lluK/%lluK(peak %lluK) "
                   "ib=%lluK/%lluK(peak %lluK) ubo=%lluK/%lluK(peak %lluK) "
                   "sbo=%lluK/%lluK(peak %lluK)\n",
                   g_frame.num, g_frame.draws,
                   (unsigned long long)(used_vb >> 10),
                   (unsigned long long)((VbRingBytes() / 2) >> 10),
                   (unsigned long long)(peak_vb >> 10),
                   (unsigned long long)(used_ib >> 10),
                   (unsigned long long)((kIbRing / 2) >> 10),
                   (unsigned long long)(peak_ib >> 10),
                   (unsigned long long)(used_ubo >> 10),
                   (unsigned long long)((kUboRing / 2) >> 10),
                   (unsigned long long)(peak_ubo >> 10),
                   (unsigned long long)(used_sbo >> 10),
                   (unsigned long long)((kSboRing / 2) >> 10),
                   (unsigned long long)(peak_sbo >> 10));
    prev_vb = g_frame.slot_idx * (VbRingBytes() / 2);
    prev_ib = g_frame.slot_idx * (kIbRing / 2);
    prev_ubo = g_frame.slot_idx * (kUboRing / 2);
    prev_sbo = g_frame.slot_idx * (kSboRing / 2);
  }
  const VkDeviceSize vb_base = g_frame.slot_idx * (VbRingBytes() / 2);
  const VkDeviceSize ib_base = g_frame.slot_idx * (kIbRing / 2);
  const VkDeviceSize ubo_base = g_frame.slot_idx * (kUboRing / 2);
  g_ring.vb_offset = vb_base;
  g_ring.vb_end = vb_base + VbRingBytes() / 2;
  g_ring.ib_offset = ib_base;
  g_ring.ib_end = ib_base + kIbRing / 2;
  g_ring.ubo_offset = ubo_base;
  g_ring.ubo_end = ubo_base + kUboRing / 2;
  // Window 0 of the cbuffer ring is a permanently-zero window: every binding a
  // draw does not use points there (dynamic offset 0), so DrawRecomp only
  // writes the windows it actually fills instead of zeroing 8 windows per
  // draw. Slot 0's usable range starts after it; nothing ever writes it again.
  if (g_ring.ubo_map) {
    if (!g_ring.zero_window_initialized) {
      g_ring.zero_window_initialized = true;
      std::memset(g_ring.ubo_map, 0, kCbufWindow);
    }
    if (g_frame.slot_idx == 0)
      g_ring.ubo_offset = (kCbufWindow + g_ring.ubo_align - 1) &
                          ~(VkDeviceSize)(g_ring.ubo_align - 1);
  }
  // Raw-buffer ring: same slot split and same permanently-zero window 0, which
  // is where a binding whose descriptor did not resolve points.
  const VkDeviceSize sbo_base = g_frame.slot_idx * (kSboRing / 2);
  g_ring.sbo_offset = sbo_base;
  g_ring.sbo_end = sbo_base + kSboRing / 2;
  if (g_frame.slot_idx == 0)
    g_ring.sbo_offset = g_ring.sbo_stride;
  g_region.cur_rt = 0;
  g_region.cur_depth = 0;
  g_region.cur_stencil = 0;
  g_region.open = false;
  g_region.last_rt = 0;
  g_region.busiest_rt = 0;
  g_region.busiest_rt_draws = 0;
  g_frame.had_room = false;
  g_frame.room_bake = false;
  for (auto& kv : g_rts) {
    kv.second.used_this_frame = false;
    kv.second.draws = 0;
    // An orphaned lazy clear must not wipe persistent content when an unrelated
    // incremental draw touches this RT in a later frame.
    kv.second.clear_pending = false;
    kv.second.clear_src = "none";
  }
  for (auto& kv : g_depths) {
    kv.second.used_this_frame = false;
    kv.second.stencil_used_this_frame = false;
  }

  vkResetCommandBuffer(g_frame.cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g_frame.cmd, &bi);
  CmdBeginLabel(g_frame.cmd, "frame %llu", (unsigned long long)g_frame.num);
  if (slot.timestamps) {
    vkCmdResetQueryPool(g_frame.cmd, slot.timestamps, 0, 2);
    vkCmdWriteTimestamp(g_frame.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        slot.timestamps, 0);
  }
  g_frame.recording = true;
}

// DELTA_GPU_MEMWATCH=<guest addr>: scan a page there every frame and report the
// first frame it is ever non-zero. A surface that ten passes sample but nothing
// in the command stream writes is either never written at all or written
// through a path the PM4 traces cannot see; only watching the MEMORY over time
// separates those, and a per-frame scan does it without touching the guest's
// mappings.
void WatchGuestMem() {
  if (!kMemWatch)
    return;
  const uint64_t a = kMemWatch.get();
  if (!gpu::IsReadableRange(a, 4096))
    return;
  const auto* p = reinterpret_cast<const uint8_t*>(a);
  uint64_t nz = 0;
  for (uint32_t i = 0; i < 4096; i++)
    nz += p[i] != 0;
  static uint64_t seen_nz = 0;
  static int reported = 0;
  if (nz)
    seen_nz++;
  if (reported < 6 && (nz || (g_frame.num % 500) == 0)) {
    reported++;
    std::fprintf(stderr,
                 "[memwatch] f%d %#lx nonzero=%lu/4096 (frames-nonzero=%lu)\n",
                 g_frame.num, (unsigned long)a, (unsigned long)nz,
                 (unsigned long)seen_nz);
  }
}

void EndFrame(Renderer& renderer, uint64_t scanout_base) {
  WatchGuestMem();
  if (!renderer.available() || !g_frame.recording)
    return;
  // Bound CS-write staleness for guest CPU readers. Only a device fault (the
  // flush nulls renderer.state) is fatal; a range that could not be written
  // back just stays stale, as it always did.
  //
  // DELTA_GPU_CS_LAZY_FLUSH=1 drops this blanket flush and leaves the writeback
  // to the targeted FlushCsWritesRange calls the guest readers already make
  // (textures, vertex/index data, constant buffers, CP DMA sources). It exists
  // because executing SotC's async compute means whole 4 MiB material arenas
  // are written back EVERY frame whether anything reads them or not -- 80 ms a
  // frame of memcpy in a 1.3 fps frame. The risk it takes is a guest CPU read
  // that goes through none of those hooks seeing a stale range.
  if (!kCsLazyFlush && !FlushCsWrites(renderer) && !renderer.available()) {
    g_frame.recording = false;
    return;
  }
  g_frame.recording = false;
  ReportFps();
  ScopeNs end_timer(&g_ns_end);
  EndRegion();  // close any open region

  // Present the scanout RT (the flip buffer); fall back to the last RT
  // rendered.
  uint64_t present_base =
      g_rts.count(scanout_base) ? scanout_base : g_region.last_rt;
  // Debug: present the busiest RT (the scene) instead of the composited
  // scanout.
  if (kPresentScene && g_region.busiest_rt)
    present_base = g_region.busiest_rt;
  if (kPresentFirst && g_region.first_rt)
    present_base = g_region.first_rt;
  // Debug: present the first RT matching DELTA_GPU_PRESENT_RTW x RTH (inspect a
  // specific render target, e.g. the 832x512 room buffer).
  if (kWantW && kWantH) {
    int best =
        -1000000;  // pick the FRESHEST match (room buffers cycle addresses)
    for (auto& kv : g_rts)
      if ((int)kv.second.w == kWantW && (int)kv.second.h == kWantH &&
          kv.second.last_frame > best) {
        best = kv.second.last_frame;
        present_base = kv.first;
      }
  }
  // Debug: present a specific RT by guest address (addresses are stable per
  // build).
  if (kWantAddr && g_rts.count(kWantAddr))
    present_base = kWantAddr;
  // Debug: present the freshest offscreen target instead of the scanout, to see
  // what a composite pass was actually given. Addresses move between runs, so
  // PRESENT_ADDR cannot name it.
  if (kWantOffscreen) {
    int best = -1000000;
    for (auto& kv : g_rts)
      if (kv.first != scanout_base && kv.second.last_frame > best) {
        best = kv.second.last_frame;
        present_base = kv.first;
      }
  }
  if (kPresentTrace)
    std::fprintf(
        stderr, "[present] f%d scanout=%#lx -> present=%#lx%s\n", g_frame.num,
        (unsigned long)scanout_base, (unsigned long)present_base,
        (scanout_base && present_base == scanout_base) ? ""
                                                       : " (fallback last_rt)");
  auto it = g_rts.find(present_base);

  // Record the presented RT's readback copy into this frame's slot, submit it,
  // and DON'T wait: the (software) GPU rasterizes this frame while the guest
  // emulates the next one. The fence is waited one EndFrame later, where the
  // slot's pixels are presented (one frame of latency). DELTA_GPU_SYNC=1
  // restores wait-here (FramePipelined()).
  FrameSlot& cur = g_frame.slots[g_frame.slot_idx];
  cur.presentable = false;
  const bool kNeedsCpuCapture =
      kSnapAt || kSnapSeqN || kSnapEvery || kOverlayDump;
  const bool present_to_window = !kNoPresent && gfx::canPresent();
  const bool need_scanout = present_to_window || g_dump || kNeedsCpuCapture;
  cur.present_to_window = present_to_window;
  if (it != g_rts.end() && need_scanout) {
    RTarget& rt = it->second;
    EnsureReadback(rt.w, rt.h, rt.fmt);
    if (kClearRedTransfer) {
      VkAccessFlags src_access =
          rt.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
              ? VK_ACCESS_SHADER_READ_BIT
          : rt.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
              ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
          : rt.layout == VK_IMAGE_LAYOUT_GENERAL ? VK_ACCESS_SHADER_WRITE_BIT
                                                 : 0;
      ImageBarrier(g_frame.cmd, rt.image, rt.layout,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, src_access,
                   VK_ACCESS_TRANSFER_WRITE_BIT);
      rt.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      VkClearColorValue red{{1.0f, 0.0f, 0.0f, 1.0f}};
      VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vkCmdClearColorImage(g_frame.cmd, rt.image, rt.layout, &red, 1, &range);
    }
    const VkAccessFlags present_src =
        rt.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
            ? VK_ACCESS_TRANSFER_WRITE_BIT
        : rt.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
            ? VK_ACCESS_TRANSFER_READ_BIT
        : rt.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            ? VK_ACCESS_SHADER_READ_BIT
        : rt.layout == VK_IMAGE_LAYOUT_GENERAL
            ? VK_ACCESS_SHADER_WRITE_BIT
            : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    CmdInsertLabel(g_frame.cmd, "present readback rt=%#llx %ux%u",
                   (unsigned long long)present_base, rt.w, rt.h);
    ImageBarrier(g_frame.cmd, rt.image, rt.layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, present_src,
                 VK_ACCESS_TRANSFER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {rt.w, rt.h, 1};
    vkCmdCopyImageToBuffer(g_frame.cmd, rt.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           g_frame.readback, 1, &copy);
    cur.presentable = true;
    cur.w = rt.w;
    cur.h = rt.h;
    cur.fmt = rt.fmt;
  }
  {
    ScopeNs submit_timer(&g_ns_submit);
    ScopeNs frame_submit_timer(&g_fr_submit);
    if (cur.timestamps)
      vkCmdWriteTimestamp(g_frame.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          cur.timestamps, 1);
    CmdEndLabel(g_frame.cmd);  // close the "frame N" scope
    const VkResult end_result = vkEndCommandBuffer(g_frame.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_frame.cmd;
    VkResult submit_result = end_result;
    if (submit_result == VK_SUCCESS)
      submit_result = vkResetFences(g_dev.device, 1, &cur.fence);
    if (submit_result == VK_SUCCESS)
      submit_result = vkQueueSubmit(g_dev.queue, 1, &si, cur.fence);
    if (end_result != VK_SUCCESS || submit_result != VK_SUCCESS)
      std::fprintf(stderr, "[gpuvk] frame submit failed: end=%d submit=%d\n",
                   (int)end_result, (int)submit_result);
    cur.submitted = submit_result == VK_SUCCESS;
  }
  if (!cur.submitted) {
    renderer.state = nullptr;
    return;
  }
  // Stamp the layout each color target will hold once this submission
  // executes: the anchor a mid-frame readback (the compute path staging an
  // RT-backed CS input) chains its barriers from.
  for (auto& rt_entry : g_rts)
    rt_entry.second.submitted_layout = rt_entry.second.layout;
  for (auto& depth_entry : g_depths) {
    depth_entry.second.submitted_layout = depth_entry.second.layout;
    depth_entry.second.submitted_stencil_layout =
        depth_entry.second.stencil_layout;
  }
  cur.frame_num = g_frame.num;
  cur.frame_draws = g_frame.draws;
  cur.frame_max_idx = g_frame.max_idx;
  cur.frame_had_room = g_frame.had_room;
  cur.present_base = present_base;
  cur.scanout_base = scanout_base;
  // EnsureReadback may have (re)created the aliased buffer; store it back.
  cur.readback = g_frame.readback;
  cur.readback_mem = g_frame.readback_mem;
  cur.readback_map = g_frame.readback_map;
  cur.readback_size = g_frame.readback_size;

  // Close an armed capture: this frame is submitted, so its mid-frame
  // snapshots and its final targets can be read back and written out.
  trace::FrameEnd(scanout_base);

  // Gameplay latches judge the just-recorded frame's command stream (no pixels
  // involved): sustained room frames with real draw counts, or a huge-index 3D
  // draw, mean a run is underway -- stop the headless autoskip mashing menus.
  static int room_streak = 0;
  if (g_frame.had_room && g_frame.draws > 20 && ++room_streak >= 4)
    gfx::setInGameplay(true);  // latch fast, before the autoskip re-pauses
  if (g_frame.max_idx >= 1500)
    gfx::setInGameplay(true);

  // Finish a completed frame: the previous slot when pipelined (its raster ran
  // while this frame recorded), this frame's own when synchronous.
  const uint32_t finish_idx =
      FramePipelined() ? (g_frame.slot_idx ^ 1) : g_frame.slot_idx;
  if (FramePipelined())
    g_frame.slot_idx ^= 1;
  FrameSlot& fin = g_frame.slots[finish_idx];
  const bool waited = fin.submitted;
  if (fin.submitted) {
    uint64_t _tr0 = NowNs();
    const VkResult fin_wait =
        vkWaitForFences(g_dev.device, 1, &fin.fence, VK_TRUE, UINT64_MAX);
    if (fin_wait != VK_SUCCESS) {
      std::fprintf(stderr,
                   "[gpuvk] frame %d fence DEVICE FAULT: wait=%d draws=%u\n",
                   fin.frame_num, (int)fin_wait, fin.frame_draws);
      ReportDeviceFault(g_dev);
      renderer.state = nullptr;
      return;
    }
    uint64_t dt = NowNs() - _tr0;
    g_ns_readback += dt;
    g_fr_wait += dt;
    if (fin.timestamps) {
      uint64_t timestamps[2] = {};
      if (vkGetQueryPoolResults(g_dev.device, fin.timestamps, 0, 2,
                                sizeof(timestamps), timestamps,
                                sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        const uint64_t mask =
            g_dev.timestamp_valid_bits >= 64
                ? UINT64_MAX
                : (uint64_t{1} << g_dev.timestamp_valid_bits) - 1;
        const uint64_t ticks = (timestamps[1] - timestamps[0]) & mask;
        g_ns_gpu_exec += static_cast<uint64_t>(static_cast<double>(ticks) *
                                               g_dev.timestamp_period);
        g_gpu_exec_samples++;
      }
    }
    fin.submitted = false;
  }
  PushStageSample();
  if (waited && (g_dump || kDeclines) && fin.frame_num % 30 == 0)
    ReportDeclines();
  if (!waited || !fin.presentable) {
    if (!ReportRtContents(cur)) {
      renderer.state = nullptr;
      return;
    }
    if (RdocFrame() && GetRdocApi() && GetRdocApi()->IsFrameCapturing()) {
      uint32_t ok = GetRdocApi()->EndFrameCapture(RdocDevice(), nullptr);
      std::fprintf(stderr, "[rdoc] capture %s\n", ok ? "written" : "FAILED");
      if (ok && kRdocExit) {
        std::fflush(nullptr);
        std::_Exit(0);
      }
    }
    return;
  }

  // Readback transform (DELTA_GPU_FLIP: 0=none 1=Y 2=X 3=XY). Default 0 (none):
  // the y-up (negative-height) viewport already stores render-target content
  // upright, and the scene->scanout copy runs the game's real recompiled shader
  // (which samples that upright content correctly), so the presented image
  // needs no flip. (The old default Y-flip existed only to undo the heuristic
  // composite's upside-down output.)
  static std::vector<uint8_t> flipped;
  auto* rb = static_cast<uint8_t*>(fin.readback_map);
  // DELTA_GPU_RBTRACE: whether the bytes this present is about to show are
  // actually non-zero, and which slot's mapping they came from. "A black window"
  // has two very different causes that look identical downstream -- the readback
  // failing, or the target genuinely being black -- and every consumer below
  // (present, WritePpm, SNAP) inherits the answer silently. `nz` separates them
  // in one line. The presented slot's map differing from the currently bound
  // one is NORMAL: presentation runs one frame behind recording.
  static const bool kRbTrace2 = std::getenv("DELTA_GPU_RBTRACE") != nullptr;
  if (kRbTrace2) {
    uint64_t nz = 0, sampled = 0;
    const size_t n = (size_t)fin.w * fin.h * 4;
    // Stride is deliberately not a multiple of 4, so the sample walks all four
    // channels rather than reporting on one of them.
    for (size_t i = 0; rb && i < n; i += 4099, sampled++)
      nz += rb[i] != 0;
    std::fprintf(stderr,
                 "[rb] present f%d nz=%llu/%llu rt=%#lx %ux%u map=%p%s\n",
                 fin.frame_num, (unsigned long long)nz,
                 (unsigned long long)sampled, (unsigned long)fin.present_base,
                 fin.w, fin.h, fin.readback_map,
                 fin.readback_map == g_frame.readback_map ? " (bound)" : "");
  }
  uint8_t* pixels;
  if (kFlipMode == 0 && fin.fmt == VK_FORMAT_B8G8R8A8_UNORM) {
    // Common case: the readback is already BGRA8 in presentation order; the
    // consumers below (WritePpm/present) read it in place, so skip the 8 MB
    // per-pixel convert-and-copy entirely. ReportRtContents (the only other
    // readback-buffer user) runs after the last consumer.
    pixels = rb;
  } else {
    flipped.resize(static_cast<size_t>(fin.w) * fin.h * 4);
    const uint32_t src_bytes = FormatBytes(fin.fmt);
    const uint32_t src_stride = fin.w * src_bytes;
    for (uint32_t y = 0; y < fin.h; y++) {
      uint32_t sy = (kFlipMode & 1) ? (fin.h - 1 - y) : y;
      const uint8_t* srow = rb + static_cast<size_t>(sy) * src_stride;
      uint8_t* drow = flipped.data() + static_cast<size_t>(y) * fin.w * 4;
      for (uint32_t x = 0; x < fin.w; x++) {
        uint32_t sx = (kFlipMode & 2) ? (fin.w - 1 - x) : x;
        ReadbackPixelBgra(srow + static_cast<size_t>(sx) * src_bytes, fin.fmt,
                          drow + static_cast<size_t>(x) * 4);
      }
    }
    pixels = flipped.data();
  }
  // Minimal single-shot capture (DELTA_GPU_SNAP=N): write ONE ppm of the
  // presented scanout to <dumpdir>/gpu_snap.ppm at the first drawing frame >=
  // N, then never again. For verifying gfx without the rolling DELTA_GPU_DUMP
  // firehose (hundreds of MB per run). DELTA_GPU_SNAP_ROOM=1 waits for a
  // gameplay-room frame.
  // Wait for a frame with at least this many draws before capturing, so a busy
  // scene frame is grabbed instead of a sparse HUD/transition frame (e.g.
  // Doom64 gameplay where only some frames carry the full level geometry).
  // DELTA_GPU_SNAP_MININDICES: require a frame to contain a draw with at least
  // this many indices (3D level geometry, e.g. Doom64 with ~2400-index draws)
  // instead of counting draws -- a level frame can have few draws but huge
  // index counts that a draw-count gate (kSnapMinDraws/kSnapBest) misses.
  // DELTA_GPU_SNAP_BEST: instead of capturing the first qualifying frame, keep
  // re-capturing whenever this frame has more draws than any seen so far (after
  // kSnapAt). The final gpu_snap.ppm is then the busiest frame of the run -- a
  // real scene frame, not a sparse HUD/transition one, without guessing a frame
  // number.
  static int snap_best_draws = 0;
  static bool snapped = false;
  bool snap_now = kSnapAt && fin.frame_num >= kSnapAt && fin.frame_draws > 0 &&
                  (int)fin.frame_draws >= kSnapMinDraws &&
                  (int)fin.frame_max_idx >= kSnapMinIdx &&
                  (!kSnapRoom || (fin.frame_had_room && fin.frame_draws > 20));
  // With a min-indices gate, "best" tracks the largest index count seen (the
  // busiest 3D frame) rather than the draw count.
  if (kSnapBest && kSnapMinIdx)
    snap_now = snap_now && (int)fin.frame_max_idx > snap_best_draws;
  else if (kSnapBest)
    snap_now = snap_now && (int)fin.frame_draws > snap_best_draws;
  else
    snap_now = snap_now && !snapped;
  if (snap_now) {
    snap_best_draws =
        kSnapMinIdx ? (int)fin.frame_max_idx : (int)fin.frame_draws;
    char p[256];
    std::snprintf(p, sizeof p, "%s/gpu_snap.ppm", DumpDir());
    WritePpm(p, pixels, fin.w, fin.h);
    if (FILE* alpha = std::fopen("/tmp/gpu_snap_alpha.pgm", "wb")) {
      std::fprintf(alpha, "P5\n%u %u\n255\n", fin.w, fin.h);
      for (uint64_t i = 0; i < static_cast<uint64_t>(fin.w) * fin.h; i++)
        std::fputc(pixels[i * 4 + 3], alpha);
      std::fclose(alpha);
    }
    std::fprintf(
        stderr, "[snap] wrote %s (f%d %ux%u draws=%u rt=%#lx scanout=%#lx)\n",
        p, fin.frame_num, fin.w, fin.h, fin.frame_draws,
        (unsigned long)fin.present_base, (unsigned long)fin.scanout_base);
    snapped = true;
  }
  // Sequence capture (DELTA_GPU_SNAPSEQ=K): write up to K numbered
  // gameplay-room frames, one every ~250 frames, to seq_NN.ppm. Bounded (K*6MB,
  // cleaned up after); lets a long explore run be inspected for non-start rooms
  // without the firehose.
  static int seq_done = 0, seq_last_frame = -10000;
  if (kSnapSeqN && seq_done < kSnapSeqN && fin.frame_had_room &&
      fin.frame_draws > 20 && fin.frame_num - seq_last_frame >= 250) {
    char p[256];
    std::snprintf(p, sizeof p, "%s/seq_%02d.ppm", DumpDir(), seq_done);
    WritePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr, "[snapseq] %d -> f%d draws=%u\n", seq_done,
                 fin.frame_num, fin.frame_draws);
    seq_done++;
    seq_last_frame = fin.frame_num;
  }

  // DELTA_GPU_SNAPEVERY=N: write a numbered ppm every N drawing frames,
  // unconditionally. A filmstrip of one run shows every screen the title passed
  // through (and every transition), which a single best-frame snap cannot.
  // DELTA_GPU_SNAPEVERY_MAX caps how many are written.
  static int every_done = 0, every_last = -1000000;
  if (kSnapEvery && every_done < kSnapEveryMax && fin.frame_draws > 0 &&
      fin.frame_num - every_last >= kSnapEvery) {
    char p[256];
    std::snprintf(p, sizeof(p), "%s/every_%03d.ppm", DumpDir(), every_done);
    WritePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr, "[snapevery] %d -> f%d draws=%u\n", every_done,
                 fin.frame_num, fin.frame_draws);
    every_done++;
    every_last = fin.frame_num;
  }

  // Deterministic room capture: whenever this frame sampled a room RT, roll the
  // presented image to /tmp/gpu_room.ppm (atomic). The last write is guaranteed
  // a gameplay frame regardless of when the flaky autoskip enters/leaves a run.
  // Skip sparse transition frames (few draws) so the capture is representative
  // gameplay.
  if (g_dump && fin.frame_had_room && fin.frame_draws > 20) {
    char p[256], tmp[256];
    std::snprintf(p, sizeof(p), "%s/gpu_room.ppm", DumpDir());
    std::snprintf(tmp, sizeof(tmp), "%s/gpu_room.tmp", DumpDir());
    WritePpm(tmp, pixels, fin.w, fin.h);
    std::rename(tmp, p);
  }
  if (g_dump && fin.frame_num >= 1000 && fin.frame_num % 2000 == 0 &&
      fin.frame_draws > 0)
    DumpPpm(pixels, fin.w, fin.h);
  // Rolling latest-frame capture (uncapped) so late transitions (menu/gameplay)
  // can be inspected from a long headless run without knowing the frame number.
  if (g_dump && fin.frame_num % kLatestEvery == 0 && fin.frame_draws > 0) {
    char latest[256];
    std::snprintf(latest, sizeof(latest), "%s/gpu_latest.ppm", DumpDir());
    WritePpm(latest, pixels, fin.w, fin.h);
  }
  if (g_dump && fin.frame_num % 200 == 0) {
    std::fprintf(
        stderr,
        "[gpuvk] frame %d draws=%u heuristic=%u rt=%#lx %ux%u  scanout=%#lx\n",
        fin.frame_num, fin.frame_draws, g_frame.heuristic,
        (unsigned long)fin.present_base, fin.w, fin.h,
        (unsigned long)fin.scanout_base);
    for (auto& kv : g_rts)
      if (kv.second.used_this_frame)
        std::fprintf(stderr, "[gpuvk]    RT %#lx %ux%u draws=%u%s\n",
                     (unsigned long)kv.first, kv.second.w, kv.second.h,
                     kv.second.draws,
                     kv.first == fin.scanout_base ? " <-SCANOUT" : "");
  }
  // Perf overlay, drawn into the presented buffer only -- the PPM capture
  // paths above already consumed `pixels`, so dumps stay clean.
  DrawPerfOverlay(pixels, fin.w, fin.h);
  // DELTA_GPU_OVERLAY_DUMP: one post-overlay ppm (visual check of the overlay
  // itself, which the clean capture paths above deliberately exclude).
  static bool overlay_dumped = false;
  if (kOverlayDump && !overlay_dumped && fin.frame_num >= 600) {
    overlay_dumped = true;
    char p[256];
    std::snprintf(p, sizeof p, "%s/gpu_overlay.ppm", DumpDir());
    WritePpm(p, pixels, fin.w, fin.h);
    std::fprintf(stderr, "[overlay] wrote %s\n", p);
  }
  // Present the rendered scanout into the window the VideoOut HLE opened. When
  // there is no display (headless) the window was never created, so we skip
  // present and rely on the readback/PPM path. DELTA_GPU_NOPRESENT forces that
  // headless path even on a display.
  // Bring the window up on the first presentable frame: the videoout HLE only
  // creates it from its own scanout-present path, which the GPU (Gnm) title
  // never takes, so the renderer owns window creation here. ensure() is
  // idempotent and runs on this (the presenting) thread.
  if (fin.present_to_window) {
    ScopeNs present_timer(&g_ns_present);
    ScopeNs frame_present_timer(&g_fr_present);
    if (kSyncPresent) {
      if (gfx::ensure("prosperity", fin.w, fin.h) && gfx::pumpEvents())
        gfx::present(pixels, fin.w, fin.h, fin.w * 4, gfx::PixelFormat::bgra8);
    } else if (pixels == flipped.data()) {
      renderer.state->presenter.Present(std::move(flipped), fin.w, fin.h);
    } else {
      renderer.state->presenter.Present(pixels, fin.w, fin.h);
    }
  }

  // Runs last: reuses (and clobbers) the readback buffer the present path
  // above already consumed.
  if (!ReportRtContents(cur)) {
    renderer.state = nullptr;
    return;
  }
  if (snapped && kSnapExit) {
    std::fflush(nullptr);
    std::_Exit(0);
  }

  if (RdocFrame() && GetRdocApi() && GetRdocApi()->IsFrameCapturing()) {
    uint32_t ok = GetRdocApi()->EndFrameCapture(RdocDevice(), nullptr);
    std::fprintf(stderr, "[rdoc] capture %s\n", ok ? "written" : "FAILED");
    if (ok && kRdocExit) {
      std::fflush(nullptr);
      std::_Exit(0);
    }
  }
}

}  // namespace gpu::rhi
