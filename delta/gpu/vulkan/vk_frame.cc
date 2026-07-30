/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_frame.h"

#include "gfx/gfx.h"
#include "gpu/gcn/gcn_translate.h"
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
#include "gpu/vulkan/vk_upload_ring.h"

#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
  static const bool sync = [] {
    const char* e = std::getenv("DELTA_GPU_SYNC");
    return (e && e[0] && e[0] != '0') ||
           std::getenv("DELTA_GPU_RTSTAT") != nullptr;
  }();
  return !sync;
}

void EnsureReadback(uint32_t w, uint32_t h, VkFormat fmt) {
  VkDeviceSize need = (VkDeviceSize)w * h * FormatBytes(fmt);
  if (g_frame.readback && need <= g_frame.readback_size)
    return;
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
  static const int f = [] {
    const char* e = std::getenv("DELTA_RDOC_FRAME");
    return e ? std::atoi(e) : 0;
  }();
  return f;
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
  static const bool enabled = std::getenv("DELTA_GPU_RTSTAT") != nullptr;
  static const bool dump = std::getenv("DELTA_GPU_RTDUMP") != nullptr;
  static const bool all = std::getenv("DELTA_GPU_RTSTAT_ALL") != nullptr;
  static const int kReportFrame = [] {
    const char* e = std::getenv("DELTA_GPU_RTSTAT_FRAME");
    return e ? std::atoi(e) : 0;
  }();
  // DELTA_GPU_RTSTAT_EVERY=<n>: sample every n frames instead of every 200, so
  // a per-frame flicker can be told apart from a slow animation.
  static const int every = [] {
    const char* e = std::getenv("DELTA_GPU_RTSTAT_EVERY");
    return e ? std::max(1, std::atoi(e)) : 200;
  }();
  if (!enabled ||
      (kReportFrame ? g_frame.num != kReportFrame : g_frame.num % every != 0))
    return true;
  int reported = 0;
  for (auto& kv : g_rts) {
    RTarget& rt = kv.second;
    if ((!rt.used_this_frame && !(all && rt.ever_rendered)) || reported >= 32)
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
    const uint32_t* px = static_cast<const uint32_t*>(g_frame.readback_map);
    const uint64_t n = static_cast<uint64_t>(rt.w) * rt.h;
    const uint64_t step = n > 16384 ? n / 16384 : 1;
    uint64_t nz = 0, rgb_nz = 0, samples = 0, luma_sum = 0, nan_half = 0;
    uint32_t distinct[4] = {};
    uint32_t num_distinct = 0;
    for (uint64_t i = 0; i < n; i += step, samples++) {
      const uint32_t v = px[i];
      // Mean brightness of the sampled grid: a count of non-zero pixels cannot
      // show a target drifting brighter frame over frame, which is what a
      // runaway exposure or an accumulating pass looks like.
      luma_sum += ((v & 0xFF) + ((v >> 8) & 0xFF) + ((v >> 16) & 0xFF)) / 3;
      if (v)
        nz++;
      if (v & 0x00FFFFFFu)
        rgb_nz++;  // ignores an opaque-black alpha channel
      // A half-float target poisoned with NaN reads back as black through
      // every later unorm pass, so count NaN halves separately from "bright".
      if (rt.fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
        for (uint32_t half = 0; half < 2; half++) {
          const uint32_t h = (v >> (16 * half)) & 0xFFFFu;
          if ((h & 0x7C00u) == 0x7C00u && (h & 0x03FFu))
            nan_half++;
        }
      }
      bool seen = false;
      for (uint32_t k = 0; k < num_distinct; k++)
        seen |= distinct[k] == v;
      if (!seen && num_distinct < 4)
        distinct[num_distinct++] = v;
    }
    std::fprintf(
        stderr,
        "[rtstat] f%d RT %#lx %ux%u draws=%u nz=%lu rgbnz=%lu/%lu "
        "mean=%lu nan=%lu vs=%#lx ps=%#lx cb=%#x rb=%#x "
        "vals=%08x %08x %08x %08x\n",
        g_frame.num, (unsigned long)kv.first, rt.w, rt.h, rt.draws,
        (unsigned long)nz, (unsigned long)rgb_nz, (unsigned long)samples,
        (unsigned long)(samples ? luma_sum / samples : 0),
        (unsigned long)nan_half, (unsigned long)rt.last_vs,
        (unsigned long)rt.last_ps, rt.last_cbuf_mask, rt.last_rawbuf_mask,
        distinct[0], distinct[1], distinct[2], distinct[3]);
    // DELTA_GPU_RTSTAT_DIS: disassemble the pixel shader that produced a
    // NaN-poisoned half-float target. Guest shader addresses differ from run to
    // run, so the shader has to be named in the same run that observed the NaN.
    static const bool kNanDis = std::getenv("DELTA_GPU_RTSTAT_DIS") != nullptr;
    if (kNanDis && nan_half && rt.last_ps) {
      static std::vector<uint64_t> seen;
      if (std::find(seen.begin(), seen.end(), rt.last_ps) == seen.end()) {
        seen.push_back(rt.last_ps);
        gcn::DisassembleAt(rt.last_ps, "nan.PS");
      }
    }
    if (dump) {
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
      static const bool kDumpRaw =
          std::getenv("DELTA_GPU_RTDUMP_RAW") != nullptr;
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
  g_frame.draws = 0;
  g_frame.heuristic = 0;
  g_frame.max_idx = 0;
  g_frame.num++;
  // DELTA_GPU_FORCECLEAR=<rt address>: clear that target at the top of every
  // frame. Diagnostic for a target the title clears by a means we do not see --
  // additive passes into it otherwise accumulate frame over frame.
  static const uint64_t kForceClear = [] {
    const char* e = std::getenv("DELTA_GPU_FORCECLEAR");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  if (kForceClear) {
    auto it = g_rts.find(kForceClear);
    if (it != g_rts.end())
      it->second.clear_pending = true;
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
  const VkDeviceSize vb_base = g_frame.slot_idx * (kVbRing / 2);
  const VkDeviceSize ib_base = g_frame.slot_idx * (kIbRing / 2);
  const VkDeviceSize ubo_base = g_frame.slot_idx * (kUboRing / 2);
  g_ring.vb_offset = vb_base;
  g_ring.vb_end = vb_base + kVbRing / 2;
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
  }
  for (auto& kv : g_depths)
    kv.second.used_this_frame = false;

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

void EndFrame(Renderer& renderer, uint64_t scanout_base) {
  if (!renderer.available() || !g_frame.recording)
    return;
  // Bound CS-write staleness for guest CPU readers. Only a device fault (the
  // flush nulls renderer.state) is fatal; a range that could not be written
  // back just stays stale, as it always did.
  if (!FlushCsWrites(renderer) && !renderer.available()) {
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
  static const bool kPresentScene =
      std::getenv("DELTA_GPU_PRESENT_SCENE") != nullptr;
  if (kPresentScene && g_region.busiest_rt)
    present_base = g_region.busiest_rt;
  static const bool kPresentFirst =
      std::getenv("DELTA_GPU_PRESENT_FIRST_RT") != nullptr;
  if (kPresentFirst && g_region.first_rt)
    present_base = g_region.first_rt;
  // Debug: present the first RT matching DELTA_GPU_PRESENT_RTW x RTH (inspect a
  // specific render target, e.g. the 832x512 room buffer).
  static const int kWantW = [] {
    const char* e = std::getenv("DELTA_GPU_PRESENT_RTW");
    return e ? std::atoi(e) : 0;
  }();
  static const int kWantH = [] {
    const char* e = std::getenv("DELTA_GPU_PRESENT_RTH");
    return e ? std::atoi(e) : 0;
  }();
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
  static const uint64_t kWantAddr = [] {
    const char* e = std::getenv("DELTA_GPU_PRESENT_ADDR");
    return e ? strtoull(e, nullptr, 0) : 0ull;
  }();
  if (kWantAddr && g_rts.count(kWantAddr))
    present_base = kWantAddr;
  // Debug: present the freshest offscreen target instead of the scanout, to see
  // what a composite pass was actually given. Addresses move between runs, so
  // PRESENT_ADDR cannot name it.
  static const bool kWantOffscreen =
      std::getenv("DELTA_GPU_PRESENT_OFFSCREEN") != nullptr;
  if (kWantOffscreen) {
    int best = -1000000;
    for (auto& kv : g_rts)
      if (kv.first != scanout_base && kv.second.last_frame > best) {
        best = kv.second.last_frame;
        present_base = kv.first;
      }
  }
  static const bool kPresentTrace =
      std::getenv("DELTA_PRESENT_TRACE") != nullptr;
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
  static const bool kNoPresent = std::getenv("DELTA_GPU_NOPRESENT") != nullptr;
  static const bool kNeedsCpuCapture = [] {
    return std::getenv("DELTA_GPU_SNAP") || std::getenv("DELTA_GPU_SNAPSEQ") ||
           std::getenv("DELTA_GPU_OVERLAY_DUMP");
  }();
  const bool present_to_window = !kNoPresent && gfx::canPresent();
  const bool need_scanout = present_to_window || g_dump || kNeedsCpuCapture;
  cur.present_to_window = present_to_window;
  if (it != g_rts.end() && need_scanout) {
    RTarget& rt = it->second;
    EnsureReadback(rt.w, rt.h, rt.fmt);
    static const bool kClearRedTransfer =
        std::getenv("DELTA_GPU_CLEARRED") != nullptr;
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
  for (auto& depth_entry : g_depths)
    depth_entry.second.submitted_layout = depth_entry.second.layout;
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
      if (ok && std::getenv("DELTA_RDOC_EXIT")) {
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
  static const int kFlipMode = [] {
    const char* e = std::getenv("DELTA_GPU_FLIP");
    return e ? std::atoi(e) : 0;
  }();
  static std::vector<uint8_t> flipped;
  auto* rb = static_cast<uint8_t*>(fin.readback_map);
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
  static const int kSnapAt = [] {
    const char* e = std::getenv("DELTA_GPU_SNAP");
    return e ? std::atoi(e) : 0;
  }();
  static const bool kSnapRoom = std::getenv("DELTA_GPU_SNAP_ROOM") != nullptr;
  // Wait for a frame with at least this many draws before capturing, so a busy
  // scene frame is grabbed instead of a sparse HUD/transition frame (e.g.
  // Doom64 gameplay where only some frames carry the full level geometry).
  static const int kSnapMinDraws = [] {
    const char* e = std::getenv("DELTA_GPU_SNAP_MINDRAWS");
    return e ? std::atoi(e) : 0;
  }();
  // DELTA_GPU_SNAP_MININDICES: require a frame to contain a draw with at least
  // this many indices (3D level geometry, e.g. Doom64 with ~2400-index draws)
  // instead of counting draws -- a level frame can have few draws but huge
  // index counts that a draw-count gate (kSnapMinDraws/kSnapBest) misses.
  static const int kSnapMinIdx = [] {
    const char* e = std::getenv("DELTA_GPU_SNAP_MININDICES");
    return e ? std::atoi(e) : 0;
  }();
  // DELTA_GPU_SNAP_BEST: instead of capturing the first qualifying frame, keep
  // re-capturing whenever this frame has more draws than any seen so far (after
  // kSnapAt). The final gpu_snap.ppm is then the busiest frame of the run -- a
  // real scene frame, not a sparse HUD/transition one, without guessing a frame
  // number.
  static const bool kSnapBest = std::getenv("DELTA_GPU_SNAP_BEST") != nullptr;
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
  static const int kSnapSeqN = [] {
    const char* e = std::getenv("DELTA_GPU_SNAPSEQ");
    return e ? std::atoi(e) : 0;
  }();
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
  static const int kLatestEvery = [] {
    const char* e = std::getenv("DELTA_GPU_LATEST_EVERY");
    return e ? std::atoi(e) : 300;
  }();
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
  static const bool kOverlayDump =
      std::getenv("DELTA_GPU_OVERLAY_DUMP") != nullptr;
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
    static const bool kSyncPresent = [] {
      const char* e = std::getenv("DELTA_GPU_SYNCPRESENT");
      return e && e[0] && e[0] != '0';
    }();
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
  if (snapped && std::getenv("DELTA_GPU_SNAP_EXIT")) {
    std::fflush(nullptr);
    std::_Exit(0);
  }

  if (RdocFrame() && GetRdocApi() && GetRdocApi()->IsFrameCapturing()) {
    uint32_t ok = GetRdocApi()->EndFrameCapture(RdocDevice(), nullptr);
    std::fprintf(stderr, "[rdoc] capture %s\n", ok ? "written" : "FAILED");
    if (ok && std::getenv("DELTA_RDOC_EXIT")) {
      std::fflush(nullptr);
      std::_Exit(0);
    }
  }
}

}  // namespace gpu::rhi
