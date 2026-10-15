/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/frame.h"
#include "base/arch.h"
#include "gpu/render/image/bindless.h"
#include "gpu/render/memory/buffer_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/debug/capture.h"
#include "gpu/render/compute/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/draw/draw_recomp.h"
#include "gpu/render/image/guest_format.h"
#include "gpu/render/debug/labels.h"
#include "gpu/render/debug/perf.h"
#include "gpu/render/draw/pipeline_cache.h"
#include "gpu/render/present.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/renderer_state.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/render/debug/trace.h"
#include "gpu/render/memory/upload_ring.h"
#include "host/window.h"

#include <dlfcn.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "gpu/render/debug/gpu_timeline.h"
#include "options/options.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(bool, kGpuSync, "DELTA_GPU_SYNC", false);
DELTA_OPTION(int, kRdocFrame, "DELTA_RDOC_FRAME", 0);
DELTA_OPTION(bool, kFbDump, "DELTA_GPU_FBDUMP", false);
DELTA_OPTION(u64, kMemWatch, "DELTA_GPU_MEMWATCH", 0);
DELTA_OPTION(bool, kRdocExit, "DELTA_RDOC_EXIT", false);
DELTA_OPTION(int, kReportFrame, "DELTA_GPU_RTSTAT_FRAME", 0);
DELTA_OPTION(int, kRtStatEvery, "DELTA_GPU_RTSTAT_EVERY", 200);
DELTA_OPTION(int, kRtStatMax, "DELTA_GPU_RTSTAT_MAX", 32);
DELTA_OPTION(u64, kForceClear, "DELTA_GPU_FORCECLEAR", 0);
DELTA_OPTION(int, kWantW, "DELTA_GPU_PRESENT_RTW", 0);
DELTA_OPTION(int, kWantH, "DELTA_GPU_PRESENT_RTH", 0);
DELTA_OPTION(u64, kWantAddr, "DELTA_GPU_PRESENT_ADDR", 0);
DELTA_OPTION(int, kFlipMode, "DELTA_GPU_FLIP", 0);
DELTA_OPTION(bool, kCsLazyFlush, "DELTA_GPU_CS_LAZY_FLUSH", true);
DELTA_OPTION(int, kSnapAt, "DELTA_GPU_SNAP", 0);
DELTA_OPTION(u32, kLdsDump, "DELTA_GPU_LDSDUMP", 0);
constexpr u32 kLdsPoison = 0xCDCDCDCD;
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
DELTA_OPTION(u64, kGpuDbStat, "DELTA_GPU_DBSTAT", 0);
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

namespace gpu::render {

DELTA_OPTION(bool, kListProf, "DELTA_GPU_LISTPROF", false);

namespace {
base::Vector<rhi::CommandList*> g_free_lists;

rhi::CommandList* AcquireList() {
  if (!g_free_lists.empty()) {
    rhi::CommandList* list = g_free_lists.back();
    g_free_lists.pop_back();
    return list;
  }
  return Device().CreateCommandList();
}
}  // namespace

rhi::Device& Device() {
  return *g_backend.device;
}

rhi::CommandList* BeginImmediate() {
  rhi::CommandList* list = AcquireList();
  if (list) {
    list->SetProfileTag(kListProf ? 3 : 0);
    list->Begin();
  }
  return list;
}

bool EndImmediate(rhi::CommandList* list, const char* why) {
  DELTA_ZONE_DYN(why);
  list->End();
  const bool ok = Device().Wait(Device().Submit(list));
  g_free_lists.push_back(list);
  return ok;
}

namespace {
DELTA_OPTION(bool, kPassProf, "DELTA_GPU_PASSPROF", false);
DELTA_OPTION(bool, kDrawProf, "DELTA_GPU_DRAWPROF", false);
#if defined(DELTA_TRACY)
constexpr bool kTracyBuilt = true;
#else
constexpr bool kTracyBuilt = false;
#endif
constexpr u32 kMaxProfiledPasses = 4096;
u32 g_pass_open = ~0u;
u32 g_pass_draws_at_open = 0;

struct PassTotal {
  double ns = 0;
  u64 passes = 0, draws = 0;
};
base::HashMap<u64, PassTotal> g_pass_totals;
u32 g_pass_frames = 0;

void CollectPassTimes(FrameSlot& slot) {
  if (!slot.pass_timestamps || slot.pass_marks.empty())
    return;
  const u32 n = static_cast<u32>(slot.pass_marks.size());
  base::Vector<u64> stamps(n * 2);
  const bool read =
      Device().ReadTimestamps(slot.pass_timestamps, 0, n * 2, stamps.data());
  for (u32 i = 0; i < n; i++) {
    const FrameSlot::PassMark& mark = slot.pass_marks[i];
    if (!mark.traced)
      continue;
    timeline::Time(mark.query, read ? stamps[i * 2] : 0);
    timeline::Time(mark.query + 1, read ? stamps[i * 2 + 1] : 0);
  }
  if (read && (kPassProf || kDrawProf)) {
    const rhi::Caps& caps = Device().caps();
    const u64 mask = caps.timestamp_bits >= 64
                         ? UINT64_MAX
                         : (u64{1} << caps.timestamp_bits) - 1;
    for (u32 i = 0; i < n; i++) {
      PassTotal& t = g_pass_totals[slot.pass_marks[i].target];
      t.ns += ((stamps[i * 2 + 1] - stamps[i * 2]) & mask) *
              caps.timestamp_period_ns;
      t.passes++;
      t.draws += slot.pass_marks[i].draws;
    }
  }
  slot.pass_marks.clear();
  if (!(kPassProf || kDrawProf) || ++g_pass_frames < 120)
    return;
  base::Vector<base::Pair<u64, PassTotal>> sorted;
  for (const auto& [target, pass] : g_pass_totals)
    sorted.push_back({target, pass});
  base::Sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.second.ns > b.second.ns;
  });
  double total = 0;
  for (const auto& [target, t] : sorted)
    total += t.ns;
  BASE_LOGI("passprof", "{} {:.2f} ms/frame over {} {}",
            kDrawProf ? "draws" : "regions", total / g_pass_frames / 1e6,
            sorted.size(), kDrawProf ? "pixel shaders" : "targets");
  for (size_t i = 0; i < sorted.size() && i < 24; i++) {
    const PassTotal& t = sorted[i].second;
    BASE_LOGI("passprof", "  {:#x} {:.2f} ms/f  passes={:.1f}/f draws={:.0f}/f",
              (unsigned long long)sorted[i].first, t.ns / g_pass_frames / 1e6,
              double(t.passes) / g_pass_frames,
              double(t.draws) / g_pass_frames);
  }
  g_pass_totals.clear();
  g_pass_frames = 0;
}
}  // namespace

void DrawProfBegin(u64 ps) {
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  if (!kDrawProf || !slot.pass_timestamps ||
      slot.pass_marks.size() >= kMaxProfiledPasses)
    return;
  g_pass_open = static_cast<u32>(slot.pass_marks.size());
  g_pass_draws_at_open = g_frame.draws;
  slot.pass_marks.push_back({ps, 0, false, 0});
  g_frame.list->WriteTimestamp(slot.pass_timestamps, g_pass_open * 2, true);
}

void DrawProfEnd() {
  if (!kDrawProf || g_pass_open == ~0u)
    return;
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  slot.pass_marks[g_pass_open].draws = 1;
  g_frame.list->WriteTimestamp(slot.pass_timestamps, g_pass_open * 2 + 1,
                               false);
  g_pass_open = ~0u;
}

void PassProfBegin(u64 target) {
  if (kDrawProf)
    return;
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  if (!slot.pass_timestamps || slot.pass_marks.size() >= kMaxProfiledPasses)
    return;
  const bool traced = timeline::Active();
  if (!kPassProf && !traced)
    return;
  u16 query = 0;
  if (traced) {
    char name[48];
    std::snprintf(name, sizeof(name), "rt %#llx", (unsigned long long)target);
    query = timeline::Begin(name);
  }
  g_pass_open = static_cast<u32>(slot.pass_marks.size());
  g_pass_draws_at_open = g_frame.draws;
  slot.pass_marks.push_back({target, 0, traced, query});
  g_frame.list->WriteTimestamp(slot.pass_timestamps, g_pass_open * 2, true);
}

void PassProfEnd() {
  if (kDrawProf || g_pass_open == ~0u)
    return;
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  FrameSlot::PassMark& mark = slot.pass_marks[g_pass_open];
  mark.draws = g_frame.draws - g_pass_draws_at_open;
  if (mark.traced)
    timeline::End(mark.query);
  g_frame.list->WriteTimestamp(slot.pass_timestamps, g_pass_open * 2 + 1,
                               false);
  g_pass_open = ~0u;
}

bool CreateFrameSlots() {
  const bool timestamps = Device().caps().timestamps;
  for (auto& slot : g_frame.slots) {
    slot.list = AcquireList();
    if (!slot.list)
      return false;
    if (timestamps)
      slot.timestamps = Device().CreateTimestampPool(2);
    if (timestamps && (kPassProf || kTracyBuilt))
      slot.pass_timestamps =
          Device().CreateTimestampPool(kMaxProfiledPasses * 2);
  }
  g_frame.list = g_frame.slots[0].list;
  return true;
}

bool FramePipelined() {
  return !(kGpuSync || kGpuRtstat);
}

void EnsureReadback(u32 w, u32 h, rhi::Format fmt) {
  u64 need = (u64)w * h * FormatBytes(fmt);
  if (g_frame.readback && need <= g_frame.readback_size)
    return;
  static const bool kRbTrace = std::getenv("DELTA_GPU_RBTRACE") != nullptr;
  if (kRbTrace)
    BASE_LOGI("rb",
              "realloc for {}x{} fmt={} need={} have={} old_map={:p} "
              "old_buf={:p}",
              w, h, (int)fmt, (unsigned long long)need,
              (unsigned long long)g_frame.readback_size, g_frame.readback_map,
              static_cast<void*>(g_frame.readback));
  Device().WaitIdle();
  Device().Destroy(g_frame.readback);
  rhi::BufferDesc desc;
  desc.size = need;
  desc.usage = rhi::kBufferCopyDst;
  desc.memory = rhi::MemoryKind::kReadback;
  desc.name = "frame readback";
  g_frame.readback = Device().CreateBuffer(desc);
  g_frame.readback_size = g_frame.readback ? need : 0;
  g_frame.readback_map =
      g_frame.readback ? g_frame.readback->mapped() : nullptr;
}

rhi::Texture* PresentConvertTarget(u32 w, u32 h) {
  static rhi::Texture* image = nullptr;
  static u32 image_w = 0, image_h = 0;
  if (image && image_w == w && image_h == h)
    return image;
  if (image) {
    Device().WaitIdle();
    Device().Destroy(image);
  }
  rhi::TextureDesc desc;
  desc.format = rhi::Format::kBGRA8Unorm;
  desc.width = w;
  desc.height = h;
  desc.usage = rhi::kTextureCopySrc | rhi::kTextureCopyDst;
  desc.name = "present convert";
  image = Device().CreateTexture(desc);
  image_w = image ? w : 0;
  image_h = image ? h : 0;
  return image;
}

bool EnsureSharedImage(FrameSlot& slot, u32 w, u32 h, rhi::Format fmt) {
  const bool bgra =
      fmt == rhi::Format::kBGRA8Unorm || fmt == rhi::Format::kBGRA8Srgb;
  const bool rgba =
      fmt == rhi::Format::kRGBA8Unorm || fmt == rhi::Format::kRGBA8Srgb;
  if (kFlipMode != 0 || kSyncPresent || (!bgra && !rgba) ||
      !Device().caps().memory_export || !g_backend.presenter.SharedWorks() ||
      !host::CanPresentShared(Device().caps().device_uuid))
    return false;
  const rhi::Format want =
      bgra ? rhi::Format::kBGRA8Unorm : rhi::Format::kRGBA8Unorm;
  if (slot.shared && slot.shared->desc().width == w &&
      slot.shared->desc().height == h && slot.shared->desc().format == want)
    return true;
  static u32 next_id = 0;
  if (slot.shared) {
    Device().WaitIdle();
    Device().Destroy(slot.shared);
    close(slot.shared_fd);
    slot.shared = nullptr;
    slot.shared_fd = -1;
  }
  rhi::TextureDesc desc;
  desc.format = want;
  desc.width = w;
  desc.height = h;
  desc.usage =
      rhi::kTextureCopySrc | rhi::kTextureCopyDst | rhi::kTextureExportable;
  desc.name = "present shared";
  slot.shared = Device().CreateTexture(desc);
  if (!slot.shared)
    return false;
  slot.shared_fd = Device().ExportFd(slot.shared, &slot.shared_bytes);
  if (slot.shared_fd < 0) {
    Device().Destroy(slot.shared);
    slot.shared = nullptr;
    return false;
  }
  slot.shared_id = ++next_id;
  BASE_LOGI("present", "shared frame image {} {}x{}", slot.shared_id, w, h);
  return true;
}

namespace {

// NOLINTBEGIN(readability-identifier-naming)
struct RdocApi {
  void* entry0[19];
  void (*StartFrameCapture)(void* dev, void* wnd);
  u32 (*IsFrameCapturing)();
  u32 (*EndFrameCapture)(void* dev, void* wnd);
};
// NOLINTEND(readability-identifier-naming)

RdocApi* GetRdocApi() {
  static RdocApi* api = []() -> RdocApi* {
    using GetApi = int (*)(u32 version, void** out);
    void* lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    auto get = lib ? reinterpret_cast<GetApi>(dlsym(lib, "RENDERDOC_GetAPI"))
                   : nullptr;
    RdocApi* a = nullptr;
    if (!get || get(10000, reinterpret_cast<void**>(&a)) != 1) {
      BASE_LOGI("rdoc", "no capture layer attached");
      return nullptr;
    }
    return a;
  }();
  return api;
}

int RdocFrame() {
  return kRdocFrame;
}

void* RdocDevice() {
  return Device().CaptureHandle();
}
bool ReportRtContents(FrameSlot& owner) {
  if (kLdsDump && g_ring.lds_map &&
      g_frame.num % base::Max(1, kRtStatEvery.get()) == 0) {
    const u32* dw = reinterpret_cast<const u32*>(g_ring.lds_map);
    const u32 n =
        base::Min<u32>(kLdsDump.get(), (u32)(kLdsScratch / sizeof(u32)));
    char line[1024];
    int at = 0;
    u32 nonzero = 0;
    for (u32 i = 0; i < n; i++) {
      if (dw[i] == kLdsPoison)
        continue;
      nonzero++;
      if (at < (int)sizeof(line) - 24)
        at += std::snprintf(line + at, sizeof(line) - at, " %u:%08x", i, dw[i]);
    }
    char marks[512];
    int mat = 0;
    const u32* tr = dw + gpu::gcn::kLdsTraceBase;
    for (u32 i = 0; i + 1 < 1024 && mat < (int)sizeof(marks) - 24; i += 2)
      if (tr[i] != kLdsPoison && tr[i] != 0)
        mat += std::snprintf(marks + mat, sizeof(marks) - mat, " %x/vcc=%x",
                             tr[i] & 0xFFFF, tr[i + 1]);
    marks[mat] = 0;
    const u32* pos = tr + 0x3F0;
    float px[4];
    for (u32 i = 0; i < 4; i++) {
      const u32 raw = pos[i] == kLdsPoison ? 0u : pos[i];
      std::memcpy(&px[i], &raw, 4);
    }
    if (pos[0] != kLdsPoison || pos[3] != kLdsPoison)
      BASE_LOGI("ldsdump", "f{} exported POS0 = {} {} {} {}", g_frame.num,
                px[0], px[1], px[2], px[3]);
    BASE_LOGI("ldsdump", "f{} {} of {} dwords written:{} | ds pcs:{}",
              g_frame.num, nonzero, n, line, marks);
  }
  const int every = base::Max(1, kRtStatEvery.get());
  if (!kGpuRtstat ||
      (kReportFrame ? g_frame.num != kReportFrame : g_frame.num % every != 0))
    return true;
  const size_t max_scored = base::Max(1, kRtStatMax.get());
  base::Vector<base::Pair<u64, RTarget*>> order;
  for (auto& kv : g_rts)
    if (kv.second.used_this_frame || (kGpuRtstatAll && kv.second.ever_rendered))
      order.emplace_back(kv.first, &kv.second);
  base::Sort(order.begin(), order.end());
  if (order.size() > max_scored) {
    char dropped[512] = {};
    int at = 0;
    for (size_t i = max_scored;
         i < order.size() && at < (int)sizeof(dropped) - 16; i++)
      at += std::snprintf(dropped + at, sizeof(dropped) - at, " %#lx",
                          (unsigned long)order[i].first);
    BASE_LOGI("rtstat", "f{} {} targets, scoring {}, not scored:{}",
              g_frame.num, (unsigned)order.size(), (unsigned)max_scored,
              dropped);
    order.resize(max_scored);
  }
  for (auto& kv : order) {
    RTarget& rt = *kv.second;
    EnsureReadback(rt.w, rt.h, rt.fmt);
    owner.readback = g_frame.readback;
    rhi::CommandList* list = BeginImmediate();
    if (!list)
      return false;
    const rhi::TextureState old_layout = rt.layout;
    TransitionImage(list, rt.texture, rt.layout, rhi::TextureState::kCopySrc);
    rhi::BufferTextureCopy copy;
    copy.region.width = rt.w;
    copy.region.height = rt.h;
    list->CopyTextureToBuffer(g_frame.readback, rt.texture, &copy, 1);
    list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessHostRead);
    if (!EndImmediate(list, "imm.rt_report")) {
      rt.layout = old_layout;
      BASE_LOGI("rtstat", "readback submit failed");
      return false;
    }
    const auto* px8 = static_cast<const u8*>(g_frame.readback_map);
    const u32 bpp = base::Max(1u, (u32)FormatBytes(rt.fmt));
    const bool is_h4 = rt.fmt == rhi::Format::kRGBA16Float;
    const bool is_h2 = rt.fmt == rhi::Format::kRG16Float;
    const bool is_half = is_h4 || is_h2;
    const u32 chans = is_h4 ? 4u : (is_h2 ? 2u : (bpp >= 4 ? 4u : bpp));
    const u64 n = static_cast<u64>(rt.w) * rt.h;
    const u64 step = n > 16384 ? n / 16384 : 1;
    u64 nz = 0, rgb_nz = 0, samples = 0, nan_half = 0, inf_half = 0, hot = 0,
        a_nz = 0;
    double a_sum = 0.0;
    double luma_sum = 0.0;
    float lum_max = 0.f;
    u32 max_x = 0, max_y = 0;
    u64 hi = 0;
    u32 hx0 = 0xffffffffu, hy0 = 0xffffffffu, hx1 = 0, hy1 = 0;
    float a_at_max = -1.f;
    double hi_a_sum = 0.0;
    float hi_a_min = 1e30f, hi_a_max = -1e30f;
    base::Vector<float> lums;
    u64 tone[8] = {};
    u32 distinct[4] = {};
    u32 num_distinct = 0;
    const auto half_val = [](u32 h) -> float {
      const u32 e = (h >> 10) & 0x1F, mn = h & 0x3FF;
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
    for (u64 i = 0; i < n; i += step, samples++) {
      const u8* t = px8 + i * bpp;
      u32 v = 0;
      std::memcpy(&v, t, base::Min<u32>(bpp, 4));
      float ch[4] = {0.f, 0.f, 0.f, 1.f};
      bool finite = true;
      for (u32 c = 0; c < chans; c++) {
        if (is_half) {
          const u32 h = (u32)t[2 * c] | ((u32)t[2 * c + 1] << 8);
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
      for (u32 c = 0; c < chans; c++) {
        any |= ch[c] != 0.f;
        any_rgb |= c < 3 && ch[c] != 0.f;
      }
      nz += any;
      rgb_nz += any_rgb;
      a_nz += ch[3] != 0.f;
      a_sum += ch[3];
      const float lum = base::Max({ch[0], ch[1], ch[2]});
      if (finite) {
        luma_sum += base::Min(1.f, base::Max(0.f, lum));
        if (lum > lum_max) {
          lum_max = lum;
          max_x = (u32)(i % rt.w);
          max_y = (u32)(i / rt.w);
          a_at_max = ch[3];
        }
        if (lum > 100.f) {
          hi++;
          const u32 hx = (u32)(i % rt.w), hy = (u32)(i / rt.w);
          hx0 = base::Min(hx0, hx);
          hy0 = base::Min(hy0, hy);
          hx1 = base::Max(hx1, hx);
          hy1 = base::Max(hy1, hy);
          hi_a_sum += ch[3];
          hi_a_min = base::Min(hi_a_min, ch[3]);
          hi_a_max = base::Max(hi_a_max, ch[3]);
        }
        lums.push_back(lum);
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
        if (is_half ? lum > 1.f : lum >= 1.f)
          hot++;
      }
      bool seen = false;
      for (u32 k = 0; k < num_distinct; k++)
        seen |= distinct[k] == v;
      if (!seen && num_distinct < 4)
        distinct[num_distinct++] = v;
    }
    if (kFbDump && is_h4) {
      char rp2[256];
      std::snprintf(rp2, sizeof(rp2), "%s/rt_%lx.ppm", DumpDir(),
                    (unsigned long)kv.first);
      if (FILE* rf2 = std::fopen(rp2, "wb")) {
        std::fprintf(rf2, "P6\n%u %u\n255\n", rt.w, rt.h);
        for (u64 q = 0; q < (u64)rt.w * rt.h; q++) {
          const u8* t8 = px8 + q * bpp;
          float c4[4];
          for (u32 k = 0; k < 4; k++)
            c4[k] = half_val((u32)t8[2 * k] | ((u32)t8[2 * k + 1] << 8));
          const float l = base::Max({c4[0], c4[1], c4[2]});
          u8 px3[3];
          if (l > 100.f) {
            px3[0] = 255;
            px3[1] = 0;
            px3[2] = 0;
          } else {
            const float g = l <= 0.f ? 0.f : std::log10(1.f + l * 9.f);
            const int v8 = (int)(base::Min(1.f, g) * 255.f);
            px3[0] = px3[1] = px3[2] = (u8)v8;
          }
          std::fwrite(px3, 1, 3, rf2);
        }
        std::fclose(rf2);
      }
    }
    if (rt.feedback_texture && is_h4) {
      if (rhi::CommandList* flist = BeginImmediate()) {
        const rhi::TextureState fold = rt.feedback_layout;
        TransitionImage(flist, rt.feedback_texture, rt.feedback_layout,
                        rhi::TextureState::kCopySrc);
        rhi::BufferTextureCopy fcp;
        fcp.region.width = rt.w;
        fcp.region.height = rt.h;
        flist->CopyTextureToBuffer(g_frame.readback, rt.feedback_texture, &fcp,
                                   1);
        TransitionImage(flist, rt.feedback_texture, rt.feedback_layout, fold);
        flist->Barrier(rhi::kAccessCopyWrite, rhi::kAccessHostRead);
        if (EndImmediate(flist, "imm.rt_report")) {
          u64 fhi = 0, below = 0;
          float fa_min = 1e30f, fa_max = -1e30f, fmax = 0.f;
          double fa_sum = 0.0;
          for (u64 i = 0; i < n; i += step) {
            const u8* t8 = px8 + i * bpp;
            float c4[4];
            for (u32 k = 0; k < 4; k++)
              c4[k] = half_val((u32)t8[2 * k] | ((u32)t8[2 * k + 1] << 8));
            const float l = base::Max({c4[0], c4[1], c4[2]});
            fmax = base::Max(fmax, l);
            if (l > 100.f) {
              fhi++;
              fa_sum += c4[3];
              fa_min = base::Min(fa_min, c4[3]);
              fa_max = base::Max(fa_max, c4[3]);
              if (c4[3] < 0.89990f)
                below++;
            }
          }
          float at[4] = {0, 0, 0, 0};
          {
            const u64 idx = (u64)max_y * rt.w + (u64)max_x;
            if (idx < n) {
              const u8* t8 = px8 + idx * bpp;
              for (u32 k = 0; k < 4; k++)
                at[k] = half_val((u32)t8[2 * k] | ((u32)t8[2 * k + 1] << 8));
            }
          }
          if (kFbDump) {
            char fp[256];
            std::snprintf(fp, sizeof(fp), "%s/fb_%lx.ppm", DumpDir(),
                          (unsigned long)kv.first);
            if (FILE* pf = std::fopen(fp, "wb")) {
              std::fprintf(pf, "P6\n%u %u\n255\n", rt.w, rt.h);
              for (u64 q = 0; q < (u64)rt.w * rt.h; q++) {
                const u8* t8 = px8 + q * bpp;
                float c4[4];
                for (u32 k = 0; k < 4; k++)
                  c4[k] = half_val((u32)t8[2 * k] | ((u32)t8[2 * k + 1] << 8));
                const float l = base::Max({c4[0], c4[1], c4[2]});
                u8 px3[3];
                if (l > 100.f) {
                  px3[0] = 255;
                  px3[1] = 0;
                  px3[2] = 0;
                } else {
                  const float g = l <= 0.f ? 0.f : std::log10(1.f + l * 9.f);
                  const int v8 = (int)(base::Min(1.f, g) * 255.f);
                  px3[0] = px3[1] = px3[2] = (u8)v8;
                }
                std::fwrite(px3, 1, 3, pf);
              }
              std::fclose(pf);
              BASE_LOGI("rtstat-fb", "wrote {}", fp);
            }
          }
          BASE_LOGI("rtstat-fb",
                    "{:#x} max={:.4g} hi100={} fbA={:.4g}/{:.4g}/{:.4g} "
                    "below0.8999={} at({},{})={:.4g},{:.4g},{:.4g} "
                    "a={:.4g} {}",
                    (unsigned long)kv.first, fmax, (unsigned long)fhi,
                    fhi ? fa_min : 0.f, fhi ? fa_sum / (double)fhi : 0.0,
                    fhi ? fa_max : 0.f, (unsigned long)below, max_x, max_y,
                    at[0], at[1], at[2], at[3],
                    at[3] >= 0.89990f ? "RESET" : "AMPLIFY");
        }
      }
    }
    u64 guest_nz = 0, guest_bytes = 0;
    {
      const u64 fp = RtByteSize(rt);
      if (fp && fp <= (64u << 20) && gpu::IsReadableRange(kv.first, fp)) {
        const auto* gp = reinterpret_cast<const u8*>(kv.first);
        const u64 gstep = fp > 65536 ? fp / 65536 : 1;
        for (u64 k = 0; k < fp; k += gstep, guest_bytes++)
          guest_nz += gp[k] != 0;
      }
    }
    base::Sort(lums.begin(), lums.end());
    const auto pct = [&](double q) -> float {
      if (lums.empty())
        return 0.f;
      size_t i = (size_t)(q * (double)(lums.size() - 1));
      return lums[i];
    };
    BASE_LOGI(
        "rtstat",
        "f{} RT {:#x} {}x{} draws={} nz={} rgbnz={}/{} anz={} amean={:.4g} "
        "mean={} "
        "tone={}/{}/{}/{}/{}/{}/{}/{} hot={} nan={} inf={} "
        "max={:.4g}@{},{}(a={:.4g}) hi100={} hibox={},{}-{},{} "
        "hiA={:.4g}/{:.4g}/{:.4g} p99={:.4g} p999={:.4g} guestnz={}/{} "
        "vs={:#x} ps={:#x} cb={:#x} rb={:#x} vals={:08x} {:08x} {:08x} "
        "{:08x}",
        g_frame.num, (unsigned long)kv.first, rt.w, rt.h, rt.draws,
        (unsigned long)nz, (unsigned long)rgb_nz, (unsigned long)samples,
        (unsigned long)a_nz, samples ? a_sum / (double)samples : 0.0,
        (unsigned long)(samples ? luma_sum / (double)samples * 255.0 : 0.0),
        (unsigned long)tone[0], (unsigned long)tone[1], (unsigned long)tone[2],
        (unsigned long)tone[3], (unsigned long)tone[4], (unsigned long)tone[5],
        (unsigned long)tone[6], (unsigned long)tone[7], (unsigned long)hot,
        (unsigned long)nan_half, (unsigned long)inf_half, lum_max, max_x, max_y,
        a_at_max, (unsigned long)hi, hi ? hx0 : 0, hi ? hy0 : 0, hx1, hy1,
        hi ? hi_a_min : 0.f, hi ? hi_a_sum / (double)hi : 0.0,
        hi ? hi_a_max : 0.f, pct(0.99), pct(0.999), (unsigned long)guest_nz,
        (unsigned long)guest_bytes, (unsigned long)rt.last_vs,
        (unsigned long)rt.last_ps, rt.last_cbuf_mask, rt.last_rawbuf_mask,
        distinct[0], distinct[1], distinct[2], distinct[3]);
    if (kNanDis && nan_half && rt.last_ps) {
      static base::Vector<u64> seen;
      static const guest::SessionReset reset_seen(
          [] { guest::ResetResource(seen); });
      if (base::Find(seen.begin(), seen.end(), rt.last_ps) == seen.end()) {
        seen.push_back(rt.last_ps);
        gcn::DisassembleAt(rt.last_ps, "nan.PS");
      }
    }
    if (kGpuRtdump) {
      base::Vector<u8> bgra(n * 4);
      const auto* src = static_cast<const u8*>(g_frame.readback_map);
      const u32 src_bytes = FormatBytes(rt.fmt);
      for (u64 i = 0; i < n; i++)
        ReadbackPixelBgra(src + i * src_bytes, rt.fmt, bgra.data() + i * 4);
      char path[256];
      std::snprintf(path, sizeof(path), "%s/rt_f%d_%#lx_%ux%u.ppm", DumpDir(),
                    g_frame.num, (unsigned long)kv.first, rt.w, rt.h);
      WritePpm(path, bgra.data(), rt.w, rt.h);
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
  base::Vector<base::Pair<u64, DepthTarget*>> depth_list;
  for (auto& kv : g_depths)
    depth_list.emplace_back(kv.first, &kv.second);
  for (auto& kv : g_depth_variants)
    for (DepthTarget& v : kv.second)
      depth_list.emplace_back(kv.first, &v);
  if (!kGpuDbStat)
    depth_list.clear();
  else if (kGpuDbStat != 1)
    depth_list.erase(base::RemoveIf(depth_list.begin(), depth_list.end(),
                                    [](const base::Pair<u64, DepthTarget*>& e) {
                                      return e.first != (u64)kGpuDbStat;
                                    }),
                     depth_list.end());
  for (auto& entry : depth_list) {
    DepthTarget& d = *entry.second;
    if (!d.texture || !d.w || !d.h ||
        d.layout == rhi::TextureState::kUndefined ||
        (!d.used_this_frame && !kGpuRtstatAll))
      continue;
    EnsureReadback(d.w, d.h, rhi::Format::kR32Float);
    owner.readback = g_frame.readback;
    rhi::CommandList* list = BeginImmediate();
    if (!list)
      continue;
    const rhi::TextureState old_layout = d.layout;
    TransitionImage(list, d.texture, d.layout, rhi::TextureState::kCopySrc,
                    rhi::kAspectDepth, d.layers);
    rhi::BufferTextureCopy copy;
    copy.region.aspect = rhi::kAspectDepth;
    copy.region.width = d.w;
    copy.region.height = d.h;
    list->CopyTextureToBuffer(g_frame.readback, d.texture, &copy, 1);
    TransitionImage(list, d.texture, d.layout, old_layout, rhi::kAspectDepth,
                    d.layers);
    list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessHostRead);
    if (!EndImmediate(list, "imm.rt_report"))
      continue;
    const float* z = static_cast<const float*>(g_frame.readback_map);
    const u64 n = static_cast<u64>(d.w) * d.h;
    const u64 step = n > 16384 ? n / 16384 : 1;
    u64 samples = 0, zero = 0, one = 0;
    float lo = 1e30f, hi = -1e30f;
    double sum = 0;
    for (u64 i = 0; i < n; i += step, samples++) {
      const float v = z[i];
      zero += v == 0.f;
      one += v == 1.f;
      lo = base::Min(lo, v);
      hi = base::Max(hi, v);
      sum += v;
    }
    BASE_LOGI("dbstat",
              "f{} DEPTH {:#x} {}x{} min={} max={} mean={} zero={}/{} one={}",
              g_frame.num, (unsigned long)entry.first, d.w, d.h, lo, hi,
              samples ? sum / samples : 0.0, (unsigned long)zero,
              (unsigned long)samples, (unsigned long)one);
    if (kGpuRtdump && hi > lo) {
      base::Vector<u8> bgra(n * 4);
      for (u64 i = 0; i < n; i++) {
        const float t = (z[i] - lo) / (hi - lo);
        const u8 g =
            static_cast<u8>(base::Min(255.f, base::Max(0.f, t * 255.f)));
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

void StampSubmittedLayouts() {
  for (auto& rt_entry : g_rts)
    rt_entry.second.submitted_layout = rt_entry.second.layout;
  for (auto& parked : g_rt_variants)
    for (RTarget& v : parked.second)
      v.submitted_layout = v.layout;
  for (auto& depth_entry : g_depths) {
    depth_entry.second.submitted_layout = depth_entry.second.layout;
    depth_entry.second.submitted_stencil_layout =
        depth_entry.second.stencil_layout;
  }
  for (auto& parked : g_depth_variants)
    for (DepthTarget& v : parked.second) {
      v.submitted_layout = v.layout;
      v.submitted_stencil_layout = v.stencil_layout;
    }
}

bool SubmitFrameChunk() {
  if (!g_frame.recording)
    return true;
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  EndRegion();
  CmdEndLabel(g_frame.list);
  g_frame.list->End();
  slot.submission = Device().Submit(g_frame.list);
  slot.chunks.push_back(g_frame.list);
  rhi::CommandList* next = AcquireList();
  if (!next)
    return false;
  next->SetProfileTag(kListProf ? 1 : 0);
  next->Begin();
  slot.list = next;
  g_frame.list = next;
  CsFrameListChanged();
  g_frame.chunk_seq++;
  g_frame.draws_at_chunk = g_frame.draws;
  CmdBeginLabel(g_frame.list, "frame %llu chunk %llu",
                (unsigned long long)g_frame.num,
                (unsigned long long)g_frame.chunk_seq);
  StampSubmittedLayouts();
  return true;
}

}  // namespace gpu::render

namespace gpu::render {

void BeginFrame(Renderer& renderer) {
  DELTA_ZONE("gpu.begin_frame");
  timeline::Refresh();
  if (!renderer.available())
    return;
  if (!g_ring.ubo_bytes)
    g_ring.ubo_bytes = UboRingBytes();
  {
    DELTA_ZONE("gpu.bf_release");
    ReleaseRetiredTextures();
    ReleaseRetiredTargets();
    ReleaseRetiredCsBuffers();
  }
  CsSubmitBatchBeforeFrame();
  if (!CreateTextureDescriptors())
    return;
  trace::FrameBegin(g_frame.num + 1);
  g_frame.draws = 0;
  g_frame.draws_at_chunk = 0;
  g_frame.max_idx = 0;
  g_frame.num++;
  ReleaseDrawBufferBindings();
  if (kForceClear) {
    auto it = g_rts.find(kForceClear);
    if (it != g_rts.end()) {
      it->second.clear_pending = true;
      it->second.clear_src = "forceclear-knob";
    }
  }
  if (RdocFrame() && g_frame.num == RdocFrame() && GetRdocApi()) {
    GetRdocApi()->StartFrameCapture(RdocDevice(), nullptr);
    BASE_LOGI("rdoc", "capturing frame {}", g_frame.num);
  }
  FrameSlot& slot = g_frame.slots[g_frame.slot_idx];
  if (renderer.state) {
    DELTA_ZONE("gpu.bf_wait_present");
    renderer.state->presenter.WaitForBorrowed();
  }
  g_frame.list = slot.list;
  g_frame.readback = slot.readback;
  g_frame.readback_map = slot.readback ? slot.readback->mapped() : nullptr;
  g_frame.readback_size = slot.readback ? slot.readback->desc().size : 0;
  ResetTextureUploads(g_frame.slot_idx);
  static const bool kRingHwm = std::getenv("DELTA_GPU_RINGHWM") != nullptr;
  if (kRingHwm) {
    static u64 prev_vb = 0, prev_ib = 0, prev_ubo = 0, prev_sbo = 0;
    static u64 peak_vb = 0, peak_ib = 0, peak_ubo = 0, peak_sbo = 0;
    const u64 used_vb = g_ring.vb_offset - prev_vb;
    const u64 used_ib = g_ring.ib_offset - prev_ib;
    const u64 used_ubo = g_ring.ubo_map ? g_ring.ubo_offset - prev_ubo : 0;
    const u64 used_sbo = g_ring.sbo_map ? g_ring.sbo_offset - prev_sbo : 0;
    peak_vb = base::Max(peak_vb, used_vb);
    peak_ib = base::Max(peak_ib, used_ib);
    peak_ubo = base::Max(peak_ubo, used_ubo);
    peak_sbo = base::Max(peak_sbo, used_sbo);
    if (g_frame.num % 10 == 0)
      BASE_LOGI("ringhwm",
                "f{} draws={} vb={}K/{}K(peak {}K) ib={}K/{}K(peak {}K) "
                "ubo={}K/{}K(peak {}K) sbo={}K/{}K(peak {}K)",
                g_frame.num, g_frame.draws, (unsigned long long)(used_vb >> 10),
                (unsigned long long)((VbRingBytes() / 2) >> 10),
                (unsigned long long)(peak_vb >> 10),
                (unsigned long long)(used_ib >> 10),
                (unsigned long long)((kIbRing / 2) >> 10),
                (unsigned long long)(peak_ib >> 10),
                (unsigned long long)(used_ubo >> 10),
                (unsigned long long)((UboRingBytes() / 2) >> 10),
                (unsigned long long)(peak_ubo >> 10),
                (unsigned long long)(used_sbo >> 10),
                (unsigned long long)((kSboRing / 2) >> 10),
                (unsigned long long)(peak_sbo >> 10));
    prev_vb = g_frame.slot_idx * (VbRingBytes() / 2);
    prev_ib = g_frame.slot_idx * (kIbRing / 2);
    prev_ubo = g_frame.slot_idx * (UboRingBytes() / 2);
    prev_sbo = g_frame.slot_idx * (kSboRing / 2);
  }
  const u64 vb_base = g_frame.slot_idx * (VbRingBytes() / 2);
  const u64 ib_base = g_frame.slot_idx * (kIbRing / 2);
  const u64 ubo_base = g_frame.slot_idx * (UboRingBytes() / 2);
  g_ring.vb_offset = vb_base;
  g_ring.vb_end = vb_base + VbRingBytes() / 2;
  g_ring.ib_offset = ib_base;
  g_ring.ib_end = ib_base + kIbRing / 2;
  g_ring.ubo_offset = ubo_base + g_ring.ubo_stride;
  g_ring.ubo_end = ubo_base + UboRingBytes() / 2;
  const u64 sbo_base = g_frame.slot_idx * (kSboRing / 2);
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
    kv.second.clear_pending = false;
    kv.second.clear_src = "none";
  }
  for (auto& kv : g_depths) {
    kv.second.used_this_frame = false;
    kv.second.stencil_used_this_frame = false;
  }

  g_frame.list->SetProfileTag(kListProf ? 1 : 0);
  g_frame.list->Begin();
  CmdBeginLabel(g_frame.list, "frame %llu", (unsigned long long)g_frame.num);
  if (g_ring.lds_buf)
    g_frame.list->FillBuffer(g_ring.lds_buf, 0, kLdsScratch,
                             kLdsDump ? kLdsPoison : 0u);
  if (slot.timestamps) {
    g_frame.list->ResetTimestamps(slot.timestamps, 0, 2);
    g_frame.list->WriteTimestamp(slot.timestamps, 0, true);
  }
  if (slot.pass_timestamps) {
    g_frame.list->ResetTimestamps(slot.pass_timestamps, 0,
                                  kMaxProfiledPasses * 2);
    slot.pass_marks.clear();
  }
  g_frame.recording = true;
  bindless::FrameBegin(g_frame.list);
}

void WatchGuestMem() {
  if (!kMemWatch)
    return;
  const u64 a = kMemWatch.get();
  if (!gpu::IsReadableRange(a, 4096))
    return;
  const auto* p = reinterpret_cast<const u8*>(a);
  u64 nz = 0;
  for (u32 i = 0; i < 4096; i++)
    nz += p[i] != 0;
  static u64 seen_nz = 0;
  static int reported = 0;
  if (nz)
    seen_nz++;
  if (reported < 6 && (nz || (g_frame.num % 500) == 0)) {
    reported++;
    BASE_LOGI("memwatch", "f{} {:#x} nonzero={}/4096 (frames-nonzero={})",
              g_frame.num, (unsigned long)a, (unsigned long)nz,
              (unsigned long)seen_nz);
  }
}

namespace {
DELTA_OPTION(u64, kSlowEndMs, "DELTA_GPU_SLOW_END_MS", 300);
struct EndFramePhases {
  u64 start = NowNs(), last = start;
  u64 ns[6] = {};
  void Mark(int phase) {
    const u64 now = NowNs();
    ns[phase] += now - last;
    last = now;
  }
  ~EndFramePhases() {
    if (NowNs() - start < kSlowEndMs * 1'000'000)
      return;
    BASE_LOGW("gpuvk",
              "slow frame end {} ms: tracker {} buffers {} cs-flush {} "
              "record+submit {} gpu-wait {} convert {} present+maintain {}",
              (NowNs() - start) / 1'000'000, ns[0] / 1'000'000,
              ns[1] / 1'000'000, ns[2] / 1'000'000, ns[3] / 1'000'000,
              ns[4] / 1'000'000, ns[5] / 1'000'000,
              (NowNs() - last) / 1'000'000);
  }
};
}  // namespace

void EndFrame(Renderer& renderer, u64 scanout_base) {
  DELTA_ZONE("gpu.end_frame");
  DELTA_FRAME_MARK();
  DELTA_PLOT("gpu.draws", static_cast<int64_t>(g_frame.draws));
  EndFramePhases phases;
  WatchGuestMem();
  GuestWriteTracker().EndFrame();
  phases.Mark(0);
  BufferCacheEndFrame();
  phases.Mark(1);
  if (!renderer.available() || !g_frame.recording)
    return;
  if (!FlushCsWritesFrameEnd(renderer, !kCsLazyFlush) &&
      !renderer.available()) {
    g_frame.recording = false;
    return;
  }
  g_frame.recording = false;
  phases.Mark(2);
  ReportFps();
  ScopeNs end_timer(&g_stats.ns_end);
  EndRegion();

  u64 present_base =
      g_rts.count(scanout_base) ? scanout_base : g_region.last_rt;
  if (kPresentScene && g_region.busiest_rt)
    present_base = g_region.busiest_rt;
  if (kPresentFirst && g_region.first_rt)
    present_base = g_region.first_rt;
  if (kWantW && kWantH) {
    int best = -1000000;
    for (auto& kv : g_rts)
      if ((int)kv.second.w == kWantW && (int)kv.second.h == kWantH &&
          kv.second.last_frame > best) {
        best = kv.second.last_frame;
        present_base = kv.first;
      }
  }
  if (kWantAddr && g_rts.count(kWantAddr))
    present_base = kWantAddr;
  if (kWantOffscreen) {
    int best = -1000000;
    for (auto& kv : g_rts)
      if (kv.first != scanout_base && kv.second.last_frame > best) {
        best = kv.second.last_frame;
        present_base = kv.first;
      }
  }
  if (kPresentTrace)
    BASE_LOGI("present", "f{} scanout={:#x} -> present={:#x}{}", g_frame.num,
              (unsigned long)scanout_base, (unsigned long)present_base,
              (scanout_base && present_base == scanout_base)
                  ? ""
                  : " (fallback last_rt)");
  auto it = g_rts.find(present_base);

  FrameSlot& cur = g_frame.slots[g_frame.slot_idx];
  cur.presentable = false;
  const bool kNeedsCpuCapture =
      kSnapAt || kSnapSeqN || kSnapEvery || kOverlayDump;
  const bool present_to_window = !kNoPresent && host::CanPresent();
  const bool need_scanout = present_to_window || g_dump || kNeedsCpuCapture;
  cur.present_to_window = present_to_window;
  if (it != g_rts.end() && need_scanout) {
    RTarget& rt = it->second;
    rhi::Texture* convert = nullptr;
    if (kFlipMode == 0 && rt.fmt != rhi::Format::kBGRA8Unorm &&
        rt.fmt != rhi::Format::kRGBA8Unorm &&
        rt.fmt != rhi::Format::kBGRA8Srgb &&
        rt.fmt != rhi::Format::kRGBA8Srgb &&
        Device().SupportsBlit(rt.fmt, rhi::Format::kBGRA8Unorm))
      convert = PresentConvertTarget(rt.w, rt.h);
    const rhi::Format read_fmt = convert ? rhi::Format::kBGRA8Unorm : rt.fmt;
    const bool shared = present_to_window && !g_dump && !kNeedsCpuCapture &&
                        EnsureSharedImage(cur, rt.w, rt.h, read_fmt);
    if (!shared)
      EnsureReadback(rt.w, rt.h, read_fmt);
    if (kClearRedTransfer) {
      TransitionImage(g_frame.list, rt.texture, rt.layout,
                      rhi::TextureState::kCopyDst);
      g_frame.list->ClearTexture(rt.texture, rhi::TextureState::kCopyDst, {},
                                 rhi::ClearColor{{1.0f, 0.0f, 0.0f, 1.0f}});
    }
    g_frame.list->ProfileBegin(6);
    CmdInsertLabel(g_frame.list, "present readback rt=%#llx %ux%u",
                   (unsigned long long)present_base, rt.w, rt.h);
    TransitionImage(g_frame.list, rt.texture, rt.layout,
                    rhi::TextureState::kCopySrc);
    rhi::Texture* read_src = rt.texture;
    if (convert) {
      rhi::TextureRegion region;
      region.width = rt.w;
      region.height = rt.h;
      rhi::TextureBarrier barrier;
      barrier.texture = convert;
      barrier.before = rhi::TextureState::kUndefined;
      barrier.after = rhi::TextureState::kCopyDst;
      g_frame.list->Barrier(0, 0, &barrier, 1);
      g_frame.list->BlitTexture(convert, region, rt.texture, region,
                                rhi::Filter::kNearest);
      barrier.before = rhi::TextureState::kCopyDst;
      barrier.after = rhi::TextureState::kCopySrc;
      g_frame.list->Barrier(0, 0, &barrier, 1);
      read_src = convert;
    }
    if (shared) {
      rhi::TextureRegion region;
      region.width = rt.w;
      region.height = rt.h;
      rhi::TextureBarrier barrier;
      barrier.texture = cur.shared;
      barrier.before = rhi::TextureState::kUndefined;
      barrier.after = rhi::TextureState::kCopyDst;
      g_frame.list->Barrier(0, 0, &barrier, 1);
      g_frame.list->CopyTexture(cur.shared, region, read_src, region);
      barrier.before = rhi::TextureState::kCopyDst;
      barrier.after = rhi::TextureState::kCopySrc;
      g_frame.list->Barrier(0, 0, &barrier, 1);
    } else {
      rhi::BufferTextureCopy copy;
      copy.region.width = rt.w;
      copy.region.height = rt.h;
      g_frame.list->CopyTextureToBuffer(g_frame.readback, read_src, &copy, 1);
    }
    cur.shared_frame = shared;
    cur.presentable = true;
    cur.w = rt.w;
    cur.h = rt.h;
    cur.fmt = read_fmt;
  }
  {
    ScopeNs submit_timer(&g_stats.ns_submit);
    ScopeNs frame_submit_timer(&g_frame_ns.submit);
    g_frame.list->ProfileEnd();
    bindless::FrameEnd(g_frame.list, g_frame.slot_idx);
    if (cur.timestamps)
      g_frame.list->WriteTimestamp(cur.timestamps, 1, false);
    CmdEndLabel(g_frame.list);
    g_frame.list->End();
    const u64 before = Device().LastSubmission();
    cur.submission = Device().Submit(g_frame.list);
    cur.submitted = cur.submission != before;
    if (!cur.submitted)
      BASE_LOGI("gpuvk", "frame submit failed");
  }
  if (!cur.submitted) {
    renderer.state = nullptr;
    return;
  }
  StampSubmittedLayouts();
  cur.frame_num = g_frame.num;
  cur.frame_draws = g_frame.draws;
  cur.frame_max_idx = g_frame.max_idx;
  cur.frame_had_room = g_frame.had_room;
  cur.present_base = present_base;
  cur.scanout_base = scanout_base;
  cur.readback = g_frame.readback;

  trace::FrameEnd(scanout_base);

  static int room_streak = 0;
  if (g_frame.had_room && g_frame.draws > 20 && ++room_streak >= 4)
    host::SetInGameplay(true);
  if (g_frame.max_idx >= 1500)
    host::SetInGameplay(true);

  const u32 finish_idx =
      FramePipelined() ? (g_frame.slot_idx ^ 1) : g_frame.slot_idx;
  if (FramePipelined())
    g_frame.slot_idx ^= 1;
  FrameSlot& fin = g_frame.slots[finish_idx];
  const bool waited = fin.submitted;
  phases.Mark(3);
  if (fin.submitted) {
    u64 tr0 = NowNs();
    DELTA_ZONE("gpu.frame_fence");
    if (!Device().Wait(fin.submission)) {
      BASE_LOGI("gpuvk", "frame {} DEVICE FAULT: draws={}", fin.frame_num,
                fin.frame_draws);
      renderer.state = nullptr;
      return;
    }
    u64 dt = NowNs() - tr0;
    g_stats.ns_readback += dt;
    g_frame_ns.wait += dt;
    if (fin.timestamps) {
      u64 timestamps[2] = {};
      if (Device().ReadTimestamps(fin.timestamps, 0, 2, timestamps)) {
        const u32 bits = Device().caps().timestamp_bits;
        const u64 mask = bits >= 64 ? UINT64_MAX : (u64{1} << bits) - 1;
        const u64 ticks = (timestamps[1] - timestamps[0]) & mask;
        g_stats.ns_gpu_exec += static_cast<u64>(
            static_cast<double>(ticks) * Device().caps().timestamp_period_ns);
        g_stats.gpu_exec_samples++;
      }
    }
    phases.Mark(4);
    bindless::FrameFinished(finish_idx);
    CollectPassTimes(fin);
    for (rhi::CommandList* c : fin.chunks)
      g_free_lists.push_back(c);
    fin.chunks.clear();
    fin.submitted = false;
  }
  PushStageSample();
  if (waited && (g_dump || kDeclines) && fin.frame_num % 30 == 0)
    ReportDeclines();
  if (!waited || !fin.presentable || fin.shared_frame) {
    if (waited && fin.shared_frame && fin.present_to_window) {
      ScopeNs present_timer(&g_stats.ns_present);
      ScopeNs frame_present_timer(&g_frame_ns.present);
      renderer.state->presenter.PresentShared(
          fin.shared, fin.shared_id, fin.shared_fd, fin.shared_bytes, fin.w,
          fin.h,
          fin.shared->desc().format == rhi::Format::kRGBA8Unorm
              ? host::PixelFormat::kRgba8
              : host::PixelFormat::kBgra8);
      Device().Maintain();
    }
    if (!ReportRtContents(cur)) {
      renderer.state = nullptr;
      return;
    }
    if (RdocFrame() && GetRdocApi() && GetRdocApi()->IsFrameCapturing()) {
      u32 ok = GetRdocApi()->EndFrameCapture(RdocDevice(), nullptr);
      BASE_LOGI("rdoc", "capture {}", ok ? "written" : "FAILED");
      if (ok && kRdocExit) {
        std::fflush(nullptr);
        std::_Exit(0);
      }
    }
    return;
  }

  static base::Vector<u8> flipped;
  static const guest::SessionReset reset_flipped(
      [] { guest::ResetResource(flipped); });
  auto* rb = fin.readback ? fin.readback->mapped() : nullptr;
  static const bool kRbTrace2 = std::getenv("DELTA_GPU_RBTRACE") != nullptr;
  if (kRbTrace2) {
    u64 nz = 0, sampled = 0;
    const size_t n = (size_t)fin.w * fin.h * 4;
    for (size_t i = 0; rb && i < n; i += 4099, sampled++)
      nz += rb[i] != 0;
    BASE_LOGI("rb", "present f{} nz={}/{} rt={:#x} {}x{} map={:p}{}",
              fin.frame_num, (unsigned long long)nz,
              (unsigned long long)sampled, (unsigned long)fin.present_base,
              fin.w, fin.h, static_cast<void*>(rb),
              fin.readback == g_frame.readback ? " (bound)" : "");
  }
  u8* pixels;
  host::PixelFormat pixel_fmt = host::PixelFormat::kBgra8;
  if (kFlipMode == 0 && (fin.fmt == rhi::Format::kBGRA8Unorm ||
                         fin.fmt == rhi::Format::kBGRA8Srgb)) {
    pixels = rb;
  } else if (kFlipMode == 0 && (fin.fmt == rhi::Format::kRGBA8Unorm ||
                                fin.fmt == rhi::Format::kRGBA8Srgb)) {
    pixels = rb;
    pixel_fmt = host::PixelFormat::kRgba8;
  } else {
    flipped.resize(static_cast<size_t>(fin.w) * fin.h * 4);
    const u32 src_bytes = FormatBytes(fin.fmt);
    const u32 src_stride = fin.w * src_bytes;
    for (u32 y = 0; y < fin.h; y++) {
      u32 sy = (kFlipMode & 1) ? (fin.h - 1 - y) : y;
      const u8* srow = rb + static_cast<size_t>(sy) * src_stride;
      u8* drow = flipped.data() + static_cast<size_t>(y) * fin.w * 4;
      for (u32 x = 0; x < fin.w; x++) {
        u32 sx = (kFlipMode & 2) ? (fin.w - 1 - x) : x;
        ReadbackPixelBgra(srow + static_cast<size_t>(sx) * src_bytes, fin.fmt,
                          drow + static_cast<size_t>(x) * 4);
      }
    }
    pixels = flipped.data();
  }
  phases.Mark(5);
  const auto to_bgra = [&]() {
    if (pixel_fmt == host::PixelFormat::kBgra8)
      return;
    flipped.resize(static_cast<size_t>(fin.w) * fin.h * 4);
    const size_t texels = static_cast<size_t>(fin.w) * fin.h;
    for (size_t i = 0; i < texels; i++) {
      flipped[i * 4 + 0] = pixels[i * 4 + 2];
      flipped[i * 4 + 1] = pixels[i * 4 + 1];
      flipped[i * 4 + 2] = pixels[i * 4 + 0];
      flipped[i * 4 + 3] = pixels[i * 4 + 3];
    }
    pixels = flipped.data();
    pixel_fmt = host::PixelFormat::kBgra8;
  };
  static int snap_best_draws = 0;
  static bool snapped = false;
  bool snap_now = kSnapAt && fin.frame_num >= kSnapAt && fin.frame_draws > 0 &&
                  (int)fin.frame_draws >= kSnapMinDraws &&
                  (int)fin.frame_max_idx >= kSnapMinIdx &&
                  (!kSnapRoom || (fin.frame_had_room && fin.frame_draws > 20));
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
    to_bgra();
    WritePpm(p, pixels, fin.w, fin.h);
    if (FILE* alpha = std::fopen("/tmp/gpu_snap_alpha.pgm", "wb")) {
      std::fprintf(alpha, "P5\n%u %u\n255\n", fin.w, fin.h);
      for (u64 i = 0; i < static_cast<u64>(fin.w) * fin.h; i++)
        std::fputc(pixels[i * 4 + 3], alpha);
      std::fclose(alpha);
    }
    BASE_LOGI("snap", "wrote {} (f{} {}x{} draws={} rt={:#x} scanout={:#x})", p,
              fin.frame_num, fin.w, fin.h, fin.frame_draws,
              (unsigned long)fin.present_base, (unsigned long)fin.scanout_base);
    snapped = true;
  }
  static int seq_done = 0, seq_last_frame = -10000;
  if (kSnapSeqN && seq_done < kSnapSeqN && fin.frame_had_room &&
      fin.frame_draws > 20 && fin.frame_num - seq_last_frame >= 250) {
    char p[256];
    std::snprintf(p, sizeof p, "%s/seq_%02d.ppm", DumpDir(), seq_done);
    to_bgra();
    WritePpm(p, pixels, fin.w, fin.h);
    BASE_LOGI("snapseq", "{} -> f{} draws={}", seq_done, fin.frame_num,
              fin.frame_draws);
    seq_done++;
    seq_last_frame = fin.frame_num;
  }

  static int every_done = 0, every_last = -1000000;
  if (kSnapEvery && every_done < kSnapEveryMax && fin.frame_draws > 0 &&
      fin.frame_num - every_last >= kSnapEvery) {
    char p[256];
    std::snprintf(p, sizeof(p), "%s/every_%03d.ppm", DumpDir(), every_done);
    to_bgra();
    WritePpm(p, pixels, fin.w, fin.h);
    BASE_LOGI("snapevery", "{} -> f{} draws={}", every_done, fin.frame_num,
              fin.frame_draws);
    every_done++;
    every_last = fin.frame_num;
  }

  if (g_dump && fin.frame_had_room && fin.frame_draws > 20) {
    char p[256], tmp[256];
    std::snprintf(p, sizeof(p), "%s/gpu_room.ppm", DumpDir());
    std::snprintf(tmp, sizeof(tmp), "%s/gpu_room.tmp", DumpDir());
    to_bgra();
    WritePpm(tmp, pixels, fin.w, fin.h);
    std::rename(tmp, p);
  }
  if (g_dump && fin.frame_num >= 1000 && fin.frame_num % 2000 == 0 &&
      fin.frame_draws > 0) {
    to_bgra();
    DumpPpm(pixels, fin.w, fin.h);
  }
  if (g_dump && fin.frame_num % kLatestEvery == 0 && fin.frame_draws > 0) {
    char latest[256];
    std::snprintf(latest, sizeof(latest), "%s/gpu_latest.ppm", DumpDir());
    to_bgra();
    WritePpm(latest, pixels, fin.w, fin.h);
  }
  if (g_dump && fin.frame_num % 200 == 0) {
    BASE_LOGI("gpuvk", "frame {} draws={} rt={:#x} {}x{}  scanout={:#x}",
              fin.frame_num, fin.frame_draws, (unsigned long)fin.present_base,
              fin.w, fin.h, (unsigned long)fin.scanout_base);
    for (auto& kv : g_rts)
      if (kv.second.used_this_frame)
        BASE_LOGI("gpuvk", "   RT {:#x} {}x{} draws={}{}",
                  (unsigned long)kv.first, kv.second.w, kv.second.h,
                  kv.second.draws,
                  kv.first == fin.scanout_base ? " <-SCANOUT" : "");
  }
  DrawPerfOverlay(pixels, fin.w, fin.h, pixel_fmt == host::PixelFormat::kRgba8);
  static bool overlay_dumped = false;
  if (kOverlayDump && !overlay_dumped && fin.frame_num >= 600) {
    overlay_dumped = true;
    char p[256];
    std::snprintf(p, sizeof p, "%s/gpu_overlay.ppm", DumpDir());
    to_bgra();
    WritePpm(p, pixels, fin.w, fin.h);
    BASE_LOGI("overlay", "wrote {}", p);
  }
  if (fin.present_to_window) {
    ScopeNs present_timer(&g_stats.ns_present);
    ScopeNs frame_present_timer(&g_frame_ns.present);
    if (kSyncPresent) {
      if (host::Ensure("prosperity", fin.w, fin.h) && host::PumpEvents())
        host::Present(pixels, fin.w, fin.h, fin.w * 4, pixel_fmt);
    } else if (pixels == flipped.data()) {
      renderer.state->presenter.Present(base::move(flipped), fin.w, fin.h,
                                        pixel_fmt);
    } else {
      renderer.state->presenter.Present(pixels, fin.w, fin.h, pixel_fmt);
    }
  }

  Device().Maintain();

  if (kGpuRtstat && renderer.state)
    renderer.state->presenter.WaitForBorrowed();
  if (!ReportRtContents(cur)) {
    renderer.state = nullptr;
    return;
  }
  if (snapped && kSnapExit) {
    std::fflush(nullptr);
    std::_Exit(0);
  }

  if (RdocFrame() && GetRdocApi() && GetRdocApi()->IsFrameCapturing()) {
    u32 ok = GetRdocApi()->EndFrameCapture(RdocDevice(), nullptr);
    BASE_LOGI("rdoc", "capture {}", ok ? "written" : "FAILED");
    if (ok && kRdocExit) {
      std::fflush(nullptr);
      std::_Exit(0);
    }
  }
}

namespace {
const guest::SessionReset g_session_reset([] {
  g_pass_open = ~0u;
  guest::ResetResource(g_free_lists);
  guest::ResetResource(g_pass_draws_at_open);
  guest::ResetResource(g_pass_totals);
  guest::ResetResource(g_pass_frames);
});
}  // namespace

}  // namespace gpu::render
