/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/draw/draw_recomp.h"
#include "base/arch.h"
#include "gpu/render/image/bindless.h"
#include "gpu/render/memory/buffer_cache.h"
#include "guest/session.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/compute/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/memory/guest_direct.h"
#include "gpu/render/image/guest_format.h"
#include "gpu/render/draw/index_convert.h"
#include "gpu/render/draw/index_upload.h"
#include "gpu/render/debug/labels.h"
#include "gpu/render/draw/mesh_args.h"
#include "gpu/render/draw/pipeline_cache.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/render/debug/trace.h"
#include "gpu/render/memory/upload_ring.h"
#include "gpu/write_tracker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/unordered_map.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/time/time.h"
#include "host_memory/host_memory.h"
#include "options/options.h"
#include "profile/profile.h"
#include "write_watch/write_watch.h"

namespace {
DELTA_OPTION(bool, kNoWipe, "DELTA_GPU_NOWIPE", true);
DELTA_OPTION(bool, kLazyClear2, "DELTA_GPU_LAZYCLEAR", true);
DELTA_OPTION(int, kTexBindFrame, "DELTA_GPU_TEXBIND", -1);
DELTA_OPTION(int, kSeqN, "DELTA_GPU_DRAWSEQ", 0);
DELTA_OPTION(u64, kWant, "DELTA_GPU_DRAWRT", 0);
DELTA_OPTION(int, kWantFrame, "DELTA_GPU_DRAWRT_FRAME", 0);
DELTA_OPTION(int, kBusy, "DELTA_GPU_DRAWRT_BUSY", 0);
DELTA_OPTION(int, kClearTrace, "DELTA_GPU_CLEARTRACE", 0);
DELTA_OPTION(bool, kUiWatch, "DELTA_GPU_UIWATCH", false);
DELTA_OPTION(bool, kClearRectScissor, "DELTA_GPU_CLEARRECT_SCISSOR", false);
DELTA_OPTION(bool, kDrawTrace, "DELTA_GPU_DRAWTRACE", false);
DELTA_OPTION(bool, kGpuDecltrace, "DELTA_GPU_DECLTRACE", false);
DELTA_OPTION(u64, kWhyDrop, "DELTA_GPU_WHYDROP", 0);
DELTA_OPTION(u64, kBindTrace, "DELTA_GPU_BINDTRACE", 0);
DELTA_OPTION(bool, kRawBufTrace, "DELTA_GPU_RAWBUF", false);
DELTA_OPTION(bool, kSelfTrace, "DELTA_GPU_SELFTRACE", false);
DELTA_OPTION(bool, kTightCbuf, "DELTA_GPU_TIGHTCBUF", false);
DELTA_OPTION(bool, kDirectBind, "DELTA_GPU_DIRECT_BIND", true);
DELTA_OPTION(bool, kRawBind, "DELTA_GPU_RAW_BIND", true);
DELTA_OPTION(bool, kCbufBind, "DELTA_GPU_CBUF_BIND", true);

u32 CbufCopyBytes(u64 base, u32 size) {
  const u32 planned = base::Min(size, gpu::render::kCbufWindow);
  const u64 page_end = (base + 0x1000) & ~u64{0xFFF};
  return kTightCbuf
             ? planned
             : base::Max(planned,
                         static_cast<u32>(base::Min<u64>(
                             gpu::render::kCbufWindow, page_end - base)));
}

DELTA_OPTION(u64, kCbInfo, "DELTA_GPU_CBSTAGED", 0);
DELTA_OPTION(u64, kTexForce, "DELTA_GPU_TEXFORCE", 0);
DELTA_OPTION(bool, kTint, "DELTA_GPU_RTTINT", false);
}  // namespace

namespace gpu::render {

using render::DrawInfo;
using render::FlushCsWritesRange;

namespace {

enum DeclineReason {
  kNoRecomp,
  kSelf,
  kRing,
  kGuestTex,
  kMidRegion,
  kNoPipe,
  kMaxDeclineReason
};
static const char* kDeclineName[kMaxDeclineReason] = {
    "norecomp", "self", "ring", "guesttex", "midregion", "nopipe"};
u32 g_decline[kMaxDeclineReason] = {0};
inline bool Decline(DeclineReason r) {
  g_decline[r]++;
  g_stats.win_declines++;
  if (trace::Recording())
    trace::RecordDecline(kDeclineName[r]);
  return false;
}

struct StageCacheKey {
  u64 base;
  u32 salt;
  bool operator==(const StageCacheKey&) const = default;
};

struct StageCacheKeyHash {
  mem_size operator()(const StageCacheKey& key) const {
    const u64 h = (key.base ^ (u64(key.salt) << 56)) * 0x9E3779B97F4A7C15ull;
    return static_cast<mem_size>(h ^ (h >> 32));
  }
};

DELTA_OPTION(bool, kRingTrack, "DELTA_GPU_RING_TRACK", true);

bool TrackingGuestWrites() {
  return kRingTrack && gpu::GuestWriteTracker().enabled();
}

bool ArmForCache(u64 base, u64 bytes) {
  return TrackingGuestWrites() && gpu::GuestWriteTracker().Arm(base, bytes);
}

struct StageCache {
  struct Entry {
    u64 off;
    u64 bytes;
    u64 source_bytes;
    u64 revision;
    u64 gen;
    u32 dcb;
    bool tracked;
    u32 max_index;
    u64 digest;
  };
  static constexpr u32 kBlockShift = 16;
  base::UnorderedMap<StageCacheKey, Entry, StageCacheKeyHash> map;
  base::HashMap<u64, base::Vector<StageCacheKey>> blocks;
  int frame = -1;
  u32 dcb = 0;

  bool Live(const Entry& e) const {
    return e.bytes && (e.tracked || (e.dcb == dcb &&
                                     e.gen == render::CsWritebackGeneration()));
  }
  void RollFrame() {
    if (frame != g_frame.num) {
      frame = g_frame.num;
      dcb = render::g_dcb_n;
      map.clear();
      blocks.clear();
    } else if (dcb != render::g_dcb_n) {
      dcb = render::g_dcb_n;
      if (!TrackingGuestWrites())
        map.clear();
    }
  }
  const Entry* Find(u64 base, u64 bytes, u32 salt = 0);
  void Insert(u64 base,
              u64 bytes,
              u32 salt,
              u64 off,
              bool tracked = false,
              u32 max_index = ~0u,
              u64 source_bytes = 0,
              u64 revision = 0,
              u64 digest = 0) {
    RollFrame();
    const StageCacheKey key{base, salt};
    auto& e = map[key];
    if (e.bytes >= bytes && Live(e) && e.revision == revision &&
        e.digest == digest)
      return;
    if (!source_bytes)
      source_bytes = bytes;
    e = {off, bytes,   source_bytes, revision, render::CsWritebackGeneration(),
         dcb, tracked, max_index,    digest};
    if (tracked)
      for (u64 b = base >> kBlockShift;
           b <= (base + source_bytes - 1) >> kBlockShift; b++)
        blocks[b].push_back(key);
  }
  void Invalidate(u64 first, u64 end) {
    const auto drop = [&](const base::Vector<StageCacheKey>& keys) {
      for (const StageCacheKey& key : keys) {
        Entry* e = map.find(key);
        if (e && key.base < end && first < key.base + e->source_bytes)
          map.erase(key);
      }
    };
    const u64 first_block = first >> kBlockShift;
    const u64 last_block = (end - 1) >> kBlockShift;
    if (last_block - first_block + 1 > blocks.size()) {
      for (const auto& [block, keys] : blocks)
        if (block >= first_block && block <= last_block)
          drop(keys);
      return;
    }
    for (u64 b = first_block; b <= last_block; b++)
      if (auto found = blocks.find(b); found != blocks.end())
        drop(found->second);
  }
};

StageCache g_vb_staged, g_ib_staged, g_ubo_staged, g_sbo_staged;

struct GpuStaged {
  u64 off = 0;
  u64 bytes = 0;
  u64 rev = 0;
  int frame = -1;
};
base::HashMap<u64, GpuStaged> g_sbo_gpu;
base::HashMap<u64, GpuStaged> g_ubo_gpu;

struct BufferBindingKey {
  static constexpr u32 kCount =
      kRawBufBindings > kCbufBindings ? kRawBufBindings : kCbufBindings;
  u32 count = 0;
  rhi::Buffer* buffers[kCount]{};
  u64 offsets[kCount]{};
  u32 bytes[kCount]{};
  bool operator==(const BufferBindingKey&) const = default;
};

struct BufferBindingKeyHash {
  mem_size operator()(const BufferBindingKey& key) const {
    u64 h = 1469598103934665603ull ^ key.count;
    for (u32 i = 0; i < key.count; i++) {
      h = (h ^ reinterpret_cast<u64>(key.buffers[i])) * 1099511628211ull;
      h = (h ^ key.offsets[i]) * 1099511628211ull;
      h = (h ^ key.bytes[i]) * 1099511628211ull;
    }
    return static_cast<mem_size>(h);
  }
};
using BufferBindingCache =
    base::HashMap<BufferBindingKey, rhi::BindGroup*, BufferBindingKeyHash>;
BufferBindingCache g_raw_bindings[3], g_cbuf_bindings[3];

rhi::BindGroup* BufferBindings(const BufferBindingKey& key,
                               rhi::BindGroupLayout* layout,
                               u32 count,
                               BufferBindingCache& cache) {
  if (auto found = cache.find(key); found != cache.end())
    return found->second;
  rhi::BindGroupDesc desc;
  desc.layout = layout;
  for (u32 i = 0; i < count; i++) {
    rhi::BindingWrite w;
    w.binding = i;
    w.buffer = key.buffers[i];
    w.offset = key.offsets[i];
    w.range = key.bytes[i];
    desc.writes.push_back(w);
  }
  auto* set = Device().CreateBindGroup(desc);
  if (set)
    cache.emplace(key, set);
  return set;
}

rhi::BindGroup* RawBufferBindings(const CachedBuffer* direct,
                                  const DrawInfo& d,
                                  u32 count) {
  BufferBindingKey key;
  key.count = count;
  rhi::Buffer* fallback = g_ring.sbo_buf ? g_ring.sbo_buf : g_ring.sbo_zero;
  for (u32 i = 0; i < count; i++) {
    key.buffers[i] = direct[i].buffer ? direct[i].buffer : fallback;
    key.offsets[i] = direct[i].buffer ? direct[i].offset : 0;
    key.bytes[i] = direct[i].buffer ? base::Min(d.bufs[i].size, kRawBufWindow)
                                    : kRawBufWindow;
  }
  return BufferBindings(key, GetRawBufferLayout(count), count,
                        g_raw_bindings[g_frame.num % 3]);
}

rhi::BindGroup* CbufBindings(const CachedBuffer* direct, u32 count) {
  BufferBindingKey key;
  key.count = count;
  rhi::Buffer* fallback = g_ring.ubo_buf ? g_ring.ubo_buf : g_ring.ubo_zero;
  for (u32 i = 0; i < count; i++) {
    key.buffers[i] = direct[i].buffer ? direct[i].buffer : fallback;
    key.offsets[i] = direct[i].buffer ? direct[i].offset : 0;
    key.bytes[i] = kCbufWindow;
  }
  return BufferBindings(key, GetCbufLayout(count), count,
                        g_cbuf_bindings[g_frame.num % 3]);
}

void SyncGuestWrites() {
  static int frame = -1;
  static u32 dcb = 0;
  static u64 gen = 0;
  if (!TrackingGuestWrites() ||
      (frame == g_frame.num && dcb == render::g_dcb_n &&
       gen == render::CsWritebackGeneration()))
    return;
  frame = g_frame.num;
  dcb = render::g_dcb_n;
  gen = render::CsWritebackGeneration();
  CollectGuestWrites();
}

const StageCache::Entry* StageCache::Find(u64 base, u64 bytes, u32 salt) {
  RollFrame();
  SyncGuestWrites();
  const Entry* e = map.find({base, salt});
  if (!e || e->bytes < bytes || !Live(*e) ||
      (e->revision ? CsBufferRevision(base, e->source_bytes) != e->revision
                   : render::CsRangeDirtyOverlapping(base, e->source_bytes)))
    return nullptr;
  return e;
}

u64 StagedOffset(const StageCache::Entry* e) {
  return e ? e->off : u64(-1);
}

// Four independent multiply chains, so hashing runs near memory speed.
u64 GuestDigest(u64 base, u64 bytes) {
  const auto* p = reinterpret_cast<const u8*>(base);
  constexpr u64 kMul = 0x9E3779B97F4A7C15ull;
  u64 h[4] = {bytes, kMul, ~bytes, kMul >> 1};
  u64 i = 0;
  for (; i + 32 <= bytes; i += 32)
    for (u32 k = 0; k < 4; k++) {
      u64 w;
      std::memcpy(&w, p + i + k * 8, 8);
      h[k] = (h[k] ^ w) * kMul;
      h[k] ^= h[k] >> 29;
    }
  for (; i < bytes; i++)
    h[0] = (h[0] ^ p[i]) * kMul;
  return (h[0] ^ (h[1] * 3) ^ (h[2] * 5) ^ (h[3] * 7)) * kMul;
}

// A staged copy of constants or a raw buffer is reused only while the guest
// bytes still hash the same. Write tracking cannot vouch for these: titles
// rewrite descriptor tables through a second mapping of the same memory
// (DRIVECLUB's CPU view of its GPU heap), which no protection of ours sees.
// The ring itself is write-combined, so the check never reads it.
u64 MatchingStagedOffset(const StageCache::Entry* e, u64 base) {
  if (!e || GuestDigest(base, e->bytes) != e->digest)
    return u64(-1);
  return e->off;
}

DELTA_OPTION(bool, kRingDedup, "DELTA_GPU_RING_DEDUP", true);

bool BindVertexDescriptor(u64 base,
                          u64 bytes,
                          CachedBuffer& kept,
                          CachedBuffer& direct) {
  if (!bytes)
    return false;
  if (CsResidentGeometryBuffer(base, bytes, &kept.buffer, &kept.offset))
    return true;
  if (CsHoldsGeometry(base, bytes))
    return false;
  if (kRingDedup && TrackingGuestWrites()) {
    SyncGuestWrites();
    if (FindCachedBuffer(base, bytes, kept))
      return true;
  }
  if (kDirectBind)
    direct.buffer = DirectBindAt(base, bytes, &direct.offset);
  if (!direct.buffer)
    return false;
  g_stats.bind_bytes += bytes;
  g_stats.bind_n++;
  return true;
}

bool IsReadableThisFrame(u64 base, u32 size) {
  return gpu::IsReadableRangeCached(base, size);
}

}  // namespace

namespace {
int g_collect_frame = -1;
u32 g_collect_dcb = 0;
}  // namespace

void CollectGuestWrites() {
  static base::Vector<gpu::WriteTracker::Range> written;
  static const guest::SessionReset reset_written(
      [] { guest::ResetResource(written); });
  if (!gpu::GuestWriteTracker().enabled())
    return;
  g_collect_frame = g_frame.num;
  g_collect_dcb = render::g_dcb_n;
  written.clear();
  const size_t announced = gpu::GuestWriteTracker().Collect(written);
  for (size_t i = 0; i < written.size(); i++) {
    const auto [first, end] = written[i];
    InvalidateCachedBuffers(first, end);
    g_vb_staged.Invalidate(first, end);
    g_ib_staged.Invalidate(first, end);
    g_sbo_staged.Invalidate(first, end);
    CsNoteGuestWrites(first, end);
    MirrorNoteGuestWrites(first, end, i < announced);
    InvalidateTexRange(first, end - first);
  }
  MirrorWritesCollected();
  MirrorRefreshWritten();
}

void NoteGuestRemap(u64 base, u64 bytes) {
  CsGuestRemapped(base, bytes);
  gpu::MemoryRemapped().store(true, base::memory_order_release);
  gpu::InvalidateReadable(base, bytes);
  gpu::NoteGuestRemap(base, bytes);
  NoteGuestDirectRemap(base, bytes);
  gpu::GuestMappingGeneration().fetch_add(1, base::memory_order_release);
}

void CollectGuestWritesForSubmission() {
  if (g_collect_frame != g_frame.num || g_collect_dcb != render::g_dcb_n)
    CollectGuestWrites();
}

static_assert(render::DrawInfo::kMaxBuffers == kRawBufBindings,
              "the command processor and the raw-buffer ring must agree on "
              "how many set-2 bindings exist");

static void WhyDrop(const render::DrawInfo& d, const char* where) {
  if (!kWhyDrop || d.ps_addr != (u64)kWhyDrop)
    return;
  BASE_LOGI("whydrop", "ps={:#x} exit={} rt={:#x} mrt={} depth={:#x}",
            (unsigned long)d.ps_addr, where, (unsigned long)d.rt_base,
            d.mrt_count, (unsigned long)d.depth_base);
}

void ReleaseDrawBufferBindings() {
  for (auto* caches : {g_raw_bindings, g_cbuf_bindings}) {
    auto& cache = caches[g_frame.num % 3];
    for (const auto& [key, set] : cache)
      Device().Destroy(set);
    cache.clear();
  }
}

void ReportDeclines() {
  base::String line;
  base::FormatTo(line, "  decline:");
  for (int i = 0; i < kMaxDeclineReason; i++)
    if (g_decline[i])
      base::FormatTo(line, " {}={}", kDeclineName[i], g_decline[i]);
  BASE_LOGI("gpuvk", "{}", line.c_str());
}

bool ShaderFilterDrops(u64 ps_addr) {
  static const u64 kSkipPs = [] {
    const char* e = std::getenv("DELTA_GPU_SKIP_PS");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  if (kSkipPs && ps_addr == kSkipPs)
    return true;
  static const base::Vector<u64> kOnlyPs = [] {
    base::Vector<u64> out;
    if (const char* e = std::getenv("DELTA_GPU_ONLY_PS"))
      for (const char* p = e; *p;) {
        while (*p == ',' || *p == ' ')
          p++;
        if (!*p)
          break;
        out.push_back(std::strtoull(p, nullptr, 0));
        while (*p && *p != ',')
          p++;
      }
    if (!out.empty()) {
      base::String list;
      base::FormatTo(list, "keeping only {} shader(s):", out.size());
      for (u64 v : out)
        base::FormatTo(list, " {:#x}", (unsigned long long)v);
      BASE_LOGI("onlyps", "{}", list.c_str());
    }
    return out;
  }();
  if (kOnlyPs.empty())
    return false;
  for (u64 v : kOnlyPs)
    if (v == ps_addr)
      return false;
  return true;
}

// The depth target whose stencil the draws wrote at `base` since a colour
// target at that address last changed, or 0. SotC's deferred lighting reads
// its material ids from the stencil, and the colour target there is only a
// compute clear.
u64 FresherStencil(u64 base, const DrawInfo::DrawTex& t, bool rt_eligible) {
  if (!base || !rt_eligible || t.arrayed || !g_rts.count(base))
    return 0;
  const u64 stencil = ResolveSampledStencil(base);
  return stencil &&
                 g_depths[stencil].render_serial >= g_rts[base].render_serial
             ? stencil
             : 0;
}

rhi::TextureView* BindlessTextureView(const DrawInfo::DrawTex& t,
                                      rhi::TextureState* state) {
  *state = rhi::TextureState::kShaderRead;
  if (t.null_descriptor)
    return t.is_3d     ? g_tex.zero_3d_view
           : t.arrayed ? g_tex.zero_array_view
                       : g_tex.zero_view;
  u64 base = t.base;
  const bool rt_eligible =
      (!t.arrayed ||
       (t.layers == 1 && t.base_array == 0 && t.view_layers == 1)) &&
      !t.is_3d && t.base_mip == 0 && !GuestFormatBlockCompressed(t.dfmt);
  if (base && rt_eligible && !g_rts.count(base) && !g_depths.count(base)) {
    const bool depth_format = t.dfmt == 4 && t.nfmt == 7;
    u64 resolved = depth_format ? ResolveSampledDepth(base, t.w, t.h) : 0;
    if (!resolved)
      resolved = ResolveSampledRT(base, t.w, t.h);
    if (!resolved && !depth_format)
      resolved = ResolveSampledDepth(base, t.w, t.h);
    if (resolved)
      base = resolved;
  }
  if (const u64 stencil = FresherStencil(base, t, rt_eligible)) {
    *state = rhi::TextureState::kDepthRead;
    return StencilSampledView(g_depths[stencil]);
  }
  if (base && rt_eligible && g_rts.count(base) && g_rts[base].ever_rendered) {
    rhi::Format format = rhi::Format::kUndefined;
    return SampledViewAs(g_rts[base], t.swizzle,
                         GuestTextureFormat(t.dfmt, t.nfmt), &format,
                         t.arrayed);
  }
  if (base && rt_eligible && !t.arrayed && g_depths.count(base)) {
    *state = rhi::TextureState::kDepthRead;
    return SampledView(g_depths[base], t.swizzle);
  }
  return TexViewFor(t);
}

bool DrawRecomp(render::Renderer& renderer, const DrawInfo& d) {
  DELTA_ZONE("gpu.draw_recomp");
  const u64 t_draw_start = NowNs();
  if (kWhyDrop == 1)
    BASE_LOGI("whydrop",
              "draw#{} ps={:#x} vs={:#x} rt={:#x} mrt={} tex={:#x} "
              "prim={} cnt={}",
              g_frame.draws, (unsigned long)d.ps_addr, (unsigned long)d.vs_addr,
              (unsigned long)d.rt_base, d.mrt_count, (unsigned long)d.tex_base,
              d.prim_type, d.index_data ? d.index_count : d.vertex_count);
  bool indexed = d.index_data != nullptr;
  u32 draw_count = indexed ? d.index_count : d.vertex_count;
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      BASE_LOGI("dt",
                "enter recomp count={} rt={:#x} tex={:#x} num_texs={} "
                "num_vattrs={} ok={}",
                draw_count, (unsigned long)d.rt_base, (unsigned long)d.tex_base,
                d.num_texs, d.num_vattrs, d.recomp ? d.recomp->ok : 0);
  }
  if ((d.depth_clear_draw || d.stencil_clear_draw) && d.depth_base &&
      draw_count) {
    EndRegion();
    if (!BeginRegion(d.mrt_base, d.mrt_info, d.mrt_count, d.rt_w, d.rt_h,
                     d.depth_base, d.depth_clear, d.stencil_base,
                     d.stencil_clear, false, DepthW(d), DepthH(d), d.mrt_surf_w,
                     d.mrt_surf_h, d.mrt_dcc_base, d.mrt_clear_word,
                     d.depth_htile_base, d.depth_slice))
      return true;
    const u8 aspect = (d.depth_clear_draw ? rhi::kAspectDepth : 0u) |
                      (d.stencil_clear_draw ? rhi::kAspectStencil : 0u);
    g_frame.list->ClearAttachment(~0u, {}, d.depth_clear, d.stencil_clear,
                                  aspect, 0, 0, d.rt_w, d.rt_h);
    g_depths[d.depth_base].dirty_for_read = true;
    g_frame.draws++;
    return true;
  }
  if (!d.recomp || !d.recomp->ok || !draw_count) {
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 96)
        BASE_LOGI("decl",
                  "norecomp why={} vcount={} icount={} idx={:p} "
                  "vbufs={} vattrs={} prim={:#x} rt={:#x}",
                  !d.recomp       ? "no-program"
                  : !d.recomp->ok ? "translate-failed"
                                  : "count=0",
                  d.vertex_count, d.index_count, d.index_data, d.num_vbufs,
                  d.num_vattrs, d.prim_type, (unsigned long)d.rt_base);
    }
    return Decline(kNoRecomp);
  }
  const bool mesh = !d.recomp->mesh_spirv.empty();
  const rhi::Caps& caps = Device().caps();
  if (mesh &&
      (!caps.mesh_shader || !d.recomp->mesh_input_primitives || indexed))
    return Decline(kNoRecomp);
  if (mesh) {
    const auto& limits = caps.mesh;
    if (d.recomp->mesh_threads > limits.max_threads ||
        d.recomp->mesh_threads > limits.max_threads_x ||
        d.recomp->mesh_shared_bytes > limits.max_shared_bytes ||
        d.recomp->mesh_vertices > limits.max_output_vertices ||
        d.recomp->mesh_primitives > limits.max_output_primitives)
      return Decline(kNoRecomp);
    const u32 groups = (draw_count - 1) / d.recomp->mesh_input_primitives + 1;
    const u32 instances = base::Max(d.instance_count, 1u);
    if (!d.indirect_args &&
        (groups > limits.max_groups[0] || instances > limits.max_groups[1] ||
         u64(groups) * instances > limits.max_total_groups))
      return Decline(kNoRecomp);
  }
  const bool has_storage_image =
      base::AnyOf(d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                  [](const gcn::ShaderTex& tex) { return tex.storage; });
  if (!d.mrt_count && !d.depth_base && !has_storage_image) {
    static int n = 0;
    if (kGpuDecltrace && n++ < 48)
      BASE_LOGI("decl",
                "no-target draw#{} vs={:#x} ps={:#x} tex={:#x} "
                "icount={} vcount={} mask={:#x} pstex={} storage={}",
                g_frame.draws, (unsigned long)d.vs_addr,
                (unsigned long)d.ps_addr, (unsigned long)d.tex_base,
                d.index_count, d.vertex_count, d.target_mask,
                d.recomp->ps_texs.size(),
                (size_t)base::CountIf(
                    d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                    [](const gcn::ShaderTex& t) { return t.storage; }));
    WhyDrop(d, "no-target");
    g_frame.draws++;
    return true;
  }
  u64 tex_base = d.tex_base;
  const bool force_white_tex = kTexForce && tex_base == (u64)kTexForce;
  if (force_white_tex)
    tex_base = 0;
  const bool tex_rt_eligible =
      (!d.tex_arrayed || (d.tex_layers == 1 && d.tex_base_array == 0 &&
                          d.tex_view_layers == 1)) &&
      !d.tex_is_3d && d.tex_base_mip == 0 &&
      !GuestFormatBlockCompressed(d.tex_dfmt);
  if (tex_base && tex_rt_eligible && !g_rts.count(tex_base) &&
      !g_depths.count(tex_base)) {
    bool depth_format = d.tex_dfmt == 4 && d.tex_nfmt == 7;
    u64 r = depth_format ? ResolveSampledDepth(tex_base, d.tex_w, d.tex_h) : 0;
    if (!r)
      r = ResolveSampledRT(tex_base, d.tex_w, d.tex_h);
    if (!r && !depth_format)
      r = ResolveSampledDepth(tex_base, d.tex_w, d.tex_h);
    if (r)
      tex_base = r;
  }
  bool color_as_tex = tex_rt_eligible && tex_base && tex_base != d.rt_base &&
                      g_rts.count(tex_base);
  bool feedback_as_tex = tex_rt_eligible && tex_base && tex_base == d.rt_base &&
                         g_rts.count(tex_base) && g_rts[tex_base].ever_rendered;
  const bool depth_self_read = tex_rt_eligible && !d.tex_arrayed && tex_base &&
                               tex_base == d.depth_base &&
                               !d.depth_write_enable &&
                               g_depths.count(tex_base);
  bool depth_as_tex = tex_rt_eligible && !d.tex_arrayed && tex_base &&
                      (tex_base != d.depth_base || depth_self_read) &&
                      g_depths.count(tex_base);
  bool rt_as_tex = color_as_tex || feedback_as_tex || depth_as_tex;
  if (color_as_tex && g_rts[tex_base].w >= 700 && g_rts[tex_base].w <= 900)
    g_frame.had_room = true;
  if (tex_base && !depth_self_read &&
      (tex_base == d.depth_base ||
       (tex_base == d.rt_base && !feedback_as_tex))) {
    static int n = 0;
    if (kSelfTrace && n++ < 20)
      BASE_LOGI("self",
                "draw#{} ps={:#x} rt={:#x} tex={:#x} depth={:#x} dtest={} "
                "dwrite={} ntex={}",
                g_frame.draws, (unsigned long)d.ps_addr,
                (unsigned long)d.rt_base, (unsigned long)tex_base,
                (unsigned long)d.depth_base, (int)d.depth_test_enable,
                (int)d.depth_write_enable, d.num_texs);
    return Decline(kSelf);
  }
  if ((color_as_tex || feedback_as_tex) && !CsRefreshRtFromTruth(tex_base))
    render::FlushCsWritesAt(renderer, tex_base, "rt-tex");
  const auto vtx_decline = [&](const char* why, u32 n) {
    if (kGpuDecltrace) {
      static int t = 0;
      if (t++ < 32)
        BASE_LOGI("decl",
                  "vtx why={} n={} vcount={} icount={} itype={} "
                  "vattrs={} vdata={:p} vstride={} rt={:#x}",
                  why, n, d.vertex_count, d.index_count, d.index_type,
                  d.num_vattrs, d.vertex_data, d.vertex_stride,
                  (unsigned long)d.rt_base);
    }
    return Decline(kNoRecomp);
  };
  u32 nv = d.vertex_count;
  const bool quads = indexed && d.prim_type == 19;
  // A non-indexed quad list draws through two triangles per quad indexed here.
  const bool auto_quads =
      !indexed && !d.indirect_args && d.prim_type == 19 && nv >= 4;
  const bool indirect_quads = quads && d.indirect_args;
  const u32 upload_indices = quads        ? d.index_count / 4 * 6
                             : auto_quads ? nv / 4 * 6
                                          : d.index_count;
  const bool convert_indices = indexed && (quads || d.index_type == 2);
  const u32 ib_tag = 1u + d.index_type + (quads ? 4u : 0u);
  const u64 index_bytes =
      indexed ? static_cast<u64>(upload_indices) *
                    (convert_indices ? 4
                                     : UploadedIndexElementBytes(d.index_type))
      : auto_quads ? static_cast<u64>(upload_indices) * 4
                   : 0;
  const bool descriptor_bounds =
      d.indirect_args ||
      (indexed && CsHoldsGeometry(reinterpret_cast<u64>(d.index_data),
                                  u64(d.index_count) *
                                      GuestIndexElementBytes(d.index_type)));
  bool index_scanned = false;
  if ((d.num_vattrs && !indexed && !descriptor_bounds && nv > 200000u) ||
      (d.num_vattrs && !d.vertex_data))
    return vtx_decline(nv > 200000u ? "nv-cap" : "no-vertex-data", nv);
  {
    const u32 cb_mode = (d.color_control >> 4) & 7u;
    if (cb_mode >= 2 && cb_mode <= 6) {
      WhyDrop(d, "cb-metadata-pass");
      g_frame.draws++;
      return true;
    }
  }

  if (d.is_clear_rect) {
    const u32 cb_mode = (d.color_control >> 4) & 7u;
    if (cb_mode != 1) {
      if (kClearTrace) {
        static int n = 0;
        if (n++ < kClearTrace)
          BASE_LOGI("clear",
                    "f{} draw#{} rect RT {:#x} cc={:#x} mode={} "
                    "NOT-A-CLEAR (CB metadata pass, content preserved)",
                    g_frame.num, g_frame.draws, (unsigned long)d.rt_base,
                    d.color_control, cb_mode);
      }
      WhyDrop(d, "cb-metadata-pass");
      g_frame.draws++;
      return true;
    }
    const u32 cx0 = d.clear_tl & 0x7FFF, cy0 = (d.clear_tl >> 16) & 0x7FFF;
    const u32 cx1 = d.clear_br & 0x7FFF, cy1 = (d.clear_br >> 16) & 0x7FFF;
    const bool covers_target =
        !kClearRectScissor ||
        (cx0 == 0 && cy0 == 0 && cx1 >= d.rt_w && cy1 >= d.rt_h);
    if (!covers_target) {
      if (kClearTrace) {
        static int n = 0;
        if (n++ < 16)
          BASE_LOGI("clear",
                    "rect SKIPPED rt={:#x} scissor=({},{})-({},{}) "
                    "target={}x{}",
                    (unsigned long)d.rt_base, cx0, cy0, cx1, cy1, d.rt_w,
                    d.rt_h);
      }
      WhyDrop(d, "clear-partial");
      g_frame.draws++;
      return true;
    }
    for (u32 i = 0; i < d.mrt_count && i < 8; i++) {
      auto it = g_rts.find(d.mrt_base[i]);
      if (it == g_rts.end())
        continue;
      const rhi::ClearColor clear = ColorTargetClearValue(
          d.mrt_info[i], d.mrt_clear_word[i][0], d.mrt_clear_word[i][1]);
      it->second.clear_pending = true;
      it->second.clear_src = "clear-rect";
      it->second.clear_value = clear;
      if (kClearTrace) {
        static int n = 0;
        if (n++ < kClearTrace)
          BASE_LOGI(
              "clear",
              "f{} draw#{} rect RT {:#x} info={:#x} cc={:#x} mode={} gen="
              "({},{})-({},{}) win={:#x}/{:#x} scr={:#x}/{:#x} target={}x{} "
              "CLEAR_WORD {:08x} {:08x} -> ({:g} {:g} {:g} {:g})",
              g_frame.num, g_frame.draws, (unsigned long)d.mrt_base[i],
              d.mrt_info[i], d.color_control, (d.color_control >> 4) & 7u, cx0,
              cy0, cx1, cy1, d.clear_window_tl, d.clear_window_br,
              d.clear_screen_tl, d.clear_screen_br, d.rt_w, d.rt_h,
              d.mrt_clear_word[i][0], d.mrt_clear_word[i][1], clear.f[0],
              clear.f[1], clear.f[2], clear.f[3]);
      }
    }
    if (d.depth_base) {
      auto dt = g_depths.find(d.depth_base);
      if (dt != g_depths.end()) {
        dt->second.clear_pending = true;
        dt->second.clear_value = d.depth_clear;
      }
    }
    WhyDrop(d, "clear-consumed");
    g_frame.draws++;
    return true;
  }

  if (kUiWatch) {
    static bool armed = false;
    if (!armed) {
      for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
        const auto& attr = d.vattrs[a];
        if (attr.dfmt != 10 || attr.nfmt != 0 || attr.num_comps != 4)
          continue;
        if (attr.binding >= d.num_vbufs)
          continue;
        const auto& vb = d.vbufs[attr.binding];
        const auto* p = static_cast<const u8*>(vb.data);
        if (!p || !host_memory::IsMemoryRangeMapped(p + attr.offset, 4))
          continue;
        u32 c0 = 0;
        std::memcpy(&c0, p + attr.offset, 4);
        if (c0 != 0)
          continue;
        armed = true;
        const uintptr_t at = reinterpret_cast<uintptr_t>(p) + attr.offset;
        BASE_LOGI("uiwatch",
                  "arming on UI colour {:#x} (rt={:#x}, {} verts, "
                  "stride {}), it currently reads 00000000",
                  (unsigned long)at, (unsigned long)d.rt_base, d.vertex_count,
                  vb.stride);
        write_watch::SetValueProbe(at);
        write_watch::SetChase(4);
        if (!write_watch::Arm(at & ~0xFFFull, 0x1000, 200))
          BASE_LOGI("uiwatch", "no armer registered");
        break;
      }
    }
  }

  if (kLazyClear2 && d.rt_base) {
    auto it = g_rts.find(d.rt_base);
    const bool accumulate =
        d.blend_enable && ((d.blend_control >> 8) & 0x1F) == 1u;
    if (it != g_rts.end()) {
      if (accumulate && it->second.last_frame != g_frame.num &&
          it->second.ever_rendered && it->second.accumulated) {
        it->second.clear_pending = true;
        it->second.clear_src = "accumulate-needs-reset";
        it->second.clear_value = rhi::ClearColor{};
      }
      it->second.accumulated = accumulate;
    }
  }
  {
    static const u64 kVtxRt = [] {
      const char* e = std::getenv("DELTA_GPU_VTXTRACE_RT");
      return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    static const auto kVtxStart = base::TimeTicks::Now();
    static const int kVtxAfter = [] {
      const char* e = std::getenv("DELTA_GPU_VTXTRACE_AFTER");
      return e ? std::atoi(e) : 0;
    }();
    static int vtx_n = 0;
    if (kVtxRt && d.rt_base == kVtxRt && vtx_n < 40 &&
        (base::TimeTicks::Now() - kVtxStart).InSeconds() >= kVtxAfter) {
      vtx_n++;
      base::String line;
      base::FormatTo(line, "f{} draw#{} rt={:#x} nv={} stride={} attrs={}",
                     g_frame.num, g_frame.draws, (unsigned long)d.rt_base, nv,
                     d.vertex_stride, d.num_vattrs);
      for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
        const auto& va = d.vattrs[a];
        base::FormatTo(line, " |loc={} dfmt={} nfmt={} nc={} off={} bind={}",
                       va.location, va.dfmt, va.nfmt, va.num_comps, va.offset,
                       va.binding);
        const auto& vb = d.vbufs[va.binding];
        const u64 avail = static_cast<u64>(vb.stride) * vb.num_records;
        if (vb.data && avail >= va.offset + sizeof(u32) * 4) {
          const auto* p = static_cast<const u8*>(vb.data) + va.offset;
          u32 w[4] = {};
          std::memcpy(w, p, sizeof(w));
          base::FormatTo(
              line, " raw={:08x} {:08x} {:08x} {:08x} f={:g} {:g} {:g} {:g}",
              w[0], w[1], w[2], w[3], *reinterpret_cast<const float*>(&w[0]),
              *reinterpret_cast<const float*>(&w[1]),
              *reinterpret_cast<const float*>(&w[2]),
              *reinterpret_cast<const float*>(&w[3]));
        }
      }
      BASE_LOGI("vtx", "{}", line.c_str());
    }
  }

  const u32 nbind = d.num_vattrs ? base::Min(d.num_vbufs, 8u) : 0;
  u64 bind_off[8] = {}, bind_size[8] = {};
  u64 vb_cached[8] = {};
  CachedBuffer vb_kept[8] = {};
  CachedBuffer vb_bind[8] = {};
  u64 vneed = 0;
  for (u32 j = 0; j < nbind; j++) {
    vb_cached[j] = u64(-1);
    if (indexed && !descriptor_bounds && d.vbufs[j].stride &&
        !d.vbufs[j].per_instance) {
      const u64 bytes = u64(d.vbufs[j].num_records) * d.vbufs[j].stride;
      if (BindVertexDescriptor(reinterpret_cast<u64>(d.vbufs[j].data), bytes,
                               vb_kept[j], vb_bind[j])) {
        bind_size[j] = bytes;
        continue;
      }
      if (!index_scanned) {
        if (!FlushCsWritesRange(
                renderer, reinterpret_cast<u64>(d.index_data),
                u64(d.index_count) * GuestIndexElementBytes(d.index_type),
                "index"))
          return vtx_decline("cs-index", d.index_count);
        const StageCache::Entry* staged =
            kRingDedup ? g_ib_staged.Find(reinterpret_cast<u64>(d.index_data),
                                          index_bytes, ib_tag)
                       : nullptr;
        u32 max_index;
        if (staged && staged->bytes == index_bytes &&
            staged->max_index != ~0u) {
          max_index = staged->max_index;
        } else {
          ScopeNs scan_timer(&g_stats.ns_index_scan);
          max_index = MaxGuestIndex(d.index_data, d.index_count, d.index_type);
          g_stats.index_scan_n++;
          g_stats.index_scan_bytes +=
              u64(d.index_count) * GuestIndexElementBytes(d.index_type);
        }
        if (max_index >= 200000u)
          return vtx_decline("max-index", max_index);
        nv = max_index + 1;
        index_scanned = true;
      }
    }
    if (d.vbufs[j].stride) {
      const u32 records = descriptor_bounds ? d.vbufs[j].num_records
                          : d.vbufs[j].per_instance
                              ? base::Max(1u, d.instance_count)
                              : nv;
      bind_size[j] = (u64)records * d.vbufs[j].stride;
    } else {
      u32 rec = 0;
      for (u32 a = 0; a < d.num_vattrs; a++)
        if (d.vattrs[a].binding == j)
          rec = base::Max(
              rec, d.vattrs[a].offset + VertexFormatBytes(d.vattrs[a].dfmt));
      bind_size[j] = rec;
    }
    if (bind_size[j] &&
        CsGeometryBuffer(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j],
                         &vb_kept[j].buffer, &vb_kept[j].offset))
      continue;
    if (bind_size[j] &&
        CsHoldsGeometry(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j]))
      return Decline(kRing);
    if (kRingDedup && bind_size[j] && TrackingGuestWrites()) {
      SyncGuestWrites();
      if (FindCachedBuffer(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j],
                           vb_kept[j]))
        continue;
    }
    vb_cached[j] =
        kRingDedup && bind_size[j]
            ? StagedOffset(g_vb_staged.Find(
                  reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j]))
            : u64(-1);
    if (vb_cached[j] != u64(-1))
      continue;
    if (bind_size[j] && kDirectBind) {
      u64 inner = 0;
      if (rhi::Buffer* b = DirectBindAt(reinterpret_cast<u64>(d.vbufs[j].data),
                                        bind_size[j], &inner)) {
        vb_bind[j] = {b, inner};
        g_stats.bind_bytes += bind_size[j];
        g_stats.bind_n++;
        continue;
      }
    }
    if (bind_size[j] &&
        !IsReadableThisFrame(reinterpret_cast<u64>(d.vbufs[j].data),
                             static_cast<u32>(bind_size[j])))
      return Decline(kNoRecomp);
    if (vneed)
      vneed = (vneed + 15) & ~u64(15);
    bind_off[j] = vneed;
    vneed += bind_size[j];
  }
  bool float_pos = false;
  for (u32 a = 0; a < d.num_vattrs; a++) {
    if (d.vattrs[a].location != 0)
      continue;
    const u32 df = d.vattrs[a].dfmt;
    float_pos =
        d.vattrs[a].nfmt == 7 && (df == 4 || df == 11 || df == 13 || df == 14);
    break;
  }
  if (kNoWipe && float_pos && d.vertex_data && d.num_vattrs &&
      d.recomp->ps_texs.empty() && (!indexed || index_scanned) && nv <= 8) {
    u32 cdst = (d.blend_control >> 8) & 0x1F, csrc = d.blend_control & 0x1F;
    bool replace = d.blend_enable && csrc == 1 && cdst == 0;
    if (replace) {
      const auto* vb = static_cast<const u8*>(d.vertex_data);
      bool near_black = true;
      float clear_color[4] = {0, 0, 0, 0};
      for (u32 a = 0; a < d.num_vattrs; a++) {
        if (d.vattrs[a].num_comps == 4 && d.vattrs[a].offset != 0) {
          const auto* cbuf =
              static_cast<const u8*>(d.vbufs[d.vattrs[a].binding].data);
          const u8* cb0 = cbuf + d.vattrs[a].offset;
          if (d.vattrs[a].dfmt == 10) {
            for (int i = 0; i < 4; i++)
              clear_color[i] = cb0[i] / 255.f;
          } else {
            const float* c = reinterpret_cast<const float*>(cb0);
            for (int i = 0; i < 4; i++)
              clear_color[i] = c[i];
          }
          if (clear_color[0] > 0.02f || clear_color[1] > 0.02f ||
              clear_color[2] > 0.02f)
            near_black = false;
          break;
        }
      }
      const float* m = d.mvp;
      float nx0 = 1e9f, ny0 = 1e9f, nx1 = -1e9f, ny1 = -1e9f;
      for (u32 v = 0; v < nv; v++) {
        const float* p =
            reinterpret_cast<const float*>(vb + (size_t)v * d.vertex_stride);
        float cw = m[3] * p[0] + m[7] * p[1] + m[15];
        if (cw == 0)
          cw = 1;
        float nx = (m[0] * p[0] + m[4] * p[1] + m[12]) / cw,
              ny = (m[1] * p[0] + m[5] * p[1] + m[13]) / cw;
        nx0 = nx < nx0 ? nx : nx0;
        nx1 = nx > nx1 ? nx : nx1;
        ny0 = ny < ny0 ? ny : ny0;
        ny1 = ny > ny1 ? ny : ny1;
      }
      bool fullscreen_black =
          near_black && (nx1 - nx0) >= 1.8f && (ny1 - ny0) >= 1.8f;
      if (fullscreen_black && kLazyClear2) {
        RTarget* rt = d.rt_base
                          ? GetRT(d.rt_base,
                                  RtSurfaceExtent(d.mrt_surf_w[0], d.rt_w, 256),
                                  RtSurfaceExtent(d.mrt_surf_h[0], d.rt_h, 64),
                                  ColorTargetFormat(d.mrt_info[0]))
                          : nullptr;
        if (rt) {
          if (kClearTrace) {
            static base::Atomic<u64> n{0};
            if ((n.fetch_add(1) % 500) == 0)
              BASE_LOGI("clear",
                        "lazyclear-heuristic #{} rt={:#x} mrt={} "
                        "mrt1={:#x}",
                        (unsigned long long)n.load(), (unsigned long)d.rt_base,
                        d.mrt_count,
                        (unsigned long)(d.mrt_count > 1 ? d.mrt_base[1] : 0));
          }
          rt->clear_pending = true;
          rt->clear_src = "lazyclear-heuristic";
          std::memcpy(rt->clear_value.f, clear_color, sizeof(clear_color));
          if (kClearTrace) {
            static int n = 0;
            if (n++ < kClearTrace)
              BASE_LOGI("clear",
                        "f{} draw#{} LAZYCLEAR-HEURISTIC RT {:#x} "
                        "vs={:#x} ps={:#x} color=({:g} {:g} {:g} {:g}) "
                        "blend={}/{:#x}",
                        g_frame.num, g_frame.draws, (unsigned long)d.rt_base,
                        (unsigned long)d.vs_addr, (unsigned long)d.ps_addr,
                        clear_color[0], clear_color[1], clear_color[2],
                        clear_color[3], (int)d.blend_enable, d.blend_control);
          }
        }
        if (d.depth_base && d.depth_write_enable && d.depth_func == 7) {
          DepthTarget* dt = GetDepthRT(d.depth_base, d.rt_w, d.rt_h);
          if (dt) {
            dt->clear_pending = true;
            dt->clear_value = d.depth_clear;
          }
        }
        WhyDrop(d, "lazyclear");
        g_frame.draws++;
        return true;
      }
      auto rit = g_rts.find(d.rt_base);
      if (fullscreen_black && rit != g_rts.end() && rit->second.draws > 0 &&
          rit->second.last_frame == g_frame.num) {
        WhyDrop(d, "fullscreen-black");
        g_frame.draws++;
        return true;
      }
    }
  }

  if (g_ring.vb_offset + vneed > g_ring.vb_end) {
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 32)
        BASE_LOGI("decl", "ring=VB need={} off={} end={} idx={}",
                  (unsigned long long)vneed,
                  (unsigned long long)g_ring.vb_offset,
                  (unsigned long long)g_ring.vb_end, d.index_count);
    }
    return Decline(kRing);
  }
  const u64 index_align =
      convert_indices || auto_quads || d.index_type == 1 ? 4 : 2;
  CachedBuffer ib_bind{};
  bool ib_gpu = false;
  if (indexed && !quads && d.index_type != 2) {
    ib_gpu = CsGeometryBuffer(reinterpret_cast<u64>(d.index_data), index_bytes,
                              &ib_bind.buffer, &ib_bind.offset);
    if (!ib_bind.buffer &&
        CsHoldsGeometry(reinterpret_cast<u64>(d.index_data), index_bytes))
      return Decline(kRing);
    if (!ib_bind.buffer && kDirectBind)
      ib_bind.buffer = DirectBindAt(reinterpret_cast<u64>(d.index_data),
                                    index_bytes, &ib_bind.offset);
    if (ib_bind.offset % index_align) {
      ib_bind = {};
      ib_gpu = false;
    }
    if (ib_bind.buffer) {
      g_stats.bind_bytes += index_bytes;
      g_stats.bind_n++;
    }
  }
  const u64 ib_cached =
      kRingDedup && indexed && !ib_bind.buffer && !indirect_quads
          ? StagedOffset(g_ib_staged.Find(reinterpret_cast<u64>(d.index_data),
                                          index_bytes, ib_tag))
          : u64(-1);
  const u64 index_revision =
      convert_indices
          ? CsBufferRevision(
                reinterpret_cast<u64>(d.index_data),
                u64(d.index_count) * GuestIndexElementBytes(d.index_type))
          : 0;
  const u64 aligned_ioff =
      (g_ring.ib_offset + index_align - 1) & ~(index_align - 1);
  const bool ring_indices =
      (indexed && ib_cached == u64(-1) && !ib_bind.buffer) || auto_quads;
  if (ring_indices &&
      aligned_ioff + index_bytes + (indirect_quads ? 20 : 0) > g_ring.ib_end) {
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 32)
        BASE_LOGI("decl", "ring=IB need={} off={} end={} idx={}",
                  (unsigned long long)index_bytes,
                  (unsigned long long)aligned_ioff,
                  (unsigned long long)g_ring.ib_end, d.index_count);
    }
    return Decline(kRing);
  }

  g_stats.ns_dr_pre += NowNs() - t_draw_start;
  const u64 t_pipe = NowNs();
  RecompPipe* rp = GetRecompPipe(d);
  g_stats.ns_dr_pipe += NowNs() - t_pipe;
  if (!rp || rp->failed)
    return Decline(kNoPipe);
  if (!rp->pipe) {
    g_stats.pipe_pending_draws++;
    return true;
  }
  const u64 t_tex = NowNs();
  rhi::BindGroup* tex_set = nullptr;
  if (rp->textured && !rp->multi_tex && !rt_as_tex) {
    if (!force_white_tex && GuestTextureUploadSupported(d.tex_dfmt, d.tex_nfmt))
      tex_set = (GetTexture(
          d.tex_base, d.tex_w, d.tex_h, d.tex_dfmt, d.tex_nfmt, d.tex_tiling,
          d.tex_pitch, d.tex_layers, d.tex_base_array, d.tex_view_layers,
          d.tex_mip_levels, d.tex_base_mip, d.tex_view_mips, d.tex_min_lod,
          d.tex_pow2_pad, d.tex_sampler, d.tex_sampler_valid, d.tex_arrayed,
          d.tex_force_lod_zero, d.tex_depth_compare, d.tex_swizzle, d.tex_depth,
          d.tex_is_3d));
    if (!tex_set)
      tex_set = d.tex_null_descriptor ? (d.tex_is_3d     ? g_tex.zero_3d_set
                                         : d.tex_arrayed ? g_tex.zero_array_set
                                                         : g_tex.zero_set)
                                      : (d.tex_is_3d     ? g_tex.white_3d_set
                                         : d.tex_arrayed ? g_tex.white_array_set
                                                         : g_tex.white_set);
    if (!tex_set)
      return Decline(kGuestTex);
  }

  u64 multi_color[kMaxTex] = {};
  u64 multi_depth[kMaxTex] = {};
  u64 multi_stencil_src[kMaxTex] = {};
  u64 multi_feedback[kMaxTex] = {};
  u64 multi_storage[kMaxTex] = {};
  rhi::TextureView* multi_views[kMaxTex] = {};
  rhi::TextureState multi_layouts[kMaxTex];
  bool multi_transition_source = false;
  u32 multi_n = base::Min(d.num_texs, kMaxTex);
  if (rp->multi_tex) {
    auto is_bound_target = [&](u64 base) {
      u32 count = base::Min(d.mrt_count, 8u);
      for (u32 m = 0; m < count; m++)
        if (d.mrt_base[m] == base)
          return true;
      return false;
    };
    for (u32 i = 0; i < kMaxTex; i++)
      multi_layouts[i] = rhi::TextureState::kShaderRead;
    for (u32 i = 0; i < multi_n; i++) {
      const auto& t = d.texs[i];
      u64 base = t.base;
      if (t.storage) {
        if (base && !g_rts.count(base)) {
          u64 resolved = ResolveSampledRT(base, t.w, t.h);
          if (resolved)
            base = resolved;
        }
        if (base && !g_rts.count(base)) {
          const rhi::Format format = GuestTextureFormat(t.dfmt, t.nfmt);
          if (format != rhi::Format::kUndefined)
            GetRT(base, t.w, t.h, format);
        }
        if (base && g_rts.count(base)) {
          multi_storage[i] = base;
          multi_transition_source |=
              g_rts[base].layout != rhi::TextureState::kGeneral;
        }
        continue;
      }
      const bool rt_eligible =
          (!t.arrayed ||
           (t.layers == 1 && t.base_array == 0 && t.view_layers == 1)) &&
          !t.is_3d && t.base_mip == 0 && !GuestFormatBlockCompressed(t.dfmt);
      if (base && rt_eligible && !is_bound_target(base)) {
        ActivateSampledRtVariant(base, t.w, t.h);
        if (base != d.depth_base)
          ActivateSampledDepthVariant(base, t.w, t.h);
      }
      if (base && rt_eligible && !g_rts.count(base) && !g_depths.count(base)) {
        bool depth_format = t.dfmt == 4 && t.nfmt == 7;
        u64 resolved = depth_format ? ResolveSampledDepth(base, t.w, t.h) : 0;
        if (!resolved)
          resolved = ResolveSampledRT(base, t.w, t.h);
        if (!resolved && !depth_format)
          resolved = ResolveSampledDepth(base, t.w, t.h);
        if (resolved) {
          base = resolved;
          if (base != d.depth_base)
            ActivateSampledDepthVariant(base, t.w, t.h);
        }
      }
      if (kTexForce && t.base == (u64)kTexForce) {
        static int forced = 0;
        if (forced++ < 6)
          BASE_LOGI("texforce", "ps={:#x} bind={} base={:#x} -> white",
                    (unsigned long)d.ps_addr, i, (unsigned long)t.base);
        base = 0;
        multi_color[i] = 0;
        multi_feedback[i] = 0;
        multi_depth[i] = 0;
        multi_views[i] = nullptr;
        continue;
      }
      if (base && t.is_3d && !t.storage &&
          ActivateVolumeRt(base, t.w, t.h, t.depth)) {
        multi_color[i] = base;
        multi_transition_source |=
            g_rts[base].layout != rhi::TextureState::kShaderRead ||
            g_rts[base].dirty_for_read;
        continue;
      }
      if (base && g_depths.count(base))
        CsRefreshDepthFromCs(base);
      const bool samples_target = base && g_rts.count(base);
      if (samples_target && !CsRefreshRtFromTruth(base))
        render::FlushCsWritesAt(renderer, base, "rt-tex");
      const u64 stencil_src = FresherStencil(base, t, rt_eligible);
      if (stencil_src) {
        multi_stencil_src[i] = stencil_src;
        multi_transition_source |= g_depths[stencil_src].stencil_layout !=
                                    rhi::TextureState::kDepthRead;
      } else if (base && rt_eligible && is_bound_target(base) &&
                 g_rts.count(base) && g_rts[base].ever_rendered) {
        multi_feedback[i] = base;
        multi_transition_source = true;
      } else if (base && rt_eligible && g_rts.count(base) &&
                 g_rts[base].ever_rendered) {
        multi_color[i] = base;
        multi_transition_source |=
            g_rts[base].layout != rhi::TextureState::kShaderRead ||
            g_rts[base].dirty_for_read;
      } else if (base && rt_eligible && !t.arrayed && g_depths.count(base) &&
                 (base != d.depth_base || !d.depth_write_enable)) {
        multi_depth[i] = base;
        multi_transition_source |=
            g_depths[base].layout != rhi::TextureState::kDepthRead;
      } else if (base && rt_eligible && !t.arrayed &&
                 ResolveSampledStencil(base)) {
        multi_stencil_src[i] = ResolveSampledStencil(base);
        multi_transition_source |=
            g_depths[multi_stencil_src[i]].stencil_layout !=
            rhi::TextureState::kDepthRead;
      } else {
        multi_views[i] = !t.storage && t.null_descriptor
                             ? (t.is_3d     ? g_tex.zero_3d_view
                                : t.arrayed ? g_tex.zero_array_view
                                            : g_tex.zero_view)
                             : TexViewFor(t);
        if (multi_views[i] && t.null_descriptor)
          multi_layouts[i] = rhi::TextureState::kShaderRead;
      }
    }
    if (kBindTrace && d.ps_addr == (u64)kBindTrace) {
      static int n = 0;
      if (n++ < 24) {
        for (u32 i = 0; i < multi_n; i++)
          BASE_LOGI("bind", "ps={:#x} b{} base={:#x} {}x{} -> {}",
                    (unsigned long)d.ps_addr, i, (unsigned long)d.texs[i].base,
                    d.texs[i].w, d.texs[i].h,
                    multi_storage[i]    ? "storage"
                    : multi_feedback[i] ? "feedback"
                    : multi_color[i]    ? "rt-color"
                    : multi_depth[i]    ? "rt-depth"
                                        : "GUEST-TEXTURE");
      }
    }
    for (u32 i = 0; i < multi_n; i++) {
      if (!multi_color[i])
        continue;
      for (u32 j = 0; j < multi_n; j++)
        if (multi_storage[j] == multi_color[i]) {
          multi_layouts[i] = rhi::TextureState::kGeneral;
          break;
        }
    }
  }

  {
    static int texbind_lines = 0;
    const bool texbind = kTexBindFrame >= 0 &&
                         (int)g_frame.num >= kTexBindFrame &&
                         texbind_lines < 600 && (texbind_lines++, true);
    if (texbind)
      BASE_LOGI("blend",
                "f{} draw#{} rt={:#x} vs={:#x} ps={:#x} blend={} ctl={:#x} "
                "tmask={:#x} smask={:#x} mrt={} idx={} depth={:#x} dw={}",
                g_frame.num, g_frame.draws, (unsigned long)d.rt_base,
                (unsigned long)d.vs_addr, (unsigned long)d.ps_addr,
                (int)d.blend_enable, d.blend_control, d.target_mask,
                d.shader_mask, d.mrt_count, d.index_count,
                (unsigned long)d.depth_base, (int)d.depth_write_enable);
    if (texbind && !rp->multi_tex)
      BASE_LOGI("texbind",
                "draw#{} rt={:#x} {}x{} LEGACY tex={:#x} {}x{} "
                "rtAsTex={} color={} feedback={} depth={} set={}",
                g_frame.draws, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
                (unsigned long)d.tex_base, d.tex_w, d.tex_h,
                (unsigned)rt_as_tex, (unsigned)color_as_tex,
                (unsigned)feedback_as_tex, (unsigned)depth_as_tex,
                (unsigned)(tex_set != nullptr));
    if (texbind && multi_n && rp->multi_tex) {
      base::String line;
      base::FormatTo(line, "draw#{} rt={:#x} {}x{} ntex={}:", g_frame.draws,
                     (unsigned long)d.rt_base, d.rt_w, d.rt_h, multi_n);
      for (u32 i = 0; i < multi_n; i++) {
        const char* how = multi_color[i]      ? "RT"
                          : multi_feedback[i] ? "feedback"
                          : multi_depth[i]    ? "depth"
                          : multi_storage[i]  ? "storage"
                          : multi_views[i]    ? "guest"
                                              : "WHITE";
        base::FormatTo(
            line,
            " [{}]{}@{:#x} {}x{} layers={} mips={} fmt={}/{} rt?={} er={}", i,
            how, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
            d.texs[i].layers, d.texs[i].mip_levels, d.texs[i].dfmt,
            d.texs[i].nfmt, (unsigned)g_rts.count(d.texs[i].base),
            (unsigned)(g_rts.count(d.texs[i].base)
                           ? g_rts[d.texs[i].base].ever_rendered
                           : 0));
      }
      BASE_LOGI("texbind", "{}", line.c_str());
    }
  }

  const u32 cbufs_used = rp->cbuf_count;
  CachedBuffer cb_direct[kCbufBindings]{};
  bool cb_barrier = false;
  u64 cb_gpu_off[gpu::gcn::kIndirectCbufBindings];
  base::Fill(cb_gpu_off, cb_gpu_off + base::ArraySize(cb_gpu_off), u64(-1));
  bool cb_upload = d.recomp->indirect_cbufs;
  if (cb_upload && !EnsureCbufRing())
    return Decline(kRing);
  for (u32 i = 0; i < cbufs_used; i++) {
    if (!(rp->cbuf_mask & (1ull << i)))
      continue;
    const auto& cb = d.cbufs[i];
    if (!base::Min(cb.size, kCbufWindow))
      continue;
    const u32 n = CbufCopyBytes(cb.base, cb.size);
    const u64 rev = CsBufferRevision(cb.base, n);
    if (kCbufBind && !d.recomp->indirect_cbufs) {
      CachedBuffer binding;
      if (!CsStorageBuffer(cb.base, n, &binding.buffer, &binding.offset,
                           g_ring.ubo_align, kCbufWindow) &&
          n == kCbufWindow && kDirectBind && !rev) {
        binding.buffer = DirectBindAt(cb.base, n, &binding.offset);
        cb_barrier |= binding.buffer && CsRangeDirtyOverlapping(cb.base, n);
      }
      if (binding.buffer &&
          (binding.buffer->desc().usage & rhi::kBufferUniform) &&
          !(binding.offset % g_ring.ubo_align)) {
        cb_direct[i] = binding;
        continue;
      }
    }
    if (!rev)
      continue;
    GpuStaged& staged = g_ubo_gpu[cb.base];
    if (staged.frame == g_frame.num && staged.rev == rev && staged.bytes >= n) {
      cb_gpu_off[i] = staged.off;
      continue;
    }
    if (!EnsureCbufRing())
      return Decline(kRing);
    const u64 off = (g_ring.ubo_offset + g_ring.ubo_align - 1) &
                    ~(u64)(g_ring.ubo_align - 1);
    if (off + n > g_ring.ubo_end || off + kCbufWindow > UboRingBytes() ||
        !CsSupplyBuffer(cb.base, n, g_ring.ubo_buf, off))
      continue;
    g_ring.ubo_offset = off + n;
    staged = {off, n, rev, g_frame.num};
    cb_gpu_off[i] = off;
  }
  for (u32 i = 0; i < cbufs_used && !cb_upload; ++i) {
    if (!(rp->cbuf_mask & (1ull << i)))
      continue;
    if (cb_direct[i].buffer || cb_gpu_off[i] != u64(-1))
      continue;
    const auto& cb = d.cbufs[i];
    const u32 readable = base::Min(cb.size, kCbufWindow);
    cb_upload = i == 0 || (readable && IsReadableThisFrame(cb.base, readable));
  }
  if (cb_upload) {
    if (!EnsureCbufRing())
      return Decline(kRing);
  } else if (cbufs_used && !g_ring.ubo_buf && !EnsureCbufZero() &&
             !EnsureCbufRing()) {
    return Decline(kRing);
  }

  CachedBuffer raw_direct[kRawBufBindings]{};
  bool raw_barrier = false;
  if (rp->raw_count && kRawBind) {
    const u64 align = base::Max(Device().caps().storage_offset_alignment, 1u);
    for (u32 i = 0; i < rp->raw_count; i++) {
      if (!(rp->raw_mask & (1u << i)))
        continue;
      const auto& rb = d.bufs[i];
      const u32 want = base::Min(rb.size, kRawBufWindow);
      if (!want || (want & 3))
        continue;
      CachedBuffer binding;
      if (!CsStorageBuffer(rb.base, want, &binding.buffer, &binding.offset,
                           align) &&
          kDirectBind && !CsBufferRevision(rb.base, want)) {
        binding.buffer = DirectBindAt(rb.base, want, &binding.offset);
        raw_barrier |= binding.buffer && CsRangeDirtyOverlapping(rb.base, want);
      }
      if (binding.buffer && !(binding.offset % align))
        raw_direct[i] = binding;
    }
  }

  {
    for (u32 i = 0; i < cbufs_used; i++) {
      if (!(rp->cbuf_mask & (1ull << i)))
        continue;
      const auto& cb = d.cbufs[i];
      const u32 readable = base::Min(cb.size, kCbufWindow);
      if (i < kCbufBindings && cb_direct[i].buffer) {
        const u32 n = CbufCopyBytes(cb.base, cb.size);
        if (!CsBufferRevision(cb.base, n) &&
            !FlushCsWritesForGpu(renderer, cb.base, n, "cb"))
          return Decline(kNoRecomp);
        continue;
      }
      if (cb_gpu_off[i] == u64(-1) && readable &&
          IsReadableThisFrame(cb.base, readable) &&
          !FlushCsWritesRange(renderer, cb.base,
                              CbufCopyBytes(cb.base, cb.size), "cb"))
        return Decline(kNoRecomp);
    }
    if (rp->raw_count)
      for (u32 i = 0; i < rp->raw_count; i++) {
        if (!(rp->raw_mask & (1u << i)))
          continue;
        const auto& rb = d.bufs[i];
        const u32 want = base::Min(rb.size, kRawBufWindow);
        if (want && !CsBufferRevision(rb.base, want) &&
            !(raw_direct[i].buffer
                  ? FlushCsWritesForGpu(renderer, rb.base, want, "raw")
                  : FlushCsWritesRange(renderer, rb.base, want, "raw")))
          return Decline(kNoRecomp);
      }
    for (u32 j = 0; j < nbind; j++) {
      if (!bind_size[j] || vb_cached[j] != u64(-1) || vb_kept[j].buffer)
        continue;
      const u64 base = reinterpret_cast<u64>(d.vbufs[j].data);
      if (!(vb_bind[j].buffer
                ? FlushCsWritesForGpu(renderer, base, bind_size[j], "vb")
                : FlushCsWritesRange(renderer, base, bind_size[j], "vb")))
        return Decline(kNoRecomp);
    }
    if (indexed && ib_cached == u64(-1) && !ib_gpu && !convert_indices) {
      const u64 base = reinterpret_cast<u64>(d.index_data);
      const u64 bytes =
          u64(d.index_count) * GuestIndexElementBytes(d.index_type);
      if (!(ib_bind.buffer
                ? FlushCsWritesForGpu(renderer, base, bytes, "index")
                : FlushCsWritesRange(renderer, base, bytes, "index")))
        return Decline(kNoRecomp);
    }
  }
  bool reads_synced = false;
  if ((ib_bind.buffer || raw_barrier || cb_barrier ||
       base::AnyOf(
           vb_bind, vb_bind + nbind,
           [](const CachedBuffer& b) { return b.buffer != nullptr; })) &&
      !(g_region.open && g_region.reads_synced)) {
    EndRegion();
    g_frame.list->Barrier(rhi::kAccessAllWrite,
                          rhi::kAccessVertexRead | rhi::kAccessIndexRead |
                              rhi::kAccessUniformRead | rhi::kAccessShaderRead);
    reads_synced = true;
  }
  rhi::Buffer* indirect_buf = nullptr;
  u64 indirect_off = 0;
  if (d.indirect_args) {
    const u64 bytes = indexed ? 20 : 16;
    EndRegion();
    if (!CsResidentGeometryBuffer(d.indirect_args, bytes, &indirect_buf,
                                  &indirect_off) ||
        (indirect_off & 3)) {
      indirect_buf = nullptr;
      if (GpuIndirectArgs(d.indirect_args, bytes))
        indirect_buf = IndirectArgsBuffer(d.indirect_args, &indirect_off);
      else if (FlushCsWritesForGpu(renderer, d.indirect_args, bytes,
                                   "indirect"))
        indirect_buf = StageIndirectArgs(g_frame.list, d.indirect_args, bytes,
                                         &indirect_off);
    }
    if (!indirect_buf)
      return Decline(kNoRecomp);
    if (!mesh)
      indirect_buf = ZeroIndirectBases(g_frame.list, indirect_buf,
                                       indirect_off, indexed, &indirect_off);
    if (!mesh)
      g_frame.list->Barrier(rhi::kAccessAllWrite, rhi::kAccessIndirectRead);
  }
  rhi::Buffer* mesh_tasks = nullptr;
  u64 mesh_tasks_offset = 0;
  u32 mesh_argument_word = 0;
  if (mesh && !PrepareMeshArgs(indirect_buf, indirect_off,
                               d.recomp->mesh_input_primitives, &mesh_tasks,
                               &mesh_tasks_offset, &mesh_argument_word))
    return Decline(kNoRecomp);
  if (convert_indices && ib_cached == u64(-1) &&
      !ConvertGuestIndices(renderer, reinterpret_cast<u64>(d.index_data),
                           d.index_count, d.index_type, quads, aligned_ioff,
                           indirect_quads ? &indirect_buf : nullptr,
                           indirect_quads ? &indirect_off : nullptr))
    return Decline(kNoRecomp);

  u64 raw_gpu_off[kRawBufBindings];
  base::Fill(raw_gpu_off, raw_gpu_off + base::ArraySize(raw_gpu_off), u64(-1));
  if (rp->raw_count) {
    for (u32 i = 0; i < rp->raw_count; i++) {
      if (!(rp->raw_mask & (1u << i)))
        continue;
      const auto& rb = d.bufs[i];
      const u32 want = base::Min(rb.size, kRawBufWindow);
      const u64 rev = want ? CsBufferRevision(rb.base, want) : 0;
      if (!rev || raw_direct[i].buffer)
        continue;
      if (!EnsureRawBufferRing())
        return Decline(kRing);
      GpuStaged& staged = g_sbo_gpu[rb.base];
      if (staged.frame == g_frame.num && staged.rev == rev &&
          staged.bytes >= want) {
        raw_gpu_off[i] = staged.off;
        continue;
      }
      const u64 off = (g_ring.sbo_offset + g_ring.sbo_align - 1) &
                      ~(u64)(g_ring.sbo_align - 1);
      const u64 reserve =
          rb.size > kRawBufWindow
              ? g_ring.sbo_stride
              : ((want + g_ring.sbo_align - 1) & ~(u64)(g_ring.sbo_align - 1));
      if (off + reserve > g_ring.sbo_end || off + kRawBufWindow > kSboRing ||
          !CsSupplyBuffer(rb.base, want, g_ring.sbo_buf, off))
        continue;
      if (want < reserve)
        std::memset(g_ring.sbo_map + off + want, 0,
                    static_cast<size_t>(reserve - want));
      g_ring.sbo_offset = off + reserve;
      staged = {off, want, rev, g_frame.num};
      raw_gpu_off[i] = off;
    }
    if (!g_ring.sbo_buf) {
      bool staging = false;
      for (u32 i = 0; i < rp->raw_count; ++i) {
        if (!(rp->raw_mask & (1u << i)))
          continue;
        const auto& rb = d.bufs[i];
        const u32 want = base::Min(rb.size, kRawBufWindow);
        if (want && !raw_direct[i].buffer && raw_gpu_off[i] == u64(-1) &&
            IsReadableThisFrame(rb.base, want)) {
          staging = true;
          break;
        }
      }
      if ((staging || !EnsureRawBufferZero()) && !EnsureRawBufferRing())
        return Decline(kRing);
    }
  }
  u64 voff = g_ring.vb_offset, ioff = aligned_ioff;

  if (d.recomp->direct_reads)
    PrepareDirectGuestRead();

  if (d.rt_array_base) {
    static base::HashMap<u64, int> written_frame;
    static const guest::SessionReset reset_written_frame(
        [] { guest::ResetResource(written_frame); });
    const auto first = g_rts.find(d.rt_array_base);
    int& frame = written_frame[d.rt_array_base];
    if (first != g_rts.end() && first->second.last_frame == g_frame.num &&
        frame != g_frame.num) {
      frame = g_frame.num;
      WriteRtToGuest(d.rt_array_base, d.rt_tile_mode);
    }
  }
  u32 mrt_n = base::Min(d.mrt_count, 8u);
  bool transition_source =
      rp->multi_tex
          ? multi_transition_source
          : feedback_as_tex ||
                (color_as_tex &&
                 g_rts[tex_base].layout != rhi::TextureState::kShaderRead) ||
                (depth_as_tex &&
                 g_depths[tex_base].layout != rhi::TextureState::kDepthRead);
  bool pending_depth_clear = d.depth_base && g_depths.count(d.depth_base) &&
                             g_depths[d.depth_base].clear_pending;
  bool samples_bound_depth = depth_self_read;
  for (u32 i = 0; i < multi_n && !samples_bound_depth; i++)
    samples_bound_depth = multi_depth[i] && multi_depth[i] == d.depth_base;
  bool mrt_sig_changed =
      g_region.cur_area_w != d.rt_w || g_region.cur_area_h != d.rt_h;
  for (u32 i = 0; i < mrt_n && !mrt_sig_changed; i++)
    mrt_sig_changed =
        g_region.cur_fmt[i] != (u32)ColorTargetFormat(d.mrt_info[i]) ||
        g_region.cur_w[i] != RtSurfaceExtent(d.mrt_surf_w[i], d.rt_w, 256) ||
        g_region.cur_h[i] != RtSurfaceExtent(d.mrt_surf_h[i], d.rt_h, 64);
  bool restart_region =
      g_region.cur_rt != d.rt_base || g_region.cur_mrt_count != mrt_n ||
      g_region.cur_depth != d.depth_base ||
      g_region.cur_depth_slice != d.depth_slice ||
      g_region.cur_layers != d.rt_layers ||
      g_region.cur_stencil != d.stencil_base ||
      g_region.depth_read_only != samples_bound_depth || mrt_sig_changed ||
      transition_source || pending_depth_clear;
  if (restart_region) {
    EndRegion();
    if (!rp->multi_tex && color_as_tex && transition_source) {
      auto& src = g_rts[tex_base];
      if (kTint) {
        TransitionImage(g_frame.list, src.texture, src.layout,
                        rhi::TextureState::kCopyDst);
        g_frame.list->ClearTexture(src.texture, rhi::TextureState::kCopyDst, {},
                                   rhi::ClearColor{{0.f, 0.f, 1.f, 1.f}});
      }
      if (src.layout != rhi::TextureState::kShaderRead)
        TransitionImage(g_frame.list, src.texture, src.layout,
                        rhi::TextureState::kShaderRead);
    }
    if (!rp->multi_tex && depth_as_tex && transition_source) {
      auto& src = g_depths[tex_base];
      if (src.layout != rhi::TextureState::kDepthRead)
        TransitionImage(g_frame.list, src.texture, src.layout,
                        rhi::TextureState::kDepthRead, rhi::kAspectDepth,
                        src.layers);
    }
    if (!rp->multi_tex && feedback_as_tex) {
      tex_set = SnapshotRT(g_rts[tex_base]);
      if (!tex_set)
        return Decline(kMidRegion);
    }
    if (rp->multi_tex) {
      for (u32 i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          auto& dst = g_rts[multi_storage[i]];
          if (dst.layout != rhi::TextureState::kGeneral)
            TransitionImage(g_frame.list, dst.texture, dst.layout,
                            rhi::TextureState::kGeneral);
        } else if (multi_color[i]) {
          auto& src = g_rts[multi_color[i]];
          if (kTint) {
            if (src.layout != rhi::TextureState::kCopyDst)
              TransitionImage(g_frame.list, src.texture, src.layout,
                              rhi::TextureState::kCopyDst);
            g_frame.list->ClearTexture(src.texture, rhi::TextureState::kCopyDst,
                                       {},
                                       rhi::ClearColor{{0.f, 0.f, 1.f, 1.f}});
          }
          const rhi::TextureState desired = multi_layouts[i];
          if (kBindTrace && d.ps_addr == (u64)kBindTrace) {
            static int bn = 0;
            if (bn++ < 8)
              BASE_LOGI("bindbar",
                        "b{} {:#x} layout={} desired={} dirty={} "
                        "region_restarted={}",
                        i, (unsigned long)multi_color[i], (int)src.layout,
                        (int)desired, (int)src.dirty_for_read,
                        (int)restart_region);
          }
          if (src.layout != desired || src.dirty_for_read) {
            TransitionImage(g_frame.list, src.texture, src.layout, desired);
            src.dirty_for_read = false;
          }
        } else if (multi_stencil_src[i]) {
          auto& src = g_depths[multi_stencil_src[i]];
          if (src.stencil_layout != rhi::TextureState::kDepthRead)
            TransitionImage(g_frame.list, src.texture, src.stencil_layout,
                            rhi::TextureState::kDepthRead, rhi::kAspectStencil,
                            src.layers);
        } else if (multi_depth[i]) {
          auto& src = g_depths[multi_depth[i]];
          if (src.layout != rhi::TextureState::kDepthRead)
            TransitionImage(g_frame.list, src.texture, src.layout,
                            rhi::TextureState::kDepthRead, rhi::kAspectDepth,
                            src.layers);
        } else if (multi_feedback[i]) {
          bool already_copied = false;
          for (u32 prior = 0; prior < i; prior++)
            already_copied |= multi_feedback[prior] == multi_feedback[i];
          if (!already_copied && !SnapshotRT(g_rts[multi_feedback[i]]))
            return Decline(kMidRegion);
        }
      }
    }
    if (rp->multi_tex) {
      rhi::Format multi_formats[kMaxTex] = {};
      for (u32 i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = rhi::TextureState::kGeneral;
          multi_formats[i] = g_rts[multi_storage[i]].fmt;
        } else if (multi_feedback[i]) {
          auto& src = g_rts[multi_feedback[i]];
          multi_views[i] =
              SampledView(src, d.texs[i].swizzle, true, d.texs[i].arrayed);
          multi_layouts[i] = rhi::TextureState::kShaderRead;
          multi_formats[i] = src.fmt;
        } else if (multi_color[i]) {
          multi_views[i] =
              SampledViewAs(g_rts[multi_color[i]], d.texs[i].swizzle,
                            GuestTextureFormat(d.texs[i].dfmt, d.texs[i].nfmt),
                            &multi_formats[i], d.texs[i].arrayed);
          multi_layouts[i] = rhi::TextureState::kShaderRead;
        } else if (multi_stencil_src[i]) {
          multi_views[i] = StencilSampledView(g_depths[multi_stencil_src[i]]);
          multi_layouts[i] = rhi::TextureState::kDepthRead;
          multi_formats[i] = rhi::Format::kS8Uint;
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = rhi::TextureState::kDepthRead;
        }
      }
      tex_set = rp->bindless ? bindless::Group()
                             : GetMultiTexSet(d, rp->tex_set_layout,
                                              rp->tex_bindings, multi_views,
                                              multi_layouts, multi_formats,
                                              multi_depth);
      if (!tex_set)
        return Decline(kGuestTex);
    }
    RTarget* rt =
        d.rt_base
            ? GetRT(d.rt_base, RtSurfaceExtent(d.mrt_surf_w[0], d.rt_w, 256),
                    RtSurfaceExtent(d.mrt_surf_h[0], d.rt_h, 64),
                    ColorTargetFormat(d.mrt_info[0]), d.rt_layers)
            : nullptr;
    if (d.rt_base && !rt)
      return true;
    if (!BeginRegion(d.mrt_base, d.mrt_info, mrt_n, d.rt_w, d.rt_h,
                     d.depth_base, d.depth_clear, d.stencil_base,
                     d.stencil_clear, samples_bound_depth, DepthW(d), DepthH(d),
                     d.mrt_surf_w, d.mrt_surf_h, d.mrt_dcc_base,
                     d.mrt_clear_word, d.depth_htile_base, d.depth_slice,
                     d.rt_layers, d.mrt_meta_cmask))
      return true;
    if (const auto it = g_depths.find(d.depth_base); it != g_depths.end())
      it->second.tile_mode = d.depth_tile_mode;
  }
  if (reads_synced)
    g_region.reads_synced = g_region.open;
  if (rp->multi_tex) {
    if (!tex_set && rp->bindless)
      tex_set = bindless::Group();
    if (!tex_set) {
      rhi::Format multi_formats[kMaxTex] = {};
      for (u32 i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = rhi::TextureState::kGeneral;
          multi_formats[i] = g_rts[multi_storage[i]].fmt;
        } else if (multi_feedback[i]) {
          multi_views[i] =
              SampledView(g_rts[multi_feedback[i]], d.texs[i].swizzle, true,
                          d.texs[i].arrayed);
          multi_formats[i] = g_rts[multi_feedback[i]].fmt;
        } else if (multi_color[i]) {
          multi_views[i] =
              SampledViewAs(g_rts[multi_color[i]], d.texs[i].swizzle,
                            GuestTextureFormat(d.texs[i].dfmt, d.texs[i].nfmt),
                            &multi_formats[i], d.texs[i].arrayed);
        } else if (multi_stencil_src[i]) {
          multi_views[i] = StencilSampledView(g_depths[multi_stencil_src[i]]);
          multi_layouts[i] = rhi::TextureState::kDepthRead;
          multi_formats[i] = rhi::Format::kS8Uint;
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = rhi::TextureState::kDepthRead;
        }
      }
      tex_set =
          GetMultiTexSet(d, rp->tex_set_layout, rp->tex_bindings, multi_views,
                         multi_layouts, multi_formats, multi_depth);
    }
    if (!tex_set)
      return Decline(kGuestTex);
  } else if (feedback_as_tex) {
    rhi::TextureView* views[kMaxTex] = {};
    rhi::TextureState layouts[kMaxTex] = {};
    rhi::Format formats[kMaxTex] = {};
    views[0] = SampledView(g_rts[tex_base], d.tex_swizzle, true, d.tex_arrayed);
    layouts[0] = rhi::TextureState::kShaderRead;
    formats[0] = g_rts[tex_base].fmt;
    tex_set =
        GetMultiTexSet(d, g_tex.layout, 1, views, layouts, formats, nullptr);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (color_as_tex) {
    auto& src = g_rts[tex_base];
    if (src.layout != rhi::TextureState::kShaderRead)
      return Decline(kMidRegion);
    rhi::TextureView* views[kMaxTex] = {};
    rhi::TextureState layouts[kMaxTex] = {};
    rhi::Format formats[kMaxTex] = {};
    views[0] = SampledViewAs(src, d.tex_swizzle,
                             GuestTextureFormat(d.tex_dfmt, d.tex_nfmt),
                             &formats[0], d.tex_arrayed);
    layouts[0] = rhi::TextureState::kShaderRead;
    tex_set =
        GetMultiTexSet(d, g_tex.layout, 1, views, layouts, formats, nullptr);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (depth_as_tex) {
    auto& src = g_depths[tex_base];
    if (src.layout != rhi::TextureState::kDepthRead)
      return Decline(kMidRegion);
    rhi::TextureView* views[kMaxTex] = {};
    rhi::TextureState layouts[kMaxTex] = {};
    views[0] = SampledView(src, d.tex_swizzle);
    layouts[0] = rhi::TextureState::kDepthRead;
    const u64 depth_only[kMaxTex] = {tex_base};
    rhi::Format formats[kMaxTex] = {};
    tex_set =
        GetMultiTexSet(d, g_tex.layout, 1, views, layouts, formats, depth_only);
    if (!tex_set)
      return Decline(kMidRegion);
  }

  g_stats.ns_dr_tex += NowNs() - t_tex;
  ScopeNs bind_timer(&g_stats.ns_dr_bind);
  SetGuestViewport(d);
  rhi::CommandList* list = g_frame.list;
  list->SetPipeline(rp->pipe);
  list->SetBlendConstants(d.blend_constants);
  list->SetPushConstants(0, 64, d.vs_user_data);
  list->SetPushConstants(64, 64, d.ps_user_data);
  if (mesh) {
    const u32 mesh_draw[4] = {draw_count, base::Max(d.instance_count, 1u),
                              mesh_argument_word, indirect_buf ? 1u : 0u};
    list->SetPushConstants(144, 16, mesh_draw);
  }
  if (gpu::gcn::PushCodeBase()) {
    const u32 vs_base[2] = {static_cast<u32>(d.vs_addr),
                            static_cast<u32>(d.vs_addr >> 32)};
    const u32 ps_base[2] = {static_cast<u32>(d.ps_addr),
                            static_cast<u32>(d.ps_addr >> 32)};
    list->SetPushConstants(128, 8, vs_base);
    list->SetPushConstants(136, 8, ps_base);
  }
  const bool indirect_cbufs = d.recomp->indirect_cbufs;
  const u32 cbuf_count = cbufs_used;
  u64 cb_off =
      (g_ring.ubo_offset + g_ring.ubo_align - 1) & ~(u64)(g_ring.ubo_align - 1);
  u64 cb_stride = g_ring.ubo_stride;
  if (cb_upload &&
      cb_off + cb_stride * (cbuf_count + (indirect_cbufs ? 1 : 0)) >
          g_ring.ubo_end) {
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 32)
        BASE_LOGI("decl", "ring=UBO off={} stride={} end={}",
                  (unsigned long long)cb_off, (unsigned long long)cb_stride,
                  (unsigned long long)g_ring.ubo_end);
    }
    return Decline(kRing);
  }
  u32 dyn_off[gpu::gcn::kIndirectCbufBindings]{};
  u32 cbuf_mask = 0;
  u64 next = cb_off;
  for (u32 i = 0; i < cbuf_count; i++) {
    if (!(rp->cbuf_mask & (1ull << i)))
      continue;
    const auto& cb = d.cbufs[i];
    const u32 readable = base::Min(cb.size, kCbufWindow);
    const bool resident =
        (!indirect_cbufs && cb_direct[i].buffer) || cb_gpu_off[i] != u64(-1);
    const bool have_cbuf =
        readable && (resident || IsReadableThisFrame(cb.base, readable));
    if (have_cbuf)
      cbuf_mask |= i < 32 ? 1u << i : 0;
    if (!indirect_cbufs && cb_direct[i].buffer) {
      g_stats.bind_bytes += kCbufWindow;
      g_stats.bind_n++;
      continue;
    }
    if (cb_gpu_off[i] != u64(-1)) {
      dyn_off[i] = static_cast<u32>(cb_gpu_off[i]);
      continue;
    }
    if (!have_cbuf && i != 0) {
      dyn_off[i] = 0;
      continue;
    }
    u32 cache_n = 0;
    if (have_cbuf) {
      cache_n = CbufCopyBytes(cb.base, cb.size);
      if (kRingDedup) {
        const u64 cached =
            MatchingStagedOffset(g_ubo_staged.Find(cb.base, cache_n), cb.base);
        if (cached != u64(-1)) {
          dyn_off[i] = static_cast<u32>(cached);
          continue;
        }
      }
    }
    u8* cb_dst = g_ring.ubo_map + next;
    u32 n;
    if (have_cbuf && (!FlushCsWritesRange(renderer, cb.base, cache_n, "cb") ||
                      g_frame.list != list))
      return Decline(kNoRecomp);
    if (have_cbuf) {
      n = cache_n;
      std::memcpy(cb_dst, reinterpret_cast<const void*>(cb.base), n);
      g_stats.ring_cb_bytes += n;
    } else {
      n = sizeof(d.mvp);
      std::memcpy(cb_dst, d.mvp, n);
    }
    if (kCbInfo && have_cbuf && n >= 16 &&
        (!kWantFrame || g_frame.num == kWantFrame) &&
        (kCbInfo == 1 || d.vs_addr == (u64)kCbInfo)) {
      static int shown = 0;
      if (shown++ < 40) {
        const float* f = reinterpret_cast<const float*>(cb_dst);
        base::String line;
        base::FormatTo(line, "vs={:#x} bind={} base={:#x} n={} @0: ",
                       (unsigned long)d.vs_addr, i, (unsigned long)cb.base, n);
        for (u32 k = 0; k < 16 && k * 4 < n; k++)
          base::FormatTo(line, " {:g}", f[k]);
        for (u32 off = 0x40u; off + 4 <= n; off += 0x40u) {
          base::FormatTo(line, " | @{:#x}:", off);
          for (u32 k = off / 4; k < off / 4 + 16 && k * 4 + 4 <= n; k++)
            base::FormatTo(line, " {:g}", f[k]);
        }
        BASE_LOGI("cbinfo", "{}", line.c_str());
      }
    }
    if (have_cbuf && kRingDedup)
      g_ubo_staged.Insert(cb.base, cache_n, 0, next, false, ~0u, 0, 0,
                          GuestDigest(cb.base, cache_n));
    dyn_off[i] = static_cast<u32>(next);
    next = (next + n + g_ring.ubo_align - 1) & ~(u64)(g_ring.ubo_align - 1);
  }
  if (indirect_cbufs) {
    const u64 slot_base = g_frame.slot_idx * (UboRingBytes() / 2);
    u32 offsets[gpu::gcn::kIndirectDrawDwords]{};
    for (u32 i = 0; i < cbuf_count; ++i)
      if (dyn_off[i])
        offsets[i] = static_cast<u32>((dyn_off[i] - slot_base) / sizeof(u32));
    std::memcpy(offsets + gpu::gcn::kIndirectCbufBindings, d.vs_user_data,
                sizeof(d.vs_user_data));
    std::memcpy(offsets + gpu::gcn::kIndirectCbufBindings + 32, d.ps_user_data,
                sizeof(d.ps_user_data));
    offsets[gpu::gcn::kIndirectGsUserDataAddr] =
        static_cast<u32>(d.gs_user_data_addr);
    offsets[gpu::gcn::kIndirectGsUserDataAddr + 1] =
        static_cast<u32>(d.gs_user_data_addr >> 32);
    std::memcpy(g_ring.ubo_map + next, offsets, sizeof(offsets));
    const u32 table_offset = static_cast<u32>(next);
    list->SetBindGroup(1, g_ring.indirect_cbuf_sets[g_frame.slot_idx],
                       &table_offset, 1);
    next = (next + sizeof(offsets) + g_ring.ubo_align - 1) &
           ~(u64)(g_ring.ubo_align - 1);
  } else if (cbuf_count) {
    rhi::BindGroup* set =
        base::AnyOf(cb_direct, cb_direct + cbuf_count,
                    [](const CachedBuffer& b) { return b.buffer != nullptr; })
            ? CbufBindings(cb_direct, cbuf_count)
            : GetCbufBindGroup(cbuf_count);
    if (!set)
      return Decline(kRing);
    list->SetBindGroup(1, set, dyn_off, cbuf_count);
  }
  g_ring.ubo_offset = next;
  u32 rawbuf_mask = 0;
  if (rp->raw_count) {
    u32 sbo_dyn[kRawBufBindings] = {};
    for (u32 i = 0; i < rp->raw_count; i++) {
      if (!(rp->raw_mask & (1u << i)))
        continue;
      const auto& rb = d.bufs[i];
      const u32 want = base::Min(rb.size, kRawBufWindow);
      if (!want)
        continue;
      if (raw_direct[i].buffer) {
        sbo_dyn[i] = 0;
        rawbuf_mask |= 1u << i;
        g_stats.bind_bytes += want;
        g_stats.bind_n++;
        continue;
      }
      if (raw_gpu_off[i] != u64(-1)) {
        sbo_dyn[i] = static_cast<u32>(raw_gpu_off[i]);
        rawbuf_mask |= 1u << i;
        continue;
      }
      if (!IsReadableThisFrame(rb.base, want))
        continue;
      const u64 cached =
          MatchingStagedOffset(g_sbo_staged.Find(rb.base, want), rb.base);
      if (cached != u64(-1)) {
        sbo_dyn[i] = static_cast<u32>(cached);
        rawbuf_mask |= 1u << i;
        continue;
      }
      if (!EnsureRawBufferRing())
        return Decline(kRing);
      const u64 off = (g_ring.sbo_offset + g_ring.sbo_align - 1) &
                      ~(u64)(g_ring.sbo_align - 1);
      const bool truncated = rb.size > kRawBufWindow;
      const u64 reserve =
          truncated
              ? g_ring.sbo_stride
              : ((want + g_ring.sbo_align - 1) & ~(u64)(g_ring.sbo_align - 1));
      if (off + reserve > g_ring.sbo_end || off + kRawBufWindow > kSboRing)
        return Decline(kRing);
      if (!FlushCsWritesRange(renderer, rb.base, want, "raw") ||
          g_frame.list != list)
        return Decline(kNoRecomp);
      u8* dst = g_ring.sbo_map + off;
      const bool tracked = ArmForCache(rb.base, want);
      std::memcpy(dst, reinterpret_cast<const void*>(rb.base), want);
      g_stats.ring_raw_bytes += want;
      if (want < reserve)
        std::memset(dst + want, 0, static_cast<size_t>(reserve - want));
      g_ring.sbo_offset = off + reserve;
      sbo_dyn[i] = static_cast<u32>(off);
      g_sbo_staged.Insert(rb.base, want, 0, off, tracked, ~0u, 0, 0,
                          GuestDigest(rb.base, want));
      rawbuf_mask |= 1u << i;
    }
    rhi::BindGroup* raw_set =
        base::AnyOf(raw_direct, raw_direct + rp->raw_count,
                    [](const CachedBuffer& b) { return b.buffer != nullptr; })
            ? RawBufferBindings(raw_direct, d, rp->raw_count)
            : GetRawBufferBindGroup(rp->raw_count);
    if (!raw_set)
      return Decline(kRing);
    list->SetBindGroup(2, raw_set, sbo_dyn, rp->raw_count);
    if (kRawBufTrace) {
      static int n = 0;
      if (n++ < 64)
        BASE_LOGI("rawbuf",
                  "draw#{} vs={:#x} nbufs={} staged={:#x} "
                  "sizes={}/{}/{}/{}",
                  g_frame.draws, (unsigned long)d.vs_addr, d.num_bufs,
                  rawbuf_mask, d.bufs[0].size, d.bufs[1].size, d.bufs[2].size,
                  d.bufs[3].size);
    }
  }
  if (tex_set)
    list->SetBindGroup(0, tex_set);
  if (mesh)
    list->SetBindGroup(3, MeshArgsSet());
  else if (rp->shared_lds)
    list->SetBindGroup(3, g_ring.lds_set);
  for (u32 j = 0; j < nbind; j++) {
    if (!bind_size[j] || vb_cached[j] != u64(-1) || vb_kept[j].buffer ||
        vb_bind[j].buffer)
      continue;
    if (!FlushCsWritesRange(renderer, reinterpret_cast<u64>(d.vbufs[j].data),
                            bind_size[j], "vb") ||
        g_frame.list != list)
      return Decline(kNoRecomp);
    const bool tracked =
        ArmForCache(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j]);
    if (tracked && kRingDedup &&
        CacheGuestBuffer(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j],
                         vb_kept[j])) {
      g_stats.vb_kept_bytes += bind_size[j];
      continue;
    }
    if (!EnsureVertexRing())
      return Decline(kRing);
    std::memcpy(g_ring.vb_map + voff + bind_off[j], d.vbufs[j].data,
                (size_t)bind_size[j]);
    g_stats.ring_vb_bytes += bind_size[j];
    if (kRingDedup)
      g_vb_staged.Insert(reinterpret_cast<u64>(d.vbufs[j].data), bind_size[j],
                         0, voff + bind_off[j], tracked);
  }
  if (indexed && ib_cached == u64(-1) && !ib_bind.buffer) {
    const bool tracked =
        !indirect_quads &&
        ArmForCache(reinterpret_cast<u64>(d.index_data),
                    u64(d.index_count) * GuestIndexElementBytes(d.index_type));
    if (!convert_indices) {
      if (!EnsureIndexRing())
        return Decline(kRing);
      CopyGuestIndices(g_ring.ib_map + ioff, d.index_data, d.index_count,
                       d.index_type);
      g_stats.ring_ib_bytes += index_bytes;
    }
    if (kRingDedup && !indirect_quads)
      g_ib_staged.Insert(
          reinterpret_cast<u64>(d.index_data), index_bytes, ib_tag, ioff,
          tracked, index_scanned ? nv - 1 : ~0u,
          u64(d.index_count) * GuestIndexElementBytes(d.index_type),
          index_revision);
  }
  if (auto_quads) {
    if (!EnsureIndexRing())
      return Decline(kRing);
    auto* out = reinterpret_cast<u32*>(g_ring.ib_map + ioff);
    for (u32 v = 0; v + 4 <= nv; v += 4, out += 6) {
      out[0] = v;
      out[1] = v + 1;
      out[2] = v + 2;
      out[3] = v;
      out[4] = v + 2;
      out[5] = v + 3;
    }
  }
  if (nbind) {
    rhi::Buffer* bufs[8];
    u64 offs[8];
    for (u32 j = 0; j < nbind; j++) {
      if (vb_kept[j].buffer) {
        bufs[j] = vb_kept[j].buffer;
        offs[j] = vb_kept[j].offset;
      } else if (vb_bind[j].buffer) {
        bufs[j] = vb_bind[j].buffer;
        offs[j] = vb_bind[j].offset;
      } else {
        if (!EnsureVertexRing())
          return Decline(kRing);
        bufs[j] = g_ring.vb;
        offs[j] = vb_cached[j] != u64(-1) ? vb_cached[j] : voff + bind_off[j];
      }
    }
    list->SetVertexBuffers(0, nbind, bufs, offs);
  }
  if (indexed || auto_quads)
    list->SetIndexBuffer(
        ib_bind.buffer    ? ib_bind.buffer
        : convert_indices ? g_ring.converted_ib
                          : g_ring.ib,
        ib_bind.buffer ? ib_bind.offset
                       : (ib_cached != u64(-1) ? ib_cached : ioff),
        (convert_indices || auto_quads || d.index_type == 1)
            ? rhi::IndexType::kUint32
            : rhi::IndexType::kUint16);
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      BASE_LOGI("dt", "RECOMP DREW count={} rt={:#x} nv={} multi_tex={}",
                draw_count, (unsigned long)d.rt_base, nv, (int)rp->multi_tex);
  }
  CmdInsertLabel(g_frame.list, "recomp vs=%#llx ps=%#llx n=%u%s",
                 (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr,
                 indexed ? d.index_count : d.vertex_count,
                 indexed ? " indexed" : "");
  DrawCheckpoint(g_frame.list, g_frame.num, g_frame.draws, false);
  if (const auto rt = g_rts.find(d.rt_base); rt != g_rts.end()) {
    rt->second.tile_mode = static_cast<int>(d.rt_tile_mode);
    rt->second.pitch = d.rt_surf_w;
  }
  if (d.rt_array_base) {
    g_region.write_through = d.rt_base;
    g_region.write_through_tile = d.rt_tile_mode;
  }
  DrawProfBegin(d.ps_addr);
  u64 predicate_off = 0;
  rhi::Buffer* const predicate = GuestPredicate(&predicate_off);
  if (predicate)
    list->BeginConditional(predicate, predicate_off, false);
  if (mesh && mesh_tasks)
    list->DrawMeshTasksIndirect(mesh_tasks, mesh_tasks_offset);
  else if (mesh)
    list->DrawMeshTasks((draw_count - 1) / d.recomp->mesh_input_primitives + 1,
                        d.instance_count ? d.instance_count : 1, 1);
  else if (indexed && indirect_buf)
    list->DrawIndexedIndirect(indirect_buf, indirect_off);
  else if (indirect_buf)
    list->DrawIndirect(indirect_buf, indirect_off);
  else if (indexed || auto_quads)
    list->DrawIndexed(upload_indices, d.instance_count ? d.instance_count : 1,
                      0, 0, 0);
  else
    list->Draw(d.vertex_count, d.instance_count ? d.instance_count : 1, 0, 0);
  if (predicate)
    list->EndConditional();
  DrawProfEnd();
  DrawCheckpoint(g_frame.list, g_frame.num, g_frame.draws, true);
  {
    static int seq = 0;
    if (seq < kSeqN) {
      BASE_LOGI("seq", "{} f{} draw#{} rt={:#x} tex={:#x}{}", seq++,
                g_frame.num, g_frame.draws, (unsigned long)d.rt_base,
                (unsigned long)tex_base, color_as_tex ? " RT-AS-TEX" : "");
    }
  }
  {
    static int shown = 0;
    const bool all = kWant == 1;
    static int latched = 0;
    if (all && kBusy && !kWantFrame && !latched && (int)g_frame.draws >= kBusy)
      latched = g_frame.num + 1;
    const int want_frame = kWantFrame ? kWantFrame : latched;
    const int cap = all ? (want_frame ? 100000 : 140) : 64;
    if (kWant && (all || g_region.cur_rt == kWant || d.ps_addr == kWant) &&
        shown < cap && (!want_frame || (int)g_frame.num == want_frame) &&
        (latched || (int)g_frame.draws >= kBusy)) {
      shown++;
      BASE_LOGI("drawrt",
                "f{} #{} rt={:#x} {}x{} indexed={} vcount={} "
                "icount={} inst={} prim={} tmask={:#x} num_vbufs={} stride={} "
                "mrt={} vp={:g},{:g} scale {:g},{:g} off vs={:#x} ps={:#x}",
                g_frame.num, g_frame.draws, (unsigned long)g_region.cur_rt,
                d.rt_w, d.rt_h, (int)indexed, d.vertex_count, d.index_count,
                d.instance_count, d.prim_type, d.target_mask, d.num_vbufs,
                d.vertex_stride, d.mrt_count, d.viewport_x_scale,
                d.viewport_y_scale, d.viewport_x_offset, d.viewport_y_offset,
                (unsigned long)d.vs_addr, (unsigned long)d.ps_addr);
      BASE_LOGI("drawrt",
                " blend en={} ctrl={:#x} tmask={:#x} surf={}x{} "
                "zscale={:g} zoff={:g} scissor=({},{})-({},{})",
                (int)d.blend_enable, d.blend_control, d.target_mask,
                d.rt_surf_w, d.rt_surf_h, d.viewport_z_scale,
                d.viewport_z_offset, d.scissor_tl & 0x7FFF,
                (d.scissor_tl >> 16) & 0x7FFF, d.scissor_br & 0x7FFF,
                (d.scissor_br >> 16) & 0x7FFF);
      for (u32 m = 0; m < d.mrt_count && m < 8; m++)
        BASE_LOGI("drawrt", " mrt{} base={:#x} info={:#x} fmt={} ntype={}", m,
                  (unsigned long)d.mrt_base[m], d.mrt_info[m],
                  (d.mrt_info[m] >> 2) & 0x1F, (d.mrt_info[m] >> 8) & 0x7);
      for (u32 m = 1; m < d.mrt_count && m < 8; m++)
        BASE_LOGI("drawrt", " blend{} en={} ctrl={:#x} base={:#x}", m,
                  (int)((d.mrt_blend_mask >> m) & 1u), d.mrt_blend[m],
                  (unsigned long)d.mrt_base[m]);
      BASE_LOGI(
          "drawrt",
          " depth base={:#x} valid={} test={} write={} "
          "func={} clear={:g} stencil={} cleardraw={}/{} rc={:#x} sbase={:#x}",
          (unsigned long)d.depth_base, (int)d.depth_valid,
          (int)d.depth_test_enable, (int)d.depth_write_enable, d.depth_func,
          d.depth_clear, (int)d.stencil_enable, (int)d.depth_clear_draw,
          (int)d.stencil_clear_draw, d.render_control,
          (unsigned long)d.stencil_base);
      const bool legacy_resolved =
          tex_set && tex_set != g_tex.white_set &&
          tex_set != g_tex.white_array_set && tex_set != g_tex.white_3d_set &&
          tex_set != g_tex.zero_set && tex_set != g_tex.zero_array_set &&
          tex_set != g_tex.zero_3d_set;
      const u64 shown_color[1] = {color_as_tex ? tex_base : 0};
      const u64 shown_feedback[1] = {feedback_as_tex ? tex_base : 0};
      const u64 shown_depth[1] = {depth_as_tex ? tex_base : 0};
      const u64* rc = rp->multi_tex ? multi_color : shown_color;
      const u64* rf = rp->multi_tex ? multi_feedback : shown_feedback;
      const u64* rd = rp->multi_tex ? multi_depth : shown_depth;
      const u32 shown_n = rp->multi_tex ? multi_n : base::Min(multi_n, 1u);
      BASE_LOGI("drawrt", " texs={} multi={} n={}", d.recomp->ps_texs.size(),
                (int)rp->multi_tex, multi_n);
      for (u32 i = 0; i < shown_n; i++) {
        const auto& t = d.texs[i];
        if (!t.base) {
          BASE_LOGI("drawrt", " tex{} UNRESOLVED (samples the default)", i);
          continue;
        }
        BASE_LOGI("drawrt",
                  " tex{} mips={} basemip={} viewmips={} "
                  "minlod={} layers={} arr={} lod0={} cmp={} sto={} "
                  "swz={:#x} smp={} {:08x} {:08x} {:08x} {:08x}",
                  i, t.mip_levels, t.base_mip, t.view_mips, t.min_lod, t.layers,
                  (int)t.arrayed, (int)t.force_lod_zero, (int)t.depth_compare,
                  (int)t.storage, t.swizzle, (int)t.sampler_valid, t.sampler[0],
                  t.sampler[1], t.sampler[2], t.sampler[3]);
        if (t.src && gpu::IsReadableRange(t.src, 32)) {
          const u32* tw = reinterpret_cast<const u32*>(t.src);
          BASE_LOGI("drawrt",
                    " tex{} T# src={:#x} [{:08x} {:08x} {:08x} {:08x} "
                    "{:08x} {:08x} {:08x} {:08x}]",
                    i, (unsigned long)t.src, tw[0], tw[1], tw[2], tw[3], tw[4],
                    tw[5], tw[6], tw[7]);
        }
        const bool resolved_guest =
            rp->multi_tex ? multi_views[i] != nullptr : legacy_resolved;
        BASE_LOGI("drawrt", " tex{} {:#x} {}x{} dfmt={} tiling={} -> {}{:#x}",
                  i, (unsigned long)t.base, t.w, t.h, t.dfmt, t.tiling,
                  rc[i]            ? "rt "
                  : rf[i]          ? "fb "
                  : rd[i]          ? "depth "
                  : resolved_guest ? "guest "
                                   : "MISS ",
                  (unsigned long)(rc[i]   ? rc[i]
                                  : rf[i] ? rf[i]
                                  : rd[i] ? rd[i]
                                          : 0));
        if (resolved_guest && !rc[i] && !rf[i] && !rd[i] &&
            g_rts.count(t.base)) {
          const auto& rt = g_rts[t.base];
          BASE_LOGI("drawrt",
                    " tex{} GUEST-OVER-RT: live={}x{} "
                    "ever_rendered={} variants={}",
                    i, rt.w, rt.h, (int)rt.ever_rendered,
                    g_targets.rt_variants.count(t.base)
                        ? g_targets.rt_variants[t.base].size()
                        : 0);
        }
        if (!rc[i] && !rf[i] && !rd[i] && !resolved_guest) {
          auto rt = g_rts.find(t.base);
          BASE_LOGI("drawrt",
                    " tex{} MISS why: in_rts={} live={}x{} "
                    "ever_rendered={} depth={} arrayed={} is_3d={} "
                    "resolved={:#x}",
                    i, rt != g_rts.end() ? 1 : 0,
                    rt != g_rts.end() ? rt->second.w : 0,
                    rt != g_rts.end() ? rt->second.h : 0,
                    rt != g_rts.end() ? (int)rt->second.ever_rendered : -1,
                    g_depths.count(t.base) ? 1 : 0, (int)t.arrayed,
                    (int)t.is_3d,
                    (unsigned long)ResolveSampledRT(t.base, t.w, t.h));
        }
      }
      for (u32 j = 0; j < base::Min(d.num_vbufs, 4u); j++) {
        const auto& vb = d.vbufs[j];
        if (!vb.data || !gpu::IsReadableRange((u64)(uintptr_t)vb.data, 64))
          continue;
        base::String bytes;
        const auto* p = static_cast<const u8*>(vb.data);
        for (u32 b = 0; b < base::Min<u32>(vb.stride ? vb.stride : 16, 48); b++)
          base::FormatTo(bytes, "{:02x}", p[b]);
        BASE_LOGI("drawrt", " vb{} @{:#x} stride={} v0={}", j,
                  (unsigned long)(uintptr_t)vb.data, vb.stride, bytes.c_str());
      }
      {
        base::String ud;
        for (u32 k = 0; k < 8; k++)
          base::FormatTo(ud, " {:08x}", d.vs_user_data[k]);
        const u64 table = (static_cast<u64>(d.vs_user_data[1] & 0xFFFF) << 32) |
                          d.vs_user_data[0];
        BASE_LOGI("drawrt", " vsud{} | [s0:s1]={:#x}", ud.c_str(),
                  (unsigned long)table);
        for (int row = -2; row < 6; row++) {
          const u64 at = table + static_cast<i64>(row) * 32;
          if (!gpu::IsReadableRange(at, 32))
            continue;
          const auto* w = reinterpret_cast<const u32*>(at);
          base::String line;
          for (u32 k = 0; k < 8; k++)
            base::FormatTo(line, " {:08x}", w[k]);
          for (u32 half = 0; half < 2; half++) {
            const gcn::VBuffer vb = gcn::DecodeVBuffer(w + half * 4);
            const u64 bytes =
                vb.stride ? (u64)vb.stride * vb.num_records : vb.num_records;
            if (!vb.base || !bytes)
              continue;
            const u32 scan = static_cast<u32>(base::Min<u64>(bytes, 4096));
            int nz = -1;
            if (gpu::IsReadableRange(vb.base, scan)) {
              nz = 0;
              const auto* b = reinterpret_cast<const u8*>(vb.base);
              for (u32 k = 0; k < scan; k++)
                nz += b[k] != 0;
            }
            base::FormatTo(line, " | V#[{}] {:#x} n={} nz={}", half,
                           (unsigned long)vb.base, (unsigned long)bytes, nz);
            if (nz == 0) {
              int found = 0;
              for (int k = -16; k <= 16 && !found; k++) {
                if (!k)
                  continue;
                const u64 probe = vb.base + static_cast<i64>(k) * 0x200000;
                if (!gpu::IsReadableRange(probe, 256))
                  continue;
                const auto* b = reinterpret_cast<const u8*>(probe);
                for (u32 q = 0; q < 256; q++)
                  if (b[q]) {
                    found = k;
                    break;
                  }
              }
              base::FormatTo(line, " arena{:+d}", found);
            }
          }
          BASE_LOGI("drawrt", "  eud{:+d}{}", row * 32, line.c_str());
        }
      }
      for (u32 j = 0; j < base::Min(d.num_bufs, kRawBufBindings); j++) {
        const auto& rb = d.bufs[j];
        if (!rb.base || !gpu::IsReadableRange(rb.base, 16))
          continue;
        base::String head;
        for (u32 off = 0; off < 0x600 && off + 16 <= rb.size; off += 0x180) {
          const auto* f = reinterpret_cast<const float*>(rb.base + off);
          base::FormatTo(head, " +{:#x}={:g},{:g},{:g},{:g}", off, f[0], f[1],
                         f[2], f[3]);
        }
        u32 nz = 0;
        const u32 scan = rb.size;
        const u32 step = scan > (64u << 10) ? scan / (64u << 10) : 1;
        if (gpu::IsReadableRange(rb.base, scan)) {
          const auto* p = reinterpret_cast<const u8*>(rb.base);
          for (u32 b = 0; b < scan; b += step)
            nz += p[b] != 0;
        }
        BASE_LOGI("drawrt", " buf{} @{:#x} size={} nz={} of {} sampled{}", j,
                  (unsigned long)rb.base, rb.size, nz, scan / step,
                  head.c_str());
      }
      if (d.index_data &&
          gpu::IsReadableRange((u64)(uintptr_t)d.index_data, 16)) {
        base::String idx;
        const auto* p16 = static_cast<const u16*>(d.index_data);
        const auto* p32 = static_cast<const u32*>(d.index_data);
        for (u32 k = 0; k < base::Min(d.index_count, 8u); k++)
          base::FormatTo(idx, " {}", d.index_type == 1 ? p32[k] : p16[k]);
        BASE_LOGI("drawrt", " idx @{:#x} type={} :{}",
                  (unsigned long)(uintptr_t)d.index_data, d.index_type,
                  idx.c_str());
      }
      for (u32 c = 0; c < base::Min<u32>(d.num_cbufs, kCbufBindings); c++) {
        const auto& cb = d.cbufs[c];
        if (!gpu::IsReadableRange(cb.base, base::Min(cb.size, 192u))) {
          if (cb.base || cb.size)
            BASE_LOGI("drawrt", " cb{} {:#x} sz={} UNREADABLE", c,
                      (unsigned long)cb.base, cb.size);
          else
            BASE_LOGI("drawrt", " cb{} UNBOUND", c);
          continue;
        }
        const u32* w = reinterpret_cast<const u32*>(cb.base);
        base::String line;
        base::FormatTo(line, " cb{} {:#x} sz={}:", c, (unsigned long)cb.base,
                       cb.size);
        for (u32 k = 0; k < 16 && k * 4 < cb.size; k++)
          base::FormatTo(line, " {:g}", *reinterpret_cast<const float*>(&w[k]));
        for (u32 off = 0x40u; off + 4 <= cb.size; off += 0x40u) {
          base::FormatTo(line, "\n  @{:#x}:", off);
          for (u32 k = off / 4; k < off / 4 + 16 && k * 4 + 4 <= cb.size; k++)
            base::FormatTo(line, " {:g}",
                           *reinterpret_cast<const float*>(&w[k]));
        }
        BASE_LOGI("drawrt", "{}", line.c_str());
      }
    }
  }
  for (u32 i = 0; i < multi_n; i++) {
    if (!multi_storage[i])
      continue;
    RTarget& target = g_rts[multi_storage[i]];
    target.ever_rendered = true;
    target.used_this_frame = true;
    target.last_frame = g_frame.num;
  }
  g_ring.vb_offset += vneed;
  if (ring_indices)
    g_ring.ib_offset = ioff + index_bytes + (indirect_quads ? 20 : 0);
  g_frame.draws++;
  for (u32 i = 0; i < g_region.cur_mrt_count; i++) {
    if (!g_region.cur_mrt[i])
      continue;
    auto& rt = g_rts[g_region.cur_mrt[i]];
    rt.last_vs = d.vs_addr;
    rt.last_ps = d.ps_addr;
    rt.last_cbuf_mask = cbuf_mask;
    rt.last_rawbuf_mask = rawbuf_mask;
  }
  if (g_region.cur_rt) {
    auto& rt = g_rts[g_region.cur_rt];
    if (++rt.draws > g_region.busiest_rt_draws) {
      g_region.busiest_rt_draws = rt.draws;
      g_region.busiest_rt = g_region.cur_rt;
    }
  }
  if (trace::Recording()) {
    const bool legacy_default =
        !tex_set || tex_set == g_tex.white_set ||
        tex_set == g_tex.white_array_set || tex_set == g_tex.white_3d_set ||
        tex_set == g_tex.zero_set || tex_set == g_tex.zero_array_set ||
        tex_set == g_tex.zero_3d_set;
    const u64 legacy_color = color_as_tex ? tex_base : 0;
    const u64 legacy_feedback = feedback_as_tex ? tex_base : 0;
    const u64 legacy_depth = depth_as_tex ? tex_base : 0;
    const u64 legacy_storage = 0;
    const void* legacy_view =
        (!rt_as_tex && !legacy_default) ? tex_set : nullptr;
    trace::DrawBindings bindings;
    bindings.tex_color = rp->multi_tex ? multi_color : &legacy_color;
    bindings.tex_feedback = rp->multi_tex ? multi_feedback : &legacy_feedback;
    bindings.tex_depth = rp->multi_tex ? multi_depth : &legacy_depth;
    bindings.tex_storage = rp->multi_tex ? multi_storage : &legacy_storage;
    bindings.tex_guest = rp->multi_tex
                             ? reinterpret_cast<const void* const*>(multi_views)
                             : &legacy_view;
    bindings.tex_count = rp->multi_tex ? multi_n : (d.num_texs ? 1u : 0u);
    bindings.cbuf_mask = cbuf_mask;
    bindings.rawbuf_mask = rawbuf_mask;
    trace::RecordDraw(d, "recomp", &bindings);
  }
  return true;
}

namespace {
const guest::SessionReset g_session_reset([] {
  g_collect_frame = -1;
  guest::ResetResource(g_vb_staged);
  guest::ResetResource(g_ib_staged);
  guest::ResetResource(g_ubo_staged);
  guest::ResetResource(g_sbo_staged);
  for (auto* caches : {g_raw_bindings, g_cbuf_bindings})
    for (u32 i = 0; i < 3; i++)
      guest::ResetResource(caches[i]);
  guest::ResetResource(g_sbo_gpu);
  guest::ResetResource(g_ubo_gpu);
  guest::ResetResource(g_collect_dcb);
});
}  // namespace

}  // namespace gpu::render
