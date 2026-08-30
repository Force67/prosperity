/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_draw_recomp.h"
#include "base/arch.h"

#include "gpu/guest_memory.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/gpu_perf.h"
#include "gpu/vulkan/vk_index_upload.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_trace.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/mem.h>
#include <utl/options.h>
#include <unordered_map>

namespace {
DELTA_OPTION(bool, kNoWipe, "DELTA_GPU_NOWIPE", true);
DELTA_OPTION(bool, kLazyClear2, "DELTA_GPU_LAZYCLEAR", true);
DELTA_OPTION(int, kTexBindFrame, "DELTA_GPU_TEXBIND", -1);
DELTA_OPTION(int, kSeqN, "DELTA_GPU_DRAWSEQ", 0);
DELTA_OPTION(u64, kWant, "DELTA_GPU_DRAWRT", 0);
DELTA_OPTION(int, kWantFrame, "DELTA_GPU_DRAWRT_FRAME", 0);
DELTA_OPTION(int, kBusy, "DELTA_GPU_DRAWRT_BUSY", 0);
// A COUNT, not a flag: as a bool this capped the trace at ONE line, so a
// target whose clears are all being dropped looked identical to a target
// with no clears at all.
DELTA_OPTION(int, kClearTrace, "DELTA_GPU_CLEARTRACE", 0);
// Name the guest writer of a faded-out UI vertex colour; see the arm below.
DELTA_OPTION(bool, kUiWatch, "DELTA_GPU_UIWATCH", false);
// Honour the scissor of a GNM fast clear instead of clearing the whole target.
DELTA_OPTION(bool, kClearRectScissor, "DELTA_GPU_CLEARRECT_SCISSOR", false);
DELTA_OPTION(bool, kDrawTrace, "DELTA_GPU_DRAWTRACE", false);
DELTA_OPTION(bool, kGpuDecltrace, "DELTA_GPU_DECLTRACE", false);
DELTA_OPTION(u64, kWhyDrop, "DELTA_GPU_WHYDROP", 0);
DELTA_OPTION(u64, kBindTrace, "DELTA_GPU_BINDTRACE", 0);
DELTA_OPTION(bool, kRawBufTrace, "DELTA_GPU_RAWBUF", false);
DELTA_OPTION(bool, kSelfTrace, "DELTA_GPU_SELFTRACE", false);
DELTA_OPTION(bool, kTightCbuf, "DELTA_GPU_TIGHTCBUF", false);
// DELTA_GPU_CBSTAGED=<vs addr> (or 1 for every draw): the bytes that reached
// the ring slot the descriptor points at -- what the SPIR-V actually loads,
// as opposed to the guest buffer the CPU-side trace prints.
DELTA_OPTION(u64, kCbInfo, "DELTA_GPU_CBSTAGED", 0);
// DELTA_GPU_TEXFORCE=<guest addr>: bind the 1x1 white default for exactly
// this texture address. A diagnostic, not a setting -- it answers "is THIS
// surface the one holding the frame back", which forcing every sampler
// white cannot.
DELTA_OPTION(u64, kTexForce, "DELTA_GPU_TEXFORCE", 0);
DELTA_OPTION(bool, kTint, "DELTA_GPU_RTTINT", false);
}  // namespace

namespace gpu::vk {

using rhi::DrawInfo;
using rhi::FlushCsWritesRange;

// The recompiled layout declares one push-constant range naming both stages
// (see GetRecompPipe), so every push has to name both.
constexpr VkShaderStageFlags kPcStages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

namespace {

// Why a draw declined the recompiled path (falls back to the heuristic).
// Tallied per reason so the remaining heuristic draws can be driven to zero;
// dumped with the periodic frame log.
enum DeclineReason {
  kNoRecomp,
  kNoTexPipe,
  kSelf,
  kRing,
  kGuestTex,
  kMidRegion,
  kNoPipe,
  kMaxDeclineReason
};
static const char* kDeclineName[kMaxDeclineReason] = {
    "norecomp", "notexpipe", "self", "ring", "guesttex", "midregion", "nopipe"};
u32 g_decline[kMaxDeclineReason] = {0};
inline bool Decline(DeclineReason r) {
  g_decline[r]++;
  g_win_declines++;
  if (trace::Recording())
    trace::RecordDecline(kDeclineName[r]);
  return false;
}

struct ReadableRangeKey {
  u64 base;
  u32 size;
  bool operator==(const ReadableRangeKey&) const = default;
};

struct ReadableRangeKeyHash {
  size_t operator()(const ReadableRangeKey& key) const {
    return static_cast<size_t>(key.base ^ (key.base >> 32) ^ key.size);
  }
};

// Per-frame staging dedupe. SotC redraws its whole world for the depth
// prepass, the G-buffer and the shadow cascades, and every one of those draws
// staged its vertex records, indices and cbuffer windows into the upload rings
// again -- RINGHWM showed the VB ring's whole 256 MiB frame half consumed by
// ~250 draws and the UBO ring full at ~130, after which every further draw was
// declined to the heuristic path (drawn with a guessed transform, which is
// where the exploded geometry came from). A window of guest bytes already
// staged this frame is byte-identical on repeat, so stage it once.
//
// A cached copy is current iff nothing has made its guest range stale since
// the copy: entries are stamped with rhi::CsWritebackGeneration() (compute
// results landing in guest memory bump it) and refused when a GPU-dirty
// compute range overlaps the key (dirty now means a writeback -- and a content
// change -- is still owed). CPU rewrites of the same address within one frame
// have no announcement to hook; titles ring-allocate their dynamic data so a
// rewritten buffer arrives at a new address, and the raw-buffer path has
// shipped this exact assumption since it grew its own `staged` map.
// DELTA_GPU_RING_DEDUP=0 restores the copy-per-draw behaviour for A/B.
struct StageCacheKey {
  u64 base;
  u32 salt;  // index type for the IB cache, 0 elsewhere
  bool operator==(const StageCacheKey&) const = default;
};

struct StageCacheKeyHash {
  size_t operator()(const StageCacheKey& key) const {
    return static_cast<size_t>(key.base ^ (key.base >> 32) ^ key.salt);
  }
};

struct StageCache {
  struct Entry {
    VkDeviceSize off;
    u64 bytes;
    u64 gen;
  };
  std::unordered_map<StageCacheKey, Entry, StageCacheKeyHash> map;
  int frame = -1;

  void RollFrame() {
    if (frame != g_frame.num) {
      frame = g_frame.num;
      map.clear();
    }
  }
  // Returns the cached ring offset, or -1 when absent/stale. A COPY THAT
  // COVERS the request answers it: same base, and every byte asked for is
  // already in the ring at the same offset, so the indices still land. Keying
  // on the exact length instead made a shared vertex buffer miss on every
  // draw -- GTA:SA indexes one 200k-vertex buffer, each draw reaching a few
  // hundred vertices further than the last, and re-copied ~2.4 MB per draw
  // until the per-frame ring ran out and the rest of the world was declined.
  VkDeviceSize Find(u64 base, u64 bytes, u32 salt = 0) {
    RollFrame();
    const auto it = map.find({base, salt});
    if (it == map.end() || it->second.bytes < bytes ||
        it->second.gen != rhi::CsWritebackGeneration() ||
        rhi::CsRangeDirtyOverlapping(base, bytes))
      return VkDeviceSize(-1);
    return it->second.off;
  }
  // Record a copy made at the CURRENT generation -- call after the range was
  // flushed (or was never compute-written), never before. A shorter copy never
  // replaces a longer live one: it would answer requests it does not cover.
  void Insert(u64 base, u64 bytes, u32 salt, VkDeviceSize off) {
    RollFrame();
    auto& e = map[{base, salt}];
    if (e.gen == rhi::CsWritebackGeneration() && e.bytes >= bytes)
      return;
    e = {off, bytes, rhi::CsWritebackGeneration()};
  }
};

StageCache g_vb_staged, g_ib_staged, g_ubo_staged, g_sbo_staged;

DELTA_OPTION(bool, kRingDedup, "DELTA_GPU_RING_DEDUP", true);

bool IsReadableThisFrame(u64 base, u32 size) {
  static int frame = -1;
  static std::unordered_map<ReadableRangeKey, bool, ReadableRangeKeyHash> cache;
  if (frame != g_frame.num) {
    frame = g_frame.num;
    cache.clear();
  }
  const ReadableRangeKey key{base, size};
  auto found = cache.find(key);
  if (found != cache.end())
    return found->second;
  return cache.emplace(key, gpu::IsReadableRange(base, size)).first->second;
}

}  // namespace

static_assert(rhi::DrawInfo::kMaxBuffers == kRawBufBindings,
              "the command processor and the raw-buffer ring must agree on "
              "how many set-2 bindings exist");

// DELTA_GPU_WHYDROP=<ps addr>: name the early exit that swallowed a draw. A
// draw that never reaches vkCmdDraw is invisible in every other trace, and the
// paths that consume one all `return true`.
static void WhyDrop(const rhi::DrawInfo& d, const char* where) {
  if (!kWhyDrop || d.ps_addr != (u64)kWhyDrop)
    return;
  BASE_LOGI("whydrop", "ps={:#x} exit={} rt={:#x} mrt={} depth={:#x}",
            (unsigned long)d.ps_addr, where, (unsigned long)d.rt_base,
            d.mrt_count, (unsigned long)d.depth_base);
}

void ReportDeclines() {
  base::String line;
  base::FormatTo(line, "  decline:");
  for (int i = 0; i < kMaxDeclineReason; i++)
    if (g_decline[i])
      base::FormatTo(line, " {}={}", kDeclineName[i], g_decline[i]);
  BASE_LOGI("gpuvk", "{}", line.c_str());
}

// Issue a draw running the game's recompiled VS/PS. Returns false if the draw
// can't be handled (the caller falls back to the heuristic path).
// DELTA_GPU_SKIP_PS=<hex> / DELTA_GPU_ONLY_PS=<hex>[,<hex>...]: drop every
// draw using that guest pixel shader, or every draw except those. Guest shader
// addresses are stable per build, which draw INDICES are not -- DELTA_GPU_
// ONLYDRAW looks equivalent and hands you a different pass whenever the draw
// count moves (P.T.'s light pass alone varies 29..34 draws).
//
// Answered from the RENDERER's entry point, not from the recompiled path: a
// draw whose shader never recompiled has no `d.recomp`, never reaches
// DrawRecomp at all, and would sail past the filter into the heuristic quad
// renderer -- so "isolate one pass" quietly left every un-recompiled pass in
// the frame, painting flat quads that no SPIR-V probe can touch.
bool ShaderFilterDrops(u64 ps_addr) {
  static const u64 kSkipPs = [] {
    const char* e = std::getenv("DELTA_GPU_SKIP_PS");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  if (kSkipPs && ps_addr == kSkipPs)
    return true;
  static const std::vector<u64> kOnlyPs = [] {
    std::vector<u64> out;
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

bool DrawRecomp(rhi::Renderer& renderer, const DrawInfo& d) {
  const u64 t_draw_start = NowNs();
  // DELTA_GPU_WHYDROP=1: every draw as the renderer receives it, so a slot that
  // never reaches the seq log can be identified.
  if (kWhyDrop == 1)
    BASE_LOGI("whydrop",
              "draw#{} ps={:#x} vs={:#x} rt={:#x} mrt={} tex={:#x} "
              "prim={} cnt={}",
              g_frame.draws, (unsigned long)d.ps_addr,
              (unsigned long)d.vs_addr, (unsigned long)d.rt_base,
              d.mrt_count, (unsigned long)d.tex_base, d.prim_type,
              d.index_data ? d.index_count : d.vertex_count);
  bool indexed = d.index_data && d.index_count >= 3;
  u32 draw_count = indexed ? d.index_count : d.vertex_count;
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      BASE_LOGI("dt",
                "enter recomp count={} rt={:#x} tex={:#x} num_texs={} "
                "num_vattrs={} ok={}",
                draw_count, (unsigned long)d.rt_base,
                (unsigned long)d.tex_base, d.num_texs, d.num_vattrs,
                d.recomp ? d.recomp->ok : 0);
  }
  if (!d.recomp || !d.recomp->ok || draw_count < 3) {
    // DELTA_GPU_DECLTRACE: norecomp lumps together three unrelated causes --
    // no recompiled program, a program that failed to translate, and a draw
    // whose count never made it out of the packet. Separate them, because only
    // the last one points upstream at the command processor.
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 96)
        BASE_LOGI("decl",
                  "norecomp why={} vcount={} icount={} idx={:p} "
                  "vbufs={} vattrs={} prim={:#x} rt={:#x}",
                  !d.recomp       ? "no-program"
                  : !d.recomp->ok ? "translate-failed"
                                  : "count<3",
                  d.vertex_count, d.index_count, d.index_data, d.num_vbufs,
                  d.num_vattrs, d.prim_type, (unsigned long)d.rt_base);
    }
    return Decline(kNoRecomp);
  }
  const bool has_storage_image =
      std::any_of(d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                  [](const gcn::ShaderTex& tex) { return tex.storage; });
  if (!d.mrt_count && !d.depth_base && !has_storage_image) {
    // DELTA_GPU_DECLTRACE: a draw with no target at all. Legitimate for a
    // colour-write-masked pass, but it also catches a mis-decoded write mask,
    // which silently deletes real geometry.
    static int n = 0;
    if (kGpuDecltrace && n++ < 48)
      BASE_LOGI("decl",
                "no-target draw#{} vs={:#x} ps={:#x} tex={:#x} "
                "icount={} vcount={} mask={:#x} pstex={} storage={}",
                g_frame.draws, (unsigned long)d.vs_addr,
                (unsigned long)d.ps_addr, (unsigned long)d.tex_base,
                d.index_count, d.vertex_count, d.target_mask,
                d.recomp->ps_texs.size(),
                (size_t)std::count_if(d.recomp->ps_texs.begin(),
                                      d.recomp->ps_texs.end(),
                                      [](const gcn::ShaderTex& t) {
                                        return t.storage;
                                      }));
    WhyDrop(d, "no-target");
    g_frame.draws++;
    return true;
  }
  if (!g_quad.tex_pipeline)
    return Decline(
        kNoTexPipe);  // need the descriptor infra (ds_pool/ds_layout)
  // Resolve the sampled texture address to an overlapping live RT
  // (resource-model page-table lookup), so an RT-as-texture sample binds the
  // live image for cycled/aliased RT addresses instead of stale guest memory.
  // Additive: an exact RT base resolves to itself.
  u64 tex_base = d.tex_base;
  // DELTA_GPU_TEXFORCE also has to cover the single-texture path, or it
  // silently does nothing for exactly the blit-shaped draws that path exists
  // for -- and a diagnostic that quietly no-ops reads as a negative result.
  const bool force_white_tex =
      kTexForce && tex_base == (u64)kTexForce;
  if (force_white_tex)
    tex_base = 0;
  // Render targets are plain 2D images, so only a plain 2D binding may resolve
  // to one. An array or volume binding declared a sampler type no render target
  // can satisfy.
  const bool tex_rt_eligible = !d.tex_arrayed && !d.tex_is_3d;
  if (tex_base && tex_rt_eligible && !g_rts.count(tex_base) &&
      !g_depths.count(tex_base)) {
    bool depth_format = d.tex_dfmt == 4 && d.tex_nfmt == 7;
    u64 r =
        depth_format ? ResolveSampledDepth(tex_base, d.tex_w, d.tex_h) : 0;
    if (!r)
      r = ResolveSampledRT(tex_base, d.tex_w, d.tex_h);
    if (!r && !depth_format)
      r = ResolveSampledDepth(tex_base, d.tex_w, d.tex_h);
    if (r)
      tex_base = r;
  }
  bool color_as_tex = tex_rt_eligible && tex_base && tex_base != d.rt_base &&
                      g_rts.count(tex_base);
  bool feedback_as_tex = tex_rt_eligible && tex_base &&
                         tex_base == d.rt_base && g_rts.count(tex_base) &&
                         g_rts[tex_base].ever_rendered;
  // A deferred-lighting pass binds the scene depth buffer AND samples it to
  // rebuild world position. That is legal in Vulkan whenever the pass cannot
  // write depth: the image sits in DEPTH_READ_ONLY_OPTIMAL and serves as both
  // attachment and sampled image. Declining it instead sent SotC's whole
  // lighting pass down the heuristic quad path, which produced nothing.
  const bool depth_self_read = tex_rt_eligible && tex_base &&
                               tex_base == d.depth_base &&
                               !d.depth_write_enable && g_depths.count(tex_base);
  bool depth_as_tex = tex_rt_eligible && tex_base &&
                      (tex_base != d.depth_base || depth_self_read) &&
                      g_depths.count(tex_base);
  bool rt_as_tex = color_as_tex || feedback_as_tex || depth_as_tex;
  if (color_as_tex && g_rts[tex_base].w >= 700 && g_rts[tex_base].w <= 900)
    g_frame.had_room = true;
  if (tex_base && !depth_self_read &&
      (tex_base == d.depth_base ||
       (tex_base == d.rt_base && !feedback_as_tex))) {
    // DELTA_GPU_SELFTRACE: which pass reads the target it is drawing into. We
    // drop those, and dropping one every frame leaves whatever it was meant to
    // produce stale.
    static int n = 0;
    if (kSelfTrace && n++ < 20)
      BASE_LOGI(
          "self",
          "draw#{} ps={:#x} rt={:#x} tex={:#x} depth={:#x} dtest={} "
          "dwrite={} ntex={}",
          g_frame.draws, (unsigned long)d.ps_addr, (unsigned long)d.rt_base,
          (unsigned long)tex_base, (unsigned long)d.depth_base,
          (int)d.depth_test_enable, (int)d.depth_write_enable, d.num_texs);
    return Decline(kSelf);
  }
  // Indexed draws derive the copied vertex range from their indices.
  // DRAW_INDEX_AUTO consumes the packet's sequential vertex count directly.
  // These three all report as "norecomp", which lumps a vertex-count cap in
  // with a missing program; separate them, because only the cap is ours to
  // move.
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
  if (indexed) {
    const u32 max_index =
        MaxGuestIndex(d.index_data, d.index_count, d.index_type);
    if (max_index >= 200000u)
      return vtx_decline("max-index", max_index);
    nv = max_index + 1;
  }
  if (nv > 200000u || (d.num_vattrs && (!d.vertex_data || !d.vertex_stride)))
    return vtx_decline(nv > 200000u ? "nv-cap" : "no-vertex-data", nv);
  if (d.vertex_data && d.vertex_stride &&
      !FlushCsWritesRange(renderer, reinterpret_cast<u64>(d.vertex_data),
                          static_cast<u64>(nv) * d.vertex_stride))
    return vtx_decline("cs-flush", nv);

  // GNM's fast clear (see DrawInfo::is_clear_rect): a RECT_LIST draw with no
  // pixel shader whose colour lives in CB_COLORn_CLEAR_WORD0/1. Rasterising it
  // writes nothing, so leaving it to the normal path silently turns every clear
  // into a no-op and each target keeps loading the previous frame -- SotC's
  // world colour RT accumulated one fullscreen pass per frame until its value
  // literally tracked the frame counter. Record the clear and consume the draw.
  //
  // The clear colour is encoded in each target's own format, so it needs the
  // per-format unpack in ColorTargetClearValue. Clearing to zero regardless
  // turns P.T.'s opaque white and opaque black clears into transparent black,
  // which is a hole in a deferred composite.
  if (d.is_clear_rect) {
    // ...but a RECT_LIST with no pixel shader is ALSO the shape of the CB
    // metadata passes, and those are the opposite of a clear: they preserve
    // the surface. CB_COLOR_CONTROL.MODE tells them apart --
    // CB_NORMAL(1) is the clear; CB_ELIMINATE_FAST_CLEAR(2),
    // CB_RESOLVE(3), CB_DECOMPRESS(4) and CB_FMASK_DECOMPRESS(5) resolve
    // CMASK/FMASK/DCC into the surface. We never compress a target, so for us
    // those are no-ops -- but they still have to be consumed, because with no
    // pixel shader they would otherwise rasterise nothing and drop through as
    // a normal draw.
    //
    // P.T. issues 58k of these and not one real fast clear: every frame it
    // eliminated the fast-clear state of its composite eight times, twice
    // right before the tonemap pass samples that same target through a
    // feedback copy. Taking them for clears wiped the scene a draw before it
    // was read, which is why the game presented black.
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
    // A fast clear covers the generic scissor, not necessarily the whole
    // target, and we have no way to express a partial one here -- a pending
    // clear is realised as loadOp=CLEAR over the entire attachment. SotC
    // issues twelve of these a frame against the buffer its compute resolve
    // reads, so taking each of them as "clear everything" erases the deferred
    // lighting that was rendered into it. Skip the ones that do not cover the
    // target; leaving old content is recoverable, erasing live content is not.
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
      const VkClearColorValue clear = ColorTargetClearValue(
          d.mrt_info[i], d.mrt_clear_word[i][0], d.mrt_clear_word[i][1]);
      it->second.clear_pending = true;
      it->second.clear_src = "clear-rect";
      it->second.clear_value = clear;
      if (kClearTrace) {
        static int n = 0;
        if (n++ < kClearTrace)
          BASE_LOGI("clear",
                    "f{} draw#{} rect RT {:#x} info={:#x} cc={:#x} mode={} gen="
                    "({},{})-({},{}) win={:#x}/{:#x} scr={:#x}/{:#x} target={}x{} "
                    "CLEAR_WORD {:08x} {:08x} -> ({:g} {:g} {:g} {:g})",
                    g_frame.num, g_frame.draws,
                    (unsigned long)d.mrt_base[i], d.mrt_info[i],
                    d.color_control, (d.color_control >> 4) & 7u, cx0, cy0,
                    cx1, cy1, d.clear_window_tl, d.clear_window_br,
                    d.clear_screen_tl, d.clear_screen_br, d.rt_w, d.rt_h,
                    d.mrt_clear_word[i][0], d.mrt_clear_word[i][1],
                    clear.float32[0], clear.float32[1], clear.float32[2],
                    clear.float32[3]);
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
    return true;  // consumed: must not reach the rasteriser
  }

  // DELTA_GPU_UIWATCH=1: name the guest code that writes SotC's UI vertex
  // COLOURS. Its title screen composites an opaque black plate over the scene
  // and draws every UI element with an RGBA8 colour attribute of 00000000 --
  // faded out -- so nothing reaches the screen however well the scene renders.
  // The buffer holding those colours is only knowable while a draw is
  // processed and moves every run, which is why the watch is armed from here
  // (the same route DELTA_GPU_NULLWATCH uses for a descriptor pointer).
  if (kUiWatch) {
    static bool armed = false;
    if (!armed) {
      for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
        const auto& attr = d.vattrs[a];
        if (attr.dfmt != 10 || attr.nfmt != 0 || attr.num_comps != 4)
          continue;  // not an RGBA8 colour
        if (attr.binding >= d.num_vbufs)
          continue;
        const auto& vb = d.vbufs[attr.binding];
        const auto* p = static_cast<const u8*>(vb.data);
        if (!p || !utl::isMemoryRangeMapped(p + attr.offset, 4))
          continue;
        u32 c0 = 0;
        std::memcpy(&c0, p + attr.offset, 4);
        if (c0 != 0)
          continue;  // only the faded-out ones are interesting
        armed = true;
        const uintptr_t at = reinterpret_cast<uintptr_t>(p) + attr.offset;
        BASE_LOGI("uiwatch",
                  "arming on UI colour {:#x} (rt={:#x}, {} verts, "
                  "stride {}) -- it currently reads 00000000",
                  (unsigned long)at, (unsigned long)d.rt_base,
                  d.vertex_count, vb.stride);
        utl::setWriteWatchValueProbe(at);
        utl::setWriteWatchChase(4);
        if (!utl::armWriteWatch(at & ~0xFFFull, 0x1000, 200))
          BASE_LOGI("uiwatch", "no armer registered");
        break;
      }
    }
  }

  // A target whose FIRST draw of a frame blends with COLOR_DESTBLEND == ONE is
  // being ACCUMULATED into, and accumulation from an unknown starting value is
  // meaningless -- the guest necessarily begins it from a defined state. The
  // lazy-clear heuristic (persist RT content across frames as LOAD) is right
  // for content baked once and wrong here: without a reset the accumulation
  // compounds every frame. P.T.'s light buffer is accumulated by two draws and
  // then DIVIDED by its own alpha by a third, which turns the compounding into
  // a gain of ~8 per frame and pins texels at the fp16 ceiling.
  if (kLazyClear2 && d.rt_base && d.blend_enable &&
      ((d.blend_control >> 8) & 0x1F) == 1u) {
    auto it = g_rts.find(d.rt_base);
    if (it != g_rts.end() && it->second.last_frame != g_frame.num &&
        it->second.ever_rendered) {
      it->second.clear_pending = true;
      it->second.clear_src = "accumulate-needs-reset";
      it->second.clear_value = VkClearColorValue{{0.f, 0.f, 0.f, 0.f}};
    }
  }
  // DELTA_GPU_VTXTRACE_RT=<hex>: diagnostic only. For every draw into that
  // colour target, report the vertex layout and the first vertex's raw
  // attribute words. A pass whose pixel shader just interpolates a vertex
  // colour (SotC's UI-layer fill) produces exactly that colour, so the
  // recorded target content can be checked against the guest's own data
  // instead of guessed at.
  {
    static const u64 kVtxRt = [] {
      const char* e = std::getenv("DELTA_GPU_VTXTRACE_RT");
      return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    // The menu floods the early run, so hold off until the level is up.
    static const auto kVtxStart = std::chrono::steady_clock::now();
    static const int kVtxAfter = [] {
      const char* e = std::getenv("DELTA_GPU_VTXTRACE_AFTER");
      return e ? std::atoi(e) : 0;
    }();
    static int vtx_n = 0;
    if (kVtxRt && d.rt_base == kVtxRt && vtx_n < 40 &&
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - kVtxStart)
                .count() >= kVtxAfter) {
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
        const u64 avail =
            static_cast<u64>(vb.stride) * vb.num_records;
        if (vb.data && avail >= va.offset + sizeof(u32) * 4) {
          const auto* p = static_cast<const u8*>(vb.data) + va.offset;
          u32 w[4] = {};
          std::memcpy(w, p, sizeof(w));
          base::FormatTo(line, " raw={:08x} {:08x} {:08x} {:08x} f={:g} {:g} {:g} {:g}",
                         w[0], w[1], w[2], w[3],
                         *reinterpret_cast<const float*>(&w[0]),
                         *reinterpret_cast<const float*>(&w[1]),
                         *reinterpret_cast<const float*>(&w[2]),
                         *reinterpret_cast<const float*>(&w[3]));
        }
      }
      BASE_LOGI("vtx", "{}", line.c_str());
    }
  }

  // A fullscreen, untextured, near-black REPLACE draw is the game CLEARING an
  // RT. Don't render it (that wipes the RT immediately); record a LAZY clear
  // instead -- realised as loadOp=CLEAR only when content actually redraws this
  // RT this frame (see BeginRegion). Baked-once content (the room floor) whose
  // clear and redraw land on different frames then survives. A COLOURED
  // fullscreen REPLACE is real content (e.g. the per-frame minimap redraw) and
  // must NOT be treated as a clear.
  // The extent test below reads the position attribute as float32s. A title
  // whose positions are not floats (Skyrim's UI uses 16_16_SScaled) would have
  // its geometry read as garbage, land a bogus fullscreen extent, and get
  // swallowed as a "clear" -- which is exactly what turned its whole frame
  // black. Only consider draws whose position really is float.
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
      d.recomp->ps_texs.empty() && nv <= 8) {
    u32 cdst = (d.blend_control >> 8) & 0x1F,
             csrc = d.blend_control & 0x1F;
    bool replace = d.blend_enable && csrc == 1 && cdst == 0;
    if (replace) {
      const auto* vb = static_cast<const u8*>(d.vertex_data);
      bool near_black = true;
      float clear_color[4] = {0, 0, 0, 0};
      for (u32 a = 0; a < d.num_vattrs; a++) {
        if (d.vattrs[a].num_comps == 4 && d.vattrs[a].offset != 0) {
          // Colour may live in its own binding; read from that binding's base.
          const auto* cbuf =
              static_cast<const u8*>(d.vbufs[d.vattrs[a].binding].data);
          const u8* cb0 = cbuf + d.vattrs[a].offset;  // vertex 0's colour
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
        RTarget* rt =
            d.rt_base ? GetRT(d.rt_base, RtSurfaceExtent(d.mrt_surf_w[0], d.rt_w, 256),
                              RtSurfaceExtent(d.mrt_surf_h[0], d.rt_h, 64),
                              ColorTargetFormat(d.mrt_info[0]))
                      : nullptr;
        if (rt) {
          // Counted, not sampled.
          if (kClearTrace) {
            static std::atomic<u64> n{0};
            if ((n.fetch_add(1) % 500) == 0)
              BASE_LOGI("clear",
                        "lazyclear-heuristic #{} rt={:#x} mrt={} "
                        "mrt1={:#x}",
                        (unsigned long long)n.load(),
                        (unsigned long)d.rt_base, d.mrt_count,
                        (unsigned long)(d.mrt_count > 1 ? d.mrt_base[1] : 0));
          }
          rt->clear_pending = true;
          rt->clear_src = "lazyclear-heuristic";
          std::memcpy(rt->clear_value.float32, clear_color,
                      sizeof(clear_color));
          // Which draws this heuristic decided were clears. It reclassifies a
          // fullscreen near-black draw as a clear and suppresses it, so a
          // legitimate dark fullscreen layer disappears AND takes the target's
          // previous contents with it.
          if (kClearTrace) {
            static int n = 0;
            if (n++ < kClearTrace)
              BASE_LOGI("clear",
                        "f{} draw#{} LAZYCLEAR-HEURISTIC RT {:#x} "
                        "vs={:#x} ps={:#x} color=({:g} {:g} {:g} {:g}) blend={}/{:#x}",
                        g_frame.num, g_frame.draws,
                        (unsigned long)d.rt_base, (unsigned long)d.vs_addr,
                        (unsigned long)d.ps_addr, clear_color[0],
                        clear_color[1], clear_color[2], clear_color[3],
                        (int)d.blend_enable, d.blend_control);
          }
        }
        // This draw also performs the guest's reverse-Z clear (depth write
        // enabled, ZFUNC=ALWAYS). Suppressing its color write must not discard
        // that depth effect, or stale depth rejects the following layer
        // composites.
        if (d.depth_base && d.depth_write_enable && d.depth_func == 7) {
          DepthTarget* dt = GetDepthRT(d.depth_base, d.rt_w, d.rt_h);
          if (dt) {
            dt->clear_pending = true;
            dt->clear_value = d.depth_clear;
          }
        }
        WhyDrop(d, "lazyclear");
        g_frame.draws++;
        return true;  // suppressed; the clear is applied lazily on the next
                      // redraw
      }
      // Legacy single-frame behaviour (DELTA_GPU_LAZYCLEAR=0): only suppress if
      // the RT already holds content this frame.
      auto rit = g_rts.find(d.rt_base);
      if (fullscreen_black && rit != g_rts.end() && rit->second.draws > 0 &&
          rit->second.last_frame == g_frame.num) {
        WhyDrop(d, "fullscreen-black");
        g_frame.draws++;
        return true;
      }
    }
  }

  // Lay out one contiguous ring range per vertex binding. Binding 0 sits at the
  // ring offset (single-stream draws are byte-identical to before); additional
  // bindings are 16-byte aligned so no attribute straddles a coarse boundary.
  // A binding whose guest range is already staged this frame reuses that copy
  // (vb_cached[j]) and takes no ring space at all.
  const u32 nbind = d.num_vattrs ? std::min(d.num_vbufs, 8u) : 0;
  VkDeviceSize bind_off[8] = {}, bind_size[8] = {};
  VkDeviceSize vb_cached[8] = {};
  VkDeviceSize vneed = 0;
  for (u32 j = 0; j < nbind; j++) {
    if (d.vbufs[j].stride) {
      bind_size[j] = (VkDeviceSize)nv * d.vbufs[j].stride;
    } else {
      // Stride-0 (constant) binding: upload a single record large enough to
      // cover every attribute that reads it; the pipeline binds it with stride
      // 0 so all vertices fetch this one record.
      u32 rec = 0;
      for (u32 a = 0; a < d.num_vattrs; a++)
        if (d.vattrs[a].binding == j)
          rec = std::max(
              rec, d.vattrs[a].offset + VertexFormatBytes(d.vattrs[a].dfmt));
      bind_size[j] = rec;
    }
    vb_cached[j] =
        kRingDedup && bind_size[j]
            ? g_vb_staged.Find(reinterpret_cast<u64>(d.vbufs[j].data),
                               bind_size[j])
            : VkDeviceSize(-1);
    if (vb_cached[j] != VkDeviceSize(-1))
      continue;
    if (vneed)
      vneed = (vneed + 15) & ~VkDeviceSize(15);
    bind_off[j] = vneed;
    vneed += bind_size[j];
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
  const VkDeviceSize index_bytes =
      indexed ? static_cast<VkDeviceSize>(d.index_count) *
                    UploadedIndexElementBytes(d.index_type)
              : 0;
  // The IB cache key carries the index type: CopyGuestIndices widens 16-bit
  // sources, so the same guest bytes at two types are two different uploads.
  const VkDeviceSize ib_cached =
      kRingDedup && indexed
          ? g_ib_staged.Find(reinterpret_cast<u64>(d.index_data),
                             index_bytes, 1u + d.index_type)
          : VkDeviceSize(-1);
  const VkDeviceSize index_align = d.index_type == 1 ? 4 : 2;
  const VkDeviceSize aligned_ioff =
      (g_ring.ib_offset + index_align - 1) & ~(index_align - 1);
  if (indexed && ib_cached == VkDeviceSize(-1) &&
      aligned_ioff + index_bytes > g_ring.ib_end) {
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

  g_ns_dr_pre += NowNs() - t_draw_start;
  const u64 t_pipe = NowNs();
  RecompPipe* rp = GetRecompPipe(d);
  g_ns_dr_pipe += NowNs() - t_pipe;
  if (!rp)
    return Decline(kNoPipe);
  const u64 t_tex = NowNs();
  // Guest-texture source resolved up front; an RT-as-texture source is resolved
  // after the region switch (transitioning it to readable must happen outside a
  // region).
  VkDescriptorSet tex_set = VK_NULL_HANDLE;
  if (rp->textured && !rp->multi_tex && !rt_as_tex) {
    if (!force_white_tex && GuestTextureUploadSupported(d.tex_dfmt, d.tex_nfmt))
      tex_set = GetTexture(
          d.tex_base, d.tex_w, d.tex_h, d.tex_dfmt, d.tex_nfmt, d.tex_tiling,
          d.tex_pitch, d.tex_layers, d.tex_base_array, d.tex_view_layers,
          d.tex_mip_levels, d.tex_base_mip, d.tex_view_mips, d.tex_min_lod,
          d.tex_pow2_pad, d.tex_sampler, d.tex_sampler_valid, d.tex_arrayed,
          d.tex_force_lod_zero, d.tex_depth_compare, d.tex_swizzle, d.tex_depth,
          d.tex_is_3d);
    // The fallback has to match the dimensionality the shader declared for this
    // binding, or the descriptor write is a type mismatch.
    if (!tex_set)
      tex_set = d.tex_null_descriptor
                    ? (d.tex_is_3d      ? g_tex.zero_3d_set
                       : d.tex_arrayed  ? g_tex.zero_array_set
                                        : g_tex.zero_set)
                    : (d.tex_is_3d     ? g_tex.white_3d_set
                       : d.tex_arrayed ? g_tex.white_array_set
                                       : g_tex.white_set);
    if (!tex_set)
      return Decline(kGuestTex);
  }

  u64 multi_color[kMaxTex] = {};
  u64 multi_depth[kMaxTex] = {};
  // The depth target whose STENCIL plane a binding names (see
  // ResolveSampledStencil): a separate guest surface sharing one host image.
  u64 multi_stencil_src[kMaxTex] = {};
  u64 multi_feedback[kMaxTex] = {};
  u64 multi_storage[kMaxTex] = {};
  VkImageView multi_views[kMaxTex] = {};
  VkImageLayout multi_layouts[kMaxTex];
  bool multi_transition_source = false;
  u32 multi_n = std::min(d.num_texs, kMaxTex);
  if (rp->multi_tex) {
    auto is_bound_target = [&](u64 base) {
      u32 count = std::min(d.mrt_count, 8u);
      for (u32 m = 0; m < count; m++)
        if (d.mrt_base[m] == base)
          return true;
      return false;
    };
    for (u32 i = 0; i < kMaxTex; i++)
      multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
          const VkFormat format = GuestTextureFormat(t.dfmt, t.nfmt);
          if (format != VK_FORMAT_UNDEFINED)
            GetRT(base, t.w, t.h, format);
        }
        if (base && g_rts.count(base)) {
          multi_storage[i] = base;
          multi_transition_source |=
              g_rts[base].layout != VK_IMAGE_LAYOUT_GENERAL;
        }
        continue;
      }
      // Render targets are plain 2D images, so only a plain 2D binding may
      // resolve to one. An array or volume binding declared a sampler type no
      // render target can satisfy.
      const bool rt_eligible = !t.arrayed && !t.is_3d;
      // One base can hold several render-target geometries, and only the live
      // one answers to the address. Pick the variant this sample is asking for
      // before deciding what the binding resolves to -- but never while the
      // base is a target of this same draw, where the feedback path below owns
      // the image.
      if (base && rt_eligible && !is_bound_target(base)) {
        ActivateSampledRtVariant(base, t.w, t.h);
        // Same for depth, but never for the target this draw is testing
        // against: that one has to stay the attachment.
        if (base != d.depth_base)
          ActivateSampledDepthVariant(base, t.w, t.h);
      }
      if (base && rt_eligible && !g_rts.count(base) && !g_depths.count(base)) {
        bool depth_format = t.dfmt == 4 && t.nfmt == 7;
        u64 resolved =
            depth_format ? ResolveSampledDepth(base, t.w, t.h) : 0;
        if (!resolved)
          resolved = ResolveSampledRT(base, t.w, t.h);
        if (!resolved && !depth_format)
          resolved = ResolveSampledDepth(base, t.w, t.h);
        if (resolved) {
          base = resolved;
          // Now that the address has resolved to a target, make the variant
          // this sample asked for the live one.
          if (base != d.depth_base)
            ActivateSampledDepthVariant(base, t.w, t.h);
        }
      }
      if (kTexForce && t.base == (u64)kTexForce) {
        static int forced = 0;
        if (forced++ < 6)
          BASE_LOGI("texforce", "ps={:#x} bind={} base={:#x} -> white",
                    (unsigned long)d.ps_addr, i, (unsigned long)t.base);
        base = 0;  // resolves to nothing -> the white fallback is bound
        multi_color[i] = 0;
        multi_feedback[i] = 0;
        multi_depth[i] = 0;
        multi_views[i] = VK_NULL_HANDLE;
        continue;
      }
      // A pending CS write into an RT-resolved range must reach the image
      // before this draw samples it (the flush uploads it -- see
      // UploadCsRangeToRt); guest-upload textures already get this from the
      // texture cache.
      if (base && g_rts.count(base))
        FlushCsWritesRange(renderer, base,
                           u64(g_rts[base].w) * g_rts[base].h * 8);
      if (base && rt_eligible && is_bound_target(base) && g_rts.count(base) &&
          g_rts[base].ever_rendered) {
        multi_feedback[i] = base;
        multi_transition_source = true;
      } else if (base && rt_eligible && g_rts.count(base) &&
                 g_rts[base].ever_rendered) {
        multi_color[i] = base;
        multi_transition_source |=
            g_rts[base].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
            g_rts[base].dirty_for_read;
      } else if (base && rt_eligible && g_depths.count(base) &&
                 (base != d.depth_base || !d.depth_write_enable)) {
        multi_depth[i] = base;
        multi_transition_source |=
            g_depths[base].layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      } else if (base && rt_eligible && ResolveSampledStencil(base)) {
        // A deferred lighting pass reads the material id it stencilled during
        // the G-buffer pass. That plane lives in the depth image, not in any
        // colour target, so without this the address resolved to whatever RT
        // overlapped it and the shader discarded every pixel.
        multi_stencil_src[i] = ResolveSampledStencil(base);
        multi_transition_source |=
            g_depths[multi_stencil_src[i]].stencil_layout !=
            VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
      } else {
        multi_views[i] = !t.storage && t.null_descriptor
                             ? (t.is_3d      ? g_tex.zero_3d_view
                                : t.arrayed  ? g_tex.zero_array_view
                                             : g_tex.zero_view)
                             : TexViewFor(t);
        if (multi_views[i] && t.null_descriptor)
          multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    // DELTA_GPU_BINDTRACE=<ps addr>: which bucket each sampler binding of a
    // draw landed in. A binding that falls through to the guest-texture path
    // reads memory that draws never write, i.e. black, and nothing else in the
    // pipeline reports it.
    if (kBindTrace && d.ps_addr == (u64)kBindTrace) {
      static int n = 0;
      if (n++ < 24) {
        for (u32 i = 0; i < multi_n; i++)
          BASE_LOGI("bind",
                    "ps={:#x} b{} base={:#x} {}x{} -> {}",
                    (unsigned long)d.ps_addr, i,
                    (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
                    multi_storage[i]  ? "storage"
                    : multi_feedback[i] ? "feedback"
                    : multi_color[i]  ? "rt-color"
                    : multi_depth[i]  ? "rt-depth"
                                      : "GUEST-TEXTURE");
      }
    }
    // A shader may sample an image through one binding and write the same image
    // through another storage binding. Both descriptors must use GENERAL.
    for (u32 i = 0; i < multi_n; i++) {
      if (!multi_color[i])
        continue;
      for (u32 j = 0; j < multi_n; j++)
        if (multi_storage[j] == multi_color[i]) {
          multi_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;
          break;
        }
    }
  }

  // DELTA_GPU_TEXBIND=<frame>: how each sampler of each draw resolved -- to a
  // live render target, a depth target, guest memory, or the 1x1 white default.
  // A post-processing chain that samples its own previous target reads zero the
  // moment one of those lands on guest memory.
  {
    // A frame NUMBER is not reproducible across runs (the intro's length
    // varies), so this arms at the first frame at or after it and stops on a
    // line budget rather than at a frame boundary.
    static int texbind_lines = 0;
    const bool texbind =
        kTexBindFrame >= 0 && (int)g_frame.num >= kTexBindFrame &&
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
                (unsigned)(tex_set != VK_NULL_HANDLE));
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

  VkDeviceSize voff = g_ring.vb_offset, ioff = aligned_ioff;

  // Switch render target. Re-begin when the primary target or the MRT count
  // changes (the open region's attachment count must match the pipeline's), or
  // when a new RT-as-texture source still needs a read transition. Barriers
  // cannot be recorded inside dynamic rendering; consecutive layer composites
  // often keep the same target while switching sources, so that source change
  // must also close/reopen the region.
  u32 mrt_n = std::min(d.mrt_count, 8u);
  bool transition_source =
      rp->multi_tex
          ? multi_transition_source
          : feedback_as_tex ||
                (color_as_tex &&
                 g_rts[tex_base].layout !=
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
                (depth_as_tex && g_depths[tex_base].layout !=
                                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
  bool pending_depth_clear = d.depth_base && g_depths.count(d.depth_base) &&
                             g_depths[d.depth_base].clear_pending;
  // Any binding of this draw that names the bound depth buffer forces the depth
  // attachment read-only for the whole region, so the sampled view and the
  // attachment agree on one layout.
  bool samples_bound_depth = depth_self_read;
  for (u32 i = 0; i < multi_n && !samples_bound_depth; i++)
    samples_bound_depth = multi_depth[i] && multi_depth[i] == d.depth_base;
  // The pipeline carried the format the region was opened with; a draw that
  // re-binds the same base at a different format or extent (Astro's post chain
  // recycles a target as B10G11R11 then R16G16B16A16) needs a NEW region, or
  // the pipeline and the attachment disagree -- the validation layer flags the
  // pairing and the pixels are interpreted in the wrong format.
  bool mrt_sig_changed =
      g_region.cur_area_w != d.rt_w || g_region.cur_area_h != d.rt_h;
  for (u32 i = 0; i < mrt_n && !mrt_sig_changed; i++)
    mrt_sig_changed =
        g_region.cur_fmt[i] != (u32)ColorTargetFormat(d.mrt_info[i]) ||
        g_region.cur_w[i] !=
            RtSurfaceExtent(d.mrt_surf_w[i], d.rt_w, 256) ||
        g_region.cur_h[i] != RtSurfaceExtent(d.mrt_surf_h[i], d.rt_h, 64);
  bool restart_region = g_region.cur_rt != d.rt_base ||
                         g_region.cur_mrt_count != mrt_n ||
                         g_region.cur_depth != d.depth_base ||
                         g_region.cur_stencil != d.stencil_base ||
                         g_region.depth_read_only != samples_bound_depth ||
                         mrt_sig_changed || transition_source ||
                         pending_depth_clear;
  if (restart_region) {
    EndRegion();
    if (!rp->multi_tex && color_as_tex && transition_source) {
      auto& src = g_rts[tex_base];
      // DELTA_GPU_RTTINT: overwrite the sampled source with solid blue just
      // before the reader takes it, to tell "the reader is bound to this image"
      // apart from "this image had no content".
      if (kTint) {
        ImageBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     ColorImageAccess(src.layout), VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearColorValue blue{{0.f, 0.f, 1.f, 1.f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
                                            1};
        vkCmdClearColorImage(g_frame.cmd, src.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &blue, 1,
                             &range);
        src.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      }
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        ImageBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     ColorImageAccess(src.layout), VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    if (!rp->multi_tex && depth_as_tex && transition_source) {
      auto& src = g_depths[tex_base];
      if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        DepthBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      }
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
          if (dst.layout != VK_IMAGE_LAYOUT_GENERAL) {
            // Storage images are read *and* written (imageLoad/imageStore), so
            // the destination access needs both bits, and the source access
            // must cover every layout an RT can arrive from (a hand-rolled
            // subset missed TRANSFER_DST, leaving the transition unordered
            // against the clear that put it there).
            ImageBarrier(
                g_frame.cmd, dst.image, dst.layout, VK_IMAGE_LAYOUT_GENERAL,
                ColorImageAccess(dst.layout),
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            dst.layout = VK_IMAGE_LAYOUT_GENERAL;
          }
        } else if (multi_color[i]) {
          auto& src = g_rts[multi_color[i]];
          // DELTA_GPU_RTTINT, multi-binding path: paint the source solid blue
          // right before the reader takes it. If the reader still comes out
          // black, it is not sampling this image at all.
          if (kTint) {
            if (src.layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
              ImageBarrier(g_frame.cmd, src.image, src.layout,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           ColorImageAccess(src.layout),
                           VK_ACCESS_TRANSFER_WRITE_BIT);
              src.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            }
            const VkClearColorValue blue{{0.f, 0.f, 1.f, 1.f}};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                                0, 1};
            vkCmdClearColorImage(g_frame.cmd, src.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &blue, 1,
                                 &range);
          }
          const VkImageLayout desired = multi_layouts[i];
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
            const VkAccessFlags access =
                desired == VK_IMAGE_LAYOUT_GENERAL
                    ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                    : VK_ACCESS_SHADER_READ_BIT;
            ImageBarrier(g_frame.cmd, src.image, src.layout, desired,
                         ColorImageAccess(src.layout), access);
            src.layout = desired;
            src.dirty_for_read = false;
          }
        } else if (multi_stencil_src[i]) {
          auto& src = g_depths[multi_stencil_src[i]];
          if (src.stencil_layout != VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL) {
            StencilBarrier(g_frame.cmd, src.image, src.stencil_layout,
                           VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT);
            src.stencil_layout = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
          }
        } else if (multi_depth[i]) {
          auto& src = g_depths[multi_depth[i]];
          if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
            DepthBarrier(g_frame.cmd, src.image, src.layout,
                         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
            src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
          }
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
      // The format each binding's view ended up with. A render target the T#
      // does not describe hands over its OWN format, and if that is an integer
      // one the binding may not be filtered -- the sampler is otherwise built
      // from the T#, which says nothing about it
      // (VUID-vkCmdDrawIndexed-magFilter-04553).
      VkFormat multi_formats[kMaxTex] = {};
      for (u32 i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;
          multi_formats[i] = g_rts[multi_storage[i]].fmt;
        } else if (multi_feedback[i]) {
          auto& src = g_rts[multi_feedback[i]];
          multi_views[i] = SampledView(src, d.texs[i].swizzle, true);
          multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          multi_formats[i] = src.fmt;
        } else if (multi_color[i]) {
          // Use the numeric type the T# names, not the one the attachment was
          // created with: Vulkan requires the view's numeric type to match the
          // shader's sampled type, and the two disagree whenever a pass renders
          // a plane as UNORM that a later shader reads as UINT (or vice versa).
          multi_views[i] = SampledViewAs(
              g_rts[multi_color[i]], d.texs[i].swizzle,
              GuestTextureFormat(d.texs[i].dfmt, d.texs[i].nfmt),
              &multi_formats[i]);
          multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (multi_stencil_src[i]) {
          multi_views[i] =
              StencilSampledView(g_depths[multi_stencil_src[i]]);
          multi_layouts[i] = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
          multi_formats[i] = VK_FORMAT_S8_UINT;
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      tex_set =
          GetMultiTexSet(d, rp->tex_set_layout, rp->tex_bindings, multi_views,
                         multi_layouts, multi_formats,
                         multi_depth);
      if (!tex_set)
        return Decline(kGuestTex);
    }
    RTarget* rt =
        d.rt_base ? GetRT(d.rt_base, RtSurfaceExtent(d.mrt_surf_w[0], d.rt_w, 256),
                          RtSurfaceExtent(d.mrt_surf_h[0], d.rt_h, 64),
                          ColorTargetFormat(d.mrt_info[0]))
                  : nullptr;
    if (d.rt_base && !rt)
      return true;  // RT cap hit: treat as handled (dropped)
    if (!BeginRegion(d.mrt_base, d.mrt_info, mrt_n, d.rt_w, d.rt_h,
                      d.depth_base, d.depth_clear, d.stencil_base,
                      d.stencil_clear, samples_bound_depth, DepthW(d),
                      DepthH(d), d.mrt_surf_w, d.mrt_surf_h))
      return true;
  }
  // DB_RENDER_CONTROL clear. The guest issues a RECT_LIST with no vertex
  // buffers and no pixel shader, and the hardware fills the depth/stencil
  // plane with DB_DEPTH_CLEAR / DB_STENCIL_CLEAR over it -- the shader's
  // output is not used. Running it as an ordinary draw wrote the vertex
  // shader's z instead, which for P.T. is one packet that flattened the whole
  // scene depth it had just rendered, and every pass that samples that depth
  // (its SSAO first) then had nothing to read.
  if ((d.depth_clear_draw || d.stencil_clear_draw) && d.depth_base &&
      g_depths.count(d.depth_base)) {
    VkClearAttachment ca{};
    ca.aspectMask =
        (d.depth_clear_draw ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u) |
        (d.stencil_clear_draw ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    ca.clearValue.depthStencil = {d.depth_clear, d.stencil_clear};
    VkClearRect cr{};
    cr.rect = {{0, 0}, {d.rt_w, d.rt_h}};
    cr.baseArrayLayer = 0;
    cr.layerCount = 1;
    vkCmdClearAttachments(g_frame.cmd, 1, &ca, 1, &cr);
    g_depths[d.depth_base].dirty_for_read = true;
    g_frame.draws++;
    return true;
  }
  if (rp->multi_tex) {
    if (!tex_set) {
      VkFormat multi_formats[kMaxTex] = {};
      for (u32 i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;
          multi_formats[i] = g_rts[multi_storage[i]].fmt;
        } else if (multi_feedback[i]) {
          multi_views[i] =
              SampledView(g_rts[multi_feedback[i]], d.texs[i].swizzle, true);
          multi_formats[i] = g_rts[multi_feedback[i]].fmt;
        } else if (multi_color[i]) {
          multi_views[i] = SampledViewAs(
              g_rts[multi_color[i]], d.texs[i].swizzle,
              GuestTextureFormat(d.texs[i].dfmt, d.texs[i].nfmt),
              &multi_formats[i]);
        } else if (multi_stencil_src[i]) {
          multi_views[i] =
              StencilSampledView(g_depths[multi_stencil_src[i]]);
          multi_layouts[i] = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
          multi_formats[i] = VK_FORMAT_S8_UINT;
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      tex_set =
          GetMultiTexSet(d, rp->tex_set_layout, rp->tex_bindings, multi_views,
                         multi_layouts, multi_formats,
                         multi_depth);
    }
    if (!tex_set)
      return Decline(kGuestTex);
  } else if (feedback_as_tex) {
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    VkFormat formats[kMaxTex] = {};
    views[0] = SampledView(g_rts[tex_base], d.tex_swizzle, true);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    formats[0] = g_rts[tex_base].fmt;
    tex_set =
        GetMultiTexSet(d, g_tex.ds_layout, 1, views, layouts, formats, nullptr);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (color_as_tex) {
    auto& src = g_rts[tex_base];
    if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      return Decline(kMidRegion);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    VkFormat formats[kMaxTex] = {};
    views[0] = SampledView(src, d.tex_swizzle);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    formats[0] = src.fmt;
    tex_set =
        GetMultiTexSet(d, g_tex.ds_layout, 1, views, layouts, formats, nullptr);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (depth_as_tex) {
    auto& src = g_depths[tex_base];
    if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL)
      return Decline(kMidRegion);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = SampledView(src, d.tex_swizzle);
    layouts[0] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    const u64 depth_only[kMaxTex] = {tex_base};
    VkFormat formats[kMaxTex] = {};
    tex_set = GetMultiTexSet(d, g_tex.ds_layout, 1, views, layouts, formats,
                             depth_only);
    if (!tex_set)
      return Decline(kMidRegion);
  }

  g_ns_dr_tex += NowNs() - t_tex;
  const u64 t_bind = NowNs();
  ScopeNs bind_timer(&g_ns_dr_bind);
  (void)t_bind;
  SetGuestViewport(d);
  vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->pipe);
  // The CONSTANT_* blend factors read these, and they change per draw --
  // keying a pipeline on them would multiply the cache instead.
  vkCmdSetBlendConstants(g_frame.cmd, d.blend_constants);
  // 16 user-data dwords per stage, in its own half of the shared push range:
  // both stages at offset 0 meant the second push overwrote the first.
  vkCmdPushConstants(g_frame.cmd, rp->layout, kPcStages, 0, 64,
                     d.vs_user_data);
  vkCmdPushConstants(g_frame.cmd, rp->layout, kPcStages, 64, 64,
                     d.ps_user_data);
  if (gpu::gcn::PushCodeBase()) {
    // Each stage's OWN code address, for s_getpc_b64: the modules are keyed by
    // content, so the address cannot live in the SPIR-V, and VS and PS live at
    // different addresses so each stage gets its own words (the shared-offset
    // mistake the user-data pushes already made once).
    const u32 vs_base[2] = {static_cast<u32>(d.vs_addr),
                                 static_cast<u32>(d.vs_addr >> 32)};
    const u32 ps_base[2] = {static_cast<u32>(d.ps_addr),
                                 static_cast<u32>(d.ps_addr >> 32)};
    vkCmdPushConstants(g_frame.cmd, rp->layout, kPcStages, 128,
                       8, vs_base);
    vkCmdPushConstants(g_frame.cmd, rp->layout, kPcStages, 136, 8,
                       ps_base);
  }
  // Copy each guest cbuffer window into the per-frame ring and bind set 1.
  // Vulkan requires one dynamic offset for every dynamic descriptor in the set
  // layout.
  VkDeviceSize cb_off = (g_ring.ubo_offset + g_ring.ubo_align - 1) &
                        ~(VkDeviceSize)(g_ring.ubo_align - 1);
  VkDeviceSize cb_stride = g_ring.ubo_stride;
  if (cb_off + cb_stride * kCbufBindings > g_ring.ubo_end) {
    if (kGpuDecltrace) {
      static int n = 0;
      if (n++ < 32)
        BASE_LOGI("decl", "ring=UBO off={} stride={} end={}",
                  (unsigned long long)cb_off, (unsigned long long)cb_stride,
                  (unsigned long long)g_ring.ubo_end);
    }
    return Decline(kRing);
  }
  u32 dyn_off[kCbufBindings];
  u32 cbuf_mask = 0;
  VkDeviceSize next = cb_off;
  for (u32 i = 0; i < kCbufBindings; i++) {
    const auto& cb = d.cbufs[i];
    const u32 readable = std::min(cb.size, kCbufWindow);
    const bool have_cbuf = readable && IsReadableThisFrame(cb.base, readable);
    if (have_cbuf)
      cbuf_mask |= 1u << i;
    if (!have_cbuf && i != 0) {
      dyn_off[i] = 0;  // shared zero window (see BeginFrame)
      continue;
    }
    // The copied length is a pure function of the binding (planned size,
    // widened to the page unless DELTA_GPU_TIGHTCBUF), so it doubles as the
    // cache key: a window of the same guest bytes at the same length staged
    // earlier this frame is byte-identical, and the draw just rebinds it.
    u32 cache_n = 0;
    if (have_cbuf) {
      const u32 planned = cb.size < kCbufWindow ? cb.size : kCbufWindow;
      const u64 page_end = (cb.base + 0x1000) & ~u64{0xFFF};
      const u32 avail = static_cast<u32>(
          std::min<u64>(kCbufWindow, page_end - cb.base));
      cache_n = kTightCbuf ? planned : std::max(planned, avail);
      if (kRingDedup) {
        const VkDeviceSize cached = g_ubo_staged.Find(cb.base, cache_n);
        if (cached != VkDeviceSize(-1)) {
          dyn_off[i] = static_cast<u32>(cached);
          continue;
        }
      }
    }
    u8* cb_dst = g_ring.ubo_map + next;
    u32 n;
    if (have_cbuf && !FlushCsWritesRange(renderer, cb.base, kCbufWindow))
      return Decline(kNoRecomp);
    if (have_cbuf) {
      // Upload as much of the window as the base's page holds, not just the
      // recompiler's planned size (computed above as cache_n). A shader that
      // indexes its constants dynamically -- a UI batch picking a per-quad
      // transform out of an array -- reads past the planned size, and the
      // truncated copy left those entries zero: Skyrim's menu drew its sprite
      // atlas at screen size over everything. Clamped to the page so a cbuffer
      // at the end of a mapping cannot fault.
      n = cache_n;
      std::memcpy(cb_dst, reinterpret_cast<const void*>(cb.base), n);
    } else {  // binding 0 without a resolved cbuffer: the heuristic MVP
      n = sizeof(d.mvp);
      std::memcpy(cb_dst, d.mvp, n);
    }
    // DELTA_GPU_CBINFO: what the shader will actually read. The CPU-side
    // trace prints the guest buffer; this prints the bytes that reached the
    // ring slot the descriptor points at, which is what the SPIR-V loads.
    // DELTA_GPU_DRAWRT_FRAME also gates this: unfiltered, the 40-line cap is
    // spent on startup frames, where a light buffer is legitimately all zero
    // and reads as the very corruption being looked for.
    if (kCbInfo && have_cbuf && n >= 16 &&
        (!kWantFrame || g_frame.num == kWantFrame) &&
        (kCbInfo == 1 || d.vs_addr == (u64)kCbInfo)) {
      static int shown = 0;
      if (shown++ < 40) {
        const float* f = reinterpret_cast<const float*>(cb_dst);
        base::String line;
        base::FormatTo(line, "vs={:#x} bind={} base={:#x} n={} @0: ",
                       (unsigned long)d.vs_addr, i, (unsigned long)cb.base, n);
        // Sixteen, for the same reason as the drawrt dump: the window walk
        // below starts at 0x40, so eight left bytes 32..63 unprintable.
        for (u32 k = 0; k < 16 && k * 4 < n; k++)
          base::FormatTo(line, " {:g}", f[k]);
        // Every 0x40 window, not a chosen few: the value a shader actually
        // multiplies by is as likely to sit at byte 376 (P.T.'s light pass
        // scales every light colour by a scalar there) as at 0x40 or 0xc0.
        for (u32 off = 0x40u; off + 4 <= n; off += 0x40u) {
          base::FormatTo(line, " | @{:#x}:", off);
          for (u32 k = off / 4; k < off / 4 + 16 && k * 4 + 4 <= n; k++)
            base::FormatTo(line, " {:g}", f[k]);
        }
        BASE_LOGI("cbinfo", "{}", line.c_str());
      }
    }
    const size_t window = static_cast<size_t>(next / cb_stride);
    if (window >= g_ring.ubo_written.size())
      return Decline(kRing);
    const u32 previous = g_ring.ubo_written[window];
    if (n < previous)
      std::memset(cb_dst + n, 0, previous - n);
    g_ring.ubo_written[window] = n;
    if (have_cbuf && kRingDedup)
      g_ubo_staged.Insert(cb.base, cache_n, 0, next);
    dyn_off[i] = static_cast<u32>(next);
    next += cb_stride;
  }
  g_ring.ubo_offset = next;
  vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          rp->layout, 1, 1, &g_ring.ubo_set, kCbufBindings,
                          dyn_off);
  // Stage the raw buffers the shader indexes by hand (set 2). Only the leading
  // window of each is copied -- a MUBUF address is a per-lane index with no
  // static bound, so there is no "planned size" to copy exactly -- and the
  // recompiled shader clamps into that window. Repeats within a frame (the
  // same skinning palette across a character's draws) reuse one upload.
  u32 rawbuf_mask = 0;
  if (rp->raw_bufs) {
    if (!EnsureRawBufferRing())
      return Decline(kRing);
    u32 sbo_dyn[kRawBufBindings] = {};
    for (u32 i = 0; i < kRawBufBindings; i++) {
      const auto& rb = d.bufs[i];
      const u32 want = std::min(rb.size, kRawBufWindow);
      if (!want || !IsReadableThisFrame(rb.base, want))
        continue;  // unresolved descriptor: the shared zero window at offset 0
      // Same per-frame cache as the vertex/index/cbuffer rings -- and unlike
      // the plain map this replaced, a window whose range a dispatch has
      // rewritten since it was staged (generation moved, or dirty right now)
      // is re-copied instead of served stale.
      const VkDeviceSize cached = g_sbo_staged.Find(rb.base, want);
      if (cached != VkDeviceSize(-1)) {
        sbo_dyn[i] = static_cast<u32>(cached);
        rawbuf_mask |= 1u << i;
        continue;
      }
      const VkDeviceSize off = (g_ring.sbo_offset + g_ring.sbo_align - 1) &
                               ~(VkDeviceSize)(g_ring.sbo_align - 1);
      // A resource SMALLER than the window takes only what it needs. Reserving
      // a whole window for each turned GTA:SA's world -- three buffers a draw,
      // ~50 KB between them -- into 3 MiB of ring per draw, so the frame ran
      // out after ~85 draws of several thousand and the rest was silently
      // dropped. A resource the window TRUNCATES keeps the full reservation:
      // there the shader's clamp really can land past the payload, and the
      // zero-fill below is what makes that read as zero rather than as the
      // next resource.
      const bool truncated = rb.size > kRawBufWindow;
      const VkDeviceSize reserve =
          truncated ? g_ring.sbo_stride
                    : ((want + g_ring.sbo_align - 1) &
                       ~(VkDeviceSize)(g_ring.sbo_align - 1));
      // The descriptor's range is a whole window wherever the dynamic offset
      // lands, so the window must fit in the BUFFER even when the reservation
      // is short and even in the second frame slot.
      if (off + reserve > g_ring.sbo_end || off + kRawBufWindow > kSboRing)
        return Decline(kRing);
      if (!FlushCsWritesRange(renderer, rb.base, want))
        return Decline(kNoRecomp);
      u8* dst = g_ring.sbo_map + off;
      std::memcpy(dst, reinterpret_cast<const void*>(rb.base), want);
      // Zero the reservation's tail so a read just past the payload is zero,
      // as it is past NUM_RECORDS on hardware. A truncated resource fills its
      // whole window, so this costs nothing there.
      if (want < reserve)
        std::memset(dst + want, 0, static_cast<size_t>(reserve - want));
      g_ring.sbo_offset = off + reserve;
      sbo_dyn[i] = static_cast<u32>(off);
      g_sbo_staged.Insert(rb.base, want, 0, off);
      rawbuf_mask |= 1u << i;
    }
    // One dynamic offset per dynamic descriptor the SET actually holds, which
    // is the device-derived sbo_count, not the compile-time ceiling. Passing
    // the ceiling is a spec violation on every device that reports fewer than
    // 16 (the Vulkan floor is 4).
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rp->layout, 2, 1, &g_ring.sbo_set,
                            g_ring.sbo_count, sbo_dyn);
    // DELTA_GPU_RAWBUF: which set-2 bindings a draw actually got, so a shader
    // reading zeros can be told apart from one reading real guest data.
    if (kRawBufTrace) {
      static int n = 0;
      if (n++ < 64)
        BASE_LOGI("rawbuf",
                  "draw#{} vs={:#x} nbufs={} staged={:#x} "
                  "sizes={}/{}/{}/{}",
                  g_frame.draws, (unsigned long)d.vs_addr, d.num_bufs,
                  rawbuf_mask, d.bufs[0].size, d.bufs[1].size,
                  d.bufs[2].size, d.bufs[3].size);
    }
  }
  if (tex_set)
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rp->layout, 0, 1, &tex_set, 0, nullptr);
  // The shared-LDS scratch never changes: one set, bound whenever the pipeline
  // declares it.
  if (rp->shared_lds)
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rp->layout, 3, 1, &g_ring.lds_set, 0, nullptr);
  // Commit ring uploads only after every fallible pipeline, texture, region and
  // cbuffer decision has succeeded. Bindings served by the per-frame cache
  // were copied by an earlier draw and only rebind.
  for (u32 j = 0; j < nbind; j++) {
    if (!bind_size[j] || vb_cached[j] != VkDeviceSize(-1))
      continue;
    if (!FlushCsWritesRange(renderer,
                            reinterpret_cast<u64>(d.vbufs[j].data),
                            bind_size[j]))
      return Decline(kNoRecomp);
    std::memcpy(g_ring.vb_map + voff + bind_off[j], d.vbufs[j].data,
                (size_t)bind_size[j]);
    if (kRingDedup)
      g_vb_staged.Insert(reinterpret_cast<u64>(d.vbufs[j].data),
                         bind_size[j], 0, voff + bind_off[j]);
  }
  if (indexed && ib_cached == VkDeviceSize(-1)) {
    CopyGuestIndices(g_ring.ib_map + ioff, d.index_data, d.index_count,
                     d.index_type);
    if (kRingDedup)
      g_ib_staged.Insert(reinterpret_cast<u64>(d.index_data), index_bytes,
                         1u + d.index_type, ioff);
  }
  if (nbind) {
    VkBuffer bufs[8];
    VkDeviceSize offs[8];
    for (u32 j = 0; j < nbind; j++) {
      bufs[j] = g_ring.vb;
      offs[j] = vb_cached[j] != VkDeviceSize(-1) ? vb_cached[j]
                                                 : voff + bind_off[j];
    }
    vkCmdBindVertexBuffers(g_frame.cmd, 0, nbind, bufs, offs);
  }
  if (indexed)
    vkCmdBindIndexBuffer(
        g_frame.cmd, g_ring.ib,
        ib_cached != VkDeviceSize(-1) ? ib_cached : ioff,
        d.index_type == 1 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      BASE_LOGI("dt",
                "RECOMP DREW count={} rt={:#x} nv={} multi_tex={}",
                draw_count, (unsigned long)d.rt_base, nv, (int)rp->multi_tex);
  }
  CmdInsertLabel(g_frame.cmd, "recomp vs=%#llx ps=%#llx n=%u%s",
                 (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr,
                 indexed ? d.index_count : d.vertex_count,
                 indexed ? " indexed" : "");
  if (indexed)
    vkCmdDrawIndexed(g_frame.cmd, d.index_count,
                     d.instance_count ? d.instance_count : 1, 0, 0, 0);
  else
    vkCmdDraw(g_frame.cmd, d.vertex_count,
              d.instance_count ? d.instance_count : 1, 0, 0);
  // DELTA_GPU_DRAWSEQ=<n>: the first n draws of the run in record order, with
  // the frame they belong to -- the per-frame filters cannot show that a pass
  // and the pass that reads it landed in different frames.
  {
    static int seq = 0;
    if (seq < kSeqN) {
      BASE_LOGI("seq", "{} f{} draw#{} rt={:#x} tex={:#x}{}", seq++,
                g_frame.num, g_frame.draws, (unsigned long)d.rt_base,
                (unsigned long)tex_base, color_as_tex ? " RT-AS-TEX" : "");
    }
  }
  // DELTA_GPU_DRAWRT=<base>: report what was actually issued for one target.
  {
    static int shown = 0;
    // DELTA_GPU_DRAWRT=1 logs EVERY draw (the whole frame graph) instead of one
    // target's draws, so a broken producer/consumer link is visible directly.
    const bool all = kWant == 1;
    // DELTA_GPU_DRAWRT_FRAME=N: only this frame, so the graph is a steady-state
    // frame rather than the opening composites.
    // DELTA_GPU_DRAWRT_BUSY=N: only frames that reach N draws, for screens whose
    // frame number moves between runs.
    // A composite target takes tens of draws per frame and the ones that
    // decide its final content are the LAST few, so a cap of 8 shows only the
    // ones that get overwritten.
    // With a frame filter the output is bounded by that frame's draw count, so
    // the whole graph can be printed; without one, cap it. The end of the graph
    // is the part that decides what is presented, and a low cap never reaches
    // it.
    // Printing the whole graph needs ONE frame, and a frame number is not
    // reproducible across runs while a draw count is: with DRAWRT=1 and a busy
    // threshold, latch onto the first frame that reaches it and print all of
    // that frame, from its first draw.
    static int latched = 0;
    if (all && kBusy && !kWantFrame && !latched && (int)g_frame.draws >= kBusy)
      latched = g_frame.num + 1;  // this one is already half gone
    const int want_frame = kWantFrame ? kWantFrame : latched;
    const int cap = all ? (want_frame ? 100000 : 140) : 64;
    // A pass is often easier to name by the SHADER it runs than by the target
    // it writes: a deferred light draw lands on the same address as the base
    // pass, and the target filter cannot separate them.
    if (kWant && (all || g_region.cur_rt == kWant || d.ps_addr == kWant) &&
        shown < cap &&
        (!want_frame || (int)g_frame.num == want_frame) &&
        (latched || (int)g_frame.draws >= kBusy)) {
      shown++;
      BASE_LOGI("drawrt",
                "f{} #{} rt={:#x} {}x{} indexed={} vcount={} "
                "icount={} inst={} prim={} tmask={:#x} num_vbufs={} stride={} "
                "mrt={} vp={:g},{:g} scale {:g},{:g} off vs={:#x} ps={:#x}",
                g_frame.num, g_frame.draws,
                (unsigned long)g_region.cur_rt, d.rt_w, d.rt_h, (int)indexed,
                d.vertex_count, d.index_count, d.instance_count,
                d.prim_type, d.target_mask,
                d.num_vbufs, d.vertex_stride, d.mrt_count,
                d.viewport_x_scale, d.viewport_y_scale, d.viewport_x_offset,
                d.viewport_y_offset, (unsigned long)d.vs_addr,
                (unsigned long)d.ps_addr);
      BASE_LOGI("drawrt",
                " blend en={} ctrl={:#x} tmask={:#x} surf={}x{} "
                "zscale={:g} zoff={:g} scissor=({},{})-({},{})",
                (int)d.blend_enable, d.blend_control, d.target_mask,
                d.rt_surf_w, d.rt_surf_h, d.viewport_z_scale,
                d.viewport_z_offset, d.scissor_tl & 0x7FFF,
                (d.scissor_tl >> 16) & 0x7FFF, d.scissor_br & 0x7FFF,
                (d.scissor_br >> 16) & 0x7FFF);
      // Every bound target's own CB_COLORn_INFO. The colour FORMAT and
      // NUMBER_TYPE decide whether the hardware clamps a write at 1.0 or keeps
      // it, which is the difference between a highlight and a blown one.
      for (u32 m = 0; m < d.mrt_count && m < 8; m++)
        BASE_LOGI("drawrt",
                  " mrt{} base={:#x} info={:#x} fmt={} ntype={}", m,
                  (unsigned long)d.mrt_base[m], d.mrt_info[m],
                  (d.mrt_info[m] >> 2) & 0x1F, (d.mrt_info[m] >> 8) & 0x7);
      // Every bound target's own CB_BLENDn_CONTROL. An MRT pass whose second
      // attachment blends differently from its first is invisible in the
      // single-target line above, and a wrong enable there accumulates over a
      // pass that draws the same target dozens of times.
      for (u32 m = 1; m < d.mrt_count && m < 8; m++)
        BASE_LOGI("drawrt", " blend{} en={} ctrl={:#x} base={:#x}", m,
                  (int)((d.mrt_blend_mask >> m) & 1u), d.mrt_blend[m],
                  (unsigned long)d.mrt_base[m]);
      // Whether this pass writes depth, and into what, decides whether a later
      // depth-sampling pass has anything to read.
      BASE_LOGI("drawrt",
                " depth base={:#x} valid={} test={} write={} "
                "func={} clear={:g} stencil={} cleardraw={}/{} rc={:#x} sbase={:#x}",
                (unsigned long)d.depth_base, (int)d.depth_valid,
                (int)d.depth_test_enable, (int)d.depth_write_enable,
                d.depth_func, d.depth_clear, (int)d.stencil_enable,
                (int)d.depth_clear_draw, (int)d.stencil_clear_draw,
                d.render_control, (unsigned long)d.stencil_base);
      // A single-texture pipeline never fills the multi_* arrays -- it binds
      // through color_as_tex/depth_as_tex below -- so reading them here would
      // report every such draw as a MISS that resolved fine.
      // Whether the legacy path's one binding resolved to a real guest
      // texture or fell back to a 1x1 default -- without this every
      // single-texture draw reported MISS even when its upload was fine, which
      // is a false lead pointing straight at the texture cache.
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
      const u32 shown_n = rp->multi_tex ? multi_n : std::min(multi_n, 1u);
      // How many textures the recompiler found vs how many are being reported:
      // a shader with several image_samples that took the single-texture path
      // reads only the first, and the trace would look identical to a genuine
      // one-texture blit.
      BASE_LOGI("drawrt", " texs={} multi={} n={}",
                d.recomp->ps_texs.size(), (int)rp->multi_tex, multi_n);
      for (u32 i = 0; i < shown_n; i++) {
        const auto& t = d.texs[i];
        if (!t.base) {
          // An UNRESOLVED binding is the interesting one: the shader samples
          // it and reads the default, and skipping it here made a shader with
          // thirteen samplers look like one with five.
          BASE_LOGI("drawrt", " tex{} UNRESOLVED (samples the default)", i);
          continue;
        }
        BASE_LOGI("drawrt",
                  " tex{} mips={} basemip={} viewmips={} "
                  "minlod={} layers={} arr={} lod0={} cmp={} sto={} "
                  "swz={:#x} smp={} {:08x} {:08x} {:08x} {:08x}",
                  i, t.mip_levels, t.base_mip, t.view_mips, t.min_lod,
                  t.layers, (int)t.arrayed, (int)t.force_lod_zero,
                  (int)t.depth_compare, (int)t.storage, t.swizzle,
                  (int)t.sampler_valid, t.sampler[0], t.sampler[1],
                  t.sampler[2], t.sampler[3]);
        if (t.src && gpu::IsReadableRange(t.src, 32)) {
          const u32* tw = reinterpret_cast<const u32*>(t.src);
          BASE_LOGI("drawrt",
                    " tex{} T# src={:#x} [{:08x} {:08x} {:08x} {:08x} "
                    "{:08x} {:08x} {:08x} {:08x}]",
                    i, (unsigned long)t.src, tw[0], tw[1], tw[2], tw[3],
                    tw[4], tw[5], tw[6], tw[7]);
        }
        const bool resolved_guest =
            rp->multi_tex ? multi_views[i] != VK_NULL_HANDLE : legacy_resolved;
        BASE_LOGI("drawrt",
                  " tex{} {:#x} {}x{} dfmt={} tiling={} -> {}{:#x}",
                  i, (unsigned long)t.base, t.w, t.h, t.dfmt, t.tiling,
                  rc[i]           ? "rt "
                  : rf[i]         ? "fb "
                  : rd[i]         ? "depth "
                  : resolved_guest ? "guest "
                                   : "MISS ",
                  (unsigned long)(rc[i]   ? rc[i]
                                  : rf[i] ? rf[i]
                                  : rd[i] ? rd[i]
                                          : 0));
        // A MISS is the interesting case and says nothing on its own: name the
        // render-target state the binding was matched against.
        // Also report a binding that fell through to guest memory while its
        // address IS a live render target: that is a resolution failure with a
        // plausible-looking result, which is worse than a MISS because the
        // draw silently samples stale bytes instead of the image.
        if (resolved_guest && !rc[i] && !rf[i] && !rd[i] &&
            g_rts.count(t.base)) {
          const auto& rt = g_rts[t.base];
          BASE_LOGI("drawrt",
                    " tex{} GUEST-OVER-RT: live={}x{} "
                    "ever_rendered={} variants={}",
                    i, rt.w, rt.h, (int)rt.ever_rendered,
                    g_rt_variants.count(t.base)
                        ? g_rt_variants[t.base].size()
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
      // The first vertex each binding delivers, and the indices that select it.
      // A pass that covers the screen with the wrong colour is either drawing
      // the wrong geometry or reading the wrong attributes, and only the bytes
      // behind the attribute say which.
      for (u32 j = 0; j < std::min(d.num_vbufs, 4u); j++) {
        const auto& vb = d.vbufs[j];
        if (!vb.data || !gpu::IsReadableRange((u64)(uintptr_t)vb.data, 64))
          continue;
        base::String bytes;
        const auto* p = static_cast<const u8*>(vb.data);
        for (u32 b = 0; b < std::min<u32>(vb.stride ? vb.stride : 16, 48); b++)
          base::FormatTo(bytes, "{:02x}", p[b]);
        BASE_LOGI("drawrt", " vb{} @{:#x} stride={} v0={}", j,
                  (unsigned long)(uintptr_t)vb.data, vb.stride, bytes.c_str());
      }
      // The vertex stage's user data, and what the pointer in its first pair
      // points AT. Every descriptor a shader fetches for itself hangs off one
      // of these, so a binding that resolves into dead memory is either a bad
      // table pointer or a bad slot in a good table, and nothing else in the
      // trace separates those.
      {
        base::String ud;
        for (u32 k = 0; k < 8; k++)
          base::FormatTo(ud, " {:08x}", d.vs_user_data[k]);
        const u64 table = (static_cast<u64>(d.vs_user_data[1] & 0xFFFF) << 32) |
                          d.vs_user_data[0];
        BASE_LOGI("drawrt", " vsud{} | [s0:s1]={:#x}", ud.c_str(),
                  (unsigned long)table);
        // A window either side of it, not just the target: GNM embeds a
        // shader's extended user data inline in the command buffer, so a
        // pointer that is off lands on packet headers and the distance to the
        // real block is what names the mistake.
        for (int row = -2; row < 6; row++) {
          const u64 at = table + static_cast<i64>(row) * 32;
          if (!gpu::IsReadableRange(at, 32))
            continue;
          const auto* w = reinterpret_cast<const u32*>(at);
          base::String line;
          for (u32 k = 0; k < 8; k++)
            base::FormatTo(line, " {:08x}", w[k]);
          // Each 4-dword pair read as a V#, with a census of what it names.
          // Whether the ONE descriptor a shader loads is stale is unanswerable
          // on its own; whether its NEIGHBOURS name live memory answers it.
          for (u32 half = 0; half < 2; half++) {
            const gcn::VBuffer vb = gcn::DecodeVBuffer(w + half * 4);
            const u64 bytes =
                vb.stride ? (u64)vb.stride * vb.num_records : vb.num_records;
            if (!vb.base || !bytes)
              continue;
            const u32 scan = static_cast<u32>(std::min<u64>(bytes, 4096));
            int nz = -1;
            if (gpu::IsReadableRange(vb.base, scan)) {
              nz = 0;
              const auto* b = reinterpret_cast<const u8*>(vb.base);
              for (u32 k = 0; k < scan; k++)
                nz += b[k] != 0;
            }
            base::FormatTo(line, " | V#[{}] {:#x} n={} nz={}", half,
                           (unsigned long)vb.base, (unsigned long)bytes, nz);
            // A base that reads zero may still be right-but-biased: the same
            // shape as the T# arena bias DELTA_GPU_ARENA_PROBE measures. Say
            // which neighbouring 2 MiB arena, if any, does hold data, because
            // a CONSTANT bias across every descriptor is a different bug from
            // a pool the title never fills.
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
      // A shader that fetches its own vertices reads a raw buffer instead of a
      // vertex binding, so vb0 above says nothing about the geometry it draws.
      for (u32 j = 0; j < std::min(d.num_bufs, kRawBufBindings); j++) {
        const auto& rb = d.bufs[j];
        if (!rb.base || !gpu::IsReadableRange(rb.base, 16))
          continue;
        // Sampled at several offsets, not just the head: such a buffer is
        // laid out as parallel per-component arrays, so the first floats are
        // one attribute and say nothing about the positions.
        base::String head;
        for (u32 off = 0; off < 0x600 && off + 16 <= rb.size; off += 0x180) {
          const auto* f =
              reinterpret_cast<const float*>(rb.base + off);
          base::FormatTo(head, " +{:#x}={:g},{:g},{:g},{:g}", off, f[0], f[1],
                         f[2], f[3]);
        }
        // A binding that resolves to an empty range is the interesting case:
        // it looks identical in every other trace to one carrying real data.
        u32 nz = 0;
        // The WHOLE resource, sampled: "the head is zero" and "the buffer is
        // dead" are different answers, and only the second means the address
        // is wrong rather than the offsets.
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
        for (u32 k = 0; k < std::min(d.index_count, 8u); k++)
          base::FormatTo(idx, " {}", d.index_type == 1 ? p32[k] : p16[k]);
        BASE_LOGI("drawrt", " idx @{:#x} type={} :{}",
                  (unsigned long)(uintptr_t)d.index_data, d.index_type,
                  idx.c_str());
      }
      // Only the slots this draw actually declared: the rest are unused
      // array entries, and reporting them buries the one that matters.
      for (u32 c = 0; c < std::min<u32>(d.num_cbufs, kCbufBindings);
           c++) {
        const auto& cb = d.cbufs[c];
        if (!gpu::IsReadableRange(cb.base, std::min(cb.size, 192u))) {
          // Say so rather than skipping: a silently absent binding reads as a
          // pass with fewer constant buffers than it has, and a shader whose
          // transform lives in the missing one draws with a zero matrix.
          if (cb.base || cb.size)
            BASE_LOGI("drawrt", " cb{} {:#x} sz={} UNREADABLE", c,
                      (unsigned long)cb.base, cb.size);
          else
            BASE_LOGI("drawrt", " cb{} UNBOUND", c);
          continue;
        }
        const u32* w = reinterpret_cast<const u32*>(cb.base);
        base::String line;
        base::FormatTo(line, " cb{} {:#x} sz={}:", c,
                       (unsigned long)cb.base, cb.size);
        // Sixteen, not eight: with the window walk below starting at 0x40,
        // eight left bytes 32..63 unprintable -- and P.T.'s draw #38 scales its
        // whole extinction by a scalar at byte 44.
        for (u32 k = 0; k < 16 && k * 4 < cb.size; k++)
          base::FormatTo(line, " {:g}", *reinterpret_cast<const float*>(&w[k]));
        // A shader loads its constants with s_buffer_load at whatever offset
        // the compiler chose, so show EVERY 4x4-sized window the binding is big
        // enough to hold. Showing only 0x40 and 0x80 printed a plausible matrix
        // while the value the shader actually multiplies by sat unexamined:
        // P.T.'s light pass scales every light colour by a scalar at byte 376
        // of a 464-byte buffer, past the last window this used to print.
        // Including the final PARTIAL window: P.T.'s light pass decides whether
        // to alpha-test its cookie on a scalar at byte 200 of a 208-byte
        // buffer, which every whole-window walk stops just short of.
        for (u32 off = 0x40u; off + 4 <= cb.size; off += 0x40u) {
          base::FormatTo(line, "\n  @{:#x}:", off);
          for (u32 k = off / 4; k < off / 4 + 16 && k * 4 + 4 <= cb.size;
               k++)
            base::FormatTo(line, " {:g}", *reinterpret_cast<const float*>(&w[k]));
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
  if (indexed && ib_cached == VkDeviceSize(-1))
    g_ring.ib_offset = ioff + index_bytes;
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
  // Frame debugger: the complete draw, including how every sampler binding
  // resolved -- which only this function knows. Runs last so a mid-frame
  // snapshot (which must close the open region) cannot disturb the accounting
  // above.
  if (trace::Recording()) {
    // The legacy single-texture path resolves its one binding through its own
    // variables; express it in the same shape so a capture reads the same
    // either way. A `tex_set` that is one of the 1x1 defaults resolved to
    // nothing, which is the case worth naming.
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

}  // namespace gpu::vk
