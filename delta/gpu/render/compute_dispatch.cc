/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute_internal.h"

#include <cstdio>
#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/to_string.h"
#include "base/threading/lock_guard.h"
#include "base/time/time.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/gpu_timeline.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/guest_memory_table.h"
#include "gpu/render/render_target.h"
#include "gpu/render/renderer_state.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/trace.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "profile/profile.h"

namespace gpu::render {

bool CsDeclined(const ComputeInfo& ci, const char* why) {
  static base::HashSet<base::String> reported;
  static const guest::SessionReset reset_reported(
      [] { guest::ResetResource(reported); });
  base::String key = base::ToString(ci.cs_addr) + why;
  if (reported.size() < 512 && reported.insert(key).second)
    BASE_LOGI("csgpu", "CS @{:#x} declined at exit {}", ci.cs_addr, why);
  return false;
}

bool MergeOverlapping(ComputeInfo& ci) {
  constexpr u64 kAlign = 16;
  const auto plain = [&](u32 i) {
    const auto& r = ci.res[i];
    return !r.zero_fill && !r.image_staging && r.binding < 48 &&
           !TruthEligible(r);
  };
  const auto end_of = [](const ComputeInfo::Res& r) {
    return r.base + base::Max(r.size, r.guest_size);
  };
  bool changed = false;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!plain(i))
      continue;
    u64 lo = ci.res[i].base, hi = end_of(ci.res[i]);
    u32 members = 0;
    for (bool grew = true; grew;) {
      grew = false;
      members = 0;
      for (u32 j = 0; j < ci.num_res; j++) {
        if (!plain(j) || ci.res[j].base >= hi || end_of(ci.res[j]) <= lo)
          continue;
        members++;
        if (ci.res[j].base < lo || end_of(ci.res[j]) > hi) {
          lo = base::Min(lo, ci.res[j].base);
          hi = base::Max(hi, end_of(ci.res[j]));
          grew = true;
        }
      }
    }
    bool distinct = false;
    for (u32 j = 0; j < ci.num_res; j++)
      distinct |= plain(j) && ci.res[j].base < hi && end_of(ci.res[j]) > lo &&
                  ci.res[j].base != ci.res[i].base;
    if (members < 2 || !distinct)
      continue;
    lo &= ~(kAlign - 1);
    hi = (hi + 3) & ~u64(3);
    for (u32 j = 0; j < ci.num_res; j++) {
      auto& r = ci.res[j];
      if (!plain(j) || r.base < lo || end_of(r) > hi)
        continue;
      r.view_offset = r.base - lo;
      if (r.write_bytes)
        r.write_bytes += r.view_offset;
      r.base = lo;
      r.size = r.guest_size = hi - lo;
    }
    changed = true;
  }
  return changed;
}

const ComputeInfo& MergeSameBase(const ComputeInfo& ci, ComputeInfo& merged) {
  const ComputeInfo* src = &ci;
  if (kCsMergeOverlap) {
    bool overlap = false;
    for (u32 i = 0; i < ci.num_res && !overlap; i++)
      for (u32 j = 0; j < i && !overlap; j++) {
        const auto& a = ci.res[i];
        const auto& b = ci.res[j];
        overlap = !a.zero_fill && !b.zero_fill && !a.image_staging &&
                  !b.image_staging && a.base != b.base &&
                  a.base < b.base + base::Max(b.size, b.guest_size) &&
                  b.base < a.base + base::Max(a.size, a.guest_size);
      }
    if (overlap) {
      merged = ci;
      if (MergeOverlapping(merged))
        src = &merged;
    }
  }
  const ComputeInfo& in = *src;
  bool need = false;
  for (u32 i = 0; i < in.num_res && !need; i++)
    for (u32 j = 0; j < i && !need; j++)
      need = in.res[i].base == in.res[j].base && !in.res[i].zero_fill &&
             !in.res[j].zero_fill && !in.res[i].image_staging &&
             !in.res[j].image_staging &&
             (in.res[i].size != in.res[j].size ||
              in.res[i].guest_size != in.res[j].guest_size);
  if (!need)
    return in;
  if (src != &merged)
    merged = ci;
  for (u32 i = 0; i < merged.num_res; i++)
    for (u32 j = 0; j < i; j++) {
      auto& a = merged.res[i];
      auto& b = merged.res[j];
      if (a.base != b.base || a.zero_fill || b.zero_fill || a.image_staging ||
          b.image_staging)
        continue;
      a.size = b.size = base::Max(a.size, b.size);
      a.guest_size = b.guest_size = base::Max(a.guest_size, b.guest_size);
    }
  return merged;
}

void PrebuildComputePipeline(const base::Vector<u32>& spirv,
                             u32 num_res,
                             int guest_memory_binding) {
  if (!g_backend.device || spirv.empty())
    return;
  const auto slot = base::MakeShared<PrebuiltSlot>();
  {
    base::LockGuard<base::Mutex> lock(g_prebuilt_mutex);
    const u64 key = PrebuiltKey(spirv, num_res, guest_memory_binding);
    if (g_prebuilt.count(key))
      return;
    g_prebuilt.emplace(key, slot);
  }
  PrebuiltCs built;
  built.set_layout = CsSetLayout(num_res, -1, guest_memory_binding, {});
  if (built.set_layout)
    built.layout = CsPipelineLayout(built.set_layout);
  if (built.layout)
    built.pipe = CreateCsPipeline(built.layout, spirv);
  {
    base::LockGuard<base::Mutex> lock(slot->mutex);
    slot->value = built;
    slot->ready = true;
  }
  slot->built.NotifyAll();
}

u64 GuestMirror(CsRange& e, u64& bytes) {
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(e.res, tiled, linear))
    return 0;
  const TileTable* table = GetTileTable(tiled, linear);
  if (!table || !tiled.size || tiled.size > e.guest_bytes)
    return 0;
  if (!e.mirror || e.mirror_cap < tiled.size) {
    if (e.mirror)
      g_cs_frame_retired.push_back(e.mirror);
    e.mirror_cap = (tiled.size + 0xFFFF) & ~u64(0xFFFF);
    e.mirror = CreateCsBuffer(
        e.mirror_cap, rhi::kBufferStorage | rhi::kBufferAddress,
        SplitVram() ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback);
    e.mirror_seq = 0;
    if (!e.mirror || !e.mirror->address()) {
      Device().Destroy(e.mirror);
      e.mirror = nullptr;
      e.mirror_cap = 0;
      return 0;
    }
  }
  if (e.mirror_seq != e.write_seq || e.mirror_table != table) {
    CsBatchBeginImpl();
    if (!g_cs_batch_open)
      return 0;
    RecordImageTiling(g_cs_list, *table, e.mirror, 0, tiled.size, e.buf,
                      e.res.size, false);
    e.mirror_seq = e.write_seq;
    e.mirror_table = table;
    g_guest_mirror_tiles++;
  }
  bytes = tiled.size;
  return e.mirror->address();
}

struct GuestWindowOwner {
  u64 range;
  bool mirror;
};

bool RefreshGuestWindow(u64 base, CsRange& e, u64 n) {
  e.wt_clean = GuestWriteTracker().Arm(base, n);
  CsBatchBeginImpl();
  if (!g_cs_batch_open || !RecordGuestRead(g_cs_list, base, n, e.buf))
    return false;
  g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW);
  MarkPending(e);
  e.last_validated_frame = g_frame.num;
  e.staged_stamp = ++g_cs_dirty_stamp;
  return true;
}

constexpr u64 kMissChunk = 64 * 1024;
void MaterializeGuestRange(u64 address) {
  const u64 base = address & ~(kMissChunk - 1);
  if (!gpu::IsReadableRange(base, kMissChunk))
    return;
  if (const auto blocks = g_cs_range_blocks.find(base >> kCsRangeBlockShift);
      blocks != g_cs_range_blocks.end())
    for (u64 other : blocks->second) {
      const auto found = g_cs_ranges.find(other);
      if (found != g_cs_ranges.end() && other <= address &&
          address < other + found->second.guest_bytes)
        return;
    }
  CsRange& e = g_cs_ranges[base];
  if (e.buf || !CsRangeEnsureBuffer(e, kMissChunk)) {
    if (!e.buf)
      g_cs_ranges.erase(base);
    return;
  }
  e.size = e.guest_bytes = kMissChunk;
  e.last_used_frame = g_frame.num;
  IndexCsRange(base, kMissChunk);
  if (!RefreshGuestWindow(base, e, kMissChunk)) {
    EraseCsRange(g_cs_ranges.find(base));
    return;
  }
  g_guest_misses_n++;
}

void MarkWrittenWindows(u64 cs_addr,
                        const base::Vector<GuestWindowOwner>& owners,
                        const base::Vector<u32>& written) {
  for (u32 w : written) {
    if (w >= owners.size())
      continue;
    const GuestWindowOwner& owner = owners[w];
    auto found = g_cs_ranges.find(owner.range);
    if (found == g_cs_ranges.end())
      continue;
    if (owner.mirror) {
      static int shown = 0;
      if (shown++ < 8)
        BASE_LOGW("csgpu",
                  "CS @{:#x} wrote into a mirrored image; the write is lost",
                  cs_addr);
      continue;
    }
    CsRange& e = found->second;
    IndexDirtyRange(owner.range, e.guest_bytes);
    if (!e.gpu_dirty)
      e.dirty_frame = g_frame.num;
    e.gpu_dirty = true;
    e.fill_kept.clear();
    e.write_seq++;
    e.dirty_stamp = ++g_cs_dirty_stamp;
    e.dirty_render_serial = g_render_serial;
    BumpTextureEpoch();
  }
}

constexpr u32 kQuietAfter = 16;
struct WriterHistory {
  u32 quiet = 0;
  bool writes = false;
};
base::HashMap<u64, WriterHistory> g_writer_history;
struct PendingTable {
  u32 set = 0;
  u64 batch = 0;
  u64 cs_addr = 0;
  bool check_writes = false;
  base::Vector<GuestWindowOwner> owners;
};
base::Vector<PendingTable> g_pending_tables;
constexpr size_t kMaxPendingTables = 64;

const guest::SessionReset g_writers_reset([] {
  guest::ResetResource(g_writer_history);
  guest::ResetResource(g_pending_tables);
});

struct TableSetGuard {
  u32 set = ~0u;
  ~TableSetGuard() {
    if (set != ~0u)
      ReleaseGuestMemory(set);
  }
};

void cs::HarvestGuestTables() {
  CsBatchReap();
  base::EraseIf(g_pending_tables, [](PendingTable& p) {
    if (p.batch > g_cs_batch_done)
      return false;
    for (u64 address : TakeGuestMisses(p.set))
      MaterializeGuestRange(address);
    if (p.check_writes) {
      const auto written = TakeWrittenWindows(p.set);
      if (!written.empty()) {
        BASE_LOGW("csgpu",
                  "CS @{:#x} stored through guest pointers after {} quiet "
                  "dispatches; waited for again",
                  p.cs_addr, kQuietAfter);
        g_writer_history[p.cs_addr].writes = true;
        MarkWrittenWindows(p.cs_addr, p.owners, written);
      }
    }
    ReleaseGuestMemory(p.set);
    return true;
  });
}

bool GuestWindows(const ComputeInfo& ci,
                  base::Vector<GuestMemoryWindow>& out,
                  base::Vector<GuestWindowOwner>& owners) {
  for (u64 address : ci.guest_memory_seeds)
    MaterializeGuestRange(address);
  struct Candidate {
    u64 base, skip, n;
    CsRange* e;
    bool writable;
  };
  base::Vector<Candidate> candidates;
  for (auto& [base, e] : g_cs_ranges) {
    const u64 n = e.guest_bytes ? base::Min(e.size, e.guest_bytes) : e.size;
    if (!n || !e.buf || e.imported)
      continue;
    for (const auto& r : ci.guest_memory) {
      const u64 lo = base::Max(base, r.base);
      const u64 hi = base::Min(base + n, r.base + r.size);
      if (lo < hi)
        candidates.push_back({base, lo - base, hi - lo, &e, r.writable});
    }
  }
  base::Sort(
      candidates.begin(), candidates.end(),
      [](const Candidate& a, const Candidate& b) {
        if (a.e->gpu_dirty != b.e->gpu_dirty)
          return a.e->gpu_dirty;
        const u64 ta = a.e->gpu_dirty ? a.e->dirty_stamp : a.e->staged_stamp;
        const u64 tb = b.e->gpu_dirty ? b.e->dirty_stamp : b.e->staged_stamp;
        return ta != tb ? ta > tb : a.n > b.n;
      });
  base::Vector<base::Pair<u64, u64>> taken;
  for (const Candidate& c : candidates) {
    CsRange& e = *c.e;
    u64 address = 0, n = c.skip + c.n;
    bool mirror = false;
    if (!e.image_staging || e.truth) {
      if (!e.gpu_dirty && !e.wt_clean &&
          e.last_validated_frame != g_frame.num &&
          !RefreshGuestWindow(
              c.base, e,
              e.guest_bytes ? base::Min(e.size, e.guest_bytes) : e.size))
        continue;
      address = e.buf->address();
    } else if (e.gpu_dirty) {
      address = GuestMirror(e, n);
      mirror = true;
    }
    if (!address || n <= c.skip)
      continue;
    MarkPending(e);
    u64 at = c.base + c.skip;
    const u64 end = base::Min(c.base + n, c.base + c.skip + c.n);
    size_t i = 0;
    while (i < taken.size() && taken[i].second <= at)
      i++;
    base::Vector<base::Pair<u64, u64>> pieces;
    for (; at < end; i++) {
      const u64 next = i < taken.size() ? base::Min(taken[i].first, end) : end;
      if (next > at)
        pieces.emplace_back(at, next);
      if (i >= taken.size() || taken[i].first >= end)
        break;
      at = taken[i].second;
    }
    for (const auto& [lo, hi] : pieces) {
      out.push_back({lo, hi - lo, address + (lo - c.base), c.writable, mirror});
      owners.push_back({c.base, mirror});
      auto pos = taken.begin();
      while (pos != taken.end() && pos->first < lo)
        ++pos;
      taken.insert(pos, base::Pair<u64, u64>{lo, hi});
    }
    g_guest_override_n += !pieces.empty();
  }
  base::Vector<u32> order(out.size());
  for (u32 k = 0; k < order.size(); k++)
    order[k] = k;
  base::Sort(order.begin(), order.end(),
             [&](u32 a, u32 b) { return out[a].base < out[b].base; });
  base::Vector<GuestMemoryWindow> sorted;
  base::Vector<GuestWindowOwner> sorted_owners;
  for (u32 k : order) {
    sorted.push_back(out[k]);
    sorted_owners.push_back(owners[k]);
  }
  out = base::move(sorted);
  owners = base::move(sorted_owners);
  return true;
}

struct IndirectArgs {
  rhi::Buffer* buffer = nullptr;
  u64 offset = 0;
};

bool FindIndirectArgs(u64 args, IndirectArgs& out) {
  if (!Device().caps().dispatch_indirect)
    return false;
  CsRange* found = nullptr;
  u64 range_base = 0;
  const bool several = AnyDirtyOverlapping(args, 12, [&](u64 other) {
    if (found)
      return true;
    found = &g_cs_ranges.find(other)->second;
    range_base = other;
    return false;
  });
  if (found || (kMirror && MirrorWritten(args, 12)))
    CollectGuestWrites();
  if (!several && !found && kMirror && MirrorHoldsNewest(args, 12)) {
    out = {g_mirror, MirrorOffset(args)};
    g_cs_indirect_n++;
    return true;
  }
  if (several || !found || !found->buf || found->imported ||
      !found->staged_clean || (found->image_staging && !found->truth) ||
      args < range_base ||
      args + 12 > range_base + base::Min(found->size, found->guest_bytes))
    return false;
  MarkPending(*found);
  out = {found->buf, args - range_base};
  g_cs_indirect_n++;
  return true;
}

bool CanDispatchIndirect(u64 args) {
  return Device().caps().dispatch_indirect && !(args & 3) &&
         gpu::IsReadableRangeCached(args, 12);
}

void ResolveComputeTexture(const DrawInfo::DrawTex& t,
                           rhi::TextureView** view,
                           rhi::TextureState* state,
                           rhi::Format* format) {
  if (t.null_descriptor || !t.base)
    return;
  const bool rt_eligible =
      !t.arrayed && !t.is_3d && !GuestFormatBlockCompressed(t.dfmt);
  u64 base = t.base;
  if (rt_eligible) {
    ActivateSampledRtVariant(base, t.w, t.h);
    ActivateSampledDepthVariant(base, t.w, t.h);
    if (!g_rts.count(base) && !g_depths.count(base)) {
      const bool depth_format = t.dfmt == 4 && t.nfmt == 7;
      u64 resolved = depth_format ? ResolveSampledDepth(base, t.w, t.h) : 0;
      if (!resolved)
        resolved = ResolveSampledRT(base, t.w, t.h);
      if (!resolved && !depth_format)
        resolved = ResolveSampledDepth(base, t.w, t.h);
      if (resolved) {
        base = resolved;
        ActivateSampledDepthVariant(base, t.w, t.h);
      }
    }
  }
  rhi::CommandList* const list = g_frame.list;
  if (rt_eligible && g_rts.count(base)) {
    if (!CsRefreshRtFromTruth(base))
      FlushCsWritesAt(DefaultRenderer(), base, "cs-tex");
    RTarget& rt = g_rts[base];
    if (rt.ever_rendered) {
      if (rt.layout != rhi::TextureState::kShaderRead || rt.dirty_for_read) {
        TransitionImage(list, rt.texture, rt.layout,
                        rhi::TextureState::kShaderRead);
        rt.dirty_for_read = false;
      }
      *view = SampledViewAs(rt, t.swizzle, GuestTextureFormat(t.dfmt, t.nfmt),
                            format);
      *state = rhi::TextureState::kShaderRead;
      return;
    }
  }
  if (rt_eligible && g_depths.count(base)) {
    DepthTarget& depth = g_depths[base];
    if (depth.layout != rhi::TextureState::kDepthRead)
      TransitionImage(list, depth.texture, depth.layout,
                      rhi::TextureState::kDepthRead, rhi::kAspectDepth,
                      depth.layers);
    *view = SampledView(depth, t.swizzle);
    *state = rhi::TextureState::kDepthRead;
    *format = rhi::Format::kUndefined;
    return;
  }
  if (rt_eligible) {
    if (const u64 stencil = ResolveSampledStencil(base)) {
      DepthTarget& depth = g_depths[stencil];
      if (depth.stencil_layout != rhi::TextureState::kDepthRead)
        TransitionImage(list, depth.texture, depth.stencil_layout,
                        rhi::TextureState::kDepthRead, rhi::kAspectStencil,
                        depth.layers);
      *view = StencilSampledView(depth);
      *state = rhi::TextureState::kDepthRead;
      *format = rhi::Format::kS8Uint;
      return;
    }
  }
  *view = TexViewFor(t);
  *state = rhi::TextureState::kShaderRead;
}

bool Dispatch(Renderer& renderer, const ComputeInfo& ci_in) {
  DELTA_ZONE("gpu.cs_dispatch");
  BumpTextureEpoch();
  ComputeInfo ci_merged;
  const ComputeInfo& ci = MergeSameBase(ci_in, ci_merged);
  if (g_cs_failed) {
    renderer.state = nullptr;
    return CsDeclined(ci, "1");
  }
  if (!renderer.available() || !ci.recomp || !ci.recomp->ok || !ci.num_res)
    return CsDeclined(ci, "2");
  const bool capture_dispatch = trace::CaptureDispatchInputs(renderer, ci);
  IndirectArgs indirect;
  u32 groups[3] = {ci.groups[0], ci.groups[1], ci.groups[2]};
  if (ci.indirect_args && !FindIndirectArgs(ci.indirect_args, indirect)) {
    if (!gpu::IsReadableRange(ci.indirect_args, 12))
      return CsDeclined(ci, "indirect");
    if (Device().caps().dispatch_indirect &&
        FlushCsWritesForGpu(renderer, ci.indirect_args, 12, "indirect-cs")) {
      CsBatchBeginImpl();
      if (g_cs_batch_open)
        indirect.buffer = StageIndirectArgs(g_cs_list, ci.indirect_args, 12,
                                            &indirect.offset);
    }
    if (!indirect.buffer) {
      if (!FlushCsWritesRange(renderer, ci.indirect_args, 12, "indirect-cs"))
        return CsDeclined(ci, "indirect");
      std::memcpy(groups, reinterpret_cast<const void*>(ci.indirect_args), 12);
      if (!groups[0] || !groups[1] || !groups[2])
        return true;
    }
  }
  if (CsTracksGuestWrites()) {
    for (u32 i = 0; i < ci.num_res; i++) {
      const auto found = g_cs_ranges.find(ci.res[i].base);
      if (found != g_cs_ranges.end() && found->second.gpu_dirty) {
        CollectGuestWritesForSubmission();
        break;
      }
    }
  } else if (GuestWriteTracker().enabled()) {
    CollectGuestWritesForSubmission();
  }
  const bool heavy = g_cs_heavy.contains(ci.cs_addr);
  if (heavy && g_cs_batch_count && (++g_submit_why[3], !CsBatchSubmit())) {
    renderer.state = nullptr;
    return CsDeclined(ci, "heavy");
  }
  const rhi::Caps& caps = Device().caps();
  const u32 max_resources =
      base::Min(gcn::kMaxCsResources, caps.max_compute_resources);
  if (ci.num_res + (ci.recomp->guest_memory_binding >= 0) +
          (ci.gds_binding >= 0) >
      max_resources)
    return CsDeclined(ci, "2");
  rhi::Buffer* guest_table = nullptr;
  u64 guest_table_bytes = 0;
  base::Vector<GuestMemoryWindow> windows;
  base::Vector<GuestWindowOwner> window_owners;
  TableSetGuard guest_set;
  ScopeCs cs;
  for (u32 i = 0; i < ci.num_res; i++)
    g_cs_bytes += ci.res[i].size;
  for (u32 i = 0; i < ci.num_res; i++)
    if (ci.res[i].size > caps.max_storage_buffer_range)
      return CsDeclined(ci, "3");
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    const u64 guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const u64 size = ci.res[i].size ? ((ci.res[i].size + 3) & ~u64(3)) : 4;
    for (u32 j = 0; j < i; j++) {
      if (ci.res[j].zero_fill || ci.res[j].base != ci.res[i].base)
        continue;
      const u64 other_guest_bytes =
          ci.res[j].guest_size ? ci.res[j].guest_size : ci.res[j].size;
      const u64 other_size =
          ci.res[j].size ? ((ci.res[j].size + 3) & ~u64(3)) : 4;
      if (size != other_size || guest_bytes != other_guest_bytes ||
          !SameCsResourceShape(ci.res[i], ci.res[j]))
        return CsDeclined(ci, "4");
    }
  }
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill)
      continue;
    const u64 bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    NoteDccWrite(ci.res[i].base, bytes, nullptr);
    NoteSurfaceWrite(ci.res[i].base, bytes);
    if (!ci.res[i].image_staging)
      NoteRawWrite(ci.res[i].base, bytes);
  }
  CsPipe* cp = GetCsPipe(ci);
  if (!cp)
    return CsDeclined(ci, "5");

  if (!g_cs_batches[0].list) {
    if (!CsBatchInit())
      return CsDeclined(ci, "8");
    if (CsSyncReporting()) {
      double total = 0;
      rhi::CommandList* list = g_cs_batches[0].list;
      for (int i = 0; i < 100; i++) {
        list->Begin();
        list->End();
        const u64 t0 = NowNs();
        Device().Wait(Device().Submit(list));
        total += (NowNs() - t0) / 1e6;
      }
      BASE_LOGI("csbench", "empty submit+wait: {:.3f} ms", total / 100.0);
    }
  }

  if (kCsHist) {
    struct Cell {
      u64 size, n;
    };
    static base::Mutex m;
    static base::Map<u64, Cell> tbl;
    static const guest::SessionReset reset_tbl(
        [] { guest::ResetResource(tbl); });
    static auto last = base::TimeTicks::Now();
    base::LockGuard<base::Mutex> lk(m);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].written && tbl.size() < 4096) {
        Cell& c = tbl[ci.res[i].base];
        c = {ci.res[i].size, c.n + 1};
      }
    const auto now = base::TimeTicks::Now();
    if ((now - last).InSeconds() >= kCsHist) {
      last = now;
      BASE_LOGI("cshist", "{} written ranges", tbl.size());
      for (const auto& kv : tbl)
        BASE_LOGI("cshist", "{:#x} size={:#x} x{}", (unsigned long)kv.first,
                  (unsigned long)kv.second.size, (unsigned long)kv.second.n);
      std::fflush(stderr);
    }
  }
  static u32 cs_listed = 0;
  if (kCsList && g_frame.num > 25 && cs_listed < 200) {
    cs_listed++;
    for (u32 i = 0; i < ci.num_res; i++)
      BASE_LOGI("cslist", "cs={:#x} bind={} base={:#x} size={:#x} {}{}",
                (unsigned long long)ci.cs_addr, ci.res[i].binding,
                (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                ci.res[i].image_staging ? "img"
                : ci.res[i].zero_fill   ? "zero"
                                        : "buf",
                ci.res[i].written ? " written" : "");
  }

  const u64 t_in0 = NowNs();
  rhi::Buffer* bind_buf[ComputeInfo::kMaxResources];
  u64 sz[ComputeInfo::kMaxResources];
  rhi::Buffer* truth_view[ComputeInfo::kMaxResources] = {};
  const TileTable* truth_table[ComputeInfo::kMaxResources] = {};
  u64 parent_base[ComputeInfo::kMaxResources] = {};
  u64 parent_off[ComputeInfo::kMaxResources] = {};
  g_cs_scratch_owner++;
  auto bind_truth_view = [&](u32 i, rhi::Buffer* tiled_buf, u64 off, u64 span,
                             u64 pbase) -> bool {
    gcn::TextureLayout32 tiled, linear;
    const TileTable* table = BuildCsImageLayouts(ci.res[i], tiled, linear)
                                 ? GetTileTable(tiled, linear)
                                 : nullptr;
    if (!table)
      return false;
    for (u32 j = 0; j < i; j++) {
      if (truth_view[j] && truth_table[j] == table &&
          ci.res[j].base == ci.res[i].base && parent_base[j] == pbase &&
          parent_off[j] == off && sz[j] == sz[i]) {
        truth_view[i] = truth_view[j];
        truth_table[i] = table;
        parent_base[i] = pbase;
        parent_off[i] = off;
        bind_buf[i] = truth_view[j];
        return true;
      }
    }
    auto owner_it = g_cs_ranges.find(pbase ? pbase : ci.res[i].base);
    const u64 seq =
        owner_it != g_cs_ranges.end() ? owner_it->second.write_seq : 0;
    bool fresh = true;
    rhi::Buffer* const view = AcquireView(sz[i], pbase ? pbase : ci.res[i].base,
                                          off, table, seq, &fresh);
    if (!view)
      return false;
    if (fresh) {
      CsBatchBeginImpl();
      RecordImageTiling(g_cs_list, *table, tiled_buf, off, span, view, sz[i],
                        true);
      g_cs_batch_access.erase(view);
      g_truth_detile_n++;
      StampView(view, pbase ? pbase : ci.res[i].base, off, table, seq);
    }
    truth_view[i] = view;
    truth_table[i] = table;
    parent_base[i] = pbase;
    parent_off[i] = off;
    bind_buf[i] = view;
    return true;
  };
  bool mirror_res[ComputeInfo::kMaxResources] = {};
  for (u32 i = 0; i < ci.num_res; i++) {
    const u64 bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const u64 span = base::Max<u64>(bytes, (ci.res[i].size + 3) & ~u64(3));
    mirror_res[i] = MirrorEligible(ci.res[i], span) &&
                    MirrorCopyIn(renderer, ci.res[i].base, bytes);
  }
  for (u32 i = 0; i < ci.num_res; i++) {
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~u64(3)) : 4;
    if (mirror_res[i]) {
      bind_buf[i] = g_mirror;
      continue;
    }
    if (ci.res[i].zero_fill) {
      if (!CsEnsureStage(i, sz[i]))
        return CsDeclined(ci, "10");
      bind_buf[i] = g_cs_stage[i].buf;
      continue;
    }
    const u64 base = ci.res[i].base;
    const u64 guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const bool truth_i = TruthEligible(ci.res[i]);
    u64 folds[ComputeInfo::kMaxResources];
    u32 fold_count = 0;
    if (kCsTruth && (truth_i || !ci.res[i].image_staging)) {
      CsRange* parent = nullptr;
      u64 pbase = 0;
      for (u64 cand : DirtyRangesOverlapping(base, guest_bytes, base)) {
        auto f = g_cs_ranges.find(cand);
        if (f == g_cs_ranges.end())
          continue;
        CsRange& r = f->second;
        const bool offset_ok =
            (base - cand) % caps.storage_offset_alignment == 0 ||
            ((base - cand) % 4 == 0 && ci.res[i].binding < 48 && !truth_i);
        if ((!r.truth && r.image_staging) || !r.gpu_dirty || !r.buf ||
            cand > base || base + guest_bytes > cand + r.guest_bytes ||
            !offset_ok)
          continue;
        parent = &r;
        pbase = cand;
        break;
      }
      auto own = g_cs_ranges.find(base);
      if (parent && own != g_cs_ranges.end() && own->second.gpu_dirty) {
        CsRange& s = own->second;
        if (s.image_staging && !s.truth) {
          parent = nullptr;
        } else if (s.buf) {
          const u64 n = base::Min<u64>(s.size ? s.size : s.cap, guest_bytes);
          CsBatchBeginImpl();
          g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
          g_cs_list->CopyBuffer(parent->buf, base - pbase, s.buf, 0, n);
          g_cs_list->Barrier(rhi::kAccessCopyWrite,
                             kAccessComputeRW | kAccessCopyRW);
          g_cs_batch_access.erase(parent->buf);
          UnindexDirtyRange(base, s.guest_bytes);
          s.gpu_dirty = false;
          s.last_validated_frame = -1;
          s.hash = 0;
          MarkPending(s);
          IndexDirtySpan(pbase, base, RangeEnd(base, n));
          parent->write_seq++;
          parent->dirty_stamp = ++g_cs_dirty_stamp;
          parent->dirty_render_serial = g_render_serial;
          BumpTextureEpoch();
        }
      }
      if (parent) {
        const u64 off = base - pbase;
        if (parent->chunk_ref == g_frame.chunk_seq &&
            (ci.res[i].written || ci.res[i].shader_writes) &&
            !CsSplitFrameChunk()) {
          renderer.state = nullptr;
          return CsDeclined(ci, "22");
        }
        parent->last_used_frame = g_frame.num;
        MarkPending(*parent);
        parent_base[i] = pbase;
        if (truth_i) {
          if (!bind_truth_view(i, parent->buf, off, guest_bytes, pbase))
            return CsDeclined(ci, "23");
        } else {
          parent_off[i] = off;
          bind_buf[i] = parent->buf;
        }
        g_truth_parent_n++;
        continue;
      }
    }
    {
      const u64 to = NowNs();
      const auto own = g_cs_ranges.find(base);
      const bool own_dirty = own != g_cs_ranges.end() &&
                             own->second.gpu_dirty && own->second.buf &&
                             own->second.cpu_writes.empty() &&
                             !own->second.imported;
      const bool reads_own_buffer =
          own_dirty &&
          (truth_i ? own->second.truth && own->second.guest_bytes >= guest_bytes
                   : !own->second.truth && !own->second.image_staging &&
                         !ci.res[i].image_staging && own->second.cap >= sz[i] &&
                         ((own->second.size == sz[i] &&
                           own->second.guest_bytes == guest_bytes &&
                           SameCsResourceShape(own->second.res, ci.res[i])) ||
                          (kCsSubset && own->second.size >= sz[i] &&
                           own->second.guest_bytes >= guest_bytes)));
      auto siblings = DirtyRangesOverlapping(base, guest_bytes, base);
      SortByDirtyOrder(siblings);
      for (u64 dirty : siblings) {
        auto found = g_cs_ranges.find(dirty);
        if (found == g_cs_ranges.end())
          continue;
        CsRange& s = found->second;
        if (kCsTruth && (truth_i || !ci.res[i].image_staging) && s.gpu_dirty &&
            s.buf && (s.truth || !s.image_staging) && dirty >= base &&
            dirty + s.guest_bytes <= base + guest_bytes &&
            fold_count < ComputeInfo::kMaxResources) {
          folds[fold_count++] = dirty;
          continue;
        }
        if (kCsKeepOverlap && s.gpu_dirty && !ci.res[i].read &&
            !ci.res[i].image_staging && !truth_i && !s.image_staging &&
            !s.truth) {
          g_wb_why["overlap-kept"]++;
          continue;
        }
        if (reads_own_buffer && s.gpu_dirty) {
          g_wb_why["overlap-own"]++;
          continue;
        }
        if ((truth_i || ci.res[i].image_staging) &&
            (s.truth || s.image_staging) && s.gpu_dirty &&
            !(dirty >= base && dirty + s.guest_bytes <= base + guest_bytes) &&
            !(base >= dirty && base + guest_bytes <= dirty + s.guest_bytes)) {
          g_wb_why["overlap-tiled"]++;
          continue;
        }
        if (s.gpu_dirty && !ci.res[i].read && ci.res[i].written) {
          g_wb_why["overlap-write-only"]++;
          continue;
        }
        if (s.gpu_dirty) {
          g_wb_why["overlap"]++;
          g_wb_why_bytes += s.size;
          if (kCsOverlapTrace && g_frame.num >= kCsOverlapTrace) {
            static int shown = 0;
            if (shown++ < 1500)
              BASE_LOGI(
                  "csoverlap",
                  "cs={:#x} res{} {:#x}+{:#x} {} w={} {}x{}x{} mips={} "
                  "tile={} dfmt={} | dirty {:#x}+{:#x} truth={} img={} "
                  "{}x{}x{} mips={} tile={} dfmt={}",
                  ci.cs_addr, i, base, guest_bytes,
                  ci.res[i].image_staging ? "img" : "buf",
                  (int)ci.res[i].shader_writes, ci.res[i].width,
                  ci.res[i].height, ci.res[i].layers, ci.res[i].mip_levels,
                  ci.res[i].tiling_idx, ci.res[i].dfmt, dirty, s.guest_bytes,
                  (int)s.truth, (int)s.image_staging, s.res.width, s.res.height,
                  s.res.layers, s.res.mip_levels, s.res.tiling_idx, s.res.dfmt);
          }
        }
        const WaitReaderScope reader("cs-sibling");
        if (!CsRangeFlushOne(dirty, s) && g_cs_failed)
          return CsDeclined(ci, "11");
      }
      g_in_overlap_n += !siblings.empty();
      g_in_overlap_ns += NowNs() - to;
    }
    CsRange& e = g_cs_ranges[base];
    if (e.gpu_dirty && !e.cpu_writes.empty()) {
      g_wb_why["cpu-write"]++;
      g_wb_why_bytes += e.size;
      const WaitReaderScope reader("cpu-write");
      if (!CsRangeFlushOne(base, e) && g_cs_failed)
        return CsDeclined(ci, "24");
    }
    const bool truth_same =
        truth_i && e.buf && e.truth && e.guest_bytes == guest_bytes;
    const bool truth_sub =
        truth_i && e.buf && e.truth && guest_bytes < e.guest_bytes;
    const bool raw_view = !truth_i && !ci.res[i].image_staging && e.buf &&
                          e.truth && static_cast<u64>(sz[i]) <= e.cap &&
                          guest_bytes <= e.guest_bytes;
    const bool exact_shape =
        truth_same ||
        (!truth_i && !e.truth && e.buf && e.size == static_cast<u64>(sz[i]) &&
         e.guest_bytes == guest_bytes && SameCsResourceShape(e.res, ci.res[i]));
    const bool layer_prefix = [&] {
      if (exact_shape || truth_i || e.truth || !e.buf || e.imported ||
          !e.image_staging || !ci.res[i].image_staging ||
          ci.res[i].layers >= e.res.layers || ci.res[i].mip_levels != 1 ||
          e.size < static_cast<u64>(sz[i]) || e.guest_bytes < guest_bytes)
        return false;
      ComputeInfo::Res as_staged = ci.res[i];
      as_staged.layers = e.res.layers;
      return SameCsResourceShape(e.res, as_staged);
    }();
    const bool subset =
        raw_view || truth_sub || layer_prefix ||
        (kCsSubset && !exact_shape && e.buf && !e.imported && !e.truth &&
         !ci.res[i].image_staging && !e.image_staging &&
         e.size >= static_cast<u64>(sz[i]) && e.guest_bytes >= guest_bytes);
    const bool same_shape = exact_shape || subset;
    const u64 foot = (raw_view || truth_sub) ? e.cap
                     : truth_i               ? guest_bytes
                                             : sz[i];
    const u64 stage_bytes = (raw_view || truth_sub) ? e.guest_bytes
                            : truth_i               ? guest_bytes
                                                    : 0;
    const u64 hash_bytes = stage_bytes ? stage_bytes : guest_bytes;
    if (!same_shape && e.gpu_dirty && ci.res[i].image_staging &&
        guest_bytes >= e.guest_bytes && CsAliasedBase(base) &&
        AliasedImageRenderSerial(base) != UINT64_MAX &&
        AliasedImageRenderSerial(base) > e.dirty_render_serial) {
      UnindexDirtyRange(base, e.guest_bytes);
      e.gpu_dirty = false;
      e.cpu_writes.clear();
      g_wb_why["reshape-superseded"]++;
    }
    if (!same_shape && e.gpu_dirty) {
      const u64 trs = NowNs();
      g_wb_why["reshape"]++;
      g_wb_why_bytes += e.size;
      const WaitReaderScope reader("reshape");
      const bool ok = CsRangeFlushOne(base, e);
      g_in_reshape_ns += NowNs() - trs;
      g_in_reshape_n++;
      if (!ok)
        return CsDeclined(ci, "12");
    }

    const bool buffer_reused = e.buf && e.cap >= foot;
    const u64 ta = NowNs();
    const bool ensured = CsRangeEnsureBuffer(e, foot);
    g_in_alloc_ns += NowNs() - ta;
    g_in_alloc_n += !buffer_reused;
    if (!ensured)
      return CsDeclined(ci, "14");
    if (!buffer_reused && caps.debug_labels) {
      char name[32];
      std::snprintf(name, sizeof(name), "csbuf %#llx",
                    (unsigned long long)base);
      Device().SetName(e.buf, name);
    }
    if (!buffer_reused || !same_shape)
      e.wt_clean = false;
    bool valid =
        buffer_reused && same_shape &&
        (e.gpu_dirty || e.last_validated_frame == g_frame.num || e.wt_clean);
    if (valid && !e.gpu_dirty && e.last_validated_frame != g_frame.num)
      g_in_wt_skips++;
    if (e.imported && same_shape)
      valid = true;
    const bool rt_backed = ci.res[i].image_staging && CsAliasedBase(base);
    if (rt_backed && e.gpu_dirty && e.rt_seq == e.write_seq &&
        AliasedImageRenderSerial(base) > e.refreshed_serial) {
      UnindexDirtyRange(base, e.guest_bytes);
      e.gpu_dirty = false;
      e.cpu_writes.clear();
      e.hash = 0;
      e.last_validated_frame = -1;
      g_wb_why["rt-superseded"]++;
    }
    const bool rt_stale = kCsRtCache && e.rt_sourced && e.last_rt_frame >= 0 &&
                          AliasedImageRenderSerial(base) <= e.rt_serial;
    const bool rt_attempt = rt_backed && !e.gpu_dirty && !rt_stale &&
                            !g_rt_unbridgeable.count(RtShapeKey(ci.res[i]));
    if (rt_attempt)
      valid = false;
    else if (rt_backed && !e.gpu_dirty && e.rt_sourced)
      valid = same_shape;
    if (!valid && same_shape && !rt_attempt) {
      e.wt_clean = !e.imported && GuestWriteTracker().Arm(base, hash_bytes);
      e.last_validated_frame = g_frame.num;
    }
    const bool dmem_arena = base >= 0x8000000000ull && base < 0x9000000000ull;
    if (kCsSkipUpload && !valid && dmem_arena && !ci.res[i].read &&
        ci.res[i].written && !ci.res[i].image_staging && !ci.res[i].zero_fill &&
        same_shape && !stage_bytes) {
      valid = true;
      g_cs_skip_n++;
    }
    if (e.buf && e.chunk_ref == g_frame.chunk_seq &&
        (!valid || fold_count || ci.res[i].written ||
         ci.res[i].shader_writes) &&
        !CsSplitFrameChunk()) {
      renderer.state = nullptr;
      return CsDeclined(ci, "22");
    }
    if (!valid) {
      if (kCsRename && e.pending_batch && !e.gpu_dirty && !e.res.written) {
        if (!CsRangeRename(e, sz[i])) {
          renderer.state = nullptr;
          return CsDeclined(ci, "15");
        }
        e.pending_batch = false;
      } else if (rt_attempt && e.pending_batch && g_cs_batch_open &&
                 e.batch_id >= g_cs_batch_id) {
        g_cs_chunk_needs_batch = true;
      }
      bool staged_in_frame = false;
      if (rt_attempt) {
        e.last_rt_frame = static_cast<int>(g_frame.num);
        e.rt_serial = AliasedImageRenderSerial(base);
        staged_in_frame = StageCsRangeFromRtInFrame(ci.res[i], e);
      }
      if (rt_attempt) {
        const u64 tr = NowNs();
        e.rt_sourced = staged_in_frame;
        if (e.rt_sourced)
          e.staged_clean =
              !e.imported && GuestWriteTracker().Arm(base, hash_bytes);
        g_in_rt_ns += NowNs() - tr;
        g_in_rt_n++;
        static int rt_trace_logged = 0;
        if (kCsRtTrace && rt_trace_logged < 200) {
          rt_trace_logged++;
          BASE_LOGI("csrt", "f{} base={:#x} {}x{} -> {}", (int)g_frame.num,
                    (unsigned long)base, ci.res[i].width, ci.res[i].height,
                    e.rt_sourced ? "staged-from-RT" : "guest-fallback");
        }
        if (g_cs_failed) {
          renderer.state = nullptr;
          return CsDeclined(ci, "17");
        }
      }
      if (!rt_attempt || !e.rt_sourced) {
        if (!same_shape || rt_attempt) {
          e.wt_clean = !e.imported && GuestWriteTracker().Arm(base, hash_bytes);
          e.last_validated_frame = g_frame.num;
        } else if (ci.res[i].written && CsTracksGuestWrites()) {
          GuestWriteTracker().Arm(base, hash_bytes);
        }
        e.cpu_writes.clear();
        const u64 copy_bytes = stage_bytes ? stage_bytes : ci.res[i].size;
        CsBatchBeginImpl();
        bool staged = false;
        if (g_cs_batch_open && e.buf) {
          if (stage_bytes || !ci.res[i].image_staging) {
            const u64 tail = copy_bytes & ~u64(3);
            if (!stage_bytes && sz[i] > tail)
              g_cs_list->FillBuffer(e.buf, tail, ((sz[i] + 3) & ~u64(3)) - tail,
                                    0);
            staged = RecordGuestRead(g_cs_list, base, copy_bytes, e.buf);
          } else {
            g_stage_guest_detile_bytes += sz[i];
            g_stage_guest_detile_n++;
            staged = StageImageOnGpu(ci.res[i], e);
          }
        }
        if (!staged)
          return CsDeclined(ci, "stage");
        CollectGuestWrites();
        e.staged_clean =
            !e.imported && GuestWriteTracker().Arm(base, hash_bytes);
        g_cs_list->Barrier(rhi::kAccessCopyWrite | kAccessComputeRW,
                           kAccessComputeRW);
        g_cs_batch_access.erase(e.buf);
        MarkPending(e);
        e.rt_sourced = false;
        e.staged_stamp = ++g_cs_dirty_stamp;
      }
      e.gpu_dirty = false;
      e.write_seq++;
      BumpTextureEpoch();
      g_cs_stage_n++;
      g_cs_stage_bytes += sz[i];
      if (CsSyncReporting()) {
        StageStat& st = g_stage_stats[base];
        st.bytes += sz[i];
        st.n++;
        if (!buffer_reused)
          st.first++;
        else if (!same_shape) {
          st.shape++;
          static base::HashSet<u64> shape_logged, big_logged;
          auto& logged = sz[i] >= (1u << 20) ? big_logged : shape_logged;
          if (logged.size() < 64 && logged.insert(base).second)
            BASE_LOGI(
                "csshape",
                "{:#x} was {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}"
                " now {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}",
                (unsigned long)base, e.res.width, e.res.height, e.res.pitch,
                e.res.layers, e.res.mip_levels, e.res.tiling_idx,
                e.res.elem_bytes, e.res.stage_elem_bytes, e.res.dfmt,
                (unsigned long)e.size, ci.res[i].width, ci.res[i].height,
                ci.res[i].pitch, ci.res[i].layers, ci.res[i].mip_levels,
                ci.res[i].tiling_idx, ci.res[i].elem_bytes,
                ci.res[i].stage_elem_bytes, ci.res[i].dfmt,
                (unsigned long)sz[i]);
        } else if (rt_attempt)
          st.rt++;
        else
          st.hash++;
      }
      if (ci.res[i].image_staging)
        g_stage_img_bytes += sz[i];
      else if (ci.res[i].shader_writes || ci.res[i].written)
        g_stage_rw_bytes += sz[i];
      else
        g_stage_ro_bytes += sz[i];
    }
    if (!subset) {
      e.size = truth_i ? guest_bytes : sz[i];
      if (e.guest_bytes != guest_bytes) {
        UnindexCsRange(base, e.guest_bytes);
        IndexCsRange(base, guest_bytes);
        if (e.gpu_dirty)
          IndexDirtySpan(base, DirtyLo(base), RangeEnd(base, guest_bytes));
      }
      e.guest_bytes = guest_bytes;
      e.image_staging = ci.res[i].image_staging;
      e.truth = truth_i;
      e.res = ci.res[i];
    }
    e.last_used_frame = g_frame.num;
    for (u32 k = 0; k < fold_count && e.buf; k++) {
      auto f = g_cs_ranges.find(folds[k]);
      if (f == g_cs_ranges.end())
        continue;
      CsRange& s = f->second;
      if (!s.gpu_dirty || !s.buf)
        continue;
      const u64 n = base::Min<u64>(s.size ? s.size : s.cap, s.guest_bytes);
      CsBatchBeginImpl();
      g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
      g_cs_list->CopyBuffer(e.buf, folds[k] - base, s.buf, 0, n);
      g_cs_list->Barrier(rhi::kAccessCopyWrite,
                         kAccessComputeRW | kAccessCopyRW);
      g_cs_batch_access.erase(e.buf);
      UnindexDirtyRange(folds[k], s.guest_bytes);
      s.gpu_dirty = false;
      s.last_validated_frame = -1;
      s.hash = 0;
      MarkPending(s);
      IndexDirtySpan(base, folds[k], RangeEnd(folds[k], n));
      if (!e.gpu_dirty) {
        e.gpu_dirty = true;
        e.dirty_frame = g_frame.num;
      }
      e.fill_kept.clear();
      e.write_seq++;
      e.dirty_stamp = ++g_cs_dirty_stamp;
      e.dirty_render_serial = g_render_serial;
      BumpTextureEpoch();
      MarkPending(e);
      g_truth_fold_n++;
    }
    bind_buf[i] = e.buf;
    if (truth_i) {
      if (!bind_truth_view(i, e.buf, 0, e.guest_bytes, 0))
        return CsDeclined(ci, "23");
      MarkPending(e);
    }
  }
  for (u32 i = 0; i < ci.num_res; i++)
    if (!ci.res[i].zero_fill && !truth_view[i] && !mirror_res[i])
      bind_buf[i] =
          g_cs_ranges[parent_base[i] ? parent_base[i] : ci.res[i].base].buf;
  u64 bind_off[ComputeInfo::kMaxResources] = {};

  g_ns_cs_in += NowNs() - t_in0;

  rhi::BindingWrite
      wr[ComputeInfo::kMaxResources + 2 + ComputeInfo::kMaxTextures];
  for (u32 i = 0; i < ci.num_res; i++)
    bind_off[i] = ci.res[i].zero_fill ? 0
                  : mirror_res[i]     ? MirrorOffset(ci.res[i].base)
                  : truth_view[i]     ? 0
                  : parent_base[i]
                      ? parent_off[i]
                      : g_cs_ranges[ci.res[i].base].imported_offset;
  u32 bias_dwords[ComputeInfo::kMaxResources] = {};
  const u64 align = base::Max<u64>(caps.storage_offset_alignment, 4);
  for (u32 i = 0; i < ci.num_res; i++) {
    const u64 view = ci.res[i].zero_fill ? 0 : ci.res[i].view_offset;
    const u64 off = bind_off[i] + view;
    const u64 skew = off % align;
    if (skew && (off % 4 || ci.res[i].binding >= 48 ||
                 (sz[i] - view) / 4 >= (1u << 26)))
      return CsDeclined(ci, "unaligned-view");
    bias_dwords[i] = static_cast<u32>(skew / 4);
    wr[i].binding = ci.res[i].binding;
    wr[i].buffer = bind_buf[i];
    wr[i].offset = off - skew;
    wr[i].range = sz[i] - view + skew;
  }
  u32 nwrite = ci.num_res;
  if (ci.gds_binding >= 0 && EnsureGdsBuffer()) {
    wr[nwrite].binding = static_cast<u32>(ci.gds_binding);
    wr[nwrite].buffer = g_gds.buf;
    wr[nwrite].range = GdsBuffer::kBytes;
    nwrite++;
  }
  if (ci.recomp->guest_memory_binding >= 0) {
    if (g_pending_tables.size() >= kMaxPendingTables &&
        !CsBatchWaitId(g_pending_tables.front().batch, kSyncBatchCap))
      return CsDeclined(ci, "guest-address-map");
    HarvestGuestTables();
    if (!GuestWindows(ci, windows, window_owners))
      return CsDeclined(ci, "guest-windows");
    if (!PrepareGuestMemory(windows, guest_table, guest_table_bytes,
                            ci.recomp->guest_memory_written, guest_set.set))
      return CsDeclined(ci, "guest-address-map");
    wr[nwrite].binding = ci.recomp->guest_memory_binding;
    wr[nwrite].buffer = guest_table;
    wr[nwrite].range = guest_table_bytes;
    nwrite++;
  }

  if (ci.num_tex && g_cs_batch_open && !g_cs_batches[g_cs_batch_cur].in_frame &&
      !CsBatchSubmit())
    return CsDeclined(ci, "tex-batch");
  CsBatchBeginImpl();
  if (ci.num_tex && !g_cs_batches[g_cs_batch_cur].in_frame)
    return CsDeclined(ci, "tex-out-of-frame");
  for (u32 i = 0; i < ci.num_tex; i++) {
    const DrawInfo::DrawTex& t = ci.tex[i].tex;
    rhi::TextureView* view = nullptr;
    rhi::TextureState state = rhi::TextureState::kShaderRead;
    rhi::Format format = GuestTextureFormat(t.dfmt, t.nfmt);
    ResolveComputeTexture(t, &view, &state, &format);
    CsBatchBeginImpl();
    if (!view) {
      view = DefaultTexView(t, t.nfmt == 4 || t.nfmt == 5);
      state = rhi::TextureState::kShaderRead;
    }
    rhi::BindingWrite& w = wr[nwrite++];
    w = {};
    w.binding = ci.tex[i].binding;
    w.view = view;
    w.view_state = state;
    w.sampler = SamplerForTex(t, format);
  }
  u32 zero_count = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].zero_fill)
      continue;
    zero_count++;
    g_cs_batch_access.erase(bind_buf[i]);
  }
  if (zero_count) {
    g_cs_list->Barrier(kAccessComputeRW, rhi::kAccessCopyWrite);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].zero_fill)
        g_cs_list->FillBuffer(bind_buf[i], 0, sz[i], 0);
    g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW);
  }
  if (ci.gds_binding >= 0)
    g_cs_list->Barrier(rhi::kAccessComputeWrite, kAccessComputeRW);
  g_cs_list->SetPipeline(cp->pipe);
  g_cs_list->PushBindGroup(0, cp->set_layout, wr, nwrite);
  constexpr u32 kCsPushDwords = 64;
  static_assert(16 + 48 == kCsPushDwords);
  u32 pc[kCsPushDwords];
  std::memcpy(pc, ci.user_data, sizeof(pc[0]) * 16);
  for (u32 i = 16; i < kCsPushDwords; i++)
    pc[i] = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!bind_buf[i] || ci.res[i].binding >= kCsPushDwords - 16)
      continue;
    const u64 dwords = (sz[i] - ci.res[i].view_offset) / 4;
    pc[16 + ci.res[i].binding] =
        (dwords < (1u << 26) ? static_cast<u32>(dwords) : 0) |
        (bias_dwords[i] << 26);
  }
  g_cs_list->SetPushConstants(0, sizeof(pc), pc);
  u32 hazard_src = 0, hazard_dst = 0;
  rhi::Buffer* unique_buffers[ComputeInfo::kMaxResources];
  bool unique_writes[ComputeInfo::kMaxResources] = {};
  u32 unique_count = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    u32 j = 0;
    while (j < unique_count && unique_buffers[j] != bind_buf[i])
      j++;
    if (j == unique_count) {
      unique_buffers[unique_count] = bind_buf[i];
      unique_writes[unique_count] = ci.res[i].shader_writes;
      unique_count++;
    } else {
      unique_writes[j] |= ci.res[i].shader_writes;
    }
  }
  for (u32 i = 0; i < unique_count; i++) {
    ComputeBufferAccess& prior = g_cs_batch_access[unique_buffers[i]];
    const ComputeBufferAccess current{true, unique_writes[i]};
    const bool hazard = NeedsComputeBarrier(prior, current);
    if (hazard) {
      hazard_src |= (prior.read ? rhi::kAccessComputeRead : 0) |
                    (prior.write ? rhi::kAccessComputeWrite : 0);
      hazard_dst |= rhi::kAccessComputeRead |
                    (unique_writes[i] ? rhi::kAccessComputeWrite : 0);
      prior = {};
    }
    prior.read = true;
    prior.write |= unique_writes[i];
  }
  if (hazard_src)
    g_cs_list->Barrier(hazard_src, hazard_dst);
  base::Vector<rhi::Buffer*> capture_buffers;
  if (capture_dispatch) {
    capture_buffers.resize(ci.num_res, nullptr);
    g_cs_list->Barrier(rhi::kAccessAllWrite, rhi::kAccessCopyRead);
    for (u32 i = 0; i < ci.num_res; i++) {
      rhi::Buffer* snapshot = CreateCsBuffer(wr[i].range, rhi::kBufferCopyDst,
                                             rhi::MemoryKind::kReadback);
      if (!snapshot || !snapshot->mapped()) {
        if (snapshot)
          Device().Destroy(snapshot);
        continue;
      }
      capture_buffers[i] = snapshot;
      g_cs_frame_retired.push_back(snapshot);
      g_cs_list->CopyBuffer(snapshot, 0, wr[i].buffer, wr[i].offset,
                            wr[i].range);
    }
    g_cs_list->Barrier(rhi::kAccessCopyRead | rhi::kAccessCopyWrite,
                       kAccessComputeRW | rhi::kAccessHostRead);
  }
  if (caps.debug_labels) {
    char label[96];
    std::snprintf(label, sizeof(label), "dispatch cs=%#llx %ux%ux%u res=%u",
                  (unsigned long long)ci.cs_addr, ci.groups[0], ci.groups[1],
                  ci.groups[2], ci.num_res);
    g_cs_list->InsertLabel(label);
  }
  if (ci.recomp->guest_memory_binding >= 0)
    g_cs_list->Barrier(rhi::kAccessHostWrite, rhi::kAccessComputeRead);
  const bool traced = g_cs_timestamps && timeline::Active();
  u16 query = 0;
  if (traced) {
    char name[48];
    std::snprintf(name, sizeof(name), "cs %#llx",
                  (unsigned long long)ci.cs_addr);
    query = timeline::Begin(name);
  }
  const bool stamp =
      g_cs_timestamps && (CsSyncReporting() || traced || kCsHeavyUs);
  if (stamp)
    g_cs_list->WriteTimestamp(g_cs_timestamps, g_cs_batch_count * 2, true);
  DispatchCheckpoint(g_cs_list, ci.cs_addr, false);
  g_cs_list->ProfileBegin(1);
  if (indirect.buffer) {
    g_cs_list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                       rhi::kAccessIndirectRead);
    g_cs_list->DispatchIndirect(indirect.buffer, indirect.offset);
  } else {
    g_cs_list->DispatchBase(ci.group_base[0], ci.group_base[1],
                            ci.group_base[2], groups[0], groups[1], groups[2]);
  }
  g_cs_list->ProfileEnd();
  DispatchCheckpoint(g_cs_list, ci.cs_addr, true);
  if (stamp)
    g_cs_list->WriteTimestamp(g_cs_timestamps, g_cs_batch_count * 2 + 1, false);
  if (traced)
    timeline::End(query);
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!truth_view[i] || !ci.res[i].written)
      continue;
    bool done = false;
    for (u32 j = 0; j < i && !done; j++)
      done = truth_view[j] == truth_view[i] && ci.res[j].written;
    if (done)
      continue;
    auto it =
        g_cs_ranges.find(parent_base[i] ? parent_base[i] : ci.res[i].base);
    if (it == g_cs_ranges.end() || !it->second.buf)
      continue;
    const u64 span =
        parent_base[i]
            ? (ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size)
            : it->second.guest_bytes;
    RecordImageTiling(g_cs_list, *truth_table[i], it->second.buf, parent_off[i],
                      span, truth_view[i], sz[i], false);
    g_cs_batch_access.erase(it->second.buf);
    g_truth_retile_n++;
  }
  {
    BatchedDispatch bd{ci.cs_addr, {ci.groups[0], ci.groups[1], ci.groups[2]},
                       ci.num_res, {},
                       {},         0,
                       0};
    for (u32 i = 0; i < ci.num_res && i < 4; i++) {
      bd.res_base[i] = ci.res[i].base;
      bd.res_size[i] = ci.res[i].size;
      if (ci.res[i].image_staging)
        bd.res_img |= static_cast<u8>(1u << i);
      if (ci.res[i].written)
        bd.res_write |= static_cast<u8>(1u << i);
    }
    bd.traced = traced;
    bd.query = query;
    g_cs_batch_log.push_back(bd);
  }
  if (trace::Recording())
    trace::RecordDispatch(ci);
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill || mirror_res[i])
      continue;
    auto it =
        g_cs_ranges.find(parent_base[i] ? parent_base[i] : ci.res[i].base);
    if (it != g_cs_ranges.end())
      MarkPending(it->second);
  }
  for (u32 i = 0; i < ci.num_res; i++)
    if (mirror_res[i] && ci.res[i].written) {
      const auto& res = ci.res[i];
      const u64 bytes = res.guest_size ? res.guest_size : res.size;
      MirrorWriteBack(res.base, res.write_bytes
                                    ? base::Min(res.write_bytes, bytes)
                                    : bytes);
    }
  ++g_cs_batch_count;

  const u64 t_out0 = NowNs();
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill || mirror_res[i])
      continue;
    const u64 dirty_base = parent_base[i] ? parent_base[i] : ci.res[i].base;
    auto it = g_cs_ranges.find(dirty_base);
    if (it == g_cs_ranges.end())
      continue;
    const auto& res = ci.res[i];
    if (parent_base[i] && !res.image_staging) {
      const u64 child = res.guest_size ? res.guest_size : res.size;
      const u64 lo = dirty_base + parent_off[i];
      IndexDirtySpan(
          dirty_base, lo,
          RangeEnd(
              lo, res.write_bytes ? base::Min(res.write_bytes, child) : child));
    } else {
      const u64 written =
          !parent_base[i] && !res.image_staging && res.write_bytes
              ? base::Min(res.write_bytes, it->second.guest_bytes)
              : it->second.guest_bytes;
      const u64 lo = !parent_base[i] && !res.image_staging
                         ? dirty_base + base::Min(res.view_offset, written)
                         : dirty_base;
      IndexDirtySpan(dirty_base, lo, RangeEnd(dirty_base, written));
    }
    if (!it->second.gpu_dirty)
      it->second.dirty_frame = g_frame.num;
    it->second.gpu_dirty = true;
    it->second.fill_kept.clear();
    it->second.write_seq++;
    it->second.dirty_stamp = ++g_cs_dirty_stamp;
    it->second.dirty_render_serial = g_render_serial;
    BumpTextureEpoch();
    if (truth_view[i])
      StampView(truth_view[i], dirty_base, parent_off[i], truth_table[i],
                it->second.write_seq);
  }
  if ((g_cs_batch_count >= base::Max<u32>(1, kCsBatchCap) || heavy) &&
      (++g_submit_why[4], !CsBatchSubmit())) {
    renderer.state = nullptr;
    return CsDeclined(ci, "21");
  }
  if (guest_table) {
    PendingTable pending{guest_set.set, g_cs_batch_id, ci.cs_addr};
    guest_set.set = ~0u;
    if (g_cs_batch_open)
      g_cs_list->Barrier(rhi::kAccessComputeWrite, rhi::kAccessHostRead);
    if (ci.recomp->guest_memory_written) {
      WriterHistory& history = g_writer_history[ci.cs_addr];
      if (!history.writes && history.quiet >= kQuietAfter) {
        pending.check_writes = true;
        pending.owners = base::move(window_owners);
      } else {
        CsBatchBeginImpl();
        g_cs_list->Barrier(rhi::kAccessComputeWrite, rhi::kAccessHostRead);
        if (!CsBatchFlush(kSyncBatchCap)) {
          ReleaseGuestMemory(pending.set);
          return CsDeclined(ci, "guest-writeback");
        }
        const auto written = TakeWrittenWindows(pending.set);
        if (written.empty()) {
          history.quiet++;
        } else {
          history.writes = true;
          MarkWrittenWindows(ci.cs_addr, window_owners, written);
        }
      }
    }
    g_pending_tables.push_back(base::move(pending));
  }
  g_ns_cs_out += NowNs() - t_out0;
  if (ci.expanded_htile_base)
    NoteExpandedHtile(ci.expanded_htile_base);
  if (capture_dispatch) {
    if (!FlushCsWrites(renderer, "capture-outputs") ||
        !CsBatchFlush(kSyncStageHazard))
      return false;
    trace::CaptureDispatchBindings(ci, pc, kCsPushDwords, wr,
                                   capture_buffers.data());
    trace::CaptureDispatchOutputs(ci);
  }
  return true;
}

}  // namespace gpu::render
