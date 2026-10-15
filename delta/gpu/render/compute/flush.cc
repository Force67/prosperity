/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute/internal.h"

#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/guest_page_table.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "profile/profile.h"

namespace gpu::render::cs {

bool CsRangeFlushOne(u64 base, CsRange& e) {
  if (!e.gpu_dirty)
    return true;
  DELTA_ZONE("gpu.cs_writeback");
  BumpTextureEpoch();
  if (e.imported) {
    if (!CsBatchWaitRange(e, kSyncImported))
      return false;
    NoteGuestWrite(base, e.guest_bytes);
    CsForgetGuestRange(base, e.guest_bytes);
    e.gpu_dirty = false;
    e.cpu_writes.clear();
    return true;
  }
  bool kept = false;
  if (!GpuWriteBack(base, e, &kept)) {
    static u64 failed = 0;
    if (failed++ < 8 || (failed % 4096) == 0)
      BASE_LOGI("gpuvk",
                "cs writeback #{} declined base={:#x} +{:#x} img={} truth={} "
                "dfmt={} (range stays dirty)",
                (unsigned long long)failed, (unsigned long long)base,
                (unsigned long long)e.guest_bytes, (int)e.image_staging,
                (int)e.truth, e.res.dfmt);
    return false;
  }
  g_stats.cs_flush_n++;
  UploadCsRangeToRt(base, e);
  InvalidateTexRange(base, e.guest_bytes);
  UnindexDirtyRange(base, e.guest_bytes);
  g_cs_writeback_gen++;
  CollectGuestWrites();
  CsForgetGuestRange(base, e.guest_bytes);
  e.gpu_dirty = false;
  e.cpu_writes.clear();
  e.last_validated_frame = kept ? -1 : g_frame.num;
  if (!kept && !(InMirror(base) && MirrorOwned(base & ~(kMirrorBlock - 1))))
    e.wt_clean = GuestWriteTracker().Arm(base, e.guest_bytes);
  g_cs_stats.wb_why[e.truth           ? "gpu/img"
                    : e.image_staging ? "gpu/lin"
                                      : "gpu/buf"]++;
  return true;
}

}  // namespace gpu::render::cs

namespace gpu::render {

bool FlushCsWrites(Renderer& renderer, const char* why) {
  DELTA_ZONE("gpu.cs_flush_all");
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  const WaitReaderScope reader(why);
  g_cs_stats.wb_why[base::String("all-call/") + why]++;
  for (u64 base : DirtyBases(/*ordered=*/true)) {
    auto it = g_cs_ranges.find(base);
    if (it == g_cs_ranges.end() || !it->second.gpu_dirty)
      continue;
    g_cs_stats.wb_why[base::String("all/") + why]++;
    g_cs_stats.wb_why_bytes += it->second.size;
    if (!CsRangeFlushOne(it->first, it->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      all_current = false;
    }
  }
  g_stats.ns_cs_out += NowNs() - t0;
  return all_current;
}

bool FlushCsWritesFrameEnd(Renderer& renderer, bool writeback) {
  DELTA_ZONE("gpu.cs_flush_frame_end");
  ReportGpuMemory();
  const WaitReaderScope reader("frame-end");
  HarvestGuestTables();
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  if (writeback) {
    auto stays = [](u64 base, CsRange& e) {
      if (CsLazyKept(base, e))
        return true;
      return kCsImageLazyWb && e.truth && e.gpu_dirty && !e.imported &&
             (e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e));
    };
    for (u64 base : DirtyBases(/*ordered=*/true)) {
      auto it = g_cs_ranges.find(base);
      if (it == g_cs_ranges.end() || !it->second.gpu_dirty)
        continue;
      if (stays(it->first, it->second)) {
        g_cs_stats.lazy_kept_n++;
        g_cs_stats.lazy_kept_bytes += it->second.size;
        continue;
      }
      const CsRange& fe = it->second;
      g_cs_stats
          .wb_why[base::String("frame-end") + (fe.truth           ? "/img"
                                               : fe.image_staging ? "/lin"
                                                                  : "/buf")]++;
      g_cs_stats.wb_why_bytes += fe.size;
      if (!CsRangeFlushOne(it->first, it->second)) {
        if (g_cs_failed) {
          renderer.state = nullptr;
          return false;
        }
        all_current = false;
      }
    }
  }
  if (kCsAged && g_frame.num % 120 == 0) {
    u32 logged = 0;
    for (const auto& [base, e] : g_cs_ranges) {
      if (!e.gpu_dirty || e.dirty_frame >= g_frame.num - 2 || logged++ >= 24)
        continue;
      BASE_LOGI("csaged",
                "base={:#x} +{:#x} age={} truth={} img={} {}x{} layers={} "
                "dfmt={} tiling={} rt={} rtseq={} cpu_reader={} used={}",
                (unsigned long)base, (unsigned long)e.guest_bytes,
                g_frame.num - e.dirty_frame, (int)e.truth, (int)e.image_staging,
                e.res.width, e.res.height, e.res.layers, e.res.dfmt,
                e.res.tiling_idx, (int)CsAliasedBase(base),
                (int)(e.rt_seq == e.write_seq), (int)e.cpu_reader,
                g_frame.num - e.last_used_frame);
    }
  }
  {
    DELTA_ZONE("gpu.cs_evict");
    const size_t budget = (g_cs_ranges.size() + 15) / 16;
    for (size_t n = 0; n < budget && !g_cs_ranges.empty(); n++) {
      if (g_cs_evict_next == g_cs_ranges.end())
        g_cs_evict_next = g_cs_ranges.begin();
      auto it = g_cs_evict_next++;
      CsRange& e = it->second;
      if (e.pending_batch || CsFrameReferenced(e) ||
          e.last_used_frame + 60 >= g_frame.num || (writeback && e.gpu_dirty)) {
        continue;
      }
      if (e.gpu_dirty) {
        g_cs_stats.wb_why["idle"]++;
        g_cs_stats.wb_why_bytes += e.size;
        if (!CsRangeFlushOne(it->first, e)) {
          if (g_cs_failed) {
            renderer.state = nullptr;
            return false;
          }
          continue;
        }
      }
      EraseCsRange(it);
    }
  }
  if (g_cs_batch_open && (++g_cs_stats.submit_why[5], !CsBatchSubmit())) {
    renderer.state = nullptr;
    return false;
  }
  g_stats.ns_cs_out += NowNs() - t0;
  return all_current;
}

bool FlushCsWritesAt(Renderer& renderer, u64 base, const char* why) {
  auto found = g_cs_ranges.find(base);
  if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
    return true;
  const CsRange& e = found->second;
  return FlushCsWritesRange(renderer, base,
                            e.guest_bytes ? e.guest_bytes : e.size, why);
}

bool SettleGuestWrites(u64 base, u64 bytes, const char* why) {
  if (g_pending_guest_writes.empty())
    return true;
  const WaitReaderScope reader(why);
  return WaitPendingGuestWrites(base, bytes);
}

static bool FlushDirtyRanges(Renderer& renderer,
                             u64 base,
                             u64 bytes,
                             const char* why);

DELTA_OPTION(bool, kSpeculateReads, "DELTA_GPU_SPECULATE_READS", true);
struct SpeculatedRead {
  u64 base, bytes, batch;
  u8 seen[64];
};
base::Vector<SpeculatedRead> g_speculated;
constexpr u8 kSpecMaxChanges = 3;
base::HashSet<u64> g_spec_known;
struct SpecChanges {
  u8 n = 0;
  int frame = -1;
};
base::HashMap<u64, SpecChanges> g_spec_changes;
u64 g_spec_misses = 0;
const guest::SessionReset g_spec_reset([] {
  guest::ResetResource(g_speculated);
  guest::ResetResource(g_spec_known);
  guest::ResetResource(g_spec_changes);
  guest::ResetResource(g_spec_misses);
});

bool SpecVolatile(u64 base) {
  const auto it = g_spec_changes.find(base);
  return it != g_spec_changes.end() && it->second.n >= kSpecMaxChanges;
}

bool SpeculateGuestRead(u64 base, u64 bytes) {
  if (!kSpeculateReads || !bytes || bytes > sizeof(SpeculatedRead::seen) ||
      g_speculated.size() >= 4096 || SpecVolatile(base) ||
      CsRangeDirtyOverlapping(base, bytes))
    return false;
  const u64 batch = PendingGuestWriteBatch(base, bytes);
  if (!batch || g_spec_known.insert(base).second ||
      !gpu::IsReadableRangeCached(base, bytes))
    return false;
  for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
       block += kMirrorBlock)
    if (InMirror(block) && MirrorOwned(block))
      return false;
  SpeculatedRead& r = g_speculated.emplace_back();
  r.base = base;
  r.bytes = bytes;
  r.batch = batch;
  std::memcpy(r.seen, reinterpret_cast<const void*>(base), bytes);
  return true;
}

void CheckSpeculatedReads() {
  base::EraseIf(g_speculated, [](const SpeculatedRead& r) {
    if (r.batch > g_cs_batch_done)
      return false;
    if (std::memcmp(r.seen, reinterpret_cast<const void*>(r.base), r.bytes)) {
      SpecChanges& c = g_spec_changes[r.base];
      if (c.frame != g_frame.num) {
        c.frame = g_frame.num;
        if (++c.n == kSpecMaxChanges && g_spec_misses++ < 8)
          BASE_LOGI("gpuvk",
                    "speculated read {:#x}+{} keeps changing; it waits now",
                    (unsigned long long)r.base, (unsigned long long)r.bytes);
      }
    }
    return true;
  });
}

bool FlushCsWritesForGpu(Renderer& renderer,
                         u64 base,
                         u64 bytes,
                         const char* why) {
  return FlushDirtyRanges(renderer, base, bytes, why);
}

bool FlushCsWritesRange(Renderer& renderer,
                        u64 base,
                        u64 bytes,
                        const char* why) {
  if (!g_own_unsubmitted.empty() && bytes) {
    for (u64 block = base & ~(kMirrorBlock - 1); block < base + bytes;
         block += kMirrorBlock)
      if (InMirror(block) &&
          (OwnAt(block).state.load(base::memory_order_acquire) & kOwnPending)) {
        if (!CsBatchSubmit())
          return false;
        break;
      }
  }
  return FlushDirtyRanges(renderer, base, bytes, why) &&
         SettleGuestWrites(base, bytes, why);
}

static bool FlushDirtyRanges(Renderer& renderer,
                             u64 base,
                             u64 bytes,
                             const char* why) {
  if (g_cs_dirty_pages.empty())
    return true;
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  if (!base || !bytes || g_cs_ranges.empty() ||
      (!CsRangeDirtyOverlapping(base, bytes) &&
       !(kCsFlushTrace && base == (u64)kCsFlushTrace)))
    return true;
  ScopeNs flush_timer(&g_stats.ns_cs_flush);
  const u64 t0 = NowNs();
  bool all_current = true;
  if (kCsFlushTrace && base == (u64)kCsFlushTrace) {
    static int n = 0;
    if (n++ < 12) {
      const auto ranges = DirtyRangesOverlapping(base, bytes);
      base::String line;
      base::FormatTo(line, "base={:#x} bytes={:#x} dirty={}",
                     (unsigned long)base, (unsigned long)bytes, ranges.size());
      for (u64 r : ranges) {
        auto f = g_cs_ranges.find(r);
        base::FormatTo(
            line, " [{:#x} img={} gpu_dirty={} sz={:#x}]", (unsigned long)r,
            f != g_cs_ranges.end() ? (int)f->second.image_staging : -1,
            f != g_cs_ranges.end() ? (int)f->second.gpu_dirty : -1,
            f != g_cs_ranges.end() ? (unsigned long)f->second.size : 0);
      }
      for (auto& kv : g_cs_ranges) {
        const u64 e0 = kv.first, e1 = e0 + kv.second.guest_bytes;
        if (e0 < base + bytes && base < e1)
          base::FormatTo(
              line, " OVERLAP[{:#x}+{:#x} img={} dirty={}]", (unsigned long)e0,
              (unsigned long)kv.second.guest_bytes,
              (int)kv.second.image_staging, (int)kv.second.gpu_dirty);
      }
      BASE_LOGI("csflush", "{}", line.c_str());
    }
  }
  auto overlapping = DirtyRangesOverlapping(base, bytes);
  SortByDirtyOrder(overlapping);
  if (CsSyncReporting()) {
    static int shown = 0;
    for (u64 d : overlapping) {
      auto f = g_cs_ranges.find(d);
      if (f == g_cs_ranges.end() || !f->second.gpu_dirty || shown >= 24)
        continue;
      shown++;
      const CsRange& o = f->second;
      BASE_LOGI("rtflush",
                "{} rt {:#x}+{:#x} <- range {:#x}+{:#x} img={} truth={} "
                "{}x{}x{} e={}/{} dfmt={} t={}",
                why, base, bytes, d, o.guest_bytes, (int)o.image_staging,
                (int)o.truth, o.res.width, o.res.height, o.res.layers,
                o.res.elem_bytes, o.res.stage_elem_bytes, o.res.dfmt,
                o.res.tiling_idx);
    }
  }
  const WaitReaderScope reader(why);
  struct FlushLog {
    u64 base, bytes, t0;
    const char* why;
    base::Vector<u64> ranges;
    ~FlushLog() {
      static int shown = 0;
      if (!kCsFlushFrom || g_frame.num < kCsFlushFrom || shown >= 400 ||
          ranges.empty())
        return;
      shown++;
      base::String line;
      for (u64 r : ranges) {
        auto f = g_cs_ranges.find(r);
        if (f != g_cs_ranges.end())
          base::FormatTo(line, " {:#x}+{:#x}{}", r, f->second.guest_bytes,
                         f->second.truth           ? "/img"
                         : f->second.image_staging ? "/lin"
                                                   : "/buf");
      }
      BASE_LOGI("csflush", "f{} {} {:#x}+{:#x} {:.2f}ms <-{}", (int)g_frame.num,
                why, base, bytes, (NowNs() - t0) / 1e6, line.c_str());
    }
  } flush_log{base, bytes, NowNs(), why, overlapping};
  for (u64 dirty : overlapping) {
    auto found = g_cs_ranges.find(dirty);
    if (found != g_cs_ranges.end() && found->second.truth &&
        !std::strcmp(why, "srt"))
      continue;
    if (found != g_cs_ranges.end() && found->second.gpu_dirty) {
      const CsRange& e = found->second;
      g_cs_stats.wb_why[base::String(why) + (e.truth           ? "/img"
                                             : e.image_staging ? "/lin"
                                                               : "/buf")]++;
      g_cs_stats.wb_why_bytes += e.size;
    }
    if (found != g_cs_ranges.end())
      found->second.cpu_reader = true;
    if (found != g_cs_ranges.end() && !CsRangeFlushOne(dirty, found->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      all_current = false;
    }
  }
  g_stats.ns_cs_out += NowNs() - t0;
  return all_current;
}

}  // namespace gpu::render

namespace gpu::render::cs {

void FillOverDirtyRanges(u64 base, u64 bytes) {
  const u64 end = base + bytes;
  if (g_cs_dirty_pages.empty())
    return;
  for (u64 d : DirtyRangesOverlapping(base, bytes)) {
    auto found = g_cs_ranges.find(d);
    if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
      continue;
    CsRange& e = found->second;
    if (e.imported)
      continue;
    if (d >= base && d + e.guest_bytes <= end) {
      UnindexDirtyRange(d, e.guest_bytes);
      e.gpu_dirty = false;
      e.cpu_writes.clear();
      e.hash = 0;
      e.last_validated_frame = -1;
      g_cs_stats.wb_why["fill-covered"]++;
      continue;
    }
    u64 lo = base::Max(base, d), hi = base::Min(end, d + e.guest_bytes);
    base::EraseIf(e.fill_kept, [&](const base::Pair<u64, u64>& k) {
      if (k.first > hi || lo > k.second)
        return false;
      lo = base::Min(lo, k.first);
      hi = base::Max(hi, k.second);
      return true;
    });
    e.fill_kept.emplace_back(lo, hi);
    g_cs_stats.wb_why["fill-kept"]++;
  }
}

}  // namespace gpu::render::cs

namespace gpu::render {

static bool ApplyGpuFill(u64 base, u64 bytes, u32 lo, u32 hi) {
  BumpTextureEpoch();
  if (!bytes)
    return false;
  FillOverDirtyRanges(base, bytes);
  if (!GpuFill(base, bytes, lo, hi)) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGW("gpuvk", "fill {:#x}+{:#x} cannot run on the GPU, dropped",
                (unsigned long long)base, (unsigned long long)bytes);
    return false;
  }
  CsForgetGuestRange(base, bytes);
  InvalidateTexRange(base, bytes);
  g_cs_writeback_gen++;
  return true;
}

void ApplyMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value) {
  ScopeNs fill_timer(&g_stats.ns_memory_fill);
  g_stats.memory_fill_bytes += bytes;
  g_stats.memory_fill_n++;
  if (ApplyGpuFill(base, bytes & ~u64(3), value, value))
    NoteMemoryFill(renderer, base, bytes, value);
}

void ApplyMemoryFill64(Renderer& renderer, u64 base, u64 bytes, u64 value) {
  ScopeNs fill_timer(&g_stats.ns_memory_fill);
  g_stats.memory_fill_bytes += bytes;
  g_stats.memory_fill_n++;
  if (ApplyGpuFill(base, bytes & ~u64(7), u32(value), u32(value >> 32)))
    NoteMemoryFill64(renderer, base, bytes, value);
}

bool ApplyMemoryCopy(Renderer& renderer, u64 dst, u64 src, u64 bytes) {
  if (!FlushCsWritesForGpu(renderer, src, bytes, "dma") ||
      !FlushCsWritesForGpu(renderer, dst, bytes, "dma"))
    return false;
  BumpTextureEpoch();
  if (!GpuCopy(dst, src, bytes))
    return false;
  CsForgetGuestRange(dst, bytes);
  InvalidateTexRange(dst, bytes);
  g_cs_writeback_gen++;
  return true;
}

bool GuestWriteMaybePending(u64 base, u64 bytes) {
  return bytes && !GuestPages().PendingWriteClear(base, base + bytes);
}

bool GuestBytesSettled(u64 base, u64 bytes) {
  return !CsRangeDirtyOverlapping(base, bytes) &&
         !PendingGuestWriteBatch(base, bytes) &&
         !MirrorOwnsAny(base, base + bytes);
}

}  // namespace gpu::render
