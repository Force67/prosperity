/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute/internal.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/debug/perf.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/image/texture_cache.h"
#include "gpu/rhi/write_journal.h"
#include "guest/session.h"

namespace gpu::render {

void MirrorReportCapped();

void CsSyncReport(double frames) {
  if (CsSyncReporting() && frames > 0) {
    base::Vector<base::Pair<u64, CsGpuTime>> times;
    for (const auto& [base, time] : g_cs_stats.gpu_times)
      times.push_back({base, time});
    base::Sort(times.begin(), times.end(), [](const auto& a, const auto& b) {
      return a.second.ns > b.second.ns;
    });
    for (size_t i = 0; i < base::Min<size_t>(8, times.size()); ++i)
      BASE_LOGI("cstime", "CS {:#x} {:.2f}ms/f x{:.1f}", times[i].first,
                times[i].second.ns / frames / 1e6,
                double(times[i].second.count) / frames);
    g_cs_stats.gpu_times.clear();
  }
  u64 total = 0;
  for (int i = 0; i < kSyncCount; i++)
    total += g_cs_stats.sync_n[i];
  if (!CsSyncReporting()) {
    for (int i = 0; i < kSyncCount; i++)
      g_cs_stats.sync_n[i] = g_cs_stats.sync_ns[i] = 0;
    return;
  }
  BASE_LOGI("csin",
            "hash={:.1f}ms x{:.1f} {:.1f}MB "
            "wb-hash={:.1f}ms {:.1f}MB detile={:.1f}ms "
            "rt-bridge={:.1f}ms x{:.1f} (in-frame x{:.1f}) copy={:.1f}ms "
            "tracked-clean={:.1f}x",
            g_cs_stats.in_hash_ns / frames / 1e6, g_cs_stats.in_hash_n / frames,
            g_cs_stats.in_hash_bytes / frames / 1e6,
            g_cs_stats.wb_hash_ns / frames / 1e6,
            g_cs_stats.wb_hash_bytes / frames / 1e6,
            g_cs_stats.in_detile_ns / frames / 1e6,
            g_cs_stats.in_rt_ns / frames / 1e6, g_cs_stats.in_rt_n / frames,
            g_cs_stats.in_frame_bridge_n / frames,
            g_cs_stats.in_copy_ns / frames / 1e6,
            g_cs_stats.in_wt_skips / frames);
  BASE_LOGI(
      "csin2",
      "overlap-wb={:.1f}ms x{:.1f} reshape-wb={:.1f}ms x{:.1f} "
      "alloc={:.1f}ms x{:.1f} recycled={:.1f} ranges={} {:.0f}MB "
      "membuf={:.1f}ms lazy-kept={:.1f}x {:.1f}MB | truth stage={:.1f}MB "
      "import={:.1f} detile={:.1f} retile={:.1f} parent={:.1f} "
      "fold={:.1f} rt={:.1f} view-hits={:.1f} scratch={:.0f}MB",
      g_cs_stats.in_overlap_ns / frames / 1e6, g_cs_stats.in_overlap_n / frames,
      g_cs_stats.in_reshape_ns / frames / 1e6, g_cs_stats.in_reshape_n / frames,
      g_cs_stats.in_alloc_ns / frames / 1e6, g_cs_stats.in_alloc_n / frames,
      g_cs_stats.recycled_n / frames, g_cs.ranges.size(),
      g_cs.range_bytes / 1e6, g_cs_stats.in_membuf_ns / frames / 1e6,
      g_cs_stats.lazy_kept_n / frames,
      g_cs_stats.lazy_kept_bytes / frames / 1e6,
      g_cs_stats.truth_stage_bytes / frames / 1e6,
      g_cs_stats.truth_import_n / frames, g_cs_stats.truth_detile_n / frames,
      g_cs_stats.truth_retile_n / frames, g_cs_stats.truth_parent_n / frames,
      g_cs_stats.truth_fold_n / frames, g_cs_stats.truth_rt_n / frames,
      g_cs_stats.truth_view_hits / frames, g_cs.scratch_bytes / 1e6);
  {
    base::String why;
    base::FormatTo(why, "writebacks/frame ({:.1f}MB):",
                   g_cs_stats.wb_why_bytes / frames / 1e6);
    for (const auto& kv : g_cs_stats.wb_why)
      base::FormatTo(why, " {}={:.1f}", kv.first.c_str(), kv.second / frames);
    BASE_LOGI("cswb", "{}", why.c_str());
  }
  if (kMirror) {
    const auto union_bytes = [](const base::Map<u64, u64>& m) {
      u64 uni = 0, lo = 0, hi = 0;
      for (const auto& [b0, e0] : m) {
        if (b0 > hi) {
          uni += hi - lo;
          lo = b0;
          hi = e0;
        } else {
          hi = base::Max(hi, e0);
        }
      }
      return uni + hi - lo;
    };
    const auto top = [&](const char* what, const auto& m) {
      base::Vector<base::Pair<u64, base::Pair<u64, u64>>> v;
      for (const auto& kv : m)
        v.push_back({kv.first, kv.second});
      base::Sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        return a.second.first > b.second.first;
      });
      for (size_t i = 0; i < v.size() && i < 10; i++)
        BASE_LOGI("csmirror", "{} {:#x} {:.2f}MB/f x{:.1f} ({:.0f}KB each)",
                  what, (unsigned long)v[i].first,
                  v[i].second.first / frames / 1e6, v[i].second.second / frames,
                  v[i].second.first / 1e3 / v[i].second.second);
    };
    MirrorReportCapped();
    BASE_LOGI("csmirror",
              "copy-in {:.1f}MB x{:.1f} per frame, {:.1f}MB distinct per "
              "report",
              g_cs_stats.mirror_in_bytes / frames / 1e6,
              g_cs_stats.mirror_in_n / frames,
              union_bytes(g_cs_stats.mirror_in_union) / 1e6);
    top("in", g_cs_stats.mirror_in_top);
    top("wb", g_cs_stats.mirror_wb_top);
    const u64 uni = union_bytes(g_cs_stats.mirror_wb_union);
    BASE_LOGI("csstage",
              "mirror write-back {:.1f}MB x{:.1f} per frame, "
              "{:.1f}MB distinct per report; kept {:.1f}MB, {:.1f} faults",
              g_cs_stats.mirror_wb_bytes / frames / 1e6,
              g_cs_stats.mirror_wb_n / frames, uni / 1e6,
              g_cs_stats.mirror_own_bytes / frames / 1e6,
              g_mirror_resolves.exchange(0) / frames);
  }
  BASE_LOGI("csstage",
            "per frame ro={:.1f}MB rw={:.1f}MB img={:.1f}MB (guest-detile "
            "{:.1f}MB x{:.1f})",
            g_cs_stats.stage_ro_bytes / frames / 1e6,
            g_cs_stats.stage_rw_bytes / frames / 1e6,
            g_cs_stats.stage_img_bytes / frames / 1e6,
            g_cs_stats.stage_guest_detile_bytes / frames / 1e6,
            g_cs_stats.stage_guest_detile_n / frames);
  base::Vector<base::Pair<u64, StageStat>> top;
  for (const auto& [base, stat] : g_cs_stats.stage_stats)
    top.push_back({base, stat});
  base::Sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
    return a.second.bytes + a.second.wait_ns / 8 >
           b.second.bytes + b.second.wait_ns / 8;
  });
  for (size_t i = 0; i < top.size() && i < 8; i++) {
    const StageStat& s = top[i].second;
    const auto found = g_cs.ranges.find(top[i].first);
    const ComputeInfo::Res* r =
        found == g_cs.ranges.end() ? nullptr : &found->second.res;
    BASE_LOGI("cstop",
              "{:#x} {:.1f}MB/f x{:.1f} (first={:.1f} shape={:.1f} hash={:.1f} "
              "rt={:.1f}) wait={:.1f}x {:.1f}ms | {}x{} l={} m={} e={} t={}",
              (unsigned long)top[i].first, s.bytes / frames / 1e6, s.n / frames,
              s.first / frames, s.shape / frames, s.hash / frames,
              s.rt / frames, s.wait / frames, s.wait_ns / frames / 1e6,
              r ? r->width : 0, r ? r->height : 0, r ? r->layers : 0,
              r ? r->mip_levels : 0, r ? r->elem_bytes : 0,
              r ? r->tiling_idx : 0);
  }
  BASE_LOGI(
      "csout",
      "rt-upload={:.1f}x "
      "texbridge={:.1f}x {:.1f}MB bufbridge={:.1f}x {:.1f}MB "
      "(declined {:.1f}) vtxbridge={:.1f}x chunks={:.1f} "
      "guest-override={:.1f}x "
      "mirror-tiles={:.1f}x rt-variants={:.1f}x gds-imported={:.1f}x "
      "guest-misses={:.1f}x gpu-indirect={:.1f}x staged-indirect={:.1f}x",
      g_cs_stats.out_rt_submits / frames, g_cs_stats.tex_bridge_n / frames,
      g_cs_stats.tex_bridge_bytes / frames / 1e6,
      g_cs_stats.buf_bridge_n / frames,
      g_cs_stats.buf_bridge_bytes / frames / 1e6,
      g_cs_stats.buf_bridge_declined / frames, g_cs_stats.vtx_bridge_n / frames,
      g_cs_stats.chunk_splits / frames, g_cs_stats.guest_override_n / frames,
      g_cs_stats.guest_mirror_tiles / frames,
      g_cs_stats.rt_variants_made / frames, g_cs_stats.gds_imported_n / frames,
      g_cs_stats.guest_misses_n / frames, g_cs_stats.indirect_n / frames,
      g_cs_stats.indirect_staged_n / frames);
  base::String syncs;
  base::FormatTo(syncs, "{:.1f} syncs/frame ({:.1f} submits):", total / frames,
                 g_cs_stats.submit_n / frames);
  for (int i = 0; i < kSyncCount; i++) {
    if (!g_cs_stats.sync_n[i])
      continue;
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", kCsSyncName[i],
                   g_cs_stats.sync_n[i] / frames,
                   g_cs_stats.sync_ns[i] / frames / 1e6);
  }
  base::FormatTo(syncs, " | by reader:");
  for (const auto& [reader, w] : g_cs_stats.reader_waits)
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", reader.c_str(),
                   w.first / frames, w.second / frames / 1e6);
  base::FormatTo(syncs, " | writebacks:");
  for (const auto& [reader, w] : g_cs_stats.reader_wbs)
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", reader.c_str(),
                   w.first / frames, w.second / frames / 1e6);
  base::FormatTo(
      syncs,
      " | submits: wait={:.1f} chunk={:.1f} label={:.1f} "
      "heavy={:.1f} cap={:.1f} end={:.1f}",
      g_cs_stats.submit_why[0] / frames, g_cs_stats.submit_why[1] / frames,
      g_cs_stats.submit_why[2] / frames, g_cs_stats.submit_why[3] / frames,
      g_cs_stats.submit_why[4] / frames, g_cs_stats.submit_why[5] / frames);
  BASE_LOGI("cssync", "{}", syncs.c_str());
  g_cs_stats = {};
}

DELTA_OPTION(bool, kQueueCheck, "DELTA_GPU_QCHECK", false);

bool QueueCheckArmed() {
  return kQueueCheck;
}

bool QueueCheck(const char* where) {
  if (!kQueueCheck)
    return true;
  rhi::CommandList* list = BeginImmediate();
  if (!list)
    return true;
  const bool ok = EndImmediate(list, "imm.queue_check");
  if (!ok)
    BASE_LOGI("gpuvk", "queue CHECKPOINT FAILED at {}", where);
  return ok;
}

namespace {
bool DescribeStagedRange(u64 addr, char* out, size_t out_size) {
  for (const auto& kv : g_cs.ranges) {
    const u64 base = kv.first;
    const CsRange& e = kv.second;
    const u64 n = e.guest_bytes ? e.guest_bytes : e.size;
    if (!n || addr < base || addr >= base + n)
      continue;
    std::snprintf(out, out_size,
                  "base=%#llx +%#llx (addr is base+%#llx) staged=%#llx "
                  "gpu_dirty=%d imported=%d rt_sourced=%d image=%d "
                  "last_used_frame=%d",
                  (unsigned long long)base, (unsigned long long)n,
                  (unsigned long long)(addr - base), (unsigned long long)e.size,
                  (int)e.gpu_dirty, (int)e.imported, (int)e.rt_sourced,
                  (int)e.image_staging, e.last_used_frame);
    return true;
  }
  return false;
}

bool DescribeMirrorBlock(u64 addr, char* out, size_t out_size) {
  if (!InMirror(addr))
    return false;
  const u64 block = addr & ~(kMirrorBlock - 1);
  const MirrorWrite* written = MirrorWritten(addr, 1);
  const u64 state = g_cs.own ? OwnAt(block).state.load() : 0;
  if (!written && !state)
    return false;
  u64 written_first = 0;
  if (written) {
    auto it = g_cs.mirror_writes.upper_bound(addr);
    written_first = (--it)->first;
  }
  const OwnBlock& o = OwnAt(block);
  Dl_info site{};
  const u64 written_by =
      written && written->site && dladdr(written->site, &site) &&
              site.dli_fbase
          ? reinterpret_cast<u64>(written->site) -
                reinterpret_cast<u64>(site.dli_fbase)
          : 0;
  const u64 owned_by =
      o.owned_by && dladdr(o.owned_by, &site) && site.dli_fbase
          ? reinterpret_cast<u64>(o.owned_by) -
                reinterpret_cast<u64>(site.dli_fbase)
          : 0;
  std::snprintf(out, out_size,
                "mirror block %#llx state=%#llx gpu-written=[%#llx,%#llx) "
                "(seq %llu, by +%#llx) owned-by=+%#llx merged=%d",
                (unsigned long long)block, (unsigned long long)state,
                (unsigned long long)written_first,
                (unsigned long long)(written ? written->end : 0),
                (unsigned long long)(written ? written->seq : 0),
                (unsigned long long)written_by, (unsigned long long)owned_by,
                (int)o.owned_merged);
  return true;
}

// The GPU commands that last wrote near `addr`, appended to `out`.
bool AppendJournal(u64 addr, char* out, size_t out_size) {
  rhi::write_journal::Entry hits[3];
  const u32 n = rhi::write_journal::Find(addr, 64, hits, 3);
  for (u32 i = 0; i < n; i++) {
    const size_t used = std::strlen(out);
    if (used + 1 >= out_size)
      break;
    std::snprintf(out + used, out_size - used,
                  "%s%c [%#llx,%#llx) frame %u ctx %#llx",
                  used ? " | " : "written by ", hits[i].kind,
                  (unsigned long long)hits[i].first,
                  (unsigned long long)hits[i].end, hits[i].frame,
                  (unsigned long long)hits[i].context);
  }
  return n != 0;
}
}  // namespace

bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size) {
  out[0] = 0;
  const bool known = DescribeStagedRange(addr, out, out_size) ||
                     DescribeMirrorBlock(addr, out, out_size);
  return AppendJournal(addr, out, out_size) || known;
}

DELTA_OPTION(bool, kMemStat, "DELTA_GPU_MEMSTAT", false);

void ReportGpuMemory() {
  if (!kMemStat || g_frame.num % 300)
    return;
  u64 dirty = 0, truth = 0, cold = 0;
  for (const auto& [base, e] : g_cs.ranges) {
    (void)base;
    if (e.gpu_dirty)
      dirty += e.cap;
    if (e.truth)
      truth += e.cap;
    if (e.last_used_frame + 60 < g_frame.num)
      cold += e.cap;
  }
  u64 rt = 0, rt_parked = 0, depth = 0;
  for (const auto& [base, t] : g_rts) {
    (void)base;
    rt += RtByteSize(t);
  }
  for (const auto& [base, v] : g_targets.rt_variants) {
    (void)base;
    for (const RTarget& t : v)
      rt_parked += RtByteSize(t);
  }
  for (const auto& [base, d] : g_depths) {
    (void)base;
    depth += u64(d.w) * d.h * 5 * d.layers;
  }
  for (const auto& [base, v] : g_targets.depth_variants) {
    (void)base;
    for (const DepthTarget& d : v)
      depth += u64(d.w) * d.h * 5 * d.layers;
  }
  u64 scratch = 0;
  for (const CsScratch& c : g_cs.scratch)
    scratch += c.cap;
  u64 heap_used = 0, heap_budget = 0;
  Device().QueryMemoryBudget(&heap_used, &heap_budget);
  constexpr double kMb = 1024.0 * 1024.0;
  BASE_LOGI("memstat",
            "f{} device heaps {:.0f}/{:.0f}MB | tex {:.0f}MB "
            "scratch {:.0f}MB",
            g_frame.num, heap_used / kMb, heap_budget / kMb,
            TextureImageBytes() / kMb, scratch / kMb);
  BASE_LOGI("memstat",
            "f{} cs ranges={} {:.0f}MB (dirty {:.0f} truth {:.0f} cold {:.0f}) "
            "free {:.0f}MB | rt {} {:.0f}MB parked {:.0f}MB | depth {:.0f}MB",
            g_frame.num, g_cs.ranges.size(), g_cs.range_bytes / kMb,
            dirty / kMb, truth / kMb, cold / kMb, g_cs.free_bytes / kMb,
            g_rts.size(), rt / kMb, rt_parked / kMb, depth / kMb);
}

}  // namespace gpu::render
