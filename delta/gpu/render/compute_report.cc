/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Diagnostics: the DELTA_GPU_CSSYNC report, queue checkpoints and memory
// statistics.

#include "gpu/render/compute_internal.h"

#include <cstdio>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/perf.h"
#include "gpu/render/render_target.h"
#include "guest/session.h"

namespace gpu::render {

extern u64 g_tex_image_bytes;

void CsSyncReport(double frames) {
  if (CsSyncReporting() && frames > 0) {
    base::Vector<base::Pair<u64, CsGpuTime>> times;
    for (const auto& [base, time] : g_cs_gpu_times)
      times.push_back({base, time});
    base::Sort(times.begin(), times.end(), [](const auto& a, const auto& b) {
      return a.second.ns > b.second.ns;
    });
    for (size_t i = 0; i < base::Min<size_t>(8, times.size()); ++i)
      BASE_LOGI("cstime", "CS {:#x} {:.2f}ms/f x{:.1f}", times[i].first,
                times[i].second.ns / frames / 1e6,
                double(times[i].second.count) / frames);
    g_cs_gpu_times.clear();
  }
  u64 total = 0;
  for (int i = 0; i < kSyncCount; i++)
    total += g_cs_sync_n[i];
  if (!CsSyncReporting() || !total) {
    for (int i = 0; i < kSyncCount; i++)
      g_cs_sync_n[i] = g_cs_sync_ns[i] = 0;
    return;
  }
  BASE_LOGI("csin",
            "hash={:.1f}ms x{:.1f} {:.1f}MB "
            "wb-hash={:.1f}ms {:.1f}MB detile={:.1f}ms "
            "rt-bridge={:.1f}ms x{:.1f} (in-frame x{:.1f}) copy={:.1f}ms "
            "tracked-clean={:.1f}x",
            g_in_hash_ns / frames / 1e6, g_in_hash_n / frames,
            g_in_hash_bytes / frames / 1e6,
            g_wb_hash_ns / frames / 1e6, g_wb_hash_bytes / frames / 1e6,
            g_in_detile_ns / frames / 1e6, g_in_rt_ns / frames / 1e6,
            g_in_rt_n / frames, g_in_frame_bridge_n / frames,
            g_in_copy_ns / frames / 1e6, g_in_wt_skips / frames);
  g_in_hash_ns = g_in_detile_ns = g_in_rt_ns = g_in_copy_ns = 0;
  g_in_hash_n = g_in_rt_n = g_in_frame_bridge_n = g_in_wt_skips = 0;
  g_in_hash_bytes = g_wb_hash_bytes = g_wb_hash_ns = 0;
  BASE_LOGI(
      "csin2",
      "overlap-wb={:.1f}ms x{:.1f} reshape-wb={:.1f}ms x{:.1f} "
      "alloc={:.1f}ms x{:.1f} recycled={:.1f} ranges={} {:.0f}MB "
      "membuf={:.1f}ms lazy-kept={:.1f}x {:.1f}MB | truth stage={:.1f}MB "
      "import={:.1f} detile={:.1f} retile={:.1f} parent={:.1f} "
      "fold={:.1f} rt={:.1f} view-hits={:.1f} scratch={:.0f}MB",
      g_in_overlap_ns / frames / 1e6, g_in_overlap_n / frames,
      g_in_reshape_ns / frames / 1e6, g_in_reshape_n / frames,
      g_in_alloc_ns / frames / 1e6, g_in_alloc_n / frames,
      g_cs_recycled_n / frames, g_cs_ranges.size(), g_cs_range_bytes / 1e6,
      g_in_membuf_ns / frames / 1e6,
      g_lazy_kept_n / frames, g_lazy_kept_bytes / frames / 1e6,
      g_truth_stage_bytes / frames / 1e6, g_truth_import_n / frames,
      g_truth_detile_n / frames, g_truth_retile_n / frames,
      g_truth_parent_n / frames, g_truth_fold_n / frames, g_truth_rt_n / frames,
      g_truth_view_hits / frames, g_cs_scratch_bytes / 1e6);
  g_truth_view_hits = 0;
  {
    base::String why;
    base::FormatTo(
        why, "writebacks/frame ({:.1f}MB):", g_wb_why_bytes / frames / 1e6);
    for (const auto& kv : g_wb_why)
      base::FormatTo(why, " {}={:.1f}", kv.first.c_str(), kv.second / frames);
    BASE_LOGI("cswb", "{}", why.c_str());
    g_wb_why.clear();
    g_wb_why_bytes = 0;
  }
  g_in_overlap_ns = g_in_reshape_ns = g_in_alloc_ns = 0;
  g_in_membuf_ns = 0;
  g_in_overlap_n = g_in_reshape_n = g_in_alloc_n = g_cs_recycled_n = 0;
  g_truth_stage_bytes = g_truth_import_n = g_truth_detile_n = 0;
  g_truth_retile_n = g_truth_parent_n = g_truth_fold_n = g_truth_rt_n = 0;
  g_lazy_kept_n = g_lazy_kept_bytes = 0;
  if (kMirror)
  {
    u64 uni = 0, lo = 0, hi = 0;
    for (const auto& [b0, e0] : g_mirror_wb_union) {
      if (b0 > hi) {
        uni += hi - lo;
        lo = b0;
        hi = e0;
      } else {
        hi = base::Max(hi, e0);
      }
    }
    uni += hi - lo;
    g_mirror_wb_union.clear();
    BASE_LOGI("csstage", "mirror write-back {:.1f}MB x{:.1f} per frame, "
              "{:.1f}MB distinct; kept {:.1f}MB, {:.1f} faults",
              g_mirror_wb_bytes / frames / 1e6, g_mirror_wb_n / frames,
              uni / frames / 1e6, g_mirror_own_bytes / frames / 1e6,
              g_mirror_resolves.exchange(0) / frames);
    g_mirror_own_bytes = 0;
  }
  g_mirror_wb_bytes = g_mirror_wb_n = 0;
  BASE_LOGI("csstage",
            "per frame ro={:.1f}MB rw={:.1f}MB img={:.1f}MB (guest-detile "
            "{:.1f}MB x{:.1f})",
            g_stage_ro_bytes / frames / 1e6, g_stage_rw_bytes / frames / 1e6,
            g_stage_img_bytes / frames / 1e6,
            g_stage_guest_detile_bytes / frames / 1e6,
            g_stage_guest_detile_n / frames);
  g_stage_ro_bytes = g_stage_rw_bytes = g_stage_img_bytes = 0;
  g_stage_guest_detile_bytes = g_stage_guest_detile_n = 0;
  // The ranges that actually cost the frame, most expensive first: what to fix
  // is whichever reason column is not "hash" (guest memory genuinely changed).
  base::Vector<base::Pair<u64, StageStat>> top;
  for (const auto& [base, stat] : g_stage_stats)
    top.push_back({base, stat});
  base::Sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
    return a.second.bytes + a.second.wait_ns / 8 >
           b.second.bytes + b.second.wait_ns / 8;
  });
  for (size_t i = 0; i < top.size() && i < 8; i++) {
    const StageStat& s = top[i].second;
    const auto found = g_cs_ranges.find(top[i].first);
    const ComputeInfo::Res* r =
        found == g_cs_ranges.end() ? nullptr : &found->second.res;
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
  g_stage_stats.clear();
  BASE_LOGI("csout",
            "rt-upload={:.1f}x "
            "texbridge={:.1f}x {:.1f}MB bufbridge={:.1f}x {:.1f}MB "
            "(declined {:.1f}) vtxbridge={:.1f}x chunks={:.1f} "
            "guest-override={:.1f}x "
            "mirror-tiles={:.1f}x rt-variants={:.1f}x gds-imported={:.1f}x "
            "guest-misses={:.1f}x gpu-indirect={:.1f}x",
            g_out_rt_submits / frames,
            g_tex_bridge_n / frames, g_tex_bridge_bytes / frames / 1e6,
            g_buf_bridge_n / frames, g_buf_bridge_bytes / frames / 1e6,
            g_buf_bridge_declined / frames, g_vtx_bridge_n / frames,
            g_cs_chunk_splits / frames, g_guest_override_n / frames,
            g_guest_mirror_tiles / frames, g_cs_rt_variants_made / frames,
            g_gds_imported_n / frames, g_guest_misses_n / frames,
            g_cs_indirect_n / frames);
  g_guest_override_n = 0;
  g_guest_mirror_tiles = g_cs_rt_variants_made = g_gds_imported_n = 0;
  g_guest_misses_n = g_cs_indirect_n = 0;
  g_buf_bridge_n = g_buf_bridge_bytes = g_buf_bridge_declined = 0;
  g_vtx_bridge_n = 0;
  g_tex_bridge_n = g_tex_bridge_bytes = g_cs_chunk_splits = 0;
  g_out_rt_submits = 0;
  base::String syncs;
  base::FormatTo(syncs, "{:.1f} syncs/frame ({:.1f} submits):", total / frames,
                 g_cs_submit_n / frames);
  g_cs_submit_n = 0;
  for (int i = 0; i < kSyncCount; i++) {
    if (!g_cs_sync_n[i])
      continue;
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", kCsSyncName[i],
                   g_cs_sync_n[i] / frames, g_cs_sync_ns[i] / frames / 1e6);
    g_cs_sync_n[i] = 0;
    g_cs_sync_ns[i] = 0;
  }
  base::FormatTo(syncs, " | by reader:");
  for (const auto& [reader, w] : g_cs_reader_waits)
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", reader.c_str(),
                   w.first / frames, w.second / frames / 1e6);
  g_cs_reader_waits.clear();
  base::FormatTo(syncs, " | writebacks:");
  for (const auto& [reader, w] : g_cs_reader_wbs)
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", reader.c_str(),
                   w.first / frames, w.second / frames / 1e6);
  g_cs_reader_wbs.clear();
  base::FormatTo(syncs, " | submits: wait={:.1f} chunk={:.1f} label={:.1f} heavy={:.1f} cap={:.1f} end={:.1f}",
                 g_submit_why[0] / frames, g_submit_why[1] / frames,
                 g_submit_why[2] / frames, g_submit_why[3] / frames,
                 g_submit_why[4] / frames, g_submit_why[5] / frames);
  for (u64& n : g_submit_why)
    n = 0;
  BASE_LOGI("cssync", "{}", syncs.c_str());
}

// DELTA_GPU_QCHECK=1: before every CS batch is submitted, run an EMPTY
// command list through the same queue and wait for it. The device is lost
// by whatever ran before the first failing submit, so a checkpoint that
// fails names PRIOR queue work (graphics draws, an earlier bridge copy) as
// the fault, and one that succeeds while the batch's own wait fails
// isolates the batch content. The checkpoint shares the graphics queue, so
// it runs after everything already recorded there.
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

// Declared in render/renderer.h for the kernel's crash handler.
bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size) {
  for (const auto& kv : g_cs_ranges) {
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

// DELTA_GPU_MEMSTAT: what the renderer holds in device memory, every 300
// frames.
DELTA_OPTION(bool, kMemStat, "DELTA_GPU_MEMSTAT", false);

void ReportGpuMemory() {
  if (!kMemStat || g_frame.num % 300)
    return;
  u64 dirty = 0, truth = 0, cold = 0;
  for (const auto& [base, e] : g_cs_ranges) {
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
  for (const auto& [base, v] : g_rt_variants) {
    (void)base;
    for (const RTarget& t : v)
      rt_parked += RtByteSize(t);
  }
  for (const auto& [base, d] : g_depths) {
    (void)base;
    depth += u64(d.w) * d.h * 5 * d.layers;
  }
  for (const auto& [base, v] : g_depth_variants) {
    (void)base;
    for (const DepthTarget& d : v)
      depth += u64(d.w) * d.h * 5 * d.layers;
  }
  u64 scratch = 0;
  for (const CsScratch& c : g_cs_scratch)
    scratch += c.cap;
  const u64 tiling = g_truth_bridge_scratch.cap + g_bridge.cap;
  u64 heap_used = 0, heap_budget = 0;
  Device().QueryMemoryBudget(&heap_used, &heap_budget);
  constexpr double kMb = 1024.0 * 1024.0;
  BASE_LOGI("memstat",
            "f{} device heaps {:.0f}/{:.0f}MB | tex {:.0f}MB "
            "scratch {:.0f}MB tiling {:.0f}MB",
            g_frame.num, heap_used / kMb, heap_budget / kMb,
            gpu::render::g_tex_image_bytes / kMb, scratch / kMb, tiling / kMb);
  BASE_LOGI("memstat",
            "f{} cs ranges={} {:.0f}MB (dirty {:.0f} truth {:.0f} cold {:.0f}) "
            "free {:.0f}MB | rt {} {:.0f}MB parked {:.0f}MB | depth {:.0f}MB",
            g_frame.num, g_cs_ranges.size(), g_cs_range_bytes / kMb,
            dirty / kMb, truth / kMb, cold / kMb, g_cs_free_bytes / kMb,
            g_rts.size(), rt / kMb, rt_parked / kMb, depth / kMb);
}

}  // namespace gpu::render
