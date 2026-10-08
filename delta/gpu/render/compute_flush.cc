/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Writing dispatch output back to guest memory, and memory fills over
// ranges a dispatch wrote.

#include "gpu/render/compute_internal.h"

#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/compute_writeback.h"
#include "gpu/render/frame.h"
#include "gpu/render/texture_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "profile/profile.h"

namespace gpu::render::cs {

// Write one dirty range back to guest memory (retile for images) and re-stamp
// its hash so the next validation sees guest == buffer.
bool CsRangeFlushOne(u64 base, CsRange& e) {
  if (kCsWbAudit) {
    // Counters, not a sample: the first N flushes are all startup, and the
    // question ("does a tiled-image range ever reach the retile?") is about
    // the steady state.
    static u64 n = 0, dirty_n = 0, img_n = 0, img_dirty_n = 0;
    n++;
    dirty_n += e.gpu_dirty;
    img_n += e.image_staging;
    img_dirty_n += e.gpu_dirty && e.image_staging;
    if ((n % 20000) == 0)
      BASE_LOGI("cswb",
                "flushes={} dirty={} image={} image+dirty={} "
                "staged_as_image={}",
                (unsigned long long)n, (unsigned long long)dirty_n,
                (unsigned long long)img_n, (unsigned long long)img_dirty_n,
                (unsigned long long)g_cs_image_staged);
  }
  // A clean range writes nothing, so texture answers memoized on the epoch
  // still hold. Callers sweep every overlapping sibling through here.
  if (!e.gpu_dirty)
    return true;
  DELTA_ZONE("gpu.cs_writeback");
  BumpTextureEpoch();
  if (e.imported) {  // the dispatch wrote straight into guest memory
    if (!CsBatchWaitRange(e, kSyncImported))
      return false;
    NoteGuestWrite(base, e.guest_bytes);
    CsForgetGuestRange(base, e.guest_bytes);
    e.gpu_dirty = false;
    e.cpu_writes.clear();
    return true;
  }
  if (MirrorTakeRange(base, e)) {
    UploadCsRangeToRt(base, e);
    InvalidateTexRange(base, e.guest_bytes);
    UnindexDirtyRange(base, e.guest_bytes);
    g_cs_writeback_gen++;
    // Our own write is collected while the range is still dirty, so it is not
    // mistaken for a CPU write that would restage the range next frame.
    CollectGuestWrites();
    CsForgetGuestRange(base, e.guest_bytes);
    e.gpu_dirty = false;
    e.cpu_writes.clear();
    // Guest memory is the dispatch's now, the shadow the stage-in's.
    e.shadow_valid = false;
    e.last_validated_frame = g_frame.num;
    // The buffer is what guest memory will hold once the copy lands: current
    // until the CPU writes. Owned pages are the mirror's, not trackable.
    if (!MirrorOwned(base & ~(kMirrorBlock - 1)))
      e.wt_clean = GuestWriteTracker().Arm(base, e.guest_bytes);
    g_wb_why["mirror/img"]++;
    return true;
  }
  if (CsTracksGuestWrites())
    CollectGuestWritesForSubmission();
  // Results live in VRAM: pull them into the host mirror on the same batch that
  // produced them, so the one fence wait covers the dispatch and the copy.
  // Already-recorded readbacks (CsStageReadbacks) skip this.
  if (e.device_local && !e.mirror_current && !e.readback_pending &&
      (!CanGpuTile(e.res, e) || e.truth))
    CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
  if (e.pending_batch) {
    const u64 tw = NowNs();
    const bool ok = CsBatchWaitRange(e, kSyncWriteback);
    if (CsSyncReporting()) {
      StageStat& st = g_stage_stats[base];
      st.wait++;
      st.wait_ns += NowNs() - tw;
    }
    if (!ok)
      return false;  // results must exist before readback
  }
  if (g_cs_failed)
    return false;
  g_cs_flush_n++;
  if (e.image_staging && kCsWbAudit) {
    // An image range always retiles, with no write-coverage test to report, so
    // "the writeback ran" says nothing about whether the dispatch produced
    // anything. Count what the staging actually holds: all-zero here means the
    // shader stored nothing, which is a different bug from a failed retile.
    const auto* p = static_cast<const u8*>(e.map);
    u64 nz = 0;
    for (u64 i = 0; i < e.res.size; i++)
      if (p[i])
        nz++;
    // Once per destination, not the first 24 overall: a title that uploads
    // hundreds of surfaces spends a flat cap entirely on the first few, and
    // the one surface that stays empty is never among them.
    static base::HashSet<u64> seen;
    static const guest::SessionReset reset_seen(
        [] { guest::ResetResource(seen); });
    if (seen.size() < 4096 && seen.insert(base).second)
      BASE_LOGI("cswb",
                "image base={:#x} {}x{} layers={} tiling={} staged {}/{} "
                "non-zero",
                (unsigned long long)base, e.res.width, e.res.height,
                e.res.layers, e.res.tiling_idx, (unsigned long long)nz,
                (unsigned long long)e.res.size);
  }
  const u64 t_wb = NowNs();
  // A range with a shadow merges the guest's writes word by word below. Any
  // other writeback covers whole pages, so with write tracking the pages the
  // CPU wrote since the dispatch are set aside here and put back after it.
  // Without it, a report can be of our own earlier writeback.
  const u64 linear_n =
      e.guest_bytes ? base::Min<u64>(e.size, e.guest_bytes) : e.size;
  const bool merges = (!e.image_staging || e.truth) && !kCsWbFull &&
                      !e.rt_sourced && e.shadow_valid &&
                      e.shadow.size() >= linear_n;
  struct KeptPages {
    base::Vector<base::Pair<u64, u64>> spans;
    base::Vector<u8> bytes;
    ~KeptPages() {
      u64 off = 0;
      for (const auto& [first, end] : spans) {
        std::memcpy(reinterpret_cast<void*>(first), bytes.data() + off,
                    end - first);
        off += end - first;
      }
    }
  } kept;
  if (!merges && CsTracksGuestWrites()) {
    const u64 footprint_end = base + (e.guest_bytes ? e.guest_bytes : e.size);
    for (const auto& [first, end] : e.cpu_writes) {
      const u64 lo = base::Max(first, base), hi = base::Min(end, footprint_end);
      if (lo >= hi)
        continue;
      kept.spans.emplace_back(lo, hi);
      const u64 at = kept.bytes.size();
      kept.bytes.resize(at + (hi - lo));
      std::memcpy(kept.bytes.data() + at, reinterpret_cast<const void*>(lo),
                  hi - lo);
    }
  }
  for (const auto& [lo, hi] : e.fill_kept) {
    kept.spans.emplace_back(lo, hi);
    const u64 at = kept.bytes.size();
    kept.bytes.resize(at + (hi - lo));
    std::memcpy(kept.bytes.data() + at, reinterpret_cast<const void*>(lo),
                hi - lo);
  }
  e.fill_kept.clear();
  bool adopted = !kept.spans.empty();
  // DELTA_GPU_CS_SKIP_RETILE: an image the GPU refresh below can carry needs no
  // CPU retile into guest memory. Only readers that go through guest memory
  // (the texture cache's own upload, a CP DMA, the guest CPU) lose anything,
  // and for a live image the draw path samples the image instead.
  CsAliasedImage refreshed{};
  const bool gpu_carries =
      kCsSkipRetile && e.image_staging && !e.truth &&
      FindCsAliasedImage(base, refreshed, /*for_write=*/true,
                         /*prefer_depth=*/e.res.dfmt == 4);
  if (e.image_staging && !e.truth && !gpu_carries) {
    const bool converted = ConvertCsRangeImage(e.res, e, false);
    if (!converted && e.device_local && !e.mirror_current) {
      CsCopyStaging(e, e.size, /*to_device=*/false);
      if (!CsBatchWaitRange(e, kSyncWriteback))
        return false;
    }
    if (!converted && (g_cs_failed || !WritebackCsImage(e.res, e.map))) {
      // Count as well as sample: a flat cap of 8 lines cannot tell one
      // recurring bad descriptor apart from every upload in the title failing,
      // and those need opposite responses.
      static u64 failed = 0;
      if (failed++ < 8 || (failed % 4096) == 0)
        BASE_LOGI("gpuvk",
                  "cs image writeback #{} failed base={:#x} {}x{} mips={} "
                  "layers={} tiling={} elem={}/{} (range stays stale)",
                  (unsigned long long)failed, (unsigned long long)base,
                  e.res.width, e.res.height, e.res.mip_levels, e.res.layers,
                  e.res.tiling_idx, e.res.elem_bytes, e.res.stage_elem_bytes);
      return false;
    }
  } else if (!e.image_staging || e.truth) {
    // Never write back more than the guest footprint the range was staged
    // from, and never into memory that is not the guest's: the staged size is
    // the shader's view of the resource and a descriptor with a bogus size
    // would otherwise scribble compute results over whatever follows, which
    // reads as float data turning up in the title's allocator free lists.
    const u64 n =
        e.guest_bytes ? base::Min<u64>(e.size, e.guest_bytes) : e.size;
    if (!gpu::IsReadableRange(base, n)) {
      static int logged = 0;
      if (logged++ < 8)
        BASE_LOGI("gpuvk",
                  "cs writeback out of guest memory base={:#x} +{:#x} "
                  "(dropped)",
                  (unsigned long long)base, (unsigned long long)n);
      return false;
    }
    // Write back only what the DISPATCH changed. A staged range is the shader's
    // whole view of a resource, but a shader writes a part of it, and this
    // title puts its CPU heap in the same direct memory as the buffers it hands
    // to compute, so the bytes in between belong to the allocator. Copying the
    // full range back reverts every CPU write made since stage-in, and a word
    // the CPU was writing WHILE we snapshotted comes back half updated: the
    // 5-valid-bytes-plus-3-stale-bytes free-tree pointer SotC dies on at
    // eboot+0x48ac5 when New Game is confirmed. The shadow copy taken at
    // stage-in says which words the shader touched; untouched words are left
    // alone. The merge is symmetric, so staging, shadow and guest agree again
    // when it returns:
    //   the shader wrote the block  -> guest   <- staging  (publish the result)
    //   the shader left it alone    -> staging <- guest    (adopt the CPU's
    //   word)
    // Guest-sourced linear ranges only: a range staged from a live render
    // target exists precisely so the writeback can publish that image into
    // guest memory, so "the dispatch did not write it" must not stop it there.
    //
    // A block the CPU also changed since stage-in (guest != shadow) keeps the
    // CPU's bytes even where the shader wrote it: the dispatch ran before the
    // CPU write, and a title that reuses a compute buffer's memory for its
    // descriptor tables otherwise gets old dispatch output over them.
    if (merges) {
      auto* src = static_cast<u8*>(e.map);
      u8* shd = e.shadow.data();
      auto* dst = reinterpret_cast<u8*>(base);
      const u64 blocks = n / 64;
      u64 off = blocks * 64, wrote = 0;
      // 64-byte blocks: a block the shader left alone costs one compare, which
      // keeps "the shader wrote a little of a big range" (the common case)
      // no more expensive than the unconditional copy this replaces. Blocks are
      // independent, so the sweep is split across the detiler's pool; the
      // per-lane written counts are summed rather than shared.
      constexpr u64 kMergeBlocksPerChunk = 1024;  // 64 KiB a chunk
      const u32 chunks = static_cast<u32>((blocks + kMergeBlocksPerChunk - 1) /
                                          kMergeBlocksPerChunk);
      base::Vector<ComputeWritebackCounts> counts(chunks);
      gcn::DetileParallelWork(chunks, blocks * 64, [&](u32 c0, u32 c1) {
        for (u32 c = c0; c < c1; c++) {
          const u64 first = u64(c) * kMergeBlocksPerChunk;
          const u64 last = base::Min<u64>(first + kMergeBlocksPerChunk, blocks);
          ComputeWritebackCounts local;
          for (u64 b = first; b < last; b++) {
            const u64 o = b * 64;
            MergeComputeWritebackBlock(dst + o, src + o, shd + o, local);
          }
          counts[c] = local;
        }
      });
      u64 adopted_bytes = 0;
      for (const ComputeWritebackCounts& c : counts) {
        wrote += c.wrote;
        adopted_bytes += c.adopted;
        g_cs_wb_conflicts += c.conflicts;
      }
      for (; off < n; off++) {
        if (dst[off] != shd[off]) {
          src[off] = dst[off];
          shd[off] = dst[off];
          adopted_bytes++;
        } else if (src[off] != shd[off]) {
          dst[off] = src[off];
          shd[off] = src[off];
          wrote++;
        }
      }
      adopted |= adopted_bytes != 0;
      g_cs_wb_bytes_written += wrote;
      g_cs_wb_bytes_total += n;
      if (kCsWbAudit && wrote != n) {
        static int logged = 0;
        if (logged++ < 32)
          BASE_LOGI("cswb",
                    "base={:#x} +{:#x}: shader wrote {} of {} bytes; the other "
                    "{} would have been reverted",
                    (unsigned long long)base, (unsigned long long)n,
                    (unsigned long long)wrote, (unsigned long long)n,
                    (unsigned long long)(n - wrote));
      }
    } else {
      // Full-range writeback (the pre-merge behaviour, kept for A/B). Still
      // measure how much of it the dispatch actually wrote when a shadow is
      // available, so the two modes report the same number and the A/B is a
      // comparison rather than a guess.
      u64 wrote = n;
      if (e.shadow_valid && e.shadow.size() >= n) {
        wrote = 0;
        const auto* src = static_cast<const u8*>(e.map);
        const u8* shd = e.shadow.data();
        for (u64 off = 0; off < n; off += 64) {
          const u64 blk = base::Min<u64>(64, n - off);
          if (std::memcmp(src + off, shd + off, blk) != 0)
            wrote += blk;
        }
        if (kCsWbAudit && wrote != n) {
          static int logged = 0;
          if (logged++ < 32)
            BASE_LOGI("cswb",
                      "base={:#x} +{:#x}: shader wrote {} of {} bytes; the "
                      "other {} ARE being reverted",
                      (unsigned long long)base, (unsigned long long)n,
                      (unsigned long long)wrote, (unsigned long long)n,
                      (unsigned long long)(n - wrote));
        }
      }
      // One write announced up front instead of a fault per armed page.
      host_memory::BeforeHostWrite(reinterpret_cast<void*>(base), n);
      ParallelCopy(reinterpret_cast<void*>(base), e.map, n);
      g_cs_wb_bytes_written += wrote;
      g_cs_wb_bytes_total += n;
    }
  }
  const u64 t_rt = NowNs();
  if (e.image_staging)
    g_out_retile_ns += t_rt - t_wb;
  else
    g_out_buffer_ns += t_rt - t_wb;
  g_out_retile_n += e.image_staging;
  if (CsSyncReporting()) {
    auto& wb = g_cs_reader_wbs[base::String(g_cs_wait_reader) +
                               (e.image_staging ? ":img" : ":buf")];
    wb.first++;
    wb.second += t_rt - t_wb;
  }
  UploadCsRangeToRt(base, e);  // refresh a live RT image aliasing the range
  const u64 t_inv = NowNs();
  g_out_rt_ns += t_inv - t_rt;
  InvalidateTexRange(base, e.guest_bytes);
  UnindexDirtyRange(base, e.guest_bytes);
  // Guest memory just changed under any staged copy of it: retire the draw
  // path's per-frame staging cache entries (they validate against this).
  g_cs_writeback_gen++;
  // Our own writes to the range are reported like the guest's; take them now
  // so the next dispatch does not read them as CPU writes after it.
  if (CsTracksGuestWrites())
    CollectGuestWrites();
  if (!gpu_carries)
    CsForgetGuestRange(base, e.guest_bytes);
  e.gpu_dirty = false;
  e.cpu_writes.clear();
  // The buffer lacks CPU bytes the merge adopted into guest memory: stage it
  // again before anything runs on it. Otherwise both hold the same bytes.
  e.last_validated_frame = adopted ? -1 : g_frame.num;
  g_out_tail_ns += NowNs() - t_inv;
  return true;
}

}  // namespace gpu::render::cs

namespace gpu::render {

// Make guest memory current with every GPU-written compute range. Called
// before anything that consumes guest memory: draws (vertex/texture reads at
// record time), CP DMA copies, and the end of each frame (bounds staleness
// for direct guest CPU readers to one frame). Walks only the dirty ranges;
// cold entries are evicted at frame end.
bool FlushCsWrites(Renderer& renderer, const char* why) {
  DELTA_ZONE("gpu.cs_flush_all");
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  const WaitReaderScope reader(why);
  g_wb_why[base::String("all-call/") + why]++;
  // Record every readback first: one fence wait then covers all of them,
  // instead of one submit+wait per dirty range.
  CsStageReadbacks();
  for (u64 base : DirtyBases(/*ordered=*/true)) {
    auto it = g_cs_ranges.find(base);
    if (it == g_cs_ranges.end() || !it->second.gpu_dirty)
      continue;
    g_wb_why[base::String("all/") + why]++;
    g_wb_why_bytes += it->second.size;
    if (!CsRangeFlushOne(it->first, it->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      // Writeback of this one range failed; it stays dirty. Keep flushing the
      // rest so one bad range cannot hold every other range stale forever.
      all_current = false;
    }
  }
  g_ns_cs_out += NowNs() - t0;
  return all_current;
}

bool FlushCsWritesFrameEnd(Renderer& renderer, bool writeback) {
  DELTA_ZONE("gpu.cs_flush_frame_end");
  ReportGpuMemory();
  const WaitReaderScope reader("frame-end");
  if (g_cs_failed || !SettleUnwaitedWrites()) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  if (writeback) {
    // An image range no live target aliases has no CPU reader this frame: it
    // stays in VRAM, where the next dispatch or draw takes it from, and a
    // guest reader that does turn up gets its writeback then.
    // A truth that aliases a live colour target refreshes the image here,
    // on the frame's own command buffer, and then stays like any other.
    auto stays = [](u64 base, CsRange& e) {
      if (CsLazyKept(base, e))
        return true;
      return kCsImageLazyWb && e.truth && e.gpu_dirty && !e.imported &&
             (e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e));
    };
    CsStageReadbacks(/*all=*/false);
    for (u64 base : DirtyBases(/*ordered=*/true)) {
      auto it = g_cs_ranges.find(base);
      if (it == g_cs_ranges.end() || !it->second.gpu_dirty)
        continue;
      if (stays(it->first, it->second)) {
        g_lazy_kept_n++;
        g_lazy_kept_bytes += it->second.size;
        continue;
      }
      const CsRange& fe = it->second;
      g_wb_why[base::String("frame-end") + (fe.truth           ? "/img"
                                            : fe.image_staging ? "/lin"
                                                               : "/buf")]++;
      g_wb_why_bytes += fe.size;
      if (!CsRangeFlushOne(it->first, it->second)) {
        if (g_cs_failed) {
          renderer.state = nullptr;
          return false;
        }
        all_current = false;
      }
    }
  }
  // DELTA_GPU_CS_AGED: ranges holding dispatch output no reader has asked for
  // in over two frames, i.e. what a blanket writeback would publish and the
  // lazy model does not.
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
    // Spread the cold-range walk over 16 frames. UC2 keeps over 20,000
    // ranges, most still live; walking them all cost several ms per frame.
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
        g_wb_why["idle"]++;
        g_wb_why_bytes += e.size;
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
  // The frame's command buffer may copy out of these ranges (CsSupplyTexture)
  // and is submitted next: whatever the batch still holds goes first.
  if (g_cs_batch_open && (++g_submit_why[5], !CsBatchSubmit())) {
    renderer.state = nullptr;
    return false;
  }
  g_ns_cs_out += NowNs() - t0;
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

// Targeted variant: flush only dirty ranges overlapping [base, base+bytes).
// The per-draw guest readers (texture upload, vertex copy, cbuffer ring) call
// this instead of the full flush: flushing every dirty range at every draw
// re-tiled the whole post chain ~19x/frame.
bool FlushCsWritesRange(Renderer& renderer,
                        u64 base,
                        u64 bytes,
                        const char* why) {
  // GPU-owned bytes an open batch is still to write: submitted, the read
  // faults and waits for them.
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
  if (!g_pending_guest_writes.empty()) {
    const WaitReaderScope reader(why);
    if (!WaitPendingGuestWrites(base, bytes))
      return false;
  }
  // Nothing dirty anywhere: answer without touching the page index, which
  // otherwise allocates a vector, hashes a lookup per page, then sorts and
  // dedups it. That is called once per guest read, and SotC issues 1.2M
  // DRAW_INDEX_INDIRECT packets a run, each flushing before it reads its
  // argument struct, which measured 33 seconds of a 150 s run. The index is
  // maintained on both edges (indexed when a range goes dirty, unindexed when
  // it is flushed), so empty means "no writeback outstanding".
  if (g_cs_dirty_pages.empty())
    return true;
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  // Most calls are per s_load of the descriptor replay and overlap nothing:
  // answer those before the timers and the candidate vector.
  if (!base || !bytes || g_cs_ranges.empty() ||
      (!CsRangeDirtyOverlapping(base, bytes) &&
       !(kCsFlushTrace && base == (u64)kCsFlushTrace)))
    return true;
  ScopeNs flush_timer(&g_ns_cs_flush);
  const u64 t0 = NowNs();
  bool all_current = true;
  // DELTA_GPU_CSFLUSHTRACE=<base>: what the dirty-range index finds for a
  // target a draw is about to sample. A compute result that is not found here
  // never reaches the target's image, and the draw samples black.
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
      // ...and every CS range that overlaps the target at all, found or not.
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
      BASE_LOGI("rtflush", "{} rt {:#x}+{:#x} <- range {:#x}+{:#x} img={} truth={} "
                "{}x{}x{} e={}/{} dfmt={} t={}", why, base, bytes, d, o.guest_bytes,
                (int)o.image_staging, (int)o.truth, o.res.width, o.res.height,
                o.res.layers, o.res.elem_bytes, o.res.stage_elem_bytes,
                o.res.dfmt, o.res.tiling_idx);
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
      BASE_LOGI("csflush", "f{} {} {:#x}+{:#x} {:.2f}ms <-{}",
                (int)g_frame.num, why, base, bytes, (NowNs() - t0) / 1e6,
                line.c_str());
    }
  } flush_log{base, bytes, NowNs(), why, overlapping};
  // Only the ranges this read overlaps are pulled across. Pulling every dirty
  // range along made later flushes wait-free while a blanket flush wrote them
  // all back each frame anyway; now most dirty ranges are never read on the
  // CPU, and reading them all back on every flush cost UC2 ~8% of the walk.
  if (!overlapping.empty()) {
    for (u64 dirty : overlapping) {
      auto found = g_cs_ranges.find(dirty);
      if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
        continue;
      CsRange& e = found->second;
      // The descriptor replay reads SRT tables, which never sit inside a
      // tiled image surface: a truth is not what it is after.
      if (e.truth && !std::strcmp(why, "srt"))
        continue;
      g_wb_why[base::String(why) + (e.truth           ? "/img"
                                    : e.image_staging ? "/lin"
                                                      : "/buf")]++;
      g_wb_why_bytes += e.size;
      // Reuse an eager readback already queued by the producer. Recording
      // another copy here would move the wait to a new submission.
      CsStageReadback(dirty, e, /*all=*/true);
    }
  }
  for (u64 dirty : overlapping) {
    auto found = g_cs_ranges.find(dirty);
    if (found != g_cs_ranges.end() && found->second.truth &&
        !std::strcmp(why, "srt"))
      continue;
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
  g_ns_cs_out += NowNs() - t0;
  return all_current;
}

}  // namespace gpu::render

namespace gpu::render::cs {

// The fill against each dirty range under it: `apply` false only asks.
bool FillOverDirtyRanges(u64 base, u64 bytes, bool apply) {
  const u64 end = base + bytes;
  for (const PendingGuestWrite& w : g_pending_guest_writes)
    if (w.base < end && base < w.end)
      return false;
  if (g_cs_dirty_pages.empty())
    return true;
  for (u64 d : DirtyRangesOverlapping(base, bytes)) {
    auto found = g_cs_ranges.find(d);
    if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
      continue;
    CsRange& e = found->second;
    if (e.imported)
      return false;
    if (d >= base && d + e.guest_bytes <= end) {
      // Every byte it would write back is overwritten.
      if (apply) {
        UnindexDirtyRange(d, e.guest_bytes);
        e.gpu_dirty = false;
        e.cpu_writes.clear();
        e.hash = 0;
        e.last_validated_frame = -1;
        e.mirror_current = false;
        e.shadow_valid = false;
        g_wb_why["fill-covered"]++;
      }
      continue;
    }
    // Output the fill covers in part keeps its own bytes, as it would under a
    // dispatch that only writes: tiled surfaces that share memory are not each
    // other's pixels. Only its writeback, which is older than the fill, keeps
    // the filled bytes, the way it keeps any the CPU wrote since.
    if (apply) {
      u64 lo = base::Max(base, d), hi = base::Min(end, d + e.guest_bytes);
      // Merged, or a range left dirty for many frames piles up one span per
      // fill and its writeback copies them all.
      base::EraseIf(e.fill_kept, [&](const base::Pair<u64, u64>& k) {
        if (k.first > hi || lo > k.second)
          return false;
        lo = base::Min(lo, k.first);
        hi = base::Max(hi, k.second);
        return true;
      });
      e.fill_kept.emplace_back(lo, hi);
      g_wb_why["fill-kept"]++;
    }
  }
  return true;
}

}  // namespace gpu::render::cs

namespace gpu::render {

bool CsFillStaysOnGpu(u64 base, u64 bytes) {
  return FillOverDirtyRanges(base, bytes, false);
}

void ApplyMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value) {
  ScopeNs fill_timer(&g_ns_memory_fill);
  g_memory_fill_bytes += bytes;
  g_memory_fill_n++;
  BumpTextureEpoch();
  if (!bytes)
    return;
  if (!FillOverDirtyRanges(base, bytes, false) ||
      !FillOverDirtyRanges(base, bytes, true))
    FlushCsWritesRange(renderer, base, bytes, "fill");
  if (MirrorFill(base, bytes, value)) {
    CsForgetGuestRange(base, bytes);
    NoteMemoryFill(renderer, base, bytes, value);
    g_cs_writeback_gen++;
    return;
  }
  // Announced first: armed pages open in runs, not with a fault per page.
  host_memory::BeforeHostWrite(reinterpret_cast<void*>(base), bytes);
  // Split over the detile pool: one thread memsetting ~20 MB a frame was the
  // submit thread's largest item.
  constexpr u64 kChunk = 256 * 1024;
  u32* words = reinterpret_cast<u32*>(base);
  if (bytes < kParallelCopyMin) {
    if (!value)
      std::memset(words, 0, bytes);
    else
      base::Fill(words, words + bytes / 4, value);
  } else {
    const u32 chunks = static_cast<u32>((bytes + kChunk - 1) / kChunk);
    gcn::DetileParallelWork(chunks, bytes, [&, value](u32 c0, u32 c1) {
      const u64 off = u64(c0) * kChunk / 4;
      const u64 end = base::Min<u64>(u64(c1) * kChunk, bytes) / 4;
      if (!value)
        std::memset(words + off, 0, (end - off) * sizeof(*words));
      else
        base::Fill(words + off, words + end, value);
    });
  }
  CsForgetGuestRange(base, bytes);
  InvalidateTexRange(base, bytes);
  NoteMemoryFill(renderer, base, bytes, value);
  g_cs_writeback_gen++;
}

void ApplyMemoryFill64(Renderer& renderer, u64 base, u64 bytes, u64 value) {
  ScopeNs fill_timer(&g_ns_memory_fill);
  g_memory_fill_bytes += bytes;
  g_memory_fill_n++;
  BumpTextureEpoch();
  if (!bytes)
    return;
  if (!FillOverDirtyRanges(base, bytes, false) ||
      !FillOverDirtyRanges(base, bytes, true))
    FlushCsWritesRange(renderer, base, bytes, "fill");
  if (u32(value) == u32(value >> 32) && MirrorFill(base, bytes, u32(value))) {
    CsForgetGuestRange(base, bytes);
    NoteMemoryFill64(renderer, base, bytes, value);
    g_cs_writeback_gen++;
    return;
  }
  host_memory::BeforeHostWrite(reinterpret_cast<void*>(base), bytes);
  constexpr u64 kChunk = 256 * 1024;
  auto* words = reinterpret_cast<u64*>(base);
  if (bytes < kParallelCopyMin) {
    if (!value)
      std::memset(words, 0, bytes);
    else
      base::Fill(words, words + bytes / 8, value);
  } else {
    const u32 chunks = static_cast<u32>((bytes + kChunk - 1) / kChunk);
    gcn::DetileParallelWork(chunks, bytes, [&, value](u32 c0, u32 c1) {
      const u64 off = u64(c0) * kChunk / 8;
      const u64 end = base::Min<u64>(u64(c1) * kChunk, bytes) / 8;
      if (!value)
        std::memset(words + off, 0, (end - off) * sizeof(*words));
      else
        base::Fill(words + off, words + end, value);
    });
  }
  CsForgetGuestRange(base, bytes);
  InvalidateTexRange(base, bytes);
  NoteMemoryFill64(renderer, base, bytes, value);
  g_cs_writeback_gen++;
}

}  // namespace gpu::render
