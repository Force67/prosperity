/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The compute range table and its indexes: which ranges are dirty and which
// cover a given block of guest memory.

#include "gpu/render/compute_internal.h"

#include "base/algorithm.h"
#include "base/math/value_bounds.h"
#include "gpu/guest_memory.h"
#include "gpu/guest_page_table.h"
#include "gpu/render/texture_cache.h"
#include "gpu/write_tracker.h"
#include "guest/session.h"

namespace gpu::render::cs {

base::HashMap<u64, CsRange> g_cs_ranges;
decltype(g_cs_ranges)::iterator g_cs_evict_next;
u64 g_cs_dirty_stamp = 0;

// Oldest GPU write first: the order overlapping ranges must be written back in.
void SortByDirtyOrder(base::Vector<u64>& bases) {
  if (bases.size() < 2)
    return;
  thread_local base::Vector<base::Pair<u64, u64>> keyed;
  keyed.clear();
  for (u64 b : bases) {
    const auto it = g_cs_ranges.find(b);
    keyed.emplace_back(it == g_cs_ranges.end() ? 0 : it->second.dirty_stamp, b);
  }
  base::Sort(keyed.begin(), keyed.end());
  for (size_t i = 0; i < keyed.size(); i++)
    bases[i] = keyed[i].second;
}
// Ranges MarkPending flagged, so retiring a batch visits those instead of
// every range. Map nodes do not move; CsRangeDestroy (which precedes every
// erase) drops its entry. May hold ranges no longer pending, or twice.
base::Vector<CsRange*> g_cs_pending;
u64 g_cs_range_bytes = 0;
base::HashMap<u64, base::Vector<u64>> g_cs_dirty_pages;
// How many dirty ranges touch each 64 KiB of guest memory lives in the guest
// page table: a zero answers "nothing dirty here" without a single hash map
// lookup, and without the false alarms of a hashed table. Atomic because the
// command processor's walk reads it without owning the renderer
// (CsRangeMaybeDirty); only the owner writes it.
void NoteDirtyFilter(u64 base, u64 end, int delta) {
  GuestPages().AddCsDirty(base, end, delta);
}

bool DirtyFilterClear(u64 base, u64 end) {
  return GuestPages().CsClear(base, end);
}
// The same bases, once each: what the readback staging walks instead of every
// range (thousands of them against a few dozen dirty ones).
// Dirty range base -> end of the footprint its index entries cover.
base::HashMap<u64, u64> g_cs_dirty_bases;
// Where that footprint starts when it is not the range base: a merged range
// whose dispatches wrote only one child. UC2's video decoder merges a 2 MiB
// output into a 16 MiB unbounded window that also holds its descriptor tables,
// and indexing the whole window made every 4-byte table read write 17 MiB back.
base::HashMap<u64, u64> g_cs_dirty_lo;

u64 DirtyLo(u64 base) {
  auto it = g_cs_dirty_lo.find(base);
  return it == g_cs_dirty_lo.end() ? base : it->second;
}
// Bumped whenever compute results land in guest memory (writeback or an
// executed batch of importing dispatches). The draw path's per-frame staging
// caches stamp entries with this and re-copy when it has moved: a cached
// window of guest bytes is only as fresh as the last compute visibility point.
u64 g_cs_writeback_gen = 1;

u64 RangeEnd(u64 base, u64 bytes) {
  return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
}

// Index [lo, end) of the range at `base` as dirty, growing what it covers.
void IndexDirtySpan(u64 base, u64 lo, u64 end) {
  BumpTextureEpoch();
  if (end <= lo)
    return;
  u64 old_lo = lo, old_end = 0;
  if (auto found = g_cs_dirty_bases.find(base);
      found != g_cs_dirty_bases.end()) {
    old_lo = DirtyLo(base);
    old_end = found->second;
    if (lo >= old_lo && end <= old_end)
      return;
    NoteDirtyFilter(old_lo, old_end, -1);
  }
  const u64 new_lo = old_end ? base::Min(lo, old_lo) : lo;
  const u64 new_end = base::Max(end, old_end);
  g_cs_dirty_bases[base] = new_end;
  if (new_lo == base)
    g_cs_dirty_lo.erase(base);
  else
    g_cs_dirty_lo[base] = new_lo;
  NoteDirtyFilter(new_lo, new_end, 1);
  for (u64 page = new_lo >> kCsDirtyPageShift;
       page <= (new_end - 1) >> kCsDirtyPageShift; page++) {
    const u64 page_lo = page << kCsDirtyPageShift;
    const u64 page_end = page_lo + (u64(1) << kCsDirtyPageShift);
    if (old_end && page_lo < old_end && old_lo < page_end)
      continue;  // already listed
    g_cs_dirty_pages[page].push_back(base);
  }
}

void IndexDirtyRange(u64 base, u64 bytes) {
  if (bytes)
    IndexDirtySpan(base, base, RangeEnd(base, bytes));
}

void UnindexDirtyRange(u64 base, u64 bytes) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  u64 end = RangeEnd(base, bytes);
  if (auto found = g_cs_dirty_bases.find(base);
      found != g_cs_dirty_bases.end()) {
    NoteDirtyFilter(DirtyLo(base), found->second, -1);
    end = base::Max(end, found->second);
    g_cs_dirty_bases.erase(found);
    g_cs_dirty_lo.erase(base);
  }
  for (u64 page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found == g_cs_dirty_pages.end())
      continue;
    auto& bases = found->second;
    base::EraseIf(bases, [&](u64 b) { return b == base; });
    if (bases.empty())
      g_cs_dirty_pages.erase(found);
  }
}

base::HashMap<u64, base::Vector<u64>> g_cs_range_blocks;

void IndexCsRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++)
    g_cs_range_blocks[b].push_back(base);
}

void UnindexCsRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++) {
    auto found = g_cs_range_blocks.find(b);
    if (found == g_cs_range_blocks.end())
      continue;
    base::EraseIf(found->second, [base](u64 b) { return b == base; });
    if (found->second.empty())
      g_cs_range_blocks.erase(found);
  }
}

base::Vector<u64> DirtyRangesOverlapping(u64 base,
                                         u64 bytes,
                                         u64 exclude) {
  base::Vector<u64> candidates;
  if (!bytes)
    return candidates;
  const u64 end = RangeEnd(base, bytes);
  if (DirtyFilterClear(base, end))
    return candidates;
  // Walking the QUERY's pages costs a lookup per 64 KB of it, and a texture is
  // hundreds of pages while the whole dirty index is a handful of entries.
  // Whichever side is smaller gives the same answer, because the filter below
  // rechecks every candidate's overlap exactly, so a superset is safe here.
  const u64 first_page = base >> kCsDirtyPageShift;
  const u64 last_page = (end - 1) >> kCsDirtyPageShift;
  if (last_page - first_page + 1 > g_cs_dirty_pages.size()) {
    for (const auto& [page, bases] : g_cs_dirty_pages)
      candidates.insert(candidates.end(), bases.begin(), bases.end());
  } else {
    for (u64 page = first_page; page <= last_page; page++) {
      auto found = g_cs_dirty_pages.find(page);
      if (found != g_cs_dirty_pages.end())
        candidates.insert(candidates.end(), found->second.begin(),
                          found->second.end());
    }
  }
  base::Sort(candidates.begin(), candidates.end());
  candidates.erase(base::Unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      base::RemoveIf(candidates.begin(), candidates.end(),
                     [&](u64 other) {
                       if (other == exclude)
                         return true;
                       auto indexed = g_cs_dirty_bases.find(other);
                       if (indexed == g_cs_dirty_bases.end() ||
                           DirtyLo(other) >= end || base >= indexed->second)
                         return true;
                       auto found = g_cs_ranges.find(other);
                       return found == g_cs_ranges.end() ||
                              !found->second.gpu_dirty;
                     }),
      candidates.end());
  return candidates;
}

}  // namespace gpu::render::cs

namespace gpu::render {

void CsForgetGuestRange(u64 base, u64 bytes) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  base::Vector<u64> bases;
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++)
    if (auto found = g_cs_range_blocks.find(b);
        found != g_cs_range_blocks.end())
      bases.insert(bases.end(), found->second.begin(), found->second.end());
  for (u64 range_base : bases) {
    auto found = g_cs_ranges.find(range_base);
    if (found == g_cs_ranges.end())
      continue;
    CsRange& e = found->second;
    if (range_base >= end || base >= range_base + e.guest_bytes || e.gpu_dirty)
      continue;
    e.wt_clean = false;
    e.last_validated_frame = -1;
    e.hash = 0;
    e.mirror_current = false;
    e.rt_sourced = false;
  }
}

bool CsTracksGuestWrites() {
  return kCsTrack && GuestWriteTracker().enabled();
}

void CsNoteGuestWrites(u64 first, u64 end) {
  // Untracked, the tracker also reports our own writebacks wherever a draw
  // cache armed the pages, and acting on those costs more than it saves.
  if (g_cs_range_blocks.empty() || first >= end)
    return;
  thread_local base::Vector<u64> bases;
  bases.clear();
  // A remapped reservation can span gigabytes: walk the smaller side.
  const u64 first_block = first >> kCsRangeBlockShift;
  const u64 last_block = (end - 1) >> kCsRangeBlockShift;
  if (last_block - first_block + 1 > g_cs_range_blocks.size()) {
    for (const auto& [block, list] : g_cs_range_blocks)
      if (block >= first_block && block <= last_block)
        bases.insert(bases.end(), list.begin(), list.end());
  } else {
    for (u64 b = first_block; b <= last_block; b++)
      if (auto found = g_cs_range_blocks.find(b);
          found != g_cs_range_blocks.end())
        bases.insert(bases.end(), found->second.begin(), found->second.end());
  }
  for (u64 range_base : bases) {
    auto found = g_cs_ranges.find(range_base);
    if (found == g_cs_ranges.end())
      continue;
    CsRange& e = found->second;
    const u64 range_end = RangeEnd(range_base, e.guest_bytes);
    if (range_base >= end || first >= range_end)
      continue;
    // The CPU wrote it: the staging no longer holds what guest memory does,
    // whatever this frame already checked.
    e.wt_clean = false;
    if (!e.gpu_dirty)
      e.last_validated_frame = -1;
    if (!CsTracksGuestWrites())
      continue;
    if (!e.gpu_dirty) {
      // Hash it again at the next use rather than trusting this frame's
      // check: the write may be our own writeback, which leaves it current.
      e.last_validated_frame = -1;
      continue;
    }
    const u64 lo = base::Max(first, range_base), hi = base::Min(end, range_end);
    if (!e.cpu_writes.empty() && e.cpu_writes.back().second >= lo &&
        e.cpu_writes.back().first <= hi) {
      auto& last = e.cpu_writes.back();
      last = {base::Min(last.first, lo), base::Max(last.second, hi)};
    } else {
      e.cpu_writes.emplace_back(lo, hi);
    }
  }
}

u64 CsWritebackGeneration() {
  return g_cs_writeback_gen;
}

bool CsRangeMaybeDirty(u64 base, u64 bytes) {
  return bytes && !DirtyFilterClear(base, RangeEnd(base, bytes));
}

bool CsRangeDirtyOverlapping(u64 base, u64 bytes) {
  return AnyDirtyOverlapping(base, bytes, [](u64) { return true; });
}

}  // namespace gpu::render
