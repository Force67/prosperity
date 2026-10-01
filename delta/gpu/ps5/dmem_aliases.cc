#include "gpu/ps5/dmem_aliases.h"

#include <cstring>

#include "base/algorithm.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "gpu/ps5/guest_memory_ranges.h"
#include "gpu/write_tracker.h"

namespace gpu::ps5 {
namespace {
using Range = base::Pair<u64, u64>;  // [first, end) of guest VA

// VA ranges whose backing is also mapped somewhere else, sorted and merged.
base::Vector<Range> g_aliased;
u64 g_aliased_version = ~0ull;

struct Piece {
  u64 file, offset, offset_end, va;
};

// File-backed mappings, with the pieces the write tracker's own protection
// splits a mapping into joined again, so they only change on a real remap.
base::Vector<Piece> FilePieces() {
  base::Vector<Piece> pieces;
  for (const HostMapping& m : HostMappings()) {
    if (!m.inode)
      continue;
    const u64 file = (u64(m.major) << 40) ^ (u64(m.minor) << 32) ^ m.inode;
    const u64 bytes = m.end - m.begin;
    if (!pieces.empty()) {
      Piece& last = pieces.back();
      if (last.file == file && last.va + (last.offset_end - last.offset) ==
                                   m.begin &&
          last.offset_end == m.offset) {
        last.offset_end += bytes;
        continue;
      }
    }
    pieces.push_back({file, m.offset, m.offset + bytes, m.begin});
  }
  return pieces;
}

base::Vector<Range> FindAliased(base::Vector<Piece> pieces) {
  base::Sort(pieces.begin(), pieces.end(), [](const Piece& a, const Piece& b) {
    return a.file != b.file ? a.file < b.file : a.offset < b.offset;
  });
  base::Vector<Range> aliased;
  for (size_t i = 0; i < pieces.size(); i++)
    for (size_t j = i + 1; j < pieces.size() && pieces[j].file == pieces[i].file &&
                           pieces[j].offset < pieces[i].offset_end;
         j++) {
      const u64 lo = pieces[j].offset;
      const u64 hi = base::Min(pieces[i].offset_end, pieces[j].offset_end);
      for (const Piece* p : {&pieces[i], &pieces[j]})
        aliased.push_back({p->va + (lo - p->offset), p->va + (hi - p->offset)});
    }
  base::Sort(aliased.begin(), aliased.end(),
             [](const Range& a, const Range& b) { return a.first < b.first; });
  base::Vector<Range> merged;
  for (const Range& r : aliased) {
    if (!merged.empty() && r.first <= merged.back().second)
      merged.back().second = base::Max(merged.back().second, r.second);
    else
      merged.push_back(r);
  }
  return merged;
}

bool Overlaps(const base::Vector<Range>& set, u64 first, u64 end) {
  const Range* it =
      base::UpperBound(set.begin(), set.end(), first,
                       [](u64 v, const Range& r) { return v < r.first; });
  if (it != set.begin() && (it - 1)->second > first)
    return true;
  return it != set.end() && it->first < end;
}

// Newly aliased memory may have been written through the new mapping already:
// report it, so whatever the tracker vouched for is checked again.
void Refresh() {
  HostMappings();  // parses again if a remap made them stale
  if (g_aliased_version == HostMappingsVersion())
    return;
  g_aliased_version = HostMappingsVersion();
  // Most remaps leave file-backed memory as it was: nothing to redo then.
  static base::Vector<Piece> files;
  base::Vector<Piece> now_files = FilePieces();
  if (now_files.size() == files.size() &&
      !std::memcmp(now_files.data(), files.data(),
                   files.size() * sizeof(Piece)))
    return;
  files = now_files;
  base::Vector<Range> now = FindAliased(base::move(now_files));
  for (const Range& r : now) {
    const Range* known = base::LowerBound(
        g_aliased.begin(), g_aliased.end(), r.first,
        [](const Range& a, u64 v) { return a.first < v; });
    if (known == g_aliased.end() || known->first != r.first ||
        known->second != r.second)
      GuestWriteTracker().NoteWrite(r.first, r.second - r.first);
  }
  g_aliased = base::move(now);
}

bool ArmFilter(u64 first, u64 end) {
  Refresh();
  return !Overlaps(g_aliased, first, end);
}

void OnRemap(u64, u64) {
  MarkHostMappingsStale();
}
}  // namespace

void InstallWriteTrackerPolicy() {
  WriteTracker::Policy policy;
  policy.arm_filter = &ArmFilter;
  policy.on_remap = &OnRemap;
  policy.before_collect = &Refresh;
  GuestWriteTracker().SetPolicy(policy);
  ReportRemapsToHostMappings();
}
}  // namespace gpu::ps5
