#include "gpu/ps5/guest_memory_ranges.h"
#include <charconv>
#include <cstdio>
#include <cstring>
#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/guest_memory.h"
#include "gpu/ps5/guest_address.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"

namespace gpu::ps5 {
namespace {
base::Atomic<bool> g_mappings_stale{true};
// A dispatch named an address no parsed mapping holds: parse now.
base::Atomic<bool> g_mappings_forced{false};
base::Atomic<bool> g_remaps_reported{false};
u64 g_mappings_version = 0;
}  // namespace

bool ParseHostMapping(std::string_view line, HostMapping& mapping) {
  if (line.empty())
    return false;
  const char* cursor = line.data();
  const char* end = cursor + line.size();
  const auto skip_space = [&] {
    while (cursor != end && (*cursor == ' ' || *cursor == '\t'))
      ++cursor;
  };
  const auto number = [&](auto& value, int radix) {
    skip_space();
    const auto result = std::from_chars(cursor, end, value, radix);
    cursor = result.ptr;
    return result.ec == std::errc{};
  };
  const auto separator = [&](char expected) {
    if (cursor == end || *cursor != expected)
      return false;
    ++cursor;
    return true;
  };
  HostMapping parsed{};
  if (!number(parsed.begin, 16) || !separator('-') || !number(parsed.end, 16) ||
      !separator(' ') || parsed.begin >= parsed.end)
    return false;
  skip_space();
  if (end - cursor < 5 || cursor[4] != ' ' ||
      (cursor[0] != 'r' && cursor[0] != '-') ||
      (cursor[1] != 'w' && cursor[1] != '-') ||
      (cursor[2] != 'x' && cursor[2] != '-') ||
      (cursor[3] != 'p' && cursor[3] != 's'))
    return false;
  std::memcpy(parsed.perm, cursor, 4);
  cursor += 4;
  if (!number(parsed.offset, 16) || !separator(' ') ||
      !number(parsed.major, 16) || !separator(':') ||
      !number(parsed.minor, 16) || !separator(' ') ||
      !number(parsed.inode, 10) ||
      (cursor != end && *cursor != ' ' && *cursor != '\n'))
    return false;
  mapping = parsed;
  return true;
}

void ReportRemapsToHostMappings() {
  g_remaps_reported.store(true, base::memory_order_release);
}

u64 HostMappingsVersion() {
  return g_mappings_version;
}

void MarkHostMappingsStale() {
  g_mappings_stale.store(true, base::memory_order_release);
}

// Parsing /proc/self/maps for every pointer-chasing dispatch was 6% of the
// submit thread. With remaps reported it is parsed only after one; without,
// once a frame, so a mapping created mid-frame is seen from the next frame, as
// with IsReadableRangeCached.
const base::Vector<HostMapping>& HostMappings() {
  static u64 generation = ~0ull;
  static base::Vector<HostMapping> mappings;
  static const guest::SessionReset reset_mappings(
      [] { guest::ResetResource(mappings); });
  // Remap reports alone parse at most twice a frame: a title that remaps
  // all frame long otherwise has the maps parsed for every dispatch.
  static u64 parsed_generation = ~0ull;
  static u32 parsed_in_frame = 0;
  if (parsed_generation != gpu::MemoryGeneration()) {
    parsed_generation = gpu::MemoryGeneration();
    parsed_in_frame = 0;
  }
  const bool new_frame = generation != gpu::MemoryGeneration() &&
                         !g_remaps_reported.load(base::memory_order_acquire);
  const bool forced =
      g_mappings_forced.exchange(false, base::memory_order_acq_rel);
  if (!new_frame && !forced &&
      (parsed_in_frame >= 2 ||
       !g_mappings_stale.exchange(false, base::memory_order_acq_rel)))
    return mappings;
  g_mappings_stale.store(false, base::memory_order_release);
  parsed_in_frame++;
  generation = gpu::MemoryGeneration();
  g_mappings_version++;
  mappings.clear();
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps)
    return mappings;
  char line[1024];
  while (std::fgets(line, sizeof(line), maps)) {
    HostMapping m{};
    if (ParseHostMapping(line, m) && m.perm[0] == 'r')
      mappings.push_back(m);
  }
  std::fclose(maps);
  return mappings;
}

bool IsGuestWritable(u64 base, u64 bytes) {
  u64 at = base;
  const u64 end = base + bytes;
  for (const HostMapping& m : HostMappings()) {
    if (m.end <= at || m.begin > at)
      continue;
    if (m.perm[1] != 'w' && !(m.inode && m.perm[3] == 's'))
      return false;
    at = m.end;
    if (at >= end)
      return true;
  }
  return at >= end;
}

base::Vector<render::GuestMemoryRange> GuestMemoryRanges(
    const base::Vector<u64>& addresses) {
  base::Vector<render::GuestMemoryRange> result;
#ifdef OS_LINUX
  base::Vector<NotedPools::Range> pools;
  {
    auto& noted = GpuPools();
    base::LockGuard<base::Mutex> lock(noted.lock);
    pools.assign(noted.ranges, noted.ranges + noted.count.load());
  }
  // An address the shader names outside every known mapping means the
  // mappings changed without a report: parse them again.
  const auto known = [](u64 address) {
    for (const HostMapping& m : HostMappings())
      if (address >= m.begin && address < m.end)
        return true;
    return false;
  };
  // Once per page and frame: a pointer into nothing stays unmapped, and
  // parsing again for it on every dispatch was 13% of the submit thread.
  static base::HashSet<u64> unmapped;
  static const guest::SessionReset reset_unmapped(
      [] { guest::ResetResource(unmapped); });
  static u64 unmapped_generation = ~0ull;
  if (unmapped_generation != gpu::MemoryGeneration()) {
    unmapped.clear();
    unmapped_generation = gpu::MemoryGeneration();
  }
  bool stale = false;
  for (u64 address : addresses)
    if (!known(address) && unmapped.insert(address & ~u64(0xFFF)).second)
      stale = true;
  if (stale)
    g_mappings_forced.store(true, base::memory_order_release);
  for (const HostMapping& m : HostMappings()) {
    const unsigned long long begin = m.begin, end = m.end, offset = m.offset,
                             inode = m.inode;
    const unsigned major = m.major, minor = m.minor;
    const char* perm = m.perm;
    // Direct memory is mapped read-write and shared; a read-only view of it
    // here is the write tracker's protection, not the guest's.
    const bool writable = perm[1] == 'w' || (inode && perm[3] == 's');
    u64 identity = inode;
    if (identity) {
      for (u64 word : {u64(major), u64(minor), u64(offset - begin)})
        identity = (identity ^ word) * 1099511628211ull;
      if (!identity)
        identity = 1;
    }
    const bool explicit_address = base::AnyOf(
        addresses.begin(), addresses.end(),
        [&](u64 address) { return address >= begin && address < end; });
    const auto add_range = [&](u64 lo, u64 hi) {
      const u64 backing = inode
                              ? identity
                              : host_memory::MemoryMappingIdentity(
                                    reinterpret_cast<const void*>(lo), hi - lo);
      // Separate file-backed and tracked anonymous identity domains, and
      // invalidate imports if the mapping's write permission changes.
      const u64 key =
          backing ? ((backing ^ (inode ? 1ull : 2ull)) * 1099511628211ull ^
                     u64(writable))
                  : 0;
      result.push_back(
          {lo, hi - lo, key ? key : (backing ? 1ull : 0ull), writable});
    };
    if (explicit_address) {
      add_range(begin, end);
      continue;
    }
    for (const auto& pool : pools) {
      const u64 lo = base::Max<u64>(begin, pool.base),
                hi = base::Min<u64>(end, pool.end);
      if (lo < hi)
        add_range(lo, hi);
    }
  }
  base::Sort(result.begin(), result.end(),
             [](auto& a, auto& b) { return a.base < b.base; });
  base::Vector<render::GuestMemoryRange> unique;
  for (auto range : result) {
    if (!unique.empty()) {
      const u64 prior_end = unique.back().base + unique.back().size;
      if (prior_end >= range.base && unique.back().identity == range.identity &&
          unique.back().writable == range.writable) {
        unique.back().size =
            base::Max(prior_end, range.base + range.size) - unique.back().base;
        continue;
      }
      if (prior_end >= range.base + range.size)
        continue;
      if (prior_end > range.base) {
        range.size -= prior_end - range.base;
        range.base = prior_end;
      }
    }
    unique.push_back(range);
  }
  return unique;
#else
  return result;
#endif
}

namespace {
const guest::SessionReset g_session_reset([] {
  g_mappings_stale.store(true);
  guest::ResetResource(g_mappings_forced);
  guest::ResetResource(g_remaps_reported);
  guest::ResetResource(g_mappings_version);
});
}  // namespace

}  // namespace gpu::ps5
