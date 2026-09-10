#include "gpu/ps5/guest_memory_ranges.h"
#include <algorithm>
#include <cstdio>
#include "gpu/ps5/guest_address.h"
#include "utl/mem.h"

namespace gpu::ps5 {
std::vector<rhi::GuestMemoryRange> GuestMemoryRanges(
    const std::vector<u64>& addresses) {
  std::vector<rhi::GuestMemoryRange> result;
#ifdef OS_LINUX
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps)
    return result;
  std::vector<NotedPools::Range> pools;
  {
    auto& noted = GpuPools();
    std::lock_guard<std::mutex> lock(noted.lock);
    pools.assign(noted.ranges, noted.ranges + noted.count.load());
  }
  char line[1024];
  while (std::fgets(line, sizeof(line), maps)) {
    unsigned long long begin, end, offset, inode;
    unsigned major, minor;
    char perm[5];
    if (std::sscanf(line, "%llx-%llx %4s %llx %x:%x %llu", &begin, &end, perm,
                    &offset, &major, &minor, &inode) != 7 ||
        perm[0] != 'r')
      continue;
    u64 identity = inode;
    if (identity) {
      for (u64 word : {u64(major), u64(minor), u64(offset - begin)})
        identity = (identity ^ word) * 1099511628211ull;
      if (!identity)
        identity = 1;
    }
    const bool explicit_address = std::any_of(
        addresses.begin(), addresses.end(),
        [&](u64 address) { return address >= begin && address < end; });
    const auto add_range = [&](u64 lo, u64 hi) {
      const u64 backing = inode ? identity : utl::memoryMappingIdentity(
          reinterpret_cast<const void*>(lo), hi - lo);
      // Separate file-backed and tracked anonymous identity domains, and
      // invalidate imports if the mapping's write permission changes.
      const u64 key = backing ? ((backing ^ (inode ? 1ull : 2ull)) *
                                1099511628211ull ^ u64(perm[1])) : 0;
      result.push_back({lo, hi - lo, key ? key : (backing ? 1ull : 0ull),
                        perm[1] == 'w'});
    };
    if (explicit_address) {
      add_range(begin, end);
      continue;
    }
    for (const auto& pool : pools) {
      const u64 lo = std::max<u64>(begin, pool.base),
                hi = std::min<u64>(end, pool.end);
      if (lo < hi)
        add_range(lo, hi);
    }
  }
  std::fclose(maps);
  std::sort(result.begin(), result.end(),
            [](auto& a, auto& b) { return a.base < b.base; });
  std::vector<rhi::GuestMemoryRange> unique;
  for (auto range : result) {
    if (!unique.empty()) {
      const u64 prior_end = unique.back().base + unique.back().size;
      if (prior_end >= range.base && unique.back().identity == range.identity &&
          unique.back().writable == range.writable) {
        unique.back().size =
            std::max(prior_end, range.base + range.size) - unique.back().base;
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
}  // namespace gpu::ps5
