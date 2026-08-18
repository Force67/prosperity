/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <sys/syscall.h>
#include "base/arch.h"
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <vector>

namespace gpu {

inline bool IsReadableMapping(u64 address, u64 bytes) {
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps)
    return false;

  const u64 end = address + bytes;
  u64 cursor = address;
  char line[512];
  while (cursor < end && std::fgets(line, sizeof(line), maps)) {
    unsigned long long begin, mapping_end;
    char permissions[5] = {};
    if (std::sscanf(line, "%llx-%llx %4s", &begin, &mapping_end, permissions) !=
            3 ||
        mapping_end <= cursor)
      continue;
    if (begin > cursor || permissions[0] != 'r')
      break;
    cursor = std::min<u64>(end, mapping_end);
  }
  std::fclose(maps);
  return cursor == end;
}

inline bool IsReadableRange(u64 address, u64 bytes) {
  if (!bytes || address > std::numeric_limits<u64>::max() - bytes)
    return false;
  static const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return false;
  // getpid() is a real syscall on Linux and this asks the same answer every
  // time; the probe below is already three of them.
  static const pid_t self = getpid();

  const u64 page = static_cast<u64>(page_size);
  const u64 end = address + bytes;

  // Read-probe the first and last page. This is what catches a range that is
  // mapped but not readable, which mincore below cannot see.
  unsigned char probe[2];
  iovec remote[2];
  size_t count = 0;
  remote[count++] = {reinterpret_cast<void*>(address), 1};
  const u64 first_page = address & ~(page - 1);
  const u64 last_page = (end - 1) & ~(page - 1);
  if (last_page > first_page)
    remote[count++] = {reinterpret_cast<void*>(last_page), 1};
  iovec local{probe, count};
  ssize_t read = -1;
  for (u32 attempt = 0; attempt < 2 && read != static_cast<ssize_t>(count);
       attempt++) {
    do {
      read = syscall(SYS_process_vm_readv, self, &local, 1, remote, count, 0);
    } while (read < 0 && errno == EINTR);
  }
  if (read != static_cast<ssize_t>(count))
    return IsReadableMapping(address, bytes);

  // Everything in between: mincore fails with ENOMEM when any page of the span
  // has no mapping, which is the same question the per-page walk was asking.
  // Only its success matters, not the residency bytes it writes.
  const size_t pages = static_cast<size_t>((last_page - first_page) / page + 1);
  static thread_local std::vector<unsigned char> residency;
  if (residency.size() < pages)
    residency.resize(pages);
  if (mincore(reinterpret_cast<void*>(first_page), pages * page,
              residency.data()) != 0)
    return IsReadableMapping(address, bytes);
  return true;
}

// The same answer, remembered until the generation advances.
//
// The probe above costs three syscalls, and a draw asks it about the same
// handful of ranges (its shader code, its index buffer, its descriptor tables)
// every single time -- at 92 draws a frame that was most of the per-draw cost.
// The generation is advanced once per frame, so a mapping the guest tears down
// mid-frame can still be reported readable for the rest of that frame; the
// cbuffer path in vk_draw_recomp already makes exactly that bargain. Use the
// uncached form for anything that must see a teardown immediately.
inline u64& MemoryGeneration() {
  static u64 gen = 1;
  return gen;
}

inline void NextMemoryGeneration() {
  MemoryGeneration()++;
}

inline bool IsReadableRangeCached(u64 address, u64 bytes) {
  struct Entry {
    u64 bytes = 0;
    u64 generation = 0;
    bool readable = false;
  };
  // Per thread, so the command processor and the frame loop never share it and
  // no lock is needed.
  static thread_local std::unordered_map<u64, Entry> cache;
  static thread_local u64 seen_generation = 0;
  const u64 gen = MemoryGeneration();
  if (seen_generation != gen) {
    seen_generation = gen;
    cache.clear();
  }
  auto it = cache.find(address);
  if (it != cache.end() && it->second.bytes == bytes)
    return it->second.readable;
  const bool readable = IsReadableRange(address, bytes);
  cache[address] = {bytes, gen, readable};
  return readable;
}

}  // namespace gpu
