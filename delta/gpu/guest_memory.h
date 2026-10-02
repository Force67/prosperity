/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <limits>
#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "gpu/guest_page_table.h"

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
    cursor = base::Min<u64>(end, mapping_end);
  }
  std::fclose(maps);
  return cursor == end;
}

inline bool IsReadableRange(u64 address, u64 bytes) {
  if (!bytes || address > std::numeric_limits<u64>::max() - bytes)
    return false;
  static const long kPageSize = sysconf(_SC_PAGESIZE);
  if (kPageSize <= 0)
    return false;
  // getpid() is a real syscall on Linux and this asks the same answer every
  // time; the probe below is already three of them.
  static const pid_t kSelf = getpid();

  const u64 page = static_cast<u64>(kPageSize);
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
      read = syscall(SYS_process_vm_readv, kSelf, &local, 1, remote, count, 0);
    } while (read < 0 && errno == EINTR);
  }
  // A short read or EFAULT is the answer, not a reason to ask again: parsing
  // the map costs milliseconds, and a replay chasing garbage pointers asks
  // often.
  if (read != static_cast<ssize_t>(count))
    return read < 0 && errno != EFAULT && IsReadableMapping(address, bytes);

  // Everything in between: mincore fails with ENOMEM when any page of the span
  // has no mapping, which is the same question the per-page walk was asking.
  // Only its success matters, not the residency bytes it writes.
  const size_t pages = static_cast<size_t>((last_page - first_page) / page + 1);
  static thread_local base::Vector<unsigned char> residency;
  if (residency.size() < pages)
    residency.resize(pages);
  if (mincore(reinterpret_cast<void*>(first_page), pages * page,
              residency.data()) != 0)
    return errno != ENOMEM && IsReadableMapping(address, bytes);
  return true;
}

// The same answer, remembered until the generation advances.
//
// The probe costs three syscalls and a draw asks it about the same handful of
// ranges (shader code, index buffer, descriptor tables) every time; at 92 draws
// a frame that was most of the per-draw cost. The generation advances once per
// frame, so a mapping the guest tears down mid-frame can still read as
// readable for the rest of that frame (the cbuffer path in vk_draw_recomp
// already makes that bargain). Use the uncached form when a teardown must be
// seen immediately.
inline u64& MemoryGeneration() {
  static u64 gen = 1;
  return gen;
}

inline void NextMemoryGeneration() {
  MemoryGeneration()++;
}

// The guest mapped or unmapped something since the generation last moved.
inline base::Atomic<bool>& MemoryRemapped() {
  static base::Atomic<bool> remapped{true};
  return remapped;
}

inline bool IsReadableRangeCached(u64 address, u64 bytes) {
  constexpr u64 kPageShift = GuestPageTable::kPageShift;
  // Longer spans go through the exact-address cache below.
  constexpr u64 kMaxPagedSpan = 64;
  const u32 gen =
      static_cast<u32>(MemoryGeneration()) & ~GuestPageTable::kUnreadable;
  const bool paged =
      bytes && address <= std::numeric_limits<u64>::max() - bytes &&
      ((address + bytes - 1) >> kPageShift) - (address >> kPageShift) <
          kMaxPagedSpan;
  const u64 first = address >> kPageShift;
  const u64 last = paged ? (address + bytes - 1) >> kPageShift : 0;
  GuestPageTable& table = GuestPages();
  if (paged) {
    // Descriptor and cbuffer reads land at thousands of distinct addresses a
    // frame but on few pages. A page proven unreadable is remembered too: a
    // miss on one falls back to parsing /proc/self/maps (tens of thousands of
    // lines here), and a replay chasing garbage pointers asks that about
    // thousands of addresses a frame, most of them on a few dead pages.
    bool all = true;
    for (u64 page = first; page <= last; page++) {
      const GuestPageTable::Page* p = table.Find(page << kPageShift);
      const u32 known = p ? p->readable.load(base::memory_order_relaxed) : 0;
      if (known == (gen | GuestPageTable::kUnreadable))
        return false;
      all &= known == gen;
    }
    if (all)
      return true;
  }
  struct Entry {
    u64 bytes = 0;
    u64 generation = 0;
    bool readable = false;
  };
  // Per thread, so the command processor and the frame loop never share it and
  // no lock is needed.
  static thread_local base::HashMap<u64, Entry> cache;
  static thread_local u64 seen_generation = 0;
  if (seen_generation != MemoryGeneration()) {
    seen_generation = MemoryGeneration();
    cache.clear();
  }
  auto it = cache.find(address);
  if (it != cache.end() && it->second.bytes == bytes)
    return it->second.readable;
  const bool readable = IsReadableRange(address, bytes);
  if (readable && paged) {
    for (u64 page = first; page <= last; page++)
      if (GuestPageTable::Page* p = table.At(page << kPageShift))
        p->readable.store(gen, base::memory_order_relaxed);
    return true;
  }
  // Mappings are page granular: a range within one page that is unreadable
  // means the whole page is.
  if (!readable && paged && first == last)
    if (GuestPageTable::Page* p = table.At(first << kPageShift))
      p->readable.store(gen | GuestPageTable::kUnreadable,
                        base::memory_order_relaxed);
  cache[address] = {bytes, MemoryGeneration(), readable};
  return readable;
}

}  // namespace gpu
