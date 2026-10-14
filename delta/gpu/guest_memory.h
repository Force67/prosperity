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
  static const pid_t kSelf = getpid();

  const u64 page = static_cast<u64>(kPageSize);
  const u64 end = address + bytes;

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
  if (read != static_cast<ssize_t>(count) && !GuestOwnerHolds(address, end))
    return read < 0 && errno != EFAULT && IsReadableMapping(address, bytes);

  const size_t pages = static_cast<size_t>((last_page - first_page) / page + 1);
  static thread_local base::Vector<unsigned char> residency;
  if (residency.size() < pages)
    residency.resize(pages);
  if (mincore(reinterpret_cast<void*>(first_page), pages * page,
              residency.data()) != 0)
    return errno != ENOMEM && IsReadableMapping(address, bytes);
  return true;
}

inline bool IsWritableRange(u64 address, u64 bytes) {
  if (!IsReadableRange(address, bytes))
    return false;
  static const pid_t kSelf = getpid();
  static const long kPageSize = sysconf(_SC_PAGESIZE);
  const u64 last = (address + bytes - 1) & ~(static_cast<u64>(kPageSize) - 1);
  for (u64 at : {address, last > address ? last : address}) {
    const GuestPageTable::Page* p = GuestPages().Find(at);
    if (p && p->armed.load(base::memory_order_relaxed))
      continue;
    u8 byte = *reinterpret_cast<const volatile u8*>(at);
    iovec local{&byte, 1};
    iovec remote{reinterpret_cast<void*>(at), 1};
    ssize_t wrote;
    do {
      wrote = syscall(SYS_process_vm_writev, kSelf, &local, 1, &remote, 1, 0);
    } while (wrote < 0 && errno == EINTR);
    if (wrote != 1)
      return false;
  }
  return true;
}

inline u64& MemoryGeneration() {
  static u64 gen = 1;
  return gen;
}

inline void NextMemoryGeneration() {
  MemoryGeneration()++;
}

inline base::Atomic<bool>& MemoryRemapped() {
  static base::Atomic<bool> remapped{true};
  return remapped;
}

inline base::Atomic<u64>& GuestMappingGeneration() {
  static base::Atomic<u64> generation{1};
  return generation;
}

inline void FrameMemoryGeneration() {
  static u32 frames = 0;
  if (MemoryRemapped().exchange(false, base::memory_order_acq_rel) ||
      ++frames >= 60) {
    frames = 0;
    NextMemoryGeneration();
  }
}

inline bool IsReadableRangeCached(u64 address, u64 bytes) {
  constexpr u64 kPageShift = GuestPageTable::kPageShift;
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
  if (!readable && paged && first == last)
    if (GuestPageTable::Page* p = table.At(first << kPageShift))
      p->readable.store(gen | GuestPageTable::kUnreadable,
                        base::memory_order_relaxed);
  cache[address] = {bytes, MemoryGeneration(), readable};
  return readable;
}

}  // namespace gpu
