/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "host_memory/host_memory.h"

#include <sys/mman.h>
#include <unistd.h>
#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "profile/profile.h"

namespace host_memory {

namespace {
struct Mapping {
  uintptr_t end;
  u64 identity;
};
struct MappingRegistry {
  base::Mutex lock;
  base::Map<uintptr_t, Mapping> mappings;
  u64 next_identity = 0;
  bool session = false;
  base::Map<uintptr_t, uintptr_t> owned;
};
MappingRegistry& MappingState() {
  // AllocMem can be called while another translation unit initializes.
  static MappingRegistry state;
  return state;
}

void ForgetMapping(MappingRegistry& state, uintptr_t begin, uintptr_t end) {
  auto& mappings = state.mappings;
  auto it = mappings.upper_bound(begin);
  if (it != mappings.begin())
    --it;
  while (it != mappings.end() && it->first < end) {
    const auto [base, prior] = *it;
    if (prior.end <= begin) {
      ++it;
      continue;
    }
    it = mappings.erase(it);
    if (base < begin)
      mappings.emplace(base, Mapping{begin, prior.identity});
    if (prior.end > end)
      mappings.emplace(end, Mapping{prior.end, prior.identity});
  }
}
}  // namespace

void TrackMemoryMapping(void* addr, size_t len) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(addr);
  if (!base || !len || len > UINTPTR_MAX - base)
    return;
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  ForgetMapping(state, base, base + len);
  state.mappings.emplace(base, Mapping{base + len, ++state.next_identity});
  if (state.session) {
    uintptr_t first = base, end = base + len;
    auto it = state.owned.upper_bound(first);
    if (it != state.owned.begin())
      --it;
    while (it != state.owned.end() && it->first <= end) {
      if (it->second < first) {
        ++it;
        continue;
      }
      first = base::Min(first, it->first);
      end = base::Max(end, it->second);
      it = state.owned.erase(it);
    }
    state.owned[first] = end;
  }
}

void ForgetMemoryMapping(void* addr, size_t len) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(addr);
  if (!base || !len || len > UINTPTR_MAX - base)
    return;
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  ForgetMapping(state, base, base + len);
}

u64 MemoryMappingIdentity(const void* addr, size_t len) {
  uintptr_t base = reinterpret_cast<uintptr_t>(addr);
  if (!base || !len || len > UINTPTR_MAX - base)
    return 0;
  const uintptr_t end = base + len;
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  const auto& mappings = state.mappings;
  auto it = mappings.upper_bound(base);
  if (it == mappings.begin())
    return 0;
  --it;
  u64 hash = 1469598103934665603ull;
  while (base < end) {
    if (it == mappings.end() || it->first > base || it->second.end <= base)
      return 0;
    hash = (hash ^ it->second.identity) * 1099511628211ull;
    base = it->second.end;
    ++it;
  }
  return hash ? hash : 1;
}

static int ProtectionToPosix(PageProtection prot) {
  switch (prot) {
    case PageProtection::kPriv:
      return PROT_NONE;
    case PageProtection::kR:
      return PROT_READ;
    case PageProtection::kW:
      return PROT_READ | PROT_WRITE;
    case PageProtection::kRx:
      return PROT_READ | PROT_EXEC;
    case PageProtection::kRwx:
      return PROT_READ | PROT_WRITE | PROT_EXEC;
    default:
      __builtin_trap();
      return 0;
  }
}

void* AllocMem(void* preferred_addr,
               size_t length,
               PageProtection prot,
               AllocationType type) {
  DELTA_ZONE("mem.alloc");
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  int posix_prot;

  if (type == AllocationType::kReserve) {
    // reserve: mapped but inaccessible, and don't clobber an existing mapping
    posix_prot = PROT_NONE;
    if (preferred_addr)
      flags |= MAP_FIXED_NOREPLACE;
  } else {
    // commit: overlay a sub-range of the reservation. has to be MAP_FIXED –
    // MAP_FIXED_NOREPLACE hits EEXIST and the page stays unwritable.
    posix_prot = ProtectionToPosix(prot);
    if (preferred_addr)
      flags |= MAP_FIXED;
  }

  void* p = ::mmap(preferred_addr, length, posix_prot, flags, -1, 0);
  if (p == MAP_FAILED)
    return nullptr;
  TrackMemoryMapping(p, length);
  return p;
}

void FreeMem(void* addr, size_t size) {
  if (!addr || !size)
    return;
  ::munmap(addr, size);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(addr), end = begin + size;
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  ForgetMapping(state, begin, end);
  auto it = state.owned.upper_bound(begin);
  if (it != state.owned.begin())
    --it;
  while (it != state.owned.end() && it->first < end) {
    const auto [first, last] = *it;
    if (last <= begin) {
      ++it;
      continue;
    }
    it = state.owned.erase(it);
    if (first < begin)
      state.owned.emplace(first, begin);
    if (last > end)
      state.owned.emplace(end, last);
  }
}

void FreeMem(void* addr) {
  auto& state = MappingState();
  size_t size = 0;
  {
    base::LockGuard lock(state.lock);
    const auto it = state.mappings.find(reinterpret_cast<uintptr_t>(addr));
    if (it != state.mappings.end())
      size = it->second.end - it->first;
  }
  FreeMem(addr, size);
}

void BeginMemorySession() {
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  state.session = true;
}

size_t SessionMappedBytes() {
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  size_t bytes = 0;
  for (const auto& [first, end] : state.owned)
    bytes += end - first;
  return bytes;
}

size_t EndMemorySession() {
  auto& state = MappingState();
  base::LockGuard lock(state.lock);
  size_t bytes = 0;
  for (const auto& [first, end] : state.owned) {
    ::munmap(reinterpret_cast<void*>(first), end - first);
    ForgetMapping(state, first, end);
    bytes += end - first;
  }
  state.owned = {};
  state.session = false;
  return bytes;
}

bool ProtectMem(void* addr, size_t len, PageProtection prot) {
  DELTA_ZONE("mem.protect");
  return ::mprotect(addr, len, ProtectionToPosix(prot)) == 0;
}

bool IsMemoryRangeMapped(const void* addr, size_t len) {
  DELTA_ZONE("mem.mincore");
  if (!addr || !len)
    return false;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
  if (len > UINTPTR_MAX - begin)
    return false;
  const long page_size_raw = ::sysconf(_SC_PAGE_SIZE);
  if (page_size_raw <= 0)
    return false;
  const uintptr_t page_size = static_cast<uintptr_t>(page_size_raw);
  const uintptr_t first = begin & ~(page_size - 1);
  const uintptr_t last = (begin + len - 1) & ~(page_size - 1);
  const size_t pages = static_cast<size_t>((last - first) / page_size + 1);
  base::Vector<unsigned char> residency(pages);
  return ::mincore(reinterpret_cast<void*>(first), pages * page_size,
                   residency.data()) == 0;
}

size_t MappedMemoryPrefix(const void* addr, size_t max_len) {
  DELTA_ZONE("mem.mapped_prefix");
  if (!addr || !max_len)
    return 0;
  size_t mapped = 0, remaining = max_len;
  while (remaining) {
    const size_t probe = mapped + remaining / 2 + remaining % 2;
    if (IsMemoryRangeMapped(addr, probe)) {
      mapped = probe;
      remaining = max_len - mapped;
    } else {
      remaining = probe - mapped - 1;
    }
  }
  return mapped;
}

size_t GetAvailableMem() {
  long pages = ::sysconf(_SC_PHYS_PAGES);
  long page_size = ::sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page_size <= 0)
    return static_cast<size_t>(-1);
  return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
}

namespace {
base::Atomic<WriteFaultHandler> g_write_fault_handler{nullptr};
base::Atomic<HostWriteHook> g_host_write_hook{nullptr};
}  // namespace

void SetWriteFaultHandler(WriteFaultHandler handler) {
  g_write_fault_handler.store(handler);
}

bool HandleWriteFault(uintptr_t addr) {
  const WriteFaultHandler handler = g_write_fault_handler.load();
  return handler && handler(addr);
}

void SetHostWriteHook(HostWriteHook hook) {
  g_host_write_hook.store(hook);
}

void BeforeHostWrite(void* addr, size_t len) {
  if (const HostWriteHook hook = g_host_write_hook.load(); hook && len)
    hook(addr, len);
}

}  // namespace host_memory
