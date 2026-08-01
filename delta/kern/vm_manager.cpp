
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <utl/mem.h>

#include "proc.h"
#include "vm_manager.h"

namespace krnl {
vmManager::vmManager(procInfo &info) : pinfo(info) {}

vmManager::~vmManager() {
  if (pinfo.userStack)
    utl::freeMem(pinfo.userStack);

  pinfo.userStack = nullptr;
}

bool vmManager::init() {
  /*reserve address space for the user stack*/
  pinfo.userStack = static_cast<uint8_t *>(
      utl::allocMem(nullptr, pinfo.userStackSize, utl::pageProtection::priv,
                    utl::allocationType::reserve));

  return pinfo.userStack;
}

// Erase [p, p+s) from the interval set, splitting straddlers. Caller holds
// vmlock. Keeps entries DISJOINT, which is what makes get() unambiguous: the
// guest keys real bookkeeping (SotC's streaming-heap trackers) off the exact
// [start, end) sceKernelVirtualQuery reports, so the newest mapping of a range
// must win and stale overlaps must not shadow it.
void vmManager::punchHoleLocked(uint8_t *p, size_t s) {
  uint8_t *end = p + s;
  for (size_t i = 0; i < rtPages.size();) {
    pageInfo &e = rtPages[i];
    uint8_t *eEnd = e.ptr + e.size;
    if (eEnd <= p || e.ptr >= end) {  // disjoint
      ++i;
      continue;
    }
    if (e.ptr >= p && eEnd <= end) {  // fully covered: drop
      rtPages.erase(rtPages.begin() + i);
      continue;
    }
    if (e.ptr < p && eEnd > end) {  // strictly contains: split into two
      pageInfo tail(end, static_cast<size_t>(eEnd - end), e.prot, e.sceProt,
                    e.reserved);
      tail.name = e.name;
      tail.hasPhys = e.hasPhys;
      tail.physOffset = e.physOffset + static_cast<size_t>(end - e.ptr);
      e.size = static_cast<size_t>(p - e.ptr);
      rtPages.emplace_back(tail);  // append; disjointness keeps get() correct
      ++i;
      continue;
    }
    if (e.ptr < p) {  // overlaps our head from below: truncate its tail
      e.size = static_cast<size_t>(p - e.ptr);
    } else {          // overlaps our tail from above: advance its head
      e.size = static_cast<size_t>(eEnd - end);
      e.physOffset += static_cast<size_t>(end - e.ptr);
      e.ptr = end;
    }
    ++i;
  }
}

void vmManager::add(uint8_t *ptr, size_t size, mprot prot, uint32_t sceProt,
                    bool reserved) {
  std::lock_guard lock(vmlock);
  punchHoleLocked(ptr, size);
  rtPages.emplace_back(ptr, size, prot, sceProt, reserved);
}

void vmManager::addDirect(uint8_t *ptr, size_t size, mprot prot,
                          uint32_t sceProt, uint64_t physOffset) {
  std::lock_guard lock(vmlock);
  punchHoleLocked(ptr, size);
  rtPages.emplace_back(ptr, size, prot, sceProt, false);
  rtPages.back().physOffset = physOffset;
  rtPages.back().hasPhys = true;
}

void vmManager::remove(uint8_t *ptr, size_t size) {
  std::lock_guard lock(vmlock);
  punchHoleLocked(ptr, size);
}

pageInfo *vmManager::get(uint8_t *ptr) {
  std::lock_guard lock(vmlock);
  // The kernel resolves the region *containing* an address, not just one that
  // starts there: sceKernelVirtualQuery / QueryMemoryProtection / mname all pass
  // interior pointers. Match by range so those report the right region.
  auto it = std::find_if(rtPages.begin(), rtPages.end(), [&ptr](const auto &page) {
    return ptr >= page.ptr && ptr < page.ptr + page.size;
  });

  if (it != rtPages.end())
    return &*it;

  return nullptr;
}

bool vmManager::overlaps(uint8_t *ptr, size_t size) const {
  std::lock_guard lock(vmlock);
  uint8_t *end = ptr + size;
  for (const auto &page : rtPages) {
    if (ptr < page.ptr + page.size && page.ptr < end)
      return true;
  }
  return false;
}

void vmManager::protectRange(uint8_t *ptr, size_t size, mprot prot,
                             uint32_t sceProt) {
  std::lock_guard lock(vmlock);
  uint8_t *end = ptr + size;
  for (auto &page : rtPages)
    if (ptr < page.ptr + page.size && page.ptr < end) {
      page.prot = prot;
      page.sceProt = sceProt;
    }
}

void vmManager::setRangeName(uint8_t *ptr, size_t size, const char *name) {
  std::lock_guard lock(vmlock);
  uint8_t *end = ptr + size;
  // The kernel (vm_map_set_name) allocates the name storage per entry; mirror
  // that with a strdup'd copy so the field outlives the caller's buffer. The
  // VMA never frees entry names, so repeated naming leaks a small string each
  // time -- names are handed out rarely (thread stacks), so that is fine.
  if (!name)
    return;
  size_t n = strlen(name);
  char *copy = static_cast<char *>(std::malloc(n + 1));
  if (!copy)
    return;
  std::memcpy(copy, name, n + 1);
  for (auto &page : rtPages)
    if (ptr < page.ptr + page.size && page.ptr < end)
      page.name = copy;
}

void vmManager::forEachGpuAperturePage(void (*fn)(void *, uint8_t *, size_t),
                                       void *ctx) const {
  std::lock_guard lock(vmlock);
  for (const auto &page : rtPages) {
    auto a = reinterpret_cast<uint64_t>(page.ptr);
    // Same range as gpu/ps5's GpuAddr(): everything allocLowGuest() can hand
    // out, from the lowest fixed-mapped pool slot to the 2^40 user ceiling.
    if (a >= 0x1000000000ull && a < 0x10000000000ull)
      fn(ctx, page.ptr, page.size);  // callback caps how much of a big pool it sweeps
  }
}

uint8_t *vmManager::mapMemory(uint8_t *preference, size_t size,
                              utl::pageProtection prot) {
  const auto allocType = utl::allocationType::reservecommit;

  void *ptr =
      utl::allocMem(static_cast<void *>(preference), size, prot, allocType);
  if (ptr) {
    return static_cast<uint8_t *>(ptr);
  }

  return nullptr;
}

void vmManager::unmapRtMemory(uint8_t *ptr) {
  std::lock_guard lock(vmlock);
  auto iter =
      std::find_if(rtPages.begin(), rtPages.end(),
                   [&ptr](const auto &page) { return page.ptr == ptr; });

  rtPages.erase(iter);
}
} // namespace krnl