
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdlib>
#include <cstring>
#include "base/arch.h"

#include "host_memory/host_memory.h"

#include "base/algorithm.h"
#include "base/threading/lock_guard.h"
#include "kern/process.h"
#include "kern/vm_map.h"

namespace krnl {
VmManager::VmManager(ProcInfo& info) : pinfo_(info) {}

VmManager::~VmManager() {
  if (pinfo_.user_stack)
    host_memory::FreeMem(pinfo_.user_stack);

  pinfo_.user_stack = nullptr;
}

bool VmManager::Init() {
  /*reserve address space for the user stack*/
  pinfo_.user_stack = static_cast<u8*>(host_memory::AllocMem(
      nullptr, pinfo_.user_stack_size, host_memory::PageProtection::kPriv,
      host_memory::AllocationType::kReserve));

  return pinfo_.user_stack;
}

// Erase [p, p+s) from the interval set, splitting straddlers. Caller holds
// vmlock. Keeps entries DISJOINT, which is what makes get() unambiguous: the
// guest keys real bookkeeping (SotC's streaming-heap trackers) off the exact
// [start, end) sceKernelVirtualQuery reports, so the newest mapping of a range
// must win and stale overlaps must not shadow it.
void VmManager::PunchHoleLocked(u8* p, size_t s) {
  u8* end = p + s;
  for (size_t i = 0; i < rt_pages_.size();) {
    PageInfo& e = rt_pages_[i];
    u8* e_end = e.ptr + e.size;
    if (e_end <= p || e.ptr >= end) {  // disjoint
      ++i;
      continue;
    }
    if (e.ptr >= p && e_end <= end) {  // fully covered: drop
      rt_pages_.erase(rt_pages_.begin() + i);
      continue;
    }
    if (e.ptr < p && e_end > end) {  // strictly contains: split into two
      PageInfo tail(end, static_cast<size_t>(e_end - end), e.prot, e.sce_prot,
                    e.reserved);
      tail.name = e.name;
      tail.has_phys = e.has_phys;
      tail.phys_offset = e.phys_offset + static_cast<size_t>(end - e.ptr);
      e.size = static_cast<size_t>(p - e.ptr);
      rt_pages_.emplace_back(tail);  // append; disjointness keeps get() correct
      ++i;
      continue;
    }
    if (e.ptr < p) {  // overlaps our head from below: truncate its tail
      e.size = static_cast<size_t>(p - e.ptr);
    } else {  // overlaps our tail from above: advance its head
      e.size = static_cast<size_t>(e_end - end);
      e.phys_offset += static_cast<size_t>(end - e.ptr);
      e.ptr = end;
    }
    ++i;
  }
}

namespace {
MappingChangedHook g_mapping_changed = nullptr;

void MappingChanged(u8* ptr, size_t size) {
  if (g_mapping_changed)
    g_mapping_changed(reinterpret_cast<u64>(ptr), size);
}
}  // namespace

void SetMappingChangedHook(MappingChangedHook hook) {
  g_mapping_changed = hook;
}

void VmManager::Add(u8* ptr,
                    size_t size,
                    Mprot prot,
                    u32 sce_prot,
                    bool reserved) {
  base::LockGuard lock(vmlock_);
  PunchHoleLocked(ptr, size);
  MappingChanged(ptr, size);
  rt_pages_.emplace_back(ptr, size, prot, sce_prot, reserved);
}

void VmManager::AddDirect(u8* ptr,
                          size_t size,
                          Mprot prot,
                          u32 sce_prot,
                          u64 phys_offset) {
  base::LockGuard lock(vmlock_);
  PunchHoleLocked(ptr, size);
  MappingChanged(ptr, size);
  rt_pages_.emplace_back(ptr, size, prot, sce_prot, false);
  rt_pages_.back().phys_offset = phys_offset;
  rt_pages_.back().has_phys = true;
}

void VmManager::Remove(u8* ptr, size_t size) {
  base::LockGuard lock(vmlock_);
  PunchHoleLocked(ptr, size);
  MappingChanged(ptr, size);
}

PageInfo* VmManager::Get(u8* ptr) {
  base::LockGuard lock(vmlock_);
  // The kernel resolves the region *containing* an address, not just one that
  // starts there: sceKernelVirtualQuery / QueryMemoryProtection / mname all
  // pass interior pointers. Match by range so those report the right region.
  auto it = base::FindIf(rt_pages_.begin(), rt_pages_.end(),
                         [&ptr](const auto& page) {
                           return ptr >= page.ptr && ptr < page.ptr + page.size;
                         });

  if (it != rt_pages_.end())
    return &*it;

  return nullptr;
}

bool VmManager::Overlaps(u8* ptr, size_t size) const {
  base::LockGuard lock(vmlock_);
  u8* end = ptr + size;
  for (const auto& page : rt_pages_) {
    if (ptr < page.ptr + page.size && page.ptr < end)
      return true;
  }
  return false;
}

void VmManager::ProtectRange(u8* ptr, size_t size, Mprot prot, u32 sce_prot) {
  base::LockGuard lock(vmlock_);
  u8* end = ptr + size;
  for (auto& page : rt_pages_)
    if (ptr < page.ptr + page.size && page.ptr < end) {
      page.prot = prot;
      page.sce_prot = sce_prot;
    }
}

void VmManager::SetRangeName(u8* ptr, size_t size, const char* name) {
  base::LockGuard lock(vmlock_);
  u8* end = ptr + size;
  // The kernel (vm_map_set_name) allocates the name storage per entry; mirror
  // that with a strdup'd copy so the field outlives the caller's buffer. The
  // VMA never frees entry names, so repeated naming leaks a small string each
  // time. Names are handed out rarely (thread stacks), so that is fine.
  if (!name)
    return;
  size_t n = strlen(name);
  char* copy = static_cast<char*>(std::malloc(n + 1));
  if (!copy)
    return;
  std::memcpy(copy, name, n + 1);
  for (auto& page : rt_pages_)
    if (ptr < page.ptr + page.size && page.ptr < end)
      page.name = copy;
}

void VmManager::ForEachGpuAperturePage(void (*fn)(void*, u8*, size_t),
                                       void* ctx) const {
  base::LockGuard lock(vmlock_);
  for (const auto& page : rt_pages_) {
    auto a = reinterpret_cast<u64>(page.ptr);
    // Same range as gpu/ps5's GpuAddr(): everything AllocLowGuest() can hand
    // out, from the lowest fixed-mapped pool slot to the 2^40 user ceiling.
    if (a >= 0x1000000000ull && a < 0x10000000000ull)
      fn(ctx, page.ptr,
         page.size);  // callback caps how much of a big pool it sweeps
  }
}

u8* VmManager::MapMemory(u8* preference,
                         size_t size,
                         host_memory::PageProtection prot) {
  const auto alloc_type = host_memory::AllocationType::kReservecommit;

  void* ptr = host_memory::AllocMem(static_cast<void*>(preference), size, prot,
                                    alloc_type);
  if (ptr) {
    return static_cast<u8*>(ptr);
  }

  return nullptr;
}

void VmManager::UnmapRtMemory(u8* ptr) {
  base::LockGuard lock(vmlock_);
  auto iter =
      base::FindIf(rt_pages_.begin(), rt_pages_.end(),
                   [&ptr](const auto& page) { return page.ptr == ptr; });

  rt_pages_.erase(iter);
}
}  // namespace krnl