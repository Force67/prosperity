
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <algorithm>

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

void vmManager::add(uint8_t *ptr, size_t size, mprot prot, uint32_t sceProt) {
  std::lock_guard lock(vmlock);
  rtPages.emplace_back(ptr, size, prot, sceProt);
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