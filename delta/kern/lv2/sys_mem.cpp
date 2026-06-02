
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <logger/logger.h>
#include <utl/mem.h>

#include <atomic>
#include <cstdint>
#include <sys/mman.h>

#include "../proc.h"
#include "error_table.h"
#include "sys_mem.h"

namespace krnl {
using ppt = utl::pageProtection;
using alt = utl::allocationType;

// PS4 user-space pointers must live below 2^40: libc's sceLibcMspaceCreate (and
// other allocators) reject a base whose bits >= 40 are set. The host kernel
// hands mmap(NULL) addresses far above that ceiling, so when we have to pick an
// address ourselves, carve it from a dedicated low arena instead. Bump-only and
// MAP_FIXED_NOREPLACE so we never clobber an existing mapping.
static uint8_t *allocLowGuest(size_t size) {
  constexpr uintptr_t kFloor = 0x1000000000ull;   // 64 GiB
  constexpr uintptr_t kCeil = 0x10000000000ull;   // 2^40, the PS4 user ceiling
  static std::atomic<uintptr_t> next{kFloor};
  for (int tries = 0; tries < 8192; tries++) {
    uintptr_t base = next.fetch_add(size + 0x4000) & ~uintptr_t(0x3FFF);
    if (base < kFloor || base + size > kCeil)
      return nullptr;
    void *p = ::mmap(reinterpret_cast<void *>(base), size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == reinterpret_cast<void *>(base))
      return static_cast<uint8_t *>(p);
    if (p != MAP_FAILED)
      ::munmap(p, size);  // kernel ignored the hint; release and try higher
  }
  return nullptr;
}

uint8_t *PS4ABI sys_mmap(void *addr, size_t size, uint32_t prot, uint32_t flags,
                         uint32_t fd, size_t offset) {
  auto *proc = proc::getActive();
  if (!proc)
    return reinterpret_cast<uint8_t *>(-1);

  if (flags & mFlags::stack || flags & mFlags::noextend)
    flags |= mFlags::anon;

  /*align the page*/
  size = (size + 0x3FFF) & 0xFFFFFFFFFFFFC000LL;

  // addr is a hint unless MAP_FIXED: relocate it rather than alias an existing map
  if (!(flags & mFlags::fixed)) {
    if (!addr || proc->getVma().overlaps(static_cast<uint8_t *>(addr), size))
      addr = nullptr;
  }

  if (fd != -1) {
    auto *obj = proc->getObjTable().get(fd);
    if (obj) {
      /*TODO: mmap in device!!*/
      static_cast<device *>(obj)->map(addr, size, prot, flags, offset);
    }
  }

  void *ptr = nullptr;
  if (addr) {
    ptr = utl::allocMem(addr, size, ppt::w, alt::reservecommit);
    if (!ptr && (flags & mFlags::fixed))
      ptr = utl::allocMem(addr, size, ppt::w, alt::commit);  // maybe reserved
  }
  // No usable hint (or it was taken): pick a low (<2^40) address the guest's
  // own allocators will accept, not whatever high address the host hands out.
  if (!ptr)
    ptr = allocLowGuest(size);
  if (!ptr) {
    return reinterpret_cast<uint8_t *>(-1);
  }

#if 0
		/*auto tprot = ppt::r;
		if (prot & mprotFlags::write)
			tprot = ppt::w; /*intentional*/
		if (prot & mprotFlags::exec)
			tprot = ppt::rx; */
#endif
  // FIXME: apply real protections
  auto tprot = ppt::rwx;

  if (flags & mFlags::anon)
    std::memset(ptr, 0, size);

  proc->getVma().add(static_cast<uint8_t *>(ptr), size, tprot);

  // now we apply target protection
  utl::protectMem(static_cast<void *>(ptr), size, tprot);

  std::printf("mmap %p, %x, %p\n", addr, size, _ReturnAddress());
  // LOG_WARNING("addr={}, len={}, requested by {}", fmt::ptr(addr), len,
  // fmt::ptr(_ReturnAddress()));

  if (flags & mFlags::stack)
    return &static_cast<uint8_t *>(ptr)[size];

  return static_cast<uint8_t *>(ptr);
}

int PS4ABI sys_mprotect(uint8_t *, size_t len, int prot) { 
    //TODO
    return 0;
}

int PS4ABI sys_mname(uint8_t *ptr, size_t len, const char *name, void *) {
  auto *proc = proc::getActive();
  if (!proc)
    return -1;

  auto *info = proc->getVma().get(ptr);
  if (!info) {
    LOG_WARNING("attempted to tag unknown memory ({}, {})", fmt::ptr(ptr),
                name);
    return -1;
  }

  LOG_WARNING("tagged {} with name {}", fmt::ptr(ptr), name);
  info->name = name;
  return 0;
}

struct mdbg_property {
  int32_t unk;
  int32_t unk2;
  void *addr;
  size_t areaSize;
  int64_t unk3;
  int64_t unk4;
  char name[32];
};

static_assert(sizeof(mdbg_property) == 72);

int PS4ABI sys_mdbg_service(uint32_t op, void *arg1, void *arg2, void *a3) {
  switch (op) {
  case 1: {
    auto *info = static_cast<mdbg_property *>(arg1);
    LOG_WARNING("set property {} for addr {} with size {}", info->name,
                info->addr, info->areaSize);
    /*TODO: create named object*/

    break;
  }
  }

  return 0;
}

int PS4ABI sys_dmem_container(uint32_t op) {
  if (op == -1)
    return 0;

  __debugbreak();
  return -1;
}
} // namespace krnl