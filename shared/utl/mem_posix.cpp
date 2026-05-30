/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "mem.h"

#include <sys/mman.h>
#include <unistd.h>

namespace utl {

static int protection_ToPosix(pageProtection prot) {
  switch (prot) {
  case pageProtection::priv:
    return PROT_NONE;
  case pageProtection::r:
    return PROT_READ;
  case pageProtection::w:
    return PROT_READ | PROT_WRITE;
  case pageProtection::rx:
    return PROT_READ | PROT_EXEC;
  case pageProtection::rwx:
    return PROT_READ | PROT_WRITE | PROT_EXEC;
  default:
    __builtin_trap();
    return 0;
  }
}

void* allocMem(void* preferredAddr, size_t length, pageProtection prot,
               allocationType type) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  int posix_prot;

  if (type == allocationType::reserve) {
    // reserve: mapped but inaccessible, and don't clobber an existing mapping
    posix_prot = PROT_NONE;
    if (preferredAddr)
      flags |= MAP_FIXED_NOREPLACE;
  } else {
    // commit: overlay a sub-range of the reservation. has to be MAP_FIXED --
    // MAP_FIXED_NOREPLACE hits EEXIST and the page stays unwritable.
    posix_prot = protection_ToPosix(prot);
    if (preferredAddr)
      flags |= MAP_FIXED;
  }

  void* p = ::mmap(preferredAddr, length, posix_prot, flags, -1, 0);
  if (p == MAP_FAILED)
    return nullptr;
  return p;
}

void freeMem(void* addr) {
  // Without size we can't unmap precisely; this matches the Win32 semantic
  // of "release the whole reservation" only loosely. Callers that care must
  // track size externally.
  ::munmap(addr, 0);
}

bool protectMem(void* addr, size_t len, pageProtection prot) {
  return ::mprotect(addr, len, protection_ToPosix(prot)) == 0;
}

size_t getAvailableMem() {
  long pages = ::sysconf(_SC_PHYS_PAGES);
  long page_size = ::sysconf(_SC_PAGE_SIZE);
  if (pages <= 0 || page_size <= 0)
    return static_cast<size_t>(-1);
  return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
}

}  // namespace utl
