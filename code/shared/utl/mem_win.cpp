
/*
 * UTL : The universal utility library
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#ifdef _WIN32
#include "mem.h"

#include <Windows.h>

namespace utl {
/*TODO: bugcheck on default cases*/

DWORD protection_ToWin32(pageProtection prot) {
  switch (prot) {
  case pageProtection::priv:
    return PAGE_NOACCESS;
  case pageProtection::r:
    return PAGE_READONLY;
  case pageProtection::w:
    return PAGE_READWRITE;
  case pageProtection::rx:
    return PAGE_EXECUTE_READ;
  case pageProtection::rwx:
    return PAGE_EXECUTE_READWRITE;
  default:
    __debugbreak();
    return 0;
  }
}

DWORD allocType_ToWin32(allocationType type) {
  switch (type) {
  case allocationType::commit:
    return MEM_COMMIT;
  case allocationType::reserve:
    return MEM_RESERVE;
  case allocationType::reservecommit:
    return MEM_RESERVE | MEM_COMMIT;
  default:
    __debugbreak();
    return 0;
  }
}

void *allocMem(void *preferredAddr, size_t length, pageProtection prot,
               allocationType type) {
  return VirtualAlloc(preferredAddr, length, allocType_ToWin32(type),
                      protection_ToWin32(prot));
}

void freeMem(void *Addr) { VirtualFree(Addr, 0, MEM_RELEASE); }

bool protectMem(void *addr, size_t len, pageProtection prot) {
  DWORD old;
  return VirtualProtect(addr, len, protection_ToWin32(prot), &old);
}

size_t getAvailableMem() {
  MEMORYSTATUSEX statex{};
  statex.dwLength = sizeof(statex);

  if (GlobalMemoryStatusEx(&statex))
    return statex.ullTotalPhys;

  return -1;
}
}
#endif