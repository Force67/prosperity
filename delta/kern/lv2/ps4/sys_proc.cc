/*
 * PS4Delta : PS4 emulation and research project
 *
 * The answers many Orbis rows share, and mtypeprotect as the PS4 kernel
 * checks it.
 */

#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps4/handlers.h"
#include "kern/lv2/sys_mem.h"

namespace kern {
namespace {
constexpr uintptr_t kPageMask = 0x3FFF;
constexpr u32 kMemoryTypeCount = 0x15;
// Protection bit a PS4 mapping may not ask for.
constexpr int kProtInvalid = 0x8;
}  // namespace

int PS4ABI Ps4Ok() {
  return 0;
}
int PS4ABI Ps4Nosys() {
  return -SysError::eNOSYS;
}
int PS4ABI Ps4Eperm() {
  return -SysError::ePERM;
}
int PS4ABI Ps4Einval() {
  return -SysError::eINVAL;
}
int PS4ABI Ps4Eopnotsupp() {
  return -SysError::eOPNOTSUPP;
}

int PS4ABI Ps4Mtypeprotect(u8* addr, size_t len, int mtype, int prot) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  uintptr_t end;
  if (__builtin_add_overflow(a, len, &end) || end + kPageMask < end)
    return -SysError::eINVAL;
  if (static_cast<u32>(mtype) >= kMemoryTypeCount || (prot & kProtInvalid))
    return -SysError::eINVAL;
  return sys_mprotect(addr, len, prot);
}

}  // namespace kern
