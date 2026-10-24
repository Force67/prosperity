/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero memory syscalls: the calls a PS5 kernel validates differently from
 * a PS4 one, and the ones only a PS5 has.
 */

#include <cstring>
#include "base/algorithm.h"
#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_procid.h"

namespace kern {
namespace {
constexpr uintptr_t kPageMask = 0x3FFF;
constexpr u32 kMemoryTypeCount = 0x15;
// Protection bits a PS5 mapping may not ask for (a PS4 one may not ask 0x8).
constexpr int kPs5ProtInvalid = 0xfc08;
// virtual_query flags: 1 finds the next mapping, 0x20000 asks the short
// 0x18-byte reply, 0x10000 is the third bit the kernel admits.
constexpr int kVqFlagsValid = 0x30001;
constexpr int kVqShortReply = 0x20000;
}  // namespace

int PS4ABI Ps5Mtypeprotect(u8* addr, size_t len, int mtype, int prot) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  uintptr_t end;
  if (__builtin_add_overflow(a, len, &end) || end + kPageMask < end)
    return -SysError::eINVAL;
  if (static_cast<u32>(mtype) >= kMemoryTypeCount || (prot & kPs5ProtInvalid))
    return -SysError::eINVAL;
  return sys_mprotect(addr, len, prot);
}

// libkernel only issues mmap2 for anonymous mappings (fd -1, offset 0, the
// seventh argument on the stack), adding a mapping kind in the fifth slot that
// plain mmap passes as zero: 0..2, and kind 1 limits prot to the CPU bits.
i64 PS4ABI
Ps5Mmap2(void* addr, size_t len, u32 prot, u32 flags, u64 kind, u32 fd) {
  if (kind >= 3 || ((kind & 1) && prot > 0xf))
    return -SysError::eINVAL;
  return reinterpret_cast<i64>(sys_mmap(addr, len, prot, flags, fd, 0));
}

int PS4ABI Ps5VirtualQuery(const void* addr,
                           int flags,
                           void* info,
                           size_t info_size) {
  if (flags & ~kVqFlagsValid)
    return -SysError::eINVAL;
  if (!(flags & kVqShortReply))
    return sys_virtual_query(addr, flags, info, info_size);

  // {start, end, resident count, memory type}.
  u8 full[0x48] = {};
  const int err = sys_virtual_query(addr, flags, full, sizeof(full));
  if (!info || !info_size)
    return err ? err : -SysError::eINVAL;
  u8 reply[0x18] = {};
  std::memcpy(reply, full, 0x10);
  std::memcpy(reply + 0x14, full + 0x1c, sizeof(u32));
  std::memcpy(info, reply, base::Min<size_t>(info_size, sizeof(reply)));
  return err;
}

// Two counters per memory class, of which a title may ask class 0 (flexible)
// or 10; nothing here tracks residency, so both read zero.
int PS4ABI Ps5GetResidentFmemCount2(int pid, int which, u64* count, u64* peak) {
  if (which != 0 && which != 10)
    return -SysError::eINVAL;
  if (pid && pid != sys_getpid())
    return -SysError::eSRCH;
  if (!count || !peak)
    return -SysError::eFAULT;
  *count = 0;
  *peak = 0;
  return 0;
}

// A PS5 process's map spans 128 TiB, not the PS4's 1 TiB.
int PS4ABI Ps5Mname(u8* addr, size_t len, const char* name, void* /*unused*/) {
  return NameMemoryRange(addr, len, name, 0x800000000000ull);
}

}  // namespace kern
