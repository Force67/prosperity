/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The guest picks its own virtual addresses. libkernel MAP_FIXEDs its internal
 * arena at a hard-coded 8 GiB, the GNM driver maps its register/dump areas just
 * below 64 GiB, and titles MAP_FIXED their direct/flexible memory pools at round
 * 64 GiB slots. None of that is negotiable: the guest asserts on, or silently
 * mis-indexes, an address it did not ask for.
 *
 * The host, meanwhile, is free to put an anonymous mmap wherever it likes -- and
 * with FEX in the process there are a lot of host allocations (JIT buffers, block
 * link maps, thunk pools) made before the guest has mapped anything. Whoever gets
 * there first wins, and when the host wins the guest's fixed map either fails or
 * gets relocated. A relocated mapping is the worse outcome because it looks like
 * success: libkernel quietly falls back to its internal arena, exhausts the 16 MiB,
 * prints "Internal Memory is running out" and throws std::bad_alloc.
 *
 * So claim the ranges up front, PROT_NONE and MAP_NORESERVE (address space only,
 * no commit, no RSS). A later guest MAP_FIXED replaces the placeholder, which is
 * exactly what we want; a guest mmap that only HINTS at one of these addresses is
 * handled by isGuestReservedVa() at the sys_mmap placement decision, which treats
 * our own placeholder as free rather than relocating away from it.
 *
 * Deliberately NOT reserved here: the low-guest arena (sys_mem allocLowGuest) and
 * the module region (module.cpp), both of which are bump-allocated by us with
 * MAP_FIXED_NOREPLACE. Reserving those would make our own allocators' probes fail
 * against our own placeholder.
 */

#include "guest_vaspace.h"

#include <base.h>
#include <logger/logger.h>

#include <sys/mman.h>

namespace krnl {
namespace {

struct Range {
  uintptr_t base;
  size_t size;
  const char *what;
};

constexpr size_t kMiB = 1024ull * 1024;
constexpr size_t kGiB = 1024ull * kMiB;

// x86-64 host layout. Android's 39-bit user VA cannot host these addresses at
// all (the arena there is packed differently, see allocLowGuest), so the table
// is empty and reserveGuestVaSpace() is a no-op.
#if defined(__ANDROID__)
constexpr Range kRanges[] = {};
#else
constexpr Range kRanges[] = {
    // libkernel's own arena. Hard-coded base, and the failure when it moves is
    // the bad_alloc described above.
    {0x0000'0002'0000'0000ull, 64 * kMiB, "SceKernelInternalMemory"},
    // The GNM driver's fixed areas (SceGnmGpuInfo / SceGnmDumpArea / DingDong)
    // sit just under 64 GiB.
    {0x0000'000F'C000'0000ull, 1 * kGiB, "SceGnm driver areas"},
};
// Deliberately NOT reserved: the title direct/flexible memory-pool slots at
// N * 64 GiB. Reserving that band (64..320 GiB) was tried and regressed every
// title: a guest pointer into a pool at 256 GiB faulted on the PROT_NONE
// placeholder inside an ioctl, because the paths that commit those pools probe
// with MAP_FIXED_NOREPLACE and a placeholder reads as "occupied" to them. The
// slots also do not need it -- a title MAP_FIXEDs them, and MAP_FIXED replaces
// whatever is mapped. Only ranges nothing of ours allocates into belong here.
#endif

bool g_done = false;

}  // namespace

void reserveGuestVaSpace() {
  if (g_done)
    return;
  g_done = true;
#if !defined(MAP_FIXED_NOREPLACE)
  // Without MAP_FIXED_NOREPLACE a "reservation" could silently land elsewhere
  // and we would report a range as ours while the guest gets a different one.
  LOG_WARNING("vaspace: no MAP_FIXED_NOREPLACE; guest fixed ranges unreserved");
  return;
#else
  for (const auto &r : kRanges) {
    void *p = ::mmap(reinterpret_cast<void *>(r.base), r.size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                         MAP_FIXED_NOREPLACE,
                     -1, 0);
    if (p == reinterpret_cast<void *>(r.base)) {
      LOG_INFO("vaspace: reserved {:#x}+{:#x} for {}", r.base, r.size, r.what);
      continue;
    }
    // Someone already holds it. Give the stray mapping back if the kernel
    // relocated us, and say so loudly: a guest fixed map into this range is now
    // going to fight whatever is there.
    if (p != MAP_FAILED)
      ::munmap(p, r.size);
    LOG_WARNING("vaspace: {:#x}+{:#x} ({}) already occupied", r.base, r.size,
                r.what);
  }
#endif
}

bool isGuestReservedVa(const void *addr, size_t len) {
  const auto a = reinterpret_cast<uintptr_t>(addr);
  if (!a || !len)
    return false;
  for (const auto &r : kRanges) {
    if (a >= r.base && a + len <= r.base + r.size)
      return true;
  }
  return false;
}

}  // namespace krnl
