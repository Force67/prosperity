/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5-only HLE aliases for a couple of libkernel exports that firmware 01.14.00
 * does not have. Titles built with a newer SDK ship an /app0/sce_module/libc.prx
 * that imports them; without these the resolver points the slots at libkernel's
 * badcall stub, which returns 0 without touching the out-parameter, and libc's
 * malloc-arena init then stores through a null arena (Dead Cells, PPSA15552).
 *
 * Everything else in libkernel stays LLE: NIDs are globally unique, so the
 * forced-HLE probe only ever matches the two entries below.
 */

#include "../vprx.h"  // PS4ABI (via <base.h>), MODULE_INIT_PS5

#include <cstddef>
#include <cstdint>

#include "kern/ps4/lv2/error_table.h"
#include "kern/ps4/lv2/sys_mem.h"

namespace {
// int f(void **addrOut, size_t len, int prot, size_t align, const char *name)
// The SDK libc uses this for its malloc arena; it asks for 20 MiB at 0x8000.
int PS4ABI kernelMapNamedFlexibleAligned(void **addrOut, size_t len, int prot,
                                         size_t align, const char *name) {
  if (!addrOut || !len)
    return krnl::SysError::eINVAL;

  if (align < 0x4000)
    align = 0x4000;

  auto *base = krnl::sys_mmap(nullptr, len + align, static_cast<uint32_t>(prot),
                              krnl::mFlags::anon, static_cast<uint32_t>(-1), 0);
  if (krnl::isErrnoPtr(base))
    return krnl::SysError::eNOMEM;

  auto addr = (reinterpret_cast<uintptr_t>(base) + align - 1) & ~(align - 1);
  krnl::sys_mname(reinterpret_cast<uint8_t *>(addr), len, name, nullptr);

  *addrOut = reinterpret_cast<void *>(addr);
  return 0;
}

// Diagnostic-path helper the SDK libc calls while formatting an allocator error
// report. The return value is discarded at the call site.
int PS4ABI kernelStubOk(void *, size_t, uint64_t, uint64_t) { return 0; }
}  // namespace

static const runtime::funcInfo functions[] = {
    {0xE21E85D4B2DB4E2C, (void *)&kernelMapNamedFlexibleAligned},
    {0x71FC01490CABE58B, (void *)&kernelStubOk},
};

MODULE_INIT_PS5(libkernel);

extern "C" int vprx_anchor_ps5_libkernel = 1;
