#include "ctor_probe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <utl/mem.h>

#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"

namespace krnl::ps5 {

// The SCE entry stub runs the legacy .ctors list with
//   lea rbx,[rip+disp32]      ; the LAST ctor slot
//   ...                       ; walk DOWN, skipping nulls, stop at the -1 head
// so the slot just above it is __CTOR_END__ (a zero terminator nothing walks
// into). Point the lea one slot higher and park our function there: it then runs
// first, before every real static initializer, and the list itself is untouched.
void maybePrependCtor(proc &p) {
  const char *e = std::getenv("DELTA_PS5_CTOR_PREPEND");
  if (!e || p.getPlatform() != proc::platform::ps5)
    return;
  const uint64_t off = std::strtoull(e, nullptr, 16);
  auto &info = p.getModuleList()[0]->getInfo();
  uint8_t *base = info.base;
  if (!base || !off)
    return;

  // `lea rbx,[rip+disp32]` sits at module+0x45 in the stub (entry is +0x70).
  uint8_t *lea = base + 0x45;
  if (lea[0] != 0x48 || lea[1] != 0x8d || lea[2] != 0x1d) {
    std::fprintf(stderr, "[ctorprobe] entry stub not recognised at +0x45\n");
    return;
  }
  const int32_t disp = *reinterpret_cast<int32_t *>(lea + 3);
  uint8_t *listTop = lea + 7 + disp;
  uint64_t *slot = reinterpret_cast<uint64_t *>(listTop + 8);

  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(slot) & ~0xFFFull),
                  0x2000, utl::pageProtection::rwx);
  if (*slot) {
    std::fprintf(stderr, "[ctorprobe] slot above the list is not free (%#lx)\n",
                 (unsigned long)*slot);
    return;
  }
  // A ctor is called with no arguments, so rdi holds whatever the previous one
  // left. Park a shim that zeroes it first -- the init entry points we want to
  // probe are typically `f(0)`.
  uint8_t *shim = krnl::allocLowGuest(0x40);
  if (!shim)
    return;
  const uint64_t target = reinterpret_cast<uint64_t>(base + off);
  size_t i = 0;
  shim[i++] = 0x31; shim[i++] = 0xFF;              // xor edi, edi
  shim[i++] = 0x48; shim[i++] = 0xB8;              // movabs rax, target
  std::memcpy(shim + i, &target, 8); i += 8;
  shim[i++] = 0xFF; shim[i++] = 0xD0;              // call rax
  shim[i++] = 0xC3;                                // ret
  utl::protectMem(shim, 0x1000, utl::pageProtection::rwx);
  *slot = reinterpret_cast<uint64_t>(shim);

  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(lea) & ~0xFFFull),
                  0x2000, utl::pageProtection::rwx);
  *reinterpret_cast<int32_t *>(lea + 3) = disp + 8;

  std::fprintf(stderr,
               "[ctorprobe] prepended module+%#lx (%p) via shim %p at slot %p\n",
               (unsigned long)off, static_cast<void *>(base + off),
               static_cast<void *>(shim), static_cast<void *>(slot));
}

} // namespace krnl::ps5
