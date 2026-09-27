#include "kern/lv2/ps5/ctor_probe.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/logging.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kCtorPrepend, "DELTA_PS5_CTOR_PREPEND", nullptr);
}  // namespace

namespace krnl::ps5 {

// The SCE entry stub runs the legacy .ctors list with
//   lea rbx,[rip+disp32]      ; the LAST ctor slot
//   ...                       ; walk DOWN, skipping nulls, stop at the -1 head
// so the slot just above it is __CTOR_END__ (a zero terminator nothing walks
// into). Point the lea one slot higher and park our function there: it then
// runs first, before every real static initializer, and the list itself is
// untouched.
void MaybePrependCtor(Proc& p) {
  const char* e = kCtorPrepend;
  if (!e || p.GetPlatform() != Proc::Platform::kPs5)
    return;
  const u64 off = std::strtoull(e, nullptr, 16);
  auto& info = p.GetModuleList()[0]->GetInfo();
  u8* base = info.base;
  if (!base || !off)
    return;

  // `lea rbx,[rip+disp32]` sits at module+0x45 in the stub (entry is +0x70).
  u8* lea = base + 0x45;
  if (lea[0] != 0x48 || lea[1] != 0x8d || lea[2] != 0x1d) {
    BASE_LOGI("ctorprobe", "entry stub not recognised at +0x45");
    return;
  }
  const i32 disp = *reinterpret_cast<i32*>(lea + 3);
  u8* list_top = lea + 7 + disp;
  u64* slot = reinterpret_cast<u64*>(list_top + 8);

  host_memory::ProtectMem(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~0xFFFull),
      0x2000, host_memory::PageProtection::kRwx);
  if (*slot) {
    BASE_LOGI("ctorprobe", "slot above the list is not free ({:#x})",
              (unsigned long)*slot);
    return;
  }
  // A ctor is called with no arguments, so rdi holds whatever the previous one
  // left. Park a shim that zeroes it first, since the init entry points we want
  // to probe are typically `f(0)`.
  u8* shim = krnl::AllocLowGuest(0x40);
  if (!shim)
    return;
  const u64 target = reinterpret_cast<u64>(base + off);
  size_t i = 0;
  shim[i++] = 0x31;
  shim[i++] = 0xFF;  // xor edi, edi
  shim[i++] = 0x48;
  shim[i++] = 0xB8;  // movabs rax, target
  std::memcpy(shim + i, &target, 8);
  i += 8;
  shim[i++] = 0xFF;
  shim[i++] = 0xD0;  // call rax
  shim[i++] = 0xC3;  // ret
  host_memory::ProtectMem(shim, 0x1000, host_memory::PageProtection::kRwx);
  *slot = reinterpret_cast<u64>(shim);

  host_memory::ProtectMem(
      reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(lea) & ~0xFFFull),
      0x2000, host_memory::PageProtection::kRwx);
  *reinterpret_cast<i32*>(lea + 3) = disp + 8;

  BASE_LOGI("ctorprobe",
            "prepended module+{:#x} ({:p}) via shim {:p} at slot {:p}",
            (unsigned long)off, static_cast<void*>(base + off),
            static_cast<void*>(shim), static_cast<void*>(slot));
}

}  // namespace krnl::ps5
