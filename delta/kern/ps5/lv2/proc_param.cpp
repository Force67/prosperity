#include "proc_param.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"

namespace krnl::ps5 {
namespace {

// sceLibcParam fields we care about (verified against libc's heap-init disasm).
constexpr size_t kLibcHeapSize = 0x10;       // size_t* (0 = libc default)
constexpr size_t kLibcMallocReplace = 0x30;  // SceLibcMallocReplace*
constexpr size_t kProcParamLibcParam = 0x38;

bool trace() { return std::getenv("DELTA_PROCPARAM_TRACE") != nullptr; }

} // namespace

// libc only acts on the heap configuration in sceLibcParam when the struct
// advertises version(+0x8) >= 2 and (+0xc) >= 2. Below that it ignores the block
// and the C++ operator-new arena ends up with no usable heap: Isaac
// std::bad_allocs during asset preload, and Skyrim's very first malloc -- which
// libkernel uses to allocate its own TCB -- returns null, failing process init
// with 0xa0020103. Both titles ship version 14/1, so hand libc a COPY of the
// title's param with the gate raised; measured on both, that is what produces a
// working heap.
//
// We do NOT invent a heap size: the copy carries whatever the title configured
// (Isaac asks for 240 MiB, Skyrim for nothing), along with everything else in
// the struct -- in particular the malloc(+0x30) and operator-new(+0x38)
// replacement tables a title may use to route allocation into its own manager.
//
// UNVERIFIED: whether the field layout itself is version-dependent, i.e. whether
// a v14 struct reinterpreted as v2 puts the heap size somewhere other than +0x10.
// Nothing observed so far reads +0x10 or +0x30 of either the title's param or our
// copy, so libc is likely working from an internal copy libkernel made -- which
// would also explain why no title's malloc replacement is ever consumed here.
void *procParam(void *appPP, size_t appSize) {
  static void *synth = nullptr;
  static bool done = false;
  if (done)
    return synth ? synth : appPP;
  done = true;

  auto *app = static_cast<uint8_t *>(appPP);
  const uint8_t *appLp =
      (app && appSize > kProcParamLibcParam)
          ? *reinterpret_cast<uint8_t *const *>(app + kProcParamLibcParam)
          : nullptr;
  const uint32_t appVer =
      appLp ? *reinterpret_cast<const uint32_t *>(appLp + 0x08) : 0;
  const uint32_t appVer2 =
      appLp ? *reinterpret_cast<const uint32_t *>(appLp + 0x0c) : 0;

  // Already advertises a version libc acts on, or there is no param to correct.
  if (!appLp || (appVer >= 2 && appVer2 >= 2)) {
    if (trace())
      std::fprintf(stderr, "[procparam] sceLibcParam=%p ver=%u/%u -- passed through\n",
                   static_cast<const void *>(appLp), appVer, appVer2);
    return appPP;
  }

  uint8_t *pp = krnl::allocLowGuest(0x100);
  uint8_t *lp = krnl::allocLowGuest(0x100);
  if (!pp || !lp)
    return appPP;
  std::memset(pp, 0, 0x100);
  std::memset(lp, 0, 0x100);
  std::memcpy(pp, app, appSize < 0x100 ? appSize : 0x100);
  if (*reinterpret_cast<uint64_t *>(pp) < 0x40)
    *reinterpret_cast<uint64_t *>(pp) = 0x40;  // libc requires proc[0] >= 0x40

  size_t n = *reinterpret_cast<const uint64_t *>(appLp);
  std::memcpy(lp, appLp, n && n < 0x100 ? n : 0x100);
  auto &size = *reinterpret_cast<uint64_t *>(lp + 0x00);
  if (size < 0x48) size = 0x48;
  *reinterpret_cast<uint32_t *>(lp + 0x08) = 2;
  *reinterpret_cast<uint32_t *>(lp + 0x0c) = 2;
  *reinterpret_cast<uint64_t *>(pp + kProcParamLibcParam) =
      reinterpret_cast<uint64_t>(lp);

  if (trace())
    std::fprintf(stderr,
                 "[procparam] raised sceLibcParam version %u/%u -> 2/2 "
                 "(copy=%p heapSize=%#lx mallocReplace=%#lx)\n",
                 appVer, appVer2, static_cast<void *>(lp),
                 (unsigned long)*reinterpret_cast<uint64_t *>(lp + kLibcHeapSize),
                 (unsigned long)*reinterpret_cast<uint64_t *>(lp +
                                                             kLibcMallocReplace));
  synth = pp;
  return synth;
}

} // namespace krnl::ps5
