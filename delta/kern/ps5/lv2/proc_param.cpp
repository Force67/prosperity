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

// libc only reads the heap configuration in sceLibcParam when the struct
// advertises version(+0x8) >= 2 and (+0xc) >= 2; below that it ignores the whole
// block and gives the C++ operator-new arena its default 16 MiB mspace, which
// runs out during asset preload (std::bad_alloc). Titles built against an older
// SDK ship a heap size the running libc then throws away, so raise the version on
// a COPY of the app's param -- that makes libc honour the size the title actually
// asked for. We never invent a size: a title that configured no heap is handed its
// own param untouched, because libc treats "no heap config" and "heap config"
// differently (a title replacing malloc entirely wants the former).
//
// Everything else in the app's sceLibcParam is preserved, in particular the
// malloc(+0x30) and operator-new(+0x38) replacement tables. Dropping those means
// the title's own allocator is never constructed and the first static initializer
// that allocates through it dereferences null.
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
  const uint64_t appHeap =
      appLp ? *reinterpret_cast<const uint64_t *>(appLp + kLibcHeapSize) : 0;
  const uint32_t appVer =
      appLp ? *reinterpret_cast<const uint32_t *>(appLp + 0x08) : 0;
  const uint32_t appVer2 =
      appLp ? *reinterpret_cast<const uint32_t *>(appLp + 0x0c) : 0;

  // Nothing to correct: the title either configured no heap, or already
  // advertises a version libc will act on.
  if (!appHeap || (appVer >= 2 && appVer2 >= 2)) {
    if (trace())
      std::fprintf(stderr,
                   "[procparam] app sceLibcParam=%p heap=%#lx ver=%u/%u "
                   "mallocReplace=%#lx -- passed through\n",
                   static_cast<const void *>(appLp), (unsigned long)appHeap,
                   appVer, appVer2,
                   appLp ? *reinterpret_cast<const uint64_t *>(
                               appLp + kLibcMallocReplace)
                         : 0);
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
                 "[procparam] raised sceLibcParam version %u/%u -> 2/2 so libc "
                 "honours the title's %#lx-byte heap (copy=%p)\n",
                 appVer, appVer2,
                 (unsigned long)*reinterpret_cast<const uint64_t *>(appHeap),
                 static_cast<void *>(lp));
  synth = pp;
  return synth;
}

} // namespace krnl::ps5
