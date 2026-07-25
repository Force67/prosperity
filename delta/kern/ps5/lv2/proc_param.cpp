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

// Without a sceLibcParam heap config, PS5 libc gives the C++ operator-new arena a
// small fixed 16 MiB mspace that runs out during asset preload (std::bad_alloc).
// So hand libc a proc-param copy whose sceLibcParam asks for a large fixed heap:
//   sceLibcParam+0x10 -> *heapSize (size_t)
// reached only when version(+0x8) >= 2 and (+0xc) >= 2. Extended (grow-on-demand)
// allocation is deliberately not used: that path re-enters malloc while growing a
// segment and blows the stack.
//
// The app's own sceLibcParam is COPIED rather than replaced. A title that swaps
// out malloc (+0x30) or operator new (+0x38) -- Skyrim routes every allocation
// into its own memory manager -- must keep those tables: dropping them means the
// manager is never constructed, and the first static initializer that allocates
// through it dereferences null.
void *procParam(void *appPP, size_t appSize) {
  static void *synth = nullptr;
  if (synth)
    return synth;

  uint8_t *pp = krnl::allocLowGuest(0x100);
  uint8_t *lp = krnl::allocLowGuest(0x100);
  uint8_t *vals = krnl::allocLowGuest(0x40);
  if (!pp || !lp || !vals)
    return appPP;
  std::memset(pp, 0, 0x100);
  std::memset(lp, 0, 0x100);
  std::memset(vals, 0, 0x40);

  auto *app = static_cast<uint8_t *>(appPP);
  if (app && appSize)
    std::memcpy(pp, app, appSize < 0x100 ? appSize : 0x100);
  if (*reinterpret_cast<uint64_t *>(pp) < 0x40)
    *reinterpret_cast<uint64_t *>(pp) = 0x40;  // libc requires proc[0] >= 0x40

  const uint8_t *appLp =
      (app && appSize > kProcParamLibcParam)
          ? *reinterpret_cast<uint8_t *const *>(app + kProcParamLibcParam)
          : nullptr;
  if (appLp) {
    size_t n = *reinterpret_cast<const uint64_t *>(appLp);
    std::memcpy(lp, appLp, n && n < 0x100 ? n : 0x100);
  }

  auto *heapSize = reinterpret_cast<uint64_t *>(vals);
  *heapSize = 0x60000000ull;  // 1.5 GiB

  auto &size = *reinterpret_cast<uint64_t *>(lp + 0x00);
  auto &ver = *reinterpret_cast<uint32_t *>(lp + 0x08);
  auto &ver2 = *reinterpret_cast<uint32_t *>(lp + 0x0c);
  auto &heap = *reinterpret_cast<uint64_t *>(lp + kLibcHeapSize);
  if (size < 0x48) size = 0x48;
  if (ver < 2) ver = 2;    // gate the heap-config path
  if (ver2 < 2) ver2 = 2;
  if (!heap) heap = reinterpret_cast<uint64_t>(heapSize);  // else honour the app's

  *reinterpret_cast<uint64_t *>(pp + kProcParamLibcParam) =
      reinterpret_cast<uint64_t>(lp);
  if (trace())
    std::fprintf(stderr,
                 "[procparam] sceLibcParam app=%p copy=%p mallocReplace=%#lx "
                 "heapSize=%#lx\n",
                 static_cast<const void *>(appLp), static_cast<void *>(lp),
                 *reinterpret_cast<uint64_t *>(lp + kLibcMallocReplace),
                 *reinterpret_cast<uint64_t *>(heap));
  synth = pp;
  return synth;
}

} // namespace krnl::ps5
