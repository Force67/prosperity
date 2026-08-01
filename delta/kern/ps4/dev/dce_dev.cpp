/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <atomic>
#include <base.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dce_dev.h"
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_event.h"
#include "kern/ps4/lv2/error_table.h"
#include "kern/ps4/lv2/sys_mem.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDceTrace, "DELTA_DCE_TRACE", false);
}  // namespace

namespace krnl {
dceDevice::dceDevice(proc *p) : device(p) {}

// A monotonic nanosecond timestamp and a wall-clock ~60 Hz vblank counter. The
// GameMaker runner (and sceVideoOutWaitVblank) busy-polls sceVideoOutGet-
// VblankStatus until the count advances, to pace the frame loop, so the count
// MUST tick in real time even though there is no display hardware. Without this
// the count stays 0 and the title spins forever on its first frame.
static uint64_t nowNs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
static uint64_t vblankCount() {
  using namespace std::chrono;
  static const auto start = steady_clock::now();
  auto ns = duration_cast<nanoseconds>(steady_clock::now() - start).count();
  return static_cast<uint64_t>(ns / 16666667);  // ~59.94 Hz
}
// A TSC value in the same domain/rate as the guest's rdtsc, so flip/vblank-status
// tsc fields stay comparable with rdtsc the title reads itself. On native that's
// the raw host TSC; on FEX use the 1.6 GHz wall clock the emulated rdtsc matches.
static uint64_t guestTsc() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  return nowNs() * 16 / 10;  // ns -> 1.6 GHz ticks
#endif
}

// The last flip submitted via ioctl 0xc0488204 (sceVideoOutSubmitFlip). A title
// flips display buffer N then polls GetFlipStatus until currentBuffer == N (its
// flip became the scanout). We used to report currentBuffer = 0 always, so flips
// to buffers 1/2 never matched and the title spun on each until a ~1s timeout
// (Doom64 ran at ~1fps). Record the flip here and report it in the status.
static std::atomic<uint32_t> g_dceCurrentBuffer{0};
static std::atomic<int64_t> g_dceFlipArg{0};
static std::atomic<uint64_t> g_dceFlipCount{0};
static std::atomic<uint64_t> g_dceScanoutBuffers[16]{};

uint32_t dceCurrentBuffer() { return g_dceCurrentBuffer.load(); }

int64_t dceCurrentFlipArg() { return g_dceFlipArg.load(); }

uint64_t dceScanoutBuffer(uint32_t index) {
  return index < 16 ? g_dceScanoutBuffers[index].load() : 0;
}

static bool g_dceTrace() {
  return kDceTrace;
}

// The native backend runs syscall handlers on the guest stack, so the calling
// libSceVideoOut wrapper's return address sits somewhere up the stack. Scan raw
// stack qwords for the first one landing in libSceVideoOut's .text and report it
// as base+offset, to pin the wrapper that issued each ioctl.
static void printVideoOutCaller() {
  auto *proc = proc::getActive();
  if (!proc)
    return;
  uintptr_t vbase = 0, vsize = 0;
  for (auto &m : proc->getModuleList()) {
    auto &mi = m->getInfo();
    if (std::strcmp(mi.name.c_str(), "libSceVideoOut") == 0) {
      vbase = (uintptr_t)mi.textSeg.addr;
      vsize = mi.textSeg.size;
      break;
    }
  }
  if (!vbase)
    return;
  auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
  int shown = 0;
  uintptr_t lastoff = ~0ull;
  // The caller return addresses sit within the first few frames; a wide scan can
  // walk off the top of the guest stack (an unmapped page) and fault. 128 qwords
  // (1 KiB) is well within the mapped stack and finds the wrapper reliably.
  for (int i = 0; i < 128 && shown < 6; i++) {
    uintptr_t v = sp[i];
    if (v >= vbase && v < vbase + vsize) {
      uintptr_t off = v - vbase;
      if (off == lastoff)
        continue;  // skip repeated return-address slots
      std::printf("[dce]   caller libSceVideoOut+%#lx\n", off);
      lastoff = off;
      shown++;
    }
  }
}

bool dceDevice::init(const char *, uint32_t, uint32_t) { return true; }

// A guest pointer is directly host-addressable here (in-process LLE). Guard
// dereferences to a sane userspace range so a stray field doesn't fault.
// NB: on the FEX (aarch64) backend the guest stack is a host mmap up at
// 0xffff_xxxx_xxxx, so the real libSceVideoOut passes out-slot pointers above
// the old 0x8000_0000_0000 ceiling -- accept the full 48-bit user range, else
// the dce silently drops every write to a stack out-slot (the open-op then
// mmaps an uninitialised offset/size and fails).
static bool plausiblePtr(uint64_t v) {
  return v >= 0x10000 && v < 0x0001000000000000ull;
}

// Dump an ioctl arg struct as u64s. Just the values: do NOT chase "pointers"
// (many fields are two packed u32s like {pitch, height} that look like a high
// address but aren't, and dereferencing them faults). DELTA_DCE_TRACE.
static void dumpStruct(const void *data, uint32_t len) {
  const uint64_t *q = static_cast<const uint64_t *>(data);
  uint32_t n = len / 8;
  for (uint32_t i = 0; i < n; i++)
    std::printf("    [%u] %#018llx\n", i, (unsigned long long)q[i]);
}

uint64_t dceDevice::poolAlloc(uint64_t bytes) {
  // The scanout/control pool. One big guest-addressable region, bump-allocated;
  // every sub-op 9 carve is mmap'd back via map(offset). 512 MiB covers several
  // 1080p buffer sets + control blocks.
  constexpr uint64_t kPool = 512ull * 1024 * 1024;
  if (!poolBase) {
    poolBase = allocLowGuest(kPool);
    if (!poolBase)
      return UINT64_MAX;
    poolSize = kPool;
  }
  bytes = (bytes + 0x3FFF) & ~uint64_t(0x3FFF);
  if (poolUsed + bytes > poolSize)
    return UINT64_MAX;
  uint64_t off = poolUsed;
  poolUsed += bytes;
  return off;
}

uint8_t *dceDevice::map(void *, size_t size, uint32_t, uint32_t, size_t offset) {
  // The kernel only maps /dev/dce at offsets < 0x8000: it returns the PHYSICAL
  // address of the flip-target status page (offset/0x4000 picks the target,
  // and offset 0x4000 is the data region the scanout-pool query sizes). Our
  // libSceVideoOut maps the offset sub-op 9 hands back instead. Return the
  // matching slice of the scanout pool so it gets real, zeroed, shared memory
  // (not the uninitialised anonymous fallback that made it size buffers from
  // garbage). Sub-op 9's pool offsets exceed the kernel's 0x8000 window, so
  // this is a deliberate emulator substitution that keeps the shape (map
  // fd@offset -> real backing) while giving titles CP-usable memory.
  if (poolBase && offset + size <= poolSize)
    return poolBase + offset;
  return reinterpret_cast<uint8_t *>(-1);
}

// Kernel (11.00) dce ioctl dispatch. The scanin ioctls are gated behind a
// system-credential check, so a game issuing them gets EINVAL (22); the flip
// ioctls route to the flip handler: 0xC0308203 -> flip control (sub-op switch,
// valid 0..0x19), 0xC0308206 -> register buffer, 0xC0308207 -> register buffer
// attribute, 0xC0488204 -> submit flip. 0xc0588212 is not in the dispatch at
// all (EINVAL). Sub-op arg layouts below were verified against the 11.00
// libSceVideoOut.sprx callers and the flip-control sub-op jump table.
int32_t dceDevice::ioctl(uint32_t cmd, void *data) {
  if (g_dceTrace()) {
    uint32_t len = (cmd >> 16) & 0x1FFF;
    std::printf("[dce] ioctl cmd=%#x len=%u data=%p\n", cmd, len, data);
    printVideoOutCaller();
    if (data && len && len <= 0x200)
      dumpStruct(data, len);
  }

  auto *s = static_cast<uint64_t *>(data);

  // System-only capture ("scanin") ioctls + the non-existent 0xc0588212: the
  // 11.00 kernel returns EINVAL for all of these from a game context (the
  // scanin set fails the system-credential check; 0xc0588212 falls in the
  // submit branch as an unknown cmd). A game's libSceVideoOut is built against
  // that, so EINVAL is the faithful answer.
  switch (cmd) {
  case 0x80108210:  // start dual capture mode
  case 0xC068820C:  // start capture buffer manager
  case 0xC020820E:  // get capture buffer info
  case 0xC020820F:  // execute capture command
  case 0xC004820D:  // stop capture buffer manager
  case 0xc0588212:  // was "cursor pool allocator"; not in the kernel dispatch
    return -SysError::eINVAL;
  }

  if (cmd == 0xc0308203 && data) {
    switch (s[0]) {
    case 0:
      // Open display pipe. arg[0x20] (s[4]) points at the caller's ctx+0x438
      // slot; the kernel writes the opaque display handle there. Every later
      // ioctl passes that handle as its first u64, so hand back a non-zero id.
      if (plausiblePtr(s[4]))
        *reinterpret_cast<uint64_t *>(s[4]) = nextHandle++;
      return 0;
    case 9: {
      // Allocate scanout pool. Kernel sub-op 9 (scanout-pool offset query): it
      // looks up the flip target, then writes offset=0x4000 and size=
      // (bufferCount<<14) to s[2]/s[3]; the module mmaps the dce fd at that
      // offset for that size. We substitute a bump-allocated pool slice (a
      // guest-addressable region map() hands back as real memory) -- same
      // shape, real backing.
      // s[3] is a pure OUT slot on some callers (libSceVideoOut's open path passes
      // it uninitialised) -- only treat *s[3] as a requested size when it's a
      // sane size (not stack garbage / a pointer), else use the default. Without
      // this the open-op reads a bogus huge size, poolAlloc fails ENOMEM, the
      // slots stay uninitialised, and the title mmaps garbage offset/len.
      uint64_t reqd = plausiblePtr(s[3]) ? *reinterpret_cast<uint64_t *>(s[3]) : 0;
      uint64_t want = (reqd > 0 && reqd <= 0x10000000) ? reqd  // <= 256 MiB
                                                       : 0x4000000;  // 64 MiB
      uint64_t off = poolAlloc(want);
      if (off == UINT64_MAX)
        return 12;  // ENOMEM
      if (plausiblePtr(s[2]))
        *reinterpret_cast<uint64_t *>(s[2]) = off;
      if (plausiblePtr(s[3]))
        *reinterpret_cast<uint64_t *>(s[3]) = want;
      return 0;
    }
    case 0x19: {
      // Resolution status. arg[0x10] (s[2]) = out ptr, arg[0x18] (s[3]) = out
      // size (0x30). The module copies width/height/pane (u32 @0/4/8/0xc),
      // a flag byte @0x10, refresh @0x18, aspect @0x20. Report 1920x1080.
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<uint8_t *>(s[2]);
        std::memset(o, 0, 0x30);
        *reinterpret_cast<uint32_t *>(o + 0x00) = 1920;  // width
        *reinterpret_cast<uint32_t *>(o + 0x04) = 1080;  // height
        *reinterpret_cast<uint32_t *>(o + 0x08) = 1920;  // paneWidth
        *reinterpret_cast<uint32_t *>(o + 0x0c) = 1080;  // paneHeight
        *reinterpret_cast<uint64_t *>(o + 0x18) = 60;    // refresh
      }
      return 0;
    }
    case 0xa: {
      // Get flip status. arg[0x10] (s[2]) = out ptr, arg[0x18] (s[3]) = size
      // (0x48). The module copies it into SceVideoOutFlipStatus. Kernel field
      // layout (verified against the 11.00 sceVideoOutGetFlipStatus wrapper):
      //   [0x00] flipArg [0x10] count [0x18] processTime [0x20] tsc
      //   [0x28] currentBuffer [0x2c]+[0x34] flipPendingNum [0x30] gcQueueNum
      //   [0x38] submitTsc
      // Our flip/present is synchronous, so nothing is ever pending: report
      // count == an advancing flip count and pending == 0 so the runner's
      // "is a flip still queued?" checks let it submit the next frame.
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<uint8_t *>(s[2]);
        std::memset(o, 0, 0x48);
        uint64_t now = nowNs();
        *reinterpret_cast<int64_t *>(o + 0x00) = g_dceFlipArg.load();  // flipArg
        *reinterpret_cast<uint64_t *>(o + 0x10) =
            g_dceFlipCount.load() ? g_dceFlipCount.load() : flipCount();  // count
        *reinterpret_cast<uint64_t *>(o + 0x18) = now;            // processTime
        *reinterpret_cast<uint64_t *>(o + 0x20) = guestTsc();     // tsc
        // currentBuffer = the buffer the last submitted flip displays. A title
        // waits for this to equal the index it just flipped before reusing a
        // buffer; reporting it (vs a stuck 0) is what unblocks the per-frame wait.
        *reinterpret_cast<int32_t *>(o + 0x28) = (int32_t)g_dceCurrentBuffer.load();
        *reinterpret_cast<uint64_t *>(o + 0x38) = guestTsc();     // submitTsc
        // flipPendingNum / gcQueueNum left 0 (our flip is synchronous: none pending).
      }
      return 0;
    }
    case 0xb: {
      // Get vblank status. arg[0x10] (s[2]) = out ptr, arg[0x18] (s[3]) = size
      // (0x28). Kernel field layout: [0x00] count [0x08] processTime [0x10] tsc
      // [0x18] flags. count MUST advance in real time (see vblankCount) or the
      // runner's vsync busy-poll never returns -> the title hangs on frame 1.
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<uint8_t *>(s[2]);
        std::memset(o, 0, 0x28);
        uint64_t now = nowNs();
        *reinterpret_cast<uint64_t *>(o + 0x00) = vblankCount();  // count
        *reinterpret_cast<uint64_t *>(o + 0x08) = now;            // processTime
        *reinterpret_cast<uint64_t *>(o + 0x10) = guestTsc();     // tsc
        o[0x18] = 0;                                              // flags
      }
      return 0;
    }
    case 0xc: {
      // scaler-setup query (videoout service thread). out[0x00] != 0 only when a
      // NEW scaler config is pending; the title never reconfigures the scaler after
      // boot, so report "none pending" (all zero) -- a non-zero handle here makes
      // the service spin posting bogus scaler events. (NOT the flip-done path; that
      // is the EVFILT_DISPLAY/-13 event below.)
      if (plausiblePtr(s[2]))
        std::memset(reinterpret_cast<void *>(s[2]), 0, 0x40);
      return 0;
    }
    case 0x1f:  // NOT a valid kernel sub-op: the flip-control switch only
                 // covers 0..0x19, and 0x1f (31) falls to the EINVAL default
                 // (kernel jump table -> 0x826e9679). The "open-time capability
                 // header" reading was a soft-success invention; the retail sprx
                 // sees EINVAL here, so match it.
      return -SysError::eINVAL;
    default:
      // Other query/config sub-ops (1, 6, 0xc, ...). Soft-succeed without
      // touching arg fields whose role (pointer vs scalar) we haven't pinned.
      if (g_dceTrace())
        std::printf("[dce] 0xc0308203 sub-op %#llx -> 0\n",
                    (unsigned long long)s[0]);
      return 0;
    }
  }

  if (cmd == 0xc0308206 && data) {
    // Register one scanout buffer: {display handle, buffer index, GPU base}.
    // Preserve the address so a later GC submit-and-flip can present the render
    // target selected by the DCE flip rather than whichever RT was drawn last.
    if (s[1] < 16)
      g_dceScanoutBuffers[s[1]].store(s[2]);
    return 0;
  }

  if (cmd == 0xc0308207) {
    // Register one scanout buffer's format/tiling attribute. Input only; the
    // buffer GPU VA the module flips later comes from the game's own registered
    // address, not from here. Accept it.
    return 0;
  }

  if (cmd == 0xc0488204 && data) {
    // Submit flip (72-byte arg). Our flip is
    // synchronous, so record it as immediately complete: GetFlipStatus then
    // reports currentBuffer == the index the title just flipped. This is exactly
    // Undertale's documented blocker (the flip path dropped bufferIndex/flipArg);
    // it did NOT fix Doom64's separate ~1fps busy-wait, but it is correct.
    g_dceCurrentBuffer.store(static_cast<uint32_t>(s[1]));
    g_dceFlipArg.store(static_cast<int64_t>(s[3]));
    g_dceFlipCount.fetch_add(1);  // a per-flip count reported back in GetFlipStatus
    if (g_dceTrace())
      std::printf("[dce] submitFlip buf=%d flipArg=%#llx\n", (int)s[1],
                  (unsigned long long)s[3]);
    // Report success: arg[0x40] (s[8]) holds a pointer to the status out-slot
    // the kernel copyouts 8 bytes to (0 on clean success, 88 when a flip is
    // already pending). The caller checks it for 0x58 = ok.
    if (plausiblePtr(s[8]))
      *reinterpret_cast<uint64_t *>(s[8]) = 0x58;
    return 0;
  }

  if (g_dceTrace())
    std::printf("[dce] UNHANDLED ioctl %#x -> 0\n", cmd);
  return 0;
}
}  // namespace krnl
