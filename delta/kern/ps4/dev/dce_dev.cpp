/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <atomic>
#include "base/arch.h"
#include <base.h>
#include <base/logging.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dce_dev.h"
#include "kern/proc.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDceTrace, "DELTA_DCE_TRACE", false);
}  // namespace

namespace krnl {
dceDevice::dceDevice(objectTable &objects) : device(objects) {}

// Monotonic ns timestamp + a wall-clock ~60 Hz vblank counter. The GameMaker runner
// (and sceVideoOutWaitVblank) busy-poll VblankStatus until the count advances,
// so it MUST tick in real time despite no display hardware, else the title spins
// forever on frame 1.
static u64 nowNs() {
  using namespace std::chrono;
  return static_cast<u64>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
static u64 vblankCount() {
  using namespace std::chrono;
  static const auto start = steady_clock::now();
  auto ns = duration_cast<nanoseconds>(steady_clock::now() - start).count();
  return static_cast<u64>(ns / 16666667);  // ~59.94 Hz
}
// A TSC value in the same domain/rate as the guest's rdtsc, so flip/vblank-status
// tsc fields stay comparable with rdtsc the title reads itself. On native that's
// the raw host TSC; on FEX use the 1.6 GHz wall clock the emulated rdtsc matches.
static u64 guestTsc() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  return nowNs() * 16 / 10;  // ns -> 1.6 GHz ticks
#endif
}

// The last flip submitted via 0xc0488204. A title flips buffer N then polls
// GetFlipStatus until currentBuffer == N; reporting 0 always made flips to
// buffers 1/2 spin to a ~1s timeout (Doom64 at ~1fps). Record and report it.
static std::atomic<u32> g_dceCurrentBuffer{0};
static std::atomic<i64> g_dceFlipArg{0};
static std::atomic<u64> g_dceFlipCount{0};
static std::atomic<u64> g_dceScanoutBuffers[16]{};

u32 dceCurrentBuffer() { return g_dceCurrentBuffer.load(); }

i64 dceCurrentFlipArg() { return g_dceFlipArg.load(); }

u64 dceScanoutBuffer(u32 index) {
  return index < 16 ? g_dceScanoutBuffers[index].load() : 0;
}

static bool g_dceTrace() {
  return kDceTrace;
}

// Scan the raw stack for the first return address in libSceVideoOut's .text (the
// native backend runs handlers on the guest stack) to pin which wrapper issued
// each ioctl.
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
      BASE_LOGI("dce", "  caller libSceVideoOut+{:#x}", off);
      lastoff = off;
      shown++;
    }
  }
}

bool dceDevice::init(const char *, u32, u32) { return true; }

// Guard dereferences to a sane userspace range. NB: under FEX the guest stack is a
// host mmap up at 0xffff_xxxx_xxxx, so real libSceVideoOut passes out-slot
// pointers above the old 0x8000_0000_0000 ceiling; accept the full 48-bit user
// range or every stack out-slot write is silently dropped (open then mmaps an
// uninitialised offset/size and fails).
static bool plausiblePtr(u64 v) {
  return v >= 0x10000 && v < 0x0001000000000000ull;
}

// Dump an ioctl arg struct as u64s. Just the values: do NOT chase "pointers"
// (many fields are two packed u32s like {pitch, height} that look like a high
// address but aren't, and dereferencing them faults). DELTA_DCE_TRACE.
static void dumpStruct(const void *data, u32 len) {
  const u64 *q = static_cast<const u64 *>(data);
  u32 n = len / 8;
  for (u32 i = 0; i < n; i++)
    BASE_LOGI("dce", "    [{}] {:#018x}", i, (unsigned long long)q[i]);
}

u64 dceDevice::poolAlloc(u64 bytes) {
  // The scanout/control pool. One big guest-addressable region, bump-allocated;
  // every sub-op 9 carve is mmap'd back via map(offset). 512 MiB covers several
  // 1080p buffer sets + control blocks.
  constexpr u64 kPool = 512ull * 1024 * 1024;
  if (!poolBase) {
    poolBase = allocLowGuest(kPool);
    if (!poolBase)
      return UINT64_MAX;
    poolSize = kPool;
  }
  bytes = (bytes + 0x3FFF) & ~u64(0x3FFF);
  if (poolUsed + bytes > poolSize)
    return UINT64_MAX;
  u64 off = poolUsed;
  poolUsed += bytes;
  return off;
}

u8 *dceDevice::map(void *, size_t size, u32, u32, size_t offset) {
  // The kernel only maps /dev/dce at offsets < 0x8000 (physical flip-target status
  // page; offset/0x4000 picks the target). Our libSceVideoOut maps what sub-op 9
  // hands back, whose offsets exceed that window, so substitute a slice of the
  // scanout pool: real, zeroed, shared, CP-usable memory in the same shape.
  if (poolBase && offset + size <= poolSize)
    return poolBase + offset;
  return reinterpret_cast<u8 *>(-1);
}

// 11.00 dce ioctl dispatch: scanin ioctls are system-credential-gated (a game gets
// EINVAL); 0xC0308203 -> flip control (sub-ops 0..0x19), 0xC0308206/07 ->
// register buffer(/attr), 0xC0488204 -> submit flip, 0xc0588212 not in the
// dispatch (EINVAL). Layouts verified against the 11.00 libSceVideoOut callers.
i32 dceDevice::ioctl(u32 cmd, void *data) {
  if (g_dceTrace()) {
    u32 len = (cmd >> 16) & 0x1FFF;
    BASE_LOGI("dce", "ioctl cmd={:#x} len={} data={:p}", cmd, len, data);
    printVideoOutCaller();
    if (data && len && len <= 0x200)
      dumpStruct(data, len);
  }

  auto *s = static_cast<u64 *>(data);

  // Scanin ioctls + the non-existent 0xc0588212: EINVAL from a game context, which
  // is what a game's libSceVideoOut is built against.
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
        *reinterpret_cast<u64 *>(s[4]) = nextHandle++;
      return 0;
    case 9: {
      // Sub-op 9 (scanout-pool offset query): the kernel writes offset=0x4000 and
      // size=(bufferCount<<14) to s[2]/s[3] and the module mmaps fd@offset for that
      // size; we hand back a pool slice instead (same shape, real backing). s[3] is
      // a pure OUT slot on some callers (the open path passes it uninitialised), so
      // treat *s[3] as a requested size only when sane, else default; a bogus huge
      // size made poolAlloc fail ENOMEM and left the slots uninitialised.
      u64 reqd = plausiblePtr(s[3]) ? *reinterpret_cast<u64 *>(s[3]) : 0;
      u64 want = (reqd > 0 && reqd <= 0x10000000) ? reqd  // <= 256 MiB
                                                       : 0x4000000;  // 64 MiB
      u64 off = poolAlloc(want);
      if (off == UINT64_MAX)
        return 12;  // ENOMEM
      if (plausiblePtr(s[2]))
        *reinterpret_cast<u64 *>(s[2]) = off;
      if (plausiblePtr(s[3]))
        *reinterpret_cast<u64 *>(s[3]) = want;
      return 0;
    }
    case 0x19: {
      // Resolution status. arg[0x10] (s[2]) = out ptr, arg[0x18] (s[3]) = out
      // size (0x30). The module copies width/height/pane (u32 @0/4/8/0xc),
      // a flag byte @0x10, refresh @0x18, aspect @0x20. Report 1920x1080.
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<u8 *>(s[2]);
        std::memset(o, 0, 0x30);
        *reinterpret_cast<u32 *>(o + 0x00) = 1920;  // width
        *reinterpret_cast<u32 *>(o + 0x04) = 1080;  // height
        *reinterpret_cast<u32 *>(o + 0x08) = 1920;  // paneWidth
        *reinterpret_cast<u32 *>(o + 0x0c) = 1080;  // paneHeight
        *reinterpret_cast<u64 *>(o + 0x18) = 60;    // refresh
      }
      return 0;
    }
    case 0xa: {
      // Get flip status: s[2] = out ptr (0x48). Kernel layout: [0x00] flipArg
      // [0x10] count [0x18] processTime [0x20] tsc [0x28] currentBuffer
      // [0x2c]+[0x34] flipPendingNum [0x30] gcQueueNum [0x38] submitTsc. Our flip is
      // synchronous, so report advancing count, pending 0.
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<u8 *>(s[2]);
        std::memset(o, 0, 0x48);
        u64 now = nowNs();
        *reinterpret_cast<i64 *>(o + 0x00) = g_dceFlipArg.load();  // flipArg
        *reinterpret_cast<u64 *>(o + 0x10) =
            g_dceFlipCount.load() ? g_dceFlipCount.load() : flipCount();  // count
        *reinterpret_cast<u64 *>(o + 0x18) = now;            // processTime
        *reinterpret_cast<u64 *>(o + 0x20) = guestTsc();     // tsc
        // currentBuffer = the buffer the last submitted flip displays. A title
        // waits for this to equal the index it just flipped before reusing a
        // buffer; reporting it (vs a stuck 0) is what unblocks the per-frame wait.
        *reinterpret_cast<i32 *>(o + 0x28) = (i32)g_dceCurrentBuffer.load();
        *reinterpret_cast<u64 *>(o + 0x38) = guestTsc();     // submitTsc
        // flipPendingNum / gcQueueNum left 0 (our flip is synchronous: none pending).
      }
      return 0;
    }
    case 0xb: {
      // Get vblank status: s[2] = out ptr (0x28): [0x00] count [0x08] processTime
      // [0x10] tsc [0x18] flags. count must advance in real time (see vblankCount).
      if (plausiblePtr(s[2])) {
        auto *o = reinterpret_cast<u8 *>(s[2]);
        std::memset(o, 0, 0x28);
        u64 now = nowNs();
        *reinterpret_cast<u64 *>(o + 0x00) = vblankCount();  // count
        *reinterpret_cast<u64 *>(o + 0x08) = now;            // processTime
        *reinterpret_cast<u64 *>(o + 0x10) = guestTsc();     // tsc
        o[0x18] = 0;                                              // flags
      }
      return 0;
    }
    case 0xc: {
      // Scaler-setup query: out[0x00] != 0 only when a NEW scaler config is pending;
      // titles never reconfigure after boot, so report none (a non-zero handle makes
      // the service spin posting bogus events). Flip-done is the EVFILT_DISPLAY event.
      if (plausiblePtr(s[2]))
        std::memset(reinterpret_cast<void *>(s[2]), 0, 0x40);
      return 0;
    }
    case 0x1f:  // NOT a valid kernel sub-op: the flip-control switch only
                 // Sub-ops 0..0x19; 0x1f falls to the EINVAL default (kernel jump table
                 // -> 0x826e9679). The "open-time capability header" was a soft-success
                 // invention; retail sees EINVAL here.
      return -SysError::eINVAL;
    default:
      // Other query/config sub-ops (1, 6, 0xc, ...). Soft-succeed without
      // touching arg fields whose role (pointer vs scalar) we haven't pinned.
      if (g_dceTrace())
        BASE_LOGI("dce", "0xc0308203 sub-op {:#x} -> 0",
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
    // Submit flip (72-byte arg): record it as immediately complete so GetFlipStatus
    // reports currentBuffer == the just-flipped index (Undertale's documented
    // blocker; not Doom64's separate ~1fps busy-wait).
    g_dceCurrentBuffer.store(static_cast<u32>(s[1]));
    g_dceFlipArg.store(static_cast<i64>(s[3]));
    g_dceFlipCount.fetch_add(1);  // a per-flip count reported back in GetFlipStatus
    if (g_dceTrace())
      BASE_LOGI("dce", "submitFlip buf={} flipArg={:#x}", (int)s[1],
                (unsigned long long)s[3]);
    // Report success: arg[0x40] (s[8]) holds a pointer to the status out-slot
    // the kernel copyouts 8 bytes to (0 on clean success, 88 when a flip is
    // already pending). The caller checks it for 0x58 = ok.
    if (plausiblePtr(s[8]))
      *reinterpret_cast<u64 *>(s[8]) = 0x58;
    return 0;
  }

  if (g_dceTrace())
    BASE_LOGI("dce", "UNHANDLED ioctl {:#x} -> 0", cmd);
  return 0;
}
}  // namespace krnl
