
// Copyright (C) Force67 2019

#include <base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "gc_dev.h"
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_event.h"
#include "kern/ps4/lv2/sys_mem.h"

// LLE GPU submit bridge (delta_runtime). The real libSceGnmDriver.sprx submits
// PM4 through these ioctls; forward the descriptor array to the GPU command
// processor. See prosperity_gc_submit in libSceGnmDriver.cpp.
extern "C" void prosperity_gc_submit(const void *descArray, uint32_t descCount);
extern "C" void prosperity_gc_flip(int displayBufferIndex, int64_t flipArg);
// PS5 AGC submit bridge (delta_gpu, gpu/ps5): the real libSceAgcDriver.sprx
// submits its DCB through the /dev/gc AGC ioctl 0xc0408121; forward it to the
// PS5 command processor.
extern "C" void prosperity_agc_submit(uint64_t dcbBase, uint32_t sizeBytes);

namespace krnl {
gcDevice::gcDevice(proc *p) : device(p) {}

static bool isPs5() {
  auto *p = proc::getActive();
  return p && p->getPlatform() == proc::platform::ps5;
}

bool gcDevice::init(const char *, uint32_t, uint32_t) { return true; }

static void completeFlipLabels(uint64_t flipPtr) {
  if (!flipPtr)
    return;

  auto *p = reinterpret_cast<uint32_t *>(flipPtr);
  // Gnm prepareFlip emits a PM4-like EOP-label packet:
  //   C0038000 addr_lo addr_hi value_lo value_hi
  // Complete it synchronously because the CPU-side emulated submit already
  // finished all draws before returning.
  if (p[0] == 0xC0038000u) {
    uint64_t addr = (static_cast<uint64_t>(p[2] & 0xFF) << 32) | p[1];
    uint64_t value = static_cast<uint64_t>(p[3]) |
                     (static_cast<uint64_t>(p[4]) << 32);
    if (addr) {
      *reinterpret_cast<uint64_t *>(addr) = value;
      static const bool trace = std::getenv("DELTA_GC_FLIP") != nullptr;
      if (trace)
        std::printf("[gc]   flip label [%#lx] = %#lx\n",
                    static_cast<unsigned long>(addr),
                    static_cast<unsigned long>(value));
    }
  }
}

// SCOUT (DELTA_GC_CALLER): scan the stack for the first return address landing in
// any guest module's .text and report it as <module>+offset, to pin which guest
// wrapper issued each gc ioctl (the native backend runs handlers on the guest
// stack). Off by default: the submit ioctls fire 60+/frame and the scan is slow.
static void printGuestCaller() {
  static const bool on = std::getenv("DELTA_GC_CALLER") != nullptr;
  if (!on)
    return;
  auto *proc = proc::getActive();
  if (!proc)
    return;
  auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
  int printed = 0;
  uintptr_t last = 0;
  for (int i = 0; i < 512 && printed < 6; i++) {
    uintptr_t v = sp[i];
    if (v == last)
      continue;
    for (auto &m : proc->getModuleList()) {
      auto &mi = m->getInfo();
      auto base = (uintptr_t)mi.textSeg.addr;
      if (base && v >= base && v < base + mi.textSeg.size) {
        std::printf("[gc]   caller[%d] %s+%#lx\n", printed, mi.name.c_str(),
                    v - base);
        last = v;
        printed++;
        break;
      }
    }
  }
}

/* gc_ioctl */
int32_t gcDevice::ioctl(uint32_t cmd, void *data) {
  // Graphics command submission (the LLE path: real libSceGnmDriver.sprx). The
  // arg's descriptor array is an array of 16-byte PM4 INDIRECT_BUFFER packets;
  // forward it to the GPU command processor. Handle these first (and without the
  // stack-scan SCOUT) since they fire 60+ times per frame.
  switch (cmd) {
  case 0xC0108102: {  // gc submit: {u32 a0, u32 count, u64 descPtr}
    struct argl {
      uint32_t a0;
      uint32_t count;
      uint64_t descPtr;
    };
    auto *a = static_cast<argl *>(data);
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    return 0;
  }
  case 0xC018810A: {  // gc submit (variant): {u32 a0, u32 count, u32 a2, _, u64 ptr}
    struct argl {
      uint32_t a0;
      uint32_t count;
      uint32_t a2;
      uint32_t pad;
      uint64_t descPtr;
    };
    auto *a = static_cast<argl *>(data);
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    return 0;
  }
  case 0xC020810C: {  // gc submit-and-flip: adds u64 flipPtr, u32 flag
    struct argl {
      uint32_t a0;
      uint32_t count;
      uint64_t descPtr;
      uint64_t flipPtr;
      uint32_t flag;
    };
    auto *a = static_cast<argl *>(data);
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    // SCOUT (DELTA_GC_FLIP): the flip target buffer/arg live behind flipPtr; dump
    // it the first few times so we can decode the layout. Until then present the
    // last RT (index -1) and let the videoout pump post the flip event.
    static const bool flipTrace = std::getenv("DELTA_GC_FLIP") != nullptr;
    static int flipDumps = 0;
    if (flipTrace && flipDumps < 8) {
      flipDumps++;
      printf("[gc] flip a0=%x count=%u descPtr=%lx flipPtr=%lx flag=%x\n", a->a0,
             a->count, (unsigned long)a->descPtr, (unsigned long)a->flipPtr,
             a->flag);
      if (a->flipPtr) {
        auto *fp = reinterpret_cast<const uint32_t *>(a->flipPtr);
        printf("[gc]   flipPtr[0..7]: %x %x %x %x %x %x %x %x\n", fp[0], fp[1],
               fp[2], fp[3], fp[4], fp[5], fp[6], fp[7]);
      }
    }
    completeFlipLabels(a->flipPtr);
    prosperity_gc_flip(-1, 0);
    noteFlip();  // advance the flip count + post the display event for pacing
    return 0;
  }
  case 0xC0088101:  // "switch buffer": ring double-buffer handoff. The driver
                    // issues this per submit (60+/frame) and ignores the args;
                    // our submit is synchronous so there is no ring to switch.
                    // Handle it here, silently -- it was falling through to the
                    // UNHANDLED logger below, and an unbuffered printf per submit
                    // (stdbuf -e0) throttled the whole render loop to ~1 fps.
    return 0;
  case 0xC0048116: {  // "submit done?" status word; hot (polled per submit).
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  }
  case 0xC0048114: {  // GPU status poll. The GnmDriver wrapper (libSceGnmDriver
                      // +0x5fd0) zeroes the 4-byte slot, issues this, and returns
                      // (ret == 0) -- it never reads the output back, so only the
                      // success return matters. A title's render thread polls it
                      // in a tight loop; handle it here (return success, zero the
                      // slot) so it stops falling through to the UNHANDLED logger.
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  }
  // PS5-only AGC /dev/gc protocol (libSceAgc/libSceAgcDriver). These command
  // numbers are unused on PS4; gate on the platform so the PS4 path is untouched.
  case 0x40048135:  // AGC/Gnm query: OUT dword (submit/queue id). The driver just
                    // stores the value; 0 is accepted.
    if (isPs5()) {
      if (data)
        *static_cast<uint32_t *>(data) = 0;
      return 0;
    }
    break;
  case 0xC004812E:  // AGC init: INOUT dword. 0 tells AgcDriver to map its own
                    // submit doorbell (which then succeeds); non-zero skips it.
    if (isPs5()) {
      if (data)
        *static_cast<uint32_t *>(data) = 0;
      return 0;
    }
    break;
  case 0xC0408121: {  // AGC submit (INOUT, 64 bytes). arg+0x30 = DCB GPU base
                      // address, arg+0x38 = size in bytes. Forward the DCB to the
                      // PS5 command processor (PM4 walk + completion labels), then
                      // zero the arg per the OUT convention.
    if (!isPs5())
      break;
    if (data) {
      auto *a = static_cast<uint8_t *>(data);
      uint64_t base = 0;
      uint32_t size = 0;
      std::memcpy(&base, a + 0x30, 8);
      std::memcpy(&size, a + 0x38, 4);
      static const bool trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int dumps = 0;
      if (trace && dumps < 4) {
        dumps++;
        auto *w = reinterpret_cast<uint32_t *>(a);
        std::printf("[agc] submit arg[0..15]:");
        for (int k = 0; k < 16; k++)
          std::printf(" %08x", w[k]);
        std::printf("\n[agc]   base=%#lx size=%u\n", (unsigned long)base, size);
      }
      prosperity_agc_submit(base, size);
      std::memset(a, 0, 64);
    }
    return 0;
  }
  }

  printGuestCaller();
  switch (cmd) {
  case 0xC00C8110: {

    struct argl {
      uint32_t unknown_0;
      uint32_t unknown_4;
      uint32_t unknown_8;
    };
    auto args = reinterpret_cast<argl *>(data);
    printf("gc ioctl(%x): %x, %x, %x\n", cmd, args->unknown_0, args->unknown_4,
           args->unknown_8);
    return 0;
  }
  case 0xC010810B: {
    struct argl {
      uint32_t cumask0;
      uint32_t cumask1;
      uint32_t cumask2;
      uint32_t cumask3;
    };

    /*idk what the proper value would be*/
    auto se0 = (uint16_t)1024 >> 6;
    auto se1 = (1024 >> 16) & 0x3FF;

    auto args = reinterpret_cast<argl *>(data);
    args->cumask0 = se0;
    args->cumask1 = se0;
    args->cumask2 = se1;
    args->cumask3 = se1;

    return 0;
  }
  case 0xC008811B: {
    // GNM "trace/info init": the driver passes an 8-byte out slot and stores the
    // returned pointer into its global logging-info pointer (Gnm vaddr 0x100e8),
    // then dereferences it on every submit (`cmp dword[ptr],0` = trace-enable).
    // A bogus value here makes that deref fault. Hand back a real, zeroed guest
    // struct so the deref reads trace-disabled (0) and the logger is a no-op.
    static uint8_t *traceInfo = nullptr;
    if (!traceInfo)
      traceInfo = allocLowGuest(0x100);  // zero-filled; [+0] = trace flag (off)
    auto args = static_cast<uint64_t *>(data);
    *args = reinterpret_cast<uint64_t>(traceInfo);
    printf("gc ioctl(%x): trace-info -> %p\n", cmd, (void *)traceInfo);
    return 0;
  }
  case 0xC0848119: {
    struct argl {
      uint32_t unknown_00;
      uint32_t unknown_04;
      uint32_t unknown_08;
      uint32_t unknown_0C;
      uint8_t unknown_10[112];
      uint32_t unknown_80;
    };
    auto args = static_cast<argl *>(data);
    printf("gc ioctl(%x): %x, %x, %x, %x, %x\n", cmd, args->unknown_00,
           args->unknown_04, args->unknown_08, args->unknown_0C,
           args->unknown_80);
    return 0;
  }
  }

  // SCOUT: log unknown gc ioctls and soft-succeed so the boot keeps advancing
  // instead of trapping. Lets us discover the sequence GNM actually issues.
  // Rate-limited: an unknown ioctl in the per-submit hot path would otherwise
  // flood unbuffered stderr and stall the render loop.
  static const bool gcTrace = std::getenv("DELTA_GC_TRACE") != nullptr;
  static int unhandledLogged = 0;
  if (gcTrace || unhandledLogged < 32) {
    unhandledLogged++;
    printf("[gc] UNHANDLED ioctl(%x) data=%p\n", cmd, data);
  }
  // Zero the output buffer of an unhandled OUT/INOUT ioctl. The driver reads the
  // buffer back as a query result (capability counts, status words, etc.); left
  // uninitialised it returns stack/heap garbage, which the engine then trusts --
  // e.g. a bogus huge "format count" that overruns a fixed table and smashes the
  // stack. Zero is the benign "nothing/idle/none" answer (matches the explicit
  // 0x16 submit-done handler). Length is encoded in the ioctl command (FreeBSD
  // IOCPARM_LEN). Only touch OUT ioctls (bit 0x40000000).
  if (data && (cmd & 0x40000000u)) {
    uint32_t len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

/*map to gfx memory*/
uint8_t *gcDevice::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  //__debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
