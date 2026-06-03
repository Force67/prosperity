
// Copyright (C) Force67 2019

#include <base.h>
#include <cstdio>
#include <cstring>
#include "gc_dev.h"
#include "kern/proc.h"
#include "kern/lv2/sys_mem.h"

namespace krnl {
gcDevice::gcDevice(proc *p) : device(p) {}

bool gcDevice::init(const char *, uint32_t, uint32_t) { return true; }

// SCOUT: scan the stack for the first return address landing in any guest
// module's .text and report it as <module>+offset, to pin which guest wrapper
// issued each gc ioctl (the native backend runs handlers on the guest stack).
static void printGuestCaller() {
  auto *proc = proc::getActive();
  if (!proc)
    return;
  auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
  for (int i = 0; i < 512; i++) {
    uintptr_t v = sp[i];
    for (auto &m : proc->getModuleList()) {
      auto &mi = m->getInfo();
      auto base = (uintptr_t)mi.textSeg.addr;
      if (base && v >= base && v < base + mi.textSeg.size) {
        std::printf("[gc]   caller %s+%#lx\n", mi.name.c_str(), v - base);
        return;
      }
    }
  }
}

/* gc_ioctl */
int32_t gcDevice::ioctl(uint32_t cmd, void *data) {
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
  printf("[gc] UNHANDLED ioctl(%x) data=%p\n", cmd, data);
  return 0;
}

/*map to gfx memory*/
uint8_t *gcDevice::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  //__debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
