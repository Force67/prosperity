/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdio>
#include <cstring>

#include "dce_dev.h"
#include "kern/proc.h"

namespace krnl {
dceDevice::dceDevice(proc *p) : device(p) {}

// The native backend runs syscall handlers on the guest stack, so the calling
// libSceVideoOut wrapper's return address sits somewhere up the stack. Host
// frame pointers may be omitted, so scan raw stack qwords for the first one
// landing in libSceVideoOut's .text and report it as base+offset. This pins the
// wrapper that issued each ioctl.
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
  for (int i = 0; i < 512; i++) {
    uintptr_t v = sp[i];
    if (v >= vbase && v < vbase + vsize) {
      std::printf("[dce]   caller libSceVideoOut+%#lx\n", v - vbase);
      return;
    }
  }
}

bool dceDevice::init(const char *, uint32_t, uint32_t) { return true; }

// A guest pointer is directly host-addressable here (in-process LLE). Guard
// dereferences to a sane userspace range so a stray field doesn't fault.
static bool plausiblePtr(uint64_t v) {
  return v >= 0x10000 && v < 0x0000800000000000ull;
}

// Dump an ioctl arg struct as u64s, chasing anything that looks like a pointer
// one level deep. This is how we reverse the (undocumented) DCE structs.
static void dumpStruct(const void *data, uint32_t len) {
  const uint64_t *q = static_cast<const uint64_t *>(data);
  uint32_t n = len / 8;
  for (uint32_t i = 0; i < n; i++) {
    std::printf("    [%u] %#018llx", i, (unsigned long long)q[i]);
    if (plausiblePtr(q[i])) {
      const uint64_t *p = reinterpret_cast<const uint64_t *>(q[i]);
      std::printf("  -> %#llx %#llx %#llx %#llx", (unsigned long long)p[0],
                  (unsigned long long)p[1], (unsigned long long)p[2],
                  (unsigned long long)p[3]);
    }
    std::printf("\n");
  }
}

int32_t dceDevice::ioctl(uint32_t cmd, void *data) {
  uint32_t len = (cmd >> 16) & 0x1FFF;
  uint32_t group = (cmd >> 8) & 0xff;
  uint32_t num = cmd & 0xff;
  const char *dir = (cmd & 0x80000000) ? ((cmd & 0x40000000) ? "INOUT" : "IN")
                                       : ((cmd & 0x40000000) ? "OUT" : "VOID");
  std::printf("[dce] ioctl cmd=%#x grp=%#x num=%#x len=%u %s data=%p\n", cmd,
              group, num, len, dir, data);
  printVideoOutCaller();
  if (data && len && len <= 0x200)
    dumpStruct(data, len);

  if (cmd == 0xc0308203 && data) {
    auto *s = static_cast<uint64_t *>(data);
    // 0xc0308203 is libSceVideoOut's multiplexed DCE control ioctl; s[0] selects
    // the sub-op. sceVideoOutOpen issues ops 0, 9 and 31 in sequence.
    switch (s[0]) {
    case 0:
      // Register a video-out handle. s[4] points to the caller's handle state
      // (input only); no output to fill.
      break;
    case 9:
      // Allocate scanout memory. The caller reads back *s[2] (offset) and *s[3]
      // (length), then mmaps the dce fd at that region (PROT 0x33 = CPU+GPU r/w)
      // for the framebuffer pool. Return one fixed 64 MiB region (room for
      // several 1080p buffers).
      if (plausiblePtr(s[2]))
        *reinterpret_cast<uint64_t *>(s[2]) = 0;  // offset
      if (plausiblePtr(s[3]))
        *reinterpret_cast<uint64_t *>(s[3]) = 0x4000000;  // length (64 MiB)
      break;
    case 0x1f:
      // Register a flip-label region: s[2] is a count, s[3] points into the
      // scanout memory. Input only; no output to fill.
      break;
    default:
      std::printf("[dce] unhandled 0xc0308203 sub-op %#llx\n",
                  (unsigned long long)s[0]);
      break;
    }
  }
  return 0;
}
}  // namespace krnl
