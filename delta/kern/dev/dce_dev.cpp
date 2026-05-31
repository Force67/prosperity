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

namespace krnl {
dceDevice::dceDevice(proc *p) : device(p) {}

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
  if (data && len && len <= 0x200)
    dumpStruct(data, len);

  if (cmd == 0xc0308203 && data) {
    auto *s = static_cast<uint64_t *>(data);
    // op 9: allocate display/scanout memory. libSceVideoOut reads back
    // *struct[2]=offset and *struct[3]=length and mmaps that region (PROT
    // 0x33 = CPU+GPU r/w). Return a sane fixed region so the mmap succeeds
    // (offset 0, 64 MiB -- room for several 1080p framebuffers).
    if (s[0] == 9) {
      if (plausiblePtr(s[2]))
        *reinterpret_cast<uint64_t *>(s[2]) = 0;  // offset
      if (plausiblePtr(s[3]))
        *reinterpret_cast<uint64_t *>(s[3]) = 0x4000000;  // length (64 MiB)
    }
  }
  return 0;
}
}  // namespace krnl
