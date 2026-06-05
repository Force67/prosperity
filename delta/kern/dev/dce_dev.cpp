/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dce_dev.h"
#include "kern/proc.h"
#include "kern/lv2/sys_mem.h"

namespace krnl {
dceDevice::dceDevice(proc *p) : device(p) {}

static bool g_dceTrace() {
  static const bool on = std::getenv("DELTA_DCE_TRACE") != nullptr;
  return on;
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
  // every sub-op 9 / 0xc0588212 carves a slice and the module mmaps it back via
  // map(offset). 512 MiB covers several 1080p buffer sets + control blocks.
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
  // The module mmaps the dce fd at an offset sub-op 9 handed back. Return the
  // matching slice of the pool so it gets real, zeroed, shared memory (not the
  // uninitialised anonymous fallback that made it size buffers from garbage).
  if (poolBase && offset + size <= poolSize)
    return poolBase + offset;
  return reinterpret_cast<uint8_t *>(-1);
}

// 0xc0308203 sub-op handlers. The 48-byte arg is `s[0..5]` (u64s): s[0] = sub-op.
// Layouts verified against the 11.00 libSceVideoOut.sprx callers.
int32_t dceDevice::ioctl(uint32_t cmd, void *data) {
  if (g_dceTrace()) {
    uint32_t len = (cmd >> 16) & 0x1FFF;
    std::printf("[dce] ioctl cmd=%#x len=%u data=%p\n", cmd, len, data);
    printVideoOutCaller();
    if (data && len && len <= 0x200)
      dumpStruct(data, len);
  }

  auto *s = static_cast<uint64_t *>(data);

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
      // Allocate scanout pool. arg[0x10]/arg[0x18] (s[2]/s[3]) point at the
      // caller's offset/size out-slots; it then mmaps the dce fd at that offset
      // for `size` bytes. Carve a real pool slice so the mmap maps live memory.
      uint64_t want = (plausiblePtr(s[3]) && *reinterpret_cast<uint64_t *>(s[3]))
                          ? *reinterpret_cast<uint64_t *>(s[3])
                          : 0x4000000;  // 64 MiB default
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
    case 0x1f:
      // Open-time capability/flip-control header: arg[0x10] (s[2]) = size (0xc),
      // arg[0x18] (s[3]) = pointer into the pool the kernel writes 12 bytes to.
      // Zero it (defined state); the capability predicates tolerate zeros.
      if (plausiblePtr(s[3]))
        std::memset(reinterpret_cast<void *>(s[3]), 0, 0xc);
      return 0;
    default:
      // Other query/config sub-ops (1, 6, 0xc, ...). Soft-succeed without
      // touching arg fields whose role (pointer vs scalar) we haven't pinned.
      if (g_dceTrace())
        std::printf("[dce] 0xc0308203 sub-op %#llx -> 0\n",
                    (unsigned long long)s[0]);
      return 0;
    }
  }

  if (cmd == 0xc0308206) {
    // Register one scanout buffer (GPU base path). The module already resolved
    // and validated the buffer's GPU VA via sceKernelVirtualQuery; accept it.
    return 0;
  }

  if (cmd == 0xc0308207) {
    // Register one scanout buffer's format/tiling attribute. Input only; the
    // buffer GPU VA the module flips later comes from the game's own registered
    // address, not from here. Accept it.
    return 0;
  }

  if (cmd == 0xc0588212 && data) {
    // Second scanout-memory allocator (cursor / close path), 88-byte arg. Same
    // shape as sub-op 9: offset/size out-slots a few words in. Carve a slice.
    uint64_t want = 0x100000;
    uint64_t off = poolAlloc(want);
    if (plausiblePtr(s[2]))
      *reinterpret_cast<uint64_t *>(s[2]) = off;
    if (plausiblePtr(s[3]))
      *reinterpret_cast<uint64_t *>(s[3]) = want;
    return 0;
  }

  if (cmd == 0xc0488204 && data) {
    // Submit flip (72-byte arg). Stage 1: report success so the module's flip
    // wrapper doesn't error. arg[0x40] (s[8]) points at a status out-slot the
    // caller checks for 0x58 = ok. (Presenting the frame comes next.)
    if (plausiblePtr(s[8]))
      *reinterpret_cast<uint64_t *>(s[8]) = 0x58;
    return 0;
  }

  if (g_dceTrace())
    std::printf("[dce] UNHANDLED ioctl %#x -> 0\n", cmd);
  return 0;
}
}  // namespace krnl
