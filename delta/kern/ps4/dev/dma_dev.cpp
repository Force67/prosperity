
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "dma_dev.h"
#include "kern/ps4/lv2/sys_mem.h"
#include "kern/proc.h"

namespace krnl {
dmaDevice::dmaDevice(proc *p) : device(p) {}

// PS4 direct ("physical") memory model.
//
// A title allocates a physical-offset range with sceKernelAllocateDirectMemory
// (ioctl 0xC0288001), then maps it into the virtual address space with
// sceKernelMapDirectMemory (ioctl 0x80108002), supplying the VA it already
// reserved. We don't model a real GPU physical pool, since the renderer drives
// the GPU by the virtual addresses it maps and the physical offset is only
// bookkeeping. The offset still has to be unique, non-zero and aligned. If it
// collapses to 0 on every allocation, the title's own allocator sees overlaps
// and the dependent subsystem (PT's render device) refuses to initialise.
//
// GetDirectMemorySize (ioctl 0x4008800A) is the search ceiling the title passes
// back into AllocateDirectMemory, so the old 1 KiB stub made every real
// allocation impossible.
namespace {
// PS4 user-accessible direct memory is roughly 4.5 to 5 GiB depending on the
// title budget. Report a flat large pool and bump offsets from a non-zero base
// so a test for "offset 0 means invalid" still holds.
constexpr uint64_t kDmemTotal = 0x300000000ull;  // 12 GiB (SOTTR working set)
std::atomic<uint64_t> g_dmemNext{0x10000000ull};  // first free physical offset

// Record of each direct-memory reservation so GetDirectMemoryType (ioctl
// 0xC0208004) can answer "which region owns this physical offset, and of what
// type". The renderer queries the regions it just allocated and refuses to
// initialise if they come back as a zero-length, type-0 hole.
struct DmemRegion {
  uint64_t start, end;
  uint32_t memType;
};
std::mutex g_dmemMutex;
std::vector<DmemRegion> g_dmemRegions;
}  // namespace

/* dmem_ioctl */
int32_t dmaDevice::ioctl(uint32_t cmd, void *data) {
  static const bool trace = std::getenv("DELTA_DMEM_TRACE") != nullptr;
  if (trace) {
    auto *q = static_cast<uint64_t *>(data);
    std::fprintf(stderr,
                 "[dmem-ioctl] cmd=%#x data=%p [%#llx %#llx %#llx %#llx %#llx]\n",
                 cmd, data, q ? (unsigned long long)q[0] : 0ull,
                 q ? (unsigned long long)q[1] : 0ull,
                 q ? (unsigned long long)q[2] : 0ull,
                 q ? (unsigned long long)q[3] : 0ull,
                 q ? (unsigned long long)q[4] : 0ull);
  }
  switch (cmd) {
  case 0x4008800A: {
    // GetDirectMemorySize: total physical pool available to the title.
    *static_cast<uint64_t *>(data) = kDmemTotal;
    return 0;
  }
  case 0xC0288001: {
    // AllocateDirectMemory: struct = [searchStart, searchEnd, len, align,
    // memType]; on success write the chosen physical offset back into [0].
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    uint64_t len = a[2];
    uint64_t align = a[3] ? a[3] : 0x4000;
    if (len == 0)
      return -1;
    // SCOUT (DELTA_DMEM_CALLER): on native the handler runs on the guest stack,
    // so scan it for return addresses in a loaded module's .text to pin which
    // guest code reserved this pool (e.g. the CPU heap's len constant).
    if (std::getenv("DELTA_DMEM_CALLER")) {
      std::printf("[dmem-alloc] len=%#llx memType=%#llx align=%#llx caller-chain:\n",
                  (unsigned long long)len, (unsigned long long)a[4],
                  (unsigned long long)align);
      auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
      auto *pr = proc::getActive();
      int shown = 0;
      for (int i = 0; i < 4096 && shown < 12; i++) {
        uintptr_t v = sp[i];
        if (!pr) break;
        for (auto &m : pr->getModuleList()) {
          auto &mi = m->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && v >= (uintptr_t)t && v < (uintptr_t)t + mi.textSeg.size) {
            std::printf("  sp+%-4x %s+%#lx\n", i * 8, mi.name.c_str(),
                        v - (uintptr_t)t);
            shown++;
            break;
          }
        }
      }
    }
    // Bump-allocate an aligned offset, honoring a non-zero searchStart floor.
    uint64_t off;
    for (;;) {
      uint64_t cur = g_dmemNext.load(std::memory_order_relaxed);
      uint64_t base = (cur + (align - 1)) & ~(align - 1);
      if (a[0] > base)
        base = (a[0] + (align - 1)) & ~(align - 1);
      if (g_dmemNext.compare_exchange_weak(cur, base + len,
                                            std::memory_order_relaxed)) {
        off = base;
        break;
      }
    }
    a[0] = off;  // physical offset out
    {
      std::lock_guard<std::mutex> lk(g_dmemMutex);
      g_dmemRegions.push_back({off, off + len, static_cast<uint32_t>(a[4])});
    }
    return 0;
  }
  case 0xC0288011: {
    // AllocateMainDirectMemory: struct = [offset(out), _, len, align, memType].
    // Same physical bump-allocator as AllocateDirectMemory, but the search range
    // is the whole pool (no start/end); the chosen physical offset goes back into
    // [0]. Left unhandled it fell through to `return 0` without writing an offset,
    // so every reservation aliased physical offset 0.
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    uint64_t len = a[2];
    uint64_t align = a[3] ? a[3] : 0x4000;
    if (len == 0)
      return -1;
    uint64_t off;
    for (;;) {
      uint64_t cur = g_dmemNext.load(std::memory_order_relaxed);
      uint64_t base = (cur + (align - 1)) & ~(align - 1);
      if (g_dmemNext.compare_exchange_weak(cur, base + len,
                                            std::memory_order_relaxed)) {
        off = base;
        break;
      }
    }
    a[0] = off;
    {
      std::lock_guard<std::mutex> lk(g_dmemMutex);
      g_dmemRegions.push_back({off, off + len, static_cast<uint32_t>(a[4])});
    }
    return 0;
  }
  case 0xC0208016: {
    // AvailableDirectMemorySize: struct = [searchStart, searchEnd, align, _].
    // The kernel writes the largest free size at/after searchStart into [0] and
    // its offset into [3] (layout from libkernel's wrapper). Left unhandled it
    // fell through to `return 0` without writing, reporting zero free.
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    uint64_t start = a[0], end = a[1];
    uint64_t align = a[2] ? a[2] : 0x4000;
    if (end == 0 || end > kDmemTotal)
      end = kDmemTotal;
    uint64_t base = g_dmemNext.load(std::memory_order_relaxed);
    if (start > base)
      base = start;
    base = (base + (align - 1)) & ~(align - 1);
    a[0] = end > base ? end - base : 0; // available size
    a[3] = base;                        // its physical offset
    return 0;
  }
  case 0xC0208004: {
    // GetDirectMemoryType: struct = [physAddr(in), regionStart(out),
    // regionEnd(out), memType(out, low 32b)]. Report the reservation that owns
    // physAddr, which is how the renderer reads back its GPU pool's bounds.
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    uint64_t phys = a[0];
    std::lock_guard<std::mutex> lk(g_dmemMutex);
    for (const auto &r : g_dmemRegions) {
      if (phys >= r.start && phys < r.end) {
        a[1] = r.start;
        a[2] = r.end;
        a[3] = r.memType;
        return 0;
      }
    }
    // Unknown offset: report it as a page-sized direct (WC_GARLIC) region so the
    // query still succeeds rather than returning a zero-length hole.
    a[1] = phys & ~0xFFFFull;
    a[2] = (phys & ~0xFFFFull) + 0x10000;
    a[3] = 3;
    return 0;
  }
  case 0x80108002: {
    // MapDirectMemory: struct = [virtualAddr, len, ...]. The title has already
    // reserved the VA (it probes with mmap+munmap to find a free aligned hole),
    // so it passes that VA in and expects the physical memory committed there.
    // Our identity model backs the VA with anonymous GPU-visible memory. Without
    // this, the region the title hands to the GPU stays unmapped.
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    void *va = reinterpret_cast<void *>(a[0]);
    size_t len = a[1];
    if (va && len) {
      // prot 0x33 = read|write + GPU read|write, matching the title's probe map.
      // It marks the region GPU-direct so sceKernelVirtualQuery agrees.
      uint8_t *p = sys_mmap(va, len, 0x33, mFlags::fixed | mFlags::anon,
                            static_cast<uint32_t>(-1), 0);
      if (p == reinterpret_cast<uint8_t *>(-1))
        return -1;
    }
    return 0;
  }
  }

  return 0;
}

uint8_t *dmaDevice::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  //__debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
