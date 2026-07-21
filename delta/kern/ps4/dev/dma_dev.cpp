
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
// Floor for window-less requests so physical offset 0 stays invalid ("offset 0
// means the allocation failed" checks in titles keep working).
constexpr uint64_t kDmemBase = 0x10000000ull;

// Record of each direct-memory reservation so GetDirectMemoryType (ioctl
// 0xC0208004) can answer "which region owns this physical offset, and of what
// type". The renderer queries the regions it just allocated and refuses to
// initialise if they come back as a zero-length, type-0 hole. Kept sorted by
// start so allocation can walk holes in order.
struct DmemRegion {
  uint64_t start, end;
  uint32_t memType;
};
std::mutex g_dmemMutex;
std::vector<DmemRegion> g_dmemRegions;

// First-fit hole search within [lo, hi). The window is part of the contract,
// not a hint: SotC carves the whole pool into fixed windows up front (0x220000
// tail scratch that ends exactly at pool end, a 1 GiB CPU heap, the ~11 GiB
// streaming/GPU heap between them) and derives which internal heap partition
// owns an address from the physical range. The old bump allocator satisfied
// the tail window and then bumped every later reservation past the end of the
// pool, so the two MAIN heaps lived outside their windows -- the engine's
// AllocationTracker range lookup then missed on free and the job fiber
// dereferenced the null/-1 result (Shadow_Shipping+0x189a7 / +0x8d9b7).
int dmemAllocate(uint64_t lo, uint64_t hi, uint64_t len, uint64_t align,
                 uint32_t memType, uint64_t *out) {
  if (hi == 0 || hi > kDmemTotal)
    hi = kDmemTotal;
  if (lo == 0)
    lo = kDmemBase;
  if (len == 0 || align == 0 || (align & (align - 1)) || lo >= hi)
    return -22 /*EINVAL*/;
  std::lock_guard<std::mutex> lk(g_dmemMutex);
  uint64_t cand = (lo + align - 1) & ~(align - 1);
  for (const auto &r : g_dmemRegions) {
    if (r.end <= cand)
      continue;                 // fully below the candidate
    if (r.start >= cand + len)
      break;                    // hole before this region fits (list is sorted)
    cand = ((r.end > cand ? r.end : cand) + align - 1) & ~(align - 1);
  }
  if (cand + len > hi)
    return -12 /*ENOMEM: window exhausted*/;
  auto it = g_dmemRegions.begin();
  while (it != g_dmemRegions.end() && it->start < cand)
    ++it;
  g_dmemRegions.insert(it, {cand, cand + len, memType});
  *out = cand;
  return 0;
}

// Largest free hole inside [lo, hi) (AvailableDirectMemorySize).
void dmemLargestHole(uint64_t lo, uint64_t hi, uint64_t align,
                     uint64_t *holeBase, uint64_t *holeSize) {
  if (hi == 0 || hi > kDmemTotal)
    hi = kDmemTotal;
  if (lo == 0)
    lo = kDmemBase;
  if (align == 0)
    align = 0x4000;
  *holeBase = 0;
  *holeSize = 0;
  std::lock_guard<std::mutex> lk(g_dmemMutex);
  uint64_t cur = (lo + align - 1) & ~(align - 1);
  auto consider = [&](uint64_t end) {
    if (end > cur && end - cur > *holeSize) {
      *holeBase = cur;
      *holeSize = end - cur;
    }
  };
  for (const auto &r : g_dmemRegions) {
    if (r.end <= cur)
      continue;
    if (r.start >= hi)
      break;
    consider(r.start < hi ? r.start : hi);
    cur = ((r.end > cur ? r.end : cur) + align - 1) & ~(align - 1);
  }
  consider(hi);
}
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
    uint64_t off = 0;
    int r = dmemAllocate(a[0], a[1], len, align, static_cast<uint32_t>(a[4]),
                         &off);
    if (r < 0) {
      std::fprintf(stderr,
                   "[dmem] alloc FAILED window=[%#llx,%#llx) len=%#llx -> %d\n",
                   (unsigned long long)a[0], (unsigned long long)a[1],
                   (unsigned long long)len, r);
      return r;
    }
    a[0] = off;  // physical offset out
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
    uint64_t off = 0;
    int r = dmemAllocate(0, kDmemTotal, len, align,
                         static_cast<uint32_t>(a[4]), &off);
    if (r < 0)
      return r;
    a[0] = off;
    return 0;
  }
  case 0xC0208016: {
    // AvailableDirectMemorySize: struct = [searchStart(in)/physOut, searchEnd,
    // align, sizeOut]. The kernel writes the offset of the largest free hole
    // at/after searchStart back into [0] and its size into [3]. (SotC prints
    // the available byte count it read from [3]; with the fields swapped it saw
    // our physical offset - 256 MiB - as the budget and starved its allocator.)
    auto *a = static_cast<uint64_t *>(data);
    if (!a)
      return -1;
    uint64_t base = 0, size = 0;
    dmemLargestHole(a[0], a[1], a[2], &base, &size);
    a[0] = base;                        // physical offset of the hole
    a[3] = size;                        // available size
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
