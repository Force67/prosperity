#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include <mutex>

#include <base/containers/vector.h>

#include <utl/mem.h>

namespace krnl {
struct procInfo;

using mprot = utl::pageProtection;
using alloct = utl::allocationType;

struct pageInfo {
  u8 *ptr;
  size_t size;
  mprot prot;
  // Full SCE protection as the guest requested it, including the GPU bits
  // (0x10 GPU_READ / 0x20 GPU_WRITE) that the host r/w/x `prot` drops. Reported
  // by sceKernelVirtualQuery; libSceVideoOut rejects a scanout buffer whose query
  // lacks the GPU-read bit / direct-memory type.
  u32 sceProt = 0;
  const char *name = nullptr;
  // MAP_VOID address-space reservation (no committed backing yet): virtual
  // query must report it as NOT committed / NOT flexible, and a later
  // MAP_FIXED commit inside it splits it (add() punches the hole).
  bool reserved = false;
  // Direct-memory physical offset this range maps, for a region that came from
  // sceKernelMapDirectMemory. hasPhys says the field is real: sceKernelVirtual-
  // Query reports it, and titles convert it into a block index in their own
  // heap map, so a VA substituted here lands nowhere near the right block.
  u64 physOffset = 0;
  bool hasPhys = false;

  pageInfo(u8 *p, size_t s, mprot mp, u32 sp = 0, bool rsv = false)
      : ptr(p), size(s), prot(mp), sceProt(sp), reserved(rsv) {}
};

class vmManager {
public:
  vmManager(procInfo &);
  ~vmManager();

  bool init();
  void add(u8 *ptr, size_t size, mprot, u32 sceProt = 0,
           bool reserved = false);
  // Same, for a range backed by direct memory at `physOffset`.
  void addDirect(u8 *ptr, size_t size, mprot, u32 sceProt,
                 u64 physOffset);
  // Drop bookkeeping for [ptr, ptr+size): entries fully inside vanish,
  // straddling entries are truncated/split. Host pages are the caller's
  // business (sys_munmap keeps them mapped; stale guest pointers then read
  // stable garbage instead of faulting, and the NEXT mapping there rules).
  void remove(u8 *ptr, size_t size);
  pageInfo *get(u8 *ptr);

  // Apply a protection to every tracked mapping intersecting [ptr, ptr+size)
  // (kernel vm_map_protect: a range with no tracked mapping is fine, not an
  // error). Updates both the host r/w/x prot and the full SCE prot so
  // sceKernelVirtualQuery reports the mprotect result.
  void protectRange(u8 *ptr, size_t size, mprot prot, u32 sceProt);
  // Attach a name to every tracked mapping intersecting [ptr, ptr+size)
  // (kernel vm_map_set_name). Takes a copy of the name.
  void setRangeName(u8 *ptr, size_t size, const char *name);

  // true if [ptr, ptr+size) hits a tracked mapping
  bool overlaps(u8 *ptr, size_t size) const;

  // Diagnostic: invoke `fn(ctx, ptr, size)` for every tracked mapping in the GPU
  // aperture [0x8000_0000_00, 0x8100_0000_00) that is small enough to sweep
  // (<= 4 MiB), used to locate the guest's PM4 command buffers without a
  // multi-GB scan of the big dmem pools.
  void forEachGpuAperturePage(void (*fn)(void *, u8 *, size_t),
                              void *ctx) const;

  u8 *mapMemory(u8 *preference, size_t size, utl::pageProtection);
  void unmapRtMemory(u8 *);

private:
  void punchHoleLocked(u8 *ptr, size_t size);

  procInfo &pinfo;

  // guards the page lists against concurrent sys_mmap from guest threads.
  mutable std::mutex vmlock;

  size_t codeMemTotal{0};
  size_t rtMemTotal{0};

  base::Vector<pageInfo> codePages;
  base::Vector<pageInfo> rtPages;
};
}
