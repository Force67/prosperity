#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>
#include <mutex>

#include <base/containers/vector.h>

#include <utl/mem.h>

namespace krnl {
struct procInfo;

using mprot = utl::pageProtection;
using alloct = utl::allocationType;

struct pageInfo {
  uint8_t *ptr;
  size_t size;
  mprot prot;
  // Full SCE protection as the guest requested it, including the GPU bits
  // (0x10 GPU_READ / 0x20 GPU_WRITE) that the host r/w/x `prot` drops. Reported
  // by sceKernelVirtualQuery; libSceVideoOut rejects a scanout buffer whose query
  // lacks the GPU-read bit / direct-memory type.
  uint32_t sceProt = 0;
  const char *name = nullptr;
  // MAP_VOID address-space reservation (no committed backing yet): virtual
  // query must report it as NOT committed / NOT flexible, and a later
  // MAP_FIXED commit inside it splits it (add() punches the hole).
  bool reserved = false;

  pageInfo(uint8_t *p, size_t s, mprot mp, uint32_t sp = 0, bool rsv = false)
      : ptr(p), size(s), prot(mp), sceProt(sp), reserved(rsv) {}
};

class vmManager {
public:
  vmManager(procInfo &);
  ~vmManager();

  bool init();
  void add(uint8_t *ptr, size_t size, mprot, uint32_t sceProt = 0,
           bool reserved = false);
  // Drop bookkeeping for [ptr, ptr+size): entries fully inside vanish,
  // straddling entries are truncated/split. Host pages are the caller's
  // business (sys_munmap keeps them mapped; stale guest pointers then read
  // stable garbage instead of faulting, and the NEXT mapping there rules).
  void remove(uint8_t *ptr, size_t size);
  pageInfo *get(uint8_t *ptr);

  // true if [ptr, ptr+size) hits a tracked mapping
  bool overlaps(uint8_t *ptr, size_t size) const;

  uint8_t *mapMemory(uint8_t *preference, size_t size, utl::pageProtection);
  void unmapRtMemory(uint8_t *);

private:
  void punchHoleLocked(uint8_t *ptr, size_t size);

  procInfo &pinfo;

  // guards the page lists against concurrent sys_mmap from guest threads.
  mutable std::mutex vmlock;

  size_t codeMemTotal{0};
  size_t rtMemTotal{0};

  base::Vector<pageInfo> codePages;
  base::Vector<pageInfo> rtPages;
};
}
