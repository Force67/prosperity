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

  pageInfo(uint8_t *p, size_t s, mprot mp, uint32_t sp = 0)
      : ptr(p), size(s), prot(mp), sceProt(sp) {}
};

class vmManager {
public:
  vmManager(procInfo &);
  ~vmManager();

  bool init();
  void add(uint8_t *ptr, size_t size, mprot, uint32_t sceProt = 0);
  pageInfo *get(uint8_t *ptr);

  // true if [ptr, ptr+size) hits a tracked mapping
  bool overlaps(uint8_t *ptr, size_t size) const;

  uint8_t *mapMemory(uint8_t *preference, size_t size, utl::pageProtection);
  void unmapRtMemory(uint8_t *);

private:
  procInfo &pinfo;

  // guards the page lists against concurrent sys_mmap from guest threads.
  mutable std::mutex vmlock;

  size_t codeMemTotal{0};
  size_t rtMemTotal{0};

  base::Vector<pageInfo> codePages;
  base::Vector<pageInfo> rtPages;
};
}
