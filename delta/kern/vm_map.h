#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include <optional>

#include "base/containers/vector.h"

#include "base/threading/mutex.h"
#include "host_memory/host_memory.h"

namespace kern {
struct ProcessInfo;

using Mprot = host_memory::PageProtection;
using Alloct = host_memory::AllocationType;

struct PageInfo {
  u8* ptr;
  size_t size;
  Mprot prot;
  // Full SCE protection as the guest requested it, including the GPU bits
  // (0x10 GPU_READ / 0x20 GPU_WRITE) that the host r/w/x `prot` drops. Reported
  // by sceKernelVirtualQuery; libSceVideoOut rejects a scanout buffer whose
  // query lacks the GPU-read bit / direct-memory type.
  u32 sce_prot = 0;
  const char* name = nullptr;
  // MAP_VOID address-space reservation (no committed backing yet): virtual
  // query must report it as NOT committed / NOT flexible, and a later
  // MAP_FIXED commit inside it splits it (add() punches the hole).
  bool reserved = false;
  // Direct-memory physical offset this range maps, for a region that came from
  // sceKernelMapDirectMemory. hasPhys says the field is real: sceKernelVirtual-
  // Query reports it, and titles convert it into a block index in their own
  // heap map, so a VA substituted here lands nowhere near the right block.
  u64 phys_offset = 0;
  bool has_phys = false;

  PageInfo(u8* p, size_t s, Mprot mp, u32 sp = 0, bool rsv = false)
      : ptr(p), size(s), prot(mp), sce_prot(sp), reserved(rsv) {}
};

// Told about every range whose mapping changes (mapped over, or dropped), so a
// cache of guest memory above the kernel can forget it. Installed by the
// composition root; called with the VM lock held, from any thread.
using MappingChangedHook = void (*)(u64 base, u64 bytes);
void SetMappingChangedHook(MappingChangedHook hook);

class VmMap {
 public:
  VmMap(ProcessInfo&);
  ~VmMap();

  bool Init();
  void Add(u8* ptr,
           size_t size,
           Mprot,
           u32 sce_prot = 0,
           bool reserved = false);
  // Same, for a range backed by direct memory at `physOffset`.
  void AddDirect(u8* ptr, size_t size, Mprot, u32 sce_prot, u64 phys_offset);
  // Drop bookkeeping for [ptr, ptr+size): entries fully inside vanish,
  // straddling entries are truncated/split. Direct-memory aliases are detached
  // from their backing so stale pointers cannot access a reused allocation.
  void Remove(u8* ptr, size_t size);
  void RemoveDirect(u64 phys_offset, size_t size);
  std::optional<PageInfo> Get(u8* ptr);

  // Apply a protection to every tracked mapping intersecting [ptr, ptr+size)
  // (kernel vm_map_protect: a range with no tracked mapping is fine, not an
  // error). Updates both the host r/w/x prot and the full SCE prot so
  // sceKernelVirtualQuery reports the mprotect result.
  void ProtectRange(u8* ptr, size_t size, Mprot prot, u32 sce_prot);
  // Attach a name to every tracked mapping intersecting [ptr, ptr+size)
  // (kernel vm_map_set_name). Takes a copy of the name.
  void SetRangeName(u8* ptr, size_t size, const char* name);

  // true if [ptr, ptr+size) hits a tracked mapping
  bool Overlaps(u8* ptr, size_t size) const;

  // Diagnostic: invoke `fn(ctx, ptr, size)` for every tracked mapping in the
  // GPU aperture [0x8000_0000_00, 0x8100_0000_00) that is small enough to sweep
  // (<= 4 MiB), used to locate the guest's PM4 command buffers without a
  // multi-GB scan of the big dmem pools.
  void ForEachGpuAperturePage(void (*fn)(void*, u8*, size_t), void* ctx) const;

  u8* MapMemory(u8* preference, size_t size, host_memory::PageProtection);
  void UnmapRtMemory(u8*);

 private:
  void PunchHoleLocked(u8* ptr, size_t size);

  ProcessInfo& pinfo_;

  // guards the page lists against concurrent sys_mmap from guest threads.
  mutable base::Mutex vmlock_;

  size_t code_mem_total_{0};
  size_t rt_mem_total_{0};

  base::Vector<PageInfo> code_pages_;
  base::Vector<PageInfo> rt_pages_;
};
}  // namespace kern
