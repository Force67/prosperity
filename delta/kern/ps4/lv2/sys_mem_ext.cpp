
#include <cstdlib>
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <logger/logger.h>
#include <utl/mem.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>

#include "../../proc.h"
#include "error_table.h"
#include "sys_mem.h"      // shared enums + sys_mmap (dmem maps delegate to it)
#include "sys_mem_ext.h"

namespace krnl {

// Our VM is a flat, host-backed low arena that is never compacted. We don't
// reclaim guest mappings: utl::freeMem() takes only a base (no length) so it
// can't honour a partial munmap, and a wrong base would corrupt the arena. The
// host reclaims every page at exit. Logged once so a workload that churns large
// mappings (and would actually need reclaim) is visible.
int PS4ABI sys_munmap(void *addr, size_t len) {
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    LOG_WARNING("sys_munmap: unmaps are ignored (regions leak until exit); "
                "first was {} (+{:#x})",
                fmt::ptr(addr), len);
  return 0;
}

// The guest libc allocates through mmap, so the brk is unused. A benign 0 keeps
// any stray caller satisfied without handing it a usable region.
int64_t PS4ABI sys_obreak(void *) { return 0; }
int64_t PS4ABI sys_sbrk(intptr_t) { return 0; }

// Anonymous host memory has no file backing, so there is nothing to flush.
int PS4ABI sys_msync(void *, size_t, int) { return 0; }

int PS4ABI sys_madvise(void *, size_t, int) { return 0; }

// Our guest arena is fully committed host memory that never pages out, so every
// queried page is resident. Report MINCORE_INCORE for each so a caller probing
// residency (e.g. an allocator deciding whether to madvise) sees the truth
// instead of "all paged out".
int PS4ABI sys_mincore(void *addr, size_t len, char *vec) {
  if (!vec)
    return -SysError::eFAULT;
  size_t pages = (len + 0x3FFF) >> 14;
  std::memset(vec, 0x01 /*MINCORE_INCORE*/, pages);
  return 0;
}

// All our pages are committed host memory that never pages out, so locking is a
// no-op.
int PS4ABI sys_mlock(const void *, size_t) { return 0; }
int PS4ABI sys_munlock(const void *, size_t) { return 0; }
int PS4ABI sys_mlockall(int) { return 0; }
int PS4ABI sys_munlockall() { return 0; }

// minherit only matters across fork(), which we don't model.
int PS4ABI sys_minherit(void *, size_t, int) { return 0; }

// sceKernelQueryMemoryProtection(addr, void** start, void** end, int* prot).
// Its libkernel wrapper passes a single scratch struct as arg2 that the kernel
// fills, then distributes the fields to the caller's three out-pointers. Layout
// {void* start@0; void* end@8; uint32 prot@0x10}, verified against the wrapper
// at libkernel 0x17ef0 (the prot read there masks with 0x37). An unmapped addr
// is an error, the same EACCES the wrapper translates from a failed syscall.
int PS4ABI sys_query_memory_protection(void *addr, void *info) {
  auto *proc = proc::getActive();
  if (!proc || !info)
    return -SysError::eINVAL;
  auto *region = proc->getVma().get(static_cast<uint8_t *>(addr));
  if (!region)
    return -SysError::eACCES;

  auto *qp = static_cast<uint8_t *>(info);
  std::memset(qp, 0, 0x18);
  void *start = region->ptr;
  void *end = region->ptr + region->size;
  // Full SCE prot (with GPU bits) if we kept it, else the host r/w/x bits.
  uint32_t prot = region->sceProt
                      ? region->sceProt
                      : static_cast<uint32_t>(region->prot);
  std::memcpy(qp + 0x00, &start, sizeof(void *));
  std::memcpy(qp + 0x08, &end, sizeof(void *));
  std::memcpy(qp + 0x10, &prot, sizeof(uint32_t));
  return 0;
}

// sceKernelVirtualQuery(addr, flags, SceKernelVirtualQueryInfo* info, size).
// Layout verified against the consumer at libkernel 0x2b9d0 (passes size 0x48
// and reads name at info+0x21):
//   0x00 void*  start      0x08 void*  end       0x10 uint64 offset
//   0x18 int    protection 0x1C int    memoryType
//   0x20 uint8  bits (flexible:0x01 direct:0x02 stack:0x04 pooled:0x08 committed:0x10)
//   0x21 char[32] name
// size 0x48. Our anon low-arena maps are flexible memory and always committed.
int PS4ABI sys_virtual_query(const void *addr, int /*flags*/, void *info,
                             size_t infoSize) {
  auto *proc = proc::getActive();
  if (!proc || !info || infoSize == 0)
    return -SysError::eINVAL;

  std::memset(info, 0, infoSize);
  auto *region =
      proc->getVma().get(const_cast<uint8_t *>(static_cast<const uint8_t *>(addr)));
  if (!region)
    return -SysError::eACCES;

  auto *vq = static_cast<uint8_t *>(info);
  void *start = region->ptr;
  void *end = region->ptr + region->size;
  std::memcpy(vq + 0x00, &start, sizeof(void *));
  std::memcpy(vq + 0x08, &end, sizeof(void *));
  // +0x10 = the region's direct-memory offset. We don't model a separate dmem
  // pool (guest addresses are identity-mapped), but libSceVideoOut's buffer
  // registration rejects a scanout buffer unless the offset shares the virtual
  // address's low 16 bits (a tiling-alignment check). Report the VA as the
  // offset so that holds; left zero, every scanout register failed.
  uint64_t offset = reinterpret_cast<uint64_t>(start);
  std::memcpy(vq + 0x10, &offset, sizeof(uint64_t));
  // GPU-accessible memory (the guest asked for GPU read/write, bits 0x10/0x20)
  // is direct/physical memory in SCE terms: report it as WC_GARLIC (memType 3)
  // with the direct bit set, which is what libSceVideoOut checks before it will
  // register a scanout buffer. Plain CPU memory stays flexible WB_ONION.
  bool gpu = (region->sceProt & 0x30) != 0;
  if (infoSize >= 0x1C + sizeof(int)) {
    int prot = region->sceProt ? static_cast<int>(region->sceProt)
                               : static_cast<int>(region->prot);
    std::memcpy(vq + 0x18, &prot, sizeof(int));
    int memType = gpu ? 3 : 0;  // 3 = SCE_KERNEL_WC_GARLIC, 0 = WB_ONION
    std::memcpy(vq + 0x1C, &memType, sizeof(int));
  }
  if (std::getenv("DELTA_VQ_TRACE"))
    std::printf("[vq] addr=%p region=[%p..%p) sceProt=%#x memType=%d\n", addr,
                start, end, region->sceProt, gpu ? 3 : 0);
  if (infoSize >= 0x21) {
    // flexible(0x01) | direct(0x02, GPU mem) | committed(0x10)
    vq[0x20] = 0x01 | 0x10 | (gpu ? 0x02 : 0x00);
    if (region->name) {
      size_t n = std::strlen(region->name);
      if (n > 31)
        n = 31;
      std::memcpy(vq + 0x21, region->name, n);
      vq[0x21 + n] = '\0';
    }
  }
  return 0;
}

// Applies a list of dmem map/unmap/protect ops in one call. We don't model the
// direct-memory pool, so accept and report all entries processed. Logged so a
// title that actually drives dmem through this path is visible.
int PS4ABI sys_batch_map(uint32_t /*handle*/, uint32_t /*flags*/, void *,
                         int count, int *processed) {
  LOG_WARNING("sys_batch_map: dmem batch of {} op(s) not modeled; ignoring",
              count);
  if (processed)
    *processed = count;
  return 0;
}

int PS4ABI sys_set_vm_container(uint32_t) { return 0; }

// Map a direct-memory region. We don't track physical dmem, so satisfy it with
// an ordinary anonymous low-guest mapping. sys_mmap returns (uint8_t*)-1 on
// failure, propagated as ENOMEM.
int64_t PS4ABI sys_mmap_dmem(void *addr, size_t len, int memType, int prot,
                             int /*flags*/, int64_t directMemoryStart) {
  uint8_t *p = sys_mmap(addr, len, static_cast<uint32_t>(prot), mFlags::anon,
                        static_cast<uint32_t>(-1), 0);
  if (std::getenv("DELTA_DMEM_TRACE"))
    std::fprintf(stderr, "[dmem] map req_addr=%p len=%#zx memType=%d prot=%#x dmStart=%#llx -> %p\n",
                 addr, len, memType, prot, (unsigned long long)directMemoryStart, (void *)p);
  if (p == reinterpret_cast<uint8_t *>(-1))
    return -SysError::eNOMEM;
  return reinterpret_cast<int64_t>(p);
}

int PS4ABI sys_cpuset(void *, int, int, int64_t, size_t, void *) { return 0; }

int PS4ABI sys_extend_page_table_pool() { return 0; }

int64_t PS4ABI sys_get_vm_map_timestamp() { return 0; }

int PS4ABI sys_get_map_statistics(void *info) {
  if (info)
    std::memset(info, 0, 0x40);
  return 0;
}

// Thread stacks live in the leaked flat arena, so freeing one is a no-op.
int PS4ABI sys_free_stack(void *, size_t) { return 0; }

} // namespace krnl
