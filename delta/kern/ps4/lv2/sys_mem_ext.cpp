
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
#include "../dev/dma_dev.h"  // dmemBackingFd/Size (shared physical dmem store)
#include "error_table.h"
#include "../../ps5/dev/dma_dev.h"
#include "sys_mem.h"      // shared enums + sys_mmap (dmem maps delegate to it)
#include "sys_mem_ext.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kVqTrace, "DELTA_VQ_TRACE", false);
}  // namespace

namespace krnl {

// Our VM is a flat, host-backed low arena that is never compacted, so the HOST
// pages stay mapped (utl::freeMem() takes only a base, no length, and can't
// honour a partial munmap; stale guest pointers then read stable garbage
// rather than faulting). The BOOKKEEPING, however, must be released: titles
// churn VA (SotC recycles multi-MB fiber/streaming regions constantly) and key
// real allocator state off sceKernelVirtualQuery's [start, end) — leaving dead
// entries in the VMA made later queries report stale bounds.
int PS4ABI sys_munmap(void *addr, size_t len) {
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    LOG_WARNING("sys_munmap: host pages are retained (first was {} +{:#x}); "
                "only the VMA bookkeeping is released",
                fmt::ptr(addr), len);
  if (auto *proc = proc::getActive(); proc && addr && len) {
    // One exception to "keep the host pages": a whole PROT_NONE reservation.
    // Nothing has data in it by definition, and leaving it mapped makes the
    // next reservation at that address get relocated -- V8 reserves a padded
    // region, frees it and re-reserves an exact sub-range, and a stale pointer
    // into the freed padding then reads memory that is still there instead of
    // faulting where the mistake is.
    auto *region = proc->getVma().get(static_cast<uint8_t *>(addr));
    if (region && region->ptr == addr && region->size == len &&
        region->sceProt == 0)
      ::munmap(addr, len);
    proc->getVma().remove(static_cast<uint8_t *>(addr), len);
    noteGuestReleased(static_cast<uint8_t *>(addr), len);
    forgetDmemVa(static_cast<uint8_t *>(addr), len);
  }
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
  if (!region) {
    // Worth seeing: a caller that walks its own heap this way reads the zeroed
    // struct as "not committed" and silently skips the range.
    if (kVqTrace)
      std::printf("[vq] addr=%p NOT MAPPED\n", addr);
    return -SysError::eACCES;
  }

  auto *vq = static_cast<uint8_t *>(info);
  void *start = region->ptr;
  void *end = region->ptr + region->size;
  std::memcpy(vq + 0x00, &start, sizeof(void *));
  std::memcpy(vq + 0x08, &end, sizeof(void *));
  // +0x10 = the region's direct-memory offset. For a range mapped through
  // /dev/dmem that is the physical offset the guest asked for, and it must be
  // exact: SotC turns (offset + addr - start) into a block index in its own
  // 64 KiB heap map, and marks nothing when the index lands out of range.
  // For everything else we have no dmem pool to point at (guest addresses are
  // identity-mapped), and libSceVideoOut's buffer registration rejects a
  // scanout buffer unless the offset shares the virtual address's low 16 bits
  // (a tiling-alignment check), so report the VA there; left zero, every
  // scanout register failed.
  uint64_t offset = region->hasPhys ? region->physOffset
                                    : reinterpret_cast<uint64_t>(start);
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
  if (kVqTrace)
    std::printf(
        "[vq] addr=%p region=[%p..%p) sceProt=%#x memType=%d rsv=%d off=%#llx%s\n",
        addr, start, end, region->sceProt, gpu ? 3 : 0,
        region->reserved ? 1 : 0, (unsigned long long)offset,
        region->hasPhys ? " (dmem)" : "");
  if (infoSize >= 0x21) {
    // flexible(0x01) | direct(0x02, GPU mem) | committed(0x10). A MAP_VOID
    // reservation is none of these -- titles branch on isCommitted to decide
    // whether a range still needs a real commit.
    vq[0x20] = region->reserved ? 0x00 : (0x01 | 0x10 | (gpu ? 0x02 : 0x00));
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

// Applies a list of dmem map/unmap/protect ops in one call
// (sceKernelBatchMap/BatchMap2). Entry layout from the libkernel wrapper:
//   0x00 void*  start      (VA the title already chose)
//   0x08 u64    offset     (physical dmem offset, MAP_DIRECT only)
//   0x10 u64    length
//   0x18 u8     protection (SCE prot incl. GPU bits)
//   0x19 u8     memoryType
//   0x1c u32    operation
// Ops: 0 = MAP_DIRECT, 1 = UNMAP, 2 = PROTECT, 3 = MAP_FLEXIBLE,
// 4 = TYPE_PROTECT. The maps must actually commit memory at `start`: SotC
// batch-maps its GPU pools this way, and with the old ignore-stub the PM4
// stream referenced VAs that were never backed and the submit faulted.
int PS4ABI sys_batch_map(uint32_t /*handle*/, uint32_t /*flags*/,
                         void *entries, int count, int *processed) {
  struct BatchMapEntry {
    uint64_t start;
    uint64_t offset;
    uint64_t length;
    uint8_t prot;
    uint8_t type;
    uint16_t pad;
    uint32_t operation;
  };
  static_assert(sizeof(BatchMapEntry) == 0x20, "batch-map entry is 32 bytes");

  auto *e = static_cast<BatchMapEntry *>(entries);
  int done = 0;
  for (; e && done < count; done++) {
    const auto &op = e[done];
    if (kDmemTrace)
      std::fprintf(stderr,
                   "[dmem] batch[%d/%d] op=%u start=%#llx off=%#llx len=%#llx "
                   "prot=%#x type=%u\n",
                   done, count, op.operation, (unsigned long long)op.start,
                   (unsigned long long)op.offset, (unsigned long long)op.length,
                   op.prot, op.type);
    switch (op.operation) {
    case 0:   // MAP_DIRECT: back the VA with the shared dmem store at op.offset,
              // so every VA mapping that physical offset aliases the same bytes
              // (syscall 628 already does this). Skyrim maps its GPU pools with
              // batch-map and then waits on a label the GPU writes; with private
              // anonymous pages per mapping the CPU never sees the write.
    case 3: { // MAP_FLEXIBLE: no physical offset, plain anonymous memory
      if (!op.start || !op.length) {
        if (processed)
          *processed = done;
        return -SysError::eINVAL;
      }
      auto *pr = proc::getActive();
      const bool ps5 = pr && pr->getPlatform() == proc::platform::ps5;
      const int fd = (op.operation == 0 && ps5) ? dmemBackingFd() : -1;
      if (fd >= 0 && op.offset + op.length <= dmemBackingSize()) {
        void *p = ::mmap(reinterpret_cast<void *>(op.start), op.length,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
                         static_cast<off_t>(op.offset));
        if (p != MAP_FAILED) {
          pr->getVma().add(reinterpret_cast<uint8_t *>(p), op.length,
                           utl::pageProtection::w);
          break;
        }
      }
      uint8_t *p = sys_mmap(reinterpret_cast<void *>(op.start), op.length,
                            op.prot, mFlags::fixed | mFlags::anon,
                            static_cast<uint32_t>(-1), 0);
      if (p == reinterpret_cast<uint8_t *>(-1)) {
        if (processed)
          *processed = done;
        return -SysError::eNOMEM;
      }
      break;
    }
    case 1: // UNMAP: host pages retained, bookkeeping released (see sys_munmap)
      if (auto *pr = proc::getActive(); pr && op.start && op.length)
        pr->getVma().remove(reinterpret_cast<uint8_t *>(op.start), op.length);
      break;
    case 2: // PROTECT / TYPE_PROTECT: our flat arena stays permissive; the
    case 4: // title only narrows protections it already owns
      break;
    default:
      LOG_WARNING("sys_batch_map: unknown op {} (entry {})", op.operation,
                  done);
      break;
    }
  }
  if (processed)
    *processed = done;
  return 0;
}

int PS4ABI sys_set_vm_container(uint32_t) { return 0; }

// Map a direct-memory region (sceKernelMapDirectMemory, syscall 628). Real ABI
// (verified from libkernel 01.14.00): rdi=VA hint, rsi=len, rdx=prot,
// rcx=flags (bit 0x10 = FIXED), r8=packed(alignShift<<24 | memType),
// r9=directMemoryStart (the PHYSICAL dmem offset from AllocateMainDirectMemory).
//
// The physical offset is the source of truth: map the VA to the shared dmem
// backing store at that offset (MAP_SHARED) so every VA that maps the same
// physOffset -- a CPU-written GPU command buffer and the GPU's own view of it --
// aliases the same bytes. Without this the two views were independent anonymous
// pages and the command processor read all-zero DCBs. Falls back to anonymous
// memory when the backing is unavailable. Returns the mapped VA (rax).
int64_t PS4ABI sys_mmap_dmem(void *addr, size_t len, int prot, int flags,
                             int64_t /*packed*/, int64_t physOffset) {
  const bool fixedReq = (flags & 0x10) != 0;
  auto *active = proc::getActive();
  const bool ps5 = active && active->getPlatform() == proc::platform::ps5;
  const int fd = ps5 ? dmemBackingFd() : -1;  // PS5-only shared store (WIP)
  if (kDmemTrace)
    std::fprintf(stderr,
                 "[dmem] map628 va=%p len=%#zx prot=%#x flags=%#x physOff=%#llx fixed=%d\n",
                 addr, len, prot, flags, (unsigned long long)physOffset, fixedReq ? 1 : 0);
  if (fd >= 0 && physOffset >= 0 &&
      static_cast<uint64_t>(physOffset) + len <= dmemBackingSize()) {
    const int mflags = MAP_SHARED | (fixedReq ? MAP_FIXED : 0);
    void *p = ::mmap(addr, len, PROT_READ | PROT_WRITE, mflags, fd,
                     static_cast<off_t>(physOffset));
    if (p != MAP_FAILED) {
      proc::getActive()->getVma().addDirect(reinterpret_cast<uint8_t *>(p), len,
                                            utl::pageProtection::w,
                                            static_cast<uint32_t>(prot),
                                            static_cast<uint64_t>(physOffset));
      return reinterpret_cast<int64_t>(p);
    }
  }
  // Fallback: plain anonymous mapping (loses aliasing but keeps the region live).
  uint8_t *p = sys_mmap(addr, len, PROT_READ | PROT_WRITE,
                        mFlags::anon | (fixedReq ? mFlags::fixed : 0),
                        static_cast<uint32_t>(-1), 0);
  if (p == reinterpret_cast<uint8_t *>(-1))
    return -SysError::eNOMEM;
  // Still direct memory as far as the guest is concerned: it keys its own heap
  // map off the physical offset the query reports back.
  if (physOffset >= 0)
    proc::getActive()->getVma().addDirect(p, len, utl::pageProtection::w,
                                          static_cast<uint32_t>(prot),
                                          static_cast<uint64_t>(physOffset));
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
