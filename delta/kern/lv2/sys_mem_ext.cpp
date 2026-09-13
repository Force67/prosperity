
#include <cstdlib>
#include "base/arch.h"
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <base/logging.h>
#include <logger/logger.h>
#include <utl/mem.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>

#include "kern/proc.h"
#include "kern/ps4/audio_daemon.h"
#include "kern/ps4/dev/dma_dev.h"  // dmemBackingFd/Size (shared physical dmem store)
#include "error_table.h"
#include "kern/ps5/dev/dma_dev.h"
#include "sys_mem.h"      // shared enums + sys_mmap (dmem maps delegate to it)
#include "sys_mem_ext.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kVqTrace, "DELTA_VQ_TRACE", false);
}  // namespace

namespace krnl {

// Our VM is a flat host-backed low arena, never compacted: the HOST pages stay
// mapped (utl::freeMem takes no length; stale pointers read stable garbage instead
// of faulting). The BOOKKEEPING must still be released: titles churn VA and key
// allocator state off VirtualQuery's [start,end), so dead VMA entries report stale
// bounds.
int PS4ABI sys_munmap(void *addr, size_t len) {
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    LOG_WARNING("sys_munmap: host pages are retained (first was {} +{:#x}); "
                "only the VMA bookkeeping is released",
                fmt::ptr(addr), len);
  if (auto *proc = proc::getActive(); proc && addr && len) {
    audioDaemonForgetRange(addr, len);
    // Exception to "keep the host pages": a whole PROT_NONE reservation holds no data
    // and leaving it mapped makes the next reservation there relocate. V8 reserves
    // padded, frees, re-reserves exact; a stale pointer into the padding should fault
    // where the mistake is.
    auto *region = proc->getVma().get(static_cast<u8 *>(addr));
    if (region && region->ptr == addr && region->size == len &&
        region->sceProt == 0)
      ::munmap(addr, len);
    proc->getVma().remove(static_cast<u8 *>(addr), len);
    noteGuestReleased(static_cast<u8 *>(addr), len);
    forgetDmemVa(static_cast<u8 *>(addr), len);
  }
  return 0;
}

// The guest libc allocates through mmap, so the brk is unused. A benign 0 keeps
// any stray caller satisfied without handing it a usable region.
i64 PS4ABI sys_obreak(void *) { return 0; }
i64 PS4ABI sys_sbrk(intptr_t) { return 0; }

// Anonymous host memory has no file backing, so there is nothing to flush.
int PS4ABI sys_msync(void *, size_t, int) { return 0; }

int PS4ABI sys_madvise(void *, size_t, int) { return 0; }

// The guest arena is fully committed and never pages out: report MINCORE_INCORE per
// page so a residency probe sees the truth.
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

// sceKernelQueryMemoryProtection(addr, &start, &end, &prot). The libkernel wrapper
// passes one scratch struct as arg2 for the kernel to fill, then distributes:
// {start@0, end@8, prot@0x10}, per the wrapper at libkernel 0x17ef0 (prot masked
// 0x37). Unmapped addr = the EACCES the wrapper translates.
int PS4ABI sys_query_memory_protection(void *addr, void *info) {
  auto *proc = proc::getActive();
  if (!proc || !info)
    return -SysError::eINVAL;
  auto *region = proc->getVma().get(static_cast<u8 *>(addr));
  if (!region)
    return -SysError::eACCES;

  auto *qp = static_cast<u8 *>(info);
  std::memset(qp, 0, 0x18);
  void *start = region->ptr;
  void *end = region->ptr + region->size;
  // Full SCE prot (with GPU bits) if we kept it, else the host r/w/x bits.
  u32 prot = region->sceProt
                      ? region->sceProt
                      : static_cast<u32>(region->prot);
  std::memcpy(qp + 0x00, &start, sizeof(void *));
  std::memcpy(qp + 0x08, &end, sizeof(void *));
  std::memcpy(qp + 0x10, &prot, sizeof(u32));
  return 0;
}

// The host's own view of an address, for ranges the guest VMA never recorded.
struct HostMapping {
  u64 start = 0;
  u64 end = 0;
  u32 prot = 0;  // SCE r/w/x bits, 0 when the address is not mapped at all
};

HostMapping hostMappingOf(const void *addr) {
  HostMapping out;
  const u64 want = reinterpret_cast<u64>(addr);
  std::FILE *f = std::fopen("/proc/self/maps", "re");
  if (!f)
    return out;
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    unsigned long long lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) != 3)
      continue;
    if (want < lo || want >= hi)
      continue;
    out.start = lo;
    out.end = hi;
    out.prot = (perms[0] == 'r' ? 1u : 0u) | (perms[1] == 'w' ? 2u : 0u) |
               (perms[2] == 'x' ? 4u : 0u);
    break;
  }
  std::fclose(f);
  return out;
}

// sceKernelVirtualQuery(addr, flags, info, size); layout verified against the
// consumer at libkernel 0x2b9d0 (size 0x48, name at +0x21):
// 0x00 start 0x08 end 0x10 offset 0x18 protection 0x1C memoryType
// 0x20 bits (flexible/direct/stack/pooled/committed) 0x21 char[32] name.
// Our anon low-arena maps are flexible and committed.
int PS4ABI sys_virtual_query(const void *addr, int /*flags*/, void *info,
                             size_t infoSize) {
  auto *proc = proc::getActive();
  if (!proc || !info || infoSize == 0)
    return -SysError::eINVAL;

  std::memset(info, 0, infoSize);
  auto *region =
      proc->getVma().get(const_cast<u8 *>(static_cast<const u8 *>(addr)));
  if (!region) {
    // Memory we allocated outside the guest VMA is still guest-used memory: a library
    // validating a caller's buffer with this call must not hear "unmapped". libSce
    // Videodec2 does exactly that, answering every AvPlayer call with
    // ARGUMENT_POINTER (Astro Bot's title screen sat behind a video that never started).
    if (const HostMapping host = hostMappingOf(addr); host.prot) {
      auto *vq = static_cast<u8 *>(info);
      std::memcpy(vq + 0x00, &host.start, sizeof(u64));
      std::memcpy(vq + 0x08, &host.end, sizeof(u64));
      std::memcpy(vq + 0x10, &host.start, sizeof(u64));
      if (infoSize >= 0x1C + sizeof(int)) {
        // Host PROT_READ/WRITE happen to be SCE's CPU bits, but a host mapping carries
        // no GPU bits, and a library validating hardware-bound buffers demands them.
        // libSceVdecCore (+0x17c50) rejects any range without prot & 0x30 (error 5, why
        // Astro Bot's intro video never decoded). The guest is identity-mapped for our
        // GPU: report the GPU bits alongside the CPU ones.
        int prot = static_cast<int>(host.prot);
        if (prot & 0x1)
          prot |= 0x10;  // GPU read
        if (prot & 0x2)
          prot |= 0x20;  // GPU write
        const int memType = 0;  // WB_ONION, like any other CPU mapping
        std::memcpy(vq + 0x18, &prot, sizeof(int));
        std::memcpy(vq + 0x1C, &memType, sizeof(int));
      }
      if (infoSize >= 0x21)
        vq[0x20] = 0x01 | 0x10;  // flexible + committed
      if (kVqTrace)
        BASE_LOGI("vq", "addr={:p} host mapping [{:#x}..{:#x}) prot={:#x}",
                  addr, (unsigned long long)host.start,
                  (unsigned long long)host.end, host.prot);
      return 0;
    }
    // Worth seeing: a caller that walks its own heap this way reads the zeroed
    // struct as "not committed" and silently skips the range.
    if (kVqTrace)
      BASE_LOGI("vq", "addr={:p} NOT MAPPED", addr);
    return -SysError::eACCES;
  }

  // A region with NO protection at all is a reservation, and a title committing
  // sub-ranges inside one left us reporting the outer entry (Astro Bot's decoder
  // buffer sits in a 1.9 GiB reservation; libSceVdecCore wants prot & 0x2/0x30).
  // If the host has accessible pages there, that mapping is the truth; an untouched
  // PROT_NONE reservation still reads uncommitted.
  if (!region->sceProt && !static_cast<u32>(region->prot)) {
    if (const HostMapping host = hostMappingOf(addr); host.prot) {
      auto *vq = static_cast<u8 *>(info);
      std::memcpy(vq + 0x00, &host.start, sizeof(u64));
      std::memcpy(vq + 0x08, &host.end, sizeof(u64));
      std::memcpy(vq + 0x10, &host.start, sizeof(u64));
      if (infoSize >= 0x1C + sizeof(int)) {
        int prot = static_cast<int>(host.prot);
        if (prot & 0x1)
          prot |= 0x10;  // GPU read: guest memory is identity-mapped for us
        if (prot & 0x2)
          prot |= 0x20;  // GPU write
        const int memType = 0;  // WB_ONION
        std::memcpy(vq + 0x18, &prot, sizeof(int));
        std::memcpy(vq + 0x1C, &memType, sizeof(int));
      }
      if (infoSize >= 0x21)
        vq[0x20] = 0x01 | 0x10;  // flexible + committed
      if (kVqTrace)
        BASE_LOGI("vq", "addr={:p} committed inside a reservation: "
                        "[{:#x}..{:#x}) prot={:#x}",
                  addr, (unsigned long long)host.start,
                  (unsigned long long)host.end, host.prot);
      return 0;
    }
  }

  auto *vq = static_cast<u8 *>(info);
  void *start = region->ptr;
  void *end = region->ptr + region->size;
  std::memcpy(vq + 0x00, &start, sizeof(void *));
  std::memcpy(vq + 0x08, &end, sizeof(void *));
  // +0x10 = the region's direct-memory offset, exact for /dev/dmem ranges: SotC
  // turns (offset + addr - start) into a 64 KiB heap-map block index and marks
  // nothing when out of range. Elsewhere we have no dmem pool (identity-mapped),
  // but libSceVideoOut rejects a scanout buffer unless the offset shares the VA's
  // low 16 bits (tiling alignment), so report the VA there; zero failed every
  // scanout register.
  u64 offset = region->hasPhys ? region->physOffset
                                    : reinterpret_cast<u64>(start);
  std::memcpy(vq + 0x10, &offset, sizeof(u64));
  // GPU-accessible memory (guest asked bits 0x10/0x20) is direct/physical in SCE
  // terms: report WC_GARLIC (memType 3) + direct bit, what libSceVideoOut checks
  // before registering a scanout buffer; plain CPU memory stays flexible WB_ONION.
  bool gpu = (region->sceProt & 0x30) != 0;
  // For direct memory, the reservation's type is the truth (titles route addresses
  // to heaps by it); inferring GARLIC from GPU prot bits made a CPU heap mapped
  // GPU-visible look like the GPU one.
  int memType = region->hasPhys ? dmemTypeForOffset(region->physOffset) : -1;
  if (memType < 0)
    memType = gpu ? 3 : 0;  // 3 = SCE_KERNEL_WC_GARLIC, 0 = WB_ONION
  if (infoSize >= 0x1C + sizeof(int)) {
    int prot = region->sceProt ? static_cast<int>(region->sceProt)
                               : static_cast<int>(region->prot);
    std::memcpy(vq + 0x18, &prot, sizeof(int));
    std::memcpy(vq + 0x1C, &memType, sizeof(int));
  }
  if (kVqTrace)
    BASE_LOGI("vq",
              "addr={:p} region=[{:p}..{:p}) sceProt={:#x} memType={} rsv={} "
              "off={:#x}{}",
              addr, start, end, region->sceProt, memType,
              region->reserved ? 1 : 0, (unsigned long long)offset,
              region->hasPhys ? " (dmem)" : "");
  if (infoSize >= 0x21) {
    // flexible(0x01) | direct(0x02, GPU mem) | committed(0x10). A MAP_VOID
    // reservation is none of these; titles branch on isCommitted to decide
    // whether a range still needs a real commit.
    vq[0x20] = region->reserved
                   ? 0x00
                   : (0x01 | 0x10 |
                      ((gpu || region->hasPhys) ? 0x02 : 0x00));
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

// sceKernelBatchMap/2: a list of dmem map/unmap/protect ops in one call. Entry
// layout (libkernel wrapper): 0x00 start 0x08 offset (MAP_DIRECT) 0x10 length
// 0x18 protection 0x19 memoryType 0x1c operation; ops 0 MAP_DIRECT, 1 UNMAP,
// 2 PROTECT, 3 MAP_FLEXIBLE, 4 TYPE_PROTECT. The maps must really commit at
// `start`: SotC batch-maps its GPU pools, and the old ignore-stub left PM4
// referencing never-backed VAs, faulting the submit.
int PS4ABI sys_batch_map(u32 /*handle*/, u32 /*flags*/,
                         void *entries, int count, int *processed) {
  struct BatchMapEntry {
    u64 start;
    u64 offset;
    u64 length;
    u8 prot;
    u8 type;
    u16 pad;
    u32 operation;
  };
  static_assert(sizeof(BatchMapEntry) == 0x20, "batch-map entry is 32 bytes");

  auto *e = static_cast<BatchMapEntry *>(entries);
  int done = 0;
  for (; e && done < count; done++) {
    const auto &op = e[done];
    if (kDmemTrace)
      BASE_LOGI("dmem",
                "batch[{}/{}] op={} start={:#x} off={:#x} len={:#x} prot={:#x} "
                "type={}",
                done, count, op.operation, (unsigned long long)op.start,
                (unsigned long long)op.offset, (unsigned long long)op.length,
                op.prot, op.type);
    switch (op.operation) {
    case 0:   // MAP_DIRECT: back the VA with the shared dmem store at op.offset,
              // MAP_FLEXIBLE: no physical offset, so map the shared dmem backing (syscall
              // 628 semantics): every VA mapping it aliases the same bytes. Skyrim maps GPU
              // pools this way then waits on a GPU-written label; private pages never see it.
    case 3: { // MAP_FLEXIBLE: no physical offset, plain anonymous memory
      if (!op.start || !op.length) {
        if (processed)
          *processed = done;
        return -SysError::eINVAL;
      }
      auto *pr = proc::getActive();
      // Sharing the dmem backing is NOT safe for PS4 yet: SotC maps one physical
      // offset at 1664 successive VAs and never releases, so a shared store aliases
      // every historical mapping at once and the title's memory dissolves (measured
      // twice, both times SotC stopped rendering even its intro). Needs understanding
      // of what the guest does with that offset first.
      const bool ps5 = pr && pr->getPlatform() == proc::platform::ps5;
      const int fd = (op.operation == 0 && ps5) ? dmemBackingFd() : -1;
      if (fd >= 0 && op.offset + op.length <= dmemBackingSize()) {
        void *p = ::mmap(reinterpret_cast<void *>(op.start), op.length,
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
                         static_cast<off_t>(op.offset));
        if (p != MAP_FAILED) {
          pr->getVma().add(reinterpret_cast<u8 *>(p), op.length,
                           utl::pageProtection::w);
          break;
        }
      }
      u8 *p = sys_mmap(reinterpret_cast<void *>(op.start), op.length,
                            op.prot, mFlags::fixed | mFlags::anon,
                            static_cast<u32>(-1), 0);
      if (isErrnoPtr(p)) {
        if (processed)
          *processed = done;
        return -SysError::eNOMEM;
      }
      break;
    }
    case 1: // UNMAP: host pages retained, bookkeeping released (see sys_munmap)
      if (auto *pr = proc::getActive(); pr && op.start && op.length)
        pr->getVma().remove(reinterpret_cast<u8 *>(op.start), op.length);
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

// sys_set_vm_container (559): arg == -1 returns the current vm container id;
// arg 0 or 1 selects it (requires privilege). The kernel validates: unsigned
// (arg+1) > 2 is EINVAL, and arg > 1 is EINVAL. We track the id (default 0).
int PS4ABI sys_set_vm_container(u32 op) {
  static std::atomic<u32> current{0};
  if (op == 0xFFFFFFFFu)
    return static_cast<int>(current.load());
  if (op > 1)
    return -SysError::eINVAL;
  current.store(op);
  return 0;
}

// sceKernelMapDirectMemory (628). Real ABI (libkernel 01.14.00): rdi=VA hint,
// rsi=len, rdx=prot, rcx=flags (0x10 = FIXED), r8=packed(alignShift<<24|memType),
// r9=directMemoryStart (the PHYSICAL offset from AllocateMainDirectMemory).
// The physical offset is the source of truth: map the VA to the shared dmem
// backing at that offset (MAP_SHARED) so a CPU-written command buffer and the
// GPU's view alias the same bytes (without it the CP read all-zero DCBs).
// Falls back to anonymous memory; returns the mapped VA.
i64 PS4ABI sys_mmap_dmem(void *addr, size_t len, int prot, int flags,
                             i64 /*packed*/, i64 physOffset) {
  const bool fixedReq = (flags & 0x10) != 0;
  auto *active = proc::getActive();
  const bool ps5 = active && active->getPlatform() == proc::platform::ps5;
  const int fd = ps5 ? dmemBackingFd() : -1;  // see the batch-map note above
  if (kDmemTrace)
    BASE_LOGI("dmem",
              "map628 va={:p} len={:#x} prot={:#x} flags={:#x} physOff={:#x} "
              "fixed={}",
              addr, len, prot, flags, (unsigned long long)physOffset,
              fixedReq ? 1 : 0);
  if (fd >= 0 && physOffset >= 0 &&
      static_cast<u64>(physOffset) + len <= dmemBackingSize()) {
    const int mflags = MAP_SHARED | (fixedReq ? MAP_FIXED : 0);
    void *p = ::mmap(addr, len, PROT_READ | PROT_WRITE, mflags, fd,
                     static_cast<off_t>(physOffset));
    if (p != MAP_FAILED) {
      proc::getActive()->getVma().addDirect(reinterpret_cast<u8 *>(p), len,
                                            utl::pageProtection::w,
                                            static_cast<u32>(prot),
                                            static_cast<u64>(physOffset));
      return reinterpret_cast<i64>(p);
    }
  }
  // Fallback: plain anonymous mapping (loses aliasing but keeps the region live).
  u8 *p = sys_mmap(addr, len, PROT_READ | PROT_WRITE,
                        mFlags::anon | (fixedReq ? mFlags::fixed : 0),
                        static_cast<u32>(-1), 0);
  if (isErrnoPtr(p))
    return -SysError::eNOMEM;
  // Still direct memory as far as the guest is concerned: it keys its own heap
  // map off the physical offset the query reports back.
  if (physOffset >= 0)
    proc::getActive()->getVma().addDirect(p, len, utl::pageProtection::w,
                                          static_cast<u32>(prot),
                                          static_cast<u64>(physOffset));
  return reinterpret_cast<i64>(p);
}

int PS4ABI sys_cpuset(void *, int, int, i64, size_t, void *) { return 0; }

int PS4ABI sys_extend_page_table_pool() { return 0; }

i64 PS4ABI sys_get_vm_map_timestamp() { return 0; }

int PS4ABI sys_get_map_statistics(void *info) {
  if (info)
    std::memset(info, 0, 0x40);
  return 0;
}

// Thread stacks live in the leaked flat arena, so freeing one is a no-op.
int PS4ABI sys_free_stack(void *, size_t) { return 0; }

} // namespace krnl
