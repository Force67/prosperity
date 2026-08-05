
// Copyright (C) Force67 2019

#include <base.h>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>

#include "gc_dev.h"
#include "dce_dev.h"
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_event.h"
#include "kern/ps4/lv2/sys_mem.h"
#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGcCaller, "DELTA_GC_CALLER", false);
DELTA_OPTION(bool, kGcAcb, "DELTA_GPU_ACB", false);
DELTA_OPTION(bool, kGcFlip, "DELTA_GC_FLIP", false);
DELTA_OPTION(bool, kGcTrace, "DELTA_GC_TRACE", false);
DELTA_OPTION(uint32_t, kGcTraceMax, "DELTA_GC_TRACE_MAX", 0);
}  // namespace

// LLE GPU submit bridge (delta_runtime). The real libSceGnmDriver.sprx submits
// PM4 through these ioctls; forward the descriptor array to the GPU command
// processor. See prosperity_gc_submit in libSceGnmDriver.cpp.
extern "C" void prosperity_gc_submit(const void *descArray, uint32_t descCount);
extern "C" void prosperity_gc_submit_acb(const void *commands, uint32_t bytes);
extern "C" void prosperity_gc_flip(uint64_t scanoutBase, int displayBufferIndex,
                                    int64_t flipArg);

// NOTE: the PS5 AGC /dev/gc protocol lives in a separate device,
// kern/ps5/dev/gc_dev.cpp (gcDevicePs5); this file is PS4 GNM only.

namespace krnl {
gcDevice::gcDevice(proc *p) : device(p) {}

bool gcDevice::init(const char *, uint32_t, uint32_t) { return true; }

static void completeFlipLabels(uint64_t flipPtr) {
  if (!flipPtr)
    return;
  // The flip arg block is caller-controlled; not every title passes a pointer
  // here (layout varies by GnmDriver revision). Never dereference a value that
  // isn't a mapped guest VA.
  auto *pr = proc::getActive();
  if (!pr || !pr->getVma().get(reinterpret_cast<uint8_t *>(flipPtr)))
    return;

  auto *p = reinterpret_cast<uint32_t *>(flipPtr);
  // Gnm prepareFlip emits a PM4-like EOP-label packet:
  //   C0038000 addr_lo addr_hi value_lo value_hi
  // Complete it synchronously because the CPU-side emulated submit already
  // finished all draws before returning.
  if (p[0] == 0xC0038000u) {
    uint64_t addr = (static_cast<uint64_t>(p[2] & 0xFF) << 32) | p[1];
    uint64_t value = static_cast<uint64_t>(p[3]) |
                     (static_cast<uint64_t>(p[4]) << 32);
    if (addr && !pr->getVma().get(reinterpret_cast<uint8_t *>(addr)))
      return;
    if (addr) {
      *reinterpret_cast<uint64_t *>(addr) = value;
      if (kGcFlip)
        std::printf("[gc]   flip label [%#lx] = %#lx\n",
                    static_cast<unsigned long>(addr),
                    static_cast<unsigned long>(value));
    }
  }
}

// SCOUT (DELTA_GC_CALLER): scan the stack for the first return address landing in
// any guest module's .text and report it as <module>+offset, to pin which guest
// wrapper issued each gc ioctl (the native backend runs handlers on the guest
// stack). Off by default: the submit ioctls fire 60+/frame and the scan is slow.
static void printGuestCaller() {
  if (!kGcCaller)
    return;
  auto *proc = proc::getActive();
  if (!proc)
    return;
  auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
  int printed = 0;
  uintptr_t last = 0;
  for (int i = 0; i < 512 && printed < 6; i++) {
    uintptr_t v = sp[i];
    if (v == last)
      continue;
    for (auto &m : proc->getModuleList()) {
      auto &mi = m->getInfo();
      auto base = (uintptr_t)mi.textSeg.addr;
      if (base && v >= base && v < base + mi.textSeg.size) {
        std::printf("[gc]   caller[%d] %s+%#lx\n", printed, mi.name.c_str(),
                    v - base);
        last = v;
        printed++;
        break;
      }
    }
  }
}

/* ioctl dispatch */
int32_t gcDevice::ioctl(uint32_t cmd, void *data) {
  // Graphics command submission (the LLE path: real libSceGnmDriver.sprx). The
  // arg's descriptor array is an array of 16-byte PM4 INDIRECT_BUFFER packets;
  // forward it to the GPU command processor. Handle these first (and without the
  // stack-scan SCOUT) since they fire 60+ times per frame.
  switch (cmd) {
  case 0xC0108102: {  // gc submit: {u32 a0, u32 count, u64 descPtr}
    struct argl {
      uint32_t a0;
      uint32_t count;
      uint64_t descPtr;
    };
    auto *a = static_cast<argl *>(data);
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    return 0;
  }
  case 0xC018810A: {  // gc submit (cross-process variant): {u32 pid, _, u32 count,
                      // _, u64 descPtr}. Kernel (11.00): pfind(arg+0),
                      // count at +0x08, descPtr at +0x10.
    struct argl {
      uint32_t pid;  // +0x00: submitter pid (kernel pfind()s it)
      uint32_t pad4;
      uint32_t count;  // +0x08
      uint32_t padC;
      uint64_t descPtr;  // +0x10
    };
    auto *a = static_cast<argl *>(data);
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    return 0;
  }
  case 0xC020810C: {  // gc submit + EOP: {u32 pid, u32 count, u64 descPtr,
                      //   u64 eopVal, u32 wait}. Layout per fpPS4 dev_gc.pas
                      // t_submit_args: +0x10 is the EOP completion VALUE
                      // ("submit_id | vmid<<32") -- a scalar the kernel writes
                      // on GPU completion, never a pointer. EOP label writes
                      // come from EVENT_WRITE_EOP packets inside the dcb and
                      // are the CP's job. (We used to dereference +0x10 as a
                      // prepareFlip packet pointer; SotC passes a scalar there
                      // and the read faulted.)
    struct argl {
      uint32_t pid;
      uint32_t count;
      uint64_t descPtr;
      uint64_t eopVal;
      uint32_t wait;
    };
    auto *a = static_cast<argl *>(data);
    // SCOUT (DELTA_GC_FLIP): dump the raw args + first descriptor.
    static int flipDumps = 0;
    if (kGcFlip && flipDumps < 8) {
      flipDumps++;
      printf("[gc] flip pid=%x count=%u descPtr=%lx eopVal=%lx wait=%x\n",
             a->pid, a->count, (unsigned long)a->descPtr,
             (unsigned long)a->eopVal, a->wait);
      if (a->descPtr && a->count) {
        auto *d = reinterpret_cast<const uint32_t *>(a->descPtr);
        printf("[gc]   desc[0..7]: %x %x %x %x %x %x %x %x\n", d[0], d[1], d[2],
               d[3], d[4], d[5], d[6], d[7]);
      }
    }
    prosperity_gc_submit(reinterpret_cast<const void *>(a->descPtr), a->count);
    uint32_t bufferIndex = dceCurrentBuffer();
    prosperity_gc_flip(dceScanoutBuffer(bufferIndex),
                       static_cast<int>(bufferIndex), dceCurrentFlipArg());
    noteFlip();  // advance the flip count + post the display event for pacing
    return 0;
  }
  case 0xC0088101:  // kernel: wait-suspend-done. Suspend/resume handshake; our
                    // GPU never suspends, so "done" is always true. (Was
                    // mislabeled "switch buffer" -- gc_switch_buffer_internal is
                    // unreachable in the 11.00 kernel's dispatch.)
  case 0xC0048117:  // kernel: wait-suspend-done as well (same handler).
    return 0;
  case 0xC0048116: {  // kernel: submit-completion signal. Returns 0
                      // when no submit is in flight, EBUSY (16) otherwise; hot
                      // (polled per submit). Our submits are synchronous, so
                      // "done" is always true; the kernel never touches the arg
                      // slot, we zero it as the benign idle answer.
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  }
  case 0xC0048114: {  // kernel: GRBM poll -- spins on the busy regs (0x1413/0x1414)
                      // and returns 0 WITHOUT writing the arg slot. The GnmDriver
                      // wrapper (libSceGnmDriver +0x5fd0) zeroes the slot itself
                      // and never reads it back, so only the success return
                      // matters. A title's render thread polls it in a tight loop;
                      // handle it here (return success, zero the slot) so it stops
                      // falling through to the UNHANDLED logger.
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  }
  case 0xC0048113:  // kernel: query softc->submit_count into arg slot (returns 0).
  case 0xC0048115:  // kernel: query softc->submit_idle into arg slot, then clears
                    // it. Both are query hot-paths; no async submit state to
                    // report, so the zeroed buffer is the "nothing pending" answer.
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  case 0xC0088109: {  // kernel: register a suspend-done notify pointer: stores the
                      // u64 in arg into cdevpriv and copyout's a 4-byte zero to
                      // *arg (the "no suspend pending" flag). No suspend support,
                      // so zero the pointed-to flag if it is a mapped guest VA.
    uint64_t ptr = static_cast<uint64_t *>(data) ? *static_cast<uint64_t *>(data) : 0;
    if (ptr) {
      auto *pr = proc::getActive();
      if (pr && pr->getVma().get(reinterpret_cast<uint8_t *>(ptr)))
        *reinterpret_cast<uint32_t *>(ptr) = 0;
    }
    return 0;
  }
  }

  printGuestCaller();
  switch (cmd) {
  case 0xC00C8110: {
    // kernel: spins until the GRBM busy reg (0x2004) clears, then writes regs
    // 0x2232/0x2233 from arg[0]/arg[1] (a compute kick). Our GPU is synchronous
    // (never busy) and register writes are no-ops, so just accept.
    struct argl {
      uint32_t unknown_0;
      uint32_t unknown_4;
      uint32_t unknown_8;
    };
    auto args = reinterpret_cast<argl *>(data);
    (void)args;
    return 0;
  }
  case 0xC010810B: {
    // kernel: redundant-CU query -> {se0, se0, se1, se1}. The values come
    // from a GPU PCI config read (reg 0xBC, low16>>6 and high16&0x3FF); real
    // consoles report 0 redundant CUs. The old 1024 placeholder produced se0=16,
    // which is impossible (an SE has 9 CUs) and would skew the driver's CU count.
    struct argl {
      uint32_t cumask0;
      uint32_t cumask1;
      uint32_t cumask2;
      uint32_t cumask3;
    };
    auto args = reinterpret_cast<argl *>(data);
    args->cumask0 = 0;
    args->cumask1 = 0;
    args->cumask2 = 0;
    args->cumask3 = 0;
    return 0;
  }
  case 0xC008811B: {
    // GNM "trace/info init": the driver passes an 8-byte out slot and stores the
    // returned pointer into its global logging-info pointer (Gnm vaddr 0x100e8),
    // then dereferences it on every submit (`cmp dword[ptr],0` = trace-enable).
    // A bogus value here makes that deref fault. Hand back a real, zeroed guest
    // struct so the deref reads trace-disabled (0) and the logger is a no-op.
    static uint8_t *traceInfo = nullptr;
    if (!traceInfo)
      traceInfo = allocLowGuest(0x100);  // zero-filled; [+0] = trace flag (off)
    auto args = static_cast<uint64_t *>(data);
    *args = reinterpret_cast<uint64_t>(traceInfo);
    printf("gc ioctl(%x): trace-info -> %p\n", cmd, (void *)traceInfo);
    return 0;
  }
  case 0xC030810D:   // sceGnmMapComputeQueue
  case 0xC030811A: { // sceGnmMapComputeQueueWithPriority (same 0x30 struct)
    struct Args {
      uint32_t me;
      uint32_t pipe;
      uint32_t queue;
      uint32_t vqueue;
      uint64_t ringBase;
      uint64_t readPtr;
      uint64_t state;
      uint32_t ringSizeLog2Dw;
      uint32_t reserved;
    };
    static_assert(sizeof(Args) == 0x30);
    const auto* args = static_cast<const Args*>(data);
    if (kGcTrace) {
      const auto* words = static_cast<const uint32_t*>(data);
      std::fprintf(stderr, "[gc] map compute queue ioctl=%#x:", cmd);
      for (uint32_t i = 0; i < 12; i++)
        std::fprintf(stderr, " %08x", words[i]);
      std::fprintf(stderr, "\n");
    }
    if (kGcAcb && args && args->ringSizeLog2Dw < 31) {
      const uint32_t ring_size_dw = 1u << args->ringSizeLog2Dw;
      const uint64_t ring_bytes = static_cast<uint64_t>(ring_size_dw) * 4;
      if (ring_size_dw >= 2 && utl::isMemoryRangeMapped(
                                   reinterpret_cast<const void*>(args->ringBase),
                                   ring_bytes) &&
          utl::isMemoryRangeMapped(reinterpret_cast<const void*>(args->readPtr),
                                   sizeof(uint32_t))) {
        std::lock_guard lock(computeMutex);
        ComputeQueue* slot = nullptr;
        for (ComputeQueue& entry : computeQueues) {
          if (entry.mapped && entry.me == args->me && entry.pipe == args->pipe &&
              entry.queue == args->queue) {
            slot = &entry;
            break;
          }
          if (!entry.mapped && !slot)
            slot = &entry;
        }
        if (slot) {
          *slot = {.me = args->me,
                   .pipe = args->pipe,
                   .queue = args->queue,
                   .vqueue = args->vqueue,
                   .ringBase = args->ringBase,
                   .readPtr = args->readPtr,
                   .state = args->state,
                   .ringSizeDw = ring_size_dw,
                   .readOffsetDw = 0,
                   .mapped = true};
          *reinterpret_cast<uint32_t*>(args->readPtr) = 0;
        }
      }
    }
    // Synchronous CPU submit path: no real HQD/doorbell to program. Accept the
    // mapping and return success WITHOUT touching the caller's struct --
    // vqueueId (+0x0C) is the handle the GnmDriver wrapper hands back to the
    // app for DingDong/Unmap; the UNHANDLED fallthrough used to memset the
    // whole struct, so the app saw handle 0, treated the map as failed and
    // remapped in a loop. The retail kernel validates and returns 0 with the
    // inputs intact.
    return 0;
  }
  case 0xC00C810E: { // sceGnmUnmapComputeQueue
    if (kGcAcb && data) {
      const auto* args = static_cast<const uint32_t*>(data);
      std::lock_guard lock(computeMutex);
      for (ComputeQueue& entry : computeQueues)
        if (entry.mapped && entry.me == args[0] && entry.pipe == args[1] &&
            entry.queue == args[2])
          entry = {};
    }
    return 0;
  }
  case 0xC0108108: {
    // kernel: VA->GPU-physical translate, arg[0] -> arg[1]. Used for GPU-visible buffers. Guest GPU space is identity-mapped to host,
    // so the translation is the input VA itself.
    struct argl {
      uint64_t vaddr;
      uint64_t phys;
    };
    auto args = static_cast<argl *>(data);
    args->phys = args->vaddr;
    return 0;
  }
  case 0xC010811C: // sceGnmDingDong: kicks a mapped compute queue. No-op is
                   // safe for boot but drops async-compute work; wire into the
                   // CP if a title depends on compute results.
    if (kGcTrace) {
      static uint32_t traced = 0;
      if (traced++ < 100) {
        const auto* words = static_cast<const uint32_t*>(data);
        std::fprintf(stderr, "[gc] DingDong: %08x %08x %08x %08x\n", words[0],
                     words[1], words[2], words[3]);
      }
    }
    if (kGcAcb && data) {
      const auto* args = static_cast<const uint32_t*>(data);
      std::lock_guard lock(computeMutex);
      for (ComputeQueue& entry : computeQueues) {
        if (!entry.mapped || entry.me != args[0] || entry.pipe != args[1] ||
            entry.queue != args[2])
          continue;
        const uint32_t next = args[3];
        if (next >= entry.ringSizeDw)
          break;
        const uint32_t available =
            (next - entry.readOffsetDw) & (entry.ringSizeDw - 1);
        if (available) {
          std::vector<uint32_t> commands(available);
          const auto* ring = reinterpret_cast<const uint32_t*>(entry.ringBase);
          for (uint32_t i = 0; i < available; i++)
            commands[i] = ring[(entry.readOffsetDw + i) &
                               (entry.ringSizeDw - 1)];
          if (kGcTrace) {
            // The whole run, not a sample, when a comparison needs the full
            // set of ring IBs (DELTA_GC_TRACE_MAX=0 keeps the old 100).
            static uint32_t traced_commands = 0;
            if (traced_commands++ < (kGcTraceMax ? kGcTraceMax : 100u)) {
              std::fprintf(stderr, "[gc] ACB %u/%u/%u %#x..%#x:", entry.me,
                           entry.pipe, entry.queue, entry.readOffsetDw, next);
              for (uint32_t word : commands)
                std::fprintf(stderr, " %08x", word);
              std::fprintf(stderr, "\n");
            }
          }
          prosperity_gc_submit_acb(commands.data(), available * 4);
        }
        entry.readOffsetDw = next;
        if (utl::isMemoryRangeMapped(reinterpret_cast<const void*>(entry.readPtr),
                                     sizeof(uint32_t)))
          *reinterpret_cast<uint32_t*>(entry.readPtr) = next;
        break;
      }
    }
    return 0;
  case 0xC0088111: {  // kernel: coredump dbg-reg dump. Reads arg[0] (u32 reason
                      // tag), prints the SE/SH status regs (0x2002/0x2004/0x21a1/
                      // 0x208e/0x208f) and dumps the user debug registers.
                      // Debug-only, never writes back; log the tag and succeed.
    if (kGcTrace && data)
      printf("[gc] coredump reason=%#x\n", *static_cast<uint32_t *>(data));
    return 0;
  }
  case 0xC0848119: {
    struct argl {
      uint32_t unknown_00;
      uint32_t unknown_04;
      uint32_t unknown_08;
      uint32_t unknown_0C;
      uint8_t unknown_10[112];
      uint32_t unknown_80;
    };
    auto args = static_cast<argl *>(data);
    printf("gc ioctl(%x): %x, %x, %x, %x, %x\n", cmd, args->unknown_00,
           args->unknown_04, args->unknown_08, args->unknown_0C,
           args->unknown_80);
    return 0;
  }
  }

  // SCOUT: log unknown gc ioctls and soft-succeed so the boot keeps advancing
  // instead of trapping. Lets us discover the sequence GNM actually issues.
  // Rate-limited: an unknown ioctl in the per-submit hot path would otherwise
  // flood unbuffered stderr and stall the render loop.
  static int unhandledLogged = 0;
  if (kGcTrace || unhandledLogged < 32) {
    unhandledLogged++;
    printf("[gc] UNHANDLED ioctl(%x) data=%p\n", cmd, data);
  }
  // Zero the output buffer of an unhandled OUT/INOUT ioctl. The driver reads the
  // buffer back as a query result (capability counts, status words, etc.); left
  // uninitialised it returns stack/heap garbage, which the engine then trusts --
  // e.g. a bogus huge "format count" that overruns a fixed table and smashes the
  // stack. Zero is the benign "nothing/idle/none" answer (matches the explicit
  // 0x16 submit-done handler). Length is encoded in the ioctl command (FreeBSD
  // IOCPARM_LEN). Only touch OUT ioctls (bit 0x40000000).
  if (data && (cmd & 0x40000000u)) {
    uint32_t len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

// Kernel (11.00) maps GPU-visible device memory at a fixed base + offset for
// offsets under a size cap; on retail those fields are never initialized, so
// every real mmap fails. Back the mapping with a lazily allocated GPU-visible
// pool instead: allocLowGuest hands out [512 GiB, 2^40), inside the identity
// range the CP renders into, so a title that maps /dev/gc gets real, CP-usable
// memory. Out-of-range offsets fall back to the anonymous mmap path (-1).
uint8_t *gcDevice::map(void *, size_t size, uint32_t, uint32_t, size_t offset) {
  constexpr uint64_t kPoolSize = 256ull * 1024 * 1024;
  // The pool belongs to the device, not to the descriptor: sys_open news a
  // gcDevice per open, so a per-instance pool would hand two openers different
  // memory for the same offset. sys_mmap holds no lock across device::map
  // either, so the lazy creation needs its own.
  static std::mutex poolLock;
  static uint8_t *pool = nullptr;
  std::lock_guard<std::mutex> lk(poolLock);
  if (!pool) {
    pool = allocLowGuest(kPoolSize);
    if (!pool)
      return reinterpret_cast<uint8_t *>(-1);
  }
  poolBase = pool;
  poolSize = kPoolSize;
  // The guest picks the offset, so bound each side: offset + size wraps.
  if (offset < kPoolSize && size <= kPoolSize - offset)
    return pool + offset;
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
