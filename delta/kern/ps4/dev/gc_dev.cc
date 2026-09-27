
// Copyright (C) Force67 2019

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"

#include <sys/mman.h>

#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "kern/ps4/dev/dce_dev.h"
#include "kern/ps4/dev/gc_dev.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kGcCaller, "DELTA_GC_CALLER", false);
// Execute the async-compute rings a title maps and rings doorbells on. P.T.
// streams every texture as a compute copy on ring 1/0/0 and renders nothing
// without this; no draw is declined, the packets are just never walked.
// DELTA_GPU_ACB=0 reverts.
DELTA_OPTION(bool, kGcAcb, "DELTA_GPU_ACB", true);
// Drain mapped compute rings by their contents rather than by a matching
// doorbell; see the walk in the DingDong handler.
DELTA_OPTION(u32, kGcAcbScan, "DELTA_GPU_ACB_SCAN", 0);
DELTA_OPTION(bool, kGcFlip, "DELTA_GC_FLIP", false);
DELTA_OPTION(bool, kGcTrace, "DELTA_GC_TRACE", false);
DELTA_OPTION(u32, kGcTraceMax, "DELTA_GC_TRACE_MAX", 0);
}  // namespace

// LLE GPU submit bridge: real libSceGnmDriver.sprx submits PM4 through these
// ioctls; forward the descriptor array to the GPU command processor
// (see prosperity_gc_submit in libSceGnmDriver.cpp).
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe() {
  kern::NoteGpuEndOfPipe();
}
// NOLINTEND(readability-identifier-naming)
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe_ctx(u64 ctx) {
  kern::NoteGpuEndOfPipeCtx(ctx);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_submit(const void* desc_array, u32 desc_count);
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_submit_acb(const void* commands, u32 bytes);

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_flip(u64 scanout_base,
                                   int display_buffer_index,
                                   i64 flip_arg);
// NOLINTEND(readability-identifier-naming)

// NOTE: the PS5 AGC /dev/gc protocol lives in a separate device,
// kern/ps5/dev/gc_dev.cc (GcDevicePs5); this file is PS4 GNM only.

namespace kern {
GcDevice::GcDevice(ObjectTable& objects) : Device(objects) {}

bool GcDevice::Init(const char*, u32, u32) {
  return true;
}

static void CompleteFlipLabels(u64 flip_ptr) {
  if (!flip_ptr)
    return;
  // The flip arg block is caller-controlled; not every title passes a pointer
  // here (layout varies by GnmDriver revision). Never dereference a value that
  // isn't a mapped guest VA.
  auto* pr = Process::GetActive();
  if (!pr || !pr->GetVma().Get(reinterpret_cast<u8*>(flip_ptr)))
    return;

  auto* p = reinterpret_cast<u32*>(flip_ptr);
  // Gnm prepareFlip emits a PM4-like EOP-label packet (C0038000 lo hi val_lo
  // val_hi); complete it synchronously, the emulated submit already finished
  // all draws.
  if (p[0] == 0xC0038000u) {
    u64 addr = (static_cast<u64>(p[2] & 0xFF) << 32) | p[1];
    u64 value = static_cast<u64>(p[3]) | (static_cast<u64>(p[4]) << 32);
    if (addr && !pr->GetVma().Get(reinterpret_cast<u8*>(addr)))
      return;
    if (addr) {
      *reinterpret_cast<u64*>(addr) = value;
      if (kGcFlip)
        BASE_LOGI("gc", "  flip label [{:#x}] = {:#x}",
                  static_cast<unsigned long>(addr),
                  static_cast<unsigned long>(value));
    }
  }
}

// DELTA_GC_CALLER: scan the stack for the first return address in a guest
// module's .text, to pin which guest wrapper issued each gc ioctl. Off by
// default: the submit ioctls fire 60+/frame and the scan is slow.
static void PrintGuestCaller() {
  if (!kGcCaller)
    return;
  auto* proc = Process::GetActive();
  if (!proc)
    return;
  auto* sp = reinterpret_cast<uintptr_t*>(__builtin_frame_address(0));
  int printed = 0;
  uintptr_t last = 0;
  for (int i = 0; i < 512 && printed < 6; i++) {
    uintptr_t v = sp[i];
    if (v == last)
      continue;
    for (auto& m : proc->GetModuleList()) {
      auto& mi = m->GetInfo();
      auto base = (uintptr_t)mi.text_seg.addr;
      if (base && v >= base && v < base + mi.text_seg.size) {
        BASE_LOGI("gc", "  caller[{}] {}+{:#x}", printed, mi.name.c_str(),
                  v - base);
        last = v;
        printed++;
        break;
      }
    }
  }
}

/* ioctl dispatch */
i32 GcDevice::Ioctl(u32 cmd, void* data) {
  // Graphics command submission (LLE path): the arg's descriptor array is
  // 16-byte PM4 INDIRECT_BUFFER packets; forward to the GPU command processor.
  // Handle first, without the SCOUT scan, they fire 60+ times per frame.
  switch (cmd) {
    case 0xC0108102: {  // gc submit: {u32 a0, u32 count, u64 descPtr}
      struct Argl {
        u32 a0;
        u32 count;
        u64 desc_ptr;
      };
      auto* a = static_cast<Argl*>(data);
      prosperity_gc_submit(reinterpret_cast<const void*>(a->desc_ptr),
                           a->count);
      return 0;
    }
    case 0xC018810A: {  // gc submit (cross-process variant): {u32 pid, _, u32
                        // count,
                        // _, u64 descPtr}. Kernel (11.00): pfind(arg+0),
                        // count at +0x08, descPtr at +0x10.
      struct Argl {
        u32 pid;  // +0x00: submitter pid (kernel pfind()s it)
        u32 pad4;
        u32 count;  // +0x08
        u32 pad_c;
        u64 desc_ptr;  // +0x10
      };
      auto* a = static_cast<Argl*>(data);
      prosperity_gc_submit(reinterpret_cast<const void*>(a->desc_ptr),
                           a->count);
      return 0;
    }
    case 0xC020810C: {  // gc submit + EOP: {u32 pid, u32 count, u64 descPtr,
                        //   u64 eopVal, u32 wait}. Layout per fpPS4 dev_gc.pas
                        // t_submit_args: +0x10 is the EOP completion VALUE
                        // (submit_id | vmid<<32), a scalar the kernel writes on
                        // completion, never a pointer; EOP label writes come
                        // from EVENT_WRITE_EOP packets in the dcb. (Deref'ing
                        // +0x10 as a prepareFlip pointer faulted; SotC passes a
                        // scalar.)
      struct Argl {
        u32 pid;
        u32 count;
        u64 desc_ptr;
        u64 eop_val;
        u32 wait;
      };
      auto* a = static_cast<Argl*>(data);
      // SCOUT (DELTA_GC_FLIP): dump the raw args + first descriptor.
      static int flip_dumps = 0;
      if (kGcFlip && flip_dumps < 8) {
        flip_dumps++;
        BASE_LOGI("gc",
                  "flip pid={:x} count={} descPtr={:x} eopVal={:x} wait={:x}",
                  a->pid, a->count, (unsigned long)a->desc_ptr,
                  (unsigned long)a->eop_val, a->wait);
        if (a->desc_ptr && a->count) {
          auto* d = reinterpret_cast<const u32*>(a->desc_ptr);
          BASE_LOGI("gc",
                    "  desc[0..7]: {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
        }
      }
      prosperity_gc_submit(reinterpret_cast<const void*>(a->desc_ptr),
                           a->count);
      u32 buffer_index = DceCurrentBuffer();
      prosperity_gc_flip(DceScanoutBuffer(buffer_index),
                         static_cast<int>(buffer_index), DceCurrentFlipArg());
      NoteFlip();  // advance the flip count + post the display event for pacing
      return 0;
    }
    case 0xC0088101:  // kernel: wait-suspend-done. Suspend/resume handshake;
                      // our GPU never suspends, so "done" is always true. (Was
                      // mislabeled "switch buffer"; gc_switch_buffer_internal
                      // is unreachable in the 11.00 kernel's dispatch.)
    case 0xC0048117:  // kernel: wait-suspend-done as well (same handler).
      return 0;
    case 0xC0048116: {  // kernel: submit-completion signal. Returns 0
                        // 0 when no submit is in flight, EBUSY (16) otherwise;
                        // polled per submit. Our submits are synchronous, so
                        // always report done and zero the arg slot.
      if (data)
        *static_cast<u32*>(data) = 0;
      return 0;
    }
    case 0xC0048114: {  // kernel: GRBM poll, spins on the busy regs
                        // (0x1413/0x1414) Returns 0 WITHOUT writing the arg
                        // slot (the GnmDriver wrapper +0x5fd0 zeroes it itself
                        // and never reads back). Polled in a tight loop; answer
                        // here so it stops falling through to the UNHANDLED
                        // logger.
      if (data)
        *static_cast<u32*>(data) = 0;
      return 0;
    }
    case 0xC0048113:  // kernel: query softc->submit_count into arg slot
                      // (returns 0).
    case 0xC0048115:  // kernel: query softc->submit_idle into arg slot, then
                      // clears it. Both are query hot-paths; no async submit
                      // state to report, so the zeroed buffer is the "nothing
                      // pending" answer.
      if (data)
        *static_cast<u32*>(data) = 0;
      return 0;
    case 0xC0088109: {  // kernel: register a suspend-done notify pointer:
                        // stores the u64 in arg into cdevpriv and copyout's a
                        // 4-byte zero to *arg (the "no suspend pending" flag).
                        // No suspend support, so zero the pointed-to flag if it
                        // is a mapped guest VA.
      u64 ptr = static_cast<u64*>(data) ? *static_cast<u64*>(data) : 0;
      if (ptr) {
        auto* pr = Process::GetActive();
        if (pr && pr->GetVma().Get(reinterpret_cast<u8*>(ptr)))
          *reinterpret_cast<u32*>(ptr) = 0;
      }
      return 0;
    }
  }

  PrintGuestCaller();
  switch (cmd) {
    case 0xC00C8110: {
      // kernel: spins until the GRBM busy reg (0x2004) clears, then writes regs
      // 0x2232/0x2233 from arg[0]/arg[1] (a compute kick). Our GPU is
      // synchronous (never busy) and register writes are no-ops, so just
      // accept.
      struct Argl {
        u32 unknown_0;
        u32 unknown_4;
        u32 unknown_8;
      };
      auto args = reinterpret_cast<Argl*>(data);
      (void)args;
      return 0;
    }
    case 0xC010810B: {
      // Redundant-CU query -> {se0, se0, se1, se1} from a GPU PCI config read
      // (reg 0xBC); real consoles report 0 redundant CUs. The old 1024
      // placeholder gave se0=16, impossible for a 9-CU SE, and skewed the
      // driver's CU count.
      struct Argl {
        u32 cumask0;
        u32 cumask1;
        u32 cumask2;
        u32 cumask3;
      };
      auto args = reinterpret_cast<Argl*>(data);
      args->cumask0 = 0;
      args->cumask1 = 0;
      args->cumask2 = 0;
      args->cumask3 = 0;
      return 0;
    }
    case 0xC008811B: {
      // GNM trace/info init: the driver stores the returned pointer globally
      // (Gnm vaddr 0x100e8) and derefs it on every submit (`cmp dword[ptr],0`).
      // A bogus value faults that deref; hand back a zeroed guest struct
      // (trace-disabled).
      static u8* trace_info = nullptr;
      if (!trace_info)
        trace_info =
            AllocLowGuest(0x100);  // zero-filled; [+0] = trace flag (off)
      auto args = static_cast<u64*>(data);
      *args = reinterpret_cast<u64>(trace_info);
      BASE_LOGI("gc", "ioctl({:x}): trace-info -> {:p}", cmd, trace_info);
      return 0;
    }
    case 0xC030810D:    // sceGnmMapComputeQueue
    case 0xC030811A: {  // sceGnmMapComputeQueueWithPriority (same 0x30 struct)
      struct Args {
        u32 me;
        u32 pipe;
        u32 queue;
        u32 vqueue;
        u64 ring_base;
        u64 read_ptr;
        u64 state;
        u32 ring_size_log2_dw;
        u32 reserved;
      };
      static_assert(sizeof(Args) == 0x30);
      const auto* args = static_cast<const Args*>(data);
      if (kGcTrace) {
        const auto* words = static_cast<const u32*>(data);
        base::String words_out;
        base::FormatTo(words_out, "map compute queue ioctl={:#x}:", cmd);
        for (u32 i = 0; i < 12; i++)
          base::FormatTo(words_out, " {:08x}", words[i]);
        BASE_LOGI("gc", "{}", words_out.c_str());
      }
      if (kGcAcb && args && args->ring_size_log2_dw < 31) {
        const u32 ring_size_dw = 1u << args->ring_size_log2_dw;
        const u64 ring_bytes = static_cast<u64>(ring_size_dw) * 4;
        if (ring_size_dw >= 2 &&
            host_memory::IsMemoryRangeMapped(
                reinterpret_cast<const void*>(args->ring_base), ring_bytes) &&
            host_memory::IsMemoryRangeMapped(
                reinterpret_cast<const void*>(args->read_ptr), sizeof(u32))) {
          base::LockGuard lock(g_compute_mutex);
          ComputeQueue* slot = nullptr;
          for (ComputeQueue& entry : g_compute_queues) {
            if (entry.mapped && entry.me == args->me &&
                entry.pipe == args->pipe && entry.queue == args->queue) {
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
                     .ring_base = args->ring_base,
                     .read_ptr = args->read_ptr,
                     .state = args->state,
                     .ring_size_dw = ring_size_dw,
                     .read_offset_dw = 0,
                     .mapped = true};
            *reinterpret_cast<u32*>(args->read_ptr) = 0;
          }
        }
      }
      // Synchronous CPU submit path: accept the mapping, return success WITHOUT
      // touching the caller's struct. vqueueId (+0x0C) is the handle the app
      // uses for DingDong/Unmap; the old UNHANDLED fallthrough memset the
      // struct, the app saw handle 0 and remapped in a loop.
      return 0;
    }
    case 0xC00C810E: {  // sceGnmUnmapComputeQueue
      if (kGcAcb && data) {
        const auto* args = static_cast<const u32*>(data);
        base::LockGuard lock(g_compute_mutex);
        for (ComputeQueue& entry : g_compute_queues)
          if (entry.mapped && entry.me == args[0] && entry.pipe == args[1] &&
              entry.queue == args[2])
            entry = {};
      }
      return 0;
    }
    case 0xC0108108: {
      // kernel: VA->GPU-physical translate, arg[0] -> arg[1]. Used for
      // GPU-visible buffers. Guest GPU space is identity-mapped to host, so the
      // translation is the input VA itself.
      struct Argl {
        u64 vaddr;
        u64 phys;
      };
      auto args = static_cast<Argl*>(data);
      args->phys = args->vaddr;
      return 0;
    }
    case 0xC010811C:  // sceGnmDingDong: kicks a mapped compute queue. No-op is
                      // safe for boot but drops async-compute work; wire into
                      // the CP if a title depends on compute results.
      if (kGcTrace) {
        static u32 traced = 0;
        if (traced++ < 100) {
          const auto* words = static_cast<const u32*>(data);
          BASE_LOGI("gc", "DingDong: {:08x} {:08x} {:08x} {:08x}", words[0],
                    words[1], words[2], words[3]);
        }
      }
      if (kGcAcb && data) {
        const auto* args = static_cast<const u32*>(data);
        base::LockGuard lock(g_compute_mutex);
        // Census of EVERY mapped queue, not just the kicked one: SotC maps
        // seven and only 1/0/0 arrives here. Either the other six are empty or
        // their doorbell never reaches us (the driver can ring via its /dev/gc
        // mapping); reading the first packet of each ring separates the cases.
        if (kGcTrace) {
          static u32 kicks = 0;
          if ((kicks++ % 500) == 0) {
            for (const ComputeQueue& q : g_compute_queues) {
              if (!q.mapped)
                continue;
              const auto* ring = reinterpret_cast<const u32*>(q.ring_base);
              const bool readable = host_memory::IsMemoryRangeMapped(ring, 16);
              BASE_LOGI("acbcensus",
                        "{}/{}/{} read={:#x} ring={:08x} {:08x} {:08x} "
                        "{:08x}{}",
                        q.me, q.pipe, q.queue, q.read_offset_dw,
                        readable ? ring[0] : 0, readable ? ring[1] : 0,
                        readable ? ring[2] : 0, readable ? ring[3] : 0,
                        readable ? "" : " (ring unmapped)");
            }
            std::fflush(stderr);
          }
        }
        if (kGcAcbScan)
          DrainQueues(kGcAcbScan);
        for (ComputeQueue& entry : g_compute_queues) {
          if (!entry.mapped || entry.me != args[0] || entry.pipe != args[1] ||
              entry.queue != args[2])
            continue;
          const u32 next = args[3];
          if (next >= entry.ring_size_dw)
            break;
          const u32 available =
              (next - entry.read_offset_dw) & (entry.ring_size_dw - 1);
          if (available) {
            base::Vector<u32> commands(available);
            const auto* ring = reinterpret_cast<const u32*>(entry.ring_base);
            for (u32 i = 0; i < available; i++)
              commands[i] =
                  ring[(entry.read_offset_dw + i) & (entry.ring_size_dw - 1)];
            if (kGcTrace) {
              // The whole run, not a sample, when a comparison needs the full
              // set of ring IBs (DELTA_GC_TRACE_MAX=0 keeps the old 100).
              static u32 traced_commands = 0;
              if (traced_commands++ < (kGcTraceMax ? kGcTraceMax : 100u)) {
                base::String acb_words;
                base::FormatTo(
                    acb_words, "ACB {}/{}/{} {:#x}..{:#x}:", entry.me,
                    entry.pipe, entry.queue, entry.read_offset_dw, next);
                for (u32 word : commands)
                  base::FormatTo(acb_words, " {:08x}", word);
                BASE_LOGI("gc", "{}", acb_words.c_str());
              }
            }
            prosperity_gc_submit_acb(commands.data(), available * 4);
          }
          entry.read_offset_dw = next;
          if (host_memory::IsMemoryRangeMapped(
                  reinterpret_cast<const void*>(entry.read_ptr), sizeof(u32)))
            *reinterpret_cast<u32*>(entry.read_ptr) = next;
          break;
        }
      }
      return 0;
    case 0xC0088111: {  // kernel: coredump dbg-reg dump. Reads arg[0] (u32
                        // reason tag), prints the SE/SH status regs
                        // (0x2002/0x2004/0x21a1/ 0x208e/0x208f) and dumps the
                        // user debug registers. Debug-only, never writes back;
                        // log the tag and succeed.
      if (kGcTrace && data)
        BASE_LOGI("gc", "coredump reason={:#x}", *static_cast<u32*>(data));
      return 0;
    }
    case 0xC0848119: {
      struct Argl {
        u32 unknown_00;
        u32 unknown_04;
        u32 unknown_08;
        u32 unknown_0_c;
        u8 unknown_10[112];
        u32 unknown_80;
      };
      auto args = static_cast<Argl*>(data);
      BASE_LOGI("gc", "ioctl({:x}): {:x}, {:x}, {:x}, {:x}, {:x}", cmd,
                args->unknown_00, args->unknown_04, args->unknown_08,
                args->unknown_0_c, args->unknown_80);
      return 0;
    }
  }

  // SCOUT: log unknown gc ioctls and soft-succeed so boot advances;
  // rate-limited, an unknown per-submit ioctl would flood unbuffered stderr and
  // stall the loop.
  static int unhandled_logged = 0;
  if (kGcTrace || unhandled_logged < 32) {
    unhandled_logged++;
    BASE_LOGI("gc", "UNHANDLED ioctl({:x}) data={:p}", cmd, data);
  }
  // Zero the output buffer of an unhandled OUT/INOUT ioctl (length =
  // IOCPARM_LEN of the command). The driver reads it back as a query result;
  // garbage there became a bogus huge "format count" overrunning a fixed table.
  // Zero is the benign "nothing/idle/none" answer; only OUT ioctls (bit
  // 0x40000000).
  if (data && (cmd & 0x40000000u)) {
    u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

// Kernel 11.00 maps GPU-visible device memory at a fixed base+offset under a
// size cap; on retail those fields are never initialized, so every real mmap
// fails. Back the mapping with a lazily allocated pool in [512 GiB, 2^40),
// inside the identity range the CP renders into; out-of-range offsets fall back
// to anon.
base::Array<GcDevice::ComputeQueue, 64> GcDevice::g_compute_queues{};
base::Mutex GcDevice::g_compute_mutex;

// SotC spreads async compute over seven queues; only 1/0/0 arrives as a
// DingDong ioctl, the other six sit at read=0 all run, so their doorbell
// reaches the hardware some other way (the driver can write its /dev/gc
// mapping). Consume a ring entry only when it is a complete IB packet whose
// target resolves, advance readOffsetDw past it, stop at the first non-packet
// word. Caller must hold g_compute_mutex (not recursive; locking here too
// deadlocked on the first doorbell). The ring id a doorbell names is the
// queue's VQUEUE field from the map ioctl's +0x0c (GTA:SA maps vqueue 0x29 and
// rings 41; not pipe + me*8).
void GcDevice::RingDoorbell(u32 ring_id, u32 write_offset_dw) {
  for (ComputeQueue& q : g_compute_queues) {
    if (!q.mapped || q.vqueue != ring_id)
      continue;
    if (!q.ring_size_dw || write_offset_dw >= q.ring_size_dw)
      return;
    const u32 pending =
        (write_offset_dw - q.read_offset_dw) & (q.ring_size_dw - 1);
    if (pending)
      DrainQueues(pending);
    return;
  }
}

void GcDevice::DrainQueues(u32 budget_dw) {
  for (ComputeQueue& q : g_compute_queues) {
    if (!q.mapped || !q.ring_size_dw ||
        !host_memory::IsMemoryRangeMapped(
            reinterpret_cast<const void*>(q.ring_base),
            static_cast<u64>(q.ring_size_dw) * 4))
      continue;
    const auto* ring = reinterpret_cast<const u32*>(q.ring_base);
    const u32 mask = q.ring_size_dw - 1;
    const u32 budget = base::Min<u32>(budget_dw, q.ring_size_dw);
    u32 off = q.read_offset_dw, consumed = 0;
    while (consumed < budget) {
      u32 w[4];
      for (u32 i = 0; i < 4; i++)
        w[i] = ring[(off + i) & mask];
      if ((w[0] & 0xFFFFFF00u) != 0xC0023F00u)
        break;
      const u64 addr = (static_cast<u64>(w[2] & 0xFF) << 32) | w[1];
      const u32 dw = w[3] & 0xFFFFF;
      if (!dw ||
          !host_memory::IsMemoryRangeMapped(reinterpret_cast<const void*>(addr),
                                            static_cast<u64>(dw) * 4))
        break;
      prosperity_gc_submit_acb(w, sizeof(w));
      off = (off + 4) & mask;
      consumed += 4;
    }
    if (consumed) {
      q.read_offset_dw = off;
      if (host_memory::IsMemoryRangeMapped(
              reinterpret_cast<const void*>(q.read_ptr), sizeof(u32)))
        *reinterpret_cast<u32*>(q.read_ptr) = off;
      if (kGcTrace)
        BASE_LOGI("acbscan", "{}/{}/{} drained {} dwords -> {:#x}", q.me,
                  q.pipe, q.queue, consumed, off);
    }
  }
}

u8* GcDevice::Map(void*, size_t size, u32, u32, size_t offset) {
  constexpr u64 kPoolSize = 256ull * 1024 * 1024;
  // The pool belongs to the device, not the descriptor: a GcDevice per open
  // would hand two openers different memory for the same offset; sys_mmap holds
  // no lock across device::map, so lazy creation needs its own.
  static base::Mutex pool_lock;
  static u8* pool = nullptr;
  base::LockGuard<base::Mutex> lk(pool_lock);
  if (!pool) {
    pool = AllocLowGuest(kPoolSize);
    if (!pool)
      return reinterpret_cast<u8*>(-1);
  }
  pool_base = pool;
  pool_size = kPoolSize;
  // The guest picks the offset, so bound each side: offset + size wraps.
  if (offset < kPoolSize && size <= kPoolSize - offset)
    return pool + offset;
  return reinterpret_cast<u8*>(-1);
}
}  // namespace kern

// sceGnmDingDong(ringId, offset) publishes a queue's write pointer. The real
// driver writes it into its /dev/gc mapping, so no ioctl announces it; draining
// only at flip time recycles buffers under us (SIGSEGV in Tomb Raider). Called
// once per flip so the backlog is not charged to whichever frame rang the
// doorbell.
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_dingdong(u32 ring_id, u32 offset_dw) {
  base::LockGuard lock(kern::GcDevice::g_compute_mutex);
  kern::GcDevice::RingDoorbell(ring_id, offset_dw);
}
// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gc_drain_acb(u32 budget_dw) {
  if (!budget_dw)
    return;
  base::LockGuard lock(kern::GcDevice::g_compute_mutex);
  kern::GcDevice::DrainQueues(budget_dw);
}
// NOLINTEND(readability-identifier-naming)
