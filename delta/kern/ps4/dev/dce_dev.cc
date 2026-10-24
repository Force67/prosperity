/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include "base/arch.h"
#include "base/logging.h"
#include "guest/display.h"
#include "guest/session.h"

#include "base/atomic.h"
#include "base/time/time.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_time.h"
#include "kern/process.h"
#include "kern/ps4/dev/dce_dev.h"
#include "options/options.h"
#include "host_memory/host_memory.h"
#include "gpu/ps4/cmd_processor.h"

namespace {
DELTA_OPTION(bool, kDceTrace, "DELTA_DCE_TRACE", false);
}  // namespace

namespace kern {
DceDevice::DceDevice(ObjectTable& objects) : Device(objects) {}

// Monotonic ns timestamp + a wall-clock ~60 Hz vblank counter. The GameMaker
// runner (and sceVideoOutWaitVblank) busy-poll VblankStatus until the count
// advances, so it MUST tick in real time despite no display hardware, else the
// title spins forever on frame 1.
static u64 NowNs() {
  return static_cast<u64>(base::TickClock::NowNs());
}
static u64 VblankCount() {
  return guest::display::VblankCount();
}
// A TSC value in the same domain/rate as the guest's rdtsc, so
// flip/vblank-status tsc fields stay comparable with rdtsc the title reads
// itself. On native that's the raw host TSC; on FEX use the 1.6 GHz wall clock
// the emulated rdtsc matches.
static u64 GuestTsc() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  return NowNs() * 16 / 10;  // ns -> 1.6 GHz ticks
#endif
}

// Submitted targets drive GC presentation; completed targets drive status.
static base::Atomic<u32> g_dce_current_buffer{UINT32_MAX};
static base::Atomic<i64> g_dce_flip_arg{0};
static base::Atomic<u64> g_dce_flip_count{0};
static base::Atomic<u64> g_dce_scanout_buffers[16]{};
static base::Atomic<bool> g_dce_active{false};
static base::Atomic<u32> g_dce_submitted_buffer{0};
static base::Atomic<i64> g_dce_submitted_arg{0};
static base::Atomic<u64> g_dce_submit_count{0};
static base::Atomic<u64> g_dce_flip_time{0};
static base::Atomic<u64> g_dce_flip_tsc{0};
static base::Atomic<u64> g_dce_submit_tsc{0};
struct PendingFlip {
  int index, mode;
  i64 arg;
};
static std::mutex g_dce_mutex;
static std::deque<PendingFlip> g_dce_gpu_flips;
static u64* g_dce_labels = nullptr;

bool DceActive() {
  return g_dce_active.load();
}

u32 DceSubmittedBuffer() {
  return g_dce_submitted_buffer.load();
}

i64 DceSubmittedFlipArg() {
  return g_dce_submitted_arg.load();
}

u64 DceScanoutBuffer(u32 index) {
  return index < 16 ? g_dce_scanout_buffers[index].load() : 0;
}

static bool QueueFlip(PendingFlip flip) {
  return guest::display::QueueFlip(flip.mode, [flip] {
    {
      std::lock_guard lock(g_dce_mutex);
      // PS4 releases the previous scanout buffer when its replacement flips.
      if (g_dce_labels && g_dce_current_buffer.load() < 16)
        g_dce_labels[g_dce_current_buffer.load()] = 0;
      g_dce_current_buffer.store(flip.index);
      g_dce_flip_arg.store(flip.arg);
      g_dce_flip_time.store(NowNs() / 1000);
      g_dce_flip_tsc.store(GuestTsc());
      g_dce_flip_count.fetch_add(1);
    }
    NoteFlip();
  });
}

bool DceSubmitFlip(int index, int mode, i64 arg, bool wait_for_gpu) {
  if (index < -1 || index >= 16 || mode < 0 || mode > 2)
    return false;
  NoteProcTimeFlip();
  g_dce_submitted_buffer.store(index);
  g_dce_submitted_arg.store(arg);
  g_dce_submit_tsc.store(GuestTsc());
  g_dce_submit_count.fetch_add(1);
  PendingFlip flip{index, mode, arg};
  if (wait_for_gpu) {
    std::lock_guard lock(g_dce_mutex);
    g_dce_gpu_flips.push_back(flip);
    return true;
  }
  gpu::ps4::WaitForFrame();
  if (QueueFlip(flip))
    return true;
  g_dce_submit_count.fetch_sub(1);
  return false;
}

bool DceCompleteGpuFlips() {
  std::deque<PendingFlip> flips;
  {
    std::lock_guard lock(g_dce_mutex);
    flips.swap(g_dce_gpu_flips);
  }
  size_t remaining = flips.size();
  for (auto flip : flips) {
    if (!QueueFlip(flip)) {
      g_dce_submit_count.fetch_sub(remaining);
      return false;
    }
    --remaining;
  }
  return true;
}

static bool DceTrace() {
  return kDceTrace;
}

// Scan the raw stack for the first return address in video output client's .text
// (the native backend runs handlers on the guest stack) to pin which wrapper
// issued each ioctl.
static void PrintVideoOutCaller() {
  auto* proc = Process::GetActive();
  if (!proc)
    return;
  uintptr_t vbase = 0, vsize = 0;
  for (auto& m : proc->GetModuleList()) {
    auto& mi = m->GetInfo();
    if (std::strcmp(mi.name.c_str(), "libSceVideoOut") == 0) {
      vbase = (uintptr_t)mi.text_seg.addr;
      vsize = mi.text_seg.size;
      break;
    }
  }
  if (!vbase)
    return;
  auto* sp = reinterpret_cast<uintptr_t*>(__builtin_frame_address(0));
  int shown = 0;
  uintptr_t lastoff = ~0ull;
  // The caller return addresses sit within the first few frames; a wide scan
  // can walk off the top of the guest stack (an unmapped page) and fault. 128
  // qwords (1 KiB) is well within the mapped stack and finds the wrapper
  // reliably.
  for (int i = 0; i < 128 && shown < 6; i++) {
    uintptr_t v = sp[i];
    if (v >= vbase && v < vbase + vsize) {
      uintptr_t off = v - vbase;
      if (off == lastoff)
        continue;  // skip repeated return-address slots
      BASE_LOGI("dce", "  caller libSceVideoOut+{:#x}", off);
      lastoff = off;
      shown++;
    }
  }
}

bool DceDevice::Init(const char*, u32, u32) {
  g_dce_active.store(true);
  return true;
}

// Guard dereferences to a sane userspace range. NB: under FEX the guest stack
// is a host mmap up at 0xffff_xxxx_xxxx, so real video output client passes out-slot
// pointers above the old 0x8000_0000_0000 ceiling; accept the full 48-bit user
// range or every stack out-slot write is silently dropped (open then mmaps an
// uninitialised offset/size and fails).
static bool PlausiblePtr(u64 v) {
  return v >= 0x10000 && v < 0x0001000000000000ull;
}

// Dump an ioctl arg struct as u64s. Just the values: do NOT chase "pointers"
// (many fields are two packed u32s like {pitch, height} that look like a high
// address but aren't, and dereferencing them faults). DELTA_DCE_TRACE.
static void DumpStruct(const void* data, u32 len) {
  const u64* q = static_cast<const u64*>(data);
  u32 n = len / 8;
  for (u32 i = 0; i < n; i++)
    BASE_LOGI("dce", "    [{}] {:#018x}", i, (unsigned long long)q[i]);
}

u64 DceDevice::PoolAlloc(u64 bytes) {
  // The scanout/control pool. One big guest-addressable region, bump-allocated;
  // every sub-op 9 carve is mmap'd back via map(offset). 512 MiB covers several
  // 1080p buffer sets + control blocks.
  constexpr u64 kPool = 512ull * 1024 * 1024;
  if (!pool_base_) {
    pool_base_ = AllocLowGuest(kPool);
    if (!pool_base_)
      return UINT64_MAX;
    pool_size_ = kPool;
  }
  bytes = (bytes + 0x3FFF) & ~u64(0x3FFF);
  if (pool_used_ + bytes > pool_size_)
    return UINT64_MAX;
  u64 off = pool_used_;
  pool_used_ += bytes;
  return off;
}

u8* DceDevice::Map(void*, size_t size, u32, u32, size_t offset) {
  // The kernel only maps /dev/dce at offsets < 0x8000 (physical flip-target
  // status page; offset/0x4000 picks the target). Our video output client maps what
  // sub-op 9 hands back, whose offsets exceed that window, so substitute a
  // slice of the scanout pool: real, zeroed, shared, CP-usable memory in the
  // same shape.
  if (pool_base_ && offset <= pool_used_ && size <= pool_used_ - offset)
    return pool_base_ + offset;
  return reinterpret_cast<u8*>(-1);
}

// 11.00 dce ioctl dispatch: scanin ioctls are system-credential-gated (a game
// gets EINVAL); 0xC0308203 -> flip control (sub-ops 0..0x19), 0xC0308206/07 ->
// register buffer(/attr), 0xC0488204 -> submit flip, 0xc0588212 not in the
// dispatch (EINVAL).
i32 DceDevice::Ioctl(u32 cmd, void* data) {
  if (DceTrace()) {
    u32 len = (cmd >> 16) & 0x1FFF;
    BASE_LOGI("dce", "ioctl cmd={:#x} len={} data={:p}", cmd, len, data);
    PrintVideoOutCaller();
    if (data && len && len <= 0x200)
      DumpStruct(data, len);
  }

  auto* s = static_cast<u64*>(data);

  // Scanin ioctls + the non-existent 0xc0588212: EINVAL from a game context,
  // which is what a game's video output client is built against.
  switch (cmd) {
    case 0x80108210:  // start dual capture mode
    case 0xC068820C:  // start capture buffer manager
    case 0xC020820E:  // get capture buffer info
    case 0xC020820F:  // execute capture command
    case 0xC004820D:  // stop capture buffer manager
    case 0xc0588212:  // was "cursor pool allocator"; not in the kernel dispatch
      return -SysError::eINVAL;
  }

  if (cmd == 0xc0308203 && data) {
    switch (s[0]) {
      case 0:
        // Open display pipe. arg[0x20] (s[4]) points at the caller's ctx+0x438
        // slot; the kernel writes the opaque display handle there. Every later
        // ioctl passes that handle as its first u64, so hand back a non-zero
        // id.
        if (PlausiblePtr(s[4]))
          *reinterpret_cast<u64*>(s[4]) = next_handle_++;
        return 0;
      case 6:
        // video output client's flip-rate backend passes {6, handle, rate}.
        guest::display::SetFlipRate(static_cast<int>(s[2]));
        return 0;
      case 9: {
        // Sub-op 9 (scanout-pool offset query): the kernel writes offset=0x4000
        // and size=(bufferCount<<14) to s[2]/s[3] and the module mmaps
        // fd@offset for that size; we hand back a pool slice instead (same
        // shape, real backing). s[3] is a pure OUT slot on some callers (the
        // open path passes it uninitialised), so treat *s[3] as a requested
        // size only when sane, else default; a bogus huge size made PoolAlloc
        // fail ENOMEM and left the slots uninitialised.
        u64 reqd = PlausiblePtr(s[3]) ? *reinterpret_cast<u64*>(s[3]) : 0;
        u64 want = (reqd > 0 && reqd <= 0x10000000) ? reqd        // <= 256 MiB
                                                    : 0x4000000;  // 64 MiB
        u64 off = PoolAlloc(want);
        if (off == UINT64_MAX)
          return 12;  // ENOMEM
        if (PlausiblePtr(s[2]))
          *reinterpret_cast<u64*>(s[2]) = off;
        if (PlausiblePtr(s[3]))
          *reinterpret_cast<u64*>(s[3]) = want;
        {
          std::lock_guard lock(g_dce_mutex);
          g_dce_labels = reinterpret_cast<u64*>(pool_base_ + off + 0xa0);
        }
        return 0;
      }
      case 0x19: {
        // Resolution status. arg[0x10] (s[2]) = out ptr, arg[0x18] (s[3]) = out
        // size (0x30). The module copies width/height/pane (u32 @0/4/8/0xc),
        // a flag byte @0x10, refresh @0x18, aspect @0x20. Report 1920x1080.
        if (PlausiblePtr(s[2])) {
          auto* o = reinterpret_cast<u8*>(s[2]);
          std::memset(o, 0, 0x30);
          *reinterpret_cast<u32*>(o + 0x00) = 1920;  // width
          *reinterpret_cast<u32*>(o + 0x04) = 1080;  // height
          *reinterpret_cast<u32*>(o + 0x08) = 1920;  // paneWidth
          *reinterpret_cast<u32*>(o + 0x0c) = 1080;  // paneHeight
          *reinterpret_cast<u64*>(o + 0x18) = 60;    // refresh
        }
        return 0;
      }
      case 0xa: {
        // Get flip status: s[2] = out ptr (0x48). Kernel layout: [0x00] flipArg
        // [0x10] count [0x18] processTime [0x20] tsc [0x28] currentBuffer
        // [0x2c]+[0x34] flipPendingNum [0x30] gcQueueNum [0x38] submitTsc.
        if (PlausiblePtr(s[2])) {
          auto* o = reinterpret_cast<u8*>(s[2]);
          std::memset(o, 0, 0x48);
          std::lock_guard lock(g_dce_mutex);
          *reinterpret_cast<i64*>(o + 0x00) = g_dce_flip_arg.load();  // flipArg
          *reinterpret_cast<u64*>(o + 0x10) = g_dce_flip_count.load();
          *reinterpret_cast<u64*>(o + 0x18) = g_dce_flip_time.load();
          *reinterpret_cast<u64*>(o + 0x20) = g_dce_flip_tsc.load();
          // Polling this index must not release a buffer before completion.
          *reinterpret_cast<i32*>(o + 0x28) = (i32)g_dce_current_buffer.load();
          *reinterpret_cast<u64*>(o + 0x38) = g_dce_submit_tsc.load();
          const u32 gc_pending = g_dce_gpu_flips.size();
          *reinterpret_cast<u32*>(o + 0x2c) =
              g_dce_submit_count.load() - g_dce_flip_count.load() - gc_pending;
          *reinterpret_cast<u32*>(o + 0x30) = gc_pending;
        }
        return 0;
      }
      case 0xb: {
        // Get vblank status: s[2] = out ptr (0x28): [0x00] count [0x08]
        // processTime [0x10] tsc [0x18] flags. count must advance in real time
        // (see vblankCount).
        if (PlausiblePtr(s[2])) {
          auto* o = reinterpret_cast<u8*>(s[2]);
          std::memset(o, 0, 0x28);
          u64 now = NowNs();
          *reinterpret_cast<u64*>(o + 0x00) = VblankCount();  // count
          *reinterpret_cast<u64*>(o + 0x08) = now;            // processTime
          *reinterpret_cast<u64*>(o + 0x10) = GuestTsc();     // tsc
          o[0x18] = 0;                                        // flags
        }
        return 0;
      }
      case 0xc: {
        // Scaler-setup query: out[0x00] != 0 only when a NEW scaler config is
        // pending; titles never reconfigure after boot, so report none (a
        // non-zero handle makes the service spin posting bogus events).
        // Flip-done is the EVFILT_DISPLAY event.
        if (PlausiblePtr(s[2]))
          std::memset(reinterpret_cast<void*>(s[2]), 0, 0x40);
        return 0;
      }
      case 0x1f:  // NOT a valid kernel sub-op: the flip-control switch only
                  // Sub-ops 0..0x19; 0x1f falls to the EINVAL default (kernel
                  // jump table
                  // -> 0x826e9679). The "open-time capability header" was a
                  // soft-success invention; retail sees EINVAL here.
        return -SysError::eINVAL;
      default:
        // Other query/config sub-ops (1, 6, 0xc, ...). Soft-succeed without
        // touching arg fields whose role (pointer vs scalar) we haven't pinned.
        if (DceTrace())
          BASE_LOGI("dce", "0xc0308203 sub-op {:#x} -> 0",
                    (unsigned long long)s[0]);
        return 0;
    }
  }

  if (cmd == 0xc0308206 && data) {
    // Register one scanout buffer: {display handle, buffer index, GPU base}.
    // Preserve the address so a later GC submit-and-flip can present the render
    // target selected by the DCE flip rather than whichever RT was drawn last.
    if (s[1] < 16)
      g_dce_scanout_buffers[s[1]].store(s[2]);
    return 0;
  }

  if (cmd == 0xc0308207) {
    // Register one scanout buffer's format/tiling attribute. Input only; the
    // buffer GPU VA the module flips later comes from the game's own registered
    // address, not from here. Accept it.
    return 0;
  }

  if (cmd == 0xc0488204 && data) {
    if ((s[1] >= 16 && s[1] != UINT64_MAX) || s[2] < 1 || s[2] > 6 ||
        !host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(s[8]), 8))
      return -static_cast<i32>(SysError::eINVAL);
    // The EOP out-slot belongs to the caller's stack. Retain only flip values;
    // GC queues completion after executing the submitted command buffers.
    if (!s[5])
      gpu::ps4::EndFrame(DceScanoutBuffer(static_cast<u32>(s[1])));
    if (!DceSubmitFlip(static_cast<int>(s[1]), (s[2] - 1) % 3 == 1 ? 2 : 1,
                       static_cast<i64>(s[3]), s[5] != 0))
      return -static_cast<i32>(SysError::eINTR);
    if (DceTrace())
      BASE_LOGI("dce", "submitFlip buf={} flipArg={:#x}", (int)s[1],
                (unsigned long long)s[3]);
    // A zero result is success; 0x58 makes the library refresh its scaler.
    if (PlausiblePtr(s[8]))
      *reinterpret_cast<u64*>(s[8]) = 0;
    return 0;
  }

  if (DceTrace())
    BASE_LOGI("dce", "UNHANDLED ioctl {:#x} -> 0", cmd);
  return 0;
}

namespace {
const guest::SessionReset g_session_reset([] {
  std::lock_guard lock(g_dce_mutex);
  g_dce_current_buffer.store(UINT32_MAX);
  guest::ResetResource(g_dce_flip_arg);
  guest::ResetResource(g_dce_flip_count);
  guest::ResetResource(g_dce_scanout_buffers);
  guest::ResetResource(g_dce_active);
  guest::ResetResource(g_dce_submitted_buffer);
  guest::ResetResource(g_dce_submitted_arg);
  guest::ResetResource(g_dce_submit_count);
  guest::ResetResource(g_dce_flip_time);
  guest::ResetResource(g_dce_flip_tsc);
  guest::ResetResource(g_dce_submit_tsc);
  g_dce_gpu_flips.clear();
  g_dce_labels = nullptr;
});
}  // namespace

}  // namespace kern
