/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/gc device: the libSceAgc / libSceAgcDriver AGC command protocol. This
 * is a dedicated PS5 device, split from the PS4 gcDevice (GNM PM4) so the two
 * unrelated ioctl command sets never share a switch. Forwards the AGC DCB to the
 * PS5 command processor (gpu/ps5).
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include <sys/mman.h>

#include <utl/mem.h>

#include "gc_dev.h"
#include "kern/ps4/dev/dma_dev.h"  // dmemBackingFd/Size (shared physical dmem store)
#include "kern/proc.h"
#include "kern/lv2/sys_mem.h"  // allocLowGuest, mFlags
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kAgcRingdump, "DELTA_AGC_RINGDUMP", false);
DELTA_OPTION(bool, kAgcTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kDmemTrace, "DELTA_DMEM_TRACE", false);
DELTA_OPTION(bool, kFlipTrace, "DELTA_FLIP_TRACE", false);
DELTA_OPTION(bool, kGcCensus, "DELTA_GC_CENSUS", false);
DELTA_OPTION(bool, kGcIoctlCensus, "DELTA_GC_IOCTL_CENSUS", false);
DELTA_OPTION(bool, kGcTrace, "DELTA_GC_TRACE", false);
}  // namespace

// PS5 AGC submit bridge (delta_gpu, gpu/ps5): forward the DCB to the PS5 command
// processor, which follows INDIRECT_BUFFER chains and decodes the draws.
extern "C" void prosperity_agc_submit(u64 dcbBase, u32 sizeBytes);
// PS5 present bridge: end the frame and present the rendered RT to the window.
extern "C" void prosperity_agc_flip(u64 scanoutBase);
// Is this address inside a pool the title mapped for the GPU (gpu/ps5)?
extern "C" int prosperity_gpu_is_aperture(u64 address);

// Set once the title issues the mode-1 end-of-frame ioctl. Until then the frame
// has to be ended somewhere, and the start of a state submit is the only other
// boundary available.
static std::atomic<bool> g_sawEndOfFrame{false};
// Guest address of the display buffer the game most recently flipped, resolved
// from sceVideoOutSubmitFlip*'s bufferIndex via the registered-buffer table
// (libSceVideoOut_ps5.cpp). The AGC flip ioctls below carry no buffer field, so
// they present this instead of falling back to whichever RT was drawn last.
extern "C" u64 prosperity_ps5_scanout_base();

// DELTA_FLIP_TRACE: log the scanout base each AGC flip presents, so the derived
// per-flip buffer can be checked against the registered display buffers.
static void traceFlip(const char *site, u64 base) {
  if (kFlipTrace)
    BASE_LOGI("flip", "{} present={:#x}", site, (unsigned long)base);
}

namespace krnl {
gcDevicePs5::gcDevicePs5(proc *p) : device(p) {}

bool gcDevicePs5::init(const char *, u32, u32) { return true; }

// Diagnostic (DELTA_AGC_TRACE): scan one GPU-aperture page for a draw-DCB PM4
// header to locate the command buffer if a submit-arg pointer reads zero.
static void scanPagePm4(void *ctx, u8 *p, size_t sz) {
  int *hits = static_cast<int *>(ctx);
  auto isDcb = [](u32 h) {
    if ((h >> 30) != 3) return false;
    u32 op = (h >> 8) & 0xFF;
    return op == 0x69 || op == 0x76 || op == 0x79 || op == 0x3F || op == 0x2D ||
           op == 0x27 || op == 0x35 || op == 0x15;
  };
  auto *w = reinterpret_cast<const u32 *>(p);
  u64 n = sz / 4;
  if (n > 0x400000) n = 0x400000;
  int perPage = 0;
  for (u64 j = 0; j < n && *hits < 24 && perPage < 4; j++) {
    if (isDcb(w[j])) {
      BASE_LOGI("agc", "  DCB@{:#x} hdr={:08x} op={:#x}",
                (unsigned long)(reinterpret_cast<u64>(p) + j * 4), w[j],
                (w[j] >> 8) & 0xFF);
      (*hits)++;
      perPage++;
    }
  }
}

// A guest GPU address: one question, answered by gpu/ps5/guest_address.h, which
// knows both the assumed band and the pools the title really mapped. A fixed
// band silently drops every command buffer another title allocates outside it,
// so nothing renders and the game waits forever on a GPU label the dropped
// submits would have written -- which is exactly what Astro Bot did.
static inline bool gpuAddr(u64 a) {
  return prosperity_gpu_is_aperture(a) != 0;
}

// The submit paths and trace probes below deref candidate pointers pulled out of
// a submit arg. Plenty of those words look like GPU addresses without being
// mapped, so the range check alone is not enough to read through one.
static inline bool gpuReadable(u64 a, size_t n) {
  return gpuAddr(a) && utl::isMemoryRangeMapped(reinterpret_cast<void *>(a), n);
}

// Span of ACQ ring windows the driver named in its 0xC0408121 submits, learned at
// run time rather than hardcoded (each title's ring sits wherever its driver
// mapped it). Lets the per-frame trace re-read the ring long after the 0x8121
// ioctls have stopped.
static u64 g_acqRingLo = 0, g_acqRingHi = 0;

// One ACQ ring the title created through sceAgcDriverCreateQueue (ioctl
// 0xC0408121). Steady-state submits never reach the kernel at all: the title
// writes its ring write-pointer to the queue's doorbell slot and the hardware
// command processor picks the work up from there. Recording what each create
// named is the only chance we get to learn where a queue's ring lives.
struct AcqQueue {
  u64 dcb = 0;       // command ring
  u64 ccb = 0;       // dcb + ringBytes: the ring READ POINTER (see below)
  u64 doorbell = 0;  // 8-byte write-pointer slot in the DingDong page
  u32 ringBytes = 0;
  u64 lastDoorbell = 0;
  u32 readDw = 0;  // how far we have walked, in dwords
};

static std::mutex g_queueLock;
static std::map<u32, AcqQueue> g_queues;  // by 1-based queue id

// The doorbell value is the ring write pointer. Advance our own read pointer to
// it and hand the command processor the dwords in between, unwrapping the ring.
static void drainQueue(AcqQueue &q, u64 doorbell) {
  const u32 ringDw = q.ringBytes / 4;
  if (!ringDw)
    return;
  const u32 write = static_cast<u32>(doorbell % ringDw);
  if (write == q.readDw)
    return;
  auto forward = [&](u32 firstDw, u32 dwords) {
    const u64 at = q.dcb + static_cast<u64>(firstDw) * 4;
    if (dwords && gpuReadable(at, static_cast<size_t>(dwords) * 4))
      prosperity_agc_submit(at, dwords * 4);
  };
  if (write > q.readDw) {
    forward(q.readDw, write - q.readDw);
  } else {  // wrapped
    forward(q.readDw, ringDw - q.readDw);
    forward(0, write);
  }
  q.readDw = write;
  // Report the read pointer back, or the ring only ever fills. The driver keeps
  // one dword right past the ring for it (libSceAgcDriver+0x2226 stores
  // `dcb + 0x4000` into its queue struct and zeroes the word; +0x11f0 spins on
  // it, computing free space as `rptr - (write % ringDw)` wrapped, and waits
  // for more than 8 dwords). Astro Bot's DrawThread parks in exactly that spin
  // -- holding its frame mutex, so the main thread blocks behind it and the
  // title never submits again.
  const u64 rptr = q.dcb + q.ringBytes;
  if (gpuReadable(rptr, sizeof(u32)))
    *reinterpret_cast<volatile u32 *>(rptr) = q.readDw;
}

// Poll every registered doorbell. A real command processor is woken by the
// doorbell write; we have no way to trap it cheaply, and the page is a handful
// of cache lines, so a poll costs nothing next to a frame.
static void doorbellPoller() {
  for (;;) {
    std::this_thread::sleep_for(std::chrono::microseconds(500));
    std::lock_guard<std::mutex> lk(g_queueLock);
    for (auto &[qid, q] : g_queues) {
      if (!gpuReadable(q.doorbell, sizeof(u64)))
        continue;
      const u64 now =
          *reinterpret_cast<volatile const u64 *>(q.doorbell);
      if (now == q.lastDoorbell)
        continue;
      static int rung = 0;
      if (kAgcTrace && rung < 32) {
        rung++;
        BASE_LOGI("agc", "doorbell q{} {:#x} -> {:#x} (ring {:#x} +{:#x})", qid,
                  (unsigned long)q.lastDoorbell, (unsigned long)now,
                  (unsigned long)q.dcb, q.ringBytes);
      }
      q.lastDoorbell = now;
      drainQueue(q, now);
    }
  }
}

static void registerAcqQueue(u32 qid, u64 dcb, u64 ccb, u64 doorbellBase,
                             u32 ringLog2Dw) {
  if (!qid || !gpuAddr(dcb) || ringLog2Dw > 24)
    return;
  std::lock_guard<std::mutex> lk(g_queueLock);
  AcqQueue &q = g_queues[qid];
  q.dcb = dcb;
  q.ccb = ccb;
  q.doorbell = doorbellBase + static_cast<u64>(qid - 1) * 8;
  q.ringBytes = 4u << ringLog2Dw;
  q.lastDoorbell = gpuReadable(q.doorbell, sizeof(u64))
                       ? *reinterpret_cast<volatile const u64 *>(q.doorbell)
                       : 0;
  q.readDw = static_cast<u32>(q.lastDoorbell % (q.ringBytes / 4));
  static bool polling = false;
  if (!polling) {
    polling = true;
    std::thread(doorbellPoller).detach();
  }
}

// The mode-1 submit ioctls are INOUT on firmware 13.60: libSceAgcDriver presets a
// status dword in the arg and, on return, treats the submit as FAILED unless the
// kernel has cleared it. A failed state submit makes the driver skip the 0x8132
// call that carries the title's own command buffer, so the frame is dropped
// entirely -- no draws reach us, and the completion label the title spins on is
// never written. The IN-only variants older firmware issues have no such field.
static void clearSubmitStatus(u32 cmd, void *data, u32 offset) {
  if (!data || !(cmd & 0x40000000u))
    return;
  const u32 len = (cmd >> 16) & 0x1fff;
  if (offset + sizeof(u32) > len)
    return;
  std::memset(static_cast<u8 *>(data) + offset, 0, sizeof(u32));
}

// GNM-style submit descriptor array: each 4-dword entry is an IT_INDIRECT_BUFFER
// (0xC0023F00 = dcb) / _CNST (0xC0023300 = ccb) with [hdr, addrLo, addrHi&0xFF,
// sizeDwords]. libSceAgcDriver/GnmDriver submits the pipeline+shader setup and
// draws through these on PS5 too; forward each buffer to the PS5 command processor
// (they carry the SET_SH_REG shader binding the AGC mode-1 path never emits).
static void submitGnmDescArray(u64 descPtr, u32 count) {
  const u32 *d = reinterpret_cast<const u32 *>(descPtr);
  if (!d || count > 0x1000) return;
  for (u32 i = 0; i < count; i++) {
    const u32 *e = d + i * 4;
    u32 hdr = e[0];
    u64 addr = (static_cast<u64>(e[2] & 0xFF) << 32) | e[1];
    u32 bytes = (e[3] & 0xFFFFF) * 4;
    if (bytes && (hdr == 0xC0023F00u || hdr == 0xC0023300u) &&
        gpuReadable(addr, bytes))
      prosperity_agc_submit(addr, bytes);
  }
}

i32 gcDevicePs5::ioctl(u32 cmd, void *data) {
  if (kGcIoctlCensus) {
    static std::mutex mtx;
    static std::map<u32, u64> hist;
    static auto last = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mtx);
    hist[cmd]++;
    auto now = std::chrono::steady_clock::now();
    if (now - last > std::chrono::seconds(10)) {
      last = now;
      BASE_LOGI("gcioctl", "--- 10s census ---");
      for (auto &[c, n] : hist)
        BASE_LOGI("gcioctl", "{:#x} {}", c, (unsigned long long)n);
    }
  }
  if (kGcTrace)
    BASE_LOGI("gc", "ioctl({:x}) data={:p}", cmd, data);
  switch (cmd) {
  case 0xC0108102: {  // GNM submit: {u32 a0, u32 count, u64 descPtr}
    struct argl { u32 a0; u32 count; u64 descPtr; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    return 0;
  }
  case 0xC018810A: {  // GNM submit variant: {a0, count, a2, pad, descPtr}
    struct argl { u32 a0, count, a2, pad; u64 descPtr; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    return 0;
  }
  case 0xC020810C: {  // GNM submit-and-flip: {a0, count, descPtr, flipPtr, flag}
    struct argl { u32 a0, count; u64 descPtr, flipPtr; u32 flag; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    u64 scanout = prosperity_ps5_scanout_base();
    traceFlip("0xC020810C", scanout);
    prosperity_agc_flip(scanout);  // present the flipped display buffer
    return 0;
  }
  case 0xC008811B: {
    // GNM submit-state pointer. libSceGnmDriver stores the returned address in a
    // global and reads through it on every submit path
    // (sceGnmAreSubmitsAllowed is `*p == 0`, sceGnmSubmitDone tests it, so do
    // SubmitCommandBuffers/SubmitAndFlip/DingDong). The soft-succeed default
    // zeroes the OUT slot, so the driver cached a null pointer and every submit
    // after the first one read through it. Hand back a real zeroed page:
    // [+0] == 0 means "submits allowed".
    static u8 *submitState = nullptr;
    if (!submitState)
      submitState = allocLowGuest(0x100);
    if (data)
      *static_cast<u64 *>(data) = reinterpret_cast<u64>(submitState);
    BASE_LOGI("gc", "ioctl({:x}): submit-state -> {:p}", cmd, submitState);
    return 0;
  }
  case 0xC0108139: {
    // AGC suspend-point submit, issued at the tail of every submit. The driver
    // fails the submit with 0x8A6D0107 ("checkSuspend failure") unless this
    // succeeds; its two out words are a suspend sequence number, so advance
    // them rather than reporting the same value forever.
    static std::atomic<u64> suspendSeq{0};
    if (data) {
      const u64 seq = ++suspendSeq;
      auto *a = static_cast<u8 *>(data);
      std::memset(a, 0, 0x10);
      const u32 lo = static_cast<u32>(seq);
      std::memcpy(a, &lo, 4);
      std::memcpy(a + 8, &seq, 8);
    }
    return 0;
  }
  case 0x40048135:  // AGC query: OUT dword (submit/queue id). The driver stores
                    // it; 0 is accepted.
    if (data)
      *static_cast<u32 *>(data) = 0;
    return 0;
  case 0xC004812E:  // AGC init: INOUT dword. 0 tells AgcDriver to map its own
                    // submit doorbell (which then succeeds).
    if (data)
      *static_cast<u32 *>(data) = 0;
    return 0;
  case 0xC0408121: {  // sceAgcDriverCreateQueue (IN, 64 bytes):
                      //   +0x00 me  +0x04 pipe  +0x08 queue  +0x0c 1-based qid
                      //   +0x10 dcb  +0x18 ccb (= dcb + ring)  +0x20 doorbell page
                      //   +0x28 log2(ring dwords)  +0x2c flags
                      //   +0x30 mqd  +0x38 mqd size
                      // NOT a submit: it hands the title a ring plus a doorbell
                      // slot, and every later submit is a store of the ring
                      // write pointer to that slot with no ioctl at all. Record
                      // the queue so the poller can drain it; the ring is empty
                      // now, by construction.
    if (data) {
      auto *a = static_cast<u8 *>(data);
      u64 base = 0, base2 = 0, doorbellBase = 0;
      u32 qid = 0, ringLog2Dw = 0;
      std::memcpy(&base, a + 0x10, 8);
      std::memcpy(&base2, a + 0x18, 8);
      std::memcpy(&doorbellBase, a + 0x20, 8);
      std::memcpy(&qid, a + 0x0c, 4);
      std::memcpy(&ringLog2Dw, a + 0x28, 4);
      if (gpuAddr(doorbellBase))
        registerAcqQueue(qid, base, base2, doorbellBase, ringLog2Dw);
      u32 size = 0x8000;
      if (gpuAddr(base)) {
        if (!g_acqRingLo || base < g_acqRingLo) g_acqRingLo = base;
        if (base + size > g_acqRingHi) g_acqRingHi = base + size;
      }
      static int dumps = 0;
      static u64 calls = 0;
      ++calls;
      // The first few submits are engine init, where the ring legitimately holds
      // nothing. Sample later ones too, or "the ring reads empty" only ever
      // describes start-up.
      if (kAgcTrace && (dumps < 8 || (calls % 200 == 0 && dumps < 12))) {
        dumps++;
        BASE_LOGI("agc", "--- submit #{} ---", (unsigned long long)calls);
        auto *w = reinterpret_cast<u32 *>(a);
        base::String line;
        base::FormatTo(line, "submit arg[0..15]:");
        for (int k = 0; k < 16; k++)
          base::FormatTo(line, " {:08x}", w[k]);
        base::FormatTo(line, "\n  dcb0={:#x} dcb1={:#x} size={}", (unsigned long)base,
                       (unsigned long)base2, size);
        BASE_LOGI("agc", "{}", line.c_str());
        for (int k = 0; k + 1 < 16; k++) {
          u64 p = (static_cast<u64>(w[k + 1]) << 32) | w[k];
          if (gpuReadable(p, 32)) {
            auto *pw = reinterpret_cast<const u32 *>(p);
            base::String line;
            base::FormatTo(line, "  arg[{}] ptr={:#x} ->", k, (unsigned long)p);
            for (int j = 0; j < 8; j++) base::FormatTo(line, " {:08x}", pw[j]);
            BASE_LOGI("agc", "{}", line.c_str());
          }
        }
        if (auto *pr = proc::getActive()) {
          int hits = 0;
          pr->getVma().forEachGpuAperturePage(scanPagePm4, &hits);
          if (!hits)
            BASE_LOGI("agc", "  no PM4 anywhere in the GPU aperture (empty ring?)");
        }
        // Distinguish "the title wrote nothing" from "we are reading a mapping
        // that does not see its writes": count non-zero dwords in the ring
        // window, regardless of whether they look like PM4.
        if (gpuReadable(base, 0x8000)) {
          auto *rw = reinterpret_cast<const u32 *>(base);
          u32 nz = 0, first = 0;
          for (u32 k = 0; k < 0x8000 / 4; k++)
            if (rw[k]) { if (!nz) first = k; nz++; }
          BASE_LOGI("agc", "  ring {:#x}: {}/{} dwords non-zero (first @dw {})",
                    (unsigned long)base, nz, 0x8000u / 4, first);
        }
      }
      // The window a submit names is empty AT IOCTL TIME if the driver fills it
      // afterwards and kicks the GPU through the doorbell instead. Re-read the
      // PREVIOUS submit's window here: if it has packets now, the ioctl is a
      // ring-window acquire and the submit boundary is the doorbell, not this.
      static u64 prevBase = 0;
      if (kAgcTrace && prevBase && prevBase != base && gpuReadable(prevBase, 0x8000)) {
        auto *pw = reinterpret_cast<const u32 *>(prevBase);
        u32 nz = 0;
        for (u32 k = 0; k < 0x8000 / 4; k++)
          if (pw[k]) nz++;
        if (nz && dumps <= 12) {
          base::String line;
          base::FormatTo(line, "  prev ring {:#x} now {} dwords non-zero:",
                         (unsigned long)prevBase, nz);
          for (int k = 0; k < 12; k++) base::FormatTo(line, " {:08x}", pw[k]);
          BASE_LOGI("agc", "{}", line.c_str());
        }
      }
      prevBase = base;
      // DELTA_AGC_RINGDUMP: the submit arg in full plus the ring descriptor table
      // it references, with a PM4 sniff of each buffer. The per-pass register
      // state Skyrim never seems to program (a colour target for some passes, PS
      // user data above 15) has to come from one of these.
      static int ringN = 0;
      if (kAgcRingdump && ringN < 3) {
        ringN++;
        auto *w = reinterpret_cast<const u32 *>(a);
        base::String line;
        base::FormatTo(line, "arg:");
        for (int k = 0; k < 16; k++) base::FormatTo(line, " {:08x}", w[k]);
        BASE_LOGI("ring", "{}", line.c_str());
        auto sniff = [](u64 p, const char *what) {
          if (!gpuReadable(p, 1024)) return;
          const auto *q = reinterpret_cast<const u32 *>(p);
          u32 t3 = 0;
          for (int i = 0; i < 256; i++)
            if ((q[i] >> 30) == 3) t3++;
          BASE_LOGI("ring",
                    "  {} {:#x}: {:08x} {:08x} {:08x} {:08x}  (type3 hdrs in 256 dw: {})",
                    what, (unsigned long)p, q[0], q[1], q[2], q[3], t3);
        };
        for (int k = 0; k + 1 < 16; k++)
          sniff((static_cast<u64>(w[k + 1]) << 32) | w[k], "argptr");
        const u64 tbl = 0x80014981d8ull;
        if (gpuAddr(tbl)) {
          const auto *t = reinterpret_cast<const u32 *>(tbl);
          base::String line;
          base::FormatTo(line, "table @{:#x}:", (unsigned long)tbl);
          for (int k = 0; k < 16; k++) base::FormatTo(line, " {:08x}", t[k]);
          BASE_LOGI("ring", "{}", line.c_str());
          for (int k = 0; k + 1 < 16; k += 2)
            sniff((static_cast<u64>(t[k + 1] & 0xFFFF) << 32) | t[k], "tblptr");
        }
      }
      // A ring that already holds packets belongs to a title that filled it
      // before asking for the queue; the poller would still catch it, but only
      // once its doorbell moves again.
      if (gpuReadable(base, size) &&
          *reinterpret_cast<const u32 *>(base) != 0)
        prosperity_agc_submit(base, size);
      std::memset(a, 0, 64);
    }
    return 0;
  }
  case 0xC0048125: {  // AGC submit.mode=1 completion poll (INOUT, 4 bytes). The
                      // render loop submits (0x80488131) then reads this for GPU
                      // progress; our submit is synchronous, so report a monotonic
                      // counter that always satisfies a ">= submitted id" wait. Left
                      // at 0 the title spins re-submitting forever.
    if (data) {
      static u32 s_agcDone = 0;
      *static_cast<u32 *>(data) = ++s_agcDone;
    }
    return 0;
  }
  // Firmware 13.60 issues the mode-1 family as INOUT and widens the 0x8132 arg
  // from 16 to 24 bytes. Both payloads are otherwise unchanged (0x8132's two
  // extra dwords sit past the descriptor fields read below), so the directions
  // share a case.
  case 0xC0488131:
  case 0x80488131: {  // AGC submit.mode=1 submit (IN, 72 bytes). The arg IS a small
                      // command buffer: leading filler then IT_INDIRECT_BUFFER
                      // packets pointing at the real per-frame PM4. Forward it to the
                      // command processor, which follows the IBs and renders.
    if (data) {
      // Present the previous frame's accumulated draws at the start of each new
      // frame's state submit -- but ONLY while the title has never signalled a
      // real end of frame (0x80088133). A state submit happens several times a
      // frame, so ending the frame here cuts the title's own passes in half:
      // Minecraft's UI layer gets its clear in one of our frames and its
      // content in the next, so the composite that samples it sees an empty
      // target and the UI flickers. The display buffer to scan out is the one
      // the game last flipped via the HLE videoout (bufferIndex -> registered
      // address), not whichever RT drew last.
      if (!g_sawEndOfFrame.load(std::memory_order_relaxed)) {
        u64 scanout = prosperity_ps5_scanout_base();
        traceFlip("0x80488131", scanout);
        prosperity_agc_flip(scanout);
      }
      static int s_d131 = 0;
      if (kAgcTrace && s_d131 < 6) {
        s_d131++;
        auto *w = static_cast<const u32 *>(data);
        base::String line;
        base::FormatTo(line, "8131 arg(18 dwords):");
        for (int i = 0; i < 18; i++) base::FormatTo(line, " {:08x}", w[i]);
        BASE_LOGI("agc", "{}", line.c_str());
        // The ACQ ring is only ever sampled from the 0x8121 handler, which stops
        // firing once the driver has initialised. Sample it HERE, on the actual
        // per-frame submit, or "the ring is empty" only ever describes start-up.
        for (u64 r = g_acqRingLo; r && r < g_acqRingHi; r += 0x8000) {
          if (!gpuReadable(r, 0x8000)) continue;
          auto *rw = reinterpret_cast<const u32 *>(r);
          u32 nz = 0;
          for (u32 k = 0; k < 0x8000 / 4; k++)
            if (rw[k]) nz++;
          if (!nz) continue;
          base::String line;
          base::FormatTo(line, "  acqring {:#x}: {} non-zero:", (unsigned long)r, nz);
          for (int k = 0; k < 12; k++) base::FormatTo(line, " {:08x}", rw[k]);
          BASE_LOGI("agc", "{}", line.c_str());
        }
        // Raw dump of every IT_INDIRECT_BUFFER the arg points at. The decoded walk
        // can only show what it manages to parse; the question here is whether the
        // title's own command buffer is chained in at all.
        for (int i = 0; i + 3 < 18; i++) {
          if (w[i] != 0xC0023F00u) continue;
          u64 ib = (static_cast<u64>(w[i + 2] & 0xFF) << 32) | w[i + 1];
          u32 dw = w[i + 3] & 0xFFFFF;
          if (dw > 256) dw = 256;
          if (!gpuReadable(ib, dw * 4)) continue;
          auto *iw = reinterpret_cast<const u32 *>(ib);
          base::String line;
          base::FormatTo(line, "  IB {:#x} ({} dw):", (unsigned long)ib, dw);
          for (u32 k = 0; k < dw; k++) base::FormatTo(line, " {:08x}", iw[k]);
          BASE_LOGI("agc", "{}", line.c_str());
        }
      }
      prosperity_agc_submit(reinterpret_cast<u64>(data), (cmd >> 16) & 0x1fff);
      clearSubmitStatus(cmd, data, 0x40);
    }
    return 0;
  }
  case 0xC0188132:
  case 0x80108132: {  // AGC mode-1 secondary submit (IN, 16 bytes): arg = [_, count,
                      // ptrLo, ptrHi]; ptr -> array of `count` 16-byte descriptors
                      // [addrLo, addrHi, sizeDwords, flags]. THESE carry the real
                      // rendering PM4 (SET_*_REG, draws, RELEASE_MEM) -- the
                      // 0x80488131 stream is only per-frame register state. Forward
                      // each non-null command buffer to the command processor.
    if (data) {
      auto *w = static_cast<u32 *>(data);
      u32 count = w[1];
      u64 ptr = (static_cast<u64>(w[3]) << 32) | w[2];
      static int s_d132 = 0;
      if (kAgcTrace && s_d132 < 8) {
        s_d132++;
        BASE_LOGI("agc", "8132 arg=[{:08x} {:08x} {:08x} {:08x}] ptr={:#x} count={}",
                  w[0], w[1], w[2], w[3], (unsigned long)ptr, count);
        if (ptr && count && count < 4096) {
          auto *dd = reinterpret_cast<const u32 *>(ptr);
          for (u32 i = 0; i < count && i < 24; i++) {
            BASE_LOGI("agc", "  desc[{}] = {:08x} {:08x} {:08x} {:08x}", i,
                      dd[i * 4], dd[i * 4 + 1], dd[i * 4 + 2], dd[i * 4 + 3]);
            const u64 buf = (static_cast<u64>(dd[i * 4 + 1]) << 32) | dd[i * 4];
            const u32 sz = dd[i * 4 + 2];
            if (!sz || !gpuReadable(buf, 4)) continue;
            const u32 show = sz < 48 ? sz : 48;
            if (!gpuReadable(buf, show * 4)) continue;
            auto *bw = reinterpret_cast<const u32 *>(buf);
            base::String line;
            base::FormatTo(line, "    buf {:#x} ({} dw):", (unsigned long)buf, sz);
            for (u32 k = 0; k < show; k++) base::FormatTo(line, " {:08x}", bw[k]);
            BASE_LOGI("agc", "{}", line.c_str());
          }
        }
      }
      // Census: a whole submit used to be dropped when it carried >= 64
      // descriptors, and individual buffers are skipped when the address does
      // not look like GPU memory. Both are invisible without counting them.
      static std::atomic<u64> nSubmits{0}, nDropBatch{0}, nDesc{0},
          nFwd{0}, nSkipAddr{0};
      nSubmits.fetch_add(1, std::memory_order_relaxed);
      if (count >= 64) nDropBatch.fetch_add(1, std::memory_order_relaxed);
      if (kGcCensus) {
        static std::atomic<u64> last{0};
        u64 n = nSubmits.load();
        if (n - last.load() >= 2000) {
          last.store(n);
          BASE_LOGI("gccensus",
                    "submits={} batch-dropped(count>=64)={} desc={} forwarded={} "
                    "skipped-addr={}",
                    (unsigned long long)n,
                    (unsigned long long)nDropBatch.load(),
                    (unsigned long long)nDesc.load(),
                    (unsigned long long)nFwd.load(),
                    (unsigned long long)nSkipAddr.load());
        }
      }
      if (ptr && count && count < 64) {
        auto *d = reinterpret_cast<u32 *>(ptr);
        for (u32 i = 0; i < count; i++) {
          u64 buf = (static_cast<u64>(d[i * 4 + 1]) << 32) | d[i * 4];
          u32 sz = d[i * 4 + 2];
          nDesc.fetch_add(1, std::memory_order_relaxed);
          if (sz && gpuReadable(buf, sz * 4)) {
            nFwd.fetch_add(1, std::memory_order_relaxed);
            prosperity_agc_submit(buf, sz * 4);
          } else if (sz) {
            nSkipAddr.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
      clearSubmitStatus(cmd, data, 0x10);
    }
    return 0;
  }
  case 0xC0088133:
  case 0x80088133: {  // AGC mode-1 end-of-frame / flip signal, issued once per
                    // frame after the 0x8131 state + 0x8132 draw submits. The
                    // 8-byte arg carries no buffer field (observed all-zero);
                    // present the buffer the game flipped via the HLE videoout.
                    // Both directions occur, like 0x8131 and 0x8132: newer
                    // firmware presets a status word the driver expects cleared,
                    // and only the IN form was handled, so every frame this
                    // title ends went unpresented.
    g_sawEndOfFrame.store(true, std::memory_order_relaxed);
    u64 scanout = prosperity_ps5_scanout_base();
    traceFlip("0x8133", scanout);
    prosperity_agc_flip(scanout);
    clearSubmitStatus(cmd, data, 0);
    return 0;
  }

  // The init-time family, each issued once while libSceAgcDriver and
  // libSceGnmDriver come up. Named from the drivers' own error strings; the
  // soft-succeed default already answered them, but leaving them unnamed made
  // every boot log eight UNHANDLED lines that read like gaps.
  case 0xC00C8110:  // sceGnmSetGsRingSizes {esgsRingSize, gsvsRingSize, _}: our
                    // rings are implicit in the walker.
  case 0xC0848119:  // MIP-stats report setup/reset (132-byte command block).
  case 0x80888123:  // set trap-handler resources; failing it aborts AGC init
                    // with "Can't set trap handler resources".
  case 0x80048126:  // set submit mode (1 = the AGC mode-1 path this title uses).
  case 0x80048134:  // init flag cleared immediately before the 8126 above.
    return 0;

  case 0xC010810B: {  // Get CU Mask. The driver presets four dwords to
                      // 0xFFFFFFFF and reads back the low 16 bits of each as a
                      // per-shader-engine REDUNDANT-CU mask; a real console
                      // reports none, and a non-zero answer would skew its CU
                      // count. Failing it aborts with "Get CU Mask Fails".
    if (data)
      std::memset(data, 0, 0x10);
    return 0;
  }
  case 0xC010813B: {  // GPU info query at the tail of sce_agc_initialize; the
                      // reply is memcpy'd into a SceGnmGpuInfo block the driver
                      // first fills with 0xFF, so zeros read as "nothing to
                      // override".
    if (data)
      std::memset(data, 0, 0x10);
    return 0;
  }
  }

  // DELTA_AGC_TRACE: dump the mode-1 ioctl family (0x8131 submit, 0x8132/0x8133
  // flip/label, 0x8123 setup) with embedded GPU-pointer probes, to RE the ones we
  // still soft-ok.
  if (data) {
    u32 num = cmd & 0xff;
    static int agcDumps = 0;
    if (kAgcTrace && agcDumps < 24 &&
        (num == 0x31 || num == 0x32 || num == 0x33 || num == 0x23)) {
      agcDumps++;
      u32 len = (cmd >> 16) & 0x1fff;
      auto *w = static_cast<u32 *>(data);
      base::String line;
      base::FormatTo(line, "mode1 ioctl({:x}) len={}:", cmd, len);
      for (u32 k = 0; k * 4 < len && k < 24; k++)
        base::FormatTo(line, " {:08x}", w[k]);
      BASE_LOGI("agc", "{}", line.c_str());
      for (u32 k = 0; (k + 1) * 4 < len && k < 24; k++) {
        u64 p = (static_cast<u64>(w[k + 1]) << 32) | w[k];
        // Follow both GPU-aperture pointers AND host/stack pointers (the mode-1
        // flip/wait ioctls embed a stack ptr to a label/status struct).
        bool gpu = gpuAddr(p);
        bool stk = p >= 0x7ff000000000ull && p < 0x800000000000ull;
        if (gpu || stk) {
          auto *pw = reinterpret_cast<const u32 *>(p);
          base::String line;
          base::FormatTo(line, "  +{} ptr={:#x} ->", k * 4, (unsigned long)p);
          for (int j = 0; j < 8; j++) base::FormatTo(line, " {:08x}", pw[j]);
          // If the struct holds a further (GPU) pointer, deref that too (a label).
          if (stk) {
            for (int e = 0; e < 6; e++) {
              u64 cand = (static_cast<u64>(pw[e + 1]) << 32) | pw[e];
              if (gpuAddr(cand)) {
                auto *cw = reinterpret_cast<const u32 *>(cand);
                base::FormatTo(line, "\n    [+{}] buf {:#x} sz={:08x} ->", e * 4,
                               (unsigned long)cand, pw[e + 2]);
                for (int j = 0; j < 8; j++) base::FormatTo(line, " {:08x}", cw[j]);
              }
            }
          }
          BASE_LOGI("agc", "{}", line.c_str());
        }
      }
    }
  }

  // Unknown AGC ioctl: log (rate-limited) and soft-succeed, zeroing any OUT buffer
  // so the driver reads a benign result instead of stack garbage.
  static int unhandledLogged = 0;
  if (kGcTrace || unhandledLogged < 32) {
    unhandledLogged++;
    BASE_LOGI("gc", "UNHANDLED ioctl({:x}) data={:p}", cmd, data);
  }
  if (data && (cmd & 0x40000000u)) {
    u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

// The AGC driver mmaps /dev/gc to map its GPU ring/fifo buffers (ACQRB, DingDong,
// EopFifo, ...). Back these with the shared physical-dmem store at the requested
// offset (MAP_SHARED) so the bytes the CPU writes command packets into and the
// bytes the command processor reads at submit time are the same. Places the
// mapping in the low guest aperture the GPU pointers reference.
u8 *gcDevicePs5::map(void *addr, size_t len, u32 /*prot*/, u32 flags,
                          size_t offset) {
  int fd = dmemBackingFd();
  if (fd < 0 || len == 0 ||
      static_cast<u64>(offset) + len > dmemBackingSize())
    return reinterpret_cast<u8 *>(-1);
  u8 *va = static_cast<u8 *>(addr);
  const bool fixed = (flags & mFlags::fixed) != 0;
  void *p = MAP_FAILED;
  // Same hint handling as dmaDevicePs5::map: never let the host kernel choose the
  // address, or the guest gets a merely page-aligned ring buffer.
  if (va && !fixed) {
    p = ::mmap(va, len, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_FIXED_NOREPLACE, fd, static_cast<off_t>(offset));
    if (p != MAP_FAILED && p != va) {
      ::munmap(p, len);
      p = MAP_FAILED;
    }
  }
  if (p == MAP_FAILED) {
    u8 *base = (va && fixed) ? va : allocLowGuest(len);
    if (!base)
      return reinterpret_cast<u8 *>(-1);
    p = ::mmap(base, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
  }
  if (p == MAP_FAILED)
    return reinterpret_cast<u8 *>(-1);
  static const bool trace = kGcTrace ||
                            kDmemTrace;
  if (trace)
    BASE_LOGI("gc", "devmap off={:#x} len={:#x} -> {:p} (shared)", offset, len,
              p);
  return reinterpret_cast<u8 *>(p);
}
}  // namespace krnl
