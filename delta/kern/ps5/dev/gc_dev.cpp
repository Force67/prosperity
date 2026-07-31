/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/gc device: the libSceAgc / libSceAgcDriver AGC command protocol. This
 * is a dedicated PS5 device, split from the PS4 gcDevice (GNM PM4) so the two
 * unrelated ioctl command sets never share a switch. Forwards the AGC DCB to the
 * PS5 command processor (gpu/ps5).
 */

#include <base.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>

#include <sys/mman.h>

#include <utl/mem.h>

#include "gc_dev.h"
#include "kern/ps4/dev/dma_dev.h"  // dmemBackingFd/Size (shared physical dmem store)
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"  // allocLowGuest, mFlags

// PS5 AGC submit bridge (delta_gpu, gpu/ps5): forward the DCB to the PS5 command
// processor, which follows INDIRECT_BUFFER chains and decodes the draws.
extern "C" void prosperity_agc_submit(uint64_t dcbBase, uint32_t sizeBytes);
// PS5 present bridge: end the frame and present the rendered RT to the window.
extern "C" void prosperity_agc_flip(uint64_t scanoutBase);

// Set once the title issues the mode-1 end-of-frame ioctl. Until then the frame
// has to be ended somewhere, and the start of a state submit is the only other
// boundary available.
static std::atomic<bool> g_sawEndOfFrame{false};
// Guest address of the display buffer the game most recently flipped, resolved
// from sceVideoOutSubmitFlip*'s bufferIndex via the registered-buffer table
// (libSceVideoOut_ps5.cpp). The AGC flip ioctls below carry no buffer field, so
// they present this instead of falling back to whichever RT was drawn last.
extern "C" uint64_t prosperity_ps5_scanout_base();

// DELTA_FLIP_TRACE: log the scanout base each AGC flip presents, so the derived
// per-flip buffer can be checked against the registered display buffers.
static void traceFlip(const char *site, uint64_t base) {
  static const bool on = std::getenv("DELTA_FLIP_TRACE") != nullptr;
  if (on)
    std::printf("[flip] %s present=%#lx\n", site, (unsigned long)base);
}

namespace krnl {
gcDevicePs5::gcDevicePs5(proc *p) : device(p) {}

bool gcDevicePs5::init(const char *, uint32_t, uint32_t) { return true; }

// Diagnostic (DELTA_AGC_TRACE): scan one GPU-aperture page for a draw-DCB PM4
// header to locate the command buffer if a submit-arg pointer reads zero.
static void scanPagePm4(void *ctx, uint8_t *p, size_t sz) {
  int *hits = static_cast<int *>(ctx);
  auto isDcb = [](uint32_t h) {
    if ((h >> 30) != 3) return false;
    uint32_t op = (h >> 8) & 0xFF;
    return op == 0x69 || op == 0x76 || op == 0x79 || op == 0x3F || op == 0x2D ||
           op == 0x27 || op == 0x35 || op == 0x15;
  };
  auto *w = reinterpret_cast<const uint32_t *>(p);
  uint64_t n = sz / 4;
  if (n > 0x400000) n = 0x400000;
  int perPage = 0;
  for (uint64_t j = 0; j < n && *hits < 24 && perPage < 4; j++) {
    if (isDcb(w[j])) {
      std::printf("[agc]   DCB@%#lx hdr=%08x op=%#x\n",
                  (unsigned long)(reinterpret_cast<uint64_t>(p) + j * 4), w[j],
                  (w[j] >> 8) & 0xFF);
      (*hits)++;
      perPage++;
    }
  }
}

// A guest GPU address: see GpuAddr() in gpu/ps5/cmd_processor.cc. The band must
// span everything allocLowGuest() can hand out (64 GiB slot up to the 2^40 user
// ceiling); a fixed band around one title's pool silently drops every command
// buffer another title allocates outside it, so nothing renders and the game
// waits forever on a GPU label the dropped submits would have written.
static inline bool gpuAddr(uint64_t a) {
  return a >= 0x1000000000ull && a < 0x10000000000ull;
}

// The submit paths and trace probes below deref candidate pointers pulled out of
// a submit arg. Plenty of those words look like GPU addresses without being
// mapped, so the range check alone is not enough to read through one.
static inline bool gpuReadable(uint64_t a, size_t n) {
  return gpuAddr(a) && utl::isMemoryRangeMapped(reinterpret_cast<void *>(a), n);
}

// Span of ACQ ring windows the driver named in its 0xC0408121 submits, learned at
// run time rather than hardcoded (each title's ring sits wherever its driver
// mapped it). Lets the per-frame trace re-read the ring long after the 0x8121
// ioctls have stopped.
static uint64_t g_acqRingLo = 0, g_acqRingHi = 0;

// The mode-1 submit ioctls are INOUT on firmware 13.60: libSceAgcDriver presets a
// status dword in the arg and, on return, treats the submit as FAILED unless the
// kernel has cleared it. A failed state submit makes the driver skip the 0x8132
// call that carries the title's own command buffer, so the frame is dropped
// entirely -- no draws reach us, and the completion label the title spins on is
// never written. The IN-only variants older firmware issues have no such field.
static void clearSubmitStatus(uint32_t cmd, void *data, uint32_t offset) {
  if (!data || !(cmd & 0x40000000u))
    return;
  const uint32_t len = (cmd >> 16) & 0x1fff;
  if (offset + sizeof(uint32_t) > len)
    return;
  std::memset(static_cast<uint8_t *>(data) + offset, 0, sizeof(uint32_t));
}

// GNM-style submit descriptor array: each 4-dword entry is an IT_INDIRECT_BUFFER
// (0xC0023F00 = dcb) / _CNST (0xC0023300 = ccb) with [hdr, addrLo, addrHi&0xFF,
// sizeDwords]. libSceAgcDriver/GnmDriver submits the pipeline+shader setup and
// draws through these on PS5 too; forward each buffer to the PS5 command processor
// (they carry the SET_SH_REG shader binding the AGC mode-1 path never emits).
static void submitGnmDescArray(uint64_t descPtr, uint32_t count) {
  const uint32_t *d = reinterpret_cast<const uint32_t *>(descPtr);
  if (!d || count > 0x1000) return;
  for (uint32_t i = 0; i < count; i++) {
    const uint32_t *e = d + i * 4;
    uint32_t hdr = e[0];
    uint64_t addr = (static_cast<uint64_t>(e[2] & 0xFF) << 32) | e[1];
    uint32_t bytes = (e[3] & 0xFFFFF) * 4;
    if (bytes && (hdr == 0xC0023F00u || hdr == 0xC0023300u) &&
        gpuReadable(addr, bytes))
      prosperity_agc_submit(addr, bytes);
  }
}

int32_t gcDevicePs5::ioctl(uint32_t cmd, void *data) {
  if (std::getenv("DELTA_GC_IOCTL_CENSUS")) {
    static std::mutex mtx;
    static std::map<uint32_t, uint64_t> hist;
    static auto last = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mtx);
    hist[cmd]++;
    auto now = std::chrono::steady_clock::now();
    if (now - last > std::chrono::seconds(10)) {
      last = now;
      std::printf("[gcioctl] --- 10s census ---\n");
      for (auto &[c, n] : hist)
        std::printf("[gcioctl] %#x %llu\n", c, (unsigned long long)n);
    }
  }
  if (std::getenv("DELTA_GC_TRACE"))
    std::printf("[gc] ioctl(%x) data=%p\n", cmd, data);
  switch (cmd) {
  case 0xC0108102: {  // GNM submit: {u32 a0, u32 count, u64 descPtr}
    struct argl { uint32_t a0; uint32_t count; uint64_t descPtr; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    return 0;
  }
  case 0xC018810A: {  // GNM submit variant: {a0, count, a2, pad, descPtr}
    struct argl { uint32_t a0, count, a2, pad; uint64_t descPtr; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    return 0;
  }
  case 0xC020810C: {  // GNM submit-and-flip: {a0, count, descPtr, flipPtr, flag}
    struct argl { uint32_t a0, count; uint64_t descPtr, flipPtr; uint32_t flag; };
    auto *a = static_cast<argl *>(data);
    if (a) submitGnmDescArray(a->descPtr, a->count);
    uint64_t scanout = prosperity_ps5_scanout_base();
    traceFlip("0xC020810C", scanout);
    prosperity_agc_flip(scanout);  // present the flipped display buffer
    return 0;
  }
  case 0x40048135:  // AGC query: OUT dword (submit/queue id). The driver stores
                    // it; 0 is accepted.
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  case 0xC004812E:  // AGC init: INOUT dword. 0 tells AgcDriver to map its own
                    // submit doorbell (which then succeeds).
    if (data)
      *static_cast<uint32_t *>(data) = 0;
    return 0;
  case 0xC0408121: {  // AGC submit (INOUT, 64 bytes). The DCB ring window is at
                      // arg+0x10 (= ACQRB + submitIdx*0x8000, 0x4000 bytes) with a
                      // secondary at arg+0x18 (+0x4000). Forward it, then zero the
                      // arg per the OUT convention. (This title uses the mode-1
                      // 0x80488131 path instead; this ring reads empty for it.)
    if (data) {
      auto *a = static_cast<uint8_t *>(data);
      uint64_t base = 0, base2 = 0;
      std::memcpy(&base, a + 0x10, 8);
      std::memcpy(&base2, a + 0x18, 8);
      uint32_t size = 0x8000;
      if (gpuAddr(base)) {
        if (!g_acqRingLo || base < g_acqRingLo) g_acqRingLo = base;
        if (base + size > g_acqRingHi) g_acqRingHi = base + size;
      }
      static const bool trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int dumps = 0;
      static uint64_t calls = 0;
      ++calls;
      // The first few submits are engine init, where the ring legitimately holds
      // nothing. Sample later ones too, or "the ring reads empty" only ever
      // describes start-up.
      if (trace && (dumps < 8 || (calls % 200 == 0 && dumps < 12))) {
        dumps++;
        std::printf("[agc] --- submit #%llu ---\n", (unsigned long long)calls);
        auto *w = reinterpret_cast<uint32_t *>(a);
        std::printf("[agc] submit arg[0..15]:");
        for (int k = 0; k < 16; k++)
          std::printf(" %08x", w[k]);
        std::printf("\n[agc]   dcb0=%#lx dcb1=%#lx size=%u\n", (unsigned long)base,
                    (unsigned long)base2, size);
        for (int k = 0; k + 1 < 16; k++) {
          uint64_t p = (static_cast<uint64_t>(w[k + 1]) << 32) | w[k];
          if (gpuReadable(p, 32)) {
            auto *pw = reinterpret_cast<const uint32_t *>(p);
            std::printf("[agc]   arg[%d] ptr=%#lx ->", k, (unsigned long)p);
            for (int j = 0; j < 8; j++) std::printf(" %08x", pw[j]);
            std::printf("\n");
          }
        }
        if (auto *pr = proc::getActive()) {
          int hits = 0;
          pr->getVma().forEachGpuAperturePage(scanPagePm4, &hits);
          if (!hits)
            std::printf("[agc]   no PM4 anywhere in the GPU aperture (empty ring?)\n");
        }
        // Distinguish "the title wrote nothing" from "we are reading a mapping
        // that does not see its writes": count non-zero dwords in the ring
        // window, regardless of whether they look like PM4.
        if (gpuReadable(base, 0x8000)) {
          auto *rw = reinterpret_cast<const uint32_t *>(base);
          uint32_t nz = 0, first = 0;
          for (uint32_t k = 0; k < 0x8000 / 4; k++)
            if (rw[k]) { if (!nz) first = k; nz++; }
          std::printf("[agc]   ring %#lx: %u/%u dwords non-zero (first @dw %u)\n",
                      (unsigned long)base, nz, 0x8000u / 4, first);
        }
      }
      // The window a submit names is empty AT IOCTL TIME if the driver fills it
      // afterwards and kicks the GPU through the doorbell instead. Re-read the
      // PREVIOUS submit's window here: if it has packets now, the ioctl is a
      // ring-window acquire and the submit boundary is the doorbell, not this.
      static uint64_t prevBase = 0;
      if (trace && prevBase && prevBase != base && gpuReadable(prevBase, 0x8000)) {
        auto *pw = reinterpret_cast<const uint32_t *>(prevBase);
        uint32_t nz = 0;
        for (uint32_t k = 0; k < 0x8000 / 4; k++)
          if (pw[k]) nz++;
        if (nz && dumps <= 12) {
          std::printf("[agc]   prev ring %#lx now %u dwords non-zero:",
                      (unsigned long)prevBase, nz);
          for (int k = 0; k < 12; k++) std::printf(" %08x", pw[k]);
          std::printf("\n");
        }
      }
      prevBase = base;
      // DELTA_AGC_RINGDUMP: the submit arg in full plus the ring descriptor table
      // it references, with a PM4 sniff of each buffer. The per-pass register
      // state Skyrim never seems to program (a colour target for some passes, PS
      // user data above 15) has to come from one of these.
      static const bool ringDump = std::getenv("DELTA_AGC_RINGDUMP") != nullptr;
      static int ringN = 0;
      if (ringDump && ringN < 3) {
        ringN++;
        auto *w = reinterpret_cast<const uint32_t *>(a);
        std::printf("[ring] arg:");
        for (int k = 0; k < 16; k++) std::printf(" %08x", w[k]);
        std::printf("\n");
        auto sniff = [](uint64_t p, const char *what) {
          if (!gpuReadable(p, 1024)) return;
          const auto *q = reinterpret_cast<const uint32_t *>(p);
          uint32_t t3 = 0;
          for (int i = 0; i < 256; i++)
            if ((q[i] >> 30) == 3) t3++;
          std::printf("[ring]   %s %#lx: %08x %08x %08x %08x  (type3 hdrs in 256 dw: %u)\n",
                      what, (unsigned long)p, q[0], q[1], q[2], q[3], t3);
        };
        for (int k = 0; k + 1 < 16; k++)
          sniff((static_cast<uint64_t>(w[k + 1]) << 32) | w[k], "argptr");
        const uint64_t tbl = 0x80014981d8ull;
        if (gpuAddr(tbl)) {
          const auto *t = reinterpret_cast<const uint32_t *>(tbl);
          std::printf("[ring] table @%#lx:", (unsigned long)tbl);
          for (int k = 0; k < 16; k++) std::printf(" %08x", t[k]);
          std::printf("\n");
          for (int k = 0; k + 1 < 16; k += 2)
            sniff((static_cast<uint64_t>(t[k + 1] & 0xFFFF) << 32) | t[k], "tblptr");
        }
      }
      // NOTE: the arg window reads empty for this title (mode-1 path is used); the
      // real per-frame PM4 (with the SET_SH_REG shader setup) lives in surrounding
      // ring windows listed by a descriptor table @0x80014981d8 -> 0x8002670000..
      // 0x8002698000. Forwarding that band naively CRASHES the walker (those buffers
      // chain via INDIRECT_BUFFER to sizes/addrs that need the descriptor's real
      // size, not a fixed 0x8000). NEXT: parse the descriptor table for each buffer's
      // exact addr+size and forward those. See ps5-boot-progress.
      if (gpuReadable(base, size))
        prosperity_agc_submit(base, size);
      // NOTE: forwarding the adjacent ring window (base-0x10000, which has real PM4)
      // still faults -- the window isn't fully mapped and the walker reads unmapped
      // bytes within it. NEXT: get each buffer's EXACT size from the descriptor table
      // (@0x80014981d8) and bounds-check the whole walk against forEachGpuAperturePage.
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
      static uint32_t s_agcDone = 0;
      *static_cast<uint32_t *>(data) = ++s_agcDone;
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
        uint64_t scanout = prosperity_ps5_scanout_base();
        traceFlip("0x80488131", scanout);
        prosperity_agc_flip(scanout);
      }
      static const bool agcTrace1 = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int s_d131 = 0;
      if (agcTrace1 && s_d131 < 6) {
        s_d131++;
        auto *w = static_cast<const uint32_t *>(data);
        std::printf("[agc] 8131 arg(18 dwords):");
        for (int i = 0; i < 18; i++) std::printf(" %08x", w[i]);
        std::printf("\n");
        // The ACQ ring is only ever sampled from the 0x8121 handler, which stops
        // firing once the driver has initialised. Sample it HERE, on the actual
        // per-frame submit, or "the ring is empty" only ever describes start-up.
        for (uint64_t r = g_acqRingLo; r && r < g_acqRingHi; r += 0x8000) {
          if (!gpuReadable(r, 0x8000)) continue;
          auto *rw = reinterpret_cast<const uint32_t *>(r);
          uint32_t nz = 0;
          for (uint32_t k = 0; k < 0x8000 / 4; k++)
            if (rw[k]) nz++;
          if (!nz) continue;
          std::printf("[agc]   acqring %#lx: %u non-zero:", (unsigned long)r, nz);
          for (int k = 0; k < 12; k++) std::printf(" %08x", rw[k]);
          std::printf("\n");
        }
        // Raw dump of every IT_INDIRECT_BUFFER the arg points at. The decoded walk
        // can only show what it manages to parse; the question here is whether the
        // title's own command buffer is chained in at all.
        for (int i = 0; i + 3 < 18; i++) {
          if (w[i] != 0xC0023F00u) continue;
          uint64_t ib = (static_cast<uint64_t>(w[i + 2] & 0xFF) << 32) | w[i + 1];
          uint32_t dw = w[i + 3] & 0xFFFFF;
          if (dw > 256) dw = 256;
          if (!gpuReadable(ib, dw * 4)) continue;
          auto *iw = reinterpret_cast<const uint32_t *>(ib);
          std::printf("[agc]   IB %#lx (%u dw):", (unsigned long)ib, dw);
          for (uint32_t k = 0; k < dw; k++) std::printf(" %08x", iw[k]);
          std::printf("\n");
        }
      }
      prosperity_agc_submit(reinterpret_cast<uint64_t>(data), (cmd >> 16) & 0x1fff);
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
      auto *w = static_cast<uint32_t *>(data);
      uint32_t count = w[1];
      uint64_t ptr = (static_cast<uint64_t>(w[3]) << 32) | w[2];
      static const bool agcTrace2 = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int s_d132 = 0;
      if (agcTrace2 && s_d132 < 8) {
        s_d132++;
        std::printf("[agc] 8132 arg=[%08x %08x %08x %08x] ptr=%#lx count=%u\n",
                    w[0], w[1], w[2], w[3], (unsigned long)ptr, count);
        if (ptr && count && count < 4096) {
          auto *dd = reinterpret_cast<const uint32_t *>(ptr);
          for (uint32_t i = 0; i < count && i < 24; i++)
            std::printf("[agc]   desc[%u] = %08x %08x %08x %08x\n", i,
                        dd[i * 4], dd[i * 4 + 1], dd[i * 4 + 2], dd[i * 4 + 3]);
        }
      }
      // Census: a whole submit used to be dropped when it carried >= 64
      // descriptors, and individual buffers are skipped when the address does
      // not look like GPU memory. Both are invisible without counting them.
      static std::atomic<uint64_t> nSubmits{0}, nDropBatch{0}, nDesc{0},
          nFwd{0}, nSkipAddr{0};
      nSubmits.fetch_add(1, std::memory_order_relaxed);
      if (count >= 64) nDropBatch.fetch_add(1, std::memory_order_relaxed);
      if (std::getenv("DELTA_GC_CENSUS")) {
        static std::atomic<uint64_t> last{0};
        uint64_t n = nSubmits.load();
        if (n - last.load() >= 2000) {
          last.store(n);
          std::printf("[gccensus] submits=%llu batch-dropped(count>=64)=%llu "
                      "desc=%llu forwarded=%llu skipped-addr=%llu\n",
                      (unsigned long long)n,
                      (unsigned long long)nDropBatch.load(),
                      (unsigned long long)nDesc.load(),
                      (unsigned long long)nFwd.load(),
                      (unsigned long long)nSkipAddr.load());
        }
      }
      if (ptr && count && count < 64) {
        auto *d = reinterpret_cast<uint32_t *>(ptr);
        for (uint32_t i = 0; i < count; i++) {
          uint64_t buf = (static_cast<uint64_t>(d[i * 4 + 1]) << 32) | d[i * 4];
          uint32_t sz = d[i * 4 + 2];
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
  case 0x80088133: {  // AGC mode-1 end-of-frame / flip signal (IN, 8 bytes), issued
                    // once per frame after the 0x8131 state + 0x8132 draw submits.
                    // The 8-byte arg carries no buffer field (observed all-zero);
                    // present the buffer the game flipped via the HLE videoout.
    g_sawEndOfFrame.store(true, std::memory_order_relaxed);
    uint64_t scanout = prosperity_ps5_scanout_base();
    traceFlip("0x80088133", scanout);
    prosperity_agc_flip(scanout);
    return 0;
  }
  }

  // DELTA_AGC_TRACE: dump the mode-1 ioctl family (0x8131 submit, 0x8132/0x8133
  // flip/label, 0x8123 setup) with embedded GPU-pointer probes, to RE the ones we
  // still soft-ok.
  if (data) {
    uint32_t num = cmd & 0xff;
    static const bool agcTrace = std::getenv("DELTA_AGC_TRACE") != nullptr;
    static int agcDumps = 0;
    if (agcTrace && agcDumps < 24 &&
        (num == 0x31 || num == 0x32 || num == 0x33 || num == 0x23)) {
      agcDumps++;
      uint32_t len = (cmd >> 16) & 0x1fff;
      auto *w = static_cast<uint32_t *>(data);
      std::printf("[agc] mode1 ioctl(%x) len=%u:", cmd, len);
      for (uint32_t k = 0; k * 4 < len && k < 24; k++)
        std::printf(" %08x", w[k]);
      std::printf("\n");
      for (uint32_t k = 0; (k + 1) * 4 < len && k < 24; k++) {
        uint64_t p = (static_cast<uint64_t>(w[k + 1]) << 32) | w[k];
        // Follow both GPU-aperture pointers AND host/stack pointers (the mode-1
        // flip/wait ioctls embed a stack ptr to a label/status struct).
        bool gpu = gpuAddr(p);
        bool stk = p >= 0x7ff000000000ull && p < 0x800000000000ull;
        if (gpu || stk) {
          auto *pw = reinterpret_cast<const uint32_t *>(p);
          std::printf("[agc]   +%u ptr=%#lx ->", k * 4, (unsigned long)p);
          for (int j = 0; j < 8; j++) std::printf(" %08x", pw[j]);
          // If the struct holds a further (GPU) pointer, deref that too (a label).
          if (stk) {
            for (int e = 0; e < 6; e++) {
              uint64_t cand = (static_cast<uint64_t>(pw[e + 1]) << 32) | pw[e];
              if (gpuAddr(cand)) {
                auto *cw = reinterpret_cast<const uint32_t *>(cand);
                std::printf("\n[agc]     [+%d] buf %#lx sz=%08x ->", e * 4,
                            (unsigned long)cand, pw[e + 2]);
                for (int j = 0; j < 8; j++) std::printf(" %08x", cw[j]);
              }
            }
          }
          std::printf("\n");
        }
      }
    }
  }

  // Unknown AGC ioctl: log (rate-limited) and soft-succeed, zeroing any OUT buffer
  // so the driver reads a benign result instead of stack garbage.
  static const bool gcTrace = std::getenv("DELTA_GC_TRACE") != nullptr;
  static int unhandledLogged = 0;
  if (gcTrace || unhandledLogged < 32) {
    unhandledLogged++;
    std::printf("[gc] UNHANDLED ioctl(%x) data=%p\n", cmd, data);
  }
  if (data && (cmd & 0x40000000u)) {
    uint32_t len = (cmd >> 16) & 0x1fff;
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
uint8_t *gcDevicePs5::map(void *addr, size_t len, uint32_t /*prot*/, uint32_t flags,
                          size_t offset) {
  int fd = dmemBackingFd();
  if (fd < 0 || len == 0 ||
      static_cast<uint64_t>(offset) + len > dmemBackingSize())
    return reinterpret_cast<uint8_t *>(-1);
  uint8_t *va = static_cast<uint8_t *>(addr);
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
    uint8_t *base = (va && fixed) ? va : allocLowGuest(len);
    if (!base)
      return reinterpret_cast<uint8_t *>(-1);
    p = ::mmap(base, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
               static_cast<off_t>(offset));
  }
  if (p == MAP_FAILED)
    return reinterpret_cast<uint8_t *>(-1);
  static const bool trace = std::getenv("DELTA_GC_TRACE") != nullptr ||
                            std::getenv("DELTA_DMEM_TRACE") != nullptr;
  if (trace)
    std::fprintf(stderr, "[gc] devmap off=%#zx len=%#zx -> %p (shared)\n", offset,
                 len, p);
  return reinterpret_cast<uint8_t *>(p);
}
}  // namespace krnl
