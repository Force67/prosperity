/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 /dev/gc device: the libSceAgc / libSceAgcDriver AGC command protocol. This
 * is a dedicated PS5 device, split from the PS4 gcDevice (GNM PM4) so the two
 * unrelated ioctl command sets never share a switch. Forwards the AGC DCB to the
 * PS5 command processor (gpu/ps5).
 */

#include <base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>

#include "gc_dev.h"
#include "kern/ps4/dev/dma_dev.h"  // dmemBackingFd/Size (shared physical dmem store)
#include "kern/proc.h"
#include "kern/ps4/lv2/sys_mem.h"  // allocLowGuest, mFlags

// PS5 AGC submit bridge (delta_gpu, gpu/ps5): forward the DCB to the PS5 command
// processor, which follows INDIRECT_BUFFER chains and decodes the draws.
extern "C" void prosperity_agc_submit(uint64_t dcbBase, uint32_t sizeBytes);
// PS5 present bridge: end the frame and present the rendered RT to the window.
extern "C" void prosperity_agc_flip(uint64_t scanoutBase);
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
    if (addr && bytes && (hdr == 0xC0023F00u || hdr == 0xC0023300u))
      prosperity_agc_submit(addr, bytes);
  }
}

// A guest GPU address: see gpuAddr() in gpu/ps5/cmd_processor.cpp. A title that
// batch-maps its direct memory (Skyrim) gets buffers well below Isaac's
// 0x80_xx_xx_xx_xx AGC pool, and the old fixed band silently dropped every one of
// its command buffers -- so nothing ever rendered and the game waited forever on
// a GPU label the dropped submits would have written.
static inline bool gpuAddr(uint64_t a) {
  return a >= 0x1000000000ull && a < 0x8100000000ull;
}

int32_t gcDevicePs5::ioctl(uint32_t cmd, void *data) {
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
      static const bool trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int dumps = 0;
      if (trace && dumps < 4) {
        dumps++;
        auto *w = reinterpret_cast<uint32_t *>(a);
        std::printf("[agc] submit arg[0..15]:");
        for (int k = 0; k < 16; k++)
          std::printf(" %08x", w[k]);
        std::printf("\n[agc]   dcb0=%#lx dcb1=%#lx size=%u\n", (unsigned long)base,
                    (unsigned long)base2, size);
        for (int k = 0; k + 1 < 16; k++) {
          uint64_t p = (static_cast<uint64_t>(w[k + 1]) << 32) | w[k];
          if (gpuAddr(p)) {
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
      }
      // NOTE: the arg window reads empty for this title (mode-1 path is used); the
      // real per-frame PM4 (with the SET_SH_REG shader setup) lives in surrounding
      // ring windows listed by a descriptor table @0x80014981d8 -> 0x8002670000..
      // 0x8002698000. Forwarding that band naively CRASHES the walker (those buffers
      // chain via INDIRECT_BUFFER to sizes/addrs that need the descriptor's real
      // size, not a fixed 0x8000). NEXT: parse the descriptor table for each buffer's
      // exact addr+size and forward those. See ps5-boot-progress.
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
  case 0x80488131: {  // AGC submit.mode=1 submit (IN, 72 bytes). The arg IS a small
                      // command buffer: leading filler then IT_INDIRECT_BUFFER
                      // packets pointing at the real per-frame PM4. Forward it to the
                      // command processor, which follows the IBs and renders.
    if (data) {
      // Present the previous frame's accumulated draws at the start of each new
      // frame's state submit, then submit this frame's register state. The
      // display buffer to scan out is the one the game last flipped via the HLE
      // videoout (bufferIndex -> registered address), not whichever RT drew last.
      uint64_t scanout = prosperity_ps5_scanout_base();
      traceFlip("0x80488131", scanout);
      prosperity_agc_flip(scanout);
      static const bool agcTrace1 = std::getenv("DELTA_AGC_TRACE") != nullptr;
      static int s_d131 = 0;
      if (agcTrace1 && s_d131 < 6) {
        s_d131++;
        auto *w = static_cast<const uint32_t *>(data);
        std::printf("[agc] 8131 arg(18 dwords):");
        for (int i = 0; i < 18; i++) std::printf(" %08x", w[i]);
        std::printf("\n");
      }
      prosperity_agc_submit(reinterpret_cast<uint64_t>(data), (cmd >> 16) & 0x1fff);
    }
    return 0;
  }
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
      if (ptr && count && count < 64) {
        auto *d = reinterpret_cast<uint32_t *>(ptr);
        for (uint32_t i = 0; i < count; i++) {
          uint64_t buf = (static_cast<uint64_t>(d[i * 4 + 1]) << 32) | d[i * 4];
          uint32_t sz = d[i * 4 + 2];
          if (gpuAddr(buf) && sz)
            prosperity_agc_submit(buf, sz * 4);
        }
      }
    }
    return 0;
  }
  case 0x80088133: {  // AGC mode-1 end-of-frame / flip signal (IN, 8 bytes), issued
                    // once per frame after the 0x8131 state + 0x8132 draw submits.
                    // The 8-byte arg carries no buffer field (observed all-zero);
                    // present the buffer the game flipped via the HLE videoout.
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
  bool fixed = (flags & mFlags::fixed) != 0;
  if (!va) {
    va = allocLowGuest(len);
    if (!va)
      return reinterpret_cast<uint8_t *>(-1);
    fixed = true;
  }
  int mflags = MAP_SHARED | (fixed ? MAP_FIXED : 0);
  void *p = ::mmap(va, len, PROT_READ | PROT_WRITE, mflags, fd,
                   static_cast<off_t>(offset));
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
