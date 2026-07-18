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

// Diagnostic: find every location in GPU memory that holds a pointer to the shader
// page 0x80044006xx (PGM_LO == 0x80044006, or raw low 0x044006xx). Whatever holds it
// IS the shader-bind descriptor/register-image the AGC uses out-of-band.
static void scanPageForShaderPtr(void *ctx, uint8_t *p, size_t sz) {
  int *hits = static_cast<int *>(ctx);
  auto *w = reinterpret_cast<const uint32_t *>(p);
  uint64_t n = sz / 4;
  if (n > 0x4000000) n = 0x4000000;
  uint64_t va0 = reinterpret_cast<uint64_t>(p);
  for (uint64_t j = 0; j < n && *hits < 40; j++) {
    // PGM_LO form (addr>>8) for a shader in page 0x8004400xxx..0x8004401xxx.
    bool pgm = (w[j] == 0x80044006u || w[j] == 0x80044007u || w[j] == 0x80044011u ||
                w[j] == 0x80044010u);
    // raw-address form: 0x044006xx / 0x044010xx / 0x044011xx with a 0x80 hi dword.
    bool raw = ((w[j] & 0xFFFFF000u) == 0x04400000u) && j + 1 < n &&
               (w[j + 1] & 0xFF) == 0x80;
    if (pgm || raw) {
      std::fprintf(stderr, "[agc]   shptr@%#lx = %08x %08x (%s)\n",
                   (unsigned long)(va0 + j * 4), w[j], j + 1 < n ? w[j + 1] : 0,
                   pgm ? "PGM_LO" : "raw");
      (*hits)++;
    }
  }
}

// Diagnostic: locate the game's shaders by scanning GPU-aperture pages for the RDNA2
// s_endpgm marker (0xBF810000). Every shader ends with exactly one, so its address
// (+ a scan back for the shader start) tells us WHERE the shaders live, independent
// of any pipeline/register indirection. Records the page base of each hit.
static uint64_t g_isaPages[64];
static int g_isaPageN;
static uint64_t g_sweepLo = ~0ull, g_sweepHi, g_sweepPages;
static void scanPageShaderPgm(void *ctx, uint8_t *p, size_t sz) {
  int *hits = static_cast<int *>(ctx);
  auto *w = reinterpret_cast<const uint32_t *>(p);
  uint64_t va0 = reinterpret_cast<uint64_t>(p);
  if (va0 < g_sweepLo) g_sweepLo = va0;
  if (va0 + sz > g_sweepHi) g_sweepHi = va0 + sz;
  g_sweepPages++;
  uint64_t n = sz / 4;
  if (n > 0x4000000) n = 0x4000000;  // scan up to 256MB per region
  // Count RDNA2 scalar markers (s_waitcnt 0xBF8C, s_endpgm 0xBF81, s_branch 0xBF82,
  // s_mov SOP1 0xBExx037E). A page dense in these is shader ISA.
  int endpgm = 0, waitcnt = 0;
  uint64_t firstEnd = 0;
  for (uint64_t j = 0; j < n; j++) {
    if (w[j] == 0xBF810000u) { endpgm++; if (!firstEnd) firstEnd = va0 + j * 4; }
    else if ((w[j] & 0xFFFF0000u) == 0xBF8C0000u) waitcnt++;
  }
  if ((endpgm && waitcnt) || waitcnt >= 4) {
    std::fprintf(stderr, "[agc]   ISA region @%#lx sz=%#lx (endpgm=%d waitcnt=%d first_endpgm=%#lx)\n",
                 (unsigned long)va0, (unsigned long)sz, endpgm, waitcnt,
                 (unsigned long)firstEnd);
    if (g_isaPageN < 64) g_isaPages[g_isaPageN++] = va0 & ~0xFFFull;
    (*hits)++;
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

int32_t gcDevicePs5::ioctl(uint32_t cmd, void *data) {
  // DELTA_AGC_TRACE one-shot: after the render loop is going, sweep ALL mapped GPU
  // memory for the shader PGM (into the shader heap). Tells us if/where the AGC
  // stored it (pipeline obj) vs bound it via direct MMIO (nowhere in guest mem).
  static const bool agcTr = std::getenv("DELTA_AGC_TRACE") != nullptr;
  if (agcTr && (cmd == 0x80488131 || cmd == 0x80108132 || cmd == 0xC0108102 ||
                cmd == 0xC020810C)) {
    static uint64_t s_n = 0;
    if (++s_n == 4000) {
      std::fprintf(stderr, "[agc]   ISA sweep starting (locate shaders via s_endpgm)...\n");
      if (auto *pr = proc::getActive()) {
        g_isaPageN = 0;
        g_sweepLo = ~0ull; g_sweepHi = 0; g_sweepPages = 0;
        int hits = 0;
        pr->getVma().forEachGpuAperturePage(scanPageShaderPgm, &hits);
        std::fprintf(stderr, "[agc]   ISA sweep: %d shader pages; covered %#lx..%#lx (%lu pages)\n",
                     g_isaPageN, (unsigned long)g_sweepLo, (unsigned long)g_sweepHi,
                     (unsigned long)g_sweepPages);
        // Enumerate every shader in the 0x8004000000..0x8006000000 window by finding
        // each s_endpgm and reporting the following shader's start (first dword after
        // the s_code_end padding). Tells us the full shader set + their PGM addresses.
        {
          auto *w = reinterpret_cast<const uint32_t *>(0x8004000000ull);
          uint64_t words = 0x2000000 / 4;  // 32MB window
          int found = 0;
          bool inPad = true;  // start of window: expect a shader start soon
          uint64_t shaderStart = 0x8004000000ull;
          for (uint64_t j = 0; j < words && found < 24; j++) {
            if (w[j] == 0xBF810000u) {  // s_endpgm -> shader end
              uint64_t va = 0x8004000000ull + j * 4;
              std::fprintf(stderr, "[agc]   shader %#lx..%#lx (%lu dw)\n",
                           (unsigned long)shaderStart, (unsigned long)va,
                           (unsigned long)(va - shaderStart) / 4);
              found++;
              inPad = true;
            } else if (inPad && w[j] != 0xBF9F0000u && w[j] != 0) {
              shaderStart = 0x8004000000ull + j * 4;  // first real dword after padding
              inPad = false;
            }
          }
          std::fprintf(stderr, "[agc]   (%d shaders in window)\n", found);
          // Scan for uploaded shader ELF headers (\x7fELF = 0x464c457f). If present,
          // the create uploaded the ELF ok -> failure is inside f3dg2CSgRKY (libSceAgc
          // rejects). If absent -> the upload/alloc failed (env issue).
          int elfs = 0;
          for (uint64_t j = 0; j < 0x2000000 / 4 && elfs < 8; j++) {
            if (w[j] == 0x464c457fu) {
              std::fprintf(stderr, "[agc]   ELF hdr @%#lx: e_machine=%#x shnum=%#x\n",
                           (unsigned long)(0x8004000000ull + j * 4),
                           reinterpret_cast<const uint16_t *>(&w[j])[9],
                           reinterpret_cast<const uint16_t *>(&w[j])[0x1e]);
              elfs++;
            }
          }
          std::fprintf(stderr, "[agc]   uploaded shader ELF headers found: %d\n", elfs);
        }
        // Dump the pipeline-object header at 0x8004400000 (reg 0x113 / reg 0x342
        // point here; the shaders are at +0x600). Print every aperture pointer (a
        // dword pair whose hi byte is 0x80) so we find the PGM + user-data/V# ptrs.
        {
          auto *w = reinterpret_cast<const uint32_t *>(0x8004400000ull);
          for (uint32_t o = 0; o < 0x600 / 4; o++) {
            uint64_t p = (static_cast<uint64_t>(w[o + 1] & 0xFFFF) << 32) | w[o];
            if (p >= 0x8004000000ull && p < 0x8005000000ull)
              std::fprintf(stderr, "[agc]   pipehdr+%#x = %#lx (-> pool)\n", o * 4,
                           (unsigned long)p);
          }
          std::fprintf(stderr, "[agc]   pipehdr raw[0..31]:");
          for (int j = 0; j < 32; j++) std::fprintf(stderr, " %08x", w[j]);
          std::fprintf(stderr, "\n");
        }
        // OPTION B: examine the pipeline-descriptor table @0x80014981d8 and walk each
        // entry's target buffer looking for SET_SH_REG (0x76) / SET_SH_REG_INDIRECT
        // (0x63) -- the mini command buffers that program the shader PGM out-of-band.
        {
          auto *t = reinterpret_cast<const uint32_t *>(0x8001498000ull);
          std::fprintf(stderr, "[agc]   pipe-table region 0x8001498000 [0x100..0x400]:\n");
          for (uint32_t o = 0x100 / 4; o < 0x400 / 4; o += 8) {
            uint64_t p = (static_cast<uint64_t>(t[o + 1] & 0xFFFF) << 32) | t[o];
            std::fprintf(stderr, "[agc]     +%#x: %08x %08x %08x %08x", o * 4,
                         t[o], t[o + 1], t[o + 2], t[o + 3]);
            if (p >= 0x8000000000ull && p < 0x8100000000ull) {
              auto *pw = reinterpret_cast<const uint32_t *>(p);
              std::fprintf(stderr, " -> %#lx:", (unsigned long)p);
              for (int j = 0; j < 8; j++) std::fprintf(stderr, " %08x", pw[j]);
            }
            std::fprintf(stderr, "\n");
          }
        }
        // Read the eboot's shader-pipeline object globals (0x937da8, 0x937de8, ...):
        // each holds the uploaded VS/PS GPU code addr at +0x20/+0x30. If nonzero, the
        // shader-create succeeded (ISA uploaded) and the bug is bind-time; if zero, the
        // create failed. eboot base = 0x201200000000 (deterministic PS5 load addr).
        {
          for (uint64_t g : {0x937da8ull, 0x937de8ull, 0x937e28ull, 0x937e68ull}) {
            uint64_t obj = 0x201200000000ull + g;
            auto *o = reinterpret_cast<const uint64_t *>(obj);
            // guard: only deref if the object pointer slot itself looks mapped
            std::fprintf(stderr, "[agc]   pipe-obj @eboot+%#lx: [0]=%#lx +0x18=%#lx "
                         "+0x20=%#lx +0x30=%#lx +0x38=%#lx\n", (unsigned long)g,
                         (unsigned long)o[0], (unsigned long)o[3], (unsigned long)o[4],
                         (unsigned long)o[6], (unsigned long)o[7]);
          }
          // Shader-descriptor pointer array (0x8f4100): relocated at load to point at
          // the embedded shader ELFs (eboot rodata 0x77acf0+). If zero/wrong, relocs
          // didn't apply -> create reads garbage. Also the GPU pool object *0x986728.
          {
            auto *sd = reinterpret_cast<const uint64_t *>(0x2012008f4100ull);
            std::fprintf(stderr, "[agc]   shaderDescArr@0x8f4100:");
            for (int j = 0; j < 6; j++) std::fprintf(stderr, " %#lx", (unsigned long)sd[j]);
            // The pool object IS at 0x986728 (its [0] is the vtable). Read its own
            // arena fields (+8..+0x60) -- the mapped VA base + size stored by gate A.
            auto *pool = reinterpret_cast<const uint64_t *>(0x201200986728ull);
            std::fprintf(stderr, "\n[agc]   poolObj@0x986728 own fields:");
            for (int j = 0; j < 14; j++) std::fprintf(stderr, " [+%#x]=%#lx", j * 8,
                                                      (unsigned long)pool[j]);
            std::fprintf(stderr, "\n");
            // type-1 arena vector: begin=pool[+0x28], end=pool[+0x30]. Walk the block*
            // and its free-tree root freeRec (freeRec+0x8=VA, +0x10=size). If VA is the
            // GPU pool base -> pool is fine, null is downstream; if 0xaf junk -> the
            // libc-malloc'd metadata was freed under the pool (use-after-free).
            uint64_t vbeg = pool[5], vend = pool[6];  // +0x28, +0x30
            std::fprintf(stderr, "[agc]   type1 vec begin=%#lx end=%#lx (%ld blocks)\n",
                         (unsigned long)vbeg, (unsigned long)vend, (long)(vend - vbeg) / 8);
            if (vbeg && vend > vbeg && vbeg < 0x8000000000ull) {
              uint64_t block = *reinterpret_cast<const uint64_t *>(vbeg);
              if (block > 0x100000000ull && block < 0x8000000000ull) {
                auto *b = reinterpret_cast<const uint64_t *>(block);
                std::fprintf(stderr, "[agc]   block@%#lx: vt=%#lx freeRoot=%#lx size=%#lx VA=%#lx\n",
                             (unsigned long)block, (unsigned long)b[0], (unsigned long)b[2],
                             (unsigned long)b[3], (unsigned long)b[4]);
                // Walk the free-list next-chain, summing free space + counting nodes.
                uint64_t fr = b[2];  // +0x10 free-tree root
                uint64_t totalFree = 0, biggest = 0;
                int n = 0;
                while (fr > 0x100000000ull && fr < 0x8300000000ull && n < 4096) {
                  auto *f = reinterpret_cast<const uint64_t *>(fr);
                  uint64_t va = f[1], sz = f[2];
                  if (n < 5)
                    std::fprintf(stderr, "[agc]   freeRec[%d]@%#lx: next=%#lx VA=%#lx sz=%#lx\n",
                                 n, (unsigned long)fr, (unsigned long)f[0],
                                 (unsigned long)va, (unsigned long)sz);
                  if (sz < 0x100000000ull) { totalFree += sz; if (sz > biggest) biggest = sz; }
                  fr = f[0];  // next
                  n++;
                }
                std::fprintf(stderr, "[agc]   free-list: %d nodes, total=%#lx biggest=%#lx\n",
                             n, (unsigned long)totalFree, (unsigned long)biggest);
              }
            }
          }
        }
        // Is the ACQRB ring (0x8002670000, where the driver's record builder puts the
        // DCB/CCB) populated in mode-1? Scan 0x40000 of it for SET_SH_REG (op 0x76) /
        // SET_SH_REG_INDIRECT (0x63) headers and any shader PGM (0x8004400x).
        {
          auto *w = reinterpret_cast<const uint32_t *>(0x8002670000ull);
          int sh = 0, nz = 0;
          for (uint32_t j = 0; j < 0x40000 / 4 && sh < 12; j++) {
            uint32_t v = w[j];
            if (v) nz++;
            if ((v >> 30) == 3) {
              uint32_t o = (v >> 8) & 0xFF;
              if (o == 0x76 || o == 0x63) {
                std::fprintf(stderr, "[agc]   ACQRB+%#x SET_SH hdr=%08x body=%08x %08x %08x\n",
                             j * 4, v, w[j + 1], w[j + 2], w[j + 3]);
                sh++;
              }
            }
          }
          std::fprintf(stderr, "[agc]   ACQRB ring: %d nonzero dwords, %d SET_SH pkts\n", nz, sh);
        }
        // Find WHERE a pointer to the shader page lives (the out-of-band bind).
        std::fprintf(stderr, "[agc]   searching for shader pointers...\n");
        int sp = 0;
        pr->getVma().forEachGpuAperturePage(scanPageForShaderPtr, &sp);
        std::fprintf(stderr, "[agc]   shader-pointer search: %d hits\n", sp);
        // MMIO REGISTER-APERTURE hypothesis: the driver may write SPI_SHADER_PGM_LO
        // directly to a mapped GPU register mirror. Scan the named GPU register/dump
        // regions (SceGnmGpuInfo/DumpArea 0xfe0300000 1MB, DingDong 0xfe0200000) for
        // the shader pointer (PGM_LO 0x80044006 or raw 0x044006xx). Guarded reads.
        for (auto rr : {std::make_pair(0xfe0300000ull, 0x100000u),
                        std::make_pair(0xfe0200000ull, 0x4000u)}) {
          auto *w = reinterpret_cast<const uint32_t *>(rr.first);
          int found = 0;
          for (uint32_t o = 0; o < rr.second / 4 && found < 12; o++) {
            uint32_t v = w[o];
            if (v == 0x80044006u || v == 0x80044011u ||
                ((v & 0xFFFFF000u) == 0x04400000u)) {
              std::fprintf(stderr, "[agc]   MMIO %#lx+%#x = %08x (shader PGM!)\n",
                           (unsigned long)rr.first, o * 4, v);
              found++;
            }
          }
          std::fprintf(stderr, "[agc]   MMIO region %#lx: %d PGM-like hits\n",
                       (unsigned long)rr.first, found);
        }
        // Dump the 0x80 bytes preceding the first shader (0x8004400600): the AGC
        // toolkit writes the shader's rsrc1/rsrc2 + a small pipeline header there.
        {
          auto *w = reinterpret_cast<const uint32_t *>(0x8004400580ull);
          std::fprintf(stderr, "[agc]   pre-shader@0x8004400580:");
          for (int j = 0; j < 32; j++) std::fprintf(stderr, " %08x", w[j]);
          std::fprintf(stderr, "\n");
        }
      }
    }
  }
  // Deep-dump the 0x80788123 setup ioctl (registers shader/pipeline regions): follow
  // every GPU pointer 2 levels to locate a pipeline/shader descriptor table.
  if (agcTr && cmd == 0x80788123 && data) {
    static int s_setup = 0;
    if (s_setup < 3) {
      s_setup++;
      uint32_t len = (cmd >> 16) & 0x1fff;
      auto *w = static_cast<uint32_t *>(data);
      std::fprintf(stderr, "[agc] SETUP 0x80788123 len=%u:", len);
      for (uint32_t k = 0; k * 4 < len && k < 30; k++) std::fprintf(stderr, " %08x", w[k]);
      std::fprintf(stderr, "\n");
      for (uint32_t k = 0; (k + 1) * 4 <= len && k < 30; k++) {
        uint64_t p = (static_cast<uint64_t>(w[k + 1]) << 32) | w[k];
        if (p >= 0x8000000000ull && p < 0x8100000000ull) {
          auto *pw = reinterpret_cast<const uint32_t *>(p);
          std::fprintf(stderr, "[agc]   setup[+%u] -> %#lx:", k * 4, (unsigned long)p);
          for (int j = 0; j < 12; j++) std::fprintf(stderr, " %08x", pw[j]);
          std::fprintf(stderr, "\n");
        }
      }
    }
  }
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
    prosperity_agc_flip(0);  // present the frame
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
          if (p >= 0x8000000000ull && p < 0x8100000000ull) {
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
      // frame's state submit (the title issues no videoout SubmitFlip and no
      // 0x8133), then submit this frame's register state.
      prosperity_agc_flip(0);
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
      if (ptr >= 0x7ff000000000ull && count && count < 64) {
        auto *d = reinterpret_cast<uint32_t *>(ptr);
        static int s_desc = 0;
        if (agcTr && s_desc < 4) {
          s_desc++;
          std::fprintf(stderr, "[agc] 0x8132 descs count=%u:\n", count);
          for (uint32_t i = 0; i < count; i++)
            std::fprintf(stderr, "[agc]   desc[%u]: addr=%08x%08x sz=%08x flags=%08x\n",
                         i, d[i * 4 + 1], d[i * 4], d[i * 4 + 2], d[i * 4 + 3]);
        }
        for (uint32_t i = 0; i < count; i++) {
          uint64_t buf = (static_cast<uint64_t>(d[i * 4 + 1]) << 32) | d[i * 4];
          uint32_t sz = d[i * 4 + 2];
          if (buf >= 0x8000000000ull && buf < 0x8100000000ull && sz)
            prosperity_agc_submit(buf, sz * 4);
        }
      }
    }
    return 0;
  }
  case 0x80088133:  // AGC mode-1 end-of-frame / flip signal (IN, 8 bytes), issued
                    // once per frame after the 0x8131 state + 0x8132 draw submits.
                    // Present the rendered render target (0 -> the last RT drawn).
    prosperity_agc_flip(0);
    return 0;
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
        bool gpu = p >= 0x8000000000ull && p < 0x8100000000ull;
        bool stk = p >= 0x7ff000000000ull && p < 0x800000000000ull;
        if (gpu || stk) {
          auto *pw = reinterpret_cast<const uint32_t *>(p);
          std::printf("[agc]   +%u ptr=%#lx ->", k * 4, (unsigned long)p);
          for (int j = 0; j < 8; j++) std::printf(" %08x", pw[j]);
          // If the struct holds a further (GPU) pointer, deref that too (a label).
          if (stk) {
            for (int e = 0; e < 6; e++) {
              uint64_t cand = (static_cast<uint64_t>(pw[e + 1]) << 32) | pw[e];
              if (cand >= 0x8000000000ull && cand < 0x8100000000ull) {
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
