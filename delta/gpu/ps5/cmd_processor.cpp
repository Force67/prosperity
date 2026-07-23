/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 AGC command processor. The AGC command buffer libSceAgc builds is a PM4
 * type-3 stream using the SAME IT_ opcode table as the PS4 (verified in memory
 * [[ps5-agc-gpu]]), so the packet walk + completion-label handling reuse
 * gpu/ps4/pm4.h. What differs on PS5 is the gfx10.3 register offsets (agc_regs.h)
 * and the RDNA2 shader ISA (gpu/ps5/rdna) -- both new; the Vulkan renderer
 * (gpu/vk_render) and the DrawInfo contract are shared with the PS4 path.
 *
 * The register-latch model mirrors gpu/ps4/cmd_processor.cpp: SET_*_REG packets
 * write into a flat gfx10.3 register file (masking the gfx10 selector bits), a
 * draw packet snapshots that state into gpu::vk::DrawInfo, the VS(from the merged
 * ES/GS NGG block)/PS pair is recompiled (cached), and the shared renderer runs
 * the game's real shaders. Completion labels (EOP / RELEASE_MEM / WRITE_DATA)
 * are still serviced so the engine's per-frame command-buffer fences advance.
 */

#include "cmd_processor.h"

#include "agc_regs.h"
#include "ps4/pm4.h"
#include "rdna/rdna_resource.h"
#include "rdna/rdna_translate.h"
#include "vk_render.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace gpu::ps5 {
namespace {

std::mutex g_mtx;
const bool g_trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
uint64_t g_totalSubmits = 0;

// gfx10.3 register file (persists across submits -- AGC relies on sticky state).
Regs g_regs;
uint32_t g_indexType = 0;     // 0=uint16, 1=uint32, 2=uint8 (VGT_INDEX_TYPE[1:0])
uint32_t g_numInstances = 1;  // from IT_NUM_INSTANCES
bool g_frameActive = false;

// PS5 guest allocations sit anywhere across a wide VA (eboot ~0x2014_..., GPU
// dmem tagged regions 0x8000_..., doorbell 0xfe0_...). A shader/RT/buffer address
// is host-readable if plausibly mapped and non-tiny.
inline bool inGuest(uint64_t a) {
  return a >= 0x10000ull && a < 0x1000000000000ull;
}

// EOP/RELEASE_MEM DATA_SEL 3/4 ask the GPU to write its running clock counter;
// our submit is synchronous so any advancing non-zero value reads as complete.
inline bool labelAddrOk(uint64_t a) { return inGuest(a); }
uint64_t gpuClockTs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}
void writeLabel(uint64_t addr, uint64_t value, bool is64) {
  if (!labelAddrOk(addr)) return;
  if (is64)
    *reinterpret_cast<volatile uint64_t *>(addr) = value;
  else
    *reinterpret_cast<volatile uint32_t *>(addr) = static_cast<uint32_t>(value);
}

// gfx10.3 128-bit V# (buffer descriptor). base48 = f0 | (f1[15:0] << 32);
// stride f1[29:16]; num_records f2; format f3[18:12]. The 7-bit format field is
// the GCN (nfmt<<4)|dfmt packing (gfx10.3 DecodeTBufferFormat =
// ((nfmt & 0x7) << 4) | (dfmt & 0xf)), so dfmt = fmt&0xF, nfmt = (fmt>>4)&0x7.
struct VBuffer {
  uint64_t base = 0;
  uint32_t stride = 0, numRecords = 0, dfmt = 0, nfmt = 0, gfmt = 0;
};
VBuffer decodeVBuffer(const uint32_t *p) {
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.numRecords = p[2];
  v.gfmt = (p[3] >> 12) & 0x7F;
  v.dfmt = v.gfmt & 0xF;
  v.nfmt = (v.gfmt >> 4) & 0x7;
  return v;
}

// Latch a SET_*_REG packet into the register file. body[0] is the register
// offset dword (with the gfx10 selector bits stripped), body[1..] the values.
void setRegs(uint32_t base, const uint32_t *body, uint32_t count) {
  if (count < 1) return;
  uint32_t off = base + (body[0] & ~kRegSelectorMask);
  for (uint32_t i = 1; i < count; i++)
    if (off + (i - 1) < kRegFileSize) g_regs[off + (i - 1)] = body[i];
}

// Latch a LOAD_*_REG packet: the register values live in a GPU-memory image at
// body[0..1]; body[2..] are (reg_offset, num_dwords) ranges. Each range copies
// `num` dwords from the image (image[reg_offset..]) into the reg file at
// base+reg_offset. This is how the AGC driver sets context/sh/uconfig state for
// this title (it uses LOAD_*_REG, not inline SET_*_REG), so without it CB_COLOR
// (the render target) is never bound and draws hit no visible target.
void loadRegs(uint32_t base, const uint32_t *body, uint32_t count) {
  if (count < 4) return;
  uint64_t mem = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
  if (mem < 0x8000000000ull || mem >= 0x8100000000ull) return;  // GPU aperture only
  const uint32_t *src = reinterpret_cast<const uint32_t *>(mem);
  // COHERENCY TEST: dump the LOAD image content. If a context/SH image reads all
  // zero, it's non-coherent (the driver wrote it via a different VA) -- the shader
  // would be present but invisible. If it has real reg values, the image is fine.
  static int s_img = 0;
  if (g_trace && s_img < 6) {
    s_img++;
    std::fprintf(stderr, "[agc]   LOADimg base=%#x mem=%#lx:", base, (unsigned long)mem);
    for (int j = 0; j < 16; j++) std::fprintf(stderr, " %08x", src[j]);
    std::fprintf(stderr, "\n");
  }
  // SH-image ground-truth: LOAD_SH_REG could be offset-INDEXED (image[reg_off] =
  // value, a full reg-file shadow) OR CURSOR-based (ranges packed contiguously from
  // mem). Dump both interpretations + the range list, and scan the whole image for a
  // PGM pointer (top byte 0x80), to settle where the shader PGM actually is.
  static int s_shimg = 0, s_shcnt = 0;
  // Fire on the first few SH loads AND on a batch ~5000 loads in (once the title is
  // past the loading screen and issuing real sprite draws) -- the early shadow may be
  // empty simply because rendering hasn't started.
  bool lateWindow = base == kShRegBase && s_shcnt >= 5000 && s_shcnt < 5006;
  if (base == kShRegBase) s_shcnt++;
  if (g_trace && base == kShRegBase && (s_shimg < 3 || lateWindow)) {
    s_shimg++;
    std::fprintf(stderr, "[agc] (shload #%d)\n", s_shcnt);
    std::fprintf(stderr, "[agc] SHLOAD mem=%#lx ranges:", (unsigned long)mem);
    for (uint32_t i = 2; i + 1 < count; i += 2)
      std::fprintf(stderr, " (off=%#x,num=%u)", body[i] & 0xFFFF, body[i + 1] & 0xFFFF);
    std::fprintf(stderr, "\n[agc]   indexed[0x08..0x0b]=%08x %08x %08x %08x  "
                 "indexed[0x88..0x8b]=%08x %08x %08x %08x\n",
                 src[0x08], src[0x09], src[0x0a], src[0x0b],
                 src[0x88], src[0x89], src[0x8a], src[0x8b]);
    // cursor walk: read ranges contiguously from mem, show reg_off + first two vals
    uint32_t cur = 0;
    for (uint32_t i = 2; i + 1 < count; i += 2) {
      uint32_t off = body[i] & 0xFFFF, num = body[i + 1] & 0xFFFF;
      if (num > 0x400) num = 0x400;
      std::fprintf(stderr, "[agc]   cursor off=%#x <- img[%u..]: %08x %08x\n",
                   off, cur, src[cur], num > 1 ? src[cur + 1] : 0);
      cur += num;
    }
    // scan whole image for a PGM-like pointer (val<<8 in the GPU aperture)
    for (uint32_t j = 0; j < 0x400; j++) {
      uint32_t v = src[j];
      if ((v >> 24) == 0x80) {
        uint64_t a = (uint64_t)v << 8;
        if (a >= 0x8000000000ull && a < 0x8100000000ull) {
          const uint32_t *w = reinterpret_cast<const uint32_t *>(a);
          std::fprintf(stderr, "[agc]   img[%#x]=%08x -> %#lx ISA? %08x %08x\n",
                       j, v, (unsigned long)a, w[0], w[1]);
        }
      }
    }
  }
  for (uint32_t i = 2; i + 1 < count; i += 2) {
    uint32_t off = body[i] & 0xFFFF;
    uint32_t num = body[i + 1] & 0xFFFF;
    if (num > 0x2000) num = 0x2000;  // sanity cap
    for (uint32_t j = 0; j < num; j++) {
      if (base + off + j >= kRegFileSize) break;
      uint32_t v = src[off + j];
      g_regs[base + off + j] = v;
      // Pinpoint where the shader PGM_LO lands: a shader at 0x8001xxxxxx has
      // PGM_LO ~0x800xxxxx (top byte 0x80). Log SH-space hits to find the reg.
      static int s_sh = 0;
      if (g_trace && base == kShRegBase && (v >> 24) == 0x80 && s_sh < 20) {
        s_sh++;
        std::fprintf(stderr, "[agc]   SH pgm? off=%#x val=%#x (addr~%#lx)\n",
                     off + j, v, (unsigned long)((uint64_t)v << 8));
      }
    }
  }
}

// Latch a SET_*_REG_INDIRECT packet (op 0x9f context / 0x93 sh / 0x64 uconfig).
// body = [addrLo, addrHi, mode, numDwords]; the GPU buffer at addr holds
// (reg_offset, value) PAIRS. This is how the AGC driver binds the per-draw render
// target + shaders for this title (e.g. reg 0x318 CB_COLOR0_BASE, 0x8e
// CB_TARGET_MASK), so without it draws hit no target and no shader.
void loadRegPairs(uint32_t base, const uint32_t *body, uint32_t cnt) {
  if (cnt < 4) return;
  uint64_t addr = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | (body[0] & 0xFFFFFFFCu);
  if (addr < 0x8000000000ull || addr >= 0x8100000000ull) return;
  // body[3] is the count of (reg_offset, value) register PAIRS, not dwords: each
  // iteration reads two dwords (the gfx10.3 SET_*_REG_INDIRECT handlers
  // loop `i < (buffer[3] & 0x3fff)` advancing the pointer by 2). The old
  // dword-count reading wrote nothing for a single-register indirect (num == 1),
  // so VGT_PRIMITIVE_TYPE (set this way) never landed and prim assembly died.
  uint32_t numPairs = body[3] & 0x3FFF;
  const uint32_t *p = reinterpret_cast<const uint32_t *>(addr);
  for (uint32_t i = 0; i < numPairs; i++) {
    uint32_t off = p[i * 2] & ~kRegSelectorMask;  // strip gfx10 selector bits
    if (base + off < kRegFileSize) g_regs[base + off] = p[i * 2 + 1];
  }
  // Report the first few shader binds that actually carry a nonzero PGM (SH off 0x88
  // = PGM_LO_GS, 0x08 = PGM_LO_PS) -- these are the real sprite-pipeline binds.
  static int s_shpgm = 0;
  if (g_trace && base == kShRegBase && s_shpgm < 6) {
    uint32_t gsLo = g_regs[kShRegBase + 0x88], psLo = g_regs[kShRegBase + 0x08];
    if (gsLo || psLo) {
      s_shpgm++;
      std::fprintf(stderr, "[agc] SHADER BIND @%#lx: PGM_LO_GS=%08x PGM_LO_PS=%08x\n",
                   (unsigned long)addr, gsLo, psLo);
    }
  }
}

// Render-target dimensions, derived from the viewport the title programmed: the RT
// spans [center-halfSize, center+halfSize] where halfSize = |VPORT_xSCALE|. The
// screen scissor is a "no-clip" max (e.g. 16384) and is NOT the RT size. Reading the
// viewport keeps this title-agnostic (no fixed default resolution); a 0 scale (no
// viewport yet) yields 0 so getRT skips the draw until real state is set.
uint32_t fbDim(uint32_t scaleReg) {
  float s;
  std::memcpy(&s, &g_regs[scaleReg], 4);
  if (s < 0.0f) s = -s;
  return static_cast<uint32_t>(s * 2.0f + 0.5f);
}
uint32_t fbWidth() { return fbDim(mmPA_CL_VPORT_XSCALE); }
uint32_t fbHeight() { return fbDim(mmPA_CL_VPORT_YSCALE); }

bool isDraw(uint32_t op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// Cache key: the VS/PS/fetch triple (attribute translation depends on the fetch
// program, not just the VS/PS code).
struct ShaderKey {
  uint64_t vs, ps, fetch;
  bool operator==(const ShaderKey &o) const {
    return vs == o.vs && ps == o.ps && fetch == o.fetch;
  }
};
struct ShaderKeyHash {
  size_t operator()(const ShaderKey &k) const {
    return std::hash<uint64_t>{}(k.vs) ^ (std::hash<uint64_t>{}(k.ps) << 1) ^
           (std::hash<uint64_t>{}(k.fetch) << 2);
  }
};
std::unordered_map<ShaderKey, gcn::Recompiled, ShaderKeyHash> g_shCache;

void handleDraw(uint32_t op, const uint32_t *body, uint32_t count) {
  if (!vk::available()) return;

  // gfx10.3 has no HW VS: the vertex program is the merged NGG shader, whose
  // address is written to the ES (front half, 0xC8) and/or GS (back half, 0x88)
  // PGM_LO. Some pipelines populate only the ES slot, so fall back to it when the
  // GS slot reads 0. User data (cbuffer/MVP pointers) stays in the GS block.
  uint64_t vsA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_GS);
  if (!inGuest(vsA)) vsA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_ES);
  uint64_t psA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_PS);
  const uint32_t *vud = &g_regs[mmSPI_SHADER_USER_DATA_GS_0];
  const uint32_t *pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];

  static int s_uddump = 0;
  if (g_trace && s_uddump < 4) {
    s_uddump++;
    const uint32_t *gs = &g_regs[mmSPI_SHADER_USER_DATA_GS_0];
    const uint32_t *es = &g_regs[mmSPI_SHADER_USER_DATA_ES_0];
    std::fprintf(stderr, "[agc]   UD GS:");
    for (int j = 0; j < 16; j++) std::fprintf(stderr, " %08x", gs[j]);
    std::fprintf(stderr, "\n[agc]   UD ES:");
    for (int j = 0; j < 16; j++) std::fprintf(stderr, " %08x", es[j]);
    std::fprintf(stderr, "\n");
  }

  // This title programs the shader PGM_LO/HI at non-standard SH offsets (via the
  // inline op 0x93), so the fixed GS/PS regs above read 0. Fallback: scan the SH
  // register file for PGM pairs whose address is a 256-aligned GPU-aperture pointer
  // to plausible RDNA2 ISA (first dword's top byte is a scalar/vector encoding).
  if (!inGuest(vsA) || !inGuest(psA)) {
    uint64_t found[16]; uint32_t foundReg[16]; int nf = 0;
    // Scan the SH register block for a PGM pair pointing at RDNA2 shader ISA
    // (both PGM encodings). NOTE (session 2): proven that NO register in the whole
    // file points at shader code -- the AGC binds shaders outside the PM4 stream --
    // so this stays inert until that path is decoded.
    auto tryAddr = [&](uint32_t o, uint64_t a) {
      if (nf >= 16 || a < 0x8000000000ull || a >= 0x8100000000ull || (a & 0xFF))
        return;
      uint32_t w0 = *reinterpret_cast<const uint32_t *>(a);
      if ((w0 >> 24) < 0x7e || w0 == 0xffffffffu) return;  // ISA plausibility
      for (int k = 0; k < nf; k++) if (found[k] == a) return;
      foundReg[nf] = o; found[nf++] = a;
    };
    for (uint32_t o = 0; o + 1 < 0x300; o++) {
      uint32_t lo = g_regs[kShRegBase + o], hi = g_regs[kShRegBase + o + 1];
      tryAddr(o, (static_cast<uint64_t>(lo) << 8) | ((uint64_t)(hi & 0xFF) << 40));
      tryAddr(o, (static_cast<uint64_t>(hi & 0xFFFF) << 32) | lo);
    }
    // The op 0x93 writes SH reg 0x113 = a GPU ptr (raw (HI<<32)|LO). PS5 shaders
    // have a metadata HEADER before the ISA, so the first dword isn't an opcode.
    // Dump it deeply to locate a shader binary + the ISA offset.
    static int s_hdr = 0;
    uint64_t a113 = (static_cast<uint64_t>(g_regs[kShRegBase + 0x114] & 0xFFFF) << 32) |
                    g_regs[kShRegBase + 0x113];
    if (g_trace && s_hdr < 3 && a113 >= 0x8000000000ull && a113 < 0x8100000000ull) {
      s_hdr++;
      auto *w = reinterpret_cast<const uint32_t *>(a113);
      std::fprintf(stderr, "[agc]   reg0x113 -> %#lx dump:", (unsigned long)a113);
      for (int j = 0; j < 32; j++) std::fprintf(stderr, " %08x", w[j]);
      std::fprintf(stderr, "\n");
    }
    if (nf >= 1 && !inGuest(vsA)) vsA = found[0];
    if (nf >= 2 && !inGuest(psA)) psA = found[1];
    static int s_sc = 0;
    if (g_trace && s_sc < 6 && nf) {
      s_sc++;
      std::fprintf(stderr, "[agc]   shader scan: nf=%d", nf);
      for (int k = 0; k < nf && k < 6; k++)
        std::fprintf(stderr, " [%#x]=%#lx", foundReg[k], (unsigned long)found[k]);
      std::fprintf(stderr, "\n");
    }
    // The AGC binds shaders via a pipeline/descriptor, not PGM regs. Dump the GS/PS
    // user-data (16 dwords each) and follow any GPU-aperture pointer one level to
    // look for shader ISA -- the pipeline handle/PGM likely lives in a descriptor.
    static int s_ud = 0;
    if (g_trace && s_ud < 3) {
      s_ud++;
      for (int which = 0; which < 2; which++) {
        const uint32_t *ud = which ? pud : vud;
        std::fprintf(stderr, "[agc]   %sUD:", which ? "ps" : "gs");
        for (int k = 0; k < 16; k++) std::fprintf(stderr, " %08x", ud[k]);
        std::fprintf(stderr, "\n");
        for (int k = 0; k + 1 < 16; k++) {
          uint64_t p = (static_cast<uint64_t>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
          if (p >= 0x8000000000ull && p < 0x8100000000ull) {
            auto *pw = reinterpret_cast<const uint32_t *>(p);
            std::fprintf(stderr, "[agc]     UD[%d]->%#lx:", k, (unsigned long)p);
            for (int j = 0; j < 8; j++) std::fprintf(stderr, " %08x", pw[j]);
            std::fprintf(stderr, "\n");
          }
        }
      }
    }
  }

  vk::DrawInfo d;
  d.primType = g_regs[mmVGT_PRIMITIVE_TYPE];
  d.instanceCount = g_numInstances;
  uint32_t autoVertexCount = op == IT_DRAW_INDEX_AUTO && count >= 1 ? body[0] : 0;

  // Index buffer (DRAW_INDEX_2: maxSize, baseLo, baseHi, indexCount, initiator).
  if (op == IT_DRAW_INDEX_2 && count >= 4) {
    uint64_t ibase = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
    uint32_t icount = body[3];
    if (inGuest(ibase) && icount && icount <= 0x100000) {
      d.indexData = reinterpret_cast<const void *>(ibase);
      d.indexCount = icount;
      d.indexType = g_indexType;
      static int s_idxdump = 0;
      if (g_trace && s_idxdump < 8) {
        s_idxdump++;
        const uint16_t *i16 = reinterpret_cast<const uint16_t *>(ibase);
        const uint32_t *i32 = reinterpret_cast<const uint32_t *>(ibase);
        std::fprintf(stderr, "[agc]   IDX ibase=%#lx count=%u type=%u  u16:",
                     (unsigned long)ibase, icount, g_indexType);
        for (uint32_t k = 0; k < icount && k < 8; k++)
          std::fprintf(stderr, " %u", i16[k]);
        std::fprintf(stderr, "  u32:");
        for (uint32_t k = 0; k < icount && k < 8; k++)
          std::fprintf(stderr, " %u", i32[k]);
        std::fprintf(stderr, "\n");
      }
    }
  }

  d.rtW = fbWidth();
  d.rtH = fbHeight();

  // Color targets: bind CB_COLORn only when its write mask and INFO format are
  // valid (a stale base remains programmed during depth-only passes).
  {
    uint32_t tmask = g_regs[mmCB_TARGET_MASK];
    for (int rt = 0; rt < 8; rt++) {
      uint64_t base = g_regs.cbColorBase(rt);
      uint32_t info = g_regs[mmCB_COLOR0_INFO + rt * kCbColorStride];
      if (((tmask >> (rt * 4)) & 0xF) && ((info >> 2) & 0x1F) && inGuest(base)) {
        d.mrtBase[rt] = base;
        d.mrtInfo[rt] = info;
        d.mrtCount = rt + 1;
      }
    }
    d.rtBase = d.mrtCount ? d.mrtBase[0] : 0;
    static int s_rtdbg = 0;
    if (g_trace && s_rtdbg < 12) {
      s_rtdbg++;
      std::fprintf(stderr, "[agc]   DRAW rt: cb0Base=%#lx info0=%#x tmask=%#x -> mrtCount=%u "
                   "vsA=%#lx psA=%#lx prim=%#x idx=%u\n",
                   (unsigned long)g_regs.cbColorBase(0),
                   g_regs[mmCB_COLOR0_INFO], g_regs[mmCB_TARGET_MASK], d.mrtCount,
                   (unsigned long)vsA, (unsigned long)psA, d.primType, d.indexCount);
    }
  }

  // Per-MRT blend (CB_BLENDn_CONTROL, bit 30 = enable).
  d.blendControl = g_regs[mmCB_BLEND0_CONTROL];
  d.blendEnable = (d.blendControl >> 30) & 1u;
  d.mrtBlend[0] = d.blendControl;
  if (d.blendEnable) d.mrtBlendMask |= 1u;
  for (uint32_t rt = 1; rt < 8; rt++) {
    uint32_t bc = g_regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
    d.mrtBlend[rt] = bc;
    if ((bc >> 30) & 1u) d.mrtBlendMask |= (1u << rt);
  }
  d.targetMask = g_regs[mmCB_TARGET_MASK];
  d.colorControl = g_regs[mmCB_COLOR_CONTROL];

  // Depth/stencil (gfx10 Z base = (WRITE_BASE | WRITE_BASE_HI<<32) << 8).
  {
    uint32_t dc = g_regs[mmDB_DEPTH_CONTROL];
    uint32_t zinfo = g_regs[mmDB_Z_INFO];
    uint64_t zbase = ((static_cast<uint64_t>(g_regs[mmDB_Z_WRITE_BASE_HI]) << 32) |
                      g_regs[mmDB_Z_WRITE_BASE]) << 8;
    d.depthValid = (zinfo & 0x3) != 0;
    if (d.depthValid && inGuest(zbase) && (((dc >> 1) & 1u) || ((dc >> 2) & 1u))) {
      d.depthBase = zbase;
      d.depthTestEnable = (dc >> 1) & 1u;
      d.depthWriteEnable = (dc >> 2) & 1u;
      d.depthFunc = (dc >> 4) & 0x7;
      std::memcpy(&d.depthClear, &g_regs[mmDB_DEPTH_CLEAR], 4);
      if (!(d.depthClear >= 0.0f && d.depthClear <= 1.0f)) d.depthClear = 1.0f;
    } else {
      d.depthValid = false;
    }
  }

  // Primitive setup + viewport.
  {
    uint32_t sc = g_regs[mmPA_SU_SC_MODE_CNTL];
    d.cullMode = sc & 0x3;
    d.frontCCW = ((sc >> 2) & 1u) == 0;
  }
  std::memcpy(&d.viewportXScale, &g_regs[mmPA_CL_VPORT_XSCALE], 4);
  std::memcpy(&d.viewportXOffset, &g_regs[mmPA_CL_VPORT_XOFFSET], 4);
  std::memcpy(&d.viewportYScale, &g_regs[mmPA_CL_VPORT_YSCALE], 4);
  std::memcpy(&d.viewportYOffset, &g_regs[mmPA_CL_VPORT_YOFFSET], 4);

  // Fetch-shader pointer (a heuristic default: GS user data[0..1]; the AGC
  // input-usage table is authoritative and a follow-up).
  uint64_t fetch = (static_cast<uint64_t>(vud[1] & 0xFFFF) << 32) | vud[0];

  // Recompile the VS/PS pair (cached) and resolve the live vertex-attribute
  // buffers + constant buffers from the RDNA2 descriptors in user data.
  static const bool recompOn = [] {
    const char *e = std::getenv("DELTA_PS5_RECOMP");
    return !e || std::strcmp(e, "0") != 0;
  }();
  static uint64_t s_dlN = 0;
  uint64_t myDraw = s_dlN++;
  bool dl = g_trace && s_dlN < 5000;
  if (recompOn && inGuest(vsA) && (!psA || inGuest(psA))) {
    ShaderKey key{vsA, psA, fetch};
    auto it = g_shCache.find(key);
    if (it == g_shCache.end()) {
      if (dl) {
        const uint32_t *vc = reinterpret_cast<const uint32_t *>(vsA);
        const uint32_t *pc = psA ? reinterpret_cast<const uint32_t *>(psA) : nullptr;
        // AGC shader code starts with the 0xBEEB03FF sentinel; a psA that isn't
        // (e.g. 0xffc9dfe7 poison fill) means the PS PGM_LO reg didn't land.
        std::fprintf(stderr, "[agc] DL recompile vs=%#lx (%08x) ps=%#lx (%08x %08x)...\n",
                     (unsigned long)vsA, vc[0], (unsigned long)psA,
                     pc ? pc[0] : 0, pc ? pc[1] : 0);
      }
      it = g_shCache
               .emplace(key, rdna::Recompile(reinterpret_cast<const uint32_t *>(vsA),
                                             psA ? reinterpret_cast<const uint32_t *>(psA)
                                                 : nullptr,
                                             vud, pud))
               .first;
      if (dl) std::fprintf(stderr, "[agc] DL recompile done ok=%d\n", it->second.ok);
    }
    gcn::Recompiled &rc = it->second;
    if (rc.ok) {
      // Vertex attributes: the V# is either inline in user data at table_sgpr
      // (AGC frequently passes the vertex V# directly) or reached through a table
      // pointer at {table_sgpr, +1}. vk_render binds a single interleaved buffer,
      // so resolve as many attrs as possible into it and skip any that don't
      // decode to a valid V# (a partial fetch still rasterizes).
      uint64_t base0 = 0;
      bool haveBase = false;
      if (dl)
        std::fprintf(stderr, "[agc] DL attrs=%zu vud[0..7]=%08x %08x %08x %08x %08x %08x %08x %08x\n",
                     rc.attrs.size(), vud[0], vud[1], vud[2], vud[3], vud[4], vud[5], vud[6], vud[7]);
      for (size_t i = 0; i < rc.attrs.size() && i < 8; i++) {
        auto &a = rc.attrs[i];
        if (a.table_sgpr + 3 >= 32) continue;
        VBuffer vb = decodeVBuffer(&vud[a.table_sgpr]);  // inline V#
        const char *how = "inline";
        if (!inGuest(vb.base) || !vb.stride) {  // else follow a table pointer
          uint64_t tbl = (static_cast<uint64_t>(vud[a.table_sgpr + 1] & 0xFFFF) << 32) |
                         vud[a.table_sgpr];
          if (inGuest(tbl)) {
            vb = decodeVBuffer(reinterpret_cast<const uint32_t *>(tbl + a.vbuf_dword_off * 4));
            how = "table";
          }
        }
        if (dl)
          std::fprintf(stderr, "[agc]   attr%zu loc=%u nc=%u tbl_sgpr=%u off=%u (%s) -> "
                       "base=%#lx stride=%u nrec=%u gfmt=%u -> dfmt=%u nfmt=%u\n",
                       i, a.location, a.num_comps, a.table_sgpr, a.vbuf_dword_off, how,
                       (unsigned long)vb.base, vb.stride, vb.numRecords, vb.gfmt,
                       vb.dfmt, vb.nfmt);
        if (!inGuest(vb.base) || !vb.stride) continue;  // unresolved: keep the rest
        if (!haveBase) {
          haveBase = true;
          base0 = vb.base;
          d.vertexData = reinterpret_cast<const void *>(vb.base);
          d.vertexStride = vb.stride;
          d.vertexCount = vb.numRecords;
        }
        // Single interleaved binding: only fold in attrs that sit inside the base
        // buffer's stride; a separate buffer would alias garbage at a huge offset.
        uint32_t off = (vb.base >= base0 && vb.base - base0 < d.vertexStride)
                           ? static_cast<uint32_t>(vb.base - base0) : 0;
        d.vattrs[d.nvattrs++] = {a.location, off, a.num_comps, vb.dfmt, vb.nfmt};
      }
      // A fetch VS needs at least one attribute resolved; a procedural VS (no
      // recovered attrs, seeds from VertexIndex) draws without a vertex buffer.
      bool good = d.nvattrs > 0 || rc.attrs.empty();

      // Constant buffers: resolve each SMEM descriptor. For a direct cbuf the V# is
      // inline at ud_sgpr in user data; for a chained cbuf the descriptor is reached
      // by reading the root pointer from user data and dereferencing through guest
      // memory per chain_off[] (the VS loads its transform's V# from a root
      // descriptor table this way). The bind size is the recompiler's planned dword
      // window (the V# stride/records are meaningless for an s_load pointer).
      auto resolveCbufs = [&](const std::vector<gcn::ShaderCbuf> &cbufs,
                              const uint32_t *userData, bool vertexStage) {
        for (const auto &cb : cbufs) {
          if (cb.binding >= 8) continue;
          VBuffer vb{};
          const char *how = "direct";
          if (cb.chain_len == 0) {
            if (cb.ud_sgpr + 3 >= 32) continue;
            vb = decodeVBuffer(&userData[cb.ud_sgpr]);
          } else {  // walk the s_load pointer chain to the final V#
            if (cb.ud_sgpr + 1 >= 32) continue;
            uint64_t ptr = (static_cast<uint64_t>(userData[cb.ud_sgpr + 1] & 0xFFFF) << 32) |
                           userData[cb.ud_sgpr];
            bool ok = true;
            for (uint32_t i = 0; i + 1 < cb.chain_len; i++) {  // deref intermediate pointers
              uint64_t at = (ptr + cb.chain_off[i]) & ~uint64_t{3};
              if (!inGuest(at)) { ok = false; break; }
              const uint32_t *q = reinterpret_cast<const uint32_t *>(at);
              ptr = (static_cast<uint64_t>(q[1] & 0xFFFF) << 32) | q[0];
            }
            uint64_t vAddr = (ptr + cb.chain_off[cb.chain_len - 1]) & ~uint64_t{3};
            if (!ok || !inGuest(vAddr)) continue;
            vb = decodeVBuffer(reinterpret_cast<const uint32_t *>(vAddr));
            how = "chain";
          }
          uint64_t base = vb.base & ~uint64_t{3};  // dword-align (SMEM base)
          uint64_t bytes = static_cast<uint64_t>(cb.num_dwords) * 4;
          if (dl)
            std::fprintf(stderr, "[agc]   cbuf %s bind=%u root=%u %s(len=%u) base=%#lx dwords=%u\n",
                         vertexStage ? "vs" : "ps", cb.binding, cb.ud_sgpr, how,
                         cb.chain_len, (unsigned long)base, cb.num_dwords);
          if (!inGuest(base) || !bytes) continue;
          d.cbufs[cb.binding] = {base, static_cast<uint32_t>(bytes)};
          d.nCbufs = std::max(d.nCbufs, cb.binding + 1);
          if (vertexStage && bytes >= sizeof(d.mvp)) {
            d.cbufBase = base;
            d.cbufSize = static_cast<uint32_t>(bytes);
            std::memcpy(d.mvp, reinterpret_cast<const void *>(base), sizeof(d.mvp));
          }
        }
      };
      if (good) {
        resolveCbufs(rc.vs_cbufs, vud, true);
        if (psA) resolveCbufs(rc.ps_cbufs, pud, false);
        // Textures: resolve the live gfx10.3 T#/S# each PS sampler reads, in the
        // recompiler's set-0 binding order (rdna::TrackTextures re-derives the same
        // plan). texs[i] maps to PS sampler binding i.
        if (psA && !rc.ps_texs.empty()) {
          auto texs = rdna::TrackTextures(reinterpret_cast<const uint32_t *>(psA), pud);
          for (size_t i = 0; i < texs.size() && i < 16; i++) {
            const auto &s = texs[i];
            auto &dt = d.texs[i];
            dt.base = s.valid ? s.base : 0;
            dt.w = s.width;
            dt.h = s.height;
            dt.tiling = s.tiling_idx;
            dt.pitch = s.pitch;
            dt.dfmt = s.dfmt;
            dt.nfmt = s.nfmt;
            dt.layers = s.layers;
            dt.base_array = s.base_array;
            dt.view_layers = s.view_layers;
            dt.mip_levels = s.mip_levels;
            dt.base_mip = s.base_mip;
            dt.view_mips = s.view_mips;
            dt.min_lod = s.min_lod;
            std::memcpy(dt.sampler, s.sampler, sizeof(dt.sampler));
            dt.sampler_valid = s.sampler_valid;
            dt.arrayed = s.arrayed;
            dt.force_lod_zero = s.force_lod_zero;
            dt.depth_compare = s.depth_compare;
            dt.storage = s.storage;
          }
          d.nTexs = static_cast<uint32_t>(std::min<size_t>(texs.size(), 16));
        }
        d.vsAddr = vsA;
        d.psAddr = psA;
        d.recomp = &rc;
      } else {
        d.nvattrs = 0;
      }
    }
  }

  if (autoVertexCount && autoVertexCount <= 0x100000) d.vertexCount = autoVertexCount;
  if (!d.recomp) return;  // no shader -> nothing the renderer can run yet

  // One-shot ground-truth dump of the first few resolved draws: the raw vertex
  // bytes (as float32 AND uint32, to read off the real attribute format), the
  // constant-buffer/MVP state, and the viewport -- so we can tell whether the
  // positions are garbage (wrong format), screen-space (missing projection), or
  // clip-space (a downstream/viewport issue).
  static int s_vdump = 0;
  if (g_trace && s_vdump < 8 && d.nvattrs && d.vertexData &&
      inGuest(reinterpret_cast<uint64_t>(d.vertexData))) {
    s_vdump++;
    std::fprintf(stderr,
                 "[agc] VDUMP draw#%lu nvattrs=%u stride=%u count=%u prim=%u "
                 "vp=[xs=%g xo=%g ys=%g yo=%g] nCbufs=%u cbufBase=%#lx cbufSize=%u\n",
                 (unsigned long)myDraw, d.nvattrs, d.vertexStride, d.vertexCount,
                 d.primType, d.viewportXScale, d.viewportXOffset, d.viewportYScale,
                 d.viewportYOffset, d.nCbufs, (unsigned long)d.cbufBase, d.cbufSize);
    for (uint32_t a = 0; a < d.nvattrs; a++)
      std::fprintf(stderr, "[agc]   vattr%u loc=%u off=%u nc=%u dfmt=%u nfmt=%u\n",
                   a, d.vattrs[a].location, d.vattrs[a].offset, d.vattrs[a].num_comps,
                   d.vattrs[a].dfmt, d.vattrs[a].nfmt);
    const auto *vb = reinterpret_cast<const uint8_t *>(d.vertexData);
    uint32_t nv = d.vertexCount ? d.vertexCount : 4;
    uint32_t vbytes = std::min<uint32_t>(d.vertexStride * nv, 128u);
    for (uint32_t o = 0; o + 4 <= vbytes; o += 4) {
      uint32_t u;
      float f;
      std::memcpy(&u, vb + o, 4);
      std::memcpy(&f, vb + o, 4);
      std::fprintf(stderr, "[agc]     vtx[+%02u] u=%08x f=%g\n", o, u, f);
    }
    const float *m = d.mvp;
    std::fprintf(stderr,
                 "[agc]   mvp=[%g %g %g %g / %g %g %g %g / %g %g %g %g / %g %g %g %g]\n",
                 m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10],
                 m[11], m[12], m[13], m[14], m[15]);
  }

  if (!g_frameActive) {
    if (dl) std::fprintf(stderr, "[agc] DL beginFrame...\n");
    vk::beginFrame();
    g_frameActive = true;
  }
  if (dl) std::fprintf(stderr, "[agc] DL draw#%lu vk::draw nvattrs=%d rt=%#lx...\n",
                       (unsigned long)myDraw, d.nvattrs, (unsigned long)d.rtBase);
  vk::draw(d);
  if (dl) std::fprintf(stderr, "[agc] DL draw#%lu done\n", (unsigned long)myDraw);
}

uint32_t g_opHist[256] = {};
int g_dumped = 0;

// Walk one PM4 stream, following INDIRECT_BUFFER, latching registers, decoding
// draws, and writing completion labels. depth guards a malformed self-reference.
void walk(const uint32_t *p, uint32_t words, bool dumpThis, int depth) {
  if (!p || depth > 8) return;
  uint32_t i = 0;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = pm4Type(hdr);
    if (type == Pm4Type::type3) {
      uint32_t op = pm4Opcode(hdr);
      uint32_t cnt = pm4Count(hdr);  // body dword count
      const uint32_t *body = &p[i + 1];
      // Desync recovery: a data dword misread as a huge-count packet (e.g. a
      // RELEASE_MEM trailer 0xffff1000 parsed as NOP count=16384) would abandon the
      // rest of the buffer -- and with it the SET_SH_REG_INDIRECT shader bind that
      // follows. Instead of bailing, skip one dword and resync on the next header.
      if (i + 1 + cnt > words) { i += 1; continue; }
      g_opHist[op & 0xFF]++;
      if (dumpThis) {
        std::fprintf(stderr, "[agc]   @%-5u T3 op=%#04x count=%u body:", i, op, cnt);
        uint32_t showN = (op == 0x93 || op == 0x79) ? cnt : (cnt < 6 ? cnt : 6);
        for (uint32_t b = 0; b < showN && b < 24; b++)
          std::fprintf(stderr, " %08x", body[b]);
        // INDIRECT register packets reference a GPU buffer at body[0..1]; dump it
        // so we can RE the register layout (which offset holds CB_COLOR/shaders).
        if ((op == 0x9f || op == 0x93 || op == 0x64 || op == 0x7a || op == 0x63) && cnt >= 2) {
          uint64_t a = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          if (a >= 0x8000000000ull && a < 0x8100000000ull) {
            auto *aw = reinterpret_cast<const uint32_t *>(a);
            std::fprintf(stderr, " -> buf %#lx:", (unsigned long)a);
            for (int b = 0; b < 12; b++) std::fprintf(stderr, " %08x", aw[b]);
          }
        }
        std::fprintf(stderr, "\n");
      }
      switch (op) {
      case IT_INDIRECT_BUFFER:       // baseLo, baseHi, sizeDwords(+flags)
      case 0x33: {                   // IT_INDIRECT_BUFFER_CNST (AGC constant/Cue chain
                                     // -- carries the pipeline SET_SH_REG shader setup)
        if (cnt >= 3) {
          uint64_t ib = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          uint32_t ibw = body[2] & 0xFFFFF;
          // Bounds-guard: only follow IBs into the GPU aperture with a sane size,
          // so a stale/garbage ring window can't fault the walker.
          if (ib >= 0x8000000000ull && ib < 0x8100000000ull && ibw &&
              ibw <= 0x40000)
            walk(reinterpret_cast<const uint32_t *>(ib), ibw, dumpThis, depth + 1);
        }
        break;
      }
      case IT_SET_CONTEXT_REG: setRegs(kContextRegBase, body, cnt); break;
      case IT_SET_SH_REG:      setRegs(kShRegBase, body, cnt); break;
      case IT_SET_UCONFIG_REG: setRegs(kUConfigRegBase, body, cnt); break;
      case IT_SET_CONFIG_REG:  setRegs(kConfigRegBase, body, cnt); break;
      // AGC LOAD_*_REG (registers loaded from a GPU-memory image, not inline).
      case 0x61: loadRegs(kContextRegBase, body, cnt); break;  // LOAD_CONTEXT_REG
      case 0x5f: loadRegs(kShRegBase, body, cnt); break;       // LOAD_SH_REG
      case 0x5e: loadRegs(kUConfigRegBase, body, cnt); break;  // LOAD_UCONFIG_REG
      // AGC SET_*_REG_INDIRECT (per-draw RT + shaders as (off,val) pairs in a buf).
      case 0x9f: loadRegPairs(kContextRegBase, body, cnt); break;
      case 0x64: loadRegPairs(kUConfigRegBase, body, cnt); break;
      case 0x63: loadRegPairs(kShRegBase, body, cnt); break;  // SET_SH_REG_INDIRECT (shaders)
      // op 0x93 is an INLINE SH-reg set: body[0]=reg_offset (low 16b; high bits are
      // flags/count), body[1..] the values (shader PGM_LO/HI + rsrc). Latch them.
      case 0x93: {
        if (cnt >= 2) {
          uint32_t off = kShRegBase + (body[0] & 0xFFFF);
          for (uint32_t k = 1; k < cnt; k++)
            if (off + (k - 1) < kRegFileSize) g_regs[off + (k - 1)] = body[k];
        }
        break;
      }
      case IT_INDEX_TYPE:
        if (cnt >= 1) g_indexType = body[0] & 0x3;
        break;
      case IT_NUM_INSTANCES:
        if (cnt >= 1) g_numInstances = body[0] ? body[0] : 1;
        break;
      case IT_WRITE_DATA: {  // control, dstLo, dstHi, data...
        if (cnt >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t ndw = cnt - 3;
          if (labelAddrOk(addr) && labelAddrOk(addr + static_cast<uint64_t>(ndw) * 4))
            std::memcpy(reinterpret_cast<void *>(addr), &body[3],
                        static_cast<size_t>(ndw) * 4);
        }
        break;
      }
      case IT_EVENT_WRITE_EOP: {  // eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
        if (cnt >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          uint32_t sel = (body[2] >> 29) & 0x7;
          uint64_t val = static_cast<uint64_t>(body[3]) |
                         (static_cast<uint64_t>(cnt >= 5 ? body[4] : 0) << 32);
          if (sel == 1) writeLabel(addr, val, false);
          else if (sel == 2) writeLabel(addr, val, true);
          else if (sel >= 3) writeLabel(addr, gpuClockTs(), true);
        }
        break;
      }
      case IT_RELEASE_MEM: {  // eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
        if (cnt >= 5) {
          uint32_t sel = (body[1] >> 29) & 0x7;
          uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) |
                          (body[2] & ~0x3u);
          uint64_t val = static_cast<uint64_t>(body[4]) |
                         (static_cast<uint64_t>(cnt >= 6 ? body[5] : 0) << 32);
          if (sel == 1) writeLabel(addr, val, false);
          else if (sel == 2) writeLabel(addr, val, true);
          else if (sel >= 3) writeLabel(addr, gpuClockTs(), true);
        }
        break;
      }
      case IT_EVENT_WRITE_EOS: {  // eventCtrl, addrLo, addrHi+cmd, data
        if (cnt >= 4) {
          uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                          (body[1] & ~0x3u);
          writeLabel(addr, body[3], false);
        }
        break;
      }
      default:
        if (isDraw(op)) handleDraw(op, body, cnt);
        break;
      }
      i += 1 + cnt;
    } else if (type == Pm4Type::type2 || hdr == 0) {
      i += 1;  // filler / alignment
    } else if (type == Pm4Type::type0) {
      // Type-0: write `cnt0` consecutive regs starting at the absolute dword
      // offset in the header. The walker used to SKIP these -- but the AGC driver
      // programs shader PGM_LO/HI (and other SH state) via type-0, which is why no
      // SET_SH_REG carried them. Apply them into the register file.
      uint32_t cnt0 = pm4Count(hdr);
      uint32_t base0 = hdr & 0xFFFF;
      if (i + 1 + cnt0 <= words) {
        for (uint32_t j = 0; j < cnt0; j++)
          if (base0 + j < kRegFileSize) g_regs[base0 + j] = p[i + 1 + j];
        static int s_t0 = 0;
        if (g_trace && s_t0 < 20 && base0 >= kShRegBase && base0 < kShRegBase + 0x300) {
          s_t0++;
          std::fprintf(stderr, "[agc]   type0 SH write base=%#x cnt=%u v0=%08x v1=%08x\n",
                       base0, cnt0, p[i + 1], cnt0 > 1 ? p[i + 2] : 0);
        }
      }
      i += 1 + cnt0;
    } else {
      break;  // type-1 desync
    }
  }
}

}  // namespace

void submitDcb(const void *dcb, uint32_t sizeBytes) {
  if (!dcb || sizeBytes < 4) return;
  std::lock_guard<std::mutex> lk(g_mtx);
  // Bring up the (headless) Vulkan renderer on the first submit so draws render.
  static bool s_vkTried = false;
  if (!s_vkTried) {
    s_vkTried = true;
    vk::init();
  }
  // ONE-SHOT: after some frames, scan the 2MB SceAgcRegShadow (0x8002860000) for
  // non-zero content -- if setShader wrote the shader state to a DIFFERENT shadow
  // buffer than the one LOAD_SH_REG reads, it lives here. Print any non-zero 8-dw
  // block that contains a shader-PGM-like value (top byte 0x80 => addr>>8 in aperture).
  if (g_trace) {
    static uint64_t s_scanAt = 0;
    if (++s_scanAt == 2000) {
      const uint32_t *sh = reinterpret_cast<const uint32_t *>(0x8002860000ull);
      int shown = 0;
      for (uint32_t w = 0; w < (0x200000 / 4) && shown < 16; w += 2) {
        uint32_t lo = sh[w], hi = sh[w + 1];
        uint64_t a = (static_cast<uint64_t>(lo) << 8) | ((uint64_t)(hi & 0xFF) << 40);
        if ((lo >> 24) == 0x80 && a >= 0x8000000000ull && a < 0x8100000000ull) {
          std::fprintf(stderr, "[agc]   SHADOW+%#x lo=%08x hi=%08x -> addr %#lx\n",
                       w * 4, lo, hi, (unsigned long)a);
          shown++;
        }
      }
      if (!shown) std::fprintf(stderr, "[agc]   SHADOW scan: 2MB all-zero (no PGM written)\n");
    }
  }
  uint32_t words = sizeBytes / 4;
  uint64_t sn = ++g_totalSubmits;
  // Skip the all-zero ACQRB ring submits (the 0xC0408121 path is empty for the
  // mode-1 titles); dump the first few submits that actually carry packets.
  const uint32_t *w0 = static_cast<const uint32_t *>(dcb);
  bool nonEmpty = words >= 2 && (w0[0] || w0[1]);
  bool dumpThis = g_trace && g_dumped < 6 && nonEmpty;
  if (dumpThis) {
    g_dumped++;
    const uint32_t *w = static_cast<const uint32_t *>(dcb);
    std::fprintf(stderr, "[agc] === dcb walk #%lu (size=%u words=%u hdr0=%#x) ===\n",
                 static_cast<unsigned long>(sn), sizeBytes, words, w[0]);
    uint32_t rawN = words > 100 ? 100 : words;  // dump big draw buffers fully enough
    std::fprintf(stderr, "[agc]   raw[0..%u]:", rawN);
    for (uint32_t k = 0; k < rawN; k++)
      std::fprintf(stderr, " %08x", w[k]);
    std::fprintf(stderr, "\n");
  }
  walk(static_cast<const uint32_t *>(dcb), words, dumpThis, 0);
  // Periodic global opcode census so we see draw opcodes that only appear in
  // later (undumped) submits -- tells us if the title is issuing draws yet.
  if (g_trace && nonEmpty && (sn % 200) == 0) {
    std::fprintf(stderr, "[agc] === global opcode census @submit %lu ===\n",
                 static_cast<unsigned long>(sn));
    for (int o = 0; o < 256; o++)
      if (g_opHist[o]) std::fprintf(stderr, "[agc]   op %#04x x%u\n", o, g_opHist[o]);
  }
  if (dumpThis) {
    std::fprintf(stderr, "[agc] === dcb walk done; opcode histogram ===\n");
    for (int o = 0; o < 256; o++)
      if (g_opHist[o]) std::fprintf(stderr, "[agc]   op %#04x x%u\n", o, g_opHist[o]);
  }
}

void submitCcb(const void *ccb, uint32_t sizeBytes) {
  if (!ccb || sizeBytes < 4) return;
  std::lock_guard<std::mutex> lk(g_mtx);
  walk(static_cast<const uint32_t *>(ccb), sizeBytes / 4, false, 0);
}

void endFrame(uint64_t scanoutBase) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_frameActive && vk::available()) {
    vk::endFrame(scanoutBase);
    g_frameActive = false;
  }
}

}  // namespace gpu::ps5

// LLE submit bridge: the kernel /dev/gc AGC ioctls (gc_dev.cpp) forward the DCB
// here, mirroring prosperity_gc_submit on the PS4 path.
extern "C" void prosperity_agc_submit(uint64_t dcbBase, uint32_t sizeBytes) {
  gpu::ps5::submitDcb(reinterpret_cast<const void *>(dcbBase), sizeBytes);
}

// PS5 flip bridge: the shared dce/VideoOut flip path calls this when the active
// process is PS5, so the frame the AGC submit rendered is read back + presented
// through vk::endFrame (mirrors prosperity_gc_flip on the PS4 path).
extern "C" void prosperity_agc_flip(uint64_t scanoutBase) {
  gpu::ps5::endFrame(scanoutBase);
}
