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
#include <atomic>
#include <thread>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

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
// gfx10.3 buffer V#s carry a UNIFIED 7-bit FORMAT enum (word3 [18:12]), not GCN's
// separate data/number format. Map the enum to the GCN (dfmt,nfmt) pair vk_render's
// vfmt() understands. Only the vertex-attribute formats are covered; unknowns fall
// back to the GCN-style split (harmless for descriptors that never reach vfmt()).
void gfx10VBufFormat(uint32_t gfmt, uint32_t &dfmt, uint32_t &nfmt) {
  // The gfx10.3 unified buffer-format enum (ISA spec "Buffer Format
  // Conversions"): consecutive runs of one channel layout in the order
  // UNorm, SNorm, UScaled, SScaled, UInt, SInt [, Float]. Map each run onto the
  // GCN (dfmt, nfmt) pair the shared renderer speaks. The old code only listed
  // the float formats and split the rest as GCN bitfields, which turned
  // Skyrim's 16_16_SScaled UI positions (26) into 8_8_8_8_SNorm and collapsed
  // every glyph quad to a degenerate triangle.
  struct Run { uint8_t first, count, dfmt; bool hasFloat; };
  static constexpr Run kRuns[] = {
      { 1, 6,  1, false},  // 8
      { 7, 7,  2, true },  // 16
      {14, 6,  3, false},  // 8_8
      {20, 3,  4, true },  // 32 (UInt, SInt, Float)
      {23, 7,  5, true },  // 16_16
      {30, 7,  7, true },  // 11_11_10
      {37, 7,  6, true },  // 10_11_11
      {44, 6,  9, false},  // 2_10_10_10
      {50, 6,  8, false},  // 10_10_10_2
      {56, 6, 10, false},  // 8_8_8_8
      {62, 3, 11, true },  // 32_32
      {65, 7, 12, true },  // 16_16_16_16
      {72, 3, 13, true },  // 32_32_32
      {75, 3, 14, true },  // 32_32_32_32
  };
  static constexpr uint8_t kNfmt[6] = {0, 1, 2, 3, 4, 5};
  for (const Run &r : kRuns) {
    if (gfmt < r.first || gfmt >= r.first + r.count) continue;
    const uint32_t i = gfmt - r.first;
    dfmt = r.dfmt;
    if (r.count == 3)              // UInt, SInt, Float
      nfmt = i == 0 ? 4u : i == 1 ? 5u : 7u;
    else if (r.hasFloat && i == 6) // trailing Float of a 7-wide run
      nfmt = 7;
    else
      nfmt = kNfmt[i];
    return;
  }
  dfmt = 0;
  nfmt = 0;
}
VBuffer decodeVBuffer(const uint32_t *p) {
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.numRecords = p[2];
  v.gfmt = (p[3] >> 12) & 0x7F;
  gfx10VBufFormat(v.gfmt, v.dfmt, v.nfmt);
  return v;
}
// A V# read from an unbound/garbage SGPR slot decodes to an in-range but bogus
// address with an implausible stride/record count (e.g. a depth-only pre-pass
// with an inactive vertex slot decoded stride=14915 nrec=480622080, which then
// segfaulted reading the vertex ring). Mirrors the PS4 fetch-shader sanity gate
// (gpu/ps4/gcn/gcn_resource.cpp).
bool plausibleVb(const VBuffer &v) {
  return v.stride && v.stride <= 256 && v.numRecords && v.numRecords <= 0x100000;
}

// Registers per space, so a LOAD_*_REG range can never spill into the next one.
// A LOAD_SH_REG whose range ran long wrote zeros from its (empty) shadow image
// straight over CB_COLOR0_BASE at 0xA318, unbinding the render target that
// op 0x49 had just set -- every colour draw after it hit no target and the
// frame came out black.
constexpr uint32_t kRegSpaceSize = 0x400;

inline uint32_t regSpaceLimit(uint32_t base) {
  return base + kRegSpaceSize <= kRegFileSize ? base + kRegSpaceSize
                                              : kRegFileSize;
}

// Latch a SET_*_REG packet into the register file. body[0] is the register
// offset dword (with the gfx10 selector bits stripped), body[1..] the values.
// DELTA_AGC_UDTRACE: every write to the PS user-data SGPRs above 15. Skyrim's
// grading pass reads a texture descriptor from s16..s23, and if nothing in the
// command stream programs those the shader samples whatever was left there.
static void noteUdWrite(const char *how, uint32_t reg, uint32_t val) {
  static const bool on = std::getenv("DELTA_AGC_UDTRACE") != nullptr;
  static int n = 0;
  if (on && n < 40 && val && reg >= mmSPI_SHADER_USER_DATA_PS_0 + 16 &&
      reg < mmSPI_SHADER_USER_DATA_PS_0 + 32) {
    n++;
    std::fprintf(stderr, "[agc] PS ud%u <- %08x by %s\n",
                 reg - mmSPI_SHADER_USER_DATA_PS_0, val, how);
  }
}

// DELTA_AGC_CLIPTRACE: name every packet that writes PA_CL_CLIP_CNTL. The
// clip-space convention lives there, and a title whose writes never reach us
// silently gets the reset value (OpenGL clip space).
static void noteClipWrite(const char *how, uint32_t val) {
  static const bool on = std::getenv("DELTA_AGC_CLIPTRACE") != nullptr;
  static int n = 0;
  if (on && n++ < 12)
    std::fprintf(stderr, "[agc] CLIP_CNTL <- %#x by %s\n", val, how);
}

// Draw accounting. DELTA_GPU_DRAWRT only sees draws that reach the renderer, so
// a draw dropped earlier (no usable render target, unrecoverable shader address)
// is indistinguishable from one the title never issued. Count both ends.
static std::atomic<uint64_t> g_drawsSeen{0}, g_drawsIssued{0}, g_dropNoRt{0},
    g_dropNoShader{0};

void setRegs(uint32_t base, const uint32_t *body, uint32_t count) {
  if (count < 1) return;
  uint32_t off = base + (body[0] & ~kRegSelectorMask);
  const uint32_t limit = regSpaceLimit(base);
  for (uint32_t i = 1; i < count; i++) {
    if (off + (i - 1) < limit) g_regs[off + (i - 1)] = body[i];
    if (off + (i - 1) == mmPA_CL_CLIP_CNTL) noteClipWrite("SET_REG", body[i]);
    noteUdWrite("SET_REG", off + (i - 1), body[i]);
  }
}

// A guest GPU address. Isaac's AGC pool sits in the 0x80_xx_xx_xx_xx band, but a
// title that batch-maps its direct memory gets whatever the kernel handed it --
// Skyrim's command buffers and labels land around 0x10_00_00_00_00. Test the
// whole range the guest allocator can hand out instead of one title's band.
inline bool gpuAddr(uint64_t a) {
  return a >= 0x1000000000ull && a < 0x8100000000ull;
}

// inGuest()'s bound spans essentially the whole 48-bit VA (see above), so a
// heuristically-recovered pointer (e.g. reinterpreting a V#'s raw SGPR bits as a
// table root when the inline descriptor doesn't decode) can pass it while still
// pointing at memory the guest never mapped -- an inactive/uninitialized vertex
// slot did exactly this and segfaulted the host. Probe with mincore (same
// technique as kern/crash.cpp's guest-pointer walks) before dereferencing.
inline bool mapped(uint64_t va, uint64_t bytes) {
  const long pg = sysconf(_SC_PAGESIZE);
  const uint64_t start = va & ~static_cast<uint64_t>(pg - 1);
  const uint64_t end = (va + bytes + pg - 1) & ~static_cast<uint64_t>(pg - 1);
  for (uint64_t p = start; p < end; p += static_cast<uint64_t>(pg)) {
    unsigned char vec = 0;
    if (mincore(reinterpret_cast<void *>(p), 1, &vec) != 0) return false;
  }
  return true;
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
  if (!gpuAddr(mem)) return;  // GPU aperture only
  const uint32_t *src = reinterpret_cast<const uint32_t *>(mem);
  // COHERENCY TEST: dump the LOAD image content. If a context/SH image reads all
  // zero, it's non-coherent (the driver wrote it via a different VA) -- the shader
  // would be present but invisible. If it has real reg values, the image is fine.
  // Coherency census: a title that drives its whole context through register
  // shadows is invisible if those images read zero. Sample one late.
  static int s_imgN = 0;
  if (base == kContextRegBase) s_imgN++;
  if (g_trace && base == kContextRegBase && s_imgN >= 100 && s_imgN <= 106) {
    uint32_t nz = 0, first = 0;
    for (uint32_t j = 0; j < 0x400; j++)
      if (src[j]) { nz++; if (!first) first = j; }
    std::fprintf(stderr,
                 "[agc] IMGCENSUS #%d base=%#x mem=%#lx nonzero=%u/1024 first=%#x "
                 "img[0x318]=%08x img[0x31c]=%08x ranges=%u\n",
                 s_imgN, base, (unsigned long)mem, nz, first, src[0x318],
                 src[0x31c], (count - 2) / 2);
  }
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
        if (gpuAddr(a)) {
          const uint32_t *w = reinterpret_cast<const uint32_t *>(a);
          std::fprintf(stderr, "[agc]   img[%#x]=%08x -> %#lx ISA? %08x %08x\n",
                       j, v, (unsigned long)a, w[0], w[1]);
        }
      }
    }
  }
  const uint32_t limit = regSpaceLimit(base);
  // LOAD_*_REG restores a register shadow the command processor is supposed to
  // have SAVED into. We do not model the save side, so a shadow the title never
  // wrote reads as all zeros -- and applying it wipes live state: Skyrim binds
  // CB_COLOR0_BASE with SET_CONTEXT_REG_INDIRECT (0x9f) and the very next
  // LOAD_CONTEXT_REG zeroed it again, leaving every colour draw with no render
  // target. An all-zero shadow carries nothing to restore, so skip it.
  {
    bool anyValue = false;
    for (uint32_t i = 2; i + 1 < count && !anyValue; i += 2) {
      uint32_t off = body[i] & 0xFFFF;
      uint32_t num = body[i + 1] & 0xFFFF;
      if (num > 0x2000) num = 0x2000;
      for (uint32_t j = 0; j < num; j++)
        if (base + off + j < limit && src[off + j]) { anyValue = true; break; }
    }
    if (!anyValue) return;
  }
  for (uint32_t i = 2; i + 1 < count; i += 2) {
    uint32_t off = body[i] & 0xFFFF;
    uint32_t num = body[i + 1] & 0xFFFF;
    if (num > 0x2000) num = 0x2000;  // sanity cap
    for (uint32_t j = 0; j < num; j++) {
      if (base + off + j >= limit) break;
      uint32_t v = src[off + j];
      g_regs[base + off + j] = v;
      if (base + off + j == mmPA_CL_CLIP_CNTL) noteClipWrite("LOAD_REG", v);
      noteUdWrite("LOAD_REG", base + off + j, v);
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
  if (!gpuAddr(addr)) return;
  // body[3] is the count of (reg_offset, value) register PAIRS, not dwords: each
  // iteration reads two dwords (the gfx10.3 SET_*_REG_INDIRECT handlers
  // loop `i < (buffer[3] & 0x3fff)` advancing the pointer by 2). The old
  // dword-count reading wrote nothing for a single-register indirect (num == 1),
  // so VGT_PRIMITIVE_TYPE (set this way) never landed and prim assembly died.
  uint32_t numPairs = body[3] & 0x3FFF;
  const uint32_t *p = reinterpret_cast<const uint32_t *>(addr);
  const uint32_t limit = regSpaceLimit(base);
  for (uint32_t i = 0; i < numPairs; i++) {
    uint32_t off = p[i * 2] & ~kRegSelectorMask;  // strip gfx10 selector bits
    if (base + off < limit) g_regs[base + off] = p[i * 2 + 1];
    if (base + off == mmPA_CL_CLIP_CNTL) noteClipWrite("SET_REG_INDIRECT", p[i * 2 + 1]);
    noteUdWrite("SET_REG_INDIRECT", base + off, p[i * 2 + 1]);
    // DELTA_AGC_CBTRACE: every write to the colour-target base/format, with the
    // packet's own bookkeeping, so a zero write can be traced to a mis-parse.
    static const int cbTraceFrom = [] {
      const char *e = std::getenv("DELTA_AGC_CBTRACE");
      return e ? std::atoi(e) : -1;
    }();
    static int cbN = 0;
    const bool cbTrace = cbTraceFrom >= 0 &&
                         (int)g_drawsSeen.load(std::memory_order_relaxed) >= cbTraceFrom;
    if (cbTrace && cbN < 60 &&
        (base + off == mmCB_COLOR0_BASE || base + off == mmCB_COLOR0_INFO)) {
      cbN++;
      std::fprintf(stderr,
                   "[agc] CB0 %s <- %08x  (pair %u/%u from %#lx, raw off %08x)\n",
                   base + off == mmCB_COLOR0_BASE ? "BASE" : "INFO", p[i * 2 + 1],
                   i, numPairs, (unsigned long)addr, p[i * 2]);
    }
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
  bool glClip;  // clip convention is baked into the VS (see PA_CL_CLIP_CNTL)
  bool operator==(const ShaderKey &o) const {
    return vs == o.vs && ps == o.ps && fetch == o.fetch && glClip == o.glClip;
  }
};
struct ShaderKeyHash {
  size_t operator()(const ShaderKey &k) const {
    return std::hash<uint64_t>{}(k.vs) ^ (std::hash<uint64_t>{}(k.ps) << 1) ^
           (std::hash<uint64_t>{}(k.fetch) << 2) ^ (k.glClip ? 0x9e3779b9u : 0u);
  }
};
std::unordered_map<ShaderKey, gcn::Recompiled, ShaderKeyHash> g_shCache;

// IT_DISPATCH_DIRECT: body = [dim_x, dim_y, dim_z, initiator] workgroup counts.
// The CS program address, workgroup shape, RSRC and user data come from the
// COMPUTE_* SH registers programmed before it.
const char *const kEncName[17] = {"?",    "sop1", "sop2",  "sopk", "sopc",
                                  "sopp", "smem", "vop1",  "vop2", "vop3",
                                  "vopc", "vint", "ds",    "mubuf/flat",
                                  "mtbuf", "mimg", "exp"};

void handleDispatch(const uint32_t *body, uint32_t count) {
  const uint32_t dimX = count >= 1 ? body[0] : 0;
  const uint32_t dimY = count >= 2 ? body[1] : 0;
  const uint32_t dimZ = count >= 3 ? body[2] : 0;
  const uint64_t csAddr =
      (static_cast<uint64_t>(g_regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
       g_regs[mmCOMPUTE_PGM_LO])
      << 8;
  const uint32_t tgx = g_regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF;
  const uint32_t tgy = g_regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF;
  const uint32_t tgz = g_regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF;
  const uint32_t rsrc2 = g_regs[mmCOMPUTE_PGM_RSRC2];

  static const bool csDump = std::getenv("DELTA_GPU_CSDUMP") != nullptr;
  static std::unordered_set<uint64_t> dumpedCs;
  if (csDump) {
    static uint64_t nTotal = 0, nValid = 0;
    static std::unordered_set<uint64_t> seen;
    nTotal++;
    if (inGuest(csAddr)) nValid++;
    seen.insert(csAddr);
    if ((nTotal % 2000) == 0) {
      std::fprintf(stderr, "[cs] dispatches=%lu valid=%lu unique=%zu:",
                   (unsigned long)nTotal, (unsigned long)nValid, seen.size());
      int shown = 0;
      for (uint64_t a : seen) {
        if (shown++ >= 8) break;
        std::fprintf(stderr, " %#lx", (unsigned long)a);
      }
      std::fprintf(stderr, " rsrc2=%08x tg=[%u %u %u]\n", rsrc2, tgx, tgy, tgz);
    }
  }
  if (csDump && dumpedCs.size() < 24 && inGuest(csAddr) &&
      dumpedCs.insert(csAddr).second) {
    const uint32_t *ud = &g_regs[mmCOMPUTE_USER_DATA_0];
    std::fprintf(stderr,
                 "[cs] addr=%#lx groups=[%u %u %u] tg=[%u %u %u] rsrc2=%08x\n",
                 (unsigned long)csAddr, dimX, dimY, dimZ, tgx, tgy, tgz, rsrc2);
    std::fprintf(stderr, "[cs]   user_data:");
    for (int k = 0; k < 16; k++) std::fprintf(stderr, " %08x", ud[k]);
    std::fprintf(stderr, "\n");
    // Follow each user-data pointer pair one level: the descriptor tables the CS
    // dereferences say which surfaces it actually reads and writes.
    for (int k = 0; k < 15; k++) {
      uint64_t p = (static_cast<uint64_t>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
      if (!gpuAddr(p)) continue;
      const uint32_t *tw = reinterpret_cast<const uint32_t *>(p);
      std::fprintf(stderr, "[cs]   ud%d -> %#lx:", k, (unsigned long)p);
      for (int b = 0; b < 8; b++) std::fprintf(stderr, " %08x", tw[b]);
      std::fprintf(stderr, "\n");
    }
    // Encoding census: says which instruction families a compute backend must
    // cover before any of these dispatches can run.
    const auto prog =
        rdna::DecodeShader(reinterpret_cast<const uint32_t *>(csAddr), 4096);
    uint32_t hist[24] = {};
    uint32_t flatSeg[4] = {}, flatOps[128] = {};
    for (const auto &in : prog) {
      const uint32_t e = static_cast<uint32_t>(in.enc);
      if (e < 24) hist[e]++;
      if (in.enc == gcn::Enc::kMubuf && (in.raw[0] >> 26) == 0x37) {
        flatSeg[(in.raw[0] >> 14) & 3]++;
        flatOps[(in.raw[0] >> 18) & 0x7F]++;
      }
    }
    std::fprintf(stderr, "[cs]   insts=%zu enc:", prog.size());
    for (uint32_t e = 0; e < 24; e++)
      if (hist[e]) std::fprintf(stderr, " %u=%u", e, hist[e]);
    std::fprintf(stderr, " flatseg: %u/%u/%u/%u ops:", flatSeg[0], flatSeg[1],
                 flatSeg[2], flatSeg[3]);
    for (uint32_t o = 0; o < 128; o++)
      if (flatOps[o]) std::fprintf(stderr, " %#x=%u", o, flatOps[o]);
    std::fprintf(stderr, "\n");
  }
}

// DELTA_AGC_DUMPSH=<hexaddr>: decode and print one shader by address, once.
// Shader dumps are otherwise tied to recompile time or to a draw index, neither
// of which is reachable for a steady-state pass without a full trace.
void maybeDumpShader(uint64_t addr) {
  static const uint64_t want = [] {
    const char *e = std::getenv("DELTA_AGC_DUMPSH");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  static bool done = false;
  if (!want || done || addr != want || !inGuest(addr)) return;
  done = true;
  const auto prog = rdna::DecodeShader(reinterpret_cast<const uint32_t *>(addr), 4096);
  std::fprintf(stderr, "[agc] SHADER %#lx: %zu insts\n", (unsigned long)addr,
               prog.size());
  for (const auto &in : prog) {
    std::fprintf(stderr, "[agc]  pc=%04x %-6s op=%#05x %08x", in.pc,
                 kEncName[static_cast<uint32_t>(in.enc) < 17
                              ? static_cast<uint32_t>(in.enc)
                              : 0],
                 in.opcode, in.raw[0]);
    if (in.size >= 2) std::fprintf(stderr, " %08x", in.raw[1]);
    if (in.has_literal) std::fprintf(stderr, " lit=%08x", in.literal);
    std::fprintf(stderr, "\n");
  }
}


// The T#'s four DST_SEL channel selects, packed 3 bits each for the renderer's
// image view. The identity selection (R,G,B,A) packs to 0 so views that need no
// swizzle keep sharing one cache entry.
static uint32_t packDstSel(const gcn::TImage &t) {
  const uint32_t p = (t.dst_sel[0] & 7) | ((t.dst_sel[1] & 7) << 3) |
                     ((t.dst_sel[2] & 7) << 6) | ((t.dst_sel[3] & 7) << 9);
  return p == (4u | (5u << 3) | (6u << 6) | (7u << 9)) ? 0u : p;
}

// A context register read as the float it holds (viewport scales/offsets).
static float regF(uint32_t reg) {
  float f;
  std::memcpy(&f, &g_regs[reg], 4);
  return f;
}

// Last packet that changed CB_COLOR0_BASE (see DELTA_AGC_RTPROBE).
static uint32_t g_cb0Op = 0, g_cb0Val = 0;
static uint64_t g_cb0Draw = 0;

static void drawCensus() {
  static const bool on = std::getenv("DELTA_GPU_DRAWCENSUS") != nullptr;
  if (!on)
    return;
  static const bool started = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::fprintf(stderr,
                     "[drawcensus] seen=%llu issued=%llu dropped: no-rt=%llu "
                     "no-shader=%llu\n",
                     (unsigned long long)g_drawsSeen.load(),
                     (unsigned long long)g_drawsIssued.load(),
                     (unsigned long long)g_dropNoRt.load(),
                     (unsigned long long)g_dropNoShader.load());
      }
    }).detach();
    return true;
  }();
  (void)started;
}

void handleDraw(uint32_t op, const uint32_t *body, uint32_t count) {
  size_t dropAttrCount = 0;
  g_drawsSeen.fetch_add(1, std::memory_order_relaxed);
  drawCensus();
  if (!vk::available()) return;

  // gfx10.3 has no HW VS: the vertex program is the merged NGG shader, whose
  // address is written to the ES (front half, 0xC8) and/or GS (back half, 0x88)
  // PGM_LO. Some pipelines populate only the ES slot, so fall back to it when the
  // GS slot reads 0. User data (cbuffer/MVP pointers) stays in the GS block.
  uint64_t vsA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_GS);
  if (!inGuest(vsA)) vsA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_ES);
  uint64_t psA = g_regs.shaderAddr(mmSPI_SHADER_PGM_LO_PS);
  maybeDumpShader(vsA);
  maybeDumpShader(psA);
  // DELTA_PS5_SKIPVS=hexaddr: drop draws using this VS (draw-isolation bisect).
  static const uint64_t skipVs = [] {
    const char *e = std::getenv("DELTA_PS5_SKIPVS");
    return e ? std::strtoull(e, nullptr, 16) : 0ull;
  }();
  if (skipVs && vsA == skipVs) return;
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
      if (nf >= 16 || !gpuAddr(a) || (a & 0xFF))
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
    if (g_trace && s_hdr < 3 && gpuAddr(a113)) {
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
          if (gpuAddr(p)) {
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
    // A draw whose target mask asks for colour but whose CB_COLOR0_INFO reads
    // zero keeps the last target that was valid. Skyrim's logo and menu passes
    // are bound through a path we do not see yet (the driver's default-state
    // block zeroes CB_COLOR0 and only some passes get a re-bind), and dropping
    // them entirely is certainly wrong where reusing the target is only maybe.
    // DELTA_GPU_NOSTICKYRT restores the drop.
    static const bool stickyRt = std::getenv("DELTA_GPU_NOSTICKYRT") == nullptr;
    static uint64_t lastBase = 0;
    static uint32_t lastInfo = 0, lastW = 0, lastH = 0;
    if (d.mrtCount) {
      lastBase = d.mrtBase[0];
      lastInfo = d.mrtInfo[0];
      lastW = d.rtW;
      lastH = d.rtH;
    } else if (stickyRt && (g_regs[mmCB_TARGET_MASK] & 0xF) && lastBase &&
               d.rtW == lastW && d.rtH == lastH) {
      d.mrtBase[0] = lastBase;
      d.mrtInfo[0] = lastInfo;
      d.mrtCount = 1;
      d.rtBase = lastBase;
    }
    // DELTA_AGC_RTPROBE: one line per draw naming the packet that last touched
    // CB_COLOR0_BASE. A draw with no target and a stale "last write" points at a
    // bind we never executed; one whose last write zeroed the base points at a
    // packet we execute but should not.
    static const bool rtProbe = std::getenv("DELTA_AGC_RTPROBE") != nullptr;
    static int s_probe = 0;
    if (rtProbe && s_probe < 200) {
      s_probe++;
      std::fprintf(stderr,
                   "[agc] RTPROBE draw#%lu rt=%#lx tmask=%#x info0=%#x clip=%#x "
                   "blend=%u ctl=%#x cc=%#x zscale=%g zoff=%g\n",
                   (unsigned long)g_drawsSeen.load(), (unsigned long)d.rtBase,
                   g_regs[mmCB_TARGET_MASK], g_regs[mmCB_COLOR0_INFO],
                   g_regs[mmPA_CL_CLIP_CNTL], (g_regs[mmCB_BLEND0_CONTROL] >> 30) & 1,
                   g_regs[mmCB_BLEND0_CONTROL], g_regs[mmCB_COLOR_CONTROL],
                   regF(mmPA_CL_VPORT_ZSCALE),
                   regF(mmPA_CL_VPORT_ZOFFSET));
    }
    // Why a colour draw ended up with no target: print the state that rejected
    // CB_COLOR0 (a stale/zero base, or an INFO with no format).
    static int s_nort = 0;
    if (g_trace && !d.mrtCount && (tmask & 0xF) && s_nort < 12) {
      s_nort++;
      std::fprintf(stderr,
                   "[agc]   NO-RT tmask=%#x cb0Base=%#lx info0=%#x fmt=%u\n",
                   tmask, (unsigned long)g_regs.cbColorBase(0),
                   g_regs[mmCB_COLOR0_INFO],
                   (g_regs[mmCB_COLOR0_INFO] >> 2) & 0x1F);
    }
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

  // Depth/stencil (gfx10 Z base = (WRITE_BASE | WRITE_BASE_HI<<32) << 8). Mirrors
  // the PS4 path (gpu/ps4/cmd_processor.cpp): DELTA_GPU_NODEPTH is the shared kill
  // switch, otherwise the title's own DB_Z_INFO/DB_DEPTH_CONTROL state decides. 2D
  // titles (Isaac) leave DB_Z_INFO's format field invalid so depthValid stays false
  // and no depth attachment binds (unchanged 2D path).
  {
    static const bool noDepth = std::getenv("DELTA_GPU_NODEPTH") != nullptr;
    uint32_t dc = g_regs[mmDB_DEPTH_CONTROL];
    uint32_t zinfo = noDepth ? 0 : g_regs[mmDB_Z_INFO];
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
    // DX_CLIP_SPACE_DEF (bit 19) picks the guest's clip-z convention.
    const bool glClip = !((g_regs[mmPA_CL_CLIP_CNTL] >> 19) & 1);
    ShaderKey key{vsA, psA, fetch, glClip};
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
        std::fprintf(stderr, "[agc] DL psInputEna=%#x (frag-coord/face VGPR seed)\n",
                     g_regs[mmSPI_PS_INPUT_ENA]);
      }
      it = g_shCache
               .emplace(key, rdna::Recompile(reinterpret_cast<const uint32_t *>(vsA),
                                             psA ? reinterpret_cast<const uint32_t *>(psA)
                                                 : nullptr,
                                             vud, pud, g_regs[mmSPI_PS_INPUT_ENA],
                                             glClip))
               .first;
      if (dl) std::fprintf(stderr, "[agc] DL recompile done ok=%d\n", it->second.ok);
      // A shader we cannot recompile drops its draw entirely, which is
      // indistinguishable from "the title never issued it" unless it is said out
      // loud. Report each failing pair once.
      if (!it->second.ok) {
        static int failed = 0;
        if (failed++ < 32)
          std::fprintf(stderr,
                       "[agc] recompile FAILED vs=%#lx ps=%#lx -- draw dropped\n",
                       (unsigned long)vsA, (unsigned long)psA);
      }
    }
    gcn::Recompiled &rc = it->second;
    if (rc.ok) {
      // Vertex attributes: the V# is either inline in user data at table_sgpr
      // (AGC frequently passes the vertex V# directly) or reached through a table
      // pointer at {table_sgpr, +1}. Resolve each attr's V#, then group the
      // attrs into vertex bindings: same-stride V#s within one stride of each
      // other interleave in one binding; others get their own binding (the
      // textured sprite VS streams pos/color and uv/params from two buffers, so
      // a single interleaved binding would feed the PS garbage UVs). Attrs that
      // don't decode to a valid V# are skipped (a partial fetch still rasterizes).
      VBuffer attrVbs[8];
      const gcn::ShaderAttr *attrRes[8];
      uint32_t attrN = 0;
      // The merged ES/GS NGG vertex shader reads its GS user data starting at wave
      // SGPR udBase (sgpr 0..udBase-1 are ES/system), but the AGC latches it into
      // SPI_SHADER_USER_DATA_GS_0 which we index from 0, so shift shader SGPR N ->
      // userData[N-udBase] for both attrs and cbufs. Defaults to 8 (the observed
      // merged-NGG layout); DELTA_PS5_UDBASE overrides. TODO: derive from RSRC2.
      static const uint32_t udBaseEnv = [] {
        const char *e = std::getenv("DELTA_PS5_UDBASE");
        return e ? static_cast<uint32_t>(std::atoi(e)) : 8u;
      }();
      if (dl)
        std::fprintf(stderr, "[agc] DL attrs=%zu vud[0..7]=%08x %08x %08x %08x %08x %08x %08x %08x\n",
                     rc.attrs.size(), vud[0], vud[1], vud[2], vud[3], vud[4], vud[5], vud[6], vud[7]);
      for (size_t i = 0; i < rc.attrs.size() && i < 8; i++) {
        auto &a = rc.attrs[i];
        const uint32_t ti =
            a.table_sgpr >= udBaseEnv ? a.table_sgpr - udBaseEnv : a.table_sgpr;
        if (ti + 3 >= 32) continue;
        VBuffer vb = decodeVBuffer(&vud[ti]);  // inline V#
        const char *how = "inline";
        if (!inGuest(vb.base) || !plausibleVb(vb)) {  // else follow a table pointer
          uint64_t tbl = (static_cast<uint64_t>(vud[ti + 1] & 0xFFFF) << 32) |
                         vud[ti];
          uint64_t tblAddr = tbl + a.vbuf_dword_off * 4;
          if (inGuest(tbl) && mapped(tblAddr, 16)) {
            vb = decodeVBuffer(reinterpret_cast<const uint32_t *>(tblAddr));
            how = "table";
          }
        }
        if (dl)
          std::fprintf(stderr, "[agc]   attr%zu loc=%u nc=%u tbl_sgpr=%u off=%u (%s) -> "
                       "base=%#lx stride=%u nrec=%u gfmt=%u -> dfmt=%u nfmt=%u\n",
                       i, a.location, a.num_comps, a.table_sgpr, a.vbuf_dword_off, how,
                       (unsigned long)vb.base, vb.stride, vb.numRecords, vb.gfmt,
                       vb.dfmt, vb.nfmt);
        if (!inGuest(vb.base) || !plausibleVb(vb)) continue;  // unresolved: keep the rest
        attrVbs[attrN] = vb;
        attrRes[attrN++] = &a;
      }
      // Group the resolved attrs into bindings (mirrors the PS4 recomp path).
      uint32_t attrBinding[8] = {};
      for (uint32_t i = 0; i < attrN; i++) {
        const VBuffer &vb = attrVbs[i];
        int sel = -1;
        for (uint32_t j = 0; j < d.nvbufs; j++) {
          if (d.vbufs[j].stride != vb.stride) continue;
          uint64_t b = reinterpret_cast<uint64_t>(d.vbufs[j].data);
          uint64_t lo = b < vb.base ? b : vb.base;
          uint64_t hi = b < vb.base ? vb.base : b;
          if (hi - lo < vb.stride) { sel = static_cast<int>(j); break; }
        }
        if (sel < 0) {
          if (d.nvbufs >= 8) break;
          sel = static_cast<int>(d.nvbufs);
          d.vbufs[d.nvbufs++] = {reinterpret_cast<const void *>(vb.base),
                                 vb.stride, vb.numRecords};
        } else {
          auto &bind = d.vbufs[sel];
          if (vb.base < reinterpret_cast<uint64_t>(bind.data))
            bind.data = reinterpret_cast<const void *>(vb.base);
          bind.numRecords = std::min(bind.numRecords, vb.numRecords);
        }
        attrBinding[i] = static_cast<uint32_t>(sel);
      }
      // Offsets are relative to each binding's final (lowest) base.
      for (uint32_t i = 0; i < attrN && d.nvattrs < 8; i++) {
        const VBuffer &vb = attrVbs[i];
        const uint32_t b = attrBinding[i];
        if (b >= d.nvbufs) continue;
        const uint64_t off = vb.base - reinterpret_cast<uint64_t>(d.vbufs[b].data);
        if (off >= d.vbufs[b].stride) continue;
        // A typed fetch (tbuffer_load_format_*) carries its own format and the
        // hardware ignores the V#'s; Skyrim's world positions come in that way.
        uint32_t dfmt = vb.dfmt, nfmt = vb.nfmt;
        if (attrRes[i]->inst_format)
          gfx10VBufFormat(attrRes[i]->inst_format, dfmt, nfmt);
        d.vattrs[d.nvattrs++] = {attrRes[i]->location, b, static_cast<uint32_t>(off),
                                 attrRes[i]->num_comps, dfmt, nfmt};
      }
      if (d.nvbufs) {
        // Mirror binding 0 into the legacy single-stream fields; the vertex
        // count is bounded by the smallest binding's record count.
        d.vertexData = d.vbufs[0].data;
        d.vertexStride = d.vbufs[0].stride;
        uint32_t count = UINT32_MAX;
        for (uint32_t j = 0; j < d.nvbufs; j++)
          count = std::min(count, d.vbufs[j].numRecords);
        d.vertexCount = count;
      }
      // A fetch VS needs at least one attribute resolved; a procedural VS (no
      // recovered attrs, seeds from VertexIndex) draws without a vertex buffer.
      // A fetch VS needs at least one attribute resolved; a procedural VS (no
      // recovered attrs, seeds from VertexIndex) draws without a vertex buffer.
      // A fullscreen pass binds NO vertex buffer at all even though its shader
      // contains fetch instructions (the same shader is used both ways), so
      // demanding a resolved attribute there discarded the whole pass. Let it
      // through with no vertex inputs declared instead.
      bool good = d.nvattrs > 0 || rc.attrs.empty();
      if (!good && !d.nvbufs) {
        d.nvattrs = 0;
        good = true;
      }
      dropAttrCount = rc.attrs.size();

      // Constant buffers: resolve each SMEM descriptor. For a direct cbuf the V# is
      // inline at ud_sgpr in user data; for a chained cbuf the descriptor is reached
      // by reading the root pointer from user data and dereferencing through guest
      // memory per chain_off[] (the VS loads its transform's V# from a root
      // descriptor table this way). The bind size is the recompiler's planned dword
      // window (the V# stride/records are meaningless for an s_load pointer).
      auto resolveCbufs = [&](const std::vector<gcn::ShaderCbuf> &cbufs,
                              const uint32_t *userData, bool vertexStage) {
        const uint32_t udBase = vertexStage ? udBaseEnv : 0;
        for (const auto &cb : cbufs) {
          if (cb.binding >= gpu::gcn::kMaxCbufBindings) continue;
          VBuffer vb{};
          const char *how = "direct";
          const uint32_t udi = cb.ud_sgpr >= udBase ? cb.ud_sgpr - udBase : cb.ud_sgpr;
          if (cb.chain_len == 0) {
            if (udi + 3 >= 32) continue;
            vb = decodeVBuffer(&userData[udi]);
          } else {  // walk the s_load pointer chain to the final V#
            if (udi + 1 >= 32) continue;
            uint64_t ptr = (static_cast<uint64_t>(userData[udi + 1] & 0xFFFF) << 32) |
                           userData[udi];
            bool ok = true;
            for (uint32_t i = 0; i + 1 < cb.chain_len; i++) {  // deref intermediate pointers
              uint64_t at = (ptr + cb.chain_off[i]) & ~uint64_t{3};
              if (!inGuest(at) || !mapped(at, 8)) { ok = false; break; }
              const uint32_t *q = reinterpret_cast<const uint32_t *>(at);
              ptr = (static_cast<uint64_t>(q[1] & 0xFFFF) << 32) | q[0];
            }
            uint64_t vAddr = (ptr + cb.chain_off[cb.chain_len - 1]) & ~uint64_t{3};
            if (!ok || !inGuest(vAddr) || !mapped(vAddr, 16)) continue;
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
          // USER_SGPR[5:1] with USER_SGPR_MSB[27] as bit 5: how many user-data
          // SGPRs this PS was launched with.
          const uint32_t rsrc2 = g_regs[mmSPI_SHADER_PGM_RSRC2_PS];
          // The hardware window is what RSRC2 declares. Skyrim's grading pass
          // declares 32 and keeps a T# at s16 -- resolving it is correct, but the
          // texture it names (a 3840x2160 R16G16F target) then tints the whole
          // frame magenta, so something downstream of that sample is still wrong
          // and the conservative 16-dword window renders better today.
          // DELTA_GPU_UDBOUND=0 uses the declared window, or pins any value.
          const uint32_t declared =
              ((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5);
          static const int udBound = [] {
            const char *e = std::getenv("DELTA_GPU_UDBOUND"); return e ? std::atoi(e) : 16;
          }();
          const uint32_t psUserSgprs =
              udBound > 0 ? static_cast<uint32_t>(udBound) : declared;
          auto texs = rdna::TrackTextures(reinterpret_cast<const uint32_t *>(psA), pud,
                                          psUserSgprs);
          // The single-texture render path reads the legacy tex* mirror of
          // texs[0], so populate it too (the PS4 path does the same).
          if (!texs.empty()) {
            d.texBase = texs[0].valid ? texs[0].base : 0;
            d.texSwizzle = packDstSel(texs[0]);
            d.texW = texs[0].width;
            d.texH = texs[0].height;
            d.texDfmt = texs[0].dfmt;
            d.texNfmt = texs[0].nfmt;
            d.texTiling = texs[0].tiling_idx;
            d.texPitch = texs[0].pitch;
            d.texLayers = texs[0].layers;
            d.texBaseArray = texs[0].base_array;
            d.texViewLayers = texs[0].view_layers;
            d.texMipLevels = texs[0].mip_levels;
            d.texBaseMip = texs[0].base_mip;
            d.texViewMips = texs[0].view_mips;
            d.texMinLod = texs[0].min_lod;
            std::memcpy(d.texSampler, texs[0].sampler, sizeof(d.texSampler));
            d.texPow2Pad = texs[0].pow2_pad;
            d.texSamplerValid = texs[0].sampler_valid;
            d.texArrayed = texs[0].arrayed;
            d.texForceLodZero = texs[0].force_lod_zero;
            d.texDepthCompare = texs[0].depth_compare;
          }
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
            dt.swizzle = packDstSel(s);
          }
          d.nTexs = static_cast<uint32_t>(std::min<size_t>(texs.size(), 16));
          if (dl)
            for (uint32_t i = 0; i < d.nTexs; i++)
              std::fprintf(stderr,
                           "[agc]   tex%u base=%#lx %ux%u dfmt=%u nfmt=%u tiling=%u\n",
                           i, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
                           d.texs[i].dfmt, d.texs[i].nfmt, d.texs[i].tiling);
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
  if (!d.recomp) {
    // No usable shader pair: the draw is discarded here, which looks exactly
    // like the title never issuing it. Report the addresses that failed so the
    // gap is attributable.
    g_dropNoShader.fetch_add(1, std::memory_order_relaxed);
    static int shown = 0;
    if (shown < 16 && std::getenv("DELTA_GPU_DRAWCENSUS")) {
      shown++;
      std::fprintf(stderr,
                   "[drawcensus] dropped: vs=%#lx ps=%#lx prim=%u vcount=%u "
                   "mrt=%u rt=%#lx nvattrs=%u shaderAttrs=%zu nvbufs=%u\n",
                   (unsigned long)vsA, (unsigned long)psA, d.primType,
                   d.vertexCount, d.mrtCount, (unsigned long)d.rtBase,
                   d.nvattrs, dropAttrCount, d.nvbufs);
    }
    return;
  }

  // One-shot ground-truth dump of the first few resolved draws: the raw vertex
  // bytes (as float32 AND uint32, to read off the real attribute format), the
  // constant-buffer/MVP state, and the viewport -- so we can tell whether the
  // positions are garbage (wrong format), screen-space (missing projection), or
  // clip-space (a downstream/viewport issue).
  static int s_vdump = 0;
  static const int vdumpN = [] {
    const char *e = std::getenv("DELTA_AGC_VDUMPN");
    return e ? std::atoi(e) : 8;
  }();
  // DELTA_AGC_VDUMPFROM=N: skip the opening blits and dump the draws that
  // actually shade the frame.
  static const unsigned long vdumpFrom = [] {
    const char *e = std::getenv("DELTA_AGC_VDUMPFROM");
    return e ? std::strtoul(e, nullptr, 0) : 0ul;
  }();
  // DELTA_AGC_VDUMPRT=<base>: only dump draws that target this render target.
  // Draw indices shift between runs; the target does not.
  static const uint64_t vdumpRt = [] {
    const char *e = std::getenv("DELTA_AGC_VDUMPRT");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  if (g_trace && myDraw >= vdumpFrom && (!vdumpRt || d.rtBase == vdumpRt) &&
      s_vdump < vdumpN && d.nvattrs && d.vertexData &&
      inGuest(reinterpret_cast<uint64_t>(d.vertexData))) {
    s_vdump++;
    std::fprintf(stderr,
                 "[agc] VDUMP draw#%lu nvattrs=%u stride=%u count=%u prim=%u "
                 "vp=[xs=%g xo=%g ys=%g yo=%g] nCbufs=%u cbufBase=%#lx cbufSize=%u "
                 "rt=%#lx ps=%#lx vs=%#lx\n",
                 (unsigned long)myDraw, d.nvattrs, d.vertexStride, d.vertexCount,
                 d.primType, d.viewportXScale, d.viewportXOffset, d.viewportYScale,
                 d.viewportYOffset, d.nCbufs, (unsigned long)d.cbufBase, d.cbufSize,
                 (unsigned long)d.rtBase, (unsigned long)d.psAddr,
                 (unsigned long)vsA);
    for (uint32_t a = 0; a < d.nvattrs; a++)
      std::fprintf(stderr, "[agc]   vattr%u loc=%u off=%u nc=%u dfmt=%u nfmt=%u\n",
                   a, d.vattrs[a].location, d.vattrs[a].offset, d.vattrs[a].num_comps,
                   d.vattrs[a].dfmt, d.vattrs[a].nfmt);
    // DELTA_AGC_VDUMPPROG: the decoded VS for this exact draw (shader dumps are
    // otherwise emitted once at recompile time and cannot be tied to a draw).
    if (std::getenv("DELTA_AGC_VDUMPPROG") && inGuest(vsA)) {
      const bool wantPs = std::getenv("DELTA_AGC_VDUMPPS") != nullptr;
      const uint64_t addr = wantPs ? psA : vsA;
      const auto prog =
          rdna::DecodeShader(reinterpret_cast<const uint32_t *>(addr), 4096);
      std::fprintf(stderr, "[agc]   %s %#lx: %zu insts\n", wantPs ? "PS" : "VS",
                   (unsigned long)addr, prog.size());
      for (const auto &in : prog) {
        std::fprintf(stderr, "[agc]    pc=%04x %-6s op=%#05x %08x", in.pc,
                     kEncName[static_cast<uint32_t>(in.enc) < 17
                                  ? static_cast<uint32_t>(in.enc)
                                  : 0],
                     in.opcode, in.raw[0]);
        if (in.size >= 2) std::fprintf(stderr, " %08x", in.raw[1]);
        if (in.has_literal) std::fprintf(stderr, " lit=%08x", in.literal);
        std::fprintf(stderr, "\n");
      }
    }
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
    // The VS loads its real vertex V# via SMEM from a pointer in GS user_data[4..7]
    // (pc0x23 s_load s0-3 from user_data[4..5]). Follow each such pointer one level
    // and dump what's there (a V# or raw vertices) to locate the real vertex source.
    for (int k = 4; k <= 6; k += 2) {
      uint64_t p = (static_cast<uint64_t>(vud[k + 1] & 0xFFFF) << 32) | vud[k];
      if (!gpuAddr(p)) continue;  // user data holds non-pointers too
      const uint32_t *pw = reinterpret_cast<const uint32_t *>(p);
      std::fprintf(stderr, "[agc]   ud[%d]->%#lx dwords:", k, (unsigned long)p);
      for (int j = 0; j < (k == 4 ? 32 : 8); j++)
        std::fprintf(stderr, " %08x", pw[j]);
      std::fprintf(stderr, "  floats:");
      for (int j = 0; j < 8; j++) {
        float f; std::memcpy(&f, &pw[j], 4);
        std::fprintf(stderr, " %g", f);
      }
      std::fprintf(stderr, "\n");
    }
    // DELTA_AGC_VDUMPPROJ: project this draw's first vertices on the host, using
    // the 4x3 world matrix (a 12-dword cbuffer) and the view-projection (dwords
    // 32-47 of a 48-dword one), and print the NDC the shader ought to produce.
    // Comparing that with the drawn extent says whether the transform chain in
    // the recompiled VS is the thing that is wrong.
    // DELTA_AGC_VDUMPPROJ=<world binding>:<vp binding>:<vp dword> names the
    // matrices, since a cbuffer window holds several and only the shader knows
    // which (read it off the SPIR-V's sgpr <- cbuf loads).
    if (const char *projEnv = std::getenv("DELTA_AGC_VDUMPPROJ");
        projEnv && d.nvattrs && d.vertexData) {
      uint32_t wb = 1, vb2 = 2, vdw = 32;
      std::sscanf(projEnv, "%u:%u:%u", &wb, &vb2, &vdw);
      const float *world = nullptr, *vp = nullptr;
      if (wb < d.nCbufs && d.cbufs[wb].base && inGuest(d.cbufs[wb].base))
        world = reinterpret_cast<const float *>(d.cbufs[wb].base);
      if (vb2 < d.nCbufs && d.cbufs[vb2].base && inGuest(d.cbufs[vb2].base) &&
          d.cbufs[vb2].size / 4 >= vdw + 16)
        vp = reinterpret_cast<const float *>(d.cbufs[vb2].base) + vdw;
      if (world && vp) {
        const auto *vb = reinterpret_cast<const uint8_t *>(d.vertexData);
        for (uint32_t v = 0; v < 8 && v < d.vertexCount; v++) {
          const float *p = reinterpret_cast<const float *>(vb + (size_t)v * d.vertexStride);
          float w4[4] = {0, 0, 0, 1};
          for (int r = 0; r < 3; r++)
            w4[r] = world[r * 4 + 0] * p[0] + world[r * 4 + 1] * p[1] +
                    world[r * 4 + 2] * p[2] + world[r * 4 + 3];
          float c[4];
          for (int r = 0; r < 4; r++)
            c[r] = vp[r * 4 + 0] * w4[0] + vp[r * 4 + 1] * w4[1] +
                   vp[r * 4 + 2] * w4[2] + vp[r * 4 + 3] * w4[3];
          std::fprintf(stderr,
                       "[agc]   proj v%u obj=(%g %g %g) world=(%g %g %g) "
                       "clip=(%g %g %g %g) ndc=(%g %g)\n",
                       v, p[0], p[1], p[2], w4[0], w4[1], w4[2], c[0], c[1], c[2],
                       c[3], c[3] ? c[0] / c[3] : 0.f, c[3] ? c[1] / c[3] : 0.f);
        }
      }
    }
    // DELTA_AGC_VDUMPCB=<n>: how many floats of each bound cbuffer to print. The
    // default shows the head; a transform hides further in (48-dword windows hold
    // several matrices).
    static const int cbFloats = [] {
      const char *e = std::getenv("DELTA_AGC_VDUMPCB"); return e ? std::atoi(e) : 8;
    }();
    for (uint32_t b = 0; b < d.nCbufs; b++) {
      if (!d.cbufs[b].base || !inGuest(d.cbufs[b].base)) continue;
      const float *cf = reinterpret_cast<const float *>(d.cbufs[b].base);
      const int n = std::min<int>(cbFloats, d.cbufs[b].size / 4);
      std::fprintf(stderr, "[agc]   cbuf[%u]@%#lx (%u dw) floats:", b,
                   (unsigned long)d.cbufs[b].base, d.cbufs[b].size / 4);
      for (int j = 0; j < n; j++) {
        if (j && j % 4 == 0) std::fprintf(stderr, " |");
        std::fprintf(stderr, " %g", cf[j]);
      }
      std::fprintf(stderr, "\n");
    }
  }

  if (!g_frameActive) {
    if (dl) std::fprintf(stderr, "[agc] DL beginFrame...\n");
    vk::beginFrame();
    g_frameActive = true;
  }
  if (dl) {
    std::fprintf(stderr, "[agc] DL draw#%lu vk::draw nvattrs=%d rt=%#lx "
                 "tmask=%#x cc=%#x blend=%u dv=%d db=%#lx dt=%u dw=%u df=%u ntex=%u tex0=%#lx\n",
                 (unsigned long)myDraw, d.nvattrs, (unsigned long)d.rtBase,
                 d.targetMask, d.colorControl, d.blendEnable, d.depthValid,
                 (unsigned long)d.depthBase, d.depthTestEnable,
                 d.depthWriteEnable, d.depthFunc, d.nTexs,
                 (unsigned long)(d.nTexs ? d.texs[0].base : 0));
    for (uint32_t i = 0; i < d.nTexs; i++)
      std::fprintf(stderr, "[agc]   DL tex%u base=%#lx %ux%u dfmt=%u nfmt=%u tiling=%u pitch=%u\n",
                   i, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
                   d.texs[i].dfmt, d.texs[i].nfmt, d.texs[i].tiling, d.texs[i].pitch);
    std::fprintf(stderr, "[agc]   DL vtx data=%#lx stride=%u count=%u nvbufs=%u idx=%#lx icount=%u\n",
                 (unsigned long)d.vertexData, d.vertexStride, d.vertexCount, d.nvbufs,
                 (unsigned long)d.indexData, d.indexCount);
    for (uint32_t i = 0; i < d.nvbufs; i++)
      std::fprintf(stderr, "[agc]   DL vbuf%u data=%#lx stride=%u nrec=%u\n",
                   i, (unsigned long)d.vbufs[i].data, d.vbufs[i].stride, d.vbufs[i].numRecords);
  }
  g_drawsIssued.fetch_add(1, std::memory_order_relaxed);
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
      if (i + 1 + cnt > words) {
        // DELTA_AGC_WALKSTAT: a packet whose count runs past the buffer means we
        // mis-parsed something earlier; the walker resyncs a dword at a time and
        // every packet in between is lost.
        static const bool walkStat = std::getenv("DELTA_AGC_WALKSTAT") != nullptr;
        static uint64_t resyncs = 0;
        if (walkStat && (++resyncs % 500) == 1)
          std::fprintf(stderr,
                       "[walkstat] resync #%llu at word %u/%u (hdr %08x op %#x cnt %u)\n",
                       (unsigned long long)resyncs, i, words, hdr, op, cnt);
        i += 1;
        continue;
      }
      g_opHist[op & 0xFF]++;
      // Who actually binds the render target: report the packet that first makes
      // CB_COLOR0_BASE non-zero, and every later change.
      if (g_trace) {
        static uint32_t s_lastCb = 0;
        static int s_cbLog = 0;
        uint32_t cur = g_regs[mmCB_COLOR0_BASE];
        static uint32_t s_prevOp = 0;
        if (cur != s_lastCb && s_cbLog < 16) {
          s_cbLog++;
          std::fprintf(stderr,
                       "[agc] CB0BASE %08x -> %08x by op %#04x (next %#04x)\n",
                       s_lastCb, cur, s_prevOp, op);
          s_lastCb = cur;
        }
        s_prevOp = op;
        if (false) {
        } else if (cur != s_lastCb) {
          s_lastCb = cur;
        }
      }
      if (dumpThis) {
        std::fprintf(stderr, "[agc]   @%-5u T3 op=%#04x count=%u body:", i, op, cnt);
        uint32_t showN = (op == 0x93 || op == 0x79) ? cnt : (cnt < 6 ? cnt : 6);
        for (uint32_t b = 0; b < showN && b < 24; b++)
          std::fprintf(stderr, " %08x", body[b]);
        // INDIRECT register packets reference a GPU buffer at body[0..1]; dump it
        // so we can RE the register layout (which offset holds CB_COLOR/shaders).
        if ((op == 0x9f || op == 0x93 || op == 0x64 || op == 0x7a || op == 0x63) && cnt >= 2) {
          uint64_t a = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          if (gpuAddr(a)) {
            auto *aw = reinterpret_cast<const uint32_t *>(a);
            std::fprintf(stderr, " -> buf %#lx:", (unsigned long)a);
            for (int b = 0; b < 12; b++) std::fprintf(stderr, " %08x", aw[b]);
          }
        }
        std::fprintf(stderr, "\n");
      }
      // DELTA_AGC_RTPROBE: remember which packet last changed CB_COLOR0_BASE, so a
      // draw that ends up with no colour target can name what unbound it.
      const uint32_t cb0Before = g_regs[mmCB_COLOR0_BASE];
      switch (op) {
      case IT_INDIRECT_BUFFER:       // baseLo, baseHi, sizeDwords(+flags)
      case 0x33: {                   // IT_INDIRECT_BUFFER_CNST (AGC constant/Cue chain
                                     // -- carries the pipeline SET_SH_REG shader setup)
        if (cnt >= 3) {
          uint64_t ib = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          uint32_t ibw = body[2] & 0xFFFFF;
          // Bounds-guard: only follow IBs into the GPU aperture with a sane size,
          // so a stale/garbage ring window can't fault the walker.
          if (gpuAddr(ib) && ibw &&
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
          for (uint32_t k = 1; k < cnt; k++) {
            if (off + (k - 1) < kRegFileSize) g_regs[off + (k - 1)] = body[k];
            noteUdWrite("SET_SH_INLINE(0x93)", off + (k - 1), body[k]);
          }
        }
        break;
      }
      case IT_DMA_DATA:  // CP DMA: ctrl, srcLo/Hi, dstLo/Hi, command(byteCount).
        // Skyrim never issues SET_CONTEXT_REG: it builds its context state (so
        // CB_COLOR -- the render target) into a shadow image with CP DMA and then
        // restores it with LOAD_CONTEXT_REG. Without the copy the shadow reads
        // zero, every colour draw runs with no target bound and the frame is
        // black. ctrl: SRC_SEL[30:29], DST_SEL[21:20]; sel 0/3 = memory address,
        // 2 = immediate (a fill) -- only copy true memory to memory.
        if (cnt >= 6) {
          uint32_t ctrl = body[0];
          uint32_t srcSel = (ctrl >> 29) & 0x3;
          uint32_t dstSel = (ctrl >> 20) & 0x3;
          uint64_t src = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
          uint64_t dst = (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
          uint32_t bytes = body[5] & 0x1FFFFF;
          const bool srcMem = (srcSel == 0 || srcSel == 3);
          const bool dstMem = (dstSel == 0 || dstSel == 3);
          static const bool noCopy = std::getenv("DELTA_GPU_NODMACOPY") != nullptr;
          auto memOk = [](uint64_t a) {
            return a >= 0x1000000ull && a < 0x20000000000ull;
          };
          if (!noCopy && srcMem && dstMem && bytes && bytes <= 0x1000000u &&
              src != dst && memOk(src) && memOk(src + bytes) && memOk(dst) &&
              memOk(dst + bytes))
            std::memcpy(reinterpret_cast<void *>(dst),
                        reinterpret_cast<const void *>(src), bytes);
          // srcSel 2 = the packet's own dword, repeated: a fill. That is how this
          // title clears a surface -- there is no clear packet -- so apply it to
          // guest memory and let the renderer clear any target it covers.
          if (!noCopy && srcSel == 2 && dstMem && bytes && bytes <= 0x8000000u &&
              memOk(dst) && memOk(dst + bytes)) {
            const uint32_t fill = body[1];
            auto *p32 = reinterpret_cast<uint32_t *>(dst);
            for (uint32_t k = 0; k < bytes / 4; k++) p32[k] = fill;
            vk::noteMemoryFill(dst, bytes, fill);
          }
          if (std::getenv("DELTA_GPU_DMATRACE")) {
            static int dmn = 0;
            if (dmn++ < 60)
              std::fprintf(stderr,
                           "[dma] ctrl=%#x src=%#lx dst=%#lx bytes=%u%s\n", ctrl,
                           (unsigned long)src, (unsigned long)dst, bytes,
                           (srcMem && dstMem) ? " COPIED" : "");
          }
        }
        break;
      case IT_INDEX_TYPE:
        if (cnt >= 1) g_indexType = body[0] & 0x3;
        break;
      case IT_NUM_INSTANCES:
        if (cnt >= 1) g_numInstances = body[0] ? body[0] : 1;
        break;
      case IT_DISPATCH_DIRECT: handleDispatch(body, cnt); break;
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
      if (g_regs[mmCB_COLOR0_BASE] != cb0Before) {
        g_cb0Op = op;
        g_cb0Val = g_regs[mmCB_COLOR0_BASE];
        g_cb0Draw = g_drawsSeen;
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
        for (uint32_t j = 0; j < cnt0; j++) {
          if (base0 + j < kRegFileSize) g_regs[base0 + j] = p[i + 1 + j];
          if (base0 + j == mmPA_CL_CLIP_CNTL) noteClipWrite("type0", p[i + 1 + j]);
          noteUdWrite("type0", base0 + j, p[i + 1 + j]);
        }
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
        if (gpuAddr(a)) {
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
