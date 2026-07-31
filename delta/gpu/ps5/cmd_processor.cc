/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * PS5 AGC command processor. The AGC command buffer libSceAgc builds is a PM4
 * type-3 stream using the SAME IT_ opcode table as the PS4 (verified in memory
 * [[ps5-agc-gpu]]), so the packet walk + completion-label handling reuse
 * gpu/ps4/pm4.h. What differs on PS5 is the gfx10.3 register offsets
 * (agc_regs.h) and the RDNA2 shader ISA (gpu/ps5/rdna) -- both new; the Vulkan
 * renderer (gpu/rhi) and the DrawInfo contract are shared with the PS4 path.
 *
 * The register-latch model mirrors gpu/ps4/cmd_processor.cc: SET_*_REG packets
 * write into a flat gfx10.3 register file (masking the gfx10 selector bits), a
 * draw packet snapshots that state into gpu::rhi::DrawInfo, the VS(from the
 * merged ES/GS NGG block)/PS pair is recompiled (cached), and the shared
 * renderer runs the game's real shaders. Completion labels (EOP / RELEASE_MEM /
 * WRITE_DATA) are still serviced so the engine's per-frame command-buffer
 * fences advance.
 */

#include "gpu/ps5/cmd_processor.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/agc_regs.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/rhi/renderer.h"

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utl/options.h>

namespace {
DELTA_OPTION(uint64_t, kBlkFrom, "DELTA_AGC_REGSTAT_FROM", 0);
DELTA_OPTION(int, kCbTraceFrom, "DELTA_AGC_CBTRACE", -1);
DELTA_OPTION(uint64_t, kDumpSh, "DELTA_AGC_DUMPSH", 0);
DELTA_OPTION(uint64_t, kSkipVs, "DELTA_PS5_SKIPVS", 0);
DELTA_OPTION(bool, kNoStickyRt, "DELTA_GPU_NOSTICKYRT", false);
DELTA_OPTION(bool, kRecompOn, "DELTA_PS5_RECOMP", true);
DELTA_OPTION(uint32_t, kUdBase, "DELTA_PS5_UDBASE", 8);
DELTA_OPTION(int, kVdumpN, "DELTA_AGC_VDUMPN", 8);
DELTA_OPTION(unsigned long, kVdumpFrom, "DELTA_AGC_VDUMPFROM", 0);
DELTA_OPTION(uint64_t, kVdumpRt, "DELTA_AGC_VDUMPRT", 0);
DELTA_OPTION(uint32_t, kVdumpIc, "DELTA_AGC_VDUMPIC", 0);
DELTA_OPTION(bool, kVdumpProg, "DELTA_AGC_VDUMPPROG", false);
DELTA_OPTION(const char*, kVdumpProj, "DELTA_AGC_VDUMPPROJ", nullptr);
DELTA_OPTION(int, kCbFloats, "DELTA_AGC_VDUMPCB", 8);
DELTA_OPTION(uint32_t, kOpDump, "DELTA_AGC_OPDUMP", 0xFFFF);
DELTA_OPTION(int, kRegStat, "DELTA_AGC_REGSTAT", 0);
DELTA_OPTION(bool, kAgcCliptrace, "DELTA_AGC_CLIPTRACE", false);
DELTA_OPTION(bool, kAgcUdtrace, "DELTA_AGC_UDTRACE", false);
DELTA_OPTION(bool, kAgcVdumpps, "DELTA_AGC_VDUMPPS", false);
DELTA_OPTION(bool, kCsDump, "DELTA_GPU_CSDUMP", false);
DELTA_OPTION(bool, kGpuDmatrace, "DELTA_GPU_DMATRACE", false);
DELTA_OPTION(bool, kGpuDrawcensus, "DELTA_GPU_DRAWCENSUS", false);
DELTA_OPTION(bool, kNoBlendState, "DELTA_AGC_NOBLENDSTATE", false);
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
DELTA_OPTION(bool, kNoCs, "DELTA_GPU_NOCS", false);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kResTrace, "DELTA_GPU_CSRES", false);
DELTA_OPTION(bool, kRtProbe, "DELTA_AGC_RTPROBE", false);
DELTA_OPTION(bool, kTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kWalkStat, "DELTA_AGC_WALKSTAT", false);
}  // namespace

namespace gpu::ps5 {
namespace {

std::mutex g_mtx;
uint64_t g_total_submits = 0;

// gfx10.3 register file (persists across submits -- AGC relies on sticky
// state).
Regs g_regs;
uint32_t g_index_type = 0;  // 0=uint16, 1=uint32, 2=uint8 (VGT_INDEX_TYPE[1:0])
// INDEX_BASE / INDEX_BUFFER_SIZE state. DRAW_INDEX_OFFSET_2 carries only an
// offset and a count, so the buffer it indexes has to come from these.
uint64_t g_index_base = 0;
uint32_t g_index_max = 0;
uint32_t g_num_instances = 1;  // from IT_NUM_INSTANCES
bool g_frame_active = false;

// PS5 guest allocations sit anywhere across a wide VA (eboot ~0x2014_..., GPU
// dmem tagged regions 0x8000_..., doorbell 0xfe0_...). A shader/RT/buffer
// address is host-readable if plausibly mapped and non-tiny.
inline bool InGuest(uint64_t a) {
  return a >= 0x10000ull && a < 0x1000000000000ull;
}

// EOP/RELEASE_MEM DATA_SEL 3/4 ask the GPU to write its running clock counter;
// our submit is synchronous so any advancing non-zero value reads as complete.
inline bool LabelAddrOk(uint64_t a) {
  return InGuest(a);
}
uint64_t GpuClockTs() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count());
}
void WriteLabel(uint64_t addr, uint64_t value, bool is64) {
  if (!LabelAddrOk(addr))
    return;
  if (is64)
    *reinterpret_cast<volatile uint64_t*>(addr) = value;
  else
    *reinterpret_cast<volatile uint32_t*>(addr) = static_cast<uint32_t>(value);
}

// gfx10.3 128-bit V# (buffer descriptor). base48 = f0 | (f1[15:0] << 32);
// stride f1[29:16]; num_records f2; format f3[18:12]. The 7-bit format field is
// the GCN (nfmt<<4)|dfmt packing (gfx10.3 DecodeTBufferFormat =
// ((nfmt & 0x7) << 4) | (dfmt & 0xf)), so dfmt = fmt&0xF, nfmt = (fmt>>4)&0x7.
struct VBuffer {
  uint64_t base = 0;
  uint32_t stride = 0, num_records = 0, dfmt = 0, nfmt = 0, gfmt = 0;
};
// gfx10.3 buffer V#s carry a UNIFIED 7-bit FORMAT enum (word3 [18:12]), not
// GCN's separate data/number format. Map the enum to the GCN (dfmt,nfmt) pair
// the renderer's VertexFormat() understands. Only the vertex-attribute formats
// are covered; unknowns fall back to the GCN-style split (harmless for
// descriptors that never reach VertexFormat()).
void Gfx10VBufFormat(uint32_t gfmt, uint32_t& dfmt, uint32_t& nfmt) {
  // The gfx10.3 unified buffer-format enum (ISA spec "Buffer Format
  // Conversions"): consecutive runs of one channel layout in the order
  // UNorm, SNorm, UScaled, SScaled, UInt, SInt [, Float]. Map each run onto the
  // GCN (dfmt, nfmt) pair the shared renderer speaks. The old code only listed
  // the float formats and split the rest as GCN bitfields, which turned
  // Skyrim's 16_16_SScaled UI positions (26) into 8_8_8_8_SNorm and collapsed
  // every glyph quad to a degenerate triangle.
  struct Run {
    uint8_t first, count, dfmt;
    bool has_float;
  };
  static constexpr Run kRuns[] = {
      {1, 6, 1, false},    // 8
      {7, 7, 2, true},     // 16
      {14, 6, 3, false},   // 8_8
      {20, 3, 4, true},    // 32 (UInt, SInt, Float)
      {23, 7, 5, true},    // 16_16
      {30, 7, 7, true},    // 11_11_10
      {37, 7, 6, true},    // 10_11_11
      {44, 6, 8, false},   // 10_10_10_2
      {50, 6, 9, false},   // 2_10_10_10
      {56, 6, 10, false},  // 8_8_8_8
      {62, 3, 11, true},   // 32_32
      {65, 7, 12, true},   // 16_16_16_16
      {72, 3, 13, true},   // 32_32_32
      {75, 3, 14, true},   // 32_32_32_32
  };
  static constexpr uint8_t kNfmt[6] = {0, 1, 2, 3, 4, 5};
  for (const Run& r : kRuns) {
    if (gfmt < r.first || gfmt >= r.first + r.count)
      continue;
    const uint32_t i = gfmt - r.first;
    dfmt = r.dfmt;
    if (r.count == 3)  // UInt, SInt, Float
      nfmt = i == 0 ? 4u : i == 1 ? 5u : 7u;
    else if (r.has_float && i == 6)  // trailing Float of a 7-wide run
      nfmt = 7;
    else
      nfmt = kNfmt[i];
    return;
  }
  dfmt = 0;
  nfmt = 0;
}
VBuffer DecodeVBuffer(const uint32_t* p) {
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.num_records = p[2];
  v.gfmt = (p[3] >> 12) & 0x7F;
  Gfx10VBufFormat(v.gfmt, v.dfmt, v.nfmt);
  return v;
}
// A V# read from an unbound/garbage SGPR slot decodes to an in-range but bogus
// address with an implausible stride/record count (e.g. a depth-only pre-pass
// with an inactive vertex slot decoded stride=14915 nrec=480622080, which then
// segfaulted reading the vertex ring). Mirrors the PS4 fetch-shader sanity gate
// (gpu/gcn/gcn_resource.cc).
bool PlausibleVb(const VBuffer& v) {
  return v.stride && v.stride <= 256 && v.num_records &&
         v.num_records <= 0x100000;
}

// Registers per space, so a LOAD_*_REG range can never spill into the next one.
// A LOAD_SH_REG whose range ran long wrote zeros from its (empty) shadow image
// straight over CB_COLOR0_BASE at 0xA318, unbinding the render target that
// op 0x49 had just set -- every colour draw after it hit no target and the
// frame came out black.
constexpr uint32_t kRegSpaceSize = 0x400;

inline uint32_t RegSpaceLimit(uint32_t base) {
  return base + kRegSpaceSize <= kRegFileSize ? base + kRegSpaceSize
                                              : kRegFileSize;
}

// Latch a SET_*_REG packet into the register file. body[0] is the register
// offset dword (with the gfx10 selector bits stripped), body[1..] the values.
// DELTA_AGC_UDTRACE: every write to the PS user-data SGPRs above 15. Skyrim's
// grading pass reads a texture descriptor from s16..s23, and if nothing in the
// command stream programs those the shader samples whatever was left there.
static void NoteUdWrite(const char* how, uint32_t reg, uint32_t val) {
  static int n = 0;
  if (kAgcUdtrace && n < 40 && val && reg >= mmSPI_SHADER_USER_DATA_PS_0 + 16 &&
      reg < mmSPI_SHADER_USER_DATA_PS_0 + 32) {
    n++;
    std::fprintf(stderr, "[agc] PS ud%u <- %08x by %s\n",
                 reg - mmSPI_SHADER_USER_DATA_PS_0, val, how);
  }
}

// DELTA_AGC_CLIPTRACE: name every packet that writes PA_CL_CLIP_CNTL. The
// clip-space convention lives there, and a title whose writes never reach us
// silently gets the reset value (OpenGL clip space).
static void NoteClipWrite(const char* how, uint32_t val) {
  static int n = 0;
  if (kAgcCliptrace && n++ < 12)
    std::fprintf(stderr, "[agc] CLIP_CNTL <- %#x by %s\n", val, how);
}

// Draw accounting. DELTA_GPU_DRAWRT only sees draws that reach the renderer, so
// a draw dropped earlier (no usable render target, unrecoverable shader
// address) is indistinguishable from one the title never issued. Count both
// ends.
static std::atomic<uint64_t> g_draws_seen{0}, g_draws_issued{0},
    g_drop_no_rt{0}, g_drop_no_shader{0};
// The colour target of the last draw actually issued; EndFrame presents it when
// it is a registered display buffer.
static uint64_t g_last_draw_rt = 0;

extern "C" bool prosperity_ps5_is_display_buffer(uint64_t addr);

void SetRegs(uint32_t base, const uint32_t* body, uint32_t count) {
  if (count < 1)
    return;
  uint32_t off = base + (body[0] & ~kRegSelectorMask);
  const uint32_t limit = RegSpaceLimit(base);
  for (uint32_t i = 1; i < count; i++) {
    if (off + (i - 1) < limit)
      g_regs[off + (i - 1)] = body[i];
    if (off + (i - 1) == mmPA_CL_CLIP_CNTL)
      NoteClipWrite("SET_REG", body[i]);
    NoteUdWrite("SET_REG", off + (i - 1), body[i]);
  }
}

// A guest GPU address: anything the guest allocator can hand out, from the
// lowest slot a title fixed-maps a pool at (64 GiB) up to the 2^40 user ceiling
// allocLowGuest() bumps towards. Isaac's AGC pool sits in the 0x80_xx_xx_xx_xx
// band and Skyrim's command buffers land around 0x10_00_00_00_00, but a title
// that allocates more than a few GiB runs well past either: Minecraft's bgfx
// command buffers are at 0x89_xx_xx_xx_xx, and the old 0x81_00_00_00_00 ceiling
// silently dropped every submit pointing at one.
inline bool GpuAddr(uint64_t a) {
  return a >= 0x1000000000ull && a < 0x10000000000ull;
}

// Latch a LOAD_*_REG packet: the register values live in a GPU-memory image at
// body[0..1]; body[2..] are (reg_offset, num_dwords) ranges. Each range copies
// `num` dwords from the image (image[reg_offset..]) into the reg file at
// base+reg_offset. This is how the AGC driver sets context/sh/uconfig state for
// this title (it uses LOAD_*_REG, not inline SET_*_REG), so without it CB_COLOR
// (the render target) is never bound and draws hit no visible target.
void LoadRegs(uint32_t base, const uint32_t* body, uint32_t count) {
  if (count < 4)
    return;
  uint64_t mem = (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
  if (!GpuAddr(mem))
    return;  // GPU aperture only
  const uint32_t limit = RegSpaceLimit(base);
  uint64_t source_dwords = kTrace ? 0x400 : 0;
  for (uint32_t i = 2; i + 1 < count; i += 2) {
    const uint32_t off = body[i] & 0xFFFF;
    const uint32_t num = std::min<uint32_t>(body[i + 1] & 0xFFFF, 0x2000);
    if (base + off < limit)
      source_dwords = std::max<uint64_t>(
          source_dwords, off + std::min<uint32_t>(num, limit - base - off));
  }
  if (!source_dwords ||
      !gpu::IsReadableRange(mem, source_dwords * sizeof(uint32_t)))
    return;
  const uint32_t* src = reinterpret_cast<const uint32_t*>(mem);
  // COHERENCY TEST: dump the LOAD image content. If a context/SH image reads
  // all zero, it's non-coherent (the driver wrote it via a different VA) -- the
  // shader would be present but invisible. If it has real reg values, the image
  // is fine. Coherency census: a title that drives its whole context through
  // register shadows is invisible if those images read zero. Sample one late.
  static int s_img_n = 0;
  if (base == kContextRegBase)
    s_img_n++;
  if (kTrace && base == kContextRegBase && s_img_n >= 100 && s_img_n <= 106) {
    uint32_t nz = 0, first = 0;
    for (uint32_t j = 0; j < 0x400; j++)
      if (src[j]) {
        nz++;
        if (!first)
          first = j;
      }
    std::fprintf(
        stderr,
        "[agc] IMGCENSUS #%d base=%#x mem=%#lx nonzero=%u/1024 first=%#x "
        "img[0x318]=%08x img[0x31c]=%08x ranges=%u\n",
        s_img_n, base, (unsigned long)mem, nz, first, src[0x318], src[0x31c],
        (count - 2) / 2);
  }
  static int s_img = 0;
  if (kTrace && s_img < 6) {
    s_img++;
    std::fprintf(stderr, "[agc]   LOADimg base=%#x mem=%#lx:", base,
                 (unsigned long)mem);
    for (int j = 0; j < 16; j++)
      std::fprintf(stderr, " %08x", src[j]);
    std::fprintf(stderr, "\n");
  }
  // SH-image ground-truth: LOAD_SH_REG could be offset-INDEXED (image[reg_off]
  // = value, a full reg-file shadow) OR CURSOR-based (ranges packed
  // contiguously from mem). Dump both interpretations + the range list, and
  // scan the whole image for a PGM pointer (top byte 0x80), to settle where the
  // shader PGM actually is.
  static int s_shimg = 0, s_shcnt = 0;
  // Fire on the first few SH loads AND on a batch ~5000 loads in (once the
  // title is past the loading screen and issuing real sprite draws) -- the
  // early shadow may be empty simply because rendering hasn't started.
  bool late_window = base == kShRegBase && s_shcnt >= 5000 && s_shcnt < 5006;
  if (base == kShRegBase)
    s_shcnt++;
  if (kTrace && base == kShRegBase && (s_shimg < 3 || late_window)) {
    s_shimg++;
    std::fprintf(stderr, "[agc] (shload #%d)\n", s_shcnt);
    std::fprintf(stderr, "[agc] SHLOAD mem=%#lx ranges:", (unsigned long)mem);
    for (uint32_t i = 2; i + 1 < count; i += 2)
      std::fprintf(stderr, " (off=%#x,num=%u)", body[i] & 0xFFFF,
                   body[i + 1] & 0xFFFF);
    std::fprintf(stderr,
                 "\n[agc]   indexed[0x08..0x0b]=%08x %08x %08x %08x  "
                 "indexed[0x88..0x8b]=%08x %08x %08x %08x\n",
                 src[0x08], src[0x09], src[0x0a], src[0x0b], src[0x88],
                 src[0x89], src[0x8a], src[0x8b]);
    // cursor walk: read ranges contiguously from mem, show reg_off + first two
    // vals
    uint32_t cur = 0;
    for (uint32_t i = 2; i + 1 < count; i += 2) {
      uint32_t off = body[i] & 0xFFFF, num = body[i + 1] & 0xFFFF;
      if (num > 0x400)
        num = 0x400;
      std::fprintf(stderr, "[agc]   cursor off=%#x <- img[%u..]: %08x %08x\n",
                   off, cur, src[cur], num > 1 ? src[cur + 1] : 0);
      cur += num;
    }
    // scan whole image for a PGM-like pointer (val<<8 in the GPU aperture)
    for (uint32_t j = 0; j < 0x400; j++) {
      uint32_t v = src[j];
      if ((v >> 24) == 0x80) {
        uint64_t a = (uint64_t)v << 8;
        if (GpuAddr(a) && gpu::IsReadableRange(a, 2 * sizeof(uint32_t))) {
          const uint32_t* w = reinterpret_cast<const uint32_t*>(a);
          std::fprintf(stderr, "[agc]   img[%#x]=%08x -> %#lx ISA? %08x %08x\n",
                       j, v, (unsigned long)a, w[0], w[1]);
        }
      }
    }
  }
  // LOAD_*_REG restores a register shadow the command processor is supposed to
  // have SAVED into. We do not model the save side, so a shadow the title never
  // wrote reads as all zeros -- and applying it wipes live state: Skyrim binds
  // CB_COLOR0_BASE with SET_CONTEXT_REG_INDIRECT (0x9f) and the very next
  // LOAD_CONTEXT_REG zeroed it again, leaving every colour draw with no render
  // target. An all-zero shadow carries nothing to restore, so skip it.
  {
    bool any_value = false;
    for (uint32_t i = 2; i + 1 < count && !any_value; i += 2) {
      uint32_t off = body[i] & 0xFFFF;
      uint32_t num = body[i + 1] & 0xFFFF;
      if (num > 0x2000)
        num = 0x2000;
      for (uint32_t j = 0; j < num; j++)
        if (base + off + j < limit && src[off + j]) {
          any_value = true;
          break;
        }
    }
    if (!any_value)
      return;
  }
  for (uint32_t i = 2; i + 1 < count; i += 2) {
    uint32_t off = body[i] & 0xFFFF;
    uint32_t num = body[i + 1] & 0xFFFF;
    if (num > 0x2000)
      num = 0x2000;  // sanity cap
    for (uint32_t j = 0; j < num; j++) {
      if (base + off + j >= limit)
        break;
      uint32_t v = src[off + j];
      g_regs[base + off + j] = v;
      if (base + off + j == mmPA_CL_CLIP_CNTL)
        NoteClipWrite("LOAD_REG", v);
      NoteUdWrite("LOAD_REG", base + off + j, v);
      // Pinpoint where the shader PGM_LO lands: a shader at 0x8001xxxxxx has
      // PGM_LO ~0x800xxxxx (top byte 0x80). Log SH-space hits to find the reg.
      static int s_sh = 0;
      if (kTrace && base == kShRegBase && (v >> 24) == 0x80 && s_sh < 20) {
        s_sh++;
        std::fprintf(stderr, "[agc]   SH pgm? off=%#x val=%#x (addr~%#lx)\n",
                     off + j, v, (unsigned long)((uint64_t)v << 8));
      }
    }
  }
}

// Latch a SET_*_REG_INDIRECT packet (op 0x9f context / 0x93 sh / 0x64 uconfig).
// body = [addrLo, addrHi, mode, numDwords]; the GPU buffer at addr holds
// (reg_offset, value) PAIRS. This is how the AGC driver binds the per-draw
// render target + shaders for this title (e.g. reg 0x318 CB_COLOR0_BASE, 0x8e
// CB_TARGET_MASK), so without it draws hit no target and no shader.
void LoadRegPairs(uint32_t base, const uint32_t* body, uint32_t cnt) {
  // DELTA_AGC_REGSTAT: why an indirect register load carried nothing. Every one
  // of these that bails is a whole draw's state (render target, shaders) left at
  // whatever the shadow held, which downstream looks like "the title drew
  // nothing" rather than "we dropped its state".
  struct Stat {
    uint64_t calls, short_pkt, bad_addr, unreadable, no_pairs, applied;
  };
  static Stat s_stat[3] = {};
  Stat& st = s_stat[base == kContextRegBase ? 0 : base == kShRegBase ? 1 : 2];
  const auto report = [&] {
    if (!kRegStat)
      return;
    static uint64_t n = 0;
    if ((++n % 4000) != 1)
      return;
    for (int k = 0; k < 3; k++)
      std::fprintf(stderr,
                   "[regstat] %s calls=%llu short=%llu badaddr=%llu "
                   "unreadable=%llu nopairs=%llu applied=%llu\n",
                   k == 0 ? "context" : k == 1 ? "sh" : "uconfig",
                   (unsigned long long)s_stat[k].calls,
                   (unsigned long long)s_stat[k].short_pkt,
                   (unsigned long long)s_stat[k].bad_addr,
                   (unsigned long long)s_stat[k].unreadable,
                   (unsigned long long)s_stat[k].no_pairs,
                   (unsigned long long)s_stat[k].applied);
  };
  st.calls++;
  report();
  if (cnt < 4) {
    st.short_pkt++;
    return;
  }
  uint64_t addr =
      (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | (body[0] & 0xFFFFFFFCu);
  if (!GpuAddr(addr)) {
    st.bad_addr++;
    if (kRegStat && st.bad_addr < 6)
      std::fprintf(stderr, "[regstat] bad addr %#lx (body %08x %08x)\n",
                   (unsigned long)addr, body[0], body[1]);
    return;
  }
  // body[3] is the count of (reg_offset, value) register PAIRS, not dwords:
  // each iteration reads two dwords (the gfx10.3 SET_*_REG_INDIRECT handlers
  // loop `i < (buffer[3] & 0x3fff)` advancing the pointer by 2). The old
  // dword-count reading wrote nothing for a single-register indirect (num ==
  // 1), so VGT_PRIMITIVE_TYPE (set this way) never landed and prim assembly
  // died.
  uint32_t num_pairs = body[3] & 0x3FFF;
  if (!num_pairs) {
    st.no_pairs++;
    return;
  }
  if (!gpu::IsReadableRange(
          addr, static_cast<uint64_t>(num_pairs) * 2 * sizeof(uint32_t))) {
    st.unreadable++;
    if (kRegStat && st.unreadable < 6)
      std::fprintf(stderr, "[regstat] unreadable %#lx pairs=%u\n",
                   (unsigned long)addr, num_pairs);
    return;
  }
  st.applied++;
  // DELTA_AGC_REGSTAT=2 dumps a whole state block. The (offset,value) reading
  // below is NOT the real encoding for every block this title submits: some
  // arrive as runs of entries whose offset dword repeats (0x1000_0000 with the
  // register index in the low half), which collapses onto one register instead
  // of the CB_COLORn block it means. That is why no colour target is ever bound.
  // DELTA_AGC_REGSTAT_FROM=<draw>: skip the init blocks and dump the ones a
  // steady-state frame submits, tagged with the draw they precede, so two
  // consecutive draws' state can be diffed.
  if (kRegStat >= 2 &&
      num_pairs >= 4 &&
      g_draws_seen.load(std::memory_order_relaxed) >= kBlkFrom) {
    // =3 trades the full dump for a compact non-zero digest of many more
    // blocks, so a whole steady-state frame's state stream fits in one run.
    const bool digest = kRegStat >= 3;
    static int full = 0;
    if (full++ < (digest ? 400 : 40)) {
      std::fprintf(stderr,
                   "[regstat] %s block %#lx pairs=%u mode=%08x draw=%lu\n",
                   base == kContextRegBase ? "ctx"
                   : base == kShRegBase    ? "sh"
                                           : "ucfg",
                   (unsigned long)addr, num_pairs, body[2],
                   (unsigned long)g_draws_seen.load(std::memory_order_relaxed));
      const uint32_t* q = reinterpret_cast<const uint32_t*>(addr);
      for (uint32_t k = 0; k < num_pairs; k++)
        if (!digest || q[k * 2 + 1] || (q[k * 2] & (1u << 28)))
          std::fprintf(stderr, "[regstat]   %3u: %08x %08x\n", k, q[k * 2],
                       q[k * 2 + 1]);
    }
  }
  const uint32_t* p = reinterpret_cast<const uint32_t*>(addr);
  const uint32_t limit = RegSpaceLimit(base);
  // Blend/write-mask tail. A type-1 block always ENDS with the blend object,
  // and that object's fields sit at a constant distance from the block's end:
  // over every shape this title submits (245/114/82/78 pairs, and a 21-entry
  // partial) CB_TARGET_MASK is at pairs-23 and CB_BLEND0_CONTROL at pairs-24,
  // followed by two spares, the per-target array (tags 1..0xF) and five more.
  // The object is truncatable: a shorter tail simply stops before the mask
  // field, which then means "writes nothing". That is how a clear quad says it
  // must not touch colour -- Minecraft ends every frame with one, and taking it
  // as a normal draw overwrote the composited UI with flat white.
  // The object is recognised by its per-target array: the 15 entries at
  // pairs-20..pairs-6 carry tags 1..0xF. Requiring that keeps a block which
  // merely ENDS in some OTHER type-1 object from being read as blend state --
  // doing so zeroed the write mask and silently deleted the draw.
  bool tail = !kNoBlendState && base == kContextRegBase && num_pairs >= 24;
  for (uint32_t k = 0; tail && k < 15; k++) {
    const uint32_t tag = p[(num_pairs - 20 + k) * 2];
    if (!(tag & (1u << 28)) || (tag & ~kRegSelectorMask) != k + 1)
      tail = false;
  }
  const bool tail_has_mask = tail && (p[(num_pairs - 23) * 2] & (1u << 28));
  bool has_blend_tail = tail_has_mask;
  if (has_blend_tail) {
    g_regs[mmCB_BLEND0_CONTROL] = (p[(num_pairs - 24) * 2] & (1u << 28))
                                      ? p[(num_pairs - 24) * 2 + 1]
                                      : 0;
    g_regs[mmCB_TARGET_MASK] = p[(num_pairs - 23) * 2 + 1];
  }
  // Type-1 context entries (offset dword bit 28 set) are NOT (offset,value)
  // pairs: whole runs share one offset dword, so the flat reading collapses a
  // CB_COLORn block onto CB_COLOR0_BASE and every draw ends up with no target.
  // Both observed block shapes fit reg(pair) = B + 2*pair, differing only in B,
  // and nothing in the packet or the tags carries B. Anchor it on evidence
  // instead: a pair whose value is a mapped surface address (>>8) and whose
  // pair+2 decodes as a CB_COLOR_INFO with a real FORMAT is a CB_COLOR0 bind.
  // Both fields must check out, so a wrong anchor is rejected rather than
  // binding garbage.
  bool bound_rt = false;
  if (base == kContextRegBase && num_pairs >= 15 && (p[0] & (1u << 28))) {
    for (uint32_t k = 0; k + 14 < num_pairs; k++) {
      const uint64_t rt = static_cast<uint64_t>(p[k * 2 + 1]) << 8;
      const uint32_t info = p[(k + 2) * 2 + 1];
      const uint32_t fmt = (info >> 2) & 0x1F;
      // ATTRIB2 holds MIP0_WIDTH-1 / MIP0_HEIGHT-1, so it doubles as the check
      // that the anchor is real: a wrong k gives nonsense dimensions.
      const uint32_t attrib2 = p[(k + 14) * 2 + 1];
      const uint32_t w = ((attrib2 >> 14) & 0x3FFF) + 1;
      const uint32_t h = (attrib2 & 0x3FFF) + 1;
      if (!fmt || fmt > 22 || !GpuAddr(rt) ||
          !gpu::IsReadableRange(rt, 0x1000) || w < 16 || h < 16 ||
          w > 8192 || h > 8192)
        continue;
      g_regs[mmCB_COLOR0_BASE] = p[k * 2 + 1];
      g_regs[mmCB_COLOR0_INFO] = info;
      g_regs[mmCB_COLOR0_ATTRIB2] = attrib2;
      bound_rt = true;
      // A block carrying a base AND a valid colour format IS a colour-target
      // bind, so slot 0 is written. If the block brought no blend tail its
      // write mask is unknown, and a default 0 reads as "writes nothing" --
      // which would drop the very draw this block set up.
      if (!has_blend_tail && !(g_regs[mmCB_TARGET_MASK] & 0xF))
        g_regs[mmCB_TARGET_MASK] |= 0xF;
      static int shown = 0;
      if (kRegStat && shown++ < 8)
        std::fprintf(
            stderr,
            "[regstat] ctx anchor pair=%u rt=%#lx info=%08x fmt=%u %ux%u\n", k,
            (unsigned long)rt, info, fmt, w, h);
      break;
    }
  }
  // Tail that stops before the mask field: the object says "no colour write".
  // A block whose tail is only the colour-target object instead (it binds a
  // base, so it is not a blend object at all) keeps the mask the anchor forced.
  if (tail && !tail_has_mask && !bound_rt) {
    g_regs[mmCB_BLEND0_CONTROL] = 0;
    g_regs[mmCB_TARGET_MASK] = 0;
  }
  for (uint32_t i = 0; i < num_pairs; i++) {
    uint32_t off = p[i * 2] & ~kRegSelectorMask;  // strip gfx10 selector bits
    if (base == kContextRegBase && (p[i * 2] & (1u << 28)))
      continue;  // handled by the anchor above; a flat write would undo it
    if (base + off < limit)
      g_regs[base + off] = p[i * 2 + 1];
    if (base + off == mmPA_CL_CLIP_CNTL)
      NoteClipWrite("SET_REG_INDIRECT", p[i * 2 + 1]);
    NoteUdWrite("SET_REG_INDIRECT", base + off, p[i * 2 + 1]);
    // DELTA_AGC_CBTRACE: every write to the colour-target base/format, with the
    // packet's own bookkeeping, so a zero write can be traced to a mis-parse.
    static int cb_n = 0;
    const bool cb_trace =
        kCbTraceFrom >= 0 &&
        (int)g_draws_seen.load(std::memory_order_relaxed) >= kCbTraceFrom;
    if (cb_trace && cb_n < 60 &&
        (base + off == mmCB_COLOR0_BASE || base + off == mmCB_COLOR0_INFO)) {
      cb_n++;
      std::fprintf(
          stderr,
          "[agc] CB0 %s <- %08x  (pair %u/%u from %#lx, raw off %08x)\n",
          base + off == mmCB_COLOR0_BASE ? "BASE" : "INFO", p[i * 2 + 1], i,
          num_pairs, (unsigned long)addr, p[i * 2]);
    }
  }
  // Report the first few shader binds that actually carry a nonzero PGM (SH off
  // 0x88 = PGM_LO_GS, 0x08 = PGM_LO_PS) -- these are the real sprite-pipeline
  // binds.
  static int s_shpgm = 0;
  if (kTrace && base == kShRegBase && s_shpgm < 6) {
    uint32_t gs_lo = g_regs[kShRegBase + 0x88],
             ps_lo = g_regs[kShRegBase + 0x08];
    if (gs_lo || ps_lo) {
      s_shpgm++;
      std::fprintf(stderr,
                   "[agc] SHADER BIND @%#lx: PGM_LO_GS=%08x PGM_LO_PS=%08x\n",
                   (unsigned long)addr, gs_lo, ps_lo);
    }
  }
}

// Render-target dimensions, derived from the viewport the title programmed: the
// RT spans [center-halfSize, center+halfSize] where halfSize = |VPORT_xSCALE|.
// The screen scissor is a "no-clip" max (e.g. 16384) and is NOT the RT size.
// Reading the viewport keeps this title-agnostic (no fixed default resolution);
// a 0 scale (no viewport yet) yields 0 so GetRT skips the draw until real state
// is set.
uint32_t FbDim(uint32_t scale_reg) {
  float s;
  std::memcpy(&s, &g_regs[scale_reg], 4);
  if (s < 0.0f)
    s = -s;
  return static_cast<uint32_t>(s * 2.0f + 0.5f);
}
// CB_COLOR0_ATTRIB2 carries the bound target's real size (MIP0_WIDTH-1 in
// [27:14], MIP0_HEIGHT-1 in [13:0]). Titles whose context state arrives as AGC
// state blocks never program the viewport registers, so the scale-derived size
// reads 0 and every draw is skipped for a zero-sized target; the target's own
// dimensions are the authority in that case.
uint32_t FbAttrib2Dim(bool want_width) {
  const uint32_t a2 = g_regs[mmCB_COLOR0_ATTRIB2];
  if (!a2)
    return 0;
  return want_width ? ((a2 >> 14) & 0x3FFF) + 1 : (a2 & 0x3FFF) + 1;
}
uint32_t FbWidth() {
  const uint32_t v = FbDim(mmPA_CL_VPORT_XSCALE);
  return v ? v : FbAttrib2Dim(true);
}
uint32_t FbHeight() {
  const uint32_t v = FbDim(mmPA_CL_VPORT_YSCALE);
  return v ? v : FbAttrib2Dim(false);
}

bool IsDraw(uint32_t op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// Cache key: the VS/PS/fetch triple (attribute translation depends on the fetch
// program, not just the VS/PS code).
struct ShaderKey {
  uint64_t vs, ps, fetch;
  uint32_t gs_user_sgprs, ps_user_sgprs, ps_input_ena;
  bool gl_clip;  // clip convention is baked into the VS (see PA_CL_CLIP_CNTL)
  bool operator==(const ShaderKey& o) const {
    return vs == o.vs && ps == o.ps && fetch == o.fetch &&
           gs_user_sgprs == o.gs_user_sgprs &&
           ps_user_sgprs == o.ps_user_sgprs && ps_input_ena == o.ps_input_ena &&
           gl_clip == o.gl_clip;
  }
};
struct ShaderKeyHash {
  size_t operator()(const ShaderKey& k) const {
    return std::hash<uint64_t>{}(k.vs) ^ (std::hash<uint64_t>{}(k.ps) << 1) ^
           (std::hash<uint64_t>{}(k.fetch) << 2) ^
           (static_cast<size_t>(k.gs_user_sgprs) << 3) ^
           (static_cast<size_t>(k.ps_user_sgprs) << 9) ^
           (static_cast<size_t>(k.ps_input_ena) << 15) ^
           (k.gl_clip ? 0x9e3779b9u : 0u);
  }
};
std::unordered_map<ShaderKey, gcn::Recompiled, ShaderKeyHash> g_sh_cache;

uint32_t UserSgprCount(uint32_t rsrc2) {
  const uint32_t count = ((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5);
  return std::min(count, 32u);
}

// IT_DISPATCH_DIRECT: body = [dim_x, dim_y, dim_z, initiator] workgroup counts.
// The CS program address, workgroup shape, RSRC and user data come from the
// COMPUTE_* SH registers programmed before it.
const char* const kEncName[19] = {"?",     "sop1", "sop2", "sopk", "sopc",
                                  "sopp",  "smem", "vop1", "vop2", "vop3",
                                  "vop3p", "vopc", "vint", "ds",   "mubuf",
                                  "mtbuf", "mimg", "exp",  "flat"};

// The workgroup shape and the RSRC2-derived launch state are baked into the
// recompiled module, so the code address alone cannot key the cache: the same
// CS can legally be re-dispatched with a different shape.
struct ComputeShaderKey {
  uint64_t address = 0;
  uint32_t thread_x = 0, thread_y = 0, thread_z = 0;
  uint32_t user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;

  bool operator==(const ComputeShaderKey& other) const = default;
};

struct ComputeShaderKeyHash {
  size_t operator()(const ComputeShaderKey& key) const {
    size_t hash = std::hash<uint64_t>{}(key.address);
    const auto combine = [&](uint32_t value) {
      hash ^= std::hash<uint32_t>{}(value) + 0x9e3779b9u + (hash << 6) +
              (hash >> 2);
    };
    combine(key.thread_x);
    combine(key.thread_y);
    combine(key.thread_z);
    combine(key.user_sgpr);
    combine(key.tgid_enable);
    combine(key.lds_dwords);
    return hash;
  }
};

// A dispatch dropped here is indistinguishable from one the title never issued
// (its writes just never appear), so every skip is reported. Rate-limited
// across all reasons: a title that dispatches every frame would otherwise
// flood the log.
bool CsReport() {
  static int n = 0;
  return n++ < 32;
}

void HandleDispatch(const uint32_t* body, uint32_t count) {
  const uint32_t dim_x = count >= 1 ? body[0] : 0;
  const uint32_t dim_y = count >= 2 ? body[1] : 0;
  const uint32_t dim_z = count >= 3 ? body[2] : 0;
  const uint64_t cs_addr =
      (static_cast<uint64_t>(g_regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
       g_regs[mmCOMPUTE_PGM_LO])
      << 8;
  const uint32_t tgx = g_regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF;
  const uint32_t tgy = g_regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF;
  const uint32_t tgz = g_regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF;
  const uint32_t rsrc2 = g_regs[mmCOMPUTE_PGM_RSRC2];
  const uint32_t user_sgpr = (rsrc2 >> 1) & 0x1F;
  const uint32_t tgid_enable = (rsrc2 >> 7) & 0x7;
  const uint32_t lds_dwords = (rsrc2 >> 15) & 0x1FF;

  static std::unordered_set<uint64_t> dumped_cs;
  if (kCsDump) {
    static uint64_t n_total = 0, n_valid = 0;
    static std::unordered_set<uint64_t> seen;
    n_total++;
    if (InGuest(cs_addr))
      n_valid++;
    seen.insert(cs_addr);
    if ((n_total % 2000) == 0) {
      std::fprintf(stderr, "[cs] dispatches=%lu valid=%lu unique=%zu:",
                   (unsigned long)n_total, (unsigned long)n_valid, seen.size());
      int shown = 0;
      for (uint64_t a : seen) {
        if (shown++ >= 8)
          break;
        std::fprintf(stderr, " %#lx", (unsigned long)a);
      }
      std::fprintf(stderr, " rsrc2=%08x tg=[%u %u %u]\n", rsrc2, tgx, tgy, tgz);
    }
  }
  constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
  if (kCsDump && dumped_cs.size() < 24 && InGuest(cs_addr) &&
      gpu::IsReadableRange(cs_addr, kMaxShaderBytes) &&
      dumped_cs.insert(cs_addr).second) {
    const uint32_t* ud = &g_regs[mmCOMPUTE_USER_DATA_0];
    std::fprintf(
        stderr, "[cs] addr=%#lx groups=[%u %u %u] tg=[%u %u %u] rsrc2=%08x\n",
        (unsigned long)cs_addr, dim_x, dim_y, dim_z, tgx, tgy, tgz, rsrc2);
    std::fprintf(stderr, "[cs]   user_data:");
    for (int k = 0; k < 16; k++)
      std::fprintf(stderr, " %08x", ud[k]);
    std::fprintf(stderr, "\n");
    // Follow each user-data pointer pair one level: the descriptor tables the
    // CS dereferences say which surfaces it actually reads and writes.
    for (int k = 0; k < 15; k++) {
      uint64_t p = (static_cast<uint64_t>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
      if (!GpuAddr(p) || !gpu::IsReadableRange(p, 8 * sizeof(uint32_t)))
        continue;
      const uint32_t* tw = reinterpret_cast<const uint32_t*>(p);
      std::fprintf(stderr, "[cs]   ud%d -> %#lx:", k, (unsigned long)p);
      for (int b = 0; b < 8; b++)
        std::fprintf(stderr, " %08x", tw[b]);
      std::fprintf(stderr, "\n");
    }
    // Encoding census: says which instruction families a compute backend must
    // cover before any of these dispatches can run.
    const auto prog =
        rdna::DecodeShader(reinterpret_cast<const uint32_t*>(cs_addr), 4096);
    uint32_t hist[24] = {};
    uint32_t flat_seg[4] = {}, flat_ops[128] = {};
    for (const auto& in : prog) {
      const uint32_t e = static_cast<uint32_t>(in.enc);
      if (e < 24)
        hist[e]++;
      if (in.enc == gcn::Enc::kFlat) {
        flat_seg[(in.raw[0] >> 14) & 3]++;
        flat_ops[(in.raw[0] >> 18) & 0x7F]++;
      }
    }
    std::fprintf(stderr, "[cs]   insts=%zu enc:", prog.size());
    for (uint32_t e = 0; e < 24; e++)
      if (hist[e])
        std::fprintf(stderr, " %u=%u", e, hist[e]);
    std::fprintf(stderr, " flatseg: %u/%u/%u/%u ops:", flat_seg[0], flat_seg[1],
                 flat_seg[2], flat_seg[3]);
    for (uint32_t o = 0; o < 128; o++)
      if (flat_ops[o])
        std::fprintf(stderr, " %#x=%u", o, flat_ops[o]);
    std::fprintf(stderr, "\n");
  }
  if (!InGuest(cs_addr) || !tgx || !tgy || !dim_x || !dim_y)
    return;
  if (kNoCs || !rhi::DefaultRenderer().available() ||
      !gpu::IsReadableRange(cs_addr, kMaxShaderBytes))
    return;

  // Recompile the CS to a Vulkan compute pipeline (cached), resolve the guest
  // ranges its descriptors name, and run it on the shared compute backend.
  // Minecraft builds its UI vertex buffers this way, so a skipped dispatch
  // leaves the draws that read them fetching zeros.
  const ComputeShaderKey key{cs_addr,   tgx,         tgy,       tgz,
                             user_sgpr, tgid_enable, lds_dwords};
  static std::unordered_map<ComputeShaderKey, gcn::RecompiledCs,
                            ComputeShaderKeyHash>
      cs_cache;
  auto cached = cs_cache.find(key);
  if (cached == cs_cache.end()) {
    cached = cs_cache
                 .emplace(key, rdna::RecompileCompute(
                                   reinterpret_cast<const uint32_t*>(cs_addr),
                                   tgx, tgy, tgz, user_sgpr, tgid_enable,
                                   lds_dwords))
                 .first;
    if (!cached->second.ok && CsReport())
      std::fprintf(stderr,
                   "[csgpu] unsupported CS @%#lx groups=[%u %u %u] tg=[%u %u "
                   "%u] usgpr=%u -- dispatch skipped\n",
                   (unsigned long)cs_addr, dim_x, dim_y, dim_z, tgx, tgy, tgz,
                   user_sgpr);
  }
  gcn::RecompiledCs& rc = cached->second;
  if (!rc.ok)
    return;

  const uint32_t* ud = &g_regs[mmCOMPUTE_USER_DATA_0];
  const uint32_t ud_dwords = std::min(user_sgpr, 16u);
  rhi::ComputeInfo ci;
  ci.cs_addr = cs_addr;
  ci.groups[0] = dim_x;
  ci.groups[1] = dim_y;
  ci.groups[2] = dim_z;
  ci.recomp = &rc;
  for (int k = 0; k < 16; k++)
    ci.user_data[k] = ud[k];

  // Descriptors an SRT chain produced are not in the user-data window; replay
  // the shader's scalar ops to recover the one each resource's instruction
  // actually used.
  const auto cs_resources = rdna::ResolveBuffers(
      reinterpret_cast<const uint32_t*>(cs_addr), ud, ud_dwords, 0);

  constexpr uint64_t kMaxRes = 256ull * 1024 * 1024;  // per storage buffer
  bool res_ok = true;
  for (const gcn::CsResource& r : rc.resources) {
    const uint32_t dwords = r.kind == 1 ? 8u : r.kind == 2 ? 2u : 4u;
    // Compute seeds user data straight into s0.., so a plan naming an SGPR past
    // the loaded window names one an SRT load produced, which nothing here
    // replays.
    const uint32_t* desc = nullptr;
    if (r.base_sgpr + dwords <= ud_dwords) {
      desc = &ud[r.base_sgpr];
    } else if (const auto it = cs_resources.find(r.use_pc);
               it != cs_resources.end() && it->second.descriptor_valid &&
               it->second.descriptor_dwords >= dwords) {
      desc = it->second.descriptor;
    } else {
      if (CsReport())
        std::fprintf(stderr,
                     "[csgpu] CS @%#lx bind=%u kind=%u s%u pc=%#x is outside "
                     "the %u-dword user data and did not replay -- dispatch "
                     "skipped\n",
                     (unsigned long)cs_addr, r.binding, r.kind, r.base_sgpr,
                     r.use_pc, ud_dwords);
      res_ok = false;
      break;
    }

    uint64_t base = 0, size = 0, guest_size = 0;
    gcn::TImage image;
    bool image_staging = false;
    bool zero_fill = false;
    uint32_t elem_bytes = 4, stage_elem_bytes = 4;
    if (std::all_of(desc, desc + dwords, [](uint32_t w) { return w == 0; })) {
      // A null descriptor is a real binding on a path this launch does not
      // take; the translator guards it and reads zero.
      zero_fill = true;
      size = std::max<uint64_t>(r.min_bytes, 16);
    } else if (r.kind == 1) {
      const gcn::TImage t = rdna::DecodeTImage(desc);
      const bool rgba8 = t.dfmt == 10 && (t.nfmt == 0 || t.nfmt == 4);
      const bool r32 =
          t.dfmt == 4 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
      const bool rg16f = t.dfmt == 5 && t.nfmt == 7;
      const bool r16f = t.dfmt == 2 && t.nfmt == 7;
      const bool rg8 = t.dfmt == 3 && t.nfmt == 0;
      const bool rgba16f = t.dfmt == 12 && t.nfmt == 7;
      const bool r11g11b10f = t.dfmt == 6 && t.nfmt == 7;
      elem_bytes = rgba16f ? 8u : (r16f || rg8) ? 2u : 4u;
      stage_elem_bytes = r11g11b10f ? 16u : std::max(elem_bytes, 4u);
      // tiling_idx >= 0x100 is the T# decoder's "no detiler for this gfx10
      // swizzle mode" marker; staging one would scramble the texels it copies
      // back into guest memory.
      gcn::TextureLayout32 layout;
      if ((t.type != 9 && t.type != 13) ||
          !(rgba8 || r32 || rg16f || r16f || rg8 || rgba16f || r11g11b10f) ||
          t.tiling_idx >= 0x100 || !t.valid ||
          !gcn::BuildTextureLayout32(layout, t.width, t.height, t.pitch,
                                     t.layers, t.mip_levels, t.tiling_idx,
                                     t.pow2_pad, elem_bytes)) {
        if (CsReport())
          std::fprintf(stderr,
                       "[csgpu] CS @%#lx bind=%u unsupported image base=%#lx "
                       "type=%u dfmt=%u nfmt=%u tiling=%#x %ux%u pitch=%u "
                       "-- dispatch skipped\n",
                       (unsigned long)cs_addr, r.binding, (unsigned long)t.base,
                       t.type, t.dfmt, t.nfmt, t.tiling_idx, t.width, t.height,
                       t.pitch);
        res_ok = false;
        break;
      }
      base = t.base;
      guest_size = layout.size;
      image_staging = !gcn::TilingIsLinear(t.tiling_idx) ||
                      elem_bytes != stage_elem_bytes;
      if (image_staging) {
        gcn::TextureLayout32 linear;
        if (!gcn::BuildTextureLayout32(linear, t.width, t.height, t.pitch,
                                       t.layers, t.mip_levels, 8, t.pow2_pad,
                                       stage_elem_bytes)) {
          if (CsReport())
            std::fprintf(stderr,
                         "[csgpu] CS @%#lx bind=%u image %ux%u has no linear "
                         "staging layout -- dispatch skipped\n",
                         (unsigned long)cs_addr, r.binding, t.width, t.height);
          res_ok = false;
          break;
        }
        size = linear.size;
        image = t;
      } else {
        size = guest_size;
      }
    } else if (r.kind == 2) {  // raw pointer into an SRT/descriptor table
      base = (static_cast<uint64_t>(desc[1] & 0xFFFF) << 32) | desc[0];
      size = r.min_bytes;
    } else {  // buffer V#
      const VBuffer v = DecodeVBuffer(desc);
      base = v.base;
      size = v.stride ? static_cast<uint64_t>(v.stride) * v.num_records
                      : v.num_records;
      size = std::max<uint64_t>(size, r.min_bytes);
    }
    if (!guest_size && !zero_fill)
      guest_size = size;
    if (kResTrace)
      std::fprintf(stderr,
                   "[csres] cs=%#lx bind=%u kind=%u s%u pc=%#x base=%#lx "
                   "size=%#lx guest=%#lx written=%d zero=%d\n",
                   (unsigned long)cs_addr, r.binding, r.kind, r.base_sgpr,
                   r.use_pc, (unsigned long)base, (unsigned long)size,
                   (unsigned long)guest_size, r.written ? 1 : 0,
                   zero_fill ? 1 : 0);
    // A dispatch writes guest memory, so a range that does not check out skips
    // the whole dispatch rather than binding something wrong.
    if (!zero_fill &&
        (size < r.min_bytes || size > kMaxRes || guest_size > kMaxRes ||
         !GpuAddr(base) || !gpu::IsReadableRange(base, guest_size))) {
      if (CsReport())
        std::fprintf(stderr,
                     "[csgpu] CS @%#lx bind=%u kind=%u unusable range "
                     "base=%#lx size=%#lx -- dispatch skipped\n",
                     (unsigned long)cs_addr, r.binding, r.kind,
                     (unsigned long)base, (unsigned long)guest_size);
      res_ok = false;
      break;
    }
    if (ci.num_res >= rhi::ComputeInfo::kMaxResources) {
      if (CsReport())
        std::fprintf(stderr,
                     "[csgpu] CS @%#lx needs more than %u resources -- "
                     "dispatch skipped\n",
                     (unsigned long)cs_addr, rhi::ComputeInfo::kMaxResources);
      res_ok = false;
      break;
    }
    rhi::ComputeInfo::Res& out = ci.res[ci.num_res];
    out.base = base;
    out.size = size;
    out.guest_size = guest_size;
    out.binding = r.binding;
    out.shader_writes = r.written;
    out.written = r.written && !zero_fill;
    out.zero_fill = zero_fill;
    out.image_staging = image_staging;
    if (image_staging) {
      out.width = image.width;
      out.height = image.height;
      out.pitch = image.pitch;
      out.layers = image.layers;
      out.mip_levels = image.mip_levels;
      out.tiling_idx = image.tiling_idx;
      out.elem_bytes = elem_bytes;
      out.stage_elem_bytes = stage_elem_bytes;
      out.dfmt = image.dfmt;
      out.pow2_pad = image.pow2_pad;
    }
    ci.num_res++;
  }
  if (!res_ok || !ci.num_res)
    return;
  const bool dispatched = rhi::Dispatch(rhi::DefaultRenderer(), ci);
  if (!dispatched && CsReport())
    std::fprintf(stderr, "[csgpu] CS @%#lx dispatch failed (%u resources)\n",
                 (unsigned long)cs_addr, ci.num_res);
  if (kResTrace)
    std::fprintf(stderr, "[csres] cs=%#lx dispatch %s (%u resources)\n",
                 (unsigned long)cs_addr, dispatched ? "executed" : "failed",
                 ci.num_res);
}

// DELTA_AGC_DUMPSH=<hexaddr>: decode and print one shader by address, once.
// Shader dumps are otherwise tied to recompile time or to a draw index, neither
// of which is reachable for a steady-state pass without a full trace.
void MaybeDumpShader(uint64_t addr) {
  static bool done = false;
  constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
  if (!kDumpSh || done || addr != kDumpSh || !InGuest(addr) ||
      !gpu::IsReadableRange(addr, kMaxShaderBytes))
    return;
  done = true;
  const auto prog =
      rdna::DecodeShader(reinterpret_cast<const uint32_t*>(addr), 4096);
  std::fprintf(stderr, "[agc] SHADER %#lx: %zu insts\n", (unsigned long)addr,
               prog.size());
  for (const auto& in : prog) {
    std::fprintf(stderr, "[agc]  pc=%04x %-6s op=%#05x %08x", in.pc,
                 kEncName[static_cast<uint32_t>(in.enc) < 19
                              ? static_cast<uint32_t>(in.enc)
                              : 0],
                 in.opcode, in.raw[0]);
    if (in.size >= 2)
      std::fprintf(stderr, " %08x", in.raw[1]);
    if (in.has_literal)
      std::fprintf(stderr, " lit=%08x", in.literal);
    std::fprintf(stderr, "\n");
  }
}

// The T#'s four DST_SEL channel selects, packed 3 bits each for the renderer's
// image view. The identity selection (R,G,B,A) packs to 0 so views that need no
// swizzle keep sharing one cache entry.
static uint32_t PackDstSel(const gcn::TImage& t) {
  const uint32_t p = (t.dst_sel[0] & 7) | ((t.dst_sel[1] & 7) << 3) |
                     ((t.dst_sel[2] & 7) << 6) | ((t.dst_sel[3] & 7) << 9);
  return p == (4u | (5u << 3) | (6u << 6) | (7u << 9)) ? 0u : p | (1u << 12);
}

// A context register read as the float it holds (viewport scales/offsets).
static float RegF(uint32_t reg) {
  float f;
  std::memcpy(&f, &g_regs[reg], 4);
  return f;
}

// Last packet that changed CB_COLOR0_BASE (see DELTA_AGC_RTPROBE).
static uint32_t g_cb0_op = 0, g_cb0_val = 0;
static uint64_t g_cb0_draw = 0;

static void DrawCensus() {
  if (!kGpuDrawcensus)
    return;
  static const bool kStarted = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        std::fprintf(stderr,
                     "[drawcensus] seen=%llu issued=%llu dropped: no-rt=%llu "
                     "no-shader=%llu\n",
                     (unsigned long long)g_draws_seen.load(),
                     (unsigned long long)g_draws_issued.load(),
                     (unsigned long long)g_drop_no_rt.load(),
                     (unsigned long long)g_drop_no_shader.load());
      }
    }).detach();
    return true;
  }();
  (void)kStarted;
}

void HandleDraw(uint32_t op, const uint32_t* body, uint32_t count) {
  size_t drop_attr_count = 0;
  g_draws_seen.fetch_add(1, std::memory_order_relaxed);
  DrawCensus();
  if (!rhi::DefaultRenderer().available())
    return;

  // gfx10.3 has no HW VS: the vertex program is the merged NGG shader, whose
  // address is written to the ES (front half, 0xC8) and/or GS (back half, 0x88)
  // PGM_LO. Some pipelines populate only the ES slot, so fall back to it when
  // the GS slot reads 0. User data (cbuffer/MVP pointers) stays in the GS
  // block.
  uint64_t vs_a = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_GS);
  if (!InGuest(vs_a))
    vs_a = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES);
  uint64_t ps_a = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  MaybeDumpShader(vs_a);
  MaybeDumpShader(ps_a);
  // DELTA_PS5_SKIPVS=hexaddr: drop draws using this VS (draw-isolation bisect).
  if (kSkipVs && vs_a == kSkipVs)
    return;
  const uint32_t* vud = &g_regs[mmSPI_SHADER_USER_DATA_GS_0];
  const uint32_t* pud = &g_regs[mmSPI_SHADER_USER_DATA_PS_0];

  static int s_uddump = 0;
  if (kTrace && s_uddump < 4) {
    s_uddump++;
    const uint32_t* gs = &g_regs[mmSPI_SHADER_USER_DATA_GS_0];
    const uint32_t* es = &g_regs[mmSPI_SHADER_USER_DATA_ES_0];
    std::fprintf(stderr, "[agc]   UD GS:");
    for (int j = 0; j < 16; j++)
      std::fprintf(stderr, " %08x", gs[j]);
    std::fprintf(stderr, "\n[agc]   UD ES:");
    for (int j = 0; j < 16; j++)
      std::fprintf(stderr, " %08x", es[j]);
    std::fprintf(stderr, "\n");
  }

  // This title programs the shader PGM_LO/HI at non-standard SH offsets (via
  // the inline op 0x93), so the fixed GS/PS regs above read 0. Fallback: scan
  // the SH register file for PGM pairs whose address is a 256-aligned
  // GPU-aperture pointer to plausible RDNA2 ISA (first dword's top byte is a
  // scalar/vector encoding).
  if (!InGuest(vs_a) || !InGuest(ps_a)) {
    uint64_t found[16];
    uint32_t found_reg[16];
    int nf = 0;
    // Scan the SH register block for a PGM pair pointing at RDNA2 shader ISA
    // (both PGM encodings). NOTE (session 2): proven that NO register in the
    // whole file points at shader code -- the AGC binds shaders outside the PM4
    // stream -- so this stays inert until that path is decoded.
    auto try_addr = [&](uint32_t o, uint64_t a) {
      if (nf >= 16 || !GpuAddr(a) || (a & 0xFF) ||
          !gpu::IsReadableRange(a, sizeof(uint32_t)))
        return;
      uint32_t w0 = *reinterpret_cast<const uint32_t*>(a);
      if ((w0 >> 24) < 0x7e || w0 == 0xffffffffu)
        return;  // ISA plausibility
      for (int k = 0; k < nf; k++)
        if (found[k] == a)
          return;
      found_reg[nf] = o;
      found[nf++] = a;
    };
    for (uint32_t o = 0; o + 1 < 0x300; o++) {
      uint32_t lo = g_regs[kShRegBase + o], hi = g_regs[kShRegBase + o + 1];
      try_addr(
          o, (static_cast<uint64_t>(lo) << 8) | ((uint64_t)(hi & 0xFF) << 40));
      try_addr(o, (static_cast<uint64_t>(hi & 0xFFFF) << 32) | lo);
    }
    // The op 0x93 writes SH reg 0x113 = a GPU ptr (raw (HI<<32)|LO). PS5
    // shaders have a metadata HEADER before the ISA, so the first dword isn't
    // an opcode. Dump it deeply to locate a shader binary + the ISA offset.
    static int s_hdr = 0;
    uint64_t a113 =
        (static_cast<uint64_t>(g_regs[kShRegBase + 0x114] & 0xFFFF) << 32) |
        g_regs[kShRegBase + 0x113];
    if (kTrace && s_hdr < 3 && GpuAddr(a113) &&
        gpu::IsReadableRange(a113, 32 * sizeof(uint32_t))) {
      s_hdr++;
      auto* w = reinterpret_cast<const uint32_t*>(a113);
      std::fprintf(stderr,
                   "[agc]   reg0x113 -> %#lx dump:", (unsigned long)a113);
      for (int j = 0; j < 32; j++)
        std::fprintf(stderr, " %08x", w[j]);
      std::fprintf(stderr, "\n");
    }
    if (nf >= 1 && !InGuest(vs_a))
      vs_a = found[0];
    if (nf >= 2 && !InGuest(ps_a))
      ps_a = found[1];
    static int s_sc = 0;
    if (kTrace && s_sc < 6 && nf) {
      s_sc++;
      std::fprintf(stderr, "[agc]   shader scan: nf=%d", nf);
      for (int k = 0; k < nf && k < 6; k++)
        std::fprintf(stderr, " [%#x]=%#lx", found_reg[k],
                     (unsigned long)found[k]);
      std::fprintf(stderr, "\n");
    }
    // The AGC binds shaders via a pipeline/descriptor, not PGM regs. Dump the
    // GS/PS user-data (16 dwords each) and follow any GPU-aperture pointer one
    // level to look for shader ISA -- the pipeline handle/PGM likely lives in a
    // descriptor.
    static int s_ud = 0;
    if (kTrace && s_ud < 3) {
      s_ud++;
      for (int which = 0; which < 2; which++) {
        const uint32_t* ud = which ? pud : vud;
        std::fprintf(stderr, "[agc]   %sUD:", which ? "ps" : "gs");
        for (int k = 0; k < 16; k++)
          std::fprintf(stderr, " %08x", ud[k]);
        std::fprintf(stderr, "\n");
        for (int k = 0; k + 1 < 16; k++) {
          uint64_t p =
              (static_cast<uint64_t>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
          if (GpuAddr(p) && gpu::IsReadableRange(p, 8 * sizeof(uint32_t))) {
            auto* pw = reinterpret_cast<const uint32_t*>(p);
            std::fprintf(stderr, "[agc]     UD[%d]->%#lx:", k,
                         (unsigned long)p);
            for (int j = 0; j < 8; j++)
              std::fprintf(stderr, " %08x", pw[j]);
            std::fprintf(stderr, "\n");
          }
        }
      }
    }
  }

  rhi::DrawInfo d;
  std::memcpy(d.vs_user_data, vud, sizeof(d.vs_user_data));
  std::memcpy(d.ps_user_data, pud, sizeof(d.ps_user_data));
  d.prim_type = g_regs[mmVGT_PRIMITIVE_TYPE];
  d.instance_count = g_num_instances;
  uint32_t auto_vertex_count =
      op == IT_DRAW_INDEX_AUTO && count >= 1 ? body[0] : 0;

  // Index buffer. DRAW_INDEX_2 carries the base inline (maxSize, baseLo,
  // baseHi, index_count, initiator); DRAW_INDEX_OFFSET_2 (maxSize,
  // index_offset, index_count, initiator) indexes the buffer INDEX_BASE last
  // set, starting index_offset indices in. Minecraft draws its world geometry
  // almost entirely with the latter (291k packets a run against 33k AUTO), and
  // leaving it undecoded left every one of those draws with no index buffer and
  // a vertex count taken from the V#'s num_records -- a shared ~210k-record
  // ring -- so they all tripped the vertex-count cap and were dropped.
  const uint64_t index_size =
      g_index_type == 1 ? 4 : g_index_type == 2 ? 1 : 2;
  if (op == IT_DRAW_INDEX_OFFSET_2 && count >= 3 && g_index_base) {
    const uint32_t ioff = body[1], icount = body[2];
    const uint64_t ibase = g_index_base + static_cast<uint64_t>(ioff) * index_size;
    if (InGuest(ibase) && icount && icount <= 0x100000 &&
        (!g_index_max || static_cast<uint64_t>(ioff) + icount <= g_index_max) &&
        gpu::IsReadableRange(ibase, icount * index_size)) {
      d.index_data = reinterpret_cast<const void*>(ibase);
      d.index_count = icount;
      d.index_type = g_index_type;
    }
  }
  if (op == IT_DRAW_INDEX_2 && count >= 4) {
    uint64_t ibase = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
    uint32_t icount = body[3];
    if (InGuest(ibase) && icount && icount <= 0x100000 &&
        gpu::IsReadableRange(ibase, icount * index_size)) {
      d.index_data = reinterpret_cast<const void*>(ibase);
      d.index_count = icount;
      d.index_type = g_index_type;
    }
  }
  if (kTrace && d.index_data) {
    static int s_idxdump = 0;
    const uint64_t ibase = reinterpret_cast<uint64_t>(d.index_data);
    const uint64_t dump_bytes = std::min(d.index_count, 8u) * index_size;
    if (s_idxdump < 12 && gpu::IsReadableRange(ibase, dump_bytes)) {
      s_idxdump++;
      const uint16_t* i16 = reinterpret_cast<const uint16_t*>(ibase);
      const uint32_t* i32 = reinterpret_cast<const uint32_t*>(ibase);
      std::fprintf(stderr, "[agc]   IDX op=%#x ibase=%#lx count=%u type=%u:", op,
                   (unsigned long)ibase, d.index_count, g_index_type);
      for (uint32_t k = 0; k < d.index_count && k < 8; k++)
        std::fprintf(stderr, " %u",
                     g_index_type == 1 ? i32[k] : (uint32_t)i16[k]);
      std::fprintf(stderr, "\n");
    }
  }

  d.rt_w = FbWidth();
  d.rt_h = FbHeight();

  // Color targets: bind CB_COLORn only when its write mask and INFO format are
  // valid (a stale base remains programmed during depth-only passes).
  {
    uint32_t tmask = g_regs[mmCB_TARGET_MASK];
    for (int rt = 0; rt < 8; rt++) {
      uint64_t base = g_regs.CbColorBase(rt);
      uint32_t info = g_regs[mmCB_COLOR0_INFO + rt * kCbColorStride];
      if (((tmask >> (rt * 4)) & 0xF) && ((info >> 2) & 0x1F) &&
          InGuest(base)) {
        d.mrt_base[rt] = base;
        d.mrt_info[rt] = info;
        d.mrt_count = rt + 1;
      }
    }
    d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;
    // A draw whose target mask asks for colour but whose CB_COLOR0_INFO reads
    // zero keeps the last target that was valid. Skyrim's logo and menu passes
    // are bound through a path we do not see yet (the driver's default-state
    // block zeroes CB_COLOR0 and only some passes get a re-bind), and dropping
    // them entirely is certainly wrong where reusing the target is only maybe.
    // DELTA_GPU_NOSTICKYRT restores the drop.
    static uint64_t last_base = 0;
    static uint32_t last_info = 0, last_w = 0, last_h = 0;
    if (d.mrt_count) {
      last_base = d.mrt_base[0];
      last_info = d.mrt_info[0];
      last_w = d.rt_w;
      last_h = d.rt_h;
    } else if (!kNoStickyRt && (g_regs[mmCB_TARGET_MASK] & 0xF) && last_base &&
               d.rt_w == last_w && d.rt_h == last_h) {
      d.mrt_base[0] = last_base;
      d.mrt_info[0] = last_info;
      d.mrt_count = 1;
      d.rt_base = last_base;
    }
    // DELTA_AGC_RTPROBE: one line per draw naming the packet that last touched
    // CB_COLOR0_BASE. A draw with no target and a stale "last write" points at
    // a bind we never executed; one whose last write zeroed the base points at
    // a packet we execute but should not.
    static int s_probe = 0;
    if (kRtProbe && s_probe < 200) {
      s_probe++;
      std::fprintf(
          stderr,
          "[agc] RTPROBE draw#%lu rt=%#lx tmask=%#x info0=%#x clip=%#x "
          "blend=%u ctl=%#x cc=%#x tex0=%#lx ntex=%u\n",
          (unsigned long)g_draws_seen.load(), (unsigned long)d.rt_base,
          g_regs[mmCB_TARGET_MASK], g_regs[mmCB_COLOR0_INFO],
          g_regs[mmPA_CL_CLIP_CNTL], (g_regs[mmCB_BLEND0_CONTROL] >> 30) & 1,
          g_regs[mmCB_BLEND0_CONTROL], g_regs[mmCB_COLOR_CONTROL],
          (unsigned long)(d.num_texs ? d.texs[0].base : 0), d.num_texs);
    }
    // Why a colour draw ended up with no target: print the state that rejected
    // CB_COLOR0 (a stale/zero base, or an INFO with no format).
    static int s_nort = 0;
    if (kTrace && !d.mrt_count && (tmask & 0xF) && s_nort < 12) {
      s_nort++;
      std::fprintf(
          stderr, "[agc]   NO-RT tmask=%#x cb0Base=%#lx info0=%#x fmt=%u\n",
          tmask, (unsigned long)g_regs.CbColorBase(0), g_regs[mmCB_COLOR0_INFO],
          (g_regs[mmCB_COLOR0_INFO] >> 2) & 0x1F);
    }
    static int s_rtdbg = 0;
    if (kTrace && s_rtdbg < 12) {
      s_rtdbg++;
      std::fprintf(
          stderr,
          "[agc]   DRAW rt: cb0Base=%#lx info0=%#x tmask=%#x -> mrt_count=%u "
          "vs_a=%#lx ps_a=%#lx prim=%#x idx=%u\n",
          (unsigned long)g_regs.CbColorBase(0), g_regs[mmCB_COLOR0_INFO],
          g_regs[mmCB_TARGET_MASK], d.mrt_count, (unsigned long)vs_a,
          (unsigned long)ps_a, d.prim_type, d.index_count);
    }
  }

  // Per-MRT blend (CB_BLENDn_CONTROL, bit 30 = enable).
  d.blend_control = g_regs[mmCB_BLEND0_CONTROL];
  d.blend_enable = (d.blend_control >> 30) & 1u;
  d.mrt_blend[0] = d.blend_control;
  if (d.blend_enable)
    d.mrt_blend_mask |= 1u;
  for (uint32_t rt = 1; rt < 8; rt++) {
    uint32_t bc = g_regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
    d.mrt_blend[rt] = bc;
    if ((bc >> 30) & 1u)
      d.mrt_blend_mask |= (1u << rt);
  }
  d.target_mask = g_regs[mmCB_TARGET_MASK];
  d.color_control = g_regs[mmCB_COLOR_CONTROL];

  // Depth/stencil (gfx10 Z base = (WRITE_BASE | WRITE_BASE_HI<<32) << 8).
  // Mirrors the PS4 path (gpu/ps4/cmd_processor.cc): DELTA_GPU_NODEPTH is the
  // shared kill switch, otherwise the title's own DB_Z_INFO/DB_DEPTH_CONTROL
  // state decides. 2D titles (Isaac) leave DB_Z_INFO's format field invalid so
  // depth_valid stays false and no depth attachment binds (unchanged 2D path).
  {
    uint32_t dc = g_regs[mmDB_DEPTH_CONTROL];
    uint32_t zinfo = kNoDepth ? 0 : g_regs[mmDB_Z_INFO];
    uint64_t zbase =
        ((static_cast<uint64_t>(g_regs[mmDB_Z_WRITE_BASE_HI]) << 32) |
         g_regs[mmDB_Z_WRITE_BASE])
        << 8;
    d.depth_valid = (zinfo & 0x3) != 0;
    if (d.depth_valid && InGuest(zbase) &&
        (((dc >> 1) & 1u) || ((dc >> 2) & 1u))) {
      d.depth_base = zbase;
      d.depth_test_enable = (dc >> 1) & 1u;
      d.depth_write_enable = (dc >> 2) & 1u;
      d.depth_func = (dc >> 4) & 0x7;
      std::memcpy(&d.depth_clear, &g_regs[mmDB_DEPTH_CLEAR], 4);
      if (!(d.depth_clear >= 0.0f && d.depth_clear <= 1.0f))
        d.depth_clear = 1.0f;
    } else {
      d.depth_valid = false;
    }
  }

  // Primitive setup + viewport.
  {
    uint32_t sc = g_regs[mmPA_SU_SC_MODE_CNTL];
    d.cull_mode = sc & 0x3;
    d.front_ccw = ((sc >> 2) & 1u) == 0;
  }
  std::memcpy(&d.viewport_x_scale, &g_regs[mmPA_CL_VPORT_XSCALE], 4);
  std::memcpy(&d.viewport_x_offset, &g_regs[mmPA_CL_VPORT_XOFFSET], 4);
  std::memcpy(&d.viewport_y_scale, &g_regs[mmPA_CL_VPORT_YSCALE], 4);
  std::memcpy(&d.viewport_y_offset, &g_regs[mmPA_CL_VPORT_YOFFSET], 4);
  // Titles whose context state arrives as AGC state blocks never program
  // PA_CL_VPORT_*, and SetGuestViewport rejects a zero scale -- so no viewport
  // is ever set and every draw rasterises nothing. Default to the bound
  // target's full extent, y-up (negative height) like a programmed one, so
  // RT-as-texture composites still sample aligned.
  if (!(d.viewport_x_scale > 0.0f) && d.rt_w && d.rt_h) {
    d.viewport_x_scale = d.rt_w * 0.5f;
    d.viewport_x_offset = d.rt_w * 0.5f;
    d.viewport_y_scale = d.rt_h * -0.5f;
    d.viewport_y_offset = d.rt_h * 0.5f;
  }

  const uint32_t gs_user_sgprs =
      UserSgprCount(g_regs[mmSPI_SHADER_PGM_RSRC2_GS]);
  const uint32_t ps_user_sgprs =
      UserSgprCount(g_regs[mmSPI_SHADER_PGM_RSRC2_PS]);
  // Fetch-shader pointer (a heuristic default: GS user data[0..1]; the AGC
  // input-usage table is authoritative and a follow-up).
  const uint64_t fetch =
      gs_user_sgprs >= 2
          ? (static_cast<uint64_t>(vud[1] & 0xFFFF) << 32) | vud[0]
          : 0;

  // Recompile the VS/PS pair (cached) and resolve the live vertex-attribute
  // buffers + constant buffers from the RDNA2 descriptors in user data.
  static uint64_t s_dl_n = 0;
  uint64_t my_draw = s_dl_n++;
  bool dl = kTrace && s_dl_n < 5000;
  if (kRecompOn && InGuest(vs_a) && (!ps_a || InGuest(ps_a))) {
    constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
    if (!gpu::IsReadableRange(vs_a, kMaxShaderBytes) ||
        (ps_a && !gpu::IsReadableRange(ps_a, kMaxShaderBytes)))
      return;
    // DX_CLIP_SPACE_DEF (bit 19) picks the guest's clip-z convention.
    const bool gl_clip = !((g_regs[mmPA_CL_CLIP_CNTL] >> 19) & 1);
    ShaderKey key{vs_a,          ps_a,          fetch,
                  gs_user_sgprs, ps_user_sgprs, g_regs[mmSPI_PS_INPUT_ENA],
                  gl_clip};
    auto it = g_sh_cache.find(key);
    if (it == g_sh_cache.end()) {
      if (dl) {
        const uint32_t* vc = reinterpret_cast<const uint32_t*>(vs_a);
        const uint32_t* pc =
            ps_a ? reinterpret_cast<const uint32_t*>(ps_a) : nullptr;
        // AGC shader code starts with the 0xBEEB03FF sentinel; a ps_a that
        // isn't (e.g. 0xffc9dfe7 poison fill) means the PS PGM_LO reg didn't
        // land.
        std::fprintf(
            stderr,
            "[agc] DL recompile vs=%#lx (%08x) ps=%#lx (%08x %08x)...\n",
            (unsigned long)vs_a, vc[0], (unsigned long)ps_a, pc ? pc[0] : 0,
            pc ? pc[1] : 0);
        std::fprintf(stderr,
                     "[agc] DL psInputEna=%#x (frag-coord/face VGPR seed)\n",
                     g_regs[mmSPI_PS_INPUT_ENA]);
      }
      it = g_sh_cache
               .emplace(key, rdna::Recompile(
                                 reinterpret_cast<const uint32_t*>(vs_a),
                                 ps_a ? reinterpret_cast<const uint32_t*>(ps_a)
                                      : nullptr,
                                 vud, pud, g_regs[mmSPI_PS_INPUT_ENA], gl_clip,
                                 gs_user_sgprs, ps_user_sgprs))
               .first;
      if (dl)
        std::fprintf(stderr, "[agc] DL recompile done ok=%d\n", it->second.ok);
      // A shader we cannot recompile drops its draw entirely, which is
      // indistinguishable from "the title never issued it" unless it is said
      // out loud. Report each failing pair once.
      if (!it->second.ok) {
        static int failed = 0;
        if (failed++ < 32)
          std::fprintf(
              stderr,
              "[agc] recompile FAILED vs=%#lx ps=%#lx -- draw dropped\n",
              (unsigned long)vs_a, (unsigned long)ps_a);
      }
    }
    gcn::Recompiled& rc = it->second;
    if (rc.ok) {
      // Vertex attributes: the V# is either inline in user data at table_sgpr
      // (AGC frequently passes the vertex V# directly) or reached through a
      // table pointer at {table_sgpr, +1}. Resolve each attr's V#, then group
      // the attrs into vertex bindings: same-stride V#s within one stride of
      // each other interleave in one binding; others get their own binding (the
      // textured sprite VS streams pos/color and uv/params from two buffers, so
      // a single interleaved binding would feed the PS garbage UVs). Attrs that
      // don't decode to a valid V# are skipped (a partial fetch still
      // rasterizes).
      VBuffer attr_vbs[8];
      const gcn::ShaderAttr* attr_res[8];
      uint32_t attr_n = 0;
      // The merged ES/GS NGG vertex shader reads its GS user data starting at
      // wave SGPR udBase (sgpr 0..udBase-1 are ES/system), but the AGC latches
      // it into SPI_SHADER_USER_DATA_GS_0 which we index from 0, so shift
      // shader SGPR N -> user_data[N-udBase] for both attrs and cbufs. Defaults
      // to 8 (the observed merged-NGG layout); DELTA_PS5_UDBASE overrides.
      // TODO: derive from RSRC2.
      const auto vs_resources =
          rdna::ResolveBuffers(reinterpret_cast<const uint32_t*>(vs_a), vud,
                               gs_user_sgprs, kUdBase);
      if (dl)
        std::fprintf(stderr,
                     "[agc] DL attrs=%zu gsUd=%u res=%zu vud[0..7]=%08x %08x "
                     "%08x %08x %08x %08x %08x %08x\n",
                     rc.attrs.size(), gs_user_sgprs, vs_resources.size(),
                     vud[0], vud[1], vud[2], vud[3], vud[4], vud[5], vud[6],
                     vud[7]);
      for (size_t i = 0; i < rc.attrs.size() && i < 8; i++) {
        auto& a = rc.attrs[i];
        VBuffer vb{};
        uint32_t fetch_soffset = 0;
        const char* how = "replay";
        if (a.use_pc != ~0u) {
          const auto resolved = vs_resources.find(a.use_pc);
          if (dl)
            std::fprintf(stderr,
                         "[agc]   attr%zu use_pc=%#x found=%d valid=%d\n", i,
                         a.use_pc, resolved != vs_resources.end(),
                         resolved != vs_resources.end()
                             ? (int)resolved->second.descriptor_valid
                             : -1);
          if (resolved == vs_resources.end() ||
              !resolved->second.descriptor_valid)
            continue;
          vb = DecodeVBuffer(resolved->second.descriptor);
          fetch_soffset = resolved->second.soffset;
        } else {
          const uint32_t ti = a.table_sgpr >= kUdBase
                                  ? a.table_sgpr - kUdBase
                                  : a.table_sgpr;
          if (ti + 3 >= gs_user_sgprs)
            continue;
          vb = DecodeVBuffer(&vud[ti]);
          how = "inline";
          if (!InGuest(vb.base) || !PlausibleVb(vb)) {
            const uint64_t tbl =
                (static_cast<uint64_t>(vud[ti + 1] & 0xFFFF) << 32) | vud[ti];
            const uint64_t tbl_addr = tbl + a.vbuf_dword_off * 4;
            if (InGuest(tbl) && gpu::IsReadableRange(tbl_addr, 16)) {
              vb = DecodeVBuffer(reinterpret_cast<const uint32_t*>(tbl_addr));
              how = "table";
            }
          }
        }
        if (dl)
          std::fprintf(
              stderr,
              "[agc]   attr%zu loc=%u nc=%u tbl_sgpr=%u off=%u ioff=%u soff=%u "
              "(%s) -> base=%#lx stride=%u nrec=%u gfmt=%u -> dfmt=%u "
              "nfmt=%u\n",
              i, a.location, a.num_comps, a.table_sgpr, a.vbuf_dword_off,
              a.inst_offset, fetch_soffset, how, (unsigned long)vb.base,
              vb.stride, vb.num_records, vb.gfmt, vb.dfmt, vb.nfmt);
        if (!InGuest(vb.base) || !PlausibleVb(vb))
          continue;  // unresolved: keep the rest
        // Where this attribute sits inside the vertex. The V# only names the
        // stream: the fetch adds its own byte offset, either as the immediate
        // or (Sony's compiler) through the soffset scalar. Fold it into the
        // base so the binding grouping below turns it back into a per-attribute
        // offset.
        const uint64_t field_off = a.inst_offset + fetch_soffset;
        if (field_off < vb.stride)
          vb.base += field_off;
        attr_vbs[attr_n] = vb;
        attr_res[attr_n++] = &a;
      }
      // Group the resolved attrs into bindings (mirrors the PS4 recomp path).
      uint32_t attr_binding[8] = {};
      for (uint32_t i = 0; i < attr_n; i++) {
        const VBuffer& vb = attr_vbs[i];
        int sel = -1;
        for (uint32_t j = 0; j < d.num_vbufs; j++) {
          if (d.vbufs[j].stride != vb.stride)
            continue;
          uint64_t b = reinterpret_cast<uint64_t>(d.vbufs[j].data);
          uint64_t lo = b < vb.base ? b : vb.base;
          uint64_t hi = b < vb.base ? vb.base : b;
          if (hi - lo < vb.stride) {
            sel = static_cast<int>(j);
            break;
          }
        }
        if (sel < 0) {
          if (d.num_vbufs >= 8)
            break;
          sel = static_cast<int>(d.num_vbufs);
          d.vbufs[d.num_vbufs++] = {reinterpret_cast<const void*>(vb.base),
                                    vb.stride, vb.num_records};
        } else {
          auto& bind = d.vbufs[sel];
          if (vb.base < reinterpret_cast<uint64_t>(bind.data))
            bind.data = reinterpret_cast<const void*>(vb.base);
          bind.num_records = std::min(bind.num_records, vb.num_records);
        }
        attr_binding[i] = static_cast<uint32_t>(sel);
      }
      // Offsets are relative to each binding's final (lowest) base.
      for (uint32_t i = 0; i < attr_n && d.num_vattrs < 8; i++) {
        const VBuffer& vb = attr_vbs[i];
        const uint32_t b = attr_binding[i];
        if (b >= d.num_vbufs)
          continue;
        const uint64_t off =
            vb.base - reinterpret_cast<uint64_t>(d.vbufs[b].data);
        if (off >= d.vbufs[b].stride)
          continue;
        // A typed fetch (tbuffer_load_format_*) carries its own format and the
        // hardware ignores the V#'s; Skyrim's world positions come in that way.
        uint32_t dfmt = vb.dfmt, nfmt = vb.nfmt;
        if (attr_res[i]->inst_format)
          Gfx10VBufFormat(attr_res[i]->inst_format, dfmt, nfmt);
        d.vattrs[d.num_vattrs++] = {
            attr_res[i]->location,  b,    static_cast<uint32_t>(off),
            attr_res[i]->num_comps, dfmt, nfmt};
      }
      if (d.num_vbufs) {
        // Mirror binding 0 into the legacy single-stream fields; the vertex
        // count is bounded by the smallest binding's record count.
        d.vertex_data = d.vbufs[0].data;
        d.vertex_stride = d.vbufs[0].stride;
        uint32_t count = UINT32_MAX;
        for (uint32_t j = 0; j < d.num_vbufs; j++)
          count = std::min(count, d.vbufs[j].num_records);
        d.vertex_count = count;
      }
      // A fetch VS needs at least one attribute resolved; a procedural VS (no
      // recovered attrs, seeds from VertexIndex) draws without a vertex buffer.
      // A fetch VS needs at least one attribute resolved; a procedural VS (no
      // recovered attrs, seeds from VertexIndex) draws without a vertex buffer.
      // A fullscreen pass binds NO vertex buffer at all even though its shader
      // contains fetch instructions (the same shader is used both ways), so
      // demanding a resolved attribute there discarded the whole pass. Let it
      // through with no vertex inputs declared instead.
      bool good = d.num_vattrs > 0 || rc.attrs.empty();
      if (!good && !d.num_vbufs) {
        d.num_vattrs = 0;
        good = true;
      }
      drop_attr_count = rc.attrs.size();

      const auto ps_resources =
          ps_a ? rdna::ResolveBuffers(reinterpret_cast<const uint32_t*>(ps_a),
                                      pud, ps_user_sgprs)
               : std::unordered_map<uint32_t, rdna::BufferResource>{};
      auto resolve_cbufs =
          [&](const std::vector<gcn::ShaderCbuf>& cbufs,
              const std::unordered_map<uint32_t, rdna::BufferResource>&
                  resolved,
              bool vertex_stage) {
            for (const auto& cb : cbufs) {
              if (cb.binding >= gpu::gcn::kMaxCbufBindings)
                continue;
              const auto it = resolved.find(cb.use_pc);
              if (it == resolved.end())
                continue;
              uint64_t base = it->second.base & ~uint64_t{3};
              uint64_t bytes = static_cast<uint64_t>(cb.num_dwords) * 4;
              if (dl)
                std::fprintf(
                    stderr,
                    "[agc]   cbuf %s bind=%u use_pc=%#x base=%#lx dwords=%u\n",
                    vertex_stage ? "vs" : "ps", cb.binding, cb.use_pc,
                    (unsigned long)base, cb.num_dwords);
              if (!InGuest(base) || !gpu::IsReadableRange(base, bytes))
                continue;
              d.cbufs[cb.binding] = {base, static_cast<uint32_t>(bytes)};
              d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
              if (vertex_stage && bytes >= sizeof(d.mvp)) {
                d.cbuf_base = base;
                d.cbuf_size = static_cast<uint32_t>(bytes);
                std::memcpy(d.mvp, reinterpret_cast<const void*>(base),
                            sizeof(d.mvp));
              }
            }
          };
      // Raw MUBUF buffers the shader indexes itself (set 2). Same per-pc
      // descriptor replay as the cbuffers.
      auto resolve_bufs =
          [&](const std::vector<gcn::ShaderBuffer>& bufs,
              const std::unordered_map<uint32_t, rdna::BufferResource>&
                  resolved,
              const uint32_t* ud, uint32_t nud) {
            for (const gcn::ShaderBuffer& sb : bufs) {
              if (sb.binding >= rhi::DrawInfo::kMaxBuffers)
                continue;
              VBuffer vb{};
              const auto it = resolved.find(sb.use_pc);
              if (it != resolved.end() && it->second.descriptor_valid)
                vb = DecodeVBuffer(it->second.descriptor);
              else if (sb.srsrc_sgpr + 3 < nud)
                vb = DecodeVBuffer(&ud[sb.srsrc_sgpr]);
              const uint64_t bytes =
                  vb.stride ? static_cast<uint64_t>(vb.stride) * vb.num_records
                            : vb.num_records;
              if (!InGuest(vb.base) || !bytes || bytes > 0xFFFFFFFFull)
                continue;
              d.bufs[sb.binding] = {vb.base, static_cast<uint32_t>(bytes)};
              d.num_bufs = std::max(d.num_bufs, sb.binding + 1);
            }
          };
      if (good) {
        resolve_cbufs(rc.vs_cbufs, vs_resources, true);
        resolve_bufs(rc.vs_bufs, vs_resources, vud, gs_user_sgprs);
        if (ps_a) {
          resolve_cbufs(rc.ps_cbufs, ps_resources, false);
          resolve_bufs(rc.ps_bufs, ps_resources, pud, ps_user_sgprs);
        }
        // Textures: resolve the live gfx10.3 T#/S# each PS sampler reads, in
        // the recompiler's set-0 binding order (rdna::TrackTextures re-derives
        // the same plan). texs[i] maps to PS sampler binding i.
        if (ps_a && !rc.ps_texs.empty()) {
          auto texs = rdna::TrackTextures(
              reinterpret_cast<const uint32_t*>(ps_a), pud, ps_user_sgprs);
          // The single-texture render path reads the legacy tex* mirror of
          // texs[0], so populate it too (the PS4 path does the same).
          if (!texs.empty()) {
            d.tex_base = texs[0].valid ? texs[0].base : 0;
            d.tex_swizzle = PackDstSel(texs[0]);
            d.tex_w = texs[0].width;
            d.tex_h = texs[0].height;
            d.tex_dfmt = texs[0].dfmt;
            d.tex_nfmt = texs[0].nfmt;
            d.tex_tiling = texs[0].tiling_idx;
            d.tex_pitch = texs[0].pitch;
            d.tex_layers = texs[0].layers;
            d.tex_base_array = texs[0].base_array;
            d.tex_view_layers = texs[0].view_layers;
            d.tex_mip_levels = texs[0].mip_levels;
            d.tex_base_mip = texs[0].base_mip;
            d.tex_view_mips = texs[0].view_mips;
            d.tex_min_lod = texs[0].min_lod;
            std::memcpy(d.tex_sampler, texs[0].sampler, sizeof(d.tex_sampler));
            d.tex_pow2_pad = texs[0].pow2_pad;
            d.tex_sampler_valid = texs[0].sampler_valid;
            d.tex_arrayed = texs[0].arrayed;
            d.tex_force_lod_zero = texs[0].force_lod_zero;
            d.tex_depth_compare = texs[0].depth_compare;
            d.tex_null_descriptor = texs[0].null_descriptor;
          }
          for (size_t i = 0; i < texs.size() && i < 16; i++) {
            const auto& s = texs[i];
            auto& dt = d.texs[i];
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
            dt.null_descriptor = s.null_descriptor;
            dt.swizzle = PackDstSel(s);
          }
          d.num_texs = static_cast<uint32_t>(std::min<size_t>(texs.size(), 16));
          if (dl)
            for (uint32_t i = 0; i < d.num_texs; i++)
              std::fprintf(
                  stderr,
                  "[agc]   tex%u base=%#lx %ux%u dfmt=%u nfmt=%u tiling=%u\n",
                  i, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
                  d.texs[i].dfmt, d.texs[i].nfmt, d.texs[i].tiling);
        }
        d.vs_addr = vs_a;
        d.ps_addr = ps_a;
        d.recomp = &rc;
      } else {
        d.num_vattrs = 0;
      }
    }
  }

  if (auto_vertex_count && auto_vertex_count <= 0x100000)
    d.vertex_count = auto_vertex_count;
  if (!d.recomp) {
    // No usable shader pair: the draw is discarded here, which looks exactly
    // like the title never issuing it. Report the addresses that failed so the
    // gap is attributable.
    g_drop_no_shader.fetch_add(1, std::memory_order_relaxed);
    static int shown = 0;
    if (shown < 16 && kGpuDrawcensus) {
      shown++;
      std::fprintf(
          stderr,
          "[drawcensus] dropped: vs=%#lx ps=%#lx prim=%u vcount=%u "
          "mrt=%u rt=%#lx num_vattrs=%u shaderAttrs=%zu num_vbufs=%u\n",
          (unsigned long)vs_a, (unsigned long)ps_a, d.prim_type, d.vertex_count,
          d.mrt_count, (unsigned long)d.rt_base, d.num_vattrs, drop_attr_count,
          d.num_vbufs);
    }
    return;
  }

  // One-shot ground-truth dump of the first few resolved draws: the raw vertex
  // bytes (as float32 AND uint32, to read off the real attribute format), the
  // constant-buffer/MVP state, and the viewport -- so we can tell whether the
  // positions are garbage (wrong format), screen-space (missing projection), or
  // clip-space (a downstream/viewport issue).
  static int s_vdump = 0;
  // DELTA_AGC_VDUMPFROM=N: skip the opening blits and dump the draws that
  // actually shade the frame.
  // DELTA_AGC_VDUMPRT=<base>: only dump draws that target this render target.
  // Draw indices shift between runs; the target does not.
  // DELTA_AGC_VDUMPIC=<n>: only dump draws with this index count. Draw indices
  // and render-target addresses both move between runs; an index count picks a
  // specific pass out of a frame reliably.
  const uint64_t vertex_bytes =
      std::min<uint64_t>(static_cast<uint64_t>(d.vertex_stride) *
                             (d.vertex_count ? d.vertex_count : 4),
                         128);
  if (kTrace && my_draw >= kVdumpFrom && (!kVdumpRt || d.rt_base == kVdumpRt) &&
      (!kVdumpIc || d.index_count == kVdumpIc) &&
      s_vdump < kVdumpN && d.num_vattrs && d.vertex_data &&
      InGuest(reinterpret_cast<uint64_t>(d.vertex_data)) && vertex_bytes &&
      gpu::IsReadableRange(reinterpret_cast<uint64_t>(d.vertex_data),
                           vertex_bytes)) {
    s_vdump++;
    std::fprintf(
        stderr,
        "[agc] VDUMP draw#%lu num_vattrs=%u stride=%u count=%u prim=%u "
        "vp=[xs=%g xo=%g ys=%g yo=%g] num_cbufs=%u cbuf_base=%#lx cbuf_size=%u "
        "rt=%#lx ps=%#lx vs=%#lx\n",
        (unsigned long)my_draw, d.num_vattrs, d.vertex_stride, d.vertex_count,
        d.prim_type, d.viewport_x_scale, d.viewport_x_offset,
        d.viewport_y_scale, d.viewport_y_offset, d.num_cbufs,
        (unsigned long)d.cbuf_base, d.cbuf_size, (unsigned long)d.rt_base,
        (unsigned long)d.ps_addr, (unsigned long)vs_a);
    for (uint32_t a = 0; a < d.num_vattrs; a++)
      std::fprintf(stderr,
                   "[agc]   vattr%u loc=%u off=%u nc=%u dfmt=%u nfmt=%u\n", a,
                   d.vattrs[a].location, d.vattrs[a].offset,
                   d.vattrs[a].num_comps, d.vattrs[a].dfmt, d.vattrs[a].nfmt);
    // DELTA_AGC_VDUMPPROG: the decoded VS for this exact draw (shader dumps are
    // otherwise emitted once at recompile time and cannot be tied to a draw).
    if (kVdumpProg && InGuest(vs_a)) {
      const uint64_t addr = kAgcVdumpps ? ps_a : vs_a;
      constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
      if (gpu::IsReadableRange(addr, kMaxShaderBytes)) {
        const auto prog =
            rdna::DecodeShader(reinterpret_cast<const uint32_t*>(addr), 4096);
        std::fprintf(stderr, "[agc]   %s %#lx: %zu insts\n",
                     kAgcVdumpps ? "PS" : "VS", (unsigned long)addr, prog.size());
        for (const auto& in : prog) {
          std::fprintf(stderr, "[agc]    pc=%04x %-6s op=%#05x %08x", in.pc,
                       kEncName[static_cast<uint32_t>(in.enc) < 19
                                    ? static_cast<uint32_t>(in.enc)
                                    : 0],
                       in.opcode, in.raw[0]);
          if (in.size >= 2)
            std::fprintf(stderr, " %08x", in.raw[1]);
          if (in.has_literal)
            std::fprintf(stderr, " lit=%08x", in.literal);
          std::fprintf(stderr, "\n");
        }
      }
    }
    const auto* vb = reinterpret_cast<const uint8_t*>(d.vertex_data);
    uint32_t nv = d.vertex_count ? d.vertex_count : 4;
    uint32_t vbytes = static_cast<uint32_t>(vertex_bytes);
    for (uint32_t o = 0; o + 4 <= vbytes; o += 4) {
      uint32_t u;
      float f;
      std::memcpy(&u, vb + o, 4);
      std::memcpy(&f, vb + o, 4);
      std::fprintf(stderr, "[agc]     vtx[+%02u] u=%08x f=%g\n", o, u, f);
    }
    const float* m = d.mvp;
    std::fprintf(
        stderr,
        "[agc]   mvp=[%g %g %g %g / %g %g %g %g / %g %g %g %g / %g %g %g %g]\n",
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10],
        m[11], m[12], m[13], m[14], m[15]);
    // The VS loads its real vertex V# via SMEM from a pointer in GS
    // user_data[4..7] (pc0x23 s_load s0-3 from user_data[4..5]). Follow each
    // such pointer one level and dump what's there (a V# or raw vertices) to
    // locate the real vertex source.
    for (int k = 4; k <= 6; k += 2) {
      uint64_t p = (static_cast<uint64_t>(vud[k + 1] & 0xFFFF) << 32) | vud[k];
      const uint64_t bytes = (k == 4 ? 32 : 8) * sizeof(uint32_t);
      if (!GpuAddr(p) || !gpu::IsReadableRange(p, bytes))
        continue;  // user data holds non-pointers too
      const uint32_t* pw = reinterpret_cast<const uint32_t*>(p);
      std::fprintf(stderr, "[agc]   ud[%d]->%#lx dwords:", k, (unsigned long)p);
      for (int j = 0; j < (k == 4 ? 32 : 8); j++)
        std::fprintf(stderr, " %08x", pw[j]);
      std::fprintf(stderr, "  floats:");
      for (int j = 0; j < 8; j++) {
        float f;
        std::memcpy(&f, &pw[j], 4);
        std::fprintf(stderr, " %g", f);
      }
      std::fprintf(stderr, "\n");
    }
    // DELTA_AGC_VDUMPPROJ: project this draw's first vertices on the host,
    // using the 4x3 world matrix (a 12-dword cbuffer) and the view-projection
    // (dwords 32-47 of a 48-dword one), and print the NDC the shader ought to
    // produce. Comparing that with the drawn extent says whether the transform
    // chain in the recompiled VS is the thing that is wrong.
    // DELTA_AGC_VDUMPPROJ=<world binding>:<vp binding>:<vp dword> names the
    // matrices, since a cbuffer window holds several and only the shader knows
    // which (read it off the SPIR-V's sgpr <- cbuf loads).
    if (const char* proj_env = kVdumpProj;
        proj_env && d.num_vattrs && d.vertex_data) {
      uint32_t wb = 1, vb2 = 2, vdw = 32;
      std::sscanf(proj_env, "%u:%u:%u", &wb, &vb2, &vdw);
      const float *world = nullptr, *vp = nullptr;
      if (wb < d.num_cbufs && d.cbufs[wb].size >= 12 * sizeof(float) &&
          gpu::IsReadableRange(d.cbufs[wb].base, 12 * sizeof(float)))
        world = reinterpret_cast<const float*>(d.cbufs[wb].base);
      const uint64_t vp_offset = static_cast<uint64_t>(vdw) * sizeof(float);
      if (vb2 < d.num_cbufs &&
          d.cbufs[vb2].size >= vp_offset + 16 * sizeof(float) &&
          gpu::IsReadableRange(d.cbufs[vb2].base + vp_offset,
                               16 * sizeof(float)))
        vp = reinterpret_cast<const float*>(d.cbufs[vb2].base) + vdw;
      if (world && vp) {
        const auto* vb = reinterpret_cast<const uint8_t*>(d.vertex_data);
        for (uint32_t v = 0; v < 8 && v < d.vertex_count; v++) {
          const float* p =
              reinterpret_cast<const float*>(vb + (size_t)v * d.vertex_stride);
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
                       v, p[0], p[1], p[2], w4[0], w4[1], w4[2], c[0], c[1],
                       c[2], c[3], c[3] ? c[0] / c[3] : 0.f,
                       c[3] ? c[1] / c[3] : 0.f);
        }
      }
    }
    // DELTA_AGC_VDUMPCB=<n>: how many floats of each bound cbuffer to print.
    // The default shows the head; a transform hides further in (48-dword
    // windows hold several matrices).
    for (uint32_t b = 0; b < d.num_cbufs; b++) {
      const int n = std::min<int>(kCbFloats, d.cbufs[b].size / 4);
      if (n <= 0 || !gpu::IsReadableRange(d.cbufs[b].base, n * sizeof(float)))
        continue;
      const float* cf = reinterpret_cast<const float*>(d.cbufs[b].base);
      std::fprintf(stderr, "[agc]   cbuf[%u]@%#lx (%u dw) floats:", b,
                   (unsigned long)d.cbufs[b].base, d.cbufs[b].size / 4);
      for (int j = 0; j < n; j++) {
        if (j && j % 4 == 0)
          std::fprintf(stderr, " |");
        std::fprintf(stderr, " %g", cf[j]);
      }
      std::fprintf(stderr, "\n");
    }
  }

  if (!g_frame_active) {
    if (dl)
      std::fprintf(stderr, "[agc] DL BeginFrame...\n");
    rhi::BeginFrame(rhi::DefaultRenderer());
    g_frame_active = true;
  }
  if (dl) {
    std::fprintf(stderr,
                 "[agc] DL draw#%lu rhi::Draw num_vattrs=%d rt=%#lx "
                 "tmask=%#x cc=%#x blend=%u dv=%d db=%#lx dt=%u dw=%u df=%u "
                 "ntex=%u tex0=%#lx\n",
                 (unsigned long)my_draw, d.num_vattrs, (unsigned long)d.rt_base,
                 d.target_mask, d.color_control, d.blend_enable, d.depth_valid,
                 (unsigned long)d.depth_base, d.depth_test_enable,
                 d.depth_write_enable, d.depth_func, d.num_texs,
                 (unsigned long)(d.num_texs ? d.texs[0].base : 0));
    for (uint32_t i = 0; i < d.num_texs; i++)
      std::fprintf(stderr,
                   "[agc]   DL tex%u base=%#lx %ux%u dfmt=%u nfmt=%u tiling=%u "
                   "pitch=%u\n",
                   i, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
                   d.texs[i].dfmt, d.texs[i].nfmt, d.texs[i].tiling,
                   d.texs[i].pitch);
    std::fprintf(stderr,
                 "[agc]   DL vtx data=%#lx stride=%u count=%u num_vbufs=%u "
                 "idx=%#lx icount=%u\n",
                 (unsigned long)d.vertex_data, d.vertex_stride, d.vertex_count,
                 d.num_vbufs, (unsigned long)d.index_data, d.index_count);
    for (uint32_t i = 0; i < d.num_vbufs; i++)
      std::fprintf(stderr, "[agc]   DL vbuf%u data=%#lx stride=%u nrec=%u\n", i,
                   (unsigned long)d.vbufs[i].data, d.vbufs[i].stride,
                   d.vbufs[i].num_records);
  }
  g_draws_issued.fetch_add(1, std::memory_order_relaxed);
  g_last_draw_rt = d.rt_base;
  rhi::Draw(rhi::DefaultRenderer(), d);
  if (dl)
    std::fprintf(stderr, "[agc] DL draw#%lu done\n", (unsigned long)my_draw);
}

uint32_t g_op_hist[256] = {};
int g_dumped = 0;

// Walk one PM4 stream, following INDIRECT_BUFFER, latching registers, decoding
// draws, and writing completion labels. depth guards a malformed
// self-reference.
void Walk(const uint32_t* p, uint32_t words, bool dump_this, int depth) {
  if (!p || depth > 8)
    return;
  uint32_t i = 0;
  while (i < words) {
    uint32_t hdr = p[i];
    Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType3) {
      uint32_t op = Pm4Opcode(hdr);
      uint32_t cnt = Pm4Count(hdr);  // body dword count
      const uint32_t* body = &p[i + 1];
      // Desync recovery: a data dword misread as a huge-count packet (e.g. a
      // RELEASE_MEM trailer 0xffff1000 parsed as NOP count=16384) would abandon
      // the rest of the buffer -- and with it the SET_SH_REG_INDIRECT shader
      // bind that follows. Instead of bailing, skip one dword and resync on the
      // next header.
      if (i + 1 + cnt > words) {
        // DELTA_AGC_WALKSTAT: a packet whose count runs past the buffer means
        // we mis-parsed something earlier; the walker resyncs a dword at a time
        // and every packet in between is lost.
        static uint64_t resyncs = 0;
        if (kWalkStat && (++resyncs % 500) == 1)
          std::fprintf(stderr,
                       "[walkstat] resync #%llu at word %u/%u (hdr %08x op %#x "
                       "cnt %u)\n",
                       (unsigned long long)resyncs, i, words, hdr, op, cnt);
        i += 1;
        continue;
      }
      g_op_hist[op & 0xFF]++;
      // DELTA_AGC_OPDUMP=<hex op>: the first few bodies of ONE opcode, wherever
      // it occurs. The submit-level trace only dumps early init submits, so a
      // packet that only appears in steady-state frames is otherwise invisible.
      {
        static int shown = 0;
        if (op == kOpDump && shown < 12) {
          shown++;
          std::fprintf(stderr, "[opdump] op=%#04x cnt=%u body:", op, cnt);
          for (uint32_t b = 0; b < cnt && b < 20; b++)
            std::fprintf(stderr, " %08x", body[b]);
          std::fprintf(stderr, "\n");
        }
      }
      // Who actually binds the render target: report the packet that first
      // makes CB_COLOR0_BASE non-zero, and every later change.
      if (kTrace) {
        static uint32_t s_last_cb = 0;
        static int s_cb_log = 0;
        uint32_t cur = g_regs[mmCB_COLOR0_BASE];
        static uint32_t s_prev_op = 0;
        if (cur != s_last_cb && s_cb_log < 16) {
          s_cb_log++;
          std::fprintf(stderr,
                       "[agc] CB0BASE %08x -> %08x by op %#04x (next %#04x)\n",
                       s_last_cb, cur, s_prev_op, op);
          s_last_cb = cur;
        }
        s_prev_op = op;
        if (false) {
        } else if (cur != s_last_cb) {
          s_last_cb = cur;
        }
      }
      if (dump_this) {
        std::fprintf(stderr, "[agc]   @%-5u T3 op=%#04x count=%u body:", i, op,
                     cnt);
        uint32_t show_n =
            (op == 0x93 || op == 0x79) ? cnt : (cnt < 6 ? cnt : 6);
        for (uint32_t b = 0; b < show_n && b < 24; b++)
          std::fprintf(stderr, " %08x", body[b]);
        // INDIRECT register packets reference a GPU buffer at body[0..1]; dump
        // it so we can RE the register layout (which offset holds
        // CB_COLOR/shaders).
        if ((op == 0x9f || op == 0x93 || op == 0x64 || op == 0x7a ||
             op == 0x63) &&
            cnt >= 2) {
          uint64_t a =
              (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
          if (GpuAddr(a) && gpu::IsReadableRange(a, 12 * sizeof(uint32_t))) {
            auto* aw = reinterpret_cast<const uint32_t*>(a);
            std::fprintf(stderr, " -> buf %#lx:", (unsigned long)a);
            for (int b = 0; b < 12; b++)
              std::fprintf(stderr, " %08x", aw[b]);
          }
        }
        std::fprintf(stderr, "\n");
      }
      // DELTA_AGC_RTPROBE: remember which packet last changed CB_COLOR0_BASE,
      // so a draw that ends up with no colour target can name what unbound it.
      const uint32_t cb0_before = g_regs[mmCB_COLOR0_BASE];
      switch (op) {
        case IT_INDIRECT_BUFFER:  // baseLo, baseHi, sizeDwords(+flags)
        case 0x33: {  // IT_INDIRECT_BUFFER_CNST (AGC constant/Cue chain
                      // -- carries the pipeline SET_SH_REG shader setup)
          if (cnt >= 3) {
            uint64_t ib =
                (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
            uint32_t ibw = body[2] & 0xFFFFF;
            // Bounds-guard: only follow IBs into the GPU aperture with a sane
            // size, so a stale/garbage ring window can't fault the walker.
            if (GpuAddr(ib) && ibw && ibw <= 0x40000 &&
                gpu::IsReadableRange(
                    ib, static_cast<uint64_t>(ibw) * sizeof(uint32_t)))
              Walk(reinterpret_cast<const uint32_t*>(ib), ibw, dump_this,
                   depth + 1);
          }
          break;
        }
        case IT_SET_CONTEXT_REG:
          SetRegs(kContextRegBase, body, cnt);
          break;
        case IT_SET_SH_REG:
          SetRegs(kShRegBase, body, cnt);
          break;
        case IT_SET_UCONFIG_REG:
          SetRegs(kUConfigRegBase, body, cnt);
          break;
        case IT_SET_CONFIG_REG:
          SetRegs(kConfigRegBase, body, cnt);
          break;
        // AGC LOAD_*_REG (registers loaded from a GPU-memory image, not
        // inline).
        case 0x61:
          LoadRegs(kContextRegBase, body, cnt);
          break;  // LOAD_CONTEXT_REG
        case 0x5f:
          LoadRegs(kShRegBase, body, cnt);
          break;  // LOAD_SH_REG
        case 0x5e:
          LoadRegs(kUConfigRegBase, body, cnt);
          break;  // LOAD_UCONFIG_REG
        // AGC SET_*_REG_INDIRECT (per-draw RT + shaders as (off,val) pairs in a
        // buf).
        case 0x9f:
          LoadRegPairs(kContextRegBase, body, cnt);
          break;
        case 0x64:
          LoadRegPairs(kUConfigRegBase, body, cnt);
          break;
        case 0x63:
          LoadRegPairs(kShRegBase, body, cnt);
          break;  // SET_SH_REG_INDIRECT (shaders)
        // op 0x93 is an INLINE SH-reg set: body[0]=reg_offset (low 16b; high
        // bits are flags/count), body[1..] the values (shader PGM_LO/HI +
        // rsrc). Latch them.
        case 0x93: {
          if (cnt >= 2) {
            uint32_t off = kShRegBase + (body[0] & 0xFFFF);
            for (uint32_t k = 1; k < cnt; k++) {
              if (off + (k - 1) < kRegFileSize)
                g_regs[off + (k - 1)] = body[k];
              NoteUdWrite("SET_SH_INLINE(0x93)", off + (k - 1), body[k]);
            }
          }
          break;
        }
        case IT_DMA_DATA:  // CP DMA: ctrl, srcLo/Hi, dstLo/Hi,
                           // command(byteCount).
          // Skyrim never issues SET_CONTEXT_REG: it builds its context state
          // (so CB_COLOR -- the render target) into a shadow image with CP DMA
          // and then restores it with LOAD_CONTEXT_REG. Without the copy the
          // shadow reads zero, every colour draw runs with no target bound and
          // the frame is black. ctrl: SRC_SEL[30:29], DST_SEL[21:20]; sel 0/3 =
          // memory address, 2 = immediate (a fill) -- only copy true memory to
          // memory.
          if (cnt >= 6) {
            uint32_t ctrl = body[0];
            uint32_t src_sel = (ctrl >> 29) & 0x3;
            uint32_t dst_sel = (ctrl >> 20) & 0x3;
            uint64_t src =
                (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
            uint64_t dst =
                (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
            uint32_t bytes = body[5] & 0x1FFFFF;
            const bool src_mem = (src_sel == 0 || src_sel == 3);
            const bool dst_mem = (dst_sel == 0 || dst_sel == 3);
            auto mem_ok = [](uint64_t a) {
              return a >= 0x1000000ull && a < 0x20000000000ull;
            };
            if (!kNoCopy && src_mem && dst_mem && bytes &&
                bytes <= 0x1000000u && src != dst && mem_ok(src) &&
                mem_ok(src + bytes) && mem_ok(dst) && mem_ok(dst + bytes))
              std::memcpy(reinterpret_cast<void*>(dst),
                          reinterpret_cast<const void*>(src), bytes);
            // src_sel 2 = the packet's own dword, repeated: a fill. That is how
            // this title clears a surface -- there is no clear packet -- so
            // apply it to guest memory and let the renderer clear any target it
            // covers.
            if (!kNoCopy && src_sel == 2 && dst_mem && bytes &&
                bytes <= 0x8000000u && mem_ok(dst) && mem_ok(dst + bytes)) {
              const uint32_t fill = body[1];
              auto* p32 = reinterpret_cast<uint32_t*>(dst);
              for (uint32_t k = 0; k < bytes / 4; k++)
                p32[k] = fill;
              rhi::NoteMemoryFill(rhi::DefaultRenderer(), dst, bytes, fill);
            }
            if (kGpuDmatrace) {
              static int dmn = 0;
              if (dmn++ < 60)
                std::fprintf(stderr,
                             "[dma] ctrl=%#x src=%#lx dst=%#lx bytes=%u%s\n",
                             ctrl, (unsigned long)src, (unsigned long)dst,
                             bytes, (src_mem && dst_mem) ? " COPIED" : "");
            }
          }
          break;
        case IT_INDEX_TYPE:
          if (cnt >= 1)
            g_index_type = body[0] & 0x3;
          break;
        case IT_INDEX_BASE:  // baseLo, baseHi
          if (cnt >= 2)
            g_index_base =
                (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | (body[0] & ~1u);
          break;
        case IT_INDEX_BUFFER_SIZE:
          if (cnt >= 1)
            g_index_max = body[0];
          break;
        case IT_NUM_INSTANCES:
          if (cnt >= 1)
            g_num_instances = body[0] ? body[0] : 1;
          break;
        case IT_DISPATCH_DIRECT:
          HandleDispatch(body, cnt);
          break;
        case IT_WRITE_DATA: {  // control, dstLo, dstHi, data...
          if (cnt >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            uint32_t ndw = cnt - 3;
            if (LabelAddrOk(addr) &&
                LabelAddrOk(addr + static_cast<uint64_t>(ndw) * 4))
              std::memcpy(reinterpret_cast<void*>(addr), &body[3],
                          static_cast<size_t>(ndw) * 4);
          }
          break;
        }
        case IT_EVENT_WRITE_EOP: {  // eventCtrl, addrLo, addrHi+sel, dataLo,
                                    // dataHi
          if (cnt >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            uint32_t sel = (body[2] >> 29) & 0x7;
            uint64_t val =
                static_cast<uint64_t>(body[3]) |
                (static_cast<uint64_t>(cnt >= 5 ? body[4] : 0) << 32);
            if (sel == 1)
              WriteLabel(addr, val, false);
            else if (sel == 2)
              WriteLabel(addr, val, true);
            else if (sel >= 3)
              WriteLabel(addr, GpuClockTs(), true);
          }
          break;
        }
        case IT_RELEASE_MEM: {  // eventCtrl, selBits, addrLo, addrHi, dataLo,
                                // dataHi
          if (cnt >= 5) {
            uint32_t sel = (body[1] >> 29) & 0x7;
            uint64_t addr = (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) |
                            (body[2] & ~0x3u);
            uint64_t val =
                static_cast<uint64_t>(body[4]) |
                (static_cast<uint64_t>(cnt >= 6 ? body[5] : 0) << 32);
            if (sel == 1)
              WriteLabel(addr, val, false);
            else if (sel == 2)
              WriteLabel(addr, val, true);
            else if (sel >= 3)
              WriteLabel(addr, GpuClockTs(), true);
          }
          break;
        }
        case IT_EVENT_WRITE_EOS: {  // eventCtrl, addrLo, addrHi+cmd, data
          if (cnt >= 4) {
            uint64_t addr = (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) |
                            (body[1] & ~0x3u);
            WriteLabel(addr, body[3], false);
          }
          break;
        }
        default:
          if (IsDraw(op))
            HandleDraw(op, body, cnt);
          break;
      }
      if (g_regs[mmCB_COLOR0_BASE] != cb0_before) {
        g_cb0_op = op;
        g_cb0_val = g_regs[mmCB_COLOR0_BASE];
        g_cb0_draw = g_draws_seen;
      }
      i += 1 + cnt;
    } else if (type == Pm4Type::kType2 || hdr == 0) {
      i += 1;  // filler / alignment
    } else if (type == Pm4Type::kType0) {
      // Type-0: write `cnt0` consecutive regs starting at the absolute dword
      // offset in the header. The walker used to SKIP these -- but the AGC
      // driver programs shader PGM_LO/HI (and other SH state) via type-0, which
      // is why no SET_SH_REG carried them. Apply them into the register file.
      uint32_t cnt0 = Pm4Count(hdr);
      uint32_t base0 = hdr & 0xFFFF;
      if (i + 1 + cnt0 <= words) {
        for (uint32_t j = 0; j < cnt0; j++) {
          if (base0 + j < kRegFileSize)
            g_regs[base0 + j] = p[i + 1 + j];
          if (base0 + j == mmPA_CL_CLIP_CNTL)
            NoteClipWrite("type0", p[i + 1 + j]);
          NoteUdWrite("type0", base0 + j, p[i + 1 + j]);
        }
        static int s_t0 = 0;
        if (kTrace && s_t0 < 20 && base0 >= kShRegBase &&
            base0 < kShRegBase + 0x300) {
          s_t0++;
          std::fprintf(
              stderr,
              "[agc]   type0 SH write base=%#x cnt=%u v0=%08x v1=%08x\n", base0,
              cnt0, p[i + 1], cnt0 > 1 ? p[i + 2] : 0);
        }
      }
      i += 1 + cnt0;
    } else {
      break;  // type-1 desync
    }
  }
}

}  // namespace

void SubmitDcb(const void* dcb, uint32_t size_bytes) {
  if (!dcb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  // Bring up the (headless) Vulkan renderer on the first submit so draws
  // render.
  static bool s_vk_tried = false;
  if (!s_vk_tried) {
    s_vk_tried = true;
    rhi::Init(rhi::DefaultRenderer());
  }
  // ONE-SHOT: after some frames, scan the 2MB SceAgcRegShadow (0x8002860000)
  // for non-zero content -- if setShader wrote the shader state to a DIFFERENT
  // shadow buffer than the one LOAD_SH_REG reads, it lives here. Print any
  // non-zero 8-dw block that contains a shader-PGM-like value (top byte 0x80 =>
  // addr>>8 in aperture).
  if (kTrace) {
    static uint64_t s_scan_at = 0;
    constexpr uint64_t kShadowAddress = 0x8002860000ull;
    constexpr uint64_t kShadowBytes = 0x200000;
    if (++s_scan_at == 2000 &&
        gpu::IsReadableRange(kShadowAddress, kShadowBytes)) {
      const uint32_t* sh = reinterpret_cast<const uint32_t*>(kShadowAddress);
      int shown = 0;
      for (uint32_t w = 0; w < (0x200000 / 4) && shown < 16; w += 2) {
        uint32_t lo = sh[w], hi = sh[w + 1];
        uint64_t a =
            (static_cast<uint64_t>(lo) << 8) | ((uint64_t)(hi & 0xFF) << 40);
        if (GpuAddr(a)) {
          std::fprintf(stderr,
                       "[agc]   SHADOW+%#x lo=%08x hi=%08x -> addr %#lx\n",
                       w * 4, lo, hi, (unsigned long)a);
          shown++;
        }
      }
      if (!shown)
        std::fprintf(stderr,
                     "[agc]   SHADOW scan: 2MB all-zero (no PGM written)\n");
    }
  }
  uint32_t words = size_bytes / 4;
  uint64_t sn = ++g_total_submits;
  // Skip the all-zero ACQRB ring submits (the 0xC0408121 path is empty for the
  // mode-1 titles); dump the first few submits that actually carry packets.
  const uint32_t* w0 = static_cast<const uint32_t*>(dcb);
  bool non_empty = words >= 2 && (w0[0] || w0[1]);
  bool dump_this = kTrace && g_dumped < 6 && non_empty;
  if (dump_this) {
    g_dumped++;
    const uint32_t* w = static_cast<const uint32_t*>(dcb);
    std::fprintf(stderr,
                 "[agc] === dcb walk #%lu (size=%u words=%u hdr0=%#x) ===\n",
                 static_cast<unsigned long>(sn), size_bytes, words, w[0]);
    uint32_t raw_n =
        words > 100 ? 100 : words;  // dump big draw buffers fully enough
    std::fprintf(stderr, "[agc]   raw[0..%u]:", raw_n);
    for (uint32_t k = 0; k < raw_n; k++)
      std::fprintf(stderr, " %08x", w[k]);
    std::fprintf(stderr, "\n");
  }
  Walk(static_cast<const uint32_t*>(dcb), words, dump_this, 0);
  // Periodic global opcode census so we see draw opcodes that only appear in
  // later (undumped) submits -- tells us if the title is issuing draws yet.
  if (kTrace && non_empty && (sn % 200) == 0) {
    std::fprintf(stderr, "[agc] === global opcode census @submit %lu ===\n",
                 static_cast<unsigned long>(sn));
    for (int o = 0; o < 256; o++)
      if (g_op_hist[o])
        std::fprintf(stderr, "[agc]   op %#04x x%u\n", o, g_op_hist[o]);
  }
  if (dump_this) {
    std::fprintf(stderr, "[agc] === dcb walk done; opcode histogram ===\n");
    for (int o = 0; o < 256; o++)
      if (g_op_hist[o])
        std::fprintf(stderr, "[agc]   op %#04x x%u\n", o, g_op_hist[o]);
  }
}

void SubmitCcb(const void* ccb, uint32_t size_bytes) {
  if (!ccb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lk(g_mtx);
  Walk(static_cast<const uint32_t*>(ccb), size_bytes / 4, false, 0);
}

void EndFrame(uint64_t scanout_base) {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_frame_active && rhi::DefaultRenderer().available()) {
    // The trigger that ends a frame is often the NEXT frame's state submit,
    // and the flip it reads still names the buffer before this one. When the
    // frame composited straight into a registered display buffer, that buffer
    // is what the title just finished, so present it instead -- otherwise
    // every frame presents its neighbour, which the title has already cleared.
    if (g_last_draw_rt && g_last_draw_rt != scanout_base &&
        prosperity_ps5_is_display_buffer(g_last_draw_rt))
      scanout_base = g_last_draw_rt;
    rhi::EndFrame(rhi::DefaultRenderer(), scanout_base);
    g_frame_active = false;
  }
}

}  // namespace gpu::ps5

// LLE submit bridge: the kernel /dev/gc AGC ioctls (gc_dev.cpp) forward the DCB
// here, mirroring prosperity_gc_submit on the PS4 path.
extern "C" void prosperity_agc_submit(uint64_t dcb_base, uint32_t size_bytes) {
  gpu::ps5::SubmitDcb(reinterpret_cast<const void*>(dcb_base), size_bytes);
}

// PS5 flip bridge: the shared dce/VideoOut flip path calls this when the active
// process is PS5, so the frame the AGC submit rendered is read back + presented
// through rhi::EndFrame (mirrors prosperity_gc_flip on the PS4 path).
extern "C" void prosperity_agc_flip(uint64_t scanout_base) {
  gpu::ps5::EndFrame(scanout_base);
}
