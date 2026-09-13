/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The DELTA_AGC_* / DELTA_GPU_* probes of the PS5 command stream. See
 * cmd_trace.h.
 */

#include "gpu/ps5/cmd_trace.h"
#include "base/arch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_set>
#include <utility>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_detile.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"

namespace {
DELTA_OPTION(u64, kBlkFrom, "DELTA_AGC_REGSTAT_FROM", 0);
DELTA_OPTION(int, kCbTraceFrom, "DELTA_AGC_CBTRACE", -1);
DELTA_OPTION(u64, kDumpSh, "DELTA_AGC_DUMPSH", 0);
DELTA_OPTION(int, kVdumpN, "DELTA_AGC_VDUMPN", 8);
DELTA_OPTION(unsigned long, kVdumpFrom, "DELTA_AGC_VDUMPFROM", 0);
DELTA_OPTION(u64, kVdumpRt, "DELTA_AGC_VDUMPRT", 0);
DELTA_OPTION(u32, kVdumpIc, "DELTA_AGC_VDUMPIC", 0);
DELTA_OPTION(bool, kVdumpProg, "DELTA_AGC_VDUMPPROG", false);
DELTA_OPTION(const char*, kVdumpProj, "DELTA_AGC_VDUMPPROJ", nullptr);
DELTA_OPTION(int, kCbFloats, "DELTA_AGC_VDUMPCB", 8);
DELTA_OPTION(u32, kOpDump, "DELTA_AGC_OPDUMP", 0xFFFF);
DELTA_OPTION(int, kRegStat, "DELTA_AGC_REGSTAT", 0);
DELTA_OPTION(bool, kAgcCliptrace, "DELTA_AGC_CLIPTRACE", false);
DELTA_OPTION(bool, kAgcUdtrace, "DELTA_AGC_UDTRACE", false);
DELTA_OPTION(bool, kAgcShcensus, "DELTA_AGC_SHCENSUS", false);
DELTA_OPTION(bool, kTexValid, "DELTA_GPU_TEXVALID", false);
DELTA_OPTION(bool, kAgcVdumpps, "DELTA_AGC_VDUMPPS", false);
DELTA_OPTION(bool, kCsDump, "DELTA_GPU_CSDUMP", false);
DELTA_OPTION(bool, kGpuDmatrace, "DELTA_GPU_DMATRACE", false);
DELTA_OPTION(bool, kGpuDrawcensus, "DELTA_GPU_DRAWCENSUS", false);
DELTA_OPTION(bool, kResTrace, "DELTA_GPU_CSRES", false);
DELTA_OPTION(bool, kRtProbe, "DELTA_AGC_RTPROBE", false);
DELTA_OPTION(bool, kTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kOpHist, "DELTA_AGC_OPHIST", false);
DELTA_OPTION(bool, kWalkStat, "DELTA_AGC_WALKSTAT", false);
DELTA_OPTION(bool, kOpCensus, "DELTA_AGC_OPCENSUS", false);
}  // namespace

// Whether the target of a draw is a buffer the title registered for display,
// which is what makes "thousands of draws, black frame" readable.
extern "C" bool prosperity_ps5_is_display_buffer(u64 addr);

namespace gpu::ps5 {
namespace {

constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);

// Draw accounting. The counters the census reports and the index every
// detail probe below gates on.
std::atomic<u64> g_draws_seen{0}, g_draws_issued{0}, g_drop_no_shader{0};

// Draws that have reached shader resolution. The one being reported on is the
// last, so its index is one less.
u64 g_detailed = 0;
u64 CurrentDraw() {
  return g_detailed ? g_detailed - 1 : 0;
}

// The window of draws the detail probes cover: enough to walk into steady
// state, bounded so a run does not drown in its own trace.
DELTA_OPTION(u64, kDetailDraws, "DELTA_AGC_DETAIL_DRAWS", 5000);
bool Detail() {
  return kTrace && g_detailed <= kDetailDraws;
}

// Shared budget for the dispatch-skip reports: a title that dispatches every
// frame would otherwise flood the log with one line per reason per dispatch.
bool CsReport() {
  static int n = 0;
  return n++ < 32;
}

u32 g_op_hist[256] = {};

// The opcodes the walk did not act on. An opcode nothing looked at is state or
// a draw the title expects to happen, and the only symptom downstream is
// whatever it never set -- so a skip we chose records why, and one we did not
// choose stands out as the census entry worth chasing.
bool g_skipped[256] = {};
const char* g_skip_reason[256] = {};

void DumpOpcodeHistogram() {
  for (int o = 0; o < 256; o++) {
    if (!g_op_hist[o])
      continue;
    base::String line;
    base::FormatTo(line, "  op {:#04x} x{}", o, g_op_hist[o]);
    if (g_skipped[o])
      base::FormatTo(line, "  [{}]",
                     g_skip_reason[o] ? g_skip_reason[o] : "UNHANDLED");
    BASE_LOGI("agc", "{}", line.c_str());
  }
}

const char* const kEncName[19] = {"?",     "sop1", "sop2", "sopk", "sopc",
                                  "sopp",  "smem", "vop1", "vop2", "vop3",
                                  "vop3p", "vopc", "vint", "ds",   "mubuf",
                                  "mtbuf", "mimg", "exp",  "flat"};

// One decoded shader, instruction by instruction.
base::String Words(const u32* p, u32 count);

void DumpProgram(const char* what, u64 address) {
  const auto prog =
      rdna::DecodeShader(reinterpret_cast<const u32*>(address), 4096);
  BASE_LOGI("agc", "{} {:#x}: {} insts", what, address, prog.size());
  for (const auto& in : prog) {
    base::String line;
    base::FormatTo(line, "  pc={:04x} {:<6} op={:#05x} {:08x}", in.pc,
                   kEncName[static_cast<u32>(in.enc) < 19
                                ? static_cast<u32>(in.enc)
                                : 0],
                   in.opcode, in.raw[0]);
    if (in.size >= 2)
      base::FormatTo(line, " {:08x}", in.raw[1]);
    if (in.has_literal)
      base::FormatTo(line, " lit={:08x}", in.literal);
    BASE_LOGI("agc", "{}", line.c_str());
  }
  // The dwords past the decoded end, for a branch that lands there.
  if (!prog.empty()) {
    const u32 end = prog.back().pc + prog.back().size;
    BASE_LOGI("agc", "  after {:#x}:{}", end,
              Words(reinterpret_cast<const u32*>(address) + end, 24).c_str());
  }
}

base::String Words(const u32* p, u32 count) {
  base::String line;
  for (u32 i = 0; i < count; i++)
    base::FormatTo(line, " {:08x}", p[i]);
  return line;
}

}  // namespace

// --- register writes -------------------------------------------------------

void NoteRegisterWrite(const char* source, u32 reg, u32 value) {
  if (kAgcCliptrace && reg == mmPA_CL_CLIP_CNTL) {
    static int n = 0;
    if (n++ < 12)
      BASE_LOGI("agc", "CLIP_CNTL <- {:#x} by {}", value, source);
  }
  // Skyrim's grading pass reads a texture descriptor from s16..s23; if nothing
  // in the command stream programs those, the shader samples what was left.
  if (kAgcUdtrace && value && reg >= mmSPI_SHADER_USER_DATA_PS_0 + 16 &&
      reg < mmSPI_SHADER_USER_DATA_PS_0 + 32) {
    static int n = 0;
    if (n++ < 40)
      BASE_LOGI("agc", "PS ud{} <- {:08x} by {}",
                reg - mmSPI_SHADER_USER_DATA_PS_0, value, source);
  }
  // A title that programs its shader state at offsets we do not read leaves the
  // ones we do read at zero, and then every descriptor the vertex stage names
  // resolves against nothing.
  if (kAgcShcensus && value && reg >= kShRegBase &&
      reg < kShRegBase + 0x300) {
    static std::map<u32, std::pair<u64, u32>> hist;  // offset -> {count, last}
    static std::mutex lock;
    static u64 dumps = 0;
    std::lock_guard<std::mutex> lk(lock);
    auto& e = hist[reg - kShRegBase];
    e.first++;
    e.second = value;
    if (++dumps % 20000 == 0) {
      BASE_LOGI("shcensus", "--- SH offsets written ({} distinct) ---",
                hist.size());
      for (const auto& [off, v] : hist)
        BASE_LOGI("shcensus", "  sh+{:#05x} x{} last={:08x}", off, v.first,
                  v.second);
    }
  }
  // A shader at 0x8001xxxxxx has a PGM_LO of ~0x800xxxxx: report SH-space
  // writes that look like one, to find the register a pipeline binds through.
  if (kTrace && reg >= kShRegBase && (value >> 24) == 0x80) {
    static int n = 0;
    if (n++ < 20)
      BASE_LOGI("agc", "  SH pgm? off={:#x} val={:#x} (addr~{:#x}) by {}",
                reg - kShRegBase, value, static_cast<u64>(value) << 8, source);
  }
}

// --- register images -------------------------------------------------------

u64 TraceRegImagePrefixDwords() {
  return kTrace ? 0x400 : 0;
}

void TraceRegImage(u32 base, u64 image, const u32* body, u32 count) {
  if (!kTrace)
    return;
  const u32* src = reinterpret_cast<const u32*>(image);
  // Coherency census: a title that drives its whole context through register
  // shadows is invisible if those images read zero. Sample one late, once
  // rendering has started -- an early shadow may be empty simply because it
  // has not.
  static int s_img_n = 0;
  if (base == kContextRegBase && ++s_img_n >= 100 && s_img_n <= 106) {
    u32 nz = 0, first = 0;
    for (u32 j = 0; j < 0x400; j++)
      if (src[j]) {
        nz++;
        if (!first)
          first = j;
      }
    BASE_LOGI("agc",
              "IMGCENSUS #{} base={:#x} mem={:#x} nonzero={}/1024 first={:#x} "
              "img[0x318]={:08x} img[0x31c]={:08x} ranges={}",
              s_img_n, base, image, nz, first, src[0x318], src[0x31c],
              (count - 2) / 2);
  }
  static int s_img = 0;
  if (s_img++ < 6)
    BASE_LOGI("agc", "  LOADimg base={:#x} mem={:#x}:{}", base, image,
              Words(src, 16).c_str());

  // A LOAD_SH_REG image could be offset-INDEXED (image[reg_off] = value, a full
  // reg-file shadow) or CURSOR-based (ranges packed contiguously from the
  // image). Dump both readings plus the range list, and scan the whole image
  // for a program pointer, to settle where the shader PGM actually is.
  static int s_shimg = 0, s_shcnt = 0;
  if (base != kShRegBase)
    return;
  // Fire on the first few SH loads and on a batch ~5000 in, once the title is
  // past its loading screen and issuing real draws.
  const bool late_window = s_shcnt >= 5000 && s_shcnt < 5006;
  s_shcnt++;
  if (s_shimg >= 3 && !late_window)
    return;
  s_shimg++;
  BASE_LOGI("agc", "(shload #{})", s_shcnt);
  base::String ranges;
  for (u32 i = 2; i + 1 < count; i += 2)
    base::FormatTo(ranges, " (off={:#x},num={})", body[i] & 0xFFFF,
                   body[i + 1] & 0xFFFF);
  BASE_LOGI("agc", "SHLOAD mem={:#x} ranges:{}", image, ranges.c_str());
  BASE_LOGI("agc",
            "  indexed[0x08..0x0b]={:08x} {:08x} {:08x} {:08x}  "
            "indexed[0x88..0x8b]={:08x} {:08x} {:08x} {:08x}",
            src[0x08], src[0x09], src[0x0a], src[0x0b], src[0x88], src[0x89],
            src[0x8a], src[0x8b]);
  u32 cursor = 0;
  for (u32 i = 2; i + 1 < count; i += 2) {
    const u32 off = body[i] & 0xFFFF;
    const u32 num = std::min<u32>(body[i + 1] & 0xFFFF, 0x400);
    BASE_LOGI("agc", "  cursor off={:#x} <- img[{}..]: {:08x} {:08x}", off,
              cursor, src[cursor], num > 1 ? src[cursor + 1] : 0);
    cursor += num;
  }
  for (u32 j = 0; j < 0x400; j++) {
    const u32 v = src[j];
    if ((v >> 24) != 0x80)
      continue;
    const u64 a = static_cast<u64>(v) << 8;
    if (IsGpuAddress(a) && gpu::IsReadableRange(a, 2 * sizeof(u32))) {
      const u32* w = reinterpret_cast<const u32*>(a);
      BASE_LOGI("agc", "  img[{:#x}]={:08x} -> {:#x} ISA? {:08x} {:08x}", j, v,
                a, w[0], w[1]);
    }
  }
}

// --- state blocks ----------------------------------------------------------

void NoteRegBlock(u32 base,
                  RegBlockOutcome outcome,
                  u64 address,
                  u32 num_pairs) {
  if (!kRegStat)
    return;
  struct Stat {
    u64 calls, short_pkt, bad_addr, unreadable, no_pairs, applied;
  };
  static Stat s_stat[3] = {};
  Stat& st = s_stat[base == kContextRegBase ? 0 : base == kShRegBase ? 1 : 2];
  st.calls++;
  switch (outcome) {
    case RegBlockOutcome::kApplied:
      st.applied++;
      break;
    case RegBlockOutcome::kShortPacket:
      st.short_pkt++;
      break;
    case RegBlockOutcome::kBadAddress:
      if (++st.bad_addr < 6)
        BASE_LOGI("regstat", "bad addr {:#x}", address);
      break;
    case RegBlockOutcome::kUnreadable:
      if (++st.unreadable < 6)
        BASE_LOGI("regstat", "unreadable {:#x} pairs={}", address, num_pairs);
      break;
    case RegBlockOutcome::kNoPairs:
      st.no_pairs++;
      break;
  }
  static u64 n = 0;
  if ((++n % 4000) != 1)
    return;
  for (int k = 0; k < 3; k++)
    BASE_LOGI("regstat",
              "{} calls={} short={} badaddr={} unreadable={} nopairs={} "
              "applied={}",
              k == 0 ? "context" : k == 1 ? "sh" : "uconfig", s_stat[k].calls,
              s_stat[k].short_pkt, s_stat[k].bad_addr, s_stat[k].unreadable,
              s_stat[k].no_pairs, s_stat[k].applied);
}

void TraceRegBlock(u32 base, u64 address, u32 num_pairs, u32 mode) {
  if (kRegStat < 2 || num_pairs < 4 || DrawsSeen() < kBlkFrom)
    return;
  const bool digest = kRegStat >= 3;
  static int full = 0;
  if (full++ >= (digest ? 400 : 40))
    return;
  BASE_LOGI("regstat", "{} block {:#x} pairs={} mode={:08x} draw={}",
            base == kContextRegBase ? "ctx"
            : base == kShRegBase    ? "sh"
                                    : "ucfg",
            address, num_pairs, mode, DrawsSeen());
  const u32* q = reinterpret_cast<const u32*>(address);
  for (u32 k = 0; k < num_pairs; k++)
    if (!digest || q[k * 2 + 1] || (q[k * 2] & (1u << 28)))
      BASE_LOGI("regstat", "  {:3}: {:08x} {:08x}", k, q[k * 2], q[k * 2 + 1]);
}

void TraceRegBlockAnchor(u32 pair,
                         u64 rt_base,
                         u32 info,
                         u32 width,
                         u32 height) {
  if (!kRegStat)
    return;
  static int shown = 0;
  if (shown++ < 8)
    BASE_LOGI("regstat", "ctx anchor pair={} rt={:#x} info={:08x} fmt={} {}x{}",
              pair, rt_base, info, (info >> 2) & 0x1F, width, height);
}

void TraceColorBaseWrite(u32 reg,
                         u32 value,
                         u32 pair,
                         u32 num_pairs,
                         u64 address,
                         u32 offset_dword) {
  if (kCbTraceFrom < 0 || static_cast<int>(DrawsSeen()) < kCbTraceFrom)
    return;
  if (reg != mmCB_COLOR0_BASE && reg != mmCB_COLOR0_INFO)
    return;
  static int n = 0;
  if (n++ < 60)
    BASE_LOGI("agc", "CB0 {} <- {:08x}  (pair {}/{} from {:#x}, raw off {:08x})",
              reg == mmCB_COLOR0_BASE ? "BASE" : "INFO", value, pair, num_pairs,
              address, offset_dword);
}

void TraceShaderBind(u64 address, u32 gs_pgm_lo, u32 ps_pgm_lo) {
  if (!kTrace || (!gs_pgm_lo && !ps_pgm_lo))
    return;
  static int n = 0;
  if (n++ < 6)
    BASE_LOGI("agc", "SHADER BIND @{:#x}: PGM_LO_GS={:08x} PGM_LO_PS={:08x}",
              address, gs_pgm_lo, ps_pgm_lo);
}

// --- draw accounting -------------------------------------------------------

void NoteDrawSeen() {
  g_draws_seen.fetch_add(1, std::memory_order_relaxed);
  if (!kGpuDrawcensus)
    return;
  static const bool started = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        BASE_LOGI("drawcensus", "seen={} issued={} dropped: no-shader={}",
                  g_draws_seen.load(), g_draws_issued.load(),
                  g_drop_no_shader.load());
      }
    }).detach();
    return true;
  }();
  (void)started;
}

u64 DrawsSeen() {
  return g_draws_seen.load(std::memory_order_relaxed);
}

void NoteDrawIssued(const rhi::DrawInfo& d) {
  g_draws_issued.fetch_add(1, std::memory_order_relaxed);
  if (!kGpuDrawcensus)
    return;
  // Which targets the frame actually renders into, once each. A frame that
  // presents black while thousands of draws issue means none of them landed in
  // a registered display buffer, and this is the only way to see that.
  static std::set<u64> rts;
  static std::mutex lock;
  std::lock_guard<std::mutex> lk(lock);
  if (rts.size() < 64 && rts.insert(d.rt_base).second)
    BASE_LOGI("drawcensus",
              "rt {:#x} {}x{} display={} mrt={} depth={} prim={} vcount={}",
              d.rt_base, d.rt_w, d.rt_h,
              prosperity_ps5_is_display_buffer(d.rt_base), d.mrt_count,
              d.depth_valid, d.prim_type, d.vertex_count);
}

void NoteDrawDropped(const rhi::DrawInfo& d,
                     u64 vs_addr,
                     u64 ps_addr,
                     size_t shader_attrs) {
  g_drop_no_shader.fetch_add(1, std::memory_order_relaxed);
  if (!kGpuDrawcensus)
    return;
  static int shown = 0;
  if (shown++ < 16)
    BASE_LOGI("drawcensus",
              "dropped: vs={:#x} ps={:#x} prim={} vcount={} mrt={} rt={:#x} "
              "num_vattrs={} shaderAttrs={} num_vbufs={}",
              vs_addr, ps_addr, d.prim_type, d.vertex_count, d.mrt_count,
              d.rt_base, d.num_vattrs, shader_attrs, d.num_vbufs);
}

// --- one draw, in detail ---------------------------------------------------

void TraceNggState(const Regs& regs, u64 es_addr, u64 gs_addr) {
  if (!kGpuDrawcensus || !gs_addr || es_addr == gs_addr)
    return;
  static std::unordered_set<u64> seen;
  const u64 key = gs_addr ^ (es_addr * 0x9e3779b97f4a7c15ull);
  if (seen.size() >= 128 || !seen.insert(key).second)
    return;
  BASE_LOGI("nggstate",
            "es={:#x} gs={:#x} ge_cntl={:#x} onchip={:#x} max_out={:#x} "
            "subgrp={:#x} instances={:#x} rsrc2={:#x}",
            es_addr, gs_addr, regs[0xc25b], regs[0xa291], regs[0xa2ce],
            regs[0xa2d3], regs[0xa2e4], regs[mmSPI_SHADER_PGM_RSRC2_GS]);
  BASE_LOGI("nggstate", "GS UD:{}",
            Words(regs.At(mmSPI_SHADER_USER_DATA_GS_0), 32).c_str());
  BASE_LOGI("nggstate", "ES UD:{}",
            Words(regs.At(mmSPI_SHADER_USER_DATA_ES_0), 32).c_str());
  BASE_LOGI("nggstate", "GS system UD:{}",
            Words(regs.At(mmSPI_SHADER_USER_DATA_ADDR_LO_GS), 2).c_str());
}

void NoteDrawDetail() {
  g_detailed++;
}

void TraceUserData(const u32* gs_user_data, const u32* es_user_data) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ >= 4)
    return;
  BASE_LOGI("agc", "  UD GS:{}", Words(gs_user_data, 16).c_str());
  BASE_LOGI("agc", "  UD ES:{}", Words(es_user_data, 16).c_str());
}

// Which user-data window a draw's vertex stage actually got, per shader. A
// merged NGG stage can be programmed through either the GS or the ES bank, and
// a stage that reads an empty one resolves every descriptor against zeros.
void TraceVsUserData(u64 vs_addr,
                     const u32* gs_user_data,
                     const u32* es_user_data,
                     bool chose_gs) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ >= 24)
    return;
  BASE_LOGI("agc",
            "  UD vs={:#x} chose={} gs[0..3]={:08x} {:08x} {:08x} {:08x} "
            "es[0..3]={:08x} {:08x} {:08x} {:08x}",
            vs_addr, chose_gs ? "gs" : "es", gs_user_data[0], gs_user_data[1],
            gs_user_data[2], gs_user_data[3], es_user_data[0], es_user_data[1],
            es_user_data[2], es_user_data[3]);
}

// Every non-zero SH register a draw sees. A vertex stage whose user-data window
// reads all zeros is either a stage we are reading from the wrong block or one
// the title programs somewhere else entirely, and only the whole file says
// which.
void TraceShRegs(const Regs& regs) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ >= 120)
    return;
  base::String line;
  for (u32 off = 0; off < 0x200; off++) {
    const u32 v = regs[kShRegBase + off];
    if (v)
      base::FormatTo(line, " {:03x}={:08x}", off, v);
  }
  BASE_LOGI("agc", "  SH regs:{}", line.c_str());
}

void TraceShaderScan(const u32* found_reg, const u64* found, u32 count) {
  if (!kTrace)
    return;
  static int s_sc = 0;
  if (s_sc >= 6 || !count)
    return;
  s_sc++;
  base::String scan;
  for (u32 k = 0; k < count && k < 6; k++)
    base::FormatTo(scan, " [{:#x}]={:#x}", found_reg[k], found[k]);
  BASE_LOGI("agc", "  shader scan: nf={}{}", count, scan.c_str());
}

void TraceUserDataPointers(const u32* vs_user_data, const u32* ps_user_data) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ >= 3)
    return;
  // The pipeline handle / program address likely lives in a descriptor, so
  // follow every GPU-aperture pointer in user data one level.
  for (int which = 0; which < 2; which++) {
    const u32* ud = which ? ps_user_data : vs_user_data;
    BASE_LOGI("agc", "  {}UD:{}", which ? "ps" : "gs", Words(ud, 16).c_str());
    for (int k = 0; k + 1 < 16; k++) {
      const u64 p = (static_cast<u64>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
      if (IsGpuAddress(p) && gpu::IsReadableRange(p, 8 * sizeof(u32)))
        BASE_LOGI("agc", "    UD[{}]->{:#x}:{}", k, p,
                  Words(reinterpret_cast<const u32*>(p), 8).c_str());
    }
  }
}

void TraceIndexBuffer(u32 op, const rhi::DrawInfo& d, u64 index_size) {
  if (!kTrace || !d.index_data)
    return;
  static int n = 0;
  const u64 base = reinterpret_cast<u64>(d.index_data);
  const u64 bytes = std::min(d.index_count, 8u) * index_size;
  if (n >= 12 || !gpu::IsReadableRange(base, bytes))
    return;
  n++;
  const u16* i16 = reinterpret_cast<const u16*>(base);
  const u32* i32 = reinterpret_cast<const u32*>(base);
  base::String idx;
  for (u32 k = 0; k < d.index_count && k < 8; k++)
    base::FormatTo(idx, " {}", d.index_type == 1 ? i32[k] : (u32)i16[k]);
  BASE_LOGI("agc", "  IDX op={:#x} ibase={:#x} count={} type={}:{}", op, base,
            d.index_count, d.index_type, idx.c_str());
}

void TraceRtProbe(const Regs& regs, const rhi::DrawInfo& d) {
  if (!kRtProbe)
    return;
  static int n = 0;
  if (n++ >= 200)
    return;
  BASE_LOGI("agc",
            "RTPROBE draw#{} rt={:#x} tmask={:#x} info0={:#x} clip={:#x} "
            "blend={} ctl={:#x} cc={:#x} tex0={:#x} ntex={}",
            DrawsSeen(), d.rt_base, regs[mmCB_TARGET_MASK],
            regs[mmCB_COLOR0_INFO], regs[mmPA_CL_CLIP_CNTL],
            (regs[mmCB_BLEND0_CONTROL] >> 30) & 1, regs[mmCB_BLEND0_CONTROL],
            regs[mmCB_COLOR_CONTROL], d.num_texs ? d.texs[0].base : 0,
            d.num_texs);
}

void TraceNoRenderTarget(const Regs& regs, u32 target_mask) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ < 12)
    BASE_LOGI("agc", "  NO-RT tmask={:#x} cb0Base={:#x} info0={:#x} fmt={}",
              target_mask, regs.CbColorBase(0), regs[mmCB_COLOR0_INFO],
              (regs[mmCB_COLOR0_INFO] >> 2) & 0x1F);
}

void TraceRenderTarget(const Regs& regs,
                       const rhi::DrawInfo& d,
                       u64 vs_addr,
                       u64 ps_addr) {
  if (!kTrace)
    return;
  static int n = 0;
  if (n++ < 12)
    BASE_LOGI("agc",
              "  DRAW rt: cb0Base={:#x} info0={:#x} tmask={:#x} -> "
              "mrt_count={} vs_a={:#x} ps_a={:#x} prim={:#x} idx={}",
              regs.CbColorBase(0), regs[mmCB_COLOR0_INFO],
              regs[mmCB_TARGET_MASK], d.mrt_count, vs_addr, ps_addr,
              d.prim_type, d.index_count);
}

void TraceRecompile(u64 vs_addr, u64 ps_addr, u32 ps_input_ena) {
  if (!Detail())
    return;
  const u32* vc = reinterpret_cast<const u32*>(vs_addr);
  const u32* pc = ps_addr ? reinterpret_cast<const u32*>(ps_addr) : nullptr;
  // AGC shader code starts with the 0xBEEB03FF sentinel; a ps_addr that does
  // not (e.g. the 0xffc9dfe7 poison fill) means the PS PGM_LO never landed.
  BASE_LOGI("agc", "DL recompile vs={:#x} ({:08x}) ps={:#x} ({:08x} {:08x})...",
            vs_addr, vc[0], ps_addr, pc ? pc[0] : 0, pc ? pc[1] : 0);
  BASE_LOGI("agc", "DL psInputEna={:#x} (frag-coord/face VGPR seed)",
            ps_input_ena);
}

void TraceRecompileDone(bool ok) {
  if (Detail())
    BASE_LOGI("agc", "DL recompile done ok={}", ok);
}

void TraceRecompileFailed(u64 vs_addr, u64 ps_addr) {
  static int n = 0;
  if (n++ < 32)
    BASE_LOGI("agc", "recompile FAILED vs={:#x} ps={:#x} -- draw dropped",
              vs_addr, ps_addr);
}

void TraceAttrPlan(size_t attrs,
                   u32 vs_user_sgprs,
                   size_t resolved,
                   const u32* vs_user_data) {
  if (Detail())
    BASE_LOGI("agc",
              "DL attrs={} gsUd={} res={} vud[0..7]={:08x} {:08x} {:08x} "
              "{:08x} {:08x} {:08x} {:08x} {:08x}",
              attrs, vs_user_sgprs, resolved, vs_user_data[0], vs_user_data[1],
              vs_user_data[2], vs_user_data[3], vs_user_data[4],
              vs_user_data[5], vs_user_data[6], vs_user_data[7]);
}

void TraceAttrReplay(u32 index, u32 use_pc, bool found, bool valid) {
  if (Detail())
    BASE_LOGI("agc", "  attr{} use_pc={:#x} found={} valid={}", index, use_pc,
              found, found ? (int)valid : -1);
}

void TraceAttr(u32 index,
               const gcn::ShaderAttr& attr,
               const rdna::VBuffer& vb,
               u32 fetch_soffset,
               const char* how) {
  if (Detail())
    BASE_LOGI("agc",
              "  attr{} loc={} nc={} tbl_sgpr={} off={} ioff={} soff={} "
              "({}) -> base={:#x} stride={} nrec={} gfmt={} -> dfmt={} nfmt={}",
              index, attr.location, attr.num_comps, attr.table_sgpr,
              attr.vbuf_dword_off, attr.inst_offset, fetch_soffset, how,
              vb.base, vb.stride, vb.num_records, vb.gfmt, vb.dfmt, vb.nfmt);
}

// A raw (set-2) buffer the shader indexes itself: whether the draw could name
// it at all. A shader reading an unbound one reads zeros, which is
// indistinguishable from a shader whose maths produced zero -- and an NGG
// vertex program gates its whole body on such a read.
void TraceRawBufBinding(bool vertex_stage,
                        u32 binding,
                        u32 use_pc,
                        u32 srsrc_sgpr,
                        bool replayed,
                        u64 base,
                        u64 bytes) {
  if (!Detail())
    return;
  // With the head of the window: a resolved binding says only that the draw
  // could name a buffer, and a buffer of zeros produces exactly the degenerate
  // geometry an unbound one does.
  base::String head;
  if (bytes && gpu::IsReadableRange(base, 32)) {
    const u32* w = reinterpret_cast<const u32*>(base);
    for (u32 i = 0; i < 8; i++) {
      float f;
      std::memcpy(&f, &w[i], 4);
      base::FormatTo(head, " {:08x}({})", w[i], f);
    }
  }
  BASE_LOGI("agc",
            "  {} rawbuf{} use_pc={:#x} srsrc={} replay={} -> base={:#x} "
            "bytes={} {}{}",
            vertex_stage ? "vs" : "ps", binding, use_pc, srsrc_sgpr,
            (int)replayed, base, bytes, bytes ? "" : "(UNBOUND)",
            head.c_str());
}

void TraceCbufBinding(bool vertex_stage,
                      u32 binding,
                      u32 use_pc,
                      u64 base,
                      u32 num_dwords) {
  if (!Detail())
    return;
  // With the head of the window, as hex and as the float it usually is. A
  // resolved binding says only that the shader has something to read; what it
  // reads is what decides the pixel, and a scale factor sitting at zero looks
  // exactly like a shader that never ran.
  base::String head;
  const u32 shown = std::min(num_dwords, 4u);
  if (gpu::IsReadableRange(base, static_cast<u64>(shown) * 4)) {
    const u32* w = reinterpret_cast<const u32*>(base);
    for (u32 i = 0; i < shown; i++) {
      float f;
      std::memcpy(&f, &w[i], 4);
      base::FormatTo(head, " {:08x}({})", w[i], f);
    }
  }
  BASE_LOGI("agc", "  cbuf {} bind={} use_pc={:#x} base={:#x} dwords={}:{}",
            vertex_stage ? "vs" : "ps", binding, use_pc, base, num_dwords,
            head.c_str());
}

void TraceRejectedTexture(u32 binding, const gcn::TImage& tex) {
  if (!kTexValid || tex.valid || !tex.base || !tex.width || !tex.height)
    return;
  static int n = 0;
  if (n++ < 16)
    BASE_LOGI("texvalid",
              "bind={} REJECTED but base={:#x} {}x{} dfmt={} nfmt={} "
              "tiling={} layers={} mips={}",
              binding, tex.base, tex.width, tex.height, tex.dfmt, tex.nfmt,
              tex.tiling_idx, tex.layers, tex.mip_levels);
}

void TraceDrawTextures(const rhi::DrawInfo& d) {
  if (!Detail())
    return;
  for (u32 i = 0; i < d.num_texs; i++)
    BASE_LOGI("agc", "  tex{} base={:#x} {}x{} dfmt={} nfmt={} tiling={}", i,
              d.texs[i].base, d.texs[i].w, d.texs[i].h, d.texs[i].dfmt,
              d.texs[i].nfmt, d.texs[i].tiling);
}

void TraceVertexDump(const rhi::DrawInfo& d,
                     const u32* vs_user_data,
                     u64 vs_addr,
                     u64 ps_addr) {
  if (!kTrace || CurrentDraw() < kVdumpFrom)
    return;
  if ((kVdumpRt && d.rt_base != kVdumpRt) ||
      (kVdumpIc && d.index_count != kVdumpIc))
    return;
  const u64 vertex_bytes = std::min<u64>(
      static_cast<u64>(d.vertex_stride) * (d.vertex_count ? d.vertex_count : 4),
      128);
  static int n = 0;
  // A procedural pass has no attributes and no vertex buffer, and every
  // composite and post draw in a modern title is one -- so the dump has to
  // cover them too, or the draws that matter most are the ones it cannot show.
  const bool has_vertices =
      d.num_vattrs && d.vertex_data && vertex_bytes &&
      IsGuestAddress(reinterpret_cast<u64>(d.vertex_data)) &&
      gpu::IsReadableRange(reinterpret_cast<u64>(d.vertex_data), vertex_bytes);
  if (n >= kVdumpN)
    return;
  n++;
  BASE_LOGI("agc",
            "VDUMP draw#{} num_vattrs={} stride={} count={} prim={} "
            "vp=[xs={} xo={} ys={} yo={}] num_cbufs={} cbuf_base={:#x} "
            "cbuf_size={} rt={:#x} ps={:#x} vs={:#x}",
            CurrentDraw(), d.num_vattrs, d.vertex_stride, d.vertex_count,
            d.prim_type, d.viewport_x_scale, d.viewport_x_offset,
            d.viewport_y_scale, d.viewport_y_offset, d.num_cbufs, d.cbuf_base,
            d.cbuf_size, d.rt_base, ps_addr, vs_addr);
  for (u32 a = 0; a < d.num_vattrs; a++)
    BASE_LOGI("agc", "  vattr{} loc={} off={} nc={} dfmt={} nfmt={}", a,
              d.vattrs[a].location, d.vattrs[a].offset, d.vattrs[a].num_comps,
              d.vattrs[a].dfmt, d.vattrs[a].nfmt);
  if (kVdumpProg) {
    const u64 addr = kAgcVdumpps ? ps_addr : vs_addr;
    if (IsGuestAddress(addr) && gpu::IsReadableRange(addr, kMaxShaderBytes))
      DumpProgram(kAgcVdumpps ? "  PS" : "  VS", addr);
  }
  const auto* bytes = reinterpret_cast<const u8*>(d.vertex_data);
  for (u64 o = 0; has_vertices && o + 4 <= vertex_bytes; o += 4) {
    u32 u;
    float f;
    std::memcpy(&u, bytes + o, 4);
    std::memcpy(&f, bytes + o, 4);
    BASE_LOGI("agc", "    vtx[+{:02}] u={:08x} f={}", o, u, f);
  }
  const float* m = d.mvp;
  BASE_LOGI("agc", "  mvp=[{} {} {} {} / {} {} {} {} / {} {} {} {} / {} {} {} {}]",
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10],
            m[11], m[12], m[13], m[14], m[15]);
  // The VS loads its real vertex V# via SMEM from a pointer in GS user data
  // [4..7]; follow each one level to locate the vertex source.
  for (int k = 4; k <= 6; k += 2) {
    const u64 p =
        (static_cast<u64>(vs_user_data[k + 1] & 0xFFFF) << 32) |
        vs_user_data[k];
    const int dwords = k == 4 ? 32 : 8;
    if (!IsGpuAddress(p) ||
        !gpu::IsReadableRange(p, dwords * sizeof(u32)))
      continue;  // user data holds non-pointers too
    const u32* pw = reinterpret_cast<const u32*>(p);
    base::String line;
    base::FormatTo(line, "  ud[{}]->{:#x} dwords:{}  floats:", k, p,
                   Words(pw, dwords).c_str());
    for (int j = 0; j < 8; j++) {
      float f;
      std::memcpy(&f, &pw[j], 4);
      base::FormatTo(line, " {}", f);
    }
    BASE_LOGI("agc", "{}", line.c_str());
  }
  // Project this draw's first vertices on the host with the 4x3 world matrix
  // and the view-projection, and print the NDC the shader ought to produce:
  // comparing that with the drawn extent says whether the transform chain in
  // the recompiled VS is the thing that is wrong. A cbuffer window holds
  // several matrices and only the shader knows which, so
  // <world binding>:<vp binding>:<vp dword> names them (read them off the
  // SPIR-V's sgpr <- cbuf loads).
  if (const char* proj = kVdumpProj) {
    u32 world_binding = 1, vp_binding = 2, vp_dword = 32;
    std::sscanf(proj, "%u:%u:%u", &world_binding, &vp_binding, &vp_dword);
    const float *world = nullptr, *vp = nullptr;
    if (world_binding < d.num_cbufs &&
        d.cbufs[world_binding].size >= 12 * sizeof(float) &&
        gpu::IsReadableRange(d.cbufs[world_binding].base, 12 * sizeof(float)))
      world = reinterpret_cast<const float*>(d.cbufs[world_binding].base);
    const u64 vp_offset = static_cast<u64>(vp_dword) * sizeof(float);
    if (vp_binding < d.num_cbufs &&
        d.cbufs[vp_binding].size >= vp_offset + 16 * sizeof(float) &&
        gpu::IsReadableRange(d.cbufs[vp_binding].base + vp_offset,
                             16 * sizeof(float)))
      vp = reinterpret_cast<const float*>(d.cbufs[vp_binding].base) + vp_dword;
    for (u32 v = 0; world && vp && v < 8 && v < d.vertex_count; v++) {
      const float* p = reinterpret_cast<const float*>(
          bytes + static_cast<size_t>(v) * d.vertex_stride);
      float w4[4] = {0, 0, 0, 1};
      for (int r = 0; r < 3; r++)
        w4[r] = world[r * 4 + 0] * p[0] + world[r * 4 + 1] * p[1] +
                world[r * 4 + 2] * p[2] + world[r * 4 + 3];
      float c[4];
      for (int r = 0; r < 4; r++)
        c[r] = vp[r * 4 + 0] * w4[0] + vp[r * 4 + 1] * w4[1] +
               vp[r * 4 + 2] * w4[2] + vp[r * 4 + 3] * w4[3];
      BASE_LOGI("agc",
                "  proj v{} obj=({} {} {}) world=({} {} {}) "
                "clip=({} {} {} {}) ndc=({} {})",
                v, p[0], p[1], p[2], w4[0], w4[1], w4[2], c[0], c[1], c[2],
                c[3], c[3] ? c[0] / c[3] : 0.f, c[3] ? c[1] / c[3] : 0.f);
    }
  }
  // How many floats of each bound cbuffer to print: the default shows the head,
  // but a transform hides further in (a 48-dword window holds several).
  for (u32 b = 0; b < d.num_cbufs; b++) {
    const int count = std::min<int>(kCbFloats, d.cbufs[b].size / 4);
    if (count <= 0 ||
        !gpu::IsReadableRange(d.cbufs[b].base, count * sizeof(float)))
      continue;
    const float* cf = reinterpret_cast<const float*>(d.cbufs[b].base);
    base::String line;
    base::FormatTo(line, "  cbuf[{}]@{:#x} ({} dw) floats:", b, d.cbufs[b].base,
                   d.cbufs[b].size / 4);
    for (int j = 0; j < count; j++) {
      if (j && j % 4 == 0)
        base::FormatTo(line, " |");
      base::FormatTo(line, " {}", cf[j]);
    }
    BASE_LOGI("agc", "{}", line.c_str());
  }
}

void TraceBeginFrame() {
  if (Detail())
    BASE_LOGI("agc", "DL BeginFrame...");
}

void TraceDrawSubmit(const rhi::DrawInfo& d) {
  if (!Detail())
    return;
  BASE_LOGI("agc",
            "DL draw#{} rhi::Draw num_vattrs={} rt={:#x} tmask={:#x} cc={:#x} "
            "blend={} dv={} db={:#x} dt={} dw={} df={} ntex={} tex0={:#x}",
            CurrentDraw(), d.num_vattrs, d.rt_base, d.target_mask,
            d.color_control, d.blend_enable, d.depth_valid, d.depth_base,
            d.depth_test_enable, d.depth_write_enable, d.depth_func, d.num_texs,
            d.num_texs ? d.texs[0].base : 0);
  for (u32 i = 0; i < d.num_texs; i++)
    BASE_LOGI("agc",
              "  DL tex{} base={:#x} {}x{} dfmt={} nfmt={} tiling={} pitch={}",
              i, d.texs[i].base, d.texs[i].w, d.texs[i].h, d.texs[i].dfmt,
              d.texs[i].nfmt, d.texs[i].tiling, d.texs[i].pitch);
  BASE_LOGI("agc",
            "  DL vtx data={:#x} stride={} count={} num_vbufs={} idx={:#x} "
            "icount={}",
            d.vertex_data, d.vertex_stride, d.vertex_count, d.num_vbufs,
            d.index_data, d.index_count);
  for (u32 i = 0; i < d.num_vbufs; i++)
    BASE_LOGI("agc", "  DL vbuf{} data={:#x} stride={} nrec={}", i,
              d.vbufs[i].data, d.vbufs[i].stride, d.vbufs[i].num_records);
}

void TraceDrawDone() {
  if (Detail())
    BASE_LOGI("agc", "DL draw#{} done", CurrentDraw());
}

void TraceShaderListing(u64 address) {
  static bool done = false;
  if (!kDumpSh || done || address != kDumpSh || !IsGuestAddress(address) ||
      !gpu::IsReadableRange(address, kMaxShaderBytes))
    return;
  done = true;
  DumpProgram("SHADER", address);
}

// --- compute ---------------------------------------------------------------

void NoteDispatch(u64 cs_addr, const u32 threads[3], u32 rsrc2) {
  if (!kCsDump)
    return;
  static u64 n_total = 0, n_valid = 0;
  static std::unordered_set<u64> seen;
  n_total++;
  if (IsGuestAddress(cs_addr))
    n_valid++;
  seen.insert(cs_addr);
  if ((n_total % 2000) != 0)
    return;
  base::String line;
  int shown = 0;
  for (u64 a : seen) {
    if (shown++ >= 8)
      break;
    base::FormatTo(line, " {:#x}", a);
  }
  BASE_LOGI("cs", "dispatches={} valid={} unique={}:{} rsrc2={:08x} "
                  "tg=[{} {} {}]",
            n_total, n_valid, seen.size(), line.c_str(), rsrc2, threads[0],
            threads[1], threads[2]);
}

void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 rsrc2) {
  static std::unordered_set<u64> dumped;
  if (!kCsDump || dumped.size() >= 24 || !IsGuestAddress(cs_addr) ||
      !gpu::IsReadableRange(cs_addr, kMaxShaderBytes) ||
      !dumped.insert(cs_addr).second)
    return;
  const u32* ud = regs.At(mmCOMPUTE_USER_DATA_0);
  BASE_LOGI("cs", "addr={:#x} groups=[{} {} {}] tg=[{} {} {}] rsrc2={:08x}",
            cs_addr, groups[0], groups[1], groups[2], threads[0], threads[1],
            threads[2], rsrc2);
  BASE_LOGI("cs", "  user_data:{}", Words(ud, 16).c_str());
  // Follow each user-data pointer pair one level: the descriptor tables the CS
  // dereferences say which surfaces it actually reads and writes.
  for (int k = 0; k < 15; k++) {
    const u64 p = (static_cast<u64>(ud[k + 1] & 0xFFFF) << 32) | ud[k];
    if (!IsGpuAddress(p) || !gpu::IsReadableRange(p, 8 * sizeof(u32)))
      continue;
    BASE_LOGI("cs", "  ud{} -> {:#x}:{}", k, p,
              Words(reinterpret_cast<const u32*>(p), 8).c_str());
  }
  const auto prog =
      rdna::DecodeShader(reinterpret_cast<const u32*>(cs_addr), 4096);
  u32 hist[24] = {}, flat_seg[4] = {}, flat_ops[128] = {};
  for (const auto& in : prog) {
    const u32 e = static_cast<u32>(in.enc);
    if (e < 24)
      hist[e]++;
    if (in.enc == gcn::Enc::kFlat) {
      flat_seg[(in.raw[0] >> 14) & 3]++;
      flat_ops[(in.raw[0] >> 18) & 0x7F]++;
    }
  }
  base::String enc_hist, flat_hist;
  for (u32 e = 0; e < 24; e++)
    if (hist[e])
      base::FormatTo(enc_hist, " {}={}", e, hist[e]);
  for (u32 o = 0; o < 128; o++)
    if (flat_ops[o])
      base::FormatTo(flat_hist, " {:#x}={}", o, flat_ops[o]);
  BASE_LOGI("cs", "  insts={} enc:{} flatseg: {}/{}/{}/{} ops:{}", prog.size(),
            enc_hist.c_str(), flat_seg[0], flat_seg[1], flat_seg[2],
            flat_seg[3], flat_hist.c_str());
}

void TraceCsUnsupported(u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr) {
  // Once per shader: a title dispatches the same unsupported shader every
  // frame, and repeating it would spend the whole skip budget on one of them.
  static std::unordered_set<u64> reported;
  if (reported.insert(cs_addr).second && CsReport())
    BASE_LOGI("csgpu",
              "unsupported CS @{:#x} groups=[{} {} {}] tg=[{} {} {}] usgpr={} "
              "-- binding zero-filled",
              cs_addr, groups[0], groups[1], groups[2], threads[0], threads[1],
              threads[2], user_sgpr);
}

void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res, u32 ud_dwords) {
  if (CsReport())
    BASE_LOGI("csgpu",
              "CS @{:#x} bind={} kind={} s{} pc={:#x} has no resolved descriptor "
              "({} user-data dwords) -- dispatch skipped",
              cs_addr, res.binding, res.kind, res.base_sgpr, res.use_pc,
              ud_dwords);
}

void TraceCsUnsupportedImage(u64 cs_addr,
                             u32 binding,
                             const gcn::TImage& image) {
  static std::unordered_set<u64> reported;
  if (reported.size() < 256 &&
      reported.insert((cs_addr << 8) | (binding & 0xFF)).second)
    BASE_LOGI("csgpu",
              "CS @{:#x} bind={} unsupported image base={:#x} type={} dfmt={} "
              "nfmt={} tiling={:#x} {}x{} pitch={} valid={} tiling_ok={} "
              "-- dispatch skipped",
              cs_addr, binding, image.base, image.type, image.dfmt, image.nfmt,
              image.tiling_idx, image.width, image.height, image.pitch,
              image.valid, gcn::TilingSupported(image.tiling_idx));
}

void TraceCsNoLinearStaging(u64 cs_addr,
                            u32 binding,
                            const gcn::TImage& image) {
  if (CsReport())
    BASE_LOGI("csgpu",
              "CS @{:#x} bind={} image {}x{} has no linear staging layout "
              "-- dispatch skipped",
              cs_addr, binding, image.width, image.height);
}

void TraceCsInvalidRange(u64 cs_addr,
                         const gcn::CsResource& res,
                         u64 base,
                         u64 guest_size) {
  if (CsReport())
    BASE_LOGI("csgpu",
              "CS @{:#x} bind={} kind={} unusable range base={:#x} size={:#x} "
              "-- dispatch skipped",
              cs_addr, res.binding, res.kind, base, guest_size);
}

void TraceCsTooManyResources(u64 cs_addr, u32 max_resources) {
  if (CsReport())
    BASE_LOGI("csgpu", "CS @{:#x} needs more than {} resources -- dispatch "
                       "skipped",
              cs_addr, max_resources);
}

// One line per shader: a global cap spends itself on the loading screens and
// then says nothing about the dispatch that goes missing at the frontier.
void TraceCsDispatchFailed(u64 cs_addr, u32 num_resources) {
  static std::unordered_set<u64> reported;
  if (reported.size() < 256 && reported.insert(cs_addr).second)
    BASE_LOGI("csgpu", "CS @{:#x} dispatch failed ({} resources)", cs_addr,
              num_resources);
}

void TraceCsResource(u64 cs_addr,
                     const gcn::CsResource& res,
                     u64 base,
                     u64 size,
                     u64 guest_size,
                     bool zero_fill) {
  if (kResTrace)
    BASE_LOGI("csres",
              "cs={:#x} bind={} kind={} s{} pc={:#x} base={:#x} size={:#x} "
              "guest={:#x} written={} zero={}",
              cs_addr, res.binding, res.kind, res.base_sgpr, res.use_pc, base,
              size, guest_size, res.written ? 1 : 0, zero_fill ? 1 : 0);
}

void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources) {
  if (kResTrace)
    BASE_LOGI("csres", "cs={:#x} dispatch {} ({} resources)", cs_addr,
              executed ? "executed" : "failed", num_resources);
}

// --- the packet stream -----------------------------------------------------

void NoteOpcode(u32 op) {
  g_op_hist[op & 0xFF]++;
  if (!kOpCensus)
    return;
  static const bool started = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        BASE_LOGI("agc", "=== opcode census (tick) ===");
        DumpOpcodeHistogram();
      }
    }).detach();
    return true;
  }();
  (void)started;
}

void NoteSkippedOpcode(u32 op, const char* why) {
  g_skipped[op & 0xFF] = true;
  g_skip_reason[op & 0xFF] = why;
}

void NoteUnhandledOpcode(u32 op,
                         u32 hdr,
                         u32 position,
                         u32 words,
                         const u32* body,
                         u32 count,
                         u32 prev_op) {
  g_skipped[op & 0xFF] = true;
  // One line per distinct opcode, 32 lines total per boot: the census dump
  // carries the counts beyond that.
  static u32 logged[32] = {};
  static int n_logged = 0;
  for (int k = 0; k < n_logged; k++)
    if (logged[k] == op)
      return;
  if (n_logged == 32)
    return;
  logged[n_logged++] = op;
  base::String line;
  base::FormatTo(line,
                 "unhandled op {:#04x} count={} (hdr {:08x} at {}/{} dwords) "
                 "after op {:#04x}: body:",
                 op, count, hdr, position, words, prev_op);
  for (u32 k = 0; k < count && k < 8; k++)
    base::FormatTo(line, " {:08x}", body[k]);
  BASE_LOGI("agc", "{}", line.c_str());
}

void TraceOpcodeBody(u32 op, const u32* body, u32 count) {
  if (op != kOpDump)
    return;
  static int n = 0;
  if (n++ >= 12)
    return;
  base::String line;
  base::FormatTo(line, "op={:#04x} cnt={} body:", op, count);
  for (u32 b = 0; b < count && b < 20; b++)
    base::FormatTo(line, " {:08x}", body[b]);
  BASE_LOGI("opdump", "{}", line.c_str());
}

void TraceColorBaseSource(u32 op, u32 color0_base) {
  if (!kTrace)
    return;
  static u32 last = 0, prev_op = 0;
  static int n = 0;
  if (color0_base != last) {
    if (n++ < 16)
      BASE_LOGI("agc", "CB0BASE {:08x} -> {:08x} by op {:#04x} (next {:#04x})",
                last, color0_base, prev_op, op);
    last = color0_base;
  }
  prev_op = op;
}

void TraceDcbPacket(u32 position, u32 op, const u32* body, u32 count) {
  base::String line;
  base::FormatTo(line, "  @{:<5} T3 op={:#04x} count={} body:", position, op,
                 count);
  const u32 shown = (op == 0x93 || op == 0x79) ? count : std::min(count, 6u);
  for (u32 b = 0; b < shown && b < 24; b++)
    base::FormatTo(line, " {:08x}", body[b]);
  // An indirect register packet references a GPU buffer at body[0..1]; dump it
  // so the register layout it carries can be read off.
  if ((op == 0x9f || op == 0x64 || op == 0x7a || op == 0x63) && count >= 2) {
    const u64 a = (static_cast<u64>(body[1] & 0xFFFF) << 32) | body[0];
    if (IsGpuAddress(a) && gpu::IsReadableRange(a, 12 * sizeof(u32)))
      base::FormatTo(line, " -> buf {:#x}:{}", a,
                     Words(reinterpret_cast<const u32*>(a), 12).c_str());
  }
  BASE_LOGI("agc", "{}", line.c_str());
}

// Followed vs refused INDIRECT_BUFFERs. A chain we refuse takes every draw in
// it with it, and the only symptom is a frame that renders nothing.
void TraceIndirectBuffer(u64 address, u32 words, bool followed) {
  static std::atomic<u64> ok{0}, skipped{0};
  (followed ? ok : skipped).fetch_add(1, std::memory_order_relaxed);
  if (!kWalkStat)
    return;
  const u64 n = ok.load() + skipped.load();
  if ((n % 2000) == 1)
    BASE_LOGI("walkstat", "indirect buffers: followed={} refused={} (last {:#x} x{} dw {})",
              (unsigned long long)ok.load(), (unsigned long long)skipped.load(),
              (unsigned long)address, words, followed ? "ok" : "REFUSED");
}

// The first few occlusion-query dumps: a title that never gets one waits for a
// result bit that never arrives, and that is invisible in any other log.
void TraceOcclusionQuery(u64 address, u64 value) {
  static std::atomic<u64> n{0};
  const u64 i = n.fetch_add(1);
  if (i < 3 || (i % 4000) == 0)
    BASE_LOGI("agc", "occlusion query #{} -> {:#x} = {:#x} (always visible)",
              (unsigned long long)i, (unsigned long)address,
              (unsigned long long)value);
}

void TraceResync(u32 position, u32 words, u32 hdr, u32 op, u32 count) {
  static u64 resyncs = 0;
  if (kWalkStat && (++resyncs % 500) == 1)
    BASE_LOGI("walkstat",
              "resync #{} at word {}/{} (hdr {:08x} op {:#x} cnt {})", resyncs,
              position, words, hdr, op, count);
}

void TraceType0ShaderRegs(u32 first_reg, const u32* values, u32 count) {
  if (!kTrace || first_reg < kShRegBase || first_reg >= kShRegBase + 0x300)
    return;
  static int n = 0;
  if (n++ < 20)
    BASE_LOGI("agc", "  type0 SH write base={:#x} cnt={} v0={:08x} v1={:08x}",
              first_reg, count, values[0], count > 1 ? values[1] : 0);
}

void TraceDmaData(u32 control, u64 src, u64 dst, u32 bytes, bool copied) {
  if (!kGpuDmatrace)
    return;
  static int n = 0;
  if (n++ < 60)
    BASE_LOGI("dma", "ctrl={:#x} src={:#x} dst={:#x} bytes={}{}", control, src,
              dst, bytes, copied ? " COPIED" : "");
}

// --- submissions -----------------------------------------------------------

namespace {
// The all-zero ACQRB ring submits (the 0xC0408121 path is empty for the mode-1
// titles) say nothing; a submission is interesting once it carries packets.
bool NonEmpty(const void* dcb, u32 words) {
  const u32* w = static_cast<const u32*>(dcb);
  return words >= 2 && (w[0] || w[1]);
}
}  // namespace

bool TraceSubmit(const void* dcb, u32 size_bytes, u32 words, u64 submission) {
  static int dumped = 0;
  if (!kTrace || dumped >= 6 || !NonEmpty(dcb, words))
    return false;
  dumped++;
  const u32* w = static_cast<const u32*>(dcb);
  BASE_LOGI("agc", "=== dcb walk #{} (size={} words={} hdr0={:#x}) ===",
            submission, size_bytes, words, w[0]);
  const u32 raw_n = std::min(words, 100u);  // enough of a big draw buffer
  base::String raw;
  base::FormatTo(raw, "  raw[0..{}]:{}", raw_n, Words(w, raw_n).c_str());
  BASE_LOGI("agc", "{}", raw.c_str());
  return true;
}

void TraceOpcodeCensus(const void* dcb, u32 words, u64 submission) {
  if ((!kTrace && !kOpHist) || !NonEmpty(dcb, words) ||
      (submission % 2000) != 0)
    return;
  BASE_LOGI("agc", "=== global opcode census @submit {} ===", submission);
  DumpOpcodeHistogram();
}

void TraceWalkDone() {
  BASE_LOGI("agc", "=== dcb walk done; opcode histogram ===");
  DumpOpcodeHistogram();
}

void TraceRegShadowScan() {
  if (!kTrace)
    return;
  static u64 submits = 0;
  constexpr u64 kShadowAddress = 0x8002860000ull;
  constexpr u64 kShadowBytes = 0x200000;
  if (++submits != 2000 ||
      !gpu::IsReadableRange(kShadowAddress, kShadowBytes))
    return;
  const u32* sh = reinterpret_cast<const u32*>(kShadowAddress);
  int shown = 0;
  for (u32 w = 0; w < (kShadowBytes / 4) && shown < 16; w += 2) {
    const u32 lo = sh[w], hi = sh[w + 1];
    const u64 a = (static_cast<u64>(lo) << 8) | (static_cast<u64>(hi & 0xFF) << 40);
    if (IsGpuAddress(a)) {
      BASE_LOGI("agc", "  SHADOW+{:#x} lo={:08x} hi={:08x} -> addr {:#x}",
                w * 4, lo, hi, a);
      shown++;
    }
  }
  if (!shown)
    BASE_LOGI("agc", "  SHADOW scan: 2MB all-zero (no PGM written)");
}

}  // namespace gpu::ps5
