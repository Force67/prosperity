/*
 * PS4Delta : PS4 emulation and research project
 *
 * The DELTA_GPU_* instrumentation of the PS4 command stream. See cmd_trace.h.
 */

#include "gpu/ps4/cmd_trace.h"
#include "base/arch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/mem.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/pm4.h"

// Each probe logs on its own channel, named after the tag it has always
// printed, so the lines still grep as [drawpkt], [csres] and so on, and one
// probe can be silenced with base::SetChannelMinLevel. Delivery is the async
// sink the composition root installs: a trace that blocks the submit thread
// distorts the frame timings it exists to explain.

namespace {

DELTA_OPTION(bool, kTrace, "DELTA_GPU_TRACE", false);
DELTA_OPTION(u64, kAddrWatch, "DELTA_GPU_ADDRWATCH", 0);
DELTA_OPTION(bool, kBlitDump, "DELTA_GPU_BLITDUMP", false);
DELTA_OPTION(u64, kBlitRt, "DELTA_GPU_BLIT_RT", 0);
DELTA_OPTION(bool, kCbInfoTrace, "DELTA_GPU_CBINFO", false);
DELTA_OPTION(bool, kCcbHist, "DELTA_GPU_CCBHIST", false);
DELTA_OPTION(bool, kCeTrace, "DELTA_GPU_CETRACE", false);
DELTA_OPTION(int, kCeTraceMax, "DELTA_GPU_CETRACE_MAX", 200);
DELTA_OPTION(bool, kCounterTrace, "DELTA_GPU_COUNTERTRACE", false);
DELTA_OPTION(bool, kCsDrops, "DELTA_GPU_CSDROPS", false);
DELTA_OPTION(bool, kCsDump, "DELTA_GPU_CSDUMP", false);
DELTA_OPTION(u64, kCsResTrace, "DELTA_GPU_CSRES", 0);
DELTA_OPTION(u64, kCsWatch, "DELTA_GPU_CSWATCH", 0);
DELTA_OPTION(bool, kDbTrace, "DELTA_GPU_DBTRACE", false);
DELTA_OPTION(u64, kDbWatch, "DELTA_GPU_DBWATCH", 0);
DELTA_OPTION(bool, kDcbStat, "DELTA_GPU_DCBSTAT", false);
DELTA_OPTION(bool, kDesyncTrace, "DELTA_GPU_DESYNC", false);
DELTA_OPTION(bool, kDrawList, "DELTA_GPU_DRAWLIST", false);
DELTA_OPTION(int, kDlAfter, "DELTA_GPU_DRAWLIST_AFTER", 90);
DELTA_OPTION(bool, kDrawPkt, "DELTA_GPU_DRAWPKT", false);
DELTA_OPTION(bool, kDmaTrace, "DELTA_GPU_DMATRACE", false);
DELTA_OPTION(bool, kEopTrace, "DELTA_GPU_EOPTRACE", false);
DELTA_OPTION(bool, kEudFail, "DELTA_GPU_EUDFAIL", false);
DELTA_OPTION(bool, kGeomDump, "DELTA_GPU_GEOMDUMP", false);
DELTA_OPTION(u32, kGeomMin, "DELTA_GPU_GEOMMIN", 500);
DELTA_OPTION(bool, kIbTrace, "DELTA_GPU_IBTRACE", false);
DELTA_OPTION(bool, kMaskTrace, "DELTA_GPU_MASKTRACE", false);
DELTA_OPTION(int, kMtAfter, "DELTA_GPU_MASKTRACE_AFTER", 0);
DELTA_OPTION(int, kMtMax, "DELTA_GPU_MASKTRACE_MAX", 200);
DELTA_OPTION(u64, kMtRt, "DELTA_GPU_MASKTRACE_RT", 0);
DELTA_OPTION(bool, kNoMrtTrace, "DELTA_GPU_NOMRT", false);
DELTA_OPTION(bool, kOpHist, "DELTA_GPU_OPHIST", false);
DELTA_OPTION(int, kOhAfter, "DELTA_GPU_OPHIST_AFTER", 100);
DELTA_OPTION(bool, kOpTrace, "DELTA_GPU_OPTRACE", false);
DELTA_OPTION(u64, kPsInCntl, "DELTA_GPU_PSINCNTL", 0);
DELTA_OPTION(bool, kRawBufTrace, "DELTA_GPU_RAWBUF", false);
DELTA_OPTION(int, kRegSrcFrame, "DELTA_GPU_REGSRC_FRAME", -1);
DELTA_OPTION(u64, kRegSrcPs, "DELTA_GPU_REGSRC_PS", 0);
DELTA_OPTION(u64, kRootWprotHash, "DELTA_GPU_ROOT_WPROT_HASH", 0);
DELTA_OPTION(int, kRootWprotMs, "DELTA_GPU_ROOT_WPROT_MS", 1);
DELTA_OPTION(u64, kRootWprotPs, "DELTA_GPU_ROOT_WPROT_PS", 0);
DELTA_OPTION(bool, kRootWprotStep, "DELTA_GPU_ROOT_WPROT_STEP", false);
DELTA_OPTION(bool, kShReloc, "DELTA_GPU_SHRELOC", false);
DELTA_OPTION(bool, kSpriteDis, "DELTA_GPU_SPRITEDIS", false);
DELTA_OPTION(bool, kSpriteDump, "DELTA_GPU_SPRITEDUMP", false);
DELTA_OPTION(bool, kTexfmt, "DELTA_GPU_TEXFMT", false);
DELTA_OPTION(int, kTexTrackFrame, "DELTA_GPU_TEXTRACK_FRAME", -1);
DELTA_OPTION(bool, kVattrDump, "DELTA_GPU_VATTRDUMP", false);
DELTA_OPTION(bool, kWaitTrace, "DELTA_GPU_WAITTRACE", false);

// Several probes only begin logging once the run is past its opening floods (a
// title screen, a level load). Each keeps its own epoch: they are asked
// different questions ("N seconds after the first draw", "after the first
// walk") and one shared clock would start wherever it was first read.
class Elapsed {
 public:
  double Seconds() {
    if (!started_) {
      started_ = true;
      start_ = std::chrono::steady_clock::now();
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_)
        .count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
  bool started_ = false;
};

}  // namespace

namespace gpu::ps4 {
namespace {

// Where each register was last written from, and how often the colour masks
// have been programmed. Nothing but the probes below reads either.
const u32* g_reg_sources[kRegFileSize] = {};
u32 g_shader_mask_writes = 0;
u32 g_target_mask_writes = 0;

WriteWatchFn g_write_watch = nullptr;

// Opcode histogram and per-opcode walk time of the draw command buffers.
u32 g_op_hist[256] = {};
u64 g_op_ns[256] = {};
u64 g_dcb_words = 0;
u64 g_dcb_packets = 0;
int g_dcb_seen = 0;

u32 g_ccb_hist[256] = {};
u64 g_ccb_count = 0;
int g_ccb_hist_dumps = 0;

const u32* UserData(const Regs& regs, u32 first_reg) {
  return regs.At(first_reg);
}

// base::FormatTo appends, so a line assembled a piece at a time is one log
// entry rather than one per piece.
base::String HexRun(const u32* values, u32 count) {
  base::String out;
  for (u32 i = 0; i < count; i++)
    base::FormatTo(out, " {:08x}", values[i]);
  return out;
}

void LogUserData(const char* channel, const char* label, const u32* ud) {
  BASE_LOGI(channel, "{}{}", label, HexRun(ud, 16).c_str());
}

}  // namespace

const char* RecompStatusName(RecompStatus status) {
  switch (status) {
    case RecompStatus::kOk:
      return "ok";
    case RecompStatus::kDisabled:
      return "disabled";
    case RecompStatus::kSkipped:
      return "skipsh";
    case RecompStatus::kBadAddress:
      return "bad-address";
    case RecompStatus::kRejected:
      return "rejected";
    case RecompStatus::kBadAttrs:
      return "bad-attrs";
    case RecompStatus::kAttrBindings:
      return "attr-nbind";
    case RecompStatus::kAttrOffset:
      return "attr-offset";
  }
  return "?";
}

// --- register file ---------------------------------------------------------

void NoteRegisterWrites(u32 first_reg,
                        const u32* values,
                        u32 count,
                        u32 packet_base) {
  for (u32 i = 0; i < count; i++) {
    const u32 reg = first_reg + i;
    if (reg < kRegFileSize)
      g_reg_sources[reg] = &values[i];
    if (reg == mmCB_SHADER_MASK)
      g_shader_mask_writes++;
    else if (reg == mmCB_TARGET_MASK)
      g_target_mask_writes++;
    // SET_*_REG packets only. A type-0 write (packet_base 0) reaches the same
    // registers but would spend this probe's budget on a packet class it was
    // never asked about.
    if (!kCbInfoTrace || !packet_base)
      continue;
    // Every write of a CB_COLORn_INFO, with the packet that carried it. A
    // colour target whose INFO stays zero is never bound, so a whole pass
    // renders into nothing; this says whether the title wrote a zero or we
    // never saw the write at all.
    for (int rt = 0; rt < 8; rt++) {
      if (reg != mmCB_COLOR0_INFO + rt * kCbColorStride)
        continue;
      static std::atomic<u64> nonzero{0}, zero{0};
      static int shown = 0;
      (values[i] ? nonzero : zero).fetch_add(1);
      if (shown < 24) {
        shown++;
        BASE_LOGI("cbinfo",
                  "cb{} = {:#x} (packet base={:#x} reg={:#x} count={} word={})",
                  rt, values[i], packet_base, reg, count, i);
      }
      const u64 total = nonzero.load() + zero.load();
      if ((total % 20000) == 0)
        BASE_LOGI("cbinfo", "{} non-zero, {} zero",
                  (unsigned long long)nonzero.load(),
                  (unsigned long long)zero.load());
    }
  }
}

// --- draws -----------------------------------------------------------------

void SetWriteWatch(WriteWatchFn watch) {
  g_write_watch = watch;
}

void MaybeArmRootWriteWatch(const Regs& regs,
                            u64 ps_addr,
                            u32 frame) {
  static bool armed = false;
  if (armed || !g_write_watch || (!kRootWprotPs && !kRootWprotHash))
    return;
  u64 hash = 0;
  if (kRootWprotHash && IsGuestAddress(ps_addr))
    hash = gcn::CachedCodeHash(ps_addr, 4096);
  const bool match = (kRootWprotPs && kRootWprotPs == ps_addr) ||
                     (kRootWprotHash && kRootWprotHash == hash);
  if (!match)
    return;
  const u32* user_data = UserData(regs, mmSPI_SHADER_USER_DATA_PS_0);
  const u64 root =
      (static_cast<u64>(user_data[1] & 0xFFFF) << 32) | user_data[0];
  if (!IsGuestAddress(root) ||
      !utl::isMemoryRangeMapped(reinterpret_cast<const void*>(root), 64))
    return;
  armed = true;
  constexpr size_t kRootSize = 64;
  constexpr size_t kRootPoolSpan = 64 * 1024;
  const size_t watch_size = kRootWprotStep ? kRootSize : kRootPoolSpan;
  const unsigned interval =
      static_cast<unsigned>(std::max(0, kRootWprotMs.get()));
  BASE_LOGI(
      "root-wprot",
      "f{} PS={:#x} hash={:#x} root={:#x} watching-forward={:#x} every={}ms{}",
      frame, (unsigned long)ps_addr, (unsigned long)hash, (unsigned long)root,
      watch_size, interval, kRootWprotStep ? " single-step" : "");
  g_write_watch(root, watch_size, interval, false, kRootWprotStep);
}

void TraceRegisterSources(const Regs& regs, u64 ps_addr, u32 frame) {
  if (!kRegSrcPs || ps_addr != kRegSrcPs)
    return;
  if (kRegSrcFrame >= 0 && static_cast<u32>(kRegSrcFrame) != frame)
    return;
  static u32 reports = 0;
  if (reports++ >= 64)
    return;
  base::String line;
  for (u32 i = 0; i < 16; i++) {
    const u32 reg = mmSPI_SHADER_USER_DATA_PS_0 + i;
    base::FormatTo(line, " s{}={:08x}@{:#x}", i, regs[reg],
                   reinterpret_cast<u64>(g_reg_sources[reg]));
  }
  BASE_LOGI("regsrc", "f{} PS={:#x} user_data:{}", frame, ps_addr,
            line.c_str());
}

void TraceFirstTexturedPs(const Regs& regs, u64 ps_addr) {
  static bool probed = false;
  if (!kTrace || probed || !IsGuestAddress(ps_addr))
    return;
  const auto program =
      gcn::Decode(reinterpret_cast<const u32*>(ps_addr), 256);
  int n_mimg = 0, n_smrd = 0;
  for (const auto& inst : program) {
    if (inst.enc == gcn::Enc::kMimg)
      n_mimg++;
    if (inst.enc == gcn::Enc::kSmrd)
      n_smrd++;
  }
  if (!n_mimg)
    return;
  probed = true;
  BASE_LOGI("gpu", "TEXTURED PS @{:#x}: mimg={} smrd={}",
            (unsigned long)ps_addr, n_mimg, n_smrd);
  const u32* ud = UserData(regs, mmSPI_SHADER_USER_DATA_PS_0);
  LogUserData("gpu", "  PS user_data:", ud);
  auto texs =
      gcn::TrackTextures(gcn::CachedProgram(ps_addr, 4096), ud, false, ps_addr);
  BASE_LOGI("gpu", "  TrackTextures -> {}", texs.size());
  if (texs.empty() || !texs[0].valid)
    return;
  // Dump the texture as a LINEAR interpretation (to inspect tiling).
  const auto& t = texs[0];
  FILE* f = std::fopen("/tmp/tex_raw.bin", "wb");
  if (!f)
    return;
  std::fwrite(reinterpret_cast<const u8*>(t.base), 1,
              (size_t)t.width * t.height * 4, f);
  std::fclose(f);
  BASE_LOGI("gpu", "  dumped /tmp/tex_raw.bin ({}x{} rgba)", t.width, t.height);
}

void TraceDrawOpcode(u32 op, u32 prim_type, u32 auto_count) {
  if (!kDrawPkt)
    return;
  // A histogram over the WHOLE run, not the first N draws: the counts that
  // decline downstream are not the ones a first-N sample catches.
  struct Bucket {
    u32 op, prim, auto_count, n;
  };
  static Bucket buckets[32];
  static u32 nbuckets = 0, total = 0;
  u32 i = 0;
  for (; i < nbuckets; i++)
    if (buckets[i].op == op && buckets[i].prim == prim_type &&
        buckets[i].auto_count == auto_count)
      break;
  if (i == nbuckets && nbuckets < 32)
    buckets[nbuckets++] = {op, prim_type, auto_count, 0};
  if (i < nbuckets)
    buckets[i].n++;
  if (++total % 2000 != 0)
    return;
  base::String line;
  for (u32 j = 0; j < nbuckets; j++)
    base::FormatTo(line, " op={:#x}/prim={:#x}/auto={} x{}", buckets[j].op,
                   buckets[j].prim, buckets[j].auto_count, buckets[j].n);
  BASE_LOGI("drawpkt", "{} draws:{}", total, line.c_str());
}

void TraceIndexBuffer(u64 index_base,
                      u32 index_count,
                      u32 index_type,
                      bool accepted) {
  if (!kDrawPkt)
    return;
  static int n = 0;
  if (n++ >= 128)
    return;
  BASE_LOGI("drawpkt", "DRAW_INDEX_2 ibase={:#x} icount={} itype={} {}{}",
            (unsigned long)index_base, index_count, index_type,
            accepted ? "accepted" : "REJECTED",
            accepted                 ? ""
            : index_count > 0x100000 ? " (count cap)"
            : !index_count           ? " (zero count)"
                                     : " (base out of range)");
}

void TraceIndirectArgs(u32 op,
                       u64 args_addr,
                       u64 indirect_base,
                       u32 offset,
                       bool mapped,
                       const u32 args[3]) {
  if (!kDrawPkt)
    return;
  static int n = 0;
  if (n++ >= 24)
    return;
  BASE_LOGI("drawpkt",
            "{} args={:#x} (base={:#x} +{:#x}) {} count={} inst={} start={}",
            op == IT_DRAW_INDIRECT ? "INDIRECT" : "INDEX_INDIRECT",
            (unsigned long)args_addr, (unsigned long)indirect_base, offset,
            mapped ? "mapped" : "UNMAPPED", args[0], args[1], args[2]);
}

void TraceIndexOffsetArgs(u32 max_size,
                          u32 index_offset,
                          u32 index_count,
                          u64 index_base,
                          u64 resolved_base,
                          u32 index_type,
                          bool accepted) {
  if (!kDrawPkt)
    return;
  static int n = 0;
  if (n++ >= 32)
    return;
  BASE_LOGI("drawpkt",
            "OFFSET_2 maxsz={} ioff={} icount={} g_index_base={:#x} -> "
            "ibase={:#x} itype={} {}",
            max_size, index_offset, index_count, (unsigned long)index_base,
            (unsigned long)resolved_base, index_type,
            accepted                 ? "accepted"
            : !index_base            ? "REJECTED (no INDEX_BASE set)"
            : !index_count           ? "REJECTED (zero count)"
            : index_count > 0x100000 ? "REJECTED (count cap)"
                                     : "REJECTED (unmapped)");
}

void TraceMrtSlotZeroGap(const rhi::DrawInfo& d,
                         u32 target_mask,
                         u64 vs_addr,
                         u64 ps_addr) {
  if (!kNoMrtTrace || !d.mrt_count || d.mrt_base[0])
    return;
  static std::atomic<u64> n{0};
  const u64 seen = n.fetch_add(1);
  if ((seen % 200) != 0)
    return;
  base::String live;
  for (int rt = 0; rt < 8; rt++)
    if (d.mrt_base[rt])
      base::FormatTo(live, " cb{}={:#x}", rt, d.mrt_base[rt]);
  BASE_LOGI("nomrt",
            "SLOT-0 GAP #{} tmask={:#x} count={} vs={:#x} ps={:#x} live:{}",
            seen + 1, target_mask, d.mrt_count, vs_addr, ps_addr, live.c_str());
}

void TraceNoMrtBound(const Regs& regs,
                     const rhi::DrawInfo& d,
                     u32 target_mask,
                     u64 vs_addr,
                     u64 ps_addr) {
  if (!kNoMrtTrace || !target_mask || d.mrt_count)
    return;
  static int n = 0;
  if (n >= 24)
    return;
  n++;
  BASE_LOGI("nomrt", "tmask={:#x} count={} vs={:#x} ps={:#x}", target_mask,
            d.index_data ? d.index_count : d.vertex_count,
            (unsigned long)vs_addr, (unsigned long)ps_addr);
  for (int rt = 0; rt < 8; rt++)
    BASE_LOGI("nomrt", "  cb{} base={:#x} info={:#x} pitch={:#x} slice={:#x}",
              rt, (unsigned long)regs.CbColorBase(rt),
              regs[mmCB_COLOR0_INFO + rt * kCbColorStride],
              regs[mmCB_COLOR0_PITCH + rt * kCbColorStride],
              regs[mmCB_COLOR0_SLICE + rt * kCbColorStride]);
}

void TraceDepthState(const Regs& regs, u32 z_info) {
  const u32 depth_control = regs[mmDB_DEPTH_CONTROL];
  static int n = 0;
  if (!kDbTrace || n >= 20000 || (!z_info && !depth_control))
    return;
  n++;
  BASE_LOGI("db",
            "DEPTH_CONTROL={:#x} Z_INFO={:#x} Zread={:#x} Zwrite={:#x} "
            "clear={:#x} prim={} size={:#x} slice={:#x}",
            depth_control, z_info,
            static_cast<u64>(regs[mmDB_Z_READ_BASE]) << 8,
            static_cast<u64>(regs[mmDB_Z_WRITE_BASE]) << 8,
            regs[mmDB_DEPTH_CLEAR], regs[mmVGT_PRIMITIVE_TYPE],
            regs[mmDB_DEPTH_SIZE], regs[mmDB_DEPTH_SLICE]);
}

void TraceDepthBaseWatch(const Regs& regs) {
  if (!kDbWatch)
    return;
  static int n = 0;
  const u32 want = static_cast<u32>((u64)kDbWatch >> 8);
  const u32 db_regs[] = {mmDB_Z_READ_BASE, mmDB_STENCIL_READ_BASE,
                              mmDB_Z_WRITE_BASE, mmDB_STENCIL_WRITE_BASE,
                              mmDB_HTILE_DATA_BASE};
  for (u32 reg : db_regs)
    if (regs[reg] == want && n++ < 8)
      BASE_LOGI("dbwatch", "reg {:#x} == {:#x} (shifted {:#x})", reg,
                (unsigned long)kDbWatch, want);
}

bool ShouldTraceTextureTracking(u32 frame, u64 ps_addr) {
  if (kTexTrackFrame < 0 || static_cast<int>(frame) != kTexTrackFrame)
    return false;
  BASE_LOGI("textrack", "f{} ps={:#x}", frame, (unsigned long)ps_addr);
  return true;
}

void TraceTextureFormat(const gcn::TImage& tex, const rhi::DrawInfo& d) {
  static int n = 0;
  if (!kTexfmt || n >= 24)
    return;
  n++;
  BASE_LOGI("texfmt",
            "base={:#x} {}x{} pitch={} dfmt={} nfmt={} tiling={} rt={}x{}",
            (unsigned long)tex.base, tex.width, tex.height, tex.pitch, tex.dfmt,
            tex.nfmt, tex.tiling_idx, d.rt_w, d.rt_h);
}

void TracePsInputCntl(const Regs& regs,
                      u64 ps_addr,
                      u32 ps_input_ena,
                      const u32* ps_in_cntl) {
  if (!kPsInCntl || !ps_input_ena)
    return;
  // With DELTA_GPU_PSINCNTL=1, report only shaders whose mapping is NOT the
  // identity: an identity mapping is a no-op by construction, so those are the
  // only ones honouring these registers can change.
  bool non_identity = false;
  for (u32 i = 0; i < 16; i++)
    non_identity |= (ps_in_cntl[i] & 0x1F) != i;
  if (kPsInCntl == 1 ? !non_identity : ps_addr != (u64)kPsInCntl)
    return;
  static int n = 0;
  if (n++ >= 30)
    return;
  // NUM_INTERP says how many of the 32 slots are meaningful; slots at or above
  // it are don't-care and their zero is not evidence of anything.
  base::String slots;
  for (u32 i = 0; i < 8; i++)
    base::FormatTo(slots, " {}:off={}(raw={:#x})", i,
                   regs[mmSPI_PS_INPUT_CNTL_0 + i] & 0x3F,
                   regs[mmSPI_PS_INPUT_CNTL_0 + i]);
  BASE_LOGI("psincntl", "ps={:#x} ena={:#x} numinterp={} cntl:{}", ps_addr,
            ps_input_ena, regs[mmSPI_PS_IN_CONTROL] & 0x3F, slots.c_str());
}

void TraceShaderCacheMiss(u64 vs_addr,
                          u64 ps_addr,
                          u64 vs_hash,
                          u64 ps_hash,
                          u64 fetch_hash,
                          u32 ps_input_ena,
                          u32 tex_3d_mask,
                          u32 tex_1d_mask) {
  if (!kShReloc)
    return;
  struct Seen {
    u64 addr_vs = 0, addr_ps = 0, fetch = 0;
    u32 ena = 0, t3d = 0, t1d = 0;
    std::unordered_set<u64> fetches, states;
  };
  static std::unordered_map<u64, Seen> seen;  // by content pair
  static u32 n_new = 0, n_addr = 0, n_fetch = 0, n_ena = 0, n_3d = 0,
                  n_1d = 0, n_none = 0, n_total = 0;
  const u64 pair = vs_hash ^ (ps_hash * 0x9e3779b97f4a7c15ull);
  n_total++;
  auto it = seen.find(pair);
  if (it == seen.end()) {
    n_new++;
    it = seen.emplace(pair, Seen{}).first;
  } else {
    const Seen& s = it->second;
    u32 what = 0;
    if (s.addr_vs != vs_addr || s.addr_ps != ps_addr) {
      n_addr++;
      what++;
    }
    if (s.fetch != fetch_hash) {
      n_fetch++;
      what++;
    }
    if (s.ena != ps_input_ena) {
      n_ena++;
      what++;
    }
    if (s.t3d != tex_3d_mask) {
      n_3d++;
      what++;
    }
    if (s.t1d != tex_1d_mask) {
      n_1d++;
      what++;
    }
    if (!what)
      n_none++;
    static int dumped = 0;
    if (what && dumped < 12) {
      dumped++;
      BASE_LOGI("shreloc",
                "miss vs={:#x} ps={:#x} fetch {:#x}->{:#x} ena {:#x}->{:#x} 3d "
                "{:#x}->{:#x} 1d {:#x}->{:#x} addr {:#x}/{:#x}",
                (unsigned long long)vs_hash, (unsigned long long)ps_hash,
                (unsigned long long)s.fetch, (unsigned long long)fetch_hash,
                s.ena, ps_input_ena, s.t3d, tex_3d_mask, s.t1d, tex_1d_mask,
                (unsigned long long)vs_addr, (unsigned long long)ps_addr);
    }
  }
  Seen& s = it->second;
  s.addr_vs = vs_addr;
  s.addr_ps = ps_addr;
  s.fetch = fetch_hash;
  s.ena = ps_input_ena;
  s.t3d = tex_3d_mask;
  s.t1d = tex_1d_mask;
  s.fetches.insert(fetch_hash);
  s.states.insert((static_cast<u64>(ps_input_ena) << 32) |
                  (static_cast<u64>(tex_3d_mask) << 16) | tex_1d_mask);
  if ((n_total & 31) == 0) {
    size_t max_fetch = 0, max_state = 0;
    for (const auto& [key, value] : seen) {
      max_fetch = std::max(max_fetch, value.fetches.size());
      max_state = std::max(max_state, value.states.size());
    }
    BASE_LOGI("shreloc",
              "miss fields: pair-new={} addr={} fetch={} ena={} 3d={} 1d={} "
              "none={} total={} pairs={} max-distinct fetch={} state={}",
              n_new, n_addr, n_fetch, n_ena, n_3d, n_1d, n_none, n_total,
              seen.size(), max_fetch, max_state);
  }
}

void TraceRawBuffer(const char* stage,
                    u64 vs_addr,
                    const gcn::ShaderBuffer& buffer,
                    const gcn::VBuffer& resolved,
                    u64 bytes,
                    bool accepted,
                    const rhi::DrawInfo& d) {
  if (!kRawBufTrace)
    return;
  static int n = 0;
  if (n++ >= 64)
    return;
  BASE_LOGI("rawbuf",
            "{} vs={:#x} bind={} pc={:#x} s{} base={:#x} stride={} nrec={} "
            "bytes={} vcount={} icount={} {}",
            stage, (unsigned long)vs_addr, buffer.binding, buffer.use_pc,
            buffer.srsrc_sgpr, (unsigned long)resolved.base, resolved.stride,
            resolved.num_records, (unsigned long)bytes, d.vertex_count,
            d.index_count, accepted ? "resolved" : "UNRESOLVED");
}

void TraceBlitDraw(const Regs& regs,
                   const rhi::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr) {
  static int n = 0;
  if (!kBlitDump || n >= 6)
    return;
  bool target_bound = !kBlitRt || d.rt_base == kBlitRt;
  for (u32 i = 0; kBlitRt && i < std::min(d.mrt_count, 8u); i++)
    target_bound |= d.mrt_base[i] == kBlitRt;
  if (!target_bound || (!kBlitRt && d.rt_w < 1280))
    return;
  n++;
  BASE_LOGI("blit",
            "#{} rt={:#x} {}x{} VS={:#x} PS={:#x} tex_base={:#x} {}x{} "
            "num_vattrs={} stride={} idx={} blendCtl={:#x}",
            n, (unsigned long)d.rt_base, d.rt_w, d.rt_h, (unsigned long)vs_addr,
            (unsigned long)ps_addr, (unsigned long)d.tex_base, d.tex_w, d.tex_h,
            d.num_vattrs, d.vertex_stride, d.index_count, d.blend_control);
  base::String mrt;
  for (u32 i = 0; i < std::min(d.mrt_count, 8u); i++)
    base::FormatTo(mrt, " {:#x}", d.mrt_base[i]);
  BASE_LOGI("blit", "  MRT({}):{}", d.mrt_count, mrt.c_str());
  BASE_LOGI("blit", "  CB0 pitch={:#x} slice={:#x} info={:#x} attrib={:#x}",
            regs[mmCB_COLOR0_PITCH], regs[mmCB_COLOR0_SLICE],
            regs[mmCB_COLOR0_INFO], regs[mmCB_COLOR0_ATTRIB]);
  if (!IsGuestAddress(ps_addr))
    return;
  auto texs = gcn::TrackTextures(gcn::CachedProgram(ps_addr, 4096),
                                 UserData(regs, mmSPI_SHADER_USER_DATA_PS_0),
                                 false, ps_addr);
  BASE_LOGI("blit", "  TrackTextures -> {} T#", texs.size());
  for (const auto& t : texs)
    BASE_LOGI("blit",
              "    T# base={:#x} {}x{} pitch={} dfmt={} nfmt={} tiling={}",
              (unsigned long)t.base, t.width, t.height, t.pitch, t.dfmt, t.nfmt,
              t.tiling_idx);
  const u32* ps_code = reinterpret_cast<const u32*>(ps_addr);
  const u32 ps_words = gcn::CodeLength(ps_code, 4096);
  gcn::Disassemble(ps_code, ps_words ? ps_words : 512, "blit.PS");
  if (n != 1 || !d.recomp)
    return;
  const u32* ud = UserData(regs, mmSPI_SHADER_USER_DATA_PS_0);
  LogUserData("blit", "  PS user data:", ud);
  for (const auto& cb : d.recomp->ps_cbufs) {
    const auto& resolved = d.cbufs[cb.binding];
    BASE_LOGI("blit", "  PS CB binding={} sgpr={} dwords={} base={:#x} size={}",
              cb.binding, cb.ud_sgpr, cb.num_dwords,
              (unsigned long)resolved.base, resolved.size);
    if (!resolved.base)
      continue;
    const u32* words = reinterpret_cast<const u32*>(resolved.base);
    const u32 rows =
        std::min({(cb.num_dwords + 3) / 4, resolved.size / 16, 64u});
    for (u32 row = 0; row < rows; row++) {
      float values[4];
      std::memcpy(values, words + row * 4, sizeof(values));
      BASE_LOGI("blit",
                "    {:03x}: {:08x} {:08x} {:08x} {:08x} | {:.6g} {:.6g} "
                "{:.6g} {:.6g}",
                row * 4, words[row * 4], words[row * 4 + 1], words[row * 4 + 2],
                words[row * 4 + 3], values[0], values[1], values[2], values[3]);
    }
  }
}

void TraceDrawList(const Regs& regs,
                   const rhi::DrawInfo& d,
                   u64 vs_addr,
                   u64 ps_addr,
                   u64 fetch_addr,
                   RecompStatus status) {
  // The title/menu floods the early run, so logging starts
  // DELTA_GPU_DRAWLIST_AFTER seconds (default 90) after the first draw, by
  // when the level has loaded. Then every draw, to capture the in-level
  // pattern.
  static Elapsed since_first_draw;
  const double elapsed = since_first_draw.Seconds();
  static int n = 0;
  if (!kDrawList || n >= 400 || elapsed < kDlAfter)
    return;
  n++;
  BASE_LOGI("draw",
            "count={} rt={:#x} {}x{} tex={:#x} {}x{} vd={} recomp={} "
            "num_vattrs={} prim={} VS={:#x} PS={:#x} fetch={:#x}",
            d.index_data ? d.index_count : d.vertex_count,
            (unsigned long)d.rt_base, d.rt_w, d.rt_h, (unsigned long)d.tex_base,
            d.tex_w, d.tex_h, d.vertex_data ? 1 : 0, RecompStatusName(status),
            d.num_vattrs, regs[mmVGT_PRIMITIVE_TYPE], (unsigned long)vs_addr,
            (unsigned long)ps_addr, (unsigned long)fetch_addr);
}

void TraceColorMasks(const Regs& regs,
                     const rhi::DrawInfo& d,
                     u64 ps_addr,
                     RecompStatus status) {
  if (!kMaskTrace)
    return;
  // From the first draw of the run, not the first one this reports: with
  // DELTA_GPU_MASKTRACE_RT set those are far apart.
  static Elapsed since_first_draw;
  const double elapsed = since_first_draw.Seconds();
  static int n = 0;
  bool rt_hit = !kMtRt || d.rt_base == kMtRt;
  for (u32 i = 0; kMtRt && i < std::min(d.mrt_count, 8u); i++)
    rt_hit |= d.mrt_base[i] == kMtRt;
  if (!rt_hit || n >= kMtMax || elapsed < kMtAfter)
    return;
  n++;
  const u32 tm = regs[mmCB_TARGET_MASK];
  const u32 sm = regs[mmCB_SHADER_MASK];
  const u32 cc = regs[mmCB_COLOR_CONTROL];
  BASE_LOGI(
      "mask",
      "#{} rt={:#x} {}x{} mrt={} cb0base={:#x} cb1base={:#x} "
      "info1={:#x} TARGET_MASK={:#x} SHADER_MASK={:#x} eff0={:#x} "
      "COLOR_CONTROL={:#x} mode={} rop={:#x} BLEND0={:#x} en={} "
      "psMrtMask={:#x} recomp={} info0={:#x} wr(sm/tm)={}/{} PS={:#x} "
      "count={} DEPTH_CTL={:#x} stencil_en={} gscissor={:#x}..{:#x} "
      "zbase={:#x} zvalid={} ztest={} zwrite={} zfunc={}",
      n, d.rt_base, d.rt_w, d.rt_h, d.mrt_count, regs.CbColorBase(0),
      regs.CbColorBase(1), regs[mmCB_COLOR0_INFO + kCbColorStride], tm, sm,
      (tm & sm) & 0xF, cc, (cc >> 4) & 0x7, (cc >> 16) & 0xFF, d.blend_control,
      unsigned(d.blend_enable), d.recomp ? unsigned(d.recomp->ps_mrt_mask) : 0u,
      RecompStatusName(status), d.mrt_info[0], g_shader_mask_writes,
      g_target_mask_writes, ps_addr,
      d.index_data ? d.index_count : d.vertex_count, regs[mmDB_DEPTH_CONTROL],
      regs[mmDB_DEPTH_CONTROL] & 1, regs[mmPA_SC_GENERIC_SCISSOR_TL],
      regs[mmPA_SC_GENERIC_SCISSOR_BR], d.depth_base, unsigned(d.depth_valid),
      unsigned(d.depth_test_enable), unsigned(d.depth_write_enable),
      d.depth_func);
}

void TraceSpriteDraw(const rhi::DrawInfo& d) {
  static int n = 0;
  if (!kSpriteDump || !d.recomp || d.recomp->ps_texs.empty() ||
      !d.vertex_data || n >= 12)
    return;
  n++;
  const float* m = d.mvp;
  BASE_LOGI("sprite",
            "tex={:#x} {}x{} tiling={} stride={} num_vattrs={} rt={:#x} {}x{} "
            "VS={:#x} PS={:#x} blendCtl={:#x} depth={:#x}/{}/{}/{} mvp=[{:.3f} "
            "{:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} {:.3f} / ... / {:.3f} "
            "{:.3f} {:.3f} {:.3f}]",
            (unsigned long)d.tex_base, d.tex_w, d.tex_h, d.tex_tiling,
            d.vertex_stride, d.num_vattrs, (unsigned long)d.rt_base, d.rt_w,
            d.rt_h, (unsigned long)d.vs_addr, (unsigned long)d.ps_addr,
            d.blend_control, (unsigned long)d.depth_base, d.depth_test_enable,
            d.depth_write_enable, d.depth_func, m[0], m[1], m[2], m[3], m[4],
            m[5], m[6], m[7], m[12], m[13], m[14], m[15]);
  static bool disassembled = false;
  if (kSpriteDis && !disassembled) {
    disassembled = true;
    const u32* ps = reinterpret_cast<const u32*>(d.ps_addr);
    const u32 words = gcn::CodeLength(ps, 4096);
    gcn::Disassemble(ps, words ? words : 512, "sprite.PS");
    LogUserData("sprite", "  PS user data:", d.ps_user_data);
    for (u32 i = 0; i < d.num_texs; i++) {
      const auto& tex = d.texs[i];
      BASE_LOGI("sprite",
                "  tex{}={:#x} {}x{} pitch={} dfmt={} nfmt={} tiling={} "
                "sampler={:08x}/{:08x}/{:08x}/{:08x}",
                i, (unsigned long)tex.base, tex.w, tex.h, tex.pitch, tex.dfmt,
                tex.nfmt, tex.tiling, tex.sampler[0], tex.sampler[1],
                tex.sampler[2], tex.sampler[3]);
    }
    for (const auto& inst : *gcn::CachedProgram(d.ps_addr, 4096)) {
      const u32 w = inst.raw[0], w1 = inst.raw[1];
      if (inst.enc == gcn::Enc::kVintrp) {
        BASE_LOGI("sprite", "  interp pc={:#x} op={} attr={} chan={} v{}",
                  inst.pc, (w >> 16) & 3, (w >> 10) & 0x3f, (w >> 8) & 3,
                  (w >> 18) & 0xff);
      } else if (inst.enc == gcn::Enc::kMimg) {
        BASE_LOGI("sprite",
                  "  image pc={:#x} op={:#x} dmask={:#x} vaddr=v{} vdata=v{} "
                  "srsrc=s{} ssamp=s{} da={}",
                  inst.pc, inst.opcode, (w >> 8) & 0xf, w1 & 0xff,
                  (w1 >> 8) & 0xff, ((w1 >> 16) & 0x1f) * 4,
                  ((w1 >> 21) & 0x1f) * 4, (w >> 14) & 1);
      } else if (inst.enc == gcn::Enc::kExp) {
        BASE_LOGI(
            "sprite",
            "  export pc={:#x} target={} en={:#x} compr={} v=[{} {} {} {}]",
            inst.pc, (w >> 4) & 0x3f, w & 0xf, (w >> 10) & 1, w1 & 0xff,
            (w1 >> 8) & 0xff, (w1 >> 16) & 0xff, (w1 >> 24) & 0xff);
      }
    }
  }
  for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
    const auto& attr = d.vattrs[a];
    if (attr.binding >= d.num_vbufs || !d.vbufs[attr.binding].data)
      continue;
    const auto& binding = d.vbufs[attr.binding];
    base::String vertices;
    for (u32 v = 0; v < std::min(d.vertex_count, 3u); v++) {
      const u8* raw = static_cast<const u8*>(binding.data) +
                           static_cast<size_t>(v) * binding.stride +
                           attr.offset;
      if (attr.dfmt == 10) {
        base::FormatTo(vertices, " v{}=[{:.3f} {:.3f} {:.3f} {:.3f}]", v,
                       raw[0] / 255.f, raw[1] / 255.f, raw[2] / 255.f,
                       raw[3] / 255.f);
      } else {
        const float* f = reinterpret_cast<const float*>(raw);
        base::FormatTo(vertices, " v{}=[{:.4f} {:.4f} {:.4f} {:.4f}]", v, f[0],
                       attr.num_comps > 1 ? f[1] : 0.f,
                       attr.num_comps > 2 ? f[2] : 0.f,
                       attr.num_comps > 3 ? f[3] : 0.f);
      }
    }
    BASE_LOGI("sprite",
              "  attr{} loc={} bind={} off={} stride={} nc={} dfmt={} "
              "nfmt={}{}",
              a, attr.location, attr.binding, attr.offset, binding.stride,
              attr.num_comps, attr.dfmt, attr.nfmt, vertices.c_str());
  }
}

void TraceVertexAttrs(const rhi::DrawInfo& d) {
  static int n = 0;
  if (!kVattrDump || d.num_vattrs < 2 || !d.vertex_data || d.vertex_count > 8 ||
      n >= 24)
    return;
  n++;
  BASE_LOGI("vattr", "vcount={} prim={:#x} nattrs={} nbufs={} rt={:#x}",
            d.vertex_count, d.prim_type, d.num_vattrs, d.num_vbufs,
            (unsigned long)d.rt_base);
  for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
    const auto& attr = d.vattrs[a];
    const auto& vb = d.vbufs[attr.binding < d.num_vbufs ? attr.binding : 0];
    const auto* p = static_cast<const u8*>(vb.data);
    constexpr u32 kBytes = 8;  // covers every <=64-bit attribute format
    base::String v0;
    if (p && utl::isMemoryRangeMapped(p + attr.offset, kBytes))
      for (u32 b = 0; b < kBytes; b++)
        base::FormatTo(v0, "{:02x}", p[attr.offset + b]);
    else
      v0.append("(unmapped)");
    BASE_LOGI("vattr",
              " loc={} bind={} off={} dfmt={} nfmt={} comps={} stride={} v0={}",
              attr.location, attr.binding, attr.offset, attr.dfmt, attr.nfmt,
              attr.num_comps, vb.stride, v0.c_str());
  }
}

void TraceWorldGeometry(const Regs& regs, const rhi::DrawInfo& d) {
  // Sample periodically across the WHOLE run (every 100th qualifying world
  // draw) so we can see whether the camera/view ever moves.
  static int n = 0, seen = 0;
  if (!kGeomDump || d.index_count < kGeomMin || !d.vertex_data)
    return;
  if ((seen++ % 100) != 0 || n >= 300)
    return;
  n++;
  const float* cb =
      d.cbuf_base ? reinterpret_cast<const float*>(d.cbuf_base) : nullptr;
  const auto* vb = static_cast<const u8*>(d.vertex_data);
  const float* p0 = reinterpret_cast<const float*>(vb + d.vattrs[0].offset);
  BASE_LOGI("geom",
            "idx={} ps_texs={} tex={:#x} {}x{} tiling={} pitch={} rt={:#x} "
            "{}x{} blend={:#x} mask={:#x} cc={:#x} num_vattrs={} stride={} "
            "pos0=[{:.1f} {:.1f} {:.1f}] cbuf_base={:#x}",
            d.index_count, d.recomp ? d.recomp->ps_texs.size() : 0,
            (unsigned long)d.tex_base, d.tex_w, d.tex_h, d.tex_tiling,
            d.tex_pitch, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
            d.blend_control, d.target_mask, d.color_control, d.num_vattrs,
            d.vertex_stride, p0[0], p0[1], p0[2], (unsigned long)d.cbuf_base);
  if (cb)
    BASE_LOGI("geom",
              "  cbuf=[{:.3f} {:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} "
              "{:.3f} / {:.3f} {:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} "
              "{:.3f}]",
              cb[0], cb[1], cb[2], cb[3], cb[4], cb[5], cb[6], cb[7], cb[8],
              cb[9], cb[10], cb[11], cb[12], cb[13], cb[14], cb[15]);
  // Project a handful of vertices through the cbuf MVP (both row- and
  // column-major) and count how many land in NDC [-1,1]: on-screen means the
  // black is a PS/sampling issue, off-screen a VS/cbuffer-resolution one.
  if (cb) {
    auto proj = [&](const float* p, bool col_major, float out[4]) {
      float v[4] = {p[0], p[1], p[2], 1.0f};
      for (int r = 0; r < 4; r++) {
        float s = 0;
        for (int c = 0; c < 4; c++)
          s += (col_major ? cb[c * 4 + r] : cb[r * 4 + c]) * v[c];
        out[r] = s;
      }
    };
    int on_r = 0, on_c = 0, on_rfz = 0, on_cfz = 0,
        count = d.index_count < 64 ? d.index_count : 64;
    const u16* i16 = (d.index_type == 0)
                              ? static_cast<const u16*>(d.index_data)
                              : nullptr;
    const u32* i32 = (d.index_type == 1)
                              ? static_cast<const u32*>(d.index_data)
                              : nullptr;
    float first_r[4] = {0}, first_c[4] = {0};
    auto onscreen = [](float* o) {
      if (o[3] <= 0.0001f)
        return false;
      float x = o[0] / o[3], y = o[1] / o[3], z = o[2] / o[3];
      return x >= -1 && x <= 1 && y >= -1 && y <= 1 && z >= -1 && z <= 1;
    };
    for (int i = 0; i < count; i++) {
      // Use the INDEX buffer to fetch the real vertex (these are indexed draws;
      // a linear 0..n read hits unused verts at the buffer head).
      u32 idx = i16 ? i16[i] : i32 ? i32[i] : (u32)i;
      const float* p = reinterpret_cast<const float*>(
          vb + (size_t)idx * d.vertex_stride + d.vattrs[0].offset);
      float r4[4], c4[4];
      proj(p, false, r4);
      proj(p, true, c4);
      // Same but with z negated: does flipping the position z bring the
      // geometry on-screen?
      float pf[3] = {p[0], p[1], -p[2]}, rf[4], cf[4];
      proj(pf, false, rf);
      proj(pf, true, cf);
      if (i == 0) {
        for (int k = 0; k < 4; k++) {
          first_r[k] = r4[k];
          first_c[k] = c4[k];
        }
      }
      if (onscreen(r4))
        on_r++;
      if (onscreen(c4))
        on_c++;
      if (onscreen(rf))
        on_rfz++;
      if (onscreen(cf))
        on_cfz++;
    }
    BASE_LOGI("geom",
              "  proj n={} onscreen row={} col={} | flipZ row={} col={} | "
              "v0row=[{:.2f} {:.2f} {:.2f} {:.2f}] "
              "v0col=[{:.2f} {:.2f} {:.2f} {:.2f}]",
              count, on_r, on_c, on_rfz, on_cfz, first_r[0], first_r[1],
              first_r[2], first_r[3], first_c[0], first_c[1], first_c[2],
              first_c[3]);
    // The game is in real gameplay, so a valid view transform exists. Maybe the
    // VS projects a DIFFERENT attribute than attr0. Project EACH >=3-comp attr
    // (over the indexed verts) and report which, if any, lands on-screen.
    u32 first_idx = i16 ? i16[0] : i32 ? i32[0] : 0;
    for (u32 a = 0; a < d.num_vattrs && a < 8; a++) {
      if (d.vattrs[a].num_comps < 3)
        continue;
      int on_a = 0;
      for (int i = 0; i < count; i++) {
        u32 idx = i16 ? i16[i] : i32 ? i32[i] : (u32)i;
        const float* p = reinterpret_cast<const float*>(
            vb + (size_t)idx * d.vertex_stride + d.vattrs[a].offset);
        float c4[4];
        proj(p, true, c4);
        if (onscreen(c4))
          on_a++;
      }
      const float* pv = reinterpret_cast<const float*>(
          vb + (size_t)first_idx * d.vertex_stride + d.vattrs[a].offset);
      BASE_LOGI("geom",
                "    attr{} off={} nc={} dfmt={} v0=[{:.2f} {:.2f} {:.2f}] "
                "onscreen(col)={}",
                a, d.vattrs[a].offset, d.vattrs[a].num_comps, d.vattrs[a].dfmt,
                pv[0], pv[1], pv[2], on_a);
    }
  }
  // Sample the bound texture's ALPHA: is the source genuinely alpha=0 (so the
  // PS must compute opacity elsewhere / a recompiler alpha bug) or alpha=255
  // (so our load zeroes it)? This decides why the src-alpha blend makes walls
  // invisible.
  if (IsGuestAddress(d.tex_base) && d.tex_w && d.tex_h) {
    const u32* tp = reinterpret_cast<const u32*>(d.tex_base);
    u64 total = (u64)d.tex_w * d.tex_h,
             step = total > 4096 ? total / 4096 : 1, a_nz = 0, rgb_nz = 0;
    for (u64 i = 0; i < total; i += step) {
      u32 px = tp[i];
      if (px >> 24)
        a_nz++;
      if (px & 0x00FFFFFF)
        rgb_nz++;
    }
    BASE_LOGI("geom",
              "  texAlpha: px0={:#010x} alphaNonZero={}/{} rgbNonZero={}",
              tp[0], a_nz, total / step, rgb_nz);
  }
  // The PS's texture-load pattern: SMRD (op/sdst/sbase/imm/off) + MIMG srsrc,
  // plus the first user-data dwords, to see how the T#s are loaded.
  auto ps_insts =
      gcn::Decode(reinterpret_cast<const u32*>(d.ps_addr), 4096);
  const u32* pud = UserData(regs, mmSPI_SHADER_USER_DATA_PS_0);
  BASE_LOGI("geom", "  ps_ud=[{}]", HexRun(pud, 8).c_str());
  for (const auto& inst : ps_insts) {
    if (inst.enc == gcn::Enc::kSmrd) {
      const u32 w = inst.raw[0];
      BASE_LOGI("geom", "  smrd op={} sdst={} sbase={} imm={} off={:#x}",
                (w >> 22) & 0x1F, (w >> 15) & 0x7F, (w >> 9) & 0x3F,
                (w >> 8) & 1, w & 0xFF);
    } else if (inst.enc == gcn::Enc::kMimg) {
      BASE_LOGI("geom", "  mimg srsrc={}", ((inst.raw[1] >> 16) & 0x1F) * 4);
    }
  }
  // The world VS's cbuffer reads: for each s_buffer_load, resolve the V# from
  // the VS user data and print the 16 floats it actually reads (the REAL matrix
  // the VS uses) against the heuristic cbuf above.
  if (!IsGuestAddress(d.vs_addr))
    return;
  auto vs_insts =
      gcn::Decode(reinterpret_cast<const u32*>(d.vs_addr), 4096);
  const u32* vud = UserData(regs, mmSPI_SHADER_USER_DATA_VS_0);
  BASE_LOGI("geom", "  vs_ud=[{}]", HexRun(vud, 8).c_str());
  int shown = 0;
  for (const auto& inst : vs_insts) {
    if (inst.enc != gcn::Enc::kSmrd || shown >= 6)
      continue;
    const u32 w = inst.raw[0], op = (w >> 22) & 0x1F,
                   sbase = (w >> 9) & 0x3F;
    const bool imm = (w >> 8) & 1;
    const u32 off = w & 0xFF;
    shown++;
    const u32 b2 = sbase * 2;
    const u32 boff = imm ? off * 4 : 0;
    // op<8 = s_load (64-bit pointer in ud[b2..b2+1]); op>=8 = s_buffer_load (V#
    // in ud[b2..b2+3], base in the low 44 bits). A 2nd matrix here would be the
    // missing view transform.
    u64 base = 0;
    if (b2 + 1 < 16) {
      if (op < 0x08)
        base = ((u64)vud[b2 + 1] << 32 | vud[b2]);
      else
        base = ((u64)(vud[b2 + 1] & 0xFFF) << 32 | vud[b2]);
    }
    base::String matrix;
    if (base >= 0x1000000ull && base < kGuestEnd) {
      const float* m = reinterpret_cast<const float*>(base + boff);
      base::FormatTo(matrix,
                     " mtx=[{:.2f} {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f} "
                     "{:.2f} / {:.2f} {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} "
                     "{:.2f} {:.2f}]",
                     m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9],
                     m[10], m[11], m[12], m[13], m[14], m[15]);
    }
    BASE_LOGI("geom",
              "  vs_smrd op={} {} sbase={}(ud{}) off={:#x} -> base={:#x}{}", op,
              op < 0x08 ? "sload" : "sbufload", sbase, b2, boff, base,
              matrix.c_str());
  }
}

namespace {

// The embedded "OrbShdr" BinaryInfo in a shader's GCN code, which carries its
// length and hash. Layout: if code[0] == 0xBEEB03FF the info is at code +
// (code[1]+1)*2 dwords; else scan for the 7-byte signature.
void ProbeShaderBinaryInfo(const char* tag, u64 addr) {
  const auto* code = reinterpret_cast<const u32*>(addr);
  const u8* info = nullptr;
  if (code[0] == 0xBEEB03FFu) {
    info = reinterpret_cast<const u8*>(code + (code[1] + 1) * 2);
  } else {
    const auto* bytes = reinterpret_cast<const u8*>(code);
    for (int k = 0; k < 0x4000; k++)
      if (std::memcmp(bytes + k, "OrbShdr", 7) == 0) {
        info = bytes + k;
        break;
      }
  }
  if (!info) {
    BASE_LOGI("gpu", "  {} shader @{:#x}: no OrbShdr (code0={:#x})", tag,
              (unsigned long)addr, code[0]);
    return;
  }
  u32 len_field;
  std::memcpy(&len_field, info + 8, 4);
  u64 hash;
  std::memcpy(&hash, info + 0xC, 8);  // approx offsets
  BASE_LOGI("gpu", "  {} shader @{:#x} OrbShdr len={} hash={:#x}", tag,
            (unsigned long)addr, len_field & 0xFFFFFF, (unsigned long)hash);
}

// The user-data SGPRs, decoding any dword pair that forms a plausible guest
// pointer as a V# (base44, stride, num_records).
void DumpUserDataDescriptors(const char* tag, const u32* ud) {
  LogUserData("gpu", base::Format("  {} user_data:", tag).c_str(), ud);
  for (int k = 0; k + 1 < 16; k += 2) {
    const u64 base = ((u64)(ud[k + 1] & 0xFFF) << 32) | ud[k];
    if (!IsGuestAddress(base))
      continue;
    const u32 stride = (ud[k + 1] >> 16) & 0x3FFF;
    const u32 nrec = ud[k + 2];
    BASE_LOGI("gpu", "    sgpr[{}..]: ptr={:#x} stride={} nrec={} fmt={:#x}", k,
              (unsigned long)base, stride, nrec, ud[k + 3]);
    // A small vertex buffer (a quad): dump it as floats to learn the layout.
    if (!stride || stride > 64 || !nrec || nrec > 8)
      continue;
    const auto* f = reinterpret_cast<const float*>(base);
    const auto* u = reinterpret_cast<const u32*>(base);
    for (u32 v = 0; v < nrec; v++) {
      base::String values;
      for (u32 c = 0; c < stride / 4; c++)
        base::FormatTo(values, " {:g}({:08x})", f[v * (stride / 4) + c],
                       u[v * (stride / 4) + c]);
      BASE_LOGI("gpu", "      v{}:{}", v, values.c_str());
    }
  }
}

// A descriptor table a user-data pointer points at: V# (4 dwords: base48,
// stride, num_records) and T# (8 dwords: base + width/height) heuristics.
void DumpDescriptorTable(const char* tag, u64 ptr) {
  if (!IsGuestAddress(ptr))
    return;
  const auto* t = reinterpret_cast<const u32*>(ptr);
  BASE_LOGI("gpu", "  table {} @{:#x}:", tag, (unsigned long)ptr);
  for (int k = 0; k < 32; k += 4) {
    const u64 b = ((u64)(t[k + 1] & 0xFFFF) << 32) | t[k];
    const u32 stride = (t[k + 1] >> 16) & 0x3FFF;
    if (IsGuestAddress(b) && stride && stride <= 256)
      BASE_LOGI("gpu",
                "    +{:02x} V#? base={:#x} stride={} nrec={} dfmt={:#x}",
                k * 4, (unsigned long)b, stride, t[k + 2], t[k + 3]);
    // T# heuristic: dword2 has width-1[0:13], height-1[14:27]
    const u64 tb = ((u64)(t[k + 1] & 0xFFFFFF) << 32) | t[k];
    const u32 w = (t[k + 2] & 0x3FFF) + 1,
                   h = ((t[k + 2] >> 14) & 0x3FFF) + 1;
    if (IsGuestAddress(tb) && w > 4 && w <= 8192 && h > 4 && h <= 8192)
      BASE_LOGI("gpu", "    +{:02x} T#? base={:#x} {}x{} dfmt={:#x}", k * 4,
                (unsigned long)tb, w, h, (t[k + 1] >> 20) & 0x3F);
  }
}

}  // namespace

void TraceDrawRegisters(const Regs& regs,
                        u32 op,
                        const u32* body,
                        u32 count) {
  if (!kTrace)
    return;
  const u64 vs = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  const u64 ps = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  const u32 sc_tl = regs[mmPA_SC_SCREEN_SCISSOR_TL];
  const u32 sc_br = regs[mmPA_SC_SCREEN_SCISSOR_BR];
  BASE_LOGI("gpu",
            "DRAW op={:#x} prim={} indices={} | RT={:#x} info={:#x} "
            "attrib={:#x} scissor=[{},{}..{},{}] VS={:#x} PS={:#x}",
            op, regs[mmVGT_PRIMITIVE_TYPE], count >= 1 ? body[0] : 0,
            (unsigned long)regs.CbColorBase(0), regs[mmCB_COLOR0_INFO],
            regs[mmCB_COLOR0_ATTRIB], sc_tl & 0xFFFF, sc_tl >> 16,
            sc_br & 0xFFFF, sc_br >> 16, (unsigned long)vs, (unsigned long)ps);

  static bool probed = false;
  if (probed || !vs || !ps)
    return;
  probed = true;
  ProbeShaderBinaryInfo("VS", vs);
  ProbeShaderBinaryInfo("PS", ps);
  DumpUserDataDescriptors("VS", UserData(regs, mmSPI_SHADER_USER_DATA_VS_0));
  DumpUserDataDescriptors("PS", UserData(regs, mmSPI_SHADER_USER_DATA_PS_0));

  const u32* vud = UserData(regs, mmSPI_SHADER_USER_DATA_VS_0);
  DumpDescriptorTable("VS.sgpr0", ((u64)(vud[1] & 0xFFFF) << 32) | vud[0]);
  DumpDescriptorTable("VS.sgpr2", ((u64)(vud[3] & 0xFFFF) << 32) | vud[2]);

  // The fetch shader (sgpr0 ptr, just past the VS code) does the s_load(V#
  // table) + buffer_load(attributes).
  gcn::Disassemble(reinterpret_cast<const u32*>(vs), 512, "VS");
  gcn::Disassemble(reinterpret_cast<const u32*>(ps), 512, "PS");
  const u64 fetch = ((u64)(vud[1] & 0xFFFF) << 32) | vud[0];
  if (!IsGuestAddress(fetch))
    return;
  gcn::Disassemble(reinterpret_cast<const u32*>(fetch), 128, "VS.fetch");
  auto vbs = gcn::TrackVertexBuffers(*gcn::CachedProgram(fetch, 64), vud);
  for (size_t i = 0; i < vbs.size(); i++) {
    const auto& v = vbs[i];
    BASE_LOGI("gpu", "  VB{} base={:#x} stride={} nrec={}", i,
              (unsigned long)v.base, v.stride, v.num_records);
    const auto* f = reinterpret_cast<const float*>(v.base);
    for (u32 r = 0; r < v.num_records && r < 6; r++) {
      base::String values;
      for (u32 c = 0; c < v.stride / 4 && c < 8; c++)
        base::FormatTo(values, " {:g}", f[r * (v.stride / 4) + c]);
      BASE_LOGI("gpu", "    r{}:{}", r, values.c_str());
    }
  }
}

// --- compute ---------------------------------------------------------------

void TraceComputeShader(const Regs& regs,
                        u64 cs_addr,
                        const u32 groups[3],
                        const u32 threads[3],
                        u32 user_sgpr,
                        u32 tgid_enable,
                        u32 lds_dwords) {
  static std::unordered_set<u64> dumped;
  if (!kCsDump || dumped.size() >= 32 || !IsGuestAddress(cs_addr) ||
      !dumped.insert(cs_addr).second)
    return;
  BASE_LOGI(
      "cs",
      "addr={:#x} groups=[{} {} {}] tg=[{} {} {}] usgpr={} tgiden={} lds={}",
      (unsigned long)cs_addr, groups[0], groups[1], groups[2], threads[0],
      threads[1], threads[2], user_sgpr, tgid_enable, lds_dwords);
  LogUserData("cs", "  user_data:", UserData(regs, mmCOMPUTE_USER_DATA_0));
  gcn::Disassemble(reinterpret_cast<const u32*>(cs_addr), 1024, "cs");
}

void TraceDroppedDispatch(u64 cs_addr,
                          const u32 groups[3],
                          const u32 threads[3],
                          const char* reason) {
  if (!kCsDrops)
    return;
  static std::atomic<u64> dropped{0};
  const u64 n = dropped.fetch_add(1) + 1;
  if (n <= 64 || (n % 512) == 0)
    BASE_LOGI("csdrop", "#{} cs={:#x} groups=[{} {} {}] tg=[{} {} {}] ({})", n,
              cs_addr, groups[0], groups[1], groups[2], threads[0], threads[1],
              threads[2], reason);
}

bool ShouldTraceCsResources(u64 cs_addr) {
  if (!kCsResTrace)
    return false;
  if (kCsResTrace > 1)
    return cs_addr == (u64)kCsResTrace;
  static std::unordered_set<u64> traced;
  return traced.size() < 64 && traced.insert(cs_addr).second;
}

bool CsWatchCovers(u64 base, u64 size) {
  return kCsWatch && base <= (u64)kCsWatch &&
         (u64)kCsWatch < base + std::max<u64>(size, 1);
}

void TraceCsUnresolved(u64 cs_addr, const gcn::CsResource& res) {
  BASE_LOGI(
      "csres", "cs={:#x} bind={} kind={} s{} pc={:#x} unresolved; using dummy",
      (unsigned long)cs_addr, res.binding, res.kind, res.base_sgpr, res.use_pc);
}

void TraceCsCode(u64 cs_addr) {
  static std::unordered_set<u64> dumped;
  if (!kEudFail || !dumped.insert(cs_addr).second)
    return;
  const u32* code = reinterpret_cast<const u32*>(cs_addr);
  BASE_LOGI("csdump", "cs={:#x} first 0x360 dwords:", (unsigned long)cs_addr);
  for (u32 k = 0; k < 0x360; k += 8)
    BASE_LOGI("csdump",
              "{:04x}: {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
              k, code[k], code[k + 1], code[k + 2], code[k + 3], code[k + 4],
              code[k + 5], code[k + 6], code[k + 7]);
}

void TraceCsZeroFill(u64 cs_addr,
                     u32 binding,
                     const gcn::TImage& image,
                     const u32* descriptor) {
  BASE_LOGI("csres",
            "cs={:#x} bind={} zero-fill type={} dfmt={} nfmt={} words=[{:08x} "
            "{:08x} {:08x} {:08x}]",
            (unsigned long)cs_addr, binding, image.type, image.dfmt, image.nfmt,
            descriptor[0], descriptor[1], descriptor[2], descriptor[3]);
}

void TraceCsUnsupportedImage(u64 cs_addr,
                             u32 binding,
                             const gcn::TImage& image,
                             const u32* descriptor) {
  BASE_LOGI("csres",
            "cs={:#x} bind={} unsupported image valid={} base={:#x} type={} "
            "dfmt={} nfmt={} tiling={} {}x{} pitch={} layers={} words=[{:08x} "
            "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}]",
            (unsigned long)cs_addr, binding, image.valid ? 1 : 0,
            (unsigned long)image.base, image.type, image.dfmt, image.nfmt,
            image.tiling_idx, image.width, image.height, image.pitch,
            image.layers, descriptor[0], descriptor[1], descriptor[2],
            descriptor[3], descriptor[4], descriptor[5], descriptor[6],
            descriptor[7]);
}

void TraceCsWindowedBuffer(u64 cs_addr,
                           u32 binding,
                           u64 declared_size,
                           u64 mapped_size) {
  BASE_LOGI("csres",
            "cs={:#x} bind={} windowed buffer declared={:#x} mapped={:#x}",
            (unsigned long)cs_addr, binding, (unsigned long)declared_size,
            (unsigned long)mapped_size);
}

void TraceCsResource(u64 cs_addr,
                     const gcn::CsResource& res,
                     u64 base,
                     u64 size,
                     u64 guest_size,
                     bool image_staging,
                     const gcn::TImage& image,
                     u32 elem_bytes,
                     u32 stage_elem_bytes) {
  // Non-zero bytes currently in the guest range. A copy whose SOURCE is empty
  // and one whose destination never receives the write look the same from the
  // descriptor alone.
  u64 nonzero = 0;
  if (base && guest_size && guest_size <= (1u << 24) &&
      IsGuestRange(base, guest_size) &&
      utl::isMemoryRangeMapped(reinterpret_cast<const void*>(base),
                               guest_size)) {
    const u8* p = reinterpret_cast<const u8*>(base);
    for (u64 i = 0; i < guest_size; i++)
      nonzero += p[i] != 0;
  }
  BASE_LOGI("csres",
            "cs={:#x} bind={} kind={} s{} pc={:#x} base={:#x} size={:#x} "
            "guest={:#x} written={} img={} tile={} elem={}/{} nz={}",
            (unsigned long)cs_addr, res.binding, res.kind, res.base_sgpr,
            res.use_pc, (unsigned long)base, (unsigned long)size,
            (unsigned long)guest_size, res.written ? 1 : 0,
            image_staging ? 1 : 0, image.tiling_idx, elem_bytes,
            stage_elem_bytes, (unsigned long)nonzero);
}

void TraceCsInvalidRange(u64 cs_addr,
                         u32 binding,
                         u64 base,
                         u64 guest_size) {
  BASE_LOGI("csres", "cs={:#x} bind={} invalid range base={:#x} size={:#x}",
            (unsigned long)cs_addr, binding, (unsigned long)base,
            (unsigned long)guest_size);
}

void TraceCsDispatch(u64 cs_addr, bool executed, u32 num_resources) {
  BASE_LOGI("csres", "cs={:#x} dispatch {} ({} resources)",
            (unsigned long)cs_addr, executed ? "executed" : "failed",
            num_resources);
}

// --- the packet stream -----------------------------------------------------

void NotePacket(u32 op) {
  g_op_hist[op & 0xFF]++;
  g_dcb_packets++;
}

void NotePacketCost(u32 op, u64 ns) {
  g_op_ns[op & 0xFF] += ns;
}

bool WantPacketCost() {
  return kDcbStat;
}

void DumpOpcodeHistogram() {
  BASE_LOGI("gpu", "dcb opcode histogram (after {} dcbs):", g_dcb_seen);
  for (int i = 0; i < 256; i++)
    if (g_op_hist[i])
      BASE_LOGI("gpu", "  op {:#04x} x{}", i, g_op_hist[i]);
}

bool CeTraceOn() {
  return kCeTrace;
}

void TraceConstRam(const char* what,
                   u32 ce_offset,
                   u32 num_dwords,
                   u64 addr,
                   const char* verdict,
                   u32 first_dword) {
  if (!kCeTrace)
    return;
  static std::atomic<u64> n{0};
  static std::atomic<u64> dst_lo{~0ull}, dst_hi{0};
  if (addr) {
    u64 lo = dst_lo.load();
    while (addr < lo && !dst_lo.compare_exchange_weak(lo, addr)) {
    }
    u64 hi = dst_hi.load();
    const u64 end = addr + (u64)num_dwords * 4;
    while (end > hi && !dst_hi.compare_exchange_weak(hi, end)) {
    }
  }
  const u64 seq = n.fetch_add(1);
  if (seq < (u64)kCeTraceMax.get())
    BASE_LOGI("ce", "{:<8} ceoff={:#x} ndw={} addr={:#x} {} data0={:#x}", what,
              ce_offset, num_dwords, (unsigned long)addr, verdict, first_dword);
  // The tail reports the destination span every dump landed in, which is what
  // names the ring.
  if (seq && (seq % 20000) == 0)
    BASE_LOGI("ce", "{} packets; dump/load span {:#x}..{:#x}",
              (unsigned long long)seq, (unsigned long)dst_lo.load(),
              (unsigned long)dst_hi.load());
}

void NoteCcbPacket(u32 op) {
  g_ccb_hist[op & 0xFF]++;
}

void TraceCcbSubmit(u32 size_bytes, u32 words) {
  if (kCcbHist && g_ccb_count == 0)
    BASE_LOGI("ccb", "first ccb: {} bytes ({} words)", size_bytes, words);
  g_ccb_count++;
}

void TraceCcbHistogram(u32 words) {
  if (!kCcbHist || g_ccb_hist_dumps >= 3 || g_ccb_count < 50)
    return;
  g_ccb_hist_dumps++;
  BASE_LOGI("ccb", "opcode histogram (after {} ccbs, this one {} words):",
            (unsigned long)g_ccb_count, words);
  for (int op = 0; op < 256; op++)
    if (g_ccb_hist[op])
      BASE_LOGI("ccb", "  op {:#04x} x{}", op, g_ccb_hist[op]);
}

void TraceWaitRegMem(bool timed_out) {
  if (!kWaitTrace)
    return;
  static std::atomic<u64> waits{0}, expired{0};
  waits.fetch_add(1);
  if (timed_out)
    expired.fetch_add(1);
  if ((waits.load() % 20000) == 0)
    BASE_LOGI("wait", "WAIT_REG_MEM: {} waited, {} timed out",
              (unsigned long long)waits.load(),
              (unsigned long long)expired.load());
}

void TraceDmaData(u32 control,
                  u32 command,
                  u32 src_sel,
                  u32 dst_sel,
                  u64 src,
                  u64 dst,
                  u32 bytes,
                  bool copied) {
  if (!kDmaTrace)
    return;
  // Shader prefetches (src==dst) outnumber real transfers by thousands to one
  // and used to eat the whole trace cap, which made "this title never CP-DMAs
  // anything" unfalsifiable. Count every packet by class and spend the cap on
  // transfers only.
  static std::atomic<u64> n_all{0}, n_prefetch{0}, n_copy{0}, n_reject{0};
  static int shown = 0;
  n_all.fetch_add(1);
  if (src == dst)
    n_prefetch.fetch_add(1);
  else if (copied)
    n_copy.fetch_add(1);
  else
    n_reject.fetch_add(1);
  if (src != dst && shown < 200) {
    shown++;
    BASE_LOGI("dma",
              "ctrl={:#x} cmd={:#x} srcSel={} dstSel={} src/data={:#x} "
              "dst={:#x} bytes={} {}",
              control, command, src_sel, dst_sel, (unsigned long)src,
              (unsigned long)dst, bytes, copied ? "copied" : "ignored");
  }
  if ((n_all.load() % 20000) == 0)
    BASE_LOGI("dma",
              "totals: {} packets, {} prefetch (src==dst), {} copied, "
              "{} rejected",
              n_all.load(), n_prefetch.load(), n_copy.load(), n_reject.load());
}

void TraceAddrWatch(const char* packet,
                    u64 dst,
                    u64 bytes,
                    u32 first_dword,
                    u32 max_lines) {
  if (!kAddrWatch || dst > (u64)kAddrWatch ||
      (u64)kAddrWatch >= dst + (bytes ? bytes : 4))
    return;
  // One budget per packet class, keyed by the caller's literal. A shared cap
  // lets whichever packet type hits first spend it all, and "nothing writes
  // this address via X" is then unfalsifiable, which is the question the knob
  // exists to answer.
  struct Site {
    const char* packet = nullptr;
    u32 hits = 0;
  };
  static Site sites[8];
  Site* site = nullptr;
  for (Site& candidate : sites) {
    if (candidate.packet == packet || !candidate.packet) {
      candidate.packet = packet;
      site = &candidate;
      break;
    }
  }
  if (!site || site->hits++ >= max_lines)
    return;
  BASE_LOGI("addrwatch", "{} dst={:#x} bytes={} data0={:08x}", packet,
            (unsigned long)dst, (unsigned long)bytes, first_dword);
}

void TraceLabelWrite(const char* packet,
                     u64 addr,
                     u32 data_sel,
                     u64 value) {
  if (!kEopTrace)
    return;
  BASE_LOGI("eop", "{} addr={:#x} sel={} val={:#x}", packet,
            (unsigned long)addr, data_sel, (unsigned long)value);
}

void TraceEosLabel(u64 addr, u32 value) {
  if (!kEopTrace)
    return;
  BASE_LOGI("eop", "EOS addr={:#x} val={:#x}", (unsigned long)addr, value);
}

void TraceDataWrite(u64 addr, u32 dwords, u32 first_dword) {
  if (!kEopTrace)
    return;
  BASE_LOGI("eop", "WRITE_DATA dst={:#x} ndw={} v0={:#x}", (unsigned long)addr,
            dwords, first_dword);
}

void TraceUnhandledOpcode(u32 op, u32 count) {
  if (!kOpTrace)
    return;
  static std::atomic<u64> seen[256];
  const u64 n = seen[op & 0xFF].fetch_add(1);
  if (n == 0 || n == 4096)
    BASE_LOGI("pm4", "unhandled op={:#x} count={} seen={}", op, count,
              (unsigned long long)(n + 1));
}

void TraceIndirectBuffer(u32 position,
                         u32 depth,
                         u32 words,
                         bool followed) {
  if (!kIbTrace)
    return;
  if (followed)
    BASE_LOGI("ib", "dcb chain @{} depth={} words={}", position, depth, words);
  else
    BASE_LOGI("ib", "dcb chain skipped @{} depth={}", position, depth);
}

void TraceCounter(const char* packet, u64 value) {
  if (!kCounterTrace)
    return;
  BASE_LOGI("cnt", "{} -> {}", packet, (unsigned long long)value);
}

void TraceWaitOnCeCounter(u32 wanted, u64 ce_counter) {
  if (!kCounterTrace)
    return;
  BASE_LOGI("cnt", "WAIT_ON_CE_COUNTER want={} (ce={}, always ok)", wanted,
            (unsigned long long)ce_counter);
}

void TraceDesync(u32 position,
                 u32 words,
                 u32 type,
                 u32 hdr,
                 bool force) {
  if (!force && !kDesyncTrace)
    return;
  BASE_LOGI("gpu", "  @{:<5}/{} STOP type{} hdr={:#x}", position, words, type,
            hdr);
}

// --- submissions -----------------------------------------------------------

bool ShouldDumpDcb(u32 size_bytes) {
  static bool dumped = false;
  if (!kTrace || dumped || size_bytes <= 4000)
    return false;
  dumped = true;
  BASE_LOGI("gpu", "=== big dcb walk (size={}) ===", size_bytes);
  return true;
}

void TraceDcbPacket(u32 position, u32 op, u32 count) {
  BASE_LOGI("gpu", "  @{:<5} T3 op={:#04x} count={}", position, op, count);
}

void TraceSubmit(const void* dcb,
                 u32 size_bytes,
                 u32 words,
                 u64 submit_number,
                 u64 draws_so_far) {
  if (!kTrace)
    return;
  if (submit_number <= 8 || submit_number % 256 == 0)
    BASE_LOGI("gpu", "submit #{} size={} draws-so-far={}",
              (unsigned long)submit_number, size_bytes,
              (unsigned long)draws_so_far);
  if (g_dcb_seen < 6)
    BASE_LOGI("gpu", "SubmitDcb dcb={} size_bytes={} words={} hdr0={:#x}", dcb,
              size_bytes, words, static_cast<const u32*>(dcb)[0]);
}

void TraceDcbWalkResult(const u32* dcb,
                        u32 words,
                        u32 words_walked) {
  BASE_LOGI("gpu", "=== big dcb walk done: {}/{} words ===", words_walked,
            words);
  // Brute-scan the whole buffer for draw-opcode headers (in case the walker
  // desynced and missed a draw), and dump raw words around the stop point.
  int found = 0;
  for (u32 w = 0; w < words; w++) {
    const u32 hdr = dcb[w];
    if ((hdr >> 30) != 3)
      continue;
    const u32 op = (hdr >> 8) & 0xFF;
    if (op == 0x2D || op == 0x27 || op == 0x35 || op == 0x30 || op == 0x15) {
      BASE_LOGI("gpu", "  SCAN found draw op={:#x} @word {}", op, w);
      if (++found > 8)
        break;
    }
  }
  if (!found)
    BASE_LOGI("gpu", "  SCAN: no draw opcode anywhere in {} words", words);
  base::String raw;
  for (u32 w = 255; w < 271 && w < words; w++)
    base::FormatTo(raw, " {:08x}", dcb[w]);
  BASE_LOGI("gpu", "  raw[255..270]:{}", raw.c_str());
}

void TraceDcbStat(u32 words) {
  if (!kDcbStat)
    return;
  g_dcb_words += words;
  static auto last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration<double>(now - last).count() < 2.0)
    return;
  last = now;
  // Not the numerous packets that cost: on SotC's load phase DISPATCH_DIRECT is
  // half the walk's time and DRAW_INDEX_INDIRECT another quarter, while NOP is
  // 37% of the PACKETS and free.
  u32 top_op[5] = {}, top_n[5] = {}, slow_op[5] = {};
  u64 slow_ns[5] = {};
  for (u32 o = 0; o < 256; o++) {
    u32 n = g_op_hist[o], op = o;
    for (int k = 0; k < 5; k++)
      if (n > top_n[k]) {
        std::swap(n, top_n[k]);
        std::swap(op, top_op[k]);
      }
    u64 ns = g_op_ns[o];
    op = o;
    for (int k = 0; k < 5; k++)
      if (ns > slow_ns[k]) {
        std::swap(ns, slow_ns[k]);
        std::swap(op, slow_op[k]);
      }
  }
  base::String line;
  for (int k = 0; k < 5; k++)
    if (top_n[k])
      base::FormatTo(line, " {:#04x} x{}", top_op[k], top_n[k]);
  line.append(" | costliest:");
  for (int k = 0; k < 5; k++)
    if (slow_ns[k])
      base::FormatTo(line, " {:#04x} {:.0f}ms", slow_op[k], slow_ns[k] / 1e6);
  BASE_LOGI("dcbstat", "{} packets / {} dwords; most:{}", g_dcb_packets,
            g_dcb_words, line.c_str());
}

void MaybeDumpOpcodeHistogram(u32 words_walked, u32 words) {
  static Elapsed since_first_submit;
  const double elapsed = since_first_submit.Seconds();
  if (kTrace && g_dcb_seen < 4)
    BASE_LOGI("gpu", "dcb done: walked {}/{} words", words_walked, words);
  if (kTrace && ++g_dcb_seen <= 4)
    DumpOpcodeHistogram();
  // Time-gated (default 100s) so the cumulative histogram includes the in-level
  // command stream (level-load compute/copies), not just the title, and only
  // once the walker has seen packets: the first buffer after the gate is often
  // empty, which used to latch the dump on nothing.
  static bool dumped = false;
  if (kOpHist && !dumped && words_walked > 0 && elapsed >= kOhAfter) {
    dumped = true;
    DumpOpcodeHistogram();
  }
}

}  // namespace gpu::ps4
