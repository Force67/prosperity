/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gpu/gcn/gcn_resource.h"
#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGpuEudfail, "DELTA_GPU_EUDFAIL", false);
DELTA_OPTION(bool, kGpuEudtrace, "DELTA_GPU_EUDTRACE", false);
DELTA_OPTION(bool, kGpuTilehist, "DELTA_GPU_TILEHIST", false);
DELTA_OPTION(bool, kTrace, "DELTA_GPU_TRACE", false);
DELTA_OPTION(u64, kSotcCompositeRt, "DELTA_GPU_SOTC_COMPOSITE_RT", 0);
DELTA_OPTION(u64, kTexSrc, "DELTA_GPU_TEXSRC", 0);
DELTA_OPTION(u64, kTscan, "DELTA_GPU_TSCAN", 0);
DELTA_OPTION(const char*, kMemFind, "DELTA_GPU_MEMFIND", nullptr);
DELTA_OPTION(u32, kSoaScan, "DELTA_GPU_SOASCAN", 0);
DELTA_OPTION(int, kTscanAfter, "DELTA_GPU_TSCAN_AFTER", 0);
DELTA_OPTION(bool, kTwatch, "DELTA_GPU_TWATCH", false);
DELTA_OPTION(bool, kNullDis, "DELTA_GPU_NULLDIS", false);
DELTA_OPTION(bool, kNullWatch, "DELTA_GPU_NULLWATCH", false);
// DELTA_GPU_ARENA_PROBE=<n>: when a descriptor reads all-zero, look for the one
// the shader wanted in the neighbouring 2 MiB resource arenas and use it.
// SotC's descriptor tables sit at a constant -3 arenas from where its own SRT
// points (24/24 probes, whole run), so the title and we disagree about which
// arena is current. This is a MEASUREMENT AID, not a fix: it proves the bias is
// the whole story without yet explaining who introduced it.
DELTA_OPTION(int, kArenaProbe, "DELTA_GPU_ARENA_PROBE", 0);
}  // namespace

namespace gpu::gcn {

void (*g_flush_guest_range)(u64 address, u64 bytes) = nullptr;

namespace {


constexpr u64 kGuestLo = 0x1000000000ull;
constexpr u64 kGuestHi = 0x20000000000ull;

bool GuestRange(u64 address, u64 size) {
  return size && address >= kGuestLo && address < kGuestHi &&
         size <= kGuestHi - address &&
         utl::isMemoryRangeMapped(reinterpret_cast<const void*>(address), size);
}

// DELTA_GPU_TSCAN=<hex surface address>: sweep every mapped guest page once for
// a texture descriptor naming that surface, and print each hit with the dwords
// around it. When a binding resolves to an all-zero T#, this is what separates
// "the title never built the descriptor" from "it built it somewhere our
// pointer chain does not reach" -- the second case shows the descriptor sitting
// in a table we never look at, and the distance to the address the shader read
// names the mistake.
u64 ScanForDescriptor(u64 want_base) {
  const u32 want_word0 = static_cast<u32>(want_base >> 8);
  const u32 want_hi = static_cast<u32>((want_base >> 40) & 0x3F);
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) {
    BASE_LOGI("tscan", "cannot read /proc/self/maps");
    return 0;
  }
  BASE_LOGI("tscan", "sweeping for base={:#x} (word0={:08x} hi={})",
            static_cast<unsigned long>(want_base), want_word0, want_hi);
  char line[512];
  u64 scanned = 0, hits = 0, first_valid = 0;
  while (std::fgets(line, sizeof(line), maps)) {
    u64 lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3)
      continue;
    if (perms[0] != 'r' || lo < kGuestLo || hi > kGuestHi || hi <= lo)
      continue;
    scanned += hi - lo;
    const u32* p = reinterpret_cast<const u32*>(lo);
    const u64 n = (hi - lo) / 4;
    for (u64 i = 0; i + 8 <= n; i++) {
      if (p[i] != want_word0 || (p[i + 1] & 0x3F) != want_hi)
        continue;
      hits++;
      if (hits > 64)
        continue;
      const u64 at = lo + i * 4;
      const TImage t = DecodeTImage(&p[i]);
      if (t.valid && !first_valid)
        first_valid = at;
      BASE_LOGI("tscan",
                "hit at={:#x} {}x{} pitch={} dfmt={} nfmt={} "
                "valid={:d} raw={:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}",
                static_cast<unsigned long>(at), t.width, t.height, t.pitch,
                t.dfmt, t.nfmt, t.valid, p[i], p[i + 1], p[i + 2], p[i + 3],
                p[i + 4], p[i + 5], p[i + 6], p[i + 7]);
    }
  }
  std::fclose(maps);
  BASE_LOGI("tscan", "done: {} hits over {} MiB",
            static_cast<unsigned long>(hits),
            static_cast<unsigned long>(scanned >> 20));
  return first_valid;
}

// DELTA_GPU_MEMFIND=<hex dword>[,<hex dword>...]: sweep every mapped guest
// page once for that dword sequence and print where it sits. The inverse of
// ScanForDescriptor -- that one starts from an address and looks for the
// descriptor naming it, this one starts from DATA the guest must have written
// and looks for where it put it, which is the only way left when a descriptor
// resolves into an arena that reads zero end to end. Reading the process from
// outside is not an option (ptrace_scope denies /proc/<pid>/mem), so it runs
// in-process like the sweep above.
void ScanForDwords(const char* spec) {
  std::vector<u32> want;
  for (const char* c = spec; *c;) {
    while (*c == ',' || *c == ' ')
      c++;
    if (!*c)
      break;
    want.push_back(static_cast<u32>(std::strtoull(c, nullptr, 16)));
    while (*c && *c != ',')
      c++;
  }
  if (want.empty())
    return;
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) {
    BASE_LOGI("memfind", "cannot read /proc/self/maps");
    return;
  }
  base::String pattern;
  for (u32 v : want)
    base::FormatTo(pattern, " {:08x}", v);
  BASE_LOGI("memfind", "sweeping for{}", pattern.c_str());
  char line[512];
  u64 scanned = 0, hits = 0;
  while (std::fgets(line, sizeof(line), maps)) {
    u64 lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3)
      continue;
    if (perms[0] != 'r' || lo < kGuestLo || hi > kGuestHi || hi <= lo)
      continue;
    scanned += hi - lo;
    const u32* p = reinterpret_cast<const u32*>(lo);
    const u64 n = (hi - lo) / 4;
    for (u64 i = 0; i + want.size() <= n; i++) {
      if (std::memcmp(&p[i], want.data(), want.size() * 4) != 0)
        continue;
      if (++hits > 64)
        continue;
      BASE_LOGI("memfind", "hit at={:#x}",
                static_cast<unsigned long>(lo + i * 4));
    }
  }
  std::fclose(maps);
  BASE_LOGI("memfind", "done: {} hits over {} MiB",
            static_cast<unsigned long>(hits),
            static_cast<unsigned long>(scanned >> 20));
}

// DELTA_GPU_SOASCAN=<byte stride>: sweep guest memory for a screen-quad
// vertex block stored as PARALLEL COLUMNS at that stride -- four clip-space
// corners whose w column is all 1.0, whose z column is all 0.0 and whose x and
// y columns are all +/-1.0. A shader that fetches its own vertices reads
// exactly that shape (GTA:SA's LUT pass reads x/y/z/w at +0x180/0x240/0x300/
// 0x3c0 of a 0xc0-strided block), and when its V# resolves into an arena that
// reads zero end to end, finding where the block REALLY is is the only thing
// that names the addressing mistake. A plain dword search cannot: the columns
// are not contiguous and "-1,-1,1,1" alone hits every static corner table in
// the modules.
void ScanForQuadColumns(u32 stride) {
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) {
    BASE_LOGI("soascan", "cannot read /proc/self/maps");
    return;
  }
  BASE_LOGI("soascan", "sweeping for a column-major screen quad, stride {:#x}",
            stride);
  const auto all = [](const float* f, bool (*pred)(float)) {
    return pred(f[0]) && pred(f[1]) && pred(f[2]) && pred(f[3]);
  };
  const auto pm1 = [](float v) { return v == 1.f || v == -1.f; };
  const auto zero = [](float v) { return v == 0.f; };
  const auto one = [](float v) { return v == 1.f; };
  char line[512];
  u64 scanned = 0, hits = 0;
  while (std::fgets(line, sizeof(line), maps)) {
    u64 lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3)
      continue;
    if (perms[0] != 'r' || lo < kGuestLo || hi > kGuestHi || hi <= lo)
      continue;
    scanned += hi - lo;
    const u64 span = static_cast<u64>(stride) * 4 + 16;
    for (u64 at = lo; at + span <= hi; at += 4) {
      const auto* w = reinterpret_cast<const float*>(at + stride * 3);
      if (!all(w, one))
        continue;
      const auto* z = reinterpret_cast<const float*>(at + stride * 2);
      const auto* x = reinterpret_cast<const float*>(at);
      const auto* y = reinterpret_cast<const float*>(at + stride);
      // w and z pin the shape. x and y are required to be CLIP COORDINATES --
      // inside [-1,1] and carrying more than one value each -- rather than
      // exactly +/-1: a pass that rasterizes a sub-box of the volume writes
      // corners that are not the unit quad's, and demanding +/-1 is an
      // assumption about the caller, not about the shape. Relaxing them all
      // the way to "finite" is useless in the other direction: 24 GiB of
      // module data then hits thousands of times.
      if (!all(z, zero))
        continue;
      // A quad's x column holds exactly two values, twice each -- its left and
      // right edge -- and so does its y column. That is true of the unit quad
      // and of any sub-rect, and it is what separates a real corner table from
      // the denormal noise that "inside [-1,1]" alone lets through.
      const auto edges = [](const float* f) {
        for (u32 k = 0; k < 4; k++) {
          if (!(f[k] >= -1.f && f[k] <= 1.f))
            return false;
          if (f[k] != 0.f && (f[k] > -1e-3f && f[k] < 1e-3f))
            return false;  // denormal or near-zero junk
        }
        u32 same_as_first = 0;
        for (u32 k = 0; k < 4; k++)
          same_as_first += f[k] == f[0];
        if (same_as_first != 2)
          return false;
        float other = f[0];
        for (u32 k = 1; k < 4; k++)
          if (f[k] != f[0]) {
            other = f[k];
            break;
          }
        u32 same_as_other = 0;
        for (u32 k = 0; k < 4; k++)
          same_as_other += f[k] == other;
        return same_as_other == 2;
      };
      if (!edges(x) || !edges(y))
        continue;
      (void)pm1;
      if (++hits > 32)
        continue;
      // The neighbouring columns too: the four that match are the position,
      // and whether this is the block a given shader reads is decided by what
      // sits either side of them (its UVs, and any per-vertex slice index).
      BASE_LOGI("soascan", "hit position columns at={:#x}",
                static_cast<unsigned long>(at));
      for (int col = -2; col <= 3; col++) {
        const u64 ca = at + static_cast<i64>(col) * stride;
        if (ca < lo || ca + 16 > hi)
          continue;
        const auto* f = reinterpret_cast<const float*>(ca);
        BASE_LOGI("soascan", "  col{:+d} ({:#x}) {:g} {:g} {:g} {:g}", col,
                  static_cast<unsigned long>(ca), f[0], f[1], f[2], f[3]);
      }
    }
  }
  std::fclose(maps);
  BASE_LOGI("soascan", "done: {} hits over {} MiB",
            static_cast<unsigned long>(hits),
            static_cast<unsigned long>(scanned >> 20));
}

// How much of the 4 MiB pool block around `address` was ever written. A block
// the title filled reads mostly non-zero; a block it only reserved reads zero
// end to end, which is what tells a stale pointer apart from a torn write.
void CensusBlock(const char* what, u64 address) {
  constexpr u64 kBlock = 0x400000;
  const u64 base = address & ~(kBlock - 1);
  if (!GuestRange(base, kBlock)) {
    BASE_LOGI("census", "{} block {:#x} not mapped", what,
              static_cast<unsigned long>(base));
    return;
  }
  const u32* p = reinterpret_cast<const u32*>(base);
  u64 nz = 0, first_nz = 0, last_nz = 0;
  for (u64 i = 0; i < kBlock / 4; i++) {
    if (!p[i])
      continue;
    nz++;
    if (!first_nz)
      first_nz = base + i * 4;
    last_nz = base + i * 4;
  }
  BASE_LOGI("census",
            "{} {:#x}: block {:#x} has {}/{} non-zero dwords, "
            "written span {:#x}..{:#x}",
            what, static_cast<unsigned long>(address),
            static_cast<unsigned long>(base), (unsigned long)nz,
            (unsigned long)(kBlock / 4), (unsigned long)first_nz,
            (unsigned long)last_nz);
}

// DELTA_GPU_TWATCH=1: remember every address a null T# was read from and
// re-read it later. A descriptor that is zero when the draw is processed but
// non-zero a moment later means the title fills the table AFTER submitting the
// draw that names it -- an ordering bug on our side, since our submit is
// synchronous -- while one that stays zero for the rest of the run means the
// pointer never named live data at all. Those two need opposite fixes, and
// nothing else distinguishes them.
struct NullSite {
  u64 address;
  u64 code_base;
  u64 draw_seen;
  bool filled;
  // Highest offset in the containing 4 MiB arena that has ever been non-zero
  // while we watched. If this never reaches the offset the shader read, the
  // pointer names a fill level the arena no longer has (a stale pointer); if it
  // passes it, we walked the command buffer at the wrong moment.
  u64 peak_watermark;
};
std::vector<NullSite> g_null_sites;
u64 g_track_draws = 0;

void NoteNullDescriptor(u64 address, u64 code_base) {
  if (!address)
    return;
  for (const NullSite& s : g_null_sites)
    if (s.address == address)
      return;
  if (g_null_sites.size() < 256)
    g_null_sites.push_back({address, code_base, g_track_draws, false, 0});
}

// Highest non-zero dword offset inside the 4 MiB arena holding `address`.
u64 BlockWatermark(u64 address) {
  constexpr u64 kBlock = 0x400000;
  const u64 base = address & ~(kBlock - 1);
  if (!GuestRange(base, kBlock))
    return 0;
  const u32* p = reinterpret_cast<const u32*>(base);
  for (u64 i = kBlock / 4; i-- > 0;)
    if (p[i])
      return i * 4;
  return 0;
}

void PollNullDescriptors() {
  u32 filled = 0, still_zero = 0;
  // The watermark sweep is 4 MiB per site, so only the first few are tracked.
  u32 watched = 0;
  for (NullSite& s : g_null_sites) {
    if (watched >= 6 || s.filled)
      continue;
    watched++;
    const u64 mark = BlockWatermark(s.address);
    if (mark > s.peak_watermark)
      s.peak_watermark = mark;
    const u64 want = s.address & 0x3FFFFF;
    BASE_LOGI("wmark", "{:#x} offset={:#x} arena peak={:#x} now={:#x} -> {}",
              static_cast<unsigned long>(s.address), (unsigned long)want,
              (unsigned long)s.peak_watermark, (unsigned long)mark,
              s.peak_watermark >= want ? "REACHED (timing)"
                                       : "never reached (stale pointer)");
  }
  for (NullSite& s : g_null_sites) {
    if (s.filled) {
      filled++;
      continue;
    }
    if (!GuestRange(s.address, 32))
      continue;
    const u32* p = reinterpret_cast<const u32*>(s.address);
    if (std::all_of(p, p + 8, [](u32 w) { return w == 0; })) {
      still_zero++;
      continue;
    }
    s.filled = true;
    filled++;
    BASE_LOGI("twatch",
              "{:#x} (read null by PS {:#x} at draw {}) is NOW "
              "{:08x}/{:08x}/{:08x}/{:08x} after {} more draws",
              static_cast<unsigned long>(s.address),
              static_cast<unsigned long>(s.code_base),
              (unsigned long)s.draw_seen, p[0], p[1], p[2], p[3],
              (unsigned long)(g_track_draws - s.draw_seen));
  }
  BASE_LOGI("twatch", "{} of {} null sites later filled, {} still zero",
            filled, static_cast<unsigned>(g_null_sites.size()), still_zero);
}

// SMRD operand fields (GFX7).
struct Smrd {
  u32 op;
  u32 sdst;
  u32 sbase;  // SGPR pair index; actual base SGPR = sbase * 2
  u32 offset;
  bool imm;
};

Smrd DecodeSmrd(u32 w) {
  return {
      .op = (w >> 22) & 0x1F,
      .sdst = (w >> 15) & 0x7F,
      .sbase = (w >> 9) & 0x3F,
      .offset = w & 0xFF,
      .imm = ((w >> 8) & 1) != 0,
  };
}

u64 UserDataPointer(const u32* user_data, u32 sgpr) {
  return (static_cast<u64>(user_data[sgpr + 1] & 0xFFFF) << 32) |
         user_data[sgpr];
}

u32 NextPow2(u32 v) {
  v -= 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

// Symbolic identity of the descriptor pair an MIMG instruction references:
// the T#/S# SGPR indices plus WHICH s_load last wrote each of them (0xFFFF =
// inline user data), plus the access-type bits that select a distinct Vulkan
// binding (arrayed / depth-compare / gather-lz). Packs into one dword-pair key.
u64 MimgDescriptorKey(u32 srsrc,
                           u32 ssamp,
                           u32 load_rsrc,
                           u32 load_samp,
                           u32 flags) {
  return (static_cast<u64>(srsrc) << 0) |
         (static_cast<u64>(ssamp) << 8) |
         (static_cast<u64>(load_rsrc & 0xFFFF) << 16) |
         (static_cast<u64>(load_samp & 0xFFFF) << 32) |
         (static_cast<u64>(flags) << 48);
}

// Concrete evaluation of a graphics stage's scalar register file. Seed s0..s15
// from the live user data, then execute the scalar loads / moves in program
// order, actually reading guest memory. Because a load's base pointer may be a
// value a PRIOR load wrote (not just direct user data), stepping the program
// resolves arbitrarily nested extended-user-data / SRT descriptor chains:
// user-data SGPR -> s_load a pointer -> s_load the descriptor through that
// pointer -> ... Callers step every instruction and read the descriptor SGPRs
// (T#/S#/V#) at the instruction that consumes them (MIMG / s_buffer_load), so
// SGPR reuse resolves each consumer against the state live at its own point.
//
// Register file: 128 SGPRs; a T# SBASE (5 bits * 4) reaches s124, +8 = s132, so
// size for the descriptor tail. `known` marks which dwords hold a resolved
// value (seeded user data, or a value read from guest memory).
// SOP1/SOP2 opcodes whose destination is an SGPR pair, so an unmodelled one
// invalidates both halves (Sea Islands numbering, as in gcn_disasm's tables).
bool Sop1DestIs64(u32 op) {
  switch (op) {
    case 0x04:  // s_mov_b64
    case 0x06:  // s_cmov_b64
    case 0x08:  // s_not_b64
    case 0x0a:  // s_wqm_b64
    case 0x0c:  // s_brev_b64
    case 0x1c:  // s_bitset0_b64
    case 0x1e:  // s_bitset1_b64
    case 0x1f:  // s_getpc_b64
    case 0x21:  // s_swappc_b64
    case 0x24:  // s_*_saveexec_b64
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2d:  // s_quadmask_b64
    case 0x2f:  // s_movrels_b64
    case 0x31:  // s_movreld_b64
      return true;
    default:
      return false;
  }
}

bool Sop2DestIs64(u32 op) {
  switch (op) {
    case 0x0b:  // s_cselect_b64
    case 0x0f:  // s_and_b64
    case 0x11:  // s_or_b64
    case 0x13:  // s_xor_b64
    case 0x15:  // s_andn2_b64
    case 0x17:  // s_orn2_b64
    case 0x19:  // s_nand_b64
    case 0x1b:  // s_nor_b64
    case 0x1d:  // s_xnor_b64
    case 0x1f:  // s_lshl_b64
    case 0x21:  // s_lshr_b64
    case 0x23:  // s_ashr_i64
    case 0x25:  // s_bfm_b64
    case 0x29:  // s_bfe_u64
    case 0x2a:  // s_bfe_i64
      return true;
    default:
      return false;
  }
}

// VOP3b: the forms that carry a second, SCALAR destination (a carry-out or a
// division-scale flag) in bits 14:8 of the first dword, where a plain VOP3a
// keeps its abs/clamp bits.
bool Vop3bWritesSdst(u32 op) {
  switch (op) {
    case 0x125:  // v_add_i32
    case 0x126:  // v_sub_i32
    case 0x127:  // v_subrev_i32
    case 0x128:  // v_addc_u32
    case 0x129:  // v_subb_u32
    case 0x12a:  // v_subbrev_u32
    case 0x16d:  // v_div_scale_f32
    case 0x16e:  // v_div_scale_f64
    case 0x176:  // v_mad_u64_u32
    case 0x177:  // v_mad_i64_i32
      return true;
    default:
      return false;
  }
}

struct ScalarEval {
  static constexpr u32 kRegs = 136;
  u32 sgpr[kRegs] = {};
  bool known[kRegs] = {};
  u64 src[kRegs] = {};  // guest address each dword was s_loaded from
  bool trace = false;
  u64 code_base = 0;  // guest address of the program, for s_getpc_b64

  // A shader that runs out of SGPRs parks scalars in the LANES of a VGPR with
  // v_writelane_b32 and reads them back with v_readlane_b32. Both of that
  // pair's scalar operands are wave-uniform by encoding -- the value must come
  // from an SGPR or an inline constant, never a VGPR, and so must the lane --
  // so the value a lane holds is exactly the scalar that was written, and the
  // walk can replay it. SotC restores descriptor-table POINTERS this way
  // (`v_readlane_b32 s82, v47, 11` then `s_load_dwordx4 s[8:11], s[82:83], 8`),
  // so a walk that skips the pair reads whatever those SGPRs held earlier and
  // decodes a descriptor from the wrong address. Keyed vgpr*64 + lane; a slot
  // that is absent is unknown and invalidates its destination.
  std::unordered_map<u32, u32> lane_spill;
  std::unordered_map<u32, u64> lane_spill_src;

  explicit ScalarEval(const u32* user_data, u64 base = 0) {
    for (u32 i = 0; i < 16; i++) {
      sgpr[i] = user_data[i];
      known[i] = true;
    }
    code_base = base;
    trace = kGpuEudtrace;
  }

  u64 Ptr(u32 s) const {  // 48-bit descriptor-table pointer pair
    return (static_cast<u64>(sgpr[s + 1] & 0xFFFF) << 32) | sgpr[s];
  }
  bool AllKnown(u32 s, u32 n) const {
    if (s + n > kRegs)
      return false;
    for (u32 i = 0; i < n; i++)
      if (!known[s + i])
        return false;
    return true;
  }
  void Set(u32 s, u32 v) {
    if (s < kRegs) {
      sgpr[s] = v;
      known[s] = true;
    }
  }
  void Clear(u32 s) {
    if (s < kRegs)
      known[s] = false;
  }
  bool Source(u32 field, u32 literal, u32& value) const {
    if (field <= 127) {
      if (!known[field])
        return false;
      value = sgpr[field];
      return true;
    }
    if (field == 128)
      value = 0;
    else if (field >= 129 && field <= 192)
      value = field - 128;
    else if (field >= 193 && field <= 208)
      value = static_cast<u32>(-static_cast<i32>(field - 192));
    else if (field == 240)
      value = 0x3f000000u;
    else if (field == 241)
      value = 0xbf000000u;
    else if (field == 242)
      value = 0x3f800000u;
    else if (field == 243)
      value = 0xbf800000u;
    else if (field == 244)
      value = 0x40000000u;
    else if (field == 245)
      value = 0xc0000000u;
    else if (field == 246)
      value = 0x40800000u;
    else if (field == 247)
      value = 0xc0800000u;
    else if (field == 255)
      value = literal;
    else
      return false;
    return true;
  }
  bool SourceHi(u32 field, u32& value) const {
    if (field <= 126)
      return Source(field + 1, 0, value);
    value = 0;
    return true;
  }

  // The vector encodings that move data between the scalar file and a VGPR's
  // lanes, plus the ones that write an SGPR the walk cannot model. Returns
  // true when the instruction was consumed here.
  bool StepLaneOp(const Inst& inst) {
    const bool vop1 = inst.enc == Enc::kVop1, vop2 = inst.enc == Enc::kVop2;
    const bool vop3 = inst.enc == Enc::kVop3;
    if (!vop1 && !vop2 && !vop3)
      return false;
    const u32 w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
    // VOP3 re-encodes the VOP1 (0x180+) and VOP2 (0x100+) opcodes and moves
    // the operands into the second dword.
    const bool readlane = (vop2 && op == 0x01) || (vop3 && op == 0x101);
    const bool writelane = (vop2 && op == 0x02) || (vop3 && op == 0x102);
    const bool readfirstlane = (vop1 && op == 0x02) || (vop3 && op == 0x182);
    const u32 dst = vop3 ? (w & 0xFF) : ((w >> 17) & 0xFF);
    const u32 src0 = vop3 ? (w1 & 0x1FF) : (w & 0x1FF);
    const u32 src1 = vop3 ? ((w1 >> 9) & 0x1FF) : ((w >> 9) & 0xFF);
    const auto forget = [&](u32 slot) {
      lane_spill.erase(slot);
      lane_spill_src.erase(slot);
    };
    if (writelane) {
      u32 value = 0, lane = 0;
      const bool lane_known = Source(src1, inst.literal, lane);
      const bool value_known = Source(src0, inst.literal, value);
      if (!lane_known) {
        for (u32 i = 0; i < 64; i++)  // could have landed anywhere
          forget(dst * 64 + i);
      } else if (!value_known) {
        forget(dst * 64 + (lane & 63));
      } else {
        lane_spill[dst * 64 + (lane & 63)] = value;
        lane_spill_src[dst * 64 + (lane & 63)] = src0 <= 127 ? src[src0] : 0;
      }
      return true;
    }
    if (readlane || readfirstlane) {
      // readfirstlane names the lowest EXEC-active lane. The walk does not
      // model EXEC, so it can only answer when the shader spilled to lane 0 --
      // which is what a spill/reload pair does when it uses one slot.
      u32 lane = 0;
      const bool lane_known = readfirstlane || Source(src1, inst.literal, lane);
      const u32 slot = (src0 - 256) * 64 + (lane & 63);
      const auto it = lane_known && src0 >= 256 && src0 < 512
                          ? lane_spill.find(slot)
                          : lane_spill.end();
      if (it == lane_spill.end()) {
        Clear(dst);
      } else {
        Set(dst, it->second);
        const auto at = lane_spill_src.find(slot);
        src[dst] = at == lane_spill_src.end() ? 0 : at->second;
      }
      return true;
    }
    // A VOP3-form compare writes its predicate to an SGPR PAIR, and a VOP3b
    // writes a carry-out there. Same rule as SOP1: an unmodelled SGPR write
    // must invalidate its destination rather than leave a stale pointer for a
    // later descriptor decode to read.
    if (vop3 && (op < 0x100 || Vop3bWritesSdst(op))) {
      const u32 sdst = op < 0x100 ? dst : ((w >> 8) & 0x7F);
      Clear(sdst);
      Clear(sdst + 1);
      return true;
    }
    return false;
  }

  // Advance the register file across one instruction. Only scalar moves and
  // pointer-relative scalar loads (the descriptor-chain ops) mutate it; every
  // other encoding leaves it unchanged. s_buffer_load (op >= 0x08) reads a
  // cbuffer through a V# and is a consumer, not a pointer op, so it is ignored
  // here.
  void Step(const Inst& inst) {
    if (inst.enc == Enc::kSop1) {
      const u32 w = inst.raw[0];
      const u32 sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (inst.opcode ==
          0x03) {  // s_mov_b32: stage a pointer via a scalar move
        u32 value;
        if (Source(ssrc0, inst.literal, value))
          Set(sdst, value);
        else
          Clear(sdst);
      } else if (inst.opcode == 0x04) {  // s_mov_b64
        u32 lo, hi;
        const bool source_known =
            Source(ssrc0, inst.literal, lo) && SourceHi(ssrc0, hi);
        if (source_known) {
          Set(sdst, lo);
          Set(sdst + 1, hi);
        } else {
          Clear(sdst);
          Clear(sdst + 1);
        }
      } else if (inst.opcode == 0x1f) {  // s_getpc_b64: address of the NEXT inst
        if (code_base) {
          const u64 pc =
              code_base + static_cast<u64>(inst.pc + inst.size) * 4;
          Set(sdst, static_cast<u32>(pc));
          Set(sdst + 1, static_cast<u32>(pc >> 32));
        } else {
          Clear(sdst);
          Clear(sdst + 1);
        }
      } else {
        // Everything else still WRITES sdst on hardware. Leaving our shadow
        // untouched kept a stale value there, and a descriptor decoded from it
        // is garbage that reads as a valid-looking T# -- worse than an
        // unresolved one, which at least falls back cleanly.
        Clear(sdst);
        if (Sop1DestIs64(inst.opcode))
          Clear(sdst + 1);
      }
      return;
    }
    if (inst.enc == Enc::kSopk) {
      const u32 w = inst.raw[0];
      const u32 sdst = (w >> 16) & 0x7F;
      const u32 simm =
          static_cast<u32>(static_cast<i32>(static_cast<i16>(w)));
      switch (inst.opcode) {
        case 0x00:  // s_movk_i32
          Set(sdst, simm);
          break;
        case 0x0f:  // s_addk_i32
          if (known[sdst])
            Set(sdst, sgpr[sdst] + simm);
          break;
        case 0x10:  // s_mulk_i32
          if (known[sdst])
            Set(sdst, sgpr[sdst] * simm);
          break;
        case 0x03:  // s_cmpk_*: SCC only, no SGPR destination
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07:
        case 0x08:
        case 0x09:
        case 0x0a:
        case 0x0b:
        case 0x0c:
        case 0x0d:
        case 0x0e:
        case 0x11:  // s_cbranch_i_fork
        case 0x13:  // s_setreg_b32
        case 0x15:  // s_setreg_imm32_b32
          break;
        default:
          Clear(sdst);
          break;
      }
      return;
    }
    if (StepLaneOp(inst))
      return;
    if (inst.enc == Enc::kSop2) {
      const u32 w = inst.raw[0], op = inst.opcode;
      const u32 sdst = (w >> 16) & 0x7F;
      const bool dest64 = Sop2DestIs64(op);
      u32 a, b;
      const bool inputs_known = Source(w & 0xFF, inst.literal, a) &&
                                Source((w >> 8) & 0xFF, inst.literal, b);
      if (!inputs_known) {
        Clear(sdst);
        if (dest64)
          Clear(sdst + 1);
        return;
      }
      u32 value;
      switch (op) {
        case 0x06:
          value = static_cast<u32>(
              std::min(static_cast<i32>(a), static_cast<i32>(b)));
          break;
        case 0x07:
          value = std::min(a, b);
          break;
        case 0x08:
          value = static_cast<u32>(
              std::max(static_cast<i32>(a), static_cast<i32>(b)));
          break;
        case 0x09:
          value = std::max(a, b);
          break;
        case 0x27: {  // s_bfe_u32: src1[4:0] offset, src1[22:16] width
          const u32 width = (b >> 16) & 0x7F;
          value = width >= 32 ? (a >> (b & 31))
                              : ((a >> (b & 31)) & ((1u << width) - 1));
          break;
        }
        case 0x00:
        case 0x02:
          value = a + b;
          break;
        case 0x01:
        case 0x03:
          value = a - b;
          break;
        case 0x0e:
          value = a & b;
          break;
        case 0x10:
          value = a | b;
          break;
        case 0x12:
          value = a ^ b;
          break;
        case 0x14:
          value = a & ~b;
          break;
        case 0x16:
          value = a | ~b;
          break;
        case 0x18:
          value = ~(a & b);
          break;
        case 0x1a:
          value = ~(a | b);
          break;
        case 0x1c:
          value = ~(a ^ b);
          break;
        case 0x1e:
          value = a << (b & 31);
          break;
        case 0x20:
          value = a >> (b & 31);
          break;
        case 0x22:
          value = static_cast<u32>(static_cast<i32>(a) >> (b & 31));
          break;
        case 0x26:
          value = a * b;
          break;
        default:
          Clear(sdst);
          if (dest64)
            Clear(sdst + 1);
          return;
      }
      Set(sdst, value);
      return;
    }
    if (inst.enc != Enc::kSmrd)
      return;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    // s_load reads a descriptor through a raw 2-dword pointer; s_buffer_load
    // (op 0x08..0x0c) reads it through a 4-dword V# resource table -- FOX and
    // other engines stash T#/pointer descriptors in a cbuffer/SRT accessed this
    // way, so following it here is what lets those bindings resolve.
    const bool buffer_load = s.op >= 0x08 && s.op <= 0x0c;
    if (s.op > 0x04 && !buffer_load)
      return;  // s_load / s_buffer_load only
    const u32 dwords = buffer_load ? (1u << (s.op - 0x08)) : (1u << s.op);
    const u32 base = s.sbase * 2;
    const u32 desc_dwords = buffer_load ? 4 : 2;
    const bool base_known = AllKnown(base, desc_dwords);
    const u64 table =
        base_known ? (buffer_load ? DecodeVBuffer(&sgpr[base]).base : Ptr(base))
                   : 0;
    bool offset_known = true;
    u64 byte_off = 0;
    const SmrdOffset so = DecodeSmrdOffset(inst);
    if (!so.in_sgpr) {
      byte_off = static_cast<u64>(so.dwords) * 4;
    } else if (so.sgpr < kRegs && known[so.sgpr]) {
      byte_off = sgpr[so.sgpr];
    } else {
      offset_known = false;
    }
    // A load rewrites its destination SGPRs even if it cannot be resolved;
    // snapshot its inputs before invalidating an overlapping destination.
    for (u32 i = 0; i < dwords; i++)
      Clear(s.sdst + i);
    if (!base_known || !offset_known) {
      if (kGpuEudfail)
        BASE_LOGI("eudfail", "s_load x{} s{} <- s{}: base_known={} off_known={}",
                  dwords, s.sdst, base, base_known, offset_known);
      return;
    }
    if (byte_off > UINT64_MAX - table)
      return;
    const u64 address = table + byte_off;
    if (!GuestRange(address, static_cast<u64>(dwords) * 4)) {
      if (kGpuEudfail)
        BASE_LOGI("eudfail",
                  "s_load UNMAPPED x{} s{} <- [s{}={:#x} + {:#x}] = {:#x}",
                  dwords, s.sdst, base, static_cast<unsigned long>(table),
                  static_cast<unsigned long>(byte_off),
                  static_cast<unsigned long>(address));
      return;
    }
    // The table this chain reads may have been filled by a compute dispatch
    // this frame -- SotC's material arenas hold the very T#s its draws reach
    // through their SRTs -- and those results sit in the CS buffer until they
    // are written back. Reading around the writeback resolves the descriptor
    // to zeros while the slot visibly holds a plausible T# a moment later
    // (TEXMISS's "src holds a valid descriptor" signature). One branch when
    // nothing is dirty anywhere; one page probe when something is.
    if (g_flush_guest_range)
      g_flush_guest_range(address, static_cast<u64>(dwords) * 4);
    const u32* mem = reinterpret_cast<const u32*>(address);
    for (u32 i = 0; i < dwords; i++) {
      Set(s.sdst + i, mem[i]);
      if (s.sdst + i < kRegs)
        src[s.sdst + i] = address + i * 4;
    }
    if (trace)
      BASE_LOGI("eudenc",
                "pc={:#x} raw={:08x} op={:#x} sdst=s{} sbase={} imm={} "
                "offset={:#x} lit={}",
                inst.pc, inst.raw[0], s.op, s.sdst, s.sbase, s.imm ? 1 : 0,
                s.offset, inst.has_literal ? 1 : 0);
    if (trace) {
      base::String bytes;
      base::FormatTo(bytes, "s_load x{} s{} <- [s{}={:#x} + {:#x}] = {:#x}",
                     dwords, s.sdst, base, static_cast<unsigned long>(table),
                     static_cast<unsigned long>(byte_off),
                     static_cast<unsigned long>(address));
      base::FormatTo(bytes, "   data:");
      for (u32 i = 0; i < dwords; i++)
        base::FormatTo(bytes, " {:08x}", mem[i]);
      BASE_LOGI("eud", "{}", bytes.c_str());
    }
  }
};

// Per-program analysis reused across draws: the MIMG binding plan plus the
// subset of instructions the scalar walk actually consumes (descriptor-chain
// scalar ops, SMRD loads, MIMG/MUBUF uses). The resolvers run once per
// draw on shaders that are mostly VALU code, so stepping only this subset --
// and planning bindings once instead of per draw -- removes the bulk of the
// per-draw analysis cost. Keyed by the Program object; the cached shared_ptr
// pins the object so the pointer cannot be reused while the entry lives. A
// shader rewrite yields a new Program from CachedProgram -> a new entry.
struct ScalarPassInfo {
  MimgBindingPlan plan;
  std::vector<Inst> insts;  // program-order subset relevant to ScalarEval users
};

// The vector instructions the walk has to see: the lane-spill pair that moves
// pointers between the scalar file and a VGPR's lanes, and the forms whose
// second destination is an SGPR the walk cannot model and must invalidate.
// Everything else in the vector encodings leaves the scalar file alone, and
// keeping it out of the subset is what makes the per-draw walk cheap.
bool VectorTouchesScalarFile(const Inst& inst) {
  switch (inst.enc) {
    case Enc::kVop1:
      return inst.opcode == 0x02;  // v_readfirstlane_b32
    case Enc::kVop2:
      return inst.opcode == 0x01 || inst.opcode == 0x02;  // read/writelane
    case Enc::kVop3:
      return inst.opcode < 0x100 ||  // a compare's SGPR-pair predicate
             inst.opcode == 0x101 || inst.opcode == 0x102 ||
             inst.opcode == 0x182 || Vop3bWritesSdst(inst.opcode);
    default:
      return false;
  }
}

const ScalarPassInfo& CachedScalarInfo(
    const std::shared_ptr<const Program>& program) {
  struct Entry {
    std::shared_ptr<const Program> pin;
    ScalarPassInfo info;
  };
  static std::unordered_map<const Program*, Entry> cache;
  auto it = cache.find(program.get());
  if (it != cache.end())
    return it->second.info;
  if (cache.size() > 512)
    cache.clear();  // unbounded-growth backstop
  Entry e;
  e.pin = program;
  const std::vector<u8> reachable = ComputeReachability(*program);
  e.info.plan = PlanMimgBindings(*program, reachable.data());
  u32 index = 0;
  for (const Inst& inst : *program) {
    if (!reachable[index++])
      continue;
    // Every SOP1/SOPK writes an SGPR, so all of them must reach the walk: the
    // ones it models advance the state, the rest invalidate their destination
    // instead of leaving a stale value for a later descriptor decode to read.
    if (inst.enc == Enc::kSop1 || inst.enc == Enc::kSopk ||
        inst.enc == Enc::kSop2 || inst.enc == Enc::kSmrd ||
        inst.enc == Enc::kMimg || inst.enc == Enc::kMubuf ||
        inst.enc == Enc::kMtbuf || VectorTouchesScalarFile(inst))
      e.info.insts.push_back(inst);
  }
  return cache.emplace(program.get(), std::move(e)).first->second.info;
}

}  // namespace

MimgBindingPlan PlanMimgBindings(const Program& program,
                                 const u8* reachable) {
  MimgBindingPlan plan;
  // Track, per SGPR range, the index of the last SMRD instruction covering it.
  struct Load {
    u32 sgpr, dwords, index;
  };
  std::vector<Load> loads;
  const auto covering_load = [&](u32 sgpr, u32 dwords) -> u32 {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->index;
    return 0xFFFF;  // inline user data (no covering load)
  };

  std::unordered_map<u64, u32> binding_of;
  u32 inst_index = 0;
  for (const Inst& inst : program) {
    const u32 idx = inst_index++;
    if (reachable && !reachable[idx])
      continue;
    if (inst.enc == Enc::kSmrd) {
      const Smrd s = DecodeSmrd(inst.raw[0]);
      if (s.op <= 0x04) {  // s_load_dword..x16 can rewrite descriptor SGPRs
        const u32 dwords = 1u << s.op;
        loads.erase(std::remove_if(loads.begin(), loads.end(),
                                   [&](const Load& ld) {
                                     return s.sdst < ld.sgpr + ld.dwords &&
                                            ld.sgpr < s.sdst + dwords;
                                   }),
                    loads.end());
        loads.push_back({s.sdst, dwords, idx});
      }
      continue;
    }
    if (inst.enc != Enc::kMimg)
      continue;
    const u32 w0 = inst.raw[0], w1 = inst.raw[1];
    const u32 op = (w0 >> 18) & 0x7F;
    const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
    const bool sampling = op >= 0x20;
    const bool storage = op == 0x08 || op == 0x09;
    const u32 ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFF;
    const u32 flags =
        (((w0 >> 14) & 1) << 0) |                      // DA
        (((op == 0x28 || op == 0x2f) ? 1 : 0) << 1) |  // dref
        ((op == 0x47 ? 1 : 0) << 2) |                  // gather4_lz
        (static_cast<u32>(storage) << 3);
    const u64 key =
        MimgDescriptorKey(srsrc, ssamp, covering_load(srsrc, 8),
                          sampling ? covering_load(ssamp, 4) : 0xFFFE, flags);
    const auto [it, inserted] = binding_of.emplace(
        key, static_cast<u32>(plan.binding_srsrc.size()));
    if (inserted) {
      plan.binding_srsrc.push_back(srsrc);
      plan.binding_storage.push_back(storage);
    }
    plan.binding_by_pc[inst.pc] = it->second;
  }
  return plan;
}

VBuffer DecodeVBuffer(const u32* p) {
  // GCN V# (buffer resource descriptor), 4 dwords:
  //  [0]  base_address[31:0]
  //  [1]  base_address[43:32] in [11:0]; [15:12] reserved;
  //       stride[13:0] in [29:16]
  //  [2]  num_records
  //  [3]  dst_sel/nfmt/dfmt/...: nfmt[14:12], dfmt[18:15]
  // The base is 44 bits, not 48: the top nibble of word 1 is reserved, and
  // Shadow of the Colossus leaves it non-zero on its per-object vertex pools.
  // Reading it as address put them at 0x7080_xxxxxxxx -- 124 TB, far outside a
  // PS4 process' ~1 TB address space -- so every descriptor carrying that
  // nibble was rejected as out of range and read back as zero.
  return {
      .base = (static_cast<u64>(p[1] & 0xFFF) << 32) | p[0],
      .stride = (p[1] >> 16) & 0x3FFF,
      .num_records = p[2],
      .dfmt = (p[3] >> 15) & 0xF,
      .nfmt = (p[3] >> 12) & 0x7,
  };
}

TImage DecodeTImage(const u32* p) {
  // GCN T# (image resource), 8 dwords:
  //  [0] base[39:8] (base = [0] << 8 with high bits from [1])
  //  [1] base_hi[5:0]; mtype_l2[7:6]; min_lod[19:8]; formats
  //  [2] width[13:0]; height[27:14]
  //  [3] dst_sel_x/y/z/w[11:0], 3 bits each; base_level[15:12];
  //      last_level[19:16]; tiling_index[24:20]; pow2_pad[25]; type[31:28]
  //  [4] depth[12:0]; pitch[26:13]
  //  [5] base_array[12:0]; last_array[25:13]
  TImage t;
  t.null_descriptor =
      std::all_of(p, p + 8, [](u32 word) { return word == 0; });
  t.base = ((static_cast<u64>(p[1] & 0x3F) << 32) | p[0]) << 8;
  t.min_lod = (p[1] >> 8) & 0xFFF;
  t.dfmt = (p[1] >> 20) & 0x3F;
  t.nfmt = (p[1] >> 26) & 0xF;
  t.width = (p[2] & 0x3FFF) + 1;
  t.height = ((p[2] >> 14) & 0x3FFF) + 1;
  for (u32 i = 0; i < 4; i++)
    t.dst_sel[i] = (p[3] >> (i * 3)) & 0x7;
  t.base_mip = (p[3] >> 12) & 0xF;
  const u32 last_mip = (p[3] >> 16) & 0xF;
  t.mip_levels = last_mip + 1;
  t.view_mips = last_mip >= t.base_mip ? last_mip - t.base_mip + 1 : 0;
  t.tiling_idx = (p[3] >> 20) & 0x1F;
  t.pow2_pad = ((p[3] >> 25) & 1) != 0;
  t.type = p[3] >> 28;
  t.pitch = ((p[4] >> 13) & 0x3FFF) + 1;
  if (t.pitch < t.width)
    t.pitch = t.width;  // fall back to width if unset
  // SQ_RSRC_IMG_2D_ARRAY / _1D_ARRAY, and CUBE. A cube is stored and sampled
  // as a 2D array whose layers are its faces -- the gfx10 decoder models it the
  // same way, and the MIMG path has already selected the face by the time the
  // address reaches the hardware. Leaving type 11 out made every PS4 cubemap
  // descriptor invalid, so the sample fell back to the 1x1 white default.
  if (t.type == 13 || t.type == 12 || t.type == 11) {
    t.layers = (p[4] & 0x1FFF) + 1;
    if (t.pow2_pad)
      t.layers = NextPow2(t.layers);
    if (t.type == 11)
      t.layers = std::max<u32>(t.layers, 6);
    t.base_array = p[5] & 0x1FFF;
    t.view_layers = 0;
    const u32 last_array = (p[5] >> 13) & 0x1FFF;
    if (t.base_array < t.layers && last_array >= t.base_array)
      t.view_layers = std::min(last_array, t.layers - 1) - t.base_array + 1;
  }
  if (t.type == 8 || t.type == 12) {  // SQ_RSRC_IMG_1D[_ARRAY]
    // Modelled as a height-1 2D image. Keep the descriptor's tiling mode so
    // Display1DThin resources use the matching detiler path.
    t.is_1d = true;
    t.height = 1;
  }
  if (t.type == 10) {  // SQ_RSRC_IMG_3D: dword 4 holds slices, not layers
    t.is_3d = true;
    t.depth = (p[4] & 0x1FFF) + 1;
    if (t.pow2_pad)
      t.depth = NextPow2(t.depth);
  }

  const bool supported_type = t.type == 8 || t.type == 9 || t.type == 10 ||
                              t.type == 11 || t.type == 12 || t.type == 13;
  const bool valid_view = (t.type != 13 && t.type != 12 && t.type != 11) ||
                          (t.base_array < t.layers && t.view_layers > 0);
  u32 max_levels = 1;
  for (u32 extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
    max_levels++;
  const bool valid_mips = t.view_mips && t.mip_levels <= max_levels;
  // A volume image only halves width/height per mip in our layout builder, so
  // the slice count of mip n would be wrong. Nothing observed ships a mipped
  // 3D texture; take only the single-level case rather than guessing a layout.
  const bool valid_3d = !t.is_3d || (t.depth <= 2048 && t.mip_levels == 1);
  t.valid = GuestRange(t.base, 1) && supported_type && t.width <= 8192 &&
            t.height <= 8192 && t.layers <= 8192 && valid_view && valid_mips &&
            valid_3d;
  return t;
}

std::vector<VBuffer> TrackVertexBuffers(const Program& fetch_program,
                                        const u32* vs_user_data) {
  std::vector<VBuffer> result;
  if (!vs_user_data)
    return result;

  // The fetch shader loads each attribute's V# with an s_load_dwordx4 whose
  // SBASE is a user-SGPR pair holding the vertex-buffer-table pointer, at byte
  // offset (offset*4 for imm). Recover the table pointer from the user data
  // and read the V# there.
  for (const Inst& inst : fetch_program) {
    if (inst.enc != Enc::kSmrd)
      continue;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    if (s.op != 0x02)
      continue;                              // s_load_dwordx4 (a 4-dword V#)
    const u32 base_sgpr = s.sbase * 2;  // user_data index of the table ptr
    if (base_sgpr + 1 >= 16)
      continue;
    const u64 table = UserDataPointer(vs_user_data, base_sgpr);
    if (!GuestRange(table, 16))
      continue;
    const SmrdOffset so = DecodeSmrdOffset(inst);
    if (so.in_sgpr)
      continue;  // a register offset needs the scalar walk, not this scan
    const u32 byte_off = so.dwords * 4;
    const VBuffer v =
        DecodeVBuffer(reinterpret_cast<const u32*>(table + byte_off));
    if (v.base >= kGuestLo && v.base < kGuestHi && v.stride &&
        v.stride <= 256 && v.num_records && v.num_records <= 0x100000) {
      if (kTrace)
        BASE_LOGI("gcnres",
                  "VB sbase=sgpr{} table={:#x} off={} -> base={:#x} "
                  "stride={} nrec={} dfmt={} nfmt={}",
                  base_sgpr, static_cast<unsigned long>(table), byte_off,
                  static_cast<unsigned long>(v.base), v.stride,
                  v.num_records, v.dfmt, v.nfmt);
      result.push_back(v);
    }
  }
  return result;
}

std::vector<TImage> TrackTextures(
    const std::shared_ptr<const Program>& ps_program,
    const u32* ps_user_data,
    bool trace,
    u64 code_base) {
  std::vector<TImage> result;
  if (!ps_program || !ps_user_data)
    return result;

  if (kTwatch && ++g_track_draws % 4000 == 0)
    PollNullDescriptors();

  // Bindings come from the shared plan (one per unique descriptor identity),
  // so this list pairs 1:1 with the recompiled shader's set-0 samplers.
  const ScalarPassInfo& cached = CachedScalarInfo(ps_program);
  const MimgBindingPlan& plan = cached.plan;

  // Step the scalar register file across the program; at each MIMG read the
  // live T#/S# straight out of the resolved SGPRs. Inline user data, a single
  // indirect load, and nested EUD chains all land here identically.
  ScalarEval eval(ps_user_data, code_base);
  eval.trace |= trace;

  for (const Inst& inst : cached.insts) {
    eval.Step(inst);
    if (inst.enc != Enc::kMimg)
      continue;
    const auto plan_it = plan.binding_by_pc.find(inst.pc);
    if (plan_it == plan.binding_by_pc.end())
      continue;  // unreachable
    const u32 binding = plan_it->second;
    const u32 word1 = inst.raw[1];
    const u32 srsrc = ((word1 >> 16) & 0x1F) * 4;  // T# base SGPR
    const u32 op = (inst.raw[0] >> 18) & 0x7F;

    // Resolve the sampler for sampling ops (used both for new bindings and to
    // backfill a binding first seen through a non-sampling op like resinfo).
    u32 sampler[4] = {};
    bool sampler_ok = false;
    if (op >= 0x20) {
      const u32 ssamp = ((word1 >> 21) & 0x1F) * 4;
      if (eval.AllKnown(ssamp, 4)) {
        std::memcpy(sampler, &eval.sgpr[ssamp], sizeof(sampler));
        sampler_ok = true;
      }
    }

    if (binding < result.size()) {  // repeat use of an existing binding
      TImage& entry = result[binding];
      if (sampler_ok && !entry.sampler_valid) {
        std::memcpy(entry.sampler, sampler, sizeof(entry.sampler));
        entry.sampler_valid = true;
      }
      continue;
    }

    TImage t;
    const bool image_ok = eval.AllKnown(srsrc, 8);
    if (image_ok) {
      t = DecodeTImage(&eval.sgpr[srsrc]);
      t.src = eval.src[srsrc];
    }
    if (kTwatch && t.null_descriptor) {
      NoteNullDescriptor(eval.src[srsrc], code_base);
      // The arenas are 2 MiB. If the descriptor the shader wanted sits a whole
      // arena away from where it looked, the title and we disagree about which
      // arena is current -- a constant bias, not a lost write.
      static int probes = 0;
      const u64 at = eval.src[srsrc];
      if (at && probes < 24) {
        probes++;
        // Widened to +-16: the registers at the write say the arena stride is
        // 0x400000 and the bad pointer is base + 8 strides, i.e. 32 MiB out,
        // which a +-4 window stepping 2 MiB could never reach.
        for (int slot = -16; slot <= 16; slot++) {
          if (!slot)
            continue;
          const u64 probe = at + static_cast<i64>(slot) * 0x200000;
          if (!GuestRange(probe, 32))
            continue;
          const u32* w = reinterpret_cast<const u32*>(probe);
          if (std::all_of(w, w + 8, [](u32 v) { return v == 0; }))
            continue;
          const TImage probe_t = DecodeTImage(w);
          BASE_LOGI("arena",
                    "null at {:#x}: arena{:+d} ({:#x}) holds {}x{} "
                    "valid={:d} raw={:08x}/{:08x}",
                    static_cast<unsigned long>(at), slot,
                    static_cast<unsigned long>(probe), probe_t.width,
                    probe_t.height, probe_t.valid, w[0], w[1]);
        }
      }
    }
    // DELTA_GPU_NULLDIS=1: disassemble the first shader that resolves a null
    // descriptor while sampling through a chain of more than one hop. Our
    // scalar walk steps the program in ORDER and ignores branches, so a shader
    // that selects its table behind a branch (or by an index we cannot fold)
    // gets a deterministically wrong address -- which is what a constant
    // offset between where the title wrote its table and where we looked
    // would look like.
    // DELTA_GPU_NULLWATCH=1: watch the SRT slot whose pointer led to a null
    // descriptor, so the guest instruction that wrote that pointer names
    // itself. This is the one address worth watching and it is not knowable
    // until a draw is processed -- it moves every run -- which is why the arm
    // goes through utl rather than an env var parsed at startup.
    if (kNullWatch && t.null_descriptor) {
      static bool armed = false;
      const u64 root = UserDataPointer(ps_user_data, 0);
      if (!armed && root && GuestRange(root, 64)) {
        armed = true;
        BASE_LOGI("nullwatch",
                  "arming on SRT {:#x} (+0x18 held the pointer "
                  "into the empty arena; T# read at {:#x})",
                  static_cast<unsigned long>(root),
                  static_cast<unsigned long>(eval.src[srsrc]));
        // +0x18 is the slot the chain read the table pointer from.
        utl::setWriteWatchValueProbe(static_cast<uintptr_t>(root) + 0x18);
        utl::setWriteWatchChase(4);  // follow it back up to four copies
        if (!utl::armWriteWatch(static_cast<uintptr_t>(root), 64, 200))
          BASE_LOGI("nullwatch", "no armer registered");
      }
    }
    if (kNullDis && t.null_descriptor) {
      static bool dumped = false;
      if (!dumped) {
        dumped = true;
        BASE_LOGI("nulldis",
                  "PS {:#x} binding {} read a null T# from {:#x} "
                  "(SRT root {:#x})",
                  static_cast<unsigned long>(code_base), binding,
                  static_cast<unsigned long>(eval.src[srsrc]),
                  static_cast<unsigned long>(
                      UserDataPointer(ps_user_data, 0)));
        DisassembleAt(code_base, "nulldis.PS");
      }
    }
    if (kTscan && t.null_descriptor) {
      static bool scanned = false;
      static const auto kScanStart = std::chrono::steady_clock::now();
      const bool due = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - kScanStart)
                           .count() >= kTscanAfter;
      if (!scanned && due) {
        scanned = true;
        BASE_LOGI("tscan",
                  "triggered by null T# in PS {:#x} binding {}, "
                  "descriptor read from {:#x}",
                  static_cast<unsigned long>(code_base), binding,
                  static_cast<unsigned long>(eval.src[srsrc]));
        CensusBlock("descriptor read", eval.src[srsrc]);
        // One level up: the SRT block the draw's user data points at. If that
        // is empty too, the title never built the resource block at all and the
        // descriptor table below it is a red herring.
        const u64 srt = UserDataPointer(ps_user_data, 0);
        BASE_LOGI("tscan", "SRT root (user s[0:1]) = {:#x}",
                  static_cast<unsigned long>(srt));
        if (GuestRange(srt, 64)) {
          const u32* p = reinterpret_cast<const u32*>(srt);
          base::String srt_words;
          base::FormatTo(srt_words, "  SRT[0..15]:");
          for (int i = 0; i < 16; i++)
            base::FormatTo(srt_words, " {:08x}", p[i]);
          BASE_LOGI("tscan", "{}", srt_words.c_str());
          CensusBlock("SRT root", srt);
        } else {
          BASE_LOGI("tscan", "  SRT root not mapped");
        }
        const u64 good = ScanForDescriptor(kTscan);
        if (good)
          CensusBlock("first valid copy", good);
      }
    }
    if (kArenaProbe && t.null_descriptor && eval.src[srsrc]) {
      const u64 at = eval.src[srsrc];
      for (int slot = -1; slot >= -kArenaProbe; slot--) {
        const u64 probe = at + static_cast<i64>(slot) * 0x200000;
        if (!GuestRange(probe, 32))
          continue;
        const u32* w = reinterpret_cast<const u32*>(probe);
        if (std::all_of(w, w + 8, [](u32 v) { return v == 0; }))
          continue;
        const TImage cand = DecodeTImage(w);
        if (!cand.valid)
          continue;
        t = cand;
        t.src = at;
        static int announced = 0;
        if (announced < 8) {
          announced++;
          BASE_LOGI("arena",
                    "substituted {:#x} -> {:#x} (arena{:+d}) {}x{}",
                    static_cast<unsigned long>(at),
                    static_cast<unsigned long>(probe), slot, t.width,
                    t.height);
          // Disassemble the shader that produced the biased pointer once: a
          // constant arena bias is most likely an address our linear scalar
          // replay computed down a path the real wave would not have taken.
          static bool dumped = false;
          if (!dumped) {
            dumped = true;
            BASE_LOGI("arena", "SRT root = {:#x}, T# read at {:#x}",
                      static_cast<unsigned long>(
                          UserDataPointer(ps_user_data, 0)),
                      static_cast<unsigned long>(at));
            DisassembleAt(code_base, "arena.PS");
          }
        }
        break;
      }
    }
    if (code_base == 0x80720da900 && binding == 0 && t.null_descriptor &&
        kSotcCompositeRt) {
      const u64 base = kSotcCompositeRt;
      const u32 descriptor[8] = {
          static_cast<u32>(base >> 8),
          static_cast<u32>((base >> 40) & 0x3f) | 0x1c400000,
          ((270 - 1) << 14) | (960 - 1),
          0x94000fac,
          (1024 - 1) << 13,
          0,
          0,
          0,
      };
      t = DecodeTImage(descriptor);
      t.src = eval.src[srsrc];
      const u64 descriptor_at = eval.src[srsrc];
      const u64 style_at = eval.src[16];
      if (GuestRange(style_at - 4, 12)) {
        auto* style = reinterpret_cast<u32*>(style_at - 4);
        style[0] = 0x3f800000;  // outline threshold (disabled below)
        style[1] = 0x3f000000;  // SDF edge threshold
        style[2] = 0x42000000;  // atlas footprint scale
      }
      if (GuestRange(descriptor_at - 32, 20)) {
        auto* outline = reinterpret_cast<u32*>(descriptor_at - 32);
        outline[0] = 0;
        outline[1] = 0;
        outline[2] = 0;
        outline[3] = 0x3f800000;
        outline[4] = 0;  // keep the optional outline branch disabled
      }
      static bool announced = false;
      if (!announced) {
        announced = true;
        BASE_LOGI("sotc-composite",
                  "substituted null T# with {:#x} 960x270 "
                  "valid={:d}",
                  static_cast<unsigned long>(base), t.valid);
      }
    }
    if (eval.trace)
      BASE_LOGI("eud",
                "MIMG pc={:#x} bind={} srsrc=s{} known={:d} at={:#x} "
                "base={:#x} {}x{} valid={:d} raw={:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/"
                "{:08x}/{:08x}",
                inst.pc, binding, srsrc, image_ok,
                static_cast<unsigned long>(eval.src[srsrc]),
                static_cast<unsigned long>(t.base), t.width, t.height,
                t.valid, image_ok ? eval.sgpr[srsrc + 0] : 0,
                image_ok ? eval.sgpr[srsrc + 1] : 0,
                image_ok ? eval.sgpr[srsrc + 2] : 0,
                image_ok ? eval.sgpr[srsrc + 3] : 0,
                image_ok ? eval.sgpr[srsrc + 4] : 0,
                image_ok ? eval.sgpr[srsrc + 5] : 0,
                image_ok ? eval.sgpr[srsrc + 6] : 0,
                image_ok ? eval.sgpr[srsrc + 7] : 0);
    // DELTA_GPU_TEXSRC=<base>: where in guest memory the T# for this surface
    // lives. A surface the GPU samples but nothing ever fills is only
    // explainable from the code that publishes its address.
    if (kTexSrc && t.base >= (u64)kTexSrc) {
      static std::set<u64> seen;
      if (seen.size() < 64 && seen.insert(t.base).second)
        BASE_LOGI("texsrc", "base={:#x} T# at {:#x} (sgpr{})",
                  static_cast<unsigned long>(t.base),
                  static_cast<unsigned long>(eval.src[srsrc]), srsrc);
    }
    if (sampler_ok) {
      std::memcpy(t.sampler, sampler, sizeof(t.sampler));
      t.sampler_valid = true;
    }
    t.arrayed = (inst.raw[0] & 0x4000) != 0;  // MIMG DA
    t.force_lod_zero = op == 0x47;            // IMAGE_GATHER4_LZ
    t.depth_compare = op == 0x28 || op == 0x2f;
    t.storage = op == 0x08 || op == 0x09;
    if (t.valid) {
      // Empirical tiling census (DELTA_GPU_TILEHIST): tally tiling_idx of
      // every sampled texture to confirm which modes are linear vs tiled.
      if (kGpuTilehist) {
        static u32 hist[32] = {0};
        static u64 n = 0, pitch_ne = 0;
        hist[t.tiling_idx & 31]++;
        if (t.pitch != t.width)
          pitch_ne++;
        if ((++n % 4000) == 0) {
          base::String line;
          base::FormatTo(line, "n={} pitch!=width={}:",
                         static_cast<unsigned long>(n),
                         static_cast<unsigned long>(pitch_ne));
          for (int i = 0; i < 32; i++)
            if (hist[i])
              base::FormatTo(line, " idx{}={}", i, hist[i]);
          BASE_LOGI("tilehist", "{}", line.c_str());
        }
      }
      if (kTrace)
        BASE_LOGI("gcnres",
                  "T# (sgpr{}) base={:#x} {}x{} pitch={} "
                  "dfmt={} nfmt={} tiling={}",
                  srsrc, static_cast<unsigned long>(t.base), t.width,
                  t.height, t.pitch, t.dfmt, t.nfmt, t.tiling_idx);
    }
    result.push_back(t);
  }
  return result;
}

std::unordered_map<u32, VBuffer> ResolveCbuffers(
    const std::shared_ptr<const Program>& program,
    const u32* user_data) {
  std::unordered_map<u32, VBuffer> result;
  if (!program || !user_data)
    return result;

  // Mirror TrackTextures: step the scalar register file, and at each
  // s_buffer_load read the live 4-dword V# out of the resolved SGPRs. FOX
  // passes cbuffer descriptors through extended user data too (s_load the V#
  // through an EUD pointer, then s_buffer_load through it), so reading the V#
  // straight from user data yields base=0. The recompiler assigns one binding
  // per base SGPR (PlanCbufs), so key by base SGPR and keep the first
  // resolvable V# seen for it.
  // One sweep per run, once the title is past its load: the data a descriptor
  // should have named exists by then, and the sweep costs a full pass over
  // every mapped guest page.
  if ((kMemFind && *kMemFind) || kSoaScan) {
    static const auto epoch = std::chrono::steady_clock::now();
    static bool swept = false;
    if (!swept && std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - epoch).count() > 60.0) {
      swept = true;
      if (kMemFind && *kMemFind)
        ScanForDwords(kMemFind);
      if (kSoaScan)
        ScanForQuadColumns(kSoaScan.get());
    }
  }
  ScalarEval eval(user_data);
  for (const Inst& inst : CachedScalarInfo(program).insts) {
    // Decode the pointer/V# from the PRE-step register state: an SMRD whose
    // destination overlaps its own source (s_buffer_load_dword s4, s[4:7])
    // clobbers the V#'s base dword in Step, and decoding afterwards turns the
    // loaded constant into the address (Tomb Raider's UI globals cbuf decoded
    // as base 0x803f800000 -- the 1.0f it had just loaded).
    const bool smrd = inst.enc == Enc::kSmrd;
    const Smrd s = smrd ? DecodeSmrd(inst.raw[0]) : Smrd{};
    const bool candidate =
        smrd && !(s.op > 0x0c || (s.op > 0x04 && s.op < 0x08));
    // s_load_dword{,x2..x16} / s_buffer_load_dword{,x2..x16}.
    // s_load addresses a raw 2-dword pointer, s_buffer_load a 4-dword V#.
    const bool pointer = s.op <= 0x04;
    const u32 base = s.sbase * 2;
    // Keyed like the translator's bindings (see CbufBindKey): the same SGPR
    // can serve as a pointer pair for one load and a V# for another.
    const u32 key = base | (pointer ? 0x100u : 0u);
    const bool known = candidate && eval.AllKnown(base, pointer ? 2 : 4);
    VBuffer v{};
    if (known) {
      if (pointer)
        v.base = eval.Ptr(base);  // size comes from the shader's plan
      else
        v = DecodeVBuffer(&eval.sgpr[base]);
    }
    eval.Step(inst);
    if (!candidate || result.count(key) || !known)
      continue;
    result.emplace(key, v);
    if (eval.trace)
      BASE_LOGI("eud", "cbuf s{} -> base={:#x} stride={} nrec={}",
                base, static_cast<unsigned long>(v.base), v.stride,
                v.num_records);
  }
  return result;
}

std::vector<VBuffer> ResolveDirectVertexBuffers(
    const std::shared_ptr<const Program>& program,
    const std::vector<ShaderAttr>& attrs,
    const u32* user_data) {
  std::vector<VBuffer> result(attrs.size());
  if (!program || !user_data || attrs.empty())
    return result;

  ScalarEval eval(user_data);
  for (const Inst& inst : CachedScalarInfo(program).insts) {
    eval.Step(inst);
    if (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf)
      continue;
    for (size_t i = 0; i < attrs.size(); i++) {
      const ShaderAttr& attr = attrs[i];
      if (!attr.direct_fetch || attr.use_pc != inst.pc ||
          !eval.AllKnown(attr.table_sgpr, 4))
        continue;
      result[i] = DecodeVBuffer(&eval.sgpr[attr.table_sgpr]);
      if (eval.trace) {
        u32 data[3] = {};
        if (GuestRange(result[i].base, sizeof(data)))
          std::memcpy(data, reinterpret_cast<const void*>(result[i].base),
                      sizeof(data));
        BASE_LOGI("eud",
                  "vattr{} pc={:#x} s{} -> base={:#x} stride={} "
                  "nrec={} fmt={}/{} inst={}/{} ioff={} "
                  "V#={:08x}/{:08x}/{:08x}/{:08x} data={:08x}/{:08x}/{:08x}",
                  i, inst.pc, attr.table_sgpr,
                  static_cast<unsigned long>(result[i].base),
                  result[i].stride, result[i].num_records, result[i].dfmt,
                  result[i].nfmt, attr.inst_dfmt, attr.inst_nfmt,
                  inst.raw[0] & 0xFFF, eval.sgpr[attr.table_sgpr],
                  eval.sgpr[attr.table_sgpr + 1],
                  eval.sgpr[attr.table_sgpr + 2],
                  eval.sgpr[attr.table_sgpr + 3], data[0], data[1], data[2]);
      }
    }
  }
  return result;
}

std::vector<VBuffer> ResolveShaderBuffers(
    const std::shared_ptr<const Program>& program,
    const std::vector<ShaderBuffer>& buffers,
    const u32* user_data) {
  std::vector<VBuffer> result(buffers.size());
  if (!program || !user_data || buffers.empty())
    return result;

  ScalarEval eval(user_data);
  for (const Inst& inst : CachedScalarInfo(program).insts) {
    eval.Step(inst);
    if (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf)
      continue;
    for (size_t i = 0; i < buffers.size(); i++) {
      const ShaderBuffer& buffer = buffers[i];
      if (buffer.use_pc != inst.pc || !eval.AllKnown(buffer.srsrc_sgpr, 4))
        continue;
      result[i] = DecodeVBuffer(&eval.sgpr[buffer.srsrc_sgpr]);
      if (eval.trace) {
        // Whether the nibble DecodeVBuffer treats as reserved is an address
        // after all: a buffer that resolves into a mapped arena and reads as
        // zeros looks exactly like one whose top bits were thrown away.
        const u64 wide =
            (static_cast<u64>(eval.sgpr[buffer.srsrc_sgpr + 1] & 0xFFFF)
             << 32) |
            eval.sgpr[buffer.srsrc_sgpr];
        BASE_LOGI("eud", "rawbuf{} 48-bit base would be {:#x}, mapped={}", i,
                  static_cast<unsigned long>(wide),
                  (int)GuestRange(wide, 16));
        BASE_LOGI("eud",
                  "rawbuf{} pc={:#x} s{} -> base={:#x} stride={} "
                  "nrec={} V#={:08x}/{:08x}/{:08x}/{:08x}",
                  i, inst.pc, buffer.srsrc_sgpr,
                  static_cast<unsigned long>(result[i].base), result[i].stride,
                  result[i].num_records, eval.sgpr[buffer.srsrc_sgpr],
                  eval.sgpr[buffer.srsrc_sgpr + 1],
                  eval.sgpr[buffer.srsrc_sgpr + 2],
                  eval.sgpr[buffer.srsrc_sgpr + 3]);
      }
    }
  }
  return result;
}

std::vector<ResolvedCsResource> ResolveCsResources(const Program& program,
                                                   const RecompiledCs& plan,
                                                   const u32* user_data) {
  std::vector<ResolvedCsResource> result(plan.resources.size());
  if (!user_data)
    return result;

  ScalarEval eval(user_data);
  for (const Inst& inst : program) {
    // Capture before Step(): an s_load may use a pointer in the same SGPR range
    // it overwrites with the loaded descriptor.
    for (const CsResource& resource : plan.resources) {
      if (resource.use_pc != inst.pc || resource.binding >= result.size())
        continue;
      const u32 dwords = resource.kind == 1   ? 8
                              : resource.kind == 2 ? 2
                                                   : 4;
      if (!eval.AllKnown(resource.base_sgpr, dwords))
        continue;
      ResolvedCsResource& resolved = result[resource.binding];
      std::memcpy(resolved.descriptor, &eval.sgpr[resource.base_sgpr],
                  dwords * sizeof(u32));
      resolved.valid = true;
    }
    eval.Step(inst);
  }
  return result;
}

}  // namespace gpu::gcn
