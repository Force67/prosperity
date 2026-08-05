/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gpu/gcn/gcn_resource.h"

#include "gpu/gcn/gcn_translate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kGpuEudfail, "DELTA_GPU_EUDFAIL", false);
DELTA_OPTION(bool, kGpuEudtrace, "DELTA_GPU_EUDTRACE", false);
DELTA_OPTION(bool, kGpuTilehist, "DELTA_GPU_TILEHIST", false);
DELTA_OPTION(bool, kTrace, "DELTA_GPU_TRACE", false);
DELTA_OPTION(uint64_t, kSotcCompositeRt, "DELTA_GPU_SOTC_COMPOSITE_RT", 0);
DELTA_OPTION(uint64_t, kTexSrc, "DELTA_GPU_TEXSRC", 0);
DELTA_OPTION(uint64_t, kTscan, "DELTA_GPU_TSCAN", 0);
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
namespace {


constexpr uint64_t kGuestLo = 0x1000000000ull;
constexpr uint64_t kGuestHi = 0x20000000000ull;

bool GuestRange(uint64_t address, uint64_t size) {
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
uint64_t ScanForDescriptor(uint64_t want_base) {
  const uint32_t want_word0 = static_cast<uint32_t>(want_base >> 8);
  const uint32_t want_hi = static_cast<uint32_t>((want_base >> 40) & 0x3F);
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) {
    std::fprintf(stderr, "[tscan] cannot read /proc/self/maps\n");
    return 0;
  }
  std::fprintf(stderr, "[tscan] sweeping for base=%#lx (word0=%08x hi=%u)\n",
               static_cast<unsigned long>(want_base), want_word0, want_hi);
  char line[512];
  uint64_t scanned = 0, hits = 0, first_valid = 0;
  while (std::fgets(line, sizeof(line), maps)) {
    uint64_t lo = 0, hi = 0;
    char perms[8] = {};
    if (std::sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3)
      continue;
    if (perms[0] != 'r' || lo < kGuestLo || hi > kGuestHi || hi <= lo)
      continue;
    scanned += hi - lo;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(lo);
    const uint64_t n = (hi - lo) / 4;
    for (uint64_t i = 0; i + 8 <= n; i++) {
      if (p[i] != want_word0 || (p[i + 1] & 0x3F) != want_hi)
        continue;
      hits++;
      if (hits > 64)
        continue;
      const uint64_t at = lo + i * 4;
      const TImage t = DecodeTImage(&p[i]);
      if (t.valid && !first_valid)
        first_valid = at;
      std::fprintf(stderr,
                   "[tscan] hit at=%#lx %ux%u pitch=%u dfmt=%u nfmt=%u "
                   "valid=%d raw=%08x/%08x/%08x/%08x/%08x/%08x/%08x/%08x\n",
                   static_cast<unsigned long>(at), t.width, t.height, t.pitch,
                   t.dfmt, t.nfmt, t.valid, p[i], p[i + 1], p[i + 2], p[i + 3],
                   p[i + 4], p[i + 5], p[i + 6], p[i + 7]);
    }
  }
  std::fclose(maps);
  std::fprintf(stderr, "[tscan] done: %lu hits over %lu MiB\n",
               static_cast<unsigned long>(hits),
               static_cast<unsigned long>(scanned >> 20));
  return first_valid;
}

// How much of the 4 MiB pool block around `address` was ever written. A block
// the title filled reads mostly non-zero; a block it only reserved reads zero
// end to end, which is what tells a stale pointer apart from a torn write.
void CensusBlock(const char* what, uint64_t address) {
  constexpr uint64_t kBlock = 0x400000;
  const uint64_t base = address & ~(kBlock - 1);
  if (!GuestRange(base, kBlock)) {
    std::fprintf(stderr, "[census] %s block %#lx not mapped\n", what,
                 static_cast<unsigned long>(base));
    return;
  }
  const uint32_t* p = reinterpret_cast<const uint32_t*>(base);
  uint64_t nz = 0, first_nz = 0, last_nz = 0;
  for (uint64_t i = 0; i < kBlock / 4; i++) {
    if (!p[i])
      continue;
    nz++;
    if (!first_nz)
      first_nz = base + i * 4;
    last_nz = base + i * 4;
  }
  std::fprintf(stderr,
               "[census] %s %#lx: block %#lx has %lu/%lu non-zero dwords, "
               "written span %#lx..%#lx\n",
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
  uint64_t address;
  uint64_t code_base;
  uint64_t draw_seen;
  bool filled;
  // Highest offset in the containing 4 MiB arena that has ever been non-zero
  // while we watched. If this never reaches the offset the shader read, the
  // pointer names a fill level the arena no longer has (a stale pointer); if it
  // passes it, we walked the command buffer at the wrong moment.
  uint64_t peak_watermark;
};
std::vector<NullSite> g_null_sites;
uint64_t g_track_draws = 0;

void NoteNullDescriptor(uint64_t address, uint64_t code_base) {
  if (!address)
    return;
  for (const NullSite& s : g_null_sites)
    if (s.address == address)
      return;
  if (g_null_sites.size() < 256)
    g_null_sites.push_back({address, code_base, g_track_draws, false, 0});
}

// Highest non-zero dword offset inside the 4 MiB arena holding `address`.
uint64_t BlockWatermark(uint64_t address) {
  constexpr uint64_t kBlock = 0x400000;
  const uint64_t base = address & ~(kBlock - 1);
  if (!GuestRange(base, kBlock))
    return 0;
  const uint32_t* p = reinterpret_cast<const uint32_t*>(base);
  for (uint64_t i = kBlock / 4; i-- > 0;)
    if (p[i])
      return i * 4;
  return 0;
}

void PollNullDescriptors() {
  uint32_t filled = 0, still_zero = 0;
  // The watermark sweep is 4 MiB per site, so only the first few are tracked.
  uint32_t watched = 0;
  for (NullSite& s : g_null_sites) {
    if (watched >= 6 || s.filled)
      continue;
    watched++;
    const uint64_t mark = BlockWatermark(s.address);
    if (mark > s.peak_watermark)
      s.peak_watermark = mark;
    const uint64_t want = s.address & 0x3FFFFF;
    std::fprintf(stderr,
                 "[wmark] %#lx offset=%#lx arena peak=%#lx now=%#lx -> %s\n",
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
    const uint32_t* p = reinterpret_cast<const uint32_t*>(s.address);
    if (std::all_of(p, p + 8, [](uint32_t w) { return w == 0; })) {
      still_zero++;
      continue;
    }
    s.filled = true;
    filled++;
    std::fprintf(stderr,
                 "[twatch] %#lx (read null by PS %#lx at draw %lu) is NOW "
                 "%08x/%08x/%08x/%08x after %lu more draws\n",
                 static_cast<unsigned long>(s.address),
                 static_cast<unsigned long>(s.code_base),
                 (unsigned long)s.draw_seen, p[0], p[1], p[2], p[3],
                 (unsigned long)(g_track_draws - s.draw_seen));
  }
  std::fprintf(stderr, "[twatch] %u of %u null sites later filled, %u still zero\n",
               filled, static_cast<unsigned>(g_null_sites.size()), still_zero);
}

// SMRD operand fields (GFX7).
struct Smrd {
  uint32_t op;
  uint32_t sdst;
  uint32_t sbase;  // SGPR pair index; actual base SGPR = sbase * 2
  uint32_t offset;
  bool imm;
};

Smrd DecodeSmrd(uint32_t w) {
  return {
      .op = (w >> 22) & 0x1F,
      .sdst = (w >> 15) & 0x7F,
      .sbase = (w >> 9) & 0x3F,
      .offset = w & 0xFF,
      .imm = ((w >> 8) & 1) != 0,
  };
}

uint64_t UserDataPointer(const uint32_t* user_data, uint32_t sgpr) {
  return (static_cast<uint64_t>(user_data[sgpr + 1] & 0xFFFF) << 32) |
         user_data[sgpr];
}

uint32_t NextPow2(uint32_t v) {
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
uint64_t MimgDescriptorKey(uint32_t srsrc,
                           uint32_t ssamp,
                           uint32_t load_rsrc,
                           uint32_t load_samp,
                           uint32_t flags) {
  return (static_cast<uint64_t>(srsrc) << 0) |
         (static_cast<uint64_t>(ssamp) << 8) |
         (static_cast<uint64_t>(load_rsrc & 0xFFFF) << 16) |
         (static_cast<uint64_t>(load_samp & 0xFFFF) << 32) |
         (static_cast<uint64_t>(flags) << 48);
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
bool Sop1DestIs64(uint32_t op) {
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

bool Sop2DestIs64(uint32_t op) {
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
bool Vop3bWritesSdst(uint32_t op) {
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
  static constexpr uint32_t kRegs = 136;
  uint32_t sgpr[kRegs] = {};
  bool known[kRegs] = {};
  uint64_t src[kRegs] = {};  // guest address each dword was s_loaded from
  bool trace = false;
  uint64_t code_base = 0;  // guest address of the program, for s_getpc_b64

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
  std::unordered_map<uint32_t, uint32_t> lane_spill;
  std::unordered_map<uint32_t, uint64_t> lane_spill_src;

  explicit ScalarEval(const uint32_t* user_data, uint64_t base = 0) {
    for (uint32_t i = 0; i < 16; i++) {
      sgpr[i] = user_data[i];
      known[i] = true;
    }
    code_base = base;
    trace = kGpuEudtrace;
  }

  uint64_t Ptr(uint32_t s) const {  // 48-bit descriptor-table pointer pair
    return (static_cast<uint64_t>(sgpr[s + 1] & 0xFFFF) << 32) | sgpr[s];
  }
  bool AllKnown(uint32_t s, uint32_t n) const {
    if (s + n > kRegs)
      return false;
    for (uint32_t i = 0; i < n; i++)
      if (!known[s + i])
        return false;
    return true;
  }
  void Set(uint32_t s, uint32_t v) {
    if (s < kRegs) {
      sgpr[s] = v;
      known[s] = true;
    }
  }
  void Clear(uint32_t s) {
    if (s < kRegs)
      known[s] = false;
  }
  bool Source(uint32_t field, uint32_t literal, uint32_t& value) const {
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
      value = static_cast<uint32_t>(-static_cast<int32_t>(field - 192));
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
  bool SourceHi(uint32_t field, uint32_t& value) const {
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
    const uint32_t w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
    // VOP3 re-encodes the VOP1 (0x180+) and VOP2 (0x100+) opcodes and moves
    // the operands into the second dword.
    const bool readlane = (vop2 && op == 0x01) || (vop3 && op == 0x101);
    const bool writelane = (vop2 && op == 0x02) || (vop3 && op == 0x102);
    const bool readfirstlane = (vop1 && op == 0x02) || (vop3 && op == 0x182);
    const uint32_t dst = vop3 ? (w & 0xFF) : ((w >> 17) & 0xFF);
    const uint32_t src0 = vop3 ? (w1 & 0x1FF) : (w & 0x1FF);
    const uint32_t src1 = vop3 ? ((w1 >> 9) & 0x1FF) : ((w >> 9) & 0xFF);
    const auto forget = [&](uint32_t slot) {
      lane_spill.erase(slot);
      lane_spill_src.erase(slot);
    };
    if (writelane) {
      uint32_t value = 0, lane = 0;
      const bool lane_known = Source(src1, inst.literal, lane);
      const bool value_known = Source(src0, inst.literal, value);
      if (!lane_known) {
        for (uint32_t i = 0; i < 64; i++)  // could have landed anywhere
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
      uint32_t lane = 0;
      const bool lane_known = readfirstlane || Source(src1, inst.literal, lane);
      const uint32_t slot = (src0 - 256) * 64 + (lane & 63);
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
      const uint32_t sdst = op < 0x100 ? dst : ((w >> 8) & 0x7F);
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
      const uint32_t w = inst.raw[0];
      const uint32_t sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (inst.opcode ==
          0x03) {  // s_mov_b32: stage a pointer via a scalar move
        uint32_t value;
        if (Source(ssrc0, inst.literal, value))
          Set(sdst, value);
        else
          Clear(sdst);
      } else if (inst.opcode == 0x04) {  // s_mov_b64
        uint32_t lo, hi;
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
          const uint64_t pc =
              code_base + static_cast<uint64_t>(inst.pc + inst.size) * 4;
          Set(sdst, static_cast<uint32_t>(pc));
          Set(sdst + 1, static_cast<uint32_t>(pc >> 32));
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
      const uint32_t w = inst.raw[0];
      const uint32_t sdst = (w >> 16) & 0x7F;
      const uint32_t simm =
          static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(w)));
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
      const uint32_t w = inst.raw[0], op = inst.opcode;
      const uint32_t sdst = (w >> 16) & 0x7F;
      const bool dest64 = Sop2DestIs64(op);
      uint32_t a, b;
      const bool inputs_known = Source(w & 0xFF, inst.literal, a) &&
                                Source((w >> 8) & 0xFF, inst.literal, b);
      if (!inputs_known) {
        Clear(sdst);
        if (dest64)
          Clear(sdst + 1);
        return;
      }
      uint32_t value;
      switch (op) {
        case 0x06:
          value = static_cast<uint32_t>(
              std::min(static_cast<int32_t>(a), static_cast<int32_t>(b)));
          break;
        case 0x07:
          value = std::min(a, b);
          break;
        case 0x08:
          value = static_cast<uint32_t>(
              std::max(static_cast<int32_t>(a), static_cast<int32_t>(b)));
          break;
        case 0x09:
          value = std::max(a, b);
          break;
        case 0x27: {  // s_bfe_u32: src1[4:0] offset, src1[22:16] width
          const uint32_t width = (b >> 16) & 0x7F;
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
          value = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31));
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
    const uint32_t dwords = buffer_load ? (1u << (s.op - 0x08)) : (1u << s.op);
    const uint32_t base = s.sbase * 2;
    const uint32_t desc_dwords = buffer_load ? 4 : 2;
    const bool base_known = AllKnown(base, desc_dwords);
    const uint64_t table =
        base_known ? (buffer_load ? DecodeVBuffer(&sgpr[base]).base : Ptr(base))
                   : 0;
    bool offset_known = true;
    uint64_t byte_off = 0;
    const SmrdOffset so = DecodeSmrdOffset(inst);
    if (!so.in_sgpr) {
      byte_off = static_cast<uint64_t>(so.dwords) * 4;
    } else if (so.sgpr < kRegs && known[so.sgpr]) {
      byte_off = sgpr[so.sgpr];
    } else {
      offset_known = false;
    }
    // A load rewrites its destination SGPRs even if it cannot be resolved;
    // snapshot its inputs before invalidating an overlapping destination.
    for (uint32_t i = 0; i < dwords; i++)
      Clear(s.sdst + i);
    if (!base_known || !offset_known) {
      if (kGpuEudfail)
        std::fprintf(
            stderr,
            "[eudfail] s_load x%u s%u <- s%u: base_known=%d off_known=%d\n",
            dwords, s.sdst, base, base_known, offset_known);
      return;
    }
    if (byte_off > UINT64_MAX - table)
      return;
    const uint64_t address = table + byte_off;
    if (!GuestRange(address, static_cast<uint64_t>(dwords) * 4)) {
      if (kGpuEudfail)
        std::fprintf(
            stderr,
            "[eudfail] s_load UNMAPPED x%u s%u <- [s%u=%#lx + %#lx] = %#lx\n",
            dwords, s.sdst, base, static_cast<unsigned long>(table),
            static_cast<unsigned long>(byte_off),
            static_cast<unsigned long>(address));
      return;
    }
    const uint32_t* mem = reinterpret_cast<const uint32_t*>(address);
    for (uint32_t i = 0; i < dwords; i++) {
      Set(s.sdst + i, mem[i]);
      if (s.sdst + i < kRegs)
        src[s.sdst + i] = address + i * 4;
    }
    if (trace)
      std::fprintf(stderr,
                   "[eudenc] pc=%#x raw=%08x op=%#x sdst=s%u sbase=%u imm=%d "
                   "offset=%#x lit=%d\n",
                   inst.pc, inst.raw[0], s.op, s.sdst, s.sbase, s.imm ? 1 : 0,
                   s.offset, inst.has_literal ? 1 : 0);
    if (trace) {
      std::fprintf(stderr, "[eud] s_load x%u s%u <- [s%u=%#lx + %#lx] = %#lx\n",
                   dwords, s.sdst, base, static_cast<unsigned long>(table),
                   static_cast<unsigned long>(byte_off),
                   static_cast<unsigned long>(address));
      std::fprintf(stderr, "[eud]   data:");
      for (uint32_t i = 0; i < dwords; i++)
        std::fprintf(stderr, " %08x", mem[i]);
      std::fprintf(stderr, "\n");
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
  const std::vector<uint8_t> reachable = ComputeReachability(*program);
  e.info.plan = PlanMimgBindings(*program, reachable.data());
  uint32_t index = 0;
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
                                 const uint8_t* reachable) {
  MimgBindingPlan plan;
  // Track, per SGPR range, the index of the last SMRD instruction covering it.
  struct Load {
    uint32_t sgpr, dwords, index;
  };
  std::vector<Load> loads;
  const auto covering_load = [&](uint32_t sgpr, uint32_t dwords) -> uint32_t {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->index;
    return 0xFFFF;  // inline user data (no covering load)
  };

  std::unordered_map<uint64_t, uint32_t> binding_of;
  uint32_t inst_index = 0;
  for (const Inst& inst : program) {
    const uint32_t idx = inst_index++;
    if (reachable && !reachable[idx])
      continue;
    if (inst.enc == Enc::kSmrd) {
      const Smrd s = DecodeSmrd(inst.raw[0]);
      if (s.op <= 0x04) {  // s_load_dword..x16 can rewrite descriptor SGPRs
        const uint32_t dwords = 1u << s.op;
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
    const uint32_t w0 = inst.raw[0], w1 = inst.raw[1];
    const uint32_t op = (w0 >> 18) & 0x7F;
    const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
    const bool sampling = op >= 0x20;
    const bool storage = op == 0x08 || op == 0x09;
    const uint32_t ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFF;
    const uint32_t flags =
        (((w0 >> 14) & 1) << 0) |                      // DA
        (((op == 0x28 || op == 0x2f) ? 1 : 0) << 1) |  // dref
        ((op == 0x47 ? 1 : 0) << 2) |                  // gather4_lz
        (static_cast<uint32_t>(storage) << 3);
    const uint64_t key =
        MimgDescriptorKey(srsrc, ssamp, covering_load(srsrc, 8),
                          sampling ? covering_load(ssamp, 4) : 0xFFFE, flags);
    const auto [it, inserted] = binding_of.emplace(
        key, static_cast<uint32_t>(plan.binding_srsrc.size()));
    if (inserted) {
      plan.binding_srsrc.push_back(srsrc);
      plan.binding_storage.push_back(storage);
    }
    plan.binding_by_pc[inst.pc] = it->second;
  }
  return plan;
}

VBuffer DecodeVBuffer(const uint32_t* p) {
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
      .base = (static_cast<uint64_t>(p[1] & 0xFFF) << 32) | p[0],
      .stride = (p[1] >> 16) & 0x3FFF,
      .num_records = p[2],
      .dfmt = (p[3] >> 15) & 0xF,
      .nfmt = (p[3] >> 12) & 0x7,
  };
}

TImage DecodeTImage(const uint32_t* p) {
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
      std::all_of(p, p + 8, [](uint32_t word) { return word == 0; });
  t.base = ((static_cast<uint64_t>(p[1] & 0x3F) << 32) | p[0]) << 8;
  t.min_lod = (p[1] >> 8) & 0xFFF;
  t.dfmt = (p[1] >> 20) & 0x3F;
  t.nfmt = (p[1] >> 26) & 0xF;
  t.width = (p[2] & 0x3FFF) + 1;
  t.height = ((p[2] >> 14) & 0x3FFF) + 1;
  for (uint32_t i = 0; i < 4; i++)
    t.dst_sel[i] = (p[3] >> (i * 3)) & 0x7;
  t.base_mip = (p[3] >> 12) & 0xF;
  const uint32_t last_mip = (p[3] >> 16) & 0xF;
  t.mip_levels = last_mip + 1;
  t.view_mips = last_mip >= t.base_mip ? last_mip - t.base_mip + 1 : 0;
  t.tiling_idx = (p[3] >> 20) & 0x1F;
  t.pow2_pad = ((p[3] >> 25) & 1) != 0;
  t.type = p[3] >> 28;
  t.pitch = ((p[4] >> 13) & 0x3FFF) + 1;
  if (t.pitch < t.width)
    t.pitch = t.width;  // fall back to width if unset
  if (t.type == 13 || t.type == 12) {  // SQ_RSRC_IMG_2D_ARRAY / _1D_ARRAY
    t.layers = (p[4] & 0x1FFF) + 1;
    if (t.pow2_pad)
      t.layers = NextPow2(t.layers);
    t.base_array = p[5] & 0x1FFF;
    t.view_layers = 0;
    const uint32_t last_array = (p[5] >> 13) & 0x1FFF;
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
                              t.type == 12 || t.type == 13;
  const bool valid_view = (t.type != 13 && t.type != 12) ||
                          (t.base_array < t.layers && t.view_layers > 0);
  uint32_t max_levels = 1;
  for (uint32_t extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
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
                                        const uint32_t* vs_user_data) {
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
    const uint32_t base_sgpr = s.sbase * 2;  // user_data index of the table ptr
    if (base_sgpr + 1 >= 16)
      continue;
    const uint64_t table = UserDataPointer(vs_user_data, base_sgpr);
    if (!GuestRange(table, 16))
      continue;
    const SmrdOffset so = DecodeSmrdOffset(inst);
    if (so.in_sgpr)
      continue;  // a register offset needs the scalar walk, not this scan
    const uint32_t byte_off = so.dwords * 4;
    const VBuffer v =
        DecodeVBuffer(reinterpret_cast<const uint32_t*>(table + byte_off));
    if (v.base >= kGuestLo && v.base < kGuestHi && v.stride &&
        v.stride <= 256 && v.num_records && v.num_records <= 0x100000) {
      if (kTrace)
        std::fprintf(stderr,
                     "[gcnres] VB sbase=sgpr%u table=%#lx off=%u -> base=%#lx "
                     "stride=%u nrec=%u dfmt=%u nfmt=%u\n",
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
    const uint32_t* ps_user_data,
    bool trace,
    uint64_t code_base) {
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
    const uint32_t binding = plan_it->second;
    const uint32_t word1 = inst.raw[1];
    const uint32_t srsrc = ((word1 >> 16) & 0x1F) * 4;  // T# base SGPR
    const uint32_t op = (inst.raw[0] >> 18) & 0x7F;

    // Resolve the sampler for sampling ops (used both for new bindings and to
    // backfill a binding first seen through a non-sampling op like resinfo).
    uint32_t sampler[4] = {};
    bool sampler_ok = false;
    if (op >= 0x20) {
      const uint32_t ssamp = ((word1 >> 21) & 0x1F) * 4;
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
      const uint64_t at = eval.src[srsrc];
      if (at && probes < 24) {
        probes++;
        // Widened to +-16: the registers at the write say the arena stride is
        // 0x400000 and the bad pointer is base + 8 strides, i.e. 32 MiB out,
        // which a +-4 window stepping 2 MiB could never reach.
        for (int slot = -16; slot <= 16; slot++) {
          if (!slot)
            continue;
          const uint64_t probe = at + static_cast<int64_t>(slot) * 0x200000;
          if (!GuestRange(probe, 32))
            continue;
          const uint32_t* w = reinterpret_cast<const uint32_t*>(probe);
          if (std::all_of(w, w + 8, [](uint32_t v) { return v == 0; }))
            continue;
          const TImage probe_t = DecodeTImage(w);
          std::fprintf(stderr,
                       "[arena] null at %#lx: arena%+d (%#lx) holds %ux%u "
                       "valid=%d raw=%08x/%08x\n",
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
      const uint64_t root = UserDataPointer(ps_user_data, 0);
      if (!armed && root && GuestRange(root, 64)) {
        armed = true;
        std::fprintf(stderr,
                     "[nullwatch] arming on SRT %#lx (+0x18 held the pointer "
                     "into the empty arena; T# read at %#lx)\n",
                     static_cast<unsigned long>(root),
                     static_cast<unsigned long>(eval.src[srsrc]));
        // +0x18 is the slot the chain read the table pointer from.
        utl::setWriteWatchValueProbe(static_cast<uintptr_t>(root) + 0x18);
        utl::setWriteWatchChase(4);  // follow it back up to four copies
        if (!utl::armWriteWatch(static_cast<uintptr_t>(root), 64, 200))
          std::fprintf(stderr, "[nullwatch] no armer registered\n");
      }
    }
    if (kNullDis && t.null_descriptor) {
      static bool dumped = false;
      if (!dumped) {
        dumped = true;
        std::fprintf(stderr,
                     "[nulldis] PS %#lx binding %u read a null T# from %#lx "
                     "(SRT root %#lx)\n",
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
        std::fprintf(stderr,
                     "[tscan] triggered by null T# in PS %#lx binding %u, "
                     "descriptor read from %#lx\n",
                     static_cast<unsigned long>(code_base), binding,
                     static_cast<unsigned long>(eval.src[srsrc]));
        CensusBlock("descriptor read", eval.src[srsrc]);
        // One level up: the SRT block the draw's user data points at. If that
        // is empty too, the title never built the resource block at all and the
        // descriptor table below it is a red herring.
        const uint64_t srt = UserDataPointer(ps_user_data, 0);
        std::fprintf(stderr, "[tscan] SRT root (user s[0:1]) = %#lx\n",
                     static_cast<unsigned long>(srt));
        if (GuestRange(srt, 64)) {
          const uint32_t* p = reinterpret_cast<const uint32_t*>(srt);
          std::fprintf(stderr, "[tscan]   SRT[0..15]:");
          for (int i = 0; i < 16; i++)
            std::fprintf(stderr, " %08x", p[i]);
          std::fprintf(stderr, "\n");
          CensusBlock("SRT root", srt);
        } else {
          std::fprintf(stderr, "[tscan]   SRT root not mapped\n");
        }
        const uint64_t good = ScanForDescriptor(kTscan);
        if (good)
          CensusBlock("first valid copy", good);
      }
    }
    if (kArenaProbe && t.null_descriptor && eval.src[srsrc]) {
      const uint64_t at = eval.src[srsrc];
      for (int slot = -1; slot >= -kArenaProbe; slot--) {
        const uint64_t probe = at + static_cast<int64_t>(slot) * 0x200000;
        if (!GuestRange(probe, 32))
          continue;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(probe);
        if (std::all_of(w, w + 8, [](uint32_t v) { return v == 0; }))
          continue;
        const TImage cand = DecodeTImage(w);
        if (!cand.valid)
          continue;
        t = cand;
        t.src = at;
        static int announced = 0;
        if (announced < 8) {
          announced++;
          std::fprintf(stderr,
                       "[arena] substituted %#lx -> %#lx (arena%+d) %ux%u\n",
                       static_cast<unsigned long>(at),
                       static_cast<unsigned long>(probe), slot, t.width,
                       t.height);
          // Disassemble the shader that produced the biased pointer once: a
          // constant arena bias is most likely an address our linear scalar
          // replay computed down a path the real wave would not have taken.
          static bool dumped = false;
          if (!dumped) {
            dumped = true;
            std::fprintf(stderr, "[arena] SRT root = %#lx, T# read at %#lx\n",
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
      const uint64_t base = kSotcCompositeRt;
      const uint32_t descriptor[8] = {
          static_cast<uint32_t>(base >> 8),
          static_cast<uint32_t>((base >> 40) & 0x3f) | 0x1c400000,
          ((270 - 1) << 14) | (960 - 1),
          0x94000fac,
          (1024 - 1) << 13,
          0,
          0,
          0,
      };
      t = DecodeTImage(descriptor);
      t.src = eval.src[srsrc];
      const uint64_t descriptor_at = eval.src[srsrc];
      const uint64_t style_at = eval.src[16];
      if (GuestRange(style_at - 4, 12)) {
        auto* style = reinterpret_cast<uint32_t*>(style_at - 4);
        style[0] = 0x3f800000;  // outline threshold (disabled below)
        style[1] = 0x3f000000;  // SDF edge threshold
        style[2] = 0x42000000;  // atlas footprint scale
      }
      if (GuestRange(descriptor_at - 32, 20)) {
        auto* outline = reinterpret_cast<uint32_t*>(descriptor_at - 32);
        outline[0] = 0;
        outline[1] = 0;
        outline[2] = 0;
        outline[3] = 0x3f800000;
        outline[4] = 0;  // keep the optional outline branch disabled
      }
      static bool announced = false;
      if (!announced) {
        announced = true;
        std::fprintf(stderr,
                     "[sotc-composite] substituted null T# with %#lx 960x270 "
                     "valid=%d\n",
                     static_cast<unsigned long>(base), t.valid);
      }
    }
    if (eval.trace)
      std::fprintf(stderr,
                   "[eud] MIMG pc=%#x bind=%u srsrc=s%u known=%d at=%#lx "
                   "base=%#lx %ux%u valid=%d raw=%08x/%08x/%08x/%08x/%08x/%08x/"
                   "%08x/%08x\n",
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
    if (kTexSrc && t.base >= (uint64_t)kTexSrc) {
      static std::set<uint64_t> seen;
      if (seen.size() < 64 && seen.insert(t.base).second)
        std::fprintf(stderr, "[texsrc] base=%#lx T# at %#lx (sgpr%u)\n",
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
        static uint32_t hist[32] = {0};
        static uint64_t n = 0, pitch_ne = 0;
        hist[t.tiling_idx & 31]++;
        if (t.pitch != t.width)
          pitch_ne++;
        if ((++n % 4000) == 0) {
          std::fprintf(stderr, "[tilehist] n=%lu pitch!=width=%lu:",
                       static_cast<unsigned long>(n),
                       static_cast<unsigned long>(pitch_ne));
          for (int i = 0; i < 32; i++)
            if (hist[i])
              std::fprintf(stderr, " idx%d=%u", i, hist[i]);
          std::fprintf(stderr, "\n");
        }
      }
      if (kTrace)
        std::fprintf(stderr,
                     "[gcnres] T# (sgpr%u) base=%#lx %ux%u pitch=%u "
                     "dfmt=%u nfmt=%u tiling=%u\n",
                     srsrc, static_cast<unsigned long>(t.base), t.width,
                     t.height, t.pitch, t.dfmt, t.nfmt, t.tiling_idx);
    }
    result.push_back(t);
  }
  return result;
}

std::unordered_map<uint32_t, VBuffer> ResolveCbuffers(
    const std::shared_ptr<const Program>& program,
    const uint32_t* user_data) {
  std::unordered_map<uint32_t, VBuffer> result;
  if (!program || !user_data)
    return result;

  // Mirror TrackTextures: step the scalar register file, and at each
  // s_buffer_load read the live 4-dword V# out of the resolved SGPRs. FOX
  // passes cbuffer descriptors through extended user data too (s_load the V#
  // through an EUD pointer, then s_buffer_load through it), so reading the V#
  // straight from user data yields base=0. The recompiler assigns one binding
  // per base SGPR (PlanCbufs), so key by base SGPR and keep the first
  // resolvable V# seen for it.
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
    const uint32_t base = s.sbase * 2;
    // Keyed like the translator's bindings (see CbufBindKey): the same SGPR
    // can serve as a pointer pair for one load and a V# for another.
    const uint32_t key = base | (pointer ? 0x100u : 0u);
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
      std::fprintf(stderr, "[eud] cbuf s%u -> base=%#lx stride=%u nrec=%u\n",
                   base, static_cast<unsigned long>(v.base), v.stride,
                   v.num_records);
  }
  return result;
}

std::vector<VBuffer> ResolveDirectVertexBuffers(
    const std::shared_ptr<const Program>& program,
    const std::vector<ShaderAttr>& attrs,
    const uint32_t* user_data) {
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
        uint32_t data[3] = {};
        if (GuestRange(result[i].base, sizeof(data)))
          std::memcpy(data, reinterpret_cast<const void*>(result[i].base),
                      sizeof(data));
        std::fprintf(stderr,
                     "[eud] vattr%zu pc=%#x s%u -> base=%#lx stride=%u "
                     "nrec=%u fmt=%u/%u inst=%u/%u ioff=%u "
                     "V#=%08x/%08x/%08x/%08x data=%08x/%08x/%08x\n",
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
    const uint32_t* user_data) {
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
      if (eval.trace)
        std::fprintf(
            stderr,
            "[eud] rawbuf%zu pc=%#x s%u -> base=%#lx stride=%u "
            "nrec=%u V#=%08x/%08x/%08x/%08x\n",
            i, inst.pc, buffer.srsrc_sgpr,
            static_cast<unsigned long>(result[i].base), result[i].stride,
            result[i].num_records, eval.sgpr[buffer.srsrc_sgpr],
            eval.sgpr[buffer.srsrc_sgpr + 1], eval.sgpr[buffer.srsrc_sgpr + 2],
            eval.sgpr[buffer.srsrc_sgpr + 3]);
    }
  }
  return result;
}

std::vector<ResolvedCsResource> ResolveCsResources(const Program& program,
                                                   const RecompiledCs& plan,
                                                   const uint32_t* user_data) {
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
      const uint32_t dwords = resource.kind == 1   ? 8
                              : resource.kind == 2 ? 2
                                                   : 4;
      if (!eval.AllKnown(resource.base_sgpr, dwords))
        continue;
      ResolvedCsResource& resolved = result[resource.binding];
      std::memcpy(resolved.descriptor, &eval.sgpr[resource.base_sgpr],
                  dwords * sizeof(uint32_t));
      resolved.valid = true;
    }
    eval.Step(inst);
  }
  return result;
}

}  // namespace gpu::gcn
