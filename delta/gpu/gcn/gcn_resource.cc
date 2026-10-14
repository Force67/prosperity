/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/gcn/gcn_resource.h"
#include "base/arch.h"
#include "guest/session.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/guest_memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/time/time.h"
#include "host_memory/host_memory.h"
#include "options/options.h"
#include "write_watch/write_watch.h"

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
DELTA_OPTION(int, kArenaProbe, "DELTA_GPU_ARENA_PROBE", 0);
}  // namespace

namespace gpu::gcn {

void (*g_flush_guest_range)(u64 address, u64 bytes) = nullptr;
void (*g_overlay_guest_range)(u64 address, u32 dwords, u32* values) = nullptr;
u64 g_direct_table = 0, g_direct_misses = 0;

namespace {

constexpr u64 kGuestLo = 0x1000000000ull;
constexpr u64 kGuestHi = 0x20000000000ull;

bool GuestRange(u64 address, u64 size) {
  return size && address >= kGuestLo && address < kGuestHi &&
         size <= kGuestHi - address && IsReadableRangeCached(address, size);
}

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
                "valid={:d} "
                "raw={:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}",
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

void ScanForDwords(const char* spec) {
  base::Vector<u32> want;
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
      if (!all(z, zero))
        continue;
      const auto edges = [](const float* f) {
        for (u32 k = 0; k < 4; k++) {
          if (!(f[k] >= -1.f && f[k] <= 1.f))
            return false;
          if (f[k] != 0.f && (f[k] > -1e-3f && f[k] < 1e-3f))
            return false;
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

struct NullSite {
  u64 address;
  u64 code_base;
  u64 draw_seen;
  bool filled;
  u64 peak_watermark;
};
base::Vector<NullSite> g_null_sites;
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
    if (base::AllOf(p, p + 8, [](u32 w) { return w == 0; })) {
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
  BASE_LOGI("twatch", "{} of {} null sites later filled, {} still zero", filled,
            static_cast<unsigned>(g_null_sites.size()), still_zero);
}

struct Smrd {
  u32 op;
  u32 sdst;
  u32 sbase;
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

u64 MimgDescriptorKey(u32 srsrc,
                      u32 ssamp,
                      u32 load_rsrc,
                      u32 load_samp,
                      u32 flags) {
  return (static_cast<u64>(srsrc) << 0) | (static_cast<u64>(ssamp) << 8) |
         (static_cast<u64>(load_rsrc & 0xFFFF) << 16) |
         (static_cast<u64>(load_samp & 0xFFFF) << 32) |
         (static_cast<u64>(flags) << 48);
}

bool Sop1DestIs64(u32 op) {
  switch (op) {
    case 0x04:
    case 0x06:
    case 0x08:
    case 0x0a:
    case 0x0c:
    case 0x1c:
    case 0x1e:
    case 0x1f:
    case 0x21:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
    case 0x2d:
    case 0x2f:
    case 0x31:
      return true;
    default:
      return false;
  }
}

bool Sop2DestIs64(u32 op) {
  switch (op) {
    case 0x0b:
    case 0x0f:
    case 0x11:
    case 0x13:
    case 0x15:
    case 0x17:
    case 0x19:
    case 0x1b:
    case 0x1d:
    case 0x1f:
    case 0x21:
    case 0x23:
    case 0x25:
    case 0x29:
    case 0x2a:
      return true;
    default:
      return false;
  }
}

bool Vop3bWritesSdst(u32 op) {
  switch (op) {
    case 0x125:
    case 0x126:
    case 0x127:
    case 0x128:
    case 0x129:
    case 0x12a:
    case 0x16d:
    case 0x16e:
    case 0x176:
    case 0x177:
      return true;
    default:
      return false;
  }
}

struct ScalarEval {
  static constexpr u32 kRegs = 136;
  u32 sgpr[kRegs] = {};
  bool known[kRegs] = {};
  u64 src[kRegs] = {};
  bool trace = false;
  u64 code_base = 0;
  bool carry_known = false;
  u32 carry = 0;

  base::HashMap<u32, u32> lane_spill;
  base::HashMap<u32, u64> lane_spill_src;

  explicit ScalarEval(const u32* user_data, u64 base = 0) {
    for (u32 i = 0; i < 16; i++) {
      sgpr[i] = user_data[i];
      known[i] = true;
    }
    code_base = base;
    trace = kGpuEudtrace;
  }

  u64 Ptr(u32 s) const {
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
  bool Meet(const ScalarEval& other) {
    bool changed = false;
    for (u32 i = 0; i < kRegs; i++) {
      if (known[i] && (!other.known[i] || other.sgpr[i] != sgpr[i])) {
        known[i] = false;
        changed = true;
      }
      if (src[i] != other.src[i])
        src[i] = 0;
    }
    const auto differs = [](const auto& theirs) {
      return [&theirs](const auto& slot) {
        const auto it = theirs.find(slot.first);
        return it == theirs.end() || it->second != slot.second;
      };
    };
    if (carry_known && (!other.carry_known || other.carry != carry)) {
      carry_known = false;
      changed = true;
    }
    changed |= base::EraseIf(lane_spill, differs(other.lane_spill)) > 0;
    base::EraseIf(lane_spill_src, differs(other.lane_spill_src));
    return changed;
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

  bool StepLaneOp(const Inst& inst) {
    const bool vop1 = inst.enc == Enc::kVop1, vop2 = inst.enc == Enc::kVop2;
    const bool vop3 = inst.enc == Enc::kVop3;
    if (!vop1 && !vop2 && !vop3)
      return false;
    const u32 w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
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
        for (u32 i = 0; i < 64; i++)
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
    if (vop3 && (op < 0x100 || Vop3bWritesSdst(op))) {
      const u32 sdst = op < 0x100 ? dst : ((w >> 8) & 0x7F);
      Clear(sdst);
      Clear(sdst + 1);
      return true;
    }
    return false;
  }

  void Step(const Inst& inst) {
    const bool had_carry = carry_known;
    carry_known = false;
    if (inst.enc == Enc::kSop1) {
      const u32 w = inst.raw[0];
      const u32 sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (inst.opcode == 0x03) {
        u32 value;
        if (Source(ssrc0, inst.literal, value))
          Set(sdst, value);
        else
          Clear(sdst);
      } else if (inst.opcode == 0x04) {
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
      } else if (inst.opcode == 0x1f) {
        if (code_base) {
          const u64 pc = code_base + static_cast<u64>(inst.pc + inst.size) * 4;
          Set(sdst, static_cast<u32>(pc));
          Set(sdst + 1, static_cast<u32>(pc >> 32));
        } else {
          Clear(sdst);
          Clear(sdst + 1);
        }
      } else {
        Clear(sdst);
        if (Sop1DestIs64(inst.opcode))
          Clear(sdst + 1);
      }
      return;
    }
    if (inst.enc == Enc::kSopk) {
      const u32 w = inst.raw[0];
      const u32 sdst = (w >> 16) & 0x7F;
      const u32 simm = static_cast<u32>(static_cast<i32>(static_cast<i16>(w)));
      switch (inst.opcode) {
        case 0x00:
          Set(sdst, simm);
          break;
        case 0x0f:
          if (known[sdst])
            Set(sdst, sgpr[sdst] + simm);
          break;
        case 0x10:
          if (known[sdst])
            Set(sdst, sgpr[sdst] * simm);
          break;
        case 0x03:
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
        case 0x11:
        case 0x13:
        case 0x15:
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
              base::Min(static_cast<i32>(a), static_cast<i32>(b)));
          break;
        case 0x07:
          value = base::Min(a, b);
          break;
        case 0x08:
          value = static_cast<u32>(
              base::Max(static_cast<i32>(a), static_cast<i32>(b)));
          break;
        case 0x09:
          value = base::Max(a, b);
          break;
        case 0x27: {
          const u32 width = (b >> 16) & 0x7F;
          value = width >= 32 ? (a >> (b & 31))
                              : ((a >> (b & 31)) & ((1u << width) - 1));
          break;
        }
        case 0x00:
        case 0x04:
        case 0x01:
        case 0x05: {
          const bool with_carry = op >= 0x04;
          if (with_carry && !had_carry) {
            Clear(sdst);
            return;
          }
          const u64 in = with_carry ? carry : 0;
          const bool add = op == 0x00 || op == 0x04;
          const u64 wide = add ? u64(a) + b + in : u64(a) - b - in;
          value = static_cast<u32>(wide);
          carry = static_cast<u32>((wide >> 32) & 1);
          Set(sdst, value);
          carry_known = true;
          return;
        }
        case 0x02:
          value = a + b;
          break;
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
    const bool buffer_load = s.op >= 0x08 && s.op <= 0x0c;
    if (s.op > 0x04 && !buffer_load)
      return;
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
    for (u32 i = 0; i < dwords; i++)
      Clear(s.sdst + i);
    if (!base_known || !offset_known) {
      if (kGpuEudfail)
        BASE_LOGI("eudfail",
                  "s_load x{} s{} <- s{}: base_known={} off_known={}", dwords,
                  s.sdst, base, base_known, offset_known);
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
    if (g_flush_guest_range)
      g_flush_guest_range(address, static_cast<u64>(dwords) * 4);
    const u32* mem = reinterpret_cast<const u32*>(address);
    u32 overlaid[16];
    if (g_overlay_guest_range && dwords <= 16) {
      std::memcpy(overlaid, mem, dwords * 4);
      g_overlay_guest_range(address, dwords, overlaid);
      mem = overlaid;
    }
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

struct ReplayOp {
  u32 index = 0;
  u32 version = 0;
  bool capture_before = false;
  bool step = false;
  bool capture_after = false;
};

struct ScalarPassInfo {
  MimgBindingPlan plan;
  base::Vector<Inst> insts;
  base::Vector<ReplayOp> replay;
  base::Vector<ReplayOp> full_replay;
};

base::Vector<ReplayOp> PlanReplay(const base::Vector<Inst>& insts, bool prune);
DELTA_OPTION(bool, kReplayCheck, "DELTA_GPU_REPLAY_CHECK", false);

bool VectorTouchesScalarFile(const Inst& inst) {
  switch (inst.enc) {
    case Enc::kVop1:
      return inst.opcode == 0x02;
    case Enc::kVop2:
      return inst.opcode == 0x01 || inst.opcode == 0x02;
    case Enc::kVop3:
      return inst.opcode < 0x100 || inst.opcode == 0x101 ||
             inst.opcode == 0x102 || inst.opcode == 0x182 ||
             Vop3bWritesSdst(inst.opcode);
    default:
      return false;
  }
}

u64 g_scalar_info_gen = 0;

const ScalarPassInfo& CachedScalarInfo(
    const base::SharedPointer<const Program>& program) {
  struct Entry {
    base::SharedPointer<const Program> pin;
    ScalarPassInfo info;
  };
  static base::HashMap<const Program*, Entry> cache;
  static const guest::SessionReset reset_cache(
      [] { guest::ResetResource(cache); });
  struct Recent {
    const Program* program = nullptr;
    const ScalarPassInfo* info = nullptr;
  };
  static Recent recent[2];
  static u64 recent_gen = 0;
  if (recent_gen != g_scalar_info_gen) {
    recent_gen = g_scalar_info_gen;
    recent[0] = recent[1] = {};
  }
  for (const Recent& r : recent)
    if (r.program == program.get())
      return *r.info;
  auto it = cache.find(program.get());
  if (it != cache.end()) {
    recent[1] = recent[0];
    recent[0] = {program.get(), &it->second.info};
    return it->second.info;
  }
  g_scalar_info_gen++;
  if (cache.size() > 4096) {
    base::EraseIf(
        cache, [](const auto& kv) { return kv.second.pin.use_count() == 1; });
    if (cache.size() > 16384)
      cache.clear();
  }
  Entry e;
  e.pin = program;
  const base::Vector<u8> reachable = ComputeReachability(*program);
  e.info.plan = PlanMimgBindings(*program, reachable.data());
  u32 index = 0;
  for (const Inst& inst : *program) {
    if (!reachable[index++])
      continue;
    if (inst.enc == Enc::kSop1 || inst.enc == Enc::kSopk ||
        inst.enc == Enc::kSop2 || inst.enc == Enc::kSmrd ||
        inst.enc == Enc::kMimg || inst.enc == Enc::kMubuf ||
        inst.enc == Enc::kMtbuf || VectorTouchesScalarFile(inst))
      e.info.insts.push_back(inst);
  }
  e.info.replay = PlanReplay(e.info.insts, true);
  if (kReplayCheck)
    e.info.full_replay = PlanReplay(e.info.insts, false);
  return cache.emplace(program.get(), base::move(e)).first->second.info;
}

struct ScalarConsumer {
  u32 index = 0;
  u32 version = 0;
  u8 first[2] = {};
  u8 count[2] = {};
  u8 known[2] = {};
  u32 value[2][8] = {};
  u64 src = 0;
};

bool IsSmrdConsumer(const Inst& inst) {
  if (inst.enc != Enc::kSmrd)
    return false;
  const u32 op = DecodeSmrd(inst.raw[0]).op;
  return op <= 0x04 || (op >= 0x08 && op <= 0x0c);
}

bool IsResourceConsumer(const Inst& inst) {
  return inst.enc == Enc::kMimg || inst.enc == Enc::kMubuf ||
         inst.enc == Enc::kMtbuf;
}

class ConsumerRegs {
 public:
  struct Sgprs {
    const ConsumerRegs* regs;
    const u32& operator[](u32 s) const { return regs->Reg(s); }
  };
  struct Srcs {
    const ConsumerRegs* regs;
    u64 operator[](u32 s) const {
      if (s == regs->c_.first[0])
        return regs->c_.src;
      *regs->uncovered_ = true;
      return 0;
    }
  };

  ConsumerRegs(const ScalarConsumer& c, bool* uncovered)
      : c_(c), uncovered_(uncovered) {}

  bool AllKnown(u32 s, u32 n) const {
    for (u32 w = 0; w < 2; w++) {
      if (s < c_.first[w] || s + n > u32(c_.first[w]) + c_.count[w])
        continue;
      const u32 bits = ((1u << n) - 1) << (s - c_.first[w]);
      return (c_.known[w] & bits) == bits;
    }
    *uncovered_ = true;
    return false;
  }
  u64 Ptr(u32 s) const {
    return (static_cast<u64>(sgpr[s + 1] & 0xFFFF) << 32) | sgpr[s];
  }

  Sgprs sgpr{this};
  Srcs src{this};
  bool trace = false;

 private:
  const u32& Reg(u32 s) const {
    static const u32 kZero = 0;
    for (u32 w = 0; w < 2; w++)
      if (s >= c_.first[w] && s < u32(c_.first[w]) + c_.count[w])
        return c_.value[w][s - c_.first[w]];
    *uncovered_ = true;
    return kZero;
  }

  const ScalarConsumer& c_;
  bool* uncovered_;
};

void Capture(ScalarConsumer& c,
             u32 w,
             u32 first,
             u32 count,
             const ScalarEval& eval) {
  count = base::Min<u32>(
      count, ScalarEval::kRegs - base::Min<u32>(first, ScalarEval::kRegs));
  c.first[w] = static_cast<u8>(first);
  c.count[w] = static_cast<u8>(count);
  for (u32 i = 0; i < count; i++) {
    if (eval.known[first + i]) {
      c.value[w][i] = eval.sgpr[first + i];
      c.known[w] |= static_cast<u8>(1u << i);
    }
  }
}

u32 ResourceSgpr(const Inst& inst) {
  return ((inst.raw[1] >> 16) & 0x1F) * 4;
}

u32 SamplerSgpr(const Inst& inst) {
  return ((inst.raw[1] >> 21) & 0x1F) * 4;
}

struct SgprSet {
  static constexpr u32 kWords = (ScalarEval::kRegs + 63) / 64;
  u64 words[kWords] = {};

  void Set(u32 i) { words[i / 64] |= 1ull << (i % 64); }
  bool Intersects(const SgprSet& other) const {
    for (u32 i = 0; i < kWords; i++)
      if (words[i] & other.words[i])
        return true;
    return false;
  }
  void Kill(const SgprSet& other) {
    for (u32 i = 0; i < kWords; i++)
      words[i] &= ~other.words[i];
  }
  void Add(const SgprSet& other) {
    for (u32 i = 0; i < kWords; i++)
      words[i] |= other.words[i];
  }
};

void AddRange(SgprSet& set, u32 first, u32 count) {
  for (u32 i = first; i < first + count && i < ScalarEval::kRegs; i++)
    set.Set(i);
}

void AddSource(SgprSet& set, u32 field) {
  if (field <= 127)
    set.Set(field);
}

struct StepEffects {
  SgprSet reads, writes, sources;
  bool keep = false;
};

StepEffects EffectsOf(const Inst& inst) {
  StepEffects e;
  const u32 w = inst.raw[0], op = inst.opcode;
  switch (inst.enc) {
    case Enc::kSop1: {
      const u32 sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (op == 0x03 || op == 0x04)
        AddSource(e.reads, ssrc0);
      if (op == 0x04 && ssrc0 <= 126)
        e.reads.Set(ssrc0 + 1);
      AddRange(e.writes, sdst, Sop1DestIs64(op) ? 2 : 1);
      return e;
    }
    case Enc::kSopk: {
      const u32 sdst = (w >> 16) & 0x7F;
      if ((op >= 0x03 && op <= 0x0e) || op == 0x11 || op == 0x13 || op == 0x15)
        return e;
      if (op == 0x0f || op == 0x10)
        e.reads.Set(sdst);
      e.writes.Set(sdst);
      return e;
    }
    case Enc::kSop2:
      AddSource(e.reads, w & 0xFF);
      AddSource(e.reads, (w >> 8) & 0xFF);
      AddRange(e.writes, (w >> 16) & 0x7F, Sop2DestIs64(op) ? 2 : 1);
      return e;
    case Enc::kVop1:
    case Enc::kVop2:
    case Enc::kVop3: {
      const bool vop1 = inst.enc == Enc::kVop1, vop2 = inst.enc == Enc::kVop2;
      const bool vop3 = inst.enc == Enc::kVop3;
      const u32 w1 = inst.raw[1];
      const u32 dst = vop3 ? (w & 0xFF) : ((w >> 17) & 0xFF);
      const u32 src0 = vop3 ? (w1 & 0x1FF) : (w & 0x1FF);
      const u32 src1 = vop3 ? ((w1 >> 9) & 0x1FF) : ((w >> 9) & 0xFF);
      if ((vop2 && op == 0x02) || (vop3 && op == 0x102)) {
        AddSource(e.reads, src0);
        AddSource(e.reads, src1);
        e.keep = true;
      } else if ((vop2 && op == 0x01) || (vop3 && op == 0x101) ||
                 (vop1 && op == 0x02) || (vop3 && op == 0x182)) {
        if (!(vop1 || op == 0x182))
          AddSource(e.reads, src1);
        AddRange(e.writes, dst, 1);
        AddRange(e.sources, dst, 1);
      } else if (vop3 && (op < 0x100 || Vop3bWritesSdst(op))) {
        AddRange(e.writes, op < 0x100 ? dst : ((w >> 8) & 0x7F), 2);
      }
      return e;
    }
    case Enc::kSmrd: {
      const Smrd s = DecodeSmrd(w);
      const bool buffer_load = s.op >= 0x08 && s.op <= 0x0c;
      if (s.op > 0x04 && !buffer_load)
        return e;
      AddRange(e.reads, s.sbase * 2, buffer_load ? 4 : 2);
      const SmrdOffset so = DecodeSmrdOffset(inst);
      if (so.in_sgpr && so.sgpr < ScalarEval::kRegs)
        e.reads.Set(so.sgpr);
      const u32 dwords = buffer_load ? (1u << (s.op - 0x08)) : (1u << s.op);
      AddRange(e.writes, s.sdst, dwords);
      AddRange(e.sources, s.sdst, dwords);
      return e;
    }
    default:
      return e;
  }
}

base::Vector<ReplayOp> PlanReplay(const base::Vector<Inst>& insts, bool prune) {
  base::Vector<ReplayOp> ops(insts.size());
  DescriptorVersions versions;
  for (u32 i = 0; i < insts.size(); i++) {
    const Inst& inst = insts[i];
    ops[i].index = i;
    if (IsSmrdConsumer(inst)) {
      const Smrd sm = DecodeSmrd(inst.raw[0]);
      ops[i].capture_before = true;
      ops[i].version = versions.Of(sm.sbase * 2, sm.op <= 0x04 ? 2 : 4);
    }
    versions.Note(inst);
    ops[i].capture_after = IsResourceConsumer(inst);
  }
  SgprSet live, live_sources;
  for (u32 i = static_cast<u32>(insts.size()); i-- > 0;) {
    const Inst& inst = insts[i];
    ReplayOp& op = ops[i];
    if (op.capture_after) {
      const u32 srsrc = ResourceSgpr(inst);
      AddRange(live, srsrc, inst.enc == Enc::kMimg ? 8 : 4);
      if (inst.enc == Enc::kMimg)
        AddRange(live, SamplerSgpr(inst), 4);
      AddRange(live_sources, srsrc, 1);
    }
    const StepEffects e = EffectsOf(inst);
    op.step = !prune || e.keep || e.writes.Intersects(live) ||
              e.sources.Intersects(live_sources);
    if (op.step) {
      live.Kill(e.writes);
      live.Add(e.reads);
    }
    if (op.capture_before)
      AddRange(live, DecodeSmrd(inst.raw[0]).sbase * 2, 4);
  }
  base::EraseIf(ops, [](const ReplayOp& op) {
    return !op.step && !op.capture_before && !op.capture_after;
  });
  return ops;
}

void BuildReplay(const base::Vector<Inst>& insts,
                 const base::Vector<ReplayOp>& plan,
                 const u32* user_data,
                 u64 code_base,
                 base::Vector<ScalarConsumer>& out) {
  out.clear();
  ScalarEval eval(user_data, code_base);
  for (const ReplayOp& op : plan) {
    const Inst& inst = insts[op.index];
    if (op.capture_before) {
      ScalarConsumer& c = out.emplace_back();
      c.index = op.index;
      c.version = op.version;
      Capture(c, 0, DecodeSmrd(inst.raw[0]).sbase * 2, 4, eval);
    }
    if (op.step)
      eval.Step(inst);
    if (!op.capture_after)
      continue;
    ScalarConsumer& c = out.emplace_back();
    c.index = op.index;
    const u32 srsrc = ResourceSgpr(inst);
    Capture(c, 0, srsrc, inst.enc == Enc::kMimg ? 8 : 4, eval);
    if (inst.enc == Enc::kMimg)
      Capture(c, 1, SamplerSgpr(inst), 4, eval);
    c.src = srsrc < ScalarEval::kRegs ? eval.src[srsrc] : 0;
  }
}

bool SameConsumer(const ScalarConsumer& a, const ScalarConsumer& b) {
  if (a.index != b.index || a.version != b.version || a.src != b.src)
    return false;
  for (u32 w = 0; w < 2; w++)
    if (a.first[w] != b.first[w] || a.count[w] != b.count[w] ||
        a.known[w] != b.known[w] ||
        std::memcmp(a.value[w], b.value[w], sizeof(a.value[w])))
      return false;
  return true;
}

void CheckReplay(const ScalarPassInfo& info,
                 const u32* user_data,
                 u64 code_base,
                 const base::Vector<ScalarConsumer>& pruned) {
  thread_local base::Vector<ScalarConsumer> full;
  BuildReplay(info.insts, info.full_replay, user_data, code_base, full);
  bool same = full.size() == pruned.size();
  for (u32 i = 0; same && i < full.size(); i++)
    same = SameConsumer(full[i], pruned[i]);
  static u64 checked = 0, mismatched = 0;
  checked++;
  if (!same && mismatched++ < 16)
    BASE_LOGW("replay",
              "pruned replay differs from the full one ({} vs {} "
              "consumers, {} of {} steps kept)",
              pruned.size(), full.size(), info.replay.size(),
              info.full_replay.size());
  if (checked % 100000 == 0)
    BASE_LOGI("replay", "checked {} replays, {} mismatched", checked,
              mismatched);
}

struct SharedReplays {
  struct Entry {
    base::SharedPointer<const Program> program;
    u64 code_base = 0;
    u32 user_data[16] = {};
    base::Vector<ScalarConsumer> consumers;
  };
  u32 depth = 0;
  u32 used = 0;
  base::Vector<Entry> entries;
};
thread_local SharedReplays g_replays;
DELTA_OPTION(bool, kSharedReplay, "DELTA_GPU_SHARED_REPLAY", true);

const base::Vector<ScalarConsumer>* SharedReplay(
    const base::SharedPointer<const Program>& program,
    const u32* user_data,
    u64 code_base) {
  if (!g_replays.depth)
    return nullptr;
  for (u32 i = 0; i < g_replays.used; i++) {
    const auto& e = g_replays.entries[i];
    if (e.program == program && e.code_base == code_base &&
        !std::memcmp(e.user_data, user_data, sizeof(e.user_data)))
      return &e.consumers;
  }
  if (g_replays.used == g_replays.entries.size())
    g_replays.entries.emplace_back();
  auto& e = g_replays.entries[g_replays.used++];
  e.program = program;
  e.code_base = code_base;
  std::memcpy(e.user_data, user_data, sizeof(e.user_data));
  const ScalarPassInfo& info = CachedScalarInfo(program);
  BuildReplay(info.insts, info.replay, user_data, code_base, e.consumers);
  if (kReplayCheck)
    CheckReplay(info, user_data, code_base, e.consumers);
  return &e.consumers;
}

template <typename Fn>
bool ForEachConsumer(const base::SharedPointer<const Program>& program,
                     const u32* user_data,
                     u64 code_base,
                     bool own,
                     bool trace,
                     Fn&& consume) {
  const auto* replay = own || trace || kGpuEudtrace || !kSharedReplay
                           ? nullptr
                           : SharedReplay(program, user_data, code_base);
  const auto& insts = CachedScalarInfo(program).insts;
  if (replay) {
    bool uncovered = false;
    for (const ScalarConsumer& c : *replay) {
      ConsumerRegs regs(c, &uncovered);
      consume(insts[c.index], regs, c.version);
      if (uncovered) {
        static int logged = 0;
        if (logged++ < 4)
          BASE_LOGI("replay", "consumer at pc={:#x} read past its snapshot",
                    insts[c.index].pc);
        return false;
      }
    }
    return true;
  }
  ScalarEval eval(user_data, code_base);
  eval.trace |= trace;
  DescriptorVersions versions;
  for (const Inst& inst : insts) {
    if (IsSmrdConsumer(inst)) {
      const Smrd sm = DecodeSmrd(inst.raw[0]);
      const u32 version = versions.Of(sm.sbase * 2, sm.op <= 0x04 ? 2 : 4);
      versions.Note(inst);
      consume(inst, eval, version);
      eval.Step(inst);
      continue;
    }
    versions.Note(inst);
    eval.Step(inst);
    if (IsResourceConsumer(inst))
      consume(inst, eval, 0u);
  }
  return true;
}

}  // namespace

ScalarReplayScope::ScalarReplayScope() {
  if (!g_replays.depth++)
    g_replays.used = 0;
}

ScalarReplayScope::~ScalarReplayScope() {
  g_replays.depth--;
}

MimgBindingPlan PlanMimgBindings(const Program& program, const u8* reachable) {
  MimgBindingPlan plan;
  struct Load {
    u32 sgpr, dwords, index;
  };
  base::Vector<Load> loads;
  const auto covering_load = [&](u32 sgpr, u32 dwords) -> u32 {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->index;
    return 0xFFFF;
  };

  base::HashMap<u64, u32> binding_of;
  u32 inst_index = 0;
  for (const Inst& inst : program) {
    const u32 idx = inst_index++;
    if (reachable && !reachable[idx])
      continue;
    if (inst.enc == Enc::kSmrd) {
      const Smrd s = DecodeSmrd(inst.raw[0]);
      if (s.op <= 0x04) {
        const u32 dwords = 1u << s.op;
        loads.erase(base::RemoveIf(loads.begin(), loads.end(),
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
        (((w0 >> 14) & 1) << 0) | (((op == 0x28 || op == 0x2f) ? 1 : 0) << 1) |
        ((op == 0x47 ? 1 : 0) << 2) | (static_cast<u32>(storage) << 3);
    const u64 key =
        MimgDescriptorKey(srsrc, ssamp, covering_load(srsrc, 8),
                          sampling ? covering_load(ssamp, 4) : 0xFFFE, flags);
    const auto [it, inserted] =
        binding_of.emplace(key, static_cast<u32>(plan.binding_srsrc.size()));
    if (inserted) {
      plan.binding_srsrc.push_back(srsrc);
      plan.binding_storage.push_back(storage);
    }
    plan.binding_by_pc[inst.pc] = it->second;
  }
  return plan;
}

VBuffer DecodeVBuffer(const u32* p) {
  return {
      .base = (static_cast<u64>(p[1] & 0xFFF) << 32) | p[0],
      .stride = (p[1] >> 16) & 0x3FFF,
      .num_records = p[2],
      .dfmt = (p[3] >> 15) & 0xF,
      .nfmt = (p[3] >> 12) & 0x7,
  };
}

TImage DecodeTImage(const u32* p) {
  TImage t;
  t.null_descriptor = base::AllOf(p, p + 8, [](u32 word) { return word == 0; });
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
    t.pitch = t.width;
  if (t.type == 13 || t.type == 12 || t.type == 11) {
    t.layers = (p[4] & 0x1FFF) + 1;
    if (t.type == 11)
      t.layers *= 6;
    if (t.pow2_pad)
      t.layers = NextPow2(t.layers);
    t.base_array = p[5] & 0x1FFF;
    t.view_layers = 0;
    const u32 last_array = (p[5] >> 13) & 0x1FFF;
    if (t.base_array < t.layers && last_array >= t.base_array)
      t.view_layers = base::Min(last_array, t.layers - 1) - t.base_array + 1;
  }
  if (t.type == 8 || t.type == 12) {
    t.is_1d = true;
    t.height = 1;
  }
  if (t.type == 10) {
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
  for (u32 extent = base::Max(t.width, t.height); extent > 1; extent >>= 1)
    max_levels++;
  const bool valid_mips = t.view_mips && t.mip_levels <= max_levels;
  const bool valid_3d = !t.is_3d || (t.depth <= 2048 && t.mip_levels == 1);
  t.valid = GuestRange(t.base, 1) && supported_type && t.width <= 8192 &&
            t.height <= 8192 && t.layers <= 8192 && valid_view && valid_mips &&
            valid_3d;
  return t;
}

void TrackVertexBuffers(base::Vector<VBuffer>& result,
                        const Program& fetch_program,
                        const u32* vs_user_data) {
  result.clear();
  if (!vs_user_data)
    return;

  for (const Inst& inst : fetch_program) {
    if (inst.enc != Enc::kSmrd)
      continue;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    if (s.op != 0x02)
      continue;
    const u32 base_sgpr = s.sbase * 2;
    if (base_sgpr + 1 >= 16)
      continue;
    const u64 table = UserDataPointer(vs_user_data, base_sgpr);
    if (!GuestRange(table, 16))
      continue;
    const SmrdOffset so = DecodeSmrdOffset(inst);
    if (so.in_sgpr)
      continue;
    const u32 byte_off = so.dwords * 4;
    if (!IsReadableRangeCached(table + byte_off, 16))
      continue;
    const VBuffer v =
        DecodeVBuffer(reinterpret_cast<const u32*>(table + byte_off));
    if (v.base >= kGuestLo && v.base < kGuestHi && v.stride &&
        v.stride <= 256 && v.num_records && v.num_records <= 0x100000) {
      if (kTrace)
        BASE_LOGI("gcnres",
                  "VB sbase=sgpr{} table={:#x} off={} -> base={:#x} "
                  "stride={} nrec={} dfmt={} nfmt={}",
                  base_sgpr, static_cast<unsigned long>(table), byte_off,
                  static_cast<unsigned long>(v.base), v.stride, v.num_records,
                  v.dfmt, v.nfmt);
      result.push_back(v);
    }
  }
  return;
}

void TrackTextures(base::Vector<TImage>& result,
                   const base::SharedPointer<const Program>& ps_program,
                   const u32* ps_user_data,
                   bool trace,
                   u64 code_base) {
  result.clear();
  if (!ps_program || !ps_user_data)
    return;

  if (kTwatch && ++g_track_draws % 4000 == 0)
    PollNullDescriptors();

  const ScalarPassInfo& cached = CachedScalarInfo(ps_program);
  const MimgBindingPlan& plan = cached.plan;

  const bool own = trace || kGpuEudtrace || kTwatch || kNullWatch || kNullDis ||
                   kTscan || kArenaProbe || kSotcCompositeRt || kTexSrc;
  const auto consume = [&](const Inst& inst, auto& eval, u32) {
    if (inst.enc != Enc::kMimg)
      return;
    const auto plan_it = plan.binding_by_pc.find(inst.pc);
    if (plan_it == plan.binding_by_pc.end())
      return;
    const u32 binding = plan_it->second;
    const u32 word1 = inst.raw[1];
    const u32 srsrc = ((word1 >> 16) & 0x1F) * 4;
    const u32 op = (inst.raw[0] >> 18) & 0x7F;

    u32 sampler[4] = {};
    bool sampler_ok = false;
    if (op >= 0x20) {
      const u32 ssamp = ((word1 >> 21) & 0x1F) * 4;
      if (eval.AllKnown(ssamp, 4)) {
        std::memcpy(sampler, &eval.sgpr[ssamp], sizeof(sampler));
        sampler_ok = true;
      }
    }

    if (binding < result.size()) {
      TImage& entry = result[binding];
      if (sampler_ok && !entry.sampler_valid) {
        std::memcpy(entry.sampler, sampler, sizeof(entry.sampler));
        entry.sampler_valid = true;
      }
      return;
    }

    TImage t;
    const bool image_ok = eval.AllKnown(srsrc, 8);
    if (image_ok) {
      t = DecodeTImage(&eval.sgpr[srsrc]);
      t.src = eval.src[srsrc];
    }
    if (kTwatch && t.null_descriptor) {
      NoteNullDescriptor(eval.src[srsrc], code_base);
      static int probes = 0;
      const u64 at = eval.src[srsrc];
      if (at && probes < 24) {
        probes++;
        for (int slot = -16; slot <= 16; slot++) {
          if (!slot)
            continue;
          const u64 probe = at + static_cast<i64>(slot) * 0x200000;
          if (!GuestRange(probe, 32))
            continue;
          const u32* w = reinterpret_cast<const u32*>(probe);
          if (base::AllOf(w, w + 8, [](u32 v) { return v == 0; }))
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
        write_watch::SetValueProbe(static_cast<uintptr_t>(root) + 0x18);
        write_watch::SetChase(4);
        if (!write_watch::Arm(static_cast<uintptr_t>(root), 64, 200))
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
                  static_cast<unsigned long>(UserDataPointer(ps_user_data, 0)));
        DisassembleAt(code_base, "nulldis.PS");
      }
    }
    if (kTscan && t.null_descriptor) {
      static bool scanned = false;
      static const auto kScanStart = base::TimeTicks::Now();
      const bool due =
          (base::TimeTicks::Now() - kScanStart).InSeconds() >= kTscanAfter;
      if (!scanned && due) {
        scanned = true;
        BASE_LOGI("tscan",
                  "triggered by null T# in PS {:#x} binding {}, "
                  "descriptor read from {:#x}",
                  static_cast<unsigned long>(code_base), binding,
                  static_cast<unsigned long>(eval.src[srsrc]));
        CensusBlock("descriptor read", eval.src[srsrc]);
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
        if (base::AllOf(w, w + 8, [](u32 v) { return v == 0; }))
          continue;
        const TImage cand = DecodeTImage(w);
        if (!cand.valid)
          continue;
        t = cand;
        t.src = at;
        static int announced = 0;
        if (announced < 8) {
          announced++;
          BASE_LOGI("arena", "substituted {:#x} -> {:#x} (arena{:+d}) {}x{}",
                    static_cast<unsigned long>(at),
                    static_cast<unsigned long>(probe), slot, t.width, t.height);
          static bool dumped = false;
          if (!dumped) {
            dumped = true;
            BASE_LOGI(
                "arena", "SRT root = {:#x}, T# read at {:#x}",
                static_cast<unsigned long>(UserDataPointer(ps_user_data, 0)),
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
        style[0] = 0x3f800000;
        style[1] = 0x3f000000;
        style[2] = 0x42000000;
      }
      if (GuestRange(descriptor_at - 32, 20)) {
        auto* outline = reinterpret_cast<u32*>(descriptor_at - 32);
        outline[0] = 0;
        outline[1] = 0;
        outline[2] = 0;
        outline[3] = 0x3f800000;
        outline[4] = 0;
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
                "base={:#x} {}x{} valid={:d} "
                "raw={:08x}/{:08x}/{:08x}/{:08x}/{:08x}/{:08x}/"
                "{:08x}/{:08x}",
                inst.pc, binding, srsrc, image_ok,
                static_cast<unsigned long>(eval.src[srsrc]),
                static_cast<unsigned long>(t.base), t.width, t.height, t.valid,
                image_ok ? eval.sgpr[srsrc + 0] : 0,
                image_ok ? eval.sgpr[srsrc + 1] : 0,
                image_ok ? eval.sgpr[srsrc + 2] : 0,
                image_ok ? eval.sgpr[srsrc + 3] : 0,
                image_ok ? eval.sgpr[srsrc + 4] : 0,
                image_ok ? eval.sgpr[srsrc + 5] : 0,
                image_ok ? eval.sgpr[srsrc + 6] : 0,
                image_ok ? eval.sgpr[srsrc + 7] : 0);
    if (kTexSrc && t.base >= (u64)kTexSrc) {
      static base::Set<u64> seen;
      static const guest::SessionReset reset_seen(
          [] { guest::ResetResource(seen); });
      if (seen.size() < 64 && seen.insert(t.base).second)
        BASE_LOGI("texsrc", "base={:#x} T# at {:#x} (sgpr{})",
                  static_cast<unsigned long>(t.base),
                  static_cast<unsigned long>(eval.src[srsrc]), srsrc);
    }
    if (sampler_ok) {
      std::memcpy(t.sampler, sampler, sizeof(t.sampler));
      t.sampler_valid = true;
    }
    t.arrayed = (inst.raw[0] & 0x4000) != 0 || t.type == 11;
    t.force_lod_zero = op == 0x47;
    t.depth_compare = op == 0x28 || op == 0x2f;
    t.storage = op == 0x08 || op == 0x09;
    if (t.valid) {
      if (kGpuTilehist) {
        static u32 hist[32] = {0};
        static u64 n = 0, pitch_ne = 0;
        hist[t.tiling_idx & 31]++;
        if (t.pitch != t.width)
          pitch_ne++;
        if ((++n % 4000) == 0) {
          base::String line;
          base::FormatTo(line,
                         "n={} pitch!=width={}:", static_cast<unsigned long>(n),
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
                  srsrc, static_cast<unsigned long>(t.base), t.width, t.height,
                  t.pitch, t.dfmt, t.nfmt, t.tiling_idx);
    }
    result.push_back(t);
  };
  if (!ForEachConsumer(ps_program, ps_user_data, code_base, own, trace,
                       consume)) {
    result.clear();
    ForEachConsumer(ps_program, ps_user_data, code_base, true, trace, consume);
  }
  return;
}

void ResolveCbuffers(CbufList& result,
                     const base::SharedPointer<const Program>& program,
                     const u32* user_data,
                     u64 code_base) {
  result.clear();
  if (!program || !user_data)
    return;

  if ((kMemFind && *kMemFind) || kSoaScan) {
    static const auto kEpoch = base::TimeTicks::Now();
    static bool swept = false;
    if (!swept && (base::TimeTicks::Now() - kEpoch).InSecondsF() > 60.0) {
      swept = true;
      if (kMemFind && *kMemFind)
        ScanForDwords(kMemFind);
      if (kSoaScan)
        ScanForQuadColumns(kSoaScan.get());
    }
  }
  const auto consume = [&](const Inst& inst, auto& eval, u32 version) {
    if (inst.enc != Enc::kSmrd)
      return;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    const bool pointer = s.op <= 0x04;
    const u32 base = s.sbase * 2;
    const u64 key = CbufKey(base, pointer, version);
    if (FindCbuf(result, key) || !eval.AllKnown(base, pointer ? 2 : 4))
      return;
    VBuffer v{};
    if (pointer)
      v.base = eval.Ptr(base);
    else
      v = DecodeVBuffer(&eval.sgpr[base]);
    result.emplace_back(key, v);
    if (eval.trace)
      BASE_LOGI("eud", "cbuf s{} -> base={:#x} stride={} nrec={}", base,
                static_cast<unsigned long>(v.base), v.stride, v.num_records);
  };
  if (!ForEachConsumer(program, user_data, code_base, false, false, consume)) {
    result.clear();
    ForEachConsumer(program, user_data, code_base, true, false, consume);
  }
  return;
}

void ResolveDirectVertexBuffers(
    base::Vector<VBuffer>& result,
    const base::SharedPointer<const Program>& program,
    const base::Vector<ShaderAttr>& attrs,
    const u32* user_data,
    u64 code_base) {
  result.assign(attrs.size(), VBuffer{});
  if (!program || !user_data ||
      base::NoneOf(attrs.begin(), attrs.end(),
                   [](const ShaderAttr& attr) { return attr.direct_fetch; }))
    return;

  const auto consume = [&](const Inst& inst, auto& eval, u32) {
    if (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf)
      return;
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
                  static_cast<unsigned long>(result[i].base), result[i].stride,
                  result[i].num_records, result[i].dfmt, result[i].nfmt,
                  attr.inst_dfmt, attr.inst_nfmt, inst.raw[0] & 0xFFF,
                  eval.sgpr[attr.table_sgpr], eval.sgpr[attr.table_sgpr + 1],
                  eval.sgpr[attr.table_sgpr + 2],
                  eval.sgpr[attr.table_sgpr + 3], data[0], data[1], data[2]);
      }
    }
  };
  if (!ForEachConsumer(program, user_data, code_base, false, false, consume)) {
    base::Fill(result.begin(), result.end(), VBuffer{});
    ForEachConsumer(program, user_data, code_base, true, false, consume);
  }
  return;
}

void ResolveShaderBuffers(base::Vector<VBuffer>& result,
                          const base::SharedPointer<const Program>& program,
                          const base::Vector<ShaderBuffer>& buffers,
                          const u32* user_data,
                          u64 code_base) {
  result.assign(buffers.size(), VBuffer{});
  if (!program || !user_data || buffers.empty())
    return;

  const auto consume = [&](const Inst& inst, auto& eval, u32) {
    if (inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf)
      return;
    for (size_t i = 0; i < buffers.size(); i++) {
      const ShaderBuffer& buffer = buffers[i];
      if (buffer.use_pc != inst.pc || !eval.AllKnown(buffer.srsrc_sgpr, 4))
        continue;
      result[i] = DecodeVBuffer(&eval.sgpr[buffer.srsrc_sgpr]);
      if (eval.trace) {
        const u64 wide =
            (static_cast<u64>(eval.sgpr[buffer.srsrc_sgpr + 1] & 0xFFFF)
             << 32) |
            eval.sgpr[buffer.srsrc_sgpr];
        BASE_LOGI("eud", "rawbuf{} 48-bit base would be {:#x}, mapped={}", i,
                  static_cast<unsigned long>(wide), (int)GuestRange(wide, 16));
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
  };
  if (!ForEachConsumer(program, user_data, code_base, false, false, consume)) {
    base::Fill(result.begin(), result.end(), VBuffer{});
    ForEachConsumer(program, user_data, code_base, true, false, consume);
  }
  return;
}

namespace {

void ResolveAlongCfg(base::Vector<ResolvedCsResource>& result,
                     const Program& program,
                     const RecompiledCs& plan,
                     const u32* user_data,
                     u64 code_base) {
  const u32 n = static_cast<u32>(program.size());
  if (!n)
    return;
  base::HashMap<u32, u32> index_of;
  for (u32 i = 0; i < n; i++)
    index_of[program[i].pc] = i;

  constexpr u32 kNone = ~0u, kUnknown = ~1u;
  base::Vector<u32> target(n, kNone);
  base::Vector<bool> leader(n, false), falls_through(n, true);
  leader[0] = true;
  for (u32 i = 0; i < n; i++) {
    const Inst& inst = program[i];
    const u32 op = inst.opcode;
    if ((inst.enc == Enc::kSop1 && (op == 0x20 || op == 0x21 || op == 0x32)) ||
        (inst.enc == Enc::kSop2 && op == 0x2b) ||
        (inst.enc == Enc::kSopk && op == 0x11)) {
      target[i] = kUnknown;
      if (i + 1 < n)
        leader[i + 1] = true;
      continue;
    }
    if (inst.enc != Enc::kSopp)
      continue;
    const bool jump = op == 0x02;
    const bool cond = (op >= 0x04 && op <= 0x09) || (op >= 0x17 && op <= 0x1a);
    if (op == 0x01 || jump)
      falls_through[i] = false;
    if (jump || cond) {
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      const auto it = index_of.find(
          static_cast<u32>(static_cast<i32>(inst.pc + inst.size) + simm));
      target[i] = it == index_of.end() ? kUnknown : it->second;
      if (it != index_of.end())
        leader[it->second] = true;
    }
    if ((op == 0x01 || jump || cond) && i + 1 < n)
      leader[i + 1] = true;
  }

  base::Vector<u32> block_of(n, 0), starts;
  for (u32 i = 0; i < n; i++) {
    if (leader[i])
      starts.push_back(i);
    block_of[i] = static_cast<u32>(starts.size() - 1);
  }
  const u32 blocks = static_cast<u32>(starts.size());
  base::Vector<ScalarEval> in(blocks, ScalarEval(user_data, code_base));
  base::Vector<bool> reached(blocks, false), queued(blocks, false);
  base::Vector<u32> work;
  reached[0] = queued[0] = true;
  work.push_back(0);
  for (u32 budget = blocks * 16; !work.empty() && budget; budget--) {
    const u32 b = work.back();
    work.pop_back();
    queued[b] = false;
    ScalarEval eval = in[b];
    const u32 end = b + 1 < blocks ? starts[b + 1] : n;
    for (u32 i = starts[b]; i < end; i++)
      eval.Step(program[i]);
    const u32 last = end - 1;
    if (target[last] == kUnknown)
      return;
    u32 succ[2] = {kNone, kNone};
    if (target[last] != kNone)
      succ[0] = block_of[target[last]];
    if (falls_through[last] && end < n)
      succ[1] = block_of[end];
    for (u32 s : succ) {
      if (s == kNone)
        continue;
      bool changed = !reached[s];
      if (changed) {
        in[s] = eval;
        reached[s] = true;
      } else {
        changed = in[s].Meet(eval);
      }
      if (changed && !queued[s]) {
        queued[s] = true;
        work.push_back(s);
      }
    }
  }
  if (!work.empty())
    return;

  for (u32 b = 0; b < blocks; b++) {
    if (!reached[b])
      continue;
    ScalarEval eval = in[b];
    const u32 end = b + 1 < blocks ? starts[b + 1] : n;
    for (u32 i = starts[b]; i < end; i++) {
      const Inst& inst = program[i];
      for (const CsResource& resource : plan.resources) {
        if (resource.use_pc != inst.pc || resource.binding >= result.size() ||
            result[resource.binding].valid)
          continue;
        const u32 dwords = resource.kind == 1 ? 8 : resource.kind == 2 ? 2 : 4;
        if (!eval.AllKnown(resource.base_sgpr, dwords))
          continue;
        ResolvedCsResource& resolved = result[resource.binding];
        std::memcpy(resolved.descriptor, &eval.sgpr[resource.base_sgpr],
                    dwords * sizeof(u32));
        resolved.valid = true;
      }
      eval.Step(inst);
    }
  }
}

}  // namespace

void ResolveCsResources(base::Vector<ResolvedCsResource>& result,
                        const Program& program,
                        const RecompiledCs& plan,
                        const u32* user_data,
                        u64 code_base) {
  result.assign(plan.resources.size(), ResolvedCsResource{});
  if (!user_data)
    return;

  ScalarEval eval(user_data, code_base);
  for (const Inst& inst : program) {
    for (const CsResource& resource : plan.resources) {
      if (resource.use_pc != inst.pc || resource.binding >= result.size())
        continue;
      const u32 dwords = resource.kind == 1 ? 8 : resource.kind == 2 ? 2 : 4;
      if (!eval.AllKnown(resource.base_sgpr, dwords))
        continue;
      ResolvedCsResource& resolved = result[resource.binding];
      std::memcpy(resolved.descriptor, &eval.sgpr[resource.base_sgpr],
                  dwords * sizeof(u32));
      resolved.valid = true;
    }
    eval.Step(inst);
  }
  for (const ResolvedCsResource& resolved : result) {
    if (!resolved.valid) {
      ResolveAlongCfg(result, program, plan, user_data, code_base);
      break;
    }
  }
}

}  // namespace gpu::gcn
