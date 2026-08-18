/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + per-draw texture tracking. See
 * rdna_resource.h.
 */

#include "gpu/ps5/rdna/rdna_resource.h"
#include "base/arch.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/guest_memory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

#include <base/logging.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kAgcTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kDbg, "DELTA_AGC_RESTRACE", false);
DELTA_OPTION(bool, kGpuSwcensus, "DELTA_GPU_SWCENSUS", false);
DELTA_OPTION(bool, kGpuTexresolve, "DELTA_GPU_TEXRESOLVE", false);
}  // namespace

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Inst;
using gpu::gcn::MimgBindingPlan;
using gpu::gcn::Program;
using gpu::gcn::TImage;

bool InGuest(u64 a) {
  return a >= 0x10000ull && a < 0x1000000000000ull;
}

bool GuestRange(u64 address, u64 bytes) {
  return bytes && InGuest(address) && bytes <= 0x1000000000000ull - address &&
         gpu::IsReadableRangeCached(address, bytes);
}

// A cube sample reaches the hardware with the face already selected (the
// shader ran v_cubeid/v_cubesc/v_cubetc), so its address is (s, t, faceId) --
// exactly a 2D-array lookup with layer = faceId.
bool MimgArrayed(u32 dim) {
  return dim == 3 || dim == 5;
}

// gfx10.3 T#s carry a 9-bit unified format enum (word1 [28:20]). Values 1..77
// are the buffer format table (GPU Shader Core ISA spec 4.x "Buffer Format
// Conversions"); 130+ are image-only (SRGB, packed 16-bit, BCn). Map the
// sampled subset onto the GCN (dfmt,nfmt) pairs the shared renderer's
// GuestTextureFormat() understands; unmapped formats yield (0,0) so the upload
// declines (white fallback) instead of misreading texels.
void Gfx10ImgFormat(u32 gfmt, u32& dfmt, u32& nfmt) {
  switch (gfmt) {
    case 1:
      dfmt = 1;
      nfmt = 0;
      break;  // 8_UNORM
    case 2:
      dfmt = 1;
      nfmt = 1;
      break;  // 8_SNORM
    case 5:
      dfmt = 1;
      nfmt = 4;
      break;  // 8_UINT
    case 6:
      dfmt = 1;
      nfmt = 5;
      break;  // 8_SINT
    case 7:
      dfmt = 2;
      nfmt = 0;
      break;  // 16_UNORM
    case 11:
      dfmt = 2;
      nfmt = 4;
      break;  // 16_UINT
    case 13:
      dfmt = 2;
      nfmt = 7;
      break;  // 16_FLOAT
    case 14:
      dfmt = 3;
      nfmt = 0;
      break;  // 8_8_UNORM
    case 18:
      dfmt = 3;
      nfmt = 4;
      break;  // 8_8_UINT
    case 20:
      dfmt = 4;
      nfmt = 4;
      break;  // 32_UINT
    case 21:
      dfmt = 4;
      nfmt = 5;
      break;  // 32_SINT
    case 22:
      dfmt = 4;
      nfmt = 7;
      break;  // 32_FLOAT (also depth-resolve key)
    case 23:
      dfmt = 5;
      nfmt = 0;
      break;  // 16_16_UNORM
    case 27:
      dfmt = 5;
      nfmt = 4;
      break;  // 16_16_UINT
    case 29:
      dfmt = 5;
      nfmt = 7;
      break;  // 16_16_FLOAT
    case 36:
      dfmt = 6;
      nfmt = 7;
      break;  // 11_11_10_FLOAT
    case 44:
      dfmt = 8;
      nfmt = 0;
      break;  // 10_10_10_2_UNORM
    case 50:
      dfmt = 9;
      nfmt = 0;
      break;  // 2_10_10_10_UNORM
    case 56:
      dfmt = 10;
      nfmt = 0;
      break;  // 8_8_8_8_UNORM
    case 62:
      dfmt = 11;
      nfmt = 4;
      break;  // 32_32_UINT
    case 64:
      dfmt = 11;
      nfmt = 7;
      break;  // 32_32_FLOAT
    case 71:
      dfmt = 12;
      nfmt = 7;
      break;  // 16_16_16_16_FLOAT
    case 74:
      dfmt = 13;
      nfmt = 7;
      break;  // 32_32_32_FLOAT
    case 77:
      dfmt = 14;
      nfmt = 7;
      break;  // 32_32_32_32_FLOAT
    case 57:
      dfmt = 10;
      nfmt = 1;
      break;  // 8_8_8_8_SNORM
    case 60:
      dfmt = 10;
      nfmt = 4;
      break;  // 8_8_8_8_UINT
    case 130:
      dfmt = 10;
      nfmt = 9;
      break;  // 8_8_8_8_SRGB
    case 169:
      dfmt = 35;
      nfmt = 0;
      break;  // BC1
    case 170:
      dfmt = 35;
      nfmt = 9;
      break;
    case 171:
      dfmt = 36;
      nfmt = 0;
      break;  // BC2
    case 172:
      dfmt = 36;
      nfmt = 9;
      break;
    case 173:
      dfmt = 37;
      nfmt = 0;
      break;  // BC3
    case 174:
      dfmt = 37;
      nfmt = 9;
      break;
    case 175:
      dfmt = 38;
      nfmt = 0;
      break;  // BC4
    case 176:
      dfmt = 38;
      nfmt = 1;
      break;
    case 177:
      dfmt = 39;
      nfmt = 0;
      break;  // BC5
    case 178:
      dfmt = 39;
      nfmt = 1;
      break;
    case 181:
      dfmt = 41;
      nfmt = 0;
      break;  // BC7
    case 182:
      dfmt = 41;
      nfmt = 9;
      break;
    default:
      dfmt = 0;
      nfmt = 0;
      break;
  }
}

struct ScalarEval {
  static constexpr u32 kRegs = 136;
  u32 sgpr[kRegs] = {};
  bool known[kRegs] = {};
  bool scc = false;
  bool scc_known = false;

  ScalarEval(const u32* user_data,
             u32 user_sgprs,
             u32 user_sgpr_base) {
    const u32 count = std::min(user_sgprs, 32u);
    for (u32 i = 0; i < count && user_sgpr_base + i < kRegs; i++) {
      sgpr[user_sgpr_base + i] = user_data[i];
      known[user_sgpr_base + i] = true;
    }
    if (user_sgpr_base == 8) {
      sgpr[3] = 0x0101;
      known[3] = true;
    }
    // EXEC. The NGG prologue derives its lane masks from it and then feeds them
    // through s_cselect into the very registers a vertex V# is patched in, so
    // leaving EXEC unknown poisons the descriptor and the draw loses every
    // attribute. Model the same one-vertex/one-primitive wave the translator
    // does (sgpr[3] above): lane 0 active.
    sgpr[126] = 1;
    sgpr[127] = 0;
    known[126] = known[127] = true;
  }

  bool AllKnown(u32 s, u32 n) const {
    if (s + n > kRegs)
      return false;
    for (u32 i = 0; i < n; i++)
      if (!known[s + i])
        return false;
    return true;
  }
  u64 Ptr(u32 s) const {
    return sgpr[s] | (static_cast<u64>(sgpr[s + 1] & 0xFFFF) << 32);
  }
  void Set(u32 s, u32 value) {
    if (s < kRegs) {
      sgpr[s] = value;
      known[s] = true;
      src_addr[s] = 0;
    }
  }
  u32 cur_pc = 0;
  u32 clear_pc[kRegs] = {};
  // Guest address an SMEM load took each dword from, 0 when it came from user
  // data or the ALU. Only a diagnostic: it is what lets a resolved descriptor
  // be compared against the memory it claims to have come from.
  u64 src_addr[kRegs] = {};
  void Clear(u32 s) {
    if (s < kRegs) {
      known[s] = false;
      clear_pc[s] = cur_pc;
      src_addr[s] = 0;
    }
  }
  void SetDest(u32 base, u32 offset, u32 value) {
    if (base != 125)
      Set(base + offset, value);
  }
  void ClearDest(u32 base, u32 offset) {
    if (base != 125)
      Clear(base + offset);
  }

  bool Source(u32 field, u32 literal, u32& value) const {
    if (field == 125) {
      value = 0;
      return true;
    }
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
    if (field == 125) {
      value = 0;
      return true;
    }
    if (field <= 126)
      return Source(field + 1, 0, value);
    value = 0;
    return true;
  }

  void Step(const Inst& inst) {
    cur_pc = inst.pc;
    if (inst.enc == Enc::kSop1) {
      const u32 sdst = (inst.raw[0] >> 16) & 0x7F,
                     ssrc = inst.raw[0] & 0xFF;
      if (inst.opcode == 0x03) {
        u32 value;
        if (Source(ssrc, inst.literal, value))
          SetDest(sdst, 0, value);
        else
          ClearDest(sdst, 0);
      } else if (inst.opcode == 0x04) {
        u32 lo, hi;
        if (Source(ssrc, inst.literal, lo) && SourceHi(ssrc, hi)) {
          SetDest(sdst, 0, lo);
          SetDest(sdst, 1, hi);
        } else {
          ClearDest(sdst, 0);
          ClearDest(sdst, 1);
        }
      } else {
        // The gfx10 SOP1 table decides how wide the destination is; a 64-bit
        // form whose second dword stayed "known" would hand a descriptor half
        // a stale user-data pair.
        const u32 dwords = Sop1WriteDwords(inst.opcode);
        for (u32 i = 0; i < dwords; i++)
          ClearDest(sdst, i);
      }
      return;
    }
    if (inst.enc == Enc::kSop2) {
      const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
      if (inst.opcode == 0x0A || inst.opcode == 0x0B) {
        if (!scc_known) {
          if (kDbg) {
            static int n = 0;
            if (n++ < 20)
              BASE_LOGI("restrace",
                        "cselect pc={:#x} clears s{} (scc unknown)",
                        inst.pc, sdst);
          }
          ClearDest(sdst, 0);
          if (inst.opcode == 0x0B)
            ClearDest(sdst, 1);
          return;
        }
        const u32 source =
            scc ? inst.raw[0] & 0xFF : (inst.raw[0] >> 8) & 0xFF;
        u32 value;
        if (Source(source, inst.literal, value)) {
          SetDest(sdst, 0, value);
        } else {
          if (kDbg) {
            static int n = 0;
            if (n++ < 20)
              BASE_LOGI("restrace",
                        "cselect pc={:#x} s{} <- src{} UNKNOWN "
                        "(scc={})",
                        inst.pc, sdst, source, (int)scc);
          }
          ClearDest(sdst, 0);
        }
        if (inst.opcode == 0x0B) {
          if (SourceHi(source, value))
            SetDest(sdst, 1, value);
          else
            ClearDest(sdst, 1);
        }
        return;
      }
      u32 a, b;
      if (!Source(inst.raw[0] & 0xFF, inst.literal, a) ||
          !Source((inst.raw[0] >> 8) & 0xFF, inst.literal, b)) {
        ClearDest(sdst, 0);
        if (DecodeScalarWrite(inst).count == 2)
          ClearDest(sdst, 1);
        scc_known = false;
        return;
      }
      // 64-bit shifts (s_lshl_b64 / s_lshr_b64 / s_ashr_i64). The NGG prologue
      // narrows EXEC with `s_lshr_b64 exec, -1, vcc_lo`; without these the
      // default case clears EXEC, and the s_cselect_b64 that patches the vertex
      // V# from it then yields an unknown descriptor dword -- which drops every
      // vertex attribute and leaves the draw with no geometry at all.
      // s_bfe_u64 / s_bfe_i64: offset = S1[5:0], width = S1[22:16]. The vertex
      // V#'s last dword is patched with `s_cselect_b32 sN+3, sN+3, vcc` and VCC
      // is built through one of these, so leaving them to the default case
      // clears VCC, then the cselect clears the descriptor dword and the whole
      // attribute is dropped.
      if (inst.opcode == 0x29 || inst.opcode == 0x2A) {
        u32 a_hi;
        if (!SourceHi(inst.raw[0] & 0xFF, a_hi)) {
          ClearDest(sdst, 0);
          ClearDest(sdst, 1);
          scc_known = false;
          return;
        }
        const u64 wide = a | (static_cast<u64>(a_hi) << 32);
        const u32 off = b & 0x3F, width = (b >> 16) & 0x7F;
        u64 r = 0;
        if (width) {
          r = wide >> off;
          if (width < 64)
            r &= (u64{1} << width) - 1;
          if (inst.opcode == 0x2A && width < 64 &&
              (r & (u64{1} << (width - 1))))
            r |= ~((u64{1} << width) - 1);  // sign-extend
        }
        SetDest(sdst, 0, static_cast<u32>(r));
        SetDest(sdst, 1, static_cast<u32>(r >> 32));
        scc = r != 0;
        scc_known = true;
        return;
      }
      if (inst.opcode == 0x1F || inst.opcode == 0x21 || inst.opcode == 0x23) {
        u32 a_hi;
        if (!SourceHi(inst.raw[0] & 0xFF, a_hi)) {
          ClearDest(sdst, 0);
          ClearDest(sdst, 1);
          scc_known = false;
          return;
        }
        const u64 wide = a | (static_cast<u64>(a_hi) << 32);
        const u32 n = b & 63;
        const u64 r =
            inst.opcode == 0x1F  ? wide << n
            : inst.opcode == 0x21
                ? wide >> n
                : static_cast<u64>(static_cast<i64>(wide) >> n);
        SetDest(sdst, 0, static_cast<u32>(r));
        SetDest(sdst, 1, static_cast<u32>(r >> 32));
        scc = r != 0;
        scc_known = true;
        return;
      }
      u32 value;
      switch (inst.opcode) {
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
        case 0x27: {
          const u32 offset = b & 31;
          const u32 width = (b >> 16) & 0x7F;
          value = width >= 32 - offset
                      ? a >> offset
                      : (a >> offset) & ((u32{1} << width) - 1);
          break;
        }
        case 0x2f:
        case 0x30:
        case 0x31:
        case 0x32:
          value = (a << (inst.opcode - 0x2e)) + b;
          break;
        default:
          ClearDest(sdst, 0);
          if (inst.opcode == 0x0B || inst.opcode == 0x29 ||
              (inst.opcode >= 0x0F && inst.opcode <= 0x23 && (inst.opcode & 1)))
            ClearDest(sdst, 1);
          scc_known = false;
          return;
      }
      SetDest(sdst, 0, value);
      if (inst.opcode <= 0x03 || (inst.opcode >= 0x2f && inst.opcode <= 0x32))
        scc_known = false;
      switch (inst.opcode) {
        case 0x0e:
        case 0x10:
        case 0x12:
        case 0x14:
        case 0x16:
        case 0x18:
        case 0x1a:
        case 0x1c:
        case 0x1e:
        case 0x20:
        case 0x22:
        case 0x27:
          scc = value != 0;
          scc_known = true;
          break;
      }
      return;
    }
    if (inst.enc == Enc::kSopk) {
      const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
      const u32 imm = inst.raw[0] & 0xFFFF;
      const u32 simm = static_cast<u32>(
          static_cast<i32>(static_cast<i16>(imm)));
      if (inst.opcode == 0x00) {
        SetDest(sdst, 0, simm);
      } else if (inst.opcode == 0x02) {
        if (!scc_known)
          ClearDest(sdst, 0);
        else if (scc)
          SetDest(sdst, 0, simm);
      } else if (inst.opcode >= 0x03 && inst.opcode <= 0x0E) {
        u32 value;
        if (!Source(sdst, 0, value)) {
          scc_known = false;
          return;
        }
        const i32 a = static_cast<i32>(value);
        const i32 b = static_cast<i32>(simm);
        switch (inst.opcode) {
          case 0x03:
            scc = a == b;
            break;
          case 0x04:
            scc = a != b;
            break;
          case 0x05:
            scc = a > b;
            break;
          case 0x06:
            scc = a >= b;
            break;
          case 0x07:
            scc = a < b;
            break;
          case 0x08:
            scc = a <= b;
            break;
          case 0x09:
            scc = value == imm;
            break;
          case 0x0A:
            scc = value != imm;
            break;
          case 0x0B:
            scc = value > imm;
            break;
          case 0x0C:
            scc = value >= imm;
            break;
          case 0x0D:
            scc = value < imm;
            break;
          case 0x0E:
            scc = value <= imm;
            break;
        }
        scc_known = true;
      } else if (inst.opcode == 0x0F || inst.opcode == 0x10) {
        u32 value;
        if (Source(sdst, 0, value))
          SetDest(sdst, 0, inst.opcode == 0x0F ? value + simm : value * simm);
        if (inst.opcode == 0x0F)
          scc_known = false;
      } else if (inst.opcode == 0x12) {
        SetDest(sdst, 0, 0);
      }
      return;
    }
    if (inst.enc == Enc::kSopc) {
      u32 a, b;
      if (!Source(inst.raw[0] & 0xFF, inst.literal, a) ||
          !Source((inst.raw[0] >> 8) & 0xFF, inst.literal, b)) {
        scc_known = false;
        return;
      }
      switch (inst.opcode) {
        case 0x00:
        case 0x06:
          scc = a == b;
          break;
        case 0x01:
        case 0x07:
          scc = a != b;
          break;
        case 0x02:
          scc = static_cast<i32>(a) > static_cast<i32>(b);
          break;
        case 0x03:
          scc = static_cast<i32>(a) >= static_cast<i32>(b);
          break;
        case 0x04:
          scc = static_cast<i32>(a) < static_cast<i32>(b);
          break;
        case 0x05:
          scc = static_cast<i32>(a) <= static_cast<i32>(b);
          break;
        case 0x08:
          scc = a > b;
          break;
        case 0x09:
          scc = a >= b;
          break;
        case 0x0a:
          scc = a < b;
          break;
        case 0x0b:
          scc = a <= b;
          break;
        case 0x0c:
          scc = ((a >> (b & 31)) & 1) == 0;
          break;
        case 0x0d:
          scc = ((a >> (b & 31)) & 1) != 0;
          break;
        default:
          scc_known = false;
          return;
      }
      scc_known = true;
      return;
    }
    if (inst.enc != Enc::kSmrd)
      return;
    const Smem smem = DecodeSmem(inst);
    const u32 dwords = SmemLoadCount(smem.op);
    if (!dwords)
      return;
    const bool buffer = smem.op >= 0x08;
    const bool base_known = AllKnown(smem.sbase, buffer ? 4 : 2);
    const u64 base = base_known ? Ptr(smem.sbase) : 0;
    u32 soffset = 0;
    const bool offset_known = Source(smem.soffset, 0, soffset);
    const i64 immediate = buffer
                                  ? static_cast<i64>(inst.raw[1] & 0xFFFFF)
                                  : static_cast<i64>(smem.offset);
    const i64 byte_offset =
        static_cast<i64>(soffset & ~3u) + (immediate & ~i64{3});
    for (u32 i = 0; i < dwords; i++)
      ClearDest(smem.sdst, i);
    if (!base_known || !offset_known || byte_offset < 0 ||
        static_cast<u64>(byte_offset) > UINT64_MAX - base)
      return;
    const u64 address = base + static_cast<u64>(byte_offset);
    if (!GuestRange(address, static_cast<u64>(dwords) * 4))
      return;
    // The table this chain reads may still be sitting in a compute dispatch's
    // buffer, written this frame but not yet copied back; reading around that
    // resolves the descriptor to whatever the slot held before.
    if (gpu::gcn::g_flush_guest_range)
      gpu::gcn::g_flush_guest_range(address, static_cast<u64>(dwords) * 4);
    const auto* src = reinterpret_cast<const u32*>(address);
    for (u32 i = 0; i < dwords; i++) {
      SetDest(smem.sdst, i, src[i]);
      if (smem.sdst != 125 && smem.sdst + i < kRegs)
        src_addr[smem.sdst + i] = address + i * 4;
    }
  }
};

// The branch classification ComputeRdnaReachability uses: 0 = falls through,
// 1 = unconditional relative, 2 = conditional/call relative, 3 = program end,
// 4 = indirect.
int BranchKind(const Inst& inst) {
  if (inst.enc == Enc::kSopk)
    return inst.opcode == 0x16 || inst.opcode == 0x1b || inst.opcode == 0x1c
               ? 2
               : 0;
  if (inst.enc == Enc::kSop1)
    return inst.opcode >= 0x20 && inst.opcode <= 0x22 ? 4 : 0;
  if (inst.enc != Enc::kSopp)
    return 0;
  switch (inst.opcode) {
    case 0x01:
    case 0x1b:
    case 0x1e:
    case 0x1f:
      return 3;
    case 0x02:
      return 1;
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07:
    case 0x08:
    case 0x09:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1a:
      return 2;
    default:
      return 0;
  }
}

// Instruction index a relative branch lands on. Targets outside the decoded
// program read as one past the end, which no interval test can match.
u32 BranchTargetIndex(const Program& program, u32 i) {
  const i32 simm = static_cast<i16>(program[i].raw[0] & 0xFFFF);
  const i64 target = static_cast<i64>(program[i].pc) +
                     static_cast<i64>(program[i].size) + simm;
  for (u32 j = 0; j < program.size(); j++)
    if (static_cast<i64>(program[j].pc) >= target)
      return j;
  return static_cast<u32>(program.size());
}

// SOP2: the logical/shift block alternates 32-bit (even) and 64-bit
// (odd) forms, and anything unrecognised counts as a pair.
u32 Sop2WriteDwords(u32 op) {
  if (op <= 0x0A || op == 0x26 || op == 0x27 || op == 0x28 || op == 0x2c ||
      (op >= 0x2f && op <= 0x37))
    return 1;
  if (op >= 0x0e && op <= 0x28)
    return (op & 1) ? 2u : 1u;
  return 2;
}

// SOP2 operations ScalarEval evaluates. The rest it clears, which reads back as
// unknown rather than as a wrong value.
bool Sop2Exact(u32 op, bool scc_trusted) {
  if (op == 0x0A || op == 0x0B)
    return scc_trusted;  // s_cselect_b32/b64
  switch (op) {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03:
    case 0x0E:
    case 0x10:
    case 0x12:
    case 0x14:
    case 0x16:
    case 0x18:
    case 0x1A:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x26:
    case 0x27:
    case 0x29:
    case 0x2A:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
      return true;
    default:
      return false;
  }
}

// VOP3B: the carry-out and 64-bit-multiply forms put a scalar destination where
// VOP3A holds abs/op_sel (mirrors the translator's own RdnaVop3HasSdst).
bool Vop3HasSdst(u32 op) {
  return op == 0x128 || op == 0x129 || op == 0x12A || op == 0x30F ||
         op == 0x310 || op == 0x319 || op == 0x16D || op == 0x16E ||
         op == 0x176 || op == 0x177;
}

// SOP1 sets SCC on most of its operations and ScalarEval never models that, so
// a later s_cselect would pick its branch from a stale flag.
bool WritesSccUnmodelled(const Inst& inst) {
  return inst.enc == Enc::kSop1 && inst.opcode != 0x03 && inst.opcode != 0x04;
}

// ...and which instructions make it faithful again. SOP1 is the only writer
// ScalarEval leaves alone: every SOPC, every SOP2 bar s_cselect (which reads
// SCC without writing it) and the SOPK compares either compute SCC or mark it
// unknown, and an unknown SCC only clears a later s_cselect's destination. So
// the distrust an SOP1 earns lasts until the next real SCC write, not to the
// end of the shader.
bool RestoresScc(const Inst& inst) {
  if (inst.enc == Enc::kSopc)
    return true;
  if (inst.enc == Enc::kSop2)
    return inst.opcode != 0x0A && inst.opcode != 0x0B;
  if (inst.enc == Enc::kSopk)
    return inst.opcode >= 0x03 && inst.opcode <= 0x0F;
  return false;
}

}  // namespace

ScalarWrites PossibleScalarWrites(const Inst& inst, bool scc_trusted) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSmrd: {
      const Smem smem = DecodeSmem(inst);
      const u32 n = SmemLoadCount(smem.op);
      // s_memtime/s_memrealtime land two dwords in SDATA; a store writes none.
      return {{{smem.sdst, n ? n : 2u, n != 0}}};
    }
    case Enc::kSop1:
      return {{{(w >> 16) & 0x7F, Sop1WriteDwords(inst.opcode),
                inst.opcode == 0x03 || inst.opcode == 0x04}}};
    case Enc::kSop2:
      return {{{(w >> 16) & 0x7F, Sop2WriteDwords(inst.opcode),
                Sop2Exact(inst.opcode, scc_trusted)}}};
    case Enc::kSopk: {
      const u32 sdst = (w >> 16) & 0x7F;
      if (inst.opcode == 0x16)  // s_call_b64 stashes the return address
        return {{{sdst, 2, false}}};
      const bool exact = inst.opcode == 0x00 || inst.opcode == 0x0F ||
                         inst.opcode == 0x10 ||
                         (inst.opcode == 0x02 && scc_trusted);
      // 0x12 is s_getreg_b32, which ScalarEval answers with a flat zero.
      if (exact || inst.opcode == 0x02 || inst.opcode == 0x12)
        return {{{sdst, 1, exact}}};
      return {};
    }
    case Enc::kVop1:
      if (inst.opcode == 0x02)  // v_readfirstlane_b32
        return {{{(w >> 17) & 0xFF, 1, false}}};
      return {};
    case Enc::kVop2:
      // v_add_co_ci_u32 and its sub forms write VCC.
      if (inst.opcode >= 0x28 && inst.opcode <= 0x2A)
        return {{{106, 2, false}}};
      return {};
    case Enc::kVopc:
      // SDWA's SD bit picks an explicit SDST over VCC.
      if (inst.extension == gpu::gcn::InstExtension::kSdwa)
        return {{{106, 2, false}, {(w1 >> 8) & 0x7F, 2, false}}};
      return {{{106, 2, false}}};
    case Enc::kVop3: {
      ScalarWrites out;
      if (inst.opcode < 0x100)
        out.range[0] = {w & 0xFF, 2, false};  // VOPC alias: mask lands in VDST
      else if (inst.opcode == 0x182 || inst.opcode == 0x360)
        out.range[0] = {w & 0xFF, 1, false};  // v_readfirstlane / v_readlane
      if (Vop3HasSdst(inst.opcode))
        out.range[1] = {(w >> 8) & 0x7F, 2, false};
      return out;
    }
    default:
      return {};
  }
}

ScalarReplayPlan::Loss ScalarReplayPlan::LossAt(u32 sgpr,
                                               u32 dwords,
                                               u32 use_index) const {
  if (sgpr + dwords > kRegs)
    return kUnmodelled;
  // EXEC is seeded with a fictional one-lane mask, not read from the dispatch.
  if (sgpr < 128 && sgpr + dwords > 126)
    return kUnmodelled;
  // A branch target inside (write, use] means some path reaches the use
  // without the write; a back edge spanning the write means the wave ran it
  // more often than the replay's single walk did.
  const auto skippable = [&](u32 write) {
    for (u32 target : targets)
      if (target > write && target <= use_index)
        return true;
    for (const BackEdge& edge : back_edges)
      if (edge.target <= write && write <= edge.source)
        return true;
    return false;
  };
  // A write the program places after the use still runs before it if a back
  // edge carries execution from the write back to the use.
  const auto carried = [&](u32 write) {
    for (const BackEdge& edge : back_edges)
      if (edge.target <= use_index && edge.source >= write)
        return true;
    return false;
  };

  Loss worst = kCovered;
  const auto note = [&](Loss loss) { worst = std::max(worst, loss); };
  for (u32 reg = sgpr; reg < sgpr + dwords; reg++) {
    const Write* last = nullptr;
    for (const Write& write : writes) {
      if (reg < write.first || reg >= write.first + write.count)
        continue;
      if (write.index < use_index)
        last = &write;  // writes are recorded in program order
      else if (indirect || carried(write.index))
        note(kLoopCarried);
    }
    if (!last)
      continue;  // never written: still the user data the dispatch seeded
    if (!last->exact)
      note(kUnmodelled);
    else if (indirect || skippable(last->index))
      note(kConditional);
  }
  return worst;
}

ScalarReplayPlan PlanScalarReplay(const Program& program) {
  ScalarReplayPlan plan;
  bool scc_trusted = true;
  u32 index = 0;
  for (const Inst& inst : program) {
    const int kind = BranchKind(inst);
    if (kind == 4)
      plan.indirect = true;
    if (kind == 1 || kind == 2) {
      const u32 target = BranchTargetIndex(program, index);
      plan.targets.push_back(target);
      if (target <= index)
        plan.back_edges.push_back({target, index});
    }
    for (const ScalarWrites::Range& range :
         PossibleScalarWrites(inst, scc_trusted).range)
      if (range.count)
        plan.writes.push_back({index, range.first, range.count, range.exact});
    if (WritesSccUnmodelled(inst))
      scc_trusted = false;
    else if (RestoresScc(inst))
      scc_trusted = true;
    index++;
  }
  return plan;
}

MimgBindingPlan RdnaPlanMimg(const Program& program) {
  MimgBindingPlan plan;
  struct BindingKey {
    u32 srsrc;
    u32 ssamp;
    u32 flags;
    u32 versions[12];
  };
  std::vector<BindingKey> keys;
  u32 versions[136] = {};
  u32 generation = 1;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kMimg) {
      const u32 w0 = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
      const u32 dim = (w0 >> 3) & 0x7;
      const bool r128 = (w0 >> 15) & 1;
      const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
      const bool sampling = op >= 0x20;
      const bool storage = op == 0x08 || op == 0x09;
      const u32 ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFFu;
      BindingKey key{
          .srsrc = srsrc,
          .ssamp = ssamp,
          .flags = static_cast<u32>(MimgArrayed(dim)) |
                   ((op == 0x28 || op == 0x2f ? 1u : 0u) << 1) |
                   ((op == 0x47 ? 1u : 0u) << 2) |
                   (static_cast<u32>(storage) << 3) |
                   (static_cast<u32>(r128) << 4),
      };
      for (u32 i = 0; i < (r128 ? 4u : 8u); i++)
        key.versions[i] = versions[srsrc + i];
      if (sampling)
        for (u32 i = 0; i < 4; i++)
          key.versions[8 + i] = versions[ssamp + i];

      u32 binding = static_cast<u32>(keys.size());
      for (u32 i = 0; i < keys.size(); i++)
        if (std::memcmp(&keys[i], &key, sizeof(key)) == 0) {
          binding = i;
          break;
        }
      if (binding == keys.size()) {
        keys.push_back(key);
        plan.binding_srsrc.push_back(srsrc);
        plan.binding_storage.push_back(storage);
      }
      plan.binding_by_pc[inst.pc] = binding;
    }

    const ScalarWrite write = DecodeScalarWrite(inst);
    for (u32 i = 0; i < write.count && write.first + i < 136; i++)
      versions[write.first + i] = generation;
    generation++;
  }
  return plan;
}

void DecodeBufferFormat(u32 gfmt, u32& dfmt, u32& nfmt) {
  // The gfx10.3 unified buffer-format enum (ISA spec "Buffer Format
  // Conversions") is consecutive runs of one channel layout, each in the order
  // UNorm, SNorm, UScaled, SScaled, UInt, SInt [, Float]. Map each run onto the
  // GCN (dfmt, nfmt) pair. Listing only the float formats and splitting the
  // rest as GCN bitfields turned Skyrim's 16_16_SScaled UI positions (26) into
  // 8_8_8_8_SNorm and collapsed every glyph quad to a degenerate triangle.
  struct Run {
    u8 first, count, dfmt;
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
  static constexpr u8 kNfmt[6] = {0, 1, 2, 3, 4, 5};
  for (const Run& r : kRuns) {
    if (gfmt < r.first || gfmt >= r.first + r.count)
      continue;
    const u32 i = gfmt - r.first;
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

VBuffer DecodeVBuffer(const u32* d) {
  VBuffer v;
  v.base = (static_cast<u64>(d[1] & 0xFFFF) << 32) | d[0];
  v.stride = (d[1] >> 16) & 0x3FFF;
  v.num_records = d[2];
  v.gfmt = (d[3] >> 12) & 0x7F;
  DecodeBufferFormat(v.gfmt, v.dfmt, v.nfmt);
  return v;
}

bool PlausibleVBuffer(const VBuffer& v) {
  return v.stride && v.stride <= 256 && v.num_records &&
         v.num_records <= 0x100000;
}

TImage DecodeTImage(const u32* d, bool r128) {
  TImage t;
  const u32 descriptor_dwords = r128 ? 4 : 8;
  t.null_descriptor = std::all_of(
      d, d + descriptor_dwords, [](u32 word) { return word == 0; });
  const u64 base_units = d[0] | (static_cast<u64>(d[1] & 0xFF) << 32);
  t.base = base_units << 8;
  t.min_lod = (d[1] >> 8) & 0xFFF;
  t.width = (((d[1] >> 30) & 0x3) | ((d[2] & 0xFFF) << 2)) + 1;
  t.height = ((d[2] >> 14) & 0x3FFF) + 1;
  t.base_mip = (d[3] >> 12) & 0xF;
  const u32 last_level = (d[3] >> 16) & 0xF;
  const u32 sw_mode = (d[3] >> 20) & 0x1F;
  t.type = (d[3] >> 28) & 0xF;
  for (int i = 0; i < 4; i++)
    t.dst_sel[i] = (d[3] >> (i * 3)) & 0x7;
  const u32 depth = r128 ? 0 : d[4] & 0x1fff;
  t.base_array = r128 ? 0 : (d[4] >> 16) & 0x1fff;
  const u32 max_mip = r128 ? last_level : (d[5] >> 4) & 0xf;

  t.pitch = t.width;
  // Cube (type 11) is sampled as a 6-layer 2D array; see MimgArrayed.
  t.arrayed = t.type == 11 || t.type == 12 || t.type == 13;
  const bool volumetric = t.type == 10;  // 3D
  t.layers = (t.arrayed || volumetric) ? depth + 1 : 1;
  if (t.type == 11)
    t.layers = std::max<u32>(t.layers, 6);
  t.view_layers =
      t.arrayed
          ? std::max<u32>(t.layers - std::min(t.base_array, t.layers), 1)
          : 1;
  t.mip_levels = max_mip + 1;
  t.view_mips = std::max<u32>(last_level + 1 - t.base_mip, 1);
  const bool valid_array = !t.arrayed || t.base_array <= depth;
  const u32 gfmt = (d[1] >> 20) & 0x1FF;
  Gfx10ImgFormat(gfmt, t.dfmt, t.nfmt);
  // gfx10 swizzle mode 0 = SW_LINEAR -> the renderer's linear index. The
  // "standard" modes (256 B / 4 KiB / 64 KiB, ids 1/5/9) map onto the gfx10
  // detiler's own id range. Everything else (Z/D/R, the _X pipe-XOR and _T
  // variants) has no detiler yet, so it is shifted past the valid range:
  // BuildTextureLayout32 rejects it and the draw gets the white fallback
  // instead of scrambled texels.
  // DELTA_GPU_SWCENSUS: which gfx10 swizzle modes this title's textures use --
  // the detiler only covers linear and the three "standard" modes, and anything
  // else is rejected into the white fallback (flat-coloured quads).
  if (kGpuSwcensus) {
    static u32 seen[64] = {};
    if (sw_mode < 64 && seen[sw_mode]++ == 0)
      BASE_LOGI("swcensus", "sw_mode={} first seen ({}x{} gfmt={})",
                sw_mode, t.width, t.height, gfmt);
    static u32 seen_sel[4096] = {};
    const u32 packed = (t.dst_sel[0]) | (t.dst_sel[1] << 3) |
                            (t.dst_sel[2] << 6) | (t.dst_sel[3] << 9);
    if (packed < 4096 && seen_sel[packed]++ == 0)
      BASE_LOGI(
          "swcensus",
          "dst_sel = {},{},{},{} (word3={:08x}) on {}x{} gfmt={}",
          t.dst_sel[0], t.dst_sel[1], t.dst_sel[2], t.dst_sel[3], d[3], t.width,
          t.height, gfmt);
  }
  // gfx10 swizzle modes have their own address equations rather than the
  // Liverpool ones; BuildTextureLayout32 applies the 256-byte linear row
  // alignment itself, from the element size the staging path picked.
  t.tiling_idx = gcn::kGfx10TilingBase + sw_mode;
  if (kAgcTrace) {
    static u32 seen[32], n_seen = 0;
    const u32 key = (gfmt << 8) | sw_mode;
    bool is_new = true;
    for (u32 i = 0; i < n_seen; i++)
      if (seen[i] == key) {
        is_new = false;
        break;
      }
    if (is_new && n_seen < 32) {
      seen[n_seen++] = key;
      BASE_LOGI("agc",
                "T# base={:#x} {}x{} gfmt={} sw={} type={} mips={} -> "
                "dfmt={} nfmt={} tiling={} pitch={}",
                (unsigned long)t.base, t.width, t.height, gfmt, sw_mode,
                t.type, t.mip_levels, t.dfmt, t.nfmt, t.tiling_idx, t.pitch);
    }
  }
  u32 max_levels = 1;
  for (u32 extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
    max_levels++;
  const bool valid_mips = t.base_mip <= last_level && last_level < max_levels &&
                          t.base_mip + t.view_mips <= t.mip_levels;
  const bool valid_word4 = r128 || !(d[4] & 0xe000c000u);
  // Word 6's compression fields describe DCC/metadata the hardware would use to
  // read the surface. We never compress a render target, and one sampled back
  // resolves through the address page table to the image we rendered into, so
  // those bits say nothing about whether the descriptor is well formed. Taking
  // them as invalid rejected every G-buffer Demon's Souls sampled, and a
  // rejected T# loses its base, which then reads as "never written".
  const bool valid_compression = true;
  const bool valid_compact_type =
      !r128 || t.type == 8 || t.type == 9 || t.type == 14;
  t.valid = InGuest(t.base) && t.dfmt && t.width <= 16384 &&
            t.height <= 16384 && t.layers <= 16384 && valid_array &&
            valid_mips && valid_word4 && valid_compression &&
            valid_compact_type;
  return t;
}

std::vector<TImage> TrackTextures(const u32* ps_code,
                                  const u32* pud,
                                  u32 user_sgprs) {
  std::vector<TImage> out;
  if (!ps_code || !pud || !InGuest(reinterpret_cast<u64>(ps_code)))
    return out;
  const auto prog_ref = CachedReachableProgram(ps_code, 4096);
  const Program& prog = *prog_ref;
  // The plan is a pure function of the program, so the cached program's own
  // identity says whether a cached plan still describes it.
  static std::unordered_map<u64, std::pair<std::shared_ptr<const Program>,
                                           MimgBindingPlan>>
      plan_cache;
  const u64 code_addr = reinterpret_cast<u64>(ps_code);
  auto plan_it = plan_cache.find(code_addr);
  if (plan_it == plan_cache.end() || plan_it->second.first != prog_ref) {
    if (plan_cache.size() > 512)
      plan_cache.clear();
    plan_it = plan_cache.insert_or_assign(code_addr,
                                          std::make_pair(prog_ref,
                                                         RdnaPlanMimg(prog)))
                  .first;
  }
  const MimgBindingPlan& plan = plan_it->second.second;
  out.resize(plan.binding_srsrc.size());
  std::vector<bool> filled(out.size(), false);
  ScalarEval eval(pud, user_sgprs, 0);

  for (const Inst& in : prog) {
    eval.Step(in);
    if (in.enc != Enc::kMimg)
      continue;
    auto it = plan.binding_by_pc.find(in.pc);
    if (it == plan.binding_by_pc.end() || filled[it->second])
      continue;
    const u32 b = it->second;
    const u32 w0 = in.raw[0], w1 = in.raw[1], op = in.opcode;
    // DELTA_GPU_TEXRESOLVE: which SGPR quad each sampler's T# comes from, and
    // whether it resolved. A binding that reads back base 0 is either a null
    // descriptor or a chain we failed to walk.
    const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
    const bool r128 = (w0 >> 15) & 1;
    const u32 resource_dwords = r128 ? 4 : 8;
    if (kGpuTexresolve && !eval.AllKnown(srsrc, resource_dwords)) {
      BASE_LOGI("texres",
                "  mimg pc={:04x} w0={:08x} w1={:08x} op={:#x} nsa={} "
                "srsrc_field={} ssamp_field={} vaddr={} vdata={}",
                in.pc, w0, w1, op, (w0 >> 1) & 3, (w1 >> 16) & 0x1F,
                (w1 >> 21) & 0x1F, w1 & 0xFF, (w1 >> 8) & 0xFF);
      // An "inline" descriptor that user data never programmed has to come from
      // somewhere else in the shader: list every scalar write to its registers.
      for (const Inst& w : prog) {
        u32 d0 = 0xFFFF, cnt = 1;
        if (w.enc == Enc::kSop1) {
          d0 = (w.raw[0] >> 16) & 0x7F;
          cnt = w.opcode == 0x04 ? 2 : 1;
        } else if (w.enc == Enc::kSop2)
          d0 = (w.raw[0] >> 16) & 0x7F;
        else if (w.enc == Enc::kSmrd && w.opcode <= 0x0C) {
          d0 = (w.raw[0] >> 6) & 0x7F;
          cnt = 8;
        }
        if (d0 == 0xFFFF)
          continue;
        if (d0 + cnt <= srsrc || d0 >= srsrc + resource_dwords)
          continue;
        BASE_LOGI(
            "texres",
            "  writer pc={:04x} enc={} op={:#x} -> s{}..{} ({:08x} {:08x})",
            w.pc, (int)w.enc, w.opcode, d0, d0 + cnt - 1, w.raw[0], w.raw[1]);
      }
    }
    if (kGpuTexresolve)
      BASE_LOGI("texres", "binding={} srsrc=s{} known={:d}", b, srsrc,
                eval.AllKnown(srsrc, resource_dwords));
    if (eval.AllKnown(srsrc, resource_dwords)) {
      out[b] = DecodeTImage(&eval.sgpr[srsrc], r128);
      // A descriptor that decodes with sane extents but a zero base is either
      // genuinely unpatched in guest memory or read from the wrong place. Print
      // both sides of that comparison: what the replay produced, and what the
      // address it loaded from holds right now.
      if (kGpuTexresolve && !out[b].base && out[b].width > 1 &&
          out[b].height > 1) {
        static u32 printed = 0;
        if (printed++ < 6) {
          std::string replayed, in_memory;
          const u64 from = eval.src_addr[srsrc];
          const bool mapped = from && GuestRange(from, resource_dwords * 4);
          const auto* mem = reinterpret_cast<const u32*>(from);
          char word[16];
          for (u32 i = 0; i < resource_dwords; i++) {
            std::snprintf(word, sizeof(word), "%08x ", eval.sgpr[srsrc + i]);
            replayed += word;
            std::snprintf(word, sizeof(word), "%08x ", mapped ? mem[i] : 0u);
            in_memory += word;
          }
          BASE_LOGI("texres",
                    "zero-base binding={} srsrc=s{} {}x{} replayed=[ {}] "
                    "from={:#x} mapped={:d} memory=[ {}]",
                    b, srsrc, out[b].width, out[b].height, replayed.c_str(),
                    static_cast<unsigned long>(from), mapped ? 1 : 0,
                    mapped ? in_memory.c_str() : "unreadable ");
        }
      }
      out[b].arrayed = MimgArrayed((w0 >> 3) & 0x7);
      out[b].depth_compare = op == 0x28 || op == 0x2f;
      out[b].force_lod_zero = op == 0x47;
      out[b].storage = op == 0x08 || op == 0x09;
      const u32 ssamp = ((w1 >> 21) & 0x1F) * 4;
      if (op >= 0x20 && eval.AllKnown(ssamp, 4)) {
        std::memcpy(out[b].sampler, &eval.sgpr[ssamp], sizeof(out[b].sampler));
        out[b].sampler_valid = true;
      }
    }
    filled[b] = true;
  }
  return out;
}

std::unordered_map<u32, BufferResource> ResolveBuffers(
    const u32* code,
    const u32* user_data,
    u32 user_sgprs,
    u32 user_sgpr_base) {
  std::unordered_map<u32, BufferResource> out;
  if (!code || !user_data || !InGuest(reinterpret_cast<u64>(code)))
    return out;
  ScalarEval eval(user_data, user_sgprs, user_sgpr_base);
  for (const Inst& inst : *CachedReachableProgram(code, 4096)) {
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      // s_buffer_load reads through a V#, s_load through a bare 64-bit pointer.
      // A compute dispatch binds the range either names as a storage buffer, so
      // the descriptor travels with the base, not just the address it decodes
      // to (a graphics cbuffer only wants the latter).
      const u32 dwords = smem.op >= 0x08 ? 4u : 2u;
      if (eval.AllKnown(smem.sbase, dwords)) {
        BufferResource resource;
        resource.base = eval.Ptr(smem.sbase);
        resource.descriptor_dwords = dwords;
        std::memcpy(resource.descriptor, &eval.sgpr[smem.sbase],
                    dwords * sizeof(u32));
        resource.descriptor_valid = true;
        out.emplace(inst.pc, resource);
      }
    } else if (inst.enc == Enc::kMimg) {
      // An image T# an SRT chain produced: the dispatch path needs its extents
      // and format to stage the surface, and cannot read them out of user data.
      const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
      const u32 dwords = ((inst.raw[0] >> 15) & 1) ? 4u : 8u;
      if (eval.AllKnown(srsrc, dwords)) {
        BufferResource resource;
        resource.base = eval.Ptr(srsrc);
        resource.descriptor_dwords = dwords;
        std::memcpy(resource.descriptor, &eval.sgpr[srsrc],
                    dwords * sizeof(u32));
        resource.descriptor_valid = true;
        out.emplace(inst.pc, resource);
      }
    } else if (inst.enc == Enc::kFlat) {
      // A global_ access addresses s[SADDR:SADDR+1] + v[ADDR] + offset, so the
      // scalar pair is the resource and the VGPR is a byte offset into it.
      // gfx10.3 FLAT: w0[15:14] SEG (2 = global), w1[22:16] SADDR, with 0x7d
      // (and gfx9's 0x7f) reading as NULL. That pair sits well past the
      // user-data window -- s[48:49] in this title -- so the replay is the only
      // thing that can recover it.
      const u32 seg = (inst.raw[0] >> 14) & 0x3;
      const u32 saddr = (inst.raw[1] >> 16) & 0x7F;
      if (seg == 2 && saddr != 0x7D && saddr != 0x7F &&
          eval.AllKnown(saddr, 2)) {
        BufferResource resource;
        resource.base = eval.Ptr(saddr);
        resource.descriptor_dwords = 2;
        std::memcpy(resource.descriptor, &eval.sgpr[saddr], 2 * sizeof(u32));
        resource.descriptor_valid = true;
        out.emplace(inst.pc, resource);
      }
    } else if (inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) {
      // Stores too, not just the loads a graphics stage fetches with: a compute
      // dispatch's output buffer reaches it through the same V#.
      const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
      if (kDbg) {
        static int n = 0;
        if (n++ < 40)
          BASE_LOGI("restrace",
                    "fetch pc={:#x} srsrc=s{} known={}{}{}{} "
                    "clearedby={:#x}/{:#x}/{:#x}/{:#x}",
                    inst.pc, srsrc, (int)eval.known[srsrc],
                    (int)eval.known[srsrc + 1], (int)eval.known[srsrc + 2],
                    (int)eval.known[srsrc + 3], eval.clear_pc[srsrc],
                    eval.clear_pc[srsrc + 1], eval.clear_pc[srsrc + 2],
                    eval.clear_pc[srsrc + 3]);
      }
      if (eval.AllKnown(srsrc, 4)) {
        BufferResource resource;
        resource.base = eval.Ptr(srsrc);
        resource.descriptor_dwords = 4;
        std::memcpy(resource.descriptor, &eval.sgpr[srsrc], 4 * sizeof(u32));
        resource.descriptor_valid = true;
        const u32 soff = (inst.raw[1] >> 24) & 0xFF;
        if (soff == 128) {
          resource.soffset_valid = true;
        } else if (soff > 128 && soff <= 192) {
          resource.soffset = soff - 128;
          resource.soffset_valid = true;
        } else if (soff < ScalarEval::kRegs && eval.known[soff]) {
          resource.soffset = eval.sgpr[soff];
          resource.soffset_valid = true;
        }
        out.emplace(inst.pc, resource);
      }
    }
    eval.Step(inst);
  }
  return out;
}

}  // namespace gpu::rdna
