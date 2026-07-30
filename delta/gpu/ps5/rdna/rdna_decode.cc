/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) instruction decoder. See rdna_decode.h. Produces the shared
 * gpu::gcn::Inst representation; the RDNA2-specific opcode semantics are
 * applied by the dispatch in rdna_translate.cc.
 */

#include "gpu/ps5/rdna/rdna_decode.h"

#include <algorithm>
#include <cstdio>

namespace gpu::rdna {
namespace {

// The 'OrbShdr'-style ShaderBinaryInfo signature at byte offset `off`. The AGC
// toolchain reuses the same 7-byte-signature footer scheme; we only need its
// position (== code length), not its bytes, so match either the PS4 magic or a
// generic printable 7-byte token beginning with a capital letter.
bool BinaryInfoAt(const uint32_t* code, uint32_t off) {
  const char* s = reinterpret_cast<const char*>(code) + off;
  // PS4/Orbis: "OrbShdr". PS5 keeps a 7-byte signature; accept the Orbis magic
  // (still emitted by many AGC blobs) as the reliable anchor.
  return s[0] == 'O' && s[1] == 'r' && s[2] == 'b' && s[3] == 'S' &&
         s[4] == 'h' && s[5] == 'd' && s[6] == 'r';
}

// A source-operand field selects a trailing 32-bit literal when it == 255.
// Scalar sources are 8-bit ([7:0] / [15:8]); vector src0 is 9-bit ([8:0]).
bool Scalar2HasLit(uint32_t w) {
  return (w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255;
}
bool Scalar1HasLit(uint32_t w) {
  return (w & 0xFF) == 255;
}
// VALU src0 (9-bit [8:0]) values that append an extension dword. DPP forms are
// valid only for VOP1/VOP2; SDWA and literals are also valid for VOPC.
gpu::gcn::InstExtension ValuExtension(Enc enc, uint32_t w) {
  switch (w & 0x1FF) {
    case 233:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp8
                 : gpu::gcn::InstExtension::kNone;
    case 234:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp8Fi
                 : gpu::gcn::InstExtension::kNone;
    case 249:
      return gpu::gcn::InstExtension::kSdwa;
    case 250:
      return enc == Enc::kVop1 || enc == Enc::kVop2 || enc == Enc::kVopc
                 ? gpu::gcn::InstExtension::kDpp
                 : gpu::gcn::InstExtension::kNone;
    case 255:
      return gpu::gcn::InstExtension::kLiteral;
    default:
      return gpu::gcn::InstExtension::kNone;
  }
}

// Family classification, mirroring the gfx10.3 encoding-family dispatch. The
// VALU branch (bit31 == 0) further resolves VOP1/VOPC from the VOP2 opcode
// field; we resolve it here so the shared translator sees the right Enc.
Enc Classify(uint32_t w, uint32_t& opcode) {
  if ((w & 0x80000000u) ==
      0u) {  // VALU 32-bit: VOP2, or VOP1/VOPC via op field
    uint32_t vop2_op = (w >> 25) & 0x3F;
    if (vop2_op == 0x3F) {  // VOP1
      opcode = (w >> 9) & 0xFF;
      return Enc::kVop1;
    }
    if (vop2_op == 0x3E) {  // VOPC
      opcode = (w >> 17) & 0xFF;
      return Enc::kVopc;
    }
    opcode = vop2_op;
    return Enc::kVop2;
  }
  if ((w & 0xC0000000u) == 0x80000000u) {  // scalar
    uint32_t sub = (w >> 23) & 0x7F;
    if (sub == 0x7D) {
      opcode = (w >> 8) & 0xFF;
      return Enc::kSop1;
    }
    if (sub == 0x7E) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopc;
    }
    if (sub == 0x7F) {
      opcode = (w >> 16) & 0x7F;
      return Enc::kSopp;
    }
    if (sub >= 0x60) {
      opcode = (w >> 23) & 0x1F;
      return Enc::kSopk;
    }
    opcode = (w >> 23) & 0x7F;
    return Enc::kSop2;
  }
  switch (w >> 26) {  // top bits 11
    case 0x32:
      opcode = (w >> 16) & 0x3;
      return Enc::kVintrp;
    case 0x33:
      if ((w >> 24) != 0xCC || (w & (1u << 23))) {
        opcode = 0;
        return Enc::kUnknown;
      }
      opcode = (w >> 16) & 0x7F;
      return Enc::kVop3p;
    case 0x35:
      opcode = (w >> 16) & 0x3FF;
      return Enc::kVop3;  // 10-bit opcode
    case 0x36:
      // gfx10 moved the DS opcode up a bit (GDS took bit 17), so the gfx9
      // field reads as op*2 + gds here.
      opcode = (w >> 18) & 0xFF;
      return Enc::kDs;
    case 0x37:
      opcode = (w >> 18) & 0x7F;
      return Enc::kFlat;
    case 0x38:
      opcode = ((w >> 18) & 0x7F) | (((w >> 25) & 1) << 7);
      return Enc::kMubuf;
    // MTBUF carries op[2:0] here and op[3] in word1[21]; Decode() adds that bit
    // once the second dword is in hand. word0[25:19] is the unified format.
    case 0x3A:
      opcode = (w >> 16) & 0x7;
      return Enc::kMtbuf;
    case 0x3C:
      opcode = ((w >> 18) & 0x7F) | ((w & 1) << 7);
      return Enc::kMimg;
    case 0x3D:
      opcode = (w >> 18) & 0xFF;
      return Enc::kSmrd;  // SMEM (replaces GCN SMRD)
    // Observed NGG streams use the reserved 0b110001 prefix only for null
    // exports. Do not let non-null forms acquire ordinary EXP semantics.
    case 0x31:
      if (w & 0xF) {
        opcode = 0;
        return Enc::kUnknown;
      }
      opcode = (w >> 4) & 0x3F;
      return Enc::kExp;
    // RDNA2 EXP is 0b111110 (0xf8 prefix), same slot as GCN; target/en live in
    // [9:4]/[3:0] (see EmitExport).
    case 0x3E:
      opcode = (w >> 4) & 0x3F;
      return Enc::kExp;
    default:
      opcode = 0;
      return Enc::kUnknown;
  }
}

// Base dword count (excluding any trailing literal). RDNA2 two-dword encodings:
// VOP3/VOP3P, SMEM, DS, MUBUF/MTBUF, FLAT, MIMG, EXP. MIMG adds NSA words.
uint32_t BaseSize(Enc e, uint32_t w) {
  if ((w >> 26) == 0x31)
    return 2;  // reserved NGG export slot retains the EXP-shaped width
  switch (e) {
    case Enc::kVop3:
    case Enc::kVop3p:
    case Enc::kSmrd:  // SMEM is 2 dwords on RDNA2 (was 1 on GCN SMRD)
    case Enc::kDs:
    case Enc::kFlat:
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kExp:
      return 2;
    case Enc::kMimg: {
      uint32_t nsa =
          (w >> 1) & 0x3;  // NSA (non-sequential address) extra dwords
      return 2 + nsa;
    }
    default:
      return 1;
  }
}

uint32_t Vop3SourceCountImpl(Enc enc, uint32_t op) {
  if (enc == Enc::kVop3p) {
    switch (op) {
      case 0x00:
      case 0x09:
      case 0x0e:
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x20:
      case 0x21:
      case 0x22:
        return 3;
      default:
        return 2;
    }
  }
  if (op < 0x100)
    return 2;  // VOPC aliases
  if (op == 0x101 || (op >= 0x128 && op <= 0x12a))
    return 3;
  if (op >= 0x100 && op < 0x140)
    return 2;  // VOP2 aliases
  if (op >= 0x180 && op < 0x200)
    return 1;  // VOP1 aliases
  if (op >= 0x270 && op <= 0x272)
    return 2;  // VINTRP aliases
  return 3;
}

std::vector<uint8_t> ComputeRdnaReachability(const Program& program) {
  std::vector<uint8_t> reachable(program.size(), 0);
  if (program.empty())
    return reachable;

  // 0=ordinary, 1=unconditional relative, 2=conditional/call relative,
  // 3=program end, 4=indirect control flow.
  const auto branch_kind = [](const Inst& inst) {
    if (inst.enc == Enc::kSopk) {
      if (inst.opcode == 0x16 || inst.opcode == 0x1b || inst.opcode == 0x1c)
        return 2;  // s_call_b64 / subvector loops
      return 0;
    }
    if (inst.enc == Enc::kSop1) {
      if (inst.opcode >= 0x20 && inst.opcode <= 0x22)
        return 4;  // setpc/swappc/rfe
      return 0;
    }
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
  };

  const uint32_t max_pc = program.back().pc + program.back().size;
  std::vector<uint32_t> starts{0};
  for (const Inst& inst : program) {
    const int kind = branch_kind(inst);
    if (!kind)
      continue;
    starts.push_back(inst.pc + inst.size);
    if (kind == 1 || kind == 2) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xffff);
      starts.push_back(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                             static_cast<int32_t>(inst.size) +
                                             simm));
    }
  }
  std::sort(starts.begin(), starts.end());
  starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
  starts.erase(std::remove_if(starts.begin(), starts.end(),
                              [max_pc](uint32_t pc) { return pc >= max_pc; }),
               starts.end());
  const auto block_of = [&](uint32_t pc) {
    uint32_t block = 0;
    for (uint32_t i = 0; i < starts.size() && starts[i] <= pc; i++)
      block = i;
    return block;
  };

  std::vector<uint8_t> block_reachable(starts.size(), 0);
  std::vector<uint32_t> worklist{0};
  while (!worklist.empty()) {
    const uint32_t block = worklist.back();
    worklist.pop_back();
    if (block >= starts.size() || block_reachable[block])
      continue;
    block_reachable[block] = 1;
    const uint32_t block_end =
        block + 1 < starts.size() ? starts[block + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < starts[block] || inst.pc >= block_end)
        continue;
      const int kind = branch_kind(inst);
      if (!kind)
        continue;
      terminated = true;
      if (kind == 3)
        break;
      if (kind == 4) {
        std::fill(block_reachable.begin(), block_reachable.end(), 1);
        worklist.clear();
        break;
      }
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xffff);
      const uint32_t target =
          static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                static_cast<int32_t>(inst.size) + simm);
      if (target < max_pc)
        worklist.push_back(block_of(target));
      if (kind == 2)
        worklist.push_back(block + 1);
      break;
    }
    if (!terminated)
      worklist.push_back(block + 1);
  }

  for (uint32_t i = 0; i < program.size(); i++)
    reachable[i] = block_reachable[block_of(program[i].pc)];
  return reachable;
}

}  // namespace

uint32_t Vop3SourceCount(Enc enc, uint32_t op) {
  return Vop3SourceCountImpl(enc, op);
}

uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords) {
  if (!code || max_dwords < 2)
    return 0;
  // Fast path: the toolchain emits "s_mov_b32 vcc_hi/null, #imm" (0xBEEB03FF)
  // as the first instruction, with the ShaderBinaryInfo footer at
  // code[(imm+1)*2].
  if (code[0] == 0xBEEB03FFu) {
    uint64_t d = static_cast<uint64_t>(code[1] + 1) * 2;
    if (d >= 2 && d + 2 <= max_dwords &&
        BinaryInfoAt(code, static_cast<uint32_t>(d) * 4))
      return static_cast<uint32_t>(d);
  }
  for (uint32_t d = 1; d + 2 <= max_dwords; d++)
    if (BinaryInfoAt(code, d * 4))
      return d;
  return 0;
}

Program Decode(const uint32_t* code, uint32_t max_dwords, bool stop_at_endpgm) {
  Program out;
  if (!code)
    return out;
  uint32_t i = 0;
  while (i < max_dwords) {
    Inst in;
    in.pc = i;
    in.raw[0] = code[i];
    in.enc = Classify(code[i], in.opcode);
    in.size = BaseSize(in.enc, code[i]);
    if (i + in.size > max_dwords) {
      in.enc = Enc::kUnknown;
      in.opcode = 0;
      in.size = max_dwords - i;
      for (uint32_t d = 1; d < in.size; d++)
        in.raw[d] = code[i + d];
      out.push_back(in);
      break;
    }
    for (uint32_t d = 1; d < in.size; d++)
      in.raw[d] = code[i + d];
    if (in.enc == Enc::kMtbuf)
      in.opcode |= ((in.raw[1] >> 21) & 1) << 3;

    // Trailing 32-bit literal (the 1-dword ALU encodings, and VOP3 when a
    // source selects LITERAL_CONST). VOP2 madmk/madak/fmamk/fmaak always carry
    // a K.
    bool lit = false;
    switch (in.enc) {
      case Enc::kSop2:
      case Enc::kSopc:
        lit = Scalar2HasLit(code[i]);
        break;
      case Enc::kSop1:
        lit = Scalar1HasLit(code[i]);
        break;
      case Enc::kSopk:
        lit = in.opcode == 0x15;  // s_setreg_imm32_b32
        break;
      case Enc::kVop2:
        in.extension = ValuExtension(in.enc, code[i]);
        lit = in.extension != gpu::gcn::InstExtension::kNone ||
              in.opcode == 0x20 || in.opcode == 0x21 || in.opcode == 0x2C ||
              in.opcode == 0x2D || in.opcode == 0x37 || in.opcode == 0x38;
        break;
      case Enc::kVop1:
      case Enc::kVopc:
        in.extension = ValuExtension(in.enc, code[i]);
        lit = in.extension != gpu::gcn::InstExtension::kNone;
        break;
      case Enc::kVop3:
      case Enc::kVop3p:
        // Only architecturally used sources can select the shared literal.
        for (uint32_t src = 0; src < Vop3SourceCount(in.enc, in.opcode); src++)
          lit |= ((in.raw[1] >> (src * 9)) & 0x1ff) == 255;
        break;
      default:
        break;
    }
    if (lit && i + in.size < max_dwords) {
      const uint32_t extra = code[i + in.size];
      if (in.extension != gpu::gcn::InstExtension::kNone &&
          in.extension != gpu::gcn::InstExtension::kLiteral) {
        in.raw[in.size] = extra;
      } else {
        in.has_literal = true;
        in.literal = extra;
        in.extension = gpu::gcn::InstExtension::kLiteral;
      }
      in.size += 1;
    } else if (lit) {
      in.enc = Enc::kUnknown;
      in.opcode = 0;
    }
    if (in.size == 0)
      in.size = 1;  // never stall

    out.push_back(in);

    if (in.enc == Enc::kSopp) {
      // s_code_end is a hard executable-code marker. Ordinary and ordered PS
      // endpgm operations stop an unbounded scan, but a footer-bounded decode
      // retains later branch targets.
      if (in.opcode == 0x1F ||
          (stop_at_endpgm &&
           (in.opcode == 0x01 || in.opcode == 0x1B || in.opcode == 0x1E)))
        break;
    }
    i += in.size;
  }
  return out;
}

Program DecodeShader(const uint32_t* code, uint32_t max_dwords) {
  uint32_t len = CodeLength(code, max_dwords);
  if (len && len <= max_dwords)
    return Decode(code, len, /*stop_at_endpgm=*/false);
  return Decode(code, max_dwords, /*stop_at_endpgm=*/true);
}

Program ReachableProgram(const Program& program) {
  const std::vector<uint8_t> reachable = ComputeRdnaReachability(program);
  Program out;
  out.reserve(program.size());
  for (uint32_t i = 0; i < program.size(); i++)
    if (reachable[i])
      out.push_back(program[i]);
  return out;
}

}  // namespace gpu::rdna
