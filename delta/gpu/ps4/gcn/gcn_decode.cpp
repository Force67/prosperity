/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN GFX6/7 instruction decoder. See gcn_decode.h.
 */

#include "gcn_decode.h"

#include <cstdio>

namespace gpu::gcn {
namespace {

// The 'OrbShdr' ShaderBinaryInfo signature at byte offset `off` in `code`?
bool orbShdrAt(const uint32_t *code, uint32_t off) {
  const char *s = reinterpret_cast<const char *>(code) + off;
  return s[0] == 'O' && s[1] == 'r' && s[2] == 'b' && s[3] == 'S' &&
         s[4] == 'h' && s[5] == 'd' && s[6] == 'r';
}

// A 32-bit-encoded scalar/vector op carries a trailing 32-bit literal when a
// source-operand field selects LITERAL_CONST (255).
bool sop2HasLit(uint32_t w) {
  uint32_t s0 = w & 0xFF, s1 = (w >> 8) & 0xFF;
  return s0 == 255 || s1 == 255;
}
bool sop1HasLit(uint32_t w) { return (w & 0xFF) == 255; }
bool sopcHasLit(uint32_t w) {
  uint32_t s0 = w & 0xFF, s1 = (w >> 8) & 0xFF;
  return s0 == 255 || s1 == 255;
}
// VOP src0 is 9 bits [8:0]; 255 selects a literal.
bool vopHasLit(uint32_t w) { return (w & 0x1FF) == 255; }

Enc classify(uint32_t w, uint32_t &opcode) {
  if ((w >> 30) == 0x2) {  // 10b => scalar
    uint32_t top9 = w >> 23;
    if (top9 == 0x17F) { opcode = (w >> 16) & 0x7F; return Enc::sopp; }
    if (top9 == 0x17E) { opcode = (w >> 16) & 0x7F; return Enc::sopc; }
    if (top9 == 0x17D) { opcode = (w >> 8) & 0xFF; return Enc::sop1; }
    if ((w >> 28) == 0xB) { opcode = (w >> 23) & 0x1F; return Enc::sopk; }
    opcode = (w >> 23) & 0x7F; return Enc::sop2;
  }
  if ((w >> 27) == 0x18) { opcode = (w >> 22) & 0x1F; return Enc::smrd; }
  if ((w >> 26) == 0x34) { opcode = (w >> 17) & 0x1FF; return Enc::vop3; }
  if ((w >> 26) == 0x32) { opcode = (w >> 16) & 0x3; return Enc::vintrp; }
  if ((w >> 26) == 0x36) { opcode = (w >> 18) & 0xFF; return Enc::ds; }
  if ((w >> 26) == 0x38) { opcode = (w >> 18) & 0x7F; return Enc::mubuf; }
  if ((w >> 26) == 0x3A) { opcode = (w >> 16) & 0x7; return Enc::mtbuf; }
  if ((w >> 26) == 0x3C) { opcode = (w >> 18) & 0x7F; return Enc::mimg; }
  if ((w >> 26) == 0x3E) { opcode = (w >> 16) & 0x3F; return Enc::exp; }
  if ((w >> 25) == 0x3F) { opcode = (w >> 9) & 0xFF; return Enc::vop1; }
  if ((w >> 25) == 0x3E) { opcode = (w >> 17) & 0xFF; return Enc::vopc; }
  if ((w >> 31) == 0x0) { opcode = (w >> 25) & 0x3F; return Enc::vop2; }
  opcode = 0;
  return Enc::unknown;
}

// Dwords occupied (excluding any trailing literal).
uint32_t baseSize(Enc e) {
  switch (e) {
  case Enc::vop3:
  case Enc::ds:
  case Enc::mubuf:
  case Enc::mtbuf:
  case Enc::mimg:
  case Enc::exp:
    return 2;
  default:
    return 1;
  }
}

}  // namespace

uint32_t codeLength(const uint32_t *code, uint32_t maxDwords) {
  if (!code || maxDwords < 2)
    return 0;
  // Fast path: the toolchain emits "s_mov_b32 vcc_hi, #imm" (0xBEEB03FF) as the
  // first instruction, where the ShaderBinaryInfo footer sits at code[(imm+1)*2].
  if (code[0] == 0xBEEB03FFu) {
    uint64_t d = (uint64_t)(code[1] + 1) * 2;
    if (d >= 2 && d + 2 <= maxDwords && orbShdrAt(code, (uint32_t)d * 4))
      return (uint32_t)d;
  }
  // General case: scan (dword-aligned) for the footer signature. The GCN code ends
  // exactly where its footer begins, so the footer's dword offset is the length.
  for (uint32_t d = 1; d + 2 <= maxDwords; d++)
    if (orbShdrAt(code, d * 4))
      return d;
  return 0;
}

std::vector<Inst> decode(const uint32_t *code, uint32_t maxDwords,
                         bool stopAtEndpgm) {
  std::vector<Inst> out;
  if (!code)
    return out;
  uint32_t i = 0;
  while (i < maxDwords) {
    Inst in;
    in.pc = i;
    in.raw[0] = code[i];
    in.enc = classify(code[i], in.opcode);
    in.size = baseSize(in.enc);
    if (in.size == 2 && i + 1 < maxDwords)
      in.raw[1] = code[i + 1];

    // Trailing 32-bit literal for the 1-dword ALU encodings.
    bool lit = false;
    switch (in.enc) {
    case Enc::sop2: lit = sop2HasLit(code[i]); break;
    case Enc::sop1: lit = sop1HasLit(code[i]); break;
    case Enc::sopc: lit = sopcHasLit(code[i]); break;
    case Enc::vop2:
      // V_MADMK_F32 (0x20) and V_MADAK_F32 (0x21) always carry a trailing 32-bit
      // literal (the K constant), independent of the src0=LITERAL_CONST signalling.
      lit = vopHasLit(code[i]) || in.opcode == 0x20 || in.opcode == 0x21;
      break;
    case Enc::vop1:
    case Enc::vopc: lit = vopHasLit(code[i]); break;
    default: break;
    }
    if (lit && i + in.size < maxDwords) {
      in.hasLiteral = true;
      in.literal = code[i + in.size];
      in.size += 1;
    }
    if (in.size == 0)
      in.size = 1;  // safety: never stall

    out.push_back(in);

    // s_endpgm (SOPP opcode 1) terminates a basic block. When bounded by the real
    // code length it is not an end-of-stream marker, so keep decoding: a block
    // reached only after an early-out s_endpgm must still be lifted.
    if (stopAtEndpgm && in.enc == Enc::sopp && in.opcode == 1)
      break;
    i += in.size;
  }
  return out;
}

const char *mnemonic(const Inst &i) {
  switch (i.enc) {
  case Enc::sop1:
    switch (i.opcode) {
    case 0x03: return "s_mov_b32";
    case 0x04: return "s_mov_b64";
    default: return "s_op1";
    }
  case Enc::sop2:
    switch (i.opcode) {
    case 0x00: return "s_add_u32";
    case 0x02: return "s_sub_u32";
    default: return "s_op2";
    }
  case Enc::sopp:
    switch (i.opcode) {
    case 0x00: return "s_nop";
    case 0x01: return "s_endpgm";
    case 0x0c: return "s_waitcnt";
    default: return "s_opp";
    }
  case Enc::sopk: return "s_movk/sopk";
  case Enc::sopc: return "s_cmp";
  case Enc::smrd:
    switch (i.opcode) {
    case 0x00: return "s_load_dword";
    case 0x01: return "s_load_dwordx2";
    case 0x02: return "s_load_dwordx4";
    case 0x03: return "s_load_dwordx8";
    case 0x04: return "s_load_dwordx16";
    case 0x08: return "s_buffer_load_dword";
    case 0x0a: return "s_buffer_load_dwordx4";
    case 0x0c: return "s_buffer_load_dwordx16";
    default: return "smrd";
    }
  case Enc::vop1: return "v_op1";
  case Enc::vop2: return "v_op2";
  case Enc::vop3: return "v_op3";
  case Enc::vopc: return "v_cmp";
  case Enc::vintrp: return "v_interp";
  case Enc::ds: return "ds";
  case Enc::mubuf: return "mubuf";
  case Enc::mtbuf: return "tbuffer";
  case Enc::mimg: return "image";
  case Enc::exp: return "exp";
  default: return "?";
  }
}

void disassemble(const uint32_t *code, uint32_t maxDwords, const char *tag) {
  auto insts = decode(code, maxDwords);
  std::fprintf(stderr, "[gcn] %s: %zu instructions\n", tag, insts.size());
  for (const auto &in : insts) {
    std::fprintf(stderr, "[gcn]   %04x: %08x", in.pc, in.raw[0]);
    if (in.size >= 2) std::fprintf(stderr, " %08x", in.raw[1]);
    std::fprintf(stderr, "  %s op=%#x", mnemonic(in), in.opcode);
    if (in.hasLiteral) std::fprintf(stderr, " lit=%#x", in.literal);
    std::fprintf(stderr, "\n");
  }
}

}  // namespace gpu::gcn
