/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) instruction decoder. See rdna_decode.h. Produces the shared
 * gpu::gcn::Inst representation; the RDNA2-specific opcode semantics are applied
 * by the dispatch in rdna_translate.cpp.
 */

#include "rdna_decode.h"

#include <cstdio>

namespace gpu::rdna {
namespace {

// The 'OrbShdr'-style ShaderBinaryInfo signature at byte offset `off`. The AGC
// toolchain reuses the same 7-byte-signature footer scheme; we only need its
// position (== code length), not its bytes, so match either the PS4 magic or a
// generic printable 7-byte token beginning with a capital letter.
bool binaryInfoAt(const uint32_t* code, uint32_t off) {
  const char* s = reinterpret_cast<const char*>(code) + off;
  // PS4/Orbis: "OrbShdr". PS5 keeps a 7-byte signature; accept the Orbis magic
  // (still emitted by many AGC blobs) as the reliable anchor.
  return s[0] == 'O' && s[1] == 'r' && s[2] == 'b' && s[3] == 'S' &&
         s[4] == 'h' && s[5] == 'd' && s[6] == 'r';
}

// A source-operand field selects a trailing 32-bit literal when it == 255.
// Scalar sources are 8-bit ([7:0] / [15:8]); vector src0 is 9-bit ([8:0]).
bool scalar2HasLit(uint32_t w) {
  return (w & 0xFF) == 255 || ((w >> 8) & 0xFF) == 255;
}
bool scalar1HasLit(uint32_t w) { return (w & 0xFF) == 255; }
// VALU src0 (9-bit [8:0]) encodings that append an extra dword: 255 selects a
// 32-bit literal, 249 selects an SDWA control word, 250 selects a DPP control
// word. Missing the SDWA/DPP dword desyncs the decoder (the control word is then
// read as the next instruction) -- 249/250 are reserved src0 values, never real
// operands, so treating them as size-extending is always correct (matches
// published gfx10.3 references, which decode SDWA/DPP as 2-word instructions).
bool valuSrc0Extra(uint32_t w) {
  const uint32_t s = w & 0x1FF;
  return s == 255 || s == 249 || s == 250;
}

// Family classification, mirroring the gfx10.3 encoding-family dispatch. The
// VALU branch (bit31 == 0) further resolves VOP1/VOPC from the VOP2 opcode
// field; we resolve it here so the shared translator sees the right Enc.
Enc classify(uint32_t w, uint32_t& opcode) {
  if ((w & 0x80000000u) == 0u) {  // VALU 32-bit: VOP2, or VOP1/VOPC via op field
    uint32_t vop2Op = (w >> 25) & 0x3F;
    if (vop2Op == 0x3F) {  // VOP1
      opcode = (w >> 9) & 0xFF;
      return Enc::kVop1;
    }
    if (vop2Op == 0x3E) {  // VOPC
      opcode = (w >> 17) & 0xFF;
      return Enc::kVopc;
    }
    opcode = vop2Op;
    return Enc::kVop2;
  }
  if ((w & 0xC0000000u) == 0x80000000u) {  // scalar
    uint32_t sub = (w >> 23) & 0x7F;
    if (sub == 0x7D) { opcode = (w >> 8) & 0xFF; return Enc::kSop1; }
    if (sub == 0x7E) { opcode = (w >> 16) & 0x7F; return Enc::kSopc; }
    if (sub == 0x7F) { opcode = (w >> 16) & 0x7F; return Enc::kSopp; }
    if (sub >= 0x60) { opcode = (w >> 23) & 0x1F; return Enc::kSopk; }
    opcode = (w >> 23) & 0x7F;
    return Enc::kSop2;
  }
  switch (w >> 26) {  // top bits 11
    case 0x32: opcode = (w >> 16) & 0x3; return Enc::kVintrp;
    // VOP3P (packed 16-bit math) shares the 2-dword shape but has different
    // semantics; tag it above the 10-bit VOP3 opcode space so the translator
    // warns rather than misinterpreting it as a scalar VOP3 op.
    case 0x33: opcode = ((w >> 16) & 0x7F) | 0x400; return Enc::kVop3;
    case 0x35: opcode = (w >> 16) & 0x3FF; return Enc::kVop3;  // 10-bit opcode
    case 0x36: opcode = (w >> 18) & 0xFF; return Enc::kDs;
    case 0x37: opcode = (w >> 18) & 0x7F; return Enc::kMubuf;  // FLAT/GLOBAL/SCRATCH -> buffer slot
    case 0x38: opcode = ((w >> 18) & 0x7F) | (((w >> 25) & 1) << 7); return Enc::kMubuf;
    // MTBUF: the opcode MSB actually lives in word1[21] (not word0); classify()
    // only sees word0, so this opcode is approximate. Harmless while MTBUF is not
    // emitted (RdnaEmitInst warns on it); revisit if graphics MTBUF is added.
    case 0x3A: opcode = ((w >> 16) & 0x7) | (((w >> 21) & 1) << 3); return Enc::kMtbuf;
    case 0x3C: opcode = ((w >> 18) & 0x7F) | ((w & 1) << 7); return Enc::kMimg;
    case 0x3D: opcode = (w >> 18) & 0xFF; return Enc::kSmrd;  // SMEM (replaces GCN SMRD)
    // RDNA2 EXP is 0b111110 (0xf8 prefix), same slot as GCN; target/en live in
    // [9:4]/[3:0] (see EmitExport). NGG streams also emit a null export at
    // 0b110001 (en=0, a no-op) -- route it here too so it is not "unsupported".
    case 0x31:
    case 0x3E: opcode = (w >> 4) & 0x3F; return Enc::kExp;
    default: opcode = 0; return Enc::kUnknown;
  }
}

// Base dword count (excluding any trailing literal). RDNA2 two-dword encodings:
// VOP3/VOP3P, SMEM, DS, MUBUF/MTBUF, FLAT, MIMG, EXP. MIMG adds NSA words.
uint32_t baseSize(Enc e, uint32_t w) {
  switch (e) {
    case Enc::kVop3:
    case Enc::kSmrd:  // SMEM is 2 dwords on RDNA2 (was 1 on GCN SMRD)
    case Enc::kDs:
    case Enc::kMubuf:
    case Enc::kMtbuf:
    case Enc::kExp:
      return 2;
    case Enc::kMimg: {
      uint32_t nsa = (w >> 1) & 0x3;  // NSA (non-sequential address) extra dwords
      return 2 + nsa;
    }
    default:
      return 1;
  }
}

}  // namespace

uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords) {
  if (!code || max_dwords < 2) return 0;
  // Fast path: the toolchain emits "s_mov_b32 vcc_hi/null, #imm" (0xBEEB03FF) as
  // the first instruction, with the ShaderBinaryInfo footer at code[(imm+1)*2].
  if (code[0] == 0xBEEB03FFu) {
    uint64_t d = static_cast<uint64_t>(code[1] + 1) * 2;
    if (d >= 2 && d + 2 <= max_dwords && binaryInfoAt(code, static_cast<uint32_t>(d) * 4))
      return static_cast<uint32_t>(d);
  }
  for (uint32_t d = 1; d + 2 <= max_dwords; d++)
    if (binaryInfoAt(code, d * 4)) return d;
  return 0;
}

Program Decode(const uint32_t* code, uint32_t max_dwords, bool stop_at_endpgm) {
  Program out;
  if (!code) return out;
  uint32_t i = 0;
  while (i < max_dwords) {
    Inst in;
    in.pc = i;
    in.raw[0] = code[i];
    in.enc = classify(code[i], in.opcode);
    in.size = baseSize(in.enc, code[i]);
    if (in.size >= 2 && i + 1 < max_dwords) in.raw[1] = code[i + 1];

    // Trailing 32-bit literal (the 1-dword ALU encodings, and VOP3 when a source
    // selects LITERAL_CONST). VOP2 madmk/madak/fmamk/fmaak always carry a K.
    bool lit = false;
    switch (in.enc) {
      case Enc::kSop2:
      case Enc::kSopc: lit = scalar2HasLit(code[i]); break;
      case Enc::kSop1: lit = scalar1HasLit(code[i]); break;
      case Enc::kVop2:
        lit = valuSrc0Extra(code[i]) || in.opcode == 0x20 || in.opcode == 0x21 ||
              in.opcode == 0x2C || in.opcode == 0x2D;
        break;
      case Enc::kVop1:
      case Enc::kVopc: lit = valuSrc0Extra(code[i]); break;
      case Enc::kVop3:
        // A src field == 255 selects a literal after the two base words.
        if (i + 1 < max_dwords) {
          uint32_t w1 = code[i + 1];
          lit = (w1 & 0x1FF) == 255 || ((w1 >> 9) & 0x1FF) == 255 ||
                ((w1 >> 18) & 0x1FF) == 255;
        }
        break;
      default: break;
    }
    if (lit && i + in.size < max_dwords) {
      in.has_literal = true;
      in.literal = code[i + in.size];
      in.size += 1;
    }
    if (in.size == 0) in.size = 1;  // never stall

    out.push_back(in);

    // s_endpgm (SOPP opcode 1) terminates the stream unless bounded by a real
    // length (then a block after an early-out endpgm is still decoded).
    if (stop_at_endpgm && in.enc == Enc::kSopp && in.opcode == 1) break;
    i += in.size;
  }
  return out;
}

Program DecodeShader(const uint32_t* code, uint32_t max_dwords) {
  uint32_t len = CodeLength(code, max_dwords);
  if (len && len <= max_dwords) return Decode(code, len, /*stop_at_endpgm=*/false);
  return Decode(code, max_dwords, /*stop_at_endpgm=*/true);
}

}  // namespace gpu::rdna
