#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (Sea Islands / GFX6-7, the PS4 Liverpool ISA) instruction decoder. Decodes
 * the variable-length 32-bit instruction stream into a flat instruction list we
 * can analyse (resource tracking) and translate to SPIR-V. This is the front end
 * of the shader recompiler.
 */

#include <cstdint>
#include <vector>

namespace gpu::gcn {

// Instruction encoding families (top-bit dispatch). Determines operand layout
// and whether a 32-bit literal/inline constant dword follows.
enum class Enc : uint8_t {
  unknown,
  sop1, sop2, sopk, sopc, sopp,  // scalar ALU
  smrd,                          // scalar memory (loads V#/T# tables)
  vop1, vop2, vop3, vopc,        // vector ALU
  vintrp,                        // interpolation
  ds,                            // LDS/GDS
  mubuf, mtbuf,                  // buffer load/store (vertex fetch via V#)
  mimg,                          // image sample (uses T#/S#)
  exp,                           // export (PS color / VS position out)
};

struct Inst {
  Enc enc = Enc::unknown;
  uint32_t opcode = 0;       // encoding-relative opcode
  uint32_t raw[2] = {0, 0};  // up to 2 dwords (some need a literal)
  uint32_t size = 1;         // length in dwords (incl. literal)
  uint32_t pc = 0;          // dword offset within the program
  bool hasLiteral = false;
  uint32_t literal = 0;
};

// Decode a GCN program. `code` points at the bytecode (guest, host-readable),
// `maxDwords` bounds the scan (use codeLength() / the BinaryInfo length).
// s_endpgm is a basic-block terminator, not an end-of-stream marker: with
// stopAtEndpgm=false the whole program is decoded so blocks reached only after an
// early-out s_endpgm are still lifted. stopAtEndpgm=true (default) keeps the
// legacy "stop at the first s_endpgm" behaviour for callers that want it.
std::vector<Inst> decode(const uint32_t *code, uint32_t maxDwords,
                         bool stopAtEndpgm = true);

// Recover the real GCN code length (in dwords) from the trailing Gnm
// ShaderBinaryInfo ("OrbShdr") footer that the Orbis toolchain appends after the
// bytecode. Returns 0 if no footer is found within `maxDwords`. Use this to bound
// decode() so a shader with an early s_endpgm is not truncated.
uint32_t codeLength(const uint32_t *code, uint32_t maxDwords);

// Human-readable mnemonic for an instruction (best-effort; "?" for unmapped).
const char *mnemonic(const Inst &i);

// Disassemble to stderr (debug aid).
void disassemble(const uint32_t *code, uint32_t maxDwords, const char *tag);

}  // namespace gpu::gcn
