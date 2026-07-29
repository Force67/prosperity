#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3 / Oberon, the PS5 AGC ISA) instruction decoder. Decodes the
 * variable-length 32-bit instruction stream into the SAME flat `gpu::gcn::Inst`
 * / `Program` representation the PS4 GCN path uses, so the shared SPIR-V
 * translator infrastructure (register-file model, ALU/scalar emitters,
 * control-flow lowering) can be reused. Only the encoding families and opcode
 * numbers are RDNA2-specific; see gpu/ps5/rdna/rdna_translate.cc for the
 * per-instruction dispatch that maps RDNA2 opcodes onto the shared emitters.
 *
 * Encoding-family dispatch and opcode tables verified against the a gfx10.3
 * reference gfx10.3 decoder (src/graphics/shader/recompiler/ShaderDecoder.cpp).
 */

#include <cstdint>

#include "gpu/ps4/gcn/gcn_decode.h"

namespace gpu::rdna {

// Reuse the GCN decoder's instruction/encoding representation: the encoding
// families line up (RDNA2 keeps the SOP*/VOP*/MIMG/EXP taxonomy), and the
// shared translator consumes gpu::gcn::Inst. RDNA2's SMEM family is recorded as
// gcn::Enc::kSmrd (the rdna dispatch decodes its RDNA2 fields).
using gpu::gcn::Enc;
using gpu::gcn::Inst;
using gpu::gcn::Program;

// Decode an RDNA2 program bounded by `max_dwords`. Without a reliable shader
// length, stop_at_endpgm stops at the first program-ending instruction. Pass
// false only when the caller has a real code bound and needs blocks after an
// early-out s_endpgm.
Program Decode(const uint32_t* code,
               uint32_t max_dwords,
               bool stop_at_endpgm = true);

// Recover the real code length (in dwords) from the trailing ShaderBinaryInfo
// footer the AGC toolchain appends. The locator matches PS4: the code begins
// with the sentinel dword 0xBEEB03FF (s_mov_b32 vcc_hi/null, #imm) whose imm
// gives the footer dword offset. Returns 0 if no footer is found.
uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords);

// Decode bounded by the recovered code length when a footer is present, else
// fall back to the stop-at-first-endpgm scan.
Program DecodeShader(const uint32_t* code, uint32_t max_dwords);

// Remove instructions in statically unreachable basic blocks while preserving
// original PCs for branch and resource-plan lookup.
Program ReachableProgram(const Program& program);

}  // namespace gpu::rdna
