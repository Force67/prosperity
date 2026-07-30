#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (GFX7 / Sea Islands) disassembler: full mnemonic tables + operand
 * rendering for every encoding the decoder classifies. Debug/diagnostic
 * surface only -- the translator never consumes disassembly.
 *
 * Opcode numbering follows the Sea Islands ISA (the decoder's encoding-
 * relative opcode fields):
 * https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
 */

#include <cstdint>
#include <string>

#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

// Mnemonic only (e.g. "v_mad_f32"). Unmapped opcodes render as
// "<enc>_op0x<n>" rather than failing, so unknown-op reports stay greppable.
std::string Mnemonic(const Inst& inst);

// Full one-line disassembly: mnemonic + rendered operands
// (e.g. "v_mac_f32 v3, s2, v1" or "image_sample v[0:3], v[4:5], s[8:15],
// s[16:19] dmask:0xf"). Best effort: operand fields are always printed from
// the encoding even when the opcode itself is unmapped.
std::string DisasmInst(const Inst& inst);

// Formatted listing line: "pc: raw-words  disassembly". `pc_width` pads the
// pc column (dword offsets).
std::string DisasmLine(const Inst& inst);

// Disassemble a code range to stderr (debug aid; decodes with the
// stop-at-endpgm heuristic exactly like the old raw dump did).
void Disassemble(const uint32_t* code, uint32_t max_dwords, const char* tag);

}  // namespace gpu::gcn
