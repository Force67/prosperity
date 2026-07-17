#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN (Sea Islands / GFX7, the PS4 Liverpool ISA) instruction decoder. Decodes
 * the variable-length 32-bit instruction stream into a flat instruction list
 * (`Program`) that the resource tracker and the SPIR-V translator consume.
 *
 * Instruction-family constants and operand field layouts:
 * https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
 * https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h
 */

#include <cstdint>
#include <memory>
#include <vector>

namespace gpu::gcn {

// Instruction encoding families (top-bit dispatch). Determines operand layout
// and whether a 32-bit literal/inline constant dword follows.
enum class Enc : uint8_t {
  kUnknown,
  kSop1, kSop2, kSopk, kSopc, kSopp,  // scalar ALU
  kSmrd,                              // scalar memory (loads V#/T# tables)
  kVop1, kVop2, kVop3, kVopc,         // vector ALU
  kVintrp,                            // interpolation
  kDs,                                // LDS/GDS
  kMubuf, kMtbuf,                     // buffer load/store (vertex fetch via V#)
  kMimg,                              // image sample/load/store (uses T#/S#)
  kExp,                               // export (PS color / VS position out)
};

struct Inst {
  Enc enc = Enc::kUnknown;
  uint32_t opcode = 0;       // encoding-relative opcode
  uint32_t raw[2] = {0, 0};  // up to 2 dwords (some need a literal)
  uint32_t size = 1;         // length in dwords (incl. literal)
  uint32_t pc = 0;           // dword offset within the program
  bool has_literal = false;
  uint32_t literal = 0;
};

// A decoded shader: the flat instruction list in program order.
using Program = std::vector<Inst>;

// Decode a GCN program. `code` points at the bytecode (guest, host-readable),
// `max_dwords` bounds the scan (use CodeLength() / the BinaryInfo length).
// s_endpgm is a basic-block terminator, not an end-of-stream marker: with
// stop_at_endpgm=false the whole program is decoded so blocks reached only
// after an early-out s_endpgm are still lifted. stop_at_endpgm=true stops at
// the first s_endpgm for callers without a reliable length bound.
Program Decode(const uint32_t* code, uint32_t max_dwords,
               bool stop_at_endpgm = true);

// Recover the real GCN code length (in dwords) from the trailing Gnm
// ShaderBinaryInfo ("OrbShdr") footer that the Orbis toolchain appends after
// the bytecode. Returns 0 if no footer is found within `max_dwords`.
uint32_t CodeLength(const uint32_t* code, uint32_t max_dwords);

// Decode a shader bounded by its real code length (from the OrbShdr footer) so
// an early-out s_endpgm does not truncate the stream. Falls back to the
// stop-at-first-endpgm scan when no footer is present (e.g. a driver-generated
// sub-shader), which never over-reads into the footer/padding.
Program DecodeShader(const uint32_t* code, uint32_t max_dwords);

// Shared, cached DecodeShader for per-draw analysis (resource tracking runs on
// every draw; decoding 4K dwords each time is measurable). The cache key is the
// guest address; entries revalidate against a hash of the code so an in-place
// shader rewrite is picked up. Returns a shared_ptr so entries stay valid even
// if the cache evicts. Not thread-safe: callers already serialize on the
// command-processor lock.
std::shared_ptr<const Program> CachedProgram(uint64_t addr,
                                             uint32_t max_dwords);

// Human-readable mnemonic for an instruction (best-effort; "?" for unmapped).
const char* Mnemonic(const Inst& inst);

// Disassemble to stderr (debug aid).
void Disassemble(const uint32_t* code, uint32_t max_dwords, const char* tag);

}  // namespace gpu::gcn
