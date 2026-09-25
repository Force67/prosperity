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

#include "base/arch.h"
#include <memory>
#include <vector>

namespace gpu::gcn {

enum class IsaMode : u8 { kBase, kNeo };

// Instruction encoding families (top-bit dispatch). Determines operand layout
// and whether a 32-bit literal/inline constant dword follows.
enum class Enc : u8 {
  kUnknown,
  kSop1,
  kSop2,
  kSopk,
  kSopc,
  kSopp,  // scalar ALU
  kSmrd,  // scalar memory (loads V#/T# tables)
  kVop1,
  kVop2,
  kVop3,
  kVop3p,
  kVopc,    // vector ALU
  kVintrp,  // interpolation
  kDs,      // LDS/GDS
  kMubuf,
  kMtbuf,  // buffer load/store (vertex fetch via V#)
  kMimg,   // image sample/load/store (uses T#/S#)
  kExp,    // export (PS color / VS position out)
  kFlat,   // RDNA flat/global/scratch memory
};

enum class InstExtension : u8 {
  kNone,
  kLiteral,
  kSdwa,
  kDpp,
  kDpp8,
  kDpp8Fi,
};

struct Inst {
  IsaMode isa = IsaMode::kBase;
  Enc enc = Enc::kUnknown;
  u32 opcode = 0;   // encoding-relative opcode
  u32 raw[5] = {};  // largest base encoding: RDNA MIMG with NSA=3
  u32 size = 1;     // length in dwords (incl. literal)
  u32 pc = 0;       // dword offset within the program
  bool has_literal = false;
  u32 literal = 0;
  InstExtension extension = InstExtension::kNone;
};

// A decoded shader: the flat instruction list in program order.
using Program = std::vector<Inst>;

// Where an SMRD load takes its offset from; the three forms do NOT share units, and
// the wrong stride resolves a T#/V# out of unrelated buffer bytes:
    // IMM=1              OFFSET = 8-bit DWORD offset.
    // IMM=0, OFFSET=0xFF trailing 32-bit literal, also DWORDs (Sea Islands wide form;
                       // LLVM encodes it through the same byte>>2 conversion). The ISA
                       // prose reads the literal as bytes, but implementing that measured
                       // no difference (SotC: 64 unresolved bindings either way); find a
                       // separating case before revisiting, texmiss won't show it.
    // IMM=0 otherwise    OFFSET = SGPR holding a BYTE offset.
struct SmrdOffset {
  u32 dwords = 0;    // resolved offset, when it is not in an SGPR
  u32 sgpr = 0;      // SGPR index carrying a byte offset
  bool in_sgpr = false;   // read `sgpr` instead of `dwords`
};

inline SmrdOffset DecodeSmrdOffset(const Inst& inst) {
  const u32 w = inst.raw[0], field = w & 0xFF;
  SmrdOffset o;
  if ((w >> 8) & 1)
    o.dwords = field;
  else if (field == 0xFF && inst.has_literal)
    o.dwords = inst.literal;
  else {
    o.in_sgpr = true;
    o.sgpr = field;
  }
  return o;
}

// Which scalar load last filled a descriptor's SGPRs, as that load's pc. A
// shader that reloads the same SGPR quad with a second V# reads a second
// resource through it, so the pair (base SGPR, version) is what names one.
// Linear program order, the same for every consumer of it.
class DescriptorVersions {
 public:
  static constexpr u32 kInline = 0xFFFFFFFFu;  // still the user data

  u32 Of(u32 sgpr, u32 dwords) const {
    for (auto it = loads_.rbegin(); it != loads_.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->pc;
    return kInline;
  }

  // Record an SMRD's destination; call after reading the version it consumes.
  void Note(const Inst& inst) {
    if (inst.enc != Enc::kSmrd)
      return;
    // s_load_dword{,x2..x16} 0x00-0x04, s_buffer_load_dword{,x2..x16} 0x08-0x0c.
    const u32 op = inst.opcode;
    const u32 n = op <= 0x04 ? 1u << op
                  : op >= 0x08 && op <= 0x0c ? 1u << (op - 0x08)
                                             : 0u;
    if (!n)
      return;
    const u32 sdst = (inst.raw[0] >> 15) & 0x7F;
    std::erase_if(loads_, [&](const Load& ld) {
      return sdst < ld.sgpr + ld.dwords && ld.sgpr < sdst + n;
    });
    loads_.push_back({sdst, n, inst.pc});
  }

 private:
  struct Load {
    u32 sgpr, dwords, pc;
  };
  std::vector<Load> loads_;
};

// Decode a GCN program (`code` guest bytecode, `max_dwords` bounds the scan; use
// CodeLength()/BinaryInfo). s_endpgm is a block terminator, not end-of-stream:
// stop_at_endpgm=false lifts blocks after an early-out endpgm too; true stops at
// the first, for callers without a reliable length bound.
IsaMode DefaultIsaMode();
void SetDefaultIsaMode(IsaMode mode);

Program Decode(const u32* code,
               u32 max_dwords,
               bool stop_at_endpgm = true,
               IsaMode mode = DefaultIsaMode());

// Recover the real GCN code length (in dwords) from the trailing Gnm
// ShaderBinaryInfo ("OrbShdr") footer that the Orbis toolchain appends after
// the bytecode. Returns 0 if no footer is found within `max_dwords`.
u32 CodeLength(const u32* code, u32 max_dwords);

// Decode a shader bounded by its real code length (from the OrbShdr footer) so
// an early-out s_endpgm does not truncate the stream. Falls back to the
// stop-at-first-endpgm scan when no footer is present (e.g. a driver-generated
// sub-shader), which never over-reads into the footer/padding.
Program DecodeShader(const u32* code,
                     u32 max_dwords,
                     IsaMode mode = DefaultIsaMode());

// Mark instructions reachable from the entry block. Shader binaries may
// contain footer padding after an early s_endpgm; decoded dead data must not
// influence translation or resource planning.
std::vector<u8> ComputeReachability(const Program& program);

// Shared, cached DecodeShader for per-draw analysis (decoding 4K dwords per draw is
// measurable). Keyed by guest address, revalidated against a code hash so an
// in-place rewrite is picked up; shared_ptr keeps entries valid past eviction.
// Not thread-safe: callers hold the command-processor lock.
std::shared_ptr<const Program> CachedProgram(u64 addr,
                                             u32 max_dwords);

// Content hash of the shader at `addr`, instructions only (footer body, or up to the
// terminator for runtime-generated shaders with no footer), so a cache key can name
// the CODE, not the scratch address it sits at this draw. Cached per address per
// generation, like CachedProgram.
u64 CachedCodeHash(u64 addr, u32 max_dwords);

// Does this program transfer to a fetch shader (s_swappc_b64)? The pointer parks in
// s[0:1], but s[0:1] holds something in every VS (SotC keeps its SRT pointer there),
// so the call is the only ground truth. Without it, constant data got hashed into
// the recompile key (95% of misses) and ParseFetch read phantom attributes.
bool CallsFetchShader(const Program& program);

// Advance the CachedProgram revalidation generation (once per frame, end-of-frame).
// Entries revalidate at most once per generation; within a frame, pure map hits.
// Uploads happen between frames, so a same-address rewrite is still caught.
void NextProgramCacheGeneration();

// Mnemonics and full disassembly live in gcn_disasm.h.

}  // namespace gpu::gcn
