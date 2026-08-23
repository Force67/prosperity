#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + tracking: the gfx10.3 128/256-bit T# layout
 * and the per-draw resolution of the T#/S# a pixel shader samples. Kept out of
 * the SPIR-V-gated translator so the command processor can resolve textures
 * even in a build without the recompiler backend.
 */

#include "base/arch.h"
#include <unordered_map>
#include <vector>

#include "gpu/gcn/gcn_resource.h"
#include "gpu/ps5/rdna/rdna_decode.h"

namespace gpu::rdna {

struct Smem {
  u32 op;
  u32 sdst;
  u32 sbase;
  u32 soffset;
  i32 offset;
};

struct BufferResource {
  u64 base = 0;
  // 4 dwords for a V#, 8 for an image T#; descriptor_dwords says which.
  u32 descriptor[8] = {};
  u32 descriptor_dwords = 0;
  bool descriptor_valid = false;
  // Byte offset the fetch adds on top of the descriptor. For a structured V#
  // the hardware computes base + soffset + index*stride, and Sony's compiler
  // uses it to place each attribute inside the vertex: the three attributes of
  // one stream share a base and differ only here.
  u32 soffset = 0;
  bool soffset_valid = false;
};

inline Smem DecodeSmem(const gpu::gcn::Inst& inst) {
  return {
      .op = inst.opcode,
      .sdst = (inst.raw[0] >> 6) & 0x7F,
      .sbase = (inst.raw[0] & 0x3F) * 2,
      .soffset = (inst.raw[1] >> 25) & 0x7F,
      .offset = static_cast<i32>(inst.raw[1] << 11) >> 11,
  };
}

// Dwords an SMEM s_load / s_buffer_load reads (x1/x2/x4/x8/x16). 0 = not a
// load.
inline u32 SmemLoadCount(u32 op) {
  switch (op) {
    case 0x00:
    case 0x08:
      return 1;
    case 0x01:
    case 0x09:
      return 2;
    case 0x02:
    case 0x0A:
      return 4;
    case 0x03:
    case 0x0B:
      return 8;
    case 0x04:
    case 0x0C:
      return 16;
    default:
      return 0;
  }
}

struct ScalarWrite {
  u32 first = 0;
  u32 count = 0;
};

// SGPRs an SOP1 writes, from the gfx10 opcode table (LLVM SOPInstructions.td
// SOP1_Real_gfx10). RDNA renumbered the whole block: the 64-bit saveexec family
// GFX7 puts at 0x24-0x2b is still there, but gfx10 added the wave32 (b32) forms
// at 0x3c-0x47 and four more 64-bit ones at 0x37-0x3b -- including
// s_andn1_saveexec_b64, which Demon's Souls' G-buffer shaders use. Reading one
// of those as a single-dword write leaves the second SGPR looking like live
// user data, which is exactly how a stale value reaches a descriptor.
// Everything unrecognised counts as a pair for the same reason.
inline u32 Sop1WriteDwords(u32 op) {
  switch (op) {
    case 0x03:  // s_mov_b32
    case 0x05:  // s_cmov_b32
    case 0x07:  // s_not_b32
    case 0x09:  // s_wqm_b32
    case 0x0b:  // s_brev_b32
    case 0x0d:  // s_bcnt0_i32_b32
    case 0x0e:  // s_bcnt0_i32_b64   (64-bit source, 32-bit result)
    case 0x0f:  // s_bcnt1_i32_b32
    case 0x10:  // s_bcnt1_i32_b64
    case 0x11:  // s_ff0_i32_b32
    case 0x12:  // s_ff0_i32_b64
    case 0x13:  // s_ff1_i32_b32
    case 0x14:  // s_ff1_i32_b64
    case 0x15:  // s_flbit_i32_b32
    case 0x16:  // s_flbit_i32_b64
    case 0x17:  // s_flbit_i32
    case 0x18:  // s_flbit_i32_i64
    case 0x19:  // s_sext_i32_i8
    case 0x1a:  // s_sext_i32_i16
    case 0x1b:  // s_bitset0_b32
    case 0x1d:  // s_bitset1_b32
    case 0x2c:  // s_quadmask_b32
    case 0x2e:  // s_movrels_b32
    case 0x30:  // s_movreld_b32
    case 0x34:  // s_abs_i32
    case 0x3c:  // s_and_saveexec_b32 .. s_andn2_wrexec_b32
    case 0x3d:
    case 0x3e:
    case 0x3f:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x49:  // s_movrelsd_2_b32
      return 1;
    case 0x20:  // s_setpc_b64 writes no SGPR at all
    case 0x22:  // s_rfe_b64
      return 0;
    default:
      return 2;
  }
}

inline ScalarWrite DecodeScalarWrite(const gpu::gcn::Inst& inst) {
  using gpu::gcn::Enc;
  if (inst.enc == Enc::kSmrd) {
    const u32 sdst = DecodeSmem(inst).sdst;
    return sdst == 125 ? ScalarWrite{}
                       : ScalarWrite{.first = sdst,
                                     .count = SmemLoadCount(inst.opcode)};
  }
  if (inst.enc == Enc::kSopk) {
    const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
    if (sdst == 125)
      return {};
    if (inst.opcode == 0x00 || inst.opcode == 0x02 ||
        (inst.opcode >= 0x0F && inst.opcode <= 0x10) || inst.opcode == 0x12)
      return {.first = sdst, .count = 1};
    return {};
  }
  if (inst.enc == Enc::kSop1) {
    const u32 count = Sop1WriteDwords(inst.opcode);
    const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
    return sdst == 125 || !count
               ? ScalarWrite{}
               : ScalarWrite{.first = sdst, .count = count};
  }
  if (inst.enc == Enc::kSop2) {
    const bool wide =
        inst.opcode == 0x0B || inst.opcode == 0x0F || inst.opcode == 0x11 ||
        inst.opcode == 0x13 || inst.opcode == 0x15 || inst.opcode == 0x17 ||
        inst.opcode == 0x19 || inst.opcode == 0x1B || inst.opcode == 0x1D ||
        inst.opcode == 0x1F || inst.opcode == 0x21 || inst.opcode == 0x23 ||
        inst.opcode == 0x29;
    const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
    return sdst == 125 ? ScalarWrite{}
                       : ScalarWrite{.first = sdst, .count = wide ? 2u : 1u};
  }
  return {};
}

// Every SGPR an instruction may write, over-approximated, with `exact` set when
// the dispatch-time replay (ScalarEval) either reproduces the value or clears
// the register. DecodeScalarWrite above reports only the scalar ALU's exact
// destinations; this one also reports the vector ops that land in an SGPR and
// the wider forms, which is what stops a register the shader clobbered from
// passing for user data. scc_trusted says whether SCC itself is still faithful,
// which is what makes s_cselect exact.
struct ScalarWrites {
  struct Range {
    u32 first = 0;
    u32 count = 0;
    bool exact = false;
  };
  Range range[2];
};
ScalarWrites PossibleScalarWrites(const gpu::gcn::Inst& inst,
                                  bool scc_trusted = false);

// SGPRs whose live value ResolveBuffers reproduces faithfully. The replay walks
// the instruction list once and in order, so what it holds at a use is what the
// LAST write before that use left behind. That is the wave's value only when
// that write ran on every path to the use, ran exactly once, and is an
// operation ScalarEval models. A compute dispatch writes guest memory, so a
// descriptor built out of a register that fails any of the three is declined
// rather than guessed -- and which one it failed is reported, so the gap is
// visible rather than just a count.
struct ScalarReplayPlan {
  static constexpr u32 kRegs = 136;
  enum Loss : u8 {
    kCovered = 0,
    kUnmodelled,   // written by an op ScalarEval does not evaluate
    kConditional,  // written under a branch that a path around it skips
    kLoopBody,     // written inside a loop, so the wave ran it more than once
    kLoopCarried,  // a later write a back edge can run before the use
  };
  struct Write {
    u32 index;
    u32 first;
    u32 count;
    bool exact;
    // Inside a loop, does this write produce the same value on every
    // iteration? A descriptor s_loaded from a user-data table at a constant
    // offset does, and that is most of what a loop body holds; only a write
    // whose own inputs the loop changes is beyond the replay.
    bool loop_invariant;
  };
  // Instruction indices. A back edge is a branch whose target precedes it.
  struct BackEdge {
    u32 target;
    u32 source;
  };
  std::vector<Write> writes;
  std::vector<BackEdge> back_edges;
  std::vector<u32> targets;
  bool indirect = false;  // s_setpc: control may reach anywhere from anywhere
  // Basic blocks and their dominator tree. "A branch lands between the write
  // and the use" is not the question -- a descriptor is routinely built in one
  // arm of an if and used at the join, and every path still runs the write.
  // The question is whether the write DOMINATES the use, which is what this
  // answers. Astro Bot's frame is largely compute, and the blunt test declined
  // its two biggest passes.
  static constexpr u32 kNoBlock = ~0u;
  std::vector<u32> block_of;  // instruction index -> block id
  std::vector<u32> idom;      // block id -> immediate dominator (entry: self)
  bool Dominates(u32 write_index, u32 use_index) const;

  // Does the replay hold the live value of sgpr[dwords] at instruction
  // use_index, and if not, why not?
  Loss LossAt(u32 sgpr, u32 dwords, u32 use_index) const;
  bool Covers(u32 sgpr, u32 dwords, u32 use_index) const {
    return LossAt(sgpr, dwords, use_index) == kCovered;
  }
};

ScalarReplayPlan PlanScalarReplay(const Program& program);

// MIMGs reading the same T#/S# descriptor share one set-0 binding, in first-use
// order. Both the recompiler (EmitMimg declarations) and TrackTextures pair
// against this, so the resolved textures line up 1:1 with the shader's
// samplers.
gpu::gcn::MimgBindingPlan RdnaPlanMimg(const Program& program);

// Decode a gfx10.3 image resource. R128 selects the compact four-dword form;
// ordinary resources contain eight dwords.
gpu::gcn::TImage DecodeTImage(const u32* dwords, bool r128 = false);

// A decoded gfx10.3 buffer resource (V#, four dwords): base48 = w0 |
// (w1[15:0] << 32), stride w1[29:16], num_records w2, format w3[18:12].
struct VBuffer {
  u64 base = 0;
  u32 stride = 0;
  u32 num_records = 0;
  // The GCN (data, number) format pair the shared renderer speaks, mapped from
  // `gfmt` -- which is what the descriptor actually encodes.
  u32 dfmt = 0;
  u32 nfmt = 0;
  u32 gfmt = 0;
};

// gfx10.3 buffer V#s carry a UNIFIED 7-bit format enum where GCN has separate
// data and number formats, so every descriptor and every typed fetch that
// reaches the shared renderer has to be mapped back onto the GCN pair. Only the
// vertex-attribute formats are covered; unknowns yield (0, 0), which is
// harmless for descriptors that never reach VertexFormat().
void DecodeBufferFormat(u32 gfmt, u32& dfmt, u32& nfmt);

VBuffer DecodeVBuffer(const u32* dwords);

// A V# read from an unbound or stale SGPR slot decodes to an in-range but bogus
// buffer: a depth-only pre-pass with an inactive vertex slot yielded stride
// 14915 over 480622080 records, which then segfaulted reading the vertex ring.
// Mirrors the PS4 fetch-shader sanity gate (gpu/gcn/gcn_resource.cc).
bool PlausibleVBuffer(const VBuffer& v);

// Resolve the live T#/S# each MIMG in a pixel shader samples, in binding order.
// user_sgprs is how many user-data SGPRs the stage was launched with
// (SPI_SHADER_PGM_RSRC2_*.USER_SGPR): a descriptor inline beyond that window is
// not user data at all, just whatever the previous draw left in those
// registers.
// ud_base is the SGPR the stage's user data starts at: 0 for a PS, 8 for the
// merged NGG stage a vertex program runs as.
std::vector<gpu::gcn::TImage> TrackTextures(const u32* ps_code,
                                            const u32* ps_user_data,
                                            u32 user_sgprs,
                                            u32 ud_base = 0);

// Resolve buffer bases and complete V#s at their consuming instruction PCs.
std::unordered_map<u32, BufferResource> ResolveBuffers(
    const u32* code,
    const u32* user_data,
    u32 user_sgprs,
    u32 user_sgpr_base = 0);

}  // namespace gpu::rdna
