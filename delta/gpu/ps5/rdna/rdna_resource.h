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
    if (inst.opcode == 0x20 || inst.opcode == 0x21)
      return {};
    const bool wide = inst.opcode == 0x04 || inst.opcode == 0x06 ||
                      inst.opcode == 0x08 || inst.opcode == 0x0A ||
                      (inst.opcode >= 0x24 && inst.opcode <= 0x2B);
    const u32 sdst = (inst.raw[0] >> 16) & 0x7F;
    return sdst == 125 ? ScalarWrite{}
                       : ScalarWrite{.first = sdst, .count = wide ? 2u : 1u};
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
// the instruction list once and in order, so it only tells the truth about the
// shader's unconditional prologue: a register written under a branch or inside
// a loop body keeps whatever value that single walk left behind, and one a
// vector op (or a scalar op ScalarEval does not model) wrote keeps a stale
// value instead of reading back as unknown. A compute dispatch writes guest
// memory, so a descriptor built out of an untrusted register is declined rather
// than guessed.
struct ScalarReplayPlan {
  static constexpr u32 kRegs = 136;
  static constexpr u32 kNever = 0xFFFFFFFFu;
  // Instruction index of the first write to each SGPR the replay loses, and
  // where its single linear walk stops matching execution (the first branch, or
  // the first instruction a back edge repeats).
  u32 first_lost[kRegs];
  u32 prefix_end = 0;
  // Does the replay hold the live value of sgpr[dwords] at instruction
  // use_index?
  bool Covers(u32 sgpr, u32 dwords, u32 use_index) const;
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

// Resolve the live T#/S# each MIMG in a pixel shader samples, in binding order.
// user_sgprs is how many user-data SGPRs the stage was launched with
// (SPI_SHADER_PGM_RSRC2_*.USER_SGPR): a descriptor inline beyond that window is
// not user data at all, just whatever the previous draw left in those
// registers.
std::vector<gpu::gcn::TImage> TrackTextures(const u32* ps_code,
                                            const u32* ps_user_data,
                                            u32 user_sgprs);

// Resolve buffer bases and complete V#s at their consuming instruction PCs.
std::unordered_map<u32, BufferResource> ResolveBuffers(
    const u32* code,
    const u32* user_data,
    u32 user_sgprs,
    u32 user_sgpr_base = 0);

}  // namespace gpu::rdna
