#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
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
  u32 descriptor[8] = {};
  u32 descriptor_dwords = 0;
  bool descriptor_valid = false;
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

inline u32 Sop1WriteDwords(u32 op) {
  switch (op) {
    case 0x03:
    case 0x05:
    case 0x07:
    case 0x09:
    case 0x0b:
    case 0x0d:
    case 0x0e:
    case 0x0f:
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17:
    case 0x18:
    case 0x19:
    case 0x1a:
    case 0x1b:
    case 0x1d:
    case 0x2c:
    case 0x2e:
    case 0x30:
    case 0x34:
    case 0x3c:
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
    case 0x49:
      return 1;
    case 0x20:
    case 0x22:
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
    return sdst == 125 || !count ? ScalarWrite{}
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

base::HashMap<u32, base::Vector<u32>> PlanMovrelIndices(
    const gpu::gcn::Program& program);

struct ScalarReplayPlan {
  static constexpr u32 kRegs = 136;
  enum Loss : u8 {
    kCovered = 0,
    kUnmodelled,
    kConditional,
    kLoopBody,
    kLoopCarried,
  };
  struct Write {
    u32 index;
    u32 first;
    u32 count;
    bool exact;
    bool loop_invariant;
  };
  struct BackEdge {
    u32 target;
    u32 source;
  };
  base::Vector<Write> writes;
  base::Vector<BackEdge> back_edges;
  base::Vector<u32> targets;
  bool indirect = false;
  static constexpr u32 kNoBlock = ~0u;
  base::Vector<u32> block_of;
  base::Vector<u32> idom;
  bool Dominates(u32 write_index, u32 use_index) const;

  Loss LossAt(u32 sgpr, u32 dwords, u32 use_index) const;
  bool Covers(u32 sgpr, u32 dwords, u32 use_index) const {
    return LossAt(sgpr, dwords, use_index) == kCovered;
  }
};

ScalarReplayPlan PlanScalarReplay(const Program& program);

gpu::gcn::MimgBindingPlan RdnaPlanMimg(const Program& program);

gpu::gcn::TImage DecodeTImage(const u32* dwords, bool r128 = false);

struct VBuffer {
  u64 base = 0;
  u32 stride = 0;
  u32 num_records = 0;
  u32 dfmt = 0;
  u32 nfmt = 0;
  u32 gfmt = 0;
};

void DecodeBufferFormat(u32 gfmt, u32& dfmt, u32& nfmt);

VBuffer DecodeVBuffer(const u32* dwords);

bool PlausibleVBuffer(const VBuffer& v);

base::Vector<gpu::gcn::TImage> TrackTextures(const u32* ps_code,
                                             const u32* ps_user_data,
                                             u32 user_sgprs,
                                             u32 ud_base = 0,
                                             u64 system_user_data_addr = 0,
                                             u32 max_dwords = 4096);

base::HashMap<u32, BufferResource> ResolveBuffers(
    const u32* code,
    const u32* user_data,
    u32 user_sgprs,
    u32 user_sgpr_base = 0,
    u32 max_dwords = 4096,
    u64 system_user_data_addr = 0);

}  // namespace gpu::rdna
