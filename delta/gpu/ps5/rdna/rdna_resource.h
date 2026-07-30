#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + tracking: the gfx10.3 128/256-bit T# layout
 * and the per-draw resolution of the T#/S# a pixel shader samples. Kept out of
 * the SPIR-V-gated translator so the command processor can resolve textures
 * even in a build without the recompiler backend.
 */

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gpu/gcn/gcn_resource.h"
#include "gpu/ps5/rdna/rdna_decode.h"

namespace gpu::rdna {

struct Smem {
  uint32_t op;
  uint32_t sdst;
  uint32_t sbase;
  uint32_t soffset;
  int32_t offset;
};

struct BufferResource {
  uint64_t base = 0;
  uint32_t descriptor[4] = {};
  bool descriptor_valid = false;
};

inline Smem DecodeSmem(const gpu::gcn::Inst& inst) {
  return {
      .op = inst.opcode,
      .sdst = (inst.raw[0] >> 6) & 0x7F,
      .sbase = (inst.raw[0] & 0x3F) * 2,
      .soffset = (inst.raw[1] >> 25) & 0x7F,
      .offset = static_cast<int32_t>(inst.raw[1] << 11) >> 11,
  };
}

// Dwords an SMEM s_load / s_buffer_load reads (x1/x2/x4/x8/x16). 0 = not a
// load.
inline uint32_t SmemLoadCount(uint32_t op) {
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
  uint32_t first = 0;
  uint32_t count = 0;
};

inline ScalarWrite DecodeScalarWrite(const gpu::gcn::Inst& inst) {
  using gpu::gcn::Enc;
  if (inst.enc == Enc::kSmrd) {
    const uint32_t sdst = DecodeSmem(inst).sdst;
    return sdst == 125 ? ScalarWrite{}
                       : ScalarWrite{.first = sdst,
                                     .count = SmemLoadCount(inst.opcode)};
  }
  if (inst.enc == Enc::kSopk) {
    const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F;
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
    const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F;
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
    const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F;
    return sdst == 125 ? ScalarWrite{}
                       : ScalarWrite{.first = sdst, .count = wide ? 2u : 1u};
  }
  return {};
}

// MIMGs reading the same T#/S# descriptor share one set-0 binding, in first-use
// order. Both the recompiler (EmitMimg declarations) and TrackTextures pair
// against this, so the resolved textures line up 1:1 with the shader's
// samplers.
gpu::gcn::MimgBindingPlan RdnaPlanMimg(const Program& program);

// Decode a gfx10.3 image resource. R128 selects the compact four-dword form;
// ordinary resources contain eight dwords.
gpu::gcn::TImage DecodeTImage(const uint32_t* dwords, bool r128 = false);

// Resolve the live T#/S# each MIMG in a pixel shader samples, in binding order.
// user_sgprs is how many user-data SGPRs the stage was launched with
// (SPI_SHADER_PGM_RSRC2_*.USER_SGPR): a descriptor inline beyond that window is
// not user data at all, just whatever the previous draw left in those
// registers.
std::vector<gpu::gcn::TImage> TrackTextures(const uint32_t* ps_code,
                                            const uint32_t* ps_user_data,
                                            uint32_t user_sgprs);

// Resolve buffer bases and complete V#s at their consuming instruction PCs.
std::unordered_map<uint32_t, BufferResource> ResolveBuffers(
    const uint32_t* code,
    const uint32_t* user_data,
    uint32_t user_sgprs,
    uint32_t user_sgpr_base = 0);

}  // namespace gpu::rdna
