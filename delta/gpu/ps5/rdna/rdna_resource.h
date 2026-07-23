#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + tracking: the gfx10.3 128/256-bit T# layout
 * and the per-draw resolution of the T#/S# a pixel shader samples. Kept out of
 * the SPIR-V-gated translator so the command processor can resolve textures even
 * in a build without the recompiler backend.
 */

#include <cstdint>
#include <vector>

#include "ps4/gcn/gcn_resource.h"
#include "rdna_decode.h"

namespace gpu::rdna {

// Dwords an SMEM s_load / s_buffer_load reads (x1/x2/x4/x8/x16). 0 = not a load.
inline uint32_t SmemLoadCount(uint32_t op) {
  switch (op) {
    case 0x00: case 0x08: return 1;
    case 0x01: case 0x09: return 2;
    case 0x02: case 0x0A: return 4;
    case 0x03: case 0x0B: return 8;
    case 0x04: case 0x0C: return 16;
    default: return 0;
  }
}

// MIMGs reading the same T#/S# descriptor share one set-0 binding, in first-use
// order. Both the recompiler (EmitMimg declarations) and TrackTextures pair
// against this, so the resolved textures line up 1:1 with the shader's samplers.
gpu::gcn::MimgBindingPlan RdnaPlanMimg(const Program& program);

// Decode a gfx10.3 256-bit T# (8 dwords). Only the geometric fields are
// authoritative here; dfmt/nfmt default to RGBA8_UNORM (the gfx10.3 unified
// 9-bit format enum is not recoverable from the ISA spec text).
gpu::gcn::TImage DecodeTImage(const uint32_t* dwords);

// Resolve the live T#/S# each MIMG in a pixel shader samples, in binding order.
// Descriptors that are inline in user data or loaded one level from a user-data
// table pointer resolve; deeper SGPR dataflow does not (returns valid=false).
std::vector<gpu::gcn::TImage> TrackTextures(const uint32_t* ps_code,
                                            const uint32_t* ps_user_data);

}  // namespace gpu::rdna
