#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking: extract the V# (buffer) / T# (image) / S# (sampler)
 * "sharps" a shader uses by analysing how it loads them out of the user-data
 * SGPRs. The renderer uses this per draw to resolve the live guest resources
 * behind a decoded shader's bindings.
 *
 * Descriptor field layouts:
 * https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_sh_mask.h
 * https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h
 */

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gcn_decode.h"

namespace gpu::gcn {

// A decoded vertex-buffer resource (GCN V#, 4 dwords).
struct VBuffer {
  uint64_t base = 0;         // guest address of the vertex data
  uint32_t stride = 0;       // bytes per vertex
  uint32_t num_records = 0;  // vertex count
  uint32_t dfmt = 0;         // data format
  uint32_t nfmt = 0;         // numeric format
};

// A decoded image resource (GCN T#, 8 dwords).
struct TImage {
  uint64_t base = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pitch = 0;        // surface pitch in pixels (T#.pitch+1)
  uint32_t layers = 1;       // physical array layers (T#.depth+1 for 2D arrays)
  uint32_t base_array = 0;   // first layer exposed by this descriptor view
  uint32_t view_layers = 1;  // number of layers exposed by this descriptor view
  uint32_t mip_levels = 1;   // physical levels in storage (LAST_LEVEL + 1)
  uint32_t base_mip = 0;     // first level exposed by this descriptor view
  uint32_t view_mips = 1;    // levels exposed by this descriptor view
  uint32_t min_lod = 0;      // T# MIN_LOD clamp in U4.8 fixed-point
  uint32_t dfmt = 0;
  uint32_t nfmt = 0;
  uint32_t type = 0;         // SQ_RSRC_IMG_* (9 = 2D, 13 = 2D array)
  uint32_t tiling_idx = 0;   // 8/31 = linear; everything else is tiled
  uint32_t sampler[4] = {};  // S# used by the sampling MIMG instruction
  bool pow2_pad = false;     // pad physical mip dims/layers to powers of two
  bool sampler_valid = false;
  bool arrayed = false;      // MIMG DA bit: address carries an array layer
  bool force_lod_zero = false;  // gather4_lz: implicit gather clamped to mip 0
  bool depth_compare = false;   // MIMG _C uses the sampler's compare function
  bool valid = false;
};

// Decode a V# from 4 consecutive dwords.
VBuffer DecodeVBuffer(const uint32_t* dwords);

// Decode a T# from 8 consecutive dwords.
TImage DecodeTImage(const uint32_t* dwords);

// Sampler-binding plan for a pixel shader: MIMG instructions that reference
// the same descriptor (same T#/S# SGPRs, written by the same s_load -- or
// inline user data -- and used with the same access type) share one binding.
// Bindings are numbered in first-appearance order. This is the contract
// between the recompiler's set-0 sampler declarations and TrackTextures'
// per-binding result: both derive from this one plan so they cannot drift.
struct MimgBindingPlan {
  // MIMG instruction pc -> binding id.
  std::unordered_map<uint32_t, uint32_t> binding_by_pc;
  // Per binding: the T# base SGPR of its first-use MIMG.
  std::vector<uint32_t> binding_srsrc;
};
MimgBindingPlan PlanMimgBindings(const Program& program);

// Recover the image(s) a pixel shader references, by tracking its
// s_load_dwordx4/x8/x16 of descriptor tables out of the user-data SGPRs.
// The result preserves MIMG order (it is the shader's set-0 binding order);
// unresolved entries are returned with valid=false so later bindings are not
// compacted. Pass a CachedProgram()/DecodeShader() of the PS code.
std::vector<TImage> TrackTextures(const Program& ps_program,
                                  const uint32_t* ps_user_data);

// Given a decoded fetch shader and the VS user-data SGPRs (16 dwords), recover
// the vertex-attribute buffers it fetches, in attribute order. Handles the
// common Gnm fetch-shader pattern (s_load_dwordx4 of a V# from the
// vertex-buffer table a user SGPR points at, then buffer_load_format per
// attribute).
std::vector<VBuffer> TrackVertexBuffers(const Program& fetch_program,
                                        const uint32_t* vs_user_data);

}  // namespace gpu::gcn
