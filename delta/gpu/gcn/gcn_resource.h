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

#include "base/arch.h"
#include <memory>
#include <unordered_map>
#include <vector>

#include "gpu/gcn/gcn_decode.h"

namespace gpu::gcn {

struct RecompiledCs;
struct ShaderAttr;
struct ShaderBuffer;

// A decoded vertex-buffer resource (GCN V#, 4 dwords).
struct VBuffer {
  u64 base = 0;         // guest address of the vertex data
  u32 stride = 0;       // bytes per vertex
  u32 num_records = 0;  // vertex count
  u32 dfmt = 0;         // data format
  u32 nfmt = 0;         // numeric format
};

// A decoded image resource (GCN T#, 8 dwords).
struct TImage {
  u64 base = 0;
  u32 width = 0;
  u32 height = 0;
  u32 pitch = 0;        // surface pitch in pixels (T#.pitch+1)
  u32 depth = 1;        // slices of a volume image (T#.depth+1 for 3D)
  u32 layers = 1;       // physical array layers (T#.depth+1 for 2D arrays)
  u32 base_array = 0;   // first layer exposed by this descriptor view
  u32 view_layers = 1;  // number of layers exposed by this descriptor view
  u32 mip_levels = 1;   // physical levels in storage (LAST_LEVEL + 1)
  u32 base_mip = 0;     // first level exposed by this descriptor view
  u32 view_mips = 1;    // levels exposed by this descriptor view
  u32 min_lod = 0;      // T# MIN_LOD clamp in U4.8 fixed-point
  u32 dfmt = 0;
  u32 nfmt = 0;
  u32 type = 0;        // SQ_RSRC_IMG_* (8/12 = 1D, 9 = 2D, 10 = 3D, 13 = 2D array)
  u32 tiling_idx = 0;  // 8/31 = linear; everything else is tiled
  // T# DST_SEL_X/Y/Z/W: which source channel (or constant) each sampled
  // component reads. 0=0, 1=1, 4=R, 5=G, 6=B, 7=A. A single-channel mask (a
  // font atlas) is declared with its coverage selected into the components the
  // shader reads; ignoring that samples alpha as 1 and fills every glyph solid.
  u32 dst_sel[4] = {4, 5, 6, 7};
  u32 sampler[4] = {};  // S# used by the sampling MIMG instruction
  bool pow2_pad = false;     // pad physical mip dims/layers to powers of two
  bool sampler_valid = false;
  bool is_3d = false;           // SQ_RSRC_IMG_3D: sampled with a w coordinate
  bool is_1d = false;           // SQ_RSRC_IMG_1D[_ARRAY]: x (+layer) only
  bool arrayed = false;         // MIMG DA bit: address carries an array layer
  bool force_lod_zero = false;  // gather4_lz: implicit gather clamped to mip 0
  bool depth_compare = false;   // MIMG _C uses the sampler's compare function
  bool storage = false;         // image_store target
  bool null_descriptor = false; // all-zero T# samples transparent zero
  bool valid = false;
  u64 src = 0;  // guest address dword 0 was s_loaded from (0 = user data)
};

// DST_SEL in the form the renderer's image views take it (x | y<<3 | z<<6 |
// w<<9).
inline u32 PackDstSel(const u32 dst_sel[4]) {
  return (dst_sel[0] & 7) | ((dst_sel[1] & 7) << 3) | ((dst_sel[2] & 7) << 6) |
         ((dst_sel[3] & 7) << 9);
}

// Decode a V# from 4 consecutive dwords.
VBuffer DecodeVBuffer(const u32* dwords);

// Decode a T# from 8 consecutive dwords.
TImage DecodeTImage(const u32* dwords);

// Sampler-binding plan: MIMGs referencing the same descriptor (same SGPRs, same
// producer, same access) share one binding, numbered in first-appearance order.
// Contract between the recompiler's set-0 declarations and TrackTextures, so they
// cannot drift.
struct MimgBindingPlan {
  // MIMG instruction pc -> binding id.
  std::unordered_map<u32, u32> binding_by_pc;
  // Per binding: the T# base SGPR of its first-use MIMG.
  std::vector<u32> binding_srsrc;
  // Per binding: true when the descriptor is an image_store target.
  std::vector<bool> binding_storage;
};
MimgBindingPlan PlanMimgBindings(const Program& program,
                                 const u8* reachable = nullptr);

// Recover the image(s) a PS references by tracking s_load_dwordx4/x8/x16 of
// descriptor tables out of the user-data SGPRs. Preserves MIMG order (set-0 binding
// order); unresolved entries carry valid=false. Pass CachedProgram(): it keys a
// per-program cache of plan + scalar-relevant subset, skipping re-planning per draw.
// code_base resolves s_getpc_b64 against a table embedded after the code; 0 = unknown.
std::vector<TImage> TrackTextures(
    const std::shared_ptr<const Program>& ps_program,
    const u32* ps_user_data,
    bool trace = false,
    u64 code_base = 0);

// Resolve the live descriptor behind each cbuffer, following the same extended-
// user-data / SRT chains as TrackTextures: a 4-dword V# (s_buffer_load) or a
// 2-dword flat pointer (s_load, .base only, no size field). Keyed by base SGPR
// OR'd 0x100 for a pointer (one SGPR can serve both). FOX passes cbuffer
// descriptors through EUD, so the chain is walked and read at the point of load.
std::unordered_map<u32, VBuffer> ResolveCbuffers(
    const std::shared_ptr<const Program>& program,
    const u32* user_data);

// Resolve attributes fetched by MUBUF in the main VS: the descriptor SGPRs may start
// as inline user data or be overwritten by an SMRD load, so capture each V# from the
// scalar state live at its MUBUF. Index-aligned with attrs; unresolved zeroed.
std::vector<VBuffer> ResolveDirectVertexBuffers(
    const std::shared_ptr<const Program>& program,
    const std::vector<ShaderAttr>& attrs,
    const u32* user_data);

// Resolve the live V# behind each raw MUBUF buffer (see ShaderBuffer): same replay
// as ResolveDirectVertexBuffers, captured at the consuming instruction.
// Index-aligned with `buffers`; unresolved zeroed.
std::vector<VBuffer> ResolveShaderBuffers(
    const std::shared_ptr<const Program>& program,
    const std::vector<ShaderBuffer>& buffers,
    const u32* user_data);

// Replay a compute shader's uniform scalar descriptor loads and capture each
// planned V#/T#/pointer at the instruction where it is consumed. This resolves
// SRT chains into SGPRs above the direct COMPUTE_USER_DATA range.
struct ResolvedCsResource {
  bool valid = false;
  u32 descriptor[8] = {};
};
std::vector<ResolvedCsResource> ResolveCsResources(const Program& program,
                                                   const RecompiledCs& plan,
                                                   const u32* user_data);

// Given a decoded fetch shader + the 16 VS user-data SGPRs, recover the vertex-
// attribute buffers in attribute order (the common s_load_dwordx4 from the
// vertex-buffer table + buffer_load_format pattern).
std::vector<VBuffer> TrackVertexBuffers(const Program& fetch_program,
                                        const u32* vs_user_data);

// A descriptor table the replay is about to read may still be sitting in a
// compute buffer, unwritten-back. The recompiler cannot ask a renderer that
// itself (it is the bottom of the stack), so whoever owns both installs this;
// left null, the replay just reads guest memory as it finds it.
extern void (*g_flush_guest_range)(u64 address, u64 bytes);

}  // namespace gpu::gcn
