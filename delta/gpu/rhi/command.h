#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The work a command processor hands the renderer: one decoded draw, or one
 * decoded compute dispatch. Backend-agnostic by construction: nothing here
 * names a graphics API type, so the PS4 (PM4/GCN) and PS5 (AGC/RDNA2) command
 * processors compile without seeing the backend at all.
 *
 * Addresses are guest addresses (identity-mapped, host-readable).
 */

#include "base/arch.h"
#include <vector>

namespace gpu::gcn {
struct Recompiled;
struct RecompiledCs;
}  // namespace gpu::gcn

namespace gpu::rhi {

// One vertex attribute for the recompiled-shader path: where the recompiled VS reads
// input `location` from within a vertex buffer binding (binding indexes
// DrawInfo::vbufs; interleaved attributes share a binding, separate streams don't).
struct VertexAttr {
  u32 location = 0;
  u32 binding = 0;    // index into DrawInfo::vbufs
  u32 offset = 0;     // byte offset within the binding's vertex record
  u32 num_comps = 0;  // 1..4
  u32 dfmt = 0;       // GCN data format (selects the backend format)
  u32 nfmt = 0;       // GCN number format
};

// One vertex buffer binding for the recompiled-shader path. Each distinct guest
// V# base + stride becomes one vertex binding; its records are uploaded into
// the renderer's vertex ring and bound for the draw.
struct VertexBinding {
  const void* data = nullptr;  // guest base of this binding's vertex data
  u32 stride = 0;         // bytes per record
  u32 num_records = 0;    // records available in the source buffer
};

// Per-draw inputs extracted by the command processor (resource-tracked from the
// shader + register state).
struct DrawInfo {
  const void* vertex_data = nullptr;  // base of attribute-0 (position) buffer
  u32 vertex_count = 0;
  u32 vertex_stride = 0;  // bytes per vertex in the source buffer
  u32 pos_offset = 0;     // byte offset of the float2 position
  u32 prim_type = 0;      // VGT_PRIMITIVE_TYPE (4 = triangle list)

  // Index buffer (DRAW_INDEX_2); non-null means indexed (type: 0=16, 1=32, 2=8 bit).
  // Without one the draw is sequential (DRAW_INDEX_AUTO).
  const void* index_data = nullptr;
  u32 index_count = 0;
  u32 index_type = 0;
  u32 instance_count =
      1;  // from IT_NUM_INSTANCES (tilemaps draw instanced)
  float mvp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  // Legacy transform fields for heuristic rendering, mirroring the first resolved
  // VS cbuffer; recompiled shaders use cbufs[] (set 1, bindings 0..7). mvp[] stays
  // binding 0's fallback when the VS descriptor cannot be resolved.
  u64 cbuf_base = 0;
  u32 cbuf_size = 0;
  struct DrawCbuf {
    u64 base = 0;
    u32 size = 0;
  };
  DrawCbuf cbufs[32];
  u32 num_cbufs = 0;
  // Raw MUBUF buffers the recompiled VS/PS read by hand (skinning palette, instance
  // table, self-indexed vertex data), each staged into a set-2 storage window with
  // binding == index. Empty when the vertex-input state covers every fetch.
  static constexpr u32 kMaxBuffers = 16;
  struct DrawBuffer {
    u64 base = 0;
    u32 size =
        0;  // bytes the descriptor describes (may exceed the window)
  };
  DrawBuffer bufs[kMaxBuffers];
  u32 num_bufs = 0;
  u64 rt_base = 0;  // CB_COLOR0 address; the draw's render target
  u32 rt_w = 0,
           rt_h = 0;  // render-target dimensions (shared by all MRT targets)

  // Multiple render targets (CB_COLOR0..7); mrt_base[0] mirrors rt_base. A target is
  // bound when its CB_TARGET_MASK nibble, CB_COLORn_INFO format and base are valid;
  // mrt_info preserves the format so attachments match the guest surface.
  u64 mrt_base[8] = {0};
  u32 mrt_info[8] = {0};
  u32 mrt_count = 0;
  // Bit n set = slot n is bound. Not the same as "n < mrt_count": a pass may
  // leave a lower slot masked off while a higher one is live, and a shader
  // exporting to a slot nothing is bound at writes into nothing.
  u32 mrt_bound_mask = 0;

  // Texturing (optional). tex_base preserves every valid T# address so the
  // renderer can resolve non-RGBA guest formats to live render targets. Direct
  // guest-memory uploads remain limited to formats the upload path can decode.
  const void* uv_data = nullptr;
  u32 uv_stride = 0;
  u32 uv_offset = 0;  // byte offset of the float2 uv within the vertex
  u32 color_offset =
      0xFFFFFFFFu;  // byte offset of float3 color; ~0 = white
  u64 tex_base = 0;
  u32 tex_w = 0, tex_h = 0;
  u32 tex_dfmt = 0, tex_nfmt = 0;
  u32 tex_tiling = 8;       // T# tiling_index (8/31 = linear; else tiled)
  u32 tex_pitch = 0;        // T# surface pitch in pixels (0 = use tex_w)
  u32 tex_depth = 1;        // slices of a volume image (1 = not 3D)
  u32 tex_layers = 1;       // physical layers in the image allocation
  u32 tex_base_array = 0;   // first layer exposed by the image view
  u32 tex_view_layers = 1;  // layers exposed by the image view
  u32 tex_mip_levels = 1;   // physical mip levels in the image allocation
  u32 tex_base_mip = 0;     // first mip exposed by the image view
  u32 tex_view_mips = 1;    // mip levels exposed by the image view
  u32 tex_min_lod = 0;      // T# MIN_LOD clamp in U4.8 fixed-point
  u32 tex_sampler[4] =
      {};                     // guest sampler descriptor for this MIMG binding
  bool tex_pow2_pad = false;  // physical mip dimensions/layers use POW2_PAD
  bool tex_sampler_valid = false;
  bool tex_arrayed = false;  // MIMG DA: shader consumes a layer coordinate
  bool tex_is_3d = false;    // T# type 10: sampled with a w coordinate
  bool tex_is_1d = false;    // T# type 8/12: height-1 image, x (+layer) only
  bool tex_force_lod_zero = false;
  bool tex_depth_compare = false;
  bool tex_null_descriptor = false;
  u32 tex_swizzle = 0;  // packed T# DST_SEL for the legacy single texture

  // Multi-texture (Doom64 walls: diffuse + lightmap + ... from the EUD table);
  // texs[0] mirrors tex_base; num_texs > 1 binds an N-sampler descriptor set.
  struct DrawTex {
    u64 base = 0;
    u32 w = 0, h = 0, tiling = 8, pitch = 0;
    u32 dfmt = 0, nfmt = 0;
    u32 depth = 1;
    u32 layers = 1, base_array = 0, view_layers = 1;
    u32 mip_levels = 1, base_mip = 0, view_mips = 1;
    u32 min_lod = 0;
    u32 sampler[4] = {};
    bool pow2_pad = false;
    bool sampler_valid = false;
    bool arrayed = false;
    bool is_3d = false;
    bool is_1d = false;
    bool force_lod_zero = false;
    bool depth_compare = false;
    bool storage = false;
    bool null_descriptor = false;
    u32 swizzle = 0;  // packed T# DST_SEL_X/Y/Z/W (0 = identity)
    // Guest address the descriptor was s_loaded from (0 = inline user data).
    // A binding that fails to resolve is only diagnosable from the memory
    // behind it: whether the slot is zero, stale, or never the shader's.
    u64 src = 0;
  };
  static constexpr u32 kMaxDrawTextures = 24;  // == gpu::vk::kMaxTex
  DrawTex texs[kMaxDrawTextures];
  u32 num_texs = 0;

  // Per-draw blend state, decoded from CB_BLEND0_CONTROL (raw dword) + whether
  // blending is enabled for color target 0. The renderer maps the GNM blend
  // factors/functions to a pipeline (cached per unique state).
  u32 blend_control = 0;
  bool blend_enable = false;
  // Per-MRT blend (CB_BLENDn_CONTROL + enable mask); [0]/bit0 mirror the single-RT
  // fields, and an MRT draw gets each target's own blend instead of target 0's.
  u32 mrt_blend[8] = {0};
  u32 mrt_blend_mask = 0;
  // CB_BLEND_RED/GREEN/BLUE/ALPHA, the operand of the CONSTANT_* blend
  // factors. Left at zero these turn a constant-blended pass black, which is
  // indistinguishable from a shader that computed nothing.
  float blend_constants[4] = {0.f, 0.f, 0.f, 0.f};
  // CB_TARGET_MASK (MRT0 = bits[3:0]) + CB_COLOR_CONTROL MODE [6:4], honoured as the
  // colour write mask so a draw the game masks off writes nothing.
  u32 target_mask = 0xF;
  // CB_SHADER_MASK: which channels of each target the PS actually exports. The
  // hardware writes a channel only when this and target_mask both enable it,
  // and leaves the rest of the target untouched.
  u32 shader_mask = 0;
  u32 color_control = 0;

  // Depth/stencil: valid depth_base + depth_valid binds a depth attachment keyed by
  // depth_base; 2D titles leave it 0 (unchanged path).
  u64 depth_base = 0;
  // DB_HTILE_DATA_BASE: the depth surface's compression metadata. A write
  // over it is the depth fast clear (see NoteDccWrite).
  u64 depth_htile_base = 0;
  bool depth_valid = false;         // DB_Z_INFO format != 0
  bool depth_test_enable = false;   // DB_DEPTH_CONTROL Z_ENABLE
  bool depth_write_enable = false;  // DB_DEPTH_CONTROL Z_WRITE_ENABLE
  u32 depth_func =
      7;  // DB_DEPTH_CONTROL ZFUNC (maps 1:1 to the compare op)
  float depth_clear = 1.0f;  // DB_DEPTH_CLEAR (fast-clear value)
  // DB_RENDER_CONTROL clear bits: this "draw" is a hardware fill of the
  // depth/stencil plane with the clear value, not geometry.
  bool depth_clear_draw = false;
  bool stencil_clear_draw = false;
  u32 render_control = 0;  // raw DB_RENDER_CONTROL, for diagnosis
  // The Z surface's OWN padded geometry (DB_DEPTH_SIZE), not the colour target's:
  // a half-resolution depth on a full pass, sized from the colour target, swallows
  // unrelated addresses in the sampled-address page table.
  u32 depth_w = 0, depth_h = 0;
  u64 stencil_base = 0;
  bool stencil_enable = false;
  bool stencil_backface_enable = false;
  u32 depth_control = 0;
  u8 stencil_clear = 0;
  u32 stencil_control = 0;
  u32 stencil_refmask = 0;
  u32 stencil_refmask_bf = 0;

  // GNM fast clear: a RECT_LIST draw, no PS, no attributes, colour in
  // CB_COLORn_CLEAR_WORD0/1. Rasterising it writes nothing, so an unrecognising
  // backend leaves the target holding the previous frame (SotC accumulated one
  // fullscreen pass per frame until its value tracked the frame counter).
  bool is_clear_rect = false;
  // The rectangle a fast clear covers, from the generic scissor; a GNM fast clear
  // has no vertex attributes, so this is the ONLY size hint (a partial clear
  // treated as whole-attachment erases everything else in the target).
  // MRT0's real surface geometry from CB_COLOR0_PITCH/SLICE TILE_MAX; the screen
  // scissor is the DRAWN region and can be a fraction of the surface.
  u32 rt_surf_w = 0, rt_surf_h = 0;
  // The same, per bound target. MRT slots can have different geometries, and
  // the image each one needs is its own surface, not the drawn region.
  u32 mrt_surf_w[8] = {}, mrt_surf_h[8] = {};
  // Per-viewport scissor 0 (PA_SC_VPORT_SCISSOR_0_TL/BR), x in the low half
  // and y in the high half of each word. This is the per-DRAW scissor.
  u32 scissor_tl = 0, scissor_br = 0;
  u32 clear_tl = 0, clear_br = 0;
  // The other two scissors (hardware intersects three); a title leaving the generic
  // one at reset (P.T.: (0,0)-(0,0) every fast clear) is pinned by the others.
  u32 clear_window_tl = 0, clear_window_br = 0;
  u32 clear_screen_tl = 0, clear_screen_br = 0;
  u32 mrt_clear_word[8][2] = {};
  // CB_COLORn_DCC_BASE: where each target's compression metadata lives. A
  // title clears a compressed target by writing a clear code over this, not
  // by touching the pixels (see NoteDccWrite).
  u64 mrt_dcc_base[8] = {};

  // Primitive-setup: raster topology + face culling, from VGT_PRIMITIVE_TYPE
  // and PA_SU_SC_MODE_CNTL. 2D titles draw triangle lists with no culling
  // (unchanged).
  u32 cull_mode = 0;  // PA_SU_SC_MODE_CNTL: CULL_FRONT[0] CULL_BACK[1]
  bool front_ccw = true;   // FACE[2] == 0

  // XY viewport transform from PA_CL_VPORT_0_*.
  float viewport_x_scale = 0, viewport_x_offset = 0;
  float viewport_y_scale = 0, viewport_y_offset = 0;
  // Depth range: window_z = ndc_z * z_scale + z_offset; ignoring it hands every
  // depth-sampling pass the wrong numbers, not just a shifted test.
  float viewport_z_scale = 1.0f, viewport_z_offset = 0.0f;

  // Recompiled-shader path: non-null recomp runs the game's actual VS/PS instead of
  // the heuristic quad; vertex_data/stride is the raw interleaved buffer, vattrs
  // the inputs, mvp the pushed constant buffer, tex_base the sampler.
  u64 vs_addr = 0, ps_addr = 0;  // pipeline cache key
  // gfx10.3 merged NGG: both half addresses the draw programmed, and which one
  // vs_addr picked. Captured so a pass that never rasterizes can name the half
  // the geometry actually lives in. The PS4 walk leaves both 0.
  u64 es_addr = 0, gs_addr = 0;
  u64 gs_user_data_addr = 0;
  bool ps4_neo = false;
  u32 vs_user_data[32] = {};
  u32 ps_user_data[32] = {};
  const gcn::Recompiled* recomp = nullptr;
  // Vulkan guarantees 16 vertex input attributes; stopping at 8 left later inputs
  // with no Location (VUID-VkGraphicsPipelineCreateInfo-Input-07904) reading
  // undefined.
  static constexpr u32 kMaxVertexAttrs = 16;
  VertexAttr vattrs[kMaxVertexAttrs];
  u32 num_vattrs = 0;
  // Vertex buffer bindings; vbufs[0] mirrors vertex_data/stride so the single-
  // binding fast path is unchanged; a multi-stream draw fills one per distinct V#.
  VertexBinding vbufs[8];
  u32 num_vbufs = 0;
};

// A compute dispatch resolved by the command processor: the recompiled CS, the live
// guest memory ranges its descriptors point at, and the raw user data (pushed).
// The renderer stages each range into a storage buffer, runs the dispatch, and
// copies written ranges back for the graphics texture path.
struct GuestMemoryRange {
  u64 base = 0, size = 0;
  // Backing identity (file offset or tracked allocation), invalidates remaps.
  u64 identity = 0;
  bool writable = false;
};

struct ComputeInfo {
  static constexpr u32 kMaxResources = 128;

  u64 cs_addr = 0;            // pipeline cache key
  u32 groups[3] = {1, 1, 1};  // workgroup counts
  u32 group_base[3] = {};     // first workgroup ID in each dimension
  const gcn::RecompiledCs* recomp = nullptr;
  u32 user_data[16] = {};  // COMPUTE_USER_DATA_0..15 (push constants)
  struct Res {
    u64 base = 0;        // guest address the storage buffer aliases
    u64 size = 0;        // bytes staged in linear SSBO layout
    u64 guest_size = 0;  // physical guest bytes (same as size when linear)
    u32 binding = 0;
    bool shader_writes = false;  // shader access, including dummy resources
    bool written = false;        // copy back to guest after the dispatch
    // The dispatch reads it. A written-but-never-read range does not need
    // staging in from guest memory: the shader supplies every byte it will
    // then write back. See DELTA_GPU_CS_SKIP_UPLOAD.
    bool read = true;
    bool zero_fill =
        false;  // inactive/null descriptor: bind zeroed dummy storage
    // Back this range with the guest pages themselves rather than a staged
    // copy. A window sized to a whole allocation is far too big to copy per
    // dispatch, and the shader only touches part of it.
    bool prefer_import = false;
    bool image_staging = false;  // detile and/or expand compact texels
    u32 width = 0, height = 0, pitch = 0;
    u32 layers = 0, mip_levels = 0, tiling_idx = 0;
    u32 elem_bytes = 4;
    u32 stage_elem_bytes = 4;
    u32 dfmt = 0;
    u32 nfmt = 0;
    bool pow2_pad = false;
  };
  Res res[kMaxResources];
  u32 num_res = 0;
  // Binding of the GDS scratchpad, past the resources, or -1 when the shader
  // has no ds_append/ds_consume.
  int gds_binding = -1;
  std::vector<GuestMemoryRange> guest_memory;
};

// How long a command processor spent walking a submitted command buffer, and
// how many it walked. The console-specific processors write these and the
// renderer's per-frame report reads them, so neither has to include the other.
extern u64 g_ns_dcb, g_ns_dcb_lock;
// The AGC queue whose ring the command processor walks (0 = not via a ring), for
// frame capture: out-of-order work across queues is otherwise indistinguishable
// from work that is simply wrong.
extern u32 g_submit_queue;
extern u32 g_dcb_n;

}  // namespace gpu::rhi
