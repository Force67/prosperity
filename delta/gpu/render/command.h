#pragma once

#include <cstddef>
#include <cstring>
#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/meta/traits.h"

namespace gpu::gcn {
struct Recompiled;
struct RecompiledCs;
}  // namespace gpu::gcn

namespace gpu::render {

struct VertexAttr {
  u32 location = 0;
  u32 binding = 0;
  u32 offset = 0;
  u32 num_comps = 0;
  u32 dfmt = 0;
  u32 nfmt = 0;
};

struct VertexBinding {
  const void* data = nullptr;
  u32 stride = 0;
  u32 num_records = 0;
  bool per_instance = false;
};

struct DrawInfo {
  const void* vertex_data = nullptr;
  u32 vertex_count = 0;
  u32 vertex_stride = 0;
  u32 prim_type = 0;

  const void* index_data = nullptr;
  u32 index_count = 0;
  u32 index_type = 0;
  u32 instance_count = 1;
  u64 indirect_args = 0;
  float mvp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  u64 cbuf_base = 0;
  u32 cbuf_size = 0;
  struct DrawCbuf {
    u64 base = 0;
    u32 size = 0;
  };
  DrawCbuf cbufs[64];
  u32 num_cbufs = 0;
  static constexpr u32 kMaxBuffers = 16;
  struct DrawBuffer {
    u64 base = 0;
    u32 size = 0;
    u32 stride = 0;
  };
  DrawBuffer bufs[kMaxBuffers];
  u32 num_bufs = 0;
  u64 rt_base = 0;
  u32 rt_layers = 1;
  u64 rt_array_base = 0;
  u32 rt_tile_mode = 0;
  u32 rt_w = 0, rt_h = 0;

  u64 mrt_base[8] = {0};
  u32 mrt_info[8] = {0};
  u32 mrt_count = 0;
  u32 mrt_bound_mask = 0;

  u64 tex_base = 0;
  u32 tex_w = 0, tex_h = 0;
  u32 tex_dfmt = 0, tex_nfmt = 0;
  u32 tex_tiling = 8;
  u32 tex_pitch = 0;
  u32 tex_depth = 1;
  u32 tex_layers = 1;
  u32 tex_base_array = 0;
  u32 tex_view_layers = 1;
  u32 tex_mip_levels = 1;
  u32 tex_base_mip = 0;
  u32 tex_view_mips = 1;
  u32 tex_min_lod = 0;
  u32 tex_sampler[4] = {};
  bool tex_pow2_pad = false;
  bool tex_sampler_valid = false;
  bool tex_arrayed = false;
  bool tex_is_3d = false;
  bool tex_is_1d = false;
  bool tex_force_lod_zero = false;
  bool tex_depth_compare = false;
  bool tex_null_descriptor = false;
  u32 tex_swizzle = 0;

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
    u32 swizzle = 0;
    u64 src = 0;
  };
  static constexpr u32 kMaxDrawTextures = 64;
  u32 num_texs = 0;

  u32 blend_control = 0;
  bool blend_enable = false;
  u32 mrt_blend[8] = {0};
  u32 mrt_blend_mask = 0;
  float blend_constants[4] = {0.f, 0.f, 0.f, 0.f};
  u32 target_mask = 0xF;
  u32 shader_mask = 0;
  u32 col_format = 0;
  u32 color_control = 0;

  u64 depth_base = 0;
  u64 depth_htile_base = 0;
  u32 depth_slice = 0;
  bool depth_valid = false;
  bool depth_test_enable = false;
  bool depth_write_enable = false;
  u32 depth_func = 7;
  float depth_clear = 1.0f;
  bool depth_clear_draw = false;
  bool stencil_clear_draw = false;
  u32 render_control = 0;
  u32 depth_w = 0, depth_h = 0;
  u64 stencil_base = 0;
  bool stencil_enable = false;
  bool stencil_backface_enable = false;
  u32 depth_control = 0;
  u8 stencil_clear = 0;
  u32 stencil_control = 0;
  u32 stencil_refmask = 0;
  u32 stencil_refmask_bf = 0;

  bool is_clear_rect = false;
  u32 rt_surf_w = 0, rt_surf_h = 0;
  u32 mrt_surf_w[8] = {}, mrt_surf_h[8] = {};
  u32 scissor_tl = 0, scissor_br = 0;
  u32 clear_tl = 0, clear_br = 0;
  u32 clear_window_tl = 0, clear_window_br = 0;
  u32 clear_screen_tl = 0, clear_screen_br = 0;
  u32 mrt_clear_word[8][2] = {};
  u64 mrt_dcc_base[8] = {};
  bool mrt_meta_cmask = false;

  u32 cull_mode = 0;
  bool front_ccw = true;

  float viewport_x_scale = 0, viewport_x_offset = 0;
  float viewport_y_scale = 0, viewport_y_offset = 0;
  float viewport_z_scale = 1.0f, viewport_z_offset = 0.0f;

  u64 vs_addr = 0, ps_addr = 0;
  u64 es_addr = 0, gs_addr = 0;
  u64 gs_user_data_addr = 0;
  bool ps4_neo = false;
  u32 vs_user_data[32] = {};
  u32 ps_user_data[32] = {};
  const gcn::Recompiled* recomp = nullptr;
  static constexpr u32 kMaxVertexAttrs = 16;
  u32 num_vattrs = 0;
  u32 num_vbufs = 0;

  DrawTex texs[kMaxDrawTextures];
  VertexAttr vattrs[kMaxVertexAttrs];
  VertexBinding vbufs[8];

  void Reset();
};

static_assert(base::is_trivially_copyable_v<DrawInfo>);

inline void DrawInfo::Reset() {
  static const DrawInfo kDefault{};
  std::memcpy(static_cast<void*>(this), &kDefault, offsetof(DrawInfo, texs));
}

struct GuestMemoryRange {
  u64 base = 0, size = 0;
  u64 identity = 0;
  bool writable = false;
};

struct ComputeInfo {
  static constexpr u32 kMaxResources = 128;

  u64 cs_addr = 0;
  u64 expanded_htile_base = 0;
  u32 groups[3] = {1, 1, 1};
  u32 group_base[3] = {};
  u64 indirect_args = 0;
  const gcn::RecompiledCs* recomp = nullptr;
  u32 user_data[16] = {};
  struct Res {
    u64 base = 0;
    u64 size = 0;
    u64 guest_size = 0;
    u32 binding = 0;
    bool shader_writes = false;
    bool written = false;
    bool read = true;
    bool zero_fill = false;
    bool image_staging = false;
    u32 width = 0, height = 0, pitch = 0;
    u32 layers = 0, mip_levels = 0, tiling_idx = 0;
    u32 elem_bytes = 4;
    u32 stage_elem_bytes = 4;
    u32 dfmt = 0;
    u32 nfmt = 0;
    bool pow2_pad = false;
    u64 view_offset = 0;
    u64 write_bytes = 0;
  };
  Res res[kMaxResources];
  u32 num_res = 0;
  int gds_binding = -1;
  base::Vector<GuestMemoryRange> guest_memory;
  base::Vector<u64> guest_memory_seeds;
  struct Tex {
    u32 binding = 0;
    DrawInfo::DrawTex tex;
  };
  static constexpr u32 kMaxTextures = 64;
  Tex tex[kMaxTextures];
  u32 num_tex = 0;
};

extern u32 g_submit_queue;
// Command buffers walked so far. Caches key on (frame, g_dcb_n), so it must
// never go back.
extern u32 g_dcb_n;

}  // namespace gpu::render
