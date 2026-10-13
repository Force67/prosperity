/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

struct RTarget {
  rhi::Texture* texture = nullptr;
  rhi::TextureView* view = nullptr;
  rhi::BindGroup* set = nullptr;
  rhi::Texture* feedback_texture = nullptr;
  rhi::TextureView* feedback_view = nullptr;
  rhi::BindGroup* feedback_set = nullptr;
  base::HashMap<u32, rhi::TextureView*> sampled_views;
  base::HashMap<u32, rhi::TextureView*> alias_views;
  base::HashMap<u32, rhi::TextureView*> feedback_sampled_views;
  rhi::TextureState feedback_layout = rhi::TextureState::kUndefined;
  u32 w = 0, h = 0;
  u32 depth = 1;
  rhi::TextureView* volume_view = nullptr;
  rhi::Format fmt = rhi::Format::kBGRA8Unorm;
  bool is_depth = false;
  rhi::TextureState layout = rhi::TextureState::kUndefined;
  bool dirty_for_read = false;
  rhi::TextureState submitted_layout = rhi::TextureState::kUndefined;
  bool used_this_frame = false;
  u32 draws = 0;
  int last_frame = -1000;
  u64 render_serial = 0;
  bool accumulated = false;
  bool clear_pending = false;
  rhi::ClearColor clear_value{};
  const char* clear_src = "none";
  bool ever_rendered = false;
  u64 last_vs = 0, last_ps = 0;
  u32 last_cbuf_mask = 0;
  u32 last_rawbuf_mask = 0;
  u64 dcc_base = 0;
  bool dcc_is_cmask = false;
  bool dcc_clear_pending = false;
  rhi::Buffer* clear_pred = nullptr;
  u64 clear_pred_offset = 0;
  bool dcc_code_known = false;
  u32 dcc_clear_code = 0;
  u32 cb_info = 0;
};

extern base::HashMap<u64, RTarget>& g_rts;

constexpr rhi::Format kDepthFormat = rhi::Format::kD32FloatS8Uint;
struct DepthTarget {
  rhi::Texture* texture = nullptr;
  rhi::TextureView* view = nullptr;
  rhi::TextureView* stencil_view = nullptr;
  rhi::TextureView* attachment_view = nullptr;
  u32 layers = 1;
  base::Vector<rhi::TextureView*> layer_views;
  u32 clear_layers = 0;
  rhi::BindGroup* set = nullptr;
  base::HashMap<u32, rhi::TextureView*> sampled_views;
  u32 w = 0, h = 0;
  u32 guest_w = 0, guest_h = 0;
  rhi::TextureState layout = rhi::TextureState::kUndefined;
  rhi::TextureState stencil_layout = rhi::TextureState::kUndefined;
  u64 stencil_base = 0;
  bool dirty_for_read = false;
  rhi::TextureState submitted_layout = rhi::TextureState::kUndefined;
  rhi::TextureState submitted_stencil_layout = rhi::TextureState::kUndefined;
  u64 render_serial = 0;
  int last_frame = -1000;
  bool used_this_frame = false;
  bool stencil_used_this_frame = false;
  bool clear_pending = false;
  float clear_value = 1.0f;
  u64 htile_base = 0;
  bool htile_clear_pending = false;
  rhi::Buffer* clear_pred = nullptr;
  u64 clear_pred_offset = 0;
  bool htile_code_known = false;
  u32 htile_clear_code = 0;
};

extern base::HashMap<u64, DepthTarget>& g_depths;
extern u64 g_render_serial;
extern base::HashMap<u64, base::Vector<DepthTarget>> g_depth_variants;
extern base::HashMap<u64, base::Vector<RTarget>> g_rt_variants;

constexpr u32 kRtPageShift = 16;
extern base::HashMap<u64, base::Vector<u64>>& g_rt_pages;

u64 RtByteSizeWH(u32 w, u32 h, rhi::Format fmt);
u64 RtByteSize(const RTarget& rt);

RTarget* GetRT(u64 base, u32 w, u32 h, rhi::Format fmt, u32 depth = 1);
void ReleaseRetiredTargets();

DepthTarget* GetDepthRT(u64 base,
                        u32 w,
                        u32 h,
                        u64 stencil_base = 0,
                        u32 layers = 1);

rhi::BindGroup* SnapshotRT(RTarget& rt);

void TransitionImage(rhi::CommandList* list,
                     rhi::Texture* texture,
                     rhi::TextureState& current,
                     rhi::TextureState state,
                     u8 aspect = rhi::kAspectColor,
                     u32 layers = 1);

rhi::TextureView* SampledView(RTarget& rt,
                              u32 swizzle,
                              bool feedback = false,
                              bool arrayed = false);
rhi::TextureView* SampledViewAs(RTarget& rt,
                                u32 swizzle,
                                rhi::Format want,
                                rhi::Format* used = nullptr,
                                bool arrayed = false);
rhi::TextureView* SampledView(DepthTarget& depth, u32 swizzle);

bool ActivateSampledRtVariant(u64 base, u32 w, u32 h);
bool ActivateVolumeRt(u64 base, u32 w, u32 h, u32 depth);
bool ActivateSampledDepthVariant(u64 base, u32 w, u32 h);

u64 ResolveSampledRT(u64 addr, u32 w, u32 h);
u64 ResolveSampledDepth(u64 addr, u32 w, u32 h);
u64 ResolveSampledStencil(u64 addr);
rhi::TextureView* StencilSampledView(DepthTarget& depth);

bool PreserveCsDepthBeforeClear(u64 base);

void NoteMetadataBase(u64 base);
void NoteDccWrite(u64 base, u64 bytes, const u32* fill);
void NoteExpandedHtile(u64 base);

void NoteSurfaceWrite(u64 base, u64 bytes);

void NoteRawWrite(u64 base, u64 bytes);
bool OverlapsLiveTarget(u64 base, u64 bytes);

bool ActivateWrittenRtVariant(u64 base, u32 w, u32 h);
bool ActivateRtVariantAs(u64 base, u32 w, u32 h, rhi::Format fmt);

struct RenderRegion {
  u64 cur_rt = 0;
  u64 cur_mrt[8] = {0};
  u32 cur_mrt_count = 0;
  u32 cur_fmt[8] = {0};
  u32 cur_w[8] = {0};
  u32 cur_h[8] = {0};
  u32 cur_area_w = 0;
  u32 cur_area_h = 0;
  u64 cur_depth = 0;
  u32 cur_depth_slice = 0;
  u32 cur_layers = 1;
  u64 cur_stencil = 0;
  u64 last_rt = 0;
  u64 first_rt = 0;
  u64 busiest_rt = 0;
  u32 busiest_rt_draws = 0;
  bool open = false;
  bool depth_read_only = false;
  u64 write_through = 0;
  u32 write_through_tile = 0;
};

extern RenderRegion& g_region;

bool BeginRegion(const u64* mrt_base,
                 const u32* mrt_info,
                 u32 mrt_count,
                 u32 w,
                 u32 h,
                 u64 depth_base = 0,
                 float depth_clear = 1.0f,
                 u64 stencil_base = 0,
                 u8 stencil_clear = 0,
                 bool depth_read_only = false,
                 u32 depth_w = 0,
                 u32 depth_h = 0,
                 const u32* mrt_surf_w = nullptr,
                 const u32* mrt_surf_h = nullptr,
                 const u64* mrt_dcc_base = nullptr,
                 const u32 (*mrt_clear_word)[2] = nullptr,
                 u64 depth_htile_base = 0,
                 u32 depth_slice = 0,
                 u32 layers = 1,
                 bool meta_cmask = false);

u32 RtSurfaceExtent(u32 surface, u32 drawn, u32 tile);

inline u32 DepthW(const render::DrawInfo& d) {
  return d.depth_w ? d.depth_w : d.rt_w;
}
inline u32 DepthH(const render::DrawInfo& d) {
  return d.depth_h ? d.depth_h : d.rt_h;
}
void EndRegion();
void SetGuestViewport(const render::DrawInfo& d);

bool WriteRtToGuest(u64 base, u32 tile_mode);

}  // namespace gpu::render
