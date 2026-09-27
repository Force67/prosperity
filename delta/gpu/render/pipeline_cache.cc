/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/pipeline_cache.h"
#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/shaders/quad_frag_spv.h"
#include "gpu/shaders/quad_vert_spv.h"
#include "gpu/shaders/tex_frag_spv.h"
#include "gpu/shaders/tex_vert_spv.h"
#include "gpu/render/labels.h"
#include "gpu/render/device.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/frame.h"
#include "gpu/render/hash.h"
#include "gpu/gpu_perf.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/upload_ring.h"

#include <cstdio>
#include <cstdlib>

#include <base/containers/array.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/options.h>
#include <base/algorithm.h>
#include <base/containers/vector.h>
#include <base/math/value_bounds.h>
#include <base/memory/move.h>

namespace {
DELTA_OPTION(bool, kDoCull, "DELTA_GPU_CULL", false);
DELTA_OPTION(bool, kDepthClamp, "DELTA_GPU_DEPTHCLAMP", false);
// DELTA_GPU_PIPETRACE=1 traces the first pipelines built; =<ps addr> traces
// only that shader's. Pipelines are built in load order, so a flat cap only
// ever shows the loading screens, so a negative from it says nothing about the
// draw you care about.
DELTA_OPTION(u64, kGpuPipetrace, "DELTA_GPU_PIPETRACE", 0);
DELTA_OPTION(bool, kNoMaskDiag, "DELTA_GPU_NOMASK", false);
DELTA_OPTION(bool, kNoRectGs, "DELTA_GPU_NORECTGS", false);
// A depth prepass leaves the shaded pass testing ZFUNC=EQUAL against depth we
// cannot reproduce bit-exactly; on by default because rejecting the whole scene
// is never the better failure. DELTA_GPU_ZEQUAL=strict restores the raw op.
DELTA_OPTION(bool, kRelaxDepthEqual, "DELTA_GPU_RELAX_ZEQUAL", true);
// DELTA_GPU_NOZTEST=1: build every pipeline with the depth test off. Tells
// "this pass produced nothing because the depth test rejected it" apart from
// "its shader computed nothing", which look identical in an empty target.
DELTA_OPTION(bool, kNoZTest, "DELTA_GPU_NOZTEST", false);
// DELTA_GPU_NOZTEST_PS=<ps guest addr>: the same, for ONE pass. NOZTEST is a
// blunt instrument: it disables the depth test on every pipeline, so a frame
// that improves under it has told you only that SOME depth test was responsible,
// and every other pass is now drawing over everything at the same time. Naming
// one shader answers the question the blunt version cannot: whether THIS pass is
// being rejected by the comparison, or is computing nothing to begin with.
DELTA_OPTION(const char*, kNoZTestPs, "DELTA_GPU_NOZTEST_PS", nullptr);
// DELTA_GPU_NOZWRITE_PS=<list>: the twin of the above for depth WRITES. The pair
// separates the two things a depth-tested, depth-writing pass can get wrong:
// disabling the TEST asks whether the pass is being rejected by what it reads
// from the plane; disabling the WRITE asks whether what it puts INTO the plane
// is what breaks a later pass. Neither question is answerable from the other.
DELTA_OPTION(const char*, kNoZWritePs, "DELTA_GPU_NOZWRITE_PS", nullptr);
// DELTA_GPU_ZWRITE_PS=<list>: force depth WRITE ON for named shaders. The point
// is measurement, not correctness: a pass that only ever READS the depth plane
// leaves no trace of the z it computed, so there is no way to find out where its
// geometry actually lands. Force its write, turn its test off with
// DELTA_GPU_NOZTEST_PS so every fragment gets through, and the depth plane then
// holds that pass's own z for you to read back. It corrupts the plane for
// everything downstream, so it is a probe and nothing else.
DELTA_OPTION(const char*, kZWritePs, "DELTA_GPU_ZWRITE_PS", nullptr);

// Comma-separated list, because a target is routinely written by more than one
// pass: P.T.'s light buffers take 34 draws from one shader and 28 from another,
// and disabling the depth test on either alone proves nothing about the pair.
base::Vector<u64> ParsePsList(const char* e, const char* tag) {
  base::Vector<u64> out;
  if (e)
    for (const char* p = e; *p;) {
      while (*p == ',' || *p == ' ')
        p++;
      if (!*p)
        break;
      out.push_back(std::strtoull(p, nullptr, 0));
      while (*p && *p != ',')
        p++;
    }
  if (!out.empty()) {
    base::String armed;
    base::FormatTo(armed, "armed for {} shader(s):", out.size());
    for (u64 v : out)
      base::FormatTo(armed, " {:#x}", (unsigned long long)v);
    BASE_LOGI(tag, "{}", armed.c_str());
  }
  return out;
}

bool NoZTestForPs(u64 ps) {
  static const base::Vector<u64> list = ParsePsList(kNoZTestPs, "nozps");
  if (list.empty() || !ps)
    return false;
  for (u64 v : list)
    if (v == ps)
      return true;
  return false;
}

bool NoZWriteForPs(u64 ps) {
  static const base::Vector<u64> list = ParsePsList(kNoZWritePs, "nozwps");
  if (list.empty() || !ps)
    return false;
  for (u64 v : list)
    if (v == ps)
      return true;
  return false;
}

bool ForceZWriteForPs(u64 ps) {
  static const base::Vector<u64> list = ParsePsList(kZWritePs, "zwps");
  if (list.empty() || !ps)
    return false;
  for (u64 v : list)
    if (v == ps)
      return true;
  return false;
}
}  // namespace

namespace gpu::render {

using render::DrawInfo;

rhi::StencilOp StencilOp(u32 op) {
  switch (op & 0xF) {
    case 1:
      return rhi::StencilOp::kZero;
    case 2:
    case 3:
    case 4:
      return rhi::StencilOp::kReplace;
    case 5:
      return rhi::StencilOp::kIncrementClamp;
    case 6:
      return rhi::StencilOp::kDecrementClamp;
    case 7:
      return rhi::StencilOp::kInvert;
    case 8:
      return rhi::StencilOp::kIncrementWrap;
    case 9:
      return rhi::StencilOp::kDecrementWrap;
    default:
      return rhi::StencilOp::kKeep;
  }
}

rhi::StencilFace StencilState(const DrawInfo& d, bool back) {
  const u32 shift = back ? 12 : 0;
  const u32 refmask = back ? d.stencil_refmask_bf : d.stencil_refmask;
  rhi::StencilFace state;
  state.fail = StencilOp(d.stencil_control >> shift);
  state.pass = StencilOp(d.stencil_control >> (shift + 4));
  state.depth_fail = StencilOp(d.stencil_control >> (shift + 8));
  // ZFUNC and STENCILFUNC use the same order as rhi::CompareOp.
  state.compare = static_cast<rhi::CompareOp>(
      (d.depth_control >> (back ? 20 : 8)) & 0x7);
  state.compare_mask = (refmask >> 8) & 0xFF;
  state.write_mask = (refmask >> 16) & 0xFF;
  state.reference = refmask & 0xFF;
  return state;
}

RecompPipe* RecompiledPipelineCache::Find(u64 key) {
  const auto it = pipelines_.find(key);
  return it == pipelines_.end() ? nullptr : &it->second;
}

RecompPipe* RecompiledPipelineCache::Store(u64 key, RecompPipe pipeline) {
  return &pipelines_.emplace(key, base::move(pipeline)).first->second;
}

// Build a graphics pipeline for the colored (textured=false) or textured quad
// with the given colour-blend attachment. Shaders + layout selected by
// `textured`.
rhi::Pipeline* BuildPipeline(bool textured,
                             const rhi::BlendAttachment& blend,
                             rhi::Format color_format) {
  rhi::GraphicsPipelineDesc desc;
  desc.layout = textured ? g_quad.tex_layout : g_quad.layout;
  desc.vertex = textured ? rhi::ShaderCode{tex_vert_spv, base::ArraySize(tex_vert_spv)}
                         : rhi::ShaderCode{quad_vert_spv,
                                           base::ArraySize(quad_vert_spv)};
  desc.fragment = textured
                      ? rhi::ShaderCode{tex_frag_spv, base::ArraySize(tex_frag_spv)}
                      : rhi::ShaderCode{quad_frag_spv,
                                        base::ArraySize(quad_frag_spv)};
  // Interleaved repacked vertex: pos.xy@0, color.rgba@8, uv.xy@24, stride 32.
  desc.vertex_buffers = {{32, false}};
  desc.vertex_attributes = {{0, 0, rhi::Format::kRG32Float, 0},
                            {1, 0, rhi::Format::kRGBA32Float, 8},
                            {2, 0, rhi::Format::kRG32Float, 24}};
  // GNM draws are indexed triangle LISTS (VGT_PRIMITIVE_TYPE 4); the previous
  // hardcoded strip connected separate sprites into long diagonal triangles.
  desc.topology = rhi::Topology::kTriangleList;
  desc.color_count = 1;
  desc.color_formats[0] = (color_format);
  desc.blend[0] = blend;
  desc.name = textured ? "quad textured" : "quad";
  return Device().CreateGraphicsPipeline(desc);
}

// Pipeline for a draw's blend state, cached. Returns the default src-alpha
// pipeline when the per-state build fails so a draw never silently drops.
rhi::Pipeline* GetPipeline(bool textured,
                           u32 bc,
                           bool en,
                           rhi::Format color_format) {
  u64 key = (textured ? 1ull : 0) | (en ? 2ull : 0) |
                 ((u64)(en ? (bc & 0x7FFFFFFFu) : 0u) << 2);
  key = HashWord(key, static_cast<u64>(color_format));
  auto it = g_quad.cache.find(key);
  if (it != g_quad.cache.end())
    return it->second;
  rhi::Pipeline* p =
      BuildPipeline(textured, BlendAttachment(bc, en), color_format);
  if (!p && color_format == kDefaultRtFormat)
    p = textured ? g_quad.tex_pipeline : g_quad.pipeline;
  g_quad.cache[key] = p;
  return p;
}

namespace {

// Classic src-alpha over (the default quad blend).
rhi::BlendAttachment SrcAlphaOver(rhi::BlendFactor dst_alpha) {
  rhi::BlendAttachment b;
  b.enable = true;
  b.src_color = rhi::BlendFactor::kSrcAlpha;
  b.dst_color = rhi::BlendFactor::kOneMinusSrcAlpha;
  b.src_alpha = rhi::BlendFactor::kOne;
  b.dst_alpha = dst_alpha;
  return b;
}

}  // namespace

bool CreatePipeline() {
  if (g_quad.pipeline)
    return true;
  rhi::PipelineLayoutDesc layout;
  layout.push_constant_bytes = 64;  // mat4
  layout.push_constant_stages = rhi::kStageVertex;
  g_quad.layout = Device().CreatePipelineLayout(layout);
  if (!g_quad.layout)
    return false;
  // Default colored pipeline: classic src-alpha (used as the fallback / for
  // draws that don't enable blend the cache builds an opaque one on demand).
  g_quad.pipeline =
      BuildPipeline(false, SrcAlphaOver(rhi::BlendFactor::kZero),
                    kDefaultRtFormat);
  if (!g_quad.pipeline) {
    BASE_LOGI("gpuvk", "pipeline failed");
    return false;
  }
  return true;
}

bool CreateTexPipeline() {
  if (g_quad.tex_pipeline)
    return true;
  if (!CreateTextureDescriptors())
    return false;
  rhi::PipelineLayoutDesc layout;
  layout.groups = {g_tex.layout};
  layout.push_constant_bytes = 68;  // mat4 + clipUV flag
  layout.push_constant_stages = rhi::kStageVertex;
  g_quad.tex_layout = Device().CreatePipelineLayout(layout);
  if (!g_quad.tex_layout)
    return false;
  // Default textured pipeline: src-alpha over (the common sprite blend).
  // Per-draw blend states build their own pipeline on demand via GetPipeline().
  g_quad.tex_pipeline = BuildPipeline(
      true, SrcAlphaOver(rhi::BlendFactor::kOneMinusSrcAlpha),
      kDefaultRtFormat);
  if (!g_quad.tex_pipeline) {
    BASE_LOGI("gpuvk", "tex pipeline failed");
    return false;
  }
  return true;
}

// Build (or fetch) the pipeline for a recompiled draw, keyed by the shader pair
// + blend state + vertex layout.
RecompPipe* GetRecompPipe(const DrawInfo& d) {
  if (d.recomp->ps_texs.size() > kMaxTex)
    return nullptr;
  u32 mrt_n = base::Min(d.mrt_count, 8u);
  // Depth + primitive-setup state folded into the pipeline key (mixed through
  // an FNV prime so it spreads across the whole 64-bit space, away from the
  // blend/stride bits).
  u32 dstate = (d.depth_base ? 1u : 0u) | (d.depth_test_enable ? 2u : 0u) |
                    (d.depth_write_enable ? 4u : 0u) |
                    ((d.depth_func & 7u) << 3) | ((d.prim_type & 0x1Fu) << 6) |
                    ((d.cull_mode & 3u) << 11) |
                    (d.front_ccw ? 0u : (1u << 13));
  u64 key =
      d.vs_addr * 0x9e3779b97f4a7c15ull ^ d.ps_addr ^
      ((u64)(d.blend_enable ? (d.blend_control & 0x7FFFFFFFu) : 0) << 1) ^
      ((u64)d.vertex_stride << 33) ^ ((u64)mrt_n << 60) ^
      ((u64)dstate * 0x100000001b3ull);
  // The MODULE, not the code address. A pipeline embeds the shader modules it
  // was created with, and the recompiler builds a different module for the same
  // code whenever a descriptor-derived mask differs (tex_3d, tex_1d, tex_uint,
  // mrt_uint, the PS input mapping). Keying on the addresses and only SOME of
  // those masks let a draw bind a pipeline holding another draw's module, whose
  // declarations then disagreed with the descriptor set built for this one (a
  // binding the executing module reads as integer took the UNORM default,
  // VUID-vkCmdDrawIndexed-format-07753). The Recompiled is cached per (code,
  // every mask) and never evicted, so its address IS that identity.
  key = HashWord(key, reinterpret_cast<u64>(d.recomp));
  key = HashWord(key, d.ps4_neo ? 1 : 0);
  key = HashWord(key, d.stencil_enable ? d.depth_control : 0);
  key = HashWord(key, d.stencil_enable ? d.stencil_control : 0);
  key = HashWord(key, d.stencil_enable ? d.stencil_refmask : 0);
  key = HashWord(key, d.stencil_enable && d.stencil_backface_enable
                          ? d.stencil_refmask_bf
                          : 0);
  key = HashWord(key, d.num_vattrs);
  // A sampler reading a volume image translates the same PS address to a
  // different module, whose image types the pipeline layout has to match.
  u64 tex_3d_mask = 0, tex_1d_mask = 0;
  for (u32 i = 0; i < d.num_texs && i < kMaxTex; i++) {
    if (d.texs[i].is_3d)
      tex_3d_mask |= 1ull << i;
    if (d.texs[i].is_1d)
      tex_1d_mask |= 1ull << i;
  }
  key = HashWord(key, tex_3d_mask);
  key = HashWord(key, tex_1d_mask);
  key = HashWord(key, d.target_mask);
  key = HashWord(key, d.shader_mask);
  for (u32 i = 0; i < mrt_n; i++)
    key = HashWord(key, static_cast<u64>(ColorTargetFormat(d.mrt_info[i])));
  // The vertex-input layout (binding count + per-binding strides + per-attr
  // binding assignment) is baked into the pipeline, so it must be part of the
  // key or a later multi-stream draw would reuse a single-stream pipeline (or
  // vice versa) for the same shader pair.
  key = HashWord(key, d.num_vbufs);
  for (u32 j = 0; j < d.num_vbufs; j++)
    key = HashWord(key, d.vbufs[j].stride |
                            (d.vbufs[j].per_instance ? 1ull << 32 : 0));
  for (u32 i = 0; i < d.num_vattrs; i++) {
    key = HashWord(key, d.vattrs[i].location);
    key = HashWord(key, d.vattrs[i].binding);
    key = HashWord(key, d.vattrs[i].offset);
    key = HashWord(key, d.vattrs[i].num_comps);
    key = HashWord(key, d.vattrs[i].dfmt);
    key = HashWord(key, d.vattrs[i].nfmt);
  }
  if (RecompPipe* pipeline = g_recomp_cache.Find(key))
    return pipeline;
  RecompPipe rp;
  const bool mesh = !d.recomp->mesh_spirv.empty();
  const rhi::Caps& caps = Device().caps();
  if (mesh && !caps.mesh_shader)
    return nullptr;
  const u32 vertex_stage = mesh ? rhi::kStageMesh : rhi::kStageVertex;
  rp.textured = !d.recomp->ps_texs.empty() || !d.recomp->vs_texs.empty();
  const bool has_storage =
      base::AnyOf(d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                  [](const gcn::ShaderTex& tex) { return tex.storage; });
  // A vertex texture fetch takes a binding of its own in set 0, so the exact
  // per-binding layout below is the only one that can describe the draw.
  const size_t n_tex = d.recomp->ps_texs.size() + d.recomp->vs_texs.size();
  rp.multi_tex = n_tex > 1 || has_storage || !d.recomp->vs_texs.empty();

  // set 0 = texture(s) (or an empty layout when untextured), set 1 = cbuffer
  // UBO. Multi/storage shaders use an exact per-binding descriptor layout;
  // single-sampler shaders retain the shared one-binding layout.
  rhi::BindGroupLayout* set0 = !rp.textured ? g_ring.empty_layout : g_tex.layout;
  if (rp.multi_tex) {
    rhi::BindGroupLayoutDesc desc;
    const u32 n_bind = static_cast<u32>(base::Min(n_tex, size_t(kMaxTex)));
    for (u32 i = 0; i < n_bind; i++) {
      const bool is_vs = i >= d.recomp->ps_texs.size();
      const bool storage = !is_vs && d.recomp->ps_texs[i].storage;
      desc.bindings.push_back(
          {i,
           storage ? rhi::BindingType::kStorageTexture
                   : rhi::BindingType::kSampledTexture,
           is_vs ? vertex_stage : rhi::kStageFragment});
    }
    rp.tex_set_layout = Device().CreateBindGroupLayout(desc);
    if (!rp.tex_set_layout)
      return nullptr;
    rp.tex_bindings = n_bind;
    set0 = rp.tex_set_layout;
  }
  rp.raw_bufs = !d.recomp->vs_bufs.empty() || !d.recomp->ps_bufs.empty();
  // A stage that stages through LDS needs the set-3 scratch, and a descriptor
  // set is positional: taking it means taking set 2 as well, whether or not
  // the shader reads a raw buffer.
  rp.shared_lds = d.recomp->shared_lds && EnsureLdsScratch();
  rhi::BindGroupLayout* cbuf_layout = d.recomp->indirect_cbufs
                                          ? g_ring.indirect_cbuf_layout
                                          : g_ring.ubo_layout;
  if (!cbuf_layout)
    return nullptr;
  rhi::PipelineLayoutDesc layout;
  layout.groups = {set0, cbuf_layout};
  if (rp.raw_bufs || rp.shared_lds)
    layout.groups.push_back(g_ring.sbo_layout);
  if (rp.shared_lds)
    layout.groups.push_back(g_ring.lds_layout);
  // One 64-byte window per stage: 16 user-data dwords each, 128 bytes total
  // (the guaranteed minimum push-constant size), plus each stage's own
  // code-address words (VS 128..135, PS 136..143) pushed per draw for
  // s_getpc_b64.
  // ONE range naming both stages, not one range per stage. The two stages'
  // windows interleave (VS owns [0,64) and [128,136), PS [64,128) and
  // [136,144)), so no pair of per-stage ranges can cover that without
  // overlapping, and Vulkan requires a push to name every stage of every range
  // it overlaps. Every push names both stages.
  const bool pc_base = gpu::gcn::PushCodeBase();
  layout.push_constant_bytes = mesh ? 160u : pc_base ? 144u : 128u;
  layout.push_constant_stages = vertex_stage | rhi::kStageFragment;
  rp.layout = Device().CreatePipelineLayout(layout);
  if (!rp.layout)
    return nullptr;

  rhi::GraphicsPipelineDesc pd;
  pd.layout = rp.layout;
  pd.mesh = mesh;
  pd.vertex = rhi::Code(mesh ? d.recomp->mesh_spirv : d.recomp->vs_spirv);
  pd.fragment = rhi::Code(d.recomp->fs_spirv);
  // RECTLIST is primitive type 17 on GFX7 but 7 on gfx10.3 (PrimitiveType::
  // kRectList; 17 is kRectListLegacy there). Missing the gfx10 number rendered
  // every PS5 fullscreen pass as a single triangle covering half the rect.
  const bool is_rect_list = d.prim_type == 17 || d.prim_type == 7;
  bool use_gs = !mesh && is_rect_list && !kNoRectGs && caps.geometry_shader &&
                !d.recomp->gs_spirv.empty();
  if (d.recomp->guest_gs) {
    if (!caps.geometry_shader)
      return nullptr;
    use_gs = true;
  }
  if (use_gs)
    pd.geometry = rhi::Code(d.recomp->gs_spirv);

  // One binding per resolved vertex buffer (single-stream draws stay a single
  // binding, identical to before); attributes reference their binding.
  const u32 nbind = d.num_vattrs ? base::Min(d.num_vbufs, 8u) : 0;
  for (u32 j = 0; j < nbind; j++)
    pd.vertex_buffers.push_back({d.vbufs[j].stride, d.vbufs[j].per_instance});
  for (u32 i = 0; i < d.num_vattrs; i++)
    pd.vertex_attributes.push_back(
        {d.vattrs[i].location, d.vattrs[i].binding,
         (VertexFormat(d.vattrs[i].dfmt, d.vattrs[i].nfmt)),
         d.vattrs[i].offset});
  pd.topology = PrimitiveTopology(d.prim_type);
  // Face culling from PA_SU_SC_MODE_CNTL (CULL_FRONT[0]/CULL_BACK[1] map 1:1
  // onto the cull-mode bits). The render region uses a negative-height (y-up)
  // viewport to match GCN rasterisation, which flips triangle winding in
  // framebuffer space, so the guest's front-face sense is inverted here to
  // compensate. Culling is opt-in (DELTA_GPU_CULL=1) until the winding can be
  // validated against visible 3D geometry; the default stays cull-none so
  // correctly-drawn faces are never dropped.
  pd.cull = kDoCull ? static_cast<rhi::CullMode>(d.cull_mode & 0x3)
                    : rhi::CullMode::kNone;
  pd.front_ccw = !d.front_ccw;
  pd.depth_clamp = kDepthClamp;
  // Depth test/write from DB_DEPTH_CONTROL (only when the draw bound a Z
  // buffer; 2D draws leave depth_base 0 so this stays fully disabled, unchanged
  // from before).
  if (d.depth_base) {
    const bool per_ps = NoZTestForPs(d.ps_addr);
    const bool skip_ztest = kNoZTest || per_ps;
    // Say so, once per shader. A knob that silently does not match its target
    // produces a null result indistinguishable from a real one, and this
    // title's record is full of exactly that mistake.
    if (per_ps) {
      static base::Vector<u64> said;
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI("nozps",
                  "depth test disabled for ps={:#x} (test_enable was {}, func {})",
                  (unsigned long long)d.ps_addr, (int)d.depth_test_enable,
                  d.depth_func & 0x7);
      }
    }
    pd.depth_test = d.depth_test_enable && !skip_ztest;
    const bool no_write = NoZWriteForPs(d.ps_addr);
    if (no_write) {
      static base::Vector<u64> said;
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI("nozwps", "depth write disabled for ps={:#x} (was {})",
                  (unsigned long long)d.ps_addr, (int)d.depth_write_enable);
      }
    }
    const bool force_write = ForceZWriteForPs(d.ps_addr);
    if (force_write) {
      static base::Vector<u64> said;
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI("zwps", "depth write FORCED for ps={:#x} (was {})",
                  (unsigned long long)d.ps_addr, (int)d.depth_write_enable);
      }
    }
    pd.depth_write = (d.depth_write_enable || force_write) && !no_write;
    pd.depth_compare = static_cast<rhi::CompareOp>(d.depth_func & 0x7);
    // A depth-prepass title re-draws its geometry with ZFUNC=EQUAL against the
    // depth the prepass laid down. That only works when both passes compute
    // gl_Position bit-identically, which a hardware driver guarantees for one
    // shader but we cannot: the title uses a position-only VS for the prepass
    // and the full VS for the shaded pass, and our two SPIR-V modules are
    // optimised independently, so the interpolated depth differs by an ULP and
    // EQUAL rejects the whole scene. Widen EQUAL to the direction the prepass
    // wrote, which admits exactly the surface the prepass kept (nothing can be
    // nearer than the nearest surface), so the visible result matches.
    if (kRelaxDepthEqual && pd.depth_compare == rhi::CompareOp::kEqual &&
        !d.depth_write_enable)
      pd.depth_compare = d.depth_clear <= 0.5f
                             ? rhi::CompareOp::kGreaterEqual
                             : rhi::CompareOp::kLessEqual;
    if (d.stencil_enable) {
      pd.stencil_test = true;
      pd.stencil_front = StencilState(d, false);
      pd.stencil_back =
          d.stencil_backface_enable ? StencilState(d, true) : pd.stencil_front;
    }
  }
  // One blend attachment per bound MRT target, each from its own
  // CB_BLENDn_CONTROL (mrt_blend[i] / mrt_blend_mask bit i); target 0 mirrors
  // blend_control/blend_enable so the single-RT path is unchanged. Targets the
  // PS does not export to are write-masked off so they keep their loaded
  // content.
  pd.color_count = mrt_n;
  for (u32 i = 0; i < mrt_n; i++) {
    u32 bc = i == 0 ? d.blend_control : d.mrt_blend[i];
    bool en = i == 0 ? d.blend_enable : ((d.mrt_blend_mask >> i) & 1u);
    // Blending an integer attachment is invalid, and the hardware agrees:
    // CB_COLORn_INFO sets BLEND_BYPASS on exactly these targets.
    if (IsIntegerColorFormat(ColorTargetFormat(d.mrt_info[i])))
      en = false;
    pd.blend[i] = BlendAttachment(bc, en);
    // Only exported targets may write, and CB_TARGET_MASK always gates each
    // component, including when the frontend omits CB_SHADER_MASK (AGC).
    if (!kNoMaskDiag)
      pd.blend[i].write_mask = ColorWriteMask(
          d.target_mask, d.shader_mask, d.recomp->ps_mrt_mask, i);
    pd.color_formats[i] = (ColorTargetFormat(d.mrt_info[i]));
  }
  // DELTA_GPU_PIPETRACE: the colour-blend state a pipeline is actually built
  // with, next to the PS's export mask. The two have to agree or an
  // attachment is silently write-masked off (or written unblended).
  if (kGpuPipetrace &&
      (kGpuPipetrace == 1 || d.ps_addr == kGpuPipetrace)) {
    static int n = 0;
    if (n++ < 24)
      BASE_LOGI("pipe",
                "ps={:#x} mrtN={} psMrtMask={:#x} tmask={:#x} smask={:#x} "
                "att0: en={} src={} dst={} src_a={} dst_a={} writeMask={:#x}",
                (unsigned long)d.ps_addr, mrt_n, d.recomp->ps_mrt_mask,
                d.target_mask, d.shader_mask, (int)pd.blend[0].enable,
                (int)pd.blend[0].src_color, (int)pd.blend[0].dst_color,
                (int)pd.blend[0].src_alpha, (int)pd.blend[0].dst_alpha,
                pd.blend[0].write_mask);
  }
  if (d.depth_base)
    pd.depth_format = (kDepthFormat);
  if (d.stencil_enable)
    pd.stencil_format = (kDepthFormat);
  char name[64];
  std::snprintf(name, sizeof(name), "recomp vs=%#llx ps=%#llx",
                (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr);
  pd.name = name;
  rp.pipe = Device().CreateGraphicsPipeline(pd);
  if (!rp.pipe)
    return nullptr;
  return g_recomp_cache.Store(key, base::move(rp));
}

}  // namespace gpu::render
