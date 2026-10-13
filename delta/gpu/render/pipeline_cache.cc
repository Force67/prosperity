/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/pipeline_cache.h"
#include "base/arch.h"
#include "guest/session.h"
#include "memory_debug/memory_debug.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/hash.h"
#include "gpu/render/labels.h"
#include "gpu/render/mesh_args.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/upload_ring.h"

#include <cstdio>
#include <cstdlib>

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kDoCull, "DELTA_GPU_CULL", false);
DELTA_OPTION(bool, kDepthClamp, "DELTA_GPU_DEPTHCLAMP", false);
DELTA_OPTION(u64, kGpuPipetrace, "DELTA_GPU_PIPETRACE", 0);
DELTA_OPTION(bool, kNoMaskDiag, "DELTA_GPU_NOMASK", false);
DELTA_OPTION(bool, kNoRectGs, "DELTA_GPU_NORECTGS", false);
DELTA_OPTION(bool, kRelaxDepthEqual, "DELTA_GPU_RELAX_ZEQUAL", true);
DELTA_OPTION(bool, kNoZTest, "DELTA_GPU_NOZTEST", false);
DELTA_OPTION(const char*, kNoZTestPs, "DELTA_GPU_NOZTEST_PS", nullptr);
DELTA_OPTION(const char*, kNoZWritePs, "DELTA_GPU_NOZWRITE_PS", nullptr);
DELTA_OPTION(const char*, kZWritePs, "DELTA_GPU_ZWRITE_PS", nullptr);

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
  static const base::Vector<u64> kList = ParsePsList(kNoZTestPs, "nozps");
  if (kList.empty() || !ps)
    return false;
  for (u64 v : kList)
    if (v == ps)
      return true;
  return false;
}

bool NoZWriteForPs(u64 ps) {
  static const base::Vector<u64> kList = ParsePsList(kNoZWritePs, "nozwps");
  if (kList.empty() || !ps)
    return false;
  for (u64 v : kList)
    if (v == ps)
      return true;
  return false;
}

bool ForceZWriteForPs(u64 ps) {
  static const base::Vector<u64> kList = ParsePsList(kZWritePs, "zwps");
  if (kList.empty() || !ps)
    return false;
  for (u64 v : kList)
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
  state.compare =
      static_cast<rhi::CompareOp>((d.depth_control >> (back ? 20 : 8)) & 0x7);
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

namespace {
DELTA_OPTION(bool, kAsyncPipelines, "DELTA_GPU_ASYNC_PIPELINES", true);

void AdoptBuilt(RecompPipe& rp) {
  if (!rp.pending || !rp.pending->done.load(base::memory_order_acquire))
    return;
  rp.pipe = rp.pending->pipe;
  rp.failed = !rp.pipe;
  rp.pending = nullptr;
}

void BuildLater(RecompPipe& rp, const rhi::GraphicsPipelineDesc& pd) {
  auto pending = base::MakeShared<PendingPipeline>();
  const auto copy = [](const rhi::ShaderCode& code, base::Vector<u32>& to) {
    to.assign(code.words, code.words + code.count);
    return rhi::Code(to);
  };
  pending->desc = pd;
  pending->desc.vertex = copy(pd.vertex, pending->vertex);
  pending->desc.geometry = copy(pd.geometry, pending->geometry);
  pending->desc.fragment = copy(pd.fragment, pending->fragment);
  pending->name = pd.name ? pd.name : "";
  pending->desc.name = pending->name.c_str();
  rp.pending = pending;
  gcn::PostShaderWork([pending] {
    pending->pipe = Device().CreateGraphicsPipeline(pending->desc);
    pending->vertex = {};
    pending->geometry = {};
    pending->fragment = {};
    pending->done.store(true, base::memory_order_release);
  });
}
}  // namespace

RecompPipe* GetRecompPipe(const DrawInfo& d) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
  if (d.recomp->ps_texs.size() > kMaxTex)
    return nullptr;
  u32 mrt_n = base::Min(d.mrt_count, 8u);
  u32 dstate = (d.depth_base ? 1u : 0u) | (d.depth_test_enable ? 2u : 0u) |
               (d.depth_write_enable ? 4u : 0u) | ((d.depth_func & 7u) << 3) |
               ((d.prim_type & 0x1Fu) << 6) | ((d.cull_mode & 3u) << 11) |
               (d.front_ccw ? 0u : (1u << 13));
  u64 key = d.vs_addr * 0x9e3779b97f4a7c15ull ^ d.ps_addr ^
            ((u64)(d.blend_enable ? (d.blend_control & 0x7FFFFFFFu) : 0) << 1) ^
            ((u64)d.vertex_stride << 33) ^ ((u64)mrt_n << 60) ^
            ((u64)dstate * 0x100000001b3ull);
  key = HashWord(key, reinterpret_cast<u64>(d.recomp));
  key = HashWord(key, d.ps4_neo ? 1 : 0);
  key = HashWord(key, d.stencil_enable ? d.depth_control : 0);
  key = HashWord(key, d.stencil_enable ? d.stencil_control : 0);
  key = HashWord(key, d.stencil_enable ? d.stencil_refmask : 0);
  key = HashWord(key, d.stencil_enable && d.stencil_backface_enable
                          ? d.stencil_refmask_bf
                          : 0);
  key = HashWord(key, d.num_vattrs);
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
  key = HashWord(key, d.num_vbufs);
  for (u32 j = 0; j < d.num_vbufs; j++)
    key = HashWord(
        key, d.vbufs[j].stride | (d.vbufs[j].per_instance ? 1ull << 32 : 0));
  for (u32 i = 0; i < d.num_vattrs; i++) {
    key = HashWord(key, d.vattrs[i].location);
    key = HashWord(key, d.vattrs[i].binding);
    key = HashWord(key, d.vattrs[i].offset);
    key = HashWord(key, d.vattrs[i].num_comps);
    key = HashWord(key, d.vattrs[i].dfmt);
    key = HashWord(key, d.vattrs[i].nfmt);
  }
  if (RecompPipe* pipeline = g_recomp_cache.Find(key)) {
    AdoptBuilt(*pipeline);
    return pipeline;
  }
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
  const size_t n_tex = d.recomp->ps_texs.size() + d.recomp->vs_texs.size();
  rp.multi_tex = n_tex > 1 || has_storage || !d.recomp->vs_texs.empty();

  rhi::BindGroupLayout* set0 =
      !rp.textured ? g_ring.empty_layout : g_tex.layout;
  if (rp.multi_tex) {
    rhi::BindGroupLayoutDesc desc;
    const u32 n_bind = static_cast<u32>(base::Min(n_tex, size_t(kMaxTex)));
    for (u32 i = 0; i < n_bind; i++) {
      const bool is_vs = i >= d.recomp->ps_texs.size();
      const bool storage = !is_vs && d.recomp->ps_texs[i].storage;
      desc.bindings.push_back({i,
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
  for (const auto* plans :
       {&d.recomp->vs_cbufs, &d.recomp->gs_cbufs, &d.recomp->ps_cbufs}) {
    for (const auto& cb : *plans) {
      const u32 limit =
          d.recomp->indirect_cbufs ? gcn::kIndirectCbufBindings : kCbufBindings;
      if (cb.binding >= limit)
        return nullptr;
      rp.cbuf_count = base::Max(rp.cbuf_count, cb.binding + 1);
      rp.cbuf_mask |= 1ull << cb.binding;
    }
  }
  for (const auto* plans : {&d.recomp->vs_bufs, &d.recomp->ps_bufs}) {
    for (const auto& buffer : *plans) {
      if (buffer.binding >= g_ring.sbo_count)
        return nullptr;
      rp.raw_count = base::Max(rp.raw_count, buffer.binding + 1);
      rp.raw_mask |= 1u << buffer.binding;
    }
  }
  rp.shared_lds = d.recomp->shared_lds && EnsureLdsScratch();
  rhi::BindGroupLayout* cbuf_layout =
      d.recomp->indirect_cbufs ? g_ring.indirect_cbuf_layout
      : rp.cbuf_count          ? GetCbufLayout(rp.cbuf_count)
                               : g_ring.empty_layout;
  if (!cbuf_layout)
    return nullptr;
  rhi::PipelineLayoutDesc layout;
  layout.groups = {set0, cbuf_layout};
  if (rp.raw_count || rp.shared_lds || mesh) {
    auto* raw_layout =
        rp.raw_count ? GetRawBufferLayout(rp.raw_count) : g_ring.empty_layout;
    if (!raw_layout)
      return nullptr;
    layout.groups.push_back(raw_layout);
  }
  if (rp.shared_lds || mesh)
    layout.groups.push_back(mesh ? MeshArgsLayout() : g_ring.lds_layout);
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

  const u32 nbind = d.num_vattrs ? base::Min(d.num_vbufs, 8u) : 0;
  for (u32 j = 0; j < nbind; j++)
    pd.vertex_buffers.push_back({d.vbufs[j].stride, d.vbufs[j].per_instance});
  for (u32 i = 0; i < d.num_vattrs; i++)
    pd.vertex_attributes.push_back(
        {d.vattrs[i].location, d.vattrs[i].binding,
         (VertexFormat(d.vattrs[i].dfmt, d.vattrs[i].nfmt)),
         d.vattrs[i].offset});
  pd.topology = PrimitiveTopology(d.prim_type);
  pd.cull = kDoCull ? static_cast<rhi::CullMode>(d.cull_mode & 0x3)
                    : rhi::CullMode::kNone;
  pd.front_ccw = !d.front_ccw;
  pd.depth_clamp = kDepthClamp;
  if (d.depth_base) {
    const bool per_ps = NoZTestForPs(d.ps_addr);
    const bool skip_ztest = kNoZTest || per_ps;
    if (per_ps) {
      static base::Vector<u64> said;
      static const guest::SessionReset reset_said(
          [] { guest::ResetResource(said); });
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI(
            "nozps",
            "depth test disabled for ps={:#x} (test_enable was {}, func {})",
            (unsigned long long)d.ps_addr, (int)d.depth_test_enable,
            d.depth_func & 0x7);
      }
    }
    pd.depth_test = d.depth_test_enable && !skip_ztest;
    const bool no_write = NoZWriteForPs(d.ps_addr);
    if (no_write) {
      static base::Vector<u64> said;
      static const guest::SessionReset reset_said(
          [] { guest::ResetResource(said); });
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI("nozwps", "depth write disabled for ps={:#x} (was {})",
                  (unsigned long long)d.ps_addr, (int)d.depth_write_enable);
      }
    }
    const bool force_write = ForceZWriteForPs(d.ps_addr);
    if (force_write) {
      static base::Vector<u64> said;
      static const guest::SessionReset reset_said(
          [] { guest::ResetResource(said); });
      if (base::Find(said.begin(), said.end(), d.ps_addr) == said.end()) {
        said.push_back(d.ps_addr);
        BASE_LOGI("zwps", "depth write FORCED for ps={:#x} (was {})",
                  (unsigned long long)d.ps_addr, (int)d.depth_write_enable);
      }
    }
    pd.depth_write = (d.depth_write_enable || force_write) && !no_write;
    pd.depth_compare = static_cast<rhi::CompareOp>(d.depth_func & 0x7);
    // Depth-prepass EQUAL needs bit-identical gl_Position across two optimised
    // modules.
    if (kRelaxDepthEqual && pd.depth_compare == rhi::CompareOp::kEqual &&
        !d.depth_write_enable)
      pd.depth_compare = d.depth_clear <= 0.5f ? rhi::CompareOp::kGreaterEqual
                                               : rhi::CompareOp::kLessEqual;
    if (d.stencil_enable) {
      pd.stencil_test = true;
      pd.stencil_front = StencilState(d, false);
      pd.stencil_back =
          d.stencil_backface_enable ? StencilState(d, true) : pd.stencil_front;
    }
  }
  pd.color_count = mrt_n;
  for (u32 i = 0; i < mrt_n; i++) {
    u32 bc = i == 0 ? d.blend_control : d.mrt_blend[i];
    bool en = i == 0 ? d.blend_enable : ((d.mrt_blend_mask >> i) & 1u);
    // Integer attachments cannot blend; the hardware sets BLEND_BYPASS for
    // them.
    if (IsIntegerColorFormat(ColorTargetFormat(d.mrt_info[i])))
      en = false;
    pd.blend[i] = BlendAttachment(bc, en);
    if (!kNoMaskDiag)
      pd.blend[i].write_mask = ColorWriteMask(d.target_mask, d.shader_mask,
                                              d.recomp->ps_mrt_mask, i);
    pd.color_formats[i] = (ColorTargetFormat(d.mrt_info[i]));
  }
  if (kGpuPipetrace && (kGpuPipetrace == 1 || d.ps_addr == kGpuPipetrace)) {
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
  if (kAsyncPipelines) {
    BuildLater(rp, pd);
    return g_recomp_cache.Store(key, base::move(rp));
  }
  rp.pipe = Device().CreateGraphicsPipeline(pd);
  if (!rp.pipe)
    return nullptr;
  return g_recomp_cache.Store(key, base::move(rp));
}

}  // namespace gpu::render
