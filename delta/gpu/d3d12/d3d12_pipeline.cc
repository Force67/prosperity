/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Root signatures, pipeline state objects and the shader cache.
 */

#include <cstring>

#include "base/logging.h"

#include "base/containers/vector.h"
#include "base/hashing/hash.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "gpu/d3d12/d3d12_internal.h"
#include "ui/shader_activity.h"

namespace gpu::d3d12::impl {

namespace {

// The root signature budget, in DWORDs.
constexpr u32 kRootBudget = 64;

D3D12_BLEND ToBlend(rhi::BlendFactor f, bool alpha) {
  switch (f) {
    case rhi::BlendFactor::kZero:
      return D3D12_BLEND_ZERO;
    case rhi::BlendFactor::kOne:
      return D3D12_BLEND_ONE;
    case rhi::BlendFactor::kSrcColor:
      return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
    case rhi::BlendFactor::kOneMinusSrcColor:
      return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
    case rhi::BlendFactor::kDstColor:
      return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
    case rhi::BlendFactor::kOneMinusDstColor:
      return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
    case rhi::BlendFactor::kSrcAlpha:
      return D3D12_BLEND_SRC_ALPHA;
    case rhi::BlendFactor::kOneMinusSrcAlpha:
      return D3D12_BLEND_INV_SRC_ALPHA;
    case rhi::BlendFactor::kDstAlpha:
      return D3D12_BLEND_DEST_ALPHA;
    case rhi::BlendFactor::kOneMinusDstAlpha:
      return D3D12_BLEND_INV_DEST_ALPHA;
    // D3D12 has one blend factor register for all four channels; a constant
    // alpha factor reads the constant's own channel, not its alpha.
    case rhi::BlendFactor::kConstantColor:
    case rhi::BlendFactor::kConstantAlpha:
      return D3D12_BLEND_BLEND_FACTOR;
    case rhi::BlendFactor::kOneMinusConstantColor:
    case rhi::BlendFactor::kOneMinusConstantAlpha:
      return D3D12_BLEND_INV_BLEND_FACTOR;
    case rhi::BlendFactor::kSrcAlphaSaturate:
      return D3D12_BLEND_SRC_ALPHA_SAT;
    case rhi::BlendFactor::kSrc1Color:
      return alpha ? D3D12_BLEND_SRC1_ALPHA : D3D12_BLEND_SRC1_COLOR;
    case rhi::BlendFactor::kOneMinusSrc1Color:
      return alpha ? D3D12_BLEND_INV_SRC1_ALPHA : D3D12_BLEND_INV_SRC1_COLOR;
    case rhi::BlendFactor::kSrc1Alpha:
      return D3D12_BLEND_SRC1_ALPHA;
    case rhi::BlendFactor::kOneMinusSrc1Alpha:
      return D3D12_BLEND_INV_SRC1_ALPHA;
  }
  return D3D12_BLEND_ONE;
}

D3D12_BLEND_OP ToBlendOp(rhi::BlendOp op) {
  return static_cast<D3D12_BLEND_OP>(static_cast<u32>(op) + 1);
}

D3D12_COMPARISON_FUNC ToCompare(rhi::CompareOp op) {
  return static_cast<D3D12_COMPARISON_FUNC>(static_cast<u32>(op) + 1);
}

D3D12_DEPTH_STENCILOP_DESC ToStencil(const rhi::StencilFace& f) {
  auto op = [](rhi::StencilOp o) {
    return static_cast<D3D12_STENCIL_OP>(static_cast<u32>(o) + 1);
  };
  return {op(f.fail), op(f.depth_fail), op(f.pass), ToCompare(f.compare)};
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE TopologyType(rhi::Topology t) {
  switch (t) {
    case rhi::Topology::kPointList:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    case rhi::Topology::kLineList:
    case rhi::Topology::kLineStrip:
    case rhi::Topology::kLineListAdjacency:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    default:
      return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  }
}

D3D_PRIMITIVE_TOPOLOGY Topology(rhi::Topology t) {
  switch (t) {
    case rhi::Topology::kPointList:
      return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
    case rhi::Topology::kLineList:
      return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
    case rhi::Topology::kLineStrip:
      return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
    case rhi::Topology::kTriangleList:
    case rhi::Topology::kTriangleFan:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    case rhi::Topology::kTriangleStrip:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    case rhi::Topology::kLineListAdjacency:
      return D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
    case rhi::Topology::kTriangleListAdjacency:
      return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ;
  }
  return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
}

D3D12_DESCRIPTOR_RANGE_TYPE RangeType(const GroupEntry& e) {
  switch (e.type) {
    case rhi::BindingType::kSampledTexture:
      return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case rhi::BindingType::kStorageTexture:
      return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case rhi::BindingType::kUniformBuffer:
    case rhi::BindingType::kUniformBufferDynamic:
      return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case rhi::BindingType::kStorageBuffer:
    case rhi::BindingType::kStorageBufferDynamic:
      return e.read_only ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                         : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  }
  return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
}

D3D12_ROOT_PARAMETER Constants(u32 reg, u32 count) {
  D3D12_ROOT_PARAMETER p{};
  p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  p.Constants = {reg, kInternalSpace, count};
  p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  return p;
}

const char* kBlitHlsl = R"(
cbuffer Blit : register(b0) {
  float2 uv0;
  float2 uv_scale;
  uint linear_filter;
};
Texture2DArray<float4> src : register(t0);
SamplerState point_sampler : register(s0);
SamplerState linear_sampler : register(s1);

float4 vs(uint id : SV_VertexID, out float2 uv : TEXCOORD0) : SV_Position {
  float2 p = float2((id << 1) & 2, id & 2);
  uv = uv0 + p * uv_scale;
  return float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
}

float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
  float3 c = float3(uv, 0.0);
  return linear_filter != 0 ? src.SampleLevel(linear_sampler, c, 0.0)
                            : src.SampleLevel(point_sampler, c, 0.0);
}
)";

}  // namespace

ID3D12RootSignature* D3D12Device::SerializeRoot(
    const D3D12_ROOT_SIGNATURE_DESC& desc) {
  ID3DBlob* blob = nullptr;
  ID3DBlob* error = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                         &blob, &error))) {
    BASE_LOGI(
        "gpud3d12", "root signature: {}",
        error ? static_cast<const char*>(error->GetBufferPointer()) : "failed");
    SafeRelease(error);
    return nullptr;
  }
  ID3D12RootSignature* root = nullptr;
  if (FAILED(device->CreateRootSignature(
          0, blob->GetBufferPointer(), blob->GetBufferSize(),
          IID_ID3D12RootSignature, reinterpret_cast<void**>(&root))))
    root = nullptr;
  SafeRelease(blob);
  SafeRelease(error);
  return root;
}

rhi::PipelineLayout* D3D12Device::CreatePipelineLayout(
    const rhi::PipelineLayoutDesc& desc) {
  auto layout = base::MakeUnique<D3D12PipelineLayout>(desc);
  // Whole 16-byte rows: the shader's cbuffer is sized in rows.
  layout->push_dwords = (desc.push_constant_bytes + 15) / 16 * 4;

  // What the root costs with every dynamic uniform buffer as a root CBV and
  // push constants as root constants; over budget, the buffers move into
  // their tables first, then push constants move into a root CBV.
  u32 tables = 0, dynamic_ubos = 0;
  for (rhi::BindGroupLayout* g : desc.groups) {
    if (!g)
      continue;
    auto* gl = static_cast<D3D12BindGroupLayout*>(g);
    tables += (gl->table_size ? 1 : 0) + (gl->sampler_count ? 1 : 0);
    for (const GroupEntry& e : gl->entries)
      dynamic_ubos += e.type == rhi::BindingType::kUniformBufferDynamic;
  }
  constexpr u32 kInternal = 1 + 2 + 3 + 3;
  bool root_ubos = true;
  bool push_root_constants = true;
  if (layout->push_dwords + kInternal + tables + 2 * dynamic_ubos >
      kRootBudget) {
    root_ubos = false;
    if (layout->push_dwords + kInternal + tables > kRootBudget)
      push_root_constants = false;
  }

  base::Vector<D3D12_ROOT_PARAMETER> params;
  base::Vector<base::Vector<D3D12_DESCRIPTOR_RANGE>> ranges;
  ranges.reserve(desc.groups.size() * 2);
  if (layout->push_dwords) {
    D3D12_ROOT_PARAMETER p = Constants(kPushRegister, layout->push_dwords);
    if (!push_root_constants) {
      p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
      p.Descriptor = {kPushRegister, kInternalSpace};
      layout->push_cbv = static_cast<i32>(params.size());
    } else {
      layout->push_constants = static_cast<i32>(params.size());
    }
    params.push_back(p);
  }
  layout->raster = static_cast<i32>(params.size());
  params.push_back(Constants(kRasterRegister, 1));
  layout->draw = static_cast<i32>(params.size());
  params.push_back(Constants(kDrawRegister, 2));
  layout->dispatch = static_cast<i32>(params.size());
  params.push_back(Constants(kDispatchRegister, 3));
  layout->group_base = static_cast<i32>(params.size());
  params.push_back(Constants(kGroupBaseRegister, 3));

  layout->groups.resize(desc.groups.size());
  for (u32 set = 0; set < desc.groups.size(); set++) {
    auto* gl = static_cast<D3D12BindGroupLayout*>(desc.groups[set]);
    if (!gl)
      continue;
    RootGroup& rg = layout->groups[set];
    rg.root_cbv.assign(gl->dynamic_entries.size(), -1);
    base::Vector<D3D12_DESCRIPTOR_RANGE> views, samplers;
    for (const GroupEntry& e : gl->entries) {
      if ((e.type == rhi::BindingType::kStorageBuffer ||
           e.type == rhi::BindingType::kStorageBufferDynamic) &&
          e.read_only)
        layout->read_only_storage.emplace_back(set, e.binding);
      if (e.type == rhi::BindingType::kUniformBufferDynamic && root_ubos) {
        D3D12_ROOT_PARAMETER p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor = {e.binding, set};
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        rg.root_cbv[e.dynamic_index] = static_cast<i32>(params.size());
        params.push_back(p);
        continue;
      }
      views.push_back({RangeType(e), 1, e.binding, set, e.slot});
      if (e.sampler_slot != ~0u)
        samplers.push_back({D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, e.binding,
                            set, e.sampler_slot});
    }
    for (auto* list : {&views, &samplers}) {
      if (list->empty())
        continue;
      ranges.push_back(base::move(*list));
      D3D12_ROOT_PARAMETER p{};
      p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      p.DescriptorTable = {static_cast<UINT>(ranges.back().size()),
                           ranges.back().data()};
      p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      (list == &views ? rg.table : rg.sampler_table) =
          static_cast<i32>(params.size());
      params.push_back(p);
    }
  }

  D3D12_ROOT_SIGNATURE_DESC rd{};
  rd.NumParameters = static_cast<UINT>(params.size());
  rd.pParameters = params.data();
  rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
  layout->root = SerializeRoot(rd);
  if (!layout->root)
    return nullptr;
  return gpu::rhi::Release(layout, this);
}

bool D3D12Device::CompileStage(const rhi::ShaderCode& code,
                               const LowerOptions& options,
                               base::Vector<u8>* dxil,
                               LoweredShader* info) {
  u64 key = HashWords(code.words, code.count, code.count);
  const u32 opts[] = {
      u32(options.stage),         options.shader_model,
      u32(options.flip_y),        options.uint_inputs,
      options.sint_inputs,        options.swap_rb_inputs,
      u32(options.dispatch_base), u32(options.emulate_barycentrics)};
  key = HashWords(opts, sizeof(opts) / 4, key);
  for (const auto& [set, binding] : options.read_only_storage) {
    const u32 sb[] = {set, binding};
    key = HashWords(sb, 2, key);
  }
  for (char c : options.producer_outputs) {
    const u32 w = static_cast<u8>(c);
    key = HashWords(&w, 1, key);
  }
  {
    base::LockGuard<base::Mutex> lock(shader_mutex_);
    auto it = shaders_.find(key);
    if (it != shaders_.end()) {
      *dxil = it->second.dxil;
      *info = it->second.info;
      return it->second.ok;
    }
  }
  ShaderEntry entry;
  entry.ok = LowerToHlsl(code.words, code.count, options, &entry.info);
  base::String error = entry.info.error;
  if (entry.ok)
    entry.ok =
        Dxc::Compile(entry.info.hlsl, entry.info.profile, &entry.dxil, &error);
  if (!entry.ok)
    BASE_LOGI("gpud3d12", "shader ({:#x}) failed: {}", key, error.c_str());
  entry.info.hlsl.clear();
  *dxil = entry.dxil;
  *info = entry.info;
  base::LockGuard<base::Mutex> lock(shader_mutex_);
  shaders_[key] = base::move(entry);
  return shaders_[key].ok;
}

bool D3D12Device::CompileHlsl(const base::String& hlsl,
                              const char* profile,
                              base::Vector<u8>* dxil) {
  const base::String keyed = hlsl + profile;
  const u64 key = base::HashBytes(keyed.data(), keyed.size());
  {
    base::LockGuard<base::Mutex> lock(shader_mutex_);
    auto it = shaders_.find(key);
    if (it != shaders_.end()) {
      *dxil = it->second.dxil;
      return it->second.ok;
    }
  }
  ShaderEntry entry;
  base::String error;
  entry.ok = Dxc::Compile(hlsl, profile, &entry.dxil, &error);
  if (!entry.ok)
    BASE_LOGI("gpud3d12", "internal shader failed: {}", error.c_str());
  *dxil = entry.dxil;
  base::LockGuard<base::Mutex> lock(shader_mutex_);
  shaders_[key] = base::move(entry);
  return shaders_[key].ok;
}

rhi::Pipeline* D3D12Device::CreateGraphicsPipeline(
    const rhi::GraphicsPipelineDesc& desc) {
  const ui::ShaderCompilation compilation;
  auto* layout = static_cast<D3D12PipelineLayout*>(desc.layout);
  if (desc.mesh) {
    BASE_LOGI("gpud3d12", "mesh pipelines are not supported");
    return nullptr;
  }
  auto pipeline = base::MakeUnique<D3D12Pipeline>();
  pipeline->layout = layout;

  LowerOptions vo;
  vo.stage = rhi::kStageVertex;
  vo.shader_model = shader_model_;
  vo.flip_y = desc.geometry.empty();
  vo.read_only_storage = layout->read_only_storage;
  base::Vector<D3D12_INPUT_ELEMENT_DESC> elements;
  for (const rhi::VertexAttribute& a : desc.vertex_attributes) {
    const DxgiInfo& fi = Dxgi(a.format);
    if (a.location < 32) {
      if (fi.scaled == 1)
        vo.uint_inputs |= 1u << a.location;
      if (fi.scaled == 2)
        vo.sint_inputs |= 1u << a.location;
      if (fi.swap_rb)
        vo.swap_rb_inputs |= 1u << a.location;
    }
    const bool instance = a.buffer < desc.vertex_buffers.size() &&
                          desc.vertex_buffers[a.buffer].per_instance;
    elements.push_back({"TEXCOORD", a.location, VertexFormat(a.format),
                        a.buffer, a.offset,
                        instance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                 : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                        instance ? 1u : 0u});
  }
  base::Vector<u8> vs, gs, ps;
  LoweredShader vinfo, ginfo, pinfo;
  if (!CompileStage(desc.vertex, vo, &vs, &vinfo))
    return nullptr;
  pipeline->uses_draw_params = vinfo.uses_draw_params;
  pipeline->uses_raster = vinfo.uses_raster;
  base::String outputs = vinfo.outputs;
  if (!desc.geometry.empty()) {
    LowerOptions go = vo;
    go.stage = rhi::kStageGeometry;
    go.flip_y = true;
    go.uint_inputs = go.sint_inputs = go.swap_rb_inputs = 0;
    go.producer_outputs = outputs;
    if (!CompileStage(desc.geometry, go, &gs, &ginfo))
      return nullptr;
    pipeline->uses_raster = ginfo.uses_raster;
    outputs = ginfo.outputs;
  }
  // Without SM 6.1, a fragment shader's barycentrics come from a generated
  // geometry shader (triangles only).
  constexpr u32 kBaryCoord = 5286, kBaryCoordNoPersp = 5287;
  const bool barycentrics =
      !caps_.fragment_barycentric && desc.geometry.empty() &&
      !desc.fragment.empty() &&
      TopologyType(desc.topology) == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE &&
      desc.topology != rhi::Topology::kTriangleListAdjacency &&
      (DeclaresBuiltIn(desc.fragment.words, desc.fragment.count, kBaryCoord) ||
       DeclaresBuiltIn(desc.fragment.words, desc.fragment.count,
                       kBaryCoordNoPersp));
  if (barycentrics &&
      !CompileHlsl(BarycentricGeometryShader(vinfo.outputs, &outputs), "gs_6_0",
                   &gs))
    return nullptr;
  if (!desc.fragment.empty()) {
    LowerOptions fo;
    fo.stage = rhi::kStageFragment;
    fo.shader_model = shader_model_;
    fo.read_only_storage = layout->read_only_storage;
    fo.producer_outputs = outputs;
    fo.emulate_barycentrics = barycentrics;
    if (!CompileStage(desc.fragment, fo, &ps, &pinfo))
      return nullptr;
  }

  D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = layout->root;
  pd.VS = {vs.data(), vs.size()};
  pd.GS = {gs.data(), gs.size()};
  pd.PS = {ps.data(), ps.size()};
  pd.BlendState.IndependentBlendEnable = TRUE;
  // Culling both faces draws nothing; D3D12 cannot cull both, so the
  // pipeline keeps its triangles and writes nothing instead.
  const bool cull_all = desc.cull == rhi::CullMode::kFrontAndBack;
  for (u32 i = 0; i < desc.color_count && i < 8; i++) {
    const rhi::BlendAttachment& b = desc.blend[i];
    D3D12_RENDER_TARGET_BLEND_DESC& t = pd.BlendState.RenderTarget[i];
    t.BlendEnable = b.enable;
    t.SrcBlend = ToBlend(b.src_color, false);
    t.DestBlend = ToBlend(b.dst_color, false);
    t.BlendOp = ToBlendOp(b.color_op);
    t.SrcBlendAlpha = ToBlend(b.src_alpha, true);
    t.DestBlendAlpha = ToBlend(b.dst_alpha, true);
    t.BlendOpAlpha = ToBlendOp(b.alpha_op);
    t.LogicOp = D3D12_LOGIC_OP_NOOP;
    t.RenderTargetWriteMask = cull_all ? 0 : b.write_mask & 0xF;
    pd.RTVFormats[i] = Dxgi(desc.color_formats[i]).format;
  }
  pd.SampleMask = ~0u;
  D3D12_RASTERIZER_DESC& rs = pd.RasterizerState;
  rs.FillMode = D3D12_FILL_MODE_SOLID;
  rs.CullMode = desc.cull == rhi::CullMode::kFront  ? D3D12_CULL_MODE_FRONT
                : desc.cull == rhi::CullMode::kBack ? D3D12_CULL_MODE_BACK
                                                    : D3D12_CULL_MODE_NONE;
  rs.FrontCounterClockwise = desc.front_ccw;
  rs.DepthClipEnable = !desc.depth_clamp;
  D3D12_DEPTH_STENCIL_DESC& ds = pd.DepthStencilState;
  ds.DepthEnable = desc.depth_test && !cull_all;
  ds.DepthWriteMask = desc.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL
                                       : D3D12_DEPTH_WRITE_MASK_ZERO;
  ds.DepthFunc = ToCompare(desc.depth_compare);
  ds.StencilEnable = desc.stencil_test && !cull_all;
  ds.StencilReadMask = desc.stencil_front.compare_mask;
  ds.StencilWriteMask = desc.stencil_front.write_mask;
  ds.FrontFace = ToStencil(desc.stencil_front);
  ds.BackFace = ToStencil(desc.stencil_back);
  pd.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};
  pd.IBStripCutValue = desc.primitive_restart
                           ? D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
                           : D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
  pd.PrimitiveTopologyType = TopologyType(desc.topology);
  pd.NumRenderTargets = desc.color_count;
  const rhi::Format ds_format = desc.depth_format != rhi::Format::kUndefined
                                    ? desc.depth_format
                                    : desc.stencil_format;
  pd.DSVFormat = ds_format != rhi::Format::kUndefined ? Dxgi(ds_format).format
                                                      : DXGI_FORMAT_UNKNOWN;
  pd.SampleDesc.Count = 1;
  if (FAILED(device->CreateGraphicsPipelineState(
          &pd, IID_ID3D12PipelineState,
          reinterpret_cast<void**>(&pipeline->pso)))) {
    BASE_LOGI("gpud3d12", "graphics pipeline {} failed",
              desc.name ? desc.name : "");
    return nullptr;
  }
  if (desc.primitive_restart) {
    pd.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
    device->CreateGraphicsPipelineState(
        &pd, IID_ID3D12PipelineState,
        reinterpret_cast<void**>(&pipeline->pso_restart32));
  }
  pipeline->topology = Topology(desc.topology);
  pipeline->fan = desc.topology == rhi::Topology::kTriangleFan;
  for (size_t i = 0; i < desc.vertex_buffers.size() && i < 16; i++)
    pipeline->strides[i] = desc.vertex_buffers[i].stride;
  pipeline->stencil_ref = desc.stencil_front.reference;
  if (desc.name)
    SetName(&*pipeline, desc.name);
  return gpu::rhi::Release(pipeline, this);
}

rhi::Pipeline* D3D12Device::CreateComputePipeline(
    const rhi::ComputePipelineDesc& desc) {
  const ui::ShaderCompilation compilation;
  auto* layout = static_cast<D3D12PipelineLayout*>(desc.layout);
  LowerOptions co;
  co.stage = rhi::kStageCompute;
  co.shader_model = shader_model_;
  co.read_only_storage = layout->read_only_storage;
  co.dispatch_base = desc.dispatch_base;
  base::Vector<u8> cs;
  LoweredShader info;
  if (!CompileStage(desc.code, co, &cs, &info))
    return nullptr;
  auto pipeline = base::MakeUnique<D3D12Pipeline>();
  pipeline->layout = layout;
  pipeline->compute = true;
  pipeline->uses_workgroup_count = info.uses_workgroup_count;
  pipeline->dispatch_base = desc.dispatch_base;
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = layout->root;
  pd.CS = {cs.data(), cs.size()};
  if (FAILED(device->CreateComputePipelineState(
          &pd, IID_ID3D12PipelineState,
          reinterpret_cast<void**>(&pipeline->pso)))) {
    BASE_LOGI("gpud3d12", "compute pipeline {} failed",
              desc.name ? desc.name : "");
    return nullptr;
  }
  if (desc.name)
    SetName(&*pipeline, desc.name);
  return gpu::rhi::Release(pipeline, this);
}

bool D3D12Device::InitBlit() {
  base::String error;
  auto compile = [&](const char* entry, const char* profile,
                     base::Vector<u8>* out) {
    // Dxc::Compile always takes "main": rename the entry point in the text.
    const base::String source = kBlitHlsl;
    const base::String decl = base::String(" ") + entry + "(";
    const mem_size at = source.find(decl);
    const base::String text =
        source.substr(0, at) + " main(" + source.substr(at + decl.size());
    return Dxc::Compile(text, profile, out, &error);
  };
  if (!compile("vs", "vs_6_0", &blit_vs_) ||
      !compile("ps", "ps_6_0", &blit_ps_)) {
    BASE_LOGI("gpud3d12", "blit shaders: {}", error.c_str());
    return false;
  }
  D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
  D3D12_ROOT_PARAMETER params[2]{};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  params[0].Constants = {0, 0, 5};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[1].DescriptorTable = {1, &range};
  D3D12_STATIC_SAMPLER_DESC samplers[2]{};
  for (u32 i = 0; i < 2; i++) {
    samplers[i].Filter =
        i ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplers[i].AddressU = samplers[i].AddressV = samplers[i].AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[i].MaxLOD = 1000.0f;
    samplers[i].ShaderRegister = i;
    samplers[i].ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
  }
  D3D12_ROOT_SIGNATURE_DESC rd{};
  rd.NumParameters = 2;
  rd.pParameters = params;
  rd.NumStaticSamplers = 2;
  rd.pStaticSamplers = samplers;
  blit_root_ = SerializeRoot(rd);
  return blit_root_ != nullptr;
}

ID3D12PipelineState* D3D12Device::BlitPipeline(DXGI_FORMAT format) {
  base::LockGuard<base::Mutex> lock(blit_mutex_);
  auto it = blit_psos_.find(format);
  if (it != blit_psos_.end())
    return it->second;
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
  pd.pRootSignature = blit_root_;
  pd.VS = {blit_vs_.data(), blit_vs_.size()};
  pd.PS = {blit_ps_.data(), blit_ps_.size()};
  pd.BlendState.RenderTarget[0].RenderTargetWriteMask =
      D3D12_COLOR_WRITE_ENABLE_ALL;
  pd.SampleMask = ~0u;
  pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
  pd.RasterizerState.DepthClipEnable = TRUE;
  pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
  pd.NumRenderTargets = 1;
  pd.RTVFormats[0] = format;
  pd.SampleDesc.Count = 1;
  ID3D12PipelineState* pso = nullptr;
  if (FAILED(device->CreateGraphicsPipelineState(
          &pd, IID_ID3D12PipelineState, reinterpret_cast<void**>(&pso))))
    pso = nullptr;
  blit_psos_[format] = pso;
  return pso;
}

}  // namespace gpu::d3d12::impl
