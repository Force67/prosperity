// Developer tool, not a test: runs real recompiler SPIR-V through the D3D12
// backend's shader path. For each module it derives a binding layout from the
// decorations, lowers to HLSL (SPIRV-Cross), compiles to DXIL (DXC) and, with
// --pso, builds a pipeline on the D3D12 device (vkd3d on Linux, which also
// translates the DXIL). Prints per-stage counts and the top failure reasons.
//
//   rhi_hlsl_corpus [--pso] [--dump | --fails] [--limit N] [--sm 61] [--jobs 8]
//                   [dir|file.spv]...
//
// The default input is ~/.cache/ps4delta/spirv.

#include <spirv_cross.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "base/arch.h"
#include "gpu/d3d12/d3d12_internal.h"
#include "gpu/d3d12/d3d12_shader.h"
#include "gpu/render/backend.h"

namespace {

namespace fs = std::filesystem;
namespace sc = spirv_cross;
using namespace gpu;
using namespace gpu::d3d12;

struct Stats {
  int total = 0, lowered = 0, compiled = 0, pso = 0;
};

struct Module {
  std::vector<u32> words;
  rhi::ShaderStage stage = rhi::kStageVertex;
  bool barycentric = false;
  bool buffer_address = false;
  std::map<u32, std::vector<rhi::BindingLayout>> sets;
  u32 push_bytes = 0;
  std::vector<std::pair<u32, rhi::Format>> inputs;  // vertex: location, format
  std::vector<std::pair<u32, DXGI_FORMAT>> outputs;  // fragment targets
  bool writes_depth = false;
  u32 separate = 0;  // separate images/samplers the rhi cannot bind
};

std::vector<u32> ReadWords(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  std::vector<u32> words(bytes.size() / 4);
  std::memcpy(words.data(), bytes.data(), words.size() * 4);
  return words;
}

rhi::Format AttributeFormat(const sc::SPIRType& t) {
  static const rhi::Format kFloat[] = {
      rhi::Format::kR32Float, rhi::Format::kRG32Float,
      rhi::Format::kRGB32Float, rhi::Format::kRGBA32Float};
  static const rhi::Format kUint[] = {
      rhi::Format::kR32Uint, rhi::Format::kRG32Uint, rhi::Format::kRGBA32Uint,
      rhi::Format::kRGBA32Uint};
  static const rhi::Format kSint[] = {
      rhi::Format::kR32Sint, rhi::Format::kRG32Sint, rhi::Format::kRGBA32Sint,
      rhi::Format::kRGBA32Sint};
  const u32 n = std::clamp(t.vecsize, 1u, 4u) - 1;
  if (t.basetype == sc::SPIRType::UInt)
    return kUint[n];
  if (t.basetype == sc::SPIRType::Int)
    return kSint[n];
  return kFloat[n];
}

bool Reflect(const std::vector<u32>& words, Module* m, std::string* error) {
  try {
    sc::Compiler c(words);
    switch (c.get_execution_model()) {
      case spv::ExecutionModelVertex:
        m->stage = rhi::kStageVertex;
        break;
      case spv::ExecutionModelGeometry:
        m->stage = rhi::kStageGeometry;
        break;
      case spv::ExecutionModelFragment:
        m->stage = rhi::kStageFragment;
        break;
      case spv::ExecutionModelGLCompute:
        m->stage = rhi::kStageCompute;
        break;
      case spv::ExecutionModelMeshEXT:
        m->stage = rhi::kStageMesh;
        break;
      default:
        *error = "unsupported execution model";
        return false;
    }
    for (spv::Capability cap : c.get_declared_capabilities()) {
      if (cap == spv::CapabilityFragmentBarycentricKHR)
        m->barycentric = true;
      if (cap == spv::CapabilityPhysicalStorageBufferAddresses)
        m->buffer_address = true;
    }
    const sc::ShaderResources r = c.get_shader_resources();
    auto add = [&](const sc::Resource& res, rhi::BindingType type,
                   bool read_only) {
      const u32 set = c.get_decoration(res.id, spv::DecorationDescriptorSet);
      const u32 binding = c.get_decoration(res.id, spv::DecorationBinding);
      rhi::BindingLayout b;
      b.binding = binding;
      b.type = type;
      b.stages = m->stage;
      b.read_only = read_only;
      m->sets[set].push_back(b);
    };
    for (const sc::Resource& res : r.uniform_buffers)
      add(res, rhi::BindingType::kUniformBuffer, false);
    for (const sc::Resource& res : r.storage_buffers)
      add(res, rhi::BindingType::kStorageBuffer,
          c.get_buffer_block_flags(res.id).get(spv::DecorationNonWritable));
    for (const sc::Resource& res : r.sampled_images)
      add(res, rhi::BindingType::kSampledTexture, false);
    for (const sc::Resource& res : r.storage_images)
      add(res, rhi::BindingType::kStorageTexture, false);
    // Separate images and samplers: bound by position only, as SRVs and
    // samplers of the same registers. The rhi has no such binding type.
    for (const sc::Resource& res : r.separate_images) {
      add(res, rhi::BindingType::kSampledTexture, false);
      m->separate++;
    }
    m->separate += static_cast<u32>(r.separate_samplers.size());
    for (const sc::Resource& res : r.push_constant_buffers)
      m->push_bytes = static_cast<u32>(
          c.get_declared_struct_size(c.get_type(res.base_type_id)));
    for (const sc::Resource& res : r.stage_inputs)
      if (m->stage == rhi::kStageVertex)
        m->inputs.emplace_back(c.get_decoration(res.id, spv::DecorationLocation),
                               AttributeFormat(c.get_type(res.type_id)));
    for (const sc::Resource& res : r.stage_outputs) {
      if (m->stage != rhi::kStageFragment)
        continue;
      const sc::SPIRType& t = c.get_type(res.type_id);
      const DXGI_FORMAT f = t.basetype == sc::SPIRType::UInt
                                ? DXGI_FORMAT_R32G32B32A32_UINT
                            : t.basetype == sc::SPIRType::Int
                                ? DXGI_FORMAT_R32G32B32A32_SINT
                                : DXGI_FORMAT_R32G32B32A32_FLOAT;
      m->outputs.emplace_back(
          c.get_decoration(res.id, spv::DecorationLocation), f);
    }
    for (const sc::BuiltInResource& b : r.builtin_outputs)
      if (b.builtin == spv::BuiltInFragDepth)
        m->writes_depth = true;
    return true;
  } catch (const std::exception& e) {
    *error = e.what();
    return false;
  }
}

// The first line of a diagnostic, without the file:line: prefix or numbers
// that would split one cause into many.
std::string Reason(const std::string& text) {
  size_t start = text.find("error: ");
  start = start == std::string::npos ? 0 : text.rfind('\n', start) + 1;
  std::string line = text.substr(start, text.find('\n', start) - start);
  const size_t err = line.find("error: ");
  if (err != std::string::npos)
    line = line.substr(err + 7);
  std::string out;
  for (size_t i = 0; i < line.size(); i++) {
    if (std::isdigit(static_cast<unsigned char>(line[i]))) {
      if (out.empty() || out.back() != '#')
        out += '#';
      continue;
    }
    out += line[i];
  }
  if (out.size() > 120)
    out.resize(120);
  return out;
}

class PsoChecker {
 public:
  explicit PsoChecker(rhi::Device* device) : device_(device) {
    std::string error;
    Dxc::Compile(
        "float4 main(uint id : SV_VertexID) : SV_Position { return 0; }",
        "vs_6_0", &passthrough_vs_, &error);
  }

  // Empty on success.
  std::string Check(const Module& m, const std::vector<u8>& dxil) {
    std::vector<rhi::Object*> objects;
    std::string result = Build(m, dxil, objects);
    for (auto it = objects.rbegin(); it != objects.rend(); ++it)
      device_->Destroy(*it);
    return result;
  }

 private:
  std::string Build(const Module& m,
                    const std::vector<u8>& dxil,
                    std::vector<rhi::Object*>& objects) {
    rhi::PipelineLayoutDesc pl;
    const u32 sets = m.sets.empty() ? 0 : m.sets.rbegin()->first + 1;
    for (u32 s = 0; s < sets; s++) {
      rhi::BindGroupLayoutDesc gd;
      auto it = m.sets.find(s);
      if (it != m.sets.end())
        gd.bindings = it->second;
      rhi::BindGroupLayout* g = device_->CreateBindGroupLayout(gd);
      objects.push_back(g);
      pl.groups.push_back(g);
    }
    pl.push_constant_bytes = m.push_bytes;
    pl.push_constant_stages = m.stage;
    rhi::PipelineLayout* layout = device_->CreatePipelineLayout(pl);
    if (!layout)
      return "root signature";
    objects.push_back(layout);
    ShaderRoot root = Root(layout);

    if (m.stage == rhi::kStageCompute) {
      rhi::ComputePipelineDesc cp;
      cp.layout = layout;
      cp.code = {m.words.data(), m.words.size()};
      rhi::Pipeline* p = device_->CreateComputePipeline(cp);
      if (!p)
        return "compute pipeline";
      objects.push_back(p);
      return "";
    }
    if (m.stage == rhi::kStageVertex) {
      rhi::GraphicsPipelineDesc gp;
      gp.layout = layout;
      gp.vertex = {m.words.data(), m.words.size()};
      for (size_t i = 0; i < m.inputs.size(); i++) {
        gp.vertex_buffers.push_back({16, false});
        gp.vertex_attributes.push_back(
            {m.inputs[i].first, static_cast<u32>(i), m.inputs[i].second, 0});
      }
      rhi::Pipeline* p = device_->CreateGraphicsPipeline(gp);
      if (!p)
        return "vertex pipeline";
      objects.push_back(p);
      return "";
    }
    // Fragment and geometry stages go straight to D3D12 with a stand-in
    // vertex shader: only the stage under test matters here.
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.root;
    pd.VS = {passthrough_vs_.data(), passthrough_vs_.size()};
    if (m.stage == rhi::kStageGeometry)
      pd.GS = {dxil.data(), dxil.size()};
    else
      pd.PS = {dxil.data(), dxil.size()};
    pd.SampleMask = ~0u;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    for (const auto& [location, format] : m.outputs)
      if (location < 8) {
        pd.RTVFormats[location] = format;
        pd.NumRenderTargets = std::max<UINT>(pd.NumRenderTargets, location + 1);
        pd.BlendState.RenderTarget[location].RenderTargetWriteMask = 0xF;
      }
    for (UINT i = 0; i < pd.NumRenderTargets; i++)
      if (pd.RTVFormats[i] == DXGI_FORMAT_UNKNOWN)
        pd.RTVFormats[i] = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (m.writes_depth) {
      pd.DSVFormat = DXGI_FORMAT_D32_FLOAT;
      pd.DepthStencilState.DepthEnable = TRUE;
      pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
      pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    }
    pd.SampleDesc.Count = 1;
    ID3D12PipelineState* pso = nullptr;
    if (FAILED(root.device->CreateGraphicsPipelineState(
            &pd, IID_ID3D12PipelineState, reinterpret_cast<void**>(&pso))))
      return m.stage == rhi::kStageGeometry ? "geometry pipeline"
                                            : "fragment pipeline";
    pso->Release();
    return "";
  }

  struct ShaderRoot {
    ID3D12Device* device = nullptr;
    ID3D12RootSignature* root = nullptr;
  };
  ShaderRoot Root(rhi::PipelineLayout* layout) {
    return {static_cast<impl::D3D12Device*>(device_)->device,
            static_cast<impl::D3D12PipelineLayout*>(layout)->root};
  }

  rhi::Device* device_;
  std::vector<u8> passthrough_vs_;
};

const char* StageName(rhi::ShaderStage s) {
  switch (s) {
    case rhi::kStageVertex:
      return "vs";
    case rhi::kStageGeometry:
      return "gs";
    case rhi::kStageFragment:
      return "fs";
    case rhi::kStageCompute:
      return "cs";
    case rhi::kStageMesh:
      return "ms";
    default:
      return "?";
  }
}

void Top(const char* title, const std::map<std::string, int>& reasons) {
  std::vector<std::pair<int, std::string>> sorted;
  for (const auto& [r, n] : reasons)
    sorted.emplace_back(n, r);
  std::sort(sorted.rbegin(), sorted.rend());
  std::printf("\n%s:\n", title);
  for (size_t i = 0; i < sorted.size() && i < 25; i++)
    std::printf("  %6d  %s\n", sorted[i].first, sorted[i].second.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  bool pso = false, dump = false, fails = false;
  size_t limit = ~size_t(0);
  u32 shader_model = 60;
  u32 jobs = 8;
  std::vector<fs::path> inputs;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--pso"))
      pso = true;
    else if (!std::strcmp(argv[i], "--dump"))
      dump = true;
    else if (!std::strcmp(argv[i], "--fails"))
      fails = true;
    else if (!std::strcmp(argv[i], "--limit") && i + 1 < argc)
      limit = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--sm") && i + 1 < argc)
      shader_model = static_cast<u32>(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--jobs") && i + 1 < argc)
      jobs = std::max(1, std::atoi(argv[++i]));
    else
      inputs.emplace_back(argv[i]);
  }
  if (inputs.empty()) {
    const char* home = std::getenv("HOME");
    inputs.emplace_back(fs::path(home ? home : ".") / ".cache/ps4delta/spirv");
  }
  std::vector<fs::path> files;
  for (const fs::path& p : inputs) {
    if (fs::is_directory(p)) {
      for (const auto& e : fs::directory_iterator(p))
        if (e.path().extension() == ".spv")
          files.push_back(e.path());
    } else {
      files.push_back(p);
    }
  }
  std::sort(files.begin(), files.end());
  if (files.size() > limit)
    files.resize(limit);
  if (!Dxc::Load()) {
    std::fprintf(stderr, "cannot load DXC\n");
    return 1;
  }
  std::unique_ptr<rhi::Device> device;
  std::unique_ptr<PsoChecker> checker;
  if (pso) {
    device = render::CreateBackendDevice(rhi::Backend::kD3D12);
    if (!device) {
      std::fprintf(stderr, "no D3D12 device\n");
      return 1;
    }
    checker = std::make_unique<PsoChecker>(device.get());
  }

  std::map<std::string, Stats> stats;
  std::map<std::string, int> lower_fail, dxc_fail, pso_fail;
  int barycentric = 0, buffer_address = 0, separate = 0;
  std::mutex mutex;
  std::atomic<size_t> next{0}, done{0};
  auto process = [&](size_t n) {
    Module m;
    m.words = ReadWords(files[n]);
    std::string error;
    if (!Reflect(m.words, &m, &error)) {
      std::lock_guard<std::mutex> lock(mutex);
      stats["??"].total++;
      lower_fail["reflect: " + Reason(error)]++;
      return;
    }
    const std::string stage = StageName(m.stage);
    {
      std::lock_guard<std::mutex> lock(mutex);
      stats[stage].total++;
      barycentric += m.barycentric;
      buffer_address += m.buffer_address;
      separate += m.separate != 0;
    }
    LowerOptions o;
    o.stage = m.stage;
    o.shader_model = std::max(shader_model, m.barycentric ? 61u : 60u);
    if (m.stage == rhi::kStageMesh)
      o.shader_model = std::max(o.shader_model, 65u);
    o.flip_y = m.stage == rhi::kStageVertex || m.stage == rhi::kStageGeometry;
    for (const auto& [set, bindings] : m.sets)
      for (const rhi::BindingLayout& b : bindings)
        if (b.type == rhi::BindingType::kStorageBuffer && b.read_only)
          o.read_only_storage.emplace_back(set, b.binding);
    LoweredShader lowered;
    const bool ok = LowerToHlsl(m.words.data(), m.words.size(), o, &lowered);
    if (dump || (fails && !ok)) {
      std::lock_guard<std::mutex> lock(mutex);
      if (ok)
        std::printf("// %s\n%s\n", files[n].c_str(), lowered.hlsl.c_str());
      else
        std::printf("%s: lower: %s\n", files[n].c_str(),
                    lowered.error.c_str());
    }
    if (!ok) {
      std::lock_guard<std::mutex> lock(mutex);
      lower_fail[stage + ": " + Reason(lowered.error)]++;
      return;
    }
    std::vector<u8> dxil;
    const bool compiled =
        Dxc::Compile(lowered.hlsl, lowered.profile, &dxil, &error);
    {
      std::lock_guard<std::mutex> lock(mutex);
      stats[stage].lowered++;
      if (!compiled) {
        dxc_fail[stage + ": " + Reason(error)]++;
        if (dump || fails)
          std::printf("%s: dxc: %s\n", files[n].c_str(), error.c_str());
        return;
      }
      stats[stage].compiled++;
    }
    if (!checker)
      return;
    std::string failure;
    if (m.barycentric)
      failure = "barycentrics (device lacks SM 6.1)";
    else if (m.buffer_address)
      failure = "buffer device address";
    else
      failure = checker->Check(m, dxil);
    std::lock_guard<std::mutex> lock(mutex);
    if (failure.empty())
      stats[stage].pso++;
    else
      pso_fail[stage + ": " + failure]++;
    if (fails && !failure.empty())
      std::printf("%s: pipeline: %s\n", files[n].c_str(), failure.c_str());
  };
  std::vector<std::thread> threads;
  for (u32 t = 0; t < jobs; t++)
    threads.emplace_back([&] {
      for (size_t n; (n = next++) < files.size();) {
        process(n);
        if (++done % 1000 == 0)
          std::fprintf(stderr, "%zu / %zu\n", done.load(), files.size());
      }
    });
  for (std::thread& t : threads)
    t.join();

  std::printf("%zu modules; %d use barycentrics, %d buffer addresses, %d "
              "separate images/samplers\n",
              files.size(), barycentric, buffer_address, separate);
  std::printf("stage   total  lowered  compiled%s\n", pso ? "  pipeline" : "");
  for (const auto& [name, s] : stats)
    if (pso)
      std::printf("%-5s %7d %8d %9d %9d\n", name.c_str(), s.total, s.lowered,
                  s.compiled, s.pso);
    else
      std::printf("%-5s %7d %8d %9d\n", name.c_str(), s.total, s.lowered,
                  s.compiled);
  Top("lowering failures", lower_fail);
  Top("DXC failures", dxc_fail);
  if (pso)
    Top("pipeline failures", pso_fail);
  return 0;
}
