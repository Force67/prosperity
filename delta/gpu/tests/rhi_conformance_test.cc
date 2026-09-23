/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The same work, run through every backend this build has, compared pixel for
 * pixel. Each scenario states what the image should be, so a single backend
 * failing is caught on its own; the backends are then compared against each
 * other, which is what catches the differences neither one thinks is an error
 * (a flipped viewport, a swapped blend factor, a descriptor bound one slot
 * out).
 *
 * The shaders come from one HLSL file (rhi_conformance.hlsl): SPIR-V compiled
 * at build time by glslc, DXBC compiled at run time by D3DCompile. A pixel
 * difference is therefore a backend difference, not a shader difference.
 *
 * Needs a working device, so this is a developer test rather than a CI one: it
 * reports success with no assertions when no backend can be created.
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "gpu/d3d12/d3d12_rhi.h"
#include "gpu/rhi/device.h"
#include "gpu/vulkan/vk_rhi.h"

// Generated from rhi_conformance.hlsl by the build (see tests/CMakeLists.txt).
#include "rhi_conformance_hlsl.h"

namespace gpu::rhi {
namespace {

const u32 kVsSolidSpv[] =
#include "vs_solid.spv.inc"
    ;
const u32 kPsSolidSpv[] =
#include "ps_solid.spv.inc"
    ;
const u32 kPsMrtSpv[] =
#include "ps_mrt.spv.inc"
    ;
const u32 kVsTexSpv[] =
#include "vs_tex.spv.inc"
    ;
const u32 kPsTexSpv[] =
#include "ps_tex.spv.inc"
    ;
const u32 kCsFillSpv[] =
#include "cs_fill.spv.inc"
    ;

constexpr u32 kSize = 64;  // targets are kSize x kSize; a row is 256 bytes,
                           // which is the pitch D3D12 copies demand
constexpr u32 kPixels = kSize * kSize;
constexpr u32 kImageBytes = kPixels * 4;

struct Rgba {
  u8 r = 0, g = 0, b = 0, a = 0;

  bool operator==(const Rgba& o) const {
    return r == o.r && g == o.g && b == o.b && a == o.a;
  }
};

Rgba At(const std::vector<u8>& image, u32 x, u32 y) {
  const u8* p = image.data() + (y * kSize + x) * 4;
  return {p[0], p[1], p[2], p[3]};
}

// A shader in both the forms the backends consume. Each takes the one it can
// use and ignores the other.
ShaderDesc Shader(u32 stage,
                  const u32* spirv,
                  u64 spirv_bytes,
                  const char* entry,
                  const char* target) {
  ShaderDesc desc;
  desc.stage = stage;
  desc.spirv = spirv;
  desc.spirv_bytes = spirv_bytes;
  desc.hlsl = kConformanceHlsl;
  desc.entry = entry;
  desc.hlsl_target = target;
  desc.name = entry;
  return desc;
}

std::unique_ptr<Texture> MakeTarget(Device& device, const char* name) {
  TextureDesc desc;
  desc.width = desc.height = kSize;
  desc.format = Format::RGBA8Unorm;
  desc.usage = kTextureRenderTarget | kTextureCopySrc;
  desc.name = name;
  return device.CreateTexture(desc);
}

std::unique_ptr<Buffer> MakeReadback(Device& device, u64 bytes) {
  BufferDesc desc;
  desc.size = bytes;
  desc.usage = kBufferCopyDst;
  desc.memory = MemoryKind::Readback;
  desc.name = "readback";
  return device.CreateBuffer(desc);
}

// Upload buffer holding `bytes` of `data`, ready to be bound or copied from.
std::unique_ptr<Buffer> MakeUpload(Device& device,
                                   const void* data,
                                   u64 bytes,
                                   u32 usage,
                                   const char* name) {
  BufferDesc desc;
  desc.size = bytes;
  desc.usage = usage | kBufferCopySrc;
  desc.memory = MemoryKind::Upload;
  desc.name = name;
  auto buffer = device.CreateBuffer(desc);
  if (buffer && data)
    memcpy(buffer->Map(), data, bytes);
  return buffer;
}

void CopyTargetToReadback(CommandList& list, Texture& target, Buffer& into) {
  list.Transition(&target, ResourceState::CopySrc);
  list.Transition(&into, ResourceState::CopyDst);
  TextureRegion region;
  region.width = kSize;
  region.height = kSize;
  list.CopyTextureToBuffer(&into, 0, kSize * 4, &target, region);
}

std::vector<u8> Drain(Buffer& readback, u64 bytes) {
  std::vector<u8> out(bytes);
  memcpy(out.data(), readback.Map(), bytes);
  return out;
}

// Every backend this build can bring up. Devices are expensive and are shared
// by all scenarios; a backend that cannot be created is simply absent.
struct Backend {
  const char* name;
  std::unique_ptr<Device> device;
};

std::vector<Backend>& Backends() {
  static std::vector<Backend> backends = [] {
    std::vector<Backend> found;
    if (auto vulkan = vk::CreateVulkanDevice())
      found.push_back({"vulkan", std::move(vulkan)});
    if (auto d3d12 = d3d12::CreateD3D12Device())
      found.push_back({"d3d12", std::move(d3d12)});
    return found;
  }();
  return backends;
}

// Runs `scenario` on every backend, hands each result to `check`, and requires
// every backend to have produced the same image.
template <typename Scenario, typename Check>
void ForEachBackend(Scenario scenario, Check check, u32 tolerance = 0) {
  std::vector<Backend>& backends = Backends();
  if (backends.empty())
    GTEST_SKIP() << "no RHI backend could be created";

  std::vector<u8> reference;
  const char* reference_name = nullptr;
  for (Backend& backend : backends) {
    SCOPED_TRACE(backend.name);
    std::vector<u8> image = scenario(*backend.device);
    ASSERT_FALSE(image.empty()) << backend.name << " produced nothing";
    check(image);

    if (!reference_name) {
      reference = std::move(image);
      reference_name = backend.name;
      continue;
    }
    ASSERT_EQ(image.size(), reference.size());
    u32 differing = 0;
    for (size_t i = 0; i < image.size(); i++) {
      const int delta = static_cast<int>(image[i]) - reference[i];
      if (delta > static_cast<int>(tolerance) ||
          delta < -static_cast<int>(tolerance)) {
        differing++;
      }
    }
    EXPECT_EQ(differing, 0u)
        << backend.name << " and " << reference_name << " disagree on "
        << differing << " of " << image.size() << " bytes";
  }
}

std::vector<u8> ClearOnly(Device& device) {
  auto target = MakeTarget(device, "target");
  auto readback = MakeReadback(device, kImageBytes);
  auto list = device.CreateCommandList();
  if (!target || !readback || !list)
    return {};

  RenderPassDesc pass;
  pass.width = pass.height = kSize;
  pass.num_color = 1;
  pass.color[0].texture = target.get();
  pass.color[0].load = LoadOp::Clear;
  // Values that are exactly representable in 8 bits, so the expectation is
  // about the backend rather than about rounding.
  pass.color[0].clear[0] = 64.0f / 255.0f;
  pass.color[0].clear[1] = 128.0f / 255.0f;
  pass.color[0].clear[2] = 192.0f / 255.0f;
  pass.color[0].clear[3] = 1.0f;

  list->Begin();
  list->BeginRenderPass(pass);
  list->EndRenderPass();
  CopyTargetToReadback(*list, *target, *readback);
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kImageBytes);
}

struct SolidVertex {
  float x, y, z;
  float r, g, b, a;
};

// A triangle large enough to cover the whole target, so every pixel is
// determined and no edge rule can make the two backends differ.
const SolidVertex kCoverRed[3] = {
    {-1.0f, -1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
    {3.0f, -1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
    {-1.0f, 3.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
};

// The same shape moved so it covers only the +Y half of clip space. This is
// what pins the orientation: the covered half has to come back as the TOP half
// of the image, and a backend that leaves clip space Y-down fills the bottom.
const SolidVertex kTopHalf[3] = {
    {-1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
    {3.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
    {-1.0f, 4.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
};

// A different colour at each corner, for comparing how the two backends
// interpolate. The -Y corner is the only vertex inside the viewport, so it is
// the only pixel whose colour is a vertex colour rather than a blend.
const SolidVertex kCornerColors[3] = {
    {-1.0f, -1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f},
    {3.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f},
    {-1.0f, 3.0f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f},
};

GraphicsPipelineDesc SolidPipelineDesc(const ShaderModule* vs,
                                       const ShaderModule* ps) {
  GraphicsPipelineDesc desc;
  desc.vertex = vs;
  desc.pixel = ps;
  desc.num_vertex_buffers = 1;
  desc.vertex_buffers[0].stride = sizeof(SolidVertex);
  desc.num_attributes = 2;
  desc.attributes[0] = {0, 0, 0, Format::RGB32Float};
  desc.attributes[1] = {1, 0, 12, Format::RGBA32Float};
  desc.num_color_targets = 1;
  desc.color_formats[0] = Format::RGBA8Unorm;
  desc.name = "solid";
  return desc;
}

std::vector<u8> SolidTriangle(Device& device, const SolidVertex* vertices) {
  auto vs = device.CreateShader(Shader(kStageVertex, kVsSolidSpv,
                                       sizeof(kVsSolidSpv), "vs_solid",
                                       "vs_5_1"));
  auto ps = device.CreateShader(Shader(kStagePixel, kPsSolidSpv,
                                       sizeof(kPsSolidSpv), "ps_solid",
                                       "ps_5_1"));
  if (!vs || !ps)
    return {};
  auto pipeline = device.CreateGraphicsPipeline(
      SolidPipelineDesc(vs.get(), ps.get()));
  auto target = MakeTarget(device, "target");
  auto readback = MakeReadback(device, kImageBytes);
  auto vbuf = MakeUpload(device, vertices, sizeof(SolidVertex) * 3,
                         kBufferVertex, "verts");
  const u16 indices[3] = {0, 1, 2};
  auto ibuf =
      MakeUpload(device, indices, sizeof(indices), kBufferIndex, "indices");
  auto list = device.CreateCommandList();
  if (!pipeline || !target || !readback || !vbuf || !ibuf || !list)
    return {};

  RenderPassDesc pass;
  pass.width = pass.height = kSize;
  pass.num_color = 1;
  pass.color[0].texture = target.get();
  pass.color[0].load = LoadOp::Clear;

  list->Begin();
  list->BeginRenderPass(pass);
  list->SetPipeline(pipeline.get());
  list->SetVertexBuffer(0, vbuf.get(), 0);
  list->SetIndexBuffer(ibuf.get(), 0, IndexType::U16);
  list->SetViewport(0, 0, kSize, kSize, 0.0f, 1.0f);
  list->SetScissor(0, 0, kSize, kSize);
  list->DrawIndexed(3, 1, 0, 0, 0);
  list->EndRenderPass();
  CopyTargetToReadback(*list, *target, *readback);
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kImageBytes);
}

std::vector<u8> DepthOrdering(Device& device) {
  auto vs = device.CreateShader(Shader(kStageVertex, kVsSolidSpv,
                                       sizeof(kVsSolidSpv), "vs_solid",
                                       "vs_5_1"));
  auto ps = device.CreateShader(Shader(kStagePixel, kPsSolidSpv,
                                       sizeof(kPsSolidSpv), "ps_solid",
                                       "ps_5_1"));
  if (!vs || !ps)
    return {};
  GraphicsPipelineDesc pipeline_desc = SolidPipelineDesc(vs.get(), ps.get());
  pipeline_desc.depth.test_enable = true;
  pipeline_desc.depth.write_enable = true;
  pipeline_desc.depth.compare = CompareOp::Less;
  pipeline_desc.depth_format = Format::D32Float;
  pipeline_desc.name = "depth";
  auto pipeline = device.CreateGraphicsPipeline(pipeline_desc);

  TextureDesc depth_desc;
  depth_desc.width = depth_desc.height = kSize;
  depth_desc.format = Format::D32Float;
  depth_desc.usage = kTextureDepthStencil;
  depth_desc.name = "depth";
  auto depth = device.CreateTexture(depth_desc);
  auto target = MakeTarget(device, "target");
  auto readback = MakeReadback(device, kImageBytes);
  auto list = device.CreateCommandList();
  if (!pipeline || !depth || !target || !readback || !list)
    return {};

  // Near enough to pass, then behind (rejected), then nearest of all.
  SolidVertex draws[3][3];
  const float depths[3] = {0.5f, 0.75f, 0.25f};
  const float colors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  std::unique_ptr<Buffer> buffers[3];
  for (u32 i = 0; i < 3; i++) {
    memcpy(draws[i], kCoverRed, sizeof(kCoverRed));
    for (SolidVertex& v : draws[i]) {
      v.z = depths[i];
      v.r = colors[i][0];
      v.g = colors[i][1];
      v.b = colors[i][2];
    }
    buffers[i] = MakeUpload(device, draws[i], sizeof(draws[i]), kBufferVertex,
                            "verts");
    if (!buffers[i])
      return {};
  }

  RenderPassDesc pass;
  pass.width = pass.height = kSize;
  pass.num_color = 1;
  pass.color[0].texture = target.get();
  pass.color[0].load = LoadOp::Clear;
  pass.has_depth = true;
  pass.depth.texture = depth.get();
  pass.depth.load = LoadOp::Clear;
  pass.depth.clear_depth = 1.0f;

  list->Begin();
  list->BeginRenderPass(pass);
  list->SetPipeline(pipeline.get());
  list->SetViewport(0, 0, kSize, kSize, 0.0f, 1.0f);
  list->SetScissor(0, 0, kSize, kSize);
  for (u32 i = 0; i < 3; i++) {
    list->SetVertexBuffer(0, buffers[i].get(), 0);
    list->Draw(3, 1, 0, 0);
  }
  list->EndRenderPass();
  CopyTargetToReadback(*list, *target, *readback);
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kImageBytes);
}

// Both colour targets of an MRT pass, one after the other in the result.
std::vector<u8> MultipleRenderTargets(Device& device) {
  auto vs = device.CreateShader(Shader(kStageVertex, kVsSolidSpv,
                                       sizeof(kVsSolidSpv), "vs_solid",
                                       "vs_5_1"));
  auto ps = device.CreateShader(
      Shader(kStagePixel, kPsMrtSpv, sizeof(kPsMrtSpv), "ps_mrt", "ps_5_1"));
  if (!vs || !ps)
    return {};
  GraphicsPipelineDesc pipeline_desc = SolidPipelineDesc(vs.get(), ps.get());
  pipeline_desc.num_color_targets = 2;
  pipeline_desc.color_formats[1] = Format::RGBA8Unorm;
  pipeline_desc.name = "mrt";
  auto pipeline = device.CreateGraphicsPipeline(pipeline_desc);

  auto first = MakeTarget(device, "mrt0");
  auto second = MakeTarget(device, "mrt1");
  auto readback = MakeReadback(device, kImageBytes * 2);
  SolidVertex vertices[3];
  memcpy(vertices, kCoverRed, sizeof(kCoverRed));
  for (SolidVertex& v : vertices) {
    v.r = 1.0f;
    v.g = 128.0f / 255.0f;
    v.b = 64.0f / 255.0f;
    v.a = 0.0f;
  }
  auto vbuf =
      MakeUpload(device, vertices, sizeof(vertices), kBufferVertex, "verts");
  auto list = device.CreateCommandList();
  if (!pipeline || !first || !second || !readback || !vbuf || !list)
    return {};

  RenderPassDesc pass;
  pass.width = pass.height = kSize;
  pass.num_color = 2;
  pass.color[0].texture = first.get();
  pass.color[0].load = LoadOp::Clear;
  pass.color[1].texture = second.get();
  pass.color[1].load = LoadOp::Clear;

  list->Begin();
  list->BeginRenderPass(pass);
  list->SetPipeline(pipeline.get());
  list->SetVertexBuffer(0, vbuf.get(), 0);
  list->SetViewport(0, 0, kSize, kSize, 0.0f, 1.0f);
  list->SetScissor(0, 0, kSize, kSize);
  list->Draw(3, 1, 0, 0);
  list->EndRenderPass();

  list->Transition(first.get(), ResourceState::CopySrc);
  list->Transition(second.get(), ResourceState::CopySrc);
  list->Transition(readback.get(), ResourceState::CopyDst);
  TextureRegion region;
  region.width = kSize;
  region.height = kSize;
  list->CopyTextureToBuffer(readback.get(), 0, kSize * 4, first.get(), region);
  list->CopyTextureToBuffer(readback.get(), kImageBytes, kSize * 4,
                            second.get(), region);
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kImageBytes * 2);
}

struct TexVertex {
  float x, y;
  float u, v;
};

// Two triangles covering the target, uv running 0..1 with (0,0) at the corner
// that ends up top-left in the image.
const TexVertex kQuad[6] = {
    {-1.0f, 1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
    {-1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 0.0f},
    {1.0f, -1.0f, 1.0f, 1.0f}, {-1.0f, -1.0f, 0.0f, 1.0f},
};

// A textured draw whose constant buffer is selected by a dynamic offset and
// whose colour is scaled by push constants. `entry` picks which of the two
// offsets in the buffer the draw reads: entry 0 leaves the quad on screen,
// entry 1 pushes it off.
std::vector<u8> TexturedQuad(Device& device, u32 entry, const float tint[4]) {
  auto vs = device.CreateShader(
      Shader(kStageVertex, kVsTexSpv, sizeof(kVsTexSpv), "vs_tex", "vs_5_1"));
  auto ps = device.CreateShader(
      Shader(kStagePixel, kPsTexSpv, sizeof(kPsTexSpv), "ps_tex", "ps_5_1"));
  if (!vs || !ps)
    return {};

  BindGroupLayoutDesc texture_layout_desc;
  texture_layout_desc.index = 0;
  texture_layout_desc.name = "texture";
  texture_layout_desc.num_entries = 2;
  texture_layout_desc.entries[0] = {0, BindingType::SampledTexture,
                                    kStagePixel, false};
  texture_layout_desc.entries[1] = {1, BindingType::Sampler, kStagePixel,
                                    false};
  auto texture_layout = device.CreateBindGroupLayout(texture_layout_desc);

  BindGroupLayoutDesc xform_layout_desc;
  xform_layout_desc.index = 1;
  xform_layout_desc.name = "xform";
  xform_layout_desc.num_entries = 1;
  xform_layout_desc.entries[0] = {0, BindingType::UniformBuffer, kStageVertex,
                                  true};
  auto xform_layout = device.CreateBindGroupLayout(xform_layout_desc);
  if (!texture_layout || !xform_layout)
    return {};

  GraphicsPipelineDesc pipeline_desc;
  pipeline_desc.vertex = vs.get();
  pipeline_desc.pixel = ps.get();
  pipeline_desc.num_vertex_buffers = 1;
  pipeline_desc.vertex_buffers[0].stride = sizeof(TexVertex);
  pipeline_desc.num_attributes = 2;
  pipeline_desc.attributes[0] = {0, 0, 0, Format::RG32Float};
  pipeline_desc.attributes[1] = {1, 0, 8, Format::RG32Float};
  pipeline_desc.num_color_targets = 1;
  pipeline_desc.color_formats[0] = Format::RGBA8Unorm;
  pipeline_desc.bind_groups[0] = texture_layout.get();
  pipeline_desc.bind_groups[1] = xform_layout.get();
  pipeline_desc.num_bind_groups = 2;
  pipeline_desc.push_constant_bytes = 16;
  pipeline_desc.name = "textured";
  auto pipeline = device.CreateGraphicsPipeline(pipeline_desc);

  // A 2x2 texture whose four texels are distinguishable, staged through a
  // buffer with the row pitch D3D12 requires.
  const u32 pitch = device.caps().texture_row_pitch_alignment > 8
                        ? device.caps().texture_row_pitch_alignment
                        : 8;
  std::vector<u8> texels(pitch * 2, 0);
  const Rgba corner[4] = {{255, 0, 0, 255},
                          {0, 255, 0, 255},
                          {0, 0, 255, 255},
                          {255, 255, 255, 255}};
  memcpy(&texels[0], &corner[0], 4);
  memcpy(&texels[4], &corner[1], 4);
  memcpy(&texels[pitch], &corner[2], 4);
  memcpy(&texels[pitch + 4], &corner[3], 4);
  auto staging =
      MakeUpload(device, texels.data(), texels.size(), 0, "texel staging");

  TextureDesc texture_desc;
  texture_desc.width = texture_desc.height = 2;
  texture_desc.format = Format::RGBA8Unorm;
  texture_desc.usage = kTextureSampled | kTextureCopyDst;
  texture_desc.name = "texture";
  auto texture = device.CreateTexture(texture_desc);

  SamplerDesc sampler_desc;
  sampler_desc.min_filter = FilterMode::Nearest;
  sampler_desc.mag_filter = FilterMode::Nearest;
  sampler_desc.mip_filter = FilterMode::Nearest;
  sampler_desc.address_u = AddressMode::ClampToEdge;
  sampler_desc.address_v = AddressMode::ClampToEdge;
  auto sampler = device.CreateSampler(sampler_desc);

  // Two float4 transforms, one per dynamic offset. The alignment is the
  // device's, because it is the constraint that decides the stride.
  const u32 stride = device.caps().constant_buffer_offset_alignment;
  std::vector<u8> xforms(stride * 2, 0);
  const float on_screen[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float off_screen[4] = {2.0f, 0.0f, 0.0f, 0.0f};
  memcpy(&xforms[0], on_screen, sizeof(on_screen));
  memcpy(&xforms[stride], off_screen, sizeof(off_screen));
  auto xform_buffer = MakeUpload(device, xforms.data(), xforms.size(),
                                 kBufferConstant, "xforms");

  auto vbuf = MakeUpload(device, kQuad, sizeof(kQuad), kBufferVertex, "quad");
  auto target = MakeTarget(device, "target");
  auto readback = MakeReadback(device, kImageBytes);
  auto list = device.CreateCommandList();
  if (!pipeline || !staging || !texture || !sampler || !xform_buffer ||
      !vbuf || !target || !readback || !list) {
    return {};
  }

  BindGroupDesc texture_group_desc;
  texture_group_desc.layout = texture_layout.get();
  texture_group_desc.name = "texture group";
  texture_group_desc.num_bindings = 2;
  texture_group_desc.bindings[0].binding = 0;
  texture_group_desc.bindings[0].texture = texture.get();
  texture_group_desc.bindings[1].binding = 1;
  texture_group_desc.bindings[1].sampler = sampler.get();
  auto texture_group = device.CreateBindGroup(texture_group_desc);

  BindGroupDesc xform_group_desc;
  xform_group_desc.layout = xform_layout.get();
  xform_group_desc.name = "xform group";
  xform_group_desc.num_bindings = 1;
  xform_group_desc.bindings[0].binding = 0;
  xform_group_desc.bindings[0].buffer = xform_buffer.get();
  xform_group_desc.bindings[0].size = 16;
  auto xform_group = device.CreateBindGroup(xform_group_desc);
  if (!texture_group || !xform_group)
    return {};

  list->Begin();
  list->Transition(staging.get(), ResourceState::CopySrc);
  list->Transition(texture.get(), ResourceState::CopyDst);
  TextureRegion upload;
  upload.width = upload.height = 2;
  list->CopyBufferToTexture(texture.get(), upload, staging.get(), 0, pitch);
  list->Transition(texture.get(), ResourceState::ShaderRead);

  RenderPassDesc pass;
  pass.width = pass.height = kSize;
  pass.num_color = 1;
  pass.color[0].texture = target.get();
  pass.color[0].load = LoadOp::Clear;

  const u32 dynamic_offset = entry * stride;
  list->BeginRenderPass(pass);
  list->SetPipeline(pipeline.get());
  list->SetBindGroup(0, texture_group.get());
  list->SetBindGroup(1, xform_group.get(), &dynamic_offset, 1);
  list->SetPushConstants(0, 16, tint);
  list->SetVertexBuffer(0, vbuf.get(), 0);
  list->SetViewport(0, 0, kSize, kSize, 0.0f, 1.0f);
  list->SetScissor(0, 0, kSize, kSize);
  list->Draw(6, 1, 0, 0);
  list->EndRenderPass();
  CopyTargetToReadback(*list, *target, *readback);
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kImageBytes);
}

// The compute scenario returns its buffer rather than an image.
std::vector<u8> ComputeFill(Device& device) {
  auto cs = device.CreateShader(Shader(kStageCompute, kCsFillSpv,
                                       sizeof(kCsFillSpv), "cs_fill",
                                       "cs_5_1"));
  if (!cs)
    return {};

  BindGroupLayoutDesc layout_desc;
  layout_desc.index = 2;
  layout_desc.name = "storage";
  layout_desc.num_entries = 1;
  layout_desc.entries[0] = {0, BindingType::StorageBuffer, kStageCompute,
                            false};
  auto layout = device.CreateBindGroupLayout(layout_desc);
  if (!layout)
    return {};

  ComputePipelineDesc pipeline_desc;
  pipeline_desc.compute = cs.get();
  // Groups 0 and 1 are declared but unused, so the group index the shader
  // names still selects the right table.
  pipeline_desc.bind_groups[2] = layout.get();
  pipeline_desc.num_bind_groups = 3;
  pipeline_desc.push_constant_bytes = 16;
  pipeline_desc.name = "fill";
  auto pipeline = device.CreateComputePipeline(pipeline_desc);

  constexpr u32 kValues = 64;
  BufferDesc storage_desc;
  storage_desc.size = kValues * sizeof(u32);
  storage_desc.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  storage_desc.name = "out";
  auto storage = device.CreateBuffer(storage_desc);
  auto readback = MakeReadback(device, kValues * sizeof(u32));
  auto list = device.CreateCommandList();
  if (!pipeline || !storage || !readback || !list)
    return {};

  BindGroupDesc group_desc;
  group_desc.layout = layout.get();
  group_desc.name = "storage group";
  group_desc.num_bindings = 1;
  group_desc.bindings[0].binding = 0;
  group_desc.bindings[0].buffer = storage.get();
  auto group = device.CreateBindGroup(group_desc);
  if (!group)
    return {};

  // The shader reads its multiplier out of the push constants as raw bits.
  u8 push[16] = {};
  const u32 multiplier = 3;
  memcpy(push, &multiplier, sizeof(multiplier));

  list->Begin();
  list->Transition(storage.get(), ResourceState::UnorderedAccess);
  list->SetPipeline(pipeline.get());
  list->SetBindGroup(2, group.get());
  list->SetPushConstants(0, 16, push);
  list->Dispatch(kValues / 8, 1, 1);
  list->Transition(storage.get(), ResourceState::CopySrc);
  list->Transition(readback.get(), ResourceState::CopyDst);
  list->CopyBuffer(readback.get(), 0, storage.get(), 0,
                   kValues * sizeof(u32));
  list->End();
  device.Wait(device.Submit(list.get()));
  return Drain(*readback, kValues * sizeof(u32));
}

TEST(RhiConformance, AtLeastOneBackendExists) {
  if (Backends().empty())
    GTEST_SKIP() << "no RHI backend could be created";
  for (const Backend& backend : Backends())
    EXPECT_STREQ(backend.name, backend.device->backend_name());
}

TEST(RhiConformance, EveryFormatResolvesOnEveryBackend) {
  if (Backends().empty())
    GTEST_SKIP() << "no RHI backend could be created";
  for (const Backend& backend : Backends()) {
    SCOPED_TRACE(backend.name);
    for (u32 i = 1; i < static_cast<u32>(Format::kCount); i++) {
      const Format format = static_cast<Format>(i);
      // RGB32Float is a vertex format on both APIs and a texture format on
      // neither, so it is the one entry with nothing to create here.
      if (format == Format::RGB32Float)
        continue;
      TextureDesc desc;
      // A multiple of the block size, so a compressed format is whole blocks.
      desc.width = desc.height = 8;
      desc.format = format;
      desc.usage = kTextureSampled | kTextureCopyDst;
      desc.name = FormatName(format);
      if (FormatIsDepth(format))
        desc.usage |= kTextureDepthStencil;
      EXPECT_NE(backend.device->CreateTexture(desc), nullptr)
          << FormatName(format);
    }
  }
}

TEST(RhiConformance, ClearFillsTheWholeTarget) {
  ForEachBackend(ClearOnly, [](const std::vector<u8>& image) {
    for (u32 y = 0; y < kSize; y++) {
      for (u32 x = 0; x < kSize; x++) {
        ASSERT_TRUE(At(image, x, y) == (Rgba{64, 128, 192, 255}))
            << "at " << x << "," << y;
      }
    }
  });
}

TEST(RhiConformance, ACoveringTriangleReachesEveryPixel) {
  ForEachBackend(
      [](Device& device) { return SolidTriangle(device, kCoverRed); },
      [](const std::vector<u8>& image) {
        for (u32 i = 0; i < kPixels; i++) {
          ASSERT_TRUE(At(image, i % kSize, i / kSize) ==
                      (Rgba{255, 0, 0, 255}))
              << "at pixel " << i;
        }
      });
}

TEST(RhiConformance, ClipSpaceHasYUpOnEveryBackend) {
  ForEachBackend(
      [](Device& device) { return SolidTriangle(device, kTopHalf); },
      [](const std::vector<u8>& image) {
        for (u32 x = 0; x < kSize; x++) {
          for (u32 y = 0; y < kSize / 2; y++) {
            ASSERT_TRUE(At(image, x, y) == (Rgba{255, 0, 0, 255}))
                << "top half at " << x << "," << y;
          }
          for (u32 y = kSize / 2; y < kSize; y++) {
            ASSERT_TRUE(At(image, x, y) == (Rgba{0, 0, 0, 0}))
                << "bottom half at " << x << "," << y;
          }
        }
      });
}

TEST(RhiConformance, InterpolationAgreesAcrossBackends) {
  ForEachBackend(
      [](Device& device) { return SolidTriangle(device, kCornerColors); },
      [](const std::vector<u8>& image) {
        // The red vertex sits exactly on the bottom-left pixel of the image.
        const Rgba corner = At(image, 0, kSize - 1);
        EXPECT_GT(corner.r, 240);
        EXPECT_LT(corner.g, 16);
      },
      /*tolerance=*/2);
}

TEST(RhiConformance, DepthTestKeepsTheNearestDraw) {
  ForEachBackend(DepthOrdering, [](const std::vector<u8>& image) {
    for (u32 i = 0; i < kPixels; i++) {
      ASSERT_TRUE(At(image, i % kSize, i / kSize) == (Rgba{0, 0, 255, 255}))
          << "at pixel " << i;
    }
  });
}

TEST(RhiConformance, EachRenderTargetGetsItsOwnShaderOutput) {
  ForEachBackend(MultipleRenderTargets, [](const std::vector<u8>& image) {
    ASSERT_EQ(image.size(), kImageBytes * 2u);
    const u8* first = image.data();
    const u8* second = image.data() + kImageBytes;
    for (u32 i = 0; i < kPixels; i++) {
      ASSERT_EQ(first[i * 4 + 0], 255) << "at pixel " << i;
      ASSERT_EQ(first[i * 4 + 1], 128) << "at pixel " << i;
      ASSERT_EQ(first[i * 4 + 2], 64) << "at pixel " << i;
      ASSERT_EQ(first[i * 4 + 3], 0) << "at pixel " << i;
      // The second target takes the same colour reversed.
      ASSERT_EQ(second[i * 4 + 0], 0) << "at pixel " << i;
      ASSERT_EQ(second[i * 4 + 1], 64) << "at pixel " << i;
      ASSERT_EQ(second[i * 4 + 2], 128) << "at pixel " << i;
      ASSERT_EQ(second[i * 4 + 3], 255) << "at pixel " << i;
    }
  });
}

TEST(RhiConformance, ATexturedQuadSamplesEveryTexel) {
  const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  ForEachBackend(
      [&](Device& device) { return TexturedQuad(device, 0, white); },
      [](const std::vector<u8>& image) {
        const u32 near_edge = kSize / 4;
        const u32 far_edge = kSize - kSize / 4;
        EXPECT_TRUE(At(image, near_edge, near_edge) ==
                    (Rgba{255, 0, 0, 255}));
        EXPECT_TRUE(At(image, far_edge, near_edge) == (Rgba{0, 255, 0, 255}));
        EXPECT_TRUE(At(image, near_edge, far_edge) == (Rgba{0, 0, 255, 255}));
        EXPECT_TRUE(At(image, far_edge, far_edge) ==
                    (Rgba{255, 255, 255, 255}));
      });
}

TEST(RhiConformance, ADynamicOffsetSelectsTheConstantsTheDrawReads) {
  const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  // Entry 1 translates the quad clear of the viewport, so nothing is drawn.
  // A backend that ignored the offset would draw the quad instead.
  ForEachBackend(
      [&](Device& device) { return TexturedQuad(device, 1, white); },
      [](const std::vector<u8>& image) {
        for (u32 i = 0; i < kPixels; i++)
          ASSERT_TRUE(At(image, i % kSize, i / kSize) == (Rgba{0, 0, 0, 0}))
              << "at pixel " << i;
      });
}

TEST(RhiConformance, PushConstantsReachThePixelShader) {
  const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  ForEachBackend(
      [&](Device& device) { return TexturedQuad(device, 0, black); },
      [](const std::vector<u8>& image) {
        // The quad is drawn, but the tint zeroes its colour. Alpha survives,
        // which is what separates this from the quad not being drawn at all.
        EXPECT_TRUE(At(image, kSize / 2, kSize / 2) == (Rgba{0, 0, 0, 255}));
      });
}

TEST(RhiConformance, AComputeDispatchWritesItsStorageBuffer) {
  ForEachBackend(ComputeFill, [](const std::vector<u8>& raw) {
    ASSERT_EQ(raw.size(), 64u * sizeof(u32));
    const u32* values = reinterpret_cast<const u32*>(raw.data());
    for (u32 i = 0; i < 64; i++)
      ASSERT_EQ(values[i], i * 3 + 1) << "at index " << i;
  });
}

}  // namespace
}  // namespace gpu::rhi
