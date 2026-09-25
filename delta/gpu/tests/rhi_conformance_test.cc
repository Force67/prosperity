// Runs the same work through every rhi backend the build has and checks the
// pixels against fixed expectations, so a difference is a backend bug rather
// than a difference of opinion between two backends. Needs a GPU; a backend
// that cannot create a device is skipped.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/arch.h"
#include "gpu/render/backend.h"
#include "gpu/rhi/device.h"
#include "gpu/tests/rhi_conformance_spv.h"
#include "gpu/tests/rhi_conformance_extra_spv.h"

namespace {

using namespace gpu::rhi;

constexpr u32 kW = 64, kH = 64;

template <size_t N>
ShaderCode Spv(const u32 (&words)[N]) {
  return {words, N};
}

class RhiConformance : public ::testing::TestWithParam<Backend> {
 protected:
  void SetUp() override {
    // One device per backend for the whole run, never destroyed: the Vulkan
    // backend leaves its VkDevice to process exit, and a device per test
    // exhausts the driver after a few dozen tests.
    static std::map<Backend, Device*> devices;
    Device*& shared = devices[GetParam()];
    if (!shared)
      shared = gpu::render::CreateBackendDevice(GetParam()).release();
    device_.reset(shared);
    if (!device_)
      GTEST_SKIP() << BackendName(GetParam()) << " unavailable";
    list_ = device_->CreateCommandList();
    ASSERT_NE(list_, nullptr);
  }
  void TearDown() override {
    if (!device_)
      return;
    device_->WaitIdle();
    for (Object* o : objects_)
      device_->Destroy(o);
    device_->Destroy(list_);
    device_.release();  // NOLINT: owned by SetUp's table
  }

  template <typename T>
  T* Own(T* object) {
    EXPECT_NE(object, nullptr);
    objects_.push_back(object);
    return object;
  }

  Texture* Target(Format format = Format::kRGBA8Unorm) {
    TextureDesc td;
    td.format = format;
    td.width = kW;
    td.height = kH;
    td.usage = kTextureColorTarget | kTextureSampled | kTextureCopySrc |
               kTextureCopyDst;
    return Own(device_->CreateTexture(td));
  }

  TextureView* View(Texture* t) {
    TextureViewDesc vd;
    vd.format = t->desc().format;
    const FormatInfo& fi = GetFormatInfo(vd.format);
    vd.aspect = fi.is_depth ? kAspectDepth : kAspectColor;
    return Own(device_->CreateView(t, vd));
  }

  Buffer* Readback(u64 bytes) {
    BufferDesc bd;
    bd.size = bytes;
    bd.usage = kBufferCopyDst;
    bd.memory = MemoryKind::kReadback;
    return Own(device_->CreateBuffer(bd));
  }

  Buffer* Upload(const void* data, u64 bytes, u32 usage) {
    BufferDesc bd;
    bd.size = bytes;
    bd.usage = usage | kBufferCopySrc;
    bd.memory = MemoryKind::kUpload;
    Buffer* b = Own(device_->CreateBuffer(bd));
    if (b && data)
      std::memcpy(b->mapped(), data, bytes);
    return b;
  }

  // Copy a kW x kH RGBA8 target (in `state`) to host memory.
  std::vector<u32> ReadPixels(Texture* t, TextureState state) {
    Buffer* rb = Readback(kW * kH * 4);
    TextureBarrier b{t, state, TextureState::kCopySrc};
    list_->Barrier(0, 0, &b, 1);
    BufferTextureCopy c;
    c.region.width = kW;
    c.region.height = kH;
    list_->CopyTextureToBuffer(rb, t, &c, 1);
    list_->Barrier(kAccessCopyWrite, kAccessHostRead);
    list_->End();
    EXPECT_TRUE(device_->Wait(device_->Submit(list_)));
    std::vector<u32> px(kW * kH);
    std::memcpy(px.data(), rb->mapped(), px.size() * 4);
    list_->Begin();
    return px;
  }

  void Submit() {
    list_->End();
    EXPECT_TRUE(device_->Wait(device_->Submit(list_)));
    list_->Begin();
  }

  static u32 Rgba(u8 r, u8 g, u8 b, u8 a) {
    return r | (g << 8) | (b << 16) | (u32(a) << 24);
  }

  std::unique_ptr<Device> device_;
  CommandList* list_ = nullptr;
  std::vector<Object*> objects_;
};

TEST_P(RhiConformance, ClearByLoadOp) {
  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(0, 0, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = v;
  rp.colors[0].load = LoadOp::kClear;
  rp.colors[0].clear = {{1.0f, 0.0f, 1.0f, 1.0f}};
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[0], Rgba(255, 0, 255, 255));
  EXPECT_EQ(px[kW * kH - 1], Rgba(255, 0, 255, 255));
}

// A triangle over the upper half of clip space, through a y-up viewport, must
// land in the top rows of the target (row 0 = the lowest address), with
// vertex colour, a dynamic-offset UBO, push constants and MRT all applied.
TEST_P(RhiConformance, TriangleOrientationMrtUboPush) {
  Texture* t0 = Target();
  Texture* t1 = Target();
  TextureView* v0 = View(t0);
  TextureView* v1 = View(t1);

  BindGroupLayoutDesc empty;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(empty));
  BindGroupLayoutDesc ubo_desc;
  ubo_desc.bindings.push_back({0, BindingType::kUniformBufferDynamic,
                               kStageVertex});
  BindGroupLayout* set1 = Own(device_->CreateBindGroupLayout(ubo_desc));
  PipelineLayoutDesc pl;
  pl.groups = {set0, set1};
  pl.push_constant_bytes = 16;
  pl.push_constant_stages = kStageVertex | kStageFragment;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));

  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_tri_vert_spv);
  gp.fragment = Spv(k_tri_frag_spv);
  gp.vertex_buffers = {{24, false}};
  gp.vertex_attributes = {{0, 0, Format::kRG32Float, 0},
                          {1, 0, Format::kRGBA32Float, 8}};
  gp.color_count = 2;
  gp.color_formats[0] = gp.color_formats[1] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);

  // pos.xy, rgba: covers x in [-1,1], y in [0,1] (before the push offset).
  const float verts[] = {-1.0f, 0.0f, 1, 0, 0, 1, 3.0f, 0.0f, 1, 0, 0, 1,
                         -1.0f, 2.0f, 1, 0, 0, 1};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  // Two 256-byte windows; the draw binds the second, a unit tint.
  std::vector<float> ubo(128, 0.0f);
  ubo[0] = ubo[1] = ubo[2] = ubo[3] = 9.0f;
  ubo[64] = 1.0f;
  ubo[65] = 1.0f;
  ubo[66] = 1.0f;
  ubo[67] = 1.0f;
  Buffer* ub = Upload(ubo.data(), ubo.size() * 4, kBufferUniform);
  BindGroupDesc gd;
  gd.layout = set1;
  BindingWrite w;
  w.binding = 0;
  w.buffer = ub;
  w.range = 16;
  gd.writes.push_back(w);
  BindGroup* group = Own(device_->CreateBindGroup(gd));
  BindGroupDesc gd0;
  gd0.layout = set0;
  BindGroup* group0 = Own(device_->CreateBindGroup(gd0));

  list_->Begin();
  TextureBarrier b[2] = {
      {t0, TextureState::kUndefined, TextureState::kColorTarget},
      {t1, TextureState::kUndefined, TextureState::kColorTarget}};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead | kAccessUniformRead, b,
                 2);
  RenderPassDesc rp;
  rp.color_count = 2;
  rp.colors[0].view = v0;
  rp.colors[1].view = v1;
  rp.colors[0].load = rp.colors[1].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  list_->SetViewport(0, kH, kW, -static_cast<float>(kH), 0, 1);
  list_->SetScissor(0, 0, kW, kH);
  const float push[4] = {0.0f, 0.0f, 0.5f, 0.0f};
  list_->SetPushConstants(0, 16, push);
  list_->SetBindGroup(0, group0);
  const u32 offset = 256;
  list_->SetBindGroup(1, group, &offset, 1);
  u64 zero = 0;
  list_->SetVertexBuffers(0, 1, &vb, &zero);
  list_->Draw(3, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> p0 = ReadPixels(t0, TextureState::kColorTarget);
  std::vector<u32> p1 = ReadPixels(t1, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(p0[4 * kW + 4], Rgba(255, 0, 0, 255)) << "top row not covered";
  EXPECT_EQ(p0[(kH - 4) * kW + 4], 0u) << "bottom row covered";
  EXPECT_EQ(p1[4 * kW + 4], Rgba(0, 255, 255, 0)) << "second target";
}

TEST_P(RhiConformance, SampledTextureUpload) {
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.width = 2;
  td.height = 2;
  td.usage = kTextureSampled | kTextureCopyDst;
  Texture* src = Own(device_->CreateTexture(td));
  TextureView* src_view = View(src);
  const u32 texels[4] = {Rgba(255, 0, 0, 255), Rgba(0, 255, 0, 255),
                         Rgba(0, 0, 255, 255), Rgba(255, 255, 255, 255)};
  Buffer* staging = Upload(texels, sizeof(texels), 0);
  SamplerDesc sd;
  sd.mag = sd.min = Filter::kNearest;
  Sampler* sampler = Own(device_->CreateSampler(sd));

  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kSampledTexture, kStageFragment});
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_fullscreen_vert_spv);
  gp.fragment = Spv(k_tex_frag_spv);
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  BindGroupDesc gd;
  gd.layout = set0;
  BindingWrite w;
  w.binding = 0;
  w.view = src_view;
  w.sampler = sampler;
  gd.writes.push_back(w);
  BindGroup* group = Own(device_->CreateBindGroup(gd));

  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier up{src, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, &up, 1);
  BufferTextureCopy c;
  c.region.width = 2;
  c.region.height = 2;
  list_->CopyBufferToTexture(src, staging, &c, 1);
  TextureBarrier b[2] = {
      {src, TextureState::kCopyDst, TextureState::kShaderRead},
      {t, TextureState::kUndefined, TextureState::kColorTarget}};
  list_->Barrier(0, 0, b, 2);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = v;
  rp.colors[0].load = LoadOp::kDontCare;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  list_->SetViewport(0, kH, kW, -static_cast<float>(kH), 0, 1);
  list_->SetScissor(0, 0, kW, kH);
  list_->SetBindGroup(0, group);
  list_->Draw(3, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[2 * kW + 2], texels[0]) << "top-left";
  EXPECT_EQ(px[2 * kW + kW - 3], texels[1]) << "top-right";
  EXPECT_EQ(px[(kH - 3) * kW + 2], texels[2]) << "bottom-left";
}

TEST_P(RhiConformance, ComputeStorageBufferAndPush) {
  constexpr u32 kN = 256;
  std::vector<u32> data(kN);
  for (u32 i = 0; i < kN; i++)
    data[i] = i;
  BufferDesc bd;
  bd.size = kN * 4;
  bd.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  Buffer* staging = Upload(data.data(), kN * 4, 0);
  Buffer* rb = Readback(kN * 4);

  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kStorageBuffer, kStageCompute});
  ld.push = true;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  pl.push_constant_bytes = 4;
  pl.push_constant_stages = kStageCompute;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  ComputePipelineDesc cp;
  cp.layout = layout;
  cp.code = Spv(k_double_comp_spv);
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);

  list_->Begin();
  list_->CopyBuffer(buf, 0, staging, 0, kN * 4);
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead | kAccessShaderWrite);
  list_->SetPipeline(pipe);
  BindingWrite w;
  w.binding = 0;
  w.buffer = buf;
  w.range = kN * 4;
  list_->PushBindGroup(0, set0, &w, 1);
  const u32 add = 7;
  list_->SetPushConstants(0, 4, &add);
  list_->Dispatch(kN / 64, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, buf, 0, kN * 4);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* out = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(out[0], 7u);
  EXPECT_EQ(out[100], 207u);
  EXPECT_EQ(out[kN - 1], (kN - 1) * 2 + 7);
}

TEST_P(RhiConformance, StorageImageWrite) {
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.width = kW;
  td.height = kH;
  td.usage = kTextureStorage | kTextureCopySrc;
  Texture* t = Own(device_->CreateTexture(td));
  TextureView* v = View(t);
  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kStorageTexture, kStageCompute});
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  ComputePipelineDesc cp;
  cp.layout = layout;
  cp.code = Spv(k_storage_image_comp_spv);
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);
  BindGroupDesc gd;
  gd.layout = set0;
  BindingWrite w;
  w.binding = 0;
  w.view = v;
  w.view_state = TextureState::kGeneral;
  gd.writes.push_back(w);
  BindGroup* group = Own(device_->CreateBindGroup(gd));

  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kGeneral};
  list_->Barrier(0, 0, &b, 1);
  list_->SetPipeline(pipe);
  list_->SetBindGroup(0, group);
  list_->Dispatch(kW / 8, kH / 8, 1);
  std::vector<u32> px = ReadPixels(t, TextureState::kGeneral);
  list_->End();
  EXPECT_EQ(px[3 * kW + 5], Rgba(5, 3, 0, 255));
  EXPECT_EQ(px[60 * kW + 40], Rgba(40, 60, 0, 255));
}

TEST_P(RhiConformance, CopyClearAndBlit) {
  Texture* a = Target();
  Texture* b = Target();
  list_->Begin();
  TextureBarrier to_dst{a, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(0, 0, &to_dst, 1);
  TextureRange all;
  ClearColor green{};
  green.f[1] = 1.0f;
  green.f[3] = 1.0f;
  list_->ClearTexture(a, TextureState::kCopyDst, all, green);
  TextureBarrier bs[2] = {
      {a, TextureState::kCopyDst, TextureState::kCopySrc},
      {b, TextureState::kUndefined, TextureState::kCopyDst}};
  list_->Barrier(0, 0, bs, 2);
  ClearColor red{};
  red.f[0] = 1.0f;
  red.f[3] = 1.0f;
  list_->ClearTexture(b, TextureState::kCopyDst, all, red);
  list_->Barrier(kAccessCopyWrite, kAccessCopyWrite);
  TextureRegion src;
  src.width = 16;
  src.height = 16;
  TextureRegion dst = src;
  dst.x = 8;
  dst.y = 8;
  list_->CopyTexture(b, dst, a, src);
  if (device_->caps().texture_blit) {
    list_->Barrier(kAccessCopyWrite, kAccessCopyWrite);
    TextureRegion bsrc;
    bsrc.width = 4;
    bsrc.height = 4;
    TextureRegion bdst;
    bdst.x = 40;
    bdst.y = 40;
    bdst.width = 16;
    bdst.height = 16;
    list_->BlitTexture(b, bdst, a, bsrc, Filter::kNearest);
  }
  std::vector<u32> px = ReadPixels(b, TextureState::kCopyDst);
  list_->End();
  EXPECT_EQ(px[0], Rgba(255, 0, 0, 255));
  EXPECT_EQ(px[10 * kW + 10], Rgba(0, 255, 0, 255));
  EXPECT_EQ(px[30 * kW + 30], Rgba(255, 0, 0, 255));
  if (device_->caps().texture_blit)
    EXPECT_EQ(px[50 * kW + 50], Rgba(0, 255, 0, 255));
}

std::string Name(const ::testing::TestParamInfo<Backend>& info) {
  return BackendName(info.param);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         RhiConformance,
                         ::testing::ValuesIn(gpu::render::CompiledBackends()),
                         Name);

}  // namespace

// ---------------------------------------------------------------------------
// Scenarios added with the D3D12 backend: depth, stencil, blending, view
// swizzles and format aliases, texture arrays / 3D / cubes, buffer fills and
// updates, timestamps, draw parameters, culling, geometry shaders, unaligned
// copies, storage bindings with subgroups, varying linkage, scaled vertex
// formats and triangle fans. Shaders: rhi_shaders/extra/.
// ---------------------------------------------------------------------------

namespace {

using namespace gpu::rhi;

struct QuadPush {
  float rect[4];
  float color[4];
  float z[4];
};

void YUp(CommandList* list) {
  list->SetViewport(0, kH, kW, -static_cast<float>(kH), 0, 1);
  list->SetScissor(0, 0, kW, kH);
}

void DrawQuad(CommandList* list,
              float x0,
              float y0,
              float x1,
              float y1,
              const float (&color)[4],
              float z = 0.0f) {
  QuadPush p{{x0, y0, x1, y1},
             {color[0], color[1], color[2], color[3]},
             {z, 0, 0, 0}};
  list->SetPushConstants(0, sizeof(p), &p);
  list->Draw(4, 1, 0, 0);
}

bool Near(u32 a, u32 b, int tolerance = 1) {
  for (int c = 0; c < 4; c++) {
    const int x = (a >> (8 * c)) & 0xff, y = (b >> (8 * c)) & 0xff;
    if (x - y > tolerance || y - x > tolerance)
      return false;
  }
  return true;
}

constexpr float kRed[4] = {1, 0, 0, 1};
constexpr float kGreen[4] = {0, 1, 0, 1};
constexpr float kBlue[4] = {0, 0, 1, 1};

}  // namespace

namespace {

// A layout with only push constants, for the quad shaders.
PipelineLayout* QuadLayout(Device* device) {
  PipelineLayoutDesc pl;
  pl.push_constant_bytes = sizeof(QuadPush);
  pl.push_constant_stages = kStageVertex | kStageFragment;
  return device->CreatePipelineLayout(pl);
}

GraphicsPipelineDesc QuadPipeline(PipelineLayout* layout) {
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_x_quad_vert_spv);
  gp.fragment = Spv(k_x_color_frag_spv);
  gp.topology = Topology::kTriangleStrip;
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  return gp;
}

TEST_P(RhiConformance, DepthTestAndReadback) {
  if (!device_->SupportsFormat(Format::kD32Float, kTextureDepthTarget))
    GTEST_SKIP() << "no D32 depth target";
  TextureDesc dd;
  dd.format = Format::kD32Float;
  dd.width = kW;
  dd.height = kH;
  dd.usage = kTextureDepthTarget | kTextureCopySrc;
  Texture* depth = Own(device_->CreateTexture(dd));
  TextureView* dv = View(depth);
  Texture* color = Target();
  TextureView* cv = View(color);
  PipelineLayout* layout = Own(QuadLayout(device_.get()));
  GraphicsPipelineDesc gp = QuadPipeline(layout);
  gp.depth_format = Format::kD32Float;
  gp.depth_test = gp.depth_write = true;
  gp.depth_compare = CompareOp::kLess;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);

  list_->Begin();
  TextureBarrier b[2] = {
      {color, TextureState::kUndefined, TextureState::kColorTarget},
      {depth, TextureState::kUndefined, TextureState::kDepthTarget}};
  b[1].range.aspect = kAspectDepth;
  list_->Barrier(0, 0, b, 2);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = cv;
  rp.colors[0].load = LoadOp::kClear;
  rp.depth.view = dv;
  rp.depth.depth_load = LoadOp::kClear;
  rp.depth.clear_depth = 1.0f;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  YUp(list_);
  DrawQuad(list_, -1, -1, 1, 1, kRed, 0.5f);
  DrawQuad(list_, -1, -1, 1, 1, kGreen, 0.7f);
  DrawQuad(list_, -1, -1, 0, 1, kBlue, 0.3f);
  list_->EndRenderPass();
  TextureBarrier to_src{depth, TextureState::kDepthTarget,
                        TextureState::kCopySrc};
  to_src.range.aspect = kAspectDepth;
  list_->Barrier(kAccessDepthWrite, kAccessCopyRead, &to_src, 1);
  Buffer* rb = Readback(kW * kH * 4);
  BufferTextureCopy c;
  c.region.aspect = kAspectDepth;
  c.region.width = kW;
  c.region.height = kH;
  list_->CopyTextureToBuffer(rb, depth, &c, 1);
  std::vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 8], Rgba(0, 0, 255, 255)) << "nearer quad";
  EXPECT_EQ(px[32 * kW + 56], Rgba(255, 0, 0, 255)) << "farther quad won";
  const float* z = reinterpret_cast<const float*>(rb->mapped());
  EXPECT_FLOAT_EQ(z[32 * kW + 8], 0.3f);
  EXPECT_FLOAT_EQ(z[32 * kW + 56], 0.5f);
}

TEST_P(RhiConformance, StencilMask) {
  if (!device_->SupportsFormat(Format::kD32FloatS8Uint, kTextureDepthTarget))
    GTEST_SKIP() << "no D32S8 depth target";
  TextureDesc dd;
  dd.format = Format::kD32FloatS8Uint;
  dd.width = kW;
  dd.height = kH;
  dd.usage = kTextureDepthTarget;
  Texture* ds = Own(device_->CreateTexture(dd));
  TextureViewDesc vd;
  vd.aspect = kAspectDepth | kAspectStencil;
  TextureView* dsv = Own(device_->CreateView(ds, vd));
  Texture* color = Target();
  TextureView* cv = View(color);
  PipelineLayout* layout = Own(QuadLayout(device_.get()));
  GraphicsPipelineDesc mark = QuadPipeline(layout);
  mark.depth_format = mark.stencil_format = Format::kD32FloatS8Uint;
  mark.stencil_test = true;
  mark.stencil_front.pass = StencilOp::kReplace;
  mark.stencil_front.reference = 1;
  mark.stencil_back = mark.stencil_front;
  mark.blend[0].write_mask = 0;
  GraphicsPipelineDesc test = mark;
  test.stencil_front = {};
  test.stencil_front.compare = CompareOp::kEqual;
  test.stencil_front.reference = 1;
  test.stencil_back = test.stencil_front;
  test.blend[0].write_mask = 0xF;
  Pipeline* mark_pipe = Own(device_->CreateGraphicsPipeline(mark));
  Pipeline* test_pipe = Own(device_->CreateGraphicsPipeline(test));
  ASSERT_NE(mark_pipe, nullptr);
  ASSERT_NE(test_pipe, nullptr);

  list_->Begin();
  TextureBarrier b[2] = {
      {color, TextureState::kUndefined, TextureState::kColorTarget},
      {ds, TextureState::kUndefined, TextureState::kDepthTarget}};
  b[1].range.aspect = kAspectDepth | kAspectStencil;
  list_->Barrier(0, 0, b, 2);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = cv;
  rp.colors[0].load = LoadOp::kClear;
  rp.depth.view = dsv;
  rp.depth.stencil = true;
  rp.depth.depth_load = rp.depth.stencil_load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  YUp(list_);
  list_->SetPipeline(mark_pipe);
  DrawQuad(list_, -1, -1, 0, 1, kRed);
  list_->SetPipeline(test_pipe);
  DrawQuad(list_, -1, -1, 1, 1, kGreen);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 8], Rgba(0, 255, 0, 255)) << "stencil pass";
  EXPECT_EQ(px[32 * kW + 56], 0u) << "stencil fail";
}

TEST_P(RhiConformance, BlendAndConstants) {
  Texture* color = Target();
  TextureView* cv = View(color);
  PipelineLayout* layout = Own(QuadLayout(device_.get()));
  GraphicsPipelineDesc alpha = QuadPipeline(layout);
  alpha.blend[0].enable = true;
  alpha.blend[0].src_color = BlendFactor::kSrcAlpha;
  alpha.blend[0].dst_color = BlendFactor::kOneMinusSrcAlpha;
  GraphicsPipelineDesc constant = alpha;
  constant.blend[0].src_color = BlendFactor::kConstantColor;
  constant.blend[0].dst_color = BlendFactor::kZero;
  constant.blend[0].src_alpha = BlendFactor::kConstantAlpha;
  Pipeline* alpha_pipe = Own(device_->CreateGraphicsPipeline(alpha));
  Pipeline* constant_pipe = Own(device_->CreateGraphicsPipeline(constant));
  ASSERT_NE(alpha_pipe, nullptr);
  ASSERT_NE(constant_pipe, nullptr);

  list_->Begin();
  TextureBarrier b{color, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(0, 0, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = cv;
  rp.colors[0].load = LoadOp::kClear;
  rp.colors[0].clear = {{0.0f, 0.0f, 1.0f, 1.0f}};
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  YUp(list_);
  list_->SetPipeline(alpha_pipe);
  DrawQuad(list_, -1, -1, 0, 1, {1, 0, 0, 0.25f});
  list_->SetPipeline(constant_pipe);
  const float k[4] = {0.5f, 0.25f, 1.0f, 0.5f};
  list_->SetBlendConstants(k);
  DrawQuad(list_, 0, -1, 1, 1, {1, 1, 1, 1});
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_TRUE(Near(px[32 * kW + 8], Rgba(64, 0, 191, 64)))
      << std::hex << px[32 * kW + 8];
  EXPECT_TRUE(Near(px[32 * kW + 56], Rgba(128, 64, 255, 128)))
      << std::hex << px[32 * kW + 56];
}

// Draws a fullscreen triangle sampling `view` with `fragment` into a fresh
// target and returns its centre texel.
u32 SampleCenter(Device* device,
                 CommandList* list,
                 std::vector<Object*>& objects,
                 const ShaderCode& fragment,
                 TextureView* view,
                 TextureState view_state,
                 const float (&coord)[4]) {
  auto own = [&](Object* o) {
    objects.push_back(o);
    return o;
  };
  SamplerDesc sd;
  sd.mag = sd.min = Filter::kNearest;
  auto* sampler = static_cast<Sampler*>(own(device->CreateSampler(sd)));
  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kSampledTexture, kStageFragment});
  auto* set0 =
      static_cast<BindGroupLayout*>(own(device->CreateBindGroupLayout(ld)));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  pl.push_constant_bytes = 16;
  pl.push_constant_stages = kStageFragment;
  auto* layout =
      static_cast<PipelineLayout*>(own(device->CreatePipelineLayout(pl)));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_fullscreen_vert_spv);
  gp.fragment = fragment;
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  auto* pipe = static_cast<Pipeline*>(own(device->CreateGraphicsPipeline(gp)));
  if (!pipe)
    return 0xdeadbeef;
  BindGroupDesc gd;
  gd.layout = set0;
  BindingWrite w;
  w.binding = 0;
  w.view = view;
  w.sampler = sampler;
  w.view_state = view_state;
  gd.writes.push_back(w);
  auto* group = static_cast<BindGroup*>(own(device->CreateBindGroup(gd)));
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.width = kW;
  td.height = kH;
  td.usage = kTextureColorTarget | kTextureCopySrc;
  auto* target = static_cast<Texture*>(own(device->CreateTexture(td)));
  TextureViewDesc vd;
  auto* tv = static_cast<TextureView*>(own(device->CreateView(target, vd)));
  BufferDesc bd;
  bd.size = 4;
  bd.usage = kBufferCopyDst;
  bd.memory = MemoryKind::kReadback;
  auto* rb = static_cast<Buffer*>(own(device->CreateBuffer(bd)));

  list->Begin();
  TextureBarrier b{target, TextureState::kUndefined,
                   TextureState::kColorTarget};
  list->Barrier(0, 0, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = tv;
  rp.colors[0].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list->BeginRenderPass(rp);
  list->SetPipeline(pipe);
  YUp(list);
  list->SetBindGroup(0, group);
  list->SetPushConstants(0, 16, coord);
  list->Draw(3, 1, 0, 0);
  list->EndRenderPass();
  TextureBarrier to_src{target, TextureState::kColorTarget,
                        TextureState::kCopySrc};
  list->Barrier(kAccessColorWrite, kAccessCopyRead, &to_src, 1);
  BufferTextureCopy c;
  c.region.x = kW / 2;
  c.region.y = kH / 2;
  list->CopyTextureToBuffer(rb, target, &c, 1);
  list->Barrier(kAccessCopyWrite, kAccessHostRead);
  list->End();
  device->Wait(device->Submit(list));
  u32 texel;
  std::memcpy(&texel, rb->mapped(), 4);
  return texel;
}

u8 SrgbToLinear(u8 c) {
  const double v = c / 255.0;
  const double l = v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
  return static_cast<u8>(l * 255.0 + 0.5);
}

TEST_P(RhiConformance, ViewSwizzleAndFormatAlias) {
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.usage = kTextureSampled | kTextureCopyDst | kTextureMutableFormat;
  Texture* tex = Own(device_->CreateTexture(td));
  const u32 texel = Rgba(64, 128, 192, 255);
  Buffer* staging = Upload(&texel, 4, 0);
  list_->Begin();
  TextureBarrier up{tex, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, &up, 1);
  BufferTextureCopy c;
  list_->CopyBufferToTexture(tex, staging, &c, 1);
  TextureBarrier rd{tex, TextureState::kCopyDst, TextureState::kShaderRead};
  list_->Barrier(0, 0, &rd, 1);
  list_->End();
  ASSERT_TRUE(device_->Wait(device_->Submit(list_)));

  TextureViewDesc plain;
  TextureViewDesc swizzled;
  swizzled.swizzle[0] = Swizzle::kB;
  swizzled.swizzle[2] = Swizzle::kR;
  swizzled.swizzle[3] = Swizzle::kOne;
  TextureViewDesc alias;
  alias.format = Format::kRGBA8Srgb;
  TextureView* views[3] = {Own(device_->CreateView(tex, plain)),
                           Own(device_->CreateView(tex, swizzled)),
                           Own(device_->CreateView(tex, alias))};
  u32 got[3];
  for (int i = 0; i < 3; i++)
    got[i] = SampleCenter(device_.get(), list_, objects_, Spv(k_tex_frag_spv),
                          views[i], TextureState::kShaderRead, {0, 0, 0, 0});
  EXPECT_EQ(got[0], texel);
  EXPECT_EQ(got[1], Rgba(192, 128, 64, 255));
  EXPECT_TRUE(Near(got[2], Rgba(SrgbToLinear(64), SrgbToLinear(128),
                                SrgbToLinear(192), 255)))
      << std::hex << got[2];
}

TEST_P(RhiConformance, TextureArrays3DAndCubes) {
  // Two layers, two slices, six faces: one texel each.
  TextureDesc ad;
  ad.format = Format::kRGBA8Unorm;
  ad.layers = 2;
  ad.usage = kTextureSampled | kTextureCopyDst;
  Texture* array = Own(device_->CreateTexture(ad));
  TextureDesc vd3 = ad;
  vd3.dim = TextureDim::k3D;
  vd3.layers = 1;
  vd3.depth = 2;
  Texture* volume = Own(device_->CreateTexture(vd3));
  TextureDesc cd = ad;
  cd.layers = 6;
  cd.usage |= kTextureCubeCompatible;
  Texture* cube = Own(device_->CreateTexture(cd));

  u32 texels[10];
  texels[0] = Rgba(255, 0, 0, 255);
  texels[1] = Rgba(0, 255, 0, 255);
  texels[2] = Rgba(0, 0, 255, 255);
  texels[3] = Rgba(255, 255, 255, 255);
  for (u32 f = 0; f < 6; f++)
    texels[4 + f] = Rgba(u8(f * 40), u8(255 - f * 40), u8(f * 20), 255);
  Buffer* staging = Upload(texels, sizeof(texels), 0);
  list_->Begin();
  TextureBarrier up[3] = {
      {array, TextureState::kUndefined, TextureState::kCopyDst},
      {volume, TextureState::kUndefined, TextureState::kCopyDst},
      {cube, TextureState::kUndefined, TextureState::kCopyDst}};
  up[0].range.layers = 2;
  up[2].range.layers = 6;
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, up, 3);
  BufferTextureCopy c;
  c.region.layers = 2;
  list_->CopyBufferToTexture(array, staging, &c, 1);
  c = {};
  c.buffer_offset = 8;
  c.region.depth = 2;
  list_->CopyBufferToTexture(volume, staging, &c, 1);
  c = {};
  c.buffer_offset = 16;
  c.region.layers = 6;
  list_->CopyBufferToTexture(cube, staging, &c, 1);
  TextureBarrier rd[3] = {
      {array, TextureState::kCopyDst, TextureState::kShaderRead},
      {volume, TextureState::kCopyDst, TextureState::kShaderRead},
      {cube, TextureState::kCopyDst, TextureState::kShaderRead}};
  rd[0].range.layers = 2;
  rd[2].range.layers = 6;
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead, rd, 3);
  list_->End();
  ASSERT_TRUE(device_->Wait(device_->Submit(list_)));

  TextureViewDesc layer1;
  layer1.base_layer = 1;
  TextureViewDesc arrayed;
  arrayed.dim = ViewDim::k2DArray;
  arrayed.layers = 2;
  TextureViewDesc v3d;
  v3d.dim = ViewDim::k3D;
  TextureViewDesc vcube;
  vcube.dim = ViewDim::kCube;
  vcube.layers = 6;
  TextureView* layer1_view = Own(device_->CreateView(array, layer1));
  TextureView* array_view = Own(device_->CreateView(array, arrayed));
  TextureView* volume_view = Own(device_->CreateView(volume, v3d));
  TextureView* cube_view = Own(device_->CreateView(cube, vcube));
  const TextureState s = TextureState::kShaderRead;
  auto sample = [&](const ShaderCode& fs, TextureView* v,
                    const float (&coord)[4]) {
    return SampleCenter(device_.get(), list_, objects_, fs, v, s, coord);
  };
  EXPECT_EQ(sample(Spv(k_tex_frag_spv), layer1_view, {0, 0, 0, 0}),
            texels[1])
      << "2D view of layer 1";
  EXPECT_EQ(sample(Spv(k_x_texarray_frag_spv), array_view, {0, 0, 0, 0}),
            texels[0]);
  EXPECT_EQ(sample(Spv(k_x_texarray_frag_spv), array_view, {1, 0, 0, 0}),
            texels[1]);
  EXPECT_EQ(sample(Spv(k_x_tex3d_frag_spv), volume_view, {0.25f, 0, 0, 0}),
            texels[2]);
  EXPECT_EQ(sample(Spv(k_x_tex3d_frag_spv), volume_view, {0.75f, 0, 0, 0}),
            texels[3]);
  EXPECT_EQ(sample(Spv(k_x_cube_frag_spv), cube_view, {-1, 0, 0, 0}),
            texels[4 + 1])
      << "-X";
  EXPECT_EQ(sample(Spv(k_x_cube_frag_spv), cube_view, {0, 1, 0, 0}),
            texels[4 + 2])
      << "+Y";
  EXPECT_EQ(sample(Spv(k_x_cube_frag_spv), cube_view, {0, 0, -1, 0}),
            texels[4 + 5])
      << "-Z";
}

TEST_P(RhiConformance, FillUpdateAndCopyBuffer) {
  BufferDesc bd;
  bd.size = 1024;
  bd.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  Buffer* rb = Readback(1024);
  list_->Begin();
  list_->FillBuffer(buf, 0, 1024, 0x01020304u);
  list_->Barrier(kAccessCopyWrite, kAccessCopyWrite);
  const u32 update[2] = {0xaaaaaaaau, 0xbbbbbbbbu};
  list_->UpdateBuffer(buf, 16, sizeof(update), update);
  list_->Barrier(kAccessCopyWrite, kAccessCopyWrite);
  list_->FillBuffer(buf, 512, 64, 0x55555555u);
  list_->Barrier(kAccessCopyWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, buf, 0, 1024);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* out = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(out[0], 0x01020304u);
  EXPECT_EQ(out[4], 0xaaaaaaaau);
  EXPECT_EQ(out[5], 0xbbbbbbbbu);
  EXPECT_EQ(out[6], 0x01020304u);
  EXPECT_EQ(out[128], 0x55555555u);
  EXPECT_EQ(out[143], 0x55555555u);
  EXPECT_EQ(out[144], 0x01020304u);
  EXPECT_EQ(out[255], 0x01020304u);
}

TEST_P(RhiConformance, Timestamps) {
  if (!device_->caps().timestamps)
    GTEST_SKIP() << "no timestamps";
  TimestampPool* pool = Own(device_->CreateTimestampPool(2));
  BufferDesc bd;
  bd.size = 1 << 20;
  bd.usage = kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  list_->Begin();
  list_->ResetTimestamps(pool, 0, 2);
  list_->WriteTimestamp(pool, 0);
  list_->FillBuffer(buf, 0, bd.size, 7);
  list_->WriteTimestamp(pool, 1);
  Submit();
  list_->End();
  u64 t[2] = {};
  ASSERT_TRUE(device_->ReadTimestamps(pool, 0, 2, t));
  EXPECT_GE(t[1], t[0]);
  EXPECT_LT(double(t[1] - t[0]) * device_->caps().timestamp_period_ns, 1e9);
}

// gl_VertexIndex counts from the first vertex (or adds the vertex offset of
// an indexed draw) and gl_InstanceIndex from the first instance.
TEST_P(RhiConformance, DrawParameters) {
  PipelineLayoutDesc pl;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_x_drawid_vert_spv);
  gp.fragment = Spv(k_x_color_frag_spv);
  gp.topology = Topology::kPointList;
  gp.vertex_buffers = {{4, false}, {4, true}};
  gp.vertex_attributes = {{0, 0, Format::kR32Float, 0},
                          {1, 1, Format::kR32Float, 0}};
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  float slots[16];
  for (int i = 0; i < 16; i++)
    slots[i] = static_cast<float>(i);
  const float instance_slots[4] = {0, 10, 20, 30};
  const u16 indices[3] = {0, 1, 2};
  Buffer* vb = Upload(slots, sizeof(slots), kBufferVertex);
  Buffer* ib = Upload(instance_slots, sizeof(instance_slots), kBufferVertex);
  Buffer* index = Upload(indices, sizeof(indices), kBufferIndex);
  Texture* color = Target();
  TextureView* cv = View(color);

  list_->Begin();
  TextureBarrier b{color, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead | kAccessIndexRead, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = cv;
  rp.colors[0].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  YUp(list_);
  Buffer* vbs[2] = {vb, ib};
  const u64 offsets[2] = {0, 0};
  list_->SetVertexBuffers(0, 2, vbs, offsets);
  list_->Draw(3, 2, 2, 1);
  list_->SetIndexBuffer(index, 0, IndexType::kUint16);
  list_->DrawIndexed(3, 1, 0, 5, 3);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[12], Rgba(2, 1, 0, 255));
  EXPECT_EQ(px[14], Rgba(4, 1, 0, 255));
  EXPECT_EQ(px[22], Rgba(2, 2, 0, 255));
  EXPECT_EQ(px[24], Rgba(4, 2, 0, 255));
  EXPECT_EQ(px[35], Rgba(5, 3, 0, 255)) << "indexed, vertex offset 5";
  EXPECT_EQ(px[37], Rgba(7, 3, 0, 255));
}

// A counter-clockwise quad in y-up clip space faces front through a y-up
// viewport and back through a y-down one, which also puts +y at the bottom.
TEST_P(RhiConformance, CullingAndViewportDirection) {
  PipelineLayout* layout = Own(QuadLayout(device_.get()));
  GraphicsPipelineDesc none = QuadPipeline(layout);
  GraphicsPipelineDesc back = none;
  back.cull = CullMode::kBack;
  back.front_ccw = true;
  Pipeline* none_pipe = Own(device_->CreateGraphicsPipeline(none));
  Pipeline* back_pipe = Own(device_->CreateGraphicsPipeline(back));
  ASSERT_NE(none_pipe, nullptr);
  ASSERT_NE(back_pipe, nullptr);
  Texture* t[3] = {Target(), Target(), Target()};
  std::vector<u32> px[3];
  for (int i = 0; i < 3; i++) {
    TextureView* v = View(t[i]);
    list_->Begin();
    TextureBarrier b{t[i], TextureState::kUndefined,
                     TextureState::kColorTarget};
    list_->Barrier(0, 0, &b, 1);
    RenderPassDesc rp;
    rp.color_count = 1;
    rp.colors[0].view = v;
    rp.colors[0].load = LoadOp::kClear;
    rp.width = kW;
    rp.height = kH;
    list_->BeginRenderPass(rp);
    list_->SetPipeline(i == 0 ? none_pipe : back_pipe);
    if (i == 1)
      YUp(list_);
    else
      list_->SetViewport(0, 0, kW, kH, 0, 1);
    list_->SetScissor(0, 0, kW, kH);
    if (i == 0)
      DrawQuad(list_, -1, 0, 1, 1, kRed);
    else
      DrawQuad(list_, -1, -1, 1, 1, kGreen);
    list_->EndRenderPass();
    px[i] = ReadPixels(t[i], TextureState::kColorTarget);
    list_->End();
  }
  EXPECT_EQ(px[0][60 * kW + 4], Rgba(255, 0, 0, 255)) << "y-down: +y at bottom";
  EXPECT_EQ(px[0][4 * kW + 4], 0u);
  EXPECT_EQ(px[1][32 * kW + 32], Rgba(0, 255, 0, 255)) << "y-up: front face";
  EXPECT_EQ(px[2][32 * kW + 32], 0u) << "y-down: back face culled";
}

TEST_P(RhiConformance, GeometryShaderOrientation) {
  if (!device_->caps().geometry_shader)
    GTEST_SKIP() << "no geometry shaders";
  Texture* t = Target();
  TextureView* v = View(t);
  BindGroupLayoutDesc empty;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(empty));
  BindGroupLayoutDesc ubo_desc;
  ubo_desc.bindings.push_back({0, BindingType::kUniformBuffer, kStageVertex});
  BindGroupLayout* set1 = Own(device_->CreateBindGroupLayout(ubo_desc));
  PipelineLayoutDesc pl;
  pl.groups = {set0, set1};
  pl.push_constant_bytes = 16;
  pl.push_constant_stages = kStageVertex;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_tri_vert_spv);
  gp.geometry = Spv(k_x_passthrough_geom_spv);
  gp.fragment = Spv(k_x_color_frag_spv);
  gp.vertex_buffers = {{24, false}};
  gp.vertex_attributes = {{0, 0, Format::kRG32Float, 0},
                          {1, 0, Format::kRGBA32Float, 8}};
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  const float verts[] = {-1.0f, 0.0f, 1, 0, 0, 1, 3.0f, 0.0f, 1, 0, 0, 1,
                         -1.0f, 2.0f, 1, 0, 0, 1};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  const float tint[4] = {1, 1, 1, 1};
  Buffer* ub = Upload(tint, sizeof(tint), kBufferUniform);
  BindGroupDesc gd0;
  gd0.layout = set0;
  BindGroup* group0 = Own(device_->CreateBindGroup(gd0));
  BindGroupDesc gd;
  gd.layout = set1;
  BindingWrite w;
  w.binding = 0;
  w.buffer = ub;
  w.range = 16;
  gd.writes.push_back(w);
  BindGroup* group1 = Own(device_->CreateBindGroup(gd));

  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead | kAccessUniformRead, &b,
                 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = v;
  rp.colors[0].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  YUp(list_);
  const float push[4] = {0, 0, 0.5f, 0};
  list_->SetPushConstants(0, 16, push);
  list_->SetBindGroup(0, group0);
  list_->SetBindGroup(1, group1);
  u64 zero = 0;
  list_->SetVertexBuffers(0, 1, &vb, &zero);
  list_->Draw(3, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[4 * kW + 4], Rgba(0, 0, 255, 255)) << "top row, swizzled";
  EXPECT_EQ(px[(kH - 4) * kW + 4], 0u) << "bottom row covered";
}

// Tightly packed and oddly pitched buffer layouts, at offsets no copy
// engine aligns for.
TEST_P(RhiConformance, UnalignedTextureCopies) {
  constexpr u32 w = 5, h = 3;
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.width = w;
  td.height = h;
  td.usage = kTextureCopySrc | kTextureCopyDst;
  Texture* tex = Own(device_->CreateTexture(td));
  u32 data[1 + w * h];
  data[0] = 0;
  for (u32 y = 0; y < h; y++)
    for (u32 x = 0; x < w; x++)
      data[1 + y * w + x] = Rgba(u8(x * 10), u8(y * 10), 7, 255);
  Buffer* staging = Upload(data, sizeof(data), 0);
  Buffer* tight = Readback(w * h * 4);
  Buffer* pitched = Readback(512);
  std::memset(pitched->mapped(), 0, 512);
  list_->Begin();
  TextureBarrier up{tex, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, &up, 1);
  BufferTextureCopy c;
  c.buffer_offset = 4;
  c.region.width = w;
  c.region.height = h;
  list_->CopyBufferToTexture(tex, staging, &c, 1);
  TextureBarrier src{tex, TextureState::kCopyDst, TextureState::kCopySrc};
  list_->Barrier(kAccessCopyWrite, kAccessCopyRead, &src, 1);
  c.buffer_offset = 0;
  list_->CopyTextureToBuffer(tight, tex, &c, 1);
  BufferTextureCopy sub;
  sub.buffer_offset = 12;
  sub.row_length = 7;
  sub.region.x = 1;
  sub.region.y = 1;
  sub.region.width = 3;
  sub.region.height = 2;
  list_->CopyTextureToBuffer(pitched, tex, &sub, 1);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  EXPECT_EQ(std::memcmp(tight->mapped(), data + 1, w * h * 4), 0);
  const u32* p = reinterpret_cast<const u32*>(pitched->mapped() + 12);
  for (u32 r = 0; r < 2; r++)
    for (u32 x = 0; x < 3; x++)
      EXPECT_EQ(p[r * 7 + x], data[1 + (1 + r) * w + 1 + x])
          << "row " << r << " col " << x;
  EXPECT_EQ(p[3], 0u) << "past the copied row";
}

TEST_P(RhiConformance, StorageBindingsAndSubgroups) {
  constexpr u32 kN = 128;
  std::vector<u32> input(kN);
  for (u32 i = 0; i < kN; i++)
    input[i] = i + 1;
  BufferDesc bd;
  bd.size = 256 + kN * 4;
  bd.usage = kBufferStorage | kBufferCopyDst;
  Buffer* src = Own(device_->CreateBuffer(bd));
  bd.size = 1024;
  bd.usage = kBufferStorage | kBufferCopySrc;
  Buffer* dst = Own(device_->CreateBuffer(bd));
  Buffer* staging = Upload(input.data(), kN * 4, 0);
  std::vector<u32> ubo(64, 0);
  ubo[0] = 3;
  Buffer* ub = Upload(ubo.data(), 256, kBufferUniform);
  Buffer* rb = Readback(1024);

  BindGroupLayoutDesc ld;
  ld.bindings.push_back({2, BindingType::kUniformBuffer, kStageCompute});
  ld.bindings.push_back(
      {0, BindingType::kStorageBufferDynamic, kStageCompute, true});
  ld.bindings.push_back({1, BindingType::kStorageBufferDynamic, kStageCompute});
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  ComputePipelineDesc cp;
  cp.layout = layout;
  cp.code = Spv(k_x_storage_comp_spv);
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);
  BindGroupDesc gd;
  gd.layout = set0;
  BindingWrite w;
  w.binding = 0;
  w.buffer = src;
  w.range = kN * 4;
  gd.writes.push_back(w);
  w.binding = 1;
  w.buffer = dst;
  w.range = 768;
  gd.writes.push_back(w);
  w.binding = 2;
  w.buffer = ub;
  w.range = 16;
  gd.writes.push_back(w);
  BindGroup* group = Own(device_->CreateBindGroup(gd));

  list_->Begin();
  list_->Barrier(kAccessHostWrite, kAccessCopyRead);
  list_->CopyBuffer(src, 256, staging, 0, kN * 4);
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead);
  list_->SetPipeline(pipe);
  const u32 offsets[2] = {256, 0};  // binding 0, binding 1
  list_->SetBindGroup(0, group, offsets, 2);
  list_->Dispatch(kN / 64, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, dst, 0, 1024);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* out = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(out[0], 1u * 3 + 2);
  EXPECT_EQ(out[kN - 1], kN * 3 + 2);
  EXPECT_EQ(out[128], kN) << "runtime array length";
  EXPECT_EQ(out[129], device_->caps().subgroup_size) << "subgroup size";
}

// The fragment shader reads one of three vertex outputs.
TEST_P(RhiConformance, VaryingSubset) {
  PipelineLayoutDesc pl;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_x_varyings_vert_spv);
  gp.fragment = Spv(k_x_varyings_frag_spv);
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(0, 0, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = v;
  rp.colors[0].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  YUp(list_);
  list_->Draw(3, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 32], Rgba(0, 255, 0, 255));
}

TEST_P(RhiConformance, ScaledVertexFormatAndFan) {
  PipelineLayoutDesc pl;
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  GraphicsPipelineDesc gp;
  gp.layout = layout;
  gp.vertex = Spv(k_x_scaled_vert_spv);
  gp.fragment = Spv(k_x_color_frag_spv);
  gp.topology = Topology::kTriangleFan;
  gp.vertex_buffers = {{12, false}};
  gp.vertex_attributes = {{0, 0, Format::kRG32Float, 0},
                          {1, 0, Format::kRGBA8Uscaled, 8}};
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  struct Vertex {
    float x, y;
    u8 rgba[4];
  };
  const Vertex verts[4] = {{-1, -1, {0, 255, 0, 255}},
                           {1, -1, {0, 255, 0, 255}},
                           {1, 1, {0, 255, 0, 255}},
                           {-1, 1, {0, 255, 0, 255}}};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead, &b, 1);
  RenderPassDesc rp;
  rp.color_count = 1;
  rp.colors[0].view = v;
  rp.colors[0].load = LoadOp::kClear;
  rp.width = kW;
  rp.height = kH;
  list_->BeginRenderPass(rp);
  list_->SetPipeline(pipe);
  YUp(list_);
  u64 zero = 0;
  list_->SetVertexBuffers(0, 1, &vb, &zero);
  list_->Draw(4, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[2 * kW + 2], Rgba(0, 255, 0, 255));
  EXPECT_EQ(px[(kH - 3) * kW + kW - 3], Rgba(0, 255, 0, 255));
  EXPECT_EQ(px[2 * kW + kW - 3], Rgba(0, 255, 0, 255));
}

TEST_P(RhiConformance, DispatchBase) {
  if (!device_->caps().dispatch_base)
    GTEST_SKIP() << "no dispatch base";
  BufferDesc bd;
  bd.size = 64;
  bd.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  Buffer* rb = Readback(64);
  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kStorageBuffer, kStageCompute});
  ld.push = true;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  PipelineLayout* layout = Own(device_->CreatePipelineLayout(pl));
  ComputePipelineDesc cp;
  cp.layout = layout;
  cp.code = Spv(k_x_base_comp_spv);
  cp.dispatch_base = true;
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);
  list_->Begin();
  list_->FillBuffer(buf, 0, 64, 0);
  list_->Barrier(kAccessCopyWrite, kAccessShaderWrite);
  list_->SetPipeline(pipe);
  BindingWrite w;
  w.binding = 0;
  w.buffer = buf;
  w.range = 64;
  list_->PushBindGroup(0, set0, &w, 1);
  list_->DispatchBase(2, 0, 0, 1, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, buf, 0, 64);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* out = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(out[1], 0u) << "workgroup 0 ran";
  EXPECT_EQ(out[8], 2008u);
  EXPECT_EQ(out[11], 2011u);
  EXPECT_EQ(out[12], 0u);
}

}  // namespace
