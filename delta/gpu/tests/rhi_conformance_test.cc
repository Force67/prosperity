// Runs the same work through every rhi backend the build has and checks the
// pixels against fixed expectations, so a difference is a backend bug rather
// than a difference of opinion between two backends. Needs a GPU; a backend
// that cannot create a device is skipped.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "base/arch.h"
#include "gpu/render/backend.h"
#include "gpu/rhi/device.h"
#include "gpu/tests/rhi_conformance_spv.h"

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
    device_ = gpu::render::CreateBackendDevice(GetParam());
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
