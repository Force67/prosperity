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

  // pos.vert: per-vertex position and RGBA8 colour, per-instance offset.
  struct Vertex {
    float x, y, z;
    u32 color;
  };

  PipelineLayout* EmptyLayout() {
    return Own(device_->CreatePipelineLayout(PipelineLayoutDesc{}));
  }

  GraphicsPipelineDesc PosPipeline(PipelineLayout* layout) {
    GraphicsPipelineDesc gp;
    gp.layout = layout;
    gp.vertex = Spv(k_pos_vert_spv);
    gp.fragment = Spv(k_color_frag_spv);
    gp.vertex_buffers = {{sizeof(Vertex), false}, {4, true}};
    gp.vertex_attributes = {{0, 0, Format::kRGB32Float, 0},
                            {1, 0, Format::kRGBA8Unorm, 12},
                            {2, 1, Format::kRG16Sscaled, 0}};
    gp.color_count = 1;
    gp.color_formats[0] = Format::kRGBA8Unorm;
    return gp;
  }

  void BindVertices(Buffer* vertices, Buffer* instances) {
    Buffer* buffers[2] = {vertices, instances};
    const u64 offsets[2] = {};
    list_->SetVertexBuffers(0, 2, buffers, offsets);
  }

  // Offsets in whole clip-space units, two per instance.
  Buffer* Instances(std::vector<i16> offsets) {
    return Upload(offsets.data(), offsets.size() * 2, kBufferVertex);
  }

  void YUpViewport() {
    list_->SetViewport(0, kH, kW, -static_cast<float>(kH), 0, 1);
    list_->SetScissor(0, 0, kW, kH);
  }

  RenderPassDesc ClearPass(TextureView* view) {
    RenderPassDesc rp;
    rp.color_count = 1;
    rp.colors[0].view = view;
    rp.colors[0].load = LoadOp::kClear;
    rp.width = kW;
    rp.height = kH;
    return rp;
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

// Twenty storage buffers in one stage, windows of one buffer: past the
// per-stage limit of some GL drivers, with an atomic and a runtime length
// among the ones over it.
TEST_P(RhiConformance, ManyStorageBuffers) {
  constexpr u32 kBuffers = 20, kWindow = 256;
  BufferDesc bd;
  bd.size = kBuffers * kWindow;
  bd.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  Buffer* rb = Readback(bd.size);
  BindGroupLayoutDesc ld;
  for (u32 i = 0; i < kBuffers; i++)
    ld.bindings.push_back({i, BindingType::kStorageBuffer, kStageCompute});
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  ComputePipelineDesc cp;
  cp.layout = Own(device_->CreatePipelineLayout(pl));
  cp.code = Spv(k_many_buffers_comp_spv);
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);
  BindGroupDesc gd;
  gd.layout = set0;
  for (u32 i = 0; i < kBuffers; i++) {
    BindingWrite w;
    w.binding = i;
    w.buffer = buf;
    w.offset = i * kWindow;
    w.range = kWindow;
    gd.writes.push_back(w);
  }
  BindGroup* group = Own(device_->CreateBindGroup(gd));

  list_->Begin();
  for (u32 i = 0; i < kBuffers; i++)
    list_->FillBuffer(buf, i * kWindow, kWindow, i);
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead | kAccessShaderWrite);
  list_->SetPipeline(pipe);
  list_->SetBindGroup(0, group);
  list_->Dispatch(1, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, buf, 0, bd.size);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* words = reinterpret_cast<const u32*>(rb->mapped());
  constexpr u32 kPerWindow = kWindow / 4;
  EXPECT_EQ(words[19 * kPerWindow + 5], 136u) << "sum of windows 0..16";
  EXPECT_EQ(words[18 * kPerWindow], 18u + 64) << "atomic";
  EXPECT_EQ(words[17 * kPerWindow], kPerWindow) << "runtime length";
  EXPECT_EQ(words[17 * kPerWindow + 1], 17u) << "untouched";
}

// Workgroup ids start at the base; a plain dispatch of the same pipeline
// starts at zero again.
TEST_P(RhiConformance, DispatchBase) {
  if (!device_->caps().dispatch_base)
    GTEST_SKIP() << "no dispatch base";
  constexpr u32 kN = 16;
  BufferDesc bd;
  bd.size = kN * 4;
  bd.usage = kBufferStorage | kBufferCopySrc | kBufferCopyDst;
  Buffer* buf = Own(device_->CreateBuffer(bd));
  Buffer* rb = Readback(kN * 4);
  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kStorageBuffer, kStageCompute});
  ld.push = true;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  ComputePipelineDesc cp;
  cp.layout = Own(device_->CreatePipelineLayout(pl));
  cp.code = Spv(k_dispatch_base_comp_spv);
  cp.dispatch_base = true;
  Pipeline* pipe = Own(device_->CreateComputePipeline(cp));
  ASSERT_NE(pipe, nullptr);

  list_->Begin();
  list_->FillBuffer(buf, 0, kN * 4, ~0u);
  list_->Barrier(kAccessCopyWrite, kAccessShaderWrite);
  list_->SetPipeline(pipe);
  BindingWrite w;
  w.buffer = buf;
  w.range = kN * 4;
  list_->PushBindGroup(0, set0, &w, 1);
  list_->DispatchBase(2, 0, 0, 1, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessShaderWrite);
  list_->Dispatch(1, 1, 1);
  list_->Barrier(kAccessShaderWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, buf, 0, kN * 4);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();
  const u32* v = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(v[0], 0u) << "plain dispatch";
  EXPECT_EQ(v[3], 3u);
  EXPECT_EQ(v[4], ~0u) << "workgroup 1 never ran";
  EXPECT_EQ(v[8], 200u) << "based dispatch";
  EXPECT_EQ(v[11], 203u);
  EXPECT_EQ(v[12], ~0u);
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

// Culling follows the winding in framebuffer space, so flipping the viewport
// flips which of two opposite triangles survives.
TEST_P(RhiConformance, CullWindingFollowsViewport) {
  PipelineLayout* layout = EmptyLayout();
  GraphicsPipelineDesc gp = PosPipeline(layout);
  gp.cull = CullMode::kBack;
  gp.front_ccw = true;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  const u32 red = Rgba(255, 0, 0, 255);
  // Left: counter-clockwise in clip space (y up). Right: clockwise.
  const Vertex verts[] = {
      {-0.9f, -0.5f, 0, red}, {-0.1f, -0.5f, 0, red}, {-0.5f, 0.5f, 0, red},
      {0.1f, -0.5f, 0, red},  {0.5f, 0.5f, 0, red},   {0.9f, -0.5f, 0, red}};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  Buffer* inst = Instances({0, 0});
  Texture* up = Target();
  Texture* down = Target();
  TextureView* up_view = View(up);
  TextureView* down_view = View(down);

  list_->Begin();
  TextureBarrier b[2] = {
      {up, TextureState::kUndefined, TextureState::kColorTarget},
      {down, TextureState::kUndefined, TextureState::kColorTarget}};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead, b, 2);
  for (int pass = 0; pass < 2; pass++) {
    list_->BeginRenderPass(ClearPass(pass ? down_view : up_view));
    list_->SetPipeline(pipe);
    if (pass)
      list_->SetViewport(0, 0, kW, kH, 0, 1);
    else
      list_->SetViewport(0, kH, kW, -static_cast<float>(kH), 0, 1);
    list_->SetScissor(0, 0, kW, kH);
    BindVertices(vb, inst);
    list_->Draw(6, 1, 0, 0);
    list_->EndRenderPass();
  }
  std::vector<u32> pu = ReadPixels(up, TextureState::kColorTarget);
  std::vector<u32> pd = ReadPixels(down, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(pu[32 * kW + 16], red) << "y-up: front face culled";
  EXPECT_EQ(pu[32 * kW + 48], 0u) << "y-up: back face drawn";
  EXPECT_EQ(pd[32 * kW + 16], 0u) << "y-down: back face drawn";
  EXPECT_EQ(pd[32 * kW + 48], red) << "y-down: front face culled";
}

// Depth test and write, stencil replace then equal, additive blending and an
// indexed draw with a vertex offset, then both depth planes read back.
TEST_P(RhiConformance, DepthStencilBlendIndexed) {
  constexpr Format kDs = Format::kD32FloatS8Uint;
  if (!device_->SupportsFormat(kDs, kTextureDepthTarget | kTextureCopySrc))
    GTEST_SKIP() << "no D32S8 target";
  Texture* color = Target();
  TextureView* color_view = View(color);
  TextureDesc dd;
  dd.format = kDs;
  dd.width = kW;
  dd.height = kH;
  dd.usage = kTextureDepthTarget | kTextureCopySrc;
  Texture* depth = Own(device_->CreateTexture(dd));
  TextureViewDesc dv;
  dv.format = kDs;
  dv.aspect = kAspectDepth | kAspectStencil;
  TextureView* depth_view = Own(device_->CreateView(depth, dv));

  PipelineLayout* layout = EmptyLayout();
  GraphicsPipelineDesc base = PosPipeline(layout);
  base.depth_format = base.stencil_format = kDs;
  GraphicsPipelineDesc g1 = base;
  g1.depth_test = g1.depth_write = true;
  g1.depth_compare = CompareOp::kLess;
  g1.stencil_test = true;
  g1.stencil_front.pass = StencilOp::kReplace;
  g1.stencil_front.reference = 3;
  g1.stencil_back = g1.stencil_front;
  GraphicsPipelineDesc g2 = base;
  g2.depth_test = g2.depth_write = true;
  g2.depth_compare = CompareOp::kLess;
  GraphicsPipelineDesc g3 = base;
  g3.stencil_test = true;
  g3.stencil_front.compare = CompareOp::kEqual;
  g3.stencil_front.reference = 3;
  g3.stencil_back = g3.stencil_front;
  g3.blend[0].enable = true;
  g3.blend[0].dst_color = g3.blend[0].dst_alpha = BlendFactor::kOne;
  Pipeline* p1 = Own(device_->CreateGraphicsPipeline(g1));
  Pipeline* p2 = Own(device_->CreateGraphicsPipeline(g2));
  Pipeline* p3 = Own(device_->CreateGraphicsPipeline(g3));
  ASSERT_TRUE(p1 && p2 && p3);

  const u32 red = Rgba(255, 0, 0, 255), green = Rgba(0, 255, 0, 255),
            blue = Rgba(0, 0, 255, 255);
  const Vertex verts[] = {
      // Skipped by the vertex offset.
      {9, 9, 0, 0}, {9, 9, 0, 0}, {9, 9, 0, 0}, {9, 9, 0, 0},
      // Left half at z 0.5.
      {-1, -1, 0.5f, red}, {0, -1, 0.5f, red}, {0, 1, 0.5f, red},
      {-1, 1, 0.5f, red},
      // Full screen at z 0.8, then at z 0.1.
      {-1, -1, 0.8f, green}, {3, -1, 0.8f, green}, {-1, 3, 0.8f, green},
      {-1, -1, 0.1f, blue}, {3, -1, 0.1f, blue}, {-1, 3, 0.1f, blue}};
  const u16 indices[] = {0, 1, 2, 0, 2, 3};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  Buffer* ib = Upload(indices, sizeof(indices), kBufferIndex);
  Buffer* inst = Instances({0, 0});

  list_->Begin();
  TextureBarrier b[2] = {
      {color, TextureState::kUndefined, TextureState::kColorTarget},
      {depth, TextureState::kUndefined, TextureState::kDepthTarget}};
  b[1].range.aspect = kAspectDepth | kAspectStencil;
  list_->Barrier(kAccessHostWrite, kAccessVertexRead | kAccessIndexRead, b, 2);
  RenderPassDesc rp = ClearPass(color_view);
  rp.depth.view = depth_view;
  rp.depth.stencil = true;
  rp.depth.depth_load = rp.depth.stencil_load = LoadOp::kClear;
  rp.depth.clear_depth = 1.0f;
  rp.depth.clear_stencil = 0;
  list_->BeginRenderPass(rp);
  YUpViewport();
  BindVertices(vb, inst);
  list_->SetIndexBuffer(ib, 0, IndexType::kUint16);
  list_->SetPipeline(p1);
  list_->DrawIndexed(6, 1, 0, 4, 0);
  list_->SetPipeline(p2);
  list_->Draw(3, 1, 8, 0);
  list_->SetPipeline(p3);
  list_->Draw(3, 1, 11, 0);
  list_->EndRenderPass();

  Buffer* depth_rb = Readback(kW * kH * 5);
  TextureBarrier to_src{depth, TextureState::kDepthTarget,
                        TextureState::kCopySrc};
  to_src.range.aspect = kAspectDepth | kAspectStencil;
  list_->Barrier(kAccessDepthWrite, kAccessCopyRead, &to_src, 1);
  BufferTextureCopy copies[2];
  copies[0].region.aspect = kAspectDepth;
  copies[0].region.width = kW;
  copies[0].region.height = kH;
  copies[1] = copies[0];
  copies[1].region.aspect = kAspectStencil;
  copies[1].buffer_offset = kW * kH * 4;
  list_->CopyTextureToBuffer(depth_rb, depth, &copies[0], 1);
  list_->CopyTextureToBuffer(depth_rb, depth, &copies[1], 1);
  std::vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 8], Rgba(255, 0, 255, 255)) << "left";
  EXPECT_EQ(px[32 * kW + 56], green) << "right";
  const float* d = reinterpret_cast<const float*>(depth_rb->mapped());
  const u8* st = depth_rb->mapped() + kW * kH * 4;
  EXPECT_FLOAT_EQ(d[32 * kW + 8], 0.5f);
  EXPECT_FLOAT_EQ(d[32 * kW + 56], 0.8f);
  EXPECT_EQ(st[32 * kW + 8], 3);
  EXPECT_EQ(st[32 * kW + 56], 0);
}

// first_instance selects the per-instance data, and a scaled attribute
// converts its integers to float without normalizing.
TEST_P(RhiConformance, InstancingAndScaledAttributes) {
  PipelineLayout* layout = EmptyLayout();
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(PosPipeline(layout)));
  ASSERT_NE(pipe, nullptr);
  const u32 white = Rgba(255, 255, 255, 255);
  const Vertex verts[] = {{0.3f, 0.3f, 0, white},
                          {0.7f, 0.3f, 0, white},
                          {0.5f, 0.7f, 0, white}};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  Buffer* inst = Instances({0, 0, -1, 0, 0, -1});
  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead, &b, 1);
  list_->BeginRenderPass(ClearPass(v));
  list_->SetPipeline(pipe);
  YUpViewport();
  BindVertices(vb, inst);
  list_->Draw(3, 2, 0, 1);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[17 * kW + 48], 0u) << "instance 0 drawn";
  EXPECT_EQ(px[17 * kW + 16], white) << "instance 1";
  EXPECT_EQ(px[49 * kW + 48], white) << "instance 2";
}

// Sparse attribute locations beyond 16, and an attribute the shader does not
// read.
TEST_P(RhiConformance, HighAttributeLocations) {
  GraphicsPipelineDesc gp = PosPipeline(EmptyLayout());
  gp.vertex = Spv(k_high_locations_vert_spv);
  gp.vertex_attributes = {{20, 0, Format::kRGB32Float, 0},
                          {27, 0, Format::kRGBA8Unorm, 12},
                          {3, 1, Format::kRG16Sscaled, 0}};
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  const u32 cyan = Rgba(0, 255, 255, 255);
  const Vertex verts[] = {
      {-1, -1, 0, cyan}, {3, -1, 0, cyan}, {-1, 3, 0, cyan}};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  Buffer* inst = Instances({0, 0});
  Texture* t = Target();
  TextureView* v = View(t);
  list_->Begin();
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead, &b, 1);
  list_->BeginRenderPass(ClearPass(v));
  list_->SetPipeline(pipe);
  YUpViewport();
  BindVertices(vb, inst);
  list_->Draw(3, 1, 0, 0);
  list_->EndRenderPass();
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 32], cyan);
}

// A swizzled single-layer view of an array texture, BGRA memory order, and
// rendering into one slice of a 3D texture through a 2D view.
TEST_P(RhiConformance, ViewsSwizzleBgraAnd3DSlice) {
  TextureDesc ad;
  ad.format = Format::kRGBA8Unorm;
  ad.width = ad.height = 2;
  ad.layers = 2;
  ad.usage = kTextureSampled | kTextureCopyDst | kTextureMutableFormat;
  Texture* arr = Own(device_->CreateTexture(ad));
  TextureViewDesc lv;
  lv.dim = ViewDim::k2D;
  lv.format = Format::kRGBA8Unorm;
  lv.base_layer = 1;
  lv.swizzle[0] = Swizzle::kB;
  lv.swizzle[1] = Swizzle::kG;
  lv.swizzle[2] = Swizzle::kR;
  lv.swizzle[3] = Swizzle::kOne;
  TextureView* layer_view = Own(device_->CreateView(arr, lv));
  TextureDesc bd = ad;
  bd.format = Format::kBGRA8Unorm;
  bd.layers = 1;
  bd.usage = kTextureSampled | kTextureCopyDst;
  Texture* bgra = Own(device_->CreateTexture(bd));
  TextureView* bgra_view = View(bgra);
  TextureDesc cd = ad;
  cd.width = cd.height = 1;
  cd.layers = 6;
  cd.usage = kTextureSampled | kTextureCubeCompatible;
  Texture* cube = Own(device_->CreateTexture(cd));
  TextureViewDesc cv;
  cv.dim = ViewDim::kCube;
  cv.layers = 6;
  EXPECT_NE(Own(device_->CreateView(cube, cv)), nullptr) << "cube view";

  u32 texels[8] = {};
  texels[4] = Rgba(10, 20, 30, 40);  // layer 1, top-left
  const u8 bgra_bytes[16] = {1, 2, 3, 4};
  Buffer* staging = Upload(texels, sizeof(texels), 0);
  Buffer* bgra_staging = Upload(bgra_bytes, sizeof(bgra_bytes), 0);

  TextureDesc vd;
  vd.dim = TextureDim::k3D;
  vd.format = Format::kRGBA8Unorm;
  vd.width = kW;
  vd.height = kH;
  vd.depth = 2;
  vd.usage = kTextureColorTarget | kTextureCopySrc | kTextureCopyDst |
             kTextureArrayCompatible;
  Texture* vol = Own(device_->CreateTexture(vd));
  TextureViewDesc sv;
  sv.dim = ViewDim::k2D;
  sv.format = Format::kRGBA8Unorm;
  sv.base_layer = 1;
  TextureView* slice = Own(device_->CreateView(vol, sv));
  ASSERT_NE(slice, nullptr);

  SamplerDesc sd;
  sd.mag = sd.min = Filter::kNearest;
  Sampler* sampler = Own(device_->CreateSampler(sd));
  BindGroupLayoutDesc ld;
  ld.bindings.push_back({0, BindingType::kSampledTexture, kStageFragment});
  ld.push = true;
  BindGroupLayout* set0 = Own(device_->CreateBindGroupLayout(ld));
  PipelineLayoutDesc pl;
  pl.groups = {set0};
  GraphicsPipelineDesc gp;
  gp.layout = Own(device_->CreatePipelineLayout(pl));
  gp.vertex = Spv(k_fullscreen_vert_spv);
  gp.fragment = Spv(k_tex_frag_spv);
  gp.color_count = 1;
  gp.color_formats[0] = Format::kRGBA8Unorm;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);

  Texture* t0 = Target();
  Texture* t1 = Target();
  TextureView* v0 = View(t0);
  TextureView* v1 = View(t1);
  list_->Begin();
  TextureBarrier up[3] = {
      {arr, TextureState::kUndefined, TextureState::kCopyDst},
      {bgra, TextureState::kUndefined, TextureState::kCopyDst},
      {vol, TextureState::kUndefined, TextureState::kCopyDst}};
  up[0].range.layers = 2;
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, up, 3);
  BufferTextureCopy c;
  c.region.width = c.region.height = 2;
  c.region.layers = 2;
  list_->CopyBufferToTexture(arr, staging, &c, 1);
  c.region.layers = 1;
  list_->CopyBufferToTexture(bgra, bgra_staging, &c, 1);
  TextureRange all;
  ClearColor grey{};
  grey.f[0] = grey.f[1] = grey.f[2] = grey.f[3] = 0.2f;
  list_->ClearTexture(vol, TextureState::kCopyDst, all, grey);
  TextureBarrier ready[5] = {
      {arr, TextureState::kCopyDst, TextureState::kShaderRead},
      {bgra, TextureState::kCopyDst, TextureState::kShaderRead},
      {vol, TextureState::kCopyDst, TextureState::kColorTarget},
      {t0, TextureState::kUndefined, TextureState::kColorTarget},
      {t1, TextureState::kUndefined, TextureState::kColorTarget}};
  ready[0].range.layers = 2;
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead | kAccessColorWrite,
                 ready, 5);
  TextureView* sources[2] = {layer_view, bgra_view};
  TextureView* targets[2] = {v0, v1};
  for (int i = 0; i < 2; i++) {
    BindingWrite w;
    w.view = sources[i];
    w.sampler = sampler;
    list_->BeginRenderPass(ClearPass(targets[i]));
    list_->SetPipeline(pipe);
    YUpViewport();
    list_->PushBindGroup(0, set0, &w, 1);
    list_->Draw(3, 1, 0, 0);
    list_->EndRenderPass();
  }
  RenderPassDesc rp = ClearPass(slice);
  rp.colors[0].clear = {{0.0f, 0.0f, 1.0f, 1.0f}};
  list_->BeginRenderPass(rp);
  list_->EndRenderPass();
  Buffer* vol_rb = Readback(8);
  TextureBarrier to_src{vol, TextureState::kColorTarget,
                        TextureState::kCopySrc};
  list_->Barrier(kAccessColorWrite, kAccessCopyRead, &to_src, 1);
  BufferTextureCopy vc;
  vc.row_length = vc.image_height = 1;
  vc.region.depth = 2;
  list_->CopyTextureToBuffer(vol_rb, vol, &vc, 1);
  std::vector<u32> p0 = ReadPixels(t0, TextureState::kColorTarget);
  std::vector<u32> p1 = ReadPixels(t1, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(p0[2 * kW + 2], Rgba(30, 20, 10, 255)) << "swizzled layer 1";
  EXPECT_EQ(p1[2 * kW + 2], Rgba(3, 2, 1, 4)) << "BGRA memory order";
  const u32* vol_px = reinterpret_cast<const u32*>(vol_rb->mapped());
  EXPECT_EQ(vol_px[0], Rgba(51, 51, 51, 51)) << "slice 0 cleared";
  EXPECT_EQ(vol_px[1], Rgba(0, 0, 255, 255)) << "slice 1 rendered";
}

TEST_P(RhiConformance, BufferOpsAndTextureCopies) {
  BufferDesc bd;
  bd.size = 256;
  bd.usage = kBufferCopySrc | kBufferCopyDst;
  Buffer* dev = Own(device_->CreateBuffer(bd));
  Buffer* rb = Readback(256);
  const u32 update[4] = {1, 2, 3, 4};

  // 3D: 4x4x2 texels valued by index; read back a 2x2 window of slice 1
  // into rows of 4 texels.
  TextureDesc vd;
  vd.dim = TextureDim::k3D;
  vd.format = Format::kRGBA8Unorm;
  vd.width = vd.height = 4;
  vd.depth = 2;
  vd.usage = kTextureCopySrc | kTextureCopyDst;
  Texture* vol = Own(device_->CreateTexture(vd));
  u32 vol_texels[32];
  for (u32 i = 0; i < 32; i++)
    vol_texels[i] = i;
  Buffer* vol_staging = Upload(vol_texels, sizeof(vol_texels), 0);
  Buffer* vol_rb = Readback(64);

  // Layer 4 of a cube-compatible array, copied to a plain 2D texture.
  TextureDesc cd;
  cd.format = Format::kRGBA8Unorm;
  cd.layers = 6;
  cd.usage = kTextureSampled | kTextureCopySrc | kTextureCopyDst |
             kTextureCubeCompatible;
  Texture* cube = Own(device_->CreateTexture(cd));
  TextureDesc pd = cd;
  pd.layers = 1;
  pd.usage = kTextureCopySrc | kTextureCopyDst;
  Texture* plain = Own(device_->CreateTexture(pd));
  const u32 marker = Rgba(7, 8, 9, 10);
  Buffer* marker_staging = Upload(&marker, 4, 0);
  Buffer* plain_rb = Readback(4);

  // Depth cleared outside a pass.
  TextureDesc dd;
  dd.format = Format::kD32Float;
  dd.width = dd.height = 4;
  dd.usage = kTextureDepthTarget | kTextureCopySrc | kTextureCopyDst;
  Texture* depth = Own(device_->CreateTexture(dd));
  Buffer* depth_rb = Readback(64);

  // BC1 blocks round-trip unchanged.
  Texture* bc = nullptr;
  u8 blocks[32];
  for (u32 i = 0; i < 32; i++)
    blocks[i] = static_cast<u8>(i * 37 + 5);
  Buffer* bc_staging = Upload(blocks, sizeof(blocks), 0);
  Buffer* bc_rb = Readback(32);
  if (device_->SupportsFormat(Format::kBC1Unorm,
                              kTextureSampled | kTextureCopySrc)) {
    TextureDesc td;
    td.format = Format::kBC1Unorm;
    td.width = td.height = 8;
    td.usage = kTextureSampled | kTextureCopySrc | kTextureCopyDst;
    bc = Own(device_->CreateTexture(td));
  }

  list_->Begin();
  list_->FillBuffer(dev, 0, 256, 0x11111111u);
  list_->Barrier(kAccessCopyWrite, kAccessCopyWrite);
  list_->FillBuffer(dev, 16, 32, 0xdeadbeefu);
  list_->UpdateBuffer(dev, 64, sizeof(update), update);
  list_->Barrier(kAccessCopyWrite, kAccessCopyRead);
  list_->CopyBuffer(rb, 0, dev, 0, 256);

  TextureBarrier to_dst[4] = {
      {vol, TextureState::kUndefined, TextureState::kCopyDst},
      {cube, TextureState::kUndefined, TextureState::kCopyDst},
      {plain, TextureState::kUndefined, TextureState::kCopyDst},
      {depth, TextureState::kUndefined, TextureState::kCopyDst}};
  to_dst[1].range.layers = 6;
  to_dst[3].range.aspect = kAspectDepth;
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, to_dst, 4);
  BufferTextureCopy vc;
  vc.region.width = vc.region.height = 4;
  vc.region.depth = 2;
  list_->CopyBufferToTexture(vol, vol_staging, &vc, 1);
  BufferTextureCopy mc;
  mc.region.base_layer = 4;
  list_->CopyBufferToTexture(cube, marker_staging, &mc, 1);
  TextureRange depth_range;
  depth_range.aspect = kAspectDepth;
  list_->ClearDepthStencil(depth, TextureState::kCopyDst, depth_range, 0.25f,
                           0);
  if (bc) {
    TextureBarrier bb{bc, TextureState::kUndefined, TextureState::kCopyDst};
    list_->Barrier(0, 0, &bb, 1);
    BufferTextureCopy bcc;
    bcc.region.width = bcc.region.height = 8;
    list_->CopyBufferToTexture(bc, bc_staging, &bcc, 1);
    bb = {bc, TextureState::kCopyDst, TextureState::kCopySrc};
    list_->Barrier(kAccessCopyWrite, kAccessCopyRead, &bb, 1);
    list_->CopyTextureToBuffer(bc_rb, bc, &bcc, 1);
  }
  TextureBarrier to_src[3] = {
      {vol, TextureState::kCopyDst, TextureState::kCopySrc},
      {cube, TextureState::kCopyDst, TextureState::kCopySrc},
      {depth, TextureState::kCopyDst, TextureState::kCopySrc}};
  to_src[1].range.layers = 6;
  to_src[2].range.aspect = kAspectDepth;
  list_->Barrier(kAccessCopyWrite, kAccessCopyRead, to_src, 3);
  BufferTextureCopy window;
  window.buffer_offset = 16;
  window.row_length = 4;
  window.region.x = window.region.y = window.region.z = 1;
  window.region.width = window.region.height = 2;
  list_->CopyTextureToBuffer(vol_rb, vol, &window, 1);
  TextureRegion src;
  src.base_layer = 4;
  list_->CopyTexture(plain, TextureRegion{}, cube, src);
  BufferTextureCopy dc;
  dc.region.aspect = kAspectDepth;
  dc.region.width = dc.region.height = 4;
  list_->CopyTextureToBuffer(depth_rb, depth, &dc, 1);
  TextureBarrier ps{plain, TextureState::kCopyDst, TextureState::kCopySrc};
  list_->Barrier(kAccessCopyWrite, kAccessCopyRead, &ps, 1);
  BufferTextureCopy one;
  list_->CopyTextureToBuffer(plain_rb, plain, &one, 1);
  list_->Barrier(kAccessCopyWrite, kAccessHostRead);
  Submit();
  list_->End();

  const u32* words = reinterpret_cast<const u32*>(rb->mapped());
  EXPECT_EQ(words[0], 0x11111111u);
  EXPECT_EQ(words[4], 0xdeadbeefu);
  EXPECT_EQ(words[11], 0xdeadbeefu);
  EXPECT_EQ(words[12], 0x11111111u);
  EXPECT_EQ(words[16], 1u);
  EXPECT_EQ(words[19], 4u);
  EXPECT_EQ(words[63], 0x11111111u);
  const u32* win = reinterpret_cast<const u32*>(vol_rb->mapped());
  EXPECT_EQ(win[4], 16u + 4 + 1) << "slice 1, row 1, texel 1";
  EXPECT_EQ(win[5], 16u + 4 + 2);
  EXPECT_EQ(win[8], 16u + 8 + 1) << "one row length on";
  EXPECT_EQ(*reinterpret_cast<const u32*>(plain_rb->mapped()), marker);
  EXPECT_FLOAT_EQ(reinterpret_cast<const float*>(depth_rb->mapped())[5],
                  0.25f);
  if (bc)
    EXPECT_EQ(std::memcmp(bc_rb->mapped(), blocks, sizeof(blocks)), 0);
}

// A geometry shader expanding a point, a clear of part of an attachment
// inside the pass, and GPU timestamps around the draw.
TEST_P(RhiConformance, GeometryClearAttachmentTimestamps) {
  if (!device_->caps().geometry_shader)
    GTEST_SKIP() << "no geometry shaders";
  PipelineLayout* layout = EmptyLayout();
  GraphicsPipelineDesc gp = PosPipeline(layout);
  gp.geometry = Spv(k_quad_geom_spv);
  gp.topology = Topology::kPointList;
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(gp));
  ASSERT_NE(pipe, nullptr);
  const u32 green = Rgba(0, 255, 0, 255);
  const Vertex point{0, 0, 0, green};
  Buffer* vb = Upload(&point, sizeof(point), kBufferVertex);
  Buffer* inst = Instances({0, 0});
  TimestampPool* pool = nullptr;
  if (device_->caps().timestamps)
    pool = Own(device_->CreateTimestampPool(2));
  Texture* t = Target();
  TextureView* v = View(t);

  list_->Begin();
  if (pool)
    list_->ResetTimestamps(pool, 0, 2);
  TextureBarrier b{t, TextureState::kUndefined, TextureState::kColorTarget};
  list_->Barrier(kAccessHostWrite, kAccessVertexRead, &b, 1);
  if (pool)
    list_->WriteTimestamp(pool, 0, true);
  list_->BeginRenderPass(ClearPass(v));
  list_->SetPipeline(pipe);
  YUpViewport();
  BindVertices(vb, inst);
  list_->Draw(1, 1, 0, 0);
  ClearColor blue{};
  blue.f[2] = blue.f[3] = 1.0f;
  list_->ClearAttachment(0, blue, 0, 0, kAspectColor, 0, 0, 8, 8);
  list_->EndRenderPass();
  if (pool)
    list_->WriteTimestamp(pool, 1, false);
  std::vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 48], green) << "right half";
  EXPECT_EQ(px[32 * kW + 16], 0u) << "left half";
  EXPECT_EQ(px[2 * kW + 2], Rgba(0, 0, 255, 255)) << "cleared rectangle";
  EXPECT_EQ(px[2 * kW + 50], green) << "outside the rectangle";
  if (pool) {
    u64 ts[2] = {};
    ASSERT_TRUE(device_->ReadTimestamps(pool, 0, 2, ts));
    EXPECT_GE(ts[1], ts[0]);
    EXPECT_NE(ts[0], 0u);
  }
}

std::string Name(const ::testing::TestParamInfo<Backend>& info) {
  return BackendName(info.param);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         RhiConformance,
                         ::testing::ValuesIn(gpu::render::CompiledBackends()),
                         Name);

}  // namespace
