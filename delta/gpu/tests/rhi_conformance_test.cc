// Runs the same work through every rhi backend the build has and checks the
// pixels against fixed expectations, so a difference is a backend bug rather
// than a difference of opinion between two backends. Needs a GPU; a backend
// that cannot create a device is skipped.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include "base/arch.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "gpu/render/backend.h"
#include "gpu/rhi/device.h"
#include "gpu/tests/rhi_conformance_extra_spv.h"
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
    // One device per backend for the whole run, never destroyed: the Vulkan
    // backend leaves its VkDevice to process exit, and a device per test
    // exhausts the driver after a few dozen tests.
    static base::Map<Backend, Device*> devices;
    Device*& shared = devices[GetParam()];
    if (!shared) {
      auto created = gpu::render::CreateBackendDevice(GetParam());
      shared = gpu::rhi::Release(created);
    }
    device_ = shared;
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
    device_ = nullptr;
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
  base::Vector<u32> ReadPixels(Texture* t, TextureState state) {
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
    base::Vector<u32> px(kW * kH);
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
  Buffer* Instances(base::Vector<i16> offsets) {
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

  Device* device_ = nullptr;  // owned by SetUp's table
  CommandList* list_ = nullptr;
  base::Vector<Object*> objects_;
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
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
  ubo_desc.bindings.push_back(
      {0, BindingType::kUniformBufferDynamic, kStageVertex});
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
  const float verts[] = {-1.0f, 0.0f, 1, 0,     0,    1, 3.0f, 0.0f, 1,
                         0,     0,    1, -1.0f, 2.0f, 1, 0,    0,    1};
  Buffer* vb = Upload(verts, sizeof(verts), kBufferVertex);
  // Two 256-byte windows; the draw binds the second, a unit tint.
  base::Vector<float> ubo(128, 0.0f);
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
  base::Vector<u32> p0 = ReadPixels(t0, TextureState::kColorTarget);
  base::Vector<u32> p1 = ReadPixels(t1, TextureState::kColorTarget);
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[2 * kW + 2], texels[0]) << "top-left";
  EXPECT_EQ(px[2 * kW + kW - 3], texels[1]) << "top-right";
  EXPECT_EQ(px[(kH - 3) * kW + 2], texels[2]) << "bottom-left";
}

TEST_P(RhiConformance, ComputeStorageBufferAndPush) {
  constexpr u32 kN = 256;
  base::Vector<u32> data(kN);
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kGeneral);
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
  base::Vector<u32> px = ReadPixels(b, TextureState::kCopyDst);
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
  const Vertex verts[] = {{-0.9f, -0.5f, 0, red}, {-0.1f, -0.5f, 0, red},
                          {-0.5f, 0.5f, 0, red},  {0.1f, -0.5f, 0, red},
                          {0.5f, 0.5f, 0, red},   {0.9f, -0.5f, 0, red}};
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
  base::Vector<u32> pu = ReadPixels(up, TextureState::kColorTarget);
  base::Vector<u32> pd = ReadPixels(down, TextureState::kColorTarget);
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
  const Vertex verts[] = {// Skipped by the vertex offset.
                          {9, 9, 0, 0},
                          {9, 9, 0, 0},
                          {9, 9, 0, 0},
                          {9, 9, 0, 0},
                          // Left half at z 0.5.
                          {-1, -1, 0.5f, red},
                          {0, -1, 0.5f, red},
                          {0, 1, 0.5f, red},
                          {-1, 1, 0.5f, red},
                          // Full screen at z 0.8, then at z 0.1.
                          {-1, -1, 0.8f, green},
                          {3, -1, 0.8f, green},
                          {-1, 3, 0.8f, green},
                          {-1, -1, 0.1f, blue},
                          {3, -1, 0.1f, blue},
                          {-1, 3, 0.1f, blue}};
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
  base::Vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 8], Rgba(255, 0, 255, 255)) << "left";
  EXPECT_EQ(px[32 * kW + 56], green) << "right";
  const float* d = reinterpret_cast<const float*>(depth_rb->mapped());
  const u8* st = depth_rb->mapped() + kW * kH * 4;
  EXPECT_FLOAT_EQ(d[32 * kW + 8], 0.5f);
  EXPECT_FLOAT_EQ(d[32 * kW + 56], 0.8f);
  // vkd3d copies only plane 0 of a planar texture.
  if (!std::strstr(device_->caps().device_name, "vkd3d")) {
    EXPECT_EQ(st[32 * kW + 8], 3);
    EXPECT_EQ(st[32 * kW + 56], 0);
  }
}

// first_instance selects the per-instance data, and a scaled attribute
// converts its integers to float without normalizing.
TEST_P(RhiConformance, InstancingAndScaledAttributes) {
  PipelineLayout* layout = EmptyLayout();
  Pipeline* pipe = Own(device_->CreateGraphicsPipeline(PosPipeline(layout)));
  ASSERT_NE(pipe, nullptr);
  const u32 white = Rgba(255, 255, 255, 255);
  const Vertex verts[] = {
      {0.3f, 0.3f, 0, white}, {0.7f, 0.3f, 0, white}, {0.5f, 0.7f, 0, white}};
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
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
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead | kAccessColorWrite, ready,
                 5);
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
  base::Vector<u32> p0 = ReadPixels(t0, TextureState::kColorTarget);
  base::Vector<u32> p1 = ReadPixels(t1, TextureState::kColorTarget);
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
  EXPECT_FLOAT_EQ(reinterpret_cast<const float*>(depth_rb->mapped())[5], 0.25f);
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
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
  QuadPush p{
      {x0, y0, x1, y1}, {color[0], color[1], color[2], color[3]}, {z, 0, 0, 0}};
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
  PipelineLayout* layout = Own(QuadLayout(device_));
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
  base::Vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
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
  PipelineLayout* layout = Own(QuadLayout(device_));
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
  base::Vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[32 * kW + 8], Rgba(0, 255, 0, 255)) << "stencil pass";
  EXPECT_EQ(px[32 * kW + 56], 0u) << "stencil fail";
}

TEST_P(RhiConformance, BlendAndConstants) {
  Texture* color = Target();
  TextureView* cv = View(color);
  PipelineLayout* layout = Own(QuadLayout(device_));
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
  base::Vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
  list_->End();
  EXPECT_TRUE(Near(px[32 * kW + 8], Rgba(64, 0, 191, 64)))
      << base::Format("{:#x}", px[32 * kW + 8]).c_str();
  EXPECT_TRUE(Near(px[32 * kW + 56], Rgba(128, 64, 255, 128)))
      << base::Format("{:#x}", px[32 * kW + 56]).c_str();
}

// Draws a fullscreen triangle sampling `view` with `fragment` into a fresh
// target and returns its centre texel.
u32 SampleCenter(Device* device,
                 CommandList* list,
                 base::Vector<Object*>& objects,
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
  const double l =
      v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
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
    got[i] = SampleCenter(device_, list_, objects_, Spv(k_tex_frag_spv),
                          views[i], TextureState::kShaderRead, {0, 0, 0, 0});
  EXPECT_EQ(got[0], texel);
  EXPECT_EQ(got[1], Rgba(192, 128, 64, 255));
  EXPECT_TRUE(Near(got[2], Rgba(SrgbToLinear(64), SrgbToLinear(128),
                                SrgbToLinear(192), 255)))
      << base::Format("{:#x}", got[2]).c_str();
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
    return SampleCenter(device_, list_, objects_, fs, v, s, coord);
  };
  EXPECT_EQ(sample(Spv(k_tex_frag_spv), layer1_view, {0, 0, 0, 0}), texels[1])
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

// Sampling an integer texture is a nearest fetch with the sampler's
// addressing (D3D12 before SM 6.7 emulates it).
TEST_P(RhiConformance, IntegerTextureSampling) {
  TextureDesc td;
  td.format = Format::kR32Uint;
  td.width = td.height = 2;
  td.usage = kTextureSampled | kTextureCopyDst;
  Texture* tex = Own(device_->CreateTexture(td));
  const u32 texels[4] = {10, 20, 30, 40};
  Buffer* staging = Upload(texels, sizeof(texels), 0);
  list_->Begin();
  TextureBarrier up{tex, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, &up, 1);
  BufferTextureCopy c;
  c.region.width = c.region.height = 2;
  list_->CopyBufferToTexture(tex, staging, &c, 1);
  TextureBarrier rd{tex, TextureState::kCopyDst, TextureState::kShaderRead};
  list_->Barrier(kAccessCopyWrite, kAccessShaderRead, &rd, 1);
  list_->End();
  ASSERT_TRUE(device_->Wait(device_->Submit(list_)));
  TextureView* view = Own(device_->CreateView(tex, {}));
  auto sample = [&](float u, float v) {
    return SampleCenter(device_, list_, objects_, Spv(k_x_itex_frag_spv), view,
                        TextureState::kShaderRead, {u, v, 0, 0});
  };
  EXPECT_EQ(sample(0.25f, 0.25f), Rgba(10, 0, 0, 1));
  EXPECT_EQ(sample(0.75f, 0.25f), Rgba(20, 0, 0, 1));
  EXPECT_EQ(sample(0.25f, 0.75f), Rgba(30, 0, 0, 1));
  EXPECT_EQ(sample(0.9f, 0.6f), Rgba(40, 0, 0, 1));
  EXPECT_EQ(sample(1.3f, 0.1f), Rgba(20, 0, 0, 1)) << "clamped";
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
  list_->WriteTimestamp(pool, 0, true);
  list_->FillBuffer(buf, 0, bd.size, 7);
  list_->WriteTimestamp(pool, 1, false);
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
  base::Vector<u32> px = ReadPixels(color, TextureState::kColorTarget);
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
  PipelineLayout* layout = Own(QuadLayout(device_));
  GraphicsPipelineDesc none = QuadPipeline(layout);
  GraphicsPipelineDesc back = none;
  back.cull = CullMode::kBack;
  back.front_ccw = true;
  Pipeline* none_pipe = Own(device_->CreateGraphicsPipeline(none));
  Pipeline* back_pipe = Own(device_->CreateGraphicsPipeline(back));
  ASSERT_NE(none_pipe, nullptr);
  ASSERT_NE(back_pipe, nullptr);
  Texture* t[3] = {Target(), Target(), Target()};
  base::Vector<u32> px[3];
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
  const float verts[] = {-1.0f, 0.0f, 1, 0,     0,    1, 3.0f, 0.0f, 1,
                         0,     0,    1, -1.0f, 2.0f, 1, 0,    0,    1};
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[4 * kW + 4], Rgba(0, 0, 255, 255)) << "top row, swizzled";
  EXPECT_EQ(px[(kH - 4) * kW + 4], 0u) << "bottom row covered";
}

// Tightly packed and oddly pitched buffer layouts, at offsets no copy
// engine aligns for.
TEST_P(RhiConformance, UnalignedTextureCopies) {
  constexpr u32 kW = 5, kH = 3;
  TextureDesc td;
  td.format = Format::kRGBA8Unorm;
  td.width = kW;
  td.height = kH;
  td.usage = kTextureCopySrc | kTextureCopyDst;
  Texture* tex = Own(device_->CreateTexture(td));
  u32 data[1 + kW * kH];
  data[0] = 0;
  for (u32 y = 0; y < kH; y++)
    for (u32 x = 0; x < kW; x++)
      data[1 + y * kW + x] = Rgba(u8(x * 10), u8(y * 10), 7, 255);
  Buffer* staging = Upload(data, sizeof(data), 0);
  Buffer* tight = Readback(kW * kH * 4);
  Buffer* pitched = Readback(512);
  std::memset(pitched->mapped(), 0, 512);
  list_->Begin();
  TextureBarrier up{tex, TextureState::kUndefined, TextureState::kCopyDst};
  list_->Barrier(kAccessHostWrite, kAccessCopyRead, &up, 1);
  BufferTextureCopy c;
  c.buffer_offset = 4;
  c.region.width = kW;
  c.region.height = kH;
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
  EXPECT_EQ(std::memcmp(tight->mapped(), data + 1, kW * kH * 4), 0);
  const u32* p = reinterpret_cast<const u32*>(pitched->mapped() + 12);
  for (u32 r = 0; r < 2; r++)
    for (u32 x = 0; x < 3; x++)
      EXPECT_EQ(p[r * 7 + x], data[1 + (1 + r) * kW + 1 + x])
          << "row " << r << " col " << x;
  EXPECT_EQ(p[3], 0u) << "past the copied row";
}

TEST_P(RhiConformance, StorageBindingsAndSubgroups) {
  constexpr u32 kN = 128;
  base::Vector<u32> input(kN);
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
  base::Vector<u32> ubo(64, 0);
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
  EXPECT_EQ(out[130], 5u) << "ballot find LSB";
  EXPECT_EQ(out[131], 6u) << "ballot find MSB";
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
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
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  EXPECT_EQ(px[2 * kW + 2], Rgba(0, 255, 0, 255));
  EXPECT_EQ(px[(kH - 3) * kW + kW - 3], Rgba(0, 255, 0, 255));
  EXPECT_EQ(px[2 * kW + kW - 3], Rgba(0, 255, 0, 255));
}

// gl_BaryCoordEXT over a strip: which vertex each weight belongs to, in the
// even and the odd triangle. D3D12 without SM 6.1 emulates it.
TEST_P(RhiConformance, FragmentBarycentrics) {
  if (!device_->caps().fragment_barycentric &&
      device_->backend() != Backend::kD3D12)
    GTEST_SKIP() << "no fragment barycentrics";
  PipelineLayout* layout = Own(QuadLayout(device_));
  GraphicsPipelineDesc gp = QuadPipeline(layout);
  gp.fragment = Spv(k_x_bary_frag_spv);
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
  DrawQuad(list_, -1, -1, 1, 1, kRed);
  list_->EndRenderPass();
  base::Vector<u32> px = ReadPixels(t, TextureState::kColorTarget);
  list_->End();
  // The dominant weight near each corner. The odd triangle of the strip
  // orders its vertices (1, 3, 2), keeping the winding.
  auto dominant = [&](u32 x, u32 y) {
    const u32 p = px[y * kW + x];
    const u32 r = p & 0xff, g = (p >> 8) & 0xff, bl = (p >> 16) & 0xff;
    return r > g && r > bl ? 0 : g > bl ? 1 : 2;
  };
  EXPECT_EQ(dominant(1, kH - 2), 0) << "vertex 0, bottom left";
  EXPECT_EQ(dominant(kW - 4, kH - 2), 1) << "vertex 1, bottom right";
  EXPECT_EQ(dominant(1, 1), 2) << "vertex 2, top left";
  EXPECT_EQ(dominant(kW - 2, 1), 1) << "vertex 3, top right";
  EXPECT_EQ(dominant(kW - 2, kH * 3 / 4), 0) << "odd triangle, vertex 1";
}

}  // namespace
