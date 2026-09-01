/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * What the renderer decides, asserted without a GPU. The recording device
 * turns a sequence of RHI calls into a log, so a test can state what a piece
 * of work should have produced -- which pass, which barriers, in which order
 * -- instead of standing up a device and looking at pixels.
 */

#include "gpu/rhi/null_device.h"

#include <gtest/gtest.h>

#include "gpu/rhi/types.h"

namespace gpu::rhi {
namespace {

TEST(RhiFormat, TableAgreesWithTheEnum) {
  // Every format answers, and only the unknown one answers with nothing.
  for (u32 i = 1; i < static_cast<u32>(Format::kCount); i++) {
    const Format f = static_cast<Format>(i);
    EXPECT_GT(FormatTexelBytes(f), 0u) << FormatName(f);
    EXPECT_GT(FormatBlockDim(f), 0u) << FormatName(f);
    EXPECT_STRNE(FormatName(f), "Unknown");
  }
  EXPECT_EQ(FormatTexelBytes(Format::Unknown), 0u);
}

TEST(RhiFormat, RowBytesRoundsCompressedFormatsUpToWholeBlocks) {
  EXPECT_EQ(FormatRowBytes(Format::RGBA8Unorm, 64), 256u);
  EXPECT_EQ(FormatRowBytes(Format::R8Unorm, 3), 3u);
  // A BC1 row of 4 texels is one 8-byte block; 5 texels still needs two.
  EXPECT_EQ(FormatRowBytes(Format::BC1Unorm, 4), 8u);
  EXPECT_EQ(FormatRowBytes(Format::BC1Unorm, 5), 16u);
}

TEST(RhiFormat, DepthFormatsAreMarkedAndOnlyOneCarriesStencil) {
  EXPECT_TRUE(FormatIsDepth(Format::D32Float));
  EXPECT_FALSE(FormatHasStencil(Format::D32Float));
  EXPECT_TRUE(FormatHasStencil(Format::D32FloatS8Uint));
  EXPECT_FALSE(FormatIsDepth(Format::RGBA8Unorm));
}

// A device plus the handful of objects most tests here need.
class NullRecording : public ::testing::Test {
 protected:
  std::unique_ptr<Texture> MakeTarget(const char* name) {
    TextureDesc desc;
    desc.width = desc.height = 16;
    desc.format = Format::RGBA8Unorm;
    desc.usage = kTextureRenderTarget | kTextureSampled | kTextureCopySrc;
    desc.name = name;
    return device_.CreateTexture(desc);
  }

  NullDevice device_;
};

TEST_F(NullRecording, ARepeatedTransitionRecordsNothing) {
  auto target = MakeTarget("scene");
  auto list = device_.CreateCommandList();
  list->Begin();
  list->Transition(target.get(), ResourceState::CopySrc);
  list->Transition(target.get(), ResourceState::CopySrc);
  list->Transition(target.get(), ResourceState::ShaderRead);
  list->End();

  EXPECT_EQ(device_.LogText(),
            "begin\n"
            "barrier scene undefined->copysrc\n"
            "barrier scene copysrc->shaderread\n"
            "end\n");
}

TEST_F(NullRecording, RepeatingAnUnorderedAccessTransitionStillBarriers) {
  BufferDesc desc;
  desc.size = 256;
  desc.usage = kBufferStorage;
  desc.name = "scratch";
  auto buffer = device_.CreateBuffer(desc);

  auto list = device_.CreateCommandList();
  list->Begin();
  // Two dispatches writing the same buffer need the second to see the first,
  // which is the one case where a transition into the current state is real
  // work rather than a no-op.
  list->Transition(buffer.get(), ResourceState::UnorderedAccess);
  list->Transition(buffer.get(), ResourceState::UnorderedAccess);
  list->End();

  EXPECT_EQ(device_.LogText(),
            "begin\n"
            "barrier scratch common->uav\n"
            "uavbarrier scratch\n"
            "end\n");
}

TEST_F(NullRecording, APassTransitionsItsOwnAttachments) {
  auto color = MakeTarget("scene");
  auto list = device_.CreateCommandList();

  RenderPassDesc pass;
  pass.width = pass.height = 16;
  pass.num_color = 1;
  pass.color[0].texture = color.get();
  pass.color[0].load = LoadOp::Clear;
  pass.color[0].clear[1] = 1.0f;

  list->Begin();
  list->BeginRenderPass(pass);
  list->EndRenderPass();
  list->End();

  const auto barriers = device_.LogMatching("barrier");
  ASSERT_EQ(barriers.size(), 1u);
  EXPECT_EQ(barriers[0], "barrier scene undefined->rendertarget");
  EXPECT_EQ(device_.LogMatching("pass")[0],
            "pass 16x16 color0=scene:clear(0.000,1.000,0.000,0.000)");
  EXPECT_EQ(color->state(), ResourceState::RenderTarget);
}

TEST_F(NullRecording, EveryDynamicOffsetReachesTheBinding) {
  BindGroupLayoutDesc layout_desc;
  layout_desc.index = 1;
  layout_desc.name = "cbufs";
  layout_desc.num_entries = 2;
  layout_desc.entries[0] = {0, BindingType::UniformBuffer, kStageVertex, true};
  layout_desc.entries[1] = {1, BindingType::UniformBuffer, kStagePixel, true};
  auto layout = device_.CreateBindGroupLayout(layout_desc);

  BufferDesc buffer_desc;
  buffer_desc.size = 1024;
  buffer_desc.usage = kBufferConstant;
  buffer_desc.memory = MemoryKind::Upload;
  buffer_desc.name = "cbuf";
  auto buffer = device_.CreateBuffer(buffer_desc);

  BindGroupDesc group_desc;
  group_desc.layout = layout.get();
  group_desc.name = "draw";
  group_desc.num_bindings = 2;
  group_desc.bindings[0].binding = 0;
  group_desc.bindings[0].buffer = buffer.get();
  group_desc.bindings[0].size = 256;
  group_desc.bindings[1].binding = 1;
  group_desc.bindings[1].buffer = buffer.get();
  group_desc.bindings[1].size = 256;
  auto group = device_.CreateBindGroup(group_desc);

  const u32 offsets[2] = {0, 256};
  auto list = device_.CreateCommandList();
  list->Begin();
  list->SetBindGroup(1, group.get(), offsets, 2);
  list->End();

  EXPECT_EQ(device_.LogMatching("bindgroup")[0], "bindgroup 1 draw +0 +256");
}

TEST_F(NullRecording, MappedMemoryIsRealSoATestCanStageData) {
  BufferDesc desc;
  desc.size = 64;
  desc.usage = kBufferVertex;
  desc.memory = MemoryKind::Upload;
  auto buffer = device_.CreateBuffer(desc);

  auto* words = static_cast<u32*>(buffer->Map());
  ASSERT_NE(words, nullptr);
  words[0] = 0xABCDEF01u;
  EXPECT_EQ(static_cast<u32*>(buffer->Map())[0], 0xABCDEF01u);
}

TEST_F(NullRecording, TheWholeOfADrawIsRecordedInOrder) {
  auto color = MakeTarget("scene");
  GraphicsPipelineDesc pipeline_desc;
  pipeline_desc.name = "solid";
  auto pipeline = device_.CreateGraphicsPipeline(pipeline_desc);

  BufferDesc vertex_desc;
  vertex_desc.size = 256;
  vertex_desc.usage = kBufferVertex;
  vertex_desc.memory = MemoryKind::Upload;
  vertex_desc.name = "verts";
  auto vertices = device_.CreateBuffer(vertex_desc);

  RenderPassDesc pass;
  pass.width = pass.height = 16;
  pass.num_color = 1;
  pass.color[0].texture = color.get();
  pass.color[0].load = LoadOp::Clear;

  auto list = device_.CreateCommandList();
  list->Begin();
  list->BeginRenderPass(pass);
  list->SetPipeline(pipeline.get());
  list->SetVertexBuffer(0, vertices.get(), 0);
  list->SetViewport(0, 0, 16, 16, 0, 1);
  list->SetScissor(0, 0, 16, 16);
  list->Draw(3, 1, 0, 0);
  list->EndRenderPass();
  list->Transition(color.get(), ResourceState::CopySrc);
  list->End();
  device_.Submit(list.get());

  EXPECT_EQ(device_.LogText(),
            "begin\n"
            "barrier scene undefined->rendertarget\n"
            "pass 16x16 color0=scene:clear(0.000,0.000,0.000,0.000)\n"
            "pipeline solid\n"
            "vbuf 0 verts+0\n"
            "viewport 0.0,0.0 16.0x16.0 z=0.000..1.000\n"
            "scissor 0,0 16x16\n"
            "draw v=3 i=1 first=0,0\n"
            "endpass\n"
            "barrier scene rendertarget->copysrc\n"
            "end\n"
            "submit 1\n");
}

}  // namespace
}  // namespace gpu::rhi
