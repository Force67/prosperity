#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_translate.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/ps5/draw_state.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_draw_recomp.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_upload_ring.h"

extern "C" void prosperity_gpu_end_of_pipe() {}
extern "C" bool prosperity_ps5_is_display_buffer(u64) { return false; }

TEST(VkDraw, ConstantBufferBudgetAppliesAfterDeviceInitialization) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  if (gpu::vk::g_ring.ubo_buf)
    GTEST_SKIP() << "Run this initialization test in a fresh renderer";
  base::OptionBase* budget = nullptr;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_GPU_UBORING_MB") == 0)
      budget = const_cast<base::OptionBase*>(option);
  });
  ASSERT_NE(budget, nullptr);
  ASSERT_TRUE(budget->SetFromString("512"));  // Title settings load after Init.
  gpu::rhi::BeginFrame(renderer);
  const auto capacity = gpu::vk::UboRingBytes();
  EXPECT_EQ(capacity, std::min<VkDeviceSize>(512ull << 20,
      VkDeviceSize(gpu::vk::g_dev.max_storage_buffer_range) * 2));
  ASSERT_NE(gpu::vk::g_ring.ubo_map, nullptr);
  VkMemoryRequirements memory{};
  vkGetBufferMemoryRequirements(gpu::vk::g_dev.device,
                                gpu::vk::g_ring.ubo_buf, &memory);
  EXPECT_GE(memory.size, capacity);
  // Later option changes cannot move offsets beyond the existing allocation.
  ASSERT_TRUE(budget->SetFromString("1024"));
  EXPECT_EQ(gpu::vk::UboRingBytes(), capacity);
  budget->Reset();
  gpu::rhi::EndFrame(renderer, 0);
}

TEST(RdnaResources, RepeatedDescriptorLoadsReuseTextureBindings) {
  std::vector<u32> code;
  for (u32 i = 0; i < 32; ++i) {
    code.insert(code.end(), {0xf408040e, 0xfa000030,  // s_load s[16:19], s[28:29], 48
                              0xf09c0108, 0x00040f05});  // sample with s[16:23]
  }
  code.push_back(0xbf810000);
  const auto plan = gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(plan.binding_srsrc.size(), 1u);
  EXPECT_EQ(plan.binding_by_pc.size(), 32u);

  // A changed table pointer or offset must keep the samples distinct.
  code.insert(code.begin() + 4, 0xbe9c0381);  // s_mov_b32 s28, 1
  auto changed = gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 2u);
  code[6] = 0xfa000040;
  changed = gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 3u);

  // A memory-writing shader cannot assume a repeated load sees the same data.
  code.insert(code.end() - 1, {0xf0200108, 0x00000400});  // image_store
  changed = gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 33u);
}

// Pipelines use Recompiled addresses as module identities. Keep programs alive
// for the renderer lifetime, matching the production shader cache.

TEST(VkDraw, MeshWorkgroupsExpandInputPointsIntoTriangles) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 vs[64] = {0x7e000280, 0x7e0202f2,
                      0xf80008cf, 0x01000000, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  static auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  using gpu::gcn::spirv::Id;
  gpu::gcn::spirv::Module m;
  m.Capability(spv::Capability::MeshShadingEXT);
  m.Extension("SPV_EXT_mesh_shader");
  const Id u = m.TypeInt(), f = m.TypeFloat(), v4 = m.TypeVec(f, 4);
  const auto uc = [&](u32 v) { return m.ConstU32(v); };
  const auto fc = [&](float v) { return m.ConstF32(v); };
  const Id uv3 = m.TypeVec(u, 3);
  const Id group = m.Variable(m.TypePointer(spv::StorageClass::Input, uv3),
                             spv::StorageClass::Input);
  m.Decorate(group, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::WorkgroupId)});
  const Id pos = m.Variable(
      m.TypePointer(spv::StorageClass::Output, m.TypeArray(v4, 3)),
      spv::StorageClass::Output);
  m.Decorate(pos, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::Position)});
  const Id prim = m.Variable(
      m.TypePointer(spv::StorageClass::Output, m.TypeArray(uv3, 1)),
      spv::StorageClass::Output);
  m.Decorate(prim, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::PrimitiveTriangleIndicesEXT)});
  const Id main = m.BeginFunction(m.TypeVoid(), m.TypeFunction(m.TypeVoid(), {}));
  const Id gx = m.CompositeExtract(u, m.Load(uv3, group), 0);
  const Id offset = m.Emit(spv::Op::OpFSub, f,
                          {m.Emit(spv::Op::OpConvertUToF, f, {gx}), fc(.5f)});
  m.EmitVoid(spv::Op::OpSetMeshOutputsEXT, {uc(3), uc(1)});
  for (u32 i = 0; i < 3; ++i) {
    const float x[] = {-.4f, .4f, 0.f}, y[] = {-.5f, -.5f, .5f};
    const Id p = m.CompositeConstruct(v4,
        {m.Emit(spv::Op::OpFAdd, f, {offset, fc(x[i])}), fc(y[i]), fc(0), fc(1)});
    m.Store(m.AccessChain(m.TypePointer(spv::StorageClass::Output, v4),
                          pos, {uc(i)}), p);
  }
  m.Store(m.AccessChain(m.TypePointer(spv::StorageClass::Output, uv3),
                        prim, {uc(0)}),
           m.CompositeConstruct(uv3, {uc(0), uc(1), uc(2)}));
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::MeshEXT, main, "main", {group, pos, prim});
  m.ExecMode(main, spv::ExecutionMode::LocalSize, {1, 1, 1});
  m.ExecMode(main, spv::ExecutionMode::OutputVertices, {3});
  m.ExecMode(main, spv::ExecutionMode::OutputPrimitivesEXT, {1});
  m.ExecMode(main, spv::ExecutionMode::OutputTrianglesEXT, {});
  std::string error;
  ASSERT_TRUE(gpu::gcn::spirv::Finalize(m.Assemble(), &program.mesh_spirv, &error))
      << error;
  program.vs_spirv.clear();
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 2;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 12) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 0);
}

TEST(VkDraw, SplitNggStagesTransferLdsAndExportConnectivity) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 es[64] = {
      0x34000a82,              // v0 = ES vertex ID * 4
      0xd8340000, 0x00000500,  // LDS[v0] = ES vertex ID
      0xbe802006};             // s_setpc_b64 s[6:7] -> separately bound GS
  const u32 gs[64] = {
      0xd8d80000, 0x02000000,  // v2 = LDS[primitive vertex offset]
      0x7e000d02,              // v0 = float(v2)
      0x100000f0, 0x060000f1,  // x = id * .5 - .5
      0x7e0202f1, 0x7e0602f0,  // y = -.5, alternate = .5
      0x7d840481, 0x02020701,  // y = id == 1 ? .5 : -.5
      0x7e040280, 0x7e0602f2,  // z = 0, w = 1
      0xf80008cf, 0x03020100,  // POS0
      0x7e0802ff, 0x00200400,  // packed triangle indices 0, 1, 2
      0xbefe0481,              // only lane zero exports the primitive
      0xf8000941, 0x00000004,
      0xbefe04c1,
      0xbefc03ff, 0x00001003,  // allocation: 3 vertices, 1 primitive
      0xbf900009, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128};
  static const auto program = gpu::rdna::Recompile(es, ps, user_data, user_data,
                                            0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(program.ok);
  ASSERT_FALSE(program.mesh_spirv.empty());
  ASSERT_TRUE(program.vs_spirv.empty());
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
  EXPECT_EQ(pixels[0], 0);
}

TEST(VkDraw, UnifiedNggProgramExportsTriangleConnectivity) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 gs[64] = {
      0x7e040305,              // v2 = input vertex ID
      0x7e000d02,              // v0 = float(v2)
      0x100000f0, 0x060000f1,  // x = id * .5 - .5
      0x7e0202f1, 0x7e0602f0,  // y = -.5, alternate = .5
      0x7d840481, 0x02020701,  // y = id == 1 ? .5 : -.5
      0x7e040280, 0x7e0602f2,  // z = 0, w = 1
      0xf80008cf, 0x03020100,  // POS0
      0x7e0802ff, 0x00200400,  // packed triangle indices 0, 1, 2
      0xbefe0481,              // only lane zero exports the primitive
      0xf8000941, 0x00000004,
      0xbefe04c1,
      0xbefc03ff, 0x00001003,  // allocation: 3 vertices, 1 primitive
      0xbf900009, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128, 0, false};
  static const auto program = gpu::rdna::Recompile(gs, ps, user_data, user_data,
                                            0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(program.ok);
  EXPECT_TRUE(gpu::rdna::HasNggPrimitiveExports(gs));
  EXPECT_FALSE(gpu::rdna::HasNggTransfer(gs));
  ASSERT_FALSE(program.mesh_spirv.empty());
  ASSERT_TRUE(program.vs_spirv.empty());
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
  EXPECT_EQ(pixels[0], 0);
}

TEST(VkDraw, NggPointBatchesPreserveEveryExpandedQuad) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  std::array<u32, 4096> code{};
  u32 pc = 0;
  const auto v1 = [&](u32 op, u32 dst, u32 src) {
    code[pc++] = 0x7e000000 | (dst << 17) | (op << 9) | src;
  };
  const auto v2 = [&](u32 op, u32 dst, u32 src, u32 other) {
    code[pc++] = (op << 25) | (dst << 17) | (other << 9) | src;
  };
  const auto literal = [&](u32 op, u32 dst, u32 value, u32 other) {
    v2(op, dst, 255, other); code[pc++] = value;
  };
  // Four output vertices and two triangles for each independent input point.
  // v0 initially carries local_id*4, v5 carries group_base+local_id.
  v2(0x16, 1, 130, 0);             // local_id
  v2(0x16, 2, 130, 1);             // local input = local_id / 4
  v2(0x26, 3, 261, 1);             // group base = v5 - local_id
  v2(0x25, 3, 258, 3);             // global input point
  v2(0x1b, 4, 129, 1);             // corner x bit
  v2(0x16, 5, 129, 1);
  v2(0x1b, 5, 129, 5);             // corner y bit
  v1(6, 10, 259);
  literal(8, 10, 0x3e800000, 10);   // x = input*.25 - .75
  literal(3, 10, 0xbf400000, 10);
  v1(6, 4, 260);
  literal(8, 4, 0x3e000000, 4);     // quad width .125
  v2(3, 10, 260, 10);
  v1(6, 11, 261);
  v2(8, 11, 240, 11);              // quad height .5
  literal(3, 11, 0xbe800000, 11);
  v1(1, 12, 128); v1(1, 13, 242);
  code[pc++] = 0xf80008cf; code[pc++] = 0x0d0c0b0a;
  v2(0x16, 6, 129, 1);
  v2(0x1a, 6, 130, 6);             // primitive's vertex base
  v2(0x1b, 7, 129, 1);             // triangle within quad
  v2(0x1a, 8, 129, 7);
  v2(0x25, 8, 262, 8);
  v2(0x25, 8, 129, 8);             // i1 = base + 1 + triangle*2
  v2(0x25, 9, 130, 6);             // i2 = base + 2
  v2(0x25, 6, 263, 6);             // i0 = base + triangle
  v2(0x1a, 8, 138, 8); v2(0x1a, 9, 148, 9);
  v2(0x1c, 6, 264, 6); v2(0x1c, 6, 265, 6);
  code[pc++] = 0xf8000941; code[pc++] = 6;
  v1(1, 8, 3); literal(0x1b, 8, 255, 8); // live input count from wave info
  v2(0x1a, 9, 130, 8); v2(0x1a, 8, 141, 8);
  v2(0x1c, 9, 264, 9);
  v1(2, 0, 265);                   // M0 = vertices | (primitives << 12)
  code[pc++] = 0xbefc0300; code[pc++] = 0xbf900009; code[pc++] = 0xbf810000;
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280,
                        0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  const u32 batches[] = {6, 2, 4}; // one group, three groups, partial final group
  static std::array<gpu::gcn::Recompiled, 3> programs;
  alignas(65536) static std::array<std::array<u8, 65536>, 3> targets{};
  std::array<u8, 64 * 16 * 4> reference{};
  for (u32 i = 0; i < 3; ++i) {
    SCOPED_TRACE(batches[i]);
    const gpu::rdna::NggConfig cfg{code.data(), 64, batches[i], batches[i] * 4,
                                    batches[i] * 2, 128, 0, false};
    programs[i] = gpu::rdna::Recompile(code.data(), ps, user_data, user_data,
                                       0, false, 0, 0, nullptr, 0, &cfg);
    ASSERT_TRUE(programs[i].ok);
    gpu::rhi::DrawInfo draw;
    draw.recomp = &programs[i]; draw.prim_type = 1; draw.vertex_count = 6;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[i].data());
    draw.rt_w = 64; draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1; draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    for (u32 q = 0; q < 6; ++q) {
      EXPECT_EQ(pixels[(8 * 64 + 10 + q * 8) * 4], 255);
      EXPECT_EQ(pixels[(8 * 64 + 14 + q * 8) * 4], 0);
    }
    if (!i) std::copy_n(pixels, reference.size(), reference.begin());
    else EXPECT_TRUE(std::equal(reference.begin(), reference.end(), pixels));
  }
}

TEST(VkDraw, MeshConstantWindowsExceedTheDynamicDescriptorLimit) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 es[64] = {0x34000a82, 0xd8340000, 0x00000500, 0xbe802006};
  std::array<u32, 256> gs{};
  u32 pc = 0;
  // One scalar load in each of 17 disjoint constant-buffer windows. The
  // final load controls the triangle position, beyond the 15 dynamic UBOs.
  for (u32 window = 0; window < 17; ++window) {
    gs[pc++] = 0xf4200004;  // s_buffer_load_dword s0, s[8:11], offset
    gs[pc++] = 0xfa000000 | (window * gpu::gcn::kCbufDwords * 4);
  }
  const u32 body[] = {
      0xd8d80000, 0x02000000, 0x7e000d02,
      0x100000f0, 0x060000f1, 0x06000000,  // x += s0
      0x7e0202f1, 0x7e0602f0, 0x7d840481, 0x02020701,
      0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100,
      0x7e0802ff, 0x00200400, 0xbefe0481, 0xf8000941, 0x00000004,
      0xbefe04c1, 0xbefc03ff, 0x00001003, 0xbf900009, 0xbf810000};
  std::copy(std::begin(body), std::end(body), gs.begin() + pc);
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs.data(), 64, 3, 3, 1, 128};
  static const auto program = gpu::rdna::Recompile(es, ps, user_data, user_data,
                                            0, false, 4, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(program.ok);
  ASSERT_EQ(program.vs_cbufs.size(), 17u);
  ASSERT_TRUE(program.indirect_cbufs);
  static std::array<float, 17 * gpu::gcn::kCbufDwords> constants{};
  alignas(65536) static std::array<std::array<u8, 65536>, 3> targets{};
  for (u32 frame = 0; frame < 3; ++frame) {
    constants[16 * gpu::gcn::kCbufDwords] = frame == 1 ? .5f : -.5f;
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.prim_type = 1;
    draw.vertex_count = 3;
    for (const auto& cb : program.vs_cbufs) {
      draw.cbufs[cb.binding] = {
          reinterpret_cast<u64>(constants.data() + cb.first_dword),
          cb.num_dwords * 4};
      draw.num_cbufs = std::max(draw.num_cbufs, cb.binding + 1);
    }
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    const u32 expected_x = frame == 1 ? 12 : 4;
    EXPECT_EQ(pixels[(8 * 16 + expected_x) * 4], 255) << "frame " << frame;
    EXPECT_EQ(pixels[(8 * 16 + (16 - expected_x)) * 4], 0) << "frame " << frame;
  }
}

TEST(VkDraw, SplitNggUserWindowsAndHighPixelRegistersRemainDistinct) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 es[64] = {
      0x34000a83,              // v0 = vertex ID * 8
      0x7e0c0208,              // v6 = ES user data in s8
      0xd8340000, 0x00000500,  // LDS[v0] = vertex ID
      0xd8340004, 0x00000600,  // LDS[v0+4] = ES offset
      0xbe802006};
  const u32 gs[64] = {
      0x34000a83,              // address matching the ES record
      0xd8d80000, 0x02000000,
      0xd8d80004, 0x06000000,
      0x7e000d02, 0x100000f0, 0x060000f1,
      0x06000000, 0x06000106,  // x += GS user s0 + offset written by ES
      0x7e0202f1, 0x7e0602f0, 0x7d840481, 0x02020701,
      0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100,
      0x7e0802ff, 0x00200400, 0xbefe0481, 0xf8000941, 0x00000004,
      0xbefe04c1, 0xbefc03ff, 0x00001003, 0xbf900009, 0xbf810000};
  const u32 ps[64] = {
      0x7e00021d,              // red = user s29, beyond the old push window
      0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
  u32 user_data[32]{};
  user_data[0] = 0x3e800000; // ES contributes +.25; GS changes independently
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128};
  static const auto program = gpu::rdna::Recompile(es, ps, user_data, user_data,
                                            0, false, 4, 30, nullptr, 0, &cfg);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<std::array<u8, 65536>, 2> targets{};
  for (u32 frame = 0; frame < 2; ++frame) {
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.prim_type = 1;
    draw.vertex_count = 3;
    draw.vs_user_data[0] = user_data[0];
    draw.gs_user_data_addr = frame ? 0xbe800000 : 0x3e800000;
    draw.ps_user_data[29] = frame ? 0x3f000000 : 0x3f800000;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    EXPECT_NEAR(pixels[(8 * 16 + (frame ? 8 : 12)) * 4],
                 frame ? 128 : 255, 1);
    EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 0);
  }
}

TEST(VkDraw, GeometryResourceReplayUsesSystemUserDataAddress) {
  // The vertex user window contains a V#, while s0:s1 names a different table.
  const u32 code[64] = {0xf4080200, 0xfa000000, // s_load_dwordx4 s8, s0, 0
                        0xf4200304, 0xfa000000, // s_buffer_load s12, s8, 0
                        0xbf810000};
  static const u32 values[2] = {0x12345678, 0x87654321};
  const u64 base = reinterpret_cast<u64>(values);
  const u32 descriptor[4] = {static_cast<u32>(base),
                            static_cast<u32>(base >> 32) | (4u << 16),
                            2, 0x0004dfac};
  const u32 user_data[4] = {0, 0, 0, 0};
  const auto resources = gpu::rdna::ResolveBuffers(code, user_data, 4, 8,
      5, reinterpret_cast<u64>(descriptor));
  ASSERT_TRUE(resources.count(0));
  ASSERT_TRUE(resources.count(2));
  EXPECT_EQ(resources.at(0).base, reinterpret_cast<u64>(descriptor));
  EXPECT_EQ(resources.at(2).base, base);
  EXPECT_TRUE(resources.at(2).descriptor_valid);
}

TEST(VkDraw, VertexAndPixelUserDataPastSixteenDoNotOverlap) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {0x7e00021f, 0x7e0202ff, 0x3d800000,
                      0x7e040280, 0x7e0602f2,
                      0xf80008cf, 0x03020100, 0xbf810000}; // x = s31 (user23)
  const u32 ps[64] = {0x7e00021d, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000}; // red = s29
  const u32 user_data[32]{};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data,
                                            0, false, 24, 30);
  ASSERT_TRUE(program.ok);
  ASSERT_TRUE(program.indirect_cbufs);
  ASSERT_TRUE(program.mesh_spirv.empty());
  alignas(65536) static std::array<std::array<u8, 65536>, 2> targets{};
  for (u32 frame = 0; frame < 2; ++frame) {
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.vs_user_data[23] = frame ? 0xbee00000 : 0x3f100000;
    draw.ps_user_data[29] = frame ? 0x3f000000 : 0x3f800000;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    // Place the point at a pixel center, avoiding rasterization edge ties.
    const u32 x = frame ? 4 : 12;
    EXPECT_NEAR(pixels[(7 * 16 + x) * 4], frame ? 128 : 255, 1);
    EXPECT_EQ(pixels[(7 * 16 + (16 - x)) * 4], 0);
  }
}

TEST(VkDraw, PixelShaderUsesDppOperand) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {0x7e0002ff, 0x3d800000, 0x7e040280, 0x7e0602f2,
                      0xf80008cf, 0x03020000, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, // v0 = 1
                      0x7e0002fa, 0xff00e400, // v0 = quad_perm[0,1,2,3](v0)
                      0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data,
                                                  0, false, 0, 0);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  EXPECT_EQ(pixels[(7 * 16 + 8) * 4], 255);
}

TEST(VkDraw, SdwaAluWritesSelectedBytesAndPreservesOtherBits) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  struct Case { u32 select, unused, input, expected; };
  // Convert 1234.75 to 1234 (0x4d2), or -1.75 to -1. Expected packed
  // values independently check placement, padding, sign extension and merging.
  const Case cases[] = {
      {0, 0, 0x449a5800, 0x000000d2}, {1, 0, 0x449a5800, 0x0000d200},
      {2, 0, 0x449a5800, 0x00d20000}, {3, 0, 0x449a5800, 0xd2000000},
      {4, 0, 0x449a5800, 0x000004d2}, {5, 0, 0x449a5800, 0x04d20000},
      {0, 2, 0x449a5800, 0xabcdefd2}, {1, 2, 0x449a5800, 0xabcdd212},
      {2, 2, 0x449a5800, 0xabd2ef12}, {3, 2, 0x449a5800, 0xd2cdef12},
      {4, 2, 0x449a5800, 0xabcd04d2}, {5, 2, 0x449a5800, 0x04d2ef12},
      {0, 1, 0xbfe00000, 0xffffffff}, {1, 1, 0xbfe00000, 0xffffff00},
      {2, 1, 0xbfe00000, 0xffff0000}, {3, 1, 0xbfe00000, 0xff000000},
      {4, 1, 0xbfe00000, 0xffffffff}, {5, 1, 0xbfe00000, 0xffff0000},
      {0, 1, 0x4301c000, 0xffffff81}, {4, 1, 0x449a5800, 0x000004d2}};
  const u32 vs[64] = {0x7e0002ff, 0x3d800000, 0x7e040280, 0x7e0602f2,
                      0xf80008cf, 0x03020000, 0xbf810000};
  static std::array<gpu::gcn::Recompiled, std::size(cases) * 2> programs;
  alignas(65536) static std::array<std::array<u8, 65536>, std::size(cases) * 2> targets{};
  for (u32 i = 0; i < std::size(programs); ++i) {
    const auto& c = cases[i % std::size(cases)];
    const bool binary = i >= std::size(cases);
    SCOPED_TRACE(i);
    u32 ps[64] = {0x7e000200, 0x7e020201}; // old bits and float input
    u32 pc = 2;
    if (binary) {
      ps[pc++] = 0x7e021101; // convert v1 to integer before the binary op
      ps[pc++] = 0x7e040280; // v2 = 0
    }
    ps[pc++] = binary ? 0x3a0004f9 : 0x7e0010f9; // XOR or CVT with SDWA
    ps[pc++] = 0x00060001 | (c.select << 8) | (c.unused << 11) |
               (binary ? 0x06000000 : 0);
    const u32 tail[] = {
        0x7d840002,              // compare packed integer result with s2
        0x7e0402f2, 0x02000480,  // red = result matches ? 1 : 0
        0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
    std::copy_n(tail, std::size(tail), ps + pc);
    const u32 user_data[32] = {0xabcdef12, c.input, c.expected};
    programs[i] = gpu::rdna::Recompile(vs, ps, user_data, user_data,
                                       0, false, 0, 3);
    ASSERT_TRUE(programs[i].ok);
    gpu::rhi::DrawInfo draw;
    draw.recomp = &programs[i];
    draw.prim_type = 1;
    draw.vertex_count = 1;
    std::copy_n(user_data, 32, draw.ps_user_data);
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[i].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    EXPECT_EQ(pixels[(7 * 16 + 8) * 4], 255);
  }
}

TEST(VkDraw, NggWavesKeepIndependentBranchesAndFullBallots) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 es[64] = {0x34000a82, 0xd8340000, 0x00000500, 0xbe802006};
  // Two guest waves take different scalar branches. Their triangles use
  // vertices in the upper half of each wave and a ballot spanning both halves.
  const u32 gs[128] = {
      0x9380ff03, 0x00040018,  // s0 = wave index from merged_wave_info
      0xbf068000,              // s_cmp_eq_u32 s0, 0
      0xbf850003,              // wave zero jumps to the negative offset
      0xbe8403f0,              // s4 = .5
      0xbf820002,
      0xbf800000,
      0xbe8403f1,              // s4 = -.5
      0xbf800000,              // join
      0xbf8a0000,              // both waves must reach the same barrier
      0xd8d80000, 0x02000000,  // v2 = ES vertex ID
      0x360404ff, 63,          // v2 &= 63
      0x7d840481,              // ballot lane == 1
      0xbe86046a,              // s[6:7] = vcc
      0x7d8404a3,              // ballot lane == 35
      0x88ea066a,              // vcc |= s[6:7]
      0xd7650008, 0x0001006a,  // mbcnt_lo(vcc_lo, 0)
      0xd7660008, 0x0002106b,  // mbcnt_hi(vcc_hi, v8)
      0x7e100d08,              // v8 = float(rank), must be 1 in lanes 32..34
      0x061010f3,              // rank correction = rank - 1
      0x7e000d02,              // v0 = float(lane)
      0x060000ff, 0xc2000000,  // v0 -= 32
      0x100000ff, 0x3ecccccd,  // v0 *= .4
      0x060000ff, 0xbecccccd,  // v0 -= .4
      0x06000004,              // v0 += wave's scalar offset
      0x06000108,              // v0 += rank correction
      0x7e0202f1, 0x7e0602f0,  // y = -.5 / .5
      0x7d8404a1, 0x02020701,  // y = lane == 33 ? .5 : -.5
      0x7e040280, 0x7e0602f2,
      0xf80008cf, 0x03020100,
      0x7e0802ff, 0x02208420,  // triangle 32, 33, 34
      0x7e0c02ff, 0x06218460,  // triangle 96, 97, 98
      0x7d840a81,              // ES vertex ID v5 == 1
      0x02080d04,              // choose connectivity for output primitive 1
      0xf8000941, 0x00000004,
      0xbefc03ff, 0x00002063,  // 99 vertices, 2 primitives
      0xbf900009, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 128, 99, 128, 2, 128};
  static const auto program = gpu::rdna::Recompile(es, ps, user_data, user_data,
                                            0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 99;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 12) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 0);
}

TEST(VkDraw, FetchedVerticesKeepNggVertexAndInstanceIds) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // Read the launch IDs before an inline fetch overwrites v5. Two vertices
  // in two instances should cover four separate points, one per quadrant.
  const u32 vs[64] = {
      0x7e280d05, 0x7e2a0d08,  // v20=float(v5), v21=float(v8)
      0xf4080404, 0,          // s_load_dwordx4 s[16:19], s[8:9], 0
      0xe0002000, 0x80040505, // buffer_load_format_x v5, v5, s[16:19], 0
      0x062828f1, 0x062a2af1, // v20+=-0.5, v21+=-0.5
      0x7e2c0280,             // v22=0
      0xf80008cf, 0x05161514, // position=(v20,v21,v22,v5)
      0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const std::array<float, 2> vertices = {1.f, 1.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[4] = {u32(vb), u32(vb >> 32) | (4u << 16), 2,
                            0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  ASSERT_EQ(program.attrs.size(), 1u);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  draw.prim_type = 1;
  draw.vertex_count = draw.instance_count = 2;
  draw.num_vbufs = draw.num_vattrs = 1;
  draw.vertex_data = vertices.data();
  draw.vertex_stride = 4;
  draw.vbufs[0] = {vertices.data(), 4, 2};
  draw.vattrs[0] = {0, 0, 0, 1, 4, 7}; // R32_FLOAT
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  std::array<u32, 4> quadrants{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++quadrants[(y / 8) * 2 + x / 8];
  EXPECT_EQ(quadrants, (std::array<u32, 4>{1, 1, 1, 1}));
}

TEST(VkDraw, IndexedRawVerticesUseLoadedDescriptorStride) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {
      0xf4080404, 0xfa000000,  // s[16:19] = descriptor from s[8:9]
      0xe0302000, 0x80040005,  // v0 = buffer[vertex ID], using descriptor stride
      0x7e020280, 0x7e040280, 0x7e0602f2,
      0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const std::array<float, 4> vertices = {-.5f, 99.f, .5f, 99.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[4] = {u32(vb), u32(vb >> 32) | (8u << 16), 2,
                            0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  gpu::rdna::NextProgramGeneration();
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  std::copy_n(user_data, 32, draw.vs_user_data);
  draw.prim_type = 1;
  draw.vertex_count = 2;
  draw.bufs[0] = {vb, sizeof(vertices)};
  draw.num_bufs = 1;
  draw.cbufs[0] = {table, sizeof(descriptor)};
  draw.num_cbufs = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  std::array<u32, 2> halves{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++halves[x / 8];
  EXPECT_EQ(halves, (std::array<u32, 2>{1, 1}));
}

TEST(VkDraw, Ps5DepthClearUsesRegisterValueInsteadOfVertexDepth) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(256) static const u32 vs[4096] = {
      0x7e000280, 0x7e020280, 0x7e0402ff, 0x3f400000,
      0x7e0602f2, 0xf80008cf, 0x03020100, 0xbf810000};
  alignas(256) static const u32 ps[4096] = {
      0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
  alignas(65536) static std::array<u8, 65536> target{}, depth{};
  using namespace gpu::ps5;
  Regs regs;
  const auto address = [&](u32 reg, const void* ptr) {
    const u64 value = reinterpret_cast<u64>(ptr);
    regs[reg] = u32(value >> 8);
    regs[reg + 1] = u32(value >> 40);
  };
  address(mmSPI_SHADER_PGM_LO_ES, vs);
  address(mmSPI_SHADER_PGM_LO_PS, ps);
  regs[mmVGT_PRIMITIVE_TYPE] = 1;
  regs[mmCB_TARGET_MASK] = 15;
  regs[mmCB_COLOR0_BASE] = u32(reinterpret_cast<u64>(target.data()) >> 8);
  regs[mmCB_COLOR0_BASE_EXT] = u32(reinterpret_cast<u64>(target.data()) >> 40);
  regs[mmCB_COLOR0_INFO] = 10u << 2;
  regs[mmCB_COLOR0_ATTRIB2] = (15u << 14) | 15;
  regs[mmPA_CL_CLIP_CNTL] = 1u << 19;
  regs[mmPA_CL_VPORT_XSCALE] = regs[mmPA_CL_VPORT_XOFFSET] = std::bit_cast<u32>(8.f);
  regs[mmPA_CL_VPORT_YSCALE] = std::bit_cast<u32>(-8.f);
  regs[mmPA_CL_VPORT_YOFFSET] = std::bit_cast<u32>(8.f);
  regs[mmDB_Z_INFO] = 3;
  regs[mmDB_Z_WRITE_BASE] = u32(reinterpret_cast<u64>(depth.data()) >> 8);
  regs[mmDB_Z_WRITE_BASE_HI] = u32(reinterpret_cast<u64>(depth.data()) >> 40);
  regs[mmDB_DEPTH_CLEAR] = std::bit_cast<u32>(1.f);
  regs[mmDB_DEPTH_CONTROL] = (7u << 4) | 6;  // ALWAYS, test and write
  regs[mmDB_RENDER_CONTROL] = 1;
  const u32 body[] = {1, 0};
  const DrawPacket packet{0x2d, body, 2};  // DRAW_INDEX_AUTO
  gpu::rhi::DrawInfo clear;
  ASSERT_TRUE(BuildDrawInfo(regs, packet, clear));
  ASSERT_TRUE(clear.depth_clear_draw);
  // Skyrim retains its previous postprocess resources during the clear.
  // They must not turn the clear into a rejected depth-feedback draw.
  clear.tex_base = clear.depth_base;
  clear.num_texs = 1;
  clear.texs[0].base = clear.depth_base;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, clear));
  regs[mmDB_RENDER_CONTROL] = 0;
  regs[mmDB_DEPTH_CONTROL] = (1u << 4) | 6;  // LESS: .75 passes only after clear=1
  gpu::rhi::DrawInfo point;
  ASSERT_TRUE(BuildDrawInfo(regs, packet, point));
  ASSERT_FALSE(point.depth_clear_draw);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, point));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, point.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  u32 red = 0;
  for (u32 i = 0; i < 16 * 16; ++i)
    red += pixels[i * 4] == 255;
  EXPECT_EQ(red, 1u);
}

TEST(VkDraw, IndexedTypedLoadUsesLiveStrideAndInstructionFormat) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // Inline V# at s[8:11], indexed by v5. The instruction requests float4
  // even though the descriptor advertises R32_UINT. Padding must not become
  // vertex data, and changing the stride must work with the same module.
  const u32 vs[64] = {0xea6b2000, 0x80020005,
                      0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  std::array<float, 32> vertices;
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  u32 user_data[32] = {u32(vb), u32(vb >> 32) | (32u << 16), 2,
                       0x21014fac};
  gpu::rdna::NextProgramGeneration();
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  ASSERT_TRUE(program.attrs.empty());
  ASSERT_EQ(program.vs_bufs.size(), 1u);
  alignas(65536) static std::array<std::array<u8, 65536>, 2> targets{};
  for (u32 pass = 0; pass < 2; ++pass) {
    const u32 stride = pass ? 48 : 32;
    SCOPED_TRACE(stride);
    vertices.fill(99.f);
    const std::array<float, 4> left = {-.5f, 0.f, 0.f, 1.f};
    const std::array<float, 4> right = {.5f, 0.f, 0.f, 1.f};
    std::copy(left.begin(), left.end(), vertices.begin());
    std::copy(right.begin(), right.end(), vertices.begin() + stride / 4);
    user_data[1] = u32(vb >> 32) | (stride << 16);
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    std::copy_n(user_data, 32, draw.vs_user_data);
    draw.prim_type = 1;
    draw.vertex_count = 2;
    draw.bufs[0] = {vb, sizeof(vertices)};
    draw.num_bufs = 1;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[pass].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    std::array<u32, 2> halves{};
    for (u32 y = 0; y < 16; ++y)
      for (u32 x = 0; x < 16; ++x)
        if (pixels[(y * 16 + x) * 4] == 255)
          ++halves[x / 8];
    EXPECT_EQ(halves, (std::array<u32, 2>{1, 1}));
  }
}

TEST(VkDraw, BulkDescriptorLoadPreservesInteriorBufferStride) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {
      0xf4100204, 0xfa000000,  // s[8:23] = four descriptors from s[8:9]
      0xf4200804, 0xfa000000,  // a scalar load uses the first descriptor
      0xe0302000, 0x80040005,  // v0 = buffer[vertex ID], using descriptor stride
      0x7e020280, 0x7e040280, 0x7e0602f2,
      0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  const std::array<float, 4> vertices = {-.5f, 99.f, .5f, 99.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[16] = {
      u32(vb), u32(vb >> 32) | (8u << 16), 2, 0x21014fac,
      0, 0, 0, 0,
      u32(vb), u32(vb >> 32) | (8u << 16), 2, 0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  gpu::rdna::NextProgramGeneration();
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  std::copy_n(user_data, 32, draw.vs_user_data);
  draw.prim_type = 1;
  draw.vertex_count = 2;
  draw.bufs[0] = {vb, sizeof(vertices)};
  draw.num_bufs = 1;
  const auto resources = gpu::rdna::ResolveBuffers(vs, user_data, 32, 8);
  for (const auto& cb : program.vs_cbufs) {
    ASSERT_TRUE(resources.count(cb.use_pc));
    draw.cbufs[cb.binding] = {resources.at(cb.use_pc).base + cb.first_dword * 4,
                              cb.num_dwords * 4};
    draw.num_cbufs = std::max(draw.num_cbufs, cb.binding + 1);
  }
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  std::array<u32, 2> halves{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++halves[x / 8];
  EXPECT_EQ(halves, (std::array<u32, 2>{1, 1}));
}

TEST(VkDraw, PassthroughInterpolationPreservesPackedVertexValues) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer) || !gpu::vk::g_dev.barycentric_available)
    GTEST_SKIP() << "Fragment barycentrics are required";
  static const u32 vs[4096] = {
      0xe0382000, 0x80020005,  // position.xyz + packed attribute, indexed by v5
      0x7e0802f2, 0xf80008cf, 0x04020100,
      0xf8000201, 0x00000003, 0xbf810000};
  static const u32 ps[4096] = {
      0xc8020002, 0xc8060000, 0xc80a0001,  // P0, P10, P20
      0x7e0602f2, 0x7e0e0280,
      0x7d840000, 0x02080680,  // v4 = first vertex matches s0
      0x7d840201, 0x020a0680,  // v5 = second vertex matches s1
      0x7d840402, 0x020c0680,  // v6 = third vertex matches s2
      0xf800080f, 0x03060504, 0xbf810000};
  struct Vertex { float x, y, z; u32 packed; };
  const Vertex vertices[] = {{-.75f, -.75f, 0, 0x3f800123},
                            {.75f, -.75f, 0, 0x7fc0ffff},
                            {0, .75f, 0, 0xabcdef12}};
  const u64 vb = reinterpret_cast<u64>(vertices);
  const u32 vs_ud[32] = {u32(vb), u32(vb >> 32) | (16u << 16), 3, 0x21014fac};
  const u32 ps_ud[32] = {vertices[0].packed, vertices[1].packed, vertices[2].packed};
  alignas(65536) static std::array<std::array<u8, 65536>, 2> targets{};
  const gpu::gcn::Recompiled* previous = nullptr;
  for (u32 passthrough = 0; passthrough < 2; ++passthrough) {
    const u32 input_control[32] = {passthrough ? 0x420u : 0u};
    const auto& program = gpu::ps5::GetGraphicsShader({
        .vs_addr = reinterpret_cast<u64>(vs), .ps_addr = reinterpret_cast<u64>(ps),
        .vs_user_sgprs = 4, .ps_user_sgprs = 3,
        .ps_in_cntl = input_control, .ps_num_interp = 1,
        .vs_user_data = vs_ud, .ps_user_data = ps_ud});
    ASSERT_TRUE(program.ok);
    if (previous)
      EXPECT_NE(&program, previous) << "Interpolation mode changes the shader";
    previous = &program;
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    std::copy_n(vs_ud, 32, draw.vs_user_data);
    std::copy_n(ps_ud, 32, draw.ps_user_data);
    draw.prim_type = 4;
    draw.vertex_count = 3;
    draw.bufs[0] = {vb, sizeof(vertices)};
    draw.num_bufs = 1;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[passthrough].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 3], 255);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 1], passthrough ? 255 : 0);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 2], passthrough ? 255 : 0);
  }
}

TEST(VkDraw, PixelTypedLoadConvertsPackedDataWithByteOffsets) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {0x7e000280, 0x7e0202f2,
                      0xf80008cf, 0x01000000, 0xbf810000};
  const u32 ps[64] = {
      0x7e000284,                        // v0 = byte offset 4
      0xe8001004 | (56u << 19) | (3u << 16), 0x88020000,
      // tbuffer_load_format_xyzw v[0:3], v0, s[8:11], 8 offen offset:4
      0xf800080f, 0x03020100, 0xbf810000};
  const u32 data[8] = {0, 0, 0, 0, 0xffff8040, 0, 0, 0};
  const u64 base = reinterpret_cast<u64>(data);
  const u32 user_data[32] = {u32(base), u32(base >> 32), sizeof(data), 0x14fac};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  ASSERT_EQ(program.ps_bufs.size(), 1u);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  std::copy_n(user_data, 32, draw.ps_user_data);
  draw.prim_type = draw.vertex_count = 1;
  draw.bufs[0] = {base, sizeof(data)};
  draw.num_bufs = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  u32 colored = 0;
  for (u32 i = 0; i < 256; ++i) {
    if (!pixels[i * 4 + 2])
      continue;
    ++colored;
    EXPECT_EQ(pixels[i * 4], 64);
    EXPECT_EQ(pixels[i * 4 + 1], 128);
    EXPECT_EQ(pixels[i * 4 + 2], 255);
    EXPECT_EQ(pixels[i * 4 + 3], 255);
  }
  EXPECT_EQ(colored, 1u);
}

TEST(VkDraw, PixelOneDimensionalSampleSurvivesAddtidSpill) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {0x7e000280, 0x7e0202f2,
                      0xf80008cf, 0x01000000, 0xbf810000};
  const u32 ps[64] = {
      0x7e0002f0, 0xf09c0f00, 0x00400400,  // sample_lz 1D at x=.5
      0xbefc03ff, 256,                    // M0 = 256
      0xdac00000, 0x00000400,              // spill red at M0 + lane*4
      0xbefc0380,                         // M0 = 0
      0xdac40100, 0x00000000,              // reload with offset:256
      0xf800080f, 0x07060500, 0xbf810000};
  const u32 user_data[32]{};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> texture{64, 128, 255, 255}, target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  draw.prim_type = draw.vertex_count = 1;
  draw.num_texs = 1;
  auto& tex = draw.texs[0];
  tex.base = reinterpret_cast<u64>(texture.data());
  tex.w = tex.h = tex.pitch = 1;
  tex.dfmt = 10;
  tex.tiling = 256;  // RDNA linear
  tex.is_1d = true;
  draw.tex_base = tex.base;
  draw.tex_w = draw.tex_h = draw.tex_pitch = 1;
  draw.tex_dfmt = 10;
  draw.tex_tiling = 256;
  draw.tex_is_1d = true;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  u32 colored = 0;
  for (u32 i = 0; i < 256; ++i) {
    if (!pixels[i * 4 + 2])
      continue;
    ++colored;
    EXPECT_EQ(pixels[i * 4], 64);
    EXPECT_EQ(pixels[i * 4 + 1], 128);
    EXPECT_EQ(pixels[i * 4 + 2], 255);
  }
  EXPECT_EQ(colored, 1u);
}

TEST(VkDraw, RawVertexBufferReadsPastOneMiB) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[64] = {
      0xd5690000, 0x00020aff, 0x001fffff,  // v0 = vertex ID * (2M - 1)
      0x4a000081,                        // v0 += 1
      0xe0302000, 0x80020000,  // v0 = raw buffer s[8:11][v0]
      0x7e020280, 0x7e040280, 0x7e0602f2,
      0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                      0xf800080f, 0x00010100, 0xbf810000};
  std::vector<float> vertices(4 * 1024 * 1024);
  const u32 indices[] = {1, 2 * 1024 * 1024, u32(vertices.size() - 1)};
  const float positions[] = {-.5625f, .0625f, .5625f};
  for (u32 i = 0; i < 3; ++i)
    vertices[indices[i]] = positions[i];
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 user_data[32] = {
      u32(vb), u32(vb >> 32) | (4u << 16), u32(vertices.size()), 0x21014fac};
  gpu::rdna::NextProgramGeneration();
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<u8, 65536> target{};
  gpu::rhi::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  std::copy_n(user_data, 32, draw.vs_user_data);
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.bufs[0] = {vb, u32(vertices.size() * sizeof(float))};
  draw.num_bufs = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::rhi::BeginFrame(renderer);
  ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::vk::g_frame.slot_idx;
  gpu::rhi::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::vk::g_frame.slots[slot_index];
  ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                           VK_TRUE, UINT64_MAX), VK_SUCCESS);
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = static_cast<const u8*>(slot.readback_map);
  for (u32 x : {3u, 8u, 12u}) {
    u32 colored = 0;
    for (u32 y = 0; y < 16; ++y)
      colored += pixels[(y * 16 + x) * 4] == 255;
    EXPECT_EQ(colored, 1u) << "vertex at pixel x=" << x;
  }
}

TEST(VkDraw, SinglePointRendersWithAndWithoutIndices) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // A procedural point at the origin, shaded opaque red.
  const u32 vs[64] = {0x7e000280, 0x7e0202f2,
                       0xf80008cf, 0x01000000, 0xbf810000};
  const u32 ps[64] = {0x7e0002f2, 0x7e020280,
                       0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32] = {};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<std::array<u8, 65536>, 2> targets{};
  const u16 index = 0;
  for (u32 indexed = 0; indexed < 2; ++indexed) {
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    draw.prim_type = 1;
    draw.vertex_count = indexed ? 0 : 1;
    draw.index_data = indexed ? &index : nullptr;
    draw.index_count = indexed ? 1 : 0;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[indexed].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;  // RGBA8_UNORM
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw)) << "indexed=" << indexed;
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    ASSERT_NE(slot.readback_map, nullptr);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    u32 colored = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (pixels[i * 4] || pixels[i * 4 + 1] || pixels[i * 4 + 2])
        ++colored;
    EXPECT_EQ(colored, 1u) << "indexed=" << indexed;
    EXPECT_EQ(gpu::vk::g_frame.heuristic, 0u);

    // Read the point through a compute image, swap red/green, and publish it
    // back into the same live render target. Guest bytes still contain zeros:
    // this must bridge the actual GPU image in both directions.
    alignas(256) static std::array<u32, 4096> cs{};
    const std::array<u32, 10> code = {
        0x7e020287,               // v1 = row 7
        0xf0000f08, 0x00000400,   // image_load v[4:7], v[0:1], s[0:7]
        0x7e100304, 0x7e080305, 0x7e0a0308,  // swap v4/v5 through v8
        0xf0200f08, 0x00000400,   // image_store v[4:7], v[0:1], s[0:7]
        0xbf810000, 0};
    std::copy(code.begin(), code.end(), cs.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::NoteGpuPool(draw.rt_base, targets[indexed].size());
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(cs.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 16;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud] = draw.rt_base >> 8;
    regs[ud + 1] = ((draw.rt_base >> 40) & 0xff) | (56u << 20) | (3u << 30);
    regs[ud + 2] = 3 | (15u << 14);
    regs[ud + 3] = 0x90000fac;  // 2D, LINEAR, RGBA8_UNORM
    gpu::rhi::BeginFrame(renderer);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    const u32 compute_slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& compute_slot = gpu::vk::g_frame.slots[compute_slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &compute_slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(compute_slot.presentable);
    const auto* result = static_cast<const u8*>(compute_slot.readback_map);
    u32 green = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (result[i * 4 + 1] == 255 && result[i * 4] == 0 &&
          result[i * 4 + 2] == 0)
        ++green;
    EXPECT_EQ(green, 1u) << "indexed=" << indexed;
  }
}

TEST(VkDraw, NarrowRenderTargetsRoundTripThroughCompute) {
  utl::initOptions();
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // A half-red point in an R8/R16 target, whose guest bytes remain zero.
  const u32 vs[64] = {0x7e000280, 0x7e0202f2,
                       0xf80008cf, 0x01000000, 0xbf810000};
  const u32 ps[64] = {0x7e0002f0, 0x7e020280,
                       0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32] = {};
  static const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  alignas(65536) static std::array<std::array<u8, 65536>, 3> targets{};
  for (u32 format_index = 0; format_index < 3; ++format_index) {
    gpu::rhi::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.index_data = nullptr;
    draw.index_count = 0;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[format_index].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = ((format_index ? 2u : 1u) << 2) |
                       (format_index == 2 ? 7u << 8 : 0);  // R8/R16_UNORM, R16_FLOAT
    gpu::rhi::BeginFrame(renderer);
    ASSERT_TRUE(gpu::vk::DrawRecomp(renderer, draw)) << "format=" << format_index;
    const u32 slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::vk::g_frame.slots[slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(slot.presentable);
    ASSERT_NE(slot.readback_map, nullptr);
    const auto* pixels = static_cast<const u8*>(slot.readback_map);
    const auto red = [&](const u8* data, u32 i) -> u32 {
      if (!format_index) return data[i];
      u16 value;
      std::memcpy(&value, data + i * 2, 2);
      return value;
    };
    const u32 half = format_index == 2 ? 0x3800 : format_index ? 32768 : 128;
    const u32 quarter = format_index == 2 ? 0x3400 : half / 2;
    u32 colored = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (red(pixels, i) >= half - 1 && red(pixels, i) <= half)
        ++colored;
    EXPECT_EQ(colored, 1u) << "format=" << format_index;
    EXPECT_EQ(gpu::vk::g_frame.heuristic, 0u);

    // Halve the live red channel in compute, then present the modified image.
    alignas(256) static std::array<u32, 4096> cs{};
    const std::array<u32, 7> code = {
        0x7e020287,               // v1 = row 7
        0xf0000108, 0x00000400,   // image_load v4, v[0:1], s[0:7]
        0x100808f0,              // v_mul_f32 v4, 0.5, v4
        0xf0200108, 0x00000400,   // image_store v4, v[0:1], s[0:7]
        0xbf810000};
    std::copy(code.begin(), code.end(), cs.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::NoteGpuPool(draw.rt_base, targets[format_index].size());
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(cs.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 16;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud] = draw.rt_base >> 8;
    regs[ud + 1] = ((draw.rt_base >> 40) & 0xff) | ((format_index == 2 ? 13u : format_index ? 7u : 1u) << 20) | (3u << 30);
    regs[ud + 2] = 3 | (15u << 14);
    regs[ud + 3] = 0x90000fac;  // 2D, LINEAR
    gpu::rhi::BeginFrame(renderer);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    const u32 compute_slot_index = gpu::vk::g_frame.slot_idx;
    gpu::rhi::EndFrame(renderer, draw.rt_base);
    const auto& compute_slot = gpu::vk::g_frame.slots[compute_slot_index];
    ASSERT_EQ(vkWaitForFences(gpu::vk::g_dev.device, 1, &compute_slot.fence,
                             VK_TRUE, UINT64_MAX), VK_SUCCESS);
    ASSERT_TRUE(compute_slot.presentable);
    const auto* result = static_cast<const u8*>(compute_slot.readback_map);
    u32 quarter_red = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (red(result, i) >= quarter - 1 && red(result, i) <= quarter)
        ++quarter_red;
    EXPECT_EQ(quarter_red, 1u) << "format=" << format_index;
  }
}
