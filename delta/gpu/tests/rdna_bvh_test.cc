#include <array>
#include <bit>
#include <limits>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/shader_cache.h"

namespace {
constexpr u32 kInvalid = 0xffffffff;
u32 Bits(float value) {
  return std::bit_cast<u32>(value);
}
float Float(u32 value) {
  return std::bit_cast<float>(value);
}

class RdnaBvh : public testing::Test {
 protected:
  void SetUp() override {
    if (!gpu::rhi::Init(gpu::rhi::DefaultRenderer()))
      GTEST_SKIP() << "A Vulkan device is required";
  }
  // Every dispatch owns distinct guest memory; the renderer can cache inputs.
  std::array<u32, 4> Run(const std::array<u32, 48>& nodes,
                         u32 ptr,
                         std::array<float, 3> origin = {0, 0, 0},
                         std::array<float, 3> direction = {0, 0, 1},
                         bool sort = true,
                         bool wide = false,
                         bool nsa = false,
                         bool a16 = false,
                         u32 last_node = 2,
                         u32 high = 0,
                         float extent = 100,
                         u32 grow = 0,
                         u32 extra_bindings = 0,
                         std::span<const u32> epilogue = {},
                         bool runtime = false,
                         bool changing = false,
                         bool indirect = false) {
    alignas(65536) static std::array<std::array<u32, 16384>, 256> memory{};
    static u32 allocation = 0;
    auto& source = memory.at(allocation++);
    auto& output = memory.at(allocation++);
    std::copy(nodes.begin(), nodes.end(), source.begin());
    output.fill(0xa5a5a5a5);
    if (changing) {
      std::copy(nodes.begin(), nodes.end(), source.begin() + 64);
      for (u32 i = 0; i < 4; ++i)
        source[64 + 16 + i] += 0x100;
    }
    if (allocation == 2)
      gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()),
                            sizeof(memory));
    alignas(256) static std::array<u32, 4096> code{};
    code.fill(0);
    std::array<u32, 12> args{};
    const u32 shift = wide ? 1 : 0;
    args[0] = ptr;
    if (wide)
      args[1] = high;
    args[1 + shift] = Bits(extent);
    for (u32 i = 0; i < 3; i++) {
      args[2 + shift + i] = Bits(origin[i]);
      args[5 + shift + i] = Bits(direction[i]);
      args[8 + shift + i] = Bits(1.f / direction[i]);
    }
    if (a16) {
      // A16 is used below with direction (0,0,1); packed inverse is
      // (inf,inf,1).
      EXPECT_EQ(direction, (std::array<float, 3>{0, 0, 1}));
      args[5 + shift] = 0;
      args[6 + shift] = 0x7c003c00;
      args[7 + shift] = 0x3c007c00;
    }
    constexpr std::array<u32, 12> registers = {0,  61, 65, 66, 68, 62,
                                               64, 67, 69, 70, 71, 72};
    const u32 count = (a16 ? 8 : 11) + shift;
    u32 pc = 0;
    if (extra_bindings) {
      code[pc++] = 0xbe880404;  // s_mov_b64 s[8:9], s[4:5]
      code[pc++] = 0xbe8a0406;  // s_mov_b64 s[10:11], s[6:7]
      for (u32 i = 0; i < extra_bindings; i++) {
        code[pc++] = 0xbe880304;  // new descriptor version in s8
        code[pc++] = 0xe0300000;
        code[pc++] = 0x80026400;  // buffer_load_dword v100, off, s[8:11], 0
      }
    }
    if (changing) {
      code[pc++] = 0xbe8c0380;  // s12 = output byte offset
      code[pc++] = 0xbe8e0382;  // s14 = loop count
    }
    const u32 loop_pc = pc;
    if (indirect) {
      code[pc++] = 0xf4000004;  // s_load_dword s0, s[8:9], 0
      code[pc++] = 0xfa000000;
    }
    for (u32 i = 0; i < count; i++) {
      code[pc++] = 0x7e0002ff | ((nsa ? registers[i] : i) << 17);
      code[pc++] = args[i];
    }
    if (runtime) {
      code[pc++] = 0x7ec80200;  // v_mov_b32 v100, s0
      code[pc++] = 0x7e000564;  // v_readfirstlane_b32 s0, v100
    }
    const u32 bvh_pc = pc;
    code[pc++] = (wide ? 0xf19c9f01 : 0xf1989f01) | (nsa ? 6 : 0);
    code[pc++] = a16 ? 1u << 30 : 0;
    if (nsa) {
      for (u32 i = 1; i < count; i++)
        code[pc + (i - 1) / 4] |= registers[i] << (((i - 1) % 4) * 8);
      pc += 3;
    }
    for (u32 word : epilogue)
      code[pc++] = word;
    code[pc++] = 0xe0780000;  // buffer_store_dwordx4 v[0:3], off, s[4:7], 0
    code[pc++] = changing ? 0x0c010000 : 0x80010000;
    if (changing) {
      code[pc++] = indirect ? 0x80088408 : 0x80008100;  // next pointer / BVH
      code[pc++] = 0x800c900c;                          // output offset += 16
      code[pc++] = 0x808e810e;                          // --loop count
      code[pc++] = 0xbf07800e;                          // s_cmp_lg_u32 s14, 0
      const u32 branch = pc++;
      code[branch] = 0xbf850000 | ((loop_pc - pc) & 0xffff);
    }
    code[pc++] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    const auto compiled = gpu::rdna::RecompileCompute(code.data(), 1, 1, 1,
                                                      indirect ? 10 : 8, 0, 0);
    EXPECT_TRUE(compiled.ok) << "BVH at " << bvh_pc;
    if (!compiled.ok)
      return {};
    EXPECT_EQ(compiled.resources.size(), 2 + extra_bindings + indirect);
    EXPECT_EQ(compiled.resources[extra_bindings + indirect].kind, 3);
    EXPECT_EQ(compiled.resources[extra_bindings + indirect].runtime_address,
              runtime);
    gpu::ps5::Regs regs;
    const u64 address = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = address >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = address >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = (indirect ? 10 : 8) << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 base = reinterpret_cast<u64>(source.data());
    regs[ud] = base >> 8;
    regs[ud + 1] =
        ((base >> 40) & 255) | (sort ? 0x80000000 : 0) | (grow << 23);
    regs[ud + 2] = last_node;
    regs[ud + 3] = 0x81000000;  // BVH type, barycentric return
    const u64 dest = reinterpret_cast<u64>(output.data());
    regs[ud + 4] = dest;
    regs[ud + 5] = (dest >> 32) & 0xffff;
    regs[ud + 6] = changing ? 32 : 16;
    regs[ud + 7] = 0x00020000;
    if (indirect) {
      source[128] = base >> 8;
      source[129] = (base + 256) >> 8;
      const u64 pointers = reinterpret_cast<u64>(&source[128]);
      regs[ud + 8] = pointers;
      regs[ud + 9] = pointers >> 32;
    }
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(gpu::rhi::DefaultRenderer(), regs, dispatch, 4);
    EXPECT_TRUE(gpu::rhi::FlushCsWrites(gpu::rhi::DefaultRenderer()));
    if (changing) {
      EXPECT_EQ(
          (std::array<u32, 4>{output[4], output[5], output[6], output[7]}),
          (std::array<u32, 4>{0x188, 0x198, 0x180, kInvalid}));
    }
    EXPECT_EQ(output[changing ? 8 : 4], 0xa5a5a5a5);
    return {output[0], output[1], output[2], output[3]};
  }

  std::array<u32, 48> Boxes(bool half) {
    std::array<u32, 48> nodes{};
    // Node 1: four children, one misses, distances intentionally out of order.
    const float bounds[4][6] = {{-1, -1, 8, 1, 1, 9},
                                {-1, -1, 2, 1, 1, 3},
                                {2, 2, 1, 3, 3, 2},
                                {-1, -1, 4, 1, 1, 5}};
    const auto fp16 = [](float v) -> u32 {
      switch (static_cast<int>(v)) {
        case -1:
          return 0xbc00;
        case 1:
          return 0x3c00;
        case 2:
          return 0x4000;
        case 3:
          return 0x4200;
        case 4:
          return 0x4400;
        case 5:
          return 0x4500;
        case 8:
          return 0x4800;
        case 9:
          return 0x4880;
      }
      return 0;
    };
    for (u32 c = 0; c < 4; c++) {
      nodes[16 + c] = 0x80 + c * 8;
      for (u32 i = 0; i < 6; i++) {
        if (half)
          nodes[20 + c * 3 + i / 2] |= fp16(bounds[c][i]) << ((i % 2) * 16);
        else
          nodes[20 + c * 6 + i] = Bits(bounds[c][i]);
      }
    }
    return nodes;
  }
};

TEST_F(RdnaBvh, RuntimeDescriptorReadsGuestPagesAndChecksWholeNodes) {
  for (bool half : {false, true}) {
    EXPECT_EQ(Run(Boxes(half), 8 + (half ? 4 : 5), {0, 0, 0}, {0, 0, 1}, true,
                  false, true, false, 2, 0, 100, 0, 0, {}, true),
              (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
  }
  EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, true, false, true,
                false, 1, 0, 100, 0, 0, {}, true),
            (std::array<u32, 4>{kInvalid, kInvalid, kInvalid, kInvalid}));
}

TEST_F(RdnaBvh, RuntimeDescriptorChangesBetweenTraversalIterations) {
  EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, true, false, false,
                false, 2, 0, 100, 0, 0, {}, true, true),
            (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
}

TEST_F(RdnaBvh, RuntimeScalarLoadsFollowChangingPointerChains) {
  EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, true, false, false,
                false, 2, 0, 100, 0, 0, {}, true, true, true),
            (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
}

TEST_F(RdnaBvh, DynamicRawBuffersAndScalarPointersWithoutBvh) {
  alignas(65536) static std::array<std::array<u32, 16384>, 4> memory{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  for (bool scalar : {false, true}) {
    auto& input = memory[scalar ? 2 : 0];
    auto& output = memory[scalar ? 3 : 1];
    input[0] = 123;
    input[1] = 456;
    input[2] = 789;
    input[3] = 1011;
    output.fill(0xa5a5a5a5);
    code.fill(0);
    u32 pc = 0;
    for (u32 word : {0x7ec80208u, 0x7e000564u, 0x7eca0209u, 0x7e020565u})
      code[pc++] = word;  // live base comes through lane-dependent SGPR writes
    if (scalar) {
      for (u32 word : {0xf4080000u, 0xfa000000u, 0x7e000200u, 0x7e020201u,
                       0x7e040202u, 0x7e060203u})
        code[pc++] = word;
    } else {
      code[pc++] = 0xe0380000;
      code[pc++] = 0x80000000;
    }
    code[pc++] = 0xe0780000;
    code[pc++] = 0x80010000;
    code[pc++] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    const auto compiled =
        gpu::rdna::RecompileCompute(code.data(), 1, 1, 1, 10, 0, 0);
    ASSERT_TRUE(compiled.ok);
    ASSERT_EQ(compiled.resources.size(), 2);
    EXPECT_TRUE(compiled.resources[0].runtime_address);
    gpu::ps5::Regs regs;
    const u64 program = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = program >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = program >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 10 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud + 2] = 8;  // V# permits only the first two dwords
    regs[ud + 3] = 0x20000;
    const u64 dst = reinterpret_cast<u64>(output.data());
    regs[ud + 4] = dst;
    regs[ud + 5] = (dst >> 32) & 0xffff;
    regs[ud + 6] = 16;
    regs[ud + 7] = 0x20000;
    const u64 src = reinterpret_cast<u64>(input.data());
    regs[ud + 8] = src;
    regs[ud + 9] = src >> 32;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(gpu::rhi::DefaultRenderer(), regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(gpu::rhi::DefaultRenderer()));
    for (u32 i = 0; i < 4; ++i)
      EXPECT_EQ(output[i], scalar || i < 2 ? input[i] : 0u);
    EXPECT_EQ(output[4], 0xa5a5a5a5);
  }
}

TEST_F(RdnaBvh, BoxesSortCullAndDecodeAllOperandForms) {
  for (bool half : {false, true})
    for (bool wide : {false, true})
      for (bool nsa : {false, true})
        for (bool a16 : {false, true}) {
          SCOPED_TRACE(testing::Message() << half << wide << nsa << a16);
          EXPECT_EQ(Run(Boxes(half), 8 + (half ? 4 : 5), {0, 0, 0}, {0, 0, 1},
                        true, wide, nsa, a16),
                    (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
        }
  EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, false),
            (std::array<u32, 4>{0x80, 0x88, kInvalid, 0x98}));
}

TEST_F(RdnaBvh, BoundsInvalidNodesAndParallelSlabBoundary) {
  const std::array<u32, 4> miss = {kInvalid, kInvalid, kInvalid, kInvalid};
  const auto boxes = Boxes(false);
  for (u32 ptr : {kInvalid, 0xfffffffdu, 29u, 14u, 15u})
    EXPECT_EQ(Run(boxes, ptr), miss);
  // A 128-byte node must fit completely, including its second 64-byte unit.
  EXPECT_EQ(Run(boxes, 13, {0, 0, 0}, {0, 0, 1}, true, false, false, false, 1),
            miss);
  EXPECT_EQ(
      Run(boxes, 13, {0, 0, 0}, {0, 0, 1}, true, true, false, false, 2, 1),
      miss);
  EXPECT_EQ(Run(boxes, 13, {-1, 0, 0}),
            (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
  EXPECT_EQ(Run(boxes, 13, {-2, 0, 0}), miss);
  EXPECT_EQ(Run(boxes, 13, {0, 0, 10}, {0, 0, -1}),
            (std::array<u32, 4>{0x80, 0x98, 0x88, kInvalid}));
  EXPECT_EQ(
      Run(boxes, 13, {0, 0, 0}, {0, 0, 1}, true, false, false, false, 2, 0, 3),
      (std::array<u32, 4>{0x88, kInvalid, kInvalid, kInvalid}));
}

TEST_F(RdnaBvh, BoxGrowthAndNaNs) {
  auto boxes = Boxes(false);
  boxes[22] = Bits(1.f) + 1;  // near just beyond the ray extent
  const std::array<u32, 4> miss = {kInvalid, kInvalid, kInvalid, kInvalid};
  EXPECT_EQ(Run(boxes, 13, {0, 0, 0}, {0, 0, 1}, true, false, false, false, 2,
                0, 1, 0),
            miss);
  EXPECT_EQ(Run(boxes, 13, {0, 0, 0}, {0, 0, 1}, true, false, false, false, 2,
                0, 1, 2),
            (std::array<u32, 4>{0x80, kInvalid, kInvalid, kInvalid}));
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(Run(boxes, 13, {nan, 0, 0}), miss);
  EXPECT_EQ(Run(boxes, 13, {0, 0, 0}, {nan, 0, 1}), miss);
  boxes[20] = Bits(nan);
  EXPECT_EQ(Run(boxes, 13),
            (std::array<u32, 4>{0x88, 0x98, kInvalid, kInvalid}));
}

TEST_F(RdnaBvh, BindingsBeyondPushConstantsAndOldResourceLimit) {
  for (u32 extra : {48u, 64u}) {
    EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, true, false, false,
                  false, 2, 0, 100, 0, extra),
              (std::array<u32, 4>{0x88, 0x98, 0x80, kInvalid}));
    EXPECT_EQ(Run(Boxes(false), 13, {0, 0, 0}, {0, 0, 1}, true, false, false,
                  false, 1, 0, 100, 0, extra),
              (std::array<u32, 4>{kInvalid, kInvalid, kInvalid, kInvalid}));
  }
}

TEST_F(RdnaBvh,
       RayShaderWideComparisonsReadBothHalvesAndSignExtendInlineConstants) {
  std::vector<u32> tail = {
      0x7e2802ff, 0,          0x7e2a02ff, 1,  // v[20:21] = 0x100000000
      0x7e2c02ff, 0xffffffff, 0x7e2e02ff, 0,  // v[22:23] = 0xffffffff
      0xbf920001};  // trap without a handler is a hardware NOP
  const auto compare = [&](u32 op, u32 a, u32 b, u32 dest) {
    tail.push_back(0xd400006a | (op << 16));
    tail.push_back(a | (b << 9));
    tail.push_back(0x7e00026a | (dest << 17));  // v_mov_b32 vD, vcc_lo
  };
  compare(0xe4, 276, 278, 0);  // unsigned A > B: high half decides
  compare(0xe4, 276, 193, 1);  // unsigned A > -1: inline high half is all ones
  compare(0xa1, 193, 276, 2);  // signed -1 < A
  compare(0xe2, 278, 278, 3);  // unsigned B == B
  EXPECT_EQ(Run({}, 8, {0, 0, 0}, {0, 0, 1}, true, false, false, false, 2, 0,
                100, 0, 0, tail),
            (std::array<u32, 4>{1, 0, 1, 1}));
}

TEST(RdnaBvhCompile, RejectsUnsupportedControlFields) {
  alignas(256) std::array<u32, 4096> code = {0xf1989f01, 0, 0xe0780000,
                                             0x80010000, 0xbf810000};
  for (u32 bit : {3u, 8u, 12u, 15u, 16u, 17u}) {
    code[0] = 0xf1989f01 ^ (1u << bit);
    EXPECT_FALSE(gpu::rdna::RecompileCompute(code.data(), 1, 1, 1, 8, 0, 0).ok);
  }
  code[0] = 0xf1989f01;
  code[1] = 1u << 31;  // D16 is not a BVH return encoding
  EXPECT_FALSE(gpu::rdna::RecompileCompute(code.data(), 1, 1, 1, 8, 0, 0).ok);
}

TEST(RdnaBvhCompile, LaneDependentBvhDescriptorRequiresRuntimeBinding) {
  alignas(256) static std::array<u32, 4096> code = {
      0x7e000500,  // v_readfirstlane_b32 s0, v0
      0xf1989f01, 0, 0xe0780000, 0x80010000, 0xbf810000};
  const auto compiled =
      gpu::rdna::RecompileCompute(code.data(), 1, 1, 1, 8, 0, 0);
  ASSERT_TRUE(compiled.ok);
  ASSERT_EQ(compiled.resources.size(), 2);
  EXPECT_TRUE(compiled.resources[0].runtime_address);
  EXPECT_FALSE(compiled.resources[1].runtime_address);
}

TEST(RdnaBvhCompile, TrapHandlerStateIsPartOfTheShaderCacheKey) {
  alignas(256) static std::array<u32, 4096> code = {0xbf920001, 0xe0780000,
                                                    0x80010000, 0xbf810000};
  gpu::ps5::ComputeShaderState state;
  state.cs_addr = reinterpret_cast<u64>(code.data());
  state.thread_x = state.thread_y = state.thread_z = 1;
  state.user_sgpr = 8;
  EXPECT_TRUE(gpu::ps5::GetComputeShader(state).ok);
  state.trap_present = true;
  EXPECT_FALSE(gpu::ps5::GetComputeShader(state).ok);
  state.trap_present = false;
  EXPECT_TRUE(gpu::ps5::GetComputeShader(state).ok);
}

TEST_F(RdnaBvh, AllCompressedTrianglesAndBarycentricRotations) {
  constexpr u32 slots[4][3] = {{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}};
  constexpr float vertices[3][3] = {{0, 0, 5}, {2, 0, 5}, {0, 2, 5}};
  for (u32 type = 0; type < 4; type++) {
    std::array<u32, 48> nodes{};
    for (u32 v = 0; v < 3; v++)
      for (u32 c = 0; c < 3; c++)
        nodes[16 + slots[type][v] * 3 + c] = Bits(vertices[v][c]);
    for (u32 rotation = 0; rotation < 3; rotation++) {
      const u32 i = (1 + rotation) % 3, j = (2 + rotation) % 3;
      nodes[31] = (i | (j << 2)) << (type * 8);
      const auto result =
          Run(nodes, 8 + type, {.5f, .25f, 0}, {0, 0, 1}, true, type & 1, true,
              type & 2, 2, 0, 1);  // triangles aren't clipped to extent
      const float den = Float(result[1]);
      const std::array<float, 3> bary = {.625f, .25f, .125f};
      EXPECT_FLOAT_EQ(Float(result[0]) / den, 5);
      EXPECT_FLOAT_EQ(Float(result[2]) / den, bary[i]);
      EXPECT_FLOAT_EQ(Float(result[3]) / den, bary[j]);
      EXPECT_LT(den, 0);  // preserve facing sign in numerator and denominator
    }
    const auto outside = Run(nodes, 8 + type, {3, 3, 0});
    EXPECT_EQ(outside[0], Bits(std::numeric_limits<float>::infinity()));
    EXPECT_EQ(outside[1], Bits(1));
    const auto behind = Run(nodes, 8 + type, {.5f, .25f, 6});
    EXPECT_EQ(behind[0], Bits(std::numeric_limits<float>::infinity()));
  }
  const auto degenerate = Run({}, 8);
  EXPECT_EQ(degenerate[0], Bits(std::numeric_limits<float>::infinity()));
  EXPECT_EQ(degenerate[1], Bits(1));
}
}  // namespace

namespace gpu::rhi {
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0, g_dcb_n = 0;
}  // namespace gpu::rhi
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}
