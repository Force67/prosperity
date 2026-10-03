#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include "options/options.h"

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/bit_cast.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/ps4/compute_dispatch.h"
#include "gpu/ps4/render_queue.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/draw_state.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/render/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/draw_recomp.h"
#include "gpu/render/frame.h"
#include "gpu/render/renderer.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/upload_ring.h"
#include "host_memory/host_memory.h"

// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe() {}
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}

TEST(VkDraw, ConstantBufferBudgetAppliesAfterDeviceInitialization) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  if (gpu::render::g_ring.ubo_buf)
    GTEST_SKIP() << "Run this initialization test in a fresh renderer";
  base::OptionBase* budget = nullptr;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_GPU_UBORING_MB") == 0)
      budget = const_cast<base::OptionBase*>(option);
  });
  ASSERT_NE(budget, nullptr);
  ASSERT_TRUE(budget->SetFromString("512"));  // Title settings load after Init.
  gpu::render::BeginFrame(renderer);
  const auto capacity = gpu::render::UboRingBytes();
  EXPECT_EQ(capacity,
            base::Min<u64>(
                512ull << 20,
                gpu::render::Device().caps().max_storage_buffer_range * 2));
  ASSERT_NE(gpu::render::g_ring.ubo_map, nullptr);
  EXPECT_GE(gpu::render::g_ring.ubo_buf->desc().size, capacity);
  // Later option changes cannot move offsets beyond the existing allocation.
  ASSERT_TRUE(budget->SetFromString("1024"));
  EXPECT_EQ(gpu::render::UboRingBytes(), capacity);
  budget->Reset();
  gpu::render::EndFrame(renderer, 0);
}

TEST(RenderQueue, ExecutesAcrossIdleWakesAndCommandWraparound) {
  gpu::ps4::RenderQueue queue;
  queue.Start(gpu::render::DefaultRenderer());
  u32 completed = 0;
  for (u32 batch = 0; batch < 256; ++batch) {
    for (u32 i = 0; i < 65; ++i) {
      const u32 expected = batch * 65 + i;
      queue.PushCall([&, expected] {
        EXPECT_EQ(completed, expected);
        ++completed;
      });
    }
    queue.Drain();
    EXPECT_EQ(completed, (batch + 1) * 65);
  }
  queue.Stop();
}

TEST(GcnCompute, LockstepBranchSeesEitherHalfOfWave) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static u32 dest[5][64] = {};
  alignas(256) static u32 code[4096] = {};
  static gpu::gcn::RecompiledCs compiled_shaders[5];
  const u32 lanes[] = {0, 31, 32, 63, 64};
  for (u32 trial = 0; trial < 5; trial++) {
    // The LDS round trip requires lockstep control flow. Only one lane sets
    // VCC, but every lane must take the same scalar branch and write one.
    const u32 shader[] = {0x34020082, 0xd8340000, 0x00000001,
                          0xd8d80000, 0x02000001, 0x7d840080u + lanes[trial],
                          0xbf860003, 0x7e020281, 0xe0702000,
                          0x80000100, 0xbf810000};
    std::memcpy(code, shader, sizeof(shader));
    const auto previous_isa = gpu::gcn::DefaultIsaMode();
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    auto& compiled = compiled_shaders[trial];
    compiled = gpu::gcn::RecompileCompute(code, 64, 1, 1, 4, 0, 1);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(compiled.ok);
    ASSERT_EQ(compiled.resources.size(), 1u);
    gpu::render::ComputeInfo ci;
    ci.cs_addr = reinterpret_cast<u64>(code);
    ci.recomp = &compiled;
    ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
    ci.num_res = 1;
    auto& resource = ci.res[0];
    resource.base = reinterpret_cast<u64>(dest[trial]);
    resource.size = resource.guest_size = sizeof(dest[trial]);
    resource.binding = compiled.resources[0].binding;
    resource.written = resource.shader_writes = true;
    ci.user_data[0] = static_cast<u32>(resource.base);
    ci.user_data[1] = (resource.base >> 32) | (4u << 16);
    ci.user_data[2] = 64;
    ci.user_data[3] = (4u << 15) | (4u << 12);
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    for (u32 lane = 0; lane < 64; lane++)
      EXPECT_EQ(dest[trial][lane], lanes[trial] < 64 ? 1u : 0u)
          << "predicate lane " << lanes[trial] << ", output lane " << lane;
  }
}

TEST(GcnCompute, MultiplyAddPreservesSeparateRounding) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static u32 dest[2][64] = {};
  alignas(256) static u32 code[2][4096] = {};
  static gpu::gcn::RecompiledCs programs[2];
  for (u32 trial = 0; trial < 2; ++trial) {
    const u32 shader[] = {
        0x7e020204, 0x7e040205, 0x7e0602f3,
        trial ? 0xd2960001u : 0xd2820001u,
        257u | (258u << 9) | (259u << 18),
        0xe0702000, 0x80000100, 0xbf810000};
    std::memcpy(code[trial], shader, sizeof(shader));
    const auto previous_isa = gpu::gcn::DefaultIsaMode();
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    programs[trial] =
        gpu::gcn::RecompileCompute(code[trial], 64, 1, 1, 6, 0, 0);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(programs[trial].ok);
    ASSERT_EQ(programs[trial].resources.size(), 1u);
    gpu::render::ComputeInfo ci;
    ci.cs_addr = reinterpret_cast<u64>(code[trial]);
    ci.recomp = &programs[trial];
    ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
    ci.num_res = 1;
    auto& resource = ci.res[0];
    resource.base = reinterpret_cast<u64>(dest[trial]);
    resource.size = resource.guest_size = sizeof(dest[trial]);
    resource.binding = programs[trial].resources[0].binding;
    resource.written = resource.shader_writes = true;
    ci.user_data[0] = static_cast<u32>(resource.base);
    ci.user_data[1] = (resource.base >> 32) | (4u << 16);
    ci.user_data[2] = 64;
    ci.user_data[3] = (4u << 15) | (4u << 12);
    ci.user_data[4] = 0x3f800001;
    ci.user_data[5] = 0x3f7ffffe;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    for (u32 value : dest[trial])
      EXPECT_EQ(value, trial ? 0xa8800000u : 0u) << "fused=" << trial;
  }
}

TEST(GcnCompute, BlockImageWritesSupplyCompressedTexturesWithoutWriteback) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  auto* memory = static_cast<u8*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1210000000ull), 1048576,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(memory, nullptr);
  for (u32 trial = 0; trial < 3; ++trial) {
    const u32 elem = trial ? 16 : 8;
    const u32 width = trial == 2 ? 1 : trial == 1 ? 16 : 13;
    const u32 height = trial == 2 ? 1 : trial == 1 ? 8 : 9;
    const u32 layers = trial == 1 ? 2 : 1;
    const u32 block_w = (width + 3) / 4, block_h = (height + 3) / 4;
    const u32 dfmt = trial ? 14 : 11;
    const u32 words[] = {0x0000ffff, 0x12345678, 0x0000f800, 0x87654321};
    const u32 shader[] = {0x7e0802ff, words[0],
                          0x7e0a02ff, words[1],
                          0x7e0c02ff, words[2],
                          0x7e0e02ff, words[3],
                          0x4a080902, trial ? 0xf0205f00u : 0xf0204300u,
                          0x00000400, 0xbf810000};
    auto* code = memory + trial * 4096;
    std::memcpy(code, shader, sizeof(shader));
    gpu::gcn::TextureLayout32 tiled;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(tiled, block_w, block_h, 8,
                                               layers, 1, 14, false, elem));
    auto* guest = memory + 65536 + trial * 262144;
    std::memset(guest, 0xa5, 262144);
    const u64 base = reinterpret_cast<u64>(guest);
    const u64 pc = reinterpret_cast<u64>(code);
    gpu::Regs regs;
    regs[gpu::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::mmCOMPUTE_NUM_THREAD_X] = 8;
    regs[gpu::mmCOMPUTE_NUM_THREAD_Y] = 8;
    regs[gpu::mmCOMPUTE_NUM_THREAD_Z] = 2;
    regs[gpu::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
    const u32 ud = gpu::mmCOMPUTE_USER_DATA_0;
    regs[ud] = base >> 8;
    regs[ud + 1] = ((base >> 40) & 0xff) | (dfmt << 20) | (4u << 26);
    regs[ud + 2] = (block_w - 1) | ((block_h - 1) << 14);
    regs[ud + 3] = 0xfac | (14u << 20) | (13u << 28);
    regs[ud + 4] = (7u << 13) | (layers - 1);
    regs[ud + 5] = (layers - 1) << 13;
    gpu::render::BeginFrame(renderer);
    const u32 groups[] = {1, 1, 1, 1};
    gpu::ps4::DispatchCompute(renderer, regs, groups, 4);
    ASSERT_TRUE(gpu::render::CsRangeDirtyOverlapping(base, tiled.size));
    ASSERT_TRUE(gpu::render::CsSupplyTexture(
        base, tiled, width, height, nullptr, gpu::rhi::TextureState::kUndefined,
        nullptr, true));
    EXPECT_FALSE(gpu::render::CsSupplyTexture(
        base, tiled, width + 4, height, nullptr,
        gpu::rhi::TextureState::kUndefined, nullptr, true));
    auto wrong_pitch = tiled;
    wrong_pitch.mips[0].pitch += 8;
    EXPECT_FALSE(gpu::render::CsSupplyTexture(
        base, wrong_pitch, width, height, nullptr,
        gpu::rhi::TextureState::kUndefined, nullptr, true));
    auto sampled = tiled;
    const u32 mips = trial == 0 ? 3 : 1;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(sampled, block_w, block_h, 8,
                                               layers, mips, 14, false, elem));
    u32 supplied_mips = 0;
    ASSERT_TRUE(gpu::render::CsSupplyTexture(
        base, sampled, width, height, nullptr,
        gpu::rhi::TextureState::kUndefined, nullptr, true, &supplied_mips));
    EXPECT_EQ(supplied_mips, 1u);
    if (mips > 1)
      EXPECT_FALSE(gpu::render::CsSupplyTexture(
          base, sampled, width, height, nullptr,
          gpu::rhi::TextureState::kUndefined, nullptr, true));
    gpu::rhi::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.layers = layers;
    desc.mips = mips;
    desc.format = trial == 0   ? gpu::rhi::Format::kBC1Unorm
                  : trial == 1 ? gpu::rhi::Format::kBC3Unorm
                               : gpu::rhi::Format::kBC3Srgb;
    desc.usage = gpu::rhi::kTextureSampled | gpu::rhi::kTextureCopyDst |
                 gpu::rhi::kTextureCopySrc;
    auto* image = gpu::render::Device().CreateTexture(desc);
    ASSERT_NE(image, nullptr);
    gpu::rhi::Buffer* tail = nullptr;
    if (mips > 1) {
      gpu::rhi::BufferDesc upload;
      upload.size = 32;
      upload.usage = gpu::rhi::kBufferCopySrc;
      upload.memory = gpu::rhi::MemoryKind::kUpload;
      tail = gpu::render::Device().CreateBuffer(upload);
      ASSERT_NE(tail, nullptr);
      std::memset(tail->mapped(), 0x3c, upload.size);
      gpu::rhi::TextureBarrier init{image, gpu::rhi::TextureState::kUndefined,
                                    gpu::rhi::TextureState::kCopyDst};
      init.range.mips = mips;
      gpu::render::g_frame.list->Barrier(gpu::rhi::kAccessHostWrite,
                                         gpu::rhi::kAccessCopyRead, &init, 1);
      gpu::rhi::BufferTextureCopy copies[2]{};
      for (u32 mip = 1, offset = 0; mip < mips; ++mip) {
        auto& copy = copies[mip - 1];
        copy.buffer_offset = offset;
        copy.region.mip = mip;
        copy.region.width = base::Max(width >> mip, 1u);
        copy.region.height = base::Max(height >> mip, 1u);
        offset += ((copy.region.width + 3) / 4) *
                  ((copy.region.height + 3) / 4) * elem;
      }
      gpu::render::g_frame.list->CopyBufferToTexture(image, tail, copies, 2);
      init.before = gpu::rhi::TextureState::kCopyDst;
      init.after = gpu::rhi::TextureState::kShaderRead;
      gpu::render::g_frame.list->Barrier(0, 0, &init, 1);
    }
    u64 seq = 0;
    ASSERT_TRUE(gpu::render::CsSupplyTexture(
        base, sampled, width, height, image,
        mips > 1 ? gpu::rhi::TextureState::kShaderRead
                 : gpu::rhi::TextureState::kUndefined,
        &seq, true, &supplied_mips));
    if (mips > 1) {
      const u32 raw_shader[] = {0x7e020204, 0xe0702000, 0x80000100, 0xbf810000};
      auto* tail_code = code + 256;
      std::memcpy(tail_code, raw_shader, sizeof(raw_shader));
      const u64 tail_pc = reinterpret_cast<u64>(tail_code);
      const u64 tail_base = base + sampled.mips[1].offset;
      const u64 tail_bytes = sampled.size - sampled.mips[1].offset;
      gpu::Regs raw;
      raw[gpu::mmCOMPUTE_PGM_LO] = tail_pc >> 8;
      raw[gpu::mmCOMPUTE_PGM_HI] = tail_pc >> 40;
      raw[gpu::mmCOMPUTE_NUM_THREAD_X] = tail_bytes / 4;
      raw[gpu::mmCOMPUTE_NUM_THREAD_Y] = 1;
      raw[gpu::mmCOMPUTE_NUM_THREAD_Z] = 1;
      raw[gpu::mmCOMPUTE_PGM_RSRC2] = 5 << 1;
      raw[ud] = tail_base;
      raw[ud + 1] = (tail_base >> 32) | (4u << 16);
      raw[ud + 2] = tail_bytes / 4;
      raw[ud + 3] = (4u << 15) | (4u << 12);
      raw[ud + 4] = 0x5a5a5a5a;
      const u32 raw_groups[] = {1, 1, 1, 1};
      gpu::ps4::DispatchCompute(renderer, raw, raw_groups, 4);
      ASSERT_TRUE(gpu::render::CsSupplyTexture(
          base, sampled, width, height, nullptr,
          gpu::rhi::TextureState::kShaderRead, nullptr, true, &supplied_mips));
      EXPECT_EQ(supplied_mips, mips);
      auto* tail_guest = reinterpret_cast<u8*>(tail_base);
      tail_guest[0] = 0x11;
      EXPECT_FALSE(gpu::render::CsSupplyTexture(
          base, sampled, width, height, nullptr,
          gpu::rhi::TextureState::kShaderRead, nullptr, true, &supplied_mips));
      tail_guest[0] = 0xa5;
      ASSERT_TRUE(gpu::render::CsSupplyTexture(
          base, sampled, width, height, image,
          gpu::rhi::TextureState::kShaderRead, &seq, true, &supplied_mips));
      EXPECT_EQ(tail_guest[0], 0xa5);
    }
    EXPECT_NE(seq, 0u);
    EXPECT_EQ(guest[0], 0xa5);
    gpu::rhi::BufferDesc bd;
    const u32 mip0_bytes = block_w * block_h * elem * layers;
    bd.size = mip0_bytes + (mips > 1 ? 24 : 0);
    bd.usage = gpu::rhi::kBufferCopyDst;
    bd.memory = gpu::rhi::MemoryKind::kReadback;
    auto* result = gpu::render::Device().CreateBuffer(bd);
    ASSERT_NE(result, nullptr);
    gpu::rhi::TextureBarrier barrier{image, gpu::rhi::TextureState::kShaderRead,
                                     gpu::rhi::TextureState::kCopySrc};
    barrier.range.mips = mips;
    gpu::render::g_frame.list->Barrier(0, 0, &barrier, 1);
    gpu::rhi::BufferTextureCopy copies[3]{};
    for (u32 mip = 0, offset = 0; mip < mips; ++mip) {
      auto& copy = copies[mip];
      copy.buffer_offset = offset;
      copy.region.mip = mip;
      copy.region.width = base::Max(width >> mip, 1u);
      copy.region.height = base::Max(height >> mip, 1u);
      copy.region.layers = layers;
      offset += ((copy.region.width + 3) / 4) * ((copy.region.height + 3) / 4) *
                elem * layers;
    }
    gpu::render::g_frame.list->CopyTextureToBuffer(result, image, copies, mips);
    gpu::render::g_frame.list->Barrier(gpu::rhi::kAccessCopyWrite,
                                       gpu::rhi::kAccessHostRead);
    const u32 slot = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, 0);
    ASSERT_TRUE(gpu::render::Device().Wait(
        gpu::render::g_frame.slots[slot].submission));
    const auto* data = reinterpret_cast<const u32*>(result->mapped());
    for (u32 i = 0; i < mip0_bytes / 4; ++i) {
      const u32 word = i % (elem / 4);
      const u32 layer = i / (block_w * block_h * elem / 4);
      EXPECT_EQ(data[i], words[word] + (word == 0 ? layer : 0))
          << "trial=" << trial << "layer=" << layer;
    }
    for (u32 i = mip0_bytes / 4; i < bd.size / 4; ++i)
      EXPECT_EQ(data[i], 0x5a5a5a5au) << "tail word=" << i;
    EXPECT_EQ(guest[0], 0xa5);
    gpu::render::Device().Destroy(tail);
    gpu::render::Device().Destroy(result);
    gpu::render::Device().Destroy(image);
  }
  ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
  host_memory::FreeMem(memory, 1048576);
}

TEST(VkDraw, CachedTextureLayoutsPreserveMipsLayersAndVolumeUploads) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  auto* guest = static_cast<u8*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1213000000ull), 1048576,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(guest, nullptr);
  for (u32 trial = 0; trial < 3; ++trial) {
    const bool bc = trial == 1, volume = trial == 2;
    const u32 width = bc ? 100 : volume ? 16 : 32;
    const u32 height = bc ? 52 : volume ? 8 : 16;
    const u32 layers = volume ? 4 : bc ? 1 : 2;
    const u32 mips = volume ? 1 : bc ? 5 : 4;
    const u32 elem = bc ? 16 : 4;
    const u32 tiling = bc ? gpu::gcn::kGfx10TilingBase : volume ? 8 : 14;
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
        layout, bc ? (width + 3) / 4 : width, bc ? (height + 3) / 4 : height,
        bc ? (width + 3) / 4 : width, layers, mips, tiling, false, elem));
    if (bc)
      for (u32 mip = 0; mip < mips; ++mip) {
        layout.mips[mip].width = (base::Max(width >> mip, 1u) + 3) / 4;
        layout.mips[mip].height = (base::Max(height >> mip, 1u) + 3) / 4;
      }
    ASSERT_LE(layout.size, 1048576u);
    const u64 base = reinterpret_cast<u64>(guest);
    gpu::render::DrawInfo::DrawTex texture;
    texture.base = base;
    texture.w = width;
    texture.h = height;
    texture.pitch = width;
    texture.dfmt = bc ? 37 : 10;
    texture.nfmt = 0;
    texture.tiling = tiling;
    texture.layers = volume ? 1 : layers;
    texture.depth = volume ? layers : 1;
    texture.is_3d = volume;
    texture.arrayed = !volume && layers > 1;
    texture.view_layers = texture.layers;
    texture.mip_levels = texture.view_mips = mips;
    texture.swizzle = 0xfac;
    gpu::rhi::Texture* cached_image = nullptr;
    for (u32 pass = 0; pass < 2; ++pass) {
      std::memset(guest, 0, layout.size);
      base::Vector<u8> expected;
      gpu::rhi::BufferTextureCopy copies[16]{};
      for (u32 mip = 0; mip < mips; ++mip) {
        const auto& level = layout.mips[mip];
        const u32 bytes = level.width * level.height * elem;
        auto& copy = copies[mip];
        copy.buffer_offset = expected.size();
        copy.region.mip = mip;
        copy.region.width = base::Max(width >> mip, 1u);
        copy.region.height = base::Max(height >> mip, 1u);
        copy.region.layers = volume ? 1 : layers;
        copy.region.depth = volume ? layers : 1;
        for (u32 layer = 0; layer < layers; ++layer) {
          base::Vector<u8> pixels(bytes);
          for (u32 i = 0; i < bytes; ++i)
            pixels[i] = static_cast<u8>(i + pass * 73 + mip * 17 + layer * 31);
          ASSERT_TRUE(gpu::gcn::RetileTextureMip32(pixels.data(), guest, layout,
                                                   mip, layer));
          expected.insert(expected.end(), pixels.begin(), pixels.end());
        }
      }
      gpu::render::InvalidateTexRange(base, layout.size);
      gpu::render::BeginFrame(renderer);
      auto* view = gpu::render::TexViewFor(texture);
      ASSERT_NE(view, nullptr);
      if (cached_image)
        EXPECT_EQ(view->texture(), cached_image);
      cached_image = view->texture();
      gpu::render::BumpTextureEpoch();
      EXPECT_EQ(gpu::render::TexViewFor(texture), view);
      gpu::rhi::BufferDesc bd;
      bd.size = expected.size();
      bd.usage = gpu::rhi::kBufferCopyDst;
      bd.memory = gpu::rhi::MemoryKind::kReadback;
      auto* result = gpu::render::Device().CreateBuffer(bd);
      ASSERT_NE(result, nullptr);
      gpu::rhi::TextureBarrier barrier{cached_image,
                                       gpu::rhi::TextureState::kShaderRead,
                                       gpu::rhi::TextureState::kCopySrc};
      barrier.range.mips = mips;
      barrier.range.layers = volume ? 1 : layers;
      gpu::render::g_frame.list->Barrier(
          gpu::rhi::kAccessShaderRead, gpu::rhi::kAccessCopyRead, &barrier, 1);
      gpu::render::g_frame.list->CopyTextureToBuffer(result, cached_image,
                                                     copies, mips);
      barrier.before = gpu::rhi::TextureState::kCopySrc;
      barrier.after = gpu::rhi::TextureState::kShaderRead;
      gpu::render::g_frame.list->Barrier(
          gpu::rhi::kAccessCopyRead, gpu::rhi::kAccessShaderRead, &barrier, 1);
      gpu::render::g_frame.list->Barrier(gpu::rhi::kAccessCopyWrite,
                                         gpu::rhi::kAccessHostRead);
      const u32 slot = gpu::render::g_frame.slot_idx;
      gpu::render::EndFrame(renderer, 0);
      ASSERT_TRUE(gpu::render::Device().Wait(
          gpu::render::g_frame.slots[slot].submission));
      EXPECT_EQ(std::memcmp(result->mapped(), expected.data(), expected.size()),
                0)
          << "trial " << trial << ", pass " << pass;
      gpu::render::Device().Destroy(result);
    }
  }
  host_memory::FreeMem(guest, 1048576);
}

TEST(GcnCompute, CpuReadSeesLatestOfTwoWrites) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static u32 dest[64] = {};
  alignas(256) static const u32 code[4096] = {0x7e020204, 0xe0702000,
                                              0x80000100, 0xbf810000};
  static gpu::gcn::RecompiledCs compiled;
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  compiled = gpu::gcn::RecompileCompute(code, 64, 1, 1, 5, 0, 0);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(compiled.ok);
  ASSERT_EQ(compiled.resources.size(), 1u);
  gpu::render::ComputeInfo ci;
  ci.cs_addr = reinterpret_cast<u64>(code);
  ci.recomp = &compiled;
  ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
  ci.num_res = 1;
  auto& resource = ci.res[0];
  resource.base = reinterpret_cast<u64>(dest);
  resource.size = resource.guest_size = sizeof(dest);
  resource.binding = compiled.resources[0].binding;
  resource.written = resource.shader_writes = true;
  ci.user_data[0] = static_cast<u32>(resource.base);
  ci.user_data[1] = (resource.base >> 32) | (4u << 16);
  ci.user_data[2] = 64;
  ci.user_data[3] = (4u << 15) | (4u << 12);
  for (u32 trial = 0; trial < 3; trial++) {
    ci.user_data[4] = trial * 2 + 1;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    // A second write supersedes any readback queued by the first dispatch.
    ci.user_data[4]++;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    ASSERT_TRUE(gpu::render::FlushCsWritesRange(
        renderer, resource.base, resource.size, "test-readback"));
    for (u32 value : dest)
      EXPECT_EQ(value, ci.user_data[4]);
  }
}

TEST(GcnCompute, MergingChildInvalidatesParentReadback) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  base::OptionBase* merge = nullptr;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_GPU_CS_TRUTH") == 0)
      merge = const_cast<base::OptionBase*>(option);
  });
  ASSERT_NE(merge, nullptr);
  struct ResetOption {
    base::OptionBase* option;
    ~ResetOption() { option->Reset(); }
  } reset{merge};
  alignas(65536) static u32 dest[2][64] = {};
  alignas(256) static const u32 write_code[4096] = {0x7e020204, 0xe0702000,
                                                    0x80000100, 0xbf810000};
  alignas(256) static const u32 read_code[4096] = {0xe0302000, 0x80000100,
                                                   0xbf810000};
  static gpu::gcn::RecompiledCs writer, reader;
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  writer = gpu::gcn::RecompileCompute(write_code, 64, 1, 1, 5, 0, 0);
  reader = gpu::gcn::RecompileCompute(read_code, 64, 1, 1, 4, 0, 0);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(writer.ok);
  ASSERT_TRUE(reader.ok);
  ASSERT_EQ(writer.resources.size(), 1u);
  ASSERT_EQ(reader.resources.size(), 1u);
  for (u32 trial = 0; trial < 2; trial++) {
    const u64 parent = reinterpret_cast<u64>(dest[trial]);
    auto dispatch = [&](u64 base, u32 count, bool write, u32 value) {
      gpu::render::ComputeInfo ci;
      ci.cs_addr = reinterpret_cast<u64>(write ? write_code : read_code);
      ci.recomp = write ? &writer : &reader;
      ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
      ci.num_res = 1;
      auto& resource = ci.res[0];
      resource.base = base;
      resource.size = resource.guest_size = count * sizeof(u32);
      resource.binding = ci.recomp->resources[0].binding;
      resource.written = resource.shader_writes = write;
      resource.read = !write;
      ci.user_data[0] = static_cast<u32>(base);
      ci.user_data[1] = (base >> 32) | (4u << 16);
      ci.user_data[2] = count;
      ci.user_data[3] = (4u << 15) | (4u << 12);
      ci.user_data[4] = value;
      return gpu::render::Dispatch(renderer, ci);
    };
    ASSERT_TRUE(dispatch(parent, 64, true, 1));
    ASSERT_TRUE(
        gpu::render::FlushCsWritesRange(renderer, parent, 256, "test-merge"));
    ASSERT_TRUE(dispatch(parent, 64, true, 3));
    // Create independent dirty ranges, then merge through either read path.
    ASSERT_TRUE(merge->SetFromString("0"));
    ASSERT_TRUE(dispatch(parent + 64, 16, true, 2));
    ASSERT_TRUE(merge->SetFromString("1"));
    ASSERT_TRUE(dispatch(parent + (trial ? 64 : 0), trial ? 16 : 64, false, 0));
    ASSERT_TRUE(
        gpu::render::FlushCsWritesRange(renderer, parent, 256, "test-merge"));
    for (u32 lane = 0; lane < 64; lane++)
      EXPECT_EQ(dest[trial][lane], lane >= 16 && lane < 32 ? 2u : 3u)
          << "read path " << trial << ", lane " << lane;
  }
}

TEST(GcnCompute, DirtyExtentTracksStoresRatherThanStagingPadding) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static u32 dest[2][32768] = {};
  alignas(256) static const u32 code[2][4096] = {
      {0x7e020204, 0xe0702000, 0x80000100, 0xbf810000},
      {0x7e020204, 0xea242000, 0x80000100, 0xbf810000}};
  static gpu::gcn::RecompiledCs shaders[2];
  for (u32 trial = 0; trial < 2; trial++) {
    SCOPED_TRACE(trial);
    auto& compiled = shaders[trial];
    const auto previous_isa = gpu::gcn::DefaultIsaMode();
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    compiled = gpu::gcn::RecompileCompute(code[trial], 64, 1, 1, 5, 0, 0);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(compiled.ok);
    ASSERT_EQ(compiled.resources.size(), 1u);
    gpu::render::ComputeInfo ci;
    ci.cs_addr = reinterpret_cast<u64>(code[trial]);
    ci.recomp = &compiled;
    ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
    ci.num_res = 1;
    auto& resource = ci.res[0];
    const u64 base = resource.base = reinterpret_cast<u64>(dest[trial]);
    resource.size = resource.guest_size = sizeof(dest[trial]);
    resource.binding = compiled.resources[0].binding;
    resource.written = resource.shader_writes = true;
    ci.user_data[0] = static_cast<u32>(base);
    ci.user_data[1] = (base >> 32) | (4u << 16);
    ci.user_data[3] = (4u << 15) | (4u << 12);
    auto dispatch = [&](u32 records, u64 extent, u32 value) {
      ci.user_data[2] = records;
      ci.user_data[4] = value;
      resource.write_bytes = extent;
      return gpu::render::Dispatch(renderer, ci);
    };
    ASSERT_TRUE(dispatch(32, 128, 1));
    EXPECT_TRUE(gpu::render::CsRangeDirtyOverlapping(base + 127, 1));
    EXPECT_FALSE(gpu::render::CsRangeDirtyOverlapping(base + 128, 1));
    EXPECT_FALSE(gpu::render::CsRangeMaybeDirty(base + 65536, 4));
    dest[trial][16384] = 9;
    const u64 submitted = gpu::render::Device().LastSubmission();
    ASSERT_TRUE(gpu::render::FlushCsWritesRange(renderer, base + 65536, 4,
                                               "test-padding"));
    EXPECT_EQ(gpu::render::Device().LastSubmission(), submitted);
    EXPECT_EQ(dest[trial][0], 0u);
    ASSERT_TRUE(dispatch(64, 256, 2));
    ASSERT_TRUE(dispatch(32, 128, 3));
    EXPECT_TRUE(gpu::render::CsRangeDirtyOverlapping(base + 255, 1));
    EXPECT_FALSE(gpu::render::CsRangeDirtyOverlapping(base + 256, 1));
    ASSERT_TRUE(gpu::render::FlushCsWritesRange(renderer, base, 256,
                                               "test-extent"));
    for (u32 lane = 0; lane < 64; lane++)
      EXPECT_EQ(dest[trial][lane], lane < 32 ? 3u : 2u);
    // An unknown bound covers the whole footprint until its results are read.
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(dispatch(64, 0, 4));
    EXPECT_TRUE(gpu::render::CsHoldsVertices(base, sizeof(dest[trial])));
    ASSERT_TRUE(dispatch(32, 128, 5));
    EXPECT_TRUE(gpu::render::CsRangeDirtyOverlapping(base + 65536, 4));
    EXPECT_TRUE(gpu::render::CsRangeMaybeDirty(base + 65536, 4));
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    EXPECT_FALSE(gpu::render::CsRangeDirtyOverlapping(base, sizeof(dest[trial])));
    EXPECT_FALSE(gpu::render::CsRangeMaybeDirty(base + 65536, 4));
    for (u32 lane = 0; lane < 64; lane++)
      EXPECT_EQ(dest[trial][lane], lane < 32 ? 5u : 4u);
    EXPECT_EQ(dest[trial][16384], 9u);
    gpu::render::EndFrame(renderer, 0);
    ASSERT_TRUE(dispatch(0, 128, 6));
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    for (u32 lane = 0; lane < 64; lane++)
      EXPECT_EQ(dest[trial][lane], lane < 32 ? 5u : 4u);
  }
}

TEST(RdnaResources, RepeatedDescriptorLoadsReuseTextureBindings) {
  base::Vector<u32> code;
  for (u32 i = 0; i < 32; ++i) {
    code.insert(code.end(),
                {0xf408040e, 0xfa000030,    // s_load s[16:19], s[28:29], 48
                 0xf09c0108, 0x00040f05});  // sample with s[16:23]
  }
  code.push_back(0xbf810000);
  const auto plan =
      gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(plan.binding_srsrc.size(), 1u);
  EXPECT_EQ(plan.binding_by_pc.size(), 32u);

  // A changed table pointer or offset must keep the samples distinct.
  code.insert(code.begin() + 4, 0xbe9c0381);  // s_mov_b32 s28, 1
  auto changed =
      gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 2u);
  code[6] = 0xfa000040;
  changed =
      gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 3u);

  // A memory-writing shader cannot assume a repeated load sees the same data.
  code.insert(code.end() - 1, {0xf0200108, 0x00000400});  // image_store
  changed =
      gpu::rdna::RdnaPlanMimg(gpu::rdna::Decode(code.data(), code.size()));
  EXPECT_EQ(changed.binding_srsrc.size(), 33u);
}

// Pipelines use Recompiled addresses as module identities. Keep programs alive
// for the renderer lifetime, matching the production shader cache.

TEST(VkDraw, MeshWorkgroupsExpandInputPointsIntoTriangles) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                      0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
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
  const Id pos =
      m.Variable(m.TypePointer(spv::StorageClass::Output, m.TypeArray(v4, 3)),
                 spv::StorageClass::Output);
  m.Decorate(pos, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::Position)});
  const Id prim =
      m.Variable(m.TypePointer(spv::StorageClass::Output, m.TypeArray(uv3, 1)),
                 spv::StorageClass::Output);
  m.Decorate(prim, spv::Decoration::BuiltIn,
             {static_cast<u32>(spv::BuiltIn::PrimitiveTriangleIndicesEXT)});
  const Id main =
      m.BeginFunction(m.TypeVoid(), m.TypeFunction(m.TypeVoid(), {}));
  const Id gx = m.CompositeExtract(u, m.Load(uv3, group), 0);
  const Id offset = m.Emit(spv::Op::OpFSub, f,
                           {m.Emit(spv::Op::OpConvertUToF, f, {gx}), fc(.5f)});
  m.EmitVoid(spv::Op::OpSetMeshOutputsEXT, {uc(3), uc(1)});
  for (u32 i = 0; i < 3; ++i) {
    const float x[] = {-.4f, .4f, 0.f}, y[] = {-.5f, -.5f, .5f};
    const Id p = m.CompositeConstruct(
        v4, {m.Emit(spv::Op::OpFAdd, f, {offset, fc(x[i])}), fc(y[i]), fc(0),
             fc(1)});
    m.Store(m.AccessChain(m.TypePointer(spv::StorageClass::Output, v4), pos,
                          {uc(i)}),
            p);
  }
  m.Store(m.AccessChain(m.TypePointer(spv::StorageClass::Output, uv3), prim,
                        {uc(0)}),
          m.CompositeConstruct(uv3, {uc(0), uc(1), uc(2)}));
  m.ReturnVoid();
  m.EndFunction();
  m.EntryPoint(spv::ExecutionModel::MeshEXT, main, "main", {group, pos, prim});
  m.ExecMode(main, spv::ExecutionMode::LocalSize, {1, 1, 1});
  m.ExecMode(main, spv::ExecutionMode::OutputVertices, {3});
  m.ExecMode(main, spv::ExecutionMode::OutputPrimitivesEXT, {1});
  m.ExecMode(main, spv::ExecutionMode::OutputTrianglesEXT, {});
  base::String error;
  ASSERT_TRUE(
      gpu::gcn::spirv::Finalize(m.Assemble(), &program.mesh_spirv, &error))
      << error.c_str();
  program.vs_spirv.clear();
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &program;
  draw.prim_type = 1;
  draw.vertex_count = 2;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 12) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 0);
}

TEST(VkDraw, SplitNggStagesTransferLdsAndExportConnectivity) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling the split NGG mesh shader";
  const u32 es[4096] = {0x34000a82,              // v0 = ES vertex ID * 4
                      0xd8340000, 0x00000500,  // LDS[v0] = ES vertex ID
                      0xbe802006};  // s_setpc_b64 s[6:7] -> separately bound GS
  const u32 gs[4096] = {
      0xd8d80000, 0x02000000,  // v2 = LDS[primitive vertex offset]
      0x7e000d02,              // v0 = float(v2)
      0x100000f0, 0x060000f1,  // x = id * .5 - .5
      0x7e0202f1, 0x7e0602f0,  // y = -.5, alternate = .5
      0x7d840481, 0x02020701,  // y = id == 1 ? .5 : -.5
      0x7e040280, 0x7e0602f2,  // z = 0, w = 1
      0xf80008cf, 0x03020100,  // POS0
      0x7e0802ff, 0x00200400,  // packed triangle indices 0, 1, 2
      0xbefe0481,              // only lane zero exports the primitive
      0xf8000941, 0x00000004, 0xbefe04c1,
      0xbefc03ff, 0x00001003,  // allocation: 3 vertices, 1 primitive
      0xbf900009, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128};
  static const auto kProgram = gpu::rdna::Recompile(
      es, ps, user_data, user_data, 0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_FALSE(kProgram.mesh_spirv.empty());
  ASSERT_TRUE(kProgram.vs_spirv.empty());
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
  EXPECT_EQ(pixels[0], 0);
}

TEST(VkDraw, UnifiedNggProgramExportsTriangleConnectivity) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling the unified NGG mesh shader";
  // The shader probe reads 16 KiB and caches its result by address.
  alignas(16384) static const u32 gs[4096] = {
      0x7e040305,              // v2 = input vertex ID
      0x7e000d02,              // v0 = float(v2)
      0x100000f0, 0x060000f1,  // x = id * .5 - .5
      0x7e0202f1, 0x7e0602f0,  // y = -.5, alternate = .5
      0x7d840481, 0x02020701,  // y = id == 1 ? .5 : -.5
      0x7e040280, 0x7e0602f2,  // z = 0, w = 1
      0xf80008cf, 0x03020100,  // POS0
      0x7e0802ff, 0x00200400,  // packed triangle indices 0, 1, 2
      0xbefe0481,              // only lane zero exports the primitive
      0xf8000941, 0x00000004, 0xbefe04c1,
      0xbefc03ff, 0x00001003,  // allocation: 3 vertices, 1 primitive
      0xbf900009, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128, 0, false};
  static const auto kProgram = gpu::rdna::Recompile(
      gs, ps, user_data, user_data, 0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(kProgram.ok);
  EXPECT_TRUE(gpu::rdna::HasNggPrimitiveExports(gs));
  EXPECT_FALSE(gpu::rdna::HasNggTransfer(gs));
  ASSERT_FALSE(kProgram.mesh_spirv.empty());
  ASSERT_TRUE(kProgram.vs_spirv.empty());
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
  EXPECT_EQ(pixels[0], 0);
}

TEST(VkDraw, NggPointBatchesPreserveEveryExpandedQuad) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling NGG mesh shaders";
  base::Array<u32, 4096> code{};
  u32 pc = 0;
  const auto v1 = [&](u32 op, u32 dst, u32 src) {
    code[pc++] = 0x7e000000 | (dst << 17) | (op << 9) | src;
  };
  const auto v2 = [&](u32 op, u32 dst, u32 src, u32 other) {
    code[pc++] = (op << 25) | (dst << 17) | (other << 9) | src;
  };
  const auto literal = [&](u32 op, u32 dst, u32 value, u32 other) {
    v2(op, dst, 255, other);
    code[pc++] = value;
  };
  // Four output vertices and two triangles for each independent input point.
  // v0 initially carries local_id*4, v5 carries group_base+local_id.
  v2(0x16, 1, 130, 0);  // local_id
  v2(0x16, 2, 130, 1);  // local input = local_id / 4
  v2(0x26, 3, 261, 1);  // group base = v5 - local_id
  v2(0x25, 3, 258, 3);  // global input point
  v2(0x1b, 4, 129, 1);  // corner x bit
  v2(0x16, 5, 129, 1);
  v2(0x1b, 5, 129, 5);  // corner y bit
  v1(6, 10, 259);
  literal(8, 10, 0x3e800000, 10);  // x = input*.25 - .75
  literal(3, 10, 0xbf400000, 10);
  v1(6, 4, 260);
  literal(8, 4, 0x3e000000, 4);  // quad width .125
  v2(3, 10, 260, 10);
  v1(6, 11, 261);
  v2(8, 11, 240, 11);  // quad height .5
  literal(3, 11, 0xbe800000, 11);
  v1(1, 12, 128);
  v1(1, 13, 242);
  code[pc++] = 0xf80008cf;
  code[pc++] = 0x0d0c0b0a;
  v2(0x16, 6, 129, 1);
  v2(0x1a, 6, 130, 6);  // primitive's vertex base
  v2(0x1b, 7, 129, 1);  // triangle within quad
  v2(0x1a, 8, 129, 7);
  v2(0x25, 8, 262, 8);
  v2(0x25, 8, 129, 8);  // i1 = base + 1 + triangle*2
  v2(0x25, 9, 130, 6);  // i2 = base + 2
  v2(0x25, 6, 263, 6);  // i0 = base + triangle
  v2(0x1a, 8, 138, 8);
  v2(0x1a, 9, 148, 9);
  v2(0x1c, 6, 264, 6);
  v2(0x1c, 6, 265, 6);
  code[pc++] = 0xf8000941;
  code[pc++] = 6;
  v1(1, 8, 3);
  literal(0x1b, 8, 255, 8);  // live input count from wave info
  v2(0x1a, 9, 130, 8);
  v2(0x1a, 8, 141, 8);
  v2(0x1c, 9, 264, 9);
  v1(2, 0, 265);  // M0 = vertices | (primitives << 12)
  code[pc++] = 0xbefc0300;
  code[pc++] = 0xbf900009;
  code[pc++] = 0xbf810000;
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                        0xbf810000};
  const u32 user_data[32]{};
  const u32 batches[] = {6, 2,
                         4};  // one group, three groups, partial final group
  static base::Array<gpu::gcn::Recompiled, 3> programs;
  alignas(65536) static base::Array<base::Array<u8, 65536>, 3> targets{};
  base::Array<u8, 64 * 16 * 4> reference{};
  for (u32 i = 0; i < 3; ++i) {
    SCOPED_TRACE(batches[i]);
    const gpu::rdna::NggConfig cfg{
        code.data(),    64,  batches[i], batches[i] * 4,
        batches[i] * 2, 128, 0,          false};
    programs[i] = gpu::rdna::Recompile(code.data(), ps, user_data, user_data, 0,
                                       false, 0, 0, nullptr, 0, &cfg);
    ASSERT_TRUE(programs[i].ok);
    gpu::render::DrawInfo draw;
    draw.recomp = &programs[i];
    draw.prim_type = 1;
    draw.vertex_count = 6;
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[i].data());
    draw.rt_w = 64;
    draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    for (u32 q = 0; q < 6; ++q) {
      EXPECT_EQ(pixels[(8 * 64 + 10 + q * 8) * 4], 255);
      EXPECT_EQ(pixels[(8 * 64 + 14 + q * 8) * 4], 0);
    }
    if (!i)
      base::CopyN(pixels, reference.size(), reference.begin());
    else
      EXPECT_TRUE(std::equal(reference.begin(), reference.end(), pixels));
  }
}

TEST(VkDraw, MeshConstantWindowsExceedTheDynamicDescriptorLimit) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling the mesh constant-window shader";
  const u32 es[4096] = {0x34000a82, 0xd8340000, 0x00000500, 0xbe802006};
  base::Array<u32, 256> gs{};
  u32 pc = 0;
  // One scalar load in each of 17 disjoint constant-buffer windows. The
  // final load controls the triangle position, beyond the 15 dynamic UBOs.
  for (u32 window = 0; window < 17; ++window) {
    gs[pc++] = 0xf4200004;  // s_buffer_load_dword s0, s[8:11], offset
    gs[pc++] = 0xfa000000 | (window * gpu::gcn::kCbufDwords * 4);
  }
  const u32 body[] = {0xd8d80000, 0x02000000, 0x7e000d02, 0x100000f0,
                      0x060000f1, 0x06000000,  // x += s0
                      0x7e0202f1, 0x7e0602f0, 0x7d840481, 0x02020701,
                      0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100,
                      0x7e0802ff, 0x00200400, 0xbefe0481, 0xf8000941,
                      0x00000004, 0xbefe04c1, 0xbefc03ff, 0x00001003,
                      0xbf900009, 0xbf810000};
  base::Copy(body, body + base::ArraySize(body), gs.begin() + pc);
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs.data(), 64, 3, 3, 1, 128};
  static const auto kProgram = gpu::rdna::Recompile(
      es, ps, user_data, user_data, 0, false, 4, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_EQ(kProgram.vs_cbufs.size(), 17u);
  ASSERT_TRUE(kProgram.indirect_cbufs);
  static base::Array<float, 17 * gpu::gcn::kCbufDwords> constants{};
  alignas(65536) static base::Array<base::Array<u8, 65536>, 3> targets{};
  for (u32 frame = 0; frame < 3; ++frame) {
    constants[16 * gpu::gcn::kCbufDwords] = frame == 1 ? .5f : -.5f;
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.prim_type = 1;
    draw.vertex_count = 3;
    for (const auto& cb : kProgram.vs_cbufs) {
      draw.cbufs[cb.binding] = {
          reinterpret_cast<u64>(constants.data() + cb.first_dword),
          cb.num_dwords * 4};
      draw.num_cbufs = base::Max(draw.num_cbufs, cb.binding + 1);
    }
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    const u32 expected_x = frame == 1 ? 12 : 4;
    EXPECT_EQ(pixels[(8 * 16 + expected_x) * 4], 255) << "frame " << frame;
    EXPECT_EQ(pixels[(8 * 16 + (16 - expected_x)) * 4], 0) << "frame " << frame;
  }
}

TEST(VkDraw, SplitNggUserWindowsAndHighPixelRegistersRemainDistinct) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling NGG mesh shaders";
  const u32 es[4096] = {0x34000a83,              // v0 = vertex ID * 8
                      0x7e0c0208,              // v6 = ES user data in s8
                      0xd8340000, 0x00000500,  // LDS[v0] = vertex ID
                      0xd8340004, 0x00000600,  // LDS[v0+4] = ES offset
                      0xbe802006};
  const u32 gs[4096] = {0x34000a83,  // address matching the ES record
                      0xd8d80000, 0x02000000, 0xd8d80004, 0x06000000,
                      0x7e000d02, 0x100000f0, 0x060000f1, 0x06000000,
                      0x06000106,  // x += GS user s0 + offset written by ES
                      0x7e0202f1, 0x7e0602f0, 0x7d840481, 0x02020701,
                      0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100,
                      0x7e0802ff, 0x00200400, 0xbefe0481, 0xf8000941,
                      0x00000004, 0xbefe04c1, 0xbefc03ff, 0x00001003,
                      0xbf900009, 0xbf810000};
  const u32 ps[4096] = {0x7e00021d,  // red = user s29, beyond the old push window
                      0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
  u32 user_data[32]{};
  user_data[0] = 0x3e800000;  // ES contributes +.25; GS changes independently
  const gpu::rdna::NggConfig cfg{gs, 64, 3, 3, 1, 128};
  static const auto kProgram = gpu::rdna::Recompile(
      es, ps, user_data, user_data, 0, false, 4, 30, nullptr, 0, &cfg);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<base::Array<u8, 65536>, 2> targets{};
  for (u32 frame = 0; frame < 2; ++frame) {
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.prim_type = 1;
    draw.vertex_count = 3;
    draw.vs_user_data[0] = user_data[0];
    draw.gs_user_data_addr = frame ? 0xbe800000 : 0x3e800000;
    draw.ps_user_data[29] = frame ? 0x3f000000 : 0x3f800000;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    EXPECT_NEAR(pixels[(8 * 16 + (frame ? 8 : 12)) * 4], frame ? 128 : 255, 1);
    EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 0);
  }
}

TEST(VkDraw, GeometryResourceReplayUsesSystemUserDataAddress) {
  // The vertex user window contains a V#, while s0:s1 names a different table.
  const u32 code[64] = {0xf4080200, 0xfa000000,  // s_load_dwordx4 s8, s0, 0
                        0xf4200304, 0xfa000000,  // s_buffer_load s12, s8, 0
                        0xbf810000};
  static const u32 kValues[2] = {0x12345678, 0x87654321};
  const u64 base = reinterpret_cast<u64>(kValues);
  const u32 descriptor[4] = {static_cast<u32>(base),
                             static_cast<u32>(base >> 32) | (4u << 16), 2,
                             0x0004dfac};
  const u32 user_data[4] = {0, 0, 0, 0};
  const auto resources = gpu::rdna::ResolveBuffers(
      code, user_data, 4, 8, 5, reinterpret_cast<u64>(descriptor));
  ASSERT_TRUE(resources.count(0));
  ASSERT_TRUE(resources.count(2));
  EXPECT_EQ(resources.at(0).base, reinterpret_cast<u64>(descriptor));
  EXPECT_EQ(resources.at(2).base, base);
  EXPECT_TRUE(resources.at(2).descriptor_valid);
}

TEST(VkDraw, VertexAndPixelUserDataPastSixteenDoNotOverlap) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {0x7e00021f, 0x7e0202ff, 0x3d800000,
                      0x7e040280, 0x7e0602f2, 0xf80008cf,
                      0x03020100, 0xbf810000};  // x = s31 (user23)
  const u32 ps[4096] = {0x7e00021d, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};  // red = s29
  const u32 user_data[32]{};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data, 0, false, 24, 30);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_TRUE(kProgram.indirect_cbufs);
  ASSERT_TRUE(kProgram.mesh_spirv.empty());
  alignas(65536) static base::Array<base::Array<u8, 65536>, 2> targets{};
  for (u32 frame = 0; frame < 2; ++frame) {
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.vs_user_data[23] = frame ? 0xbee00000 : 0x3f100000;
    draw.ps_user_data[29] = frame ? 0x3f000000 : 0x3f800000;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[frame].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    // Place the point at a pixel center, avoiding rasterization edge ties.
    const u32 x = frame ? 4 : 12;
    EXPECT_NEAR(pixels[(7 * 16 + x) * 4], frame ? 128 : 255, 1);
    EXPECT_EQ(pixels[(7 * 16 + (16 - x)) * 4], 0);
  }
}

TEST(VkDraw, PixelShaderUsesDppOperand) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {0x7e0002ff, 0x3d800000, 0x7e040280, 0x7e0602f2,
                      0xf80008cf, 0x03020000, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2,              // v0 = 1
                      0x7e0002fa, 0xff00e400,  // v0 = quad_perm[0,1,2,3](v0)
                      0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
  const u32 user_data[32]{};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data, 0, false, 0, 0);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.prim_type = 1;
  draw.vertex_count = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  EXPECT_EQ(pixels[(7 * 16 + 8) * 4], 255);
}

TEST(VkDraw, SdwaAluWritesSelectedBytesAndPreservesOtherBits) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  struct Case {
    u32 select, unused, input, expected;
  };
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
  const u32 vs[4096] = {0x7e0002ff, 0x3d800000, 0x7e040280, 0x7e0602f2,
                      0xf80008cf, 0x03020000, 0xbf810000};
  static base::Array<gpu::gcn::Recompiled, base::ArraySize(cases) * 2> programs;
  alignas(65536) static base::Array<base::Array<u8, 65536>,
                                    base::ArraySize(cases) * 2>
      targets{};
  for (u32 i = 0; i < programs.size(); ++i) {
    const auto& c = cases[i % base::ArraySize(cases)];
    const bool binary = i >= base::ArraySize(cases);
    SCOPED_TRACE(i);
    u32 ps[4096] = {0x7e000200, 0x7e020201};  // old bits and float input
    u32 pc = 2;
    if (binary) {
      ps[pc++] = 0x7e021101;  // convert v1 to integer before the binary op
      ps[pc++] = 0x7e040280;  // v2 = 0
    }
    ps[pc++] = binary ? 0x3a0004f9 : 0x7e0010f9;  // XOR or CVT with SDWA
    ps[pc++] = 0x00060001 | (c.select << 8) | (c.unused << 11) |
               (binary ? 0x06000000 : 0);
    const u32 tail[] = {0x7d840002,  // compare packed integer result with s2
                        0x7e0402f2, 0x02000480,  // red = result matches ? 1 : 0
                        0x7e020280, 0xf800080f, 0x00010100, 0xbf810000};
    base::CopyN(tail, base::ArraySize(tail), ps + pc);
    const u32 user_data[32] = {0xabcdef12, c.input, c.expected};
    programs[i] =
        gpu::rdna::Recompile(vs, ps, user_data, user_data, 0, false, 0, 3);
    ASSERT_TRUE(programs[i].ok);
    gpu::render::DrawInfo draw;
    draw.recomp = &programs[i];
    draw.prim_type = 1;
    draw.vertex_count = 1;
    base::CopyN(user_data, 32, draw.ps_user_data);
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[i].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    EXPECT_EQ(pixels[(7 * 16 + 8) * 4], 255);
  }
}

TEST(VkDraw, NggWavesKeepIndependentBranchesAndFullBallots) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) || !gpu::render::Device().caps().mesh_shader)
    GTEST_SKIP() << "Mesh shading is required";
  if (std::strstr(gpu::render::Device().caps().device_name, "llvmpipe"))
    GTEST_SKIP() << "LLVMpipe crashes compiling NGG mesh shaders";
  const u32 es[4096] = {0x34000a82, 0xd8340000, 0x00000500, 0xbe802006};
  // Two guest waves take different scalar branches. Their triangles use
  // vertices in the upper half of each wave and a ballot spanning both halves.
  const u32 gs[4096] = {
      0x9380ff03, 0x00040018,  // s0 = wave index from merged_wave_info
      0xbf068000,              // s_cmp_eq_u32 s0, 0
      0xbf850003,              // wave zero jumps to the negative offset
      0xbe8403f0,              // s4 = .5
      0xbf820002, 0xbf800000,
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
      0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100, 0x7e0802ff,
      0x02208420,              // triangle 32, 33, 34
      0x7e0c02ff, 0x06218460,  // triangle 96, 97, 98
      0x7d840a81,              // ES vertex ID v5 == 1
      0x02080d04,              // choose connectivity for output primitive 1
      0xf8000941, 0x00000004, 0xbefc03ff, 0x00002063,  // 99 vertices, 2
                                                       // primitives
      0xbf900009, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32]{};
  const gpu::rdna::NggConfig cfg{gs, 128, 99, 128, 2, 128};
  static const auto kProgram = gpu::rdna::Recompile(
      es, ps, user_data, user_data, 0, false, 0, 0, nullptr, 0, &cfg);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.prim_type = 1;
  draw.vertex_count = 99;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  EXPECT_EQ(pixels[(8 * 16 + 4) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 12) * 4], 255);
  EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 0);
}

TEST(VkDraw, FetchedVerticesKeepNggVertexAndInstanceIds) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // Read the launch IDs before an inline fetch overwrites v5. Two vertices
  // in two instances should cover four separate points, one per quadrant.
  const u32 vs[4096] = {
      0x7e280d05, 0x7e2a0d08,  // v20=float(v5), v21=float(v8)
      0xf4080404, 0,           // s_load_dwordx4 s[16:19], s[8:9], 0
      0xe0002000, 0x80040505,  // buffer_load_format_x v5, v5, s[16:19], 0
      0x062828f1, 0x062a2af1,  // v20+=-0.5, v21+=-0.5
      0x7e2c0280,              // v22=0
      0xf80008cf, 0x05161514,  // position=(v20,v21,v22,v5)
      0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const base::Array<float, 2> vertices = {1.f, 1.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[4] = {u32(vb), u32(vb >> 32) | (4u << 16), 2,
                             0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_EQ(kProgram.attrs.size(), 1u);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  draw.prim_type = 1;
  draw.vertex_count = draw.instance_count = 2;
  draw.num_vbufs = draw.num_vattrs = 1;
  draw.vertex_data = vertices.data();
  draw.vertex_stride = 4;
  draw.vbufs[0] = {vertices.data(), 4, 2};
  draw.vattrs[0] = {0, 0, 0, 1, 4, 7};  // R32_FLOAT
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  base::Array<u32, 4> quadrants{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++quadrants[(y / 8) * 2 + x / 8];
  EXPECT_EQ(quadrants, (base::Array<u32, 4>{1, 1, 1, 1}));
}

TEST(VkDraw, IndexedRawVerticesUseLoadedDescriptorStride) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {
      0xf4080404, 0xfa000000,  // s[16:19] = descriptor from s[8:9]
      0xe0302000, 0x80040005,  // v0 = buffer[vertex ID], using descriptor
                               // stride
      0x7e020280, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const base::Array<float, 4> vertices = {-.5f, 99.f, .5f, 99.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[4] = {u32(vb), u32(vb >> 32) | (8u << 16), 2,
                             0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  gpu::rdna::NextProgramGeneration();
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  base::CopyN(user_data, 32, draw.vs_user_data);
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
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  base::Array<u32, 2> halves{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++halves[x / 8];
  EXPECT_EQ(halves, (base::Array<u32, 2>{1, 1}));
}

TEST(VkDraw, Ps5DepthClearUsesRegisterValueInsteadOfVertexDepth) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(256) static const u32 kVs[4096] = {0x7e000280, 0x7e020280, 0x7e0402ff,
                                             0x3f400000, 0x7e0602f2, 0xf80008cf,
                                             0x03020100, 0xbf810000};
  alignas(256) static const u32 kPs[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f,
                                             0x00010100, 0xbf810000};
  alignas(65536) static base::Array<u8, 65536> target{}, depth{};
  using namespace gpu::ps5;
  Regs regs;
  const auto address = [&](u32 reg, const void* ptr) {
    const u64 value = reinterpret_cast<u64>(ptr);
    regs[reg] = u32(value >> 8);
    regs[reg + 1] = u32(value >> 40);
  };
  address(mmSPI_SHADER_PGM_LO_ES, kVs);
  address(mmSPI_SHADER_PGM_LO_PS, kPs);
  regs[mmVGT_PRIMITIVE_TYPE] = 1;
  regs[mmCB_TARGET_MASK] = 15;
  regs[mmCB_COLOR0_BASE] = u32(reinterpret_cast<u64>(target.data()) >> 8);
  regs[mmCB_COLOR0_BASE_EXT] = u32(reinterpret_cast<u64>(target.data()) >> 40);
  regs[mmCB_COLOR0_INFO] = 10u << 2;
  regs[mmCB_COLOR0_ATTRIB2] = (15u << 14) | 15;
  regs[mmPA_CL_CLIP_CNTL] = 1u << 19;
  regs[mmPA_CL_VPORT_XSCALE] = regs[mmPA_CL_VPORT_XOFFSET] =
      base::BitCast<u32>(8.f);
  regs[mmPA_CL_VPORT_YSCALE] = base::BitCast<u32>(-8.f);
  regs[mmPA_CL_VPORT_YOFFSET] = base::BitCast<u32>(8.f);
  regs[mmDB_Z_INFO] = 3;
  regs[mmDB_Z_WRITE_BASE] = u32(reinterpret_cast<u64>(depth.data()) >> 8);
  regs[mmDB_Z_WRITE_BASE_HI] = u32(reinterpret_cast<u64>(depth.data()) >> 40);
  regs[mmDB_DEPTH_CLEAR] = base::BitCast<u32>(1.f);
  regs[mmDB_DEPTH_CONTROL] = (7u << 4) | 6;  // ALWAYS, test and write
  regs[mmDB_RENDER_CONTROL] = 1;
  const u32 body[] = {1, 0};
  const DrawPacket packet{0x2d, body, 2};  // DRAW_INDEX_AUTO
  gpu::render::DrawInfo clear;
  ASSERT_TRUE(BuildDrawInfo(regs, packet, clear));
  ASSERT_TRUE(clear.depth_clear_draw);
  // Skyrim retains its previous postprocess resources during the clear.
  // They must not turn the clear into a rejected depth-feedback draw.
  clear.tex_base = clear.depth_base;
  clear.num_texs = 1;
  clear.texs[0].base = clear.depth_base;
  regs[mmDB_RENDER_CONTROL] = 0;
  gpu::render::DrawInfo previous;
  ASSERT_TRUE(BuildDrawInfo(regs, packet, previous));
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, previous));
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, clear));
  regs[mmDB_RENDER_CONTROL] = 0;
  regs[mmDB_DEPTH_CONTROL] =
      (1u << 4) | 6;  // LESS: .75 passes only after clear=1
  gpu::render::DrawInfo point;
  ASSERT_TRUE(BuildDrawInfo(regs, packet, point));
  ASSERT_FALSE(point.depth_clear_draw);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, point));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, point.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  u32 red = 0;
  for (u32 i = 0; i < 16 * 16; ++i)
    red += pixels[i * 4] == 255;
  EXPECT_EQ(red, 1u);
}

TEST(VkDraw, IndexedTypedLoadUsesLiveStrideAndInstructionFormat) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // Inline V# at s[8:11], indexed by v5. The instruction requests float4
  // even though the descriptor advertises R32_UINT. Padding must not become
  // vertex data, and changing the stride must work with the same module.
  const u32 vs[4096] = {0xea6b2000, 0x80020005, 0xf80008cf, 0x03020100,
                      0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  base::Array<float, 32> vertices;
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  u32 user_data[32] = {u32(vb), u32(vb >> 32) | (32u << 16), 2, 0x21014fac};
  gpu::rdna::NextProgramGeneration();
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_TRUE(kProgram.attrs.empty());
  ASSERT_EQ(kProgram.vs_bufs.size(), 1u);
  alignas(65536) static base::Array<base::Array<u8, 65536>, 2> targets{};
  for (u32 pass = 0; pass < 2; ++pass) {
    const u32 stride = pass ? 48 : 32;
    SCOPED_TRACE(stride);
    vertices.fill(99.f);
    const base::Array<float, 4> left = {-.5f, 0.f, 0.f, 1.f};
    const base::Array<float, 4> right = {.5f, 0.f, 0.f, 1.f};
    base::Copy(left.begin(), left.end(), vertices.begin());
    base::Copy(right.begin(), right.end(), vertices.begin() + stride / 4);
    user_data[1] = u32(vb >> 32) | (stride << 16);
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    base::CopyN(user_data, 32, draw.vs_user_data);
    draw.prim_type = 1;
    draw.vertex_count = 2;
    draw.bufs[0] = {vb, sizeof(vertices)};
    draw.num_bufs = 1;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[pass].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    base::Array<u32, 2> halves{};
    for (u32 y = 0; y < 16; ++y)
      for (u32 x = 0; x < 16; ++x)
        if (pixels[(y * 16 + x) * 4] == 255)
          ++halves[x / 8];
    EXPECT_EQ(halves, (base::Array<u32, 2>{1, 1}));
  }
}

TEST(VkDraw, BulkDescriptorLoadPreservesInteriorBufferStride) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {
      0xf4100204, 0xfa000000,  // s[8:23] = four descriptors from s[8:9]
      0xf4200804, 0xfa000000,  // a scalar load uses the first descriptor
      0xe0302000, 0x80040005,  // v0 = buffer[vertex ID], using descriptor
                               // stride
      0x7e020280, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const base::Array<float, 4> vertices = {-.5f, 99.f, .5f, 99.f};
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 descriptor[16] = {
      u32(vb), u32(vb >> 32) | (8u << 16), 2, 0x21014fac, 0, 0, 0, 0,
      u32(vb), u32(vb >> 32) | (8u << 16), 2, 0x21014fac};
  const u64 table = reinterpret_cast<u64>(descriptor);
  const u32 user_data[32] = {u32(table), u32(table >> 32)};
  gpu::rdna::NextProgramGeneration();
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  base::CopyN(user_data, 32, draw.vs_user_data);
  draw.prim_type = 1;
  draw.vertex_count = 2;
  draw.bufs[0] = {vb, sizeof(vertices)};
  draw.num_bufs = 1;
  const auto resources = gpu::rdna::ResolveBuffers(vs, user_data, 32, 8);
  for (const auto& cb : kProgram.vs_cbufs) {
    ASSERT_TRUE(resources.count(cb.use_pc));
    draw.cbufs[cb.binding] = {resources.at(cb.use_pc).base + cb.first_dword * 4,
                              cb.num_dwords * 4};
    draw.num_cbufs = base::Max(draw.num_cbufs, cb.binding + 1);
  }
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  base::Array<u32, 2> halves{};
  for (u32 y = 0; y < 16; ++y)
    for (u32 x = 0; x < 16; ++x)
      if (pixels[(y * 16 + x) * 4] == 255)
        ++halves[x / 8];
  EXPECT_EQ(halves, (base::Array<u32, 2>{1, 1}));
}

TEST(VkDraw, PassthroughInterpolationPreservesPackedVertexValues) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer) ||
      !gpu::render::Device().caps().fragment_barycentric)
    GTEST_SKIP() << "Fragment barycentrics are required";
  static const u32 kVs[4096] = {
      0xe0382000, 0x80020005,  // position.xyz + packed attribute, indexed by v5
      0x7e0802f2, 0xf80008cf, 0x04020100, 0xf8000201, 0x00000003, 0xbf810000};
  static const u32 kPs[4096] = {
      0xc8020002, 0xc8060000, 0xc80a0001,  // P0, P10, P20
      0x7e0602f2, 0x7e0e0280, 0x7d840000,
      0x02080680,              // v4 = first vertex matches s0
      0x7d840201, 0x020a0680,  // v5 = second vertex matches s1
      0x7d840402, 0x020c0680,  // v6 = third vertex matches s2
      0xf800080f, 0x03060504, 0xbf810000};
  struct Vertex {
    float x, y, z;
    u32 packed;
  };
  const Vertex vertices[] = {{-.75f, -.75f, 0, 0x3f800123},
                             {.75f, -.75f, 0, 0x7fc0ffff},
                             {0, .75f, 0, 0xabcdef12}};
  const u64 vb = reinterpret_cast<u64>(vertices);
  const u32 vs_ud[32] = {u32(vb), u32(vb >> 32) | (16u << 16), 3, 0x21014fac};
  const u32 ps_ud[32] = {vertices[0].packed, vertices[1].packed,
                         vertices[2].packed};
  alignas(65536) static base::Array<base::Array<u8, 65536>, 2> targets{};
  const gpu::gcn::Recompiled* previous = nullptr;
  for (u32 passthrough = 0; passthrough < 2; ++passthrough) {
    const u32 input_control[32] = {passthrough ? 0x420u : 0u};
    const auto& program =
        gpu::ps5::GetGraphicsShader({.vs_addr = reinterpret_cast<u64>(kVs),
                                     .ps_addr = reinterpret_cast<u64>(kPs),
                                     .vs_user_sgprs = 4,
                                     .ps_user_sgprs = 3,
                                     .ps_in_cntl = input_control,
                                     .ps_num_interp = 1,
                                     .vs_user_data = vs_ud,
                                     .ps_user_data = ps_ud});
    ASSERT_TRUE(program.ok);
    if (previous)
      EXPECT_NE(&program, previous) << "Interpolation mode changes the shader";
    previous = &program;
    gpu::render::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(kVs);
    draw.ps_addr = reinterpret_cast<u64>(kPs);
    base::CopyN(vs_ud, 32, draw.vs_user_data);
    base::CopyN(ps_ud, 32, draw.ps_user_data);
    draw.prim_type = 4;
    draw.vertex_count = 3;
    draw.bufs[0] = {vb, sizeof(vertices)};
    draw.num_bufs = 1;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[passthrough].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 3], 255);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4], 255);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 1], passthrough ? 255 : 0);
    EXPECT_EQ(pixels[(8 * 16 + 8) * 4 + 2], passthrough ? 255 : 0);
  }
}

TEST(VkDraw, PixelTypedLoadConvertsPackedDataWithByteOffsets) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                      0xbf810000};
  const u32 ps[4096] = {
      0x7e000284,  // v0 = byte offset 4
      0xe8001004 | (56u << 19) | (3u << 16), 0x88020000,
      // tbuffer_load_format_xyzw v[0:3], v0, s[8:11], 8 offen offset:4
      0xf800080f, 0x03020100, 0xbf810000};
  const u32 data[8] = {0, 0, 0, 0, 0xffff8040, 0, 0, 0};
  const u64 base = reinterpret_cast<u64>(data);
  const u32 user_data[32] = {u32(base), u32(base >> 32), sizeof(data), 0x14fac};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  ASSERT_EQ(kProgram.ps_bufs.size(), 1u);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  base::CopyN(user_data, 32, draw.ps_user_data);
  draw.prim_type = draw.vertex_count = 1;
  draw.bufs[0] = {base, sizeof(data)};
  draw.num_bufs = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
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
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                      0xbf810000};
  const u32 ps[4096] = {
      0x7e0002f0, 0xf09c0f00, 0x00400400,  // sample_lz 1D at x=.5
      0xbefc03ff, 256,                     // M0 = 256
      0xdac00000, 0x00000400,              // spill red at M0 + lane*4
      0xbefc0380,                          // M0 = 0
      0xdac40100, 0x00000000,              // reload with offset:256
      0xf800080f, 0x07060500, 0xbf810000};
  const u32 user_data[32]{};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> texture{64, 128, 255, 255},
      target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
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
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
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
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  const u32 vs[4096] = {
      0xd5690000, 0x00020aff, 0x001fffff,  // v0 = vertex ID * (2M - 1)
      0x4a000081,                          // v0 += 1
      0xe0302000, 0x80020000,              // v0 = raw buffer s[8:11][v0]
      0x7e020280, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020100, 0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  base::Vector<float> vertices(4 * 1024 * 1024);
  const u32 indices[] = {1, 2 * 1024 * 1024, u32(vertices.size() - 1)};
  const float positions[] = {-.5625f, .0625f, .5625f};
  for (u32 i = 0; i < 3; ++i)
    vertices[indices[i]] = positions[i];
  const u64 vb = reinterpret_cast<u64>(vertices.data());
  const u32 user_data[32] = {u32(vb), u32(vb >> 32) | (4u << 16),
                             u32(vertices.size()), 0x21014fac};
  gpu::rdna::NextProgramGeneration();
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<u8, 65536> target{};
  gpu::render::DrawInfo draw;
  draw.recomp = &kProgram;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  base::CopyN(user_data, 32, draw.vs_user_data);
  draw.prim_type = 1;
  draw.vertex_count = 3;
  draw.bufs[0] = {vb, u32(vertices.size() * sizeof(float))};
  draw.num_bufs = 1;
  draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(target.data());
  draw.rt_w = draw.rt_h = 16;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, draw.rt_base);
  const auto& slot = gpu::render::g_frame.slots[slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  const auto* pixels = slot.readback->mapped();
  for (u32 x : {3u, 8u, 12u}) {
    u32 colored = 0;
    for (u32 y = 0; y < 16; ++y)
      colored += pixels[(y * 16 + x) * 4] == 255;
    EXPECT_EQ(colored, 1u) << "vertex at pixel x=" << x;
  }
}

TEST(VkDraw, SrgbScanoutPreservesEncodedBytes) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  static const u32 kVs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                                0xbf810000};
  static const u32 kPs[4096] = {0x7e0002ff, 0x3e800000, 0x7e020280, 0x7e0402f2,
                                0xf800080f, 0x02010100, 0xbf810000};
  static const u32 kGreen[4096] = {0x7e0002ff, 0x3e800000, 0x7e0202f2,
                                   0x7e0402f2, 0xf800080f, 0x02010100,
                                   0xbf810000};
  const u32 user_data[32]{};
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  static const auto program =
      gpu::gcn::Recompile(kVs, kPs, user_data, user_data);
  static const auto green =
      gpu::gcn::Recompile(kVs, kGreen, user_data, user_data);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(program.ok);
  ASSERT_TRUE(green.ok);
  alignas(65536) static base::Array<base::Array<u8, 65536>, 4> targets{};
  alignas(65536) static base::Array<base::Array<u8, 65536>, 4> metadata{};
  for (u32 trial = 0; trial < 4; ++trial) {
    const u32 bgra = trial & 1;
    const bool alias = trial >= 2;
    SCOPED_TRACE(alias);
    SCOPED_TRACE(bgra);
    gpu::render::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(kVs);
    draw.ps_addr = reinterpret_cast<u64>(kPs);
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[trial].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = (10u << 2) | (6u << 8) | (bgra << 11);
    gpu::render::BeginFrame(renderer);
    if (alias) {
      draw.mrt_meta_cmask = true;
      draw.mrt_dcc_base[0] = reinterpret_cast<u64>(metadata[trial].data());
      auto* parked = gpu::render::GetRT(
          draw.rt_base, 16, 16,
          bgra ? gpu::rhi::Format::kBGRA8Unorm : gpu::rhi::Format::kRGBA8Unorm);
      ASSERT_TRUE(parked);
      parked->dcc_base = draw.mrt_dcc_base[0];
      const u32 clear = 0;
      gpu::render::NoteDccWrite(draw.mrt_dcc_base[0], 4, &clear);
    }
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    if (alias) {
      draw.recomp = &green;
      draw.ps_addr = reinterpret_cast<u64>(kGreen);
      draw.mrt_info[0] = (10u << 2) | (bgra << 11);
      draw.target_mask = 2;
      ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
      gpu::render::EndRegion();
      ASSERT_TRUE(gpu::render::GetRT(
          draw.rt_base, 16, 16,
          bgra ? gpu::rhi::Format::kBGRA8Srgb : gpu::rhi::Format::kRGBA8Srgb));
    }
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    EXPECT_EQ(slot.fmt,
              bgra ? gpu::rhi::Format::kBGRA8Srgb
                   : gpu::rhi::Format::kRGBA8Srgb);
    const auto* pixels = slot.readback->mapped();
    u32 colored = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (pixels[i * 4 + 3] == 255) {
        EXPECT_NEAR(pixels[i * 4 + (bgra ? 2 : 0)], 137, 1);
        EXPECT_EQ(pixels[i * 4 + 1], alias ? 255 : 0);
        EXPECT_EQ(pixels[i * 4 + (bgra ? 0 : 2)], 0);
        ++colored;
      }
    EXPECT_EQ(colored, 1u);
  }
}

TEST(VkDraw, GcnPixelExportsRespectExec) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  static const u32 kVs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                                0xbf810000};
  static const u32 kPs[4][4096] = {
      {0x7e0002f2, 0x7e020280, 0x7e0402f2, 0xf800080f, 0x00010201, 0xbf810000},
      {0x7e0002f2, 0x7e020280, 0x7e0402f2, 0xf800000f, 0x00010201, 0xbefe0480,
       0xf800080f, 0x00010100, 0xbf810000},
      {0x7e0002f2, 0x7e020280, 0xbefe0480, 0xf800080f, 0x00010100, 0xbf810000},
      {0x7e0002f2, 0x7e020280, 0x7e0402f2, 0xf800000f, 0x00010201,
       0xbefe0480, 0xf800180f, 0x00010100, 0xbf810000}};
  const u32 user_data[32]{};
  static base::Array<gpu::gcn::Recompiled, 4> programs;
  alignas(65536) static base::Array<base::Array<u8, 65536>, 4> targets{};
  for (u32 trial = 0; trial < 4; ++trial) {
    SCOPED_TRACE(trial);
    const auto previous_isa = gpu::gcn::DefaultIsaMode();
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    programs[trial] =
        gpu::gcn::Recompile(kVs, kPs[trial], user_data, user_data);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(programs[trial].ok);
    gpu::render::DrawInfo draw;
    draw.recomp = &programs[trial];
    draw.vs_addr = reinterpret_cast<u64>(kVs);
    draw.ps_addr = reinterpret_cast<u64>(kPs[trial]);
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[trial].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    u32 green = 0;
    for (u32 i = 0; i < 16 * 16; ++i) {
      EXPECT_EQ(pixels[i * 4], 0);
      EXPECT_EQ(pixels[i * 4 + 2], 0);
      green += pixels[i * 4 + 1] == 255;
    }
    EXPECT_EQ(green, trial < 2 ? 1u : 0u);
  }
}

TEST(VkDraw, SinglePointRendersWithAndWithoutIndices) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // A procedural point at the origin, shaded opaque red.
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                        0xbf810000};
  const u32 ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32] = {};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<base::Array<u8, 65536>, 2> targets{};
  const u16 index = 0;
  for (u32 indexed = 0; indexed < 2; ++indexed) {
    base::Fill(targets[indexed].begin(), targets[indexed].end(), u8{0xA5});
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    draw.prim_type = 1;
    draw.vertex_count = indexed ? 0 : 1;
    draw.index_data = indexed ? &index : nullptr;
    draw.index_count = indexed ? 1 : 0;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[indexed].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;  // RGBA8_UNORM
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw))
        << "indexed=" << indexed;
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    ASSERT_NE(slot.readback, nullptr);
    const auto* pixels = slot.readback->mapped();
    u32 colored = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (pixels[i * 4] || pixels[i * 4 + 1] || pixels[i * 4 + 2])
        ++colored;
    EXPECT_EQ(colored, 1u) << "indexed=" << indexed;
    EXPECT_EQ(gpu::render::g_frame.heuristic, 0u);

    // Read the point through a compute image, swap red/green, and publish it
    // back into the same live render target. Guest bytes still contain 0xA5:
    // this must bridge the actual GPU image in both directions.
    alignas(256) static base::Array<u32, 4096> cs{};
    const base::Array<u32, 10> code = {
        0x7e020287,              // v1 = row 7
        0xf0000f08, 0x00000400,  // image_load v[4:7], v[0:1], s[0:7]
        0x7e100304, 0x7e080305, 0x7e0a0308,  // swap v4/v5 through v8
        0xf0200f08, 0x00000400,  // image_store v[4:7], v[0:1], s[0:7]
        0xbf810000, 0};
    base::Copy(code.begin(), code.end(), cs.begin());
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
    gpu::render::BeginFrame(renderer);
    const u32 dispatch[] = {1, 1, 1, 1};
    alignas(256) static const u32 kReader[4096] = {
        0x7e020287, 0xf0000f08, 0x00000400, 0xbf810000};
    const u64 reader_pc = reinterpret_cast<u64>(kReader);
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = reader_pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = reader_pc >> 40;
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    const u32 compute_slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& compute_slot = gpu::render::g_frame.slots[compute_slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(compute_slot.submission));
    ASSERT_TRUE(compute_slot.presentable);
    const auto* result = compute_slot.readback->mapped();
    u32 green = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (result[i * 4 + 1] == 255 && result[i * 4] == 0 &&
          result[i * 4 + 2] == 0)
        ++green;
    EXPECT_EQ(green, 1u) << "indexed=" << indexed;
  }
}

TEST(VkDraw, NarrowRenderTargetsRoundTripThroughCompute) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // A half-red point in an R8/R16 target, whose guest bytes remain zero.
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                      0xbf810000};
  const u32 ps[4096] = {0x7e0002f0, 0x7e020280, 0xf800080f, 0x00010100,
                      0xbf810000};
  const u32 user_data[32] = {};
  static const auto kProgram =
      gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(kProgram.ok);
  alignas(65536) static base::Array<base::Array<u8, 65536>, 3> targets{};
  for (u32 format_index = 0; format_index < 3; ++format_index) {
    gpu::render::DrawInfo draw;
    draw.recomp = &kProgram;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    draw.prim_type = 1;
    draw.vertex_count = 1;
    draw.index_data = nullptr;
    draw.index_count = 0;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[format_index].data());
    draw.rt_w = draw.rt_h = 16;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] =
        ((format_index ? 2u : 1u) << 2) |
        (format_index == 2 ? 7u << 8 : 0);  // R8/R16_UNORM, R16_FLOAT
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw))
        << "format=" << format_index;
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    ASSERT_NE(slot.readback, nullptr);
    const auto* pixels = slot.readback->mapped();
    // The scanout is converted to BGRA8 on the GPU before the readback.
    ASSERT_EQ(slot.fmt, gpu::rhi::Format::kBGRA8Unorm);
    const auto red = [](const u8* data, u32 i) -> u32 { return data[i * 4 + 2]; };
    const u32 half = 128, quarter = 64;
    u32 colored = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (red(pixels, i) >= half - 1 && red(pixels, i) <= half)
        ++colored;
    EXPECT_EQ(colored, 1u) << "format=" << format_index;
    EXPECT_EQ(gpu::render::g_frame.heuristic, 0u);

    // Halve the live red channel in compute, then present the modified image.
    alignas(256) static base::Array<u32, 4096> cs{};
    const base::Array<u32, 7> code = {
        0x7e020287,              // v1 = row 7
        0xf0000108, 0x00000400,  // image_load v4, v[0:1], s[0:7]
        0x100808f0,              // v_mul_f32 v4, 0.5, v4
        0xf0200108, 0x00000400,  // image_store v4, v[0:1], s[0:7]
        0xbf810000};
    base::Copy(code.begin(), code.end(), cs.begin());
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
    regs[ud + 1] = ((draw.rt_base >> 40) & 0xff) |
                   ((format_index == 2 ? 13u
                     : format_index    ? 7u
                                       : 1u)
                    << 20) |
                   (3u << 30);
    regs[ud + 2] = 3 | (15u << 14);
    regs[ud + 3] = 0x90000fac;  // 2D, LINEAR
    gpu::render::BeginFrame(renderer);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::render::FlushCsWrites(renderer));
    const u32 compute_slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& compute_slot = gpu::render::g_frame.slots[compute_slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(compute_slot.submission));
    ASSERT_TRUE(compute_slot.presentable);
    const auto* result = compute_slot.readback->mapped();
    u32 quarter_red = 0;
    for (u32 i = 0; i < 16 * 16; ++i)
      if (red(result, i) >= quarter - 1 && red(result, i) <= quarter)
        ++quarter_red;
    EXPECT_EQ(quarter_red, 1u) << "format=" << format_index;
  }
}

TEST(GcnCompute, ExpandedHtileAvoidsReadbackAndRejectsChangedKernel) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  auto* memory = static_cast<u32*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1212000000ull), 1048576,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(memory, nullptr);
  const u32 shader[] = {
      0xbeeb03ff, 0x00000092, 0x8f008310, 0x8f018311, 0x4a000000, 0x4a020201,
      0xc2800d04, 0xc2880d00, 0xbf8c007f, 0x90008200, 0x81148700, 0x90148314,
      0xb780003f, 0xb7940001, 0xd2d20002, 0x00002100, 0xd2d20003, 0x00002101,
      0x90008600, 0x8714c214, 0xb0150000, 0x7e080280, 0x7e0a02c1, 0xbf0b1510,
      0xbf85008d, 0x4a0c0415, 0x2c0e0c86, 0x2c100c82, 0xd2900009, 0x02110506,
      0x36140c83, 0x2c141481, 0x36160c82, 0x2c180c81, 0x2c1a0c83, 0x361c0c87,
      0xb0160000, 0xbf0b1610, 0xbf85007d, 0x4a1e0616, 0xb7960001, 0xbf008013,
      0xbf85000d, 0x2c201e83, 0xd2d20010, 0x00002910, 0x4a20210d, 0x36222087,
      0x361e1e87, 0x362020c8, 0x34222283, 0x4a1e210f, 0x4a20230e, 0x341e1e86,
      0x4a1e1f10, 0xbf820027, 0xd2900010, 0x0211050f, 0x34202084, 0x2c221e86,
      0x4a202109, 0x2c241e81, 0xd2d20011, 0x00000111, 0x2c261e82, 0x34202081,
      0x3a28150f, 0x3a1e1f06, 0x3a24250c, 0x4a222307, 0x362020c4, 0xd2940014,
      0x042e2881, 0x3a1e1f0c, 0x36242481, 0x3a262708, 0x34222289, 0x38202910,
      0x361e1e81, 0x34242481, 0x36262681, 0x4a202111, 0x34222682, 0x381e250f,
      0x34242082, 0x381e1f11, 0x362224ff, 0x000000ff, 0x341e1e88, 0xd2900010,
      0x02610d10, 0x4a1e1f11, 0x3420208b, 0x4a1e210f, 0x2c1e1e82, 0xe0002000,
      0x80010f0f, 0xbf8c0f70, 0x36201e8f, 0xd10a0018, 0x00011b10, 0x7d0a208f,
      0x87ea6a18, 0xbe98246a, 0x8afe7e18, 0xbf880038, 0xbf008011, 0xbf850030,
      0xd2900010, 0x0239250f, 0xd2900011, 0x0219190f, 0x7d882290, 0xbe9a246a,
      0x8afe7e1a, 0x362422a0, 0xbf88001b, 0x7d0a2480, 0xbe9c246a, 0xd290000f,
      0x02111d0f, 0xbf880009, 0x36222283, 0x4a241ec4, 0x4a1e1ec2, 0x32222511,
      0x321e1e81, 0x381e1f11, 0xd23c0011, 0x00010112, 0x3822230f, 0x8afe7e1c,
      0xd290000f, 0x020d1f0f, 0xbf880009, 0x36222287, 0x4a241ec1, 0x4a1e1e82,
      0x32222511, 0x321e1e81, 0x381e1f11, 0xd23c0011, 0x00010112, 0x3822230f,
      0xbefe041a, 0xbf008012, 0xbf850003, 0x4c1e2310, 0x24221e80, 0xbf82000a,
      0x4a1e2310, 0x221e1eff, 0x00003fff, 0x7e220310, 0x7e20030f, 0xbf820004,
      0xd2900011, 0x0239090f, 0xd2900010, 0x0239250f, 0x260a2305, 0x28082104,
      0xbefe0418, 0xbf82ff81, 0xb7950001, 0xbf82ff71, 0xbf008001, 0xbf850023,
      0x4c040b04, 0x7d860490, 0xbe80246a, 0x7e067302, 0xbf880014, 0x4c06069f,
      0x7d0a0480, 0x000606c1, 0xd1820004, 0x00010d03, 0x4a0c06c2, 0xbe842404,
      0x340c0c83, 0xbf880004, 0x4a0606c3, 0x2a040702, 0x36040487, 0x38040506,
      0x8afe7e04, 0x34060682, 0xbf880004, 0x4a060688, 0x2a040d02, 0x36040483,
      0x38040503, 0xbefe0400, 0xd10a006a, 0x00010002, 0x00060905, 0x34060692,
      0x360404bf, 0x3404048c, 0x38040503, 0x3804048f, 0xbf820006, 0x34040892,
      0x36060aff, 0x00003fff, 0x34060684, 0x38040702, 0x3804048f, 0xbf008003,
      0xbf850017, 0xc2000d08, 0xbf8c007f, 0x90008200, 0xb7800007, 0x90008300,
      0xb7800001, 0x2c060283, 0x90008100, 0xd2d20003, 0x00000103, 0x2c080083,
      0x34060681, 0x4a060704, 0x36080687, 0x36020287, 0x360606c8, 0x36000087,
      0x34080883, 0x4a020701, 0x4a000900, 0x34020286, 0x4a000300, 0xbf820034,
      0xc2000d08, 0xbf8c007f, 0x90008200, 0xd2900003, 0x02110501, 0x36080083,
      0xb780003f, 0xd2900005, 0x02110500, 0x34060684, 0x2c080881, 0x2c0c0286,
      0x90008600, 0x4a060705, 0x3a080901, 0x2c0a0081, 0x2c0e0281, 0x2c100086,
      0xd2d20006, 0x00000106, 0x2c120082, 0x2c140282, 0x34060681, 0x36080881,
      0x3a020300, 0x3a0e0f05, 0x4a0c0d08, 0x360606c4, 0xd2940000, 0x04120082,
      0x3a020b01, 0x36080e81, 0x3a0a1509, 0x340c0c89, 0x38000103, 0x36020281,
      0x34060881, 0x36080a81, 0x4a000106, 0x34080882, 0x38020701, 0x34060082,
      0x38020304, 0x360606ff, 0x000000ff, 0x34020288, 0xd2900000, 0x02610d00,
      0x4a020303, 0x3400008b, 0x4a000101, 0x2c000082, 0x7d88000a, 0xbe80246a,
      0xeba42000, 0x80020200, 0xbf810000,
  };
  for (u32 trial = 0; trial < 3; ++trial) {
    u32* code = memory + trial * 1024;
    std::memcpy(code, shader, sizeof(shader));
    if (trial == 2) {
      code[0xc9] = 0x38040480;
      code[0xd0] = 0x38040480;
    }
    u32* params = memory + 4096 + trial * 64;
    const u32 config[] = {2, 1, 0, 0, 1024, 1, 0, trial & 1, 512};
    std::memcpy(params, config, sizeof(config));
    u32* source = memory + 16384 + trial * 16384;
    std::fill(source, source + 16384, 0xfffffff0);
    u32* output = memory + 131072 + trial * 16384;
    std::fill(output, output + 16384, 0xa5a5a5a5);
    const u64 base = reinterpret_cast<u64>(output);
    auto* depth = gpu::render::GetDepthRT(
        reinterpret_cast<u64>(memory + 204800 + trial * 4096), 8, 8);
    ASSERT_NE(depth, nullptr);
    depth->htile_base = base;
    depth->clear_pending = false;
    gpu::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code);
    regs[gpu::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::mmCOMPUTE_NUM_THREAD_X] = 8;
    regs[gpu::mmCOMPUTE_NUM_THREAD_Y] = 8;
    regs[gpu::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::mmCOMPUTE_PGM_RSRC2] = (16 << 1) | (3 << 7);
    const auto descriptor = [&](u32 offset, u32* data, u32 stride, u32 records,
                                u32 format) {
      const u64 addr = reinterpret_cast<u64>(data);
      regs[gpu::mmCOMPUTE_USER_DATA_0 + offset] = addr;
      regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 1] =
          (addr >> 32) | (stride << 16);
      regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 2] = records;
      regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 3] = format;
    };
    descriptor(4, source, 4, 16384, 0x20024004);
    descriptor(8, output, 4, 16384, 0x18024004);
    descriptor(12, params, 16, 3, 0x20077fac);
    const u32 groups[] = {1, 1, 1, 1};
    gpu::render::BeginFrame(renderer);
    gpu::ps4::DispatchCompute(renderer, regs, groups, 4);
    EXPECT_EQ(depth->htile_clear_pending, trial == 2);
    EXPECT_FALSE(depth->clear_pending);
    EXPECT_EQ(output[0], 0xa5a5a5a5u);
    EXPECT_TRUE(gpu::render::CsRangeDirtyOverlapping(base, 4));
    ASSERT_TRUE(
        gpu::render::FlushCsWritesRange(renderer, base, 4, "test-htile"));
    EXPECT_EQ(output[0] & 15, trial == 2 ? 0u : 15u);
    gpu::render::NoteDccWrite(base, 4, nullptr);
    EXPECT_TRUE(depth->htile_clear_pending);
    gpu::render::EndFrame(renderer, 0);
  }
  host_memory::FreeMem(memory, 1048576);
}

TEST(GcnCompute, RepeatFillMatchesGpuAndClearsRenderTargetVariants) {
  options::Init();
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  base::OptionBase* cpu_fill = nullptr;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_GPU_CPU_FILL") == 0)
      cpu_fill = const_cast<base::OptionBase*>(option);
  });
  ASSERT_NE(cpu_fill, nullptr);
  auto* memory = static_cast<u32*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1200000000ull), 65536,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(memory, nullptr);
  const u32 code[] = {
      0xbeeb03ff, 0x0000001c, 0x8f008610, 0x4a000000, 0xc2400d00, 0xbf8c007f,
      0x7d880000, 0xbe82246a, 0x7e020c01, 0xbf88002c, 0x7e025501, 0x100202ff,
      0x4f800000, 0x7e020f01, 0xd2d20002, 0x00020201, 0xd2d40003, 0x00020201,
      0x4c080480, 0xd10a0002, 0x00020680, 0xd2000002, 0x000a0504, 0xd2d40002,
      0x00020302, 0x4c060501, 0x4a020501, 0xd2000001, 0x000a0701, 0xd2d40001,
      0x00020101, 0xd2d20001, 0x00000301, 0x4c040300, 0xd18c0002, 0x00020300,
      0xd18c000c, 0x00000302, 0x4e020401, 0x87ea0c02, 0x00020302, 0x4a040201,
      0xd2000001, 0x000a0302, 0xd10a006a, 0x00000280, 0x000202c1, 0xe0002000,
      0x80010101, 0x7d88000a, 0xbe80246a, 0xbf8c0f70, 0xe0102000, 0x80020100,
      0xbf810000,
  };
  std::memcpy(memory, code, sizeof(code));
  u32* source = memory + 256;
  u32* params = memory + 512;
  u32* result = memory + 1024;
  source[0] = 0x12345678;
  source[1] = 0x87654321;
  params[1] = 1;
  gpu::Regs regs;
  const u64 address = reinterpret_cast<u64>(memory);
  regs[gpu::mmCOMPUTE_PGM_LO] = static_cast<u32>(address >> 8);
  regs[gpu::mmCOMPUTE_PGM_HI] = static_cast<u32>(address >> 40);
  regs[gpu::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::mmCOMPUTE_PGM_RSRC2] = (16 << 1) | (1 << 7);
  auto descriptor = [&](u32 offset, u32* base, u32 stride, u32 count) {
    const u64 addr = reinterpret_cast<u64>(base);
    regs[gpu::mmCOMPUTE_USER_DATA_0 + offset] = static_cast<u32>(addr);
    regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 1] =
        static_cast<u32>(addr >> 32) | (stride << 16);
    regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 2] = count;
    regs[gpu::mmCOMPUTE_USER_DATA_0 + offset + 3] = 0x20024004;
  };
  descriptor(4, source, 4, 1);
  descriptor(8, result, 4, 103);
  descriptor(12, params, 16, 1);
  const u32 groups[] = {2, 1, 1, 1};
  struct Case {
    u32 limit, records, pattern;
  };
  const Case cases[] = {
      {97, 103, 1}, {200, 103, 1}, {200, 160, 1}, {97, 103, 2}};
  u32 trial = 0;
  u32 case_index = 0;
  for (const auto& c : cases) {
    params = memory + 512 + case_index++ * 16;
    descriptor(12, params, 16, 1);
    params[0] = c.limit;
    params[1] = c.pattern;
    descriptor(4, source, 4, c.pattern);
    descriptor(8, result, 4, c.records);
    for (const char* enabled : {"0", "1"}) {
      result = memory + 1024 + trial++ * 512;
      descriptor(8, result, 4, c.records);
      ASSERT_TRUE(cpu_fill->SetFromString(enabled));
      std::fill(result, result + 256, 0xfeedface);
      gpu::render::BeginFrame(renderer);
      gpu::ps4::DispatchCompute(renderer, regs, groups, 4);
      ASSERT_TRUE(gpu::render::FlushCsWritesRange(
          renderer, reinterpret_cast<u64>(result), 1024, "test-fill"));
      gpu::render::EndFrame(renderer, 0);
      for (u32 i = 0; i < 256; i++) {
        const u32 count = std::min(std::min(c.limit, c.records), 128u);
        EXPECT_EQ(result[i], i < count ? source[i % c.pattern] : 0xfeedface)
            << "limit=" << c.limit << " cpu=" << enabled << " i=" << i;
      }
    }
  }
  result = memory + 8192;
  const u64 target_base = reinterpret_cast<u64>(result);
  source[0] = 0x38003400;
  source[1] = 0x3c003a00;
  params = memory + 768;
  params[0] = 256;
  params[1] = 2;
  descriptor(4, source, 4, 2);
  descriptor(8, result, 4, 256);
  descriptor(12, params, 16, 1);
  ASSERT_TRUE(cpu_fill->SetFromString("1"));
  gpu::render::BeginFrame(renderer);
  for (u32 width : {32u, 16u}) {
    auto* rt = gpu::render::GetRT(target_base, width, 4,
                                  gpu::rhi::Format::kRGBA16Float);
    ASSERT_NE(rt, nullptr);
    rt->cb_info = (12u << 2) | (7u << 8);
  }
  ASSERT_TRUE(gpu::render::CanClearMemoryFill64(target_base, 1024));
  EXPECT_FALSE(gpu::render::CanClearMemoryFill64(target_base, 512));
  EXPECT_FALSE(gpu::render::CanClearMemoryFill64(target_base + 4, 1024));
  const u32 wide_groups[] = {4, 1, 1, 1};
  gpu::ps4::DispatchCompute(renderer, regs, wide_groups, 4);
  for (u32 i = 0; i < 256; i++)
    EXPECT_EQ(result[i], source[i & 1]);
  gpu::rhi::BufferDesc readback_desc;
  readback_desc.size = 1536;
  readback_desc.usage = gpu::rhi::kBufferCopyDst;
  readback_desc.memory = gpu::rhi::MemoryKind::kReadback;
  auto* readback = gpu::render::Device().CreateBuffer(readback_desc);
  ASSERT_NE(readback, nullptr);
  u64 offset = 0;
  for (u32 width : {32u, 16u}) {
    ASSERT_TRUE(gpu::render::ActivateRtVariantAs(
        target_base, width, 4, gpu::rhi::Format::kRGBA16Float));
    auto& rt = gpu::render::g_rts[target_base];
    gpu::rhi::TextureBarrier barrier{rt.texture, rt.layout,
                                     gpu::rhi::TextureState::kCopySrc};
    gpu::render::g_frame.list->Barrier(0, 0, &barrier, 1);
    rt.layout = gpu::rhi::TextureState::kCopySrc;
    gpu::rhi::BufferTextureCopy copy;
    copy.buffer_offset = offset;
    copy.region.width = width;
    copy.region.height = 4;
    gpu::render::g_frame.list->CopyTextureToBuffer(readback, rt.texture, &copy,
                                                   1);
    offset += u64(width) * 4 * 8;
  }
  gpu::render::g_frame.list->Barrier(gpu::rhi::kAccessCopyWrite,
                                     gpu::rhi::kAccessHostRead);
  const u32 slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, 0);
  ASSERT_TRUE(gpu::render::Device().Wait(
      gpu::render::g_frame.slots[slot_index].submission));
  const auto* pixels = reinterpret_cast<const u32*>(readback->mapped());
  for (u32 i = 0; i < 1536 / 4; i++)
    EXPECT_EQ(pixels[i], source[i & 1]) << "word=" << i;
  gpu::render::Device().Destroy(readback);
  cpu_fill->Reset();
  host_memory::FreeMem(memory, 65536);
}
