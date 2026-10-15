#include <gtest/gtest.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include "base/containers/array.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gcn/spirv/spv_emit.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/guest_page_table.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/render/compute/internal.h"
#include "gpu/render/device.h"
#include "gpu/render/draw/draw_recomp.h"
#include "gpu/render/frame.h"
#include "gpu/render/memory/guest_direct.h"
#include "gpu/render/draw/index_convert.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/memory/upload_ring.h"
#include "gpu/write_tracker.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

extern "C" void prosperity_gpu_end_of_pipe() {}
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}

namespace {

struct DirectMemory {
  static constexpr u64 kBase = 0x2001000000ull;
  static constexpr u64 kLateBase = kBase + (32ull << 20);
  static constexpr u64 kHighBase = 0x8001000000ull;
  static constexpr u64 kArenaBase = 0x9001000000ull;
  static constexpr u64 kModuleBase = 0xc001000000ull;
  static constexpr u64 kEndBase = 0xffff800000ull;
  static constexpr u64 kBytes = 4ull << 20;
  static constexpr u64 kMappedBytes = 2 * kBytes;
  int fd = memfd_create("delta_dmem_test", 0);
  void* memory = MAP_FAILED;
  void* high_memory = MAP_FAILED;
  void* extra_memory[3] = {MAP_FAILED, MAP_FAILED, MAP_FAILED};
  void* late_memory = MAP_FAILED;

  DirectMemory() {
    if (fd >= 0 && !ftruncate(fd, kMappedBytes))
      memory =
          mmap(reinterpret_cast<void*>(kBase), kMappedBytes,
               PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
    if (memory != MAP_FAILED)
      high_memory =
          mmap(reinterpret_cast<void*>(kHighBase), kMappedBytes,
               PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
    const u64 extra_bases[] = {kArenaBase, kModuleBase, kEndBase};
    if (memory != MAP_FAILED)
      for (u32 i = 0; i < base::ArraySize(extra_bases); ++i)
        extra_memory[i] = mmap(reinterpret_cast<void*>(extra_bases[i]),
                               kMappedBytes, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
  }

  ~DirectMemory() {
    for (const u64 base : {kBase, kHighBase, kLateBase}) {
      void* mapping = base == kBase       ? memory
                      : base == kHighBase ? high_memory
                                          : late_memory;
      const u64 bytes =
          base == kLateBase ? gpu::render::kMirrorBlock : kMappedBytes;
      if (mapping == MAP_FAILED)
        continue;
      if (gpu::render::g_mirror) {
        gpu::render::cs::CsBatchSubmit();
        for (u64 at = base; at < base + bytes; at += gpu::render::kMirrorBlock)
          if (gpu::render::MirrorOwned(at)) {
            const u32 value = *reinterpret_cast<volatile const u32*>(at);
            (void)value;
          }
      }
      gpu::GuestWriteTracker().Release(base, bytes);
      munmap(mapping, bytes);
      gpu::render::NoteGuestRemap(base, bytes);
    }
    const u64 extra_bases[] = {kArenaBase, kModuleBase, kEndBase};
    for (u32 i = 0; i < base::ArraySize(extra_bases); ++i) {
      if (extra_memory[i] == MAP_FAILED)
        continue;
      gpu::GuestWriteTracker().Release(extra_bases[i], kMappedBytes);
      munmap(extra_memory[i], kMappedBytes);
      gpu::render::NoteGuestRemap(extra_bases[i], kMappedBytes);
    }
    if (fd >= 0)
      close(fd);
  }
};

struct HostMemory {
  static constexpr u64 kBase = 0x2100000000ull;
  static constexpr u64 kBytes = 3 * gpu::render::kMirrorBlock;
  void* memory = MAP_FAILED;
  HostMemory() {
    memory = mmap(reinterpret_cast<void*>(kBase), kBytes, PROT_NONE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (memory != MAP_FAILED)
      host_memory::TrackMemoryMapping(memory, kBytes, true);
  }
  bool Commit(u64 offset, u64 bytes) {
    void* address = reinterpret_cast<void*>(kBase + offset);
    if (mmap(address, bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != address)
      return false;
    host_memory::TrackMemoryMapping(address, bytes);
    gpu::render::NoteGuestRemap(kBase + offset, bytes);
    return true;
  }
  ~HostMemory() {
    if (memory == MAP_FAILED)
      return;
    gpu::render::Device().WaitIdle();
    gpu::GuestWriteTracker().Release(kBase, kBytes);
    munmap(memory, kBytes);
    host_memory::ForgetMemoryMapping(memory, kBytes);
    gpu::render::NoteGuestRemap(kBase, kBytes);
  }
};

TEST(VkMirrorBinding, OwnedOutputStaysResidentAcrossDrawsAndCpuWrites) {
  options::Init();
  DirectMemory memory;
  ASSERT_NE(memory.memory, MAP_FAILED);
  ASSERT_NE(memory.high_memory, MAP_FAILED);
  for (void* alias : memory.extra_memory)
    ASSERT_NE(alias, MAP_FAILED);
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A graphics device is required";
  const auto& caps = gpu::render::Device().caps();
  if (!caps.sparse_buffer || !caps.host_import || !caps.buffer_address ||
      !caps.copy_after)
    GTEST_SKIP()
        << "Sparse buffers, host imports and ordered copies are required";
  gpu::GuestWriteTracker().Enable();
  static u8 fault_stack[65536];
  stack_t stack{};
  stack.ss_sp = fault_stack;
  stack.ss_size = sizeof(fault_stack);
  ASSERT_EQ(sigaltstack(&stack, nullptr), 0);
  struct sigaction fault{};
  fault.sa_flags = SA_SIGINFO | SA_ONSTACK;
  fault.sa_sigaction = [](int, siginfo_t* si, void*) {
    if (!host_memory::HandleWriteFault(
            reinterpret_cast<uintptr_t>(si->si_addr)))
      std::abort();
  };
  ASSERT_EQ(sigaction(SIGSEGV, &fault, nullptr), 0);
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                        0xbf810000};
  const u32 ps[4096] = {0x7e000204, 0xe8001000 | (56u << 19) | (3u << 16),
                        0x80020000, 0xf800080f,
                        0x03020100, 0xbf810000};
  const u64 base = DirectMemory::kBase;
  const u32 user_data[32] = {u32(base), u32(base >> 32), DirectMemory::kBytes,
                             0x14fac};
  const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  ASSERT_EQ(program.ps_bufs.size(), 1u);
  const u32 cbuf_ps[4096] = {0xf4000100, 0xfa000000, 0x7e000204,
                             0x7e020280, 0x7e040280, 0x7e0602f2,
                             0xf800080f, 0x03020100, 0xbf810000};
  const auto constants =
      gpu::rdna::Recompile(vs, cbuf_ps, user_data, user_data, 0, false, 8, 8);
  ASSERT_TRUE(constants.ok);
  ASSERT_EQ(constants.ps_cbufs.size(), 1u);
  ASSERT_FALSE(constants.indirect_cbufs);
  const u32 mixed_ps[4096] = {0xf4000100, 0xfa000000, 0xf4000140, 0xfa002000,
                              0x7e000204, 0x7e020205, 0x7e040280, 0x7e0602f2,
                              0xf800080f, 0x03020100, 0xbf810000};
  const auto mixed_constants =
      gpu::rdna::Recompile(vs, mixed_ps, user_data, user_data, 0, false, 8, 8);
  ASSERT_TRUE(mixed_constants.ok);
  ASSERT_EQ(mixed_constants.ps_cbufs.size(), 1u);
  ASSERT_FALSE(mixed_constants.indirect_cbufs);
  const u32 cs[4096] = {0x7e020204, 0xe0702000, 0x80000100, 0xbf810000};
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto compute = gpu::gcn::RecompileCompute(cs, 64, 1, 1, 5, 0, 0);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(compute.ok);
  ASSERT_EQ(compute.resources.size(), 1u);
  const u32 red_ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                            0xbf810000};
  auto geometry = gpu::rdna::Recompile(vs, red_ps, user_data, user_data);
  ASSERT_TRUE(geometry.ok);
  using gpu::gcn::spirv::Id;
  gpu::gcn::spirv::Module module;
  const Id f = module.TypeFloat(), v4 = module.TypeVec(f, 4);
  const Id vertex =
      module.Variable(module.TypePointer(spv::StorageClass::Input, f),
                      spv::StorageClass::Input);
  module.Decorate(vertex, spv::Decoration::Location, {0});
  const Id position =
      module.Variable(module.TypePointer(spv::StorageClass::Output, v4),
                      spv::StorageClass::Output);
  module.Decorate(position, spv::Decoration::BuiltIn,
                  {static_cast<u32>(spv::BuiltIn::Position)});
  const Id point_size =
      module.Variable(module.TypePointer(spv::StorageClass::Output, f),
                      spv::StorageClass::Output);
  module.Decorate(point_size, spv::Decoration::BuiltIn,
                  {static_cast<u32>(spv::BuiltIn::PointSize)});
  const Id main = module.BeginFunction(
      module.TypeVoid(), module.TypeFunction(module.TypeVoid(), {}));
  module.Store(position,
               module.Emit(spv::Op::OpCompositeConstruct, v4,
                           {module.Load(f, vertex), module.ConstF32(0),
                            module.ConstF32(0), module.ConstF32(1)}));
  module.Store(point_size, module.ConstF32(1));
  module.ReturnVoid();
  module.EndFunction();
  module.EntryPoint(spv::ExecutionModel::Vertex, main, "main",
                    {vertex, position, point_size});
  base::String error;
  ASSERT_TRUE(
      gpu::gcn::spirv::Finalize(module.Assemble(), &geometry.vs_spirv, &error))
      << error.c_str();
  ASSERT_TRUE(gpu::render::InitGuestDirect());
  u64 late_offset = 0, first_offset = 0;
  ASSERT_EQ(gpu::render::DirectBindAt(DirectMemory::kLateBase, 4, &late_offset),
            nullptr);
  memory.late_memory = mmap(reinterpret_cast<void*>(DirectMemory::kLateBase),
                            gpu::render::kMirrorBlock, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_FIXED_NOREPLACE, memory.fd, 0);
  ASSERT_NE(memory.late_memory, MAP_FAILED);
  gpu::render::NoteGuestRemap(DirectMemory::kLateBase,
                              gpu::render::kMirrorBlock);
  auto* late_import =
      gpu::render::DirectBindAt(DirectMemory::kLateBase, 4, &late_offset);
  ASSERT_NE(late_import, nullptr);
  EXPECT_EQ(late_import, gpu::render::DirectBindAt(base, 4, &first_offset));
  EXPECT_EQ(late_offset, first_offset);
  ASSERT_EQ(mmap(memory.late_memory, gpu::render::kMirrorBlock,
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, memory.fd,
                 gpu::render::kMirrorBlock),
            memory.late_memory);
  gpu::render::NoteGuestRemap(DirectMemory::kLateBase,
                              gpu::render::kMirrorBlock);
  EXPECT_EQ(late_import, gpu::render::DirectBindAt(DirectMemory::kLateBase, 4,
                                                   &late_offset));
  EXPECT_EQ(late_offset, first_offset + gpu::render::kMirrorBlock);
  alignas(65536) static base::Array<u8, 65536> targets[68]{};
  *static_cast<u32*>(memory.memory) = 0x3e800000;
  gpu::render::DrawInfo imported;
  imported.recomp = &constants;
  imported.vs_addr = reinterpret_cast<u64>(vs);
  imported.ps_addr = reinterpret_cast<u64>(cbuf_ps);
  std::memcpy(imported.ps_user_data, user_data, sizeof(user_data));
  imported.prim_type = imported.vertex_count = 1;
  imported.cbufs[0] = {base, gpu::render::kCbufWindow};
  imported.num_cbufs = 1;
  imported.rt_base = imported.mrt_base[0] =
      reinterpret_cast<u64>(targets[3].data());
  imported.rt_w = imported.rt_h = 16;
  imported.viewport_x_scale = imported.viewport_y_scale = 8;
  imported.viewport_x_offset = imported.viewport_y_offset = 8;
  imported.mrt_count = imported.mrt_bound_mask = 1;
  imported.mrt_info[0] = 10u << 2;
  gpu::render::BeginFrame(renderer);
  const u64 imported_bytes = gpu::g_stats.bind_bytes;
  const u64 imported_copies = gpu::g_stats.ring_cb_bytes;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, imported));
  EXPECT_EQ(gpu::g_stats.bind_bytes, imported_bytes + gpu::render::kCbufWindow);
  EXPECT_EQ(gpu::g_stats.ring_cb_bytes, imported_copies);
  const u32 imported_slot = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, imported.rt_base);
  const auto& slot = gpu::render::g_frame.slots[imported_slot];
  ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
  ASSERT_TRUE(slot.presentable);
  u32 imported_pixels = 0;
  for (u32 i = 0; i < 256; ++i)
    if (const u8 red = slot.readback->mapped()[i * 4]) {
      EXPECT_EQ(red, 64);
      ++imported_pixels;
    }
  EXPECT_EQ(imported_pixels, 1u);
  for (u32 block : {2u, 3u})
    ASSERT_TRUE(gpu::render::GpuFill(base + block * gpu::render::kMirrorBlock,
                                     gpu::render::kMirrorBlock, 0, 0));
  EXPECT_NE(gpu::render::MirrorWritten(base + 2 * gpu::render::kMirrorBlock,
                                       2 * gpu::render::kMirrorBlock),
            nullptr);
  gpu::render::ComputeInfo ci;
  ci.cs_addr = reinterpret_cast<u64>(cs);
  ci.recomp = &compute;
  ci.groups[0] = ci.groups[1] = ci.groups[2] = 1;
  ci.num_res = 1;
  auto& resource = ci.res[0];
  resource.base = base;
  resource.size = resource.guest_size = DirectMemory::kBytes;
  resource.binding = compute.resources[0].binding;
  resource.written = resource.shader_writes = true;
  ci.user_data[0] = u32(base);
  ci.user_data[1] = (base >> 32) | (4u << 16);
  ci.user_data[2] = DirectMemory::kBytes / 4;
  ci.user_data[3] = (4u << 15) | (4u << 12);
  u64 latest_revision = 0;
  for (u32 trial = 0; trial < 3; ++trial) {
    SCOPED_TRACE(trial);
    gpu::render::BeginFrame(renderer);
    if (trial)
      gpu::render::RecordComputeInFrame();
    ci.user_data[4] = 0xffff8040;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    ASSERT_TRUE(gpu::render::MirrorOwned(base));
    ASSERT_TRUE(gpu::render::cs::g_pending_guest_writes.empty());
    ASSERT_TRUE(gpu::render::CsHoldsGeometry(base, DirectMemory::kBytes));
    gpu::rhi::Buffer* buffer = nullptr;
    u64 offset = 0;
    ASSERT_TRUE(gpu::render::CsGeometryBuffer(base, DirectMemory::kBytes,
                                              &buffer, &offset));
    EXPECT_EQ(buffer, gpu::render::g_mirror);
    EXPECT_EQ(offset, gpu::render::MirrorOffset(base));
    gpu::render::DrawInfo draw;
    draw.recomp = &program;
    draw.vs_addr = reinterpret_cast<u64>(vs);
    draw.ps_addr = reinterpret_cast<u64>(ps);
    std::memcpy(draw.ps_user_data, user_data, sizeof(user_data));
    draw.prim_type = draw.vertex_count = 1;
    draw.bufs[0] = {base, DirectMemory::kBytes};
    draw.num_bufs = 1;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[trial].data());
    draw.rt_w = draw.rt_h = 16;
    draw.viewport_x_scale = draw.viewport_y_scale = 8;
    draw.viewport_x_offset = 4;
    draw.viewport_y_offset = 8;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    const u64 cpu_copies = gpu::g_stats.ring_raw_bytes;
    const u64 cpu_constants = gpu::g_stats.ring_cb_bytes;
    const u64 bound = gpu::g_stats.bind_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    ci.user_data[4] = 0xff804020;
    if (trial == 2)
      ASSERT_TRUE(gpu::render::GpuFill(base, DirectMemory::kBytes,
                                       ci.user_data[4], ci.user_data[4]));
    else
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    draw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    gpu::render::DrawInfo cb_draw = draw;
    cb_draw.recomp = &constants;
    cb_draw.ps_addr = reinterpret_cast<u64>(cbuf_ps);
    cb_draw.cbufs[0] = {base, gpu::render::kCbufWindow};
    cb_draw.num_cbufs = 1;
    cb_draw.num_bufs = 0;
    cb_draw.bufs[0] = {};
    cb_draw.viewport_x_offset = cb_draw.viewport_y_offset = 4;
    ci.user_data[4] = 0x3e800000;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, cb_draw));
    ci.user_data[4] = 0x3f400000;
    if (trial == 2)
      ASSERT_TRUE(gpu::render::GpuFill(base, DirectMemory::kBytes,
                                       ci.user_data[4], ci.user_data[4]));
    else
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    cb_draw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, cb_draw));
    EXPECT_EQ(gpu::render::cs::g_cs_stats.buf_bridge_bytes, copies);
    EXPECT_EQ(gpu::g_stats.ring_raw_bytes, cpu_copies);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, cpu_constants);
    EXPECT_EQ(gpu::g_stats.bind_bytes,
              bound + 2 * DirectMemory::kBytes + 2 * gpu::render::kCbufWindow);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    latest_revision = gpu::render::CsBufferRevision(base, DirectMemory::kBytes);
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    u32 colored = 0;
    for (u32 i = 0; i < 256; ++i) {
      if (!pixels[i * 4])
        continue;
      ++colored;
      const bool first = i % 16 < 8;
      const bool cbuf = i / 16 < 6;
      EXPECT_EQ(pixels[i * 4], first ? 64 : cbuf ? 191 : 32);
      EXPECT_EQ(pixels[i * 4 + 1], cbuf ? 0 : first ? 128 : 64);
      EXPECT_EQ(pixels[i * 4 + 2], cbuf ? 0 : first ? 255 : 128);
      EXPECT_EQ(pixels[i * 4 + 3], 255);
    }
    EXPECT_EQ(colored, 4u);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    *static_cast<volatile u32*>(memory.memory) = 0xff0000ff;
    gpu::render::CollectGuestWrites();
    EXPECT_FALSE(gpu::render::CsHoldsGeometry(base, DirectMemory::kBytes));
    EXPECT_GT(gpu::render::cs::g_mirror_resolves.load(), resolves);
  }
  for (u32 mode = 0; mode < 2; ++mode) {
    SCOPED_TRACE(mode);
    gpu::render::cs::g_cs_in_frame = mode != 0;
    ASSERT_TRUE(gpu::render::GpuFill(base + gpu::render::kMirrorBlock,
                                     gpu::render::kMirrorBlock, 0x3f400000,
                                     0x3f400000));
    ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
    *static_cast<volatile u32*>(memory.memory) = 0x3e800000;
    gpu::render::CollectGuestWrites();
    gpu::render::BeginFrame(renderer);
    EXPECT_GT(gpu::render::CsBufferRevision(base, DirectMemory::kBytes),
              latest_revision);
    ASSERT_TRUE(gpu::render::MirrorOwned(base + gpu::render::kMirrorBlock));
    const u64 mixed_resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 mixed_copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    const u64 mixed_cpu_raw = gpu::g_stats.ring_raw_bytes;
    const u64 mixed_cpu_cb = gpu::g_stats.ring_cb_bytes;
    gpu::render::DrawInfo mixed = imported;
    mixed.rt_base = mixed.mrt_base[0] =
        reinterpret_cast<u64>(targets[4 + mode].data());
    mixed.viewport_x_offset = mixed.viewport_y_offset = 4;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, mixed));
    gpu::render::DrawInfo raw = mixed;
    raw.recomp = &program;
    raw.ps_addr = reinterpret_cast<u64>(ps);
    raw.num_cbufs = 0;
    raw.cbufs[0] = {};
    raw.bufs[0] = {base, DirectMemory::kBytes};
    raw.num_bufs = 1;
    raw.viewport_y_offset = 12;
    const u64 raw_ring = gpu::render::g_ring.sbo_offset;
    const u64 raw_bound = gpu::g_stats.bind_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, raw));
    raw.ps_user_data[4] = gpu::render::kMirrorBlock;
    raw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, raw));
    EXPECT_EQ(gpu::render::g_ring.sbo_offset, raw_ring);
    EXPECT_EQ(gpu::g_stats.bind_bytes, raw_bound + 2 * DirectMemory::kBytes);
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 256, 256, 256, 1, 1, 8,
                                               false, 4));
    gpu::rhi::TextureDesc texture_desc;
    texture_desc.width = texture_desc.height = 256;
    texture_desc.format = gpu::rhi::Format::kRGBA8Unorm;
    texture_desc.usage = gpu::rhi::kTextureSampled | gpu::rhi::kTextureCopyDst |
                         gpu::rhi::kTextureCopySrc;
    auto* texture = gpu::render::Device().CreateTexture(texture_desc);
    ASSERT_NE(texture, nullptr);
    gpu::rhi::BufferDesc readback_desc;
    readback_desc.size = 2 * layout.size;
    readback_desc.usage = gpu::rhi::kBufferCopyDst;
    readback_desc.memory = gpu::rhi::MemoryKind::kReadback;
    auto* readback = gpu::render::Device().CreateBuffer(readback_desc);
    ASSERT_NE(readback, nullptr);
    u64 texture_revision = 0;
    auto texture_state = gpu::rhi::TextureState::kUndefined;
    const u64 texture_copies = gpu::render::cs::g_cs_stats.tex_bridge_bytes;
    const auto supply_texture = [&](u32 pass) {
      if (!gpu::render::MirrorSupplyTexture(base, layout, layout, texture,
                                            texture_state, &texture_revision))
        return false;
      gpu::rhi::TextureBarrier barrier{texture,
                                       gpu::rhi::TextureState::kShaderRead,
                                       gpu::rhi::TextureState::kCopySrc};
      auto* list = gpu::render::g_frame.list;
      list->Barrier(gpu::rhi::kAccessShaderRead, gpu::rhi::kAccessCopyRead,
                    &barrier, 1);
      gpu::rhi::BufferTextureCopy copy;
      copy.buffer_offset = pass * layout.size;
      copy.region.width = copy.region.height = 256;
      list->CopyTextureToBuffer(readback, texture, &copy, 1);
      barrier.before = gpu::rhi::TextureState::kCopySrc;
      barrier.after = gpu::rhi::TextureState::kShaderRead;
      list->Barrier(gpu::rhi::kAccessCopyRead, gpu::rhi::kAccessShaderRead,
                    &barrier, 1);
      texture_state = gpu::rhi::TextureState::kShaderRead;
      return true;
    };
    ASSERT_TRUE(supply_texture(0));
    const u64 before_fill =
        gpu::render::CsBufferRevision(base, DirectMemory::kBytes);
    ASSERT_TRUE(gpu::render::GpuFill(base + gpu::render::kMirrorBlock,
                                     gpu::render::kMirrorBlock, 0x3f800000,
                                     0x3f800000));
    EXPECT_GT(gpu::render::CsBufferRevision(base, DirectMemory::kBytes),
              before_fill);
    EXPECT_NE(gpu::render::MirrorWritten(base + gpu::render::kMirrorBlock,
                                         2 * gpu::render::kMirrorBlock),
              nullptr);
    ASSERT_TRUE(gpu::render::SubmitFrameChunk());
    const auto& mixed_slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    ASSERT_TRUE(gpu::render::Device().Wait(mixed_slot.submission));
    const u64 before_cpu_write =
        gpu::render::CsBufferRevision(base, DirectMemory::kBytes);
    const auto* page = gpu::GuestPages().Find(base);
    ASSERT_NE(page, nullptr);
    const bool tracked = page->armed.load(base::memory_order_acquire) != 0;
    *static_cast<volatile u32*>(memory.memory) = 0x3f000000;
    gpu::render::CollectGuestWrites();
    if (tracked)
      EXPECT_GT(gpu::render::CsBufferRevision(base, DirectMemory::kBytes),
                before_cpu_write);
    mixed.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, mixed));
    ASSERT_TRUE(supply_texture(1));
    gpu::render::g_frame.list->Barrier(gpu::rhi::kAccessCopyWrite,
                                       gpu::rhi::kAccessHostRead);
    EXPECT_EQ(gpu::g_stats.ring_raw_bytes, mixed_cpu_raw);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, mixed_cpu_cb);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), mixed_resolves);
    EXPECT_EQ(gpu::render::cs::g_cs_stats.buf_bridge_bytes,
              mixed_copies + (tracked ? 1 : 2) * DirectMemory::kBytes +
                  2 * gpu::render::kCbufWindow);
    EXPECT_EQ(gpu::render::cs::g_cs_stats.tex_bridge_bytes,
              texture_copies + 2 * layout.size);
    gpu::render::EndFrame(renderer, mixed.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(mixed_slot.submission));
    ASSERT_TRUE(mixed_slot.presentable);
    u32 mixed_pixels = 0;
    for (u32 i = 0; i < 256; ++i) {
      const auto* pixel = mixed_slot.readback->mapped() + i * 4;
      if (!pixel[3])
        continue;
      ++mixed_pixels;
      const bool first = i % 16 < 8;
      if (i / 16 < 8) {
        EXPECT_NEAR(pixel[0], first ? 64 : 128, 1);
        EXPECT_EQ(pixel[1], 0);
        EXPECT_EQ(pixel[2], 0);
        EXPECT_EQ(pixel[3], 255);
      } else {
        EXPECT_EQ(pixel[0], 0);
        EXPECT_EQ(pixel[1], 0);
        EXPECT_EQ(pixel[2], first ? 128 : 64);
        EXPECT_EQ(pixel[3], first ? 62 : 63);
      }
    }
    EXPECT_EQ(mixed_pixels, 4u);
    for (u32 pass = 0; pass < 2; ++pass) {
      u32 first, owned;
      std::memcpy(&first, readback->mapped() + pass * layout.size,
                  sizeof(first));
      std::memcpy(
          &owned,
          readback->mapped() + pass * layout.size + gpu::render::kMirrorBlock,
          sizeof(owned));
      EXPECT_EQ(first, pass ? 0x3f000000u : 0x3e800000u);
      EXPECT_EQ(owned, pass ? 0x3f800000u : 0x3f400000u);
    }
    gpu::render::Device().Destroy(texture);
    gpu::render::Device().Destroy(readback);
  }
  alignas(65536) static u32 host_indices[2][32768]{};
  for (u32 trial = 0; trial < 9; ++trial) {
    SCOPED_TRACE(trial);
    const bool vertices = trial % 3 == 0;
    const bool host = trial >= 7;
    const bool partial = (trial >= 3 && trial < 6) || host;
    const u64 geometry_base =
        host ? reinterpret_cast<u64>(host_indices[trial - 7])
             : base + (trial == 6 ? 7ull << 20
                       : partial  ? u64(trial + 1) << 20
                                  : 0);
    const u32 element = trial % 3 == 1 ? 2 : 4;
    const u64 bytes = 2 * gpu::render::kMirrorBlock;
    gpu::render::cs::g_cs_in_frame = trial % 2 == 0;
    const u32 initial = vertices ? 0x3f000000 : element == 2 ? 0x00010001 : 1;
    if (!host) {
      ASSERT_TRUE(gpu::render::GpuFill(
          geometry_base + (partial ? gpu::render::kMirrorBlock : 0),
          partial ? gpu::render::kMirrorBlock : bytes, initial, initial));
      ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
      ASSERT_TRUE(gpu::render::SettleGuestWrites(geometry_base, bytes, "test"));
    }
    if (vertices)
      *reinterpret_cast<volatile float*>(geometry_base) = -.5f;
    else
      std::memset(reinterpret_cast<void*>(geometry_base), 0,
                  gpu::render::kMirrorBlock);
    auto* auxiliary = reinterpret_cast<u32*>(base + (2ull << 20));
    if (vertices) {
      auxiliary[0] = 0;
      auxiliary[1] = gpu::render::kMirrorBlock / 16;
    } else {
      const float positions[] = {-.5f, 0, .5f};
      std::memcpy(auxiliary, positions, sizeof(positions));
    }
    gpu::render::CollectGuestWrites();
    gpu::render::BeginFrame(renderer);
    if (host)
      ASSERT_TRUE(
          gpu::render::GpuFill(geometry_base + gpu::render::kMirrorBlock,
                               gpu::render::kMirrorBlock, initial, initial));
    EXPECT_TRUE(gpu::render::CsHoldsGeometry(geometry_base, bytes));
    if (partial)
      EXPECT_EQ(gpu::render::MirrorRevision(geometry_base, bytes), 0u);
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 cpu_vertices = gpu::g_stats.ring_vb_bytes;
    const u64 cpu_indices = gpu::g_stats.ring_ib_bytes;
    const u64 copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    gpu::render::DrawInfo draw;
    draw.recomp = &geometry;
    draw.prim_type = 1;
    draw.vertex_count = vertices ? 1 : 3;
    draw.vertex_data = reinterpret_cast<const void*>(
        vertices ? geometry_base : reinterpret_cast<u64>(auxiliary));
    draw.vertex_stride = vertices ? 16 : 4;
    draw.num_vbufs = draw.num_vattrs = 1;
    draw.vbufs[0] = {draw.vertex_data, draw.vertex_stride,
                     vertices ? u32(bytes / 16) : 3u};
    draw.vattrs[0] = {0, 0, 0, 1, 4, 7};
    draw.index_data =
        vertices ? auxiliary : reinterpret_cast<void*>(geometry_base);
    draw.index_count = vertices ? 2 : bytes / element;
    draw.index_type = vertices || element == 4 ? 1 : 0;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[6 + trial].data());
    draw.rt_w = draw.rt_h = 16;
    draw.viewport_x_scale = draw.viewport_y_scale = 8;
    draw.viewport_x_offset = 8.5f;
    draw.viewport_y_offset = 4.5f;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u64 repeat_copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    if (trial == 6)
      EXPECT_EQ(gpu::render::cs::g_cs_stats.buf_bridge_bytes, repeat_copies);
    if (partial)
      EXPECT_GT(gpu::render::cs::g_cs_stats.buf_bridge_bytes, repeat_copies);
    EXPECT_GT(gpu::render::cs::g_cs_stats.buf_bridge_bytes, copies);
    const u32 updated = vertices ? 0xbe800000 : element == 2 ? 0x00020002 : 2;
    ASSERT_TRUE(gpu::render::GpuFill(geometry_base + gpu::render::kMirrorBlock,
                                     gpu::render::kMirrorBlock, updated,
                                     updated));
    ASSERT_TRUE(gpu::render::SubmitFrameChunk());
    const auto& slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    if (vertices)
      *reinterpret_cast<volatile float*>(geometry_base) = 0;
    else if (element == 2)
      std::fill_n(reinterpret_cast<u16*>(geometry_base),
                  gpu::render::kMirrorBlock / 2, 1);
    else
      std::fill_n(reinterpret_cast<u32*>(geometry_base),
                  gpu::render::kMirrorBlock / 4, 1);
    gpu::render::CollectGuestWrites();
    draw.viewport_y_offset = 12.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_vb_bytes, cpu_vertices);
    EXPECT_EQ(gpu::g_stats.ring_ib_bytes, cpu_indices);
    gpu::render::EndFrame(renderer, draw.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const u32 expected[] = {4 * 16 + 4, 4 * 16 + (vertices ? 12u : 8u),
                            12 * 16 + (vertices ? 6u : 8u),
                            12 * 16 + (vertices ? 8u : 12u)};
    for (u32 pixel : expected)
      EXPECT_EQ(slot.readback->mapped()[pixel * 4], 255);
    u32 pixels = 0;
    for (u32 i = 0; i < 256; ++i)
      pixels += slot.readback->mapped()[i * 4] != 0;
    EXPECT_EQ(pixels, 4u);
  }
  for (u32 mode = 0; mode < 2; ++mode) {
    SCOPED_TRACE(mode);
    const u64 raw_base = base + (u64(4 + mode) << 20);
    const u64 owned = raw_base + gpu::render::kMirrorBlock;
    const u32 bytes = 2 * gpu::render::kMirrorBlock;
    gpu::render::cs::g_cs_in_frame = mode != 0;
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0xff804020, 0xff804020));
    ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
    ASSERT_TRUE(gpu::render::SettleGuestWrites(owned, gpu::render::kMirrorBlock,
                                               "test"));
    *reinterpret_cast<volatile u32*>(raw_base) = 0xffff8040;
    gpu::render::CollectGuestWrites();
    gpu::render::BeginFrame(renderer);
    ASSERT_EQ(gpu::render::MirrorRevision(raw_base, bytes), 0u);
    ASSERT_TRUE(gpu::render::MirrorOwned(owned));
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 cpu_copies = gpu::g_stats.ring_raw_bytes;
    const u64 ring = gpu::render::g_ring.sbo_offset;
    const u64 bound = gpu::g_stats.bind_bytes;
    gpu::render::DrawInfo draw = imported;
    draw.recomp = &program;
    draw.ps_addr = reinterpret_cast<u64>(ps);
    draw.num_cbufs = 0;
    draw.cbufs[0] = {};
    draw.bufs[0] = {raw_base, bytes};
    draw.num_bufs = 1;
    draw.ps_user_data[0] = u32(raw_base);
    draw.ps_user_data[1] = u32(raw_base >> 32);
    draw.ps_user_data[2] = bytes;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[17 + mode].data());
    draw.viewport_x_offset = draw.viewport_y_offset = 4.5f;
    draw.ps_user_data[4] = 0;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    draw.viewport_x_offset = 12.5f;
    draw.ps_user_data[4] = gpu::render::kMirrorBlock;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0xff402010, 0xff402010));
    ASSERT_TRUE(gpu::render::SubmitFrameChunk());
    const auto& slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    *reinterpret_cast<volatile u32*>(raw_base) = 0xff201008;
    gpu::render::CollectGuestWrites();
    draw.viewport_y_offset = 12.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    draw.viewport_x_offset = 4.5f;
    draw.ps_user_data[4] = 0;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_raw_bytes, cpu_copies);
    EXPECT_EQ(gpu::render::g_ring.sbo_offset, ring);
    EXPECT_EQ(gpu::g_stats.bind_bytes, bound + 4 * bytes);
    gpu::render::EndFrame(renderer, draw.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const u32 locations[] = {4 * 16 + 4, 4 * 16 + 12, 12 * 16 + 4,
                             12 * 16 + 12};
    const u32 colors[] = {0xffff8040, 0xff804020, 0xff201008, 0xff402010};
    for (u32 i = 0; i < 4; ++i)
      for (u32 channel = 0; channel < 4; ++channel)
        EXPECT_EQ(slot.readback->mapped()[locations[i] * 4 + channel],
                  (colors[i] >> (channel * 8)) & 255);
  }
  for (u32 mode = 0; mode < 2; ++mode) {
    SCOPED_TRACE(mode);
    const u64 owned = base + (7ull << 20) + gpu::render::kMirrorBlock;
    const u64 constants_base = owned - gpu::render::kCbufWindow / 2;
    gpu::render::cs::g_cs_in_frame = mode != 0;
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0x3f400000, 0x3f400000));
    ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
    ASSERT_TRUE(gpu::render::SettleGuestWrites(owned, gpu::render::kMirrorBlock,
                                               "test"));
    *reinterpret_cast<volatile u32*>(constants_base) = 0x3e800000;
    gpu::render::CollectGuestWrites();
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::MirrorOwned(owned));
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 cpu_copies = gpu::g_stats.ring_cb_bytes;
    const u64 ring = gpu::render::g_ring.ubo_offset;
    const u64 bound = gpu::g_stats.bind_bytes;
    gpu::render::DrawInfo draw = imported;
    draw.recomp = &mixed_constants;
    draw.ps_addr = reinterpret_cast<u64>(mixed_ps);
    draw.ps_user_data[0] = u32(constants_base);
    draw.ps_user_data[1] = u32(constants_base >> 32);
    draw.cbufs[0] = {constants_base, gpu::render::kCbufWindow};
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[15 + mode].data());
    draw.viewport_x_offset = 4;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0x3f800000, 0x3f800000));
    ASSERT_TRUE(gpu::render::SubmitFrameChunk());
    const auto& slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    *reinterpret_cast<volatile u32*>(constants_base) = 0x3f000000;
    gpu::render::CollectGuestWrites();
    draw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, cpu_copies);
    EXPECT_EQ(gpu::render::g_ring.ubo_offset, ring);
    EXPECT_EQ(gpu::g_stats.bind_bytes, bound + 2 * gpu::render::kCbufWindow);
    gpu::render::EndFrame(renderer, draw.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    u32 pixels = 0;
    for (u32 i = 0; i < 256; ++i) {
      const auto* pixel = slot.readback->mapped() + i * 4;
      if (!pixel[3])
        continue;
      ++pixels;
      EXPECT_NEAR(pixel[0], i % 16 < 8 ? 64 : 128, 1);
      EXPECT_EQ(pixel[1], i % 16 < 8 ? 191 : 255);
      EXPECT_EQ(pixel[2], 0);
      EXPECT_EQ(pixel[3], 255);
    }
    EXPECT_EQ(pixels, 2u);
  }
  for (u32 mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    const u64 owned = base + (7ull << 20) + gpu::render::kMirrorBlock;
    const u64 constants_base = owned + 4;
    const u32 bytes = mode < 2 ? gpu::render::kCbufWindow : 4096;
    gpu::render::cs::g_cs_in_frame = (mode & 1) != 0;
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0x3e800000, 0x3e800000));
    ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
    gpu::render::BeginFrame(renderer);
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 cpu_copies = gpu::g_stats.ring_cb_bytes;
    const u64 copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    const u64 ring = gpu::render::g_ring.ubo_offset;
    const u64 bound = gpu::g_stats.bind_bytes;
    gpu::render::DrawInfo draw = imported;
    draw.ps_user_data[0] = u32(constants_base);
    draw.ps_user_data[1] = u32(constants_base >> 32);
    draw.cbufs[0] = {constants_base, bytes};
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[19 + mode].data());
    draw.viewport_x_offset = 4;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u64 repeat_copies = gpu::render::cs::g_cs_stats.buf_bridge_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_cs_stats.buf_bridge_bytes, repeat_copies);
    ASSERT_TRUE(gpu::render::GpuFill(owned, gpu::render::kMirrorBlock,
                                     0x3f400000, 0x3f400000));
    draw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, cpu_copies);
    EXPECT_EQ(gpu::render::g_ring.ubo_offset, ring);
    const bool direct = bytes == gpu::render::kCbufWindow &&
                        constants_base % gpu::render::g_ring.ubo_align == 0;
    EXPECT_EQ(gpu::render::cs::g_cs_stats.buf_bridge_bytes,
              copies + (direct ? 0 : 2 * bytes));
    EXPECT_EQ(gpu::g_stats.bind_bytes, bound + 3 * gpu::render::kCbufWindow);
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    u32 pixels = 0;
    for (u32 i = 0; i < 256; ++i) {
      const auto* pixel = slot.readback->mapped() + i * 4;
      if (!pixel[3])
        continue;
      ++pixels;
      EXPECT_EQ(pixel[0], i % 16 < 8 ? 64 : 191);
      EXPECT_EQ(pixel[1], 0);
      EXPECT_EQ(pixel[2], 0);
      EXPECT_EQ(pixel[3], 255);
    }
    EXPECT_EQ(pixels, 2u);
  }
  auto* native_vertices = reinterpret_cast<float*>(base + (6ull << 20));
  std::fill_n(native_vertices, 262144, .25f);
  gpu::render::CollectGuestWrites();
  alignas(65536) static u32 native_indices[16384] = {200001};
  gpu::render::BeginFrame(renderer);
  gpu::render::DrawInfo native;
  native.recomp = &geometry;
  native.prim_type = native.vertex_count = native.index_count = 1;
  native.vertex_data = native_vertices;
  native.vertex_stride = 4;
  native.num_vbufs = native.num_vattrs = 1;
  native.vbufs[0] = {native_vertices, 4, 262144};
  native.vattrs[0] = {0, 0, 0, 1, 4, 7};
  native.index_data = native_indices;
  native.index_type = 1;
  native.rt_base = native.mrt_base[0] =
      reinterpret_cast<u64>(targets[23].data());
  native.rt_w = native.rt_h = 16;
  native.viewport_x_scale = native.viewport_y_scale = 8;
  native.viewport_x_offset = 2.5f;
  native.viewport_y_offset = 8.5f;
  native.mrt_count = native.mrt_bound_mask = 1;
  native.mrt_info[0] = 10u << 2;
  const u64 native_scans = gpu::g_stats.index_scan_n;
  const u64 native_copies = gpu::g_stats.ring_vb_bytes;
  const u64 native_bound = gpu::g_stats.bind_bytes;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, native));
  ASSERT_TRUE(gpu::render::SubmitFrameChunk());
  const auto& native_slot =
      gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
  ASSERT_TRUE(gpu::render::Device().Wait(native_slot.submission));
  native_vertices[200001] = -.25f;
  gpu::render::CollectGuestWrites();
  native.viewport_x_offset = 12.5f;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, native));
  EXPECT_EQ(gpu::g_stats.index_scan_n, native_scans);
  EXPECT_EQ(gpu::g_stats.ring_vb_bytes, native_copies);
  EXPECT_EQ(gpu::g_stats.bind_bytes, native_bound + 2 * 262144 * sizeof(float));
  gpu::render::EndFrame(renderer, native.rt_base);
  ASSERT_TRUE(gpu::render::Device().Wait(native_slot.submission));
  ASSERT_TRUE(native_slot.presentable);
  for (u32 x : {4u, 10u}) {
    const auto* pixel = native_slot.readback->mapped() + (8 * 16 + x) * 4;
    EXPECT_EQ(pixel[0], 255);
    EXPECT_EQ(pixel[1], 0);
    EXPECT_EQ(pixel[2], 0);
    EXPECT_EQ(pixel[3], 255);
  }
  const u64 conversion_base = base + (7ull << 20);
  gpu::render::BeginFrame(renderer);
  const u64 conversion_offset = gpu::render::g_ring.ib_offset;
  const u64 conversion_copies = gpu::g_stats.index_convert_copy_bytes;
  const u64 conversion_resolves = gpu::render::cs::g_mirror_resolves.load();
  for (u32 trial = 0; trial < 2; ++trial) {
    const u32 pattern = trial ? 0x07060504 : 0x03020100;
    ASSERT_TRUE(gpu::render::GpuFill(conversion_base, gpu::render::kMirrorBlock,
                                     pattern, pattern));
    ASSERT_TRUE(gpu::render::ConvertGuestIndices(
        renderer, conversion_base, 4, 2, true, conversion_offset + trial * 64));
  }
  EXPECT_EQ(gpu::g_stats.index_convert_copy_bytes, conversion_copies);
  EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), conversion_resolves);
  gpu::rhi::BufferDesc conversion_desc;
  conversion_desc.size = 128;
  conversion_desc.usage = gpu::rhi::kBufferCopyDst;
  conversion_desc.memory = gpu::rhi::MemoryKind::kReadback;
  auto* conversion_readback =
      gpu::render::Device().CreateBuffer(conversion_desc);
  ASSERT_NE(conversion_readback, nullptr);
  auto* conversion_list = gpu::render::g_frame.list;
  conversion_list->Barrier(gpu::rhi::kAccessComputeWrite,
                           gpu::rhi::kAccessCopyRead);
  conversion_list->CopyBuffer(conversion_readback, 0,
                              gpu::render::g_ring.converted_ib,
                              conversion_offset, conversion_desc.size);
  conversion_list->Barrier(gpu::rhi::kAccessCopyWrite,
                           gpu::rhi::kAccessHostRead);
  const auto& conversion_slot =
      gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
  gpu::render::EndFrame(renderer, 0);
  ASSERT_TRUE(gpu::render::Device().Wait(conversion_slot.submission));
  constexpr u32 corners[] = {0, 1, 2, 0, 2, 3};
  for (u32 trial = 0; trial < 2; ++trial) {
    const auto* result = reinterpret_cast<const u32*>(
        conversion_readback->mapped() + trial * 64);
    for (u32 i = 0; i < base::ArraySize(corners); ++i)
      EXPECT_EQ(result[i], corners[i] + trial * 4);
  }
  gpu::render::Device().Destroy(conversion_readback);
  gpu::render::Device().WaitIdle();
  const u32 literal_ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                                0xbf810000};
  const auto indirect_program =
      gpu::rdna::Recompile(vs, literal_ps, user_data, user_data);
  ASSERT_TRUE(indirect_program.ok);
  ASSERT_TRUE(indirect_program.vs_bufs.empty());
  ASSERT_TRUE(indirect_program.ps_bufs.empty());
  ASSERT_FALSE(indirect_program.direct_reads);
  auto* draw_arguments = reinterpret_cast<u32*>(base + (7ull << 20) +
                                                2 * gpu::render::kMirrorBlock);
  draw_arguments[0] = 0;
  draw_arguments[1] = 1;
  draw_arguments[2] = draw_arguments[3] = 0;
  gpu::render::CollectGuestWrites();
  gpu::render::cs::g_cs_in_frame = false;
  gpu::render::BeginFrame(renderer);
  const u64 argument_address = reinterpret_cast<u64>(draw_arguments);
  ASSERT_TRUE(gpu::render::GpuFill(argument_address, 4, 1, 1));
  gpu::render::DrawInfo indirect_draw = imported;
  indirect_draw.recomp = &indirect_program;
  indirect_draw.cbufs[0] = {};
  indirect_draw.num_cbufs = 0;
  indirect_draw.indirect_args = argument_address;
  indirect_draw.rt_base = indirect_draw.mrt_base[0] =
      reinterpret_cast<u64>(targets[24].data());
  indirect_draw.viewport_x_offset = 4.5f;
  indirect_draw.viewport_y_offset = 8.5f;
  const u64 argument_resolves = gpu::render::cs::g_mirror_resolves.load();
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, indirect_draw));
  const u64 argument_chunk = gpu::render::g_frame.chunk_seq;
  ASSERT_TRUE(gpu::render::GpuFill(argument_address, 4, 0, 0));
  EXPECT_GT(gpu::render::g_frame.chunk_seq, argument_chunk);
  indirect_draw.viewport_x_offset = 12.5f;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, indirect_draw));
  EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), argument_resolves);
  const auto& argument_slot =
      gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
  gpu::render::EndFrame(renderer, indirect_draw.rt_base);
  ASSERT_TRUE(gpu::render::Device().Wait(argument_slot.submission));
  ASSERT_TRUE(argument_slot.presentable);
  const auto* argument_pixels = argument_slot.readback->mapped();
  EXPECT_EQ(argument_pixels[(8 * 16 + 4) * 4], 255);
  EXPECT_EQ(argument_pixels[(8 * 16 + 12) * 4], 0);

  const auto direct_program =
      gpu::rdna::Recompile(vs, cbuf_ps, user_data, user_data, 0, false, 8, 8);
  ASSERT_TRUE(direct_program.ok);
  ASSERT_TRUE(direct_program.direct);
  ASSERT_TRUE(direct_program.direct_reads);
  ASSERT_TRUE(direct_program.ps_cbufs.empty());
  const u64 direct_base = base + (7ull << 20) + 3 * gpu::render::kMirrorBlock;
  gpu::render::DrawInfo direct_draw = imported;
  direct_draw.recomp = &direct_program;
  direct_draw.cbufs[0] = {};
  direct_draw.num_cbufs = 0;
  direct_draw.ps_user_data[0] = u32(direct_base);
  direct_draw.ps_user_data[1] = u32(direct_base >> 32);
  direct_draw.rt_base = direct_draw.mrt_base[0] =
      reinterpret_cast<u64>(targets[25].data());
  direct_draw.viewport_x_offset = 4.5f;
  direct_draw.viewport_y_offset = 8.5f;
  gpu::render::BeginFrame(renderer);
  gpu::render::RecordComputeInFrame();
  ASSERT_TRUE(gpu::render::GpuFill(direct_base, gpu::render::kMirrorBlock,
                                   0x3e800000, 0x3e800000));
  const u64 direct_resolves = gpu::render::cs::g_mirror_resolves.load();
  const u64 direct_copies = gpu::g_stats.ring_cb_bytes;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
  ASSERT_TRUE(gpu::render::cs::CsBatchSubmit());
  const auto& direct_slot =
      gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
  ASSERT_TRUE(gpu::render::Device().Wait(direct_slot.submission));
  *reinterpret_cast<volatile u32*>(direct_base) = 0x3f400000;
  gpu::render::CollectGuestWrites();
  const u64 direct_cpu_resolves = gpu::render::cs::g_mirror_resolves.load();
  EXPECT_EQ(direct_cpu_resolves, direct_resolves + 1);
  direct_draw.viewport_x_offset = 12.5f;
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
  EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), direct_cpu_resolves);
  EXPECT_EQ(gpu::g_stats.ring_cb_bytes, direct_copies);
  gpu::render::EndFrame(renderer, direct_draw.rt_base);
  ASSERT_TRUE(gpu::render::Device().Wait(direct_slot.submission));
  ASSERT_TRUE(direct_slot.presentable);
  const auto* direct_pixels = direct_slot.readback->mapped();
  EXPECT_EQ(direct_pixels[(8 * 16 + 4) * 4], 64);
  EXPECT_EQ(direct_pixels[(8 * 16 + 12) * 4], 191);

  *reinterpret_cast<volatile u32*>(base + DirectMemory::kMappedBytes - 4) =
      0x3f400000;
  gpu::render::CollectGuestWrites();
  for (u64 alias : {DirectMemory::kHighBase, DirectMemory::kArenaBase,
                    DirectMemory::kModuleBase, DirectMemory::kEndBase}) {
    SCOPED_TRACE(alias);
    const u64 high_address = alias == DirectMemory::kEndBase
                                 ? 0x10000000000ull - 4
                                 : alias + direct_base - base;
    EXPECT_FALSE(gpu::render::MirrorOwned(high_address));
    direct_draw.ps_user_data[0] = u32(high_address);
    direct_draw.ps_user_data[1] = u32(high_address >> 32);
    for (u32 trial = 0; trial < 2; ++trial) {
      SCOPED_TRACE(trial);
      gpu::render::BeginFrame(renderer);
      ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
      const u32 high_slot_index = gpu::render::g_frame.slot_idx;
      gpu::render::EndFrame(renderer, direct_draw.rt_base);
      const auto& high_slot = gpu::render::g_frame.slots[high_slot_index];
      ASSERT_TRUE(gpu::render::Device().Wait(high_slot.submission));
      ASSERT_TRUE(high_slot.presentable);
      const auto* high_pixels = high_slot.readback->mapped();
      EXPECT_EQ(high_pixels[(8 * 16 + 12) * 4], trial ? 191 : 0);
      gpu::render::SyncGuestDirect();
    }
  }
  const u32 direct_vs[4096] = {0xf4000100, 0xfa000000, 0x7e000204, 0x7e020280,
                               0x7e0402f2, 0xf80008cf, 0x02010100, 0xbf810000};
  const auto direct_vertices = gpu::rdna::Recompile(
      direct_vs, literal_ps, user_data, user_data, 0, false, 8, 8);
  ASSERT_TRUE(direct_vertices.ok);
  ASSERT_TRUE(direct_vertices.direct_vs);
  ASSERT_TRUE(direct_vertices.direct_reads);
  ASSERT_TRUE(direct_vertices.vs_cbufs.empty());
  const u32 descriptor_ps[4096] = {
      0xf4080100, 0xfa000000, 0x7e000280, 0xe0301000, 0x80010000, 0x7e020280,
      0x7e040280, 0x7e0602f2, 0xf800080f, 0x03020100, 0xbf810000};
  const auto direct_descriptors = gpu::rdna::Recompile(
      vs, descriptor_ps, user_data, user_data, 0, false, 8, 8);
  ASSERT_TRUE(direct_descriptors.ok);
  ASSERT_TRUE(direct_descriptors.direct_reads);
  ASSERT_TRUE(direct_descriptors.ps_cbufs.empty());
  ASSERT_TRUE(direct_descriptors.ps_bufs.empty());
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto descriptor_compute =
      gpu::gcn::RecompileCompute(cs, 1, 1, 1, 5, 0, 0);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(descriptor_compute.ok);
  const u64 descriptor_base = direct_base + gpu::render::kMirrorBlock;
  ci.res[0].size = ci.res[0].guest_size = gpu::render::kMirrorBlock;
  ci.user_data[2] = gpu::render::kMirrorBlock / 4;
  for (u32 mode = 0; mode < 8; ++mode) {
    SCOPED_TRACE(mode);
    const bool vertex = mode == 4 || mode == 5;
    const bool descriptor = mode >= 6;
    gpu::render::cs::g_cs_in_frame = (mode & 1) != 0;
    gpu::render::BeginFrame(renderer);
    direct_draw.recomp = descriptor ? &direct_descriptors
                         : vertex   ? &direct_vertices
                                    : &direct_program;
    const u64 source = descriptor ? descriptor_base : direct_base;
    ci.recomp = descriptor ? &descriptor_compute : &compute;
    ci.res[0].base = source;
    ci.user_data[0] = u32(source);
    ci.user_data[1] = u32(source >> 32) | (4u << 16);
    std::memcpy(direct_draw.vs_user_data, ci.user_data, sizeof(ci.user_data));
    direct_draw.ps_user_data[0] = u32(source);
    direct_draw.ps_user_data[1] = u32(source >> 32);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[26 + mode].data());
    direct_draw.viewport_x_offset = 4.5f;
    direct_draw.viewport_x_scale = vertex ? 4 : 8;
    if (descriptor) {
      ASSERT_TRUE(gpu::render::GpuFill(direct_base, 4, 0x3e800000, 0x3e800000));
      ASSERT_TRUE(
          gpu::render::GpuFill(direct_base + 4, 4, 0x3f400000, 0x3f400000));
      ASSERT_TRUE(gpu::render::GpuFill(descriptor_base, 8, u32(direct_base),
                                       u32(direct_base >> 32) | (4u << 16)));
      ASSERT_TRUE(gpu::render::GpuFill(descriptor_base + 8, 8, 64, 0));
    }
    if (mode < 2) {
      ASSERT_TRUE(gpu::render::GpuFill(direct_base, gpu::render::kMirrorBlock,
                                       0x3e800000, 0x3e800000));
    } else {
      ci.user_data[4] = descriptor ? u32(direct_base) : 0x3e800000;
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    }
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 copies = gpu::g_stats.ring_cb_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    if (mode < 2) {
      ASSERT_TRUE(gpu::render::GpuFill(direct_base, gpu::render::kMirrorBlock,
                                       0x3f400000, 0x3f400000));
    } else {
      ci.user_data[4] = descriptor ? u32(direct_base + 4) : 0x3f400000;
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    }
    direct_draw.viewport_x_offset = 12.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, copies);
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    const auto* pixels = slot.readback->mapped();
    u32 pixel_count = 0;
    for (u32 i = 0; i < 256; ++i) {
      if (!pixels[i * 4 + 3])
        continue;
      ++pixel_count;
      SCOPED_TRACE(i);
      EXPECT_EQ(i / 16, 8u);
      EXPECT_TRUE(i % 16 == (vertex ? 5u : 4u) ||
                  i % 16 == (vertex ? 15u : 12u));
    }
    EXPECT_EQ(pixel_count, 2u);
    EXPECT_EQ(pixels[(8 * 16 + (vertex ? 5 : 4)) * 4], vertex ? 255 : 64);
    EXPECT_EQ(pixels[(8 * 16 + (vertex ? 15 : 12)) * 4], vertex ? 255 : 191);
  }
  const u32 gcn_scalar_ps[4096] = {0xc0020100, 0x7e000204, 0x7e020280,
                                   0x7e040280, 0x7e0602f2, 0xf800080f,
                                   0x03020100, 0xbf810000};
  const u32 gcn_offset_ps[4096] = {0xc0020008, 0x7e000204, 0x7e020280,
                                   0x7e040280, 0x7e0602f2, 0xf800080f,
                                   0x03020100, 0xbf810000};
  const u32 gcn_descriptor_ps[4096] = {
      0xc0820100, 0x7e000280, 0xe0301000, 0x80010000, 0x7e020280,
      0x7e040280, 0x7e0602f2, 0xf800080f, 0x03020100, 0xbf810000};
  const u32 gcn_typed_ps[4096] = {0x7e000280, 0xeba01000, 0x80000000,
                                  0x7e020280, 0x7e040280, 0x7e0602f2,
                                  0xf800080f, 0x03020100, 0xbf810000};
  const u32 gcn_typed_vs[4096] = {0xeba02000, 0x80000000, 0x7e020280,
                                  0x7e0402f2, 0xf80008cf, 0x02010100,
                                  0xbf810000};
  const u32* gcn_pixels[] = {gcn_scalar_ps, gcn_offset_ps, gcn_descriptor_ps,
                             gcn_typed_ps, literal_ps};
  gpu::gcn::Recompiled gcn_programs[base::ArraySize(gcn_pixels)];
  for (u32 shader = 0; shader < base::ArraySize(gcn_pixels); ++shader) {
    SCOPED_TRACE(shader);
    const bool vertex = shader == 4;
    const bool descriptor = shader == 2;
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    auto& gcn_program = gcn_programs[shader];
    gcn_program = gpu::gcn::Recompile(vertex ? gcn_typed_vs : vs,
                                      gcn_pixels[shader], user_data, user_data);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(gcn_program.ok);
    ASSERT_TRUE(gcn_program.direct);
    ASSERT_TRUE(gcn_program.direct_reads);
    ASSERT_TRUE(gcn_program.vs_cbufs.empty());
    ASSERT_TRUE(gcn_program.ps_cbufs.empty());
    ASSERT_TRUE(gcn_program.vs_bufs.empty());
    ASSERT_TRUE(gcn_program.ps_bufs.empty());
    ASSERT_TRUE(gcn_program.attrs.empty());
    for (u32 mode = 0; mode < 2; ++mode) {
      SCOPED_TRACE(mode);
      gpu::render::cs::g_cs_in_frame = mode != 0;
      gpu::render::BeginFrame(renderer);
      const u64 source = descriptor ? descriptor_base : direct_base;
      ci.recomp = &descriptor_compute;
      ci.res[0].base = source;
      ci.user_data[0] = u32(source);
      ci.user_data[1] = u32(source >> 32) | (4u << 16);
      direct_draw.recomp = &gcn_program;
      std::memcpy(direct_draw.vs_user_data, ci.user_data, sizeof(ci.user_data));
      std::memcpy(direct_draw.ps_user_data, ci.user_data, sizeof(ci.user_data));
      direct_draw.ps_user_data[8] = 0;
      direct_draw.rt_base = direct_draw.mrt_base[0] =
          reinterpret_cast<u64>(targets[34 + shader * 2 + mode].data());
      direct_draw.viewport_x_offset = 4.5f;
      direct_draw.viewport_x_scale = vertex ? 4 : 8;
      ASSERT_TRUE(
          gpu::render::GpuFill(direct_base, gpu::render::kMirrorBlock, 0, 0));
      if (descriptor) {
        ASSERT_TRUE(
            gpu::render::GpuFill(direct_base, 4, 0x3e800000, 0x3e800000));
        ASSERT_TRUE(
            gpu::render::GpuFill(direct_base + 4, 4, 0x3f400000, 0x3f400000));
        ASSERT_TRUE(gpu::render::GpuFill(descriptor_base, 8, u32(direct_base),
                                         u32(direct_base >> 32) | (4u << 16)));
        ASSERT_TRUE(gpu::render::GpuFill(descriptor_base + 8, 8, 64, 0));
      }
      ci.user_data[4] = descriptor ? u32(direct_base) : 0x3e800000;
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
      const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
      const u64 copies = gpu::g_stats.ring_cb_bytes;
      ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
      ci.user_data[4] = descriptor ? u32(direct_base + 4) : 0x3f400000;
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
      direct_draw.viewport_x_offset = 12.5f;
      ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
      EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
      EXPECT_EQ(gpu::g_stats.ring_cb_bytes, copies);
      const u32 slot_index = gpu::render::g_frame.slot_idx;
      gpu::render::EndFrame(renderer, direct_draw.rt_base);
      const auto& slot = gpu::render::g_frame.slots[slot_index];
      ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
      ASSERT_TRUE(slot.presentable);
      const auto* pixels = slot.readback->mapped();
      u32 pixel_count = 0;
      for (u32 i = 0; i < 256; ++i)
        pixel_count += pixels[i * 4 + 3] != 0;
      EXPECT_EQ(pixel_count, 2u);
      EXPECT_EQ(pixels[(8 * 16 + (vertex ? 5 : 4)) * 4], vertex ? 255 : 64);
      EXPECT_EQ(pixels[(8 * 16 + (vertex ? 15 : 12)) * 4], vertex ? 255 : 191);
    }
  }
  gpu::rhi::BufferDesc miss_desc;
  miss_desc.size = 8 + gpu::gcn::kDirectMissSlots * sizeof(u64);
  miss_desc.usage = gpu::rhi::kBufferStorage | gpu::rhi::kBufferAddress;
  miss_desc.memory = gpu::rhi::MemoryKind::kReadback;
  auto* miss_buffer = gpu::render::Device().CreateBuffer(miss_desc);
  ASSERT_NE(miss_buffer, nullptr);
  ASSERT_NE(miss_buffer->mapped(), nullptr);
  std::memset(miss_buffer->mapped(), 0, miss_desc.size);
  const u64 previous_misses = gpu::gcn::g_direct_misses;
  gpu::gcn::g_direct_misses = miss_buffer->address();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto missing_program =
      gpu::gcn::Recompile(vs, gcn_scalar_ps, user_data, user_data);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  gpu::gcn::g_direct_misses = previous_misses;
  ASSERT_TRUE(missing_program.ok);
  const u64 missing_base = base + 2 * DirectMemory::kMappedBytes;
  direct_draw.recomp = &missing_program;
  direct_draw.ps_user_data[0] = u32(missing_base);
  direct_draw.ps_user_data[1] = u32(missing_base >> 32);
  direct_draw.instance_count = 8192;
  u32 full_count = 0;
  for (u32 trial = 0; trial < 2; ++trial) {
    SCOPED_TRACE(trial);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[44 + trial].data());
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    const u32 count = *reinterpret_cast<const u32*>(miss_buffer->mapped());
    EXPECT_GE(count, gpu::gcn::kDirectMissSlots);
    if (trial)
      EXPECT_EQ(count, full_count);
    full_count = count;
    const auto* addresses =
        reinterpret_cast<const u64*>(miss_buffer->mapped() + 8);
    for (u32 i = 0; i < gpu::gcn::kDirectMissSlots; ++i)
      EXPECT_EQ(addresses[i], missing_base);
    const auto* pixel = slot.readback->mapped() + (8 * 16 + 12) * 4;
    EXPECT_EQ(pixel[0], 0);
    EXPECT_EQ(pixel[3], 255);
  }
  std::memset(miss_buffer->mapped(), 0, miss_desc.size);
  direct_draw.ps_user_data[0] = direct_draw.ps_user_data[1] = 0;
  direct_draw.rt_base = direct_draw.mrt_base[0] =
      reinterpret_cast<u64>(targets[46].data());
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
  const u32 null_slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, direct_draw.rt_base);
  const auto& null_slot = gpu::render::g_frame.slots[null_slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(null_slot.submission));
  EXPECT_EQ(*reinterpret_cast<const u32*>(miss_buffer->mapped()), 0u);
  const auto* null_pixel = null_slot.readback->mapped() + (8 * 16 + 12) * 4;
  EXPECT_EQ(null_pixel[0], 0);
  EXPECT_EQ(null_pixel[3], 255);
  direct_draw.ps_user_data[0] = u32(missing_base);
  direct_draw.ps_user_data[1] = u32(missing_base >> 32);
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
  const u32 reset_slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, direct_draw.rt_base);
  const auto& reset_slot = gpu::render::g_frame.slots[reset_slot_index];
  ASSERT_TRUE(gpu::render::Device().Wait(reset_slot.submission));
  EXPECT_GE(*reinterpret_cast<const u32*>(miss_buffer->mapped()),
            gpu::gcn::kDirectMissSlots);

  HostMemory host;
  ASSERT_NE(host.memory, MAP_FAILED);
  ASSERT_TRUE(host.Commit(0, gpu::render::kMirrorBlock));
  const u64 partial = HostMemory::kBase + gpu::render::kMirrorBlock;
  const u64 readonly = partial + gpu::render::kMirrorBlock;
  ASSERT_TRUE(host.Commit(gpu::render::kMirrorBlock + 4096, 4096));
  ASSERT_TRUE(host.Commit(gpu::render::kMirrorBlock + 32768, 4096));
  ASSERT_TRUE(
      host.Commit(2 * gpu::render::kMirrorBlock, gpu::render::kMirrorBlock));
  for (const u64 address :
       {HostMemory::kBase, partial + 4096, partial + 32768, readonly})
    *reinterpret_cast<float*>(address) = .25f;
  ASSERT_EQ(mprotect(reinterpret_cast<void*>(partial + 32768), 4096, PROT_READ),
            0);
  ASSERT_EQ(mprotect(reinterpret_cast<void*>(readonly),
                     gpu::render::kMirrorBlock, PROT_READ),
            0);
  gpu::render::NoteGuestRemap(partial + 32768, 4096);
  gpu::render::NoteGuestRemap(readonly, gpu::render::kMirrorBlock);
  gpu::render::SyncGuestDirect();
  direct_draw.instance_count = 1;
  direct_draw.recomp = &gcn_programs[0];
  direct_draw.ps_user_data[0] = u32(HostMemory::kBase);
  direct_draw.ps_user_data[1] = u32(HostMemory::kBase >> 32);
  direct_draw.rt_base = direct_draw.mrt_base[0] =
      reinterpret_cast<u64>(targets[47].data());
  gpu::render::BeginFrame(renderer);
  ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
  const u32 warm_slot_index = gpu::render::g_frame.slot_idx;
  gpu::render::EndFrame(renderer, direct_draw.rt_base);
  ASSERT_TRUE(gpu::render::Device().Wait(
      gpu::render::g_frame.slots[warm_slot_index].submission));
  gpu::render::SyncGuestDirect();
  const u64 host_addresses[] = {HostMemory::kBase, partial + 4096,
                                partial + 32768, readonly, partial + 16384};
  for (u32 trial = 0; trial < base::ArraySize(host_addresses); ++trial) {
    SCOPED_TRACE(trial);
    const u64 address = host_addresses[trial];
    direct_draw.recomp = trial < 2 ? &gcn_programs[0] : &direct_program;
    direct_draw.ps_user_data[0] = u32(address);
    direct_draw.ps_user_data[1] = u32(address >> 32);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[48 + trial].data());
    direct_draw.viewport_x_offset = 4.5f;
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 copies = gpu::g_stats.ring_cb_bytes;
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    ASSERT_TRUE(gpu::render::SubmitFrameChunk());
    const auto& slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    if (trial < 2) {
      *reinterpret_cast<volatile float*>(address) = .75f;
      gpu::render::CollectGuestWrites();
    }
    direct_draw.viewport_x_offset = 12.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, copies);
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    const auto* pixels = slot.readback->mapped();
    EXPECT_EQ(pixels[(8 * 16 + 4) * 4], trial == 4 ? 0 : 64);
    EXPECT_EQ(pixels[(8 * 16 + 12) * 4], trial < 2 ? 191 : trial == 4 ? 0 : 64);
    EXPECT_EQ(pixels[(8 * 16 + 4) * 4 + 3], 255);
    EXPECT_EQ(pixels[(8 * 16 + 12) * 4 + 3], 255);
  }
  const u32 tail_pixels[][4096] = {
      {0x7e000280, 0xe0301000, 0x80000000, 0x7e020280, 0x7e040280, 0x7e0602f2,
       0xf800080f, 0x03020100, 0xbf810000},
      {0x7e000280, 0xeba01000, 0x80000000, 0x7e020280, 0x7e040280, 0x7e0602f2,
       0xf800080f, 0x03020100, 0xbf810000},
      {0x7e000280, 0xe00c1000, 0x80000000, 0x7e020280, 0x7e040280, 0x7e0602f2,
       0xf800080f, 0x03020100, 0xbf810000},
      {0x7e000280, 0xeba31000, 0x80000000, 0xf800080f, 0x03020100, 0xbf810000}};
  gpu::gcn::Recompiled tail_programs[base::ArraySize(tail_pixels)];
  const u64 tail = partial + 8192 - 4;
  for (u32 trial = 0; trial < base::ArraySize(tail_pixels); ++trial) {
    SCOPED_TRACE(trial);
    *reinterpret_cast<u32*>(tail) = trial == 2 ? 0x00000040 : 0x3e800000;
    gpu::render::CollectGuestWrites();
    gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
    auto& tail_program = tail_programs[trial];
    tail_program =
        gpu::gcn::Recompile(vs, tail_pixels[trial], user_data, user_data);
    gpu::gcn::SetDefaultIsaMode(previous_isa);
    ASSERT_TRUE(tail_program.ok);
    ASSERT_TRUE(tail_program.direct_reads);
    direct_draw.recomp = &tail_program;
    direct_draw.ps_user_data[0] = u32(tail);
    direct_draw.ps_user_data[1] = u32(tail >> 32);
    direct_draw.ps_user_data[2] = 4;
    direct_draw.ps_user_data[3] = 10u << 15;
    direct_draw.viewport_x_offset = 4.5f;
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[64 + trial].data());
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    const auto& slot =
        gpu::render::g_frame.slots[gpu::render::g_frame.slot_idx];
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 4) * 4], 64);
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 4) * 4 + 3], 255);
  }
  direct_draw.viewport_x_offset = 12.5f;
  for (u32 trial = 0; trial < 3; ++trial) {
    SCOPED_TRACE(trial);
    const u64 address = trial ? readonly : HostMemory::kBase;
    if (!trial) {
      ASSERT_TRUE(host.Commit(0, gpu::render::kMirrorBlock));
    } else {
      ASSERT_EQ(mprotect(reinterpret_cast<void*>(readonly),
                         gpu::render::kMirrorBlock, PROT_READ | PROT_WRITE),
                0);
      gpu::render::NoteGuestRemap(readonly, gpu::render::kMirrorBlock);
    }
    *reinterpret_cast<float*>(address) = trial == 2 ? .5f : .75f;
    if (trial == 1) {
      ASSERT_EQ(mprotect(reinterpret_cast<void*>(readonly),
                         gpu::render::kMirrorBlock, PROT_READ),
                0);
      gpu::render::NoteGuestRemap(readonly, gpu::render::kMirrorBlock);
    }
    gpu::render::SyncGuestDirect();
    direct_draw.recomp = &gcn_programs[0];
    direct_draw.ps_user_data[0] = u32(address);
    direct_draw.ps_user_data[1] = u32(address >> 32);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[53 + trial].data());
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    const auto* pixel = slot.readback->mapped() + (8 * 16 + 12) * 4;
    EXPECT_NEAR(pixel[0], trial == 2 ? 128 : 191, 1);
    EXPECT_EQ(pixel[3], 255);
  }

  for (u32 mode = 0; mode < 2; ++mode) {
    SCOPED_TRACE(mode);
    gpu::render::cs::g_cs_in_frame = mode != 0;
    direct_draw.recomp = &gcn_programs[0];
    direct_draw.ps_user_data[0] = u32(HostMemory::kBase);
    direct_draw.ps_user_data[1] = u32(HostMemory::kBase >> 32);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[56 + mode].data());
    ci.recomp = &descriptor_compute;
    ci.res[0].base = HostMemory::kBase;
    ci.user_data[0] = u32(HostMemory::kBase);
    ci.user_data[1] = u32(HostMemory::kBase >> 32) | (4u << 16);
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 copies = gpu::g_stats.ring_cb_bytes;
    gpu::render::BeginFrame(renderer);
    ci.user_data[4] = 0x3e800000;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    direct_draw.viewport_x_offset = 4.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    ci.user_data[4] = 0x3f400000;
    ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    direct_draw.viewport_x_offset = 12.5f;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    EXPECT_EQ(gpu::g_stats.ring_cb_bytes, copies);
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 4) * 4], 64);
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 12) * 4], 191);
    EXPECT_EQ(*reinterpret_cast<volatile const float*>(HostMemory::kBase),
              .75f);
    *reinterpret_cast<volatile float*>(HostMemory::kBase) = .5f;
    gpu::render::CollectGuestWrites();
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[58 + mode].data());
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    const u32 cpu_slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& cpu_slot = gpu::render::g_frame.slots[cpu_slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(cpu_slot.submission));
    EXPECT_NEAR(cpu_slot.readback->mapped()[(8 * 16 + 12) * 4], 128, 1);
  }

  ASSERT_TRUE(host.Commit(gpu::render::kMirrorBlock, 4096));
  *reinterpret_cast<float*>(partial) = .25f;
  gpu::render::SyncGuestDirect();
  const u64 extended_addresses[] = {partial, HostMemory::kBase,
                                    HostMemory::kBase, HostMemory::kBase};
  for (u32 trial = 0; trial < base::ArraySize(extended_addresses); ++trial) {
    SCOPED_TRACE(trial);
    gpu::render::SyncGuestDirect();
    const u64 address = extended_addresses[trial];
    const float value = trial == 2 ? .75f : .25f;
    *reinterpret_cast<volatile float*>(address) = value;
    gpu::render::CollectGuestWrites();
    direct_draw.ps_user_data[0] = u32(address);
    direct_draw.ps_user_data[1] = u32(address >> 32);
    direct_draw.rt_base = direct_draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[60 + trial].data());
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, direct_draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, direct_draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 12) * 4],
              trial == 2 ? 191 : 64);
  }
  gpu::render::Device().Destroy(miss_buffer);
}

}  // namespace
