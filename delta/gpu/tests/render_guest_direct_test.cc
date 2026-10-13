#include <gtest/gtest.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>

#include "base/containers/array.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/render/compute_internal.h"
#include "gpu/render/device.h"
#include "gpu/render/draw_recomp.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_direct.h"
#include "gpu/render/renderer.h"
#include "gpu/write_tracker.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

extern "C" void prosperity_gpu_end_of_pipe() {}
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}

namespace {
struct GuestMemory {
  static constexpr u64 kBytes = gpu::render::kMirrorBlock;
  u64 address, bytes;
  int fd = -1;
  void* memory = MAP_FAILED;

  GuestMemory(u64 base, bool direct, u64 size = kBytes)
      : address(base), bytes(size) {
    if (direct) {
      fd = memfd_create("delta_dmem_late", 0);
      if (fd < 0 || ftruncate(fd, bytes))
        return;
    }
    memory =
        mmap(reinterpret_cast<void*>(address), bytes, PROT_READ | PROT_WRITE,
             MAP_FIXED_NOREPLACE |
                 (direct ? MAP_SHARED : MAP_PRIVATE | MAP_ANONYMOUS),
             fd, 0);
    if (memory != MAP_FAILED) {
      host_memory::TrackMemoryMapping(memory, bytes);
      gpu::render::NoteGuestRemap(address, bytes);
    }
  }

  ~GuestMemory() {
    gpu::render::Device().WaitIdle();
    if (memory != MAP_FAILED) {
      if (gpu::render::MirrorOwned(address)) {
        const u32 value = *static_cast<volatile const u32*>(memory);
        (void)value;
      }
      gpu::GuestWriteTracker().Release(address, bytes);
      munmap(memory, bytes);
      host_memory::ForgetMemoryMapping(memory, bytes);
      gpu::render::NoteGuestRemap(address, bytes);
    }
    if (fd >= 0)
      close(fd);
  }
};

TEST(VkGuestDirect, AttachesLatePoolWithoutChangingShaderTable) {
  options::Init();
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
  GuestMemory host(0x2200000000ull, false);
  ASSERT_NE(host.memory, MAP_FAILED);
  *static_cast<float*>(host.memory) = .25f;
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                        0xbf810000};
  const u32 ps[4096] = {0xc0020100, 0x7e000204, 0x7e020280, 0x7e040280,
                        0x7e0602f2, 0xf800080f, 0x03020100, 0xbf810000};
  const u32 user_data[32]{};
  const u32 rdna_ps[4096] = {0xf4000100, 0xfa000000, 0x7e000204,
                             0x7e020280, 0x7e040280, 0x7e0602f2,
                             0xf800080f, 0x03020100, 0xbf810000};
  gpu::ps5::GraphicsShaderState shader;
  shader.vs_addr = reinterpret_cast<u64>(vs);
  shader.ps_addr = reinterpret_cast<u64>(rdna_ps);
  shader.vs_user_data = shader.ps_user_data = user_data;
  shader.vs_user_sgprs = shader.ps_user_sgprs = 8;
  const auto& staged_shader = gpu::ps5::GetGraphicsShader(shader);
  ASSERT_TRUE(staged_shader.ok);
  ASSERT_FALSE(staged_shader.direct);
  ASSERT_EQ(staged_shader.ps_cbufs.size(), 1u);
  ASSERT_TRUE(gpu::render::InitGuestDirect());
  const u64 table = gpu::gcn::g_direct_table;
  ASSERT_NE(table, 0u);
  const auto& direct_shader = gpu::ps5::GetGraphicsShader(shader);
  ASSERT_NE(&staged_shader, &direct_shader);
  ASSERT_TRUE(direct_shader.direct_reads);
  ASSERT_TRUE(direct_shader.ps_cbufs.empty());
  EXPECT_EQ(&direct_shader, &gpu::ps5::GetGraphicsShader(shader));
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto program = gpu::gcn::Recompile(vs, ps, user_data, user_data);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(program.ok);
  ASSERT_TRUE(program.direct_reads);
  ASSERT_TRUE(program.ps_cbufs.empty());
  alignas(65536) static base::Array<u8, 65536> targets[32]{};
  gpu::render::DrawInfo draw;
  draw.recomp = &program;
  draw.vs_addr = reinterpret_cast<u64>(vs);
  draw.ps_addr = reinterpret_cast<u64>(ps);
  draw.prim_type = draw.vertex_count = 1;
  draw.rt_w = draw.rt_h = 16;
  draw.viewport_x_scale = draw.viewport_y_scale = 8;
  draw.viewport_x_offset = 4.5f;
  draw.viewport_y_offset = 8.5f;
  draw.mrt_count = draw.mrt_bound_mask = 1;
  draw.mrt_info[0] = 10u << 2;
  u32 frame = 0;
  const auto read = [&](u64 address, u32 expected) {
    draw.ps_user_data[0] = u32(address);
    draw.ps_user_data[1] = u32(address >> 32);
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[frame++].data());
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    EXPECT_NEAR(slot.readback->mapped()[(8 * 16 + 4) * 4], expected, 1);
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 4) * 4 + 3], 255);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    gpu::render::SyncGuestDirect();
  };
  ASSERT_NO_FATAL_FAILURE(read(host.address, 0));
  ASSERT_NO_FATAL_FAILURE(read(host.address, 64));
  ASSERT_TRUE(gpu::render::GpuFill(host.address, GuestMemory::kBytes,
                                   0x3f400000, 0x3f400000));
  ASSERT_TRUE(gpu::render::MirrorOwned(host.address));
  ASSERT_NO_FATAL_FAILURE(read(host.address, 191));

  GuestMemory arguments(0x2200100000ull, false);
  ASSERT_NE(arguments.memory, MAP_FAILED);
  const u32 literal_ps[4096] = {0x7e0002f2, 0x7e020280, 0xf800080f, 0x00010100,
                                0xbf810000};
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto indirect_program =
      gpu::gcn::Recompile(vs, literal_ps, user_data, user_data);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(indirect_program.ok);
  draw.recomp = &indirect_program;
  draw.ps_addr = reinterpret_cast<u64>(literal_ps);
  draw.indirect_args = arguments.address;
  auto* words = static_cast<volatile u32*>(arguments.memory);
  for (u32 trial = 0; trial < 4; ++trial) {
    SCOPED_TRACE(trial);
    words[0] = trial == 1 ? 0 : 1;
    words[1] = trial == 2 ? 0 : 1;
    words[2] = 0;
    words[3] = 0;
    draw.rt_base = draw.mrt_base[0] =
        reinterpret_cast<u64>(targets[frame++].data());
    const u64 staged = gpu::render::cs::g_indirect_staged_n;
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    gpu::render::BeginFrame(renderer);
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    const u32 slot_index = gpu::render::g_frame.slot_idx;
    gpu::render::EndFrame(renderer, draw.rt_base);
    const auto& slot = gpu::render::g_frame.slots[slot_index];
    ASSERT_TRUE(gpu::render::Device().Wait(slot.submission));
    ASSERT_TRUE(slot.presentable);
    EXPECT_EQ(slot.readback->mapped()[(8 * 16 + 4) * 4],
              trial == 1 || trial == 2 ? 0 : 255);
    EXPECT_LE(gpu::render::cs::g_indirect_staged_n, staged + 1);
    if (!trial)
      EXPECT_EQ(gpu::render::cs::g_indirect_staged_n, staged);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
  }
  draw.recomp = &program;
  draw.ps_addr = reinterpret_cast<u64>(ps);
  draw.indirect_args = 0;

  GuestMemory pool(0x2001000000ull, true);
  ASSERT_NE(pool.memory, MAP_FAILED);
  *static_cast<float*>(pool.memory) = .75f;
  u64 offset = 0, bytes = 0;
  ASSERT_NE(gpu::render::DirectImportAt(pool.address, &offset, &bytes),
            nullptr);
  EXPECT_EQ(offset, 0u);
  EXPECT_EQ(bytes, GuestMemory::kBytes);
  EXPECT_EQ(gpu::gcn::g_direct_table, table);
  ASSERT_NO_FATAL_FAILURE(read(pool.address, 0));
  ASSERT_NO_FATAL_FAILURE(read(pool.address, 191));
  ASSERT_NO_FATAL_FAILURE(read(host.address, 191));
  EXPECT_EQ(gpu::gcn::g_direct_table, table);
  EXPECT_EQ(&direct_shader, &gpu::ps5::GetGraphicsShader(shader));
  draw.recomp = &direct_shader;
  draw.ps_addr = reinterpret_cast<u64>(rdna_ps);
  ASSERT_NO_FATAL_FAILURE(read(host.address, 191));

  const u64 high_addresses[] = {0x10000000000ull, 0x201400000000ull,
                                0x201480000000ull, 0x201500000000ull};
  for (u32 i = 0; i < base::ArraySize(high_addresses); ++i) {
    SCOPED_TRACE(high_addresses[i]);
    GuestMemory high(high_addresses[i], false);
    ASSERT_NE(high.memory, MAP_FAILED);
    *static_cast<float*>(high.memory) = i & 1 ? .75f : .25f;
    draw.recomp = i & 1 ? &program : &direct_shader;
    draw.ps_addr = reinterpret_cast<u64>(i & 1 ? ps : rdna_ps);
    ASSERT_NO_FATAL_FAILURE(read(high.address, 0));
    ASSERT_NO_FATAL_FAILURE(read(high.address, i & 1 ? 191 : 64));
    *static_cast<float*>(high.memory) = .5f;
    gpu::render::CollectGuestWrites();
    ASSERT_NO_FATAL_FAILURE(read(high.address, 128));
  }

  GuestMemory partial(0x201600001000ull, false, 4096);
  GuestMemory readonly(0x201600008000ull, false, 4096);
  ASSERT_NE(partial.memory, MAP_FAILED);
  ASSERT_NE(readonly.memory, MAP_FAILED);
  *static_cast<float*>(partial.memory) = .25f;
  *static_cast<float*>(readonly.memory) = .75f;
  ASSERT_EQ(mprotect(readonly.memory, readonly.bytes, PROT_READ), 0);
  gpu::render::NoteGuestRemap(readonly.address, readonly.bytes);
  ASSERT_NO_FATAL_FAILURE(read(partial.address, 0));
  ASSERT_NO_FATAL_FAILURE(read(partial.address, 64));
  ASSERT_NO_FATAL_FAILURE(read(readonly.address, 191));
  ASSERT_NO_FATAL_FAILURE(read(partial.address + 12288, 0));

  gpu::rhi::BufferDesc desc;
  desc.size = GuestMemory::kBytes;
  desc.usage = gpu::rhi::kBufferStorage | gpu::rhi::kBufferAddress;
  desc.memory = gpu::rhi::MemoryKind::kUploadDevice;
  auto* redirected = gpu::render::Device().CreateBuffer(desc);
  ASSERT_NE(redirected, nullptr);
  ASSERT_NE(redirected->mapped(), nullptr);
  ASSERT_NE(redirected->address(), 0u);
  constexpr u64 last_block = 0xffffffff0000ull;
  *reinterpret_cast<float*>(redirected->mapped() + desc.size - 4) = .25f;
  gpu::render::DirectRedirect(last_block, redirected->address());
  ASSERT_NO_FATAL_FAILURE(read(last_block + desc.size - 4, 64));
  struct sigaction restore{};
  restore.sa_handler = [](int) { gpu::render::DirectRestore(last_block); };
  ASSERT_EQ(sigaction(SIGUSR1, &restore, nullptr), 0);
  ASSERT_EQ(raise(SIGUSR1), 0);
  ASSERT_NO_FATAL_FAILURE(read(last_block + desc.size - 4, 0));
  gpu::render::Device().Destroy(redirected);

  GuestMemory alias(0x201700000000ull, false);
  ASSERT_NE(alias.memory, MAP_FAILED);
  *static_cast<float*>(alias.memory) = .25f;
  ASSERT_NO_FATAL_FAILURE(read(alias.address, 0));
  ASSERT_NO_FATAL_FAILURE(read(alias.address, 64));
  ASSERT_EQ(mmap(alias.memory, alias.bytes, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_FIXED, pool.fd, 0),
            alias.memory);
  host_memory::TrackMemoryMapping(alias.memory, alias.bytes);
  gpu::render::NoteGuestRemap(alias.address, alias.bytes);
  gpu::render::SyncGuestDirect();
  ASSERT_NO_FATAL_FAILURE(read(alias.address, 191));
  EXPECT_EQ(gpu::gcn::g_direct_table, table);
}
}  // namespace
