#include <gtest/gtest.h>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include "base/containers/array.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/render/compute_internal.h"
#include "gpu/render/device.h"
#include "gpu/render/draw_recomp.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_direct.h"
#include "gpu/render/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/write_tracker.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

extern "C" void prosperity_gpu_end_of_pipe() {}
extern "C" bool prosperity_ps5_is_display_buffer(u64) { return false; }

namespace {

struct DirectMemory {
  static constexpr u64 kBase = gpu::gcn::kDirectBase + (16ull << 20);
  static constexpr u64 kBytes = 4ull << 20;
  int fd = memfd_create("delta_dmem_test", 0);
  void* memory = MAP_FAILED;

  DirectMemory() {
    if (fd >= 0 && !ftruncate(fd, kBytes))
      memory = mmap(reinterpret_cast<void*>(kBase), kBytes,
                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE,
                    fd, 0);
  }

  ~DirectMemory() {
    if (memory != MAP_FAILED) {
      if (gpu::render::g_mirror) {
        gpu::render::cs::CsBatchSubmit();
        for (u64 at = kBase; at < kBase + kBytes;
             at += gpu::render::kMirrorBlock)
          if (gpu::render::MirrorOwned(at)) {
            const u32 value = *reinterpret_cast<volatile const u32*>(at);
            (void)value;
          }
      }
      gpu::GuestWriteTracker().Release(kBase, kBytes);
      munmap(memory, kBytes);
      gpu::render::NoteGuestRemap(kBase, kBytes);
    }
    if (fd >= 0)
      close(fd);
  }
};

TEST(VkMirrorBinding, OwnedOutputStaysResidentAcrossDrawsAndCpuWrites) {
  options::Init();
  DirectMemory memory;
  ASSERT_NE(memory.memory, MAP_FAILED);
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A graphics device is required";
  const auto& caps = gpu::render::Device().caps();
  if (!caps.sparse_buffer || !caps.host_import || !caps.buffer_address ||
      !caps.copy_after)
    GTEST_SKIP() << "Sparse buffers, host imports and ordered copies are required";
  gpu::GuestWriteTracker().Enable();
  static u8 fault_stack[65536];
  stack_t stack{};
  stack.ss_sp = fault_stack;
  stack.ss_size = sizeof(fault_stack);
  ASSERT_EQ(sigaltstack(&stack, nullptr), 0);
  struct sigaction fault{};
  fault.sa_flags = SA_SIGINFO | SA_ONSTACK;
  fault.sa_sigaction = [](int, siginfo_t* si, void*) {
    if (!host_memory::HandleWriteFault(reinterpret_cast<uintptr_t>(si->si_addr)))
      std::abort();
  };
  ASSERT_EQ(sigaction(SIGSEGV, &fault, nullptr), 0);
  const u32 vs[4096] = {0x7e000280, 0x7e0202f2, 0xf80008cf, 0x01000000,
                        0xbf810000};
  const u32 ps[4096] = {0x7e000280, 0xe8001000 | (56u << 19) | (3u << 16),
                        0x80020000, 0xf800080f, 0x03020100, 0xbf810000};
  const u64 base = DirectMemory::kBase;
  const u32 user_data[32] = {u32(base), u32(base >> 32),
                            DirectMemory::kBytes, 0x14fac};
  const auto program = gpu::rdna::Recompile(vs, ps, user_data, user_data);
  ASSERT_TRUE(program.ok);
  ASSERT_EQ(program.ps_bufs.size(), 1u);
  const u32 cs[4096] = {0x7e020204, 0xe0702000, 0x80000100, 0xbf810000};
  const auto previous_isa = gpu::gcn::DefaultIsaMode();
  gpu::gcn::SetDefaultIsaMode(gpu::gcn::IsaMode::kBase);
  const auto compute = gpu::gcn::RecompileCompute(cs, 64, 1, 1, 5, 0, 0);
  gpu::gcn::SetDefaultIsaMode(previous_isa);
  ASSERT_TRUE(compute.ok);
  ASSERT_EQ(compute.resources.size(), 1u);
  ASSERT_TRUE(gpu::render::InitGuestDirect());
  alignas(65536) static base::Array<u8, 65536> targets[3]{};
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
    draw.rt_base = draw.mrt_base[0] = reinterpret_cast<u64>(targets[trial].data());
    draw.rt_w = draw.rt_h = 16;
    draw.viewport_x_scale = draw.viewport_y_scale = 8;
    draw.viewport_x_offset = 4;
    draw.viewport_y_offset = 8;
    draw.mrt_count = draw.mrt_bound_mask = 1;
    draw.mrt_info[0] = 10u << 2;
    const u64 resolves = gpu::render::cs::g_mirror_resolves.load();
    const u64 copies = gpu::render::cs::g_buf_bridge_bytes;
    const u64 cpu_copies = gpu::g_ring_raw_bytes;
    const u64 bound = gpu::g_bind_bytes;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    ci.user_data[4] = 0xff804020;
    if (trial == 2)
      ASSERT_TRUE(gpu::render::GpuFill(base, DirectMemory::kBytes,
                                      ci.user_data[4], ci.user_data[4]));
    else
      ASSERT_TRUE(gpu::render::Dispatch(renderer, ci));
    draw.viewport_x_offset = 12;
    ASSERT_TRUE(gpu::render::DrawRecomp(renderer, draw));
    EXPECT_EQ(gpu::render::cs::g_buf_bridge_bytes, copies);
    EXPECT_EQ(gpu::g_ring_raw_bytes, cpu_copies);
    EXPECT_EQ(gpu::g_bind_bytes, bound + 2 * DirectMemory::kBytes);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
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
      const bool first = i % 16 < 8;
      EXPECT_EQ(pixels[i * 4], first ? 64 : 32);
      EXPECT_EQ(pixels[i * 4 + 1], first ? 128 : 64);
      EXPECT_EQ(pixels[i * 4 + 2], first ? 255 : 128);
      EXPECT_EQ(pixels[i * 4 + 3], 255);
    }
    EXPECT_EQ(colored, 2u);
    EXPECT_EQ(gpu::render::cs::g_mirror_resolves.load(), resolves);
    *static_cast<volatile u32*>(memory.memory) = 0xff0000ff;
    gpu::render::CollectGuestWrites();
    EXPECT_FALSE(gpu::render::CsHoldsGeometry(base, DirectMemory::kBytes));
    EXPECT_GT(gpu::render::cs::g_mirror_resolves.load(), resolves);
  }
}

}  // namespace
