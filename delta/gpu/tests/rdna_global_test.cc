#include <gtest/gtest.h>
#include <array>
#include <vector>
#ifdef OS_LINUX
#include <sys/mman.h>
#endif
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/guest_memory_ranges.h"
#include "utl/mem.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/vulkan/vk_device.h"

namespace {
class RdnaGlobal : public testing::Test {
 protected:
  void SetUp() override {
    if (!gpu::rhi::Init(gpu::rhi::DefaultRenderer()))
      GTEST_SKIP() << "Vulkan is required";
  }
  using Page = std::array<u32, 16384>;
  Page& PageForDispatch() {
    alignas(65536) static std::array<Page, 32> pages{};
    static u32 next = 0;
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(pages.data()), sizeof(pages));
    auto& page = pages.at(next++);
    page.fill(0xa5a5a5a5);
    return page;
  }
  std::vector<u32> program;
  std::array<u32, 3> group_start{}, group_end{1, 1, 1};
  u32 tgid_enable = 0, initiator = 1;
  void Mov(u32 vgpr, u32 value) {
    program.push_back(0x7e0002ff | vgpr << 17);
    program.push_back(value);
  }
  void Global(u32 op, u32 addr, u32 data, u32 scalar, i32 offset = 0) {
    program.push_back(0xdc008000 | op << 18 | (u32(offset) & 0xfff));
    program.push_back(addr | scalar << 16 | data << (op >= 0x18 ? 8 : 24));
  }
  void Run(u64 src, u64 dst, u32 threads = 1) {
    alignas(256) static std::array<u32, 16384> code{};
    code.fill(0);
    std::copy(program.begin(), program.end(), code.begin());
    code[program.size()] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    const auto cs =
        gpu::rdna::RecompileCompute(code.data(), threads, 1, 1, 4, tgid_enable, 0);
    ASSERT_TRUE(cs.ok);
    gpu::ps5::Regs regs;
    const u64 address = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = address >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = address >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = threads;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = (4 << 1) | (tgid_enable << 7);
    for (u32 axis = 0; axis < 3; ++axis)
      regs[gpu::ps5::mmCOMPUTE_START_X + axis] = group_start[axis];
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud] = src;
    regs[ud + 1] = src >> 32;
    regs[ud + 2] = dst;
    regs[ud + 3] = dst >> 32;
    const u32 launch[] = {group_end[0], group_end[1], group_end[2], initiator};
    gpu::ps5::DispatchCompute(gpu::rhi::DefaultRenderer(), regs, launch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(gpu::rhi::DefaultRenderer()));
  }
};

TEST_F(RdnaGlobal, DispatchStartsUseExclusiveEndsAndPreserveWorkgroupIds) {
  auto& dest = PageForDispatch();
  program.push_back(0x7e000204);  // v_mov_b32 v0, s4 (workgroup ID)
  program.push_back(0x34000082);  // v_lshlrev_b32 v0, 2, v0
  program.push_back(0x7e040204);  // v_mov_b32 v2, s4
  Global(0x1c, 0, 2, 2);
  for (u32 axis = 0; axis < 3; ++axis) {
    tgid_enable = 1u << axis;
    group_start = {0, 0, 0};
    group_end = {1, 1, 1};
    group_start[axis] = 3;
    group_end[axis] = 6;
    initiator = 1;
    dest.fill(0xa5a5a5a5);
    Run(0, reinterpret_cast<u64>(dest.data()));
    for (u32 i = 0; i < 8; ++i)
      ASSERT_EQ(dest[i], i >= 3 && i < 6 ? i : 0xa5a5a5a5) << axis << ':' << i;

    initiator = 5;  // FORCE_START_AT_000 ignores the START registers.
    Run(0, reinterpret_cast<u64>(dest.data()));
    for (u32 i = 0; i < 8; ++i)
      ASSERT_EQ(dest[i], i < 6 ? i : 0xa5a5a5a5) << axis << ':' << i;

    initiator = 1;
    dest.fill(0xa5a5a5a5);
    for (u32 end : {2u, 3u}) {
      group_end[axis] = end;
      Run(0, reinterpret_cast<u64>(dest.data()));
      for (u32 i = 0; i < 8; ++i)
        ASSERT_EQ(dest[i], 0xa5a5a5a5) << axis << ':' << i;
    }
  }
}

TEST_F(RdnaGlobal, SignedOffsetsAndOverlappingVectorDestinations) {
  auto& source = PageForDispatch();
  auto& dest = PageForDispatch();
  for (u32 i = 0; i < 4; ++i)
    source[i] = 100 + i;
  Mov(3, 0);
  Global(0x0e, 3, 2, 0, -4);  // overwrites its own VGPR address operand
  Global(0x1e, 0, 2, 2);
  Run(reinterpret_cast<u64>(source.data()) + 4,
      reinterpret_cast<u64>(dest.data()));
  for (u32 i = 0; i < 4; ++i)
    EXPECT_EQ(dest[i], source[i]);
  EXPECT_EQ(dest[4], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, UnsignedVectorOffsetCarriesIntoHighAddress) {
  auto& source = PageForDispatch();
  auto& dest = PageForDispatch();
  source[0] = 0x12345678;
  Mov(3, 0xfffffff0);
  Global(0x0c, 3, 2, 0);
  Global(0x1c, 0, 2, 2);
  Run(reinterpret_cast<u64>(source.data()) - 0xfffffff0ull,
      reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], source[0]);
}

TEST_F(RdnaGlobal, FullVectorAddressAndStoreThenLoadAlias) {
  auto& dest = PageForDispatch();
  const u64 address = reinterpret_cast<u64>(dest.data());
  Mov(0, address);
  Mov(1, address >> 32);
  Mov(2, 0x12345678);
  Global(0x1c, 0, 2, 125);
  Global(0x0c, 0, 0, 125);  // destination aliases address low half
  Mov(3, 4);
  Global(0x1c, 3, 0, 2);
  Run(0, address);
  EXPECT_EQ(dest[0], 0x12345678);
  EXPECT_EQ(dest[1], 0x12345678);
  EXPECT_EQ(dest[2], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, ConcurrentByteStoresPreserveOtherBytesAndCopiedWriteback) {
  // Anonymous imports are rebuilt each dispatch. Force the copy path and
  // verify that its dirty bitmap writes back precisely the changed words.
  const bool imports = gpu::vk::g_dev.host_import_available;
  gpu::vk::g_dev.host_import_available = false;
  auto& dest = PageForDispatch();
  Mov(1, 0x44);
  Global(0x18, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()), 63);
  gpu::vk::g_dev.host_import_available = imports;
  for (u32 i = 0; i < 15; ++i)
    EXPECT_EQ(dest[i], 0x44444444);
  EXPECT_EQ(dest[15], 0xa5444444);
  EXPECT_EQ(dest[16], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, InvalidAddressDoesNotClampToAnotherWord) {
  auto& dest = PageForDispatch();
  Mov(1, 0x12345678);
  Mov(3, 0);
  Global(0x1c, 3, 1, 0);  // store to unmapped low memory must be discarded
  Global(0x0c, 3, 2, 0);
  Global(0x1c, 3, 2, 2);
  Run(4, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0);
  EXPECT_EQ(dest[1], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, SignedSubwordLoadsAndConcurrentHalfwordStores) {
  auto& source = PageForDispatch();
  auto& dest = PageForDispatch();
  source[0] = 0x8001ff80;
  for (u32 i = 0; i < 4; ++i) {
    Mov(3, i >= 2 ? 2 : 0);
    Global(0x08 + i, 3, 2, 0);
    Mov(3, i * 4);
    Global(0x1c, 3, 2, 2);
  }
  Run(reinterpret_cast<u64>(source.data()), reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0x80);
  EXPECT_EQ(dest[1], 0xffffff80);
  EXPECT_EQ(dest[2], 0x8001);
  EXPECT_EQ(dest[3], 0xffff8001);
  program.clear();
  auto& half = PageForDispatch();
  program.push_back(0x34000081);  // v_lshlrev_b32 v0, 1, v0
  Mov(1, 0x1234);
  Global(0x1a, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(half.data()), 31);
  for (u32 i = 0; i < 15; ++i)
    EXPECT_EQ(half[i], 0x12341234);
  EXPECT_EQ(half[15], 0xa5a51234);
  EXPECT_EQ(half[16], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, InactiveLanesCannotStore) {
  auto& dest = PageForDispatch();
  program.push_back(0xd4d2007e);  // v_cmpx_eq_u32 v0, 0
  program.push_back(0x00010100);
  Mov(1, 0x44);
  Global(0x18, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()), 4);
  EXPECT_EQ(dest[0], 0xa5a5a544);
  EXPECT_EQ(dest[1], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, AddressesBeyondTheOldSixtyFourMiBWindow) {
  alignas(65536) static std::array<u32, (65 * 1024 * 1024) / 4> data{};
  constexpr u32 offset = 64 * 1024 * 1024 + 32;
  data[offset / 4] = 0xa5a5a5a5;
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(data.data()), sizeof(data));
  Mov(3, offset);
  Mov(1, 0x76543210);
  Global(0x1c, 3, 1, 2);
  Global(0x0c, 3, 2, 2);
  Mov(3, 0);
  Global(0x1c, 3, 2, 2);
  Run(0, reinterpret_cast<u64>(data.data()));
  EXPECT_EQ(data[offset / 4], 0x76543210);
  EXPECT_EQ(data[0], 0x76543210);
  EXPECT_EQ(data[offset / 4 + 1], 0);
}

#ifdef OS_LINUX
TEST_F(RdnaGlobal, TrackedAnonymousMappingReuseAndReplacement) {
  constexpr size_t size = 65536;
  // Leave guard space so /proc/maps cannot merge this allocation with an
  // unrelated anonymous allocation. Keep it alive while Vulkan imports it.
  void* reservation = utl::allocMem(nullptr, size * 3,
      utl::pageProtection::priv, utl::allocationType::reserve);
  ASSERT_NE(reservation, nullptr);
  const u64 base = (reinterpret_cast<u64>(reservation) + size - 1) & ~(size - 1);
  auto* source = static_cast<u32*>(utl::allocMem(reinterpret_cast<void*>(base),
      size, utl::pageProtection::w, utl::allocationType::commit));
  ASSERT_NE(source, nullptr);
  gpu::ps5::NoteGpuPool(base, size);
  const auto identity = [&] {
    for (const auto& range : gpu::ps5::GuestMemoryRanges({}))
      if (range.base <= base && base < range.base + range.size)
        return range.identity;
    return u64(0);
  };
  const u64 first = identity();
  ASSERT_NE(first, 0);
  auto& dest = PageForDispatch();
  Global(0x0c, 0, 2, 0);
  Global(0x1c, 0, 2, 2);
  for (u32 value : {0x12345678u, 0x87654321u}) {
    source[0] = value;
    Run(base, reinterpret_cast<u64>(dest.data()));
    ASSERT_EQ(dest[0], value);
    ASSERT_EQ(identity(), first);
  }
  ASSERT_EQ(utl::allocMem(source, size, utl::pageProtection::w,
      utl::allocationType::commit), source);
  ASSERT_NE(identity(), first);
  source[0] = 0xaabbccdd;
  Run(base, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0xaabbccdd);
}

TEST(MemoryMappingIdentity, PartialReplacementAndUntrackedGaps) {
  auto* base = reinterpret_cast<u8*>(0x12300000000ull);
  utl::trackMemoryMapping(base, 0x10000);
  const u64 original = utl::memoryMappingIdentity(base, 0x10000);
  ASSERT_NE(original, 0);
  utl::trackMemoryMapping(base + 0x4000, 0x4000);
  EXPECT_NE(utl::memoryMappingIdentity(base, 0x10000), original);
  EXPECT_NE(utl::memoryMappingIdentity(base, 0x4000), 0);
  EXPECT_NE(utl::memoryMappingIdentity(base + 0x8000, 0x8000), 0);
  utl::forgetMemoryMapping(base + 0x4000, 0x4000);
  EXPECT_EQ(utl::memoryMappingIdentity(base, 0x10000), 0);
  EXPECT_EQ(utl::memoryMappingIdentity(base + 0x4000, 4), 0);
  EXPECT_NE(utl::memoryMappingIdentity(base + 0x8000, 0x8000), 0);
  utl::forgetMemoryMapping(base, 0x10000);
  EXPECT_EQ(utl::memoryMappingIdentity(base, 4), 0);
}

TEST_F(RdnaGlobal, ReadOnlyMappingRejectsStoresButAllowsLoads) {
  // Kept alive until process exit because Vulkan may retain a host import.
  static void* memory = mmap(nullptr, 65536, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(memory, MAP_FAILED);
  auto* source = static_cast<u32*>(memory);
  source[0] = 0x12345678;
  ASSERT_EQ(mprotect(memory, 65536, PROT_READ), 0);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory), 65536);
  auto& dest = PageForDispatch();
  Mov(1, 0xffffffff);
  Global(0x1c, 0, 1, 0);
  Global(0x0c, 0, 2, 0);
  Global(0x1c, 0, 2, 2);
  Run(reinterpret_cast<u64>(source), reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(source[0], 0x12345678);
  EXPECT_EQ(dest[0], 0x12345678);
  ASSERT_EQ(mprotect(memory, 65536, PROT_READ | PROT_WRITE), 0);
}
#endif

TEST_F(RdnaGlobal, DecoderXor3PreservesIntegerBitsAndAliasedOperands) {
  auto& dest = PageForDispatch();
  Mov(1, 0xffff8001);
  Mov(2, 0xf0f00f0f);
  Mov(3, 0x80808080);
  program.push_back(0xd5780001);
  program.push_back(257 | (258 << 9) | (259 << 18));
  Global(0x1c, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0xffff8001u ^ 0xf0f00f0fu ^ 0x80808080u);
}

TEST_F(RdnaGlobal, DecoderCodeAndHashExtendBeyondSixteenKiB) {
  auto& dest = PageForDispatch();
  program.assign(5001, 0xbf800000);
  program[0] = 0xbf820000 | 5000;  // jump over padding to the tail
  Mov(1, 123);
  Global(0x1c, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 123);
  program[5002] = 456;  // changing only the late tail must invalidate the cache
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 456);
}
}  // namespace

namespace gpu::rhi {
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0, g_dcb_n = 0;
}  // namespace gpu::rhi
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}
