#include <gtest/gtest.h>
#include <cstring>

#include "base/option.h"
#ifdef OS_LINUX
#include <sys/mman.h>
#endif
#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/pair.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/guest_memory_ranges.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/render/device.h"
#include "host_memory/host_memory.h"

namespace {
class RdnaGlobal : public testing::Test {
 protected:
  void SetUp() override {
    if (!gpu::render::Init(gpu::render::DefaultRenderer()))
      GTEST_SKIP() << "Vulkan is required";
  }
  using Page = base::Array<u32, 16384>;
  Page& PageForDispatch() {
    alignas(65536) static base::Array<Page, 32> pages{};
    static u32 next = 0;
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(pages.data()), sizeof(pages));
    auto& page = pages.at(next++);
    page.fill(0xa5a5a5a5);
    return page;
  }
  base::Vector<u32> program_;
  base::Array<u32, 3> group_start_{}, group_end_{1, 1, 1};
  u32 tgid_enable_ = 0, initiator_ = 1, lds_size_ = 0;
  void Mov(u32 vgpr, u32 value) {
    program_.push_back(0x7e0002ff | vgpr << 17);
    program_.push_back(value);
  }
  void Global(u32 op, u32 addr, u32 data, u32 scalar, i32 offset = 0) {
    program_.push_back(0xdc008000 | op << 18 | (u32(offset) & 0xfff));
    program_.push_back(addr | scalar << 16 | data << (op >= 0x18 ? 8 : 24));
  }
  void Run(u64 src, u64 dst, u32 threads = 1) {
    alignas(256) static base::Array<u32, 16384> code{};
    code.fill(0);
    base::Copy(program_.begin(), program_.end(), code.begin());
    code[program_.size()] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    const auto cs = gpu::rdna::RecompileCompute(code.data(), threads, 1, 1, 4,
                                                tgid_enable_, lds_size_, false,
                                                (initiator_ & (1u << 15)) != 0);
    ASSERT_TRUE(cs.ok);
    gpu::ps5::Regs regs;
    const u64 address = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = address >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = address >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = threads;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] =
        (4 << 1) | (tgid_enable_ << 7) | (lds_size_ << 15);
    for (u32 axis = 0; axis < 3; ++axis)
      regs[gpu::ps5::mmCOMPUTE_START_X + axis] = group_start_[axis];
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud] = src;
    regs[ud + 1] = src >> 32;
    regs[ud + 2] = dst;
    regs[ud + 3] = dst >> 32;
    const u32 launch[] = {group_end_[0], group_end_[1], group_end_[2],
                          initiator_};
    gpu::ps5::DispatchCompute(gpu::render::DefaultRenderer(), regs, launch, 4);
    ASSERT_TRUE(gpu::render::FlushCsWrites(gpu::render::DefaultRenderer()));
  }
};

TEST_F(RdnaGlobal, InactiveLaneCannotOverwriteSharedMemory) {
  auto& dest = PageForDispatch();
  lds_size_ = 1;
  Mov(0, 0);
  Mov(2, 17);
  program_.insert(program_.end(), {0xd8340000, 0x00000200});  // LDS[0] = 17
  Mov(2, 0x3f800000);
  program_.push_back(0xbefe0480);  // s_mov_b64 exec, 0
  program_.insert(program_.end(), {0xd8340000, 0x00000200});  // inactive store
  program_.push_back(0xbefe04c1);  // s_mov_b64 exec, -1
  program_.insert(program_.end(), {0xd8d80000, 0x02000000});  // v2 = LDS[0]
  Global(0x1c, 0, 2, 2);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 17u);
}

TEST_F(RdnaGlobal, MadAndMacScaleOutputBeforeClamping) {
  auto& dest = PageForDispatch();
  for (u32 op : {0x141u, 0x11fu}) {
    for (u32 omod = 0; omod < 4; ++omod) {
      const u32 result = 4 + (op == 0x141 ? 0 : 4) + omod;
      Mov(result, 0x3f000000);  // MAC accumulator = .5
      Mov(1, 0x40000000);       // source 0 = 2
      Mov(2, 0x3f000000);       // source 1 = .5
      Mov(3, 0x3f000000);       // MAD source 2 = .5
      program_.push_back(0xd4008000 | (op << 16) | result);
      program_.push_back(257 | (258 << 9) | (259 << 18) | (omod << 27));
    }
  }
  Mov(0, 0);
  for (u32 i = 0; i < 8; ++i)
    Global(0x1c, 0, i + 4, 2, i * 4);
  Run(0, reinterpret_cast<u64>(dest.data()));
  for (u32 i = 0; i < 8; ++i)
    EXPECT_EQ(dest[i], i % 4 == 3 ? 0x3f400000u : 0x3f800000u) << i;
}

TEST_F(RdnaGlobal, TypedComputeLoadDecodesRdnaFormat) {
  auto& source = PageForDispatch();
  auto& dest = PageForDispatch();
  source[0] = 0x3e800000;
  source[1] = 0xbf400000;
  source[2] = 0x3f000000;
  source[3] = 0x3f800000;
  // Inline V# at s[4:7], byte offsets v1 = lane * 8. R32G32_FLOAT
  // is GFX10 format 64, not the GCN DFMT/NFMT occupying those bits.
  program_ = {0xbe840300, 0xbe850301, 0xbe8603a0,
              0xbe870380, 0x34020083, 0xe8001000 | (64u << 19) | (1u << 16),
              0x80010401};
  Global(0x1d, 1, 4, 2);
  for (u32 format : {64u, 77u}) {
    // A two-component read is also legal with an RGBA32 format: it does not
    // require the instruction to request all four stored components.
    program_[5] = 0xe8001000 | (format << 19) | (1u << 16);
    Run(reinterpret_cast<u64>(source.data()),
        reinterpret_cast<u64>(dest.data()), 2);
    for (u32 i = 0; i < 4; ++i)
      EXPECT_EQ(dest[i], source[i]) << format << ':' << i;
  }
}

TEST_F(RdnaGlobal, GdsCounterUsesM0BaseAndReturnsPreOperationValue) {
  auto& renderer = gpu::render::DefaultRenderer();
  auto& dest = PageForDispatch();
  const u32 initial = 17;
  ASSERT_TRUE(gpu::render::WriteGds(renderer, 0xc70, &initial, 4));
  ASSERT_TRUE(gpu::render::FillGds(renderer, 0x10, 4, 99));
  // M0={base=0xc60,size=32}; append and consume the counter at +0x10.
  program_ = {0xbefc03ff, 0x0c600020, 0xd8fa0010,
              0x02000000, 0xd8f60010, 0x03000000};
  Mov(0, 0);
  Global(0x1c, 0, 2, 2);
  Global(0x1c, 0, 3, 2, 4);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 17u);
  EXPECT_EQ(dest[1], 18u);
  u32 value = 0;
  ASSERT_TRUE(gpu::render::ReadGds(renderer, 0xc70, &value, 4));
  EXPECT_EQ(value, initial);
  ASSERT_TRUE(gpu::render::ReadGds(renderer, 0x10, &value, 4));
  EXPECT_EQ(value, 99u);
  EXPECT_FALSE(gpu::render::FillGds(renderer, 65535, 4, 0));
}

TEST_F(RdnaGlobal, MaskBitCountsUseGuestWaveLaneAcrossHostSubgroups) {
  auto& dest = PageForDispatch();
  program_ = {0xd7650003, 0x000100c1,  // v_mbcnt_lo v3, -1, 0
              0xd7660003, 0x000206c1,  // v_mbcnt_hi v3, -1, v3
              0x34000082};             // v0 = local thread ID * 4
  Global(0x1c, 0, 3, 2);
  // Reuse identical code and group size; the dispatch flag must distinguish
  // cached modules when changing wave width and then switching back.
  for (u32 width : {64u, 32u, 64u}) {
    initiator_ = 1 | (width == 32 ? 1u << 15 : 0);
    Run(0, reinterpret_cast<u64>(dest.data()), 128);
    for (u32 lane = 0; lane < 128; ++lane)
      EXPECT_EQ(dest[lane], lane % width)
          << "width=" << width << " lane=" << lane;
  }
}

TEST_F(RdnaGlobal, GdsCountersBroadcastAcrossBothHalvesOfEachWave) {
  auto& renderer = gpu::render::DefaultRenderer();
  auto& dest = PageForDispatch();
  const u64 mask = (8ull << 32) | 5u;  // Three active lanes per guest wave.
  program_ = {0xbefc03ff, 0x0c600020, 0x34000082};
  Mov(2, 99);
  Mov(3, 99);
  program_.insert(program_.end(),
                  {0xbefe0400,  // exec = s[0:1]
                   0xd8fa0010, 0x02000000, 0xd8f60010, 0x03000000, 0xbefe04c1});
  Global(0x1c, 0, 2, 2);
  Global(0x1c, 0, 3, 2, 512);
  // Include a partially populated final wave, for both dispatch wave sizes.
  for (const auto [width, threads] :
       {base::Pair{64u, 128u}, base::Pair{64u, 70u}, base::Pair{32u, 80u}}) {
    SCOPED_TRACE(testing::Message()
                 << "width=" << width << " threads=" << threads);
    ASSERT_TRUE(gpu::render::FillGds(renderer, 0xc70, 4, 17));
    initiator_ = 1 | (width == 32 ? 1u << 15 : 0);
    Run(mask, reinterpret_cast<u64>(dest.data()), threads);
    base::Vector<base::Pair<u32, u32>> allocations, consumptions;
    u32 total = 0;
    for (u32 base = 0; base < threads; base += width) {
      u32 count = 0;
      for (u32 lane = base; lane < base::Min(base + width, threads); ++lane) {
        const bool active = (mask >> (lane - base)) & 1;
        count += active;
        EXPECT_EQ(dest[lane], active ? dest[base] : 99u) << lane;
        EXPECT_EQ(dest[128 + lane], active ? dest[128 + base] : 99u) << lane;
      }
      allocations.emplace_back(dest[base], count);
      consumptions.emplace_back(dest[128 + base], count);
      total += count;
    }
    // Waves may execute atomics in either order, but their allocations must
    // form a contiguous range, and consumes must return its pre-op endpoint.
    base::Sort(allocations.begin(), allocations.end());
    base::Sort(consumptions.begin(), consumptions.end(),
               [](const auto& a, const auto& b) { return b < a; });
    u32 next = 17;
    for (const auto [base, count] : allocations) {
      EXPECT_EQ(base, next);
      next += count;
    }
    next = 17 + total;
    for (const auto [base, count] : consumptions) {
      EXPECT_EQ(base, next);
      next -= count;
    }
    u32 value = 0;
    ASSERT_TRUE(gpu::render::ReadGds(renderer, 0xc70, &value, 4));
    EXPECT_EQ(value, 17u);
  }
}

TEST_F(RdnaGlobal, LiteralExecMasksSelectBothHalvesOfWave64) {
  auto& dest = PageForDispatch();
  const u64 mask = (8ull << 32) | 5u;  // lanes 0, 2 and 35
  for (bool save : {false, true}) {
    dest.fill(0xa5a5a5a5);
    program_.clear();
    program_.push_back(0x34000082);  // v0 = local thread ID * 4
    Mov(2, 17);
    // s[0:1] contains the literal mask, supplied through user data.
    program_.push_back(save ? 0xbe862400 : 0xbefe0400);
    Global(0x1c, 0, 2, 2);
    Run(mask, reinterpret_cast<u64>(dest.data()), 64);
    for (u32 lane = 0; lane < 64; ++lane)
      EXPECT_EQ(dest[lane], ((mask >> lane) & 1) ? 17u : 0xa5a5a5a5)
          << "save=" << save << " lane=" << lane;
  }
}

TEST_F(RdnaGlobal, CompareMaskControlsUpperWaveLanesAndConditionalMoves) {
  auto& dest = PageForDispatch();
  Mov(2, 1);
  Mov(3, 17);
  program_.push_back(0x7d8400a3);  // v_cmp_eq_u32 vcc, 35, v0
  program_.push_back(0x02040702);  // v_cndmask_b32 v2, v2, v3, vcc
  program_.push_back(0x34000082);  // v0 = thread ID * 4
  Global(0x1c, 0, 2, 2);
  program_.push_back(0xbefe046a);  // s_mov_b64 exec, vcc
  Global(0x1c, 0, 3, 2, 256);
  Run(0, reinterpret_cast<u64>(dest.data()), 64);
  for (u32 lane = 0; lane < 64; ++lane) {
    EXPECT_EQ(dest[lane], lane == 35 ? 17u : 1u) << lane;
    EXPECT_EQ(dest[64 + lane], lane == 35 ? 17u : 0xa5a5a5a5) << lane;
  }
}

TEST_F(RdnaGlobal, DispatchStartsUseExclusiveEndsAndPreserveWorkgroupIds) {
  auto& dest = PageForDispatch();
  program_.push_back(0x7e000204);  // v_mov_b32 v0, s4 (workgroup ID)
  program_.push_back(0x34000082);  // v_lshlrev_b32 v0, 2, v0
  program_.push_back(0x7e040204);  // v_mov_b32 v2, s4
  Global(0x1c, 0, 2, 2);
  for (u32 axis = 0; axis < 3; ++axis) {
    tgid_enable_ = 1u << axis;
    group_start_ = {0, 0, 0};
    group_end_ = {1, 1, 1};
    group_start_[axis] = 3;
    group_end_[axis] = 6;
    initiator_ = 1;
    dest.fill(0xa5a5a5a5);
    Run(0, reinterpret_cast<u64>(dest.data()));
    for (u32 i = 0; i < 8; ++i)
      ASSERT_EQ(dest[i], i >= 3 && i < 6 ? i : 0xa5a5a5a5) << axis << ':' << i;

    initiator_ = 5;  // FORCE_START_AT_000 ignores the START registers.
    Run(0, reinterpret_cast<u64>(dest.data()));
    for (u32 i = 0; i < 8; ++i)
      ASSERT_EQ(dest[i], i < 6 ? i : 0xa5a5a5a5) << axis << ':' << i;

    initiator_ = 1;
    dest.fill(0xa5a5a5a5);
    for (u32 end : {2u, 3u}) {
      group_end_[axis] = end;
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
  base::OptionBase* import = nullptr;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), "DELTA_GPU_GUEST_IMPORT") == 0)
      import = const_cast<base::OptionBase*>(option);
  });
  ASSERT_NE(import, nullptr);
  ASSERT_TRUE(import->SetFromString("0"));
  auto& dest = PageForDispatch();
  Mov(1, 0x44);
  Global(0x18, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()), 63);
  import->Reset();
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
  program_.clear();
  auto& half = PageForDispatch();
  program_.push_back(0x34000081);  // v_lshlrev_b32 v0, 1, v0
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
  program_.push_back(0xd4d2007e);  // v_cmpx_eq_u32 v0, 0
  program_.push_back(0x00010100);
  Mov(1, 0x44);
  Global(0x18, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()), 4);
  EXPECT_EQ(dest[0], 0xa5a5a544);
  EXPECT_EQ(dest[1], 0xa5a5a5a5);
}

TEST_F(RdnaGlobal, AddressesBeyondTheOldSixtyFourMiBWindow) {
  alignas(65536) static base::Array<u32, (65 * 1024 * 1024) / 4> data{};
  constexpr u32 kOffset = 64 * 1024 * 1024 + 32;
  data[kOffset / 4] = 0xa5a5a5a5;
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(data.data()), sizeof(data));
  Mov(3, kOffset);
  Mov(1, 0x76543210);
  Global(0x1c, 3, 1, 2);
  Global(0x0c, 3, 2, 2);
  Mov(3, 0);
  Global(0x1c, 3, 2, 2);
  Run(0, reinterpret_cast<u64>(data.data()));
  EXPECT_EQ(data[kOffset / 4], 0x76543210);
  EXPECT_EQ(data[0], 0x76543210);
  EXPECT_EQ(data[kOffset / 4 + 1], 0);
}

#ifdef OS_LINUX
TEST_F(RdnaGlobal, TrackedAnonymousMappingReuseAndReplacement) {
  constexpr size_t kSize = 65536;
  // Leave guard space so /proc/maps cannot merge this allocation with an
  // unrelated anonymous allocation. Keep it alive while Vulkan imports it.
  void* reservation = host_memory::AllocMem(
      nullptr, kSize * 3, host_memory::PageProtection::kPriv,
      host_memory::AllocationType::kReserve);
  ASSERT_NE(reservation, nullptr);
  const u64 base =
      (reinterpret_cast<u64>(reservation) + kSize - 1) & ~(kSize - 1);
  auto* source = static_cast<u32*>(host_memory::AllocMem(
      reinterpret_cast<void*>(base), kSize, host_memory::PageProtection::kW,
      host_memory::AllocationType::kCommit));
  ASSERT_NE(source, nullptr);
  gpu::ps5::NoteGpuPool(base, kSize);
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
  ASSERT_EQ(
      host_memory::AllocMem(source, kSize, host_memory::PageProtection::kW,
                            host_memory::AllocationType::kCommit),
      source);
  ASSERT_NE(identity(), first);
  source[0] = 0xaabbccdd;
  Run(base, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0xaabbccdd);
}

TEST(MemoryMappingIdentity, PartialReplacementAndUntrackedGaps) {
  auto* base = reinterpret_cast<u8*>(0x12300000000ull);
  host_memory::TrackMemoryMapping(base, 0x10000);
  const u64 original = host_memory::MemoryMappingIdentity(base, 0x10000);
  ASSERT_NE(original, 0);
  host_memory::TrackMemoryMapping(base + 0x4000, 0x4000);
  EXPECT_NE(host_memory::MemoryMappingIdentity(base, 0x10000), original);
  EXPECT_NE(host_memory::MemoryMappingIdentity(base, 0x4000), 0);
  EXPECT_NE(host_memory::MemoryMappingIdentity(base + 0x8000, 0x8000), 0);
  host_memory::ForgetMemoryMapping(base + 0x4000, 0x4000);
  EXPECT_EQ(host_memory::MemoryMappingIdentity(base, 0x10000), 0);
  EXPECT_EQ(host_memory::MemoryMappingIdentity(base + 0x4000, 4), 0);
  EXPECT_NE(host_memory::MemoryMappingIdentity(base + 0x8000, 0x8000), 0);
  host_memory::ForgetMemoryMapping(base, 0x10000);
  EXPECT_EQ(host_memory::MemoryMappingIdentity(base, 4), 0);
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
  program_.push_back(0xd5780001);
  program_.push_back(257 | (258 << 9) | (259 << 18));
  Global(0x1c, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 0xffff8001u ^ 0xf0f00f0fu ^ 0x80808080u);
}

TEST_F(RdnaGlobal, DecoderCodeAndHashExtendBeyondSixteenKiB) {
  auto& dest = PageForDispatch();
  program_.assign(5001, 0xbf800000);
  program_[0] = 0xbf820000 | 5000;  // jump over padding to the tail
  Mov(1, 123);
  Global(0x1c, 0, 1, 2);
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 123);
  program_[5002] =
      456;  // changing only the late tail must invalidate the cache
  Run(0, reinterpret_cast<u64>(dest.data()));
  EXPECT_EQ(dest[0], 456);
}
}  // namespace

namespace gpu::render {
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0, g_dcb_n = 0;
}  // namespace gpu::render
// NOLINTBEGIN(readability-identifier-naming): C-linkage bridge
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}
// NOLINTEND(readability-identifier-naming)
