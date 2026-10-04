#include <gtest/gtest.h>

#include <array>
#include <thread>

#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
class ShmTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::BeginSession();
    guest::ResetSessionResources();
    fd = kern::sys_shm_open("/shm-test", 0x202, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(kern::sys_ftruncate(fd, 0x8000), 0);
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    process.GetObjTable().Reset();
    guest::ResetSessionResources();
    guest::BeginSession();
  }
  u8* Map(size_t size = 0x8000, size_t offset = 0, u32 flags = 1) {
    auto* ptr = kern::sys_mmap(nullptr, size, 3, flags, fd, offset);
    EXPECT_FALSE(kern::IsErrnoPtr(ptr));
    return ptr;
  }
  kern::Process process;
  int fd = -1;
};

TEST_F(ShmTest, IndependentViewsShareWritesAndSurviveUnmap) {
  auto* first = Map();
  auto* second = Map();
  ASSERT_NE(first, second);
  first[7] = 42;
  EXPECT_EQ(second[7], 42);
  ASSERT_EQ(kern::sys_munmap(first, 0x8000), 0);
  EXPECT_FALSE(host_memory::IsMemoryRangeMapped(first, 0x8000));
  EXPECT_EQ(second[7], 42);
  auto* reopened = Map();
  EXPECT_EQ(reopened[7], 42);
  reopened[7] = 19;
  EXPECT_EQ(second[7], 19);
}

TEST_F(ShmTest, GrowthPreservesExistingMappingsAndZeroesNewPages) {
  auto* first = Map();
  first[0] = 77;
  ASSERT_EQ(kern::sys_ftruncate(fd, 0xc000), 0);
  EXPECT_EQ(kern::ShmFstatSize(fd), 0xc000u);
  auto* grown = Map(0xc000);
  EXPECT_EQ(grown[0], 77);
  EXPECT_EQ(grown[0x8000], 0);
  first[1] = 63;
  EXPECT_EQ(grown[1], 63);
}

TEST_F(ShmTest, OffsetViewAndPartialUnmapPreserveRemainingPages) {
  auto* first = Map();
  auto* tail = Map(0x4000, 0x4000);
  first[0x4000] = 81;
  EXPECT_EQ(tail[0], 81);
  ASSERT_EQ(kern::sys_munmap(first, 0x4000), 0);
  EXPECT_FALSE(host_memory::IsMemoryRangeMapped(first, 0x4000));
  EXPECT_EQ(first[0x4000], 81);
  tail[1] = 91;
  EXPECT_EQ(first[0x4001], 91);
}

TEST_F(ShmTest, UnlinkAndClosePreserveLiveViews) {
  auto* old = Map();
  old[0] = 23;
  ASSERT_EQ(kern::sys_shm_unlink("/shm-test"), 0);
  ASSERT_TRUE(process.GetObjTable().Release(fd));
  fd = kern::sys_shm_open("/shm-test", 0xa02, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 0x8000), 0);
  auto* fresh = Map();
  EXPECT_EQ(fresh[0], 0);
  EXPECT_EQ(old[0], 23);
  fresh[0] = 51;
  EXPECT_EQ(old[0], 23);
}

TEST_F(ShmTest, PrivateViewDoesNotChangeSharedBacking) {
  auto* shared = Map();
  shared[0] = 17;
  auto* private_view = Map(0x8000, 0, 2);
  EXPECT_EQ(private_view[0], 17);
  private_view[0] = 33;
  EXPECT_EQ(shared[0], 17);
  EXPECT_EQ(Map()[0], 17);
}

TEST_F(ShmTest, TruncateAndRegrowDiscardOldContents) {
  auto* old = Map();
  old[0] = 53;
  const int truncated = kern::sys_shm_open("/shm-test", 0x402, 0600);
  ASSERT_GE(truncated, 0);
  EXPECT_EQ(kern::ShmFstatSize(fd), 0u);
  ASSERT_EQ(kern::sys_ftruncate(truncated, 0x8000), 0);
  EXPECT_EQ(old[0], 0);
  EXPECT_EQ(Map()[0], 0);
}

TEST_F(ShmTest, ReportsExactFileLengthAndRejectsInvalidOffsets) {
  ASSERT_EQ(kern::sys_ftruncate(fd, 123), 0);
  EXPECT_EQ(kern::ShmFstatSize(fd), 123u);
  auto* ptr = Map(0x4000);
  EXPECT_EQ(ptr[122], 0);
  EXPECT_EQ(ptr[123], 0);
  EXPECT_EQ(kern::sys_mmap(nullptr, 0x4000, 3, 1, fd, 1),
            reinterpret_cast<u8*>(-kern::SysError::eINVAL));
  EXPECT_EQ(kern::sys_mmap(nullptr, 0x4000, 3, 3, fd, 0),
            reinterpret_cast<u8*>(-kern::SysError::eINVAL));
}

TEST_F(ShmTest, PartialFilePageRemainsAccessibleThroughGuestPageEnd) {
  ASSERT_EQ(kern::sys_ftruncate(fd, 0xf28), 0);
  auto* first = Map(0x4000);
  auto* second = Map(0x4000);
  first[0x3fff] = 94;
  EXPECT_EQ(second[0x3fff], 94);
  EXPECT_EQ(kern::ShmFstatSize(fd), 0xf28u);
}

TEST_F(ShmTest, ShrinkWithinGuestPageZeroesTruncatedTail) {
  auto* first = Map();
  first[0] = 18;
  first[0x2000] = 35;
  ASSERT_EQ(kern::sys_ftruncate(fd, 0x1000), 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 0x3000), 0);
  EXPECT_EQ(first[0], 18);
  EXPECT_EQ(first[0x2000], 0);
  EXPECT_EQ(kern::ShmFstatSize(fd), 0x3000u);
}

TEST_F(ShmTest, ReadOnlyDescriptorCannotTruncateOrWriteSharedView) {
  const int read_only = kern::sys_shm_open("/shm-test", 0, 0600);
  ASSERT_GE(read_only, 0);
  EXPECT_EQ(kern::sys_ftruncate(read_only, 0), -kern::SysError::eINVAL);
  EXPECT_EQ(kern::sys_mmap(nullptr, 0x8000, 3, 1, read_only, 0),
            reinterpret_cast<u8*>(-kern::SysError::eACCES));
  auto* shared = kern::sys_mmap(nullptr, 0x8000, 1, 1, read_only, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(shared));
  Map()[0] = 39;
  EXPECT_EQ(shared[0], 39);
  auto* private_view = kern::sys_mmap(nullptr, 0x8000, 3, 2, read_only, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(private_view));
  private_view[0] = 59;
  EXPECT_EQ(shared[0], 39);
}

TEST_F(ShmTest, FixedViewReplacesMappingAndKeepsBackingAlive) {
  auto* first = Map();
  first[0] = 62;
  ASSERT_EQ(kern::sys_shm_unlink("/shm-test"), 0);
  fd = kern::sys_shm_open("/shm-test", 0x202, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(kern::sys_ftruncate(fd, 0x8000), 0);
  auto* fixed = kern::sys_mmap(first, 0x8000, 3, 0x11, fd, 0);
  ASSERT_EQ(fixed, first);
  EXPECT_EQ(fixed[0], 0);
  fixed[0] = 73;
  EXPECT_EQ(Map()[0], 73);
}

TEST_F(ShmTest, FixedAnonymousMappingReplacesOnlyItsSharedView) {
  auto* first = Map();
  auto* second = Map();
  first[0] = 64;
  auto* anonymous = kern::sys_mmap(first, 0x8000, 3, 0x1012, -1, 0);
  ASSERT_EQ(anonymous, first);
  EXPECT_EQ(first[0], 0);
  EXPECT_EQ(second[0], 64);
  first[0] = 92;
  EXPECT_EQ(second[0], 64);
}

TEST_F(ShmTest, ServiceBackingSharesGuestViewsWithoutGuestDescriptors) {
  ASSERT_EQ(kern::ShmCreate("/service-shm-test", 17), 0);
  const int shared_fd = kern::sys_shm_open("/service-shm-test", 2, 0600);
  ASSERT_GE(shared_fd, 0);
  EXPECT_EQ(kern::ShmFstatSize(shared_fd), 17u);
  auto* guest = kern::sys_mmap(nullptr, 0x4000, 3, 1, shared_fd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(guest));
  std::array<u8, 17> bytes;
  bytes.fill(0xa5);
  ASSERT_EQ(kern::ShmRead("/service-shm-test", 0, bytes.data(), bytes.size()),
            0);
  for (u8 value : bytes)
    EXPECT_EQ(value, 0);
  const u8 value = 42;
  ASSERT_EQ(kern::ShmWrite("/service-shm-test", 16, &value, 1), 0);
  EXPECT_EQ(guest[16], value);
  ASSERT_TRUE(process.GetObjTable().Release(shared_fd));
  guest[3] = 71;
  ASSERT_EQ(kern::ShmRead("/service-shm-test", 0, bytes.data(), bytes.size()),
            0);
  EXPECT_EQ(bytes[3], 71);
  EXPECT_EQ(bytes[16], value);
}

TEST_F(ShmTest, ServiceCopiesRejectOutOfBoundsAndPreserveOutput) {
  const u8 value = 55;
  ASSERT_EQ(kern::ShmCreate("/service-shm-test", 17), 0);
  ASSERT_EQ(kern::ShmWrite("/service-shm-test", 16, &value, 1), 0);
  EXPECT_EQ(kern::ShmCreate("/service-shm-test", 0x8000),
            -kern::SysError::eEXIST);
  std::array<u8, 8> bytes;
  bytes.fill(0xa5);
  for (size_t offset : {size_t(16), size_t(17), SIZE_MAX}) {
    EXPECT_EQ(kern::ShmRead("/service-shm-test", offset, bytes.data(), 8),
              -kern::SysError::eINVAL);
    EXPECT_EQ(kern::ShmWrite("/service-shm-test", offset, bytes.data(), 8),
              -kern::SysError::eINVAL);
    for (u8 byte : bytes)
      EXPECT_EQ(byte, 0xa5);
  }
  EXPECT_EQ(kern::ShmRead("/service-shm-test", 0, nullptr, 1),
            -kern::SysError::eINVAL);
  EXPECT_EQ(kern::ShmWrite("/service-shm-test", 0, nullptr, 1),
            -kern::SysError::eINVAL);
  EXPECT_EQ(kern::ShmRead("/service-shm-test", 17, nullptr, 0), 0);
  EXPECT_EQ(kern::ShmWrite("/service-shm-test", 17, nullptr, 0), 0);
  EXPECT_EQ(kern::ShmRead("/missing-service-shm", 0, bytes.data(), 8),
            -kern::SysError::eNOENT);
  EXPECT_EQ(kern::ShmCreate(nullptr, 0), -kern::SysError::eINVAL);
  EXPECT_EQ(kern::ShmCreate("", 0), -kern::SysError::eINVAL);
  EXPECT_EQ(kern::ShmCreate("/overflow-shm", SIZE_MAX),
            -kern::SysError::eINVAL);
  ASSERT_EQ(kern::ShmRead("/service-shm-test", 16, bytes.data(), 1), 0);
  EXPECT_EQ(bytes[0], value);
}

TEST_F(ShmTest, ServiceUnlinkAndRecreatePreserveOldGuestView) {
  ASSERT_EQ(kern::ShmCreate("/service-shm-test", 17), 0);
  const int shared_fd = kern::sys_shm_open("/service-shm-test", 2, 0600);
  ASSERT_GE(shared_fd, 0);
  auto* guest = kern::sys_mmap(nullptr, 0x4000, 3, 1, shared_fd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(guest));
  guest[0] = 72;
  ASSERT_EQ(kern::sys_shm_unlink("/service-shm-test"), 0);
  u8 value = 0xa5;
  EXPECT_EQ(kern::ShmRead("/service-shm-test", 0, &value, 1),
            -kern::SysError::eNOENT);
  EXPECT_EQ(value, 0xa5);
  ASSERT_EQ(kern::ShmCreate("/service-shm-test", 17), 0);
  ASSERT_EQ(kern::ShmRead("/service-shm-test", 0, &value, 1), 0);
  EXPECT_EQ(value, 0);
  value = 39;
  ASSERT_EQ(kern::ShmWrite("/service-shm-test", 0, &value, 1), 0);
  EXPECT_EQ(guest[0], 72);
}

TEST_F(ShmTest, ServiceCopiesStayValidDuringGuestResize) {
  std::thread copier([] {
    for (u32 i = 0; i < 1000; ++i) {
      u8 value = 0xa5;
      const auto result = kern::ShmRead("/shm-test", 0, &value, 1);
      ASSERT_TRUE(result == 0 || result == -kern::SysError::eINVAL);
      if (result)
        EXPECT_EQ(value, 0xa5);
      else
        EXPECT_TRUE(value == 0 || value == 23);
      value = 23;
      const auto written = kern::ShmWrite("/shm-test", 0, &value, 1);
      EXPECT_TRUE(written == 0 || written == -kern::SysError::eINVAL);
    }
  });
  for (u32 i = 0; i < 1000; ++i) {
    EXPECT_EQ(kern::sys_ftruncate(fd, 0), 0);
    EXPECT_EQ(kern::sys_ftruncate(fd, 17), 0);
  }
  copier.join();
}

TEST_F(ShmTest, InvalidFixedViewPreservesExistingMapping) {
  auto* first = Map();
  first[0] = 83;
  EXPECT_EQ(kern::sys_mmap(first, 0x8000, 3, 0x11, fd, 1),
            reinterpret_cast<u8*>(-kern::SysError::eINVAL));
  EXPECT_EQ(first[0], 83);
  const int read_only = kern::sys_shm_open("/shm-test", 0, 0600);
  ASSERT_GE(read_only, 0);
  EXPECT_EQ(kern::sys_mmap(first, 0x8000, 3, 0x11, read_only, 0),
            reinterpret_cast<u8*>(-kern::SysError::eACCES));
  EXPECT_EQ(first[0], 83);
}
}  // namespace
