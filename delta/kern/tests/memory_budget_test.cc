#include <gtest/gtest.h>

#include "base/option_file.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"
#include "kern/memory_budget.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
constexpr u32 kAnonPrivate = 0x1002;
constexpr u32 kFixed = 0x10;
constexpr u32 kVoid = 0x100;
constexpr u32 kNoFd = static_cast<u32>(-1);

u64 Flexible() {
  return kern::ReadGuestMemoryUsage().flexible_used;
}

TEST(GuestMemoryBudget, FlexibleMappingsAreCountedAndCapped) {
  options::Init();
  base::FindOption("DELTA_FLEXIBLE_BUDGET")->SetFromString("0x100000");
  base::FindOption("DELTA_ENFORCE_GUEST_MEMORY")->SetFromString("1");
  guest::BeginSession();
  guest::ResetSessionResources();
  kern::Process process;
  ASSERT_TRUE(process.GetVma().Init());

  u8* a = kern::sys_mmap(nullptr, 0x80000, 3, kAnonPrivate, kNoFd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(a));
  EXPECT_EQ(Flexible(), 0x80000u);

  EXPECT_TRUE(kern::IsErrnoPtr(
      kern::sys_mmap(nullptr, 0x100000, 3, kAnonPrivate, kNoFd, 0)));
  EXPECT_EQ(Flexible(), 0x80000u);

  // Mapping over itself replaces, it does not add.
  EXPECT_FALSE(kern::IsErrnoPtr(
      kern::sys_mmap(a, 0x80000, 3, kAnonPrivate | kFixed, kNoFd, 0)));
  EXPECT_EQ(Flexible(), 0x80000u);

  // An address-space reservation is not flexible memory.
  u8* reserved = kern::sys_mmap(nullptr, 0x400000, 0, kAnonPrivate | kVoid,
                                kNoFd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(reserved));
  EXPECT_EQ(Flexible(), 0x80000u);

  // Unmapping the middle splits the mapping; both halves stay counted.
  ASSERT_EQ(kern::sys_munmap(a + 0x20000, 0x40000), 0);
  EXPECT_EQ(Flexible(), 0x40000u);
  ASSERT_EQ(kern::sys_munmap(a, 0x80000), 0);
  EXPECT_EQ(Flexible(), 0u);
  ASSERT_EQ(kern::sys_munmap(reserved, 0x400000), 0);

  base::FindOption("DELTA_FLEXIBLE_BUDGET")->SetFromString("0");
  base::FindOption("DELTA_ENFORCE_GUEST_MEMORY")->SetFromString("0");
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  guest::BeginSession();
}
}  // namespace
