#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>
#include <limits>
#include <new>
#include <thread>

#include "host_memory/host_memory.h"
#include "memory_debug/memory_debug.h"

namespace {
using memory_debug::Bucket;
memory_debug::Stats Stats(Bucket bucket) {
  return memory_debug::ReadSnapshot().buckets[static_cast<u32>(bucket)];
}

#if defined(DELTA_MEMORY_DEBUG)
TEST(MemoryDebug, HeapTagSurvivesScopeAndCrossThreadFree) {
  const auto before = Stats(Bucket::kMedia);
  void* pointer;
  {
    const memory_debug::Scope scope(Bucket::kMedia);
    pointer = ::operator new(1234);
  }
  const auto allocated = Stats(Bucket::kMedia);
  EXPECT_EQ(allocated.live, before.live + 1234);
  EXPECT_GE(allocated.backing, before.backing + 1234);
  EXPECT_EQ(allocated.allocations, before.allocations + 1);
  std::thread([pointer] { ::operator delete(pointer); }).join();
  const auto after = Stats(Bucket::kMedia);
  EXPECT_EQ(after.live, before.live);
  EXPECT_EQ(after.backing, before.backing);
  EXPECT_EQ(after.allocations, before.allocations);
}

TEST(MemoryDebug, AlignmentArraysAndNothrowUseTheSameTag) {
  const auto before = Stats(Bucket::kCommands);
  {
    const memory_debug::Scope scope(Bucket::kCommands);
    void* aligned = ::operator new(100, std::align_val_t(4096));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned) % 4096, 0);
    void* array = ::operator new[](50);
    void* nonthrowing = ::operator new(17, std::nothrow);
    ASSERT_NE(nonthrowing, nullptr);
    EXPECT_EQ(Stats(Bucket::kCommands).live, before.live + 167);
    ::operator delete(aligned, std::align_val_t(4096));
    ::operator delete[](array);
    ::operator delete(nonthrowing);
    const volatile size_t impossible = std::numeric_limits<size_t>::max();
    EXPECT_EQ(::operator new(impossible, std::nothrow), nullptr);
  }
  EXPECT_EQ(Stats(Bucket::kCommands).live, before.live);
  EXPECT_EQ(Stats(Bucket::kCommands).allocations, before.allocations);
}

TEST(MemoryDebug, OwnedRecordsMoveAndKeepBackingAfterPartialUse) {
  const auto before = Stats(Bucket::kUpload);
  {
    memory_debug::Record record;
    record.Set(Bucket::kUpload, 64, 256);
    memory_debug::Record moved(static_cast<memory_debug::Record&&>(record));
    record.Reset();
    moved.AddLive(32);
    moved.AddBacking(128);
    const auto current = Stats(Bucket::kUpload);
    EXPECT_EQ(current.live, before.live + 96);
    EXPECT_EQ(current.backing, before.backing + 384);
    EXPECT_EQ(current.allocations, before.allocations + 1);
  }
  const auto after = Stats(Bucket::kUpload);
  EXPECT_EQ(after.live, before.live);
  EXPECT_EQ(after.backing, before.backing);
  EXPECT_EQ(after.allocations, before.allocations);
}

TEST(MemoryDebug, MappingReplacementAndPartialUnmapDoNotDoubleCount) {
  const u64 page = sysconf(_SC_PAGESIZE);
  void* mapping = mmap(nullptr, page * 4, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  const auto before = Stats(Bucket::kJit);
  memory_debug::TrackMapping(mapping, page * 4, Bucket::kJit);
  auto* middle = static_cast<char*>(mapping) + page;
  memory_debug::TrackMapping(middle, page * 2, Bucket::kJit, true);
  EXPECT_EQ(Stats(Bucket::kJit).live, before.live + page * 4);
  memory_debug::ForgetMapping(middle, page);
  EXPECT_EQ(Stats(Bucket::kJit).live, before.live + page * 3);
  memory_debug::ForgetMapping(mapping, page * 4);
  EXPECT_EQ(Stats(Bucket::kJit).live, before.live);
  EXPECT_EQ(Stats(Bucket::kJit).allocations, before.allocations);
  EXPECT_EQ(munmap(mapping, page * 4), 0);
}

TEST(MemoryDebug, ConcurrentCounterUpdatesBalanceWithoutARegistry) {
  const auto before = Stats(Bucket::kReadback);
  std::thread workers[8];
  for (auto& worker : workers) {
    worker = std::thread([] {
      for (u32 i = 0; i < 10000; ++i) {
        memory_debug::Change(Bucket::kReadback, 256, 512, 1);
        memory_debug::Change(Bucket::kReadback, -256, -512, -1);
      }
    });
  }
  for (auto& worker : workers)
    worker.join();
  const auto after = Stats(Bucket::kReadback);
  EXPECT_EQ(after.live, before.live);
  EXPECT_EQ(after.backing, before.backing);
  EXPECT_EQ(after.allocations, before.allocations);
}
#else
TEST(MemoryDebug, DisabledInstrumentationIsInert) {
  memory_debug::Record record;
  record.Set(Bucket::kUpload, 64, 256);
  EXPECT_EQ(Stats(Bucket::kUpload).live, 0);
}
#endif

TEST(MemoryDebug, GuestReservationsAndCommitsAreSeparate) {
  const size_t page = sysconf(_SC_PAGESIZE);
  const auto before = host_memory::GetMappingStats();
  void* reserved = host_memory::AllocMem(nullptr, page * 4,
                                         host_memory::PageProtection::kPriv,
                                         host_memory::AllocationType::kReserve);
  ASSERT_NE(reserved, nullptr);
  EXPECT_EQ(host_memory::GetMappingStats().reserved,
            before.reserved + page * 4);
  EXPECT_EQ(host_memory::GetMappingStats().mapped, before.mapped);
  auto* middle = static_cast<char*>(reserved) + page;
  ASSERT_EQ(
      host_memory::AllocMem(middle, page * 2, host_memory::PageProtection::kW,
                            host_memory::AllocationType::kCommit),
      middle);
  EXPECT_EQ(host_memory::GetMappingStats().reserved,
            before.reserved + page * 2);
  EXPECT_EQ(host_memory::GetMappingStats().mapped, before.mapped + page * 2);
  host_memory::FreeMem(middle, page);
  EXPECT_EQ(host_memory::GetMappingStats().mapped, before.mapped + page);
  host_memory::FreeMem(reserved, page * 4);
  EXPECT_EQ(host_memory::GetMappingStats().mapped, before.mapped);
  EXPECT_EQ(host_memory::GetMappingStats().reserved, before.reserved);
}
}  // namespace
