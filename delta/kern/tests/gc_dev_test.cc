#include <gtest/gtest.h>

#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "kern/crash.h"
#include "kern/object_table.h"
#include "kern/ps4/dev/gc_dev.h"
#include "kern/ps4/dev/gc_submit.h"
#include "kern/ps5/dev/gc_dev.h"
#include "options/options.h"

TEST(GcDevicePs5, SubmissionCompletionClearsStatusAndPreservesRequest) {
  options::Init();
  kern::ObjectTable objects;
  auto& device = *new kern::GcDevicePs5(objects);
  ASSERT_TRUE(device.Init("gc", 0, 0));
  u32 request[] = {4, 1};
  ASSERT_EQ(device.Ioctl(0xc0088133, request), 0);
  EXPECT_EQ(request[0], 4u);
  EXPECT_EQ(request[1], 0u);
  request[1] = 7;
  ASSERT_EQ(device.Ioctl(0x80088133, request), 0);
  EXPECT_EQ(request[0], 4u);
  EXPECT_EQ(request[1], 7u);
}

TEST(GcDevice, MappedDoorbellExecutesPacketsWithoutHleAndResets) {
  options::Init();
  guest::BeginSession();
  kern::InstallCrashHandler();
  kern::ObjectTable objects;
  auto& device = *new kern::GcDevice(objects);
  ASSERT_TRUE(device.Init("gc", 0, 0));
  auto* aperture = device.Map(nullptr, 0x4000, 0x22, 1, 0);
  ASSERT_NE(aperture, reinterpret_cast<u8*>(-1));
  auto* memory = static_cast<u32*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1400000000ull), 4096,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(memory, nullptr);
  const u64 result = reinterpret_cast<u64>(memory + 64);
  struct Args {
    u32 me, pipe, queue, vqueue;
    u64 ring, read, state;
    u32 size_log2, reserved;
  } args{1,
         2,
         3,
         41,
         reinterpret_cast<u64>(memory),
         reinterpret_cast<u64>(memory + 65),
         0,
         5,
         0};
  ASSERT_EQ(device.Ioctl(0xc030810d, &args), 0);
  const u32 commands[] = {0xc0033700, 0x500, static_cast<u32>(result),
                          static_cast<u32>(result >> 32), 91};
  for (u32 i = 0; i < 5; ++i)
    memory[i] = commands[i];
  auto* doorbell = reinterpret_cast<volatile u32*>(aperture) + 2 * 8 + 3;
  *doorbell = 5;
  EXPECT_EQ(memory[64], 91u);
  EXPECT_EQ(memory[65], 5u);
  memory[64] = 0;
  *doorbell = 5;
  EXPECT_EQ(memory[64], 0u);
  EXPECT_TRUE(
      kern::GcDevice::IsDoorbellAddress(reinterpret_cast<u64>(doorbell)));
  EXPECT_FALSE(
      kern::GcDevice::WriteDoorbell(reinterpret_cast<u64>(doorbell) + 1, 5));
  u32 unmap[] = {1, 2, 3};
  ASSERT_EQ(device.Ioctl(0xc00c810e, unmap), 0);
  *doorbell = 10;
  EXPECT_EQ(memory[65], 5u);
  guest::ResetSessionResources();
  EXPECT_FALSE(
      kern::GcDevice::IsDoorbellAddress(reinterpret_cast<u64>(doorbell)));
  host_memory::FreeMem(memory, 4096);
}

TEST(GcDevice, LleDescriptorSubmitExecutesPacketsAndRejectsBadDescriptors) {
  options::Init();
  auto* words = static_cast<u32*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1300000000ull), 4096,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(words, nullptr);
  const u64 commands = reinterpret_cast<u64>(words);
  const u64 result = reinterpret_cast<u64>(words + 32);
  words[0] = 0xc0033700;
  words[1] = 0x500;
  words[2] = static_cast<u32>(result);
  words[3] = static_cast<u32>(result >> 32);
  words[4] = 73;
  u32 descriptor[] = {0xc0023f00, static_cast<u32>(commands),
                      static_cast<u32>(commands >> 32), 5};
  words[32] = 0;
  kern::SubmitGcDescriptors(descriptor, 1);
  EXPECT_EQ(words[32], 73u);
  words[32] = 0;
  kern::SubmitGcDescriptors(reinterpret_cast<void*>(1), 1);
  kern::SubmitGcDescriptors(descriptor, 4097);
  descriptor[1] = 1;
  descriptor[2] = 0;
  kern::SubmitGcDescriptors(descriptor, 1);
  EXPECT_EQ(words[32], 0u);
  host_memory::FreeMem(words, 4096);
}

TEST(GcDevice, DoorbellExecutesDirectPacketsAcrossRingWrap) {
  options::Init();
  auto* result = static_cast<u32*>(
      host_memory::AllocMem(reinterpret_cast<void*>(0x1200000000ull), 4096,
                            host_memory::PageProtection::kW,
                            host_memory::AllocationType::kReservecommit));
  ASSERT_NE(result, nullptr);
  const u64 address = reinterpret_cast<u64>(result);
  u32 ring[32]{};
  u32 read = 0;
  auto& queue = kern::GcDevice::g_compute_queues[0];
  queue = {.vqueue = 1,
           .ring_base = reinterpret_cast<u64>(ring),
           .read_ptr = reinterpret_cast<u64>(&read),
           .ring_size_dw = 32,
           .mapped = true};
  for (u32 start : {0u, 30u}) {
    queue.read_offset_dw = start;
    read = start;
    *result = 0;
    const u32 value = 42 + start;
    const u32 commands[] = {0xc0033700, 0x500, static_cast<u32>(address),
                            static_cast<u32>(address >> 32), value};
    for (u32 i = 0; i < 5; i++)
      ring[(start + i) & 31] = commands[i];
    const u32 next = (start + 5) & 31;
    kern::GcDevice::RingDoorbell(1, next);
    EXPECT_EQ(*result, value);
    EXPECT_EQ(read, next);
    EXPECT_EQ(queue.read_offset_dw, next);
    *result = 0;
    kern::GcDevice::RingDoorbell(1, next);
    EXPECT_EQ(*result, 0u);
  }
  queue = {};
  host_memory::FreeMem(result, 4096);
}
