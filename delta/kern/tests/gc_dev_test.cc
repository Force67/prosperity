#include <gtest/gtest.h>

#include "host_memory/host_memory.h"
#include "kern/ps4/dev/gc_dev.h"
#include "options/options.h"

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
