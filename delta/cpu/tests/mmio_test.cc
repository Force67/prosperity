#include <gtest/gtest.h>
#include <ucontext.h>

#include "cpu/mmio.h"

namespace {
uintptr_t g_address;
u32 g_value, g_writes;
bool Write(uintptr_t address, u32 value) {
  g_address = address;
  g_value = value;
  ++g_writes;
  return true;
}

TEST(Mmio, RegisterAndImmediateStoresAdvanceOnlyAfterDeviceAccepts) {
  struct Case {
    const u8 code[16];
    size_t size;
    uintptr_t address;
    u32 value;
  };
  const Case cases[] = {
      {{0x44, 0x89, 0x30}, 3, 0x2000, 73},
      {{0x89, 0x74, 0x88, 0xfc}, 4, 0x2010, 19},
      {{0x41, 0xc7, 0x80, 0x10, 0, 0, 0, 0x2a, 0, 0}, 11, 0x3010, 42}};
  for (const auto& item : cases) {
    ucontext_t context{};
    auto* regs = context.uc_mcontext.gregs;
    regs[REG_RIP] = reinterpret_cast<uintptr_t>(item.code);
    regs[REG_RAX] = 0x2000;
    regs[REG_RCX] = 5;
    regs[REG_RSI] = 19;
    regs[REG_R8] = 0x3000;
    regs[REG_R14] = 73;
    g_writes = 0;
    const auto rip = regs[REG_RIP];
    EXPECT_FALSE(cpu::HandleMmioWrite(&context, item.address + 1, Write));
    EXPECT_EQ(regs[REG_RIP], rip);
    EXPECT_EQ(g_writes, 0u);
    EXPECT_FALSE(cpu::HandleMmioWrite(&context, item.address,
                                      [](uintptr_t, u32) { return false; }));
    EXPECT_EQ(regs[REG_RIP], rip);
    ASSERT_TRUE(cpu::HandleMmioWrite(&context, item.address, Write));
    EXPECT_EQ(regs[REG_RIP], rip + item.size);
    EXPECT_EQ(g_address, item.address);
    EXPECT_EQ(g_value, item.value);
    EXPECT_EQ(g_writes, 1u);
  }
}

TEST(Mmio, RejectsLoadsWideStoresAndRegisterOnlyInstructions) {
  const u8 cases[][4] = {
      {0x8b, 0x30}, {0x48, 0x89, 0x30}, {0x89, 0xf0}, {0x66, 0x89, 0x30}};
  for (const auto& code : cases) {
    ucontext_t context{};
    context.uc_mcontext.gregs[REG_RIP] = reinterpret_cast<uintptr_t>(code);
    g_writes = 0;
    EXPECT_FALSE(cpu::HandleMmioWrite(&context, 0x2000, Write));
    EXPECT_EQ(g_writes, 0u);
    EXPECT_EQ(context.uc_mcontext.gregs[REG_RIP],
              reinterpret_cast<uintptr_t>(code));
  }
}
}  // namespace
