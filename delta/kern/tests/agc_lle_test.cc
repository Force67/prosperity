#include <gtest/gtest.h>
#include <xbyak.h>

#include <array>
#include <cstdlib>

#include "base/option_file.h"
#include "cpu/backend.h"
#include "guest/session.h"
#include "kern/crash.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/module.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
class AgcLleTest : public testing::Test {
 protected:
  void SetUp() override {
    const char* modules = std::getenv("DELTA_TEST_PS5_MODULES");
    if (!modules)
      GTEST_SKIP() << "Set DELTA_TEST_PS5_MODULES to native PS5 firmware";
    options::Init();
    base::FindOption("DELTA_PS5_MODULES")->SetFromString(modules);
    guest::BeginSession();
    guest::ResetSessionResources();
    kern::InstallCrashHandler();
    process.SetPlatform(kern::Process::Platform::kPs5);
    ASSERT_TRUE(process.GetVma().Init());
    agc = process.LoadModule(base::StringRef("libSceAgc"));
    ASSERT_TRUE(agc);
    driver = process.LoadModule(base::StringRef("libSceAgcDriver"));
    ASSERT_TRUE(driver);
    ASSERT_EQ(kern::sys_dynlib_process_needed_and_relocate(), 0);
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
  }
  u64 Call(uintptr_t function, u64 a0, u64 a1 = 0, u64 a2 = 0, u64 a3 = 0) {
    return cpu::GetBackend().RunGuestFunction(function, a0, a1, a2, a3);
  }
  kern::Process process;
  kern::ModulePtr agc, driver;
};

TEST_F(AgcLleTest, NativePacketEncodingCallbacksAndDriverErrors) {
#if !defined(DELTA_BACKEND_NATIVE)
  GTEST_SKIP() << "Xbyak emits x86-64 thunks only on an x86-64 host";
#else
  auto emit = agc->GetSymbolByNid("-KRzWekV120");
  ASSERT_NE(emit, 0u);
  for (const char* nid : {"T6xuVw0KUJo", "AAeX-U5-P3M", "fd5Bp5tGTgo"})
    ASSERT_NE(agc->GetSymbolByNid(nid), 0u) << nid;
  struct Context {
    u64 reserved[2];
    u32* cursor;
    u32* end;
    uintptr_t callback;
    u64 user;
    u32 reserved_words;
    u32 padding;
  } context{};
  std::array<u32, 304> words;
  words.fill(0xa5a5a5a5);
  context.cursor = words.data() + 1;
  context.end = words.data() + 303;
  for (u32 i = 0; i < 100; ++i) {
    auto* packet = words.data() + 1 + i * 3;
    ASSERT_EQ(Call(emit, reinterpret_cast<u64>(&context), i, i >> 2, i >> 1),
              reinterpret_cast<u64>(packet));
    EXPECT_EQ(context.cursor, packet + 3);
    EXPECT_EQ(packet[0], 0xc0017a00u);
    EXPECT_EQ(packet[1], 0x20000243u);
    EXPECT_EQ(packet[2], 0x400u | (i & 3) | (((i >> 2) & 3) << 6) |
                             (((i >> 1) & 1) << 14));
  }
  EXPECT_EQ(words.front(), 0xa5a5a5a5u);
  EXPECT_EQ(words.back(), 0xa5a5a5a5u);

  struct CallbackState {
    u32 required, calls;
    u64 user;
  } callback_state{};
  Xbyak::CodeGenerator reject(128);
  reject.mov(reject.rax, reinterpret_cast<uintptr_t>(&callback_state));
  reject.mov(reject.dword[reject.rax], reject.esi);
  reject.inc(reject.dword[reject.rax + 4]);
  reject.mov(reject.qword[reject.rax + 8], reject.rdx);
  reject.xor_(reject.eax, reject.eax);
  reject.ret();
  reject.ready();
  context.callback = reinterpret_cast<uintptr_t>(reject.getCode());
  context.user = 42;
  auto* cursor = context.cursor;
  EXPECT_EQ(Call(emit, reinterpret_cast<u64>(&context)), 0u);
  EXPECT_EQ(context.cursor, cursor);
  EXPECT_EQ(callback_state.required, 3u);
  EXPECT_EQ(callback_state.calls, 1u);
  EXPECT_EQ(callback_state.user, 42u);
  EXPECT_EQ(words[301], 0xa5a5a5a5u);
  EXPECT_EQ(words[302], 0xa5a5a5a5u);
  context.reserved_words = 4;
  EXPECT_EQ(Call(emit, reinterpret_cast<u64>(&context)), 0u);
  EXPECT_EQ(context.cursor, cursor);
  EXPECT_EQ(callback_state.required, 7u);
  EXPECT_EQ(callback_state.calls, 2u);
  context.reserved_words = 0;

  std::array<u32, 5> refill;
  refill.fill(0xa5a5a5a5);
  Xbyak::CodeGenerator replenish(128);
  replenish.mov(replenish.rax, reinterpret_cast<uintptr_t>(refill.data() + 1));
  replenish.mov(replenish.qword[replenish.rdi + 0x10], replenish.rax);
  replenish.add(replenish.rax, 12);
  replenish.mov(replenish.qword[replenish.rdi + 0x18], replenish.rax);
  replenish.mov(replenish.eax, 1);
  replenish.ret();
  replenish.ready();
  context.callback = reinterpret_cast<uintptr_t>(replenish.getCode());
  ASSERT_EQ(Call(emit, reinterpret_cast<u64>(&context), 1, 2, 1),
            reinterpret_cast<u64>(refill.data() + 1));
  EXPECT_EQ(context.cursor, refill.data() + 4);
  EXPECT_EQ(refill[1], 0xc0017a00u);
  EXPECT_EQ(refill[2], 0x20000243u);
  EXPECT_EQ(refill[3], 0x4481u);
  EXPECT_EQ(refill.front(), 0xa5a5a5a5u);
  EXPECT_EQ(refill.back(), 0xa5a5a5a5u);

  EXPECT_EQ(static_cast<u32>(Call(agc->GetSymbolByNid("T6xuVw0KUJo"), 0)),
            0x8a6d0003u);
  std::array<u8, 96> left{}, right{}, output;
  output.fill(0xa5);
  EXPECT_EQ(static_cast<u32>(Call(agc->GetSymbolByNid("fd5Bp5tGTgo"),
                                  reinterpret_cast<u64>(output.data()),
                                  reinterpret_cast<u64>(left.data()),
                                  reinterpret_cast<u64>(right.data()))),
            0x8a6c0008u);
  for (const u8 value : output)
    EXPECT_EQ(value, 0xa5);

  for (const char* nid : {"F0ZXt5q0ZTA", "uJziRsODk1c", "U9ueyEhSkF4"}) {
    auto query = driver->GetSymbolByNid(nid);
    ASSERT_NE(query, 0u) << nid;
    u32 output = 0xa5a5a5a5;
    EXPECT_EQ(static_cast<u32>(Call(query, reinterpret_cast<u64>(&output))),
              0x8a6c9018u);
    EXPECT_EQ(output, 0xa5a5a5a5u);
  }
#endif
}
}  // namespace
