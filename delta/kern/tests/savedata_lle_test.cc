#include <gtest/gtest.h>

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
TEST(SaveDataLle, MemoryApisRequireInitializationAndPreserveBuffers) {
  const char* modules = std::getenv("DELTA_TEST_PS4_MODULES");
  if (!modules)
    GTEST_SKIP() << "Set DELTA_TEST_PS4_MODULES to native PS4 firmware";
  options::Init();
  base::FindOption("DELTA_PS4_MODULES")->SetFromString(modules);
  guest::BeginSession();
  guest::ResetSessionResources();
  kern::InstallCrashHandler();
  kern::Process process;
  ASSERT_TRUE(process.Create(base::String(modules) + "/libSceSaveData.sprx"));
  ASSERT_EQ(kern::sys_dynlib_process_needed_and_relocate(), 0);
  auto module = process.GetModule(base::StringRef("libSceSaveData"));
  ASSERT_TRUE(module);
  std::array<u8, 512> buffer;
  buffer.fill(0xa5);
  const u64 address = reinterpret_cast<u64>(buffer.data());
  struct Case {
    const char* nid;
    std::array<u64, 4> args;
  };
  const std::array<Case, 8> cases = {{
      {"v7AAAMo0Lz4", {1, 4096, address, 0}},
      {"oQySEUfgXRA", {address, address + 256, 0, 0}},
      {"7Bt5pBC-Aco", {1, address, 64, 0}},
      {"QwOO7vegnV8", {address, 0, 0, 0}},
      {"h3YURzXGSVQ", {1, address, 64, 0}},
      {"cduy9v4YmT4", {address, 0, 0, 0}},
      {"wiT9jeC7xPw", {address, 0, 0, 0}},
      {"aoZKKNjlq3Y", {address, 0, 0, 0}},
  }};
  for (const auto& test : cases) {
    const auto function = module->GetSymbolByNid(test.nid);
    ASSERT_NE(function, 0u) << test.nid;
    for (u32 i = 0; i < 100; ++i) {
      const auto result = cpu::GetBackend().RunGuestFunction(
          function, test.args[0], test.args[1], test.args[2], test.args[3]);
      ASSERT_EQ(static_cast<u32>(result), 0x809f0001u) << test.nid;
      for (u8 value : buffer)
        ASSERT_EQ(value, 0xa5) << test.nid;
    }
  }
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  guest::BeginSession();
}
}  // namespace
