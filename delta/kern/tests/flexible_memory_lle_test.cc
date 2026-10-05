#include <gtest/gtest.h>
#include <xbyak.h>

#include <cstdlib>

#include "base/option_file.h"
#include "cpu/backend.h"
#include "guest/session.h"
#include "host_memory/host_memory.h"
#include "kern/crash.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_mem.h"
#include "kern/module.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
TEST(FlexibleMemoryLle, NativeNamedMappingsProvideWritablePages) {
  const char* modules = std::getenv("DELTA_TEST_PS5_MODULES");
  if (!modules)
    GTEST_SKIP() << "Set DELTA_TEST_PS5_MODULES to native PS5 firmware";
  options::Init();
  base::FindOption("DELTA_PS5_MODULES")->SetFromString(modules);
  guest::BeginSession();
  guest::ResetSessionResources();
  kern::InstallCrashHandler();
  kern::Process process;
  process.SetPlatform(kern::Process::Platform::kPs5);
  ASSERT_TRUE(process.GetVma().Init());
  auto kernel = process.LoadModule(base::StringRef("libkernel"));
  ASSERT_TRUE(kernel);
  ASSERT_EQ(kern::sys_dynlib_process_needed_and_relocate(), 0);
  for (const char* nid : {"mL8NDH86iQI", "kc+LEEIYakc"}) {
    const auto function = kernel->GetSymbolByNid(nid);
    ASSERT_NE(function, 0u);
    Xbyak::CodeGenerator thunk(64);
    thunk.mov(thunk.r8, reinterpret_cast<u64>("NativeFlexibleHeap"));
    thunk.mov(thunk.rax, function);
    thunk.jmp(thunk.rax);
    thunk.ready();
    u8* address = nullptr;
    const auto result = cpu::GetBackend().RunGuestFunction(
        reinterpret_cast<uintptr_t>(thunk.getCode()),
        reinterpret_cast<u64>(&address), 0x4000, 3, 0);
    ASSERT_EQ(static_cast<u32>(result), 0u) << nid;
    ASSERT_NE(address, nullptr) << nid;
    EXPECT_EQ(reinterpret_cast<u64>(address) & 0x3fff, 0u);
    EXPECT_LT(reinterpret_cast<u64>(address), 1ull << 40);
    ASSERT_TRUE(host_memory::IsMemoryRangeMapped(address, 0x4000));
    address[0] = 17;
    address[0x3fff] = 29;
    EXPECT_EQ(address[0], 17);
    EXPECT_EQ(address[0x3fff], 29);
    EXPECT_EQ(kern::sys_munmap(address, 0x4000), 0);
  }
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  guest::BeginSession();
}
}  // namespace
