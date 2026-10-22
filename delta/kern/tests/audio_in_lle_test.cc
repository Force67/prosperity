#include <gtest/gtest.h>
#include <xbyak.h>

#include <cstdlib>

#include "cpu/backend.h"
#include "guest/session.h"
#include "kern/crash.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/module.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
TEST(AudioInLle, FirmwareValidatesCaptureParameters) {
#if !defined(DELTA_BACKEND_NATIVE)
  GTEST_SKIP() << "Xbyak emits x86-64 thunks only on an x86-64 host";
#else
  const char* modules = std::getenv("DELTA_TEST_PS4_MODULES");
  if (!modules)
    GTEST_SKIP()
        << "Set DELTA_TEST_PS4_MODULES and DELTA_PS4_MODULES to firmware";
  options::Init();
  guest::BeginSession();
  guest::ResetSessionResources();
  kern::InstallCrashHandler();
  kern::Process process;
  ASSERT_TRUE(process.Create(base::String(modules) + "/libSceAudioIn.sprx"));
  ASSERT_EQ(kern::sys_dynlib_process_needed_and_relocate(), 0);
  auto module = process.GetModule(base::StringRef("libSceAudioIn"));
  ASSERT_TRUE(module);
  auto open = module->GetSymbolByNid("5NE8Sjc7VC8");
  ASSERT_NE(open, 0u);
  auto call = [&](u32 grain, u32 rate, u32 format) {
    Xbyak::CodeGenerator code(128);
    code.mov(code.rdi, 1);
    code.xor_(code.esi, code.esi);
    code.xor_(code.edx, code.edx);
    code.mov(code.ecx, grain);
    code.mov(code.r8d, rate);
    code.mov(code.r9d, format);
    code.mov(code.rax, open);
    code.jmp(code.rax);
    code.ready();
    return static_cast<u32>(cpu::GetBackend().RunGuestFunction(
        reinterpret_cast<uintptr_t>(code.getCode()), 0, 0, 0));
  };
  EXPECT_EQ(call(256, 48000, 0), 0x80260103u);
  EXPECT_EQ(call(512, 16000, 0), 0x80260102u);
  EXPECT_EQ(call(256, 16000, 2), 0x80260106u);
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  guest::BeginSession();
#endif
}
}  // namespace
