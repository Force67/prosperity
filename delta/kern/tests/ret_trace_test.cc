#include <gtest/gtest.h>
#include <xbyak.h>

#include "guest/session.h"
#include "guest_abi.h"
#include "kern/crash.h"
#include "kern/probe/probe_arm.h"
#include "options/options.h"

namespace {
TEST(RetTrace, PreservesRegistersAndInstructionFlags) {
#if !defined(DELTA_BACKEND_NATIVE)
  GTEST_SKIP() << "Native x86 return tracing";
#else
  options::Init();
  guest::BeginSession();
  guest::ResetSessionResources();
  kern::InstallCrashHandler();
  using Instruction = kern::probe::RetTraceInstruction;
  Xbyak::CodeGenerator generators[4];
  size_t index = 0;
  for (auto instruction : {Instruction::kMovEbxEax, Instruction::kMovEcxEax,
                           Instruction::kTestEax, Instruction::kTestRax}) {
    auto& code = generators[index++];
    code.push(code.rbx);
    code.mov(code.rax, code.rdi);
    code.push(0x202 | 0x8d5);
    code.popfq();
    const size_t offset = code.getSize();
    switch (instruction) {
      case Instruction::kMovEbxEax:
        code.mov(code.ebx, code.eax);
        code.mov(code.rax, code.rbx);
        break;
      case Instruction::kMovEcxEax:
        code.mov(code.ecx, code.eax);
        code.mov(code.rax, code.rcx);
        break;
      case Instruction::kTestEax:
        code.test(code.eax, code.eax);
        break;
      case Instruction::kTestRax:
        code.test(code.rax, code.rax);
        break;
    }
    code.pushfq();
    code.pop(code.rdx);
    code.mov(code.qword[code.rsi], code.rdx);
    code.pop(code.rbx);
    code.ret();
    code.ready();
    auto* site = const_cast<u8*>(code.getCode()) + offset;
    kern::probe::SetRetTrace(reinterpret_cast<uintptr_t>(site), "regression",
                             instruction);
    *site = 0xcc;
    auto function = code.getCode<u64(PS4ABI*)(u64, u64*)>();
    const bool test = instruction == Instruction::kTestEax ||
                      instruction == Instruction::kTestRax;
    const bool wide = instruction == Instruction::kTestRax;
    for (u64 input : {0ull, 1ull, 0x8000000000000000ull, 0x100000000ull,
                      0xffffffff00000001ull}) {
      u64 flags = 0;
      EXPECT_EQ(function(input, &flags),
                test ? input : static_cast<u32>(input));
      u64 expected = 0x8d5;
      if (test) {
        const u64 value = wide ? input : static_cast<u32>(input);
        expected = (value == 0 ? 0x40 : 0) |
                   (value & (wide ? 1ull << 63 : 1ull << 31) ? 0x80 : 0) |
                   (__builtin_parity(value & 0xff) == 0 ? 4 : 0);
      }
      EXPECT_EQ(flags & 0x8d5, expected);
    }
  }
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  guest::BeginSession();
#endif
}
}  // namespace
